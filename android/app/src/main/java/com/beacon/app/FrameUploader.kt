package com.beacon.app

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

/** The server's ready-to-render glasses card. */
data class DisplayCard(val level: String, val title: String, val lines: List<String>)

data class UploadStats(
    val framesReceived: Long = 0,
    val framesSent: Long = 0,
    val sendFps: Double = 0.0,
    val lastLatencyMs: Long? = null,
    val serverDropped: Long = 0,
    val skippedBusy: Long = 0,
    val errors: Long = 0,
    val lastError: String? = null,
    val lastCard: DisplayCard? = null,
    val lastPeople: Int = 0,
)

/**
 * Turns glasses frames into JPEGs and POSTs them to `<serverUrl>/frame`.
 *
 * The frame collector calls [offer] for every frame; this keeps every [SEND_EVERY_NTH]th one and
 * only while no upload is in flight, so a slow server never queues stale frames and the collector
 * is never blocked (encoding and HTTP happen on [Dispatchers.IO]).
 */
class FrameUploader(private val settings: Settings) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val client =
      OkHttpClient.Builder()
          .connectTimeout(3, TimeUnit.SECONDS)
          .readTimeout(10, TimeUnit.SECONDS)
          .build()
  private val inFlight = AtomicBoolean(false)
  private var frameCounter = 0L
  private val sendTimes = ArrayDeque<Long>()
  private var loggedFormat = false
  private var lastSizeLogAt = 0L
  private var framesAtLastSizeLog = 0L
  @Volatile private var lastJpegBytes = 0

  private val _stats = MutableStateFlow(UploadStats())
  val stats: StateFlow<UploadStats> = _stats.asStateFlow()

  fun clearError() {
    _stats.update { it.copy(lastError = null) }
  }

  fun resetStats() {
    _stats.value = UploadStats()
    frameCounter = 0
    synchronized(sendTimes) { sendTimes.clear() }
  }

  /**
   * Frames still arrive but aren't sent while this is set: with a display attached the glasses
   * keep streaming after being taken off, and nobody should be logged by glasses lying on a desk.
   */
  @Volatile var holdUploads = false

  /** Called on the frame collector's thread. Copies the buffer only for frames that are sent. */
  fun offer(buffer: ByteBuffer, width: Int, height: Int) {
    frameCounter++
    _stats.update { it.copy(framesReceived = frameCounter) }
    logFrameSize(width, height)
    if (holdUploads) return
    if (frameCounter % SEND_EVERY_NTH != 0L) return
    if (!inFlight.compareAndSet(false, true)) {
      _stats.update { it.copy(skippedBusy = it.skippedBusy + 1) }
      return
    }
    val dup = buffer.duplicate()
    val bytes = ByteArray(dup.remaining())
    dup.get(bytes)
    scope.launch {
      try {
        val jpeg = toJpeg(bytes, width, height)
        if (jpeg != null) post(jpeg)
      } finally {
        inFlight.set(false)
      }
    }
  }

  private fun post(jpeg: ByteArray) {
    lastJpegBytes = jpeg.size
    val url = settings.serverUrl.value + "/frame"
    val request =
        Request.Builder().url(url).post(jpeg.toRequestBody("image/jpeg".toMediaType())).build()
    val started = SystemClock.elapsedRealtime()
    try {
      client.newCall(request).execute().use { response ->
        val body = response.body?.string().orEmpty()
        val latency = SystemClock.elapsedRealtime() - started
        if (!response.isSuccessful) {
          fail("HTTP ${response.code} from $url")
          Log.w(TAG, "HTTP ${response.code}: ${body.take(300)}")
          return
        }
        val json = JSONObject(body)
        val dropped = json.optBoolean("dropped", false)
        val card = json.optJSONObject("display")?.let(::parseCard)
        val people = json.optJSONArray("people")?.length() ?: 0
        Log.d(TAG, "frame ${jpeg.size} B -> ${latency} ms dropped=$dropped people=$people card=$card")
        val now = SystemClock.elapsedRealtime()
        val fps =
            synchronized(sendTimes) {
              sendTimes.addLast(now)
              while (sendTimes.isNotEmpty() && now - sendTimes.first() > 5_000) sendTimes.removeFirst()
              if (sendTimes.size < 2) 0.0
              else (sendTimes.size - 1) * 1000.0 / (sendTimes.last() - sendTimes.first())
            }
        _stats.update {
          it.copy(
              framesSent = it.framesSent + 1,
              sendFps = fps,
              lastLatencyMs = latency,
              serverDropped = it.serverDropped + if (dropped) 1 else 0,
              lastError = null,
              lastCard = card ?: it.lastCard,
              lastPeople = if (dropped) it.lastPeople else people,
          )
        }
      }
    } catch (e: IOException) {
      // A cleartext-blocked or refused connection lands here; surface it instead of failing silently.
      fail("${e.javaClass.simpleName}: ${e.message} (is adb reverse running?)")
      Log.w(TAG, "POST $url failed", e)
    } catch (e: Exception) {
      fail("${e.javaClass.simpleName}: ${e.message}")
      Log.w(TAG, "POST $url failed", e)
    }
  }

  private fun fail(message: String) {
    _stats.update { it.copy(errors = it.errors + 1, lastError = message) }
  }

  /**
   * The SDK delivers decoded frames as planar I420 (Y, then U, then V). [YuvImage] wants NV21
   * (Y, then interleaved V/U), so the chroma planes are interleaved here before encoding.
   */
  private fun toJpeg(i420: ByteArray, width: Int, height: Int): ByteArray? {
    val ySize = width * height
    val chromaSize = ySize / 4
    if (i420.size < ySize + 2 * chromaSize) {
      if (!loggedFormat) {
        Log.e(TAG, "Unexpected frame buffer: ${i420.size} B for ${width}x$height (want ${ySize * 3 / 2})")
        loggedFormat = true
      }
      fail("unexpected frame format ${i420.size} B for ${width}x$height")
      return null
    }
    if (!loggedFormat) {
      Log.i(TAG, "Frames: ${width}x$height, ${i420.size} B (I420)")
      loggedFormat = true
    }
    val nv21 = ByteArray(ySize + 2 * chromaSize)
    System.arraycopy(i420, 0, nv21, 0, ySize)
    var o = ySize
    for (i in 0 until chromaSize) {
      nv21[o++] = i420[ySize + chromaSize + i] // V
      nv21[o++] = i420[ySize + i] // U
    }
    // Native size, never shrunk: distant faces need every pixel the glasses send.
    val out = ByteArrayOutputStream()
    YuvImage(nv21, ImageFormat.NV21, width, height, null)
        .compressToJpeg(Rect(0, 0, width, height), JPEG_QUALITY, out)
    return out.toByteArray()
  }

  /**
   * Every [SIZE_LOG_MS], logs the frame size that actually arrives (the SDK may lower the
   * resolution when Bluetooth is tight) with the upload numbers that decide HIGH vs MEDIUM.
   */
  private fun logFrameSize(width: Int, height: Int) {
    val now = SystemClock.elapsedRealtime()
    if (now - lastSizeLogAt < SIZE_LOG_MS) return
    val s = _stats.value
    val window = s.framesReceived - framesAtLastSizeLog
    val inFps = if (lastSizeLogAt == 0L) 0.0 else window * 1000.0 / (now - lastSizeLogAt)
    Log.i(
        TAG,
        "Frame size ${width}x$height · in %.1f fps · sent %.1f fps · last upload %s ms · last JPEG %d KB"
            .format(inFps, s.sendFps, s.lastLatencyMs ?: "-", lastJpegBytes / 1024),
    )
    lastSizeLogAt = now
    framesAtLastSizeLog = s.framesReceived
  }

  private fun parseCard(o: JSONObject): DisplayCard {
    val arr = o.optJSONArray("lines")
    val lines = (0 until (arr?.length() ?: 0)).map { arr!!.optString(it) }
    return DisplayCard(o.optString("level", "quiet"), o.optString("title"), lines)
  }

  companion object {
    private const val TAG = "Beacon/Upload"
    /** 7 fps stream / 3 = ~2.3 fps to the server. */
    const val SEND_EVERY_NTH = 3L
    private const val JPEG_QUALITY = 80
    private const val SIZE_LOG_MS = 10_000L
  }
}
