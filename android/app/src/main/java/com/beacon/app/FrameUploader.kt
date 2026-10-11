package com.beacon.app

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
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
    val errors: Long = 0,
    val lastError: String? = null,
    val lastCard: DisplayCard? = null,
    val lastPeople: Int = 0,
)

/**
 * Turns glasses frames into JPEGs and POSTs them to `<serverUrl>/frame`.
 *
 * The frame collector calls [offer] for every frame; an upload loop sends the newest one as soon
 * as the previous /frame response is back, so the upload rate follows the server's speed, stale
 * frames are never queued, and the collector is never blocked (encoding and HTTP run on IO).
 */
class FrameUploader(
    private val settings: Settings,
    private val sensors: PhoneSensors,
    private val compass: GlassesCompass,
) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val client =
      OkHttpClient.Builder()
          .connectTimeout(3, TimeUnit.SECONDS)
          .readTimeout(10, TimeUnit.SECONDS)
          .build()
  private var frameCounter = 0L
  private val sendTimes = ArrayDeque<Long>()
  private var loggedFormat = false
  // Round-trip window for logRoundTrip (only touched from the single upload loop).
  private val rtts = mutableListOf<Long>()
  private var rttBytes = 0L
  private var rttWindowStart = 0L
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
    // Restart the 10 s frame-size log too, or its first fps after a new session goes negative.
    lastSizeLogAt = 0L
    framesAtLastSizeLog = 0L
    synchronized(sendTimes) { sendTimes.clear() }
  }

  /**
   * Frames still arrive but aren't sent while this is set: with a display attached the glasses
   * keep streaming after being taken off, and nobody should be logged by glasses lying on a desk.
   */
  @Volatile var holdUploads = false

  private class PendingFrame(val bytes: ByteArray, val width: Int, val height: Int)

  /**
   * A one-slot mailbox: each new frame replaces the one waiting, so the upload loop always gets
   * the newest frame and nothing old is ever queued.
   */
  private val latest = Channel<PendingFrame>(Channel.CONFLATED)

  init {
    // The upload loop: as soon as one /frame round trip finishes, send the newest frame there is.
    // One request in flight, and at least MIN_GAP_MS between upload starts.
    scope.launch {
      var lastStart = 0L
      for (frame in latest) {
        val wait = MIN_GAP_MS - (SystemClock.elapsedRealtime() - lastStart)
        val toSend = if (wait > 0) {
          delay(wait)
          latest.tryReceive().getOrNull() ?: frame // a newer one may have landed meanwhile
        } else frame
        if (holdUploads) continue
        lastStart = SystemClock.elapsedRealtime()
        val jpeg = toJpeg(toSend.bytes, toSend.width, toSend.height) ?: continue
        post(jpeg)
      }
    }
  }

  /** Called on the frame collector's thread for every frame; never blocks it. */
  fun offer(buffer: ByteBuffer, width: Int, height: Int) {
    frameCounter++
    _stats.update { it.copy(framesReceived = frameCounter) }
    logFrameSize(width, height)
    if (holdUploads) {
      latest.tryReceive() // drop anything waiting so it isn't sent when uploads resume
      return
    }
    // The SDK reuses the buffer, so the frame is copied before it's handed over.
    val dup = buffer.duplicate()
    val bytes = ByteArray(dup.remaining())
    dup.get(bytes)
    latest.trySend(PendingFrame(bytes, width, height))
  }

  private fun post(jpeg: ByteArray) {
    lastJpegBytes = jpeg.size
    val url = frameUrl()
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
        logRoundTrip(latency, jpeg.size)
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

  /**
   * `<server>/frame` plus whatever the phone knows right now: lat/lng/acc once there's a GPS fix,
   * steps if there's a pedometer. Missing values are left out rather than sent empty.
   */
  private fun frameUrl(): String {
    val base = (settings.serverUrl.value + "/frame").toHttpUrlOrNull()
        ?: return settings.serverUrl.value + "/frame"
    return base.newBuilder().apply {
      sensors.location?.let { fix ->
        addQueryParameter("lat", "%.7f".format(Locale.US, fix.latitude))
        addQueryParameter("lng", "%.7f".format(Locale.US, fix.longitude))
        addQueryParameter("acc", "%.1f".format(Locale.US, fix.accuracy))
      }
      sensors.steps?.let { addQueryParameter("steps", it.toString()) }
      // The glasses' compass (deg clockwise from magnetic north), when Motion is delivering it.
      compass.heading?.let { addQueryParameter("heading", "%.1f".format(Locale.US, it)) }
    }.build().toString()
  }

  /** Every [RTT_LOG_EVERY] uploads: round-trip min/avg/max, upload rate and average JPEG size. */
  private fun logRoundTrip(latencyMs: Long, jpegBytes: Int) {
    rtts += latencyMs
    rttBytes += jpegBytes
    if (rtts.size < RTT_LOG_EVERY) return
    val now = SystemClock.elapsedRealtime()
    val fps = if (rttWindowStart == 0L) 0.0 else RTT_LOG_EVERY * 1000.0 / (now - rttWindowStart)
    Log.i(
        TAG,
        "Round trip (last %d): avg %d ms · min %d · max %d · %.1f uploads/s · %d KB avg"
            .format(rtts.size, rtts.average().toLong(), rtts.min(), rtts.max(), fps, rttBytes / rtts.size / 1024),
    )
    rtts.clear()
    rttBytes = 0L
    rttWindowStart = now
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
    /** Floor between upload starts, so a very fast server can't make the phone spin. */
    private const val MIN_GAP_MS = 80L
    private const val JPEG_QUALITY = 75
    /** Uploads per round-trip summary in logcat. */
    private const val RTT_LOG_EVERY = 20
    private const val SIZE_LOG_MS = 10_000L
  }
}
