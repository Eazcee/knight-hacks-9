package com.beacon.app

import android.app.Activity
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.meta.wearable.dat.camera.Camera
import com.meta.wearable.dat.camera.addCamera
import com.meta.wearable.dat.camera.removeCamera
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamError
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.core.types.DeviceCompatibility
import com.meta.wearable.dat.core.types.DeviceIdentifier
import com.meta.wearable.dat.core.types.LinkState
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.core.types.RegistrationState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** One line per known device: what the SDK says about its link and fitness for a session. */
data class GlassesDevice(
    val name: String,
    val link: LinkState,
    val compatibility: DeviceCompatibility,
    val hinge: String,
    val worn: String,
    val battery: Int,
    /** When the link last became CONNECTED (elapsedRealtime), or 0 while not connected. */
    val connectedAt: Long = 0L,
) {
  /**
   * Connected, unfolded and on someone's head: only then does a session actually start. Folded or
   * not worn, the glasses end every session within ~100 ms ("Session ended by device").
   */
  val isReady: Boolean
    get() =
        link == LinkState.CONNECTED &&
            hinge != "closed" && worn != "doffed" &&
            // Right after a reconnect both read "unknown" until something changes. Waiting a
            // moment avoids sessions on folded glasses dying every 3 s; waiting forever stranded
            // glasses that were already open and worn (they never send a change).
            ((hinge != "unknown" && worn != "unknown") || unknownGraceOver)

  /** Connected long enough that a still-"unknown" hinge/worn means "no change to report". */
  val unknownGraceOver: Boolean
    get() = connectedAt > 0 && SystemClock.elapsedRealtime() - connectedAt >= UNKNOWN_GRACE_MS

  companion object {
    const val UNKNOWN_GRACE_MS = 4_000L
  }
}

data class GlassesStatus(
    val bluetoothGranted: Boolean = false,
    val sdkReady: Boolean = false,
    val registration: RegistrationState = RegistrationState.UNAVAILABLE,
    val devices: Map<DeviceIdentifier, GlassesDevice?> = emptyMap(),
    val wantStreaming: Boolean = false,
    val session: DeviceSessionState? = null,
    val stream: StreamState? = null,
    val frameSize: String? = null,
    /** The glasses report not being worn; uploads are held. */
    val glassesOff: Boolean = false,
    val message: String? = null,
) {
  val isRegistered: Boolean =
      registration == RegistrationState.REGISTERED ||
          registration == RegistrationState.UNREGISTERING
}

/**
 * The glasses side of Beacon: registration, the device session, and the camera stream.
 *
 * Lives in the Application so a rotation never drops the session. Session pause (hinges closed,
 * glasses off) is waited out; a terminal STOPPED session is recreated after a short delay while
 * [GlassesStatus.wantStreaming] is on, since a stopped session cannot be restarted.
 */
class GlassesLink(
    private val context: Context,
    private val uploader: FrameUploader,
    private val sensors: PhoneSensors,
) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  // Frames are handled off the main thread, one at a time, in order.
  private val frameDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _status = MutableStateFlow(GlassesStatus())
  val status: StateFlow<GlassesStatus> = _status.asStateFlow()

  private var session: DeviceSession? = null
  private var camera: Camera? = null
  private var sessionJobs = mutableListOf<Job>()
  private var cameraJobs = mutableListOf<Job>()
  private var retryJob: Job? = null
  /** True once the current session has reached STARTED (so a later STARTED is a resume). */
  private var sessionStarted = false
  /** Pending "glasses have been off long enough" timer. */
  private var offJob: Job? = null
  /** Recent camera re-adds, to cap them at [MAX_REATTACHES] per [REATTACH_WINDOW_MS]. */
  private val reattachTimes = ArrayDeque<Long>()
  private var reattachJob: Job? = null
  /** Consecutive sessions that ended before reaching STARTED, for the reconnect backoff. */
  private var failedAttempts = 0
  /** Set when the glasses refused the camera: the next Start goes straight to Meta AI. */
  private var askForCameraNextStart = false

  /** The glasses display: renders each card the server returns. */
  val display = GlassesDisplay(scope)

  init {
    scope.launch {
      uploader.stats.map { it.lastCard }.distinctUntilChanged().filterNotNull().collect(display::offer)
    }
  }

  private val deviceJobs = mutableMapOf<DeviceIdentifier, Job>()

  /**
   * Called only once BLUETOOTH_CONNECT is granted. Initializing without it leaves the SDK able to
   * list the glasses (Meta AI knows them) but never able to open a session on them: every
   * createSession fails with "No eligible device available". Idempotent.
   */
  fun initialize() {
    _status.update { it.copy(bluetoothGranted = true) }
    if (_status.value.sdkReady) return
    Wearables.initialize(context)
        .onSuccess { _status.update { it.copy(sdkReady = true) } }
        .onFailure { error, _ ->
          Log.e(TAG, "Wearables.initialize failed: ${error.description}")
          note("SDK init failed: ${error.description}")
          return
        }
    scope.launch {
      Wearables.registrationState.collect { state ->
        Log.i(TAG, "Registration $state")
        _status.update { it.copy(registration = state) }
      }
    }
    scope.launch { Wearables.devices.collect(::trackDevices) }
  }

  fun onBluetoothDenied() {
    _status.update { it.copy(bluetoothGranted = false) }
  }

  /** Follows each device's metadata (link state, compatibility, hinge, worn) for the status screen. */
  private fun trackDevices(ids: Set<DeviceIdentifier>) {
    (deviceJobs.keys - ids).forEach { id -> deviceJobs.remove(id)?.cancel() }
    _status.update { s -> s.copy(devices = ids.associateWith { s.devices[it] }) }
    (ids - deviceJobs.keys).forEach { id ->
      val flow = Wearables.devicesMetadata[id] ?: return@forEach
      deviceJobs[id] =
          scope.launch {
            flow.collect { d ->
              val previous = _status.value.devices[id]
              val connectedAt =
                  when {
                    d.linkState != LinkState.CONNECTED -> 0L
                    previous != null && previous.connectedAt > 0 -> previous.connectedAt
                    else -> SystemClock.elapsedRealtime()
                  }
              val info =
                  GlassesDevice(
                      name = d.name.ifEmpty { id.toString() },
                      link = d.linkState,
                      compatibility = d.compatibility,
                      hinge = d.hingeState.name.lowercase(),
                      worn = d.donState.name.lowercase(),
                      battery = d.batteryLevel,
                      connectedAt = connectedAt,
                  )
              if (info != previous) Log.i(TAG, "Device $info")
              _status.update { s -> s.copy(devices = s.devices + (id to info)) }
              // With the display attached, taking the glasses off no longer pauses the session,
              // so hold uploads ourselves while they're off and resume (with a fresh card) when
              // they're back on.
              // The wear sensor blips to "doffed" for 2-3 s while the glasses are adjusted, so
              // only hold once they've been off for OFF_GRACE_MS (the delay the glasses
              // themselves used before pausing, without a display).
              if (info.worn == "doffed") {
                if (offJob == null && !uploader.holdUploads) {
                  offJob =
                      scope.launch {
                        delay(OFF_GRACE_MS)
                        uploader.holdUploads = true
                        _status.update { it.copy(glassesOff = true) }
                        Log.i(TAG, "Glasses off for ${OFF_GRACE_MS / 1000} s: holding uploads")
                      }
                }
              } else {
                offJob?.cancel()
                offJob = null
                if (uploader.holdUploads) {
                  uploader.holdUploads = false
                  _status.update { it.copy(glassesOff = false) }
                  Log.i(TAG, "Glasses on: uploading")
                  if (sessionStarted) display.resend()
                }
              }
              // Glasses just became ready (unfolded and worn): start the session we were holding
              // back, or give a surviving session back its camera.
              if (info.isReady && previous?.isReady != true && _status.value.wantStreaming) {
                if (session == null) {
                  retryJob?.cancel()
                  openSession()
                } else {
                  maybeReattachCamera()
                }
              }
            }
          }
    }
  }

  fun register(activity: Activity) = Wearables.startRegistration(activity)

  fun unregister(activity: Activity) = Wearables.startUnregistration(activity)

  /**
   * Checks the DAT camera permission (asking through Meta AI if needed), then opens a session.
   * [requestPermission] launches the Meta AI permission flow from the activity.
   */
  fun start(requestPermission: suspend (Permission) -> PermissionStatus) {
    if (!_status.value.sdkReady) {
      note("Allow Nearby devices (Bluetooth) first")
      return
    }
    if (!_status.value.isRegistered) {
      note("Register with Meta AI first")
      return
    }
    _status.update { it.copy(wantStreaming = true, message = null) }
    failedAttempts = 0
    StreamingService.start(context)
    sensors.start() // GPS + steps ride along with every frame while streaming
    scope.launch {
      var granted = false
      if (askForCameraNextStart) {
        askForCameraNextStart = false
        granted = requestPermission(Permission.CAMERA) == PermissionStatus.Granted
        Log.i(TAG, "Camera permission (asked after the glasses refused): $granted")
        if (!granted) {
          note("Camera permission denied in Meta AI")
          stop()
          return@launch
        }
        openSession()
        return@launch
      }
      Wearables.checkPermissionStatus(Permission.CAMERA)
          .onSuccess {
            Log.i(TAG, "Camera permission: $it")
            granted = it == PermissionStatus.Granted
          }
          .onFailure { error, _ -> note("Camera permission check failed: ${error.description}") }
      if (!granted) {
        // Checked right after the glasses reconnect, this reads Denied for a grant that
        // still stands; look again before sending the wearer into the Meta AI flow.
        delay(PERMISSION_RECHECK_MS)
        Wearables.checkPermissionStatus(Permission.CAMERA).onSuccess {
          Log.i(TAG, "Camera permission (recheck): $it")
          granted = it == PermissionStatus.Granted
        }
      }
      if (!granted) granted = requestPermission(Permission.CAMERA) == PermissionStatus.Granted
      if (!granted) {
        note("Camera permission denied in Meta AI")
        stop()
        return@launch
      }
      openSession()
    }
  }

  fun stop() {
    _status.update { it.copy(wantStreaming = false) }
    retryJob?.cancel()
    reattachJob?.cancel()
    StreamingService.stop(context)
    sensors.stop()
    val current = session
    if (current == null) {
      _status.update { it.copy(session = null, stream = null) }
    } else {
      current.stop() // STOPPED arrives through the state collector, which cleans up.
    }
  }

  private fun openSession() {
    if (session != null) return
    // Folded or unworn glasses end every session at once. Wait for them to be ready instead; the
    // metadata collector in trackDevices calls back in when they are.
    val known = _status.value.devices.values.filterNotNull()
    if (known.isNotEmpty() && known.none { it.isReady }) {
      val d = known.first()
      note(
          when {
            d.link != LinkState.CONNECTED -> "Waiting for glasses to connect (${d.link.name.lowercase()})"
            d.hinge == "closed" -> "Open the glasses hinges to start"
            d.hinge == "unknown" || d.worn == "unknown" -> "Waiting for the glasses to report in…"
            else -> "Put the glasses on to start"
          }
      )
      // No reading may ever come for glasses that are already open and worn, so look again once
      // the grace period is over instead of waiting on a metadata change.
      if (d.link == LinkState.CONNECTED && (d.hinge == "unknown" || d.worn == "unknown")) {
        val left = GlassesDevice.UNKNOWN_GRACE_MS - (SystemClock.elapsedRealtime() - d.connectedAt)
        retryJob?.cancel()
        retryJob =
            scope.launch {
              delay(left.coerceAtLeast(0L) + 100L)
              if (_status.value.wantStreaming && session == null) openSession()
            }
      }
      return
    }
    Wearables.createSession(AutoDeviceSelector())
        .onSuccess { created ->
          session = created
          uploader.resetStats()
          // Subscribe before start() so no transition is missed.
          sessionJobs += scope.launch { created.state.collect { onSessionState(created, it) } }
          sessionJobs +=
              scope.launch {
                created.errors.collect { error ->
                  Log.w(TAG, "Session error: ${error.description}")
                  note(error.description)
                }
              }
          created.start()
        }
        .onFailure { error, _ ->
          val devices = _status.value.devices.values.joinToString { "${it?.name}: ${it?.link}" }
          Log.w(TAG, "createSession failed: ${error.description} (devices: $devices)")
          note("No session: ${error.description}")
          scheduleRetry()
        }
  }

  private fun onSessionState(s: DeviceSession, state: DeviceSessionState) {
    Log.i(TAG, "Session $state")
    _status.update { it.copy(session = state) }
    when (state) {
      DeviceSessionState.STARTED -> {
        failedAttempts = 0
        // Errors from failed attempts before this session are no longer true.
        _status.update { it.copy(message = null) }
        uploader.clearError()
        if (camera == null) addCamera(s)
        if (sessionStarted) display.resend() else display.attach(s) // resumed vs. new
        sessionStarted = true
      }
      // Hinges closed / glasses off: the SDK resumes on its own; frames simply stop meanwhile.
      DeviceSessionState.PAUSED -> note("Glasses paused (folded or taken off)")
      DeviceSessionState.STOPPED -> {
        val wasStarted = sessionStarted
        teardownSession()
        if (!_status.value.wantStreaming) return
        // Folding reports hinge=closed before the session stops; Back on the glasses' root view
        // stops it while they are still worn and open. Back means "stop", so don't reconnect.
        if (wasStarted && _status.value.devices.values.any { it?.isReady == true }) {
          Log.i(TAG, "Session ended from the glasses while worn: treating as Back")
          stop()
          note("Ended from the glasses (Back). Tap Start streaming to resume.")
        } else {
          note("Glasses disconnected, reconnecting…")
          scheduleRetry()
        }
      }
      else -> Unit
    }
  }

  private fun addCamera(s: DeviceSession) {
    s.addCamera(StreamConfiguration(videoQuality = VideoQuality.HIGH, frameRate = FRAME_RATE))
        .onSuccess { added ->
          camera = added
          val stream = added.stream
          cameraJobs +=
              scope.launch(frameDispatcher) {
                stream.videoStream.collect { frame ->
                  if (frame.isCompressed) return@collect
                  val size = "${frame.width}x${frame.height}"
                  val was = _status.value.frameSize
                  if (was != size) {
                    // Every resolution change, so Bluetooth step-downs can be counted exactly.
                    Log.i(TAG, "Stream size ${was ?: "-"} -> $size")
                    _status.update { it.copy(frameSize = size) }
                  }
                  uploader.offer(frame.buffer, frame.width, frame.height)
                }
              }
          cameraJobs +=
              scope.launch {
                var wasActive = false
                stream.state.collect { state ->
                  Log.i(TAG, "Stream $state")
                  _status.update { it.copy(stream = state) }
                  val terminal = state == StreamState.STOPPED || state == StreamState.CLOSED
                  // Only a stream that really ran counts as lost; one that fails while starting
                  // (e.g. the glasses refuse the camera) must not trigger a re-add.
                  if (state == StreamState.STREAMING) wasActive = true
                  else if (wasActive) onStreamLost(s)
                }
              }
          cameraJobs +=
              scope.launch {
                stream.errorStream.collect { error ->
                  Log.w(TAG, "Stream error: ${error.description}")
                  if (error == StreamError.PERMISSIONS_DENIED) {
                    // The glasses refuse the camera even when the phone-side check says Granted
                    // (an expired "allow once"). Retrying can't fix that; Meta AI has to approve it.
                    askForCameraNextStart = true
                    stop()
                    note("The glasses refused the camera. Tap Start streaming to approve it in Meta AI.")
                  } else {
                    note("Stream: ${error.description}")
                  }
                }
              }
          stream.start().onFailure { error, _ -> note("Stream start failed: ${error.description}") }
        }
        .onFailure { error, _ ->
          // CAPABILITY_DENIED and friends are expected outcomes, not crashes.
          Log.w(TAG, "addCamera failed: ${error.description}")
          note("Camera unavailable: ${error.description}")
        }
  }

  /**
   * The stream ended but the session lives on: taking the glasses off stops the stream, while the
   * session keeps trying to restore itself for ~30 s. A stopped stream can't be restarted, so the
   * dead camera is detached and a new one is added once the session is STARTED and the glasses are
   * ready again (see [maybeReattachCamera]).
   */
  private fun onStreamLost(s: DeviceSession) {
    if (s !== session || camera == null) return
    Log.i(TAG, "Stream ended while the session is alive: detaching camera")
    cameraJobs.forEach { it.cancel() }
    cameraJobs.clear()
    camera = null
    s.removeCamera().onFailure { error, _ -> Log.w(TAG, "removeCamera failed: ${error.description}") }
    _status.update { it.copy(stream = null) }
    maybeReattachCamera()
  }

  /** Adds a fresh camera when the session is up, the glasses are ready, and none is attached. */
  private fun maybeReattachCamera() {
    val s = session ?: return
    if (camera != null || !_status.value.wantStreaming) return
    if (_status.value.session != DeviceSessionState.STARTED) return
    if (_status.value.devices.values.none { it?.isReady == true }) return
    val now = SystemClock.elapsedRealtime()
    reattachTimes.removeAll { now - it > REATTACH_WINDOW_MS }
    if (reattachTimes.size >= MAX_REATTACHES) {
      Log.w(TAG, "Camera failed ${reattachTimes.size}x in a minute: giving up")
      stop()
      note("The camera keeps failing on the glasses. Tap Start streaming to try again.")
      return
    }
    reattachTimes += now
    Log.i(TAG, "Re-adding camera to the live session in ${REATTACH_DELAY_MS} ms")
    reattachJob?.cancel()
    reattachJob =
        scope.launch {
          delay(REATTACH_DELAY_MS)
          // Re-check readiness: a fold often reports a moment after the stream ends. If the glasses
          // aren't ready, the "just became ready" path in trackDevices re-adds the camera later.
          val ready = _status.value.devices.values.any { it?.isReady == true }
          if (session === s && camera == null && _status.value.wantStreaming && ready) addCamera(s)
        }
  }

  private fun teardownSession() {
    reattachJob?.cancel()
    cameraJobs.forEach { it.cancel() }
    cameraJobs.clear()
    sessionJobs.forEach { it.cancel() }
    sessionJobs.clear()
    display.detach()
    camera = null
    session = null
    sessionStarted = false
    _status.update { it.copy(stream = null, frameSize = null) }
  }

  /** Reconnects after 3 s, doubling per consecutive failure up to 30 s; a STARTED session resets it. */
  private fun scheduleRetry() {
    val wait = (RETRY_DELAY_MS shl failedAttempts.coerceAtMost(4)).coerceAtMost(MAX_RETRY_DELAY_MS)
    failedAttempts++
    if (failedAttempts > 1) Log.i(TAG, "Reconnect attempt $failedAttempts in ${wait / 1000} s")
    retryJob?.cancel()
    retryJob =
        scope.launch {
          delay(wait)
          if (_status.value.wantStreaming && session == null) openSession()
        }
  }

  private fun note(message: String) {
    _status.update { it.copy(message = message) }
  }

  companion object {
    private const val TAG = "Beacon/Glasses"
    /** Valid SDK rates: 2, 7, 15, 24, 30. 15 = more frames of people walking by (trial). */
    private const val FRAME_RATE = 15
    private const val RETRY_DELAY_MS = 3_000L
    private const val MAX_RETRY_DELAY_MS = 30_000L
    private const val OFF_GRACE_MS = 5_000L
    private const val REATTACH_DELAY_MS = 2_000L
    private const val REATTACH_WINDOW_MS = 60_000L
    private const val MAX_REATTACHES = 3
    private const val PERMISSION_RECHECK_MS = 2_000L
  }
}
