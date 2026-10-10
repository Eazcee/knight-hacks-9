package com.beacon.app

import android.os.SystemClock
import android.util.Log
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.display.Display
import com.meta.wearable.dat.display.addDisplay
import com.meta.wearable.dat.display.types.DisplayState
import com.meta.wearable.dat.display.views.Alignment
import com.meta.wearable.dat.display.views.ContentScope
import com.meta.wearable.dat.display.views.Direction
import com.meta.wearable.dat.display.views.FlexBoxBackground
import com.meta.wearable.dat.display.views.IconName
import com.meta.wearable.dat.display.views.TextColor
import com.meta.wearable.dat.display.views.TextStyle
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class DisplayStatus(
    val state: DisplayState? = null,
    val shown: DisplayCard? = null,
    val sends: Long = 0,
    val error: String? = null,
)

/**
 * Renders the server's `display` card on the glasses.
 *
 * The card is the app's only glasses view, so it is the L0 root: there is nowhere deeper to go,
 * and the system Back gesture from it ends the session (GlassesLink treats that as "stop").
 *
 * Each sendContent replaces the whole screen, so a card is sent only when it differs from what is
 * showing. Display sleep is respected by not waking it for place-only ("quiet") updates: those are
 * sent only to replace a person/alert card, and a person/alert card is held for [HOLD_MS] so one
 * frame without a face doesn't blank it.
 */
class GlassesDisplay(private val scope: CoroutineScope) {
  private val _status = MutableStateFlow(DisplayStatus())
  val status: StateFlow<DisplayStatus> = _status.asStateFlow()

  private var display: Display? = null
  private var jobs = mutableListOf<Job>()
  private val sendLock = Mutex()
  /** Latest card from the server, whether or not it was shown. */
  private var wanted: DisplayCard? = null
  /** When the server last returned a person/alert card. */
  @Volatile private var lastPersonAt = 0L
  /** When content last reached the glasses. */
  @Volatile private var lastSentAt = 0L
  private var refreshJob: Job? = null
  /** When the server last returned an alert card. */
  @Volatile private var lastAlertAt = 0L
  private var alertHoldJob: Job? = null
  private var holdJob: Job? = null

  /** Called once the session is STARTED. A refusal (e.g. CAPABILITY_DENIED) leaves the camera running. */
  fun attach(session: DeviceSession) {
    if (display != null) return
    session
        .addDisplay()
        .onSuccess { added ->
          display = added
          jobs +=
              scope.launch {
                added.state.collect { state ->
                  Log.i(TAG, "Display $state")
                  _status.update { it.copy(state = state) }
                  // Fresh content whenever the display (re)starts: the idle root until the server
                  // has said anything, then the latest card.
                  if (state == DisplayState.STARTED) resend()
                }
              }
        }
        .onFailure { error, _ ->
          Log.w(TAG, "addDisplay failed: ${error.description}")
          _status.update { it.copy(error = "Display unavailable: ${error.description}") }
        }
  }

  fun detach() {
    jobs.forEach { it.cancel() }
    jobs.clear()
    holdJob?.cancel()
    display = null
    _status.update { DisplayStatus(sends = it.sends) }
  }

  /** Re-sends what should be showing, e.g. after the session resumes from PAUSED. */
  fun resend() {
    scope.launch { send(wanted ?: IDLE, force = true) }
  }

  /** Called with each card the server returns. */
  fun offer(card: DisplayCard) {
    wanted = card
    val now = SystemClock.elapsedRealtime()
    // Every person/alert card counts as "still in view", even an identical one that isn't resent.
    if (!card.isQuiet) lastPersonAt = now
    if (card.isAlert) lastAlertAt = now
    val shown = _status.value.shown
    // An alert outranks everything: a frame where the same face matched a known person instead of
    // the watchlist mustn't knock it off the screen. Keep it until HOLD_MS with no alert match.
    if (shown != null && shown.isAlert && !card.isAlert && now - lastAlertAt < HOLD_MS) {
      if (alertHoldJob?.isActive != true) {
        alertHoldJob =
            scope.launch {
              while (SystemClock.elapsedRealtime() - lastAlertAt < HOLD_MS) {
                delay(HOLD_MS - (SystemClock.elapsedRealtime() - lastAlertAt))
              }
              wanted?.let { if (!it.isAlert) offer(it) }
            }
      }
      return
    }
    if (card == shown) {
      holdJob?.cancel() // the person is back: drop any pending switch to quiet
      return
    }
    if (card.isQuiet) {
      // Place-only update: don't wake a sleeping display for it.
      if (shown == null || shown.isQuiet || shown == IDLE) return
      // Replacing a person/alert card: only after HOLD_MS with no face at all, so a face the
      // detector catches in only some frames doesn't make the card flicker.
      if (holdJob?.isActive == true) return
      val left = HOLD_MS - (SystemClock.elapsedRealtime() - lastPersonAt)
      holdJob?.cancel()
      holdJob =
          scope.launch {
            var wait = left
            while (wait > 0) {
              delay(wait)
              wait = HOLD_MS - (SystemClock.elapsedRealtime() - lastPersonAt)
            }
            val latest = wanted
            if (latest != null && latest.isQuiet) send(latest)
          }
      return
    }
    holdJob?.cancel()
    // Same card with only its numbers changed (match %, route distance): each sendContent redraws
    // the whole screen, so refresh it at most every REFRESH_MS instead of on every frame.
    if (shown != null && shown.level == card.level && shown.title == card.title) {
      if (refreshJob?.isActive == true) return // a pending refresh will pick up `wanted`
      val wait = REFRESH_MS - (SystemClock.elapsedRealtime() - lastSentAt)
      refreshJob =
          scope.launch {
            if (wait > 0) delay(wait)
            val latest = wanted ?: return@launch
            if (!latest.isQuiet) send(latest)
          }
      return
    }
    refreshJob?.cancel()
    scope.launch { send(card) }
  }

  private suspend fun send(card: DisplayCard, force: Boolean = false) {
    sendLock.withLock {
      val target = display ?: return
      if (_status.value.state != DisplayState.STARTED) return
      if (!force && card == _status.value.shown) return
      target
          .sendContent { render(card) }
          .onSuccess {
            lastSentAt = SystemClock.elapsedRealtime()
            Log.d(TAG, "Shown: $card")
            _status.update { it.copy(shown = card, sends = it.sends + 1, error = null) }
          }
          .onFailure { error, _ ->
            Log.w(TAG, "sendContent failed: ${error.description}")
            _status.update { it.copy(error = "Display: ${error.description}") }
          }
    }
  }

  /**
   * Title as a heading, then up to three body lines (the 600x600 display fits no more). An alert's
   * heading is a warning icon + "ALERT"; the server's own title (e.g. "ALERT: AMBER Alert · Leo · 7")
   * moves down to become the first of the three lines.
   */
  private fun ContentScope.render(card: DisplayCard) {
    flexBox(direction = Direction.COLUMN, gap = 12, padding = 24, background = FlexBoxBackground.CARD) {
      val lines: List<String>
      if (card.isAlert) {
        flexBox(direction = Direction.ROW, gap = 12, crossAlignment = Alignment.CENTER) {
          icon(name = IconName.EXCLAMATION_TRIANGLE)
          text("ALERT", style = TextStyle.HEADING)
        }
        // "ALERT: AMBER Alert · " (a label saved without a name) -> "AMBER Alert"
        val label = card.title.removePrefix("ALERT").trim(':', ' ', '·', '-')
        lines = listOfNotNull(label.ifEmpty { null }) + card.lines
      } else {
        // The server sends an empty title when it has no place yet.
        text(card.title.ifEmpty { "Beacon" }, style = TextStyle.HEADING)
        lines = card.lines
      }
      lines.take(MAX_LINES).forEach { line ->
        text(line, style = TextStyle.BODY, color = if (card.isAlert) TextColor.PRIMARY else TextColor.SECONDARY)
      }
    }
  }

  companion object {
    private const val TAG = "Beacon/Display"
    private const val MAX_LINES = 3
    private const val HOLD_MS = 3_000L
    private const val REFRESH_MS = 2_000L
    /** The L0 root before the server has sent a card. */
    val IDLE = DisplayCard(level = "quiet", title = "Beacon", lines = listOf("Watching…"))
  }
}

private val DisplayCard.isAlert get() = level == "alert"
private val DisplayCard.isQuiet get() = level == "quiet"
