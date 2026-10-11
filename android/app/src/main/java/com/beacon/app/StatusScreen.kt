package com.beacon.app

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.core.types.DeviceCompatibility
import com.meta.wearable.dat.core.types.LinkState
import com.meta.wearable.dat.display.types.DisplayState

@Composable
fun StatusScreen(
    glasses: GlassesLink,
    onGrantBluetooth: () -> Unit,
    uploader: FrameUploader,
    sensors: PhoneSensors,
    settings: Settings,
    onRegister: () -> Unit,
    onUnregister: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
  val status by glasses.status.collectAsStateWithLifecycle()
  val stats by uploader.stats.collectAsStateWithLifecycle()
  val serverUrl by settings.serverUrl.collectAsStateWithLifecycle()
  val display by glasses.display.status.collectAsStateWithLifecycle()
  val phone by sensors.status.collectAsStateWithLifecycle()

  MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
    Surface(Modifier.fillMaxSize()) {
      Column(
          Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
          verticalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        Text("Beacon", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)

        Section("Glasses") {
          StatRow(
              "Bluetooth (Nearby devices)",
              if (status.bluetoothGranted) "allowed" else "not allowed",
              if (status.bluetoothGranted) Good else MaterialTheme.colorScheme.error,
          )
          if (!status.bluetoothGranted) {
            Text(
                "Beacon can't reach the glasses without this. Tap below and choose Allow.",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )
            Button(onGrantBluetooth, modifier = Modifier.fillMaxWidth()) { Text("Allow Bluetooth") }
          }
          StatRow("Meta AI", status.registration.name.lowercase())
          StatRow("Glasses found", status.devices.size.toString())
          status.devices.values.filterNotNull().forEach { d ->
            StatRow(
                d.name,
                "${d.link.name.lowercase()} · ${d.compatibility.name.lowercase()}",
                if (d.link == LinkState.CONNECTED && d.compatibility == DeviceCompatibility.COMPATIBLE) Good else Warn,
            )
            Text(
                "hinges ${d.hinge} · ${d.worn} · battery ${d.battery}%",
                style = MaterialTheme.typography.bodySmall,
            )
          }
          StatRow("Session", status.session?.name?.lowercase() ?: "none", sessionColor(status.session))
          StatRow("Stream", status.stream?.name?.lowercase() ?: "none", streamColor(status.stream))
          status.frameSize?.let { StatRow("Frame size", it) }
          if (status.glassesOff && status.session != null) {
            StatRow("Uploads", "held: glasses not worn", Warn)
          }
          status.message?.let {
            Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
          }
          Spacer(Modifier.height(4.dp))
          if (!status.isRegistered) {
            Button(onRegister, enabled = status.sdkReady, modifier = Modifier.fillMaxWidth()) {
              Text("Connect to Meta AI")
            }
          } else if (!status.wantStreaming) {
            Button(onStart, modifier = Modifier.fillMaxWidth().height(56.dp)) {
              Text("Start streaming", style = MaterialTheme.typography.titleMedium)
            }
            OutlinedButton(onUnregister, modifier = Modifier.fillMaxWidth()) { Text("Disconnect from Meta AI") }
          } else {
            Button(
                onStop,
                modifier = Modifier.fillMaxWidth().height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
            ) {
              Text("Stop streaming", style = MaterialTheme.typography.titleMedium)
            }
          }
        }

        Section("Phone (sent with every frame)") {
          StatRow(
              "GPS",
              when {
                !phone.locationAllowed -> "not allowed"
                phone.accuracyM != null -> "±%.0f m".format(phone.accuracyM)
                status.wantStreaming -> "waiting for a fix…"
                else -> "on while streaming"
              },
              when {
                !phone.locationAllowed -> MaterialTheme.colorScheme.error
                phone.accuracyM != null && phone.accuracyM!! <= 10f -> Good
                phone.accuracyM != null -> Warn
                else -> null
              },
          )
          StatRow(
              "Steps",
              when {
                !phone.hasStepSensor -> "no step sensor"
                !phone.stepsAllowed -> "not allowed"
                phone.steps != null -> phone.steps.toString()
                else -> "–"
              },
              if (phone.hasStepSensor && !phone.stepsAllowed) MaterialTheme.colorScheme.error else null,
          )
          StatRow("Heading", "from walking direction (server)")
        }

        Section("Server") {
          StatRow("Frames from glasses", stats.framesReceived.toString())
          StatRow("Frames sent", stats.framesSent.toString())
          StatRow("Send rate", "%.1f fps".format(stats.sendFps))
          StatRow("Latency", stats.lastLatencyMs?.let { "$it ms" } ?: "–")
          StatRow("Dropped by server", stats.serverDropped.toString())
          StatRow("Not sent (newer frame won)", (stats.framesReceived - stats.framesSent).coerceAtLeast(0).toString())
          StatRow("Errors", stats.errors.toString(), if (stats.errors > 0) MaterialTheme.colorScheme.error else null)
          StatRow("People in last frame", stats.lastPeople.toString())
          stats.lastError?.let {
            Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
          }
        }

        Section("Glasses display") {
          StatRow(
              "Display",
              display.state?.name?.lowercase() ?: "not attached",
              if (display.state == DisplayState.STARTED) Good else null,
          )
          StatRow("Cards sent to glasses", display.sends.toString())
          display.error?.let {
            Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
          }
          display.shown?.let { card ->
            Text("Showing on the glasses:", style = MaterialTheme.typography.bodySmall)
            CardPreview(card)
          }
          stats.lastCard?.takeIf { it != display.shown }?.let { card ->
            Text("Latest from server (not shown):", style = MaterialTheme.typography.bodySmall)
            CardPreview(card)
          }
        }

        Section("Server URL") {
          var draft by remember(serverUrl) { mutableStateOf(serverUrl) }
          OutlinedTextField(
              draft,
              { draft = it },
              singleLine = true,
              keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
              modifier = Modifier.fillMaxWidth(),
          )
          Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button({ settings.setServerUrl(draft) }, enabled = draft.trim() != serverUrl) { Text("Save") }
            OutlinedButton({ settings.setServerUrl(Settings.DEFAULT_SERVER_URL) }) { Text("USB default") }
          }
          Text(
              "USB: run  adb reverse tcp:8000 tcp:8000  on the laptop after every replug.\n" +
                  "Hotspot: enter http://<laptop hotspot IP>:8000",
              style = MaterialTheme.typography.bodySmall,
              fontFamily = FontFamily.Monospace,
          )
        }
      }
    }
  }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
  Card(Modifier.fillMaxWidth()) {
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
      Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
      HorizontalDivider()
      content()
    }
  }
}

@Composable
private fun StatRow(label: String, value: String, valueColor: Color? = null) {
  Row(
      Modifier.fillMaxWidth(),
      horizontalArrangement = Arrangement.SpaceBetween,
      verticalAlignment = Alignment.CenterVertically,
  ) {
    Text(label, style = MaterialTheme.typography.bodyLarge)
    Text(
        value,
        style = MaterialTheme.typography.bodyLarge,
        fontWeight = FontWeight.Medium,
        color = valueColor ?: Color.Unspecified,
    )
  }
}

/** Phone-side mirror of a glasses card: what the wearer sees, alert in red. */
@Composable
private fun CardPreview(card: DisplayCard) {
  val alert = card.level == "alert"
  Card(
      Modifier.fillMaxWidth(),
      colors =
          androidx.compose.material3.CardDefaults.cardColors(
              containerColor = if (alert) Color(0xFFB71C1C) else Color(0xFF1C1C1E),
              contentColor = Color.White,
          ),
  ) {
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
      Text(
          if (alert) "⚠ ALERT" else card.title,
          style = MaterialTheme.typography.titleLarge,
          fontWeight = FontWeight.Bold,
      )
      val lines = if (alert) listOf(card.title.removePrefix("ALERT").trim(':', ' ', '·', '-')) + card.lines else card.lines
      lines.filter { it.isNotEmpty() }.take(3).forEach { Text(it, style = MaterialTheme.typography.bodyLarge) }
    }
  }
}

private val Good = Color(0xFF2E7D32)
private val Warn = Color(0xFFEF6C00)

private fun sessionColor(s: DeviceSessionState?): Color? =
    when (s) {
      DeviceSessionState.STARTED -> Good
      DeviceSessionState.PAUSED, DeviceSessionState.STARTING -> Warn
      else -> null
    }

private fun streamColor(s: StreamState?): Color? =
    when (s) {
      StreamState.STREAMING -> Good
      StreamState.PAUSED, StreamState.STARTING, StreamState.STARTED -> Warn
      else -> null
    }
