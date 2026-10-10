package com.beacon.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import kotlin.coroutines.resume
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class MainActivity : ComponentActivity() {
  private val app get() = application as BeaconApplication

  // BLUETOOTH_CONNECT must be granted before Wearables.initialize: without it the SDK still lists
  // the glasses (Meta AI knows them) but every createSession fails with "No eligible device".
  private val bluetoothRequest =
      registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
          bootGlasses()
        } else {
          app.glasses.onBluetoothDenied()
          // Denied twice (or "don't ask again"): Android no longer shows the prompt.
          bluetoothBlocked =
              !shouldShowRequestPermissionRationale(Manifest.permission.BLUETOOTH_CONNECT)
        }
      }
  private var bluetoothBlocked = false

  private val notificationsRequest =
      registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

  private var wearablePermissionContinuation: CancellableContinuation<PermissionStatus>? = null
  private val wearablePermissionMutex = Mutex()
  private val wearablePermissionLauncher =
      registerForActivityResult(Wearables.RequestPermissionContract()) { result ->
        wearablePermissionContinuation?.resume(result.getOrDefault(PermissionStatus.Denied))
        wearablePermissionContinuation = null
      }

  /** Asks Meta AI for a glasses permission (e.g. CAMERA), one request at a time. */
  suspend fun requestWearablePermission(permission: Permission): PermissionStatus =
      wearablePermissionMutex.withLock {
        suspendCancellableCoroutine { cont ->
          wearablePermissionContinuation = cont
          cont.invokeOnCancellation { wearablePermissionContinuation = null }
          wearablePermissionLauncher.launch(permission)
        }
      }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (granted(Manifest.permission.BLUETOOTH_CONNECT)) {
      bootGlasses()
    } else {
      bluetoothRequest.launch(Manifest.permission.BLUETOOTH_CONNECT)
    }

    enableEdgeToEdge()
    setContent {
      StatusScreen(
          glasses = app.glasses,
          onGrantBluetooth = ::requestBluetooth,
          uploader = app.uploader,
          settings = app.settings,
          onRegister = { app.glasses.register(this) },
          onUnregister = { app.glasses.unregister(this) },
          onStart = { app.glasses.start(::requestWearablePermission) },
          onStop = { app.glasses.stop() },
      )
    }
  }

  override fun onResume() {
    super.onResume()
    // Back from the system settings page with Nearby devices now allowed.
    if (!app.glasses.status.value.sdkReady && granted(Manifest.permission.BLUETOOTH_CONNECT)) {
      bootGlasses()
    }
  }

  private fun requestBluetooth() {
    if (bluetoothBlocked) {
      startActivity(
          Intent(
              android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
              Uri.fromParts("package", packageName, null),
          )
      )
    } else {
      bluetoothRequest.launch(Manifest.permission.BLUETOOTH_CONNECT)
    }
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    if (app.glasses.status.value.sdkReady) handleRegistrationIntent(intent)
  }

  private fun bootGlasses() {
    app.glasses.initialize()
    if (app.glasses.status.value.sdkReady) handleRegistrationIntent(intent)
    if (!granted(Manifest.permission.POST_NOTIFICATIONS)) {
      notificationsRequest.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
  }

  /** Registration started from inside the Meta AI app arrives as an intent; continue it. */
  private fun handleRegistrationIntent(intent: Intent?) {
    if (intent == null) return
    Wearables.handleIntent(intent) { request -> request.continueRegistration(this) }
  }

  private fun granted(permission: String) =
      ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
}
