package com.beacon.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class PhoneSensorStatus(
    val locationAllowed: Boolean = false,
    val hasFix: Boolean = false,
    val accuracyM: Float? = null,
    val stepsAllowed: Boolean = false,
    val hasStepSensor: Boolean = true,
    val steps: Long? = null,
)

/**
 * The phone's GPS and pedometer, sent with every frame so the server can place sightings on the
 * map and dead-reckon indoors. Only runs while streaming. No heading is sent: the glasses'
 * orientation isn't documented as north-referenced, so the server derives heading from the walk.
 */
class PhoneSensors(private val context: Context) {
  private val fused = LocationServices.getFusedLocationProviderClient(context)
  private val sensorManager = context.getSystemService(SensorManager::class.java)
  private val stepSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)

  /** Latest fix and cumulative step count, read by the uploader on every frame. */
  @Volatile var location: Location? = null
    private set
  @Volatile var steps: Long? = null
    private set

  private val _status = MutableStateFlow(PhoneSensorStatus(hasStepSensor = stepSensor != null))
  val status: StateFlow<PhoneSensorStatus> = _status.asStateFlow()

  private var running = false

  private val locationCallback =
      object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
          val fix = result.lastLocation ?: return
          location = fix
          _status.value = _status.value.copy(hasFix = true, accuracyM = fix.accuracy)
        }
      }

  private val stepListener =
      object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
          // TYPE_STEP_COUNTER is cumulative since boot, which is what the server expects.
          val n = event.values.firstOrNull()?.toLong() ?: return
          steps = n
          _status.value = _status.value.copy(steps = n)
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
      }

  /** Starts both feeds as far as their permissions allow. Safe to call again after a grant. */
  @SuppressLint("MissingPermission") // checked by allowed() just above each use
  fun start() {
    if (running) return
    running = true
    val locationOk = allowed(Manifest.permission.ACCESS_FINE_LOCATION) ||
        allowed(Manifest.permission.ACCESS_COARSE_LOCATION)
    val stepsOk = allowed(Manifest.permission.ACTIVITY_RECOGNITION)
    _status.value = _status.value.copy(locationAllowed = locationOk, stepsAllowed = stepsOk)

    if (locationOk) {
      val request =
          LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, LOCATION_INTERVAL_MS).build()
      fused.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
          .addOnFailureListener { Log.w(TAG, "Location updates failed", it) }
    }
    if (stepsOk && stepSensor != null) {
      sensorManager.registerListener(stepListener, stepSensor, SensorManager.SENSOR_DELAY_NORMAL)
    }
    Log.i(TAG, "Phone sensors on: location=$locationOk steps=${stepsOk && stepSensor != null}")
  }

  fun stop() {
    if (!running) return
    running = false
    fused.removeLocationUpdates(locationCallback)
    sensorManager.unregisterListener(stepListener)
  }

  /** Called after the user answers the permission prompts, so a running session picks them up. */
  fun onPermissionsChanged() {
    if (!running) {
      _status.value = _status.value.copy(
          locationAllowed = allowed(Manifest.permission.ACCESS_FINE_LOCATION) ||
              allowed(Manifest.permission.ACCESS_COARSE_LOCATION),
          stepsAllowed = allowed(Manifest.permission.ACTIVITY_RECOGNITION),
      )
      return
    }
    stop()
    start()
  }

  private fun allowed(permission: String) =
      ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

  companion object {
    private const val TAG = "Beacon/Sensors"
    private const val LOCATION_INTERVAL_MS = 1_000L
  }
}
