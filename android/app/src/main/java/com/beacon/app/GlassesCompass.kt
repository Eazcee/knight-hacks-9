package com.beacon.app

import android.os.SystemClock
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// The compass math below is copied from Meta's BirdSpotter sample
// (samples/BirdSpotter/.../data/location/GlassesAim.kt): bearingFromMotion and the vector helpers it
// needs, unchanged apart from visibility.

data class Vector3(val x: Double, val y: Double, val z: Double)

/** One reading off the glasses' motion sensors: acceleration (m/s², gravity included) and field (µT). */
data class GlassesMotionSample(
    val acceleration: Vector3,
    val magneticField: Vector3? = null,
)

/**
 * Which way the glasses look, in their own axes: measured on a worn pair in BirdSpotter, where a
 * level head reads all of g on +Y (sky) and `-Z` is the axis the wearer looks along.
 */
private val Forward = Vector3(0.0, 0.0, -1.0)

/**
 * Which way the wearer is facing, in degrees clockwise from magnetic north, or `null` when the
 * sample cannot say. Tilt-compensated: gravity gives up, the field's horizontal part gives north,
 * their cross product gives east, and the forward axis flattened into that plane is the bearing.
 */
internal fun bearingFromMotion(sample: GlassesMotionSample): Double? {
  val field = sample.magneticField ?: return null
  val up = sample.acceleration.normalized() ?: return null
  val north = field.perpendicularTo(up)?.normalized() ?: return null
  val aim = Forward.perpendicularTo(up)?.normalized() ?: return null

  val east = north.cross(up)
  val degrees = Math.toDegrees(atan2(aim.dot(east), aim.dot(north)))
  return if (degrees < 0) degrees + 360 else degrees
}

private fun Vector3.length(): Double = sqrt(x * x + y * y + z * z)

private fun Vector3.normalized(): Vector3? {
  val length = length()
  return if (length > 1e-9) Vector3(x / length, y / length, z / length) else null
}

private fun Vector3.dot(other: Vector3): Double = x * other.x + y * other.y + z * other.z

private fun Vector3.cross(other: Vector3): Vector3 = Vector3(
    y * other.z - z * other.y,
    z * other.x - x * other.z,
    x * other.y - y * other.x,
)

private fun Vector3.perpendicularTo(axis: Vector3): Vector3? {
  val along = dot(axis)
  val flattened = Vector3(x - along * axis.x, y - along * axis.y, z - along * axis.z)
  return if (flattened.length() > 1e-9) flattened else null
}

/**
 * The glasses' compass heading, smoothed over the last [WINDOW_MS] with a circular mean (average
 * sin/cos, then atan2), so 359° and 1° average to 0° rather than 180°. Fed by the Motion collector
 * in GlassesLink, read by the uploader for every frame.
 */
class GlassesCompass {
  private val window = ArrayDeque<Pair<Long, Double>>() // (elapsedRealtime ms, bearing deg)
  @Volatile private var smoothed: Double? = null
  @Volatile private var updatedAt = 0L

  /** Latest smoothed heading 0-360, or null if there's none or it's older than [STALE_MS]. */
  val heading: Double?
    get() = smoothed?.takeIf { SystemClock.elapsedRealtime() - updatedAt <= STALE_MS }

  /** Called for each glasses Motion sample; samples without a magnetic field are skipped. */
  fun onSample(sample: GlassesMotionSample) {
    if (sample.magneticField == null) return
    val bearing = bearingFromMotion(sample) ?: return
    val now = SystemClock.elapsedRealtime()
    synchronized(window) {
      window.addLast(now to bearing)
      while (window.isNotEmpty() && now - window.first().first > WINDOW_MS) window.removeFirst()
      var s = 0.0
      var c = 0.0
      for ((_, deg) in window) {
        val r = Math.toRadians(deg)
        s += sin(r)
        c += cos(r)
      }
      val mean = Math.toDegrees(atan2(s, c))
      smoothed = if (mean < 0) mean + 360 else mean
      updatedAt = now
    }
  }

  fun clear() {
    synchronized(window) { window.clear() }
    smoothed = null
  }

  companion object {
    private const val WINDOW_MS = 1_000L
    private const val STALE_MS = 3_000L
  }
}
