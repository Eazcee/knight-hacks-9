package com.beacon.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat

/**
 * Keeps the glasses session and the uploads alive with the phone locked in a pocket (adapted from
 * the CameraAccess sample's StreamingService).
 */
class StreamingService : Service() {

  companion object {
    private const val TAG = "Beacon/Service"
    private const val CHANNEL_ID = "beacon_streaming"
    private const val NOTIFICATION_ID = 1001
    private const val WAKELOCK_TAG = "Beacon::Streaming"
    private const val WAKELOCK_TIMEOUT_MS = 2L * 60L * 60L * 1000L
    private const val ACTION_STOP = "com.beacon.app.STOP_STREAMING"

    fun start(context: Context) {
      context.startForegroundService(Intent(context, StreamingService::class.java))
    }

    // Routed through onStartCommand so startForeground() always runs before the service stops
    // (stopService() on a pending start crashes with ForegroundServiceDidNotStartInTimeException).
    fun stop(context: Context) {
      context.startForegroundService(
          Intent(context, StreamingService::class.java).setAction(ACTION_STOP)
      )
    }
  }

  private var wakeLock: PowerManager.WakeLock? = null

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onCreate() {
    super.onCreate()
    getSystemService(NotificationManager::class.java)
        .createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Glasses streaming", NotificationManager.IMPORTANCE_LOW)
        )
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    try {
      startForeground(
          NOTIFICATION_ID,
          notification(),
          ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
      )
    } catch (e: Exception) {
      Log.e(TAG, "Failed to enter foreground", e)
      stopSelf()
      return START_NOT_STICKY
    }
    if (intent?.action == ACTION_STOP) {
      releaseWakeLock()
      stopForeground(STOP_FOREGROUND_REMOVE)
      stopSelf()
      return START_NOT_STICKY
    }
    if (wakeLock == null) {
      wakeLock =
          getSystemService(PowerManager::class.java)
              .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKELOCK_TAG)
              .apply { acquire(WAKELOCK_TIMEOUT_MS) }
    }
    return START_NOT_STICKY
  }

  override fun onDestroy() {
    releaseWakeLock()
    super.onDestroy()
  }

  private fun releaseWakeLock() {
    wakeLock?.let { if (it.isHeld) it.release() }
    wakeLock = null
  }

  private fun notification(): Notification {
    val open =
        PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
    return NotificationCompat.Builder(this, CHANNEL_ID)
        .setContentTitle("Beacon is watching")
        .setContentText("Streaming glasses camera to the server")
        .setSmallIcon(R.drawable.ic_launcher_monochrome)
        .setOngoing(true)
        .setContentIntent(open)
        .setCategory(NotificationCompat.CATEGORY_SERVICE)
        .build()
  }
}
