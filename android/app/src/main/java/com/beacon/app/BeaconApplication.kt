package com.beacon.app

import android.app.Application

/**
 * Owns the long-lived pieces so the glasses session survives rotation and activity recreation.
 * [Wearables.initialize] is called from MainActivity once BLUETOOTH_CONNECT is granted (initializing
 * without it makes the SDK see no glasses for the life of the process).
 */
class BeaconApplication : Application() {
  lateinit var settings: Settings
    private set

  lateinit var sensors: PhoneSensors
    private set

  lateinit var compass: GlassesCompass
    private set

  lateinit var uploader: FrameUploader
    private set

  lateinit var glasses: GlassesLink
    private set

  override fun onCreate() {
    super.onCreate()
    settings = Settings(this)
    sensors = PhoneSensors(this)
    compass = GlassesCompass()
    uploader = FrameUploader(settings, sensors, compass)
    glasses = GlassesLink(this, uploader, sensors, compass)
  }
}
