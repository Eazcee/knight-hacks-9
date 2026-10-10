package com.beacon.app

import android.content.Context
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Persisted app settings. Default server is the laptop over `adb reverse tcp:8000 tcp:8000`. */
class Settings(context: Context) {
  private val prefs = context.getSharedPreferences("beacon", Context.MODE_PRIVATE)

  private val _serverUrl = MutableStateFlow(prefs.getString(KEY_SERVER_URL, null) ?: DEFAULT_SERVER_URL)
  val serverUrl: StateFlow<String> = _serverUrl.asStateFlow()

  fun setServerUrl(url: String) {
    val clean = url.trim().trimEnd('/').ifEmpty { DEFAULT_SERVER_URL }
    prefs.edit { putString(KEY_SERVER_URL, clean) }
    _serverUrl.value = clean
  }

  companion object {
    const val DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
    private const val KEY_SERVER_URL = "server_url"
  }
}
