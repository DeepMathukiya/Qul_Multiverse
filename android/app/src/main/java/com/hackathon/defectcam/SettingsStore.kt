package com.hackathon.defectcam

import android.content.Context

/** Persists the per-phone backend URL and device label across app restarts. */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("defectcam_settings", Context.MODE_PRIVATE)

    var backendUrl: String
        get() = prefs.getString(KEY_BACKEND_URL, Config.DEFAULT_BACKEND_URL) ?: Config.DEFAULT_BACKEND_URL
        set(value) = prefs.edit().putString(KEY_BACKEND_URL, value).apply()

    fun deviceId(context: Context): String =
        prefs.getString(KEY_DEVICE_ID, null) ?: Config.defaultDeviceId(context)

    fun setDeviceId(value: String) {
        prefs.edit().putString(KEY_DEVICE_ID, value).apply()
    }

    companion object {
        private const val KEY_BACKEND_URL = "backend_url"
        private const val KEY_DEVICE_ID = "device_id"
    }
}
