package com.hackathon.defectcam

import android.content.Context
import android.os.Build
import android.provider.Settings

object Config {
    // Shown as the default in the in-app settings bar; editable per phone without a rebuild.
    const val DEFAULT_BACKEND_URL = "http://192.168.1.100:5000/upload"

    // Minimum time between uploaded frames. Lower = more frames sent, more backend load.
    const val UPLOAD_INTERVAL_MS = 700L

    // JPEG compression quality for uploaded frames (0-100).
    const val JPEG_QUALITY = 80

    /** Stable-ish per-phone label (e.g. "Pixel_7-a1b2") so the server/dashboard can tell phones apart. */
    fun defaultDeviceId(context: Context): String {
        val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
            ?: "unknown"
        return "${Build.MODEL}-${androidId.takeLast(4)}".replace(" ", "_")
    }
}
