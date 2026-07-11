package com.hackathon.defectcam

import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Sends camera frames to the ML backend as multipart/form-data JPEG uploads,
 * tagged with a device_id so the server can tell multiple phones apart.
 * Fire-and-forget: this app only ships frames, it never reads or displays the response.
 */
class FrameUploader {

    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    fun upload(
        backendUrl: String,
        deviceId: String,
        jpegBytes: ByteArray,
        onResult: (success: Boolean) -> Unit
    ) {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("device_id", deviceId)
            .addFormDataPart("timestamp", System.currentTimeMillis().toString())
            .addFormDataPart(
                "frame",
                "${deviceId}_${System.currentTimeMillis()}.jpg",
                jpegBytes.toRequestBody("image/jpeg".toMediaType())
            )
            .build()

        val request = Request.Builder()
            .url(backendUrl)
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.w(TAG, "Frame upload failed: ${e.message}")
                onResult(false)
            }

            override fun onResponse(call: Call, response: okhttp3.Response) {
                response.close()
                onResult(response.isSuccessful)
            }
        })
    }

    companion object {
        private const val TAG = "FrameUploader"
    }
}
