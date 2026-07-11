package com.hackathon.defectcam

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.Surface
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.hackathon.defectcam.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Camera capture screen, meant to be installed on multiple phones at once.
 * Each phone streams JPEG frames tagged with its own device_id to a shared
 * ML backend; this app never reads or renders the backend's response or any
 * ML result — it is a one-way camera-to-backend pipe, fanning in to one server.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var settings: SettingsStore
    private val uploader = FrameUploader()
    private val lastUploadAtMs = AtomicLong(0L)
    private val streamingEnabled = AtomicBoolean(true)

    @Volatile private var backendUrl: String = Config.DEFAULT_BACKEND_URL
    @Volatile private var deviceId: String = "unknown"

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                binding.statusText.setText(R.string.status_permission_denied)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        settings = SettingsStore(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        backendUrl = settings.backendUrl
        deviceId = settings.deviceId(this)
        binding.serverUrlInput.setText(backendUrl)
        binding.deviceIdInput.setText(deviceId)

        binding.applyButton.setOnClickListener { applySettings() }
        binding.streamToggle.isChecked = true
        binding.streamToggle.setOnCheckedChangeListener { _, isChecked ->
            streamingEnabled.set(isChecked)
            binding.statusText.setText(
                if (isChecked) R.string.status_streaming else R.string.status_paused
            )
        }

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun applySettings() {
        val newUrl = binding.serverUrlInput.text.toString().trim()
        val newDeviceId = binding.deviceIdInput.text.toString().trim()

        if (newUrl.isNotEmpty()) {
            backendUrl = newUrl
            settings.backendUrl = newUrl
        }
        if (newDeviceId.isNotEmpty()) {
            deviceId = newDeviceId
            settings.setDeviceId(newDeviceId)
        }
        binding.statusText.text = "Applied: $deviceId -> $backendUrl"
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // No Preview use case is bound on purpose — the screen stays black,
            // only ImageAnalysis runs so frames are still captured and uploaded.
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetRotation(Surface.ROTATION_0)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        maybeUploadFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageAnalysis
                )
                runOnUiThread { binding.statusText.setText(R.string.status_streaming) }
            } catch (exc: Exception) {
                runOnUiThread { binding.statusText.text = "Camera error: ${exc.message}" }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Throttles frames to Config.UPLOAD_INTERVAL_MS and ships each as a JPEG POST. */
    private fun maybeUploadFrame(imageProxy: ImageProxy) {
        if (!streamingEnabled.get()) {
            imageProxy.close()
            return
        }

        val now = System.currentTimeMillis()
        val last = lastUploadAtMs.get()

        if (now - last < Config.UPLOAD_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        if (!lastUploadAtMs.compareAndSet(last, now)) {
            imageProxy.close()
            return
        }

        try {
            val jpegBytes = imageProxy.toJpegByteArray(Config.JPEG_QUALITY)
            uploader.upload(backendUrl, deviceId, jpegBytes) { success ->
                runOnUiThread {
                    if (streamingEnabled.get()) {
                        binding.statusText.text = if (success) {
                            getString(R.string.status_streaming)
                        } else {
                            "Upload failed – check backend URL"
                        }
                    }
                }
            }
        } finally {
            imageProxy.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
