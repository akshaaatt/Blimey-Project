package com.alphelios.blimey

import android.Manifest
import android.app.Activity
import android.app.Fragment
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Camera
import android.hardware.Camera.PreviewCallback
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.*
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.util.Size
import android.view.KeyEvent
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.ActivityCompat
import com.alphelios.blimey.ImageUtils.convertYUV420SPToARGB8888
import com.alphelios.blimey.ImageUtils.convertYUV420ToARGB8888
import com.alphelios.blimey.OverlayView.DrawCallback
import com.alphelios.blimey.activity_recognition.MainActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.util.*

abstract class CameraActivity : Activity(), OnImageAvailableListener, PreviewCallback, OnInitListener {
    var isDebug = false
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var useCamera2API = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    protected var luminanceStride = 0
    lateinit var tts: TextToSpeech
    @JvmField
    protected var previewWidth = 0
    @JvmField
    protected var previewHeight = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private lateinit var bottomnavview: BottomNavigationView

    val permissionsRequest: Int = 931
    val permissionsToGive = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(null)
        overridePendingTransition(R.anim.slide_in_up, R.anim.slide_out_up)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_camera)

        bottomnavview = findViewById(R.id.bottomnavview)
        bottomnavview.itemIconTintList = null
        bottomnavview.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener)

        tts = TextToSpeech(this, this)
        tts.language = Locale.US

        if(!askForPermissions(this, permissionsToGive, permissionsRequest)) {
            setFragment()
        }
    }

    fun askForPermissions(activity: Activity, permissions: Array<String>, requestCode: Int): Boolean {
        val permissionsToRequest: MutableList<String> = ArrayList()
        for (permission in permissions) {
            if (ActivityCompat.checkSelfPermission(activity, permission) != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(permission)
            }
        }
        if (permissionsToRequest.isEmpty()) {
            return false
        }
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(activity, permissionsToRequest.toTypedArray(), requestCode)
        }
        return true
    }


    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(!askForPermissions(this, permissionsToGive, permissionsRequest)) {
            setFragment()
        }
    }

    private lateinit var lastPreviewFrame: ByteArray
    protected fun getRgbBytes(): IntArray? {
        imageConverter!!.run()
        return rgbBytes
    }

    protected val luminance: ByteArray? get() = yuvBytes[0]

    /**
     * Callback for android.hardware.Camera API
     */
    override fun onPreviewFrame(bytes: ByteArray, camera: Camera) {
        if (isProcessingFrame) {
            return
        }
        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                val previewSize = camera.parameters.previewSize
                previewHeight = previewSize.height
                previewWidth = previewSize.width
                rgbBytes = IntArray(previewWidth * previewHeight)
                onPreviewSizeChosen(Size(previewSize.width, previewSize.height), 90)
            }
        } catch (e: Exception) {
            return
        }
        isProcessingFrame = true
        lastPreviewFrame = bytes
        yuvBytes[0] = bytes
        luminanceStride = previewWidth
        imageConverter = Runnable { convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes!!) }
        postInferenceCallback = Runnable {
            camera.addCallbackBuffer(bytes)
            isProcessingFrame = false
        }
        processImage()
    }

    /**
     * Callback for Camera2 API
     */
    override fun onImageAvailable(reader: ImageReader) {
        //We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return
            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            Trace.beginSection("imageAvailable")
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = Runnable {
                convertYUV420ToARGB8888(
                        yuvBytes[0]!!,
                        yuvBytes[1]!!,
                        yuvBytes[2]!!,
                        previewWidth,
                        previewHeight,
                        luminanceStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes!!)
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }
            processImage()
        } catch (e: Exception) {
            Trace.endSection()
            return
        }
        Trace.endSection()
    }

    @Synchronized
    public override fun onStart() {
        super.onStart()
        overridePendingTransition(R.anim.slide_in_up, R.anim.slide_out_up)
    }

    @Synchronized
    public override fun onResume() {
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    @Synchronized
    public override fun onPause() {
        if (!isFinishing) {
            finish()
        }
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
        }
        super.onPause()
    }

    @Synchronized
    public override fun onStop() {
        super.onStop()
    }

    @Synchronized
    public override fun onDestroy() {
        tts!!.shutdown()
        super.onDestroy()
    }

    @Synchronized
    protected fun runInBackground(r: Runnable?) {
        if (handler != null) {
            handler!!.post(r!!)
        }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(this@CameraActivity,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show()
            }
            requestPermissions(arrayOf(PERMISSION_CAMERA, PERMISSION_STORAGE), PERMISSIONS_REQUEST)
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
            characteristics: CameraCharacteristics, requiredLevel: Int): Boolean {
        val deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)!!
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
    }

    private fun chooseCamera(): String? {
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }
                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        ?: continue

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL
                        || isHardwareLevelSupported(characteristics,
                        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL))
                return cameraId
            }
        } catch (e: CameraAccessException) {
        }
        return null
    }

    private fun setFragment() {
        val cameraId = chooseCamera()
        if (cameraId == null) {
            Toast.makeText(this, "No Camera Detected", Toast.LENGTH_SHORT).show()
            finish()
        }
        val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment = CameraConnectionFragment.newInstance(
                    { size, rotation ->
                        previewHeight = size.height
                        previewWidth = size.width
                        onPreviewSizeChosen(size, rotation)
                    },
                    this,
                    layoutId,
                    desiredPreviewFrameSize)
            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment = LegacyCameraConnectionFragment(this, layoutId, desiredPreviewFrameSize)
        }
        fragmentManager
                .beginTransaction()
                .replace(R.id.container, fragment)
                .commit()
    }

    private fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]!!]
        }
    }

    fun requestRender() {
        val overlay = findViewById<View>(R.id.debug_overlay) as OverlayView
        overlay.postInvalidate()
    }

    fun addCallback(callback: DrawCallback?) {
        val overlay = findViewById<View>(R.id.debug_overlay) as OverlayView
        overlay.addCallback(callback!!)
    }

    open fun onSetDebug(debug: Boolean) {}
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_BUTTON_L1 || keyCode == KeyEvent.KEYCODE_DPAD_CENTER) {
            isDebug = !isDebug
            requestRender()
            onSetDebug(isDebug)
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    protected fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }

    protected val screenOrientation: Int
        get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }

    protected abstract fun processImage()
    protected abstract fun onPreviewSizeChosen(size: Size?, rotation: Int)
    protected abstract val layoutId: Int
    protected abstract val desiredPreviewFrameSize: Size?
    override fun onInit(status: Int) {}

    companion object {
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private const val PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE
        @JvmField
        var distance = 25
    }

    private var mOnNavigationItemSelectedListener = BottomNavigationView.OnNavigationItemSelectedListener { menuItem ->
        when (menuItem.itemId) {
            R.id.home -> {
                intent = Intent(applicationContext, MainActivity::class.java)
                startActivity(intent)
            }
        }
        false
    }
}