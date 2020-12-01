package com.alphelios.blimey

import android.content.Intent
import android.graphics.*
import android.media.ImageReader.OnImageAvailableListener
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.Toast
import com.alphelios.blimey.Classifier.Recognition
import com.alphelios.blimey.ImageUtils.getTransformationMatrix
import com.alphelios.blimey.ImageUtils.saveBitmap
import com.alphelios.blimey.OverlayView.DrawCallback
import com.alphelios.blimey.activity_recognition.MainActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.io.IOException
import java.util.*

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(), OnImageAvailableListener {
    private var objects: ArrayList<String>? = null
    private var prevobject: ArrayList<String>? = null
    private var prevDistance = 0

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    private enum class DetectorMode {
        TF_OD_API, MULTIBOX, YOLO
    }

    private var sensorOrientation: Int? = null
    private var detector: Classifier? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null
    private var computingDetection = false
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var tracker: MultiBoxTracker? = null
    private var luminanceCopy: ByteArray? = null
    private var borderedText: BorderedText? = null
    private var moving = ""
    private var prevrecogs: MutableList<Recognition>? = null
    public override fun onPreviewSizeChosen(size: Size?, rotation: Int) {
        val textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        var cropSize = TF_OD_API_INPUT_SIZE
        when (MODE) {
            DetectorMode.YOLO -> {
                detector = TensorFlowYoloDetector.create(
                        assets,
                        YOLO_MODEL_FILE,
                        YOLO_INPUT_SIZE,
                        YOLO_INPUT_NAME,
                        YOLO_OUTPUT_NAMES,
                        YOLO_BLOCK_SIZE)
                cropSize = YOLO_INPUT_SIZE
            }
            DetectorMode.MULTIBOX -> {
                detector = TensorFlowMultiBoxDetector.create(
                        assets,
                        MB_MODEL_FILE,
                        MB_LOCATION_FILE,
                        MB_IMAGE_MEAN,
                        MB_IMAGE_STD,
                        MB_INPUT_NAME,
                        MB_OUTPUT_LOCATIONS_NAME,
                        MB_OUTPUT_SCORES_NAME)
                cropSize = MB_INPUT_SIZE
            }
            else -> {
                try {
                    detector = TensorFlowObjectDetectionAPIModel.create(
                            assets, TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE)
                    cropSize = TF_OD_API_INPUT_SIZE
                } catch (e: IOException) {
                    val toast = Toast.makeText(
                            applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT)
                    toast.show()
                    finish()
                }
            }
        }
        previewWidth = size!!.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        frameToCropTransform = getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation!!, MAINTAIN_ASPECT)
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback(
                object : DrawCallback {
                    override fun drawCallback(canvas: Canvas?) {
                        tracker!!.draw(canvas!!)
                        if (isDebug) {
                            tracker!!.drawDebug(canvas)
                        }
                    }
                })
        addCallback(
                object : DrawCallback {
                    override fun drawCallback(canvas: Canvas?) {
                        if (!isDebug) {
                            return
                        }
                        val copy = cropCopyBitmap ?: return
                        val backgroundColor = Color.argb(100, 0, 0, 0)
                        canvas!!.drawColor(backgroundColor)
                        val matrix = Matrix()
                        val scaleFactor = 2f
                        matrix.postScale(scaleFactor, scaleFactor)
                        matrix.postTranslate(
                                canvas.width - copy.width * scaleFactor,
                                canvas.height - copy.height * scaleFactor)
                        canvas.drawBitmap(copy, matrix, Paint())
                        val lines = Vector<String?>()
                        if (detector != null) {
                            val statString = detector!!.statString
                            val statLines = statString.split("\n").toTypedArray()
                            for (line in statLines) {
                                lines.add(line)
                            }
                        }
                        lines.add("")
                        lines.add("Frame: " + previewWidth + "x" + previewHeight)
                        lines.add("Crop: " + copy.width + "x" + copy.height)
                        lines.add("View: " + canvas.width + "x" + canvas.height)
                        lines.add("Rotation: $sensorOrientation")
                        lines.add("Inference time: " + lastProcessingTimeMs + "ms")
                        borderedText!!.drawLines(canvas, 10f, (canvas.height - 10).toFloat(), lines)
                    }
                })
    }

    var trackingOverlay: OverlayView? = null
    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        val originalLuminance = luminance!!
        tracker!!.onFrame(
                previewWidth,
                previewHeight,
                luminanceStride,
                sensorOrientation!!,
                originalLuminance,
                timestamp)
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        //        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
        rgbFrameBitmap!!.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)
        if (luminanceCopy == null) {
            luminanceCopy = ByteArray(originalLuminance.size)
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.size)
        readyForNextImage()
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            saveBitmap(croppedBitmap!!)
        }
        runInBackground(
                object : Runnable {
                    override fun run() {
                        val startTime = SystemClock.uptimeMillis()
                        val results = detector!!.recognizeImage(croppedBitmap!!)
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap!!)
                        val canvas = Canvas(cropCopyBitmap!!)
                        val paint = Paint()
                        paint.color = Color.RED
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = 2.0f
                        var minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API
                        minimumConfidence = when (MODE) {
                            DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
                            DetectorMode.MULTIBOX -> MINIMUM_CONFIDENCE_MULTIBOX
                            DetectorMode.YOLO -> MINIMUM_CONFIDENCE_YOLO
                        }
                        if (objects == null) {
                            objects = ArrayList()
                            prevobject = ArrayList()
                            prevrecogs = LinkedList()
                        }
                        objects!!.clear()
                        moving = ""
                        val mappedRecognitions: MutableList<Recognition> = LinkedList()
                        var objects_title = ""
                        for (result in results) {
                            val location = result.getLocation()
                            if (result.confidence >= minimumConfidence) {
                                canvas.drawRect(location, paint)
                                objects!!.add(result.title)
                                cropToFrameTransform!!.mapRect(location)
                                result.setLocation(location)
                                mappedRecognitions.add(result)
                                var temp: Recognition? = null
                                for (t in prevrecogs!!) {
                                    if (t.title === result.title) temp = t
                                }
                                if (temp != null) {
                                    moving = if (perimeter(location) < perimeter(temp.getLocation()) - 20) " moving away" else if (perimeter(location) > perimeter(temp.getLocation()) + 20) " moving closer" else break
                                    objects_title += result.title + moving
                                } else objects_title += result.title + " ahead"
                                objects_title += " "
                            }
                        }
                        prevDistance = distance
                        prevrecogs!!.clear()
                        prevrecogs!!.addAll(results)
                        tts.speak(objects_title, TextToSpeech.QUEUE_FLUSH, null)
                        try {
                            Thread.sleep(2000)
                        } catch (e: InterruptedException) {
                            Log.e(javaClass.simpleName, "run: ", e)
                        }
                        prevobject!!.clear()
                        prevobject!!.addAll(objects!!)
                        tracker!!.trackResults(mappedRecognitions, luminanceCopy!!, currTimestamp)
                        trackingOverlay!!.postInvalidate()
                        requestRender()
                        computingDetection = false
                    }

                    fun perimeter(rectF: RectF): Float {
                        val dist = rectF.top - rectF.bottom + (rectF.right - rectF.left)
                        return dist
                    }
                })
    }

    override val layoutId: Int get() = R.layout.camera_connection_fragment_tracking
    override val desiredPreviewFrameSize = Size(640, 480)

    override fun onSetDebug(debug: Boolean) {
        detector!!.enableStatLogging(debug)
    }

    companion object {

        // Configuration values for the prepackaged multibox model.
        private const val MB_INPUT_SIZE = 224
        private const val MB_IMAGE_MEAN = 128
        private const val MB_IMAGE_STD = 128f
        private const val MB_INPUT_NAME = "ResizeBilinear"
        private const val MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape"
        private const val MB_OUTPUT_SCORES_NAME = "output_scores/Reshape"
        private const val MB_MODEL_FILE = "file:///android_asset/multibox_model.pb"
        private const val MB_LOCATION_FILE = "file:///android_asset/multibox_location_priors.txt"
        private const val TF_OD_API_INPUT_SIZE = 300
        private const val TF_OD_API_MODEL_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt"

        // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
        // must be manually placed in the assets/ directory by the user.
        // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
        // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
        // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
        private const val YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb"
        private const val YOLO_INPUT_SIZE = 416
        private const val YOLO_INPUT_NAME = "input"
        private const val YOLO_OUTPUT_NAMES = "output"
        private const val YOLO_BLOCK_SIZE = 32
        private val MODE = DetectorMode.TF_OD_API

        // Minimum detection confidence to track a detection.
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.6f
        private const val MINIMUM_CONFIDENCE_MULTIBOX = 0.1f
        private const val MINIMUM_CONFIDENCE_YOLO = 0.25f
        private val MAINTAIN_ASPECT = MODE == DetectorMode.YOLO
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}