package com.alphelios.blimey.activity_recognition

import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.res.ResourcesCompat
import com.alphelios.blimey.DetectorActivity
import com.alphelios.blimey.R
import com.alphelios.blimey.databinding.ActivityMainBinding
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.math.BigDecimal
import java.util.*

class MainActivity : AppCompatActivity(), SensorEventListener, OnInitListener {

    private lateinit var binding: ActivityMainBinding
    private var mSensorManager: SensorManager? = null
    private var mAccelerometer: Sensor? = null
    private var mGyroscope: Sensor? = null
    private var mLinearAcceleration: Sensor? = null
    private var textToSpeech: TextToSpeech? = null
    private var results: FloatArray? = null
    private var classifier: HARClassifier? = null
    private val labels = arrayOf("Biking", "Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        overridePendingTransition(R.anim.slide_in_up, R.anim.slide_out_up)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.bottomnavview.itemIconTintList = null
        binding.bottomnavview.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener)

        ax = ArrayList()
        ay = ArrayList()
        az = ArrayList()
        lx = ArrayList()
        ly = ArrayList()
        lz = ArrayList()
        gx = ArrayList()
        gy = ArrayList()
        gz = ArrayList()
        ma = ArrayList()
        ml = ArrayList()
        mg = ArrayList()

        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        mAccelerometer = mSensorManager!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        mSensorManager!!.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST)
        mLinearAcceleration = mSensorManager!!.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        mSensorManager!!.registerListener(this, mLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST)
        mGyroscope = mSensorManager!!.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        mSensorManager!!.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST)
        classifier = HARClassifier(applicationContext)
        textToSpeech = TextToSpeech(this, this)
        textToSpeech!!.language = Locale.US
    }

    override fun onInit(status: Int) {
        val timer = Timer()
        timer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                if (results == null || results!!.isEmpty()) {
                    return
                }
                var max = -1f
                var idx = -1
                for (i in results!!.indices) {
                    if (results!![i] > max) {
                        idx = i
                        max = results!![i]
                    }
                }
                if (max > 0.50 && idx != prevIdx) {
                    textToSpeech!!.speak(labels[idx], TextToSpeech.QUEUE_ADD, null,
                            Integer.toString(Random().nextInt()))
                    prevIdx = idx
                }
            }
        }, 1000, 3000)
    }

    override fun onResume() {
        super.onResume()
        sensorManager.registerListener(this, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST)
    }

    public override fun onDestroy() {
        if (textToSpeech != null) {
            textToSpeech!!.stop()
            textToSpeech!!.shutdown()
        }
        super.onDestroy()
    }

    override fun onSensorChanged(event: SensorEvent) {
        activityPrediction()
        val sensor = event.sensor
        if (sensor.type == Sensor.TYPE_ACCELEROMETER) {
            ax!!.add(event.values[0])
            ay!!.add(event.values[1])
            az!!.add(event.values[2])
        } else if (sensor.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            lx!!.add(event.values[0])
            ly!!.add(event.values[1])
            lz!!.add(event.values[2])
        } else if (sensor.type == Sensor.TYPE_GYROSCOPE) {
            gx!!.add(event.values[0])
            gy!!.add(event.values[1])
            gz!!.add(event.values[2])
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, i: Int) {}
    private fun activityPrediction() {
        val data: MutableList<Float> = ArrayList()
        if (ax!!.size >= N_SAMPLES && ay!!.size >= N_SAMPLES && az!!.size >= N_SAMPLES && lx!!.size >= N_SAMPLES && ly!!.size >= N_SAMPLES && lz!!.size >= N_SAMPLES && gx!!.size >= N_SAMPLES && gy!!.size >= N_SAMPLES && gz!!.size >= N_SAMPLES) {
            var maValue: Double
            var mgValue: Double
            var mlValue: Double
            for (i in 0 until N_SAMPLES) {
                maValue = Math.sqrt(Math.pow(ax!![i].toDouble(), 2.0) + Math.pow(ay!![i].toDouble(), 2.0) + Math.pow(az!![i].toDouble(), 2.0))
                mlValue = Math.sqrt(Math.pow(lx!![i].toDouble(), 2.0) + Math.pow(ly!![i].toDouble(), 2.0) + Math.pow(lz!![i].toDouble(), 2.0))
                mgValue = Math.sqrt(Math.pow(gx!![i].toDouble(), 2.0) + Math.pow(gy!![i].toDouble(), 2.0) + Math.pow(gz!![i].toDouble(), 2.0))
                ma!!.add(maValue.toFloat())
                ml!!.add(mlValue.toFloat())
                mg!!.add(mgValue.toFloat())
            }
            data.addAll(ax!!.subList(0, N_SAMPLES))
            data.addAll(ay!!.subList(0, N_SAMPLES))
            data.addAll(az!!.subList(0, N_SAMPLES))
            data.addAll(lx!!.subList(0, N_SAMPLES))
            data.addAll(ly!!.subList(0, N_SAMPLES))
            data.addAll(lz!!.subList(0, N_SAMPLES))
            data.addAll(gx!!.subList(0, N_SAMPLES))
            data.addAll(gy!!.subList(0, N_SAMPLES))
            data.addAll(gz!!.subList(0, N_SAMPLES))
            data.addAll(ma!!.subList(0, N_SAMPLES))
            data.addAll(ml!!.subList(0, N_SAMPLES))
            data.addAll(mg!!.subList(0, N_SAMPLES))
            results = classifier!!.predictProbabilities(toFloatArray(data))
            var max = -1f
            var idx = -1
            for (i in results!!.indices) {
                if (results!![i] > max) {
                    idx = i
                    max = results!![i]
                }
            }
            setProbabilities()
            setRowsColor(idx)
            ax!!.clear()
            ay!!.clear()
            az!!.clear()
            lx!!.clear()
            ly!!.clear()
            lz!!.clear()
            gx!!.clear()
            gy!!.clear()
            gz!!.clear()
            ma!!.clear()
            ml!!.clear()
            mg!!.clear()
        }
    }

    private fun setProbabilities() {
        binding.bikingTextView.text = round(results!![0], 2).toString()
        binding.downstairsTextView.text = round(results!![1], 2).toString()
        binding.joggingTextView.text = round(results!![2], 2).toString()
        binding.sittingTextView.text = round(results!![3], 2).toString()
        binding.standingTextView.text = round(results!![4], 2).toString()
        binding.upstairsTextView.text = round(results!![5], 2).toString()
        binding.walkingTextView.text = round(results!![6], 2).toString()
    }

    private fun setRowsColor(idx: Int) {
        binding.bikingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.joggingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.sittingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.standingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        binding.walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorTransparent, null))
        when (idx) {
            0 -> binding.bikingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            1 -> binding.downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            2 -> binding.joggingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            3 -> binding.sittingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            4 -> binding.standingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            5 -> binding.upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
            6 -> binding.walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(resources, R.color.colorBlue, null))
        }
    }

    private fun toFloatArray(list: List<Float>): FloatArray {
        var i = 0
        val array = FloatArray(list.size)
        for (f in list) {
            array[i++] = f ?: Float.NaN
        }
        return array
    }

    private val sensorManager: SensorManager get() = getSystemService(SENSOR_SERVICE) as SensorManager

    companion object {
        private const val N_SAMPLES = 100
        private var prevIdx = -1
        private var ax: MutableList<Float>? = null
        private var ay: MutableList<Float>? = null
        private var az: MutableList<Float>? = null
        private var lx: MutableList<Float>? = null
        private var ly: MutableList<Float>? = null
        private var lz: MutableList<Float>? = null
        private var gx: MutableList<Float>? = null
        private var gy: MutableList<Float>? = null
        private var gz: MutableList<Float>? = null
        private var ma: MutableList<Float>? = null
        private var ml: MutableList<Float>? = null
        private var mg: MutableList<Float>? = null
        private fun round(d: Float, decimalPlace: Int): Float {
            var bd = BigDecimal(d.toString())
            bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP)
            return bd.toFloat()
        }
    }

    override fun onStart() {
        super.onStart()
        overridePendingTransition(R.anim.slide_in_up, R.anim.slide_out_up)
    }

    private var mOnNavigationItemSelectedListener = BottomNavigationView.OnNavigationItemSelectedListener { menuItem ->
        when (menuItem.itemId) {
            R.id.profile -> {
                intent = Intent(applicationContext, DetectorActivity::class.java)
                startActivity(intent)
            }
        }
        false
    }
}