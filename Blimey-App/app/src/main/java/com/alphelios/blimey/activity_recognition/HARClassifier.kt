package com.alphelios.blimey.activity_recognition

import android.content.Context
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class HARClassifier(context: Context) {
    companion object {
        private const val MODEL_FILE = "file:///android_asset/frozen_HAR.pb"
        private const val INPUT_NODE = "LSTM_1_input"
        private val OUTPUT_NODES = arrayOf("Dense_2/Softmax")
        private const val OUTPUT_NODE = "Dense_2/Softmax"
        private val INPUT_SIZE = longArrayOf(1, 100, 12)
        private const val OUTPUT_SIZE = 7

        init {
            System.loadLibrary("tensorflow_inference")
        }
    }

    private val inferenceInterface: TensorFlowInferenceInterface
    fun predictProbabilities(data: FloatArray?): FloatArray {
        val result = FloatArray(OUTPUT_SIZE)
        inferenceInterface.feed(INPUT_NODE, data, *INPUT_SIZE)
        inferenceInterface.run(OUTPUT_NODES)
        inferenceInterface.fetch(OUTPUT_NODE, result)

        //Biking   Downstairs	 Jogging	  Sitting	Standing	Upstairs	Walking
        return result
    }

    init {
        inferenceInterface = TensorFlowInferenceInterface(context.assets, MODEL_FILE)
    }
}