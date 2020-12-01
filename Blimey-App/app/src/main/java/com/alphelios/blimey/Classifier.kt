package com.alphelios.blimey

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * Generic interface for interacting with different recognition engines.
 */
interface Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    class Recognition(
            /**
             * A unique identifier for what has been recognized. Specific to the class, not the instance of
             * the object.
             */
            val id: String,
            /**
             * Display name for the recognition.
             */
            val title: String,
            /**
             * A sortable score for how good the recognition is relative to others. Higher should be better.
             */
            val confidence: Float,
            /**
             * Optional location within the source image for the location of the recognized object.
             */
            @JvmField
            var location: RectF) {

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            resultString += "[$id] "
            resultString += "$title "
            resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            resultString += location.toString() + " "
            return resultString.trim { it <= ' ' }
        }
    }

    fun recognizeImage(bitmap: Bitmap): List<Recognition>
    fun enableStatLogging(debug: Boolean)
    val statString: String
    fun close()
}