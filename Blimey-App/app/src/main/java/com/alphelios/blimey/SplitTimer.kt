package com.alphelios.blimey

import android.os.SystemClock

/**
 * A simple utility timer for measuring CPU time and wall-clock splits.
 */
class SplitTimer(name: String?) {
    private var lastWallTime: Long = 0
    private var lastCpuTime: Long = 0
    fun newSplit() {
        lastWallTime = SystemClock.uptimeMillis()
        lastCpuTime = SystemClock.currentThreadTimeMillis()
    }

    fun endSplit(splitName: String?) {
        val currWallTime = SystemClock.uptimeMillis()
        val currCpuTime = SystemClock.currentThreadTimeMillis()
        lastWallTime = currWallTime
        lastCpuTime = currCpuTime
    }

    init {
        newSplit()
    }
}