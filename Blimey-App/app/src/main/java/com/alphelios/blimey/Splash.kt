package com.alphelios.blimey

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import com.alphelios.blimey.activity_recognition.MainActivity

class Splash : AppCompatActivity() {

    override fun onStart() {
        super.onStart()
        startActivity(Intent(this,MainActivity::class.java))
        finish()
    }
}