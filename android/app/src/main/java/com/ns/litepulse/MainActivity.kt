package com.ns.litepulse

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.ns.litepulse.camera.CameraScreen
import com.ns.litepulse.ui.theme.LitePulseTheme
import com.ns.litepulse.model.LitePulseHeartRateCalculator

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        val heartRateCalculator =
            LitePulseHeartRateCalculator()

        val testSignal =
            FloatArray(180) { index ->
                kotlin.math.sin(
                    2.0 *
                        Math.PI *
                        1.2 *
                        index /
                        30.0
                ).toFloat()
            }

        val testHeartRate =
            heartRateCalculator.calculateHeartRate(
                testSignal
            )

        Log.d(
            "LitePulseHR",
            "Synthetic test BPM = $testHeartRate"
        )

        setContent {
            LitePulseTheme {
                CameraScreen()
            }
        }
    }
}