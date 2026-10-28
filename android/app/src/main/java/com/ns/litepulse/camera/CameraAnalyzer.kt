package com.ns.litepulse.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.ns.litepulse.face.FaceLandmarkerManager

/**
 * onResult(bitmap, landmarks, sensorTimestampNs)
 *
 * The timestamp lets the preprocessor resample to a true 30 fps, because
 * ImageAnalysis with KEEP_ONLY_LATEST drops frames when processing is slow.
 */
class CameraAnalyzer(
    private val faceLandmarkerManager: FaceLandmarkerManager,
    private val onResult: (Bitmap, FaceLandmarkerResult?, Long) -> Unit
) : ImageAnalysis.Analyzer {

    private var frames = 0
    private var bitmapMs = 0L
    private var detectMs = 0L
    private var callbackMs = 0L

    override fun analyze(imageProxy: ImageProxy) {
        try {
            val timestampNs = imageProxy.imageInfo.timestamp

            val t0 = SystemClock.elapsedRealtime()
            val rotated = rotateBitmap(
                imageProxy.toBitmap(),
                imageProxy.imageInfo.rotationDegrees
            )

            val t1 = SystemClock.elapsedRealtime()
            val result = faceLandmarkerManager.detect(rotated, timestampNs / 1_000_000L)

            val t2 = SystemClock.elapsedRealtime()
            onResult(rotated, result, timestampNs)

            val t3 = SystemClock.elapsedRealtime()

            bitmapMs += t1 - t0
            detectMs += t2 - t1
            callbackMs += t3 - t2

            if (++frames == 30) {
                Log.d(
                    "LitePulseTiming",
                    "avg per frame: bitmap ${bitmapMs / 30} ms | " +
                        "face detect ${detectMs / 30} ms | " +
                        "pipeline (ROI+precheck) ${callbackMs / 30} ms"
                )
                frames = 0
                bitmapMs = 0
                detectMs = 0
                callbackMs = 0
            }

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }

        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }
}