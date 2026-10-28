package com.ns.litepulse.measurement

import android.graphics.Bitmap
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.hypot

enum class PreCheckStatus(val message: String, val hint: String) {
    NO_FACE("Wajah tidak terdeteksi", "Hadapkan wajah ke kamera depan"),
    TOO_DARK("Terlalu gelap", "Cari tempat dengan cahaya lebih terang"),
    TOO_BRIGHT("Terlalu terang", "Hindari cahaya langsung yang menyilaukan"),
    NOT_FACING("Hadap kamera", "Luruskan wajah menghadap layar"),
    NOT_CENTERED("Posisikan wajah di tengah", "Geser wajah ke tengah layar"),
    TOO_FAR("Dekatkan wajah", "Wajah terlalu kecil di layar"),
    TOO_CLOSE("Mundur sedikit", "Seluruh wajah harus terlihat"),
    MOVING("Tahan posisi", "Jangan bergerak selama pengukuran"),
    HOLD("Tahan posisi", "Pengukuran dimulai sebentar lagi"),
    READY("Siap", "")
}

data class PreCheckResult(
    val status: PreCheckStatus,
    /** Mean landmark displacement since last frame, relative to face width. */
    val motion: Float,
    val largeMotion: Boolean
) {
    val passed: Boolean get() = status == PreCheckStatus.READY
}

class PreCheckEvaluator {

    companion object {
        // All values are starting points: tune them with real recordings.
        private const val MIN_FACE_WIDTH = 0.35f      // fraction of frame width
        private const val MAX_FACE_WIDTH = 0.90f
        private const val CENTER_TOLERANCE_X = 0.20f
        private const val CENTER_TOLERANCE_Y = 0.25f
        private const val YAW_TOLERANCE = 0.15f       // nose position between eye corners
        private const val DARK_LUMA = 60f
        private const val BRIGHT_LUMA = 235f
        private const val STILL_MOTION = 0.015f       // smoothed, per frame
        private const val LARGE_MOTION = 0.05f        // single frame -> restart window

        private const val RIGHT_EYE_OUTER = 33
        private const val LEFT_EYE_OUTER = 263
        private const val NOSE_TIP = 1
    }

    private var previous: FloatArray? = null
    private var smoothedMotion = 0f

    fun reset() {
        previous = null
        smoothedMotion = 0f
    }

    fun evaluate(landmarks: List<NormalizedLandmark>, bitmap: Bitmap): PreCheckResult {

        val n = landmarks.size
        val current = FloatArray(n * 2)

        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE

        for (i in 0 until n) {
            val x = landmarks[i].x()
            val y = landmarks[i].y()
            current[i * 2] = x
            current[i * 2 + 1] = y
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }

        val faceWidth = maxX - minX
        val centerX = (minX + maxX) / 2f
        val centerY = (minY + maxY) / 2f

        // ---- motion between frames (landmark displacement) ----
        var motion = 0f
        val prev = previous
        if (prev != null && prev.size == current.size && faceWidth > 0f) {
            var sum = 0f
            for (i in 0 until n) {
                sum += hypot(
                    current[i * 2] - prev[i * 2],
                    current[i * 2 + 1] - prev[i * 2 + 1]
                )
            }
            motion = sum / n / faceWidth
        }
        previous = current
        smoothedMotion = 0.5f * smoothedMotion + 0.5f * motion
        val largeMotion = motion > LARGE_MOTION

        // ---- head orientation (nose between the outer eye corners) ----
        val yaw = if (n > LEFT_EYE_OUTER) {
            val a = landmarks[RIGHT_EYE_OUTER].x()
            val b = landmarks[LEFT_EYE_OUTER].x()
            val span = b - a
            if (abs(span) > 1e-4f) (landmarks[NOSE_TIP].x() - a) / span else 0.5f
        } else {
            0.5f
        }

        val status = when {
            faceWidth < MIN_FACE_WIDTH ->
                PreCheckStatus.TOO_FAR

            faceWidth > MAX_FACE_WIDTH ||
                minX < 0.01f || maxX > 0.99f || minY < 0.01f || maxY > 0.99f ->
                PreCheckStatus.TOO_CLOSE

            abs(centerX - 0.5f) > CENTER_TOLERANCE_X ||
                abs(centerY - 0.5f) > CENTER_TOLERANCE_Y ->
                PreCheckStatus.NOT_CENTERED

            abs(yaw - 0.5f) > YAW_TOLERANCE ->
                PreCheckStatus.NOT_FACING

            else -> {
                val luma = faceBrightness(bitmap, minX, minY, faceWidth, maxY - minY)
                when {
                    luma < DARK_LUMA -> PreCheckStatus.TOO_DARK
                    luma > BRIGHT_LUMA -> PreCheckStatus.TOO_BRIGHT
                    largeMotion || smoothedMotion > STILL_MOTION -> PreCheckStatus.MOVING
                    else -> PreCheckStatus.READY
                }
            }
        }

        return PreCheckResult(status, motion, largeMotion)
    }

    /** Mean luma (0..255) on a 10x10 grid in the central part of the face box. */
    private fun faceBrightness(
        bitmap: Bitmap,
        minX: Float,
        minY: Float,
        width: Float,
        height: Float
    ): Float {
        val grid = 10
        var sum = 0f

        for (iy in 0 until grid) {
            for (ix in 0 until grid) {
                val nx = minX + width * (0.2f + 0.6f * (ix + 0.5f) / grid)
                val ny = minY + height * (0.2f + 0.6f * (iy + 0.5f) / grid)

                val px = (nx * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
                val py = (ny * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)

                val p = bitmap.getPixel(px, py)
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                sum += 0.299f * r + 0.587f * g + 0.114f * b
            }
        }
        return sum / (grid * grid)
    }
}