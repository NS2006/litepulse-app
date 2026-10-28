package com.ns.litepulse.face

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class FaceLandmarkerManager(
    context: Context
) {

    private val faceLandmarker: FaceLandmarker

    private var lastTimestampMs = -1L

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        // VIDEO mode: the face is detected once and then tracked between frames,
        // which is much cheaper than IMAGE mode (full detection on every frame).
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(
            context,
            options
        )
    }

    /**
     * @param timestampMs frame timestamp in milliseconds. VIDEO mode requires
     *                    strictly increasing values, so duplicates are nudged forward.
     */
    fun detect(bitmap: Bitmap, timestampMs: Long): FaceLandmarkerResult {
        val ts = if (timestampMs <= lastTimestampMs) lastTimestampMs + 1 else timestampMs
        lastTimestampMs = ts

        return faceLandmarker.detectForVideo(
            BitmapImageBuilder(bitmap).build(),
            ts
        )
    }

    fun close() {
        faceLandmarker.close()
    }
}