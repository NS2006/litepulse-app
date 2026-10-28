package com.ns.litepulse.measurement

import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.ns.litepulse.face.RoiExtractor
import com.ns.litepulse.model.HeartRateResult
import com.ns.litepulse.model.LitePulseHeartRateCalculator
import com.ns.litepulse.model.LitePulseModelManager
import com.ns.litepulse.model.LitePulsePreprocessor
import com.ns.litepulse.model.SignalQuality
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.pow

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

data class MeasurementUiState(
    val faceDetected: Boolean = false,
    val faceCount: Int = 0,
    val status: PreCheckStatus = PreCheckStatus.NO_FACE,

    val bufferedFrames: Int = 0,
    /** Seconds of video in the current window (goal: 6 s). */
    val windowSeconds: Float = 0f,
    val windowReady: Boolean = false,
    val fps: Float = 0f,

    val heartRate: HeartRateResult? = null,
    /** Median of recent non-poor windows, with min / max of those windows. */
    val bpmMedian: Float? = null,
    val bpmLow: Float? = null,
    val bpmHigh: Float? = null,
    /** Session HR trend, ~1 point per second. */
    val trend: List<Float> = emptyList(),

    /** [forehead, left cheek, right cheek], sums to 1. */
    val roiWeights: List<Float> = listOf(1f / 3, 1f / 3, 1f / 3),
    /** true = real APRF weights from the model, false = estimated from the signal. */
    val weightsFromModel: Boolean = false,
    val roiPatches: Map<String, Bitmap> = emptyMap(),

    val notice: String? = null,
    val debug: String = ""
) {
    companion object {
        const val REQUIRED_SECONDS = 6f
    }

    val isMeasuring: Boolean get() = bufferedFrames > 0
    val windowFull: Boolean get() = windowReady
    val progress: Float get() = (windowSeconds / REQUIRED_SECONDS).coerceIn(0f, 1f)
    val secondsRemaining: Int
        get() = ceil(REQUIRED_SECONDS - windowSeconds).toInt().coerceAtLeast(0)
}

// ---------------------------------------------------------------------------
// ROI frame (channel-major: [R,G,B] of forehead, left cheek, right cheek)
// ---------------------------------------------------------------------------

class RoiFrame(val data: FloatArray)

object RoiFrameBuilder {

    fun build(patches: Map<String, Bitmap>): RoiFrame? {
        val size = RoiExtractor.PATCH_SIZE
        val plane = size * size

        val data = FloatArray(RoiExtractor.ROI_ORDER.size * 3 * plane)
        val pixels = IntArray(plane)

        for ((roi, name) in RoiExtractor.ROI_ORDER.withIndex()) {
            val bitmap = patches[name] ?: return null
            bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

            val base = roi * 3 * plane
            for (i in 0 until plane) {
                val p = pixels[i]
                data[base + i] = ((p shr 16) and 0xFF).toFloat()
                data[base + plane + i] = ((p shr 8) and 0xFF).toFloat()
                data[base + 2 * plane + i] = (p and 0xFF).toFloat()
            }
        }
        return RoiFrame(data)
    }
}

// ---------------------------------------------------------------------------
// ROI weights
// ---------------------------------------------------------------------------

/**
 * Fallback when the model doesn't export APRF weights: scores each ROI by how
 * pulse-like its own green trace is. Real weights (from the model) go through smooth().
 */
class RoiWeightEstimator(
    private val calculator: LitePulseHeartRateCalculator
) {
    private val equal = listOf(1f / 3, 1f / 3, 1f / 3)

    var current: List<Float> = equal
        private set

    fun reset() {
        current = equal
    }

    /** Pure: safe on a background thread. */
    fun rawWeights(greenTraces: Array<FloatArray>): List<Float> {
        val scores = greenTraces.map {
            (calculator.roiPnr(it) - 1.0).coerceAtLeast(0.05).pow(2)
        }
        val total = scores.sum()
        return scores.map { (it / total).toFloat() }
    }

    fun smooth(raw: List<Float>): List<Float> {
        current = current.zip(raw) { old, new -> 0.5f * old + 0.5f * new }
        return current
    }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/**
 * Camera thread (process): face detection, pre-check, ROI extraction, buffering.
 * Inference thread: resample -> model -> HR. Running the model on the camera
 * thread blocked frame delivery and caused ~9 fps.
 */
class MeasurementController(
    private val roiExtractor: RoiExtractor,
    private val preprocessor: LitePulsePreprocessor,
    private val model: LitePulseModelManager,
    private val calculator: LitePulseHeartRateCalculator
) {
    companion object {
        private const val TAG = "LitePulse"
        private const val STABLE_FRAMES_TO_START = 15
        private const val FAIL_TOLERANCE = 5
        // The model hogs the CPU while it runs; fewer runs = more camera frames.
        private const val INFERENCE_INTERVAL_SEC = 2.0
        private const val NOTICE_CLEAR_FRAMES = 30
        private const val MEDIAN_WINDOWS = 5
        private const val MAX_TREND_POINTS = 300
        private const val WEAK_FPS = 15f
    }

    private val lock = Any()
    private val inferenceExecutor = Executors.newSingleThreadExecutor()

    private val evaluator = PreCheckEvaluator()
    private val weightEstimator = RoiWeightEstimator(calculator)

    private var generation = 0
    private var inferenceRunning = false

    private var failStreak = 0
    private var stableStreak = 0

    private var sessionStartNs = 0L
    private var lastInferenceSec = -10.0
    private var lastTrendSec = -10.0

    private var heartRate: HeartRateResult? = null
    private var weights: List<Float> = weightEstimator.current
    private var weightsFromModel = false
    private var notice: String? = null
    private var debug = ""
    private var patches: Map<String, Bitmap> = emptyMap()

    private val recent = ArrayDeque<Float>()
    private var bpmMedian: Float? = null
    private var bpmLow: Float? = null
    private var bpmHigh: Float? = null

    private val trend = ArrayList<Float>()
    private var trendView: List<Float> = emptyList()

    fun close() {
        inferenceExecutor.shutdownNow()
    }

    fun process(
        bitmap: Bitmap,
        result: FaceLandmarkerResult?,
        timestampNs: Long
    ): MeasurementUiState = synchronized(lock) {
        processLocked(bitmap, result, timestampNs)
    }

    private fun processLocked(
        bitmap: Bitmap,
        result: FaceLandmarkerResult?,
        timestampNs: Long
    ): MeasurementUiState {

        if (sessionStartNs == 0L) sessionStartNs = timestampNs
        val nowSec = (timestampNs - sessionStartNs) / 1e9

        val landmarks = result?.faceLandmarks()?.firstOrNull()

        if (landmarks == null) {
            restart("wajah hilang")
            evaluator.reset()
            failStreak = 0
            stableStreak = 0
            patches = emptyMap()
            return snapshot(faceDetected = false, status = PreCheckStatus.NO_FACE)
        }

        val check = evaluator.evaluate(landmarks, bitmap)
        val inWindow = preprocessor.size() > 0

        if (check.passed) {
            failStreak = 0
            stableStreak++
        } else {
            failStreak++
            stableStreak = 0
        }

        if (!check.passed &&
            (check.largeMotion || failStreak >= FAIL_TOLERANCE || !inWindow)
        ) {
            if (inWindow) {
                restart(
                    if (check.largeMotion) "gerakan terlalu besar"
                    else check.status.message.lowercase()
                )
            }
            return snapshot(faceDetected = true, status = check.status)
        }

        if (!inWindow && stableStreak < STABLE_FRAMES_TO_START) {
            return snapshot(faceDetected = true, status = PreCheckStatus.HOLD)
        }

        val newPatches = roiExtractor.extractRoiPatches(bitmap, landmarks)
        patches = newPatches

        val frame = RoiFrameBuilder.build(newPatches)
            ?: return snapshot(faceDetected = true, status = check.status)

        preprocessor.push(frame.data, timestampNs)

        if (preprocessor.size() >= NOTICE_CLEAR_FRAMES) notice = null

        if (!inferenceRunning &&
            preprocessor.isReady() &&
            nowSec - lastInferenceSec >= INFERENCE_INTERVAL_SEC
        ) {
            startInference(nowSec)
        }

        return snapshot(faceDetected = true, status = check.status)
    }

    // -----------------------------------------------------------------
    // Inference (background thread)
    // -----------------------------------------------------------------

    private fun startInference(nowSec: Double) {
        val window = preprocessor.snapshot() ?: return
        val fps = preprocessor.measuredFps()
        val gen = generation

        inferenceRunning = true
        lastInferenceSec = nowSec

        inferenceExecutor.execute {
            try {
                val prepared = LitePulsePreprocessor.build(window)
                val output = model.predict(prepared.input)
                val hr = calculator.calculate(output.pulse, output.energyRatio)

                val fromModel = output.roiWeights != null
                val rawWeights = output.roiWeights?.toList()
                    ?: weightEstimator.rawWeights(prepared.roiGreen)

                synchronized(lock) {
                    if (gen == generation) {
                        applyResult(hr, rawWeights, fromModel, fps, nowSec)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed", e)
            } finally {
                synchronized(lock) { inferenceRunning = false }
            }
        }
    }

    /** Called with [lock] held. */
    private fun applyResult(
        hr: HeartRateResult,
        rawWeights: List<Float>,
        fromModel: Boolean,
        fps: Float,
        nowSec: Double
    ) {
        heartRate = hr
        weightsFromModel = fromModel
        weights = weightEstimator.smooth(rawWeights)

        if (hr.quality != SignalQuality.POOR) {
            recent.addLast(hr.bpm)
            while (recent.size > MEDIAN_WINDOWS) recent.removeFirst()
        }
        updateMedian(nowSec)

        debug = String.format(
            "PNR %.1f | band %.2f | model %s | skor %.2f | %.0f fps | bobot %s",
            hr.peakToNoise,
            hr.bandEnergyRatio,
            hr.modelBandRatio?.let { String.format("%.2f", it) } ?: "-",
            hr.score,
            fps,
            if (fromModel) "model" else "estimasi"
        )
        Log.d(TAG, "bpm=${hr.bpm} q=${hr.quality} $debug")
    }

    private fun updateMedian(nowSec: Double) {
        if (recent.isEmpty()) {
            bpmMedian = null
            bpmLow = null
            bpmHigh = null
            return
        }

        val sorted = recent.sorted()
        val mid = sorted.size / 2
        val median =
            if (sorted.size % 2 == 1) sorted[mid]
            else (sorted[mid - 1] + sorted[mid]) / 2f

        bpmMedian = median
        if (sorted.size >= 3) {
            bpmLow = sorted.first()
            bpmHigh = sorted.last()
        } else {
            bpmLow = null
            bpmHigh = null
        }

        if (nowSec - lastTrendSec >= 1.0) {
            lastTrendSec = nowSec
            trend.add(median)
            if (trend.size > MAX_TREND_POINTS) trend.removeAt(0)
            trendView = trend.toList()
        }
    }

    private fun restart(reason: String) {
        if (preprocessor.size() > 0) {
            notice = "Pengukuran diulang: $reason"
        }
        generation++                    // discard any in-flight inference
        preprocessor.clear()
        weightEstimator.reset()
        weights = weightEstimator.current
        weightsFromModel = false
        heartRate = null
        recent.clear()
        bpmMedian = null
        bpmLow = null
        bpmHigh = null
        lastInferenceSec = -10.0
    }

    private fun snapshot(faceDetected: Boolean, status: PreCheckStatus): MeasurementUiState {
        val elapsed = preprocessor.elapsedSeconds()
        val ready = preprocessor.isReady()
        val fps = preprocessor.measuredFps()

        val fpsNotice = when {
            elapsed >= MeasurementUiState.REQUIRED_SECONDS && !ready ->
                String.format("FPS kamera terlalu rendah (%.0f), tidak bisa mengukur", fps)
            ready && fps < WEAK_FPS ->
                String.format("FPS rendah (%.0f), akurasi bisa menurun", fps)
            else -> null
        }

        return MeasurementUiState(
            faceDetected = faceDetected,
            faceCount = if (faceDetected) 1 else 0,
            status = status,
            bufferedFrames = preprocessor.size(),
            windowSeconds = elapsed,
            windowReady = ready,
            fps = fps,
            heartRate = heartRate,
            bpmMedian = bpmMedian,
            bpmLow = bpmLow,
            bpmHigh = bpmHigh,
            trend = trendView,
            roiWeights = weights,
            weightsFromModel = weightsFromModel,
            roiPatches = patches,
            notice = notice ?: fpsNotice,
            debug = debug
        )
    }
}