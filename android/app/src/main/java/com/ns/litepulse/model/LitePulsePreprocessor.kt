package com.ns.litepulse.model

import kotlin.math.sqrt

/** Immutable copy of the buffered window; safe to process on another thread. */
class WindowSnapshot(
    val frames: List<FloatArray>,
    val times: List<Double>        // seconds
)

class PreparedInput(
    /** [180, 18, 36, 36] flattened. */
    val input: FloatArray,
    /** Mean green per ROI of the resampled window: [3][181]. */
    val roiGreen: Array<FloatArray>
)

/**
 * Buffers timestamped ROI frames. Once ~6 s are collected, [snapshot] + [build]
 * resample them to a uniform 30 fps grid (181 frames) and build the model input.
 *
 * The model and the BPM formula assume 30 fps. A camera running slower must be
 * resampled, otherwise the pulse frequency is scaled by (real fps / 30).
 */
class LitePulsePreprocessor {

    companion object {
        const val TEMPORAL_FRAMES = 180
        const val REQUIRED_RAW_FRAMES = 181

        const val PATCH_SIZE = 36
        const val NUM_ROIS = 3
        const val RGB_CHANNELS = 3
        const val INPUT_CHANNELS = 18

        const val TARGET_FPS = 30.0
        const val WINDOW_SECONDS = TEMPORAL_FRAMES / TARGET_FPS      // 6.0 s
        private const val MAX_BUFFER_SECONDS = 7.5
        /** ~6 fps over 6 s. Below this the signal cannot be reconstructed. */
        private const val MIN_RAW_FRAMES = 36

        private const val PLANE = PATCH_SIZE * PATCH_SIZE
        private const val PIXELS_PER_FRAME = NUM_ROIS * RGB_CHANNELS * PLANE

        /** Pure function: safe to call from a background thread. */
        fun build(snapshot: WindowSnapshot): PreparedInput {
            val green = Array(NUM_ROIS) { FloatArray(REQUIRED_RAW_FRAMES) }
            val resampled = resample(snapshot, green)
            return PreparedInput(normalize(resampled), green)
        }

        private fun resample(
            s: WindowSnapshot,
            green: Array<FloatArray>
        ): Array<FloatArray> {

            val times = s.times
            val tEnd = times.last()
            val tStart = tEnd - WINDOW_SECONDS

            val out = Array(REQUIRED_RAW_FRAMES) { FloatArray(PIXELS_PER_FRAME) }

            var j = 0
            for (k in 0 until REQUIRED_RAW_FRAMES) {
                val t = tStart + k / TARGET_FPS

                while (j < times.size - 2 && times[j + 1] < t) j++

                val t0 = times[j]
                val t1 = times[j + 1]
                val w = ((t - t0) / (t1 - t0)).coerceIn(0.0, 1.0).toFloat()

                val a = s.frames[j]
                val b = s.frames[j + 1]
                val dst = out[k]

                for (i in 0 until PIXELS_PER_FRAME) {
                    dst[i] = a[i] + (b[i] - a[i]) * w
                }

                for (r in 0 until NUM_ROIS) {
                    val base = r * RGB_CHANNELS * PLANE + PLANE      // green plane
                    var sum = 0.0
                    var n = 0
                    for (p in 0 until PLANE) {
                        val v = dst[base + p]
                        if (v != 0f) {
                            sum += v
                            n++
                        }
                    }
                    green[r][k] = if (n > 0) (sum / n).toFloat() else 0f
                }
            }
            return out
        }

        // Python equivalent:
        //   d = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + 1e-7)
        //   d = d / (d.std() + 1e-7)
        //   s = (x[:-1] - x.mean()) / (x.std() + 1e-7)
        //   concatenate([d, s], axis=1)    # [180, 18, 36, 36]
        private fun normalize(frames: Array<FloatArray>): FloatArray {

            val totalValues = REQUIRED_RAW_FRAMES.toLong() * PIXELS_PER_FRAME

            var sum = 0.0
            for (frame in frames) for (v in frame) sum += v.toDouble()
            val mean = sum / totalValues

            var squared = 0.0
            for (frame in frames) {
                for (v in frame) {
                    val d = v.toDouble() - mean
                    squared += d * d
                }
            }
            val std = sqrt(squared / totalValues)

            val diff = FloatArray(TEMPORAL_FRAMES * PIXELS_PER_FRAME)
            var diffSum = 0.0

            for (t in 0 until TEMPORAL_FRAMES) {
                val current = frames[t]
                val next = frames[t + 1]
                val base = t * PIXELS_PER_FRAME
                for (i in 0 until PIXELS_PER_FRAME) {
                    val value = (next[i] - current[i]) / (next[i] + current[i] + 1e-7f)
                    diff[base + i] = value
                    diffSum += value.toDouble()
                }
            }

            val diffMean = diffSum / diff.size
            var diffSquared = 0.0
            for (value in diff) {
                val d = value.toDouble() - diffMean
                diffSquared += d * d
            }
            val diffStd = sqrt(diffSquared / diff.size)

            val diffScale = (diffStd + 1e-7).toFloat()
            val stdScale = (std + 1e-7).toFloat()
            val meanF = mean.toFloat()

            val output = FloatArray(TEMPORAL_FRAMES * INPUT_CHANNELS * PLANE)

            for (t in 0 until TEMPORAL_FRAMES) {
                val source = frames[t]
                val diffBase = t * PIXELS_PER_FRAME

                for (i in 0 until PIXELS_PER_FRAME) {
                    val channel = i / PLANE
                    val pixel = i % PLANE

                    output[(t * INPUT_CHANNELS + channel) * PLANE + pixel] =
                        diff[diffBase + i] / diffScale

                    output[(t * INPUT_CHANNELS + 9 + channel) * PLANE + pixel] =
                        (source[i] - meanF) / stdScale
                }
            }

            return output
        }
    }

    private val frames = ArrayList<FloatArray>()
    private val times = ArrayList<Double>()   // seconds

    fun push(frame: FloatArray, timestampNs: Long) {
        require(frame.size == PIXELS_PER_FRAME) {
            "Expected $PIXELS_PER_FRAME values, got ${frame.size}"
        }

        val t = timestampNs / 1e9
        if (times.isNotEmpty() && t <= times.last()) return

        frames.add(frame)
        times.add(t)

        while (times.size > 2 && times.last() - times.first() > MAX_BUFFER_SECONDS) {
            frames.removeAt(0)
            times.removeAt(0)
        }
    }

    fun clear() {
        frames.clear()
        times.clear()
    }

    fun size(): Int = frames.size

    fun elapsedSeconds(): Float =
        if (times.size < 2) 0f else (times.last() - times.first()).toFloat()

    fun measuredFps(): Float {
        if (times.size < 2) return 0f
        return ((times.size - 1) / (times.last() - times.first())).toFloat()
    }

    fun isReady(): Boolean =
        times.size >= MIN_RAW_FRAMES &&
            times.last() - times.first() >= WINDOW_SECONDS

    /** Cheap copy (frame arrays are never modified after push). Null if not ready. */
    fun snapshot(): WindowSnapshot? {
        if (!isReady()) return null
        return WindowSnapshot(ArrayList(frames), ArrayList(times))
    }
}