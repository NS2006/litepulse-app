package com.ns.litepulse.model

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

enum class SignalQuality(val label: String) {
    GOOD("Baik"),
    FAIR("Cukup"),
    POOR("Buruk")
}

data class HeartRateResult(
    val bpm: Float,
    /** Height of the HR peak relative to the mean of the 0.75-2.5 Hz band. */
    val peakToNoise: Double,
    /** Share of spectral energy that falls inside the 0.75-2.5 Hz band. */
    val bandEnergyRatio: Double,
    /** Combined score, 0..1. */
    val score: Double,
    val quality: SignalQuality,
    /** PulseSpectralAttention energy ratio exported by the model (null if not exported). */
    val modelBandRatio: Float? = null,
    /** cumsum + detrend + bandpass signal, 180 samples @ 30 fps. */
    val waveform: FloatArray = FloatArray(0),
    /** Indices of detected pulse peaks in [waveform]. */
    val peaks: List<Int> = emptyList()
)

class LitePulseHeartRateCalculator {

    companion object {
        private const val FS = 30.0
        private const val SIGNAL_LENGTH = 180

        private const val LOW_HZ = 0.75
        private const val HIGH_HZ = 2.5

        private const val DETREND_LAMBDA = 100.0

        private const val FFT_SIZE = 16384
        private const val ROI_FFT_SIZE = 4096
        private const val RATIO_FFT_SIZE = 2048
        private const val RATIO_MIN_HZ = 0.1

        // ---- Quality thresholds (still uncalibrated, tune with the on-screen debug line) ----
        // A *perfect* sinusoid in a 6 s window has PNR ~6-7, so real signals sit far lower.
        private const val PNR_POOR = 1.8
        private const val PNR_GOOD = 3.5
        private const val RATIO_POOR = 0.15
        private const val RATIO_GOOD = 0.55
        private const val MODEL_RATIO_POOR = 0.15
        private const val MODEL_RATIO_GOOD = 0.50
        private const val SCORE_GOOD = 0.55
        private const val SCORE_FAIR = 0.25

        // scipy.signal.butter(1, [0.75/15, 2.5/15], "bandpass")
        private val B = doubleArrayOf(0.15635952, 0.0, -0.15635952)
        private val A = doubleArrayOf(1.0, -1.61758769, 0.68728096)
        private val ZI = doubleArrayOf(-0.15635952, -0.15635952)
    }

    private val detrendMatrix = buildDetrendMatrix(SIGNAL_LENGTH, DETREND_LAMBDA)

    private class Peak(val frequency: Double, val pnr: Double)

    /** Backwards compatible: BPM only. */
    fun calculateHeartRate(prediction: FloatArray): Float =
        calculate(prediction).bpm

    fun calculate(prediction: FloatArray, modelBandRatio: Float? = null): HeartRateResult {
        require(prediction.size == SIGNAL_LENGTH) {
            "Expected $SIGNAL_LENGTH values, got ${prediction.size}"
        }

        val detrended = matrixVectorMultiply(detrendMatrix, cumulativeSum(prediction))
        val filtered = filtfilt(B, A, detrended)

        val peak = findPeak(magnitudeSpectrum(filtered, FFT_SIZE), FFT_SIZE)
        val ratio = bandEnergyRatio(detrended)

        val pnrScore = ((peak.pnr - PNR_POOR) / (PNR_GOOD - PNR_POOR)).coerceIn(0.0, 1.0)
        val ratioScores = mutableListOf(
            ((ratio - RATIO_POOR) / (RATIO_GOOD - RATIO_POOR)).coerceIn(0.0, 1.0)
        )
        if (modelBandRatio != null) {
            ratioScores += ((modelBandRatio - MODEL_RATIO_POOR) /
                (MODEL_RATIO_GOOD - MODEL_RATIO_POOR)).coerceIn(0.0, 1.0)
        }
        val score = 0.6 * pnrScore + 0.4 * ratioScores.average()

        val quality = when {
            score >= SCORE_GOOD -> SignalQuality.GOOD
            score >= SCORE_FAIR -> SignalQuality.FAIR
            else -> SignalQuality.POOR
        }

        return HeartRateResult(
            bpm = (peak.frequency * 60.0).toFloat(),
            peakToNoise = peak.pnr,
            bandEnergyRatio = ratio,
            score = score,
            quality = quality,
            modelBandRatio = modelBandRatio,
            waveform = FloatArray(filtered.size) { filtered[it].toFloat() },
            peaks = detectPeaks(filtered)
        )
    }

    /** Local maxima above zero, at least 0.33 s apart (max 180 bpm). */
    private fun detectPeaks(signal: DoubleArray): List<Int> {
        val minDistance = (FS * 0.33).toInt()
        val peaks = ArrayList<Int>()

        for (i in 2 until signal.size - 2) {
            val v = signal[i]
            if (v <= 0.0 || v <= signal[i - 1] || v < signal[i + 1]) continue

            val last = peaks.lastOrNull()
            when {
                last == null || i - last >= minDistance -> peaks.add(i)
                v > signal[last] -> peaks[peaks.lastIndex] = i
            }
        }
        return peaks
    }

    /**
     * Peak-to-noise of a raw ROI intensity trace (>= 180 samples).
     * Used to estimate how pulsatile each ROI is on its own.
     */
    fun roiPnr(trace: FloatArray): Double {
        if (trace.size < SIGNAL_LENGTH) return 1.0
        val offset = trace.size - SIGNAL_LENGTH
        val x = DoubleArray(SIGNAL_LENGTH) { trace[offset + it].toDouble() }
        val filtered = filtfilt(B, A, matrixVectorMultiply(detrendMatrix, x))
        return findPeak(magnitudeSpectrum(filtered, ROI_FFT_SIZE), ROI_FFT_SIZE).pnr
    }

    private fun findPeak(spectrum: DoubleArray, fftSize: Int): Peak {
        val resolution = FS / fftSize
        val minBin = ceil(LOW_HZ / resolution).toInt()
        val maxBin = floor(HIGH_HZ / resolution).toInt()

        var bestBin = minBin
        var best = Double.NEGATIVE_INFINITY
        var sum = 0.0

        for (bin in minBin..maxBin) {
            val m = spectrum[bin]
            sum += m
            if (m > best) {
                best = m
                bestBin = bin
            }
        }

        val mean = sum / (maxBin - minBin + 1)
        return Peak(bestBin * resolution, best / (mean + 1e-12))
    }

    private fun bandEnergyRatio(signal: DoubleArray): Double {
        val spectrum = magnitudeSpectrum(signal, RATIO_FFT_SIZE)
        val resolution = FS / RATIO_FFT_SIZE
        val firstBin = ceil(RATIO_MIN_HZ / resolution).toInt()
        val bandLo = ceil(LOW_HZ / resolution).toInt()
        val bandHi = floor(HIGH_HZ / resolution).toInt()

        var total = 0.0
        var band = 0.0
        for (bin in firstBin until spectrum.size) {
            val p = spectrum[bin] * spectrum[bin]
            total += p
            if (bin in bandLo..bandHi) band += p
        }
        return if (total > 0.0) band / total else 0.0
    }

    private fun cumulativeSum(input: FloatArray): DoubleArray {
        val output = DoubleArray(input.size)
        var sum = 0.0
        for (i in input.indices) {
            sum += input[i].toDouble()
            output[i] = sum
        }
        return output
    }

    /** DETREND = I - inv(I + lambda^2 * D^T D) */
    private fun buildDetrendMatrix(size: Int, lambda: Double): Array<DoubleArray> {
        val dtD = Array(size) { DoubleArray(size) }
        val kernel = doubleArrayOf(1.0, -2.0, 1.0)
        for (i in 0 until size - 2) {
            for (r in 0 until 3) {
                for (c in 0 until 3) {
                    dtD[i + r][i + c] += kernel[r] * kernel[c]
                }
            }
        }

        val matrix = Array(size) { row ->
            DoubleArray(size) { column ->
                (if (row == column) 1.0 else 0.0) + lambda * lambda * dtD[row][column]
            }
        }

        val inverse = invertMatrix(matrix)
        return Array(size) { row ->
            DoubleArray(size) { column ->
                (if (row == column) 1.0 else 0.0) - inverse[row][column]
            }
        }
    }

    private fun matrixVectorMultiply(matrix: Array<DoubleArray>, vector: DoubleArray): DoubleArray {
        val output = DoubleArray(vector.size)
        for (row in matrix.indices) {
            var sum = 0.0
            for (column in vector.indices) sum += matrix[row][column] * vector[column]
            output[row] = sum
        }
        return output
    }

    /** scipy.signal.filtfilt (pad="odd", padlen = 3 * max(len(a), len(b))). */
    private fun filtfilt(b: DoubleArray, a: DoubleArray, input: DoubleArray): DoubleArray {
        val padLength = 3 * maxOf(a.size, b.size)
        require(input.size > padLength) { "Input must contain more than $padLength samples" }

        val padded = oddPad(input, padLength)

        val forward = filterWithInitialState(
            b, a, padded,
            doubleArrayOf(ZI[0] * padded.first(), ZI[1] * padded.first())
        )
        forward.reverse()

        val backward = filterWithInitialState(
            b, a, forward,
            doubleArrayOf(ZI[0] * forward.first(), ZI[1] * forward.first())
        )
        backward.reverse()

        return backward.copyOfRange(padLength, backward.size - padLength)
    }

    /** Direct-form II transposed, second order. */
    private fun filterWithInitialState(
        b: DoubleArray,
        a: DoubleArray,
        input: DoubleArray,
        initialState: DoubleArray
    ): DoubleArray {
        val output = DoubleArray(input.size)
        var z1 = initialState[0]
        var z2 = initialState[1]

        for (i in input.indices) {
            val x = input[i]
            val y = b[0] * x + z1
            z1 = b[1] * x - a[1] * y + z2
            z2 = b[2] * x - a[2] * y
            output[i] = y
        }
        return output
    }

    private fun oddPad(input: DoubleArray, padLength: Int): DoubleArray {
        val output = DoubleArray(input.size + 2 * padLength)
        input.copyInto(output, destinationOffset = padLength)

        for (i in 0 until padLength) {
            output[padLength - 1 - i] = 2.0 * input[0] - input[i + 1]
        }

        val last = input.lastIndex
        for (i in 0 until padLength) {
            output[padLength + input.size + i] = 2.0 * input[last] - input[last - 1 - i]
        }
        return output
    }

    private fun magnitudeSpectrum(input: DoubleArray, fftSize: Int): DoubleArray {
        val real = DoubleArray(fftSize)
        val imaginary = DoubleArray(fftSize)
        for (i in input.indices) real[i] = input[i]

        fft(real, imaginary)

        val outputSize = fftSize / 2 + 1
        return DoubleArray(outputSize) { i ->
            sqrt(real[i] * real[i] + imaginary[i] * imaginary[i])
        }
    }

    private fun fft(real: DoubleArray, imaginary: DoubleArray) {
        val n = real.size

        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit

            if (i < j) {
                val rt = real[i]; real[i] = real[j]; real[j] = rt
                val it = imaginary[i]; imaginary[i] = imaginary[j]; imaginary[j] = it
            }
        }

        var length = 2
        while (length <= n) {
            val angle = -2.0 * Math.PI / length
            val wLenReal = cos(angle)
            val wLenImaginary = sin(angle)
            val half = length / 2

            var start = 0
            while (start < n) {
                var wReal = 1.0
                var wImaginary = 0.0

                for (i in 0 until half) {
                    val even = start + i
                    val odd = start + i + half

                    val oddReal = real[odd] * wReal - imaginary[odd] * wImaginary
                    val oddImaginary = real[odd] * wImaginary + imaginary[odd] * wReal

                    val evenReal = real[even]
                    val evenImaginary = imaginary[even]

                    real[even] = evenReal + oddReal
                    imaginary[even] = evenImaginary + oddImaginary
                    real[odd] = evenReal - oddReal
                    imaginary[odd] = evenImaginary - oddImaginary

                    val nextWReal = wReal * wLenReal - wImaginary * wLenImaginary
                    wImaginary = wReal * wLenImaginary + wImaginary * wLenReal
                    wReal = nextWReal
                }
                start += length
            }
            length = length shl 1
        }
    }

    private fun invertMatrix(input: Array<DoubleArray>): Array<DoubleArray> {
        val n = input.size
        val augmented = Array(n) { row ->
            DoubleArray(n * 2).also { values ->
                for (column in 0 until n) values[column] = input[row][column]
                values[n + row] = 1.0
            }
        }

        for (column in 0 until n) {
            var pivotRow = column
            for (row in column + 1 until n) {
                if (abs(augmented[row][column]) > abs(augmented[pivotRow][column])) {
                    pivotRow = row
                }
            }

            require(abs(augmented[pivotRow][column]) > 1e-12) { "Matrix is singular" }

            if (pivotRow != column) {
                val tmp = augmented[column]
                augmented[column] = augmented[pivotRow]
                augmented[pivotRow] = tmp
            }

            val pivot = augmented[column][column]
            for (j in 0 until n * 2) augmented[column][j] /= pivot

            for (row in 0 until n) {
                if (row == column) continue
                val factor = augmented[row][column]
                if (abs(factor) < 1e-15) continue
                for (j in 0 until n * 2) augmented[row][j] -= factor * augmented[column][j]
            }
        }

        return Array(n) { row -> DoubleArray(n) { column -> augmented[row][n + column] } }
    }
}