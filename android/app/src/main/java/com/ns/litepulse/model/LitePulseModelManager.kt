package com.ns.litepulse.model

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor

/**
 * @param pulse      180-sample predicted pulse signal.
 * @param roiWeights APRF weights [forehead, left cheek, right cheek], summing to 1.
 *                   null when the exported model only returns the pulse tensor.
 */
class LitePulseOutput(
    val pulse: FloatArray,
    val roiWeights: FloatArray?,
    /** Mean PulseSpectralAttention energy ratio over feature channels, 0..1. */
    val energyRatio: Float?
)

class LitePulseModelManager(
    context: Context
) {

    private val module =
        LiteModuleLoader.load(assetFilePath(context, "litepulse.ptl"))

    fun predict(input: FloatArray): LitePulseOutput {

        val expectedSize = 180 * 18 * 36 * 36

        require(input.size == expectedSize) {
            "Expected $expectedSize input values, got ${input.size}"
        }

        val inputTensor = Tensor.fromBlob(
            input,
            longArrayOf(1, 180, 18, 36, 36)
        )

        val output = module.forward(IValue.from(inputTensor))
        Log.d("LitePulse", "model output isTuple=${output.isTuple}")

        /*
         * Supported model outputs:
         *   - Tensor                      -> pulse only (current model)
         *   - Tuple(pulse, aprf_weights)  -> pulse + per-ROI weights
         *
         * To get real APRF weights, export the model so forward() returns
         * (pulse, weights) where weights has 3 values per sample
         * (shape [3], [1, 3] or [1, T, 3]).
         */
        if (output.isTuple) {
            val items = output.toTuple()
            val pulse = items[0].toTensor().dataAsFloatArray
            val weights = if (items.size > 1 && items[1].isTensor) {
                reduceWeights(items[1].toTensor().dataAsFloatArray)
            } else {
                null
            }
            val energyRatio = if (items.size > 2 && items[2].isTensor) {
                items[2].toTensor().dataAsFloatArray.takeIf { it.isNotEmpty() }?.average()?.toFloat()
            } else {
                null
            }
            return LitePulseOutput(pulse, weights, energyRatio)
        }

        return LitePulseOutput(output.toTensor().dataAsFloatArray, null, null)
    }

    /** Averages [.., 3] weights over all leading dimensions and normalizes to sum 1. */
    private fun reduceWeights(raw: FloatArray): FloatArray? {
        if (raw.size < 3 || raw.size % 3 != 0) return null

        val groups = raw.size / 3
        val mean = FloatArray(3)
        for (g in 0 until groups) {
            for (k in 0 until 3) mean[k] += raw[g * 3 + k] / groups
        }

        val sum = mean.sum()
        if (sum <= 0f) return null
        return FloatArray(3) { mean[it] / sum }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = context.cacheDir.resolve(assetName)

        // Always overwrite: a stale cached copy would keep loading an old model
        // even after the file in assets/ was replaced.
        context.assets.open(assetName).use { input ->
            file.outputStream().use { output -> input.copyTo(output) }
        }

        return file.absolutePath
    }
}