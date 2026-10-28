package com.ns.litepulse.face

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class RoiExtractor {

    companion object {

        const val PATCH_SIZE = 36

        val ROI_ORDER = listOf(
            "forehead",
            "left_cheek",
            "right_cheek"
        )

        val LANDMARKS = mapOf(
            "forehead" to listOf(
                10, 108, 69, 104, 68, 71, 21, 54,
                103, 67, 109, 10,
                338, 297, 332, 284, 251, 301,
                293, 334, 296, 336
            ),

            "left_cheek" to listOf(
                116, 117, 118, 119, 100, 126,
                142, 36, 205, 187, 123
            ),

            "right_cheek" to listOf(
                345, 346, 347, 348, 329, 355,
                371, 266, 425, 411, 352
            )
        )
    }

    /**
     * Convert MediaPipe normalized landmarks into
     * pixel coordinates of the Bitmap given to MediaPipe.
     */
    fun getRoiPolygons(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Map<String, List<PointF>> {

        return ROI_ORDER.associateWith { roiName ->

            LANDMARKS.getValue(roiName).map { index ->

                val landmark = landmarks[index]

                PointF(
                    landmark.x() * imageWidth,
                    landmark.y() * imageHeight
                )
            }
        }
    }

    /**
     * Calculate the convex hull of a set of points.
     *
     * Equivalent to cv2.convexHull().
     */
    fun convexHull(
        points: List<PointF>
    ): List<PointF> {

        if (points.size <= 2) {
            return points
        }

        val sorted = points
            .distinctBy {
                "${it.x},${it.y}"
            }
            .sortedWith(
                compareBy<PointF> { it.x }
                    .thenBy { it.y }
            )

        if (sorted.size <= 2) {
            return sorted
        }

        fun cross(
            o: PointF,
            a: PointF,
            b: PointF
        ): Float {

            return (
                (a.x - o.x) * (b.y - o.y) -
                    (a.y - o.y) * (b.x - o.x)
                )
        }

        val lower = mutableListOf<PointF>()

        for (point in sorted) {

            while (
                lower.size >= 2 &&
                cross(
                    lower[lower.size - 2],
                    lower[lower.size - 1],
                    point
                ) <= 0f
            ) {
                lower.removeAt(
                    lower.lastIndex
                )
            }

            lower.add(point)
        }

        val upper = mutableListOf<PointF>()

        for (point in sorted.asReversed()) {

            while (
                upper.size >= 2 &&
                cross(
                    upper[upper.size - 2],
                    upper[upper.size - 1],
                    point
                ) <= 0f
            ) {
                upper.removeAt(
                    upper.lastIndex
                )
            }

            upper.add(point)
        }

        lower.removeAt(
            lower.lastIndex
        )

        upper.removeAt(
            upper.lastIndex
        )

        return lower + upper
    }

    /**
     * Get the convex-hull polygon for every ROI.
     */
    fun getConvexHullPolygons(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Map<String, List<PointF>> {

        return getRoiPolygons(
            landmarks,
            imageWidth,
            imageHeight
        ).mapValues { (_, points) ->
            convexHull(points)
        }
    }

    /**
     * Extract one ROI from the original Bitmap.
     *
     * This reproduces the Python pipeline:
     *
     * convex hull
     * -> filled polygon mask
     * -> bounding rectangle
     * -> masked crop
     * -> resize to 36x36
     */
    fun extractPatch(
        bitmap: Bitmap,
        polygon: List<PointF>
    ): Bitmap {

        if (polygon.isEmpty()) {
            throw IllegalArgumentException(
                "ROI polygon is empty"
            )
        }

        /*
         * Create polygon path.
         */
        val path = Path()

        path.moveTo(
            polygon[0].x,
            polygon[0].y
        )

        for (i in 1 until polygon.size) {

            path.lineTo(
                polygon[i].x,
                polygon[i].y
            )
        }

        path.close()

        /*
         * Find bounding rectangle.
         */
        val bounds = RectF()

        path.computeBounds(
            bounds,
            true
        )

        val left =
            bounds.left
                .toInt()
                .coerceIn(
                    0,
                    bitmap.width - 1
                )

        val top =
            bounds.top
                .toInt()
                .coerceIn(
                    0,
                    bitmap.height - 1
                )

        val right =
            bounds.right
                .toInt()
                .coerceIn(
                    left + 1,
                    bitmap.width
                )

        val bottom =
            bounds.bottom
                .toInt()
                .coerceIn(
                    top + 1,
                    bitmap.height
                )

        val cropWidth =
            right - left

        val cropHeight =
            bottom - top

        /*
         * Crop the bounding rectangle.
         */
        val cropped =
            Bitmap.createBitmap(
                bitmap,
                left,
                top,
                cropWidth,
                cropHeight
            )

        /*
         * Move the polygon into the
         * cropped-image coordinate system.
         */
        val translatedPath =
            Path()

        translatedPath.moveTo(
            polygon[0].x - left,
            polygon[0].y - top
        )

        for (i in 1 until polygon.size) {

            translatedPath.lineTo(
                polygon[i].x - left,
                polygon[i].y - top
            )
        }

        translatedPath.close()

        /*
         * Create transparent output.
         */
        val masked =
            Bitmap.createBitmap(
                cropWidth,
                cropHeight,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(masked)

        /*
         * Clear everything outside the polygon.
         */
        canvas.drawColor(
            Color.TRANSPARENT
        )

        val paint =
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                isFilterBitmap = false
            }

        canvas.save()

        canvas.clipPath(
            translatedPath
        )

        canvas.drawBitmap(
            cropped,
            0f,
            0f,
            paint
        )

        canvas.restore()

        /*
         * Resize to the model's expected
         * 36 x 36 patch.
         */
        return Bitmap.createScaledBitmap(
            masked,
            PATCH_SIZE,
            PATCH_SIZE,
            false
        )
    }

    /**
     * Extract all three ROI patches.
     */
    fun extractRoiPatches(
        bitmap: Bitmap,
        landmarks: List<NormalizedLandmark>
    ): Map<String, Bitmap> {

        val polygons =
            getConvexHullPolygons(
                landmarks,
                bitmap.width,
                bitmap.height
            )

        return ROI_ORDER.associateWith { roiName ->

            extractPatch(
                bitmap,
                polygons.getValue(roiName)
            )
        }
    }
}