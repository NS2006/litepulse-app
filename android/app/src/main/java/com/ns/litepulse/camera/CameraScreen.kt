package com.ns.litepulse.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

import androidx.core.content.ContextCompat

import com.ns.litepulse.face.FaceLandmarkerManager
import com.ns.litepulse.face.RoiExtractor
import com.ns.litepulse.measurement.MeasurementController
import com.ns.litepulse.measurement.MeasurementUiState
import com.ns.litepulse.model.HeartRateResult
import com.ns.litepulse.model.LitePulseHeartRateCalculator
import com.ns.litepulse.model.LitePulseModelManager
import com.ns.litepulse.model.LitePulsePreprocessor
import com.ns.litepulse.model.SignalQuality

import java.util.concurrent.Executors
import kotlin.math.roundToInt

/** Shows PNR / band ratio / score / fps under the ROI panel. Turn off for the demo. */
private const val SHOW_DEBUG = true

@Composable
fun CameraScreen() {

    val context = LocalContext.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { granted ->
            hasCameraPermission = granted
        }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(text = "Izin kamera diperlukan")
        }
        return
    }

    CameraContent()
}

@Composable
private fun CameraContent() {

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val faceLandmarkerManager = remember { FaceLandmarkerManager(context) }

    val controller = remember {
        MeasurementController(
            roiExtractor = RoiExtractor(),
            preprocessor = LitePulsePreprocessor(),
            model = LitePulseModelManager(context),
            calculator = LitePulseHeartRateCalculator()
        )
    }

    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }

    var ui by remember { mutableStateOf(MeasurementUiState()) }

    DisposableEffect(Unit) {
        onDispose {
            faceLandmarkerManager.close()
            analyzerExecutor.shutdown()
            controller.close()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {

        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->

                PreviewView(ctx).apply {

                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                    cameraProviderFuture.addListener({

                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder().build()
                        preview.setSurfaceProvider(surfaceProvider)

                        val imageAnalysis =
                            ImageAnalysis.Builder()
                                .setBackpressureStrategy(
                                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                )
                                // RGBA makes toBitmap() a cheap copy instead of a YUV conversion
                                .setOutputImageFormat(
                                    ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                                )
                                .build()

                        imageAnalysis.setAnalyzer(
                            analyzerExecutor,
                            CameraAnalyzer(faceLandmarkerManager) { bitmap, result, timestampNs ->
                                ui = controller.process(bitmap, result, timestampNs)
                            }
                        )

                        cameraProvider.unbindAll()

                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview,
                            imageAnalysis
                        )

                    }, ContextCompat.getMainExecutor(ctx))
                }
            }
        )

        StatusCard(
            ui = ui,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
        )

        RegionsPanel(
            ui = ui,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(12.dp)
        )
    }
}

// ---------------------------------------------------------------------
// Top card
// ---------------------------------------------------------------------

@Composable
private fun StatusCard(
    ui: MeasurementUiState,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = Color.Black.copy(alpha = 0.72f)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            ui.notice?.let {
                Text(
                    text = it,
                    color = Color(0xFFFFB74D),
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            val heartRate = ui.heartRate

            when {
                !ui.isMeasuring -> {
                    Text(
                        text = ui.status.message,
                        color = Color.White,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    if (ui.status.hint.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = ui.status.hint,
                            color = Color.White.copy(alpha = 0.65f),
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center
                        )
                    }
                }

                !ui.windowFull -> {
                    Box(contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(
                            progress = { ui.progress },
                            modifier = Modifier.size(96.dp),
                            strokeWidth = 8.dp,
                            color = Color.White,
                            trackColor = Color.White.copy(alpha = 0.2f)
                        )
                        Text(
                            text = "${ui.secondsRemaining}",
                            color = Color.White,
                            style = MaterialTheme.typography.displaySmall,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Mengukur… tahan posisi",
                        color = Color.White.copy(alpha = 0.75f),
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                heartRate == null -> {
                    Text(
                        text = "Menghitung…",
                        color = Color.White,
                        style = MaterialTheme.typography.titleLarge
                    )
                }

                heartRate.quality == SignalQuality.POOR -> {
                    Text(
                        text = "Sinyal kurang jelas",
                        color = Color.White,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Pastikan wajah terang merata, jangan berbicara, " +
                            "singkirkan rambut dari dahi, dan tahan posisi.",
                        color = Color.White.copy(alpha = 0.7f),
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    QualityChip(heartRate.quality)
                }

                else -> {
                    HeartRateReadout(ui, heartRate)
                }
            }

            // BVP waveform with peak markers (shown for any finished window)
            if (ui.windowFull && heartRate != null && heartRate.waveform.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                PulseWaveform(
                    samples = heartRate.waveform,
                    peaks = heartRate.peaks,
                    dimmed = heartRate.quality == SignalQuality.POOR,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(64.dp)
                )
            }

            // Session trend
            if (ui.trend.size >= 5) {
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "Tren HR sesi",
                    color = Color.White.copy(alpha = 0.6f),
                    style = MaterialTheme.typography.labelSmall
                )
                TrendChart(
                    values = ui.trend,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(40.dp)
                )
            }
        }
    }
}

@Composable
private fun HeartRateReadout(ui: MeasurementUiState, heartRate: HeartRateResult) {

    val shown = ui.bpmMedian ?: heartRate.bpm

    Text(
        text = "Detak Jantung",
        color = Color.White.copy(alpha = 0.75f),
        style = MaterialTheme.typography.titleMedium
    )
    Text(
        text = "${shown.roundToInt()} BPM",
        color = Color.White,
        style = MaterialTheme.typography.displayMedium,
        fontWeight = FontWeight.Bold
    )

    val low = ui.bpmLow
    val high = ui.bpmHigh
    if (low != null && high != null) {
        Text(
            text = "rentang ${low.roundToInt()}–${high.roundToInt()} bpm",
            color = Color.White.copy(alpha = 0.7f),
            style = MaterialTheme.typography.bodyMedium
        )
    }

    Spacer(modifier = Modifier.height(6.dp))
    QualityChip(heartRate.quality)

    if (heartRate.quality == SignalQuality.FAIR) {
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Perkiraan kasar, perbaiki cahaya untuk hasil lebih akurat",
            color = Color.White.copy(alpha = 0.6f),
            style = MaterialTheme.typography.bodySmall,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
private fun QualityChip(quality: SignalQuality) {

    val color = when (quality) {
        SignalQuality.GOOD -> Color(0xFF4CAF50)
        SignalQuality.FAIR -> Color(0xFFFFC107)
        SignalQuality.POOR -> Color(0xFFF44336)
    }

    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(color, RoundedCornerShape(50))
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "Kualitas sinyal: ${quality.label}",
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}

// ---------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------

@Composable
private fun PulseWaveform(
    samples: FloatArray,
    peaks: List<Int>,
    dimmed: Boolean,
    modifier: Modifier = Modifier
) {
    val lineColor = if (dimmed) Color.White.copy(alpha = 0.35f) else Color(0xFF4FC3F7)

    Canvas(modifier = modifier) {
        if (samples.size < 2) return@Canvas

        var min = samples[0]
        var max = samples[0]
        for (v in samples) {
            if (v < min) min = v
            if (v > max) max = v
        }
        val range = (max - min).coerceAtLeast(1e-6f)
        val stepX = size.width / (samples.size - 1)

        fun yOf(v: Float) =
            size.height * 0.95f - (v - min) / range * size.height * 0.9f

        val path = Path()
        samples.forEachIndexed { i, v ->
            if (i == 0) path.moveTo(0f, yOf(v)) else path.lineTo(i * stepX, yOf(v))
        }
        drawPath(path, color = lineColor, style = Stroke(width = 3f))

        if (!dimmed) {
            for (p in peaks) {
                drawCircle(
                    color = Color(0xFFFF5252),
                    radius = 6f,
                    center = Offset(p * stepX, yOf(samples[p]))
                )
            }
        }
    }
}

@Composable
private fun TrendChart(
    values: List<Float>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        if (values.size < 2) return@Canvas

        val lo = (values.min() - 3f)
        val hi = (values.max() + 3f)
        val range = (hi - lo).coerceAtLeast(1f)
        val stepX = size.width / (values.size - 1)

        val path = Path()
        values.forEachIndexed { i, v ->
            val y = size.height - (v - lo) / range * size.height
            if (i == 0) path.moveTo(0f, y) else path.lineTo(i * stepX, y)
        }
        drawPath(path, color = Color(0xFFFFB74D), style = Stroke(width = 3f))
    }
}

// ---------------------------------------------------------------------
// Bottom panel
// ---------------------------------------------------------------------

@Composable
private fun RegionsPanel(
    ui: MeasurementUiState,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = Color.Black.copy(alpha = 0.82f)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {

                Column {
                    Text(
                        text = "Bobot Region (APRF)",
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = if (ui.weightsFromModel) "Dari model"
                        else "Estimasi dari sinyal",
                        color = Color.White.copy(alpha = 0.65f),
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(
                                color = if (ui.faceDetected) Color.Green else Color.Red,
                                shape = RoundedCornerShape(50)
                            )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (ui.faceDetected) "Wajah terdeteksi" else "Tidak ada wajah",
                        color = Color.White,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(modifier = Modifier.fillMaxWidth()) {

                RoiPreview(
                    name = "Dahi",
                    bitmap = ui.roiPatches["forehead"],
                    weight = ui.roiWeights[0],
                    modifier = Modifier.weight(1f)
                )

                Spacer(modifier = Modifier.width(8.dp))

                RoiPreview(
                    name = "Pipi Kiri",
                    bitmap = ui.roiPatches["left_cheek"],
                    weight = ui.roiWeights[1],
                    modifier = Modifier.weight(1f)
                )

                Spacer(modifier = Modifier.width(8.dp))

                RoiPreview(
                    name = "Pipi Kanan",
                    bitmap = ui.roiPatches["right_cheek"],
                    weight = ui.roiWeights[2],
                    modifier = Modifier.weight(1f)
                )
            }

            if (SHOW_DEBUG && ui.debug.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = ui.debug,
                    color = Color.White.copy(alpha = 0.5f),
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    }
}

@Composable
private fun RoiPreview(
    name: String,
    bitmap: Bitmap?,
    weight: Float,
    modifier: Modifier = Modifier
) {

    val animatedWeight by animateFloatAsState(
        targetValue = weight.coerceIn(0f, 1f),
        label = "roiWeight"
    )

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(90.dp)
                .border(
                    width = 1.dp,
                    color = Color.White.copy(alpha = 0.25f),
                    shape = RoundedCornerShape(12.dp)
                )
                .background(
                    color = Color.White.copy(alpha = 0.08f),
                    shape = RoundedCornerShape(12.dp)
                ),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "$name ROI",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(4.dp)
                )
            } else {
                Text(
                    text = "Menunggu...",
                    color = Color.White.copy(alpha = 0.45f),
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        LinearProgressIndicator(
            progress = { animatedWeight },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp),
            color = Color(0xFF4FC3F7),
            trackColor = Color.White.copy(alpha = 0.15f)
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "$name ${(animatedWeight * 100).roundToInt()}%",
            color = Color.White,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium
        )
    }
}