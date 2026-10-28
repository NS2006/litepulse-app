// Page wiring: camera, frame loop and rendering (mirrors camera/CameraScreen.kt).

import { APK_URL } from './config.js';
import { createFaceLandmarker, createInferenceWorker, FramePipeline } from './pipeline.js';
import { convexHull, ROI_ORDER } from './roi.js';

const $ = (id) => document.getElementById(id);
const SHOW_DEBUG = new URLSearchParams(location.search).has('debug');
const ROI_LABELS = { forehead: 'Dahi', left_cheek: 'Pipi Kiri', right_cheek: 'Pipi Kanan' };
const ROI_COLORS = { forehead: '#4fc3f7', left_cheek: '#ffb74d', right_cheek: '#ba68c8' };
const RING_LENGTH = 2 * Math.PI * 44;

$('apk-link').href = APK_URL;

let stream = null;
let running = false;
let pipeline = null;
let worker = null;
let landmarker = null;

$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);

function setLoadStatus(text, isError = false) {
  const el = $('load-status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

async function start() {
  const button = $('start');
  button.disabled = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setLoadStatus('Kamera tidak tersedia. Buka halaman ini lewat HTTPS di browser terbaru.', true);
    button.disabled = false;
    return;
  }

  try {
    setLoadStatus('Meminta izin kamera…');
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
  } catch (error) {
    setLoadStatus('Izin kamera ditolak atau kamera tidak ditemukan.', true);
    button.disabled = false;
    return;
  }

  try {
    setLoadStatus('Memuat model (sekitar 15 MB, hanya pertama kali)…');
    if (!worker) {
      const created = createInferenceWorker();
      worker = created.worker;
      const [face] = await Promise.all([
        landmarker ? Promise.resolve({ landmarker }) : createFaceLandmarker(),
        created.ready,
      ]);
      landmarker = face.landmarker;
    }
    pipeline = new FramePipeline(landmarker, worker);
  } catch (error) {
    console.error(error);
    setLoadStatus('Gagal memuat model. Periksa koneksi lalu coba lagi.', true);
    stopStream();
    worker = null;
    button.disabled = false;
    return;
  }

  const video = $('video');
  video.srcObject = stream;
  await video.play().catch(() => {});

  setLoadStatus('');
  $('intro').hidden = true;
  $('measure').hidden = false;
  running = true;
  scheduleFrame(video);
}

function stopStream() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

function stop() {
  running = false;
  stopStream();
  $('video').srcObject = null;
  $('measure').hidden = true;
  $('intro').hidden = false;
  $('start').disabled = false;
}

// ---------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------

function scheduleFrame(video) {
  if (!running) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback((now, meta) => {
      // captureTime is when the camera took the frame (Chrome); the others are fallbacks.
      onFrame(video, meta.captureTime || meta.presentationTime || now);
      scheduleFrame(video);
    });
  } else {
    let lastTime = -1;
    const tick = (now) => {
      if (!running) return;
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        onFrame(video, now);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

function onFrame(video, timeMs) {
  if (!video.videoWidth || !running) return;
  const ui = pipeline.process(video, video.videoWidth, video.videoHeight, timeMs);
  render(ui);
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

function render(ui) {
  const notice = $('notice');
  notice.hidden = !ui.notice;
  notice.textContent = ui.notice || '';

  const hr = ui.heartRate;
  let state = 'message';
  if (!ui.isMeasuring) {
    showMessage(ui.status.message, ui.status.hint);
  } else if (!ui.windowReady) {
    state = 'progress';
    $('seconds').textContent = ui.secondsRemaining;
    $('ring-value').style.strokeDashoffset = String(RING_LENGTH * (1 - ui.progress));
  } else if (!hr) {
    showMessage('Menghitung…', '');
  } else if (hr.quality === 'POOR') {
    showMessage('Sinyal kurang jelas',
      'Pastikan wajah terang merata, jangan berbicara, singkirkan rambut dari dahi, dan tahan posisi.');
  } else {
    state = 'result';
    const shown = ui.bpmMedian != null ? ui.bpmMedian : hr.bpm;
    $('bpm').textContent = Math.round(shown);
    $('range').textContent = ui.bpmLow != null
      ? `rentang ${Math.round(ui.bpmLow)}–${Math.round(ui.bpmHigh)} bpm` : '';
    setQuality(hr.quality);
    $('quality-hint').textContent = hr.quality === 'FAIR'
      ? 'Perkiraan kasar, perbaiki cahaya untuk hasil lebih akurat' : '';
  }
  $('state-message').hidden = state !== 'message';
  $('state-progress').hidden = state !== 'progress';
  $('state-result').hidden = state !== 'result';

  const showWave = ui.windowReady && hr && hr.waveform.length > 0;
  $('waveform').hidden = !showWave;
  if (showWave) drawWaveform($('waveform'), hr.waveform, hr.peaks, hr.quality === 'POOR');

  $('trend-block').hidden = ui.trend.length < 5;
  if (ui.trend.length >= 5) drawTrend($('trend'), ui.trend);

  $('face-dot').classList.toggle('on', ui.faceDetected);
  $('face-text').textContent = ui.faceDetected ? 'Wajah terdeteksi' : 'Tidak ada wajah';

  ROI_ORDER.forEach((name, i) => {
    const pct = Math.round(ui.roiWeights[i] * 100);
    $(`bar-${name}`).style.width = `${pct}%`;
    $(`label-${name}`).textContent = `${ROI_LABELS[name]} ${pct}%`;
    if (ui.polygons) $(`roi-${name}`).getContext('2d').putImageData(ui.previews[name], 0, 0);
  });

  $('debug').textContent = SHOW_DEBUG ? ui.debug : '';
  drawOverlay(ui);
}

function showMessage(message, hint) {
  $('message').textContent = message;
  $('hint').textContent = hint;
}

function setQuality(key) {
  const el = $('quality');
  el.className = `quality ${key.toLowerCase()}`;
  $('quality-text').textContent = `Kualitas sinyal: ${{ GOOD: 'Baik', FAIR: 'Cukup', POOR: 'Buruk' }[key]}`;
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return canvas.getContext('2d');
}

function drawWaveform(canvas, samples, peaks, dimmed) {
  const ctx = fitCanvas(canvas);
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  let min = Infinity;
  let max = -Infinity;
  for (const v of samples) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  const range = Math.max(max - min, 1e-6);
  const stepX = width / (samples.length - 1);
  const y = (v) => height * 0.95 - ((v - min) / range) * height * 0.9;

  ctx.beginPath();
  samples.forEach((v, i) => (i ? ctx.lineTo(i * stepX, y(v)) : ctx.moveTo(0, y(v))));
  ctx.strokeStyle = dimmed ? 'rgba(255,255,255,0.35)' : '#4fc3f7';
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.stroke();

  if (!dimmed) {
    ctx.fillStyle = '#ff5252';
    for (const p of peaks) {
      ctx.beginPath();
      ctx.arc(p * stepX, y(samples[p]), 3.5 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawTrend(canvas, values) {
  const ctx = fitCanvas(canvas);
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const lo = Math.min(...values) - 3;
  const hi = Math.max(...values) + 3;
  const range = Math.max(hi - lo, 1);
  const stepX = width / (values.length - 1);
  ctx.beginPath();
  values.forEach((v, i) => {
    const y = height - ((v - lo) / range) * height;
    if (i) ctx.lineTo(i * stepX, y);
    else ctx.moveTo(0, y);
  });
  ctx.strokeStyle = '#ffb74d';
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.stroke();
}

/** ROI outlines over the mirrored, object-fit: cover video. */
function drawOverlay(ui) {
  const canvas = $('overlay');
  const ctx = fitCanvas(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!ui.polygons) return;

  const scale = Math.max(canvas.width / ui.frameWidth, canvas.height / ui.frameHeight);
  const offX = (canvas.width - ui.frameWidth * scale) / 2;
  const offY = (canvas.height - ui.frameHeight * scale) / 2;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);

  for (const name of ROI_ORDER) {
    const poly = convexHull(ui.polygons[name]);
    ctx.beginPath();
    poly.forEach(([x, y], i) => {
      const px = canvas.width - (x * scale + offX);
      const py = y * scale + offY;
      if (i) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.strokeStyle = ROI_COLORS[name];
    ctx.stroke();
  }
}
