import { createFaceLandmarker, createInferenceWorker, FramePipeline } from '../js/pipeline.js';

const params = new URLSearchParams(location.search);
const IMG = params.get('img') || 'tmp/face.png';
const BPMS = (params.get('bpm') || '60,75,90,110').split(',').map(Number);
const SECONDS = Number(params.get('seconds') || 12);
const FPS = 30;
const AMPLITUDE = 0.004; // 0.4 % intensity change, in the range of a real pulse
const NOISE = 1.5; // camera noise, grey levels
const MIX = [0.35, 1.0, 0.55]; // R, G, B pulse strength (green carries most of it)

const logEl = document.getElementById('log');
const lines = [];
const log = (s) => { lines.push(s); logEl.textContent = lines.join('\n'); };
window.__e2e = { done: false, runs: [] };

function gaussian() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// Yield so worker results are delivered. MessageChannel instead of setTimeout:
// background tabs throttle timers to ~1 s, which would stall the run.
const channel = new MessageChannel();
const tick = () => new Promise((resolve) => {
  channel.port1.onmessage = () => resolve();
  channel.port2.postMessage(0);
});

async function main() {
  // createImageBitmap rather than <img>.decode(): the latter can stall in a
  // page that isn't being rendered (hidden tab / preview pane).
  const img = await createImageBitmap(await (await fetch(IMG)).blob());

  const canvas = document.getElementById('frame');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const base = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const frame = new ImageData(canvas.width, canvas.height);

  let t0 = performance.now();
  const { landmarker, delegate } = await createFaceLandmarker();
  log(`MediaPipe loaded (${delegate}) in ${(performance.now() - t0).toFixed(0)} ms`);

  let clockMs = 1000;
  for (const bpm of BPMS) {
    t0 = performance.now();
    const { worker, ready } = createInferenceWorker();
    const info = await ready;
    const loadMs = performance.now() - t0;

    const results = [];
    const pipeline = new FramePipeline(landmarker, worker, (hr, timings) => results.push({ hr, timings }));
    let faceFrames = 0;
    let detectMs = 0;
    const frames = SECONDS * FPS;

    for (let f = 0; f < frames; f++) {
      const t = f / FPS;
      const p = Math.sin(2 * Math.PI * (bpm / 60) * t) + 0.3 * Math.sin(4 * Math.PI * (bpm / 60) * t + 0.5);
      const gain = MIX.map((m) => 1 + AMPLITUDE * m * p);
      const src = base.data;
      const dst = frame.data;
      for (let i = 0; i < src.length; i += 4) {
        dst[i] = src[i] * gain[0] + NOISE * gaussian();
        dst[i + 1] = src[i + 1] * gain[1] + NOISE * gaussian();
        dst[i + 2] = src[i + 2] * gain[2] + NOISE * gaussian();
        dst[i + 3] = 255;
      }
      ctx.putImageData(frame, 0, 0);

      const s = performance.now();
      const ui = pipeline.process(canvas, canvas.width, canvas.height, clockMs);
      detectMs += performance.now() - s;
      if (ui.faceDetected) faceFrames++;
      clockMs += 1000 / FPS;
      window.__e2e.progress = { bpm, frame: f, frames, status: ui.status.message, buffered: ui.bufferedFrames };
      await tick();
    }
    while (pipeline.controller.inferenceRunning) await tick();
    worker.terminate();

    const c = pipeline.controller;
    const bpms = results.map((r) => r.hr.bpm);
    const run = {
      injectedBpm: bpm,
      windows: results.length,
      perWindowBpm: bpms.map((v) => +v.toFixed(1)),
      quality: results.map((r) => r.hr.quality),
      medianBpm: c.bpmMedian,
      error: c.bpmMedian != null ? Math.abs(c.bpmMedian - bpm) : null,
      roiWeights: c.weights.map((w) => +w.toFixed(3)),
      faceDetectedPct: (100 * faceFrames) / frames,
      pipelineMsPerFrame: detectMs / frames,
      inferenceMs: results.map((r) => +r.timings.totalMs.toFixed(0)),
      workerLoadMs: +loadMs.toFixed(0),
      threads: info.threads,
    };
    window.__e2e.runs.push(run);
    log(`\ninjected ${bpm} bpm -> median ${run.medianBpm != null ? run.medianBpm.toFixed(1) : '-'} bpm ` +
      `(error ${run.error != null ? run.error.toFixed(1) : '-'})`);
    log(`  windows ${run.perWindowBpm.join(', ')}  quality ${run.quality.join(',')}`);
    log(`  face ${run.faceDetectedPct.toFixed(0)}%  frame pipeline ${run.pipelineMsPerFrame.toFixed(1)} ms  ` +
      `inference ${run.inferenceMs.join('/')} ms  APRF ${run.roiWeights.join('/')}`);
  }
  window.__e2e.done = true;
  log('\ndone');
}

main().catch((error) => {
  log(`ERROR: ${error && error.stack || error}`);
  window.__e2e.error = String(error);
  window.__e2e.done = true;
});
