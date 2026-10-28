// Inference worker: resample + normalise the window, run LitePulse with
// ONNX Runtime Web, then compute heart rate. Keeps the camera loop smooth.

import { ORT_DIST } from './config.js';
import { buildInput } from './preprocess.js';
import { HeartRateCalculator } from './heartrate.js';

const calculator = new HeartRateCalculator();
let session = null;
let ort = null;

const ready = (async () => {
  ort = await import(`${ORT_DIST}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_DIST;
  // Threads need cross-origin isolation, which static hosts like GitHub Pages
  // can't enable; single-threaded SIMD is fast enough for one run every 2 s.
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  session = await ort.InferenceSession.create(
    new URL('../models/litepulse.onnx', import.meta.url).href,
    { executionProviders: ['wasm'] },
  );
  return { threads: ort.env.wasm.numThreads };
})();

ready.then(
  (info) => self.postMessage({ type: 'ready', ...info }),
  (error) => self.postMessage({ type: 'error', message: String(error && error.message || error) }),
);

/** Averages [.., 3] weights over all frames and normalises to sum 1. */
function reduceWeights(raw) {
  const groups = raw.length / 3;
  const mean = [0, 0, 0];
  for (let g = 0; g < groups; g++) {
    for (let k = 0; k < 3; k++) mean[k] += raw[g * 3 + k] / groups;
  }
  const sum = mean[0] + mean[1] + mean[2];
  return sum > 0 ? mean.map((v) => v / sum) : null;
}

async function infer(frames, times) {
  await ready;
  const t0 = performance.now();
  const { input } = buildInput({ frames, times });
  const t1 = performance.now();
  const outputs = await session.run({ input: new ort.Tensor('float32', input, [1, 180, 18, 36, 36]) });
  const t2 = performance.now();

  const ratios = outputs.energy_ratio.data;
  let energyRatio = 0;
  for (let i = 0; i < ratios.length; i++) energyRatio += ratios[i] / ratios.length;

  const hr = calculator.calculate(outputs.bvp.data, energyRatio);
  return {
    hr,
    weights: reduceWeights(outputs.roi_weights.data),
    bvp: Float32Array.from(outputs.bvp.data),
    timings: { preprocessMs: t1 - t0, modelMs: t2 - t1, totalMs: performance.now() - t0 },
  };
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'infer') return;
  try {
    const result = await infer(data.frames, data.times);
    self.postMessage({ type: 'result', gen: data.gen, ...result });
  } catch (error) {
    self.postMessage({ type: 'error', gen: data.gen, message: String(error && error.message || error) });
  }
};
