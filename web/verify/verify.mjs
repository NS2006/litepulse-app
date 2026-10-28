// Checks the site's JavaScript pipeline against make_reference.py.
//
//   python verify/make_reference.py verify/out
//   npm install
//   node verify/verify.mjs verify/out
//
// Runs the same modules the page uses (js/roi.js, js/preprocess.js,
// js/heartrate.js) and the same ONNX file with ONNX Runtime Web's WASM backend.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { extractRoiFrame, PLANE, ROI_ORDER } from '../js/roi.js';
import { buildInput } from '../js/preprocess.js';
import { HeartRateCalculator } from '../js/heartrate.js';

const refDir = process.argv[2] || 'verify/out';
const ort = await import(process.env.ORT_WEB_MODULE ? pathToFileURL(process.env.ORT_WEB_MODULE).href : 'onnxruntime-web');
const manifest = JSON.parse(readFileSync(join(refDir, 'manifest.json'), 'utf8'));
const TYPES = { uint8: Uint8Array, float32: Float32Array, float64: Float64Array };

function load(name) {
  const { dtype, shape } = manifest.arrays[name];
  const buf = readFileSync(join(refDir, `${name}.bin`));
  const data = new TYPES[dtype](buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return { data, shape };
}

const results = [];
function check(label, ok, detail) {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(44)} ${detail}`);
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

// ---- 1. ROI extraction vs cv2 -------------------------------------------
{
  const img = load('roi_image');
  const [h, w] = img.shape;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = img.data[i * 3];
    rgba[i * 4 + 1] = img.data[i * 3 + 1];
    rgba[i * 4 + 2] = img.data[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  const imageData = { data: rgba, width: w, height: h };
  const lms = load('roi_landmarks');
  const expected = load('roi_patches');

  for (let c = 0; c < manifest.roi_cases; c++) {
    const landmarks = [];
    for (let i = 0; i < 478; i++) {
      landmarks.push({ x: lms.data[(c * 478 + i) * 2], y: lms.data[(c * 478 + i) * 2 + 1] });
    }
    const { frame } = extractRoiFrame(imageData, landmarks);
    let max = 0;
    let off = 0;
    let sum = 0;
    for (let r = 0; r < 3; r++) {
      for (let p = 0; p < PLANE; p++) {
        for (let ch = 0; ch < 3; ch++) {
          const ref = expected.data[(((c * 3 + r) * PLANE) + p) * 3 + ch];
          const got = frame[r * 3 * PLANE + ch * PLANE + p];
          const d = Math.abs(ref - got);
          max = Math.max(max, d);
          sum += d;
          if (d > 1) off++;
        }
      }
    }
    const n = 3 * 3 * PLANE;
    // cv2 resizes in fixed point, so allow +-1 from rounding and nothing more.
    check(`ROI case ${c} (${ROI_ORDER.length} ROIs) vs cv2`, max <= 1,
      `mean|d|=${(sum / n).toFixed(3)}  max|d|=${max}  >1: ${(100 * off / n).toFixed(2)}%`);
  }
}

// ---- 2. Preprocessing vs rPPG-Toolbox -----------------------------------
const jsInputs = [];
for (let c = 0; c < manifest.pre_cases; c++) {
  const frames = load(`pre_frames_${c}`).data;
  const times = load(`pre_times_${c}`).data;
  const { input } = buildInput({ frames, times });
  jsInputs.push(input);
  const ref = load(`pre_input_${c}`).data;
  const d = maxAbsDiff(input, ref);
  check(`preprocess case ${c} vs toolbox normalisation`, d < 1e-3, `max|d|=${d.toExponential(2)}`);
}

// ---- 3. ONNX Runtime Web vs PyTorch checkpoint --------------------------
{
  const modelPath = fileURLToPath(new URL('../models/litepulse.onnx', import.meta.url));
  const session = await ort.InferenceSession.create(readFileSync(modelPath), { executionProviders: ['wasm'] });
  for (let c = 0; c < manifest.pre_cases; c++) {
    for (const [label, input] of [['toolbox input', load(`pre_input_${c}`).data], ['JS input', jsInputs[c]]]) {
      const out = await session.run({ input: new ort.Tensor('float32', Float32Array.from(input), [1, 180, 18, 36, 36]) });
      const dB = maxAbsDiff(out.bvp.data, load(`model_bvp_${c}`).data);
      const dW = maxAbsDiff(out.roi_weights.data, load(`model_weights_${c}`).data);
      const dE = maxAbsDiff(out.energy_ratio.data, load(`model_energy_${c}`).data);
      check(`model case ${c}, ${label} vs .pth`, dB < 1e-3 && dW < 1e-4 && dE < 1e-4,
        `bvp ${dB.toExponential(1)}  weights ${dW.toExponential(1)}  energy ${dE.toExponential(1)}`);
    }
  }
}

// ---- 4. Heart rate vs scipy ---------------------------------------------
{
  const calc = new HeartRateCalculator();
  for (let c = 0; c < manifest.hr_cases; c++) {
    const res = calc.calculate(load(`hr_signal_${c}`).data);
    const ref = manifest.hr[c];
    const dWave = maxAbsDiff(res.waveform, load(`hr_waveform_${c}`).data);
    const ok = Math.abs(res.bpm - ref.bpm) < 1e-6 && Math.abs(res.peakToNoise - ref.pnr) < 1e-6 * ref.pnr &&
      Math.abs(res.bandEnergyRatio - ref.ratio) < 1e-6 && dWave < 1e-6;
    check(`HR case ${c} vs scipy`, ok,
      `bpm ${res.bpm.toFixed(2)} (ref ${ref.bpm.toFixed(2)})  pnr ${res.peakToNoise.toFixed(3)}  ` +
      `ratio ${res.bandEnergyRatio.toFixed(3)}  waveform ${dWave.toExponential(1)}`);
  }
}

const failed = results.filter((ok) => !ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
