// Port of model/LitePulsePreprocessor.kt: buffers timestamped ROI frames,
// resamples the last 6 s to a uniform 30 fps grid (181 frames) and builds the
// [180, 18, 36, 36] model input:
//
//   d = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + 1e-7);  d = d / (d.std() + 1e-7)
//   s = (x[:-1] - x.mean()) / (x.std() + 1e-7)
//   input = concatenate([d, s], axis=1)     # DiffNormalized, then Standardized

import { PLANE, VALUES_PER_FRAME } from './roi.js';

export const TEMPORAL_FRAMES = 180;
export const REQUIRED_RAW_FRAMES = 181;
export const INPUT_CHANNELS = 18;
export const TARGET_FPS = 30;
export const WINDOW_SECONDS = TEMPORAL_FRAMES / TARGET_FPS; // 6.0 s
const MAX_BUFFER_SECONDS = 7.5;
/** ~6 fps over 6 s. Below this the signal cannot be reconstructed. */
const MIN_RAW_FRAMES = 36;

const PIXELS_PER_FRAME = VALUES_PER_FRAME; // 3 ROIs x 3 channels x 36 x 36

export class Preprocessor {
  constructor() {
    this.frames = [];
    this.times = []; // seconds
  }

  push(frame, timeSec) {
    if (frame.length !== PIXELS_PER_FRAME) {
      throw new Error(`Expected ${PIXELS_PER_FRAME} values, got ${frame.length}`);
    }
    if (this.times.length && timeSec <= this.times[this.times.length - 1]) return;

    this.frames.push(frame);
    this.times.push(timeSec);

    while (this.times.length > 2 && this.times[this.times.length - 1] - this.times[0] > MAX_BUFFER_SECONDS) {
      this.frames.shift();
      this.times.shift();
    }
  }

  clear() {
    this.frames = [];
    this.times = [];
  }

  size() {
    return this.frames.length;
  }

  elapsedSeconds() {
    return this.times.length < 2 ? 0 : this.times[this.times.length - 1] - this.times[0];
  }

  measuredFps() {
    if (this.times.length < 2) return 0;
    return (this.times.length - 1) / (this.times[this.times.length - 1] - this.times[0]);
  }

  isReady() {
    return this.times.length >= MIN_RAW_FRAMES &&
      this.times[this.times.length - 1] - this.times[0] >= WINDOW_SECONDS;
  }

  /** Packs the buffer into transferable arrays for the inference worker. Null if not ready. */
  snapshot() {
    if (!this.isReady()) return null;
    const n = this.frames.length;
    const frames = new Float32Array(n * PIXELS_PER_FRAME);
    this.frames.forEach((f, i) => frames.set(f, i * PIXELS_PER_FRAME));
    return { frames, times: Float64Array.from(this.times) };
  }
}

/** Pure: resample + normalise a snapshot. Returns { input, roiGreen }. */
export function buildInput(snapshot) {
  const green = Array.from({ length: 3 }, () => new Float32Array(REQUIRED_RAW_FRAMES));
  const resampled = resample(snapshot, green);
  return { input: normalize(resampled), roiGreen: green };
}

function resample({ frames, times }, green) {
  const tEnd = times[times.length - 1];
  const tStart = tEnd - WINDOW_SECONDS;
  const out = [];

  let j = 0;
  for (let k = 0; k < REQUIRED_RAW_FRAMES; k++) {
    const t = tStart + k / TARGET_FPS;
    while (j < times.length - 2 && times[j + 1] < t) j++;

    const t0 = times[j];
    const t1 = times[j + 1];
    const w = Math.fround(Math.min(Math.max((t - t0) / (t1 - t0), 0), 1));

    const a = j * PIXELS_PER_FRAME;
    const b = (j + 1) * PIXELS_PER_FRAME;
    const dst = new Float32Array(PIXELS_PER_FRAME);
    for (let i = 0; i < PIXELS_PER_FRAME; i++) {
      dst[i] = frames[a + i] + (frames[b + i] - frames[a + i]) * w;
    }
    out.push(dst);

    for (let r = 0; r < 3; r++) {
      const base = r * 3 * PLANE + PLANE; // green plane
      let sum = 0;
      let count = 0;
      for (let p = 0; p < PLANE; p++) {
        const v = dst[base + p];
        if (v !== 0) {
          sum += v;
          count++;
        }
      }
      green[r][k] = count > 0 ? sum / count : 0;
    }
  }
  return out;
}

function normalize(frames) {
  const totalValues = REQUIRED_RAW_FRAMES * PIXELS_PER_FRAME;

  let sum = 0;
  for (const frame of frames) for (let i = 0; i < frame.length; i++) sum += frame[i];
  const mean = sum / totalValues;

  let squared = 0;
  for (const frame of frames) {
    for (let i = 0; i < frame.length; i++) {
      const d = frame[i] - mean;
      squared += d * d;
    }
  }
  const std = Math.sqrt(squared / totalValues);

  const diff = new Float32Array(TEMPORAL_FRAMES * PIXELS_PER_FRAME);
  let diffSum = 0;
  for (let t = 0; t < TEMPORAL_FRAMES; t++) {
    const current = frames[t];
    const next = frames[t + 1];
    const base = t * PIXELS_PER_FRAME;
    for (let i = 0; i < PIXELS_PER_FRAME; i++) {
      const value = Math.fround((next[i] - current[i]) / Math.fround(next[i] + current[i] + 1e-7));
      diff[base + i] = value;
      diffSum += value;
    }
  }

  const diffMean = diffSum / diff.length;
  let diffSquared = 0;
  for (let i = 0; i < diff.length; i++) {
    const d = diff[i] - diffMean;
    diffSquared += d * d;
  }
  const diffStd = Math.sqrt(diffSquared / diff.length);

  const diffScale = Math.fround(diffStd + 1e-7);
  const stdScale = Math.fround(std + 1e-7);
  const meanF = Math.fround(mean);

  const output = new Float32Array(TEMPORAL_FRAMES * INPUT_CHANNELS * PLANE);
  for (let t = 0; t < TEMPORAL_FRAMES; t++) {
    const source = frames[t];
    const diffBase = t * PIXELS_PER_FRAME;
    for (let i = 0; i < PIXELS_PER_FRAME; i++) {
      const channel = Math.floor(i / PLANE);
      const pixel = i % PLANE;
      output[(t * INPUT_CHANNELS + channel) * PLANE + pixel] = diff[diffBase + i] / diffScale;
      output[(t * INPUT_CHANNELS + 9 + channel) * PLANE + pixel] = (source[i] - meanF) / stdScale;
    }
  }
  return output;
}
