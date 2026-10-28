// Port of model/LitePulseHeartRateCalculator.kt: the rPPG-Toolbox post-processing
// (cumsum -> smoothness-prior detrend, lambda=100 -> Butterworth 0.75-2.5 Hz
// filtfilt -> FFT peak) plus the app's signal-quality score.

export const SignalQuality = {
  GOOD: { key: 'GOOD', label: 'Baik' },
  FAIR: { key: 'FAIR', label: 'Cukup' },
  POOR: { key: 'POOR', label: 'Buruk' },
};

const FS = 30;
const SIGNAL_LENGTH = 180;
const LOW_HZ = 0.75;
const HIGH_HZ = 2.5;
const DETREND_LAMBDA = 100;

const FFT_SIZE = 16384;
const ROI_FFT_SIZE = 4096;
const RATIO_FFT_SIZE = 2048;
const RATIO_MIN_HZ = 0.1;

// ---- Quality thresholds (still uncalibrated, tune with the on-screen debug line) ----
// A *perfect* sinusoid in a 6 s window has PNR ~6-7, so real signals sit far lower.
const PNR_POOR = 1.8;
const PNR_GOOD = 3.5;
const RATIO_POOR = 0.15;
const RATIO_GOOD = 0.55;
const MODEL_RATIO_POOR = 0.15;
const MODEL_RATIO_GOOD = 0.5;
const SCORE_GOOD = 0.55;
const SCORE_FAIR = 0.25;

// scipy.signal.butter(1, [0.75/15, 2.5/15], "bandpass")
const B = [0.15635952, 0.0, -0.15635952];
const A = [1.0, -1.61758769, 0.68728096];
const ZI = [-0.15635952, -0.15635952];

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

export class HeartRateCalculator {
  constructor() {
    this.detrendMatrix = buildDetrendMatrix(SIGNAL_LENGTH, DETREND_LAMBDA);
  }

  calculate(prediction, modelBandRatio = null) {
    if (prediction.length !== SIGNAL_LENGTH) {
      throw new Error(`Expected ${SIGNAL_LENGTH} values, got ${prediction.length}`);
    }

    const detrended = matVec(this.detrendMatrix, cumulativeSum(prediction));
    const filtered = filtfilt(B, A, detrended);

    const peak = findPeak(magnitudeSpectrum(filtered, FFT_SIZE), FFT_SIZE);
    const ratio = bandEnergyRatio(detrended);

    const pnrScore = clamp01((peak.pnr - PNR_POOR) / (PNR_GOOD - PNR_POOR));
    const ratioScores = [clamp01((ratio - RATIO_POOR) / (RATIO_GOOD - RATIO_POOR))];
    if (modelBandRatio != null) {
      ratioScores.push(clamp01((modelBandRatio - MODEL_RATIO_POOR) / (MODEL_RATIO_GOOD - MODEL_RATIO_POOR)));
    }
    const score = 0.6 * pnrScore + (0.4 * ratioScores.reduce((s, v) => s + v, 0)) / ratioScores.length;

    let quality = SignalQuality.POOR;
    if (score >= SCORE_GOOD) quality = SignalQuality.GOOD;
    else if (score >= SCORE_FAIR) quality = SignalQuality.FAIR;

    return {
      bpm: peak.frequency * 60,
      peakToNoise: peak.pnr,
      bandEnergyRatio: ratio,
      score,
      quality: quality.key,
      modelBandRatio,
      waveform: Float32Array.from(filtered),
      peaks: detectPeaks(filtered),
    };
  }

  /** Peak-to-noise of a raw ROI intensity trace (>= 180 samples). */
  roiPnr(trace) {
    if (trace.length < SIGNAL_LENGTH) return 1;
    const offset = trace.length - SIGNAL_LENGTH;
    const x = new Float64Array(SIGNAL_LENGTH);
    for (let i = 0; i < SIGNAL_LENGTH; i++) x[i] = trace[offset + i];
    const filtered = filtfilt(B, A, matVec(this.detrendMatrix, x));
    return findPeak(magnitudeSpectrum(filtered, ROI_FFT_SIZE), ROI_FFT_SIZE).pnr;
  }
}

/** Local maxima above zero, at least 0.33 s apart (max 180 bpm). */
function detectPeaks(signal) {
  const minDistance = Math.trunc(FS * 0.33);
  const peaks = [];
  for (let i = 2; i < signal.length - 2; i++) {
    const v = signal[i];
    if (v <= 0 || v <= signal[i - 1] || v < signal[i + 1]) continue;
    const last = peaks.length ? peaks[peaks.length - 1] : null;
    if (last === null || i - last >= minDistance) peaks.push(i);
    else if (v > signal[last]) peaks[peaks.length - 1] = i;
  }
  return peaks;
}

function findPeak(spectrum, fftSize) {
  const resolution = FS / fftSize;
  const minBin = Math.ceil(LOW_HZ / resolution);
  const maxBin = Math.floor(HIGH_HZ / resolution);
  let bestBin = minBin;
  let best = -Infinity;
  let sum = 0;
  for (let bin = minBin; bin <= maxBin; bin++) {
    const m = spectrum[bin];
    sum += m;
    if (m > best) {
      best = m;
      bestBin = bin;
    }
  }
  const mean = sum / (maxBin - minBin + 1);
  return { frequency: bestBin * resolution, pnr: best / (mean + 1e-12) };
}

function bandEnergyRatio(signal) {
  const spectrum = magnitudeSpectrum(signal, RATIO_FFT_SIZE);
  const resolution = FS / RATIO_FFT_SIZE;
  const firstBin = Math.ceil(RATIO_MIN_HZ / resolution);
  const bandLo = Math.ceil(LOW_HZ / resolution);
  const bandHi = Math.floor(HIGH_HZ / resolution);
  let total = 0;
  let band = 0;
  for (let bin = firstBin; bin < spectrum.length; bin++) {
    const p = spectrum[bin] * spectrum[bin];
    total += p;
    if (bin >= bandLo && bin <= bandHi) band += p;
  }
  return total > 0 ? band / total : 0;
}

function cumulativeSum(input) {
  const out = new Float64Array(input.length);
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    sum += input[i];
    out[i] = sum;
  }
  return out;
}

/** DETREND = I - inv(I + lambda^2 * D^T D) */
function buildDetrendMatrix(size, lambda) {
  const dtD = Array.from({ length: size }, () => new Float64Array(size));
  const kernel = [1, -2, 1];
  for (let i = 0; i < size - 2; i++) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) dtD[i + r][i + c] += kernel[r] * kernel[c];
    }
  }
  const matrix = Array.from({ length: size }, (_, row) => {
    const values = new Float64Array(size);
    for (let col = 0; col < size; col++) values[col] = (row === col ? 1 : 0) + lambda * lambda * dtD[row][col];
    return values;
  });
  const inverse = invertMatrix(matrix);
  return inverse.map((values, row) => values.map((v, col) => (row === col ? 1 : 0) - v));
}

function matVec(matrix, vector) {
  const out = new Float64Array(vector.length);
  for (let row = 0; row < matrix.length; row++) {
    const m = matrix[row];
    let sum = 0;
    for (let col = 0; col < vector.length; col++) sum += m[col] * vector[col];
    out[row] = sum;
  }
  return out;
}

/** scipy.signal.filtfilt (pad="odd", padlen = 3 * max(len(a), len(b))). */
function filtfilt(b, a, input) {
  const padLength = 3 * Math.max(a.length, b.length);
  if (input.length <= padLength) throw new Error(`Input must contain more than ${padLength} samples`);

  const padded = oddPad(input, padLength);
  const forward = lfilter(b, a, padded, [ZI[0] * padded[0], ZI[1] * padded[0]]);
  forward.reverse();
  const backward = lfilter(b, a, forward, [ZI[0] * forward[0], ZI[1] * forward[0]]);
  backward.reverse();
  return backward.slice(padLength, backward.length - padLength);
}

/** Direct-form II transposed, second order. */
function lfilter(b, a, input, initialState) {
  const out = new Float64Array(input.length);
  let z1 = initialState[0];
  let z2 = initialState[1];
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const y = b[0] * x + z1;
    z1 = b[1] * x - a[1] * y + z2;
    z2 = b[2] * x - a[2] * y;
    out[i] = y;
  }
  return out;
}

function oddPad(input, padLength) {
  const out = new Float64Array(input.length + 2 * padLength);
  out.set(input, padLength);
  for (let i = 0; i < padLength; i++) out[padLength - 1 - i] = 2 * input[0] - input[i + 1];
  const last = input.length - 1;
  for (let i = 0; i < padLength; i++) out[padLength + input.length + i] = 2 * input[last] - input[last - 1 - i];
  return out;
}

function magnitudeSpectrum(input, fftSize) {
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  real.set(input);
  fft(real, imag);
  const out = new Float64Array(fftSize / 2 + 1);
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  return out;
}

function fft(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);
    const half = length / 2;
    for (let start = 0; start < n; start += length) {
      let wReal = 1;
      let wImag = 0;
      for (let i = 0; i < half; i++) {
        const even = start + i;
        const odd = even + half;
        const oddReal = real[odd] * wReal - imag[odd] * wImag;
        const oddImag = real[odd] * wImag + imag[odd] * wReal;
        real[odd] = real[even] - oddReal;
        imag[odd] = imag[even] - oddImag;
        real[even] += oddReal;
        imag[even] += oddImag;
        const nextWReal = wReal * wLenReal - wImag * wLenImag;
        wImag = wReal * wLenImag + wImag * wLenReal;
        wReal = nextWReal;
      }
    }
  }
}

function invertMatrix(input) {
  const n = input.length;
  const aug = input.map((row, r) => {
    const values = new Float64Array(n * 2);
    values.set(row);
    values[n + r] = 1;
    return values;
  });

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivotRow][col])) pivotRow = row;
    }
    if (Math.abs(aug[pivotRow][col]) <= 1e-12) throw new Error('Matrix is singular');
    if (pivotRow !== col) [aug[col], aug[pivotRow]] = [aug[pivotRow], aug[col]];

    const pivotValues = aug[col];
    const pivot = pivotValues[col];
    for (let j = 0; j < n * 2; j++) pivotValues[j] /= pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      if (Math.abs(factor) < 1e-15) continue;
      const values = aug[row];
      for (let j = 0; j < n * 2; j++) values[j] -= factor * pivotValues[j];
    }
  }
  return aug.map((values) => values.slice(n));
}
