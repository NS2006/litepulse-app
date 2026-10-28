// Port of measurement/MeasurementController.kt.
// Main thread: pre-check, ROI extraction, buffering.
// Worker: resample -> model -> HR (see inference-worker.js).

import { PreCheckEvaluator, PreCheckStatus } from './precheck.js';
import { Preprocessor } from './preprocess.js';
import { extractRoiFrame } from './roi.js';

const STABLE_FRAMES_TO_START = 15;
const FAIL_TOLERANCE = 5;
const INFERENCE_INTERVAL_SEC = 2.0;
const NOTICE_CLEAR_FRAMES = 30;
const MEDIAN_WINDOWS = 5;
const MAX_TREND_POINTS = 300;
const WEAK_FPS = 15;
export const REQUIRED_SECONDS = 6;

const EQUAL_WEIGHTS = [1 / 3, 1 / 3, 1 / 3];

export class MeasurementController {
  /**
   * @param worker  inference worker (inference-worker.js)
   * @param onResult optional callback after each inference result
   */
  constructor(worker, onResult = null) {
    this.worker = worker;
    this.onResult = onResult;
    this.evaluator = new PreCheckEvaluator();
    this.preprocessor = new Preprocessor();

    this.generation = 0;
    this.inferenceRunning = false;
    this.failStreak = 0;
    this.stableStreak = 0;
    this.sessionStart = null;
    this.lastInferenceSec = -10;
    this.lastTrendSec = -10;

    this.heartRate = null;
    this.weights = EQUAL_WEIGHTS;
    this.notice = null;
    this.debug = '';
    this.polygons = null;
    this.recent = [];
    this.bpmMedian = null;
    this.bpmLow = null;
    this.bpmHigh = null;
    this.trend = [];
    this.lastTimings = null;

    worker.addEventListener('message', ({ data }) => this.onWorkerMessage(data));
  }

  /**
   * @param imageData  RGBA frame (ImageData)
   * @param landmarks  MediaPipe face landmarks of the first face, or null
   * @param timeSec    frame capture time in seconds
   * @param previews   optional {forehead, left_cheek, right_cheek} ImageData(36, 36)
   */
  process(imageData, landmarks, timeSec, previews = null) {
    if (this.sessionStart === null) this.sessionStart = timeSec;
    const nowSec = timeSec - this.sessionStart;

    if (!landmarks) {
      this.restart('wajah hilang');
      this.evaluator.reset();
      this.failStreak = 0;
      this.stableStreak = 0;
      this.polygons = null;
      return this.snapshot(false, PreCheckStatus.NO_FACE);
    }

    const check = this.evaluator.evaluate(landmarks, imageData);
    const inWindow = this.preprocessor.size() > 0;

    if (check.passed) {
      this.failStreak = 0;
      this.stableStreak++;
    } else {
      this.failStreak++;
      this.stableStreak = 0;
    }

    if (!check.passed && (check.largeMotion || this.failStreak >= FAIL_TOLERANCE || !inWindow)) {
      if (inWindow) {
        this.restart(check.largeMotion ? 'gerakan terlalu besar' : check.status.message.toLowerCase());
      }
      return this.snapshot(true, check.status);
    }

    if (!inWindow && this.stableStreak < STABLE_FRAMES_TO_START) {
      return this.snapshot(true, PreCheckStatus.HOLD);
    }

    const { frame, polygons } = extractRoiFrame(imageData, landmarks, previews);
    this.polygons = polygons;
    this.preprocessor.push(frame, timeSec);

    if (this.preprocessor.size() >= NOTICE_CLEAR_FRAMES) this.notice = null;

    if (!this.inferenceRunning &&
      this.preprocessor.isReady() &&
      nowSec - this.lastInferenceSec >= INFERENCE_INTERVAL_SEC) {
      this.startInference(nowSec);
    }

    return this.snapshot(true, check.status);
  }

  startInference(nowSec) {
    const window = this.preprocessor.snapshot();
    if (!window) return;
    this.inferenceRunning = true;
    this.lastInferenceSec = nowSec;
    this.pending = { fps: this.preprocessor.measuredFps(), nowSec };
    this.worker.postMessage(
      { type: 'infer', gen: this.generation, frames: window.frames, times: window.times },
      [window.frames.buffer, window.times.buffer],
    );
  }

  onWorkerMessage(data) {
    if (data.type === 'result') {
      this.inferenceRunning = false;
      if (data.gen === this.generation) this.applyResult(data);
    } else if (data.type === 'error' && data.gen !== undefined) {
      this.inferenceRunning = false;
      this.notice = 'Inferensi gagal: ' + data.message;
    }
  }

  applyResult({ hr, weights, timings }) {
    const { fps, nowSec } = this.pending;
    this.heartRate = hr;
    this.lastTimings = timings;
    const raw = weights || EQUAL_WEIGHTS;
    this.weights = this.weights.map((old, i) => 0.5 * old + 0.5 * raw[i]);

    if (hr.quality !== 'POOR') {
      this.recent.push(hr.bpm);
      while (this.recent.length > MEDIAN_WINDOWS) this.recent.shift();
    }
    this.updateMedian(nowSec);

    this.debug = `PNR ${hr.peakToNoise.toFixed(1)} | band ${hr.bandEnergyRatio.toFixed(2)} | ` +
      `model ${hr.modelBandRatio != null ? hr.modelBandRatio.toFixed(2) : '-'} | ` +
      `skor ${hr.score.toFixed(2)} | ${fps.toFixed(0)} fps | ` +
      `inferensi ${timings.totalMs.toFixed(0)} ms`;
    if (this.onResult) this.onResult(hr, timings);
  }

  updateMedian(nowSec) {
    if (!this.recent.length) {
      this.bpmMedian = this.bpmLow = this.bpmHigh = null;
      return;
    }
    const sorted = [...this.recent].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    this.bpmMedian = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    if (sorted.length >= 3) {
      this.bpmLow = sorted[0];
      this.bpmHigh = sorted[sorted.length - 1];
    } else {
      this.bpmLow = this.bpmHigh = null;
    }

    if (nowSec - this.lastTrendSec >= 1) {
      this.lastTrendSec = nowSec;
      this.trend.push(this.bpmMedian);
      if (this.trend.length > MAX_TREND_POINTS) this.trend.shift();
    }
  }

  restart(reason) {
    if (this.preprocessor.size() > 0) this.notice = `Pengukuran diulang: ${reason}`;
    this.generation++; // discard any in-flight inference
    this.preprocessor.clear();
    this.weights = EQUAL_WEIGHTS;
    this.heartRate = null;
    this.recent = [];
    this.bpmMedian = this.bpmLow = this.bpmHigh = null;
    this.lastInferenceSec = -10;
  }

  snapshot(faceDetected, status) {
    const elapsed = this.preprocessor.elapsedSeconds();
    const ready = this.preprocessor.isReady();
    const fps = this.preprocessor.measuredFps();

    let fpsNotice = null;
    if (elapsed >= REQUIRED_SECONDS && !ready) {
      fpsNotice = `FPS kamera terlalu rendah (${fps.toFixed(0)}), tidak bisa mengukur`;
    } else if (ready && fps < WEAK_FPS) {
      fpsNotice = `FPS rendah (${fps.toFixed(0)}), akurasi bisa menurun`;
    }

    return {
      faceDetected,
      status,
      bufferedFrames: this.preprocessor.size(),
      windowSeconds: elapsed,
      windowReady: ready,
      fps,
      heartRate: this.heartRate,
      bpmMedian: this.bpmMedian,
      bpmLow: this.bpmLow,
      bpmHigh: this.bpmHigh,
      trend: this.trend,
      roiWeights: this.weights,
      polygons: faceDetected ? this.polygons : null,
      notice: this.notice || fpsNotice,
      debug: this.debug,
      isMeasuring: this.preprocessor.size() > 0,
      progress: Math.min(Math.max(elapsed / REQUIRED_SECONDS, 0), 1),
      secondsRemaining: Math.max(Math.ceil(REQUIRED_SECONDS - elapsed), 0),
    };
  }
}
