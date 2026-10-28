// Frame pipeline shared by the page and the verification harness:
// frame -> MediaPipe Face Landmarker -> MeasurementController.

import { MAX_PROCESS_SIDE, MEDIAPIPE_BUNDLE, MEDIAPIPE_WASM } from './config.js';
import { MeasurementController } from './controller.js';
import { PATCH_SIZE, ROI_ORDER } from './roi.js';

export async function createFaceLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import(MEDIAPIPE_BUNDLE);
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = (delegate) => ({
    baseOptions: {
      modelAssetPath: new URL('../models/face_landmarker.task', import.meta.url).href,
      delegate,
    },
    // VIDEO mode tracks the face between frames instead of re-detecting it.
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  try {
    return { landmarker: await FaceLandmarker.createFromOptions(fileset, options('GPU')), delegate: 'GPU' };
  } catch (error) {
    console.warn('GPU delegate unavailable, using CPU', error);
    return { landmarker: await FaceLandmarker.createFromOptions(fileset, options('CPU')), delegate: 'CPU' };
  }
}

/** Starts the inference worker and resolves once ONNX Runtime has loaded the model. */
export function createInferenceWorker() {
  const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
  const ready = new Promise((resolve, reject) => {
    const onMessage = ({ data }) => {
      if (data.type === 'ready') {
        worker.removeEventListener('message', onMessage);
        resolve(data);
      } else if (data.type === 'error' && data.gen === undefined) {
        worker.removeEventListener('message', onMessage);
        reject(new Error(data.message));
      }
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', (event) => reject(new Error(event.message || 'Worker failed to load')));
  });
  return { worker, ready };
}

export class FramePipeline {
  constructor(landmarker, worker, onResult = null) {
    this.landmarker = landmarker;
    this.controller = new MeasurementController(worker, onResult);
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.previews = Object.fromEntries(ROI_ORDER.map((name) => [name, new ImageData(PATCH_SIZE, PATCH_SIZE)]));
    this.lastTimestampMs = -1;
  }

  /**
   * @param source  video element, canvas or image holding the current frame
   * @param timeMs  capture time of the frame in milliseconds
   */
  process(source, sourceWidth, sourceHeight, timeMs) {
    const scale = Math.min(1, MAX_PROCESS_SIDE / Math.max(sourceWidth, sourceHeight));
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.ctx.drawImage(source, 0, 0, width, height);

    // VIDEO mode needs strictly increasing timestamps.
    const ts = Math.max(this.lastTimestampMs + 1, Math.round(timeMs));
    this.lastTimestampMs = ts;
    const result = this.landmarker.detectForVideo(this.canvas, ts);
    const landmarks = result.faceLandmarks && result.faceLandmarks.length ? result.faceLandmarks[0] : null;

    const imageData = this.ctx.getImageData(0, 0, width, height);
    const ui = this.controller.process(imageData, landmarks, timeMs / 1000, this.previews);
    ui.frameWidth = width;
    ui.frameHeight = height;
    ui.previews = this.previews;
    return ui;
  }
}
