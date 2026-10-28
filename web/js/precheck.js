// Port of measurement/PreCheck.kt: face size, position, orientation,
// brightness and motion checks before (and during) a measurement.

export const PreCheckStatus = {
  NO_FACE: { message: 'Wajah tidak terdeteksi', hint: 'Hadapkan wajah ke kamera' },
  TOO_DARK: { message: 'Terlalu gelap', hint: 'Cari tempat dengan cahaya lebih terang' },
  TOO_BRIGHT: { message: 'Terlalu terang', hint: 'Hindari cahaya langsung yang menyilaukan' },
  NOT_FACING: { message: 'Hadap kamera', hint: 'Luruskan wajah menghadap layar' },
  NOT_CENTERED: { message: 'Posisikan wajah di tengah', hint: 'Geser wajah ke tengah layar' },
  TOO_FAR: { message: 'Dekatkan wajah', hint: 'Wajah terlalu kecil di layar' },
  TOO_CLOSE: { message: 'Mundur sedikit', hint: 'Seluruh wajah harus terlihat' },
  MOVING: { message: 'Tahan posisi', hint: 'Jangan bergerak selama pengukuran' },
  HOLD: { message: 'Tahan posisi', hint: 'Pengukuran dimulai sebentar lagi' },
  READY: { message: 'Siap', hint: '' },
};

// All values are starting points: tune them with real recordings.
const MIN_FACE_WIDTH = 0.35; // fraction of frame width
const MAX_FACE_WIDTH = 0.9;
const CENTER_TOLERANCE_X = 0.2;
const CENTER_TOLERANCE_Y = 0.25;
const YAW_TOLERANCE = 0.15; // nose position between eye corners
const DARK_LUMA = 60;
const BRIGHT_LUMA = 235;
const STILL_MOTION = 0.015; // smoothed, per frame
const LARGE_MOTION = 0.05; // single frame -> restart window

const RIGHT_EYE_OUTER = 33;
const LEFT_EYE_OUTER = 263;
const NOSE_TIP = 1;

export class PreCheckEvaluator {
  constructor() {
    this.reset();
  }

  reset() {
    this.previous = null;
    this.smoothedMotion = 0;
  }

  evaluate(landmarks, imageData) {
    const n = landmarks.length;
    const current = new Float32Array(n * 2);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < n; i++) {
      const { x, y } = landmarks[i];
      current[i * 2] = x;
      current[i * 2 + 1] = y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const faceWidth = maxX - minX;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // ---- motion between frames (landmark displacement) ----
    let motion = 0;
    const prev = this.previous;
    if (prev && prev.length === current.length && faceWidth > 0) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += Math.hypot(current[i * 2] - prev[i * 2], current[i * 2 + 1] - prev[i * 2 + 1]);
      }
      motion = sum / n / faceWidth;
    }
    this.previous = current;
    this.smoothedMotion = 0.5 * this.smoothedMotion + 0.5 * motion;
    const largeMotion = motion > LARGE_MOTION;

    // ---- head orientation (nose between the outer eye corners) ----
    let yaw = 0.5;
    if (n > LEFT_EYE_OUTER) {
      const a = landmarks[RIGHT_EYE_OUTER].x;
      const b = landmarks[LEFT_EYE_OUTER].x;
      const span = b - a;
      if (Math.abs(span) > 1e-4) yaw = (landmarks[NOSE_TIP].x - a) / span;
    }

    let status;
    if (faceWidth < MIN_FACE_WIDTH) {
      status = PreCheckStatus.TOO_FAR;
    } else if (faceWidth > MAX_FACE_WIDTH || minX < 0.01 || maxX > 0.99 || minY < 0.01 || maxY > 0.99) {
      status = PreCheckStatus.TOO_CLOSE;
    } else if (Math.abs(centerX - 0.5) > CENTER_TOLERANCE_X || Math.abs(centerY - 0.5) > CENTER_TOLERANCE_Y) {
      status = PreCheckStatus.NOT_CENTERED;
    } else if (Math.abs(yaw - 0.5) > YAW_TOLERANCE) {
      status = PreCheckStatus.NOT_FACING;
    } else {
      const luma = faceBrightness(imageData, minX, minY, faceWidth, maxY - minY);
      if (luma < DARK_LUMA) status = PreCheckStatus.TOO_DARK;
      else if (luma > BRIGHT_LUMA) status = PreCheckStatus.TOO_BRIGHT;
      else if (largeMotion || this.smoothedMotion > STILL_MOTION) status = PreCheckStatus.MOVING;
      else status = PreCheckStatus.READY;
    }

    return { status, motion, largeMotion, passed: status === PreCheckStatus.READY };
  }
}

/** Mean luma (0..255) on a 10x10 grid in the central part of the face box. */
function faceBrightness(imageData, minX, minY, width, height) {
  const { data, width: w, height: h } = imageData;
  const grid = 10;
  let sum = 0;
  for (let iy = 0; iy < grid; iy++) {
    for (let ix = 0; ix < grid; ix++) {
      const nx = minX + width * (0.2 + (0.6 * (ix + 0.5)) / grid);
      const ny = minY + height * (0.2 + (0.6 * (iy + 0.5)) / grid);
      const px = Math.min(Math.max(Math.trunc(nx * w), 0), w - 1);
      const py = Math.min(Math.max(Math.trunc(ny * h), 0), h - 1);
      const i = (py * w + px) * 4;
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
  }
  return sum / (grid * grid);
}
