// Pinned library versions. MediaPipe matches the Android app (tasks-vision 0.10.32)
// so the face landmarks, and therefore the ROIs, are the same.
export const MEDIAPIPE_VERSION = '0.10.32';
export const MEDIAPIPE_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
export const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

export const ORT_VERSION = '1.30.0';
export const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

/** Frames are downscaled so the long side is at most this (UBFC/PURE are 640x480). */
export const MAX_PROCESS_SIDE = 640;

/** Android download: the LitePulse.apk attached to the repository's newest GitHub Release. */
export const APK_URL = 'https://github.com/NS2006/litepulse-app/releases/latest/download/LitePulse.apk';
