# LitePulse — browser version

A static page that measures heart rate from the front camera, entirely in the
browser. It runs the same pipeline as the Android app:

| Step | Android app | Browser |
|---|---|---|
| Face landmarks | MediaPipe tasks-vision 0.10.32 | MediaPipe tasks-vision web 0.10.32 |
| ROI masks (forehead, both cheeks) | `RoiExtractor.kt` | `js/roi.js` (cv2 polygon fill + INTER_AREA, exact) |
| Resample + DiffNormalized/Standardized | `Preprocessor.kt` | `js/preprocess.js` |
| Model | `litepulse.ptl` (PyTorch Lite) | `models/litepulse.onnx` (ONNX Runtime Web, WASM, in a worker) |
| Detrend, Butterworth, FFT peak, quality | `HeartRateCalculator.kt` | `js/heartrate.js` |

Video never leaves the device. No build step: plain ES modules, libraries from jsDelivr
(pinned in `js/config.js`).

## Run locally

```bash
python verify/serve.py
```

Open http://localhost:8765 (add `?debug` to see PNR, band ratio, fps and inference time).
The camera only works on `localhost` or HTTPS, not on a LAN IP.

To try it on an Android phone over USB, keep the server running and:

```bash
adb reverse tcp:8765 tcp:8765
```

then open http://localhost:8765 in Chrome on the phone.

## Deploy

`.github/workflows/pages.yml` publishes `index.html`, `css/`, `js/` and `models/` to GitHub
Pages on every push to `main` that touches `web/`. Pages serves over HTTPS, so the camera
works. One-time setup: repository Settings → Pages → Source: **GitHub Actions**.

The "Unduh APK" link (`APK_URL` in `js/config.js`) points at
`releases/latest/download/LitePulse.apk`, so it always serves the newest GitHub Release
that has a file named exactly `LitePulse.apk`. See the root README for publishing a release.

## Regenerate the model

From `model/` (the checkpoint is in `model/pth_model/`):

```bash
pip install onnx onnxruntime
python litepulse_onnx_generator.py
```

Copy the resulting `litepulse.onnx` to `web/models/`. The script stops with an
error if the saved file does not match the PyTorch model.

`models/face_landmarker.task` is the same file as `android/app/src/main/assets/face_landmarker.task`.

## Verify against Python

`verify/make_reference.py` computes reference outputs the way training does (cv2 ROI masks,
rPPG-Toolbox normalisation, the `.pth` checkpoint, scipy filtering). `verify/verify.mjs` runs
the page's own modules and the `.onnx` file on the same inputs and compares.

```bash
python verify/make_reference.py verify/out
npm install
node verify/verify.mjs verify/out
```

Expected: `24/24 checks passed` (ROI within ±1 grey level of cv2, preprocessing within 1e-4,
model outputs within 1e-5 of the `.pth`, heart rate identical).

`verify/e2e.html` runs the whole page pipeline (MediaPipe, ROI, worker, HR) on a still photo
with a synthetic pulse at known rates, no camera needed. Put a photo of your own face at
`verify/tmp/face.png` (ignored by git), start the server, then open
http://localhost:8765/verify/e2e.html?bpm=60,75,90,110.
