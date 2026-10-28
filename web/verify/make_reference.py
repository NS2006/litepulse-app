"""
Reference outputs for verify.mjs, computed the way training does it.

    python make_reference.py <out_dir>

Writes test inputs and expected outputs for:
  1. ROI extraction    -- roi_extractor.py's cv2 pipeline (fillConvexPoly,
                          boundingRect, bitwise_and, INTER_AREA resize)
  2. Preprocessing     -- resample to 30 fps, then rPPG-Toolbox
                          diff_normalize_data / standardized_data
  3. Model             -- the PyTorch checkpoint (.pth), not the export
  4. Heart rate        -- toolbox detrend + scipy butter/filtfilt + FFT peak

Imports LitePulse.py and the checkpoint from ../../model.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy import signal, sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))
from LitePulse import LitePulse, LitePulseConfig  # noqa: E402

CKPT = os.path.join(HERE, "..", "..", "model", "pth_model",
                    "UBFC_UBFC_PURE_litepulseFDCMAEv2_epoch29.pth.zip")
LANDMARKS = {
    "forehead": [10, 108, 69, 104, 68, 71, 21, 54, 103, 67, 109, 10,
                 338, 297, 332, 284, 251, 301, 293, 334, 296, 336],
    "left_cheek": [116, 117, 118, 119, 100, 126, 142, 36, 205, 187, 123],
    "right_cheek": [345, 346, 347, 348, 329, 355, 371, 266, 425, 411, 352],
}
ROI_ORDER = ["forehead", "left_cheek", "right_cheek"]
PATCH = 36

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(7)
manifest = {}


def save(name, array):
    array = np.ascontiguousarray(array)
    array.tofile(os.path.join(out_dir, name + ".bin"))
    manifest[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}


# ---------------------------------------------------------------------------
# 1. ROI extraction (copy of roi_extractor.ROIExtractor._polygon_crop)
# ---------------------------------------------------------------------------
def polygon_crop(frame, pts, indices):
    h, w = frame.shape[:2]
    poly = pts[indices].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
    x, y, bw, bh = cv2.boundingRect(poly)
    x, y = max(x, 0), max(y, 0)
    bw, bh = max(bw, 1), max(bh, 1)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    patch = masked[y:y + bh, x:x + bw]
    if patch.size == 0:
        patch = np.zeros((PATCH, PATCH, 3), dtype=np.uint8)
    return cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_AREA)


H, W = 480, 640
yy, xx = np.mgrid[0:H, 0:W]
image = np.stack([
    120 + 60 * np.sin(xx / 37.0) + 30 * np.cos(yy / 23.0),
    90 + 50 * np.cos(xx / 29.0 + yy / 41.0),
    70 + 40 * np.sin((xx + yy) / 31.0),
], axis=-1) + rng.normal(0, 12, (H, W, 3))
image = np.clip(image, 0, 255).astype(np.uint8)
save("roi_image", image)

# (center x, center y, half width, half height) per case, in pixels.
# Covers large (area resize), small (<36 px, linear path) and off-frame ROIs.
cases = [(320, 200, 90, 40), (300, 260, 60, 50), (200, 300, 14, 12),
         (330, 240, 25, 70), (15, 20, 60, 45), (630, 470, 50, 40),
         (320, 240, 18, 18), (100, 400, 120, 33)]
landmark_sets, patches = [], []
for case in cases:
    lms = rng.uniform(0.3, 0.7, (478, 2)).astype(np.float32)
    for name in ROI_ORDER:
        cx, cy, hw, hh = case
        cx += rng.uniform(-10, 10)
        cy += rng.uniform(-10, 10)
        for i in LANDMARKS[name]:
            lms[i] = [(cx + rng.uniform(-hw, hw)) / W, (cy + rng.uniform(-hh, hh)) / H]
    pts = np.array([[float(x) * W, float(y) * H] for x, y in lms], dtype=np.float32)
    landmark_sets.append(lms)
    patches.append(np.stack([polygon_crop(image, pts, LANDMARKS[n]) for n in ROI_ORDER]))
save("roi_landmarks", np.stack(landmark_sets).astype(np.float32))   # (cases, 478, 2)
save("roi_patches", np.stack(patches).astype(np.uint8))             # (cases, 3, 36, 36, 3)

# ---------------------------------------------------------------------------
# 2. Preprocessing: irregular camera timestamps -> 30 fps -> toolbox normalisation
# ---------------------------------------------------------------------------
def diff_normalize_data(data):  # rPPG-Toolbox BaseLoader
    n, h, w, c = data.shape
    out = np.zeros((n - 1, h, w, c), dtype=np.float32)
    for j in range(n - 1):
        out[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
    out = out / np.std(out)
    out = np.append(out, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
    out[np.isnan(out)] = 0
    return out


def standardized_data(data):  # rPPG-Toolbox BaseLoader
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def resample(frames, times):
    t_end = times[-1]
    targets = t_end - 6.0 + np.arange(181) / 30.0
    j = np.clip(np.searchsorted(times, targets, side="left") - 1, 0, len(times) - 2)
    w = np.clip((targets - times[j]) / (times[j + 1] - times[j]), 0, 1).astype(np.float32)
    a, b = frames[j], frames[j + 1]
    return a + (b - a) * w[:, None]


inputs_built = []
for case in range(3):
    n = 210
    gaps = rng.uniform(1 / 40, 1 / 22, n - 1)          # ~22-40 fps, jittery
    times = np.concatenate([[100.0], 100.0 + np.cumsum(gaps)])
    t = times[:, None]
    base = rng.uniform(40, 200, (1, 3 * 3 * PATCH * PATCH))
    pulse = 1 + 0.01 * np.sin(2 * np.pi * (1.1 + 0.3 * case) * t)
    frames = (base * pulse + rng.normal(0, 3, (n, base.shape[1]))).astype(np.float32)
    frames[:, rng.choice(frames.shape[1], 400, replace=False)] = 0   # masked pixels
    frames = np.clip(np.round(frames), 0, 255).astype(np.float32)
    save(f"pre_frames_{case}", frames)
    save(f"pre_times_{case}", times.astype(np.float64))

    x = resample(frames, times)                                # (181, 3*3*36*36) roi, c, pixel
    x = x.reshape(181, 3, 3, PATCH, PATCH).transpose(0, 3, 4, 1, 2).reshape(181, PATCH, PATCH, 9)
    d = diff_normalize_data(x.copy())[:180]                     # (180, 36, 36, 9) roi-major RGB
    s = standardized_data(x.copy())[:180]
    model_in = np.concatenate([d, s], axis=-1).transpose(0, 3, 1, 2)  # (180, 18, 36, 36)
    save(f"pre_input_{case}", model_in.astype(np.float32))
    inputs_built.append(model_in.astype(np.float32))

# ---------------------------------------------------------------------------
# 3. Model: PyTorch checkpoint on the reference inputs
# ---------------------------------------------------------------------------
model = LitePulse(LitePulseConfig())
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()
bvps = []
for case, model_in in enumerate(inputs_built):
    with torch.no_grad():
        bvp, weights, _, energy = model(torch.from_numpy(model_in)[None], return_aux=True)
    save(f"model_bvp_{case}", bvp[0].numpy())
    save(f"model_weights_{case}", weights[0].numpy())
    save(f"model_energy_{case}", energy[0].numpy())
    bvps.append(bvp[0].numpy().astype(np.float64))

# ---------------------------------------------------------------------------
# 4. Heart rate from BVP (toolbox post-processing, scipy filters)
# ---------------------------------------------------------------------------
B, A = signal.butter(1, [0.75 / 15, 2.5 / 15], btype="bandpass")


def detrend(x, lam=100):  # rPPG-Toolbox post_process._detrend
    n = len(x)
    H_ = np.identity(n)
    D = sparse.spdiags(np.array([np.ones(n), -2 * np.ones(n), np.ones(n)]), [0, 1, 2], n - 2, n).toarray()
    return (H_ - np.linalg.inv(H_ + lam ** 2 * D.T @ D)) @ x


def peak(x, nfft):
    mag = np.abs(np.fft.rfft(x, nfft))
    res = 30.0 / nfft
    lo, hi = int(np.ceil(0.75 / res)), int(np.floor(2.5 / res))
    band = mag[lo:hi + 1]
    k = int(np.argmax(band))
    return (lo + k) * res * 60, band[k] / (band.mean() + 1e-12)


def band_ratio(x):
    mag = np.abs(np.fft.rfft(x, 2048)) ** 2
    res = 30.0 / 2048
    first, lo, hi = int(np.ceil(0.1 / res)), int(np.ceil(0.75 / res)), int(np.floor(2.5 / res))
    return mag[lo:hi + 1].sum() / mag[first:].sum()


hr_signals = bvps + [np.sin(2 * np.pi * 1.2 * np.arange(180) / 30)] + \
    [rng.normal(0, 1, 180) + 0.8 * np.sin(2 * np.pi * f * np.arange(180) / 30) for f in (0.9, 1.6, 2.3)]
hr_ref = []
for i, sig in enumerate(hr_signals):
    detrended = detrend(np.cumsum(sig))
    filtered = signal.filtfilt(B, A, detrended)
    bpm, pnr = peak(filtered, 16384)
    hr_ref.append({"bpm": bpm, "pnr": pnr, "ratio": band_ratio(detrended)})
    save(f"hr_signal_{i}", np.asarray(sig, dtype=np.float32))
    save(f"hr_waveform_{i}", filtered.astype(np.float64))

with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump({"arrays": manifest, "roi_cases": len(cases), "pre_cases": 3,
               "hr_cases": len(hr_signals), "hr": hr_ref}, f, indent=1)
print(f"wrote {len(manifest)} arrays to {out_dir}")
