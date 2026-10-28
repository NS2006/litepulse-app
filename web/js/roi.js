// ROI extraction: MediaPipe landmarks -> convex-hull mask -> bounding-box crop
// -> 36x36 patch, reproducing roi_extractor.py from the training repo:
//
//   poly  = pts[indices].astype(np.int32)
//   mask  = cv2.fillConvexPoly(zeros, cv2.convexHull(poly), 255)
//   x,y,w,h = cv2.boundingRect(poly)
//   patch = cv2.bitwise_and(frame, frame, mask)[y:y+h, x:x+w]
//   patch = cv2.resize(patch, (36, 36), interpolation=cv2.INTER_AREA)
//
// The Android app resizes with nearest-neighbour; this follows training instead.

export const PATCH_SIZE = 36;
export const PLANE = PATCH_SIZE * PATCH_SIZE;
export const ROI_ORDER = ['forehead', 'left_cheek', 'right_cheek'];
export const VALUES_PER_FRAME = ROI_ORDER.length * 3 * PLANE;

export const LANDMARKS = {
  forehead: [10, 108, 69, 104, 68, 71, 21, 54, 103, 67, 109, 10,
    338, 297, 332, 284, 251, 301, 293, 334, 296, 336],
  left_cheek: [116, 117, 118, 119, 100, 126, 142, 36, 205, 187, 123],
  right_cheek: [345, 346, 347, 348, 329, 355, 371, 266, 425, 411, 352],
};

/** Integer pixel polygon per ROI (float32 product, then truncation like astype(np.int32)). */
export function roiPolygons(landmarks, width, height) {
  const out = {};
  for (const name of ROI_ORDER) {
    out[name] = LANDMARKS[name].map((i) => [
      Math.trunc(Math.fround(landmarks[i].x * width)),
      Math.trunc(Math.fround(landmarks[i].y * height)),
    ]);
  }
  return out;
}

/** Andrew's monotone chain; same hull as cv2.convexHull for integer points. */
export function convexHull(points) {
  const seen = new Set();
  const pts = [];
  for (const p of points) {
    const key = p[0] + ',' + p[1];
    if (!seen.has(key)) {
      seen.add(key);
      pts.push(p);
    }
  }
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;

  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

const XY_ONE = 65536; // cv2 drawing fixed point (XY_SHIFT = 16)

/**
 * Mask of a convex polygon over the box [bx, bx+bw) x [by, by+bh), the way
 * cv2.fillConvexPoly(LINE_8, shift=0) rasterises it on a frameW x frameH image:
 * the outline first, then scanlines stepped in 16.16 fixed point from each
 * edge's upper vertex and rounded half-up. The last row is outline only.
 */
function convexPolyMask(hull, bx, by, bw, bh, frameW, frameH) {
  const mask = new Uint8Array(bw * bh);
  const set = (x, y) => {
    const mx = x - bx;
    const my = y - by;
    if (mx >= 0 && my >= 0 && mx < bw && my < bh) mask[my * bw + mx] = 1;
  };

  const n = hull.length;
  for (let i = 0; i < n; i++) cvLine8(hull[(i + n - 1) % n], hull[i], frameW, frameH, set);
  if (n < 3) return mask;

  // Non-horizontal edges as (upper, lower) with cv2's fixed-point slope.
  const edges = [];
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    let a = hull[i];
    let b = hull[(i + 1) % n];
    minY = Math.min(minY, a[1]);
    maxY = Math.max(maxY, a[1]);
    if (a[1] === b[1]) continue;
    if (a[1] > b[1]) [a, b] = [b, a];
    const dy = b[1] - a[1];
    const dx = Math.trunc(((b[0] - a[0]) * XY_ONE * 2 + dy) / (2 * dy));
    edges.push({ x: a[0] * XY_ONE, y0: a[1], y1: b[1], dx });
  }

  for (let y = Math.max(minY, 0); y < Math.min(maxY, frameH); y++) {
    let xl = Infinity;
    let xr = -Infinity;
    for (const e of edges) {
      if (y < e.y0 || y >= e.y1) continue;
      const x = e.x + (y - e.y0) * e.dx;
      xl = Math.min(xl, x);
      xr = Math.max(xr, x);
    }
    if (xl > xr) continue;
    const x0 = Math.max(Math.floor((xl + XY_ONE / 2) / XY_ONE), 0);
    const x1 = Math.min(Math.floor((xr + XY_ONE / 2) / XY_ONE), frameW - 1);
    for (let x = x0; x <= x1; x++) set(x, y);
  }
  return mask;
}

/** cv2.clipLine: clip a segment to [0, w-1] x [0, h-1]. Null if fully outside. */
function cvClipLine(w, h, p1, p2) {
  let [x1, y1] = p1;
  let [x2, y2] = p2;
  const right = w - 1;
  const bottom = h - 1;
  let c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
  let c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

  if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
    if (c1 & 12) {
      const a = c1 < 8 ? 0 : bottom;
      x1 += Math.trunc(((a - y1) * (x2 - x1)) / (y2 - y1));
      y1 = a;
      c1 = (x1 < 0) + (x1 > right) * 2;
    }
    if (c2 & 12) {
      const a = c2 < 8 ? 0 : bottom;
      x2 += Math.trunc(((a - y2) * (x2 - x1)) / (y2 - y1));
      y2 = a;
      c2 = (x2 < 0) + (x2 > right) * 2;
    }
    if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
      if (c1) {
        const a = c1 === 1 ? 0 : right;
        y1 += Math.trunc(((a - x1) * (y2 - y1)) / (x2 - x1));
        x1 = a;
        c1 = 0;
      }
      if (c2) {
        const a = c2 === 1 ? 0 : right;
        y2 += Math.trunc(((a - x2) * (y2 - y1)) / (x2 - x1));
        x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) === 0 ? [[x1, y1], [x2, y2]] : null;
}

/** cv2.line(LINE_8): clipLine, then LineIterator with leftToRight=true. */
function cvLine8(p1, p2, frameW, frameH, set) {
  const clipped = cvClipLine(frameW, frameH, p1, p2);
  if (!clipped) return;
  let [x, y] = clipped[0];
  let [x2, y2] = clipped[1];
  if (x2 < x) {
    [x, y, x2, y2] = [x2, y2, x, y];
  }
  const dx = x2 - x;
  const sy = y2 < y ? -1 : 1;
  const dy = Math.abs(y2 - y);
  const steep = dy > dx;
  const major = steep ? dy : dx;
  const minor = steep ? dx : dy;

  let err = major - 2 * minor;
  const plusDelta = 2 * major;
  const minusDelta = -2 * minor;
  for (let k = 0; k <= major; k++) {
    set(x, y);
    const stepMinor = err < 0;
    err += minusDelta + (stepMinor ? plusDelta : 0);
    if (steep) {
      y += sy;
      if (stepMinor) x += 1;
    } else {
      x += 1;
      if (stepMinor) y += sy;
    }
  }
}

/** cv2 computeResizeAreaTab: source indices and weights for one axis (downscaling). */
function areaTable(srcSize, dstSize) {
  const scale = srcSize / dstSize;
  const table = [];
  for (let d = 0; d < dstSize; d++) {
    const fsx1 = d * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, srcSize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, srcSize - 1);
    sx1 = Math.min(sx1, sx2);

    const taps = [];
    if (sx1 - fsx1 > 1e-3) taps.push([sx1 - 1, (sx1 - fsx1) / cellWidth]);
    for (let s = sx1; s < sx2; s++) taps.push([s, 1 / cellWidth]);
    if (fsx2 - sx2 > 1e-3) taps.push([sx2, Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth]);
    table.push(taps);
  }
  return table;
}

/** cv2 INTER_AREA coefficients when upscaling (2-tap, "area mode" linear). */
function areaLinearTable(srcSize, dstSize) {
  const invScale = dstSize / srcSize;
  const scale = 1 / invScale;
  const table = [];
  for (let d = 0; d < dstSize; d++) {
    let sx = Math.floor(d * scale);
    // cv2 stores this in a float before taking the fraction: 0.9999999999999964 -> 1.0 -> 0.
    let fx = Math.fround((d + 1) - (sx + 1) * invScale);
    fx = fx <= 0 ? 0 : fx - Math.floor(fx);
    if (sx >= srcSize - 1) {
      sx = srcSize - 1;
      fx = 0;
    }
    table.push([[sx, 1 - fx], [Math.min(sx + 1, srcSize - 1), fx]]);
  }
  return table;
}

/**
 * Masked crop + cv2.INTER_AREA resize into `out` (Uint8 values in a Float32Array
 * slot, channel-major: R plane, G plane, B plane) and `preview` (RGBA ImageData).
 */
function resizeMaskedArea(rgba, frameWidth, mask, cx, cy, cw, ch, bx, bw, out, outOffset, preview) {
  const area = cw >= PATCH_SIZE && ch >= PATCH_SIZE;
  const tx = area ? areaTable(cw, PATCH_SIZE) : areaLinearTable(cw, PATCH_SIZE);
  const ty = area ? areaTable(ch, PATCH_SIZE) : areaLinearTable(ch, PATCH_SIZE);
  const by = cy;

  for (let oy = 0; oy < PATCH_SIZE; oy++) {
    for (let ox = 0; ox < PATCH_SIZE; ox++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (const [sy, wy] of ty[oy]) {
        const y = cy + sy;
        for (const [sx, wx] of tx[ox]) {
          const x = cx + sx;
          if (!mask[(y - by) * bw + (x - bx)]) continue;
          const w = wx * wy;
          const i = (y * frameWidth + x) * 4;
          r += rgba[i] * w;
          g += rgba[i + 1] * w;
          b += rgba[i + 2] * w;
        }
      }
      const p = oy * PATCH_SIZE + ox;
      const R = Math.min(255, Math.max(0, Math.round(r)));
      const G = Math.min(255, Math.max(0, Math.round(g)));
      const B = Math.min(255, Math.max(0, Math.round(b)));
      out[outOffset + p] = R;
      out[outOffset + PLANE + p] = G;
      out[outOffset + 2 * PLANE + p] = B;
      if (preview) {
        preview.data[p * 4] = R;
        preview.data[p * 4 + 1] = G;
        preview.data[p * 4 + 2] = B;
        preview.data[p * 4 + 3] = 255;
      }
    }
  }
}

/**
 * One frame of model input: Float32Array(3 ROIs x 3 channels x 36 x 36),
 * ROI-major then R,G,B planes, raw 0..255 values (normalisation happens later).
 */
export function extractRoiFrame(imageData, landmarks, previews = null) {
  const { data, width, height } = imageData;
  const polygons = roiPolygons(landmarks, width, height);
  const frame = new Float32Array(VALUES_PER_FRAME);

  ROI_ORDER.forEach((name, r) => {
    const poly = polygons[name];
    const hull = convexHull(poly);

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const [x, y] of poly) {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    const bw = Math.max(maxX - minX + 1, 1);
    const bh = Math.max(maxY - minY + 1, 1);
    const x0 = Math.max(minX, 0);
    const y0 = Math.max(minY, 0);
    const x1 = Math.min(x0 + bw, width);
    const y1 = Math.min(y0 + bh, height);
    const preview = previews ? previews[name] : null;

    if (x1 <= x0 || y1 <= y0) {
      if (preview) preview.data.fill(0);
      return; // patch.size == 0 -> zeros
    }

    const mask = convexPolyMask(hull, x0, y0, x1 - x0, y1 - y0, width, height);
    resizeMaskedArea(data, width, mask, x0, y0, x1 - x0, y1 - y0, x0, x1 - x0,
      frame, r * 3 * PLANE, preview);
  });

  return { frame, polygons };
}
