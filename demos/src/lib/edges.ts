// Edges, blobs and ridges (owner: the edges day, day 2). No DOM.
//
// Built on the linear filters of lib/imageops.ts (gaussianKernel, gaussianDerivative,
// gaussianBlur, logFilter, dogFilter) and the 2 × 2 closed forms of lib/linalg.ts.
//
//  1D: correlate1d, gaussian1d, boxKernel1d (for the 1D demos gaussderiv and scale1d)
//  Canny: quantiseDirection, nmsNeighbours, nonMaxSuppression (4-direction or sub-pixel),
//         hysteresis (instrumented: the ring at which each pixel joined), canny, edgeList
//  zeroCrossings of a signed response (Marr–Hildreth edges)
//  Hessian: hessian(img, σ, normalised), classifyHessianPixel / classifyHessian (HessClass),
//           frangiValue, frangi(img, sigmas, β, c, bright) with the best-scale map
//  Structure tensor: structureTensor(img, σ, ρ) with orientation/coherence, harris, minEig
//  Scale space: scaleSpaceStack(img, sigmas, fn), normLoGStack, extrema3D, localMaxima2D
//
// Angle conventions: image coordinates have x right and y DOWN. Functions return the raw
// atan2(gy, gx) of image coordinates; demos show angles counter-clockwise on screen, i.e.
// screenAngle(gx, gy) = atan2(−gy, gx).

import { Grey, createGrey, gaussianKernel, gaussianDerivative, gaussianBlur, combine, borderIndex, Border, sampleBilinear, logFilter } from "./imageops";
import { eigSym2 } from "./linalg";

// ---------------------------------------------------------------------------------------
// 1D

/** Correlate a 1D signal with an odd-length centred kernel; the border rule as in imageops. */
export function correlate1d(f: ArrayLike<number>, k: ArrayLike<number>, border: Border = "replicate"): Float64Array {
  const n = f.length, r = k.length >> 1, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) {
      const j = borderIndex(x + i, n, border);
      if (j >= 0) s += k[i + r] * f[j];
    }
    out[x] = s;
  }
  return out;
}

/** Gaussian smoothing (order 0), slope (1) or curvature (2) of a 1D signal at scale σ (σ ≤ 0: finite differences). */
export function gaussian1d(f: ArrayLike<number>, sigma: number, order: 0 | 1 | 2 = 0, border: Border = "replicate"): Float64Array {
  if (!(sigma > 0)) {
    const k = order === 0 ? [1] : order === 1 ? [-0.5, 0, 0.5] : [1, -2, 1];
    return correlate1d(f, k, border);
  }
  return correlate1d(f, gaussianKernel(sigma, order), border);
}

/**
 * A box kernel with the same standard deviation as a Gaussian of width σ (the box of
 * width w has sd w/√12), odd length, sum 1. Used to show that box blurring can create
 * new extrema where the Gaussian never does.
 */
export function boxKernel1d(sigma: number): Float64Array {
  const half = Math.max(0, Math.round((Math.sqrt(12) * sigma - 1) / 2));
  const k = new Float64Array(2 * half + 1).fill(1 / (2 * half + 1));
  return k;
}

/** Number of local extrema (strict, interior) of a 1D signal; |difference| ≤ eps counts as flat. */
export function countExtrema(f: ArrayLike<number>, eps = 1e-9): number {
  let n = 0, prev = 0;
  for (let i = 1; i < f.length; i++) {
    const d = f[i] - f[i - 1], s = d > eps ? 1 : d < -eps ? -1 : 0;
    if (s !== 0) { if (prev !== 0 && s !== prev) n++; prev = s; }
  }
  return n;
}

// ---------------------------------------------------------------------------------------
// Gradient direction, non-maximum suppression, hysteresis

/** Angle counter-clockwise on screen (radians) of a vector (gx, gy) in image coordinates (y down). */
export const screenAngle = (gx: number, gy: number): number => Math.atan2(-gy, gx);

/** The gradient direction rounded to 0, 45, 90 or 135 degrees (on screen, counter-clockwise); returns 0..3. */
export function quantiseDirection(gx: number, gy: number): 0 | 1 | 2 | 3 {
  let a = screenAngle(gx, gy) * 180 / Math.PI;      // (-180, 180]
  if (a < 0) a += 180;                               // a direction and its opposite are the same axis
  const q = Math.round(a / 45) % 4;
  return q as 0 | 1 | 2 | 3;
}

/** Pixel offsets (dx, dy in image coordinates) of the two neighbours compared by NMS for each direction. */
export const NMS_OFFSETS: [number, number][] = [[1, 0], [1, -1], [0, 1], [1, 1]];   // 0°, 45° (up-right on screen), 90°, 135°

/** The two neighbours (x, y) of pixel (x, y) along the rounded gradient direction. */
export function nmsNeighbours(gx: number, gy: number, x: number, y: number): [[number, number], [number, number]] {
  const [dx, dy] = NMS_OFFSETS[quantiseDirection(gx, gy)];
  return [[x + dx, y + dy], [x - dx, y - dy]];
}

/**
 * Non-maximum suppression: keep a pixel's magnitude only if it is a maximum along the
 * gradient direction (else 0). 4-direction: compare with the two neighbours along the
 * rounded direction; sub-pixel: compare with the magnitude interpolated bilinearly at
 * p ± ∇L/‖∇L‖. Ties: strictly greater than the "forward" neighbour, ≥ the backward one,
 * so a plateau two pixels wide keeps one pixel. Border pixels are suppressed.
 */
export function nonMaxSuppression(mag: Grey, gx: Grey, gy: Grey, subpixel = false): Grey {
  const { width: w, height: h } = mag, out = createGrey(w, h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, m = mag.data[i];
    if (m <= 0) continue;
    let a: number, b: number;
    if (subpixel) {
      const ux = gx.data[i] / m, uy = gy.data[i] / m;
      a = sampleBilinear(mag, x + ux, y + uy); b = sampleBilinear(mag, x - ux, y - uy);
    } else {
      const [dx, dy] = NMS_OFFSETS[quantiseDirection(gx.data[i], gy.data[i])];
      a = mag.data[(y + dy) * w + x + dx]; b = mag.data[(y - dy) * w + x - dx];
    }
    if (m > a && m >= b) out.data[i] = m;
  }
  return out;
}

export interface HysteresisResult {
  /** 1 = edge pixel, 0 = not. */
  edges: Uint8Array;
  /** Ring at which a pixel joined: 0 = seed (≥ tHigh), k = k steps of growth from a seed, −1 = not an edge. */
  ring: Int32Array;
  /** Number of rings (max ring + 1; 0 if there are no seeds). */
  rings: number;
  /** Pixel counts: strong (seeds), weak (tLow ≤ v < tHigh), weak kept. */
  strong: number; weak: number; weakKept: number;
}

/**
 * Hysteresis thresholding: pixels ≥ tHigh are seeds; pixels ≥ tLow are kept if they are
 * connected (4 or 8 neighbours) through pixels ≥ tLow to a seed. Breadth-first, so `ring`
 * records the growth step by step (the canny demo replays it).
 */
export function hysteresis(v: Grey | { width: number; height: number; data: ArrayLike<number> }, tLow: number, tHigh: number, conn: 4 | 8 = 8): HysteresisResult {
  const { width: w, height: h, data } = v, n = w * h;
  const ring = new Int32Array(n).fill(-1), edges = new Uint8Array(n);
  let cur: number[] = [], strong = 0, weak = 0;
  for (let i = 0; i < n; i++) {
    if (data[i] >= tHigh) { ring[i] = 0; edges[i] = 1; cur.push(i); strong++; }
    else if (data[i] >= tLow) weak++;
  }
  const off = conn === 4 ? [[1, 0], [-1, 0], [0, 1], [0, -1]] : [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  let r = 0, weakKept = 0;
  while (cur.length) {
    const next: number[] = [];
    r++;
    for (const i of cur) {
      const x = i % w, y = (i - x) / w;
      for (const [dx, dy] of off) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (ring[j] < 0 && data[j] >= tLow) { ring[j] = r; edges[j] = 1; next.push(j); weakKept++; }
      }
    }
    cur = next;
  }
  const rings = strong ? r : 0;
  return { edges, ring, rings, strong, weak, weakKept };
}

export interface CannyResult {
  smoothed: Grey; gx: Grey; gy: Grey; mag: Grey; nms: Grey; hyst: HysteresisResult;
}

/** The four steps of Canny: smooth (σ), gradient, non-maximum suppression, hysteresis (tLow, tHigh on the magnitude). */
export function canny(img: Grey, sigma: number, tLow: number, tHigh: number, subpixel = false, border: Border = "replicate"): CannyResult {
  const smoothed = gaussianBlur(img, sigma, border);
  const gx = gaussianDerivative(img, sigma, 1, 0, border), gy = gaussianDerivative(img, sigma, 0, 1, border);
  const mag = combine(gx, gy, Math.hypot);
  const nms = nonMaxSuppression(mag, gx, gy, subpixel);
  const hyst = hysteresis(nms, tLow, tHigh, 8);
  return { smoothed, gx, gy, mag, nms, hyst };
}

export interface EdgePoint { x: number; y: number; theta: number; strength: number }

/** The edge pixels as a list {x, y, theta (gradient direction, on screen, CCW, radians), strength}, for voting (Hough). */
export function edgeList(edges: ArrayLike<number>, gx: Grey, gy: Grey, mag: Grey): EdgePoint[] {
  const out: EdgePoint[] = [], w = gx.width;
  for (let i = 0; i < edges.length; i++) if (edges[i]) {
    const x = i % w, y = (i - x) / w;
    out.push({ x, y, theta: screenAngle(gx.data[i], gy.data[i]), strength: mag.data[i] });
  }
  return out;
}

/**
 * Zero crossings of a signed response (e.g. the LoG): a pixel is marked if its value and
 * that of its right or lower neighbour have opposite signs and differ by more than
 * `minContrast` (suppresses crossings in flat noise). The pixel with the smaller |value| is marked.
 */
export function zeroCrossings(r: Grey, minContrast = 0): Uint8Array {
  const { width: w, height: h, data } = r, out = new Uint8Array(w * h);
  const test = (i: number, j: number) => {
    const a = data[i], b = data[j];
    if ((a < 0 && b >= 0) || (a >= 0 && b < 0)) {
      if (Math.abs(a - b) > minContrast) out[Math.abs(a) <= Math.abs(b) ? i : j] = 1;
    }
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (x + 1 < w) test(i, i + 1);
    if (y + 1 < h) test(i, i + w);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Hessian

export interface HessianResult {
  xx: Grey; xy: Grey; yy: Grey;
  /** Eigenvalues ordered by magnitude: |l1| ≤ |l2|. */
  l1: Grey; l2: Grey;
  /** Angle (image coordinates, atan2(vy, vx)) of the eigenvector of l1: along a ridge. */
  angle1: Grey;
}

/** Eigenvalues (|l1| ≤ |l2|) and the angle of l1's eigenvector of [[a, b], [b, c]]. */
export function hessEig(a: number, b: number, c: number): { l1: number; l2: number; angle1: number } {
  const e = eigSym2(a, b, c);
  // e.l1 ≥ e.l2 by value; reorder by magnitude
  if (Math.abs(e.l1) <= Math.abs(e.l2)) return { l1: e.l1, l2: e.l2, angle1: e.angle };
  return { l1: e.l2, l2: e.l1, angle1: Math.atan2(e.v2[1], e.v2[0]) };
}

/** Per-pixel Hessian of G_σ * I and its eigenvalues; `normalised` multiplies by σ². */
export function hessian(img: Grey, sigma: number, normalised = false, border: Border = "replicate"): HessianResult {
  const s = normalised ? sigma * sigma : 1;
  const xx = gaussianDerivative(img, sigma, 2, 0, border), xy = gaussianDerivative(img, sigma, 1, 1, border), yy = gaussianDerivative(img, sigma, 0, 2, border);
  const n = img.width * img.height, l1 = createGrey(img.width, img.height), l2 = createGrey(img.width, img.height), angle1 = createGrey(img.width, img.height);
  for (let i = 0; i < n; i++) {
    xx.data[i] *= s; xy.data[i] *= s; yy.data[i] *= s;
    const e = hessEig(xx.data[i], xy.data[i], yy.data[i]);
    l1.data[i] = e.l1; l2.data[i] = e.l2; angle1.data[i] = e.angle1;
  }
  return { xx, xy, yy, l1, l2, angle1 };
}

/** Hessian classes (bright = brighter than the surroundings). */
export enum HessClass { Flat = 0, BrightBlob = 1, DarkBlob = 2, BrightRidge = 3, DarkRidge = 4, Saddle = 5 }
export const HESS_CLASS_NAMES = ["flat", "bright blob", "dark blob", "bright ridge", "dark ridge", "saddle"];

/**
 * Class of a pixel from its eigenvalues (|l1| ≤ |l2|): flat if |l2| < t; a ridge (line) if
 * |l1| < ratio·|l2| (bright if l2 < 0); otherwise a blob (same signs; bright if negative) or a saddle.
 */
export function classifyHessianPixel(l1: number, l2: number, t: number, ratio = 0.5): HessClass {
  if (Math.abs(l2) < t) return HessClass.Flat;
  if (Math.abs(l1) < ratio * Math.abs(l2)) return l2 < 0 ? HessClass.BrightRidge : HessClass.DarkRidge;
  if (l1 * l2 < 0) return HessClass.Saddle;
  return l2 < 0 ? HessClass.BrightBlob : HessClass.DarkBlob;
}

export function classifyHessian(h: HessianResult, t: number, ratio = 0.5): Uint8Array {
  const n = h.l1.data.length, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = classifyHessianPixel(h.l1.data[i], h.l2.data[i], t, ratio);
  return out;
}

/**
 * Frangi vesselness (2D) from eigenvalues |l1| ≤ |l2|: 0 if l2 has the wrong sign
 * (l2 > 0 for bright lines), else exp(−R_B²/2β²)·(1 − exp(−S²/2c²)), R_B = l1/l2, S = √(l1² + l2²).
 */
export function frangiValue(l1: number, l2: number, beta: number, c: number, bright = true): number {
  if (bright ? l2 >= 0 : l2 <= 0) return 0;
  const rb = l1 / l2, S2 = l1 * l1 + l2 * l2;
  return Math.exp(-rb * rb / (2 * beta * beta)) * (1 - Math.exp(-S2 / (2 * c * c)));
}

/** Multi-scale Frangi: maximum over σ of the vesselness of σ²H; also the σ (index) that won. */
export function frangi(img: Grey, sigmas: number[], beta = 0.5, c = 15, bright = true): { response: Grey; bestScale: Int32Array; perScale: Grey[] } {
  const w = img.width, h = img.height, response = createGrey(w, h), bestScale = new Int32Array(w * h).fill(-1), perScale: Grey[] = [];
  sigmas.forEach((s, k) => {
    const H = hessian(img, s, true), v = createGrey(w, h);
    for (let i = 0; i < w * h; i++) {
      const f = frangiValue(H.l1.data[i], H.l2.data[i], beta, c, bright);
      v.data[i] = f;
      if (f > response.data[i]) { response.data[i] = f; bestScale[i] = k; }
    }
    perScale.push(v);
  });
  return { response, bestScale, perScale };
}

// ---------------------------------------------------------------------------------------
// Structure tensor

export interface StructureTensor {
  gx: Grey; gy: Grey;
  jxx: Grey; jxy: Grey; jyy: Grey;
  /** Eigenvalues mu1 ≥ mu2 ≥ 0. */
  mu1: Grey; mu2: Grey;
  /** Dominant gradient direction (image coordinates, atan2), in (−π/2, π/2]; the structure runs perpendicular to it. */
  angle: Grey;
  /** ((mu1 − mu2)/(mu1 + mu2))², 0 where mu1 + mu2 = 0. */
  coherence: Grey;
}

/** J = G_ρ * (∇L ∇Lᵀ) with ∇L the Gaussian gradient at scale σ (ρ ≤ 0: no averaging). */
export function structureTensor(img: Grey, sigma: number, rho: number, border: Border = "replicate"): StructureTensor {
  const gx = gaussianDerivative(img, sigma, 1, 0, border), gy = gaussianDerivative(img, sigma, 0, 1, border);
  const pxx = combine(gx, gx, (a, b) => a * b), pxy = combine(gx, gy, (a, b) => a * b), pyy = combine(gy, gy, (a, b) => a * b);
  const jxx = gaussianBlur(pxx, rho, border), jxy = gaussianBlur(pxy, rho, border), jyy = gaussianBlur(pyy, rho, border);
  const w = img.width, h = img.height, mu1 = createGrey(w, h), mu2 = createGrey(w, h), angle = createGrey(w, h), coherence = createGrey(w, h);
  for (let i = 0; i < w * h; i++) {
    const e = eigSym2(jxx.data[i], jxy.data[i], jyy.data[i]);
    const a = Math.max(0, e.l1), b = Math.max(0, e.l2);
    mu1.data[i] = a; mu2.data[i] = b; angle.data[i] = e.angle;
    coherence.data[i] = a + b > 0 ? ((a - b) / (a + b)) ** 2 : 0;
  }
  return { gx, gy, jxx, jxy, jyy, mu1, mu2, angle, coherence };
}

/** Harris corner response det J − k (tr J)². */
export function harris(J: { jxx: Grey; jxy: Grey; jyy: Grey }, k = 0.05): Grey {
  const n = J.jxx.data.length, out = createGrey(J.jxx.width, J.jxx.height);
  for (let i = 0; i < n; i++) {
    const a = J.jxx.data[i], b = J.jxy.data[i], c = J.jyy.data[i];
    out.data[i] = a * c - b * b - k * (a + c) ** 2;
  }
  return out;
}

/** Shi–Tomasi / Förstner-style corner measure: the smaller eigenvalue of J. */
export function minEig(J: { jxx: Grey; jxy: Grey; jyy: Grey }): Grey {
  const n = J.jxx.data.length, out = createGrey(J.jxx.width, J.jxx.height);
  for (let i = 0; i < n; i++) out.data[i] = Math.max(0, eigSym2(J.jxx.data[i], J.jxy.data[i], J.jyy.data[i]).l2);
  return out;
}

// ---------------------------------------------------------------------------------------
// Scale space and local maxima

/** fn(img, σ) for every σ: a stack of responses (one Grey per scale). */
export function scaleSpaceStack(img: Grey, sigmas: number[], fn: (img: Grey, sigma: number) => Grey): Grey[] {
  return sigmas.map(s => fn(img, s));
}

/** −σ²∇²L for every σ: bright blobs give positive peaks (the "Mexican hat" response). */
export function normLoGStack(img: Grey, sigmas: number[], normalised = true): Grey[] {
  return scaleSpaceStack(img, sigmas, (im, s) => {
    const r = logFilter(im, s, normalised);
    for (let i = 0; i < r.data.length; i++) r.data[i] = -r.data[i];
    return r;
  });
}

/** Geometric sequence of n scales from s0 to s1. */
export function geomScales(s0: number, s1: number, n: number): number[] {
  return Array.from({ length: n }, (_, k) => n === 1 ? s0 : s0 * Math.pow(s1 / s0, k / (n - 1)));
}

export interface Extremum { x: number; y: number; s: number; value: number }

/**
 * Local maxima in (x, y, scale) above `threshold`: larger than all 26 neighbours (ties broken
 * by position, so a plateau gives one maximum). Neighbours outside the stack are ignored;
 * pixels within `border` of the image edge are skipped. With `interiorScales`, the first and
 * last scale are not searched (a maximum there may continue beyond the stack). Sorted by
 * value, strongest first.
 */
export function extrema3D(stack: Grey[], threshold: number, border = 1, interiorScales = false): Extremum[] {
  const S = stack.length, w = stack[0].width, h = stack[0].height, out: Extremum[] = [];
  for (let s = interiorScales ? 1 : 0; s < (interiorScales ? S - 1 : S); s++) {
    const d = stack[s].data;
    for (let y = border; y < h - border; y++) for (let x = border; x < w - border; x++) {
      const v = d[y * w + x];
      if (!(v > threshold)) continue;
      let ok = true;
      for (let ds = -1; ds <= 1 && ok; ds++) {
        const t = s + ds;
        if (t < 0 || t >= S) continue;
        const e = stack[t].data;
        for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!ds && !dx && !dy) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const u = e[yy * w + xx];
          const earlier = ds < 0 || (ds === 0 && (dy < 0 || (dy === 0 && dx < 0)));
          if (u > v || (earlier && u === v)) { ok = false; break; }
        }
      }
      if (ok) out.push({ x, y, s, value: v });
    }
  }
  return out.sort((a, b) => b.value - a.value);
}

/** Local maxima in 2D above threshold within a (2r+1)² window (ties broken by position), strongest first. */
export function localMaxima2D(img: Grey, threshold: number, r = 1, border = 1): { x: number; y: number; value: number }[] {
  const { width: w, height: h, data } = img, out: { x: number; y: number; value: number }[] = [];
  for (let y = border; y < h - border; y++) for (let x = border; x < w - border; x++) {
    const v = data[y * w + x];
    if (!(v > threshold)) continue;
    let ok = true;
    for (let dy = -r; dy <= r && ok; dy++) for (let dx = -r; dx <= r; dx++) {
      if (!dx && !dy) continue;
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const u = data[yy * w + xx], earlier = dy < 0 || (dy === 0 && dx < 0);
      if (u > v || (earlier && u === v)) { ok = false; break; }
    }
    if (ok) out.push({ x, y, value: v });
  }
  return out.sort((a, b) => b.value - a.value);
}
