// Colour segmentation rules (no DOM): per-channel RGB thresholds, a hue interval (which may
// wrap through 0°) with saturation and value floors, a ΔE ball around a reference colour,
// Otsu on one derived channel, and k-means in RGB or Lab. Pixels come as planar encoded
// sRGB in [0, 1]. Otsu and k-means are written here because lib/threshold.ts and
// lib/kmeans.ts (owned by the thresholding day and the GMM elective) do not exist yet.
import { srgbToLinear, linearRgbToXyz, xyzToLab, rgbToHsv, RGB } from "../lib/colour";
import { rng } from "../lib/random";
import { deltaE2000 } from "../colourcloud/colourx";

export interface Pixels { r: Float32Array; g: Float32Array; b: Float32Array; n: number }

/** Derived channels of every pixel (computed once per image). */
export interface Derived { lab: Float32Array; hsv: Float32Array }
export function derive(px: Pixels): Derived {
  const lab = new Float32Array(3 * px.n), hsv = new Float32Array(3 * px.n);
  for (let p = 0; p < px.n; p++) {
    const e: RGB = [px.r[p], px.g[p], px.b[p]];
    const L = xyzToLab(linearRgbToXyz([srgbToLinear(e[0]), srgbToLinear(e[1]), srgbToLinear(e[2])]));
    lab.set(L, 3 * p);
    hsv.set(rgbToHsv(e), 3 * p);
  }
  return { lab, hsv };
}

/** Smallest angle between two hues in degrees (0..180). */
export const hueDist = (a: number, b: number): number => { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; };

export function rgbRule(px: Pixels, rMin: number, gMax: number, bMax: number): Uint8Array {
  const m = new Uint8Array(px.n);
  for (let p = 0; p < px.n; p++) m[p] = 255 * px.r[p] >= rMin && 255 * px.g[p] <= gMax && 255 * px.b[p] <= bMax ? 1 : 0;
  return m;
}

/** Hue within ±halfWidth of centre (circular), saturation ≥ sMin, value ≥ vMin. Pixels with
 *  undefined hue (greys) are never selected. */
export function hueRule(d: Derived, n: number, centre: number, halfWidth: number, sMin: number, vMin: number): Uint8Array {
  const m = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    const h = d.hsv[3 * p], s = d.hsv[3 * p + 1], v = d.hsv[3 * p + 2];
    m[p] = Number.isFinite(h) && hueDist(h, centre) <= halfWidth && s >= sMin && v >= vMin ? 1 : 0;
  }
  return m;
}

export type Metric = "de76" | "de2000" | "rgb";
/** Pixels within `radius` of the reference pixel: ΔE76 or ΔE2000 in Lab, or Euclidean
 *  distance of encoded RGB in 0..255 units. */
export function ballRule(px: Pixels, d: Derived, ref: number, radius: number, metric: Metric): Uint8Array {
  const m = new Uint8Array(px.n), L0 = [d.lab[3 * ref], d.lab[3 * ref + 1], d.lab[3 * ref + 2]] as RGB;
  for (let p = 0; p < px.n; p++) {
    let dist: number;
    if (metric === "rgb") dist = 255 * Math.hypot(px.r[p] - px.r[ref], px.g[p] - px.g[ref], px.b[p] - px.b[ref]);
    else {
      const L = [d.lab[3 * p], d.lab[3 * p + 1], d.lab[3 * p + 2]] as RGB;
      dist = metric === "de76" ? Math.hypot(L[0] - L0[0], L[1] - L0[1], L[2] - L0[2]) : deltaE2000(L, L0);
    }
    m[p] = dist <= radius ? 1 : 0;
  }
  return m;
}

/** Otsu's threshold on values (256 bins over their range): the cut that maximises the
 *  between-class variance. Returns the threshold value. */
export function otsu(values: ArrayLike<number>): number {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  if (!(hi > lo)) return lo;
  const B = 256, h = new Float64Array(B), w = (hi - lo) / B;
  let n = 0;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) { h[Math.min(B - 1, Math.floor((v - lo) / w))]++; n++; } }
  let sumAll = 0; for (let i = 0; i < B; i++) sumAll += i * h[i];
  let w0 = 0, sum0 = 0, best = -1, bestK = 0;
  for (let k = 0; k < B - 1; k++) {
    w0 += h[k]; sum0 += k * h[k];
    const w1 = n - w0;
    if (w0 === 0 || w1 === 0) continue;
    const m0 = sum0 / w0, m1 = (sumAll - sum0) / w1, between = w0 * w1 * (m0 - m1) ** 2;
    if (between > best) { best = between; bestK = k; }
  }
  return lo + (bestK + 1) * w;
}

export type OtsuChannel = "R" | "G" | "B" | "S" | "a" | "b" | "L";
export function channelValues(px: Pixels, d: Derived, c: OtsuChannel): Float32Array {
  const out = new Float32Array(px.n);
  for (let p = 0; p < px.n; p++) {
    out[p] = c === "R" ? 255 * px.r[p] : c === "G" ? 255 * px.g[p] : c === "B" ? 255 * px.b[p] : c === "S" ? d.hsv[3 * p + 1]
      : c === "L" ? d.lab[3 * p] : c === "a" ? d.lab[3 * p + 1] : d.lab[3 * p + 2];
  }
  return out;
}

/** k-means (Lloyd) on 3D points: seeded random pixels as the initial centres. `steps`
 *  iterations at most; stops early when no label changes. */
export function kmeans(points: Float32Array, n: number, k: number, steps: number, seed = 5): { labels: Int32Array; centres: number[][]; iterations: number; sse: number } {
  const r = rng(seed), centres: number[][] = [];
  for (let j = 0; j < k; j++) { const p = Math.floor(r.unif() * n); centres.push([points[3 * p], points[3 * p + 1], points[3 * p + 2]]); }
  const labels = new Int32Array(n).fill(-1);
  let it = 0, sse = 0;
  for (; it < steps; it++) {
    let changed = 0; sse = 0;
    for (let p = 0; p < n; p++) {
      let best = 0, bd = Infinity;
      for (let j = 0; j < k; j++) { const dd = (points[3 * p] - centres[j][0]) ** 2 + (points[3 * p + 1] - centres[j][1]) ** 2 + (points[3 * p + 2] - centres[j][2]) ** 2; if (dd < bd) { bd = dd; best = j; } }
      if (labels[p] !== best) { labels[p] = best; changed++; }
      sse += bd;
    }
    if (changed === 0) break;
    const s = centres.map(() => [0, 0, 0, 0]);
    for (let p = 0; p < n; p++) { const t = s[labels[p]]; t[0] += points[3 * p]; t[1] += points[3 * p + 1]; t[2] += points[3 * p + 2]; t[3]++; }
    s.forEach((t, j) => { if (t[3] > 0) centres[j] = [t[0] / t[3], t[1] / t[3], t[2] / t[3]]; });
  }
  return { labels, centres, iterations: it, sse };
}

/** Dice of a mask against truth == cls. */
export function diceClass(mask: Uint8Array, truth: Uint8Array, cls: number): number {
  let tp = 0, fp = 0, fn = 0;
  for (let p = 0; p < mask.length; p++) { const t = truth[p] === cls; if (mask[p] && t) tp++; else if (mask[p]) fp++; else if (t) fn++; }
  return 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 1;
}
