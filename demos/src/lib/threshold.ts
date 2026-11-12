// Thresholds (owner: thresholding day). No DOM.
//
// Global methods work on a histogram given as `counts` (one entry per bin or grey level)
// and optional bin values `x` (default x_i = i). A split at index t puts bins 0..t in
// class 0 (background) and bins t+1..L-1 in class 1 (object): the mask is [I > x_t]
// for integer grey levels. Every method returns the chosen index `t` and its criterion
// curve over t = 0..L-1 (NaN where the split is undefined, e.g. an empty class), so a
// demo can plot it on the same axis as the histogram.
//
//   percentile · otsu (+ η = σ_B²/σ_T²) · otsuMulti (3 classes, exhaustive) · isodata
//   (Ridler–Calvard, with its trace) · kittler (minimum error) · triangle · kapur
//   (maximum entropy) · local thresholds (mean − C, Niblack, Sauvola) · hysteresis ·
//   the two-Gaussian model (crossings, error of a threshold).
//
// Evaluation (Dice, ROC) is not here: see lib/metrics.ts.

import { Grey, boxFilter, createGrey } from "./imageops";
import { label2d } from "./label";

type A = ArrayLike<number>;

const xs = (n: number, x?: A): A => x ?? Float64Array.from({ length: n }, (_, i) => i);

/** Running sums: P(t) = Σ_{i≤t} p_i and M(t) = Σ_{i≤t} x_i p_i (p normalised), and the totals. */
export function runningSums(counts: A, x?: A): { P: Float64Array; M: Float64Array; Q: Float64Array; N: number; mu: number; varT: number } {
  const L = counts.length, X = xs(L, x);
  let N = 0;
  for (let i = 0; i < L; i++) N += counts[i];
  const P = new Float64Array(L), M = new Float64Array(L), Q = new Float64Array(L);
  let p = 0, m = 0, q = 0;
  for (let i = 0; i < L; i++) {
    const pi = N > 0 ? counts[i] / N : 0;
    p += pi; m += X[i] * pi; q += X[i] * X[i] * pi;
    P[i] = p; M[i] = m; Q[i] = q;
  }
  const mu = L ? M[L - 1] : 0, varT = L ? Math.max(0, Q[L - 1] - mu * mu) : 0;
  return { P, M, Q, N, mu, varT };
}

export interface SplitStats {
  w0: number; w1: number;       // class weights (fractions of the pixels)
  mu0: number; mu1: number;     // class means (NaN for an empty class)
  var0: number; var1: number;   // class variances
  mu: number; varT: number;     // total mean and variance
  varW: number; varB: number;   // within-class ω0σ0² + ω1σ1², between-class ω0ω1(μ0 − μ1)²
}

/** Class statistics of the split "bins 0..t | t+1..L-1" (computed directly, no running sums). */
export function splitStats(counts: A, t: number, x?: A): SplitStats {
  const L = counts.length, X = xs(L, x);
  let n0 = 0, s0 = 0, q0 = 0, n1 = 0, s1 = 0, q1 = 0;
  for (let i = 0; i < L; i++) {
    const c = counts[i], v = X[i];
    if (i <= t) { n0 += c; s0 += c * v; q0 += c * v * v; } else { n1 += c; s1 += c * v; q1 += c * v * v; }
  }
  const N = n0 + n1 || 1, w0 = n0 / N, w1 = n1 / N;
  const mu0 = n0 > 0 ? s0 / n0 : NaN, mu1 = n1 > 0 ? s1 / n1 : NaN;
  const var0 = n0 > 0 ? Math.max(0, q0 / n0 - mu0 * mu0) : NaN, var1 = n1 > 0 ? Math.max(0, q1 / n1 - mu1 * mu1) : NaN;
  const mu = (s0 + s1) / N, varT = Math.max(0, (q0 + q1) / N - mu * mu);
  const varW = (n0 > 0 ? w0 * var0 : 0) + (n1 > 0 ? w1 * var1 : 0);
  const varB = n0 > 0 && n1 > 0 ? w0 * w1 * (mu0 - mu1) ** 2 : 0;
  return { w0, w1, mu0, mu1, var0, var1, mu, varT, varW, varB };
}

export interface ThresholdResult {
  /** Chosen split index: bins 0..t are background. -1 if no split is possible. */
  t: number;
  /** Criterion per split index t (length L; NaN where undefined). */
  curve: Float64Array;
}

/** Index of the largest finite value (first on ties); -1 if none. */
export function argmax(a: A): number {
  let b = -1;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && (b < 0 || a[i] > a[b])) b = i;
  return b;
}

/** Index of the smallest finite value (first on ties); -1 if none. */
export function argmin(a: A): number {
  let b = -1;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && (b < 0 || a[i] < a[b])) b = i;
  return b;
}

/**
 * Percentile threshold: the smallest t with P(t) ≥ 1 − objectFraction, i.e. "keep the
 * brightest objectFraction of the pixels". The curve is the cumulative histogram P(t).
 */
export function percentileThreshold(counts: A, objectFraction: number): ThresholdResult {
  const { P } = runningSums(counts);
  let t = P.length - 1;
  for (let i = 0; i < P.length; i++) if (P[i] >= 1 - objectFraction - 1e-12) { t = i; break; }
  return { t, curve: P };
}

/**
 * Otsu (1979): maximise the between-class variance
 *   σ_B²(t) = (μ P(t) − M(t))² / (P(t)(1 − P(t))) = ω0 ω1 (μ0 − μ1)²
 * in one pass over the running sums. `eta` = σ_B²/σ_T² (0..1, the separability).
 */
export function otsu(counts: A, x?: A): ThresholdResult & { varT: number; mu: number; eta: number } {
  const { P, M, mu, varT } = runningSums(counts, x);
  const L = counts.length, curve = new Float64Array(L).fill(NaN);
  for (let t = 0; t < L - 1; t++) {
    const w0 = P[t], w1 = 1 - w0;
    if (w0 <= 1e-12 || w1 <= 1e-12) continue;
    curve[t] = (mu * w0 - M[t]) ** 2 / (w0 * w1);
  }
  const t = argmax(curve);
  return { t, curve, varT, mu, eta: t >= 0 && varT > 0 ? curve[t] / varT : 0 };
}

/**
 * Multi-level Otsu with three classes: maximise Σ_k ω_k (μ_k − μ)² over all pairs
 * t1 < t2 (classes 0..t1, t1+1..t2, t2+1..L-1). O(L²) with running sums.
 */
export function otsuMulti3(counts: A, x?: A): { t1: number; t2: number; varB: number } {
  const { P, M, mu } = runningSums(counts, x);
  const L = counts.length;
  let best = -Infinity, b1 = -1, b2 = -1;
  const term = (w: number, m: number) => (w > 1e-12 ? (m * m) / w : NaN);
  for (let t1 = 0; t1 < L - 2; t1++) {
    const a = term(P[t1], M[t1]);
    if (!Number.isFinite(a)) continue;
    for (let t2 = t1 + 1; t2 < L - 1; t2++) {
      const b = term(P[t2] - P[t1], M[t2] - M[t1]), c = term(1 - P[t2], mu - M[t2]);
      if (!Number.isFinite(b) || !Number.isFinite(c)) continue;
      const v = a + b + c - mu * mu;   // Σ ω_k μ_k² − μ²
      if (v > best) { best = v; b1 = t1; b2 = t2; }
    }
  }
  return { t1: b1, t2: b2, varB: best };
}

/**
 * Isodata / Ridler–Calvard (1978): from a start value t0 (in x units), repeat
 * t ← (μ0(t) + μ1(t))/2 with class 0 = {x_i ≤ t}, until the split stops changing.
 * This is 1D k-means with k = 2. Returns the split index and the trace of t values.
 */
export function isodata(counts: A, t0?: number, x?: A, maxIter = 100): ThresholdResult & { value: number; trace: number[] } {
  const L = counts.length, X = xs(L, x);
  let tv = t0 ?? runningSums(counts, x).mu;
  const trace = [tv];
  const idx = (v: number) => { let k = -1; for (let i = 0; i < L; i++) if (X[i] <= v) k = i; return k; };
  for (let it = 0; it < maxIter; it++) {
    const k = idx(tv), s = splitStats(counts, k, X);
    if (!(s.w0 > 0 && s.w1 > 0)) break;
    const next = (s.mu0 + s.mu1) / 2;
    trace.push(next);
    tv = next;
    if (idx(next) === k) break;   // the split reproduces itself: a fixed point
  }
  // the curve: the isodata map g(t) = (μ0 + μ1)/2 − x_t (a fixed point is where the split reproduces itself)
  const curve = new Float64Array(L).fill(NaN);
  for (let t = 0; t < L - 1; t++) { const s = splitStats(counts, t, X); if (s.w0 > 0 && s.w1 > 0) curve[t] = (s.mu0 + s.mu1) / 2; }
  return { t: idx(tv), value: tv, trace, curve };
}

/**
 * Kittler–Illingworth (1986) minimum-error threshold: fit a Gaussian to each side of
 * the split and minimise J(t) = 1 + 2(ω0 ln σ0 + ω1 ln σ1) − 2(ω0 ln ω0 + ω1 ln ω1),
 * the average negative log-likelihood (up to constants) of the two-Gaussian model.
 * Class variances get a floor of binWidth²/12 (the quantisation variance), so a class
 * that sits in one bin does not give ln 0, and splits where a class holds less than
 * `minWeight` of the pixels are skipped (a lone spike, e.g. clipped values at 0, would
 * otherwise "fit" perfectly).
 */
export function kittler(counts: A, x?: A, minWeight = 0.005): ThresholdResult {
  const L = counts.length, X = xs(L, x);
  const bw = L > 1 ? Math.abs(X[1] - X[0]) : 1, floor = bw * bw / 12;
  const { P, M, Q } = runningSums(counts, X);
  const curve = new Float64Array(L).fill(NaN);
  for (let t = 0; t < L - 1; t++) {
    const w0 = P[t], w1 = 1 - w0;
    if (w0 <= Math.max(1e-12, minWeight) || w1 <= Math.max(1e-12, minWeight)) continue;
    const m0 = M[t] / w0, m1 = (M[L - 1] - M[t]) / w1;
    const v0 = Math.max(floor, Q[t] / w0 - m0 * m0), v1 = Math.max(floor, (Q[L - 1] - Q[t]) / w1 - m1 * m1);
    curve[t] = 1 + (w0 * Math.log(v0) + w1 * Math.log(v1)) - 2 * (w0 * Math.log(w0) + w1 * Math.log(w1));
  }
  return { t: argmin(curve), curve };
}

/**
 * Triangle method (Zack et al. 1977): a line from the histogram peak to the far end of
 * the longer tail; the threshold is the bin farthest below that line. Made for a
 * unimodal histogram with a long tail of a few object pixels. The curve is the
 * (normalised) distance below the line. Works for a tail on either side; for a tail to
 * the left the objects are dark, which callers should treat as such.
 */
export function triangle(counts: A): ThresholdResult & { tailRight: boolean; peak: number; end: number } {
  const L = counts.length;
  let peak = 0, lo = -1, hi = -1;
  for (let i = 0; i < L; i++) {
    if (counts[i] > counts[peak]) peak = i;
    if (counts[i] > 0) { if (lo < 0) lo = i; hi = i; }
  }
  const curve = new Float64Array(L).fill(NaN);
  if (lo < 0) return { t: -1, curve, tailRight: true, peak, end: peak };
  const tailRight = hi - peak >= peak - lo, end = tailRight ? Math.min(L - 1, hi + 1) : Math.max(0, lo - 1);
  const hp = counts[peak], dx = end - peak;
  // distance of (i, h_i) below the line from (peak, hp) to (end, 0), in units where the
  // histogram is scaled to the same range on both axes
  const sx = 1 / Math.max(1, Math.abs(dx)), sy = 1 / (hp || 1);
  const ax = dx * sx, ay = -hp * sy, len = Math.hypot(ax, ay);
  const a = Math.min(peak, end), b = Math.max(peak, end);
  for (let i = a; i <= b; i++) {
    const px = (i - peak) * sx, py = (counts[i] - hp) * sy;
    curve[i] = (ax * py - ay * px) / len * (tailRight ? -1 : 1);
  }
  let t = argmax(curve);
  if (t < 0) t = peak;
  return { t, curve, tailRight, peak, end };
}

/**
 * Kapur, Sahoo & Wong (1985): maximise H0(t) + H1(t), the entropies of the class
 * distributions p_i/ω0 (i ≤ t) and p_i/ω1 (i > t).
 */
export function kapur(counts: A): ThresholdResult {
  const L = counts.length;
  let N = 0;
  for (let i = 0; i < L; i++) N += counts[i];
  const p = Float64Array.from(counts, c => (N > 0 ? c / N : 0));
  // running Σ p and Σ p ln p
  const P = new Float64Array(L), S = new Float64Array(L);
  let a = 0, b = 0;
  for (let i = 0; i < L; i++) { a += p[i]; b += p[i] > 0 ? p[i] * Math.log(p[i]) : 0; P[i] = a; S[i] = b; }
  const curve = new Float64Array(L).fill(NaN);
  for (let t = 0; t < L - 1; t++) {
    const w0 = P[t], w1 = 1 - w0;
    if (w0 <= 1e-12 || w1 <= 1e-12) continue;
    // H0 = −Σ (p/ω0) ln(p/ω0) = ln ω0 − S0/ω0
    curve[t] = Math.log(w0) - S[t] / w0 + Math.log(w1) - (S[L - 1] - S[t]) / w1;
  }
  return { t: argmax(curve), curve };
}

export type GlobalMethod = "otsu" | "isodata" | "kittler" | "triangle" | "kapur";

/** Run one global method by name (isodata starts at the mean). */
export function globalThreshold(method: GlobalMethod, counts: A, x?: A): ThresholdResult {
  switch (method) {
    case "otsu": return otsu(counts, x);
    case "isodata": return isodata(counts, undefined, x);
    case "kittler": return kittler(counts, x);
    case "triangle": return triangle(counts);
    case "kapur": return kapur(counts);
  }
}

// ---------------------------------------------------------------------------------------
// Local thresholds T(x), in the published form: for DARK objects on a bright background
// (text on paper, stained nuclei in bright field); the pixel is object where I(x) < T(x).
// For bright objects, apply them to the inverted image. The window is (2r+1) × (2r+1).

/** Local mean and standard deviation over the window: box(I) and sqrt(box(I²) − box(I)²). */
export function localMeanSd(img: Grey, r: number): { mean: Grey; sd: Grey } {
  const sq: Grey = { width: img.width, height: img.height, data: Float32Array.from(img.data, v => v * v) };
  const m = boxFilter(img, r), m2 = boxFilter(sq, r);
  const sd = createGrey(img.width, img.height);
  for (let i = 0; i < sd.data.length; i++) sd.data[i] = Math.sqrt(Math.max(0, m2.data[i] - m.data[i] * m.data[i]));
  return { mean: m, sd };
}

export type LocalMethod = "meanC" | "niblack" | "sauvola";

export interface LocalOptions {
  r: number;        // window radius
  C?: number;       // mean − C: offset below the local mean (default 0)
  k?: number;       // Niblack (typically −0.2) / Sauvola (typically 0.2–0.5); default −0.2 / 0.2
  R?: number;       // Sauvola: dynamic range of the standard deviation (default 128)
}

/**
 * Threshold surface of a local method (object where I < T):
 *   mean − C:  T = m − C
 *   Niblack:   T = m + k s          (k ≈ −0.2)
 *   Sauvola:   T = m (1 + k (s/R − 1))   (k ≈ 0.2–0.5, R = 128)
 */
export function localThreshold(img: Grey, method: LocalMethod, o: LocalOptions): Grey {
  const { mean, sd } = localMeanSd(img, o.r);
  const T = createGrey(img.width, img.height), C = o.C ?? 0, R = o.R ?? 128;
  const k = o.k ?? (method === "niblack" ? -0.2 : 0.2);
  for (let i = 0; i < T.data.length; i++) {
    const m = mean.data[i], s = sd.data[i];
    T.data[i] = method === "meanC" ? m - C : method === "niblack" ? m + k * s : m * (1 + k * (s / R - 1));
  }
  return T;
}

/** Mask [I > T] (bright objects), or [I < T] with `dark`, for a threshold surface or a constant. */
export function applyThreshold(img: Grey | A, T: Grey | number, dark = false): Uint8Array {
  const d = "data" in (img as Grey) ? (img as Grey).data : (img as A);
  const out = new Uint8Array(d.length);
  for (let i = 0; i < d.length; i++) {
    const t = typeof T === "number" ? T : T.data[i];
    out[i] = (dark ? d[i] < t : d[i] > t) ? 1 : 0;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Hysteresis

/**
 * Hysteresis thresholding: pixels > high are seeds; pixels > low are kept if their
 * connected component (4 or 8) of the low mask contains a seed. Uses connected
 * components from lib/label.ts.
 */
export function hysteresis(data: A, w: number, h: number, low: number, high: number, conn: 4 | 8 = 8): {
  mask: Uint8Array; seeds: Uint8Array; lowMask: Uint8Array;
} {
  const n = w * h, lowMask = new Uint8Array(n), seeds = new Uint8Array(n);
  for (let i = 0; i < n; i++) { lowMask[i] = data[i] > low ? 1 : 0; seeds[i] = data[i] > high ? 1 : 0; }
  const lab = label2d(lowMask, w, h, conn);
  const keep = new Uint8Array(lab.count + 1);
  for (let i = 0; i < n; i++) if (seeds[i] && lab.data[i] > 0) keep[lab.data[i]] = 1;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = keep[lab.data[i]] && lab.data[i] > 0 ? 1 : 0;
  return { mask, seeds, lowMask };
}

/** 1D hysteresis along a signal: keep runs above `low` that contain a sample above `high`. */
export function hysteresis1d(y: A, low: number, high: number): Uint8Array {
  const n = y.length, out = new Uint8Array(n);
  let i = 0;
  while (i < n) {
    if (!(y[i] > low)) { i++; continue; }
    let j = i, seed = false;
    while (j < n && y[j] > low) { if (y[j] > high) seed = true; j++; }
    if (seed) out.fill(1, i, j);
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Two-Gaussian model (for the minimum-error picture)

/** Standard normal cumulative distribution (Abramowitz–Stegun 7.1.26 via erf, |error| < 1.5e-7). */
export function normalCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** Normal density. */
export const normalPdf = (x: number, mu: number, sd: number): number =>
  Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

/**
 * Points where ω0 N(μ0, σ0²) = ω1 N(μ1, σ1²) (the Bayes decision boundaries): the roots
 * of a quadratic, one root for equal variances. Sorted ascending.
 */
export function gaussianCrossings(mu0: number, s0: number, mu1: number, s1: number, w1: number): number[] {
  const w0 = 1 - w1;
  // ln(w0/s0) − (x−μ0)²/2s0² = ln(w1/s1) − (x−μ1)²/2s1²
  const a = 1 / (2 * s1 * s1) - 1 / (2 * s0 * s0);
  const b = mu0 / (s0 * s0) - mu1 / (s1 * s1);
  const c = mu1 * mu1 / (2 * s1 * s1) - mu0 * mu0 / (2 * s0 * s0) + Math.log((w0 * s1) / (w1 * s0));
  // a x² + b x + c = 0
  if (Math.abs(a) < 1e-14) return Math.abs(b) < 1e-14 ? [] : [-c / b];
  const d = b * b - 4 * a * c;
  if (d < 0) return [];
  const r = Math.sqrt(d);
  return [(-b - r) / (2 * a), (-b + r) / (2 * a)].sort((p, q) => p - q);
}

/** Error rate of the rule "object if x > t" under the two-Gaussian model. */
export function thresholdError(t: number, mu0: number, s0: number, mu1: number, s1: number, w1: number): number {
  return (1 - w1) * (1 - normalCdf((t - mu0) / s0)) + w1 * normalCdf((t - mu1) / s1);
}
