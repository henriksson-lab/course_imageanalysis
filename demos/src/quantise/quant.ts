// Uniform quantisation of values in [0, 1] to b bits, with optional dithering.
// Stored code k in 0..L (L = 2^b - 1), reconstructed value k / L, so the step is
// Delta = 1 / L and 0 and 1 are exact (the convention of 8-bit images and of the GPU's
// normalised formats). Used by the quantise demo and the tests.
import { rng } from "../lib/ui";

export type Dither = "none" | "random" | "diffuse";

export const levels = (b: number) => 2 ** b;
export const step = (b: number) => 1 / (2 ** b - 1);

/**
 * Quantise `x` (row-major, w x h, values in [0, 1]) to b bits. Returns the reconstructed
 * values k / L.
 *  - none:    k = round(x L)
 *  - random:  k = round(x L + u), u uniform on [-1/2, 1/2) (fixed seed)
 *  - diffuse: Floyd-Steinberg error diffusion, left to right, top to bottom
 */
export function quantise(x: ArrayLike<number>, w: number, h: number, b: number, dither: Dither, seed = 11): Float64Array {
  const L = 2 ** b - 1;
  const n = w * h;
  const out = new Float64Array(n);
  const clampK = (k: number) => Math.min(L, Math.max(0, k));
  if (dither === "none") {
    for (let i = 0; i < n; i++) out[i] = clampK(Math.round(x[i] * L)) / L;
  } else if (dither === "random") {
    const r = rng(seed);
    for (let i = 0; i < n; i++) out[i] = clampK(Math.round(x[i] * L + r.unif() - 0.5)) / L;
  } else {
    const buf = Float64Array.from(x as ArrayLike<number>, v => v * L);
    for (let y = 0; y < h; y++) for (let c = 0; c < w; c++) {
      const i = y * w + c;
      const k = clampK(Math.round(buf[i]));
      const err = buf[i] - k;
      out[i] = k / L;
      if (c + 1 < w) buf[i + 1] += err * 7 / 16;
      if (y + 1 < h) {
        if (c > 0) buf[i + w - 1] += err * 3 / 16;
        buf[i + w] += err * 5 / 16;
        if (c + 1 < w) buf[i + w + 1] += err * 1 / 16;
      }
    }
  }
  return out;
}

export interface QStats {
  rms: number;        // RMS of (quantised - original)
  meanErr: number;
  theory: number;     // Delta / sqrt(12)
  snrDb: number;      // 10 log10(var(signal) / mean(err^2))
  distinct: number;   // number of different output values used
}

export function stats(x: ArrayLike<number>, q: ArrayLike<number>, b: number): QStats {
  const n = x.length;
  let m = 0, e1 = 0, e2 = 0;
  for (let i = 0; i < n; i++) { m += x[i]; const e = q[i] - x[i]; e1 += e; e2 += e * e; }
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  v /= n;
  const seen = new Set<number>();
  for (let i = 0; i < n; i++) seen.add(q[i]);
  return {
    rms: Math.sqrt(e2 / n), meanErr: e1 / n, theory: step(b) / Math.sqrt(12),
    snrDb: e2 > 0 ? 10 * Math.log10(v / (e2 / n)) : Infinity, distinct: seen.size,
  };
}

/** Histogram of the error in units of Delta over [-range, range], `bins` bins. */
export function errorHist(x: ArrayLike<number>, q: ArrayLike<number>, b: number, bins: number, range: number): number[] {
  const L = 2 ** b - 1;
  const h = new Array<number>(bins).fill(0);
  for (let i = 0; i < x.length; i++) {
    const e = (q[i] - x[i]) * L;
    const k = Math.floor((e + range) / (2 * range) * bins);
    if (k >= 0 && k < bins) h[k]++;
  }
  return h;
}

// ---- test images (values in [0, 1], N x N, row-major) ----

/** A ramp from 0 (left) to 1 (right) that never repeats a value: column j, row i. */
export function ramp(N: number): Float64Array {
  const out = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) out[i * N + j] = (j * N + i) / (N * N - 1);
  return out;
}

/** Smooth blobs on a slowly varying background, values in about [0.05, 0.95]. */
export function blobs(N: number): Float64Array {
  const out = new Float64Array(N * N);
  const g = [[0.32, 0.35, 0.16, 0.65], [0.7, 0.3, 0.12, 0.5], [0.55, 0.72, 0.2, 0.75], [0.2, 0.8, 0.09, 0.35]];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = (j + 0.5) / N, y = (i + 0.5) / N;
    let v = 0.08 + 0.1 * x;
    for (const [cx, cy, s, a] of g) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
    out[i * N + j] = Math.min(0.95, v);
  }
  return out;
}
