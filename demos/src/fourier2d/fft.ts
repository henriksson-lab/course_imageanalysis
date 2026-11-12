// Fourier demo helpers. The FFT itself lives in lib/fft.ts and rmse in lib/metrics.ts;
// they are re-exported here so the fourier2d demo is unchanged.
import { fft2d, freq } from "../lib/fft";
export { fft1d, fft2d, freq } from "../lib/fft";
export { rmse } from "../lib/metrics";

export type Mode = "all" | "radial" | "top";

/**
 * Build a 0/1 mask over the FFT indices (row-major, unshifted).
 * radial: keep the coefficients whose distance d from frequency 0 has rMin <= d <= rMax
 * (rMin = 0 is a low-pass, rMax = Infinity a high-pass, both a band-pass); with keepDC
 * the centre coefficient (the average brightness) is kept whatever rMin is.
 * top: keep the `keep` fraction of coefficients with the largest magnitude (`order`
 * lists indices from largest to smallest magnitude). all: keep everything.
 */
export function makeMask(n: number, mode: Mode, rMin: number, rMax: number, keepDC: boolean,
  keep: number, order: Uint32Array | null): Uint8Array {
  const m = new Uint8Array(n * n);
  if (mode === "top" && order) {
    const k = Math.max(1, Math.round(keep * n * n));
    for (let j = 0; j < k; j++) m[order[j]] = 1;
    return m;
  }
  if (mode !== "radial") return m.fill(1);
  for (let row = 0; row < n; row++) {
    const fy = freq(row, n);
    for (let col = 0; col < n; col++) {
      const fx = freq(col, n);
      const d = Math.sqrt(fx * fx + fy * fy);
      m[row * n + col] = (d >= rMin && d <= rMax) || (keepDC && d === 0) ? 1 : 0;
    }
  }
  return m;
}

/** Indices sorted by decreasing magnitude |F|. */
export function magnitudeOrder(re: Float64Array, im: Float64Array): Uint32Array {
  const n = re.length;
  const mag = new Float64Array(n);
  for (let k = 0; k < n; k++) mag[k] = re[k] * re[k] + im[k] * im[k];
  const idx = new Uint32Array(n);
  for (let k = 0; k < n; k++) idx[k] = k;
  return idx.sort((a, b) => mag[b] - mag[a] || a - b);
}

/** Inverse transform of F * mask; returns the real part. */
export function reconstruct(re: Float64Array, im: Float64Array, mask: Uint8Array, n: number): Float64Array {
  const r = new Float64Array(n * n), i = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) if (mask[k]) { r[k] = re[k]; i[k] = im[k]; }
  fft2d(r, i, n, true);
  return r;
}

/** The stripe artefact of the lecture: amp * sin(2 pi (ky*row + kx*col) / n). */
export function stripe(n: number, kx: number, ky: number, amp: number): Float64Array {
  const s = new Float64Array(n * n);
  for (let row = 0; row < n; row++)
    for (let col = 0; col < n; col++)
      s[row * n + col] = amp * Math.sin(2 * Math.PI * (ky * row + kx * col) / n);
  return s;
}
