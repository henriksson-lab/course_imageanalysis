// Radix-2 FFT in 1D and 2D (no DOM, so it can be tested with node).
// Same convention as R's fft(): forward X[k] = sum_t x[t] exp(-2 pi i k t / N),
// the inverse is unnormalised; fft2d(..., true) divides by N*N so a round trip is exact.

/** In-place 1D FFT of (re, im), length a power of two. Unnormalised in both directions. */
export function fft1d(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sgn * 2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

/** In-place 2D FFT of an n x n image stored row-major. The inverse divides by n*n. */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse = false): void {
  const r = new Float64Array(n), i = new Float64Array(n);
  for (let row = 0; row < n; row++) {
    const o = row * n;
    for (let c = 0; c < n; c++) { r[c] = re[o + c]; i[c] = im[o + c]; }
    fft1d(r, i, inverse);
    for (let c = 0; c < n; c++) { re[o + c] = r[c]; im[o + c] = i[c]; }
  }
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < n; row++) { r[row] = re[row * n + col]; i[row] = im[row * n + col]; }
    fft1d(r, i, inverse);
    for (let row = 0; row < n; row++) { re[row * n + col] = r[row]; im[row * n + col] = i[row]; }
  }
  if (inverse) {
    const s = 1 / (n * n);
    for (let k = 0; k < n * n; k++) { re[k] *= s; im[k] *= s; }
  }
}

/** Signed frequency of FFT index k (0..n-1): 0, 1, ..., n/2-1, -n/2, ..., -1. */
export const freq = (k: number, n: number): number => (k < n / 2 ? k : k - n);

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

/** Root-mean-square difference between two images. */
export function rmse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
  return Math.sqrt(s / a.length);
}

/** The stripe artefact of the lecture: amp * sin(2 pi (ky*row + kx*col) / n). */
export function stripe(n: number, kx: number, ky: number, amp: number): Float64Array {
  const s = new Float64Array(n * n);
  for (let row = 0; row < n; row++)
    for (let col = 0; col < n; col++)
      s[row * n + col] = amp * Math.sin(2 * Math.PI * (ky * row + kx * col) / n);
  return s;
}
