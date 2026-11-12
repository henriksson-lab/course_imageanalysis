// The Fourier slice theorem in discrete form (no DOM). Spectra use the centre of the image
// as the origin (x right, y up, as in lib/radon.ts), so they are smooth enough to
// interpolate: Fc(fx, fy) = sum I(col, row) exp(-2 pi i (fx x + fy y)), x = col - cen,
// y = cen - row. Images are zero-padded to P = 2n, so the spectrum is sampled every 1/P
// cycles per pixel. The grid G holds Fc at fx = (i - P/2)/P, fy = (k - P/2)/P, index k * P + i.
import { fft1d, fft2dRect } from "../lib/fft";
import * as R from "../lib/radon";

export interface Spectrum { re: Float64Array; im: Float64Array; P: number }

/** Centred spectrum of an n x n image, zero-padded to P x P. */
export function centredSpectrum(img: ArrayLike<number>, n: number, P = 2 * n): Spectrum {
  const re = new Float64Array(P * P), im = new Float64Array(P * P), cen = (n - 1) / 2;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) re[r * P + c] = img[r * n + c];
  fft2dRect(re, im, P, P);
  const G = { re: new Float64Array(P * P), im: new Float64Array(P * P), P };
  for (let k = 0; k < P; k++) for (let i = 0; i < P; i++) {
    const u = i - P / 2, v = -(k - P / 2);            // fx = u/P, fy = -v/P
    const iu = (u + P) % P, iv = (v + P) % P, src = iv * P + iu;
    const ph = 2 * Math.PI * (u + v) * cen / P, c = Math.cos(ph), s = Math.sin(ph);
    const a = re[src], b = im[src];
    G.re[k * P + i] = a * c - b * s; G.im[k * P + i] = a * s + b * c;
  }
  return G;
}

/** Centred 1D spectrum of a projection (nBins = n, bins at rho = b - cen), padded to P: value at f = m/P, m = -P/2..P/2-1. */
export function projectionSpectrum(p: ArrayLike<number>, n: number, P = 2 * n): { f: Float64Array; re: Float64Array; im: Float64Array } {
  const re = new Float64Array(P), im = new Float64Array(P), cen = (n - 1) / 2;
  for (let b = 0; b < n; b++) re[b] = p[b];
  fft1d(re, im);
  const f = new Float64Array(P), oR = new Float64Array(P), oI = new Float64Array(P);
  for (let j = 0; j < P; j++) {
    const m = j - P / 2, src = (m + P) % P;
    const ph = 2 * Math.PI * m * cen / P, c = Math.cos(ph), s = Math.sin(ph);
    f[j] = m / P; oR[j] = re[src] * c - im[src] * s; oI[j] = re[src] * s + im[src] * c;
  }
  return { f, re: oR, im: oI };
}

/** Bilinear interpolation of the grid at (fx, fy) (cycles per pixel). */
export function sampleSpectrum(G: Spectrum, fx: number, fy: number): [number, number] {
  const P = G.P, i = fx * P + P / 2, k = fy * P + P / 2;
  const i0 = Math.floor(i), k0 = Math.floor(k), a = i - i0, b = k - k0;
  let sr = 0, si = 0;
  const add = (ii: number, kk: number, w: number) => {
    if (w === 0 || ii < 0 || ii >= P || kk < 0 || kk >= P) return;
    sr += w * G.re[kk * P + ii]; si += w * G.im[kk * P + ii];
  };
  add(i0, k0, (1 - a) * (1 - b)); add(i0 + 1, k0, a * (1 - b)); add(i0, k0 + 1, (1 - a) * b); add(i0 + 1, k0 + 1, a * b);
  return [sr, si];
}

/** The 2D spectrum along the central line at angle theta, at f = m/P. */
export function sliceOf(G: Spectrum, theta: number): { re: Float64Array; im: Float64Array } {
  const P = G.P, re = new Float64Array(P), im = new Float64Array(P), c = Math.cos(theta), s = Math.sin(theta);
  for (let j = 0; j < P; j++) {
    const f = (j - P / 2) / P;
    [re[j], im[j]] = sampleSpectrum(G, f * c, f * s);
  }
  return { re, im };
}

export type Gridding = "nearest" | "bilinear";

/**
 * Direct Fourier reconstruction: the 1D spectra of the projections at `angles` placed on
 * their spokes, gridded onto the Cartesian P x P grid (nearest cell, or bilinear weights;
 * cells hit several times are averaged, cells never hit are 0), then one inverse 2D FFT.
 * Returns the gridded spectrum and the n x n image.
 */
export function directRecon(sino: ArrayLike<number>, n: number, angles: Float64Array, mode: Gridding, P = 2 * n): { grid: Spectrum; img: Float64Array; filled: number } {
  const sr = new Float64Array(P * P), si = new Float64Array(P * P), w = new Float64Array(P * P);
  for (let a = 0; a < angles.length; a++) {
    const S = projectionSpectrum(Array.from({ length: n }, (_, b) => sino[a * n + b]), n, P);
    const c = Math.cos(angles[a]), s = Math.sin(angles[a]);
    for (let j = 0; j < P; j++) {
      const f = S.f[j], i = f * c * P + P / 2, k = f * s * P + P / 2;
      if (mode === "nearest") {
        const ii = Math.round(i), kk = Math.round(k);
        if (ii < 0 || ii >= P || kk < 0 || kk >= P) continue;
        const q = kk * P + ii;
        sr[q] += S.re[j]; si[q] += S.im[j]; w[q] += 1;
      } else {
        const i0 = Math.floor(i), k0 = Math.floor(k), fa = i - i0, fb = k - k0;
        const put = (ii: number, kk: number, ww: number) => {
          if (ww <= 0 || ii < 0 || ii >= P || kk < 0 || kk >= P) return;
          const q = kk * P + ii;
          sr[q] += ww * S.re[j]; si[q] += ww * S.im[j]; w[q] += ww;
        };
        put(i0, k0, (1 - fa) * (1 - fb)); put(i0 + 1, k0, fa * (1 - fb)); put(i0, k0 + 1, (1 - fa) * fb); put(i0 + 1, k0 + 1, fa * fb);
      }
    }
  }
  let filled = 0;
  for (let q = 0; q < P * P; q++) if (w[q] > 0) { sr[q] /= w[q]; si[q] /= w[q]; filled++; }
  const grid: Spectrum = { re: sr, im: si, P };
  // back to an uncentred array spectrum and invert
  const re = new Float64Array(P * P), im = new Float64Array(P * P), cen = (n - 1) / 2;
  for (let k = 0; k < P; k++) for (let i = 0; i < P; i++) {
    const u = i - P / 2, v = -(k - P / 2), dst = ((v + P) % P) * P + (u + P) % P;
    const ph = -2 * Math.PI * (u + v) * cen / P, c = Math.cos(ph), s = Math.sin(ph);
    const a = sr[k * P + i], b = si[k * P + i];
    re[dst] = a * c - b * s; im[dst] = a * s + b * c;
  }
  fft2dRect(re, im, P, P, true);
  const img = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) img[r * n + c] = re[r * P + c];
  return { grid, img, filled };
}

/** Stripes 0.5 + 0.5 cos(2 pi (x cos a + y sin a) / period) inside the field of view. */
export function stripes(n: number, alphaDeg: number, period: number): Float64Array {
  const m = R.fovMask(n), out = new Float64Array(n * n), a = alphaDeg * Math.PI / 180;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const [x, y] = R.toWorld(c, r, n);
    out[r * n + c] = m[r * n + c] * (0.5 + 0.5 * Math.cos(2 * Math.PI * (x * Math.cos(a) + y * Math.sin(a)) / period));
  }
  return out;
}

/** Relative difference ||a - b|| / ||a|| of two complex arrays. */
export function relDiff(aR: ArrayLike<number>, aI: ArrayLike<number>, bR: ArrayLike<number>, bI: ArrayLike<number>): number {
  let num = 0, den = 0;
  for (let j = 0; j < aR.length; j++) { num += (aR[j] - bR[j]) ** 2 + (aI[j] - bI[j]) ** 2; den += aR[j] ** 2 + aI[j] ** 2; }
  return Math.sqrt(num / Math.max(den, 1e-30));
}
