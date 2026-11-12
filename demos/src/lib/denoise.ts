// Classical denoisers (no DOM). Owner: the deconvolution day. Images are row-major
// Float64Arrays of size w × h (or Grey via the wrappers); borders are mirrored (c b | a b c).
//
//  - gaussianDenoise: linear low-pass (lib/imageops gaussianBlur);
//  - medianFilter: the middle value of a (2r+1)² window (rank filter; morphology owns the family);
//  - bilateral: weights e^{-|p-q|²/2σs²} · e^{-(I_p - I_q)²/2σr²}, normalised;
//  - nonLocalMeans: weights e^{-d²/h²} with d² the MEAN squared difference of two
//    (2P+1)² patches, over a (2S+1)² search window; computed per offset with box sums, so
//    the cost is (search window) × pixels, not × patch size;
//  - wienerDenoise: the Wiener filter with H = 1 and the 1/f² signal model (lib/deconv);
//  - methodNoise: noisy − denoised.
// Brute-force references (`*Brute`) are for the tests.
import { Grey, fromArray, gaussianBlur } from "./imageops";
import { spectrum2d, signalModel } from "./deconv";
import { fft2dRect } from "./fft";

const mirror = (i: number, n: number): number => {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
};

/** Mirror-pad a w × h image by r on every side. */
export function padMirror(img: ArrayLike<number>, w: number, h: number, r: number): { data: Float64Array; W: number; H: number } {
  const W = w + 2 * r, H = h + 2 * r, out = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = img[mirror(y - r, h) * w + mirror(x - r, w)];
  return { data: out, W, H };
}

/** Gaussian smoothing with standard deviation sigma (pixels). */
export function gaussianDenoise(img: ArrayLike<number>, w: number, h: number, sigma: number): Float64Array {
  if (sigma <= 0) return Float64Array.from(img);
  return Float64Array.from(gaussianBlur(fromArray(w, h, img), sigma, "mirror").data);
}

/** Median of a (2r+1) × (2r+1) window. */
export function medianFilter(img: ArrayLike<number>, w: number, h: number, r: number): Float64Array {
  if (r <= 0) return Float64Array.from(img);
  const P = padMirror(img, w, h, r), k = 2 * r + 1, buf = new Float64Array(k * k), out = new Float64Array(w * h), mid = (k * k) >> 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let n = 0;
    for (let j = 0; j < k; j++) { const o = (y + j) * P.W + x; for (let i = 0; i < k; i++) buf[n++] = P.data[o + i]; }
    out[y * w + x] = select(buf, mid);
  }
  return out;
}

/** k-th smallest value (quickselect, reorders `a`). */
function select(a: Float64Array, k: number): number {
  let lo = 0, hi = a.length - 1;
  while (hi > lo) {
    const piv = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < piv) i++;
      while (a[j] > piv) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
  }
  return a[k];
}

/** 1D median with an odd window, mirrored ends (for the by-hand examples). */
export function median1d(x: ArrayLike<number>, window: number): Float64Array {
  const r = (window - 1) >> 1, n = x.length, out = new Float64Array(n), buf = new Float64Array(window);
  for (let i = 0; i < n; i++) {
    for (let j = -r; j <= r; j++) buf[j + r] = x[mirror(i + j, n)];
    out[i] = select(buf, r);
  }
  return out;
}

/** Bilateral filter; the window radius defaults to ceil(2σs). */
export function bilateral(img: ArrayLike<number>, w: number, h: number, sigmaS: number, sigmaR: number, radius = Math.ceil(2 * sigmaS)): Float64Array {
  const r = Math.max(1, radius), P = padMirror(img, w, h, r), out = new Float64Array(w * h);
  const ws = new Float64Array((2 * r + 1) ** 2);
  for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) ws[(j + r) * (2 * r + 1) + i + r] = Math.exp(-(i * i + j * j) / (2 * sigmaS * sigmaS));
  const kr = 1 / (2 * sigmaR * sigmaR);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = P.data[(y + r) * P.W + x + r];
    let s = 0, sw = 0, t = 0;
    for (let j = 0; j <= 2 * r; j++) {
      const o = (y + j) * P.W + x;
      for (let i = 0; i <= 2 * r; i++, t++) {
        const v = P.data[o + i], d = v - c, wt = ws[t] * Math.exp(-d * d * kr);
        s += wt * v; sw += wt;
      }
    }
    out[y * w + x] = s / sw;
  }
  return out;
}

/**
 * Non-local means. patch = P (patch (2P+1)²), search = S (window (2S+1)²), h in grey levels:
 * w(p, q) = exp(−d²(p, q)/h²), d² = mean squared difference of the two patches.
 */
export function nonLocalMeans(img: ArrayLike<number>, w: number, h: number, patch: number, search: number, hh: number): Float64Array {
  const R = patch + search, E = padMirror(img, w, h, R), W = E.W, H = E.H;
  const num = new Float64Array(w * h), den = new Float64Array(w * h);
  const D = new Float64Array(W * H), C = new Float64Array((W + 1) * (H + 1));
  const area = (2 * patch + 1) ** 2, k = 1 / (hh * hh * area);
  for (let dy = -search; dy <= search; dy++) for (let dx = -search; dx <= search; dx++) {
    // squared differences between the image and its shift by (dx, dy), over the region where both exist
    for (let y = search; y < H - search; y++) for (let x = search; x < W - search; x++) {
      const d = E.data[y * W + x] - E.data[(y + dy) * W + x + dx];
      D[y * W + x] = d * d;
    }
    // integral image of D
    for (let y = 0; y < H; y++) {
      let row = 0;
      for (let x = 0; x < W; x++) { row += D[y * W + x]; C[(y + 1) * (W + 1) + x + 1] = C[y * (W + 1) + x + 1] + row; }
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const cy = y + R, cx = x + R;   // padded coordinates
      const y0 = cy - patch, y1 = cy + patch + 1, x0 = cx - patch, x1 = cx + patch + 1;
      const ss = C[y1 * (W + 1) + x1] - C[y0 * (W + 1) + x1] - C[y1 * (W + 1) + x0] + C[y0 * (W + 1) + x0];
      const wt = Math.exp(-ss * k);
      num[y * w + x] += wt * E.data[(cy + dy) * W + cx + dx];
      den[y * w + x] += wt;
    }
  }
  for (let i = 0; i < w * h; i++) num[i] /= den[i];
  return num;
}

/** Brute-force non-local means on the same mirrored extension (reference for the tests). */
export function nonLocalMeansBrute(img: ArrayLike<number>, w: number, h: number, patch: number, search: number, hh: number): Float64Array {
  const R = patch + search, E = padMirror(img, w, h, R), W = E.W, out = new Float64Array(w * h);
  const area = (2 * patch + 1) ** 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const cy = y + R, cx = x + R;
    let s = 0, sw = 0;
    for (let dy = -search; dy <= search; dy++) for (let dx = -search; dx <= search; dx++) {
      let d2 = 0;
      for (let j = -patch; j <= patch; j++) for (let i = -patch; i <= patch; i++) {
        const d = E.data[(cy + j) * W + cx + i] - E.data[(cy + dy + j) * W + cx + dx + i];
        d2 += d * d;
      }
      const wt = Math.exp(-d2 / area / (hh * hh));
      s += wt * E.data[(cy + dy) * W + cx + dx]; sw += wt;
    }
    out[y * w + x] = s / sw;
  }
  return out;
}

/** Brute-force median (full sort) and bilateral, for the tests. */
export function medianBrute(img: ArrayLike<number>, w: number, h: number, r: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v: number[] = [];
    for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) v.push(img[mirror(y + j, h) * w + mirror(x + i, w)]);
    v.sort((a, b) => a - b);
    out[y * w + x] = v[(v.length - 1) / 2];
  }
  return out;
}
export function bilateralBrute(img: ArrayLike<number>, w: number, h: number, sigmaS: number, sigmaR: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = img[y * w + x];
    let s = 0, sw = 0;
    for (let j = -radius; j <= radius; j++) for (let i = -radius; i <= radius; i++) {
      const v = img[mirror(y + j, h) * w + mirror(x + i, w)];
      const wt = Math.exp(-(i * i + j * j) / (2 * sigmaS ** 2) - (v - c) ** 2 / (2 * sigmaR ** 2));
      s += wt * v; sw += wt;
    }
    out[y * w + x] = s / sw;
  }
  return out;
}

/**
 * Wiener denoising (no blur, H = 1): gain 1/(1 + 1/(SNR·p(f))) with the 1/f² signal model
 * p(f) of lib/deconv. The mean (f = 0) is kept almost exactly. w, h powers of two.
 */
export function wienerDenoise(img: ArrayLike<number>, w: number, h: number, snr: number): Float64Array {
  const S = spectrum2d(img, w, h), p = signalModel(w, h);
  for (let i = 0; i < w * h; i++) { const g = 1 / (1 + 1 / (snr * p[i])); S.re[i] *= g; S.im[i] *= g; }
  fft2dRect(S.re, S.im, w, h, true);
  return S.re;
}

/** Method noise: noisy − denoised. */
export function methodNoise(noisy: ArrayLike<number>, denoised: ArrayLike<number>): Float64Array {
  const out = new Float64Array(noisy.length);
  for (let i = 0; i < out.length; i++) out[i] = noisy[i] - denoised[i];
  return out;
}

/** Grey wrappers. */
export const toGrey = (a: ArrayLike<number>, w: number, h: number): Grey => fromArray(w, h, a);
