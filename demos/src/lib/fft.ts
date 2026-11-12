// Fast Fourier transforms in 1D, 2D and 3D (no DOM). Radix-2: every transformed length
// must be a power of two; use `nextPow2` + `padTo` for other sizes, or `dft1d` (O(n²)) as
// a reference for any length.
//
// Convention (as R's fft()): forward X[k] = Σ_t x[t] exp(-2πi k t / N); `fft1d` is
// unnormalised in both directions, while the 2D/3D inverses divide by the number of
// samples, so a forward + inverse round trip is exact. Arrays are row-major
// (index = y*w + x, and z*w*h + y*w + x in 3D), real and imaginary parts in two
// Float64Arrays, transformed in place.

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

/** Reference DFT of any length (O(n²)), same convention as `fft1d`; returns new arrays. */
export function dft1d(re: ArrayLike<number>, im: ArrayLike<number>, inverse = false): { re: Float64Array; im: Float64Array } {
  const n = re.length, oR = new Float64Array(n), oI = new Float64Array(n);
  const sgn = inverse ? 1 : -1;
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < n; t++) {
      const a = sgn * 2 * Math.PI * ((k * t) % n) / n, c = Math.cos(a), s = Math.sin(a);
      sr += re[t] * c - im[t] * s; si += re[t] * s + im[t] * c;
    }
    oR[k] = sr; oI[k] = si;
  }
  return { re: oR, im: oI };
}

/** In-place 2D FFT of a w × h image (both powers of two). The inverse divides by w·h. */
export function fft2dRect(re: Float64Array, im: Float64Array, w: number, h: number, inverse = false): void {
  const r = new Float64Array(w), i = new Float64Array(w);
  for (let row = 0; row < h; row++) {
    const o = row * w;
    for (let c = 0; c < w; c++) { r[c] = re[o + c]; i[c] = im[o + c]; }
    fft1d(r, i, inverse);
    for (let c = 0; c < w; c++) { re[o + c] = r[c]; im[o + c] = i[c]; }
  }
  const rc = new Float64Array(h), ic = new Float64Array(h);
  for (let col = 0; col < w; col++) {
    for (let row = 0; row < h; row++) { rc[row] = re[row * w + col]; ic[row] = im[row * w + col]; }
    fft1d(rc, ic, inverse);
    for (let row = 0; row < h; row++) { re[row * w + col] = rc[row]; im[row * w + col] = ic[row]; }
  }
  if (inverse) {
    const s = 1 / (w * h);
    for (let k = 0; k < w * h; k++) { re[k] *= s; im[k] *= s; }
  }
}

/** In-place 2D FFT of an n × n image stored row-major. The inverse divides by n·n. */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse = false): void {
  fft2dRect(re, im, n, n, inverse);
}

/** In-place 3D FFT of an nx × ny × nz volume (powers of two). The inverse divides by nx·ny·nz. */
export function fft3d(re: Float64Array, im: Float64Array, nx: number, ny: number, nz: number, inverse = false): void {
  const plane = nx * ny;
  for (let z = 0; z < nz; z++)   // each xy plane (the inverse divides by nx·ny here)
    fft2dRect(re.subarray(z * plane, (z + 1) * plane), im.subarray(z * plane, (z + 1) * plane), nx, ny, inverse);
  const rz = new Float64Array(nz), iz = new Float64Array(nz);
  for (let p = 0; p < plane; p++) {
    for (let z = 0; z < nz; z++) { rz[z] = re[z * plane + p]; iz[z] = im[z * plane + p]; }
    fft1d(rz, iz, inverse);
    const s = inverse ? 1 / nz : 1;
    for (let z = 0; z < nz; z++) { re[z * plane + p] = rz[z] * s; im[z * plane + p] = iz[z] * s; }
  }
}

/** Signed frequency of FFT index k (0..n-1): 0, 1, ..., n/2-1, -n/2, ..., -1. */
export const freq = (k: number, n: number): number => (k < n / 2 ? k : k - n);

/** Smallest power of two ≥ n. */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export const isPow2 = (n: number): boolean => n > 0 && (n & (n - 1)) === 0;

/**
 * Copy a w × h image into the top-left corner of a W × H array filled with `fill`
 * (zero-padding for linear instead of circular convolution, or to reach a power of two).
 */
export function padTo(a: ArrayLike<number>, w: number, h: number, W: number, H: number, fill = 0): Float64Array {
  const out = new Float64Array(W * H);
  if (fill !== 0) out.fill(fill);
  for (let y = 0; y < Math.min(h, H); y++)
    for (let x = 0; x < Math.min(w, W); x++) out[y * W + x] = a[y * w + x];
  return out;
}

/** The w × h window starting at (x0, y0) of a W-wide array. */
export function crop(a: ArrayLike<number>, W: number, x0: number, y0: number, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = a[(y0 + y) * W + x0 + x];
  return out;
}

/**
 * Swap quadrants so frequency 0 moves to the centre (index (h/2)·w + w/2), as for display.
 * Works for any w, h (odd sizes too). Returns a new array.
 */
export function fftshift(a: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h), sx = Math.floor(w / 2), sy = Math.floor(h / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out[((y + sy) % h) * w + (x + sx) % w] = a[y * w + x];
  return out;
}

/** Inverse of `fftshift` (they differ only for odd sizes). */
export function ifftshift(a: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h), sx = Math.ceil(w / 2), sy = Math.ceil(h / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out[((y + sy) % h) * w + (x + sx) % w] = a[y * w + x];
  return out;
}

/** Forward 2D FFT of a real w × h image (powers of two); returns new arrays. */
export function fftReal2d(img: ArrayLike<number>, w: number, h: number): { re: Float64Array; im: Float64Array } {
  const re = Float64Array.from(img as ArrayLike<number>), im = new Float64Array(w * h);
  fft2dRect(re, im, w, h, false);
  return { re, im };
}

/** Inverse 2D FFT returning only the real part (the imaginary part is dropped). */
export function ifftReal2d(re: ArrayLike<number>, im: ArrayLike<number>, w: number, h: number): Float64Array {
  const r = Float64Array.from(re), i = Float64Array.from(im);
  fft2dRect(r, i, w, h, true);
  return r;
}

/** |F| per coefficient. */
export function magnitude(re: ArrayLike<number>, im: ArrayLike<number>): Float64Array {
  const out = new Float64Array(re.length);
  for (let k = 0; k < re.length; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

/** Phase angle atan2(im, re) per coefficient, in (-π, π]. */
export function phase(re: ArrayLike<number>, im: ArrayLike<number>): Float64Array {
  const out = new Float64Array(re.length);
  for (let k = 0; k < re.length; k++) out[k] = Math.atan2(im[k], re[k]);
  return out;
}

/** log(1 + |F|), the usual display of a spectrum. */
export function logMagnitude(re: ArrayLike<number>, im: ArrayLike<number>): Float64Array {
  const out = new Float64Array(re.length);
  for (let k = 0; k < re.length; k++) out[k] = Math.log1p(Math.hypot(re[k], im[k]));
  return out;
}

/** Complex product a·b (or a·conj(b) with `conjB`), element-wise; returns new arrays. */
export function complexMul(aRe: ArrayLike<number>, aIm: ArrayLike<number>, bRe: ArrayLike<number>, bIm: ArrayLike<number>,
  conjB = false): { re: Float64Array; im: Float64Array } {
  const n = aRe.length, re = new Float64Array(n), im = new Float64Array(n), s = conjB ? -1 : 1;
  for (let k = 0; k < n; k++) {
    const br = bRe[k], bi = s * bIm[k];
    re[k] = aRe[k] * br - aIm[k] * bi;
    im[k] = aRe[k] * bi + aIm[k] * br;
  }
  return { re, im };
}

/** Complex quotient a / (b + eps) element-wise, with eps added to |b|² (regularised division). */
export function complexDiv(aRe: ArrayLike<number>, aIm: ArrayLike<number>, bRe: ArrayLike<number>, bIm: ArrayLike<number>,
  eps = 0): { re: Float64Array; im: Float64Array } {
  const n = aRe.length, re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const d = bRe[k] * bRe[k] + bIm[k] * bIm[k] + eps;
    if (d === 0) continue;
    re[k] = (aRe[k] * bRe[k] + aIm[k] * bIm[k]) / d;
    im[k] = (aIm[k] * bRe[k] - aRe[k] * bIm[k]) / d;
  }
  return { re, im };
}

/**
 * Circular convolution of two real w × h images via the FFT (powers of two):
 * out[y,x] = Σ a[v,u] b[(y-v) mod h, (x-u) mod w]. Pad first with `padTo` for linear convolution.
 */
export function convolveFFT(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number): Float64Array {
  const A = fftReal2d(a, w, h), B = fftReal2d(b, w, h);
  const P = complexMul(A.re, A.im, B.re, B.im);
  return ifftReal2d(P.re, P.im, w, h);
}

/**
 * Circular cross-correlation via the FFT: out[dy,dx] = Σ a[y,x] b[(y+dy) mod h, (x+dx) mod w].
 * The peak sits at the shift that moves a onto b.
 */
export function correlateFFT(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number): Float64Array {
  const A = fftReal2d(a, w, h), B = fftReal2d(b, w, h);
  const P = complexMul(B.re, B.im, A.re, A.im, true);
  return ifftReal2d(P.re, P.im, w, h);
}
