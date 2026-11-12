// Restoration of blurred, noisy images (no DOM). Owner: the deconvolution day.
//
//  1. Linear restoration as a gain per frequency (2D, lib/fft layout): inverse, truncated
//     inverse, Wiener (flat or 1/f² signal model), Tikhonov (energy or roughness penalty).
//     Every method is W(f) = conj(H(f))·g(f) with a real g; the restored spectrum is W·G.
//  2. Regularisation in 1D (circular, length a power of two): Tikhonov in closed form,
//     filter factors, Landweber (closed form and iterative, optionally non-negative),
//     roughness + non-negativity (projected FISTA), total variation (IRLS + conjugate
//     gradients), the L-curve corner and the discrepancy-principle λ.
//  3. Richardson–Lucy for any linear operator with a forward and an adjoint (1D direct, 2D
//     and 3D by FFT), with a background term and an iteration callback; blind RL.
//
// Notation as in the units: g = data, h = PSF, H = OTF, λ = regularisation weight,
// σ_n = noise standard deviation, SNR = signal power / noise power.
import { fft1d, fft2dRect, fft3d, freq } from "./fft";
import { toOrigin, toOrigin3d } from "./psf";

export interface Spectrum { re: Float64Array; im: Float64Array }

// ---------------------------------------------------------------------------------------
// 1. Linear restoration in Fourier space

export type LinearMethod =
  | { kind: "inverse" }
  | { kind: "truncated"; eps: number }
  | { kind: "wiener"; snr: number; model: "flat" | "1/f2" }
  | { kind: "tikhonov"; lambda: number; penalty: "energy" | "roughness" };

/** |D̂(f)|² of the 5-point Laplacian (4 − 2cos 2πfx − 2cos 2πfy)², lib/fft layout. 0 at f = 0. */
export function laplacianSymbol2(w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
    const L = 4 - 2 * Math.cos(2 * Math.PI * u / w) - 2 * Math.cos(2 * Math.PI * v / h);
    out[v * w + u] = L * L;
  }
  return out;
}

/** Corner frequency of the 1/f² signal model (cycles per pixel). */
export const F0 = 0.02;

/**
 * The 1/f² signal power model p(f) = 1/(f² + f0²), scaled so that its mean over all
 * frequencies is 1: then SNR·p(f) is the SNR per frequency for an image-wide SNR.
 */
export function signalModel(w: number, h: number, f0 = F0): Float64Array {
  const out = new Float64Array(w * h);
  let s = 0;
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
    const fx = freq(u, w) / w, fy = freq(v, h) / h, p = 1 / (fx * fx + fy * fy + f0 * f0);
    out[v * w + u] = p; s += p;
  }
  const m = s / (w * h);
  for (let i = 0; i < out.length; i++) out[i] /= m;
  return out;
}

/** Context for gains: |H|² and the precomputed symbols (computed once per size). */
export interface GainContext { H: Spectrum; H2: Float64Array; lap2: Float64Array; model: Float64Array; w: number; h: number }

export function gainContext(H: Spectrum, w: number, h: number): GainContext {
  const H2 = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) H2[i] = H.re[i] * H.re[i] + H.im[i] * H.im[i];
  return { H, H2, lap2: laplacianSymbol2(w, h), model: signalModel(w, h), w, h };
}

/** Real factor g(f) at index i, so that W = conj(H)·g. */
export function gainFactor(m: LinearMethod, c: GainContext, i: number): number {
  const h2 = c.H2[i];
  switch (m.kind) {
    case "inverse": return h2 > 1e-20 ? 1 / h2 : 0;   // |H| < 1e-10 is zero up to rounding: nothing to divide by
    case "truncated": return h2 > m.eps * m.eps ? 1 / h2 : 0;
    case "wiener": {
      const nsr = m.model === "flat" ? 1 / m.snr : 1 / (m.snr * c.model[i]);
      return 1 / (h2 + nsr);
    }
    case "tikhonov": {
      const d2 = m.penalty === "energy" ? 1 : c.lap2[i];
      const den = h2 + m.lambda * d2;
      return den > 1e-300 ? 1 / den : 0;
    }
  }
}

/** The filter W(f) = conj(H)·g(f), as a spectrum. */
export function filterOf(m: LinearMethod, c: GainContext): Spectrum {
  const n = c.w * c.h, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { const g = gainFactor(m, c, i); re[i] = c.H.re[i] * g; im[i] = -c.H.im[i] * g; }
  return { re, im };
}

/** Restore from the data spectrum G: x = F⁻¹(W·G). */
export function restoreSpectrum(G: Spectrum, m: LinearMethod, c: GainContext): Float64Array {
  const n = c.w * c.h, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = gainFactor(m, c, i), hr = c.H.re[i] * g, hi = -c.H.im[i] * g;
    re[i] = hr * G.re[i] - hi * G.im[i];
    im[i] = hr * G.im[i] + hi * G.re[i];
  }
  fft2dRect(re, im, c.w, c.h, true);
  return re;
}

/** Forward FFT of a real image. */
export function spectrum2d(img: ArrayLike<number>, w: number, h: number): Spectrum {
  const re = Float64Array.from(img), im = new Float64Array(w * h);
  fft2dRect(re, im, w, h, false);
  return { re, im };
}

/** One-call linear restoration of a w × h image with a centred PSF. */
export function linearRestore(g: ArrayLike<number>, psf: ArrayLike<number>, w: number, h: number, m: LinearMethod): Float64Array {
  const H = spectrum2d(toOrigin(psf, w, h), w, h);
  return restoreSpectrum(spectrum2d(g, w, h), m, gainContext(H, w, h));
}

/**
 * Extend a w × h image to W × H (≥) for FFT restoration without border wrap: the image sits
 * at offset (ox, oy), the margin continues the nearest edge pixel and fades to the image
 * mean with a raised-cosine taper, so the periodic copies meet smoothly.
 */
export function padTaper(img: ArrayLike<number>, w: number, h: number, W: number, H: number): { data: Float64Array; ox: number; oy: number } {
  const ox = Math.floor((W - w) / 2), oy = Math.floor((H - h) / 2), out = new Float64Array(W * H);
  let mean = 0;
  for (let i = 0; i < w * h; i++) mean += img[i];
  mean /= w * h;
  // position in the (periodic) margin: nearest image pixel and distance (in units of half the margin)
  const side = (X: number, o: number, n: number, N: number): [number, number] => {
    const x = X - o;
    if (x >= 0 && x < n) return [x, 0];
    const M = N - n, t = ((x - n) % N + N) % N;   // 0 just right of the image
    const a = t + 1, b = M - t;                     // distance to the right / left edge
    return a <= b ? [n - 1, a / (M / 2)] : [0, b / (M / 2)];
  };
  for (let Y = 0; Y < H; Y++) {
    const [y, dy] = side(Y, oy, h, H);
    for (let X = 0; X < W; X++) {
      const [x, dx] = side(X, ox, w, W);
      const t = Math.min(1, Math.max(dx, dy)), wgt = 0.5 + 0.5 * Math.cos(Math.PI * t);
      out[Y * W + X] = wgt * img[y * w + x] + (1 - wgt) * mean;
    }
  }
  return { data: out, ox, oy };
}

// ---------------------------------------------------------------------------------------
// 2. Regularisation in 1D (circular)

/** A circular 1D blur of length n (power of two) from a centred kernel. */
export interface Blur1D { n: number; re: Float64Array; im: Float64Array; H2: Float64Array }

export function blur1dOf(kernel: ArrayLike<number>): Blur1D {
  const n = kernel.length, re = new Float64Array(n), im = new Float64Array(n), c = Math.floor(n / 2);
  for (let i = 0; i < n; i++) re[(i - c + n) % n] = kernel[i];
  fft1d(re, im, false);
  const H2 = new Float64Array(n);
  for (let i = 0; i < n; i++) H2[i] = re[i] * re[i] + im[i] * im[i];
  return { n, re, im, H2 };
}

/** Centred Gaussian kernel of length n with standard deviation sigma (sum 1). */
export function gaussian1d(n: number, sigma: number): Float64Array {
  const k = new Float64Array(n), c = Math.floor(n / 2);
  if (sigma <= 0) { k[c] = 1; return k; }
  let s = 0;
  for (let i = 0; i < n; i++) { k[i] = Math.exp(-((i - c) ** 2) / (2 * sigma * sigma)); s += k[i]; }
  for (let i = 0; i < n; i++) k[i] /= s;
  return k;
}

function fwd1(x: ArrayLike<number>): Spectrum {
  const re = Float64Array.from(x), im = new Float64Array(x.length);
  fft1d(re, im, false);
  return { re, im };
}
function inv1(S: Spectrum): Float64Array {
  const n = S.re.length, re = S.re.slice(), im = S.im.slice();
  fft1d(re, im, true);
  for (let i = 0; i < n; i++) re[i] /= n;
  return re;
}
/** Multiply a spectrum by H (or conj H) and return the real signal. */
function applyH(X: Spectrum, B: Blur1D, conj: boolean): Float64Array {
  const n = B.n, re = new Float64Array(n), im = new Float64Array(n), s = conj ? -1 : 1;
  for (let i = 0; i < n; i++) {
    const hr = B.re[i], hi = s * B.im[i];
    re[i] = hr * X.re[i] - hi * X.im[i]; im[i] = hr * X.im[i] + hi * X.re[i];
  }
  return inv1({ re, im });
}

/** A x (circular blur). */
export const blur1d = (x: ArrayLike<number>, B: Blur1D): Float64Array => applyH(fwd1(x), B, false);
/** Aᵀ x (circular correlation with the kernel). */
export const blurT1d = (x: ArrayLike<number>, B: Blur1D): Float64Array => applyH(fwd1(x), B, true);

export type Penalty1D = "energy" | "gradient" | "roughness";

/** |D̂(f)|² of the 1D penalty operator: 1 (energy), first difference, second difference (roughness). */
export function penaltySymbol1d(n: number, p: Penalty1D): Float64Array {
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const c = 2 - 2 * Math.cos(2 * Math.PI * k / n);   // |first difference|²
    out[k] = p === "energy" ? 1 : p === "gradient" ? c : c * c;
  }
  return out;
}

/** Apply the penalty operator D: identity, first difference x[i+1] − x[i], or second difference x[i−1] − 2x[i] + x[i+1] (circular). */
export function penaltyOp1d(x: ArrayLike<number>, p: Penalty1D): Float64Array {
  const n = x.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (p === "energy") out[i] = x[i];
    else if (p === "gradient") out[i] = x[(i + 1) % n] - x[i];
    else out[i] = x[(i - 1 + n) % n] - 2 * x[i] + x[(i + 1) % n];
  }
  return out;
}

/** Tikhonov filter factors φ = |H|² / (|H|² + λ|D̂|²) per frequency index. */
export function tikhonovFactors(H2: ArrayLike<number>, lambda: number, D2: ArrayLike<number>): Float64Array {
  const out = new Float64Array(H2.length);
  for (let k = 0; k < H2.length; k++) { const d = H2[k] + lambda * D2[k]; out[k] = d > 0 ? H2[k] / d : 0; }
  return out;
}

/** Landweber filter factors φ = 1 − (1 − ω|H|²)^k. */
export function landweberFactors(H2: ArrayLike<number>, k: number, omega = 1): Float64Array {
  const out = new Float64Array(H2.length);
  for (let i = 0; i < H2.length; i++) out[i] = 1 - Math.pow(1 - omega * H2[i], k);
  return out;
}

/** Restore with given filter factors: X = φ·G/H (0 where H = 0). */
export function applyFactors(y: ArrayLike<number>, B: Blur1D, phi: ArrayLike<number>): Float64Array {
  const Y = fwd1(y), n = B.n, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const h2 = B.H2[i];
    if (h2 < 1e-300) continue;
    const s = phi[i] / h2, hr = B.re[i] * s, hi = -B.im[i] * s;   // φ/H = φ·conj(H)/|H|²
    re[i] = hr * Y.re[i] - hi * Y.im[i]; im[i] = hr * Y.im[i] + hi * Y.re[i];
  }
  return inv1({ re, im });
}

/** Tikhonov in closed form: minimise ‖Ax − y‖² + λ‖Dx‖², x̂ = F⁻¹(conj(H)·Y / (|H|² + λ|D̂|²)). */
export function tikhonov1d(y: ArrayLike<number>, B: Blur1D, lambda: number, D2: ArrayLike<number>): Float64Array {
  const Y = fwd1(y), n = B.n, re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const den = B.H2[i] + lambda * D2[i];
    if (den < 1e-300) continue;
    const hr = B.re[i] / den, hi = -B.im[i] / den;
    re[i] = hr * Y.re[i] - hi * Y.im[i]; im[i] = hr * Y.im[i] + hi * Y.re[i];
  }
  return inv1({ re, im });
}

/**
 * Landweber iteration x_{k+1} = x_k + ω Aᵀ(y − A x_k) from x_0 = 0, optionally projected
 * onto x ≥ 0 after every step. `onIter(k, x)` is called after each step (return false to stop).
 */
export function landweber1d(y: ArrayLike<number>, B: Blur1D, iterations: number, opts: { omega?: number; nonneg?: boolean; x0?: Float64Array; onIter?: (k: number, x: Float64Array) => boolean | void } = {}): Float64Array {
  const omega = opts.omega ?? 1, n = B.n;
  let x = opts.x0 ? opts.x0.slice() : new Float64Array(n);
  for (let k = 1; k <= iterations; k++) {
    const r = blur1d(x, B);
    for (let i = 0; i < n; i++) r[i] = y[i] - r[i];
    const g = blurT1d(r, B);
    for (let i = 0; i < n; i++) { x[i] += omega * g[i]; if (opts.nonneg && x[i] < 0) x[i] = 0; }
    if (opts.onIter && opts.onIter(k, x) === false) break;
  }
  return x;
}

/**
 * Quadratic penalty with non-negativity: minimise ‖Ax − y‖² + λ‖Dx‖² subject to x ≥ 0, by
 * projected gradient with Nesterov acceleration (FISTA). Starts from x0 (default: the
 * unconstrained solution clipped at 0).
 */
export function tikhonovNonneg1d(y: ArrayLike<number>, B: Blur1D, lambda: number, D2: ArrayLike<number>, iterations = 300, x0?: Float64Array): Float64Array {
  const n = B.n;
  let L = 0;
  for (let i = 0; i < n; i++) L = Math.max(L, B.H2[i] + lambda * D2[i]);
  const step = 1 / L;
  let x = x0 ? x0.slice() : tikhonov1d(y, B, lambda, D2).map(v => Math.max(0, v));
  let z = x.slice(), t = 1;
  const Aty = blurT1d(y, B);
  for (let it = 0; it < iterations; it++) {
    // gradient/2 of the objective at z: (AᵀA + λDᵀD) z − Aᵀy, computed in Fourier space
    const Z = fwd1(z), re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) { const m = B.H2[i] + lambda * D2[i]; re[i] = m * Z.re[i]; im[i] = m * Z.im[i]; }
    const Mz = inv1({ re, im });
    const xn = new Float64Array(n);
    for (let i = 0; i < n; i++) xn[i] = Math.max(0, z[i] - step * (Mz[i] - Aty[i]));
    const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    for (let i = 0; i < n; i++) z[i] = xn[i] + ((t - 1) / tn) * (xn[i] - x[i]);
    x = xn; t = tn;
  }
  return x;
}

/**
 * Total-variation deblurring: minimise ‖Ax − y‖² + λ Σ √((x[i+1] − x[i])² + β²), by
 * iteratively reweighted least squares: each outer step solves
 * (AᵀA + (λ/2) Dᵀ W D) x = Aᵀy with W = 1/√((Dx)² + β²), by conjugate gradients.
 */
export function tv1d(y: ArrayLike<number>, B: Blur1D, lambda: number, opts: { beta?: number; outer?: number; cg?: number; x0?: Float64Array } = {}): Float64Array {
  const n = B.n, beta = opts.beta ?? 1e-3, outer = opts.outer ?? 20, cgMax = opts.cg ?? 50;
  let x = opts.x0 ? opts.x0.slice() : Float64Array.from(y);
  const b = blurT1d(y, B);
  const wts = new Float64Array(n);
  const M = (v: Float64Array): Float64Array => {
    const V = fwd1(v), re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) { re[i] = B.H2[i] * V.re[i]; im[i] = B.H2[i] * V.im[i]; }
    const out = inv1({ re, im });
    // + (λ/2) Dᵀ W D v with D v = v[i+1] − v[i]; Dᵀ u = u[i−1] − u[i]
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) u[i] = wts[i] * (v[(i + 1) % n] - v[i]);
    for (let i = 0; i < n; i++) out[i] += (lambda / 2) * (u[(i - 1 + n) % n] - u[i]);
    return out;
  };
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < n; i++) { const d = x[(i + 1) % n] - x[i]; wts[i] = 1 / Math.sqrt(d * d + beta * beta); }
    // CG from the current x
    const Mx = M(x), r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = b[i] - Mx[i];
    const p = r.slice();
    let rr = 0;
    for (let i = 0; i < n; i++) rr += r[i] * r[i];
    const rr0 = rr;
    for (let it = 0; it < cgMax && rr > 1e-20 * rr0 && rr > 1e-30; it++) {
      const Mp = M(p);
      let pMp = 0;
      for (let i = 0; i < n; i++) pMp += p[i] * Mp[i];
      if (!(pMp > 0)) break;
      const a = rr / pMp;
      let rrn = 0;
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Mp[i]; rrn += r[i] * r[i]; }
      const bb = rrn / rr;
      for (let i = 0; i < n; i++) p[i] = r[i] + bb * p[i];
      rr = rrn;
    }
  }
  return x;
}

/** Total variation Σ|x[i+1] − x[i]| (circular). */
export function totalVariation1d(x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.abs(x[(i + 1) % x.length] - x[i]);
  return s;
}

export const norm2 = (x: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); };
export const rms = (x: ArrayLike<number>): number => norm2(x) / Math.sqrt(x.length);

/** Residual y − A x. */
export function residual1d(x: ArrayLike<number>, y: ArrayLike<number>, B: Blur1D): Float64Array {
  const r = blur1d(x, B);
  for (let i = 0; i < r.length; i++) r[i] = y[i] - r[i];
  return r;
}

/**
 * Index of the L-curve corner: the point of maximal curvature of the curve
 * (log ρ_i, log η_i), i = 0..m−1, ordered by increasing λ (ρ = residual norm,
 * η = penalty norm). Curvature from finite differences in the parameter i.
 */
export function lcurveCorner(rho: ArrayLike<number>, eta: ArrayLike<number>): number {
  const m = rho.length;
  if (m < 3) return 0;
  const X = Array.from(rho, v => Math.log(v)), Y = Array.from(eta, v => Math.log(v));
  let best = 1, bestK = -Infinity;
  for (let i = 1; i < m - 1; i++) {
    const x1 = (X[i + 1] - X[i - 1]) / 2, y1 = (Y[i + 1] - Y[i - 1]) / 2;
    const x2 = X[i + 1] - 2 * X[i] + X[i - 1], y2 = Y[i + 1] - 2 * Y[i] + Y[i - 1];
    const den = Math.pow(x1 * x1 + y1 * y1, 1.5);
    if (!(den > 0)) continue;
    // sign: with λ increasing, ρ grows and η falls; the corner bends towards the origin
    const k = (x1 * y2 - y1 * x2) / den;
    if (k > bestK) { bestK = k; best = i; }
  }
  return best;
}

/**
 * Discrepancy principle: the λ in [lo, hi] at which f(λ) (the RMS residual, increasing in
 * λ) equals `target` (the noise σ_n), by bisection on log λ. Returns lo or hi if the target
 * is out of range.
 */
export function discrepancyLambda(f: (lambda: number) => number, target: number, lo: number, hi: number, iters = 60): number {
  if (f(lo) >= target) return lo;
  if (f(hi) <= target) return hi;
  let a = Math.log(lo), b = Math.log(hi);
  for (let i = 0; i < iters; i++) {
    const m = (a + b) / 2;
    if (f(Math.exp(m)) < target) a = m; else b = m;
  }
  return Math.exp((a + b) / 2);
}

/** Interpolate where a sampled increasing curve ys(xs) crosses `target` (log-linear in x). */
export function crossing(xs: ArrayLike<number>, ys: ArrayLike<number>, target: number): number {
  for (let i = 1; i < xs.length; i++) {
    if ((ys[i - 1] - target) * (ys[i] - target) <= 0 && ys[i] !== ys[i - 1]) {
      const t = (target - ys[i - 1]) / (ys[i] - ys[i - 1]);
      return Math.exp(Math.log(xs[i - 1]) + t * (Math.log(xs[i]) - Math.log(xs[i - 1])));
    }
  }
  return NaN;
}

// ---------------------------------------------------------------------------------------
// 3. Richardson–Lucy

/** A linear blur with its adjoint (transpose). For a PSF summing to 1, adjoint(1) = 1. */
export interface Operator {
  forward(x: Float64Array): Float64Array;
  adjoint(r: Float64Array): Float64Array;
}

/** Circular convolution with a small kernel by direct summation (1D; kernel centred, odd length). */
export function directOperator1d(kernel: ArrayLike<number>, n: number): Operator {
  const m = kernel.length, c = Math.floor(m / 2);
  return {
    forward(x) {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += kernel[j] * x[((i - (j - c)) % n + n) % n]; out[i] = s; }
      return out;
    },
    adjoint(r) {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += kernel[j] * r[((i + (j - c)) % n + n) % n]; out[i] = s; }
      return out;
    },
  };
}

// Fast real convolution: the generic lib/fft routines transform real images as complex
// ones. Here two real rows share one complex FFT, only the half spectrum (u = 0..w/2) goes
// through the column (and plane) transforms, and twiddles come from tables: about 2–3×
// faster, which matters for Richardson–Lucy (two convolutions per iteration).

class Plan1D {
  readonly n: number;
  private rev: Uint32Array;
  private c: Float64Array;
  private s: Float64Array;
  constructor(n: number) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.round(Math.log2(n));
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); this.rev[i] = r; }
    this.c = new Float64Array(n / 2 || 1); this.s = new Float64Array(n / 2 || 1);
    for (let k = 0; k < n / 2; k++) { this.c[k] = Math.cos(2 * Math.PI * k / n); this.s[k] = -Math.sin(2 * Math.PI * k / n); }
  }
  /** In-place unnormalised FFT (inverse: conjugate twiddles). */
  run(re: Float64Array, im: Float64Array, inverse: boolean): void {
    const n = this.n, rev = this.rev, C = this.c, S = this.s, sg = inverse ? -1 : 1;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = C[t], wi = sg * S[t], a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}

/**
 * Circular convolution of real w × h × d volumes (powers of two; d = 1 for 2D) with a fixed
 * kernel given by its full spectrum H (lib/fft layout). `apply(x, conj)` returns x ⊛ h,
 * or the correlation (multiplication by conj H) for the adjoint.
 */
export class RealConvolver {
  private hw: number;
  private Hre: Float64Array;
  private Him: Float64Array;
  private Sre: Float64Array;   // half spectrum, (w/2+1) × h × d
  private Sim: Float64Array;
  private pw: Plan1D; private ph: Plan1D; private pd: Plan1D;
  private br: Float64Array; private bi: Float64Array;
  constructor(readonly w: number, readonly h: number, readonly d: number, H: Spectrum) {
    const hw = this.hw = w / 2 + 1, n = hw * h * d;
    this.Hre = new Float64Array(n); this.Him = new Float64Array(n);
    for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) for (let u = 0; u < hw; u++) {
      const i = (z * h + y) * hw + u, j = (z * h + y) * w + u;
      this.Hre[i] = H.re[j]; this.Him[i] = H.im[j];
    }
    this.Sre = new Float64Array(n); this.Sim = new Float64Array(n);
    this.pw = new Plan1D(w); this.ph = new Plan1D(h); this.pd = new Plan1D(d);
    const m = Math.max(w, h, d);
    this.br = new Float64Array(m); this.bi = new Float64Array(m);
  }
  apply(x: ArrayLike<number>, conj = false): Float64Array {
    const { w, h, d, hw, Sre, Sim, br, bi } = this;
    const rows = h * d, out = new Float64Array(w * h * d);
    // rows: two real rows per complex FFT, unpack to two half spectra
    for (let r = 0; r < rows; r += 2) {
      const a = r * w, b = (r + 1) * w, two = r + 1 < rows;
      for (let i = 0; i < w; i++) { br[i] = x[a + i]; bi[i] = two ? x[b + i] : 0; }
      this.pw.run(br, bi, false);
      for (let u = 0; u < hw; u++) {
        const v = (w - u) % w, zr = br[u], zi = bi[u], cr = br[v], ci = -bi[v];   // Z[u], conj Z[-u]
        Sre[r * hw + u] = (zr + cr) / 2; Sim[r * hw + u] = (zi + ci) / 2;
        if (two) { Sre[(r + 1) * hw + u] = (zi - ci) / 2; Sim[(r + 1) * hw + u] = -(zr - cr) / 2; }
      }
    }
    // columns (and planes): forward, multiply, inverse
    const col = (inv: boolean) => {
      for (let z = 0; z < d; z++) for (let u = 0; u < hw; u++) {
        for (let y = 0; y < h; y++) { const i = (z * h + y) * hw + u; br[y] = Sre[i]; bi[y] = Sim[i]; }
        this.ph.run(br.subarray(0, h), bi.subarray(0, h), inv);
        for (let y = 0; y < h; y++) { const i = (z * h + y) * hw + u; Sre[i] = br[y]; Sim[i] = bi[y]; }
      }
    };
    const pl = (inv: boolean) => {
      if (d === 1) return;
      for (let y = 0; y < h; y++) for (let u = 0; u < hw; u++) {
        for (let z = 0; z < d; z++) { const i = (z * h + y) * hw + u; br[z] = Sre[i]; bi[z] = Sim[i]; }
        this.pd.run(br.subarray(0, d), bi.subarray(0, d), inv);
        for (let z = 0; z < d; z++) { const i = (z * h + y) * hw + u; Sre[i] = br[z]; Sim[i] = bi[z]; }
      }
    };
    col(false); pl(false);
    const s = conj ? -1 : 1, Hr = this.Hre, Hi = this.Him;
    for (let i = 0; i < Sre.length; i++) {
      const hr = Hr[i], hi = s * Hi[i], a = Sre[i], b = Sim[i];
      Sre[i] = hr * a - hi * b; Sim[i] = hr * b + hi * a;
    }
    pl(true); col(true);
    // rows back: Z = A + iB over the full row (Hermitian halves), inverse FFT gives a + i b
    const scale = 1 / (w * h * d);
    for (let r = 0; r < rows; r += 2) {
      const two = r + 1 < rows;
      for (let u = 0; u < w; u++) {
        let ar: number, ai: number, bR = 0, bI = 0;
        if (u < hw) { ar = Sre[r * hw + u]; ai = Sim[r * hw + u]; if (two) { bR = Sre[(r + 1) * hw + u]; bI = Sim[(r + 1) * hw + u]; } }
        else { const v = w - u; ar = Sre[r * hw + v]; ai = -Sim[r * hw + v]; if (two) { bR = Sre[(r + 1) * hw + v]; bI = -Sim[(r + 1) * hw + v]; } }
        br[u] = ar - bI; bi[u] = ai + bR;
      }
      this.pw.run(br, bi, true);
      for (let i = 0; i < w; i++) { out[r * w + i] = br[i] * scale; if (two) out[(r + 1) * w + i] = bi[i] * scale; }
    }
    return out;
  }
}

/** Circular 2D convolution by FFT with a centred w × h PSF (w, h powers of two). */
export function fftOperator2d(psf: ArrayLike<number>, w: number, h: number): Operator & { H: Spectrum } {
  const H = spectrum2d(toOrigin(psf, w, h), w, h), C = new RealConvolver(w, h, 1, H);
  return { H, forward: x => C.apply(x, false), adjoint: r => C.apply(r, true) };
}

/** Circular 3D convolution by FFT with a centred nx × ny × nz PSF. */
export function fftOperator3d(psf: ArrayLike<number>, nx: number, ny: number, nz: number): Operator & { H: Spectrum } {
  const n = nx * ny * nz, Hre = toOrigin3d(psf, nx, ny, nz), Him = new Float64Array(n);
  fft3d(Hre, Him, nx, ny, nz, false);
  const H = { re: Hre, im: Him }, C = new RealConvolver(nx, ny, nz, H);
  return { H, forward: x => C.apply(x, false), adjoint: r => C.apply(r, true) };
}

/**
 * One Richardson–Lucy step: x ← x · Aᵀ( y / (A x + b) ) (for a PSF summing to 1).
 * Pixels where the model is ≤ 0 contribute a ratio of 0.
 */
export function rlStep(x: Float64Array, y: ArrayLike<number>, A: Operator, background = 0): Float64Array {
  const Ax = A.forward(x), n = x.length;
  for (let i = 0; i < n; i++) { const m = Ax[i] + background; Ax[i] = m > 1e-12 ? y[i] / m : 0; }
  const c = A.adjoint(Ax), out = new Float64Array(n);
  for (let i = 0; i < n; i++) { const v = x[i] * c[i]; out[i] = v > 0 ? v : 0; }
  return out;
}

/** Start value: a flat image with the data's mean (minus the background), at least a small positive value. */
export function flatStart(y: ArrayLike<number>, background = 0): Float64Array {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += y[i];
  return new Float64Array(y.length).fill(Math.max(1e-6, s / y.length - background));
}

/**
 * Richardson–Lucy: `iterations` steps from x0 (default flat). `onIter(k, x)` after each
 * step; return false to stop early.
 */
export function richardsonLucy(y: ArrayLike<number>, A: Operator, iterations: number, opts: { background?: number; x0?: Float64Array; onIter?: (k: number, x: Float64Array) => boolean | void } = {}): Float64Array {
  const b = opts.background ?? 0;
  let x = opts.x0 ? opts.x0.slice() : flatStart(y, b);
  for (let k = 1; k <= iterations; k++) {
    x = rlStep(x, y, A, b);
    if (opts.onIter && opts.onIter(k, x) === false) break;
  }
  return x;
}

/** Poisson log-likelihood Σ y log m − m (dropping log y!), with m = A x + b. */
export function poissonLogLik(y: ArrayLike<number>, model: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) { const m = model[i]; if (m > 0) s += (y[i] > 0 ? y[i] * Math.log(m) : 0) - m; else if (y[i] > 0) return -Infinity; }
  return s;
}

/**
 * One alternating step of blind Richardson–Lucy (Holmes 1992; Fish et al. 1995) on a w × h
 * image with circular convolution: an RL update of the object with the current PSF, then
 * an RL update of the PSF with the new object, then the PSF constraints: non-negative,
 * zero outside a disk of `supportRadius` pixels around the centre (if given), sum 1.
 * The PSF is centred.
 */
export function blindStep(x: Float64Array, psf: Float64Array, y: ArrayLike<number>, w: number, h: number, opts: { background?: number; supportRadius?: number } = {}): { x: Float64Array; psf: Float64Array } {
  const b = opts.background ?? 0;
  const A = fftOperator2d(psf, w, h);
  const xn = rlStep(x, y, A, b);
  // PSF update: the roles of object and PSF swap. Model m = x ⊛ h; ratio r = y / m;
  // h ← h · (x ⋆ r) / Σx, with ⋆ the correlation, evaluated at the centred PSF positions.
  let sx = 0;
  for (let i = 0; i < xn.length; i++) sx += xn[i];
  const X = fftOperator2d(centreOf(xn, w, h), w, h);   // convolution with the object
  const m = X.forward(toCentredFromOrigin(psf, w, h));
  for (let i = 0; i < m.length; i++) { const v = m[i] + b; m[i] = v > 1e-12 ? y[i] / v : 0; }
  const corr = X.adjoint(m);                            // x ⋆ r, origin-based
  const corrC = centreArr(corr, w, h);
  const out = new Float64Array(w * h), cx = w >> 1, cy = h >> 1, R = opts.supportRadius;
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) {
    const i = yy * w + xx;
    if (R !== undefined && (xx - cx) ** 2 + (yy - cy) ** 2 > R * R) continue;
    out[i] = Math.max(0, psf[i] * corrC[i] / sx);
  }
  let s = 0;
  for (let i = 0; i < out.length; i++) s += out[i];
  if (s > 0) for (let i = 0; i < out.length; i++) out[i] /= s;
  return { x: xn, psf: out };
}

// helpers for the blind step: treat an image as a "PSF" by viewing its index 0 as the centre
function centreOf(img: Float64Array, w: number, h: number): Float64Array {   // origin-based → centred view
  return centreArr(img, w, h);
}
function centreArr(a: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h), sx = w >> 1, sy = h >> 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[((y + sy) % h) * w + (x + sx) % w] = a[y * w + x];
  return out;
}
function toCentredFromOrigin(psf: Float64Array, w: number, h: number): Float64Array {   // centred PSF → origin-based image
  return toOrigin(psf, w, h);
}
