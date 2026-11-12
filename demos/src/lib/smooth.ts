// Smoothing priors for continuous signals and images (owner: the Ising / MRF day, unit gmrf/).
// No DOM: also used by the tests. Users: mrf1d demo; radon (algebraic reconstruction) and the
// wavelets comparison may use the 2D TV routine.
//
// Denoising energy on a 1D signal (free ends), with the noise variance absorbed into lambda:
//     E(x) = 1/2 sum_p (x_p - y_p)^2  +  lambda * sum_p rho(x_{p+1} - x_p)
//   quadratic (Gaussian MRF):  rho(d) = d^2 / 2   ->  (I + lambda L) x = y, L = path Laplacian
//   total variation (TV):      rho(d) = |d|        ->  Condat's direct algorithm (exact)
//   Huber:                     rho(d) = d^2/2 for |d| <= delta, delta (|d| - delta/2) beyond
//                                                  ->  iteratively reweighted least squares
// (with Gaussian noise of sd sigma_n, the posterior energy (1/2 sigma_n^2) sum (x - y)^2 +
// (lambda'/2) sum d^2 is this one with lambda = lambda' sigma_n^2.)
//
//   gmrf1d(y, 1)          // (0, 3, 0) -> (0.75, 1.5, 0.75)
//   tv1d(y, 1)            // (0, 0, 3, 3) -> (0.5, 0.5, 2.5, 2.5)
//   tv2d(img, w, h, 10)   // isotropic 2D TV (Chambolle's projection algorithm)
//
// Chain Viterbi and forward-backward live in lib/dp.ts (dynamic programming day), not here.
import { Rng } from "./random";

export type Prior = "none" | "quadratic" | "huber" | "tv";

/** Solve a tridiagonal system (Thomas algorithm): sub-diagonal a (a[0] unused), diagonal b,
 *  super-diagonal c (c[n-1] unused), right-hand side d. O(n); assumes no pivoting is needed
 *  (true for diagonally dominant matrices such as I + lambda L). */
export function thomas(a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>, d: ArrayLike<number>): Float64Array {
  const n = b.length, cp = new Float64Array(n), dp = new Float64Array(n), x = new Float64Array(n);
  if (n === 0) return x;
  cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0];
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1];
    cp[i] = i < n - 1 ? c[i] / m : 0;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
  }
  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
  return x;
}

/** Solve (I + lambda L_w) x = y on a path, where L_w is the Laplacian with edge weights wts
 *  (wts[i] joins i and i+1; omitted = all 1). */
export function weightedPathSolve(y: ArrayLike<number>, lambda: number, wts?: ArrayLike<number>): Float64Array {
  const n = y.length, a = new Float64Array(n), b = new Float64Array(n).fill(1), c = new Float64Array(n);
  for (let i = 0; i + 1 < n; i++) {
    const w = lambda * (wts ? wts[i] : 1);
    b[i] += w; b[i + 1] += w; c[i] = -w; a[i + 1] = -w;
  }
  return thomas(a, b, c, y);
}

/** Gaussian MRF / quadratic smoothing: x = (I + lambda L)^-1 y (free ends). */
export const gmrf1d = (y: ArrayLike<number>, lambda: number): Float64Array => weightedPathSolve(y, lambda);

/** The matrix I + lambda L of a path of n nodes, dense, row-major (for display). */
export function gmrfMatrix(n: number, lambda: number): Float64Array {
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) M[i * n + i] = 1;
  for (let i = 0; i + 1 < n; i++) {
    M[i * n + i] += lambda; M[(i + 1) * n + i + 1] += lambda;
    M[i * n + i + 1] -= lambda; M[(i + 1) * n + i] -= lambda;
  }
  return M;
}

/** Frequency response of quadratic smoothing on a ring: 1 / (1 + lambda (2 - 2 cos omega)), omega in radians per sample. */
export const gmrfResponse = (omega: number, lambda: number): number => 1 / (1 + lambda * (2 - 2 * Math.cos(omega)));

/** Exact 1D total-variation denoising, argmin 1/2 sum (x - y)^2 + lambda sum |x_{p+1} - x_p|
 *  (Condat 2013, "A direct algorithm for 1-D total variation denoising"; O(n) in practice). */
export function tv1d(y: ArrayLike<number>, lambda: number): Float64Array {
  const n = y.length, x = new Float64Array(n);
  if (n === 0) return x;
  if (!(lambda > 0)) { for (let i = 0; i < n; i++) x[i] = y[i]; return x; }
  let k = 0, k0 = 0, kplus = 0, kminus = 0;
  let umin = lambda, umax = -lambda;
  let vmin = y[0] - lambda, vmax = y[0] + lambda;
  const twol = 2 * lambda, minl = -lambda;
  for (;;) {
    while (k === n - 1) {
      if (umin < 0) {
        do x[k0++] = vmin; while (k0 <= kminus);
        k = kminus = k0; vmin = y[k]; umin = lambda; umax = vmin + umin - vmax;
      } else if (umax > 0) {
        do x[k0++] = vmax; while (k0 <= kplus);
        k = kplus = k0; vmax = y[k]; umax = minl; umin = vmax + umax - vmin;
      } else {
        vmin += umin / (k - k0 + 1);
        do x[k0++] = vmin; while (k0 <= k);
        return x;
      }
    }
    if ((umin += y[k + 1] - vmin) < minl) {
      do x[k0++] = vmin; while (k0 <= kminus);
      k = kplus = kminus = k0; vmin = y[k]; vmax = vmin + twol; umin = lambda; umax = minl;
    } else if ((umax += y[k + 1] - vmax) > lambda) {
      do x[k0++] = vmax; while (k0 <= kplus);
      k = kplus = kminus = k0; vmax = y[k]; vmin = vmax - twol; umin = lambda; umax = minl;
    } else {
      k++;
      if (umin >= lambda) { kminus = k; vmin += (umin - lambda) / (kminus - k0 + 1); umin = lambda; }
      if (umax <= minl) { kplus = k; vmax += (umax + lambda) / (kplus - k0 + 1); umax = minl; }
    }
  }
}

/** Huber penalty rho(d): d^2/2 for |d| <= delta, delta (|d| - delta/2) beyond. */
export function huber(d: number, delta: number): number {
  const a = Math.abs(d);
  return a <= delta ? 0.5 * d * d : delta * (a - 0.5 * delta);
}

/** Huber smoothing by iteratively reweighted least squares: each step solves
 *  (I + lambda L_w) x = y with w = min(1, delta / |d|) from the current differences d.
 *  The energy never increases (a majorise-minimise scheme). */
export function huber1d(y: ArrayLike<number>, lambda: number, delta: number, iters = 60): Float64Array {
  const n = y.length;
  let x = gmrf1d(y, lambda);
  const w = new Float64Array(Math.max(0, n - 1));
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i + 1 < n; i++) { const d = Math.abs(x[i + 1] - x[i]); w[i] = d <= delta ? 1 : delta / d; }
    x = weightedPathSolve(y, lambda, w);
  }
  return x;
}

/** Penalty rho(d) of a prior. */
export function penalty(prior: Prior, d: number, delta = 1): number {
  switch (prior) {
    case "quadratic": return 0.5 * d * d;
    case "tv": return Math.abs(d);
    case "huber": return huber(d, delta);
    default: return 0;
  }
}

/** Energy split: data 1/2 sum (x - y)^2 and prior lambda sum rho(x_{p+1} - x_p). */
export function energy1d(x: ArrayLike<number>, y: ArrayLike<number>, prior: Prior, lambda: number, delta = 1): { data: number; prior: number; total: number } {
  let data = 0, pr = 0;
  for (let i = 0; i < x.length; i++) data += 0.5 * (x[i] - y[i]) ** 2;
  if (prior !== "none") for (let i = 0; i + 1 < x.length; i++) pr += penalty(prior, x[i + 1] - x[i], delta);
  return { data, prior: lambda * pr, total: data + lambda * pr };
}

/** Minimiser of energy1d for any prior ("none" returns y). */
export function denoise1d(y: ArrayLike<number>, prior: Prior, lambda: number, delta = 1): Float64Array {
  switch (prior) {
    case "quadratic": return gmrf1d(y, lambda);
    case "tv": return tv1d(y, lambda);
    case "huber": return huber1d(y, lambda, delta);
    default: return Float64Array.from(y);
  }
}

/** A draw from the 1D prior exp(-lambda sum rho(d)) started at x0: a random walk with
 *  Gaussian increments (quadratic, sd 1/sqrt(lambda)) or Laplace increments (TV, scale 1/lambda). */
export function samplePrior1d(r: Rng, n: number, prior: "quadratic" | "tv", lambda: number, x0 = 0): Float64Array {
  const x = new Float64Array(n);
  x[0] = x0;
  for (let i = 1; i < n; i++) {
    let d: number;
    if (prior === "quadratic") d = r.norm() / Math.sqrt(lambda);
    else { const u = r.unif() - 0.5; d = -Math.sign(u) * Math.log(1 - 2 * Math.abs(u)) / lambda; }
    x[i] = x[i - 1] + d;
  }
  return x;
}

// ---------------------------------------------------------------------------------------
// 2D total variation

/** Isotropic TV of an image: sum over pixels of |forward-difference gradient| (Neumann boundary). */
export function tv2dValue(x: ArrayLike<number>, w: number, h: number): number {
  let s = 0;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const p = j * w + i;
    const gx = i < w - 1 ? x[p + 1] - x[p] : 0, gy = j < h - 1 ? x[p + w] - x[p] : 0;
    s += Math.hypot(gx, gy);
  }
  return s;
}

/** Isotropic 2D TV denoising, argmin 1/2 ||x - y||^2 + lambda TV(x), by Chambolle's (2004)
 *  projection algorithm (step 1/8, Neumann boundary). `iters` ~ 100-300 for display quality. */
export function tv2d(y: ArrayLike<number>, w: number, h: number, lambda: number, iters = 200): Float64Array {
  const n = w * h, px = new Float64Array(n), py = new Float64Array(n), div = new Float64Array(n), x = new Float64Array(n);
  const tau = 0.125;
  if (!(lambda > 0)) { for (let p = 0; p < n; p++) x[p] = y[p]; return x; }
  const computeDiv = () => {
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const p = j * w + i;
      const dx = (i < w - 1 ? px[p] : 0) - (i > 0 ? px[p - 1] : 0);
      const dy = (j < h - 1 ? py[p] : 0) - (j > 0 ? py[p - w] : 0);
      div[p] = dx + dy;
    }
  };
  for (let it = 0; it < iters; it++) {
    computeDiv();
    for (let p = 0; p < n; p++) x[p] = div[p] - y[p] / lambda;   // reuse x as the work array
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const p = j * w + i;
      const gx = i < w - 1 ? x[p + 1] - x[p] : 0, gy = j < h - 1 ? x[p + w] - x[p] : 0;
      const den = 1 + tau * Math.hypot(gx, gy);
      px[p] = (px[p] + tau * gx) / den; py[p] = (py[p] + tau * gy) / den;
    }
  }
  computeDiv();
  for (let p = 0; p < n; p++) x[p] = y[p] - lambda * div[p];
  return x;
}
