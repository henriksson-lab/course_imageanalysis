// Sparse representations (owner: the sparse representations day: sparsity/, ista/,
// compressed-sensing/, cs-mri/, dictionaries/). No DOM; runs under node.
//
// Conventions (as on the slides):
//   lasso           ½‖Ax − y‖² + λ‖x‖₁            (soft threshold at λ when A = I)
//   ridge (ℓ2)      ½‖Ax − y‖² + (λ/2)‖x‖²       (c = w/(1 + λ) when A = I)
//   ℓ0              ½‖Ax − y‖² + (λ²/2)‖x‖₀       (hard threshold at λ when A = I, so the same
//                                                  λ gives the same threshold as ℓ1)
//   ℓp (p = ½)      ½‖Ax − y‖² + λ Σ|x|^p
// Linear operators are { m, n, forward, adjoint } (lib/deconv.ts's Operator plus sizes).
// Solvers: ISTA, FISTA (Beck & Teboulle 2009) with any proximal step, OMP (Pati et al. 1993),
// lasso by coordinate descent (paths), exact 2D lasso by enumerating supports.
// Compressed sensing: sparse signals, sampling masks (1D real-Fourier rows, 2D k-space lines),
// mask point spread functions, recovery trials for the phase diagram, and the Donoho–Tanner
// ℓ1 weak threshold as a table.
import { soft, hard, Wavelet, dwt, idwt, realFourier, realFourierInverse, denoise2 } from "./wavelet";
import { dctFast, idctFast, dct1, idct1 } from "./dct";
import { fft2d, fft1d, freq } from "./fft";
import { gaussian1d, directOperator1d, fftOperator2d } from "./deconv";
import { solve } from "./linalg";
import type { Rng } from "./random";
import { rng, gauss, sampleIndices } from "./random";
import { SHEPP_LOGAN, PhantomEllipse } from "./synth";

export { soft, hard };
export type Vec = Float64Array;

// ---------------------------------------------------------------------------------------
// 1. Penalties and proximal steps

/** Soft threshold of every entry at t (or at t·w_i with per-entry weights). */
export function softVec(v: ArrayLike<number>, t: number, w?: ArrayLike<number>): Vec {
  const o = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = soft(v[i], w ? t * w[i] : t);
  return o;
}
/** Hard threshold of every entry at t. */
export function hardVec(v: ArrayLike<number>, t: number, w?: ArrayLike<number>): Vec {
  const o = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = hard(v[i], w ? t * w[i] : t);
  return o;
}
/** Projection onto x ≥ 0 (the proximal step of the non-negativity constraint). */
export const proxNonneg = (v: ArrayLike<number>): Vec => Float64Array.from(v, x => (x > 0 ? x : 0));
/** Proximal step of (t/2)‖x‖²: v/(1 + t). */
export const proxL2 = (v: ArrayLike<number>, t: number): Vec => Float64Array.from(v, x => x / (1 + t));

export type PenaltyKind = 0 | 0.5 | 1 | 2;

/** The scalar problem argmin_c ½(c − w)² + pen(c) for the penalties above (transfer curves). */
export function lassoScalar(w: number, lambda: number, p: PenaltyKind): number {
  if (p === 2) return w / (1 + lambda);
  if (p === 1) return soft(w, lambda);
  if (p === 0) return hard(w, lambda);
  // p = 1/2: compare c = 0 with the best c of the sign of w (1D search, unimodal on [c*, |w|])
  const a = Math.abs(w);
  if (a === 0 || lambda <= 0) return w;
  const f = (c: number) => 0.5 * (c - a) ** 2 + lambda * Math.sqrt(c);
  let lo = 0, hi = a;
  // the stationary point with f'' > 0 lies above c0 = (λ/4)^(2/3) (where f'' = 0)
  lo = Math.min(a, Math.pow(lambda / 4, 2 / 3));
  for (let it = 0; it < 80; it++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (f(m1) < f(m2)) hi = m2; else lo = m1;
  }
  const c = (lo + hi) / 2;
  return f(c) < f(0) ? Math.sign(w) * c : 0;
}

/** Value of the penalty (as in the conventions above) for a vector. */
export function penaltyValue(x: ArrayLike<number>, lambda: number, p: PenaltyKind): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (p === 1) s += lambda * a;
    else if (p === 2) s += 0.5 * lambda * a * a;
    else if (p === 0) s += a > 0 ? 0.5 * lambda * lambda : 0;
    else s += lambda * Math.sqrt(a);
  }
  return s;
}

// ---------------------------------------------------------------------------------------
// 2. Linear operators

export interface LinOp {
  m: number;   // rows (measurements)
  n: number;   // columns (unknowns)
  forward(x: ArrayLike<number>): Vec;
  adjoint(r: ArrayLike<number>): Vec;
}

/** A dense row-major m × n matrix as an operator. */
export function matrixOp(A: ArrayLike<number>, m: number, n: number): LinOp {
  return {
    m, n,
    forward(x) { const o = new Float64Array(m); for (let i = 0; i < m; i++) { let s = 0; const r = i * n; for (let j = 0; j < n; j++) s += A[r + j] * x[j]; o[i] = s; } return o; },
    adjoint(y) { const o = new Float64Array(n); for (let i = 0; i < m; i++) { const yi = y[i]; if (yi === 0) continue; const r = i * n; for (let j = 0; j < n; j++) o[j] += A[r + j] * yi; } return o; },
  };
}

export const identityOp = (n: number): LinOp => ({ m: n, n, forward: x => Float64Array.from(x), adjoint: r => Float64Array.from(r) });

/** Keep the pixels where mask = 1, zero elsewhere (same size; self-adjoint). */
export function maskOp(mask: ArrayLike<number>): LinOp {
  const n = mask.length;
  const f = (x: ArrayLike<number>) => { const o = new Float64Array(n); for (let i = 0; i < n; i++) if (mask[i]) o[i] = x[i]; return o; };
  return { m: n, n, forward: f, adjoint: f };
}

/** Circular Gaussian blur of a length-n signal (σ in samples; kernel cut at 4σ). */
export function blurOp1d(n: number, sigma: number): LinOp {
  if (sigma <= 0) return identityOp(n);
  const r = Math.ceil(4 * sigma), k = gaussian1d(2 * r + 1, sigma), op = directOperator1d(k, n);
  return { m: n, n, forward: x => op.forward(Float64Array.from(x)), adjoint: y => op.adjoint(Float64Array.from(y)) };
}

/** Centred Gaussian PSF of size w × h (sum 1). */
export function gaussianPsf2d(w: number, h: number, sigma: number): Vec {
  const p = new Float64Array(w * h), cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  let s = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma));
    p[y * w + x] = v; s += v;
  }
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

/** Circular 2D Gaussian blur by FFT (w, h powers of two). */
export function blurOp2d(w: number, h: number, sigma: number): LinOp {
  if (sigma <= 0) return identityOp(w * h);
  const op = fftOperator2d(gaussianPsf2d(w, h, sigma), w, h);
  return { m: w * h, n: w * h, forward: x => op.forward(Float64Array.from(x)), adjoint: y => op.adjoint(Float64Array.from(y)) };
}

/** A ∘ B (first B, then A). */
export function composeOp(A: LinOp, B: LinOp): LinOp {
  return { m: A.m, n: B.n, forward: x => A.forward(B.forward(x)), adjoint: y => B.adjoint(A.adjoint(y)) };
}

/** [A B]: x = (u, v) ↦ A u + B v (two dictionaries side by side). */
export function unionOp(A: LinOp, B: LinOp): LinOp {
  return {
    m: A.m, n: A.n + B.n,
    forward(x) {
      const u = new Float64Array(A.n), v = new Float64Array(B.n);
      for (let i = 0; i < A.n; i++) u[i] = x[i];
      for (let i = 0; i < B.n; i++) v[i] = x[A.n + i];
      const a = A.forward(u), b = B.forward(v);
      for (let i = 0; i < a.length; i++) a[i] += b[i];
      return a;
    },
    adjoint(y) { const o = new Float64Array(A.n + B.n); o.set(A.adjoint(y), 0); o.set(B.adjoint(y), A.n); return o; },
  };
}

/** Keep the rows `rows` of an orthonormal analysis transform T (x ↦ T x), e.g. the real DFT.
 *  forward = (T x)[rows], adjoint = Tᵀ (zero-filled). */
export function subsampledOp(analysis: (x: ArrayLike<number>) => Vec, synthesis: (c: ArrayLike<number>) => Vec, n: number, rows: ArrayLike<number>): LinOp {
  const m = rows.length;
  return {
    m, n,
    forward(x) { const c = analysis(x), o = new Float64Array(m); for (let i = 0; i < m; i++) o[i] = c[rows[i]]; return o; },
    adjoint(y) { const c = new Float64Array(n); for (let i = 0; i < m; i++) c[rows[i]] = y[i]; return synthesis(c); },
  };
}

/** Rows of the real orthonormal Fourier basis (lib/wavelet.ts realFourier). */
export const realFourierOp1d = (n: number, rows: ArrayLike<number>): LinOp => subsampledOp(realFourier, realFourierInverse, n, rows);
/** Samples x[rows] (random pixels). */
export const pixelOp1d = (n: number, rows: ArrayLike<number>): LinOp => subsampledOp(x => Float64Array.from(x), c => Float64Array.from(c), n, rows);

/** Frequency of row j of realFourier (0, 1, 1, 2, 2, …, n/2). */
export const realFourierFreq = (j: number, n: number): number => (j === n - 1 ? n / 2 : Math.ceil(j / 2));

/** Gaussian sensing matrix with entries N(0, 1/M) (columns of expected unit norm). */
export function gaussianMatrix(M: number, N: number, r: Rng): Vec {
  const A = new Float64Array(M * N), s = 1 / Math.sqrt(M);
  for (let i = 0; i < A.length; i++) A[i] = s * r.norm();
  return A;
}
export const gaussianMatrixOp = (M: number, N: number, r: Rng): LinOp => matrixOp(gaussianMatrix(M, N, r), M, N);

export type Basis1D = "spikes" | "dct" | "haar" | "db4" | "fourier";

/** Synthesis operator of an orthonormal 1D basis: coefficients c ↦ signal (forward), signal ↦ c (adjoint). */
export function basisOp1d(n: number, basis: Basis1D, wavelets?: { haar: Wavelet; db4: Wavelet }, levels = 5): LinOp {
  if (basis === "spikes") return identityOp(n);
  if (basis === "dct") {
    const pow2 = (n & (n - 1)) === 0;
    return { m: n, n, forward: c => (pow2 ? idctFast(c) : idct1(c)), adjoint: x => (pow2 ? dctFast(x) : dct1(x)) };
  }
  if (basis === "fourier") return { m: n, n, forward: realFourierInverse, adjoint: realFourier };
  const w = wavelets![basis];
  return { m: n, n, forward: c => idwt(c, w, levels), adjoint: x => dwt(x, w, levels) };
}

/** 2D partial Fourier sampling (unitary DFT, mask in unshifted k-space order, w·h = n). The
 *  measurement vector holds [Re | Im] of the full masked spectrum (2n numbers; zeros where
 *  not sampled). The image is real, so the adjoint is Re(F⁻¹ P y); with a mask symmetric
 *  under k ↦ −k this is the zero-filled reconstruction and ‖A‖ = 1. */
export function partialFourierOp2d(mask: ArrayLike<number>, w: number): LinOp {
  const n = mask.length, s = 1 / Math.sqrt(n);
  return {
    m: 2 * n, n,
    forward(x) {
      const re = Float64Array.from(x), im = new Float64Array(n);
      fft2d(re, im, w, false);
      const o = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) if (mask[i]) { o[i] = re[i] * s; o[n + i] = im[i] * s; }
      return o;
    },
    adjoint(y) {
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) if (mask[i]) { re[i] = y[i]; im[i] = y[n + i]; }
      fft2d(re, im, w, true);   // divides by n
      const k = n * s;
      for (let i = 0; i < n; i++) re[i] *= k;
      return re;
    },
  };
}

/** Largest eigenvalue of AᵀA (= ‖A‖², the Lipschitz constant L of the data gradient) by power iteration. */
export function powerIteration(A: LinOp, iters = 60, seed = 7): number {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; };
  let v: Vec = Float64Array.from({ length: A.n }, rnd), lam = 0;
  for (let k = 0; k < iters; k++) {
    let nv = 0; for (let i = 0; i < v.length; i++) nv += v[i] * v[i];
    nv = Math.sqrt(nv) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= nv;
    const w = A.adjoint(A.forward(v));
    lam = 0; for (let i = 0; i < v.length; i++) lam += v[i] * w[i];
    v = w;
  }
  return lam;
}

/** The dense m × n matrix of an operator (columns = forward of unit vectors). */
export function denseMatrix(A: LinOp): Vec {
  const M = new Float64Array(A.m * A.n), e = new Float64Array(A.n);
  for (let j = 0; j < A.n; j++) {
    e[j] = 1; const c = A.forward(e); e[j] = 0;
    for (let i = 0; i < A.m; i++) M[i * A.n + j] = c[i];
  }
  return M;
}

// ---------------------------------------------------------------------------------------
// 3. Solvers

const dotv = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm2 = (a: ArrayLike<number>): number => Math.sqrt(dotv(a, a));

/** ½‖Ax − y‖² + λ Σ w_i |x_i| (weights default 1). */
export function lassoObjective(A: LinOp, y: ArrayLike<number>, x: ArrayLike<number>, lambda: number, w?: ArrayLike<number>): number {
  const r = A.forward(x);
  let d = 0; for (let i = 0; i < r.length; i++) d += (r[i] - y[i]) ** 2;
  let p = 0; for (let i = 0; i < x.length; i++) p += (w ? w[i] : 1) * Math.abs(x[i]);
  return 0.5 * d + lambda * p;
}

/** λ_max = max_i |(Aᵀy)_i| / w_i: for λ ≥ λ_max the lasso solution is 0. */
export function lambdaMax(A: LinOp, y: ArrayLike<number>, w?: ArrayLike<number>): number {
  const g = A.adjoint(y);
  let m = 0;
  for (let i = 0; i < g.length; i++) { const wi = w ? w[i] : 1; if (wi > 0) m = Math.max(m, Math.abs(g[i]) / wi); }
  return m;
}

export interface ProxSolveOptions {
  lambda: number;
  iters: number;
  /** Step τ (default 1/L from power iteration). */
  step?: number;
  x0?: ArrayLike<number>;
  /** Per-entry ℓ1 weights (0 = not penalised). */
  weights?: ArrayLike<number>;
  /** Proximal step of τ·g at v; default soft threshold at τλ (with weights). */
  prox?: (v: Vec, tau: number) => Vec;
  /** Penalty g(x) for the objective history (default λ Σ w|x|). */
  penalty?: (x: Vec) => number;
  /** Called after every iteration; return true to stop. */
  onIter?: (k: number, x: Vec, objective: number) => boolean | void;
  /** Record the objective (costs one extra forward per iteration; default true). */
  history?: boolean;
}

export interface SolveResult { x: Vec; history: number[]; step: number; iters: number }

function defaults(A: LinOp, o: ProxSolveOptions) {
  const tau = o.step ?? 1 / Math.max(powerIteration(A), 1e-12);
  const prox = o.prox ?? ((v: Vec, t: number) => softVec(v, t * o.lambda, o.weights));
  const pen = o.penalty ?? ((x: Vec) => { let s = 0; for (let i = 0; i < x.length; i++) s += (o.weights ? o.weights[i] : 1) * Math.abs(x[i]); return o.lambda * s; });
  return { tau, prox, pen };
}

/** The two halves of one ISTA step from x: z = x + τAᵀ(y − Ax) (Landweber), then prox(z). */
export function istaStep(A: LinOp, y: ArrayLike<number>, x: ArrayLike<number>, tau: number, prox: (v: Vec, t: number) => Vec): { z: Vec; x: Vec } {
  const Ax = A.forward(x), r = new Float64Array(Ax.length);
  for (let i = 0; i < r.length; i++) r[i] = y[i] - Ax[i];
  const g = A.adjoint(r), z = new Float64Array(x.length);
  for (let i = 0; i < z.length; i++) z[i] = x[i] + tau * g[i];
  return { z, x: prox(z, tau) };
}

function objectiveOf(A: LinOp, y: ArrayLike<number>, x: Vec, pen: (x: Vec) => number): number {
  const r = A.forward(x);
  let d = 0; for (let i = 0; i < r.length; i++) d += (r[i] - y[i]) ** 2;
  return 0.5 * d + pen(x);
}

/** ISTA (iterative shrinkage-thresholding; Daubechies, Defrise & De Mol 2004). */
export function ista(A: LinOp, y: ArrayLike<number>, o: ProxSolveOptions): SolveResult {
  const { tau, prox, pen } = defaults(A, o);
  let x: Vec = o.x0 ? Float64Array.from(o.x0) : new Float64Array(A.n);
  const hist: number[] = [];
  let k = 0;
  for (; k < o.iters; k++) {
    x = istaStep(A, y, x, tau, prox).x;
    const f = o.history === false ? NaN : objectiveOf(A, y, x, pen);
    hist.push(f);
    if (o.onIter?.(k + 1, x, f)) { k++; break; }
  }
  return { x, history: hist, step: tau, iters: k };
}

/** FISTA (Beck & Teboulle 2009): ISTA from an extrapolated point, t_{k+1} = (1 + √(1 + 4t_k²))/2. */
export function fista(A: LinOp, y: ArrayLike<number>, o: ProxSolveOptions): SolveResult {
  const { tau, prox, pen } = defaults(A, o);
  let x: Vec = o.x0 ? Float64Array.from(o.x0) : new Float64Array(A.n);
  let z: Vec = Float64Array.from(x), t = 1;
  const hist: number[] = [];
  let k = 0;
  for (; k < o.iters; k++) {
    const xn = istaStep(A, y, z, tau, prox).x;
    const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, b = (t - 1) / tn;
    const zn = new Float64Array(xn.length);
    for (let i = 0; i < xn.length; i++) zn[i] = xn[i] + b * (xn[i] - x[i]);
    x = xn; z = zn; t = tn;
    const f = o.history === false ? NaN : objectiveOf(A, y, x, pen);
    hist.push(f);
    if (o.onIter?.(k + 1, x, f)) { k++; break; }
  }
  return { x, history: hist, step: tau, iters: k };
}

/** Least squares on the columns `support` of a dense m × n matrix (normal equations). */
export function debias(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>, support: ArrayLike<number>): Vec {
  const k = support.length, x = new Float64Array(n);
  if (!k) return x;
  const G = new Float64Array(k * k), b = new Float64Array(k);
  for (let a = 0; a < k; a++) {
    const ja = support[a];
    let s = 0; for (let i = 0; i < m; i++) s += A[i * n + ja] * y[i];
    b[a] = s;
    for (let c = a; c < k; c++) {
      const jc = support[c];
      let g = 0; for (let i = 0; i < m; i++) g += A[i * n + ja] * A[i * n + jc];
      G[a * k + c] = g; G[c * k + a] = g;
    }
  }
  let sol = solve(G, b, k);
  if (!sol) { for (let a = 0; a < k; a++) G[a * k + a] += 1e-10 * (G[a * k + a] || 1); sol = solve(G, b, k); }
  if (sol) for (let a = 0; a < k; a++) x[support[a]] = sol[a];
  return x;
}

export interface OmpResult { x: Vec; support: number[]; residualNorms: number[]; steps: Vec[] }

/** Orthogonal matching pursuit on a dense m × n matrix: pick the column most correlated with
 *  the residual (correlation / column norm), refit all chosen columns by least squares, repeat
 *  until K columns or ‖r‖ ≤ tol. `steps` holds the estimate after each pick. */
export function omp(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>, opts: { K?: number; tol?: number } = {}): OmpResult {
  const K = Math.min(opts.K ?? m, m, n), tol = opts.tol ?? 0;
  const cn = new Float64Array(n);
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += A[i * n + j] ** 2; cn[j] = Math.sqrt(s) || 1; }
  let r: Vec = Float64Array.from(y), x: Vec = new Float64Array(n);
  const support: number[] = [], res = [norm2(r)], steps: Vec[] = [];
  const used = new Uint8Array(n);
  while (support.length < K && res[res.length - 1] > tol) {
    let best = -1, bv = 0;
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      let s = 0; for (let i = 0; i < m; i++) s += A[i * n + j] * r[i];
      const v = Math.abs(s) / cn[j];
      if (v > bv + 1e-15) { bv = v; best = j; }
    }
    if (best < 0 || bv < 1e-13) break;
    used[best] = 1; support.push(best);
    x = debias(A, m, n, y, support);
    const Ax = matrixOp(A, m, n).forward(x);
    r = new Float64Array(m); for (let i = 0; i < m; i++) r[i] = y[i] - Ax[i];
    res.push(norm2(r)); steps.push(x);
  }
  return { x, support, residualNorms: res, steps };
}

/** Lasso ½‖Ax − y‖² + λ‖x‖₁ by cyclic coordinate descent on a dense matrix (small problems). */
export function lassoCD(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>, lambda: number, x0?: ArrayLike<number>, sweeps = 500, tol = 1e-12): Vec {
  const x = x0 ? Float64Array.from(x0) : new Float64Array(n);
  const cn2 = new Float64Array(n);
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += A[i * n + j] ** 2; cn2[j] = s; }
  const r = Float64Array.from(y);
  for (let j = 0; j < n; j++) if (x[j]) for (let i = 0; i < m; i++) r[i] -= A[i * n + j] * x[j];
  for (let s = 0; s < sweeps; s++) {
    let change = 0;
    for (let j = 0; j < n; j++) {
      if (!cn2[j]) continue;
      let rho = 0; for (let i = 0; i < m; i++) rho += A[i * n + j] * r[i];
      rho += cn2[j] * x[j];
      const nx = soft(rho, lambda) / cn2[j], d = nx - x[j];
      if (d !== 0) { for (let i = 0; i < m; i++) r[i] -= A[i * n + j] * d; x[j] = nx; change = Math.max(change, Math.abs(d)); }
    }
    if (change < tol) break;
  }
  return x;
}

/** Lasso solutions for each λ (warm starts, in the given order: largest first is fastest). */
export function lassoPath(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>, lambdas: ArrayLike<number>): Vec[] {
  const out: Vec[] = [];
  let x: Vec | undefined;
  for (let k = 0; k < lambdas.length; k++) { x = lassoCD(A, m, n, y, lambdas[k], x); out.push(x); }
  return out;
}

/** Ridge (Tikhonov) solution (AᵀA + λI)⁻¹Aᵀy on a dense matrix. */
export function ridge(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>, lambda: number): Vec {
  const G = new Float64Array(n * n), b = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    let s = 0; for (let i = 0; i < m; i++) s += A[i * n + a] * y[i];
    b[a] = s;
    for (let c = 0; c < n; c++) { let g = 0; for (let i = 0; i < m; i++) g += A[i * n + a] * A[i * n + c]; G[a * n + c] = g + (a === c ? lambda : 0); }
  }
  return solve(G, b, n) ?? new Float64Array(n);
}

/** Minimum-energy solution Aᵀ(AAᵀ)⁻¹y of an underdetermined system (dense, full row rank). */
export function minNormSolution(A: ArrayLike<number>, m: number, n: number, y: ArrayLike<number>): Vec {
  const G = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let c = 0; c < m; c++) { let s = 0; for (let j = 0; j < n; j++) s += A[a * n + j] * A[c * n + j]; G[a * m + c] = s; }
  const u = solve(G, y, m) ?? new Float64Array(m);
  return matrixOp(A, m, n).adjoint(u);
}

/** Mutual coherence max_{i≠j} |a_iᵀa_j| / (‖a_i‖‖a_j‖) of a dense m × n matrix. */
export function mutualCoherence(A: ArrayLike<number>, m: number, n: number): number {
  const cn = new Float64Array(n);
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += A[i * n + j] ** 2; cn[j] = Math.sqrt(s) || 1; }
  let mu = 0;
  for (let a = 0; a < n; a++) for (let c = a + 1; c < n; c++) {
    let s = 0; for (let i = 0; i < m; i++) s += A[i * n + a] * A[i * n + c];
    mu = Math.max(mu, Math.abs(s) / (cn[a] * cn[c]));
  }
  return mu;
}

// ---- the 2D picture: misfit ½(x − b)ᵀQ(x − b) (Q = AᵀA, b = least-squares point) + penalty

export type Sym2 = [number, number, number];   // Q = [[q0, q1], [q1, q2]]

/** ½(x − b)ᵀQ(x − b). */
export const misfit2 = (Q: Sym2, b: [number, number], x: [number, number]): number => {
  const u = x[0] - b[0], v = x[1] - b[1];
  return 0.5 * (Q[0] * u * u + 2 * Q[1] * u * v + Q[2] * v * v);
};

/** Exact 2D lasso: enumerate the supports and signs, keep the consistent stationary point with the
 *  lowest objective (convex problem, so this is the minimum). */
export function lasso2dExact(Q: Sym2, b: [number, number], lambda: number): [number, number] {
  const Qb = [Q[0] * b[0] + Q[1] * b[1], Q[1] * b[0] + Q[2] * b[1]];
  const cands: [number, number][] = [[0, 0]];
  for (const s of [-1, 1]) {
    const x1 = (Qb[0] - lambda * s) / Q[0]; if (Math.sign(x1) === s) cands.push([x1, 0]);
    const x2 = (Qb[1] - lambda * s) / Q[2]; if (Math.sign(x2) === s) cands.push([0, x2]);
  }
  const det = Q[0] * Q[2] - Q[1] * Q[1];
  if (Math.abs(det) > 1e-12) for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) {
    const r1 = Qb[0] - lambda * s1, r2 = Qb[1] - lambda * s2;
    const x1 = (Q[2] * r1 - Q[1] * r2) / det, x2 = (Q[0] * r2 - Q[1] * r1) / det;
    if (Math.sign(x1) === s1 && Math.sign(x2) === s2) cands.push([x1, x2]);
  }
  let best = cands[0], bf = Infinity;
  for (const c of cands) { const f = misfit2(Q, b, c) + lambda * (Math.abs(c[0]) + Math.abs(c[1])); if (f < bf - 1e-14) { bf = f; best = c; } }
  return best;
}

/** argmin ½(x − b)ᵀQ(x − b) + pen_p(x) in 2D for p = 0, ½, 1, 2. */
export function penalised2d(Q: Sym2, b: [number, number], lambda: number, p: PenaltyKind): [number, number] {
  if (lambda <= 0) return [b[0], b[1]];
  if (p === 1) return lasso2dExact(Q, b, lambda);
  const Qb = [Q[0] * b[0] + Q[1] * b[1], Q[1] * b[0] + Q[2] * b[1]];
  if (p === 2) {
    const a = Q[0] + lambda, c = Q[2] + lambda, d = a * c - Q[1] * Q[1];
    return [(c * Qb[0] - Q[1] * Qb[1]) / d, (a * Qb[1] - Q[1] * Qb[0]) / d];
  }
  const obj = (x: [number, number]) => misfit2(Q, b, x) + penaltyValue(x, lambda, p);
  if (p === 0) {
    const cands: [number, number][] = [[0, 0], [Qb[0] / Q[0], 0], [0, Qb[1] / Q[2]], [b[0], b[1]]];
    let best = cands[0], bf = Infinity;
    for (const c of cands) { const f = obj(c); if (f < bf - 1e-14) { bf = f; best = c; } }
    return best;
  }
  // p = 1/2: non-convex. Candidates: the origin, the best point on each axis (1D search) and a
  // grid search + coordinate refinement off the axes; keep the lowest objective.
  const R = Math.max(Math.abs(b[0]), Math.abs(b[1]), 1) * 1.2;
  const cands: [number, number][] = [[0, 0]];
  const axisBest = (axis: 0 | 1): [number, number] => {
    const q = axis === 0 ? Q[0] : Q[2], qb = Qb[axis];
    // ½ q t² − qb t + λ√|t| + const: scalar problem with w = qb/q, λ' = λ/q
    const t = lassoScalar(qb / q, lambda / q, 0.5);
    return axis === 0 ? [t, 0] : [0, t];
  };
  cands.push(axisBest(0), axisBest(1));
  let best: [number, number] = [b[0], b[1]], bf = obj(best);
  const G = 60;
  for (let i = 0; i <= G; i++) for (let j = 0; j <= G; j++) {
    const x: [number, number] = [-R + 2 * R * i / G, -R + 2 * R * j / G];
    if (x[0] === 0 || x[1] === 0) continue;
    const f = obj(x); if (f < bf) { bf = f; best = x; }
  }
  // coordinate refinement (each coordinate is a scalar ½-penalty problem)
  for (let it = 0; it < 60; it++) {
    for (const k of [0, 1] as const) {
      const o = 1 - k, q = k === 0 ? Q[0] : Q[2];
      const w = b[k] + Q[1] * (b[o] - best[o]) / q;
      const t = lassoScalar(w, lambda / q, 0.5);
      if (k === 0) best = [t, best[1]]; else best = [best[0], t];
    }
  }
  cands.push(best);
  let out = cands[0], of = Infinity;
  for (const c of cands) { const f = obj(c); if (f < of - 1e-14) { of = f; out = c; } }
  return out;
}

// ---------------------------------------------------------------------------------------
// 4. Compressed sensing

/** K distinct positions out of N, sorted. */
export function randomSupport(N: number, K: number, r: Rng): Int32Array {
  return Int32Array.from(sampleIndices(r, N, K)).sort();
}

/** K-sparse vector: random positions; amplitudes ±(1 … 2) ("pm") or N(0, 1) ("gauss"). */
export function sparseSignal(N: number, K: number, r: Rng, amp: "pm" | "gauss" | "pm1" = "pm"): Vec {
  const x = new Float64Array(N);
  for (const i of randomSupport(N, K, r)) {
    if (amp === "gauss") x[i] = gauss(r);
    else if (amp === "pm1") x[i] = r.unif() < 0.5 ? -1 : 1;
    else x[i] = (r.unif() < 0.5 ? -1 : 1) * (1 + r.unif());
  }
  return x;
}

/** Compressible coefficients: sorted magnitudes fall like k^(−decay), random signs and order. */
export function compressibleSignal(N: number, r: Rng, decay = 1.2): Vec {
  const x = new Float64Array(N), perm = sampleIndices(r, N, N);
  for (let k = 0; k < N; k++) x[perm[k]] = (r.unif() < 0.5 ? -1 : 1) * 2 * Math.pow(k + 1, -decay);
  return x;
}

/** A signal of a few spikes plus a few cosines (sparse only in the union of the two bases). */
export function spikesPlusSines(N: number, r: Rng, spikes = 4, sines = 3): { x: Vec; spikes: Vec; dct: Vec } {
  const sp = sparseSignal(N, spikes, r, "pm"), dc = new Float64Array(N);
  const ks = sampleIndices(r, Math.floor(N / 8) - 1, sines);
  for (const k of ks) dc[k + 1] = (r.unif() < 0.5 ? -1 : 1) * (3 + 2 * r.unif());
  const x = idctFast(dc);
  for (let i = 0; i < N; i++) x[i] += sp[i];
  return { x, spikes: sp, dct: dc };
}

/** Rows of the real Fourier basis to keep: "random" (M rows), "regular" (frequencies multiple of r),
 *  "lowpass" (the M lowest rows). */
export function fourierRows1d(N: number, kind: "random" | "regular" | "lowpass", M: number, r: Rng, every = 4): Int32Array {
  if (kind === "regular") {
    const rows: number[] = [];
    for (let j = 0; j < N; j++) if (realFourierFreq(j, N) % every === 0) rows.push(j);
    return Int32Array.from(rows);
  }
  if (kind === "lowpass") return Int32Array.from({ length: Math.min(M, N) }, (_, i) => i);
  return randomSupport(N, Math.min(M, N), r);
}

/** Point spread function of a 1D real-Fourier row selection: the zero-filled reconstruction of a
 *  spike at the centre (Σ_j φ_j(n) φ_j(c) over the kept rows). */
export function rowsPsf1d(N: number, rows: ArrayLike<number>): Vec {
  const c = new Float64Array(N), e = new Float64Array(N);
  e[N / 2] = 1;
  const full = realFourier(e);
  for (let i = 0; i < rows.length; i++) c[rows[i]] = full[rows[i]];
  return realFourierInverse(c);
}

// ---- 2D k-space masks (n × n, unshifted order: index ky*n + kx with freq(ky) in [−n/2, n/2))

/** Phase-encoding lines (rows ky) → full 2D mask. The mask is made symmetric (ky and −ky). */
export function linesToMask(lines: ArrayLike<number>, n: number): Uint8Array {
  const m = new Uint8Array(n * n);
  for (let i = 0; i < lines.length; i++) {
    for (const ky of [lines[i], (n - lines[i]) % n]) for (let kx = 0; kx < n; kx++) m[ky * n + kx] = 1;
  }
  return m;
}

/** Lines (as row indices) with |freq| < c/2: the fully sampled centre of c lines. */
export function centreLines(n: number, c: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < n; k++) if (Math.abs(freq(k, n)) < c / 2 || (c % 2 === 0 && freq(k, n) === -c / 2)) out.push(k);
  return out;
}

export type MaskKind = "full" | "centre" | "regular" | "random" | "vardens" | "radial";

/** A k-space sampling mask with about n/accel lines (symmetric under k ↦ −k).
 *  centre: the central n/accel lines; regular: every accel-th line + `centre` central lines;
 *  random: uniform random lines + centre; vardens: random with probability ∝ (1 − |k|/k_max)^q + centre;
 *  radial: `spokes` lines through the centre (spokes ≈ n·π/(2·accel)). */
export function kspaceMask(n: number, kind: MaskKind, accel: number, r: Rng, opts: { centre?: number; q?: number } = {}): Uint8Array {
  const centre = opts.centre ?? Math.max(4, Math.round(n / 16)), q = opts.q ?? 3;
  if (kind === "full" || accel <= 1) return new Uint8Array(n * n).fill(1);
  if (kind === "radial") return radialMask2d(n, Math.max(2, Math.round(n * Math.PI / (2 * accel))));
  const target = Math.max(1, Math.round(n / accel));
  if (kind === "centre") return linesToMask(centreLines(n, target), n);
  if (kind === "regular") {
    const lines = new Set<number>(centreLines(n, centre));
    for (let k = 0; k < n; k += Math.round(accel)) lines.add(k);
    return linesToMask([...lines], n);
  }
  // random lines, chosen in ± pairs (k ≥ 0 half), until the target count is reached
  const chosen = new Set<number>(centreLines(n, centre).map(k => Math.abs(freq(k, n))));
  const count = () => { let c = 0; for (const f of chosen) c += f === 0 || f === n / 2 ? 1 : 2; return c; };
  const kmax = n / 2;
  const w = new Float64Array(kmax + 1);
  for (let f = 0; f <= kmax; f++) w[f] = kind === "vardens" ? Math.pow(Math.max(1 - f / kmax, 0.02), q) : 1;
  let guard = 0;
  while (count() < target && guard++ < 100000) {
    let s = 0; for (let f = 0; f <= kmax; f++) if (!chosen.has(f)) s += w[f];
    if (s <= 0) break;
    let u = r.unif() * s, pick = -1;
    for (let f = 0; f <= kmax; f++) { if (chosen.has(f)) continue; u -= w[f]; if (u <= 0) { pick = f; break; } }
    if (pick < 0) break;
    chosen.add(pick);
  }
  const lines: number[] = [];
  for (const f of chosen) lines.push(((f % n) + n) % n);
  return linesToMask(lines, n);
}

/** Radial spokes through the centre of k-space (nearest-grid sampling), symmetric. */
export function radialMask2d(n: number, spokes: number): Uint8Array {
  const m = new Uint8Array(n * n);
  for (let s = 0; s < spokes; s++) {
    const a = Math.PI * s / spokes, c = Math.cos(a), si = Math.sin(a);
    for (let t = -n / 2; t <= n / 2; t += 0.5) {
      const kx = Math.round(t * c), ky = Math.round(t * si);
      if (kx < -n / 2 || kx >= n / 2 || ky < -n / 2 || ky >= n / 2) continue;
      for (const [u, v] of [[kx, ky], [-kx, -ky]]) {
        const ix = ((u % n) + n) % n, iy = ((v % n) + n) % n;
        m[iy * n + ix] = 1;
      }
    }
  }
  return m;
}

/** Fraction of k-space sampled. */
export const maskFraction = (m: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < m.length; i++) s += m[i] ? 1 : 0; return s / m.length; };

/** Point spread function of a 2D mask: inverse DFT of the mask, centred (fftshift), real part,
 *  scaled so that the central peak is the sampled fraction (1 for a full mask). */
export function maskPsf2d(mask: ArrayLike<number>, n: number): Vec {
  const re = Float64Array.from(mask, v => (v ? 1 : 0)), im = new Float64Array(n * n);
  fft2d(re, im, n, true);
  const out = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[((y + n / 2) % n) * n + ((x + n / 2) % n)] = re[y * n + x];
  return out;
}

/** 1D PSF of a set of k-space lines (the profile along ky through the centre). */
export function linePsf1d(mask: ArrayLike<number>, n: number): Vec {
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let ky = 0; ky < n; ky++) re[ky] = mask[ky * n] ? 1 : 0;
  fft1d(re, im, true);
  const o = new Float64Array(n);
  for (let i = 0; i < n; i++) o[(i + n / 2) % n] = re[i] / n;
  return o;
}

/** The sheppLogan ellipses plus small faint "lesions" (discs of contrast 0.02–0.05) for the
 *  "what does sparsity remove?" experiments. */
export const LESION_PHANTOM: PhantomEllipse[] = [
  ...SHEPP_LOGAN,
  { value: 0.05, a: 0.035, b: 0.035, x: -0.35, y: 0.45, angle: 0 },
  { value: 0.04, a: 0.025, b: 0.025, x: 0.38, y: 0.42, angle: 0 },
  { value: 0.03, a: 0.03, b: 0.03, x: 0.42, y: -0.32, angle: 0 },
  { value: 0.05, a: 0.02, b: 0.02, x: -0.42, y: -0.36, angle: 0 },
  { value: 0.02, a: 0.04, b: 0.04, x: 0.0, y: -0.33, angle: 0 },
];

// ---- recovery trials (phase diagram)

export type Sensing = "gaussian" | "fourier";
export type Recovery = "l1" | "omp";

export interface Trial { x: Vec; xhat: Vec; relErr: number; success: boolean; A: Vec; M: number }

/** ℓ1 recovery of a noise-free underdetermined system by FISTA with continuation in λ
 *  (λ from 0.5 λ_max down to 1e-5 λ_max, warm starts, momentum restarted per stage): approximates
 *  basis pursuit, min ‖x‖₁ subject to Ax = y. Dense loop without allocations (phase diagram). */
export function l1Recover(A: ArrayLike<number>, M: number, N: number, y: ArrayLike<number>,
  opts: { stages?: number; iters?: number; lambdaFinal?: number; polish?: boolean; snapshotEvery?: number; snapshots?: Vec[] } = {}): Vec {
  const op = matrixOp(A, M, N), L = powerIteration(op, 30) * 1.01;
  const lmax = lambdaMax(op, y), stages = opts.stages ?? 6, iters = opts.iters ?? 150;
  const lf = Math.min(opts.lambdaFinal ?? 1e-5 * lmax, 0.5 * lmax), ratio = lf / (0.5 * lmax);
  let count = 0;
  const x = new Float64Array(N), xo = new Float64Array(N), z = new Float64Array(N), r = new Float64Array(M);
  for (let s = 0; s < stages; s++) {
    const lam = lmax * 0.5 * Math.pow(ratio, s / (stages - 1)), th = lam / L;
    z.set(x); let t = 1;
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < M; i++) { let v = y[i]; const o = i * N; for (let j = 0; j < N; j++) v -= A[o + j] * z[j]; r[i] = v / L; }
      xo.set(x);
      let change = 0, size = 0;
      for (let j = 0; j < N; j++) {
        let g = z[j];
        for (let i = 0; i < M; i++) g += A[i * N + j] * r[i];
        const nx = g > th ? g - th : g < -th ? g + th : 0;
        x[j] = nx;
        change = Math.max(change, Math.abs(nx - xo[j])); size = Math.max(size, Math.abs(nx));
      }
      const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, b = (t - 1) / tn;
      for (let j = 0; j < N; j++) z[j] = x[j] + b * (x[j] - xo[j]);
      t = tn;
      if (opts.snapshots && ++count % (opts.snapshotEvery ?? 10) === 0) opts.snapshots.push(Float64Array.from(x));
      if (change <= 1e-10 * (size || 1)) break;
    }
  }
  if (opts.polish === false) return x;
  // polish: least squares on the support found (removes the remaining shrinkage bias)
  let mx = 0; for (let j = 0; j < N; j++) mx = Math.max(mx, Math.abs(x[j]));
  const S: number[] = []; for (let j = 0; j < N; j++) if (Math.abs(x[j]) > 1e-3 * mx) S.push(j);
  return S.length && S.length < M ? debias(A, M, N, y, S) : x;
}

/** One trial: K-sparse ±1 signal of length N, M measurements (Gaussian matrix or random rows of
 *  the real DFT), recovered by ℓ1 or OMP; success if the relative error is below `tol`. */
export function recoveryTrial(N: number, M: number, K: number, sensing: Sensing, solver: Recovery, seed: number, tol = 1e-2, noise = 0): Trial {
  const r = rng(seed);
  const x = sparseSignal(N, K, r, "pm1");
  let A: Vec;
  if (sensing === "gaussian") A = gaussianMatrix(M, N, r);
  else A = denseMatrix(realFourierOp1d(N, randomSupport(N, M, r)));
  const y = matrixOp(A, M, N).forward(x);
  // noise: standard deviation `noise`·‖y‖/√M per measurement (relative noise level `noise`)
  const sd = noise * norm2(y) / Math.sqrt(M);
  if (noise > 0) for (let i = 0; i < M; i++) y[i] += sd * r.norm();
  const xhat = solver === "omp"
    ? omp(A, M, N, y, { K: Math.min(M, Math.max(K, 1) * 2), tol: noise > 0 ? 1.05 * sd * Math.sqrt(M) : 1e-9 * norm2(y) }).x
    : l1Recover(A, M, N, y, noise > 0 ? { lambdaFinal: sd * meanColumnNorm(A, M, N) * Math.sqrt(2 * Math.log(N)) } : {});
  const e = new Float64Array(N); for (let i = 0; i < N; i++) e[i] = xhat[i] - x[i];
  const relErr = norm2(e) / (norm2(x) || 1);
  return { x, xhat, relErr, success: relErr < tol, A, M };
}

/** Root-mean-square column norm of a dense M × N matrix: Aᵀn has about this standard deviation
 *  per entry for white noise n of unit variance (sets the noise-level λ = σ·this·√(2 ln N)). */
export function meanColumnNorm(A: ArrayLike<number>, M: number, N: number): number {
  let s = 0; for (let i = 0; i < M * N; i++) s += A[i] * A[i];
  return Math.sqrt(s / N);
}

/** The seed of trial t of the cell (δ, ρ) (so one trial can be replayed). */
export const cellSeed = (seed0: number, delta: number, rho: number, t: number): number => seed0 * 7919 + Math.round(delta * 1000) * 101 + Math.round(rho * 1000) * 13 + t;

/** Fraction of successes over `trials` trials at δ = M/N, ρ = K/M. */
export function phaseCell(N: number, delta: number, rho: number, sensing: Sensing, solver: Recovery, trials: number, seed0 = 1, noise = 0): { rate: number; M: number; K: number } {
  const M = Math.max(1, Math.round(delta * N)), K = Math.max(1, Math.round(rho * M));
  let ok = 0;
  for (let t = 0; t < trials; t++) if (recoveryTrial(N, M, K, sensing, solver, cellSeed(seed0, delta, rho, t), noise > 0 ? 0.1 : 1e-2, noise).success) ok++;
  return { rate: ok / trials, M, K };
}

/** The ℓ1 weak phase transition ρ(δ) for signed sparse vectors and Gaussian matrices
 *  (Donoho & Tanner 2009), tabulated at δ = 0.02, 0.04, …, 1 from the state-evolution formula
 *  ρ(δ) = max_z [1 − (2/δ)((1 + z²)Φ(−z) − zφ(z))] / [1 + z² − 2((1 + z²)Φ(−z) − zφ(z))]
 *  of Donoho, Maleki & Montanari (2009, PNAS 106: 18914), which coincides with it. */
export const DONOHO_TANNER_L1: { delta: number[]; rho: number[] } = {
  delta: Array.from({ length: 50 }, (_, i) => Math.round((i + 1) * 2) / 100),
  rho: [0.1234, 0.1456, 0.1623, 0.1766, 0.1894, 0.2013, 0.2124, 0.2231, 0.2333, 0.2433, 0.2531, 0.2626, 0.2721, 0.2815, 0.2908,
    0.3001, 0.3093, 0.3186, 0.3279, 0.3373, 0.3468, 0.3563, 0.3660, 0.3758, 0.3857, 0.3958, 0.4061, 0.4166, 0.4274, 0.4384,
    0.4497, 0.4614, 0.4734, 0.4859, 0.4988, 0.5123, 0.5264, 0.5412, 0.5568, 0.5733, 0.5909, 0.6099, 0.6304, 0.6530, 0.6782,
    0.7068, 0.7405, 0.7823, 0.8401, 1.0000],
};

// ---------------------------------------------------------------------------------------
// 5. 2D restoration helpers (image-domain FISTA with a proximal step)

export type Prior2D = "none" | "tikhonov" | "haar" | "db4" | "tv";

/** Proximal step of λ·(wavelet ℓ1 of the details) for an orthonormal wavelet: soft shrinkage
 *  (lib/wavelet.ts denoise2; with shifts > 1 it averages over shifts, cycle spinning, which is
 *  a translation-invariant variant rather than the exact prox). */
export function waveletProx2d(v: ArrayLike<number>, w: number, h: number, wav: Wavelet, levels: number, t: number, shifts = 1): Vec {
  return denoise2(v, w, h, wav, levels, t, "soft", shifts).out;
}

/** Lasso-type objective in the image domain for display: ½‖Ax − y‖² + λ·g(x). */
export function dataMisfit(A: LinOp, y: ArrayLike<number>, x: ArrayLike<number>): number {
  const r = A.forward(x); let d = 0; for (let i = 0; i < r.length; i++) d += (r[i] - y[i]) ** 2; return 0.5 * d;
}
