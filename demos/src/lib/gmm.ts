// Gaussian mixture models fitted by EM (owner: Gaussian-mixture day). No DOM.
//
// Points: a flat row-major Float64Array X of n·d numbers, with optional weights w[i] ≥ 0 (a
// histogram bin is a point with weight = its count, so EM on a histogram is EM on its bins).
// A Mixture stores k weights π_k, k means (k·d) and k full covariance matrices (k·d·d,
// row-major), whatever the covariance type; the type only constrains the M-step:
//   spherical  Σ_k = σ_k² I          (1 variance per component)
//   diagonal   Σ_k = diag(σ_k1², …)  (d variances per component)
//   full       any Σ_k               (d(d+1)/2 numbers per component)
//   tied       one full Σ shared by all components
// `floor` (ε) is added to the diagonal of every Σ_k after each M-step: the variance floor that
// stops a component collapsing onto one point.
//
//   const m0 = initMixture(X, n, 2, 3, "kmeans++", rng(1), { covType: "full" });
//   const e = eStep(m0, X, n);                        // responsibilities, log p(D)
//   const m1 = mStep(X, n, 2, e.resp, 3, { covType: "full", floor: 1e-4 }, undefined, m0);
//   const fit = fitGMM(X, n, 2, 3, { covType: "full", seed: rng(1) });   // fit.trace: log p per iteration
//
// Separate eStep / mStep let demos show half-steps. Also: log-sum-exp, parameter counts,
// BIC / AIC, Mahalanobis distance, sampling from a mixture. Tests: demos/test/gmm-test.ts.
import type { Rng } from "./random";
import { choice, rng } from "./random";
import { cholesky, choleskyLogDet } from "./linalg";
import { kmeans, kmeansPlusPlus, randomCentres } from "./kmeans";

export type CovType = "spherical" | "diagonal" | "full" | "tied";
export const COV_TYPES: CovType[] = ["spherical", "diagonal", "full", "tied"];

export interface Mixture {
  k: number;
  d: number;
  weights: Float64Array;
  means: Float64Array;
  /** k full d×d covariance matrices, row-major, one after the other. */
  covs: Float64Array;
}

const LOG2PI = Math.log(2 * Math.PI);

/** Deep copy. */
export function cloneMixture(m: Mixture): Mixture {
  return { k: m.k, d: m.d, weights: Float64Array.from(m.weights), means: Float64Array.from(m.means), covs: Float64Array.from(m.covs) };
}

/** A mixture from plain arrays: weights, means [[...]], covariances [[[...]]] (or variances for d = 1). */
export function makeMixture(weights: number[], means: number[][], covs: (number[][] | number)[]): Mixture {
  const k = weights.length, d = means[0].length;
  const m: Mixture = { k, d, weights: Float64Array.from(weights), means: new Float64Array(k * d), covs: new Float64Array(k * d * d) };
  for (let j = 0; j < k; j++) {
    for (let a = 0; a < d; a++) m.means[j * d + a] = means[j][a];
    const c = covs[j];
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) m.covs[j * d * d + a * d + b] = typeof c === "number" ? (a === b ? c : 0) : c[a][b];
  }
  return m;
}

/** log Σ exp(a[off..off+k-1]), safe for large and -∞ values. */
export function logSumExp(a: ArrayLike<number>, off = 0, k = a.length): number {
  let mx = -Infinity;
  for (let j = 0; j < k; j++) if (a[off + j] > mx) mx = a[off + j];
  if (mx === -Infinity) return -Infinity;
  let s = 0;
  for (let j = 0; j < k; j++) s += Math.exp(a[off + j] - mx);
  return mx + Math.log(s);
}

/** 1D normal density. */
export function normalPdf(x: number, mu: number, variance: number): number {
  return Math.exp(-0.5 * (x - mu) * (x - mu) / variance) / Math.sqrt(2 * Math.PI * variance);
}

/** Cholesky factor of a covariance, adding a tiny ridge if it is not positive definite. */
function cholJitter(S: Float64Array, d: number): Float64Array {
  let L = cholesky(S, d);
  if (L) return L;
  let tr = 0;
  for (let a = 0; a < d; a++) tr += Math.abs(S[a * d + a]);
  let eps = 1e-12 * (tr / d || 1);
  const T = Float64Array.from(S);
  for (let tries = 0; tries < 60; tries++) {
    for (let a = 0; a < d; a++) T[a * d + a] = S[a * d + a] + eps;
    L = cholesky(T, d);
    if (L) return L;
    eps *= 10;
  }
  const I = new Float64Array(d * d);
  for (let a = 0; a < d; a++) I[a * d + a] = 1;
  return I;
}

/** Per-component Cholesky factors and log normalisers, reused for many points. */
export interface Prepared {
  L: Float64Array[];
  /** log π_k − d/2 log 2π − 1/2 log|Σ_k| */
  logc: Float64Array;
}

export function prepare(m: Mixture): Prepared {
  const { k, d } = m, L: Float64Array[] = [], logc = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    const S = m.covs.subarray(j * d * d, (j + 1) * d * d);
    const Lj = cholJitter(Float64Array.from(S), d);
    L.push(Lj);
    logc[j] = Math.log(m.weights[j]) - 0.5 * d * LOG2PI - 0.5 * choleskyLogDet(Lj, d);
  }
  return { L, logc };
}

/** Squared Mahalanobis distance of x (at offset xo) to component j, from its Cholesky factor. */
function maha2L(x: ArrayLike<number>, xo: number, mean: ArrayLike<number>, mo: number, L: Float64Array, d: number, tmp: Float64Array): number {
  let s = 0;
  for (let a = 0; a < d; a++) {
    let v = x[xo + a] - mean[mo + a];
    for (let b = 0; b < a; b++) v -= L[a * d + b] * tmp[b];
    v /= L[a * d + a];
    tmp[a] = v;
    s += v * v;
  }
  return s;
}

/** Squared Mahalanobis distance (x − μ)ᵀ Σ⁻¹ (x − μ). */
export function mahalanobis2(x: ArrayLike<number>, mean: ArrayLike<number>, cov: ArrayLike<number>, d: number): number {
  const L = cholJitter(Float64Array.from(cov), d);
  return maha2L(x, 0, mean, 0, L, d, new Float64Array(d));
}

/** Squared Mahalanobis distance of point x to every component (length k). */
export function componentMahalanobis2(m: Mixture, x: ArrayLike<number>, prep: Prepared = prepare(m)): Float64Array {
  const out = new Float64Array(m.k), tmp = new Float64Array(m.d);
  for (let j = 0; j < m.k; j++) out[j] = maha2L(x, 0, m.means, j * m.d, prep.L[j], m.d, tmp);
  return out;
}

/** log(π_k N(x_i | μ_k, Σ_k)) for every point and component: n·k values. */
export function logJoint(m: Mixture, X: ArrayLike<number>, n: number, prep: Prepared = prepare(m), out?: Float64Array): Float64Array {
  const { k, d } = m, o = out ?? new Float64Array(n * k), tmp = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) {
    o[i * k + j] = m.weights[j] > 0 ? prep.logc[j] - 0.5 * maha2L(X, i * d, m.means, j * d, prep.L[j], d, tmp) : -Infinity;
  }
  return o;
}

/** log p(x) of the mixture at one point. */
export function logDensity(m: Mixture, x: ArrayLike<number>, prep: Prepared = prepare(m)): number {
  return logSumExp(logJoint(m, x, 1, prep), 0, m.k);
}

export interface EResult {
  /** Responsibilities r_ik (n·k), each row sums to 1. */
  resp: Float64Array;
  /** log p(x_i) per point. */
  logp: Float64Array;
  /** log p(D) = Σ_i w_i log p(x_i). */
  logLik: number;
}

/** E-step: responsibilities r_ik = π_k N_k(x_i) / p(x_i), and the log-likelihood of the data
 *  under the current parameters. `hard`: one-hot at the most probable component instead. */
export function eStep(m: Mixture, X: ArrayLike<number>, n: number, w?: ArrayLike<number>, hard = false, prep: Prepared = prepare(m)): EResult {
  const k = m.k, lj = logJoint(m, X, n, prep), resp = new Float64Array(n * k), logp = new Float64Array(n);
  let L = 0;
  for (let i = 0; i < n; i++) {
    const lp = logSumExp(lj, i * k, k);
    logp[i] = lp;
    L += (w ? w[i] : 1) * lp;
    if (hard) {
      let best = 0;
      for (let j = 1; j < k; j++) if (lj[i * k + j] > lj[i * k + best]) best = j;
      resp[i * k + best] = 1;
    } else if (lp === -Infinity) {
      for (let j = 0; j < k; j++) resp[i * k + j] = 1 / k;
    } else {
      for (let j = 0; j < k; j++) resp[i * k + j] = Math.exp(lj[i * k + j] - lp);
    }
  }
  return { resp, logp, logLik: L };
}

/** log p(D) = Σ_i w_i log p(x_i). */
export function logLikelihood(m: Mixture, X: ArrayLike<number>, n: number, w?: ArrayLike<number>): number {
  return eStep(m, X, n, w).logLik;
}

export interface MOptions {
  covType?: CovType;
  /** Variance floor ε added to every diagonal (default 0). */
  floor?: number;
  /** Keep π_k = 1/k (used for "hard EM = k-means"). */
  equalWeights?: boolean;
  /** Keep the covariances of `prev` (only the weights and means are re-estimated). */
  fixCovs?: boolean;
  /** Numerical minimum variance (default 1e-10), so a collapsed component stays finite. */
  minVar?: number;
}

/** M-step: N_k = Σ_i w_i r_ik, π_k = N_k / Σ w, μ_k = weighted mean, Σ_k = weighted covariance
 *  (constrained by the covariance type), plus the floor. A component with N_k = 0 keeps the
 *  mean and covariance of `prev` (if given) with weight 0. */
export function mStep(X: ArrayLike<number>, n: number, d: number, resp: ArrayLike<number>, k: number,
  o: MOptions = {}, w?: ArrayLike<number>, prev?: Mixture): Mixture {
  const type = o.covType ?? "full", floor = o.floor ?? 0, minVar = o.minVar ?? 1e-10;
  const N = new Float64Array(k), mu = new Float64Array(k * d), S = new Float64Array(k * d * d);
  let W = 0;
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    W += wi;
    for (let j = 0; j < k; j++) {
      const r = wi * resp[i * k + j];
      if (r === 0) continue;
      N[j] += r;
      for (let a = 0; a < d; a++) mu[j * d + a] += r * X[i * d + a];
    }
  }
  const empty = (j: number) => !(N[j] > 1e-12 * Math.max(1, W));
  for (let j = 0; j < k; j++) for (let a = 0; a < d; a++) mu[j * d + a] = empty(j) ? (prev ? prev.means[j * d + a] : 0) : mu[j * d + a] / N[j];
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    for (let j = 0; j < k; j++) {
      const r = wi * resp[i * k + j];
      if (r === 0 || empty(j)) continue;
      for (let a = 0; a < d; a++) {
        const da = X[i * d + a] - mu[j * d + a];
        for (let b = a; b < d; b++) S[j * d * d + a * d + b] += r * da * (X[i * d + b] - mu[j * d + b]);
      }
    }
  }
  const covs = new Float64Array(k * d * d);
  if (type === "tied") {
    const T = new Float64Array(d * d);
    for (let j = 0; j < k; j++) for (let a = 0; a < d; a++) for (let b = a; b < d; b++) T[a * d + b] += S[j * d * d + a * d + b];
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { T[a * d + b] /= W || 1; T[b * d + a] = T[a * d + b]; }
    for (let j = 0; j < k; j++) covs.set(T, j * d * d);
  } else {
    for (let j = 0; j < k; j++) {
      const off = j * d * d;
      if (empty(j)) { if (prev) covs.set(prev.covs.subarray(off, off + d * d), off); else for (let a = 0; a < d; a++) covs[off + a * d + a] = 1; continue; }
      if (type === "full") {
        for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { const v = S[off + a * d + b] / N[j]; covs[off + a * d + b] = v; covs[off + b * d + a] = v; }
      } else if (type === "diagonal") {
        for (let a = 0; a < d; a++) covs[off + a * d + a] = S[off + a * d + a] / N[j];
      } else {
        let t = 0;
        for (let a = 0; a < d; a++) t += S[off + a * d + a];
        for (let a = 0; a < d; a++) covs[off + a * d + a] = t / (d * N[j]);
      }
    }
  }
  if (o.fixCovs && prev) covs.set(prev.covs);
  else for (let j = 0; j < k; j++) for (let a = 0; a < d; a++) {
    const q = j * d * d + a * d + a;
    covs[q] = Math.max(covs[q], minVar) + floor;
  }
  const weights = new Float64Array(k);
  for (let j = 0; j < k; j++) weights[j] = o.equalWeights ? 1 / k : N[j] / (W || 1);
  return { k, d, weights, means: mu, covs };
}

/** Mean and covariance (divided by Σ w) of weighted points. */
export function dataMoments(X: ArrayLike<number>, n: number, d: number, w?: ArrayLike<number>): { mean: Float64Array; cov: Float64Array; total: number } {
  const resp = new Float64Array(n).fill(1);
  const m = mStep(X, n, d, resp, 1, { covType: "full", minVar: 0 }, w);
  let total = 0;
  for (let i = 0; i < n; i++) total += w ? w[i] : 1;
  return { mean: m.means, cov: m.covs, total };
}

export type GmmInit = "random" | "kmeans++" | "kmeans";

/** Starting mixture: means from random points, k-means++ seeds or a full k-means run; equal
 *  weights; every covariance = the data covariance (constrained to the type) divided by k²
 *  (k-means: the covariance of each cluster). */
export function initMixture(X: ArrayLike<number>, n: number, d: number, k: number, how: GmmInit, r: Rng,
  o: MOptions & { weights?: ArrayLike<number> } = {}): Mixture {
  const w = o.weights;
  if (how === "kmeans") {
    const km = kmeans(X, n, d, k, { init: kmeansPlusPlus(X, n, d, k, r, w), weights: w });
    const resp = new Float64Array(n * k);
    for (let i = 0; i < n; i++) resp[i * k + km.assign[i]] = 1;
    const base = fromCentres(X, n, d, k, km.centres, o, w);
    return mStep(X, n, d, resp, k, { ...o, minVar: o.minVar }, w, base);
  }
  const C = how === "kmeans++" ? kmeansPlusPlus(X, n, d, k, r, w) : randomCentres(X, n, d, k, r, w);
  return fromCentres(X, n, d, k, C, o, w);
}

/** Mixture with the given centres, equal weights and covariance = data covariance / k² (typed). */
export function fromCentres(X: ArrayLike<number>, n: number, d: number, k: number, C: ArrayLike<number>,
  o: MOptions = {}, w?: ArrayLike<number>): Mixture {
  const mom = dataMoments(X, n, d, w), type = o.covType ?? "full", s = 1 / (k * k);
  const one = new Float64Array(d * d);
  let tr = 0;
  for (let a = 0; a < d; a++) tr += mom.cov[a * d + a];
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) {
    one[a * d + b] = type === "spherical" ? (a === b ? tr / d : 0) : type === "diagonal" ? (a === b ? mom.cov[a * d + a] : 0) : mom.cov[a * d + b];
    one[a * d + b] *= s;
    if (a === b) one[a * d + b] = Math.max(one[a * d + b], 1e-10) + (o.floor ?? 0);
  }
  const covs = new Float64Array(k * d * d);
  for (let j = 0; j < k; j++) covs.set(one, j * d * d);
  return { k, d, weights: new Float64Array(k).fill(1 / k), means: Float64Array.from(C), covs };
}

export interface FitOptions extends MOptions {
  init?: GmmInit | Mixture;
  seed?: Rng;
  weights?: ArrayLike<number>;
  maxIter?: number;
  /** Stop when log p(D) improves by less than tol (default 1e-6 · Σw). */
  tol?: number;
  /** Hard (classification) EM. */
  hard?: boolean;
}

export interface FitResult {
  model: Mixture;
  resp: Float64Array;
  logLik: number;
  /** log p(D) before the first M-step and after every iteration. */
  trace: number[];
  iterations: number;
  converged: boolean;
}

/** EM until convergence: E, M, E, M, … The trace holds log p(D) at every E-step. */
export function fitGMM(X: ArrayLike<number>, n: number, d: number, k: number, o: FitOptions = {}): FitResult {
  const w = o.weights;
  let m = typeof o.init === "object" ? cloneMixture(o.init) : initMixture(X, n, d, k, o.init ?? "kmeans", o.seed ?? rng(1), { ...o, weights: w });
  let e = eStep(m, X, n, w, o.hard);
  const trace = [e.logLik];
  let total = 0;
  for (let i = 0; i < n; i++) total += w ? w[i] : 1;
  const tol = o.tol ?? 1e-6 * Math.max(1, total), maxIter = o.maxIter ?? 200;
  let it = 0, converged = false;
  while (it < maxIter) {
    it++;
    m = mStep(X, n, d, e.resp, k, o, w, m);
    e = eStep(m, X, n, w, o.hard);
    trace.push(e.logLik);
    if (Math.abs(e.logLik - trace[trace.length - 2]) < tol) { converged = true; break; }
  }
  return { model: m, resp: e.resp, logLik: e.logLik, trace, iterations: it, converged };
}

/** Number of free parameters: (k − 1) weights + k·d means + the covariance numbers. */
export function nParams(k: number, d: number, type: CovType): number {
  const cov = type === "spherical" ? k : type === "diagonal" ? k * d : type === "full" ? k * d * (d + 1) / 2 : d * (d + 1) / 2;
  return k - 1 + k * d + cov;
}

/** BIC = −2 log p(D) + p ln n (smaller is better). */
export const bic = (logLik: number, p: number, n: number): number => -2 * logLik + p * Math.log(n);
/** AIC = −2 log p(D) + 2p (smaller is better). */
export const aic = (logLik: number, p: number): number => -2 * logLik + 2 * p;

/** Most probable component per point from responsibilities (n·k). */
export function hardLabels(resp: ArrayLike<number>, n: number, k: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0;
    for (let j = 1; j < k; j++) if (resp[i * k + j] > resp[i * k + b]) b = j;
    out[i] = b;
  }
  return out;
}

/** Draw n points: pick component k with probability π_k, then x = μ_k + L_k z (Σ_k = L_k L_kᵀ). */
export function sampleMixture(m: Mixture, n: number, r: Rng): { X: Float64Array; comp: Int32Array } {
  const { k, d } = m, prep = prepare(m), X = new Float64Array(n * d), comp = new Int32Array(n), z = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const j = k > 1 ? choice(r, m.weights) : 0;
    comp[i] = j;
    for (let a = 0; a < d; a++) z[a] = r.norm();
    for (let a = 0; a < d; a++) {
      let v = m.means[j * d + a];
      for (let b = 0; b <= a; b++) v += prep.L[j][a * d + b] * z[b];
      X[i * d + a] = v;
    }
  }
  return { X, comp };
}

/** 2×2 covariance of component j of a 2D (or the (a, b) block of a higher-dimensional) mixture. */
export function cov2(m: Mixture, j: number, a = 0, b = 1): [[number, number], [number, number]] {
  const d = m.d, o = j * d * d;
  return [[m.covs[o + a * d + a], m.covs[o + a * d + b]], [m.covs[o + b * d + a], m.covs[o + b * d + b]]];
}
