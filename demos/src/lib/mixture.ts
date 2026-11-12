// Gaussian mixtures with diagonal covariances, for generative models (owner: modern deep
// learning day, unit diffusion/; demos diffusion2d and genrestore). No DOM.
//
// p(x) = Σ_k π_k N(x; μ_k, diag(v_k)) in d dimensions. Provides the density, the
// responsibilities r_k(x) = P(component k | x), the score ∇ log p(x) = Σ_k r_k (μ_k − x) / v_k,
// the mixture after forward noising x_t = √ā x_0 + √(1 − ā) ε (a mixture again, with means
// √ā μ_k and variances ā v_k + 1 − ā), sampling, and the exact posterior under a linear
// measurement y = A x + n that is diagonal in the basis the mixture is written in (a_i = 0:
// coordinate i not measured). Fitting by EM lives in lib/gmm.ts (GMM elective).
import type { Rng } from "./random";

export interface DiagMixture {
  d: number;
  k: number;
  w: Float64Array;       // weights π_k (sum 1)
  mu: Float64Array;      // k × d means, row-major
  v: Float64Array;       // k × d variances
}

/** Mixture from weights (normalised here), means and variances (one number = isotropic). */
export function makeDiag(weights: ArrayLike<number>, means: ArrayLike<number>[], vars: number | ArrayLike<number>[] | ArrayLike<number>): DiagMixture {
  const k = means.length, d = means[0].length;
  let s = 0; for (let j = 0; j < k; j++) s += weights[j];
  const w = Float64Array.from({ length: k }, (_, j) => weights[j] / s);
  const mu = new Float64Array(k * d), v = new Float64Array(k * d);
  for (let j = 0; j < k; j++) for (let i = 0; i < d; i++) {
    mu[j * d + i] = means[j][i];
    const vj = typeof vars === "number" ? vars : typeof (vars as ArrayLike<number>[])[j] === "number" ? (vars as ArrayLike<number>)[j] : (vars as ArrayLike<number>[])[j][i];
    v[j * d + i] = vj;
  }
  return { d, k, w, mu, v };
}

/** log π_k + log N(x; μ_k, v_k) for every component. */
export function logJointDiag(m: DiagMixture, x: ArrayLike<number>, out = new Float64Array(m.k)): Float64Array {
  const { d } = m;
  for (let j = 0; j < m.k; j++) {
    let s = Math.log(m.w[j]);
    for (let i = 0; i < d; i++) {
      const v = m.v[j * d + i], e = x[i] - m.mu[j * d + i];
      s -= 0.5 * (e * e / v + Math.log(2 * Math.PI * v));
    }
    out[j] = s;
  }
  return out;
}

function lse(a: ArrayLike<number>): number {
  let mx = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  if (!Number.isFinite(mx)) return mx;
  let s = 0; for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - mx);
  return mx + Math.log(s);
}

/** log p(x). */
export function logDensityDiag(m: DiagMixture, x: ArrayLike<number>): number {
  return lse(logJointDiag(m, x));
}

/** Responsibilities r_k(x) = P(k | x) (sum 1). */
export function responsibilities(m: DiagMixture, x: ArrayLike<number>, out = new Float64Array(m.k)): Float64Array {
  logJointDiag(m, x, out);
  const L = lse(out);
  for (let j = 0; j < m.k; j++) out[j] = Math.exp(out[j] - L);
  return out;
}

/** Score ∇_x log p(x) = Σ_k r_k(x) (μ_k − x) / v_k. */
export function scoreDiag(m: DiagMixture, x: ArrayLike<number>, out = new Float64Array(m.d), r = new Float64Array(m.k)): Float64Array {
  responsibilities(m, x, r);
  out.fill(0);
  for (let j = 0; j < m.k; j++) {
    if (r[j] < 1e-12) continue;
    for (let i = 0; i < m.d; i++) out[i] += r[j] * (m.mu[j * m.d + i] - x[i]) / m.v[j * m.d + i];
  }
  return out;
}

/** The distribution of x_t = √ā x_0 + √(1 − ā) ε when x_0 follows m: means √ā μ_k, variances ā v_k + 1 − ā. */
export function noisedMixture(m: DiagMixture, abar: number): DiagMixture {
  const s = Math.sqrt(abar);
  return { d: m.d, k: m.k, w: m.w, mu: m.mu.map(v => s * v), v: m.v.map(v => abar * v + 1 - abar) };
}

/** The mixture restricted to some components (weights renormalised): a "condition". */
export function restrictMixture(m: DiagMixture, comps: number[]): DiagMixture {
  const d = m.d, k = comps.length;
  const w = Float64Array.from(comps, j => m.w[j]);
  const mu = new Float64Array(k * d), v = new Float64Array(k * d);
  comps.forEach((j, n) => { mu.set(m.mu.subarray(j * d, j * d + d), n * d); v.set(m.v.subarray(j * d, j * d + d), n * d); });
  let s = 0; for (const x of w) s += x;
  return { d, k, w: w.map(x => x / s), mu, v };
}

/** Pick a component index from weights (sum 1) with one uniform number. */
export function pickComponent(w: ArrayLike<number>, u: number): number {
  let c = 0;
  for (let j = 0; j < w.length; j++) { c += w[j]; if (u < c) return j; }
  return w.length - 1;
}

/** n samples (n × d, row-major) and their components. */
export function sampleDiag(m: DiagMixture, n: number, r: Rng): { X: Float64Array; comp: Int32Array } {
  const X = new Float64Array(n * m.d), comp = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    const j = pickComponent(m.w, r.unif());
    comp[s] = j;
    for (let i = 0; i < m.d; i++) X[s * m.d + i] = m.mu[j * m.d + i] + Math.sqrt(m.v[j * m.d + i]) * r.norm();
  }
  return { X, comp };
}

/** Exact posterior of x given y = A x + n under a mixture prior: again a mixture. */
export interface DiagPosterior {
  /** Posterior weights of the components (sum 1). */
  w: Float64Array;
  /** log evidence log π_k N(y; a μ_k, a² v_k + σ²) per component (up to the same constant). */
  logEvidence: Float64Array;
  mean: Float64Array;    // k × d posterior means of the components
  v: Float64Array;       // k × d posterior variances
  d: number;
  k: number;
}

/**
 * Posterior for a measurement y_i = a_i x_i + n_i, n_i ~ N(0, σ_i²), coordinate by coordinate
 * (A diagonal in this basis). a_i = 0 means "not measured". Component k gives
 *   evidence  Π_i N(y_i; a_i μ_ki, a_i² v_ki + σ_i²),
 *   mean      μ_ki + v_ki a_i (y_i − a_i μ_ki) / (a_i² v_ki + σ_i²),
 *   variance  v_ki σ_i² / (a_i² v_ki + σ_i²).
 * `comps` limits the work to a subset (others get weight 0).
 */
export function posteriorDiag(m: DiagMixture, y: ArrayLike<number>, a: ArrayLike<number>, noiseVar: number | ArrayLike<number>,
  comps?: number[]): DiagPosterior {
  const { d, k } = m;
  const logE = new Float64Array(k).fill(-Infinity), mean = new Float64Array(k * d), v = new Float64Array(k * d);
  const list = comps ?? Array.from({ length: k }, (_, j) => j);
  for (const j of list) {
    let s = Math.log(m.w[j]);
    for (let i = 0; i < d; i++) {
      const ai = a[i], vi = m.v[j * d + i], mi = m.mu[j * d + i];
      const sn = typeof noiseVar === "number" ? noiseVar : noiseVar[i];
      if (ai === 0) { mean[j * d + i] = mi; v[j * d + i] = vi; continue; }
      const c = ai * ai * vi + sn, e = y[i] - ai * mi;
      s -= 0.5 * (e * e / c + Math.log(c));
      mean[j * d + i] = mi + vi * ai * e / c;
      v[j * d + i] = vi * sn / c;
    }
    logE[j] = s;
  }
  const L = lse(logE), w = logE.map(x => Math.exp(x - L));
  return { w, logEvidence: logE, mean, v, d, k };
}

/** Posterior mean Σ_k w_k mean_k. */
export function posteriorMean(p: DiagPosterior): Float64Array {
  const out = new Float64Array(p.d);
  for (let j = 0; j < p.k; j++) {
    if (p.w[j] < 1e-12) continue;
    for (let i = 0; i < p.d; i++) out[i] += p.w[j] * p.mean[j * p.d + i];
  }
  return out;
}

/** One posterior sample: a component by its weight, then its Gaussian. */
export function posteriorSample(p: DiagPosterior, r: Rng): { x: Float64Array; comp: number } {
  const j = pickComponent(p.w, r.unif()), x = new Float64Array(p.d);
  for (let i = 0; i < p.d; i++) x[i] = p.mean[j * p.d + i] + Math.sqrt(p.v[j * p.d + i]) * r.norm();
  return { x, comp: j };
}

// ---------------------------------------------------------------------------------------
// Noise schedules (diffusion), in continuous time τ ∈ [0, 1]: ā(0) = 1 (data), ā(1) ≈ 0 (noise).

export type Schedule = "linear" | "cosine";

/**
 * ā(τ). Linear: β(τ) rising linearly from 0.1 to 20, ā = exp(−∫β) = exp(−(0.1 τ + 9.95 τ²))
 * (the continuous form of the classic 1000-step linear schedule). Cosine:
 * ā = f(τ)/f(0), f(τ) = cos²((τ + s)/(1 + s)·π/2), s = 0.008, floored at 1e-4.
 */
export function alphaBarAt(tau: number, kind: Schedule = "cosine"): number {
  if (kind === "linear") return Math.exp(-(0.1 * tau + 0.5 * (20 - 0.1) * tau * tau));
  const s = 0.008, f = (t: number) => Math.cos((t + s) / (1 + s) * Math.PI / 2) ** 2;
  return Math.max(1e-4, f(tau) / f(0));
}

/** ā at the N + 1 times τ_i = i / N (i = 0..N), for a sampler with N steps. */
export function alphaBar(N: number, kind: Schedule = "cosine"): Float64Array {
  return Float64Array.from({ length: N + 1 }, (_, i) => alphaBarAt(i / N, kind));
}
