// Diffusion samplers and helpers for the diffusion-models day (owner: the elective day
// "Diffusion models", units score-matching/, diffusion-sampling/, guidance/,
// diffusion-inverse/). No DOM. Tests: demos/test/diffusion-test.ts.
//
// Two equivalent ways to write the forward process (see score-matching/lec.qmd):
//   VP ("variance preserving", the form of diffusion/):  x_t = √ā x_0 + √(1 − ā) ε
//   VE ("noise level"):                                    x   = x_0 + σ ε,  σ² = (1 − ā)/ā
// with x_VE = x_VP / √ā. The samplers here work in the noise-level form with a DENOISER
// D(x, σ) ≈ E[x_0 | x] (the core operation); the score is (D − x)/σ² (Tweedie).
// In this form the deterministic DDIM step is exactly an Euler step of the probability-flow
// ODE dx/dσ = (x − D(x, σ))/σ, and the ancestral (DDPM) step samples q(x_σ' | x_σ, x_0 = D).
// The VP one-step formulas of diffusion/ are kept (ddimStepVP, ddpmStepVP) for the by-hand
// numbers. Schedules ā(τ) stay in lib/mixture.ts (alphaBarAt), reused here.
import { alphaBarAt, type DiagMixture, type Schedule } from "./mixture";
import type { Rng } from "./random";

// ---------------------------------------------------------------------------------------
// Noise levels

/** log-SNR = log(ā/(1 − ā)). */
export const logSnr = (abar: number): number => Math.log(abar / (1 - abar));
/** ā from the log-SNR. */
export const abarFromLogSnr = (l: number): number => 1 / (1 + Math.exp(-l));
/** Noise level σ of the VE form from ā: σ = √((1 − ā)/ā) = 1/√SNR. */
export const sigmaFromAbar = (abar: number): number => Math.sqrt((1 - abar) / abar);
/** ā from σ: ā = 1/(1 + σ²). */
export const abarFromSigma = (s: number): number => 1 / (1 + s * s);

/** Time τ at which the schedule reaches ā (inverse of alphaBarAt; clamped to [0, 1]). */
export function tauFromAbar(abar: number, kind: Schedule = "cosine"): number {
  if (abar >= 1) return 0;
  if (kind === "linear") {
    // 9.95 τ² + 0.1 τ + log ā = 0
    const c = Math.log(abar), t = (-0.1 + Math.sqrt(0.01 - 4 * 9.95 * c)) / (2 * 9.95);
    return Math.min(1, Math.max(0, t));
  }
  const s = 0.008, f0 = Math.cos(s / (1 + s) * Math.PI / 2) ** 2;
  const c = Math.sqrt(Math.min(1, abar * f0));
  return Math.min(1, Math.max(0, Math.acos(c) * 2 / Math.PI * (1 + s) - s));
}

export type Spacing = "time" | "logsnr" | "data";

/**
 * The N + 1 noise levels σ_0 > σ_1 > … > σ_N = 0 of an N-step sampler.
 *  "time":   uniform in τ of the schedule (σ from ā(τ_i), τ_i = 1 − i/N);
 *  "logsnr": uniform in log σ from σ(τ = 1) to σ_min, then 0;
 *  "data":   dense near the data (σ_i^(1/ρ) uniform, ρ = 7), then 0.
 * σ_min = sigmaMin (default 0.002); σ_max = σ at τ = 1 of the schedule (cosine: 100).
 */
export function sigmaGrid(N: number, spacing: Spacing = "time", sched: Schedule = "cosine", sigmaMin = 0.002): Float64Array {
  const smax = sigmaFromAbar(alphaBarAt(1, sched));
  const out = new Float64Array(N + 1);
  if (spacing === "time") {
    for (let i = 0; i <= N; i++) out[i] = i === N ? 0 : sigmaFromAbar(alphaBarAt(1 - i / N, sched));
    return out;
  }
  if (N === 1) { out[0] = smax; out[1] = 0; return out; }
  for (let i = 0; i < N; i++) {
    const u = i / (N - 1);
    if (spacing === "logsnr") out[i] = Math.exp(Math.log(smax) + u * (Math.log(sigmaMin) - Math.log(smax)));
    else { const r = 7, a = smax ** (1 / r), b = sigmaMin ** (1 / r); out[i] = (a + u * (b - a)) ** r; }
  }
  out[N] = 0;
  return out;
}

// ---------------------------------------------------------------------------------------
// Denoisers

/** D(x, σ) → out (≈ E[x_0 | x_0 + σ ε = x]). d-dimensional. */
export type Denoiser = (x: Float64Array, sigma: number, out: Float64Array) => void;

/** Score ∇ log p_σ(x) = (D(x, σ) − x)/σ² (Tweedie), from a denoiser. */
export function scoreFromDenoiser(D: Denoiser, x: Float64Array, sigma: number, out: Float64Array): Float64Array {
  D(x, sigma, out);
  for (let i = 0; i < x.length; i++) out[i] = (out[i] - x[i]) / (sigma * sigma);
  return out;
}

/**
 * Exact denoiser of a diagonal Gaussian mixture p(x_0) = Σ π_k N(μ_k, diag v_k), noised
 * x = x_0 + σ ε: responsibilities r_k under N(μ_k, v_k + σ²) and
 * D = Σ_k r_k (μ_k + v_k/(v_k + σ²)(x − μ_k)).
 * `comps` restricts to a subset (the conditional denoiser of a class).
 */
export function mixtureDenoiser(m: DiagMixture, comps?: number[]): Denoiser {
  const list = comps ?? Array.from({ length: m.k }, (_, j) => j);
  const lw = new Float64Array(list.length);
  let wsum = 0; for (const j of list) wsum += m.w[j];
  return (x, s, out) => {
    const d = m.d, s2 = s * s;
    let mx = -Infinity;
    for (let a = 0; a < list.length; a++) {
      const j = list[a];
      let l = Math.log(m.w[j] / wsum);
      for (let i = 0; i < d; i++) { const c = m.v[j * d + i] + s2, e = x[i] - m.mu[j * d + i]; l -= 0.5 * (e * e / c + Math.log(c)); }
      lw[a] = l; if (l > mx) mx = l;
    }
    let z = 0; for (let a = 0; a < list.length; a++) { lw[a] = Math.exp(lw[a] - mx); z += lw[a]; }
    out.fill(0);
    for (let a = 0; a < list.length; a++) {
      const r = lw[a] / z; if (r < 1e-14) continue;
      const j = list[a];
      for (let i = 0; i < d; i++) { const v = m.v[j * d + i], mu = m.mu[j * d + i]; out[i] += r * (mu + v / (v + s2) * (x[i] - mu)); }
    }
  };
}

/** Exact noised density p_σ(x) of a 1D mixture (weights, means, variances). */
export function mixtureDensity1d(w: ArrayLike<number>, mu: ArrayLike<number>, v: ArrayLike<number>, sigma: number, x: number): number {
  let p = 0, ws = 0;
  for (let k = 0; k < w.length; k++) ws += w[k];
  for (let k = 0; k < w.length; k++) { const c = v[k] + sigma * sigma; p += w[k] / ws * Math.exp(-0.5 * (x - mu[k]) ** 2 / c) / Math.sqrt(2 * Math.PI * c); }
  return p;
}

/** Classifier-free combination of two denoiser outputs: (1 + w) D_cond − w D_uncond (same for ε̂ and the score). */
export function cfgCombine(dCond: ArrayLike<number>, dUncond: ArrayLike<number>, w: number, out: Float64Array = new Float64Array(dCond.length)): Float64Array {
  for (let i = 0; i < dCond.length; i++) out[i] = (1 + w) * dCond[i] - w * dUncond[i];
  return out;
}

/** A guided denoiser: (1 + w) D_cond − w D_uncond at every noise level. */
export function guidedDenoiser(cond: Denoiser, uncond: Denoiser, w: number): Denoiser {
  let a: Float64Array = new Float64Array(0), b: Float64Array = new Float64Array(0);
  return (x, s, out) => {
    if (a.length !== x.length) { a = new Float64Array(x.length); b = new Float64Array(x.length); }
    cond(x, s, a); uncond(x, s, b); cfgCombine(a, b, w, out);
  };
}

// ---------------------------------------------------------------------------------------
// One step, VP form (as in diffusion/; used for the by-hand numbers)

/** Deterministic (DDIM, η = 0) step from x at ā to āNext > ā with predicted noise ε̂. */
export function ddimStepVP(x: number, abar: number, abarNext: number, eps: number): { x0: number; x: number } {
  const x0 = (x - Math.sqrt(1 - abar) * eps) / Math.sqrt(abar);
  return { x0, x: Math.sqrt(abarNext) * x0 + Math.sqrt(1 - abarNext) * eps };
}

/** Ancestral (DDPM) step: mean and variance of q(x_{i−1} | x_i, x̂_0). */
export function ddpmStepVP(x: number, abar: number, abarNext: number, eps: number): { x0: number; mean: number; variance: number } {
  const x0 = (x - Math.sqrt(1 - abar) * eps) / Math.sqrt(abar);
  const al = abar / abarNext, be = 1 - al;
  const mean = Math.sqrt(abarNext) * be / (1 - abar) * x0 + Math.sqrt(al) * (1 - abarNext) / (1 - abar) * x;
  return { x0, mean, variance: be * (1 - abarNext) / (1 - abar) };
}

// ---------------------------------------------------------------------------------------
// One step, noise-level (VE) form, in place on x (length d). `D` is the denoiser,
// `tmp` a scratch vector of length d (holds D(x, σ) after the call).

/** Euler step of the probability-flow ODE from σ to σn (= DDIM, deterministic). */
export function eulerOdeStep(x: Float64Array, s: number, sn: number, D: Denoiser, tmp: Float64Array): void {
  D(x, s, tmp);
  for (let i = 0; i < x.length; i++) x[i] += (sn - s) * (x[i] - tmp[i]) / s;
}

/**
 * Two-stage step (second order; midpoint in log σ, as in DPM-Solver-2): a deterministic step to
 * σm = √(σ σn) with D(x, σ), then the whole step with the denoiser evaluated there:
 * x ← Dm + (σn/σ)(x − Dm). The deterministic (Euler/DDIM) step is x ← D + (σn/σ)(x − D).
 * To σn = 0 it is an ordinary step (returns the number of denoiser calls).
 */
export function twoStageOdeStep(x: Float64Array, s: number, sn: number, D: Denoiser, tmp: Float64Array, tmp2: Float64Array): number {
  D(x, s, tmp);
  const d = x.length;
  if (sn === 0) { x.set(tmp); return 1; }
  const sm = Math.sqrt(s * sn), xm = new Float64Array(d);
  for (let i = 0; i < d; i++) xm[i] = tmp[i] + (sm / s) * (x[i] - tmp[i]);
  D(xm, sm, tmp2);
  for (let i = 0; i < d; i++) x[i] = tmp2[i] + (sn / s) * (x[i] - tmp2[i]);
  return 2;
}

/**
 * DDIM step with stochasticity η ∈ [0, 1] (0: deterministic = Euler of the ODE; 1: ancestral =
 * DDPM). Fresh noise of variance c² = η² σn² (σ² − σn²)/σ²; the remaining √(σn² − c²) keeps
 * the predicted noise direction (x − D)/σ.
 */
export function ddimStep(x: Float64Array, s: number, sn: number, eta: number, D: Denoiser, tmp: Float64Array, r: Rng): void {
  D(x, s, tmp);
  const c2 = eta * eta * sn * sn * (s * s - sn * sn) / (s * s), a = Math.sqrt(Math.max(0, sn * sn - c2)), c = Math.sqrt(c2);
  for (let i = 0; i < x.length; i++) x[i] = tmp[i] + a * (x[i] - tmp[i]) / s + (c > 0 ? c * r.norm() : 0);
}

/** Ancestral step (DDPM in the noise-level form) = ddimStep with η = 1. */
export const ddpmStep = (x: Float64Array, s: number, sn: number, D: Denoiser, tmp: Float64Array, r: Rng): void => ddimStep(x, s, sn, 1, D, tmp, r);

/** One Langevin correction at fixed σ: x += h·score + √(2h) ξ, step h = ratio·σ². Keeps p_σ for small h. */
export function langevinStep(x: Float64Array, s: number, D: Denoiser, tmp: Float64Array, r: Rng, ratio = 0.1): void {
  D(x, s, tmp);
  const h = ratio * s * s;
  for (let i = 0; i < x.length; i++) x[i] += h * (tmp[i] - x[i]) / (s * s) + Math.sqrt(2 * h) * r.norm();
}

// ---------------------------------------------------------------------------------------
// Whole runs

export interface SampleOptions {
  /** Noise levels σ_0 > … > σ_N = 0 (sigmaGrid). */
  sigmas: Float64Array;
  /** 0 = deterministic, 1 = ancestral. */
  eta?: number;
  /** Two-stage (second-order) steps for the deterministic sampler (η = 0 only). */
  twoStage?: boolean;
  /** Langevin corrections after every step (at the new σ, not at σ = 0). */
  langevin?: number;
  langevinRatio?: number;
  /** Called after every step with the step index (1..N), the new σ and x (may modify x: data consistency). */
  after?: (i: number, sigma: number, x: Float64Array, d0: Float64Array) => void;
}

/** Run one sample from a start x (at σ_0; modified in place). Returns the number of denoiser calls. */
export function sampleRun(x: Float64Array, D: Denoiser, o: SampleOptions, r: Rng, record?: (i: number, x: Float64Array) => void): number {
  const d = x.length, tmp = new Float64Array(d), tmp2 = new Float64Array(d), sg = o.sigmas, N = sg.length - 1;
  const eta = o.eta ?? 0;
  let nfe = 0;
  record?.(0, x);
  for (let i = 0; i < N; i++) {
    const s = sg[i], sn = sg[i + 1];
    if (eta === 0 && o.twoStage) nfe += twoStageOdeStep(x, s, sn, D, tmp, tmp2);
    else { ddimStep(x, s, sn, eta, D, tmp, r); nfe++; }
    if (sn > 0) for (let k = 0; k < (o.langevin ?? 0); k++) { langevinStep(x, sn, D, tmp2, r, o.langevinRatio ?? 0.1); nfe++; }
    o.after?.(i + 1, sn, x, tmp);
    record?.(i + 1, x);
  }
  return nfe;
}

/** Run the deterministic sampler backwards (data → noise): Euler steps of the ODE from σ = sigmas[N] upwards. The first step starts at the smallest positive σ. */
export function invertOde(x: Float64Array, D: Denoiser, sigmas: Float64Array, record?: (i: number, x: Float64Array) => void): void {
  const tmp = new Float64Array(x.length), N = sigmas.length - 1;
  record?.(N, x);
  // from σ_N = 0 we cannot divide by σ: start at σ_{N−1} with x unchanged (x at σ ≈ 0 is the data point)
  for (let i = N - 1; i >= 1; i--) {
    const s = sigmas[i], sn = sigmas[i - 1];
    D(x, s, tmp);
    for (let k = 0; k < x.length; k++) x[k] += (sn - s) * (x[k] - tmp[k]) / s;
    record?.(i - 1, x);
  }
}

// ---------------------------------------------------------------------------------------
// Measurement consistency (inverse problems), for a measurement y_i = a_i x_i + n_i that is
// diagonal in the basis of x (a_i = 0: not measured).

/**
 * Push towards the measurement computed from the current clean estimate D(x) (the common
 * approximation: the data term is evaluated at D(x) as if it were certain):
 * x += ζ · a_i (y_i − a_i D_i), i.e. a gradient step of ½‖y − A D‖² with the Jacobian of D
 * replaced by the identity.
 */
export function posteriorGuidedStep(x: Float64Array, d0: ArrayLike<number>, y: ArrayLike<number>, a: ArrayLike<number>, zeta: number): void {
  for (let i = 0; i < x.length; i++) if (a[i] !== 0) x[i] += zeta * a[i] * (y[i] - a[i] * d0[i]);
}

/** Replacement (inpainting): measured coordinates (mask[i] = true) are set to y_i + σ ξ, the measurement noised to the current level. */
export function replacementStep(x: Float64Array, sigma: number, y: ArrayLike<number>, mask: ArrayLike<boolean | number>, r: Rng): void {
  for (let i = 0; i < x.length; i++) if (mask[i]) x[i] = y[i] + sigma * r.norm();
}

// ---------------------------------------------------------------------------------------
// Straight paths (flow matching), 1D diagonal mixture: x_τ = (1 − τ) x_0 + τ ε, velocity
// field v(x, τ) = E[ε − x_0 | x_τ = x] (exact for a mixture of Gaussians).

/** Exact marginal velocity of the straight path for a 1D mixture at time τ ∈ (0, 1]. */
export function straightVelocity1d(w: ArrayLike<number>, mu: ArrayLike<number>, v: ArrayLike<number>, x: number, tau: number): number {
  // x = (1 − τ) x0 + τ ε with x0 ~ N(μ, v): joint Gaussian; E[x0 | x] and E[ε | x] per component
  const a = 1 - tau;
  let mx = -Infinity;
  const lw: number[] = [], e0: number[] = [], ee: number[] = [];
  for (let k = 0; k < w.length; k++) {
    const c = a * a * v[k] + tau * tau, m = a * mu[k];
    lw.push(Math.log(w[k]) - 0.5 * ((x - m) ** 2 / c + Math.log(c)));
    e0.push(mu[k] + a * v[k] / c * (x - m));
    ee.push(tau / c * (x - m));
    mx = Math.max(mx, lw[k]);
  }
  let z = 0, out = 0;
  for (let k = 0; k < w.length; k++) { const p = Math.exp(lw[k] - mx); z += p; out += p * (ee[k] - e0[k]); }
  return out / z;
}
