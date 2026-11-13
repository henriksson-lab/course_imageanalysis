// Model of the posterior1d demo (no DOM; tested in demos/test/diffusion-test.ts).
//
// Signals of 64 samples: background 0.1 plus one or two Gaussian spots. A training set of 400
// such signals defines the prior: one narrow Gaussian (sd 0.04 per sample) around each, i.e. a
// mixture with isotropic components, which stays the same mixture in any orthonormal basis.
// The measurement y = a ∘ (B x) + noise is diagonal in a basis B: the cosine basis (DCT-II)
// for a mirror-boundary Gaussian blur, an ideal low-pass (explicit cut-off: everything above
// it is the null space) and pure noise; the pixel basis for missing samples (a gap). So the
// posterior is exact (lib/mixture.ts posteriorDiag). Also: the Tikhonov estimate of the
// deconvolution day (gradient penalty, λ by the discrepancy principle), the approximate
// posterior sampler of lib/diffusion.ts (guided reverse diffusion with the clean-estimate push
// or, for a gap, replacement), coverage of the 90 % band.
import { makeDiag, posteriorDiag, posteriorMean, posteriorSample, type DiagMixture, type DiagPosterior } from "../lib/mixture";
import { dctMatrix } from "../lib/dct";
import { solve } from "../lib/linalg";
import { discrepancyLambda } from "../lib/deconv";
import { sigmaGrid, sampleRun, posteriorGuidedStep, replacementStep, type Denoiser } from "../lib/diffusion";
import { rng } from "../lib/ui";
import type { Rng } from "../lib/random";

export const N = 64, BG = 0.1, PRIOR_SD = 0.05, NTRAIN = 400;
export type PriorId = "matched" | "noclose" | "wide";
export type MeasId = "lowpass" | "blur" | "noise" | "gap";

export interface Spot { pos: number; amp: number; sd: number }

export function render(spots: Spot[]): Float64Array {
  const x = new Float64Array(N).fill(BG);
  for (const s of spots) for (let i = 0; i < N; i++) x[i] += s.amp * Math.exp(-0.5 * ((i - s.pos) / s.sd) ** 2);
  return x;
}

/** One random signal of the data-generating process (1 or 2 spots; separation 2–14 px). */
export function randomSpots(r: Rng, prior: PriorId = "matched"): Spot[] {
  const sd = prior === "wide" ? 2.6 : 1.5;
  const two = r.unif() < 0.5, c = 28 + 8 * r.unif();
  const amp = () => 0.75 + 0.2 * r.unif();
  if (!two) return [{ pos: c, amp: amp(), sd }];
  let sep = 2 + 12 * r.unif();
  if (prior === "noclose" && sep < 8) return [{ pos: c, amp: amp(), sd }];   // close pairs never seen
  return [{ pos: c - sep / 2, amp: amp(), sd }, { pos: c + sep / 2, amp: amp(), sd }];
}

/** The training signals (pixel basis), N × NTRAIN row-major per signal. */
export function trainingSet(prior: PriorId): Float64Array[] {
  const r = rng(prior === "matched" ? 401 : prior === "noclose" ? 402 : 403), out: Float64Array[] = [];
  for (let i = 0; i < NTRAIN; i++) out.push(render(randomSpots(r, prior)));
  return out;
}

let DCT: Float64Array | null = null;
export const dct = (): Float64Array => (DCT ??= dctMatrix(N));
/** Coefficients in the basis: c = B x. */
export function toBasis(x: ArrayLike<number>, pixel: boolean): Float64Array {
  if (pixel) return Float64Array.from(x);
  const D = dct(), c = new Float64Array(N);
  for (let k = 0; k < N; k++) { let s = 0; for (let n = 0; n < N; n++) s += D[k * N + n] * x[n]; c[k] = s; }
  return c;
}
export function fromBasis(c: ArrayLike<number>, pixel: boolean): Float64Array {
  if (pixel) return Float64Array.from(c);
  const D = dct(), x = new Float64Array(N);
  for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < N; k++) s += D[k * N + n] * c[k]; x[n] = s; }
  return x;
}

export interface Meas { id: MeasId; blur: number; cutoff: number; gap: [number, number]; noise: number }

/** Diagonal of the measurement in its basis. */
export function measDiag(m: Meas): { a: Float64Array; pixel: boolean } {
  const a = new Float64Array(N);
  if (m.id === "gap") { for (let i = 0; i < N; i++) a[i] = i >= m.gap[0] && i < m.gap[1] ? 0 : 1; return { a, pixel: true }; }
  if (m.id === "noise") { a.fill(1); return { a, pixel: false }; }
  if (m.id === "lowpass") { for (let k = 0; k < N; k++) a[k] = k <= m.cutoff ? 1 : 0; return { a, pixel: false }; }
  // mirror-boundary Gaussian blur: eigenvalue h0 + 2 Σ_j h_j cos(π k j / N)
  const R = Math.ceil(4 * m.blur), h: number[] = [];
  let s = 0; for (let j = -R; j <= R; j++) { const v = Math.exp(-0.5 * (j / m.blur) ** 2); h.push(v); s += v; }
  for (let k = 0; k < N; k++) { let e = h[R] / s; for (let j = 1; j <= R; j++) e += 2 * h[R + j] / s * Math.cos(Math.PI * k * j / N); a[k] = e; }
  return { a, pixel: false };
}

export function priorMixture(train: Float64Array[], pixel: boolean): DiagMixture {
  return makeDiag(new Array(train.length).fill(1), train.map(x => toBasis(x, pixel)), PRIOR_SD ** 2);
}

/** y in the measurement basis (unmeasured coordinates are 0). */
export function measure(x: Float64Array, m: Meas, seed: number): Float64Array {
  const { a, pixel } = measDiag(m), c = toBasis(x, pixel), r = rng(seed), y = new Float64Array(N);
  for (let i = 0; i < N; i++) y[i] = a[i] !== 0 ? a[i] * c[i] + m.noise * r.norm() : 0;
  return y;
}

/** The measurement shown as a signal: B^T y (low-passed / blurred, noisy; gap = NaN). */
export function measuredSignal(y: Float64Array, m: Meas): Float64Array {
  const { a, pixel } = measDiag(m);
  if (pixel) return Float64Array.from(y, (v, i) => a[i] === 0 ? NaN : v);
  return fromBasis(y, false);
}

export function posterior(train: Float64Array[], y: Float64Array, m: Meas): { post: DiagPosterior; pixel: boolean } {
  const { a, pixel } = measDiag(m);
  return { post: posteriorDiag(priorMixture(train, pixel), y, a, m.noise ** 2), pixel };
}

export const postMeanSignal = (p: { post: DiagPosterior; pixel: boolean }): Float64Array => fromBasis(posteriorMean(p.post), p.pixel);

export function postSamples(p: { post: DiagPosterior; pixel: boolean }, n: number, seed: number): Float64Array[] {
  const r = rng(seed), out: Float64Array[] = [];
  for (let i = 0; i < n; i++) out.push(fromBasis(posteriorSample(p.post, r).x, p.pixel));
  return out;
}

/**
 * Exact denoiser of an equal-weight mixture of isotropic Gaussians N(μ_k, v I) (the prior),
 * faster than the general mixtureDenoiser: ‖x − μ_k‖² = ‖x‖² − 2 x·μ_k + ‖μ_k‖², and
 * D = (σ² Σ r_k μ_k + v x)/(v + σ²). Same result (checked in the test).
 */
export function isoDenoiser(mus: Float64Array[], v: number): Denoiser {
  const K = mus.length, d = mus[0].length, n2 = mus.map(u => u.reduce((s, t) => s + t * t, 0)), lw = new Float64Array(K);
  return (x, s, out) => {
    const c = v + s * s;
    let x2 = 0; for (let i = 0; i < d; i++) x2 += x[i] * x[i];
    let mx = -Infinity;
    for (let k = 0; k < K; k++) { const u = mus[k]; let dot = 0; for (let i = 0; i < d; i++) dot += x[i] * u[i]; lw[k] = -0.5 * (x2 - 2 * dot + n2[k]) / c; if (lw[k] > mx) mx = lw[k]; }
    let z = 0; for (let k = 0; k < K; k++) { lw[k] = Math.exp(lw[k] - mx); z += lw[k]; }
    out.fill(0);
    for (let k = 0; k < K; k++) { const r = lw[k] / z; if (r < 1e-12) continue; const u = mus[k]; for (let i = 0; i < d; i++) out[i] += r * u[i]; }
    for (let i = 0; i < d; i++) out[i] = (s * s * out[i] + v * x[i]) / c;
  };
}

/**
 * Approximate posterior samples by guided reverse diffusion with the exact prior denoiser:
 * after every step a push ζ·a ∘ (y − a ∘ D(x)) (the clean-estimate approximation), or, for a
 * gap, replacement of the measured samples by the measurement noised to the current level.
 */
export function approxSamples(train: Float64Array[], y: Float64Array, m: Meas, n: number, steps: number, zeta: number, seed: number): Float64Array[] {
  const { a, pixel } = measDiag(m), D = isoDenoiser(train.map(x => toBasis(x, pixel)), PRIOR_SD ** 2);
  const sig = sigmaGrid(steps, "time", "cosine"), r = rng(seed), rn = rng(seed + 1), out: Float64Array[] = [];
  const mask = Array.from(a, v => v !== 0);
  for (let p = 0; p < n; p++) {
    const x = new Float64Array(N);
    for (let i = 0; i < N; i++) x[i] = Math.sqrt(sig[0] ** 2 + 1) * r.norm();
    sampleRun(x, D, {
      sigmas: sig, eta: 1,
      after: (_i, sn, xv, d0) => {
        if (m.id === "gap") { if (sn > 0) replacementStep(xv, sn, y, mask, rn); else for (let i = 0; i < N; i++) if (mask[i]) xv[i] = y[i]; }
        else posteriorGuidedStep(xv, d0, y, a, zeta);
      },
    }, rn);
    out.push(fromBasis(x, pixel));
  }
  return out;
}

/** Tikhonov with a gradient penalty: (AᵀA + λ DᵀD) x = Aᵀ y + λ DᵀD·0, λ by the discrepancy principle. */
export function tikhonov(y: Float64Array, m: Meas): { x: Float64Array; lambda: number } {
  const { a, pixel } = measDiag(m);
  // A in the pixel basis: Bᵀ diag(a) B
  const B = pixel ? null : dct();
  const A = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (pixel) { A[i * N + j] = i === j ? a[i] : 0; continue; }
    let s = 0; for (let k = 0; k < N; k++) s += B![k * N + i] * a[k] * B![k * N + j]; A[i * N + j] = s;
  }
  const ys = pixel ? y : fromBasis(y, false);   // Bᵀ y (pixel-basis measurement)
  const AtA = new Float64Array(N * N), Aty = new Float64Array(N);
  for (let i = 0; i < N; i++) { for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < N; k++) s += A[k * N + i] * A[k * N + j]; AtA[i * N + j] = s; } let s = 0; for (let k = 0; k < N; k++) s += A[k * N + i] * ys[k]; Aty[i] = s; }
  const DtD = new Float64Array(N * N);
  for (let i = 0; i < N - 1; i++) { DtD[i * N + i] += 1; DtD[(i + 1) * N + i + 1] += 1; DtD[i * N + i + 1] -= 1; DtD[(i + 1) * N + i] -= 1; }
  const nMeas = Array.from(a).filter(v => v !== 0).length;
  const est = (lam: number) => { const M = AtA.map((v, i) => v + lam * DtD[i] + 1e-9 * (i % (N + 1) === 0 ? 1 : 0)); return solve(M, Aty, N) ?? new Float64Array(N); };
  const rmsRes = (lam: number) => { const x = est(lam), c = toBasis(x, pixel); let s = 0; for (let i = 0; i < N; i++) if (a[i] !== 0) s += (y[i] - a[i] * c[i]) ** 2; return Math.sqrt(s / nMeas); };
  const lambda = discrepancyLambda(rmsRes, m.noise, 1e-6, 1e3);
  return { x: est(lambda), lambda };
}

/** Residual ‖y − A x‖² / (number measured · σ²): about 1 for an estimate consistent with the noise. */
export function residualRatio(x: Float64Array, y: Float64Array, m: Meas): number {
  const { a, pixel } = measDiag(m), c = toBasis(x, pixel);
  let s = 0, n = 0; for (let i = 0; i < N; i++) if (a[i] !== 0) { s += (y[i] - a[i] * c[i]) ** 2; n++; }
  return s / (n * m.noise ** 2);
}

/** The part of x the measurement does not see (|a| below 0.05: null space and coefficients far below the noise), as a signal. */
export function nullPart(x: Float64Array, m: Meas): Float64Array {
  const { a, pixel } = measDiag(m), c = toBasis(x, pixel);
  for (let i = 0; i < N; i++) if (Math.abs(a[i]) >= 0.05) c[i] = 0;
  return fromBasis(c, pixel);
}

/** Number of spots: local maxima above background + 0.3. */
export function countSpots(x: ArrayLike<number>): number {
  let c = 0;
  for (let i = 1; i < N - 1; i++) if (x[i] > BG + 0.3 && x[i] >= x[i - 1] && x[i] > x[i + 1]) c++;
  return c;
}

/** Pointwise 5 % and 95 % quantiles of a set of signals. */
export function band(S: Float64Array[]): { lo: Float64Array; hi: Float64Array } {
  const lo = new Float64Array(N), hi = new Float64Array(N), col = new Float64Array(S.length);
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < S.length; s++) col[s] = S[s][i];
    col.sort();
    const q = (p: number) => { const t = p * (S.length - 1), f = Math.floor(t), g = t - f; return f + 1 < S.length ? col[f] * (1 - g) + col[f + 1] * g : col[f]; };
    lo[i] = q(0.05); hi[i] = q(0.95);
  }
  return { lo, hi };
}

/**
 * Coverage of the 90 % band: `reps` truths drawn from the TRUE prior (the matched mixture:
 * a training signal plus its small variation), measured, posterior under `prior`, band from
 * 100 posterior samples; fraction of (position, repeat) pairs with the truth inside the band.
 */
export function coverage(prior: PriorId, m: Meas, reps = 200, seed = 900): number {
  const trueTrain = trainingSet("matched"), train = prior === "matched" ? trueTrain : trainingSet(prior);
  const r = rng(seed);
  let inside = 0;
  for (let k = 0; k < reps; k++) {
    const base = trueTrain[Math.floor(r.unif() * trueTrain.length)];
    const x = Float64Array.from(base, v => v + PRIOR_SD * r.norm());
    const y = measure(x, m, seed + 1 + k);
    const S = postSamples(posterior(train, y, m), 100, seed + 5000 + k), b = band(S);
    for (let i = 0; i < N; i++) if (x[i] >= b.lo[i] && x[i] <= b.hi[i]) inside++;
  }
  return inside / (reps * N);
}

/** The test signal of a preset: spots at the given separation, centred, amplitudes 0.9 and 0.8. */
export function testSignal(sep: number): Float64Array {
  return render(sep > 0 ? [{ pos: 32 - sep / 2, amp: 0.9, sd: 1.5 }, { pos: 32 + sep / 2, amp: 0.8, sd: 1.5 }] : [{ pos: 32, amp: 0.9, sd: 1.5 }]);
}
