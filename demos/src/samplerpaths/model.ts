// Model of the samplerpaths demo (no DOM; tested in demos/test/diffusion-test.ts).
// 1D data = a Gaussian mixture; exact denoiser at every noise level; samplers of lib/diffusion.ts
// in the noise-level form x = x₀ + σε. For display, positions are shown in the
// variance-preserving form x/√(1 + σ²) (= √ā x₀ + √(1 − ā) ε) against the time τ of the
// cosine schedule, so the whole picture stays bounded.
import { rng } from "../lib/ui";
import { sigmaGrid, sampleRun, invertOde, tauFromAbar, abarFromSigma, type Denoiser, type Spacing } from "../lib/diffusion";
import { frechetGaussian, wasserstein1d } from "../lib/genmetrics";

export type DataId = "two" | "three" | "gauss";
export const DATA_IDS: DataId[] = ["two", "three", "gauss"];
export const DATA_LABEL: Record<DataId, string> = { two: "two modes", three: "three unequal", gauss: "one Gaussian" };
export interface Mix1 { w: number[]; mu: number[]; v: number[] }

export function data(id: DataId): Mix1 {
  if (id === "two") return { w: [0.5, 0.5], mu: [-1.5, 1.5], v: [0.04, 0.04] };
  if (id === "three") return { w: [0.5, 0.3, 0.2], mu: [-1.6, 0.3, 1.8], v: [0.09, 0.02, 0.05] };
  return { w: [1], mu: [0], v: [1] };
}

/** Fast exact 1D denoiser of a mixture. */
export function denoiser(m: Mix1): Denoiser {
  const K = m.w.length, lw = new Float64Array(K);
  return (x, s, out) => {
    const s2 = s * s, xv = x[0];
    let mx = -Infinity;
    for (let k = 0; k < K; k++) { const c = m.v[k] + s2; lw[k] = Math.log(m.w[k]) - 0.5 * ((xv - m.mu[k]) ** 2 / c + Math.log(c)); if (lw[k] > mx) mx = lw[k]; }
    let z = 0, d = 0;
    for (let k = 0; k < K; k++) { const p = Math.exp(lw[k] - mx); z += p; d += p * (m.mu[k] + m.v[k] / (m.v[k] + s2) * (xv - m.mu[k])); }
    out[0] = d / z;
  };
}

/** Density of the VP form y = √ā x₀ + √(1 − ā) ε at y. */
export function densityVP(m: Mix1, abar: number, y: number): number {
  let p = 0;
  for (let k = 0; k < m.w.length; k++) { const c = abar * m.v[k] + 1 - abar, e = y - Math.sqrt(abar) * m.mu[k]; p += m.w[k] * Math.exp(-0.5 * e * e / c) / Math.sqrt(2 * Math.PI * c); }
  return p;
}

export interface RunOptions { steps: number; eta: number; spacing: Spacing; twoStage: boolean; langevin: number; seed: number; n: number; nPaths: number; startScale?: number }


export interface RunResult {
  /** τ of each grid point (0..N). */
  taus: Float64Array;
  sigmas: Float64Array;
  /** Paths (nPaths × (N + 1)) in display (VP) coordinates. */
  paths: Float64Array[];
  ends: Float64Array;
  nfe: number;
}

export const toVP = (x: number, s: number): number => x / Math.sqrt(1 + s * s);

export function run(m: Mix1, o: RunOptions): RunResult {
  const sig = sigmaGrid(o.steps, o.spacing, "cosine");
  const taus = Float64Array.from(sig, s => s === 0 ? 0 : tauFromAbar(abarFromSigma(s)));
  const D = denoiser(m), r0 = rng(o.seed), rn = rng(o.seed + 7777);
  const paths: Float64Array[] = [], ends = new Float64Array(o.n), x = new Float64Array(1);
  let nfe = 0;
  for (let p = 0; p < o.n; p++) {
    x[0] = (o.startScale ?? 1) * Math.sqrt(sig[0] * sig[0] + 1) * r0.norm();
    const path = p < o.nPaths ? new Float64Array(sig.length) : null;
    const k = sampleRun(x, D, { sigmas: sig, eta: o.eta, twoStage: o.twoStage, langevin: o.langevin }, rn, path ? (i, xv) => { path[i] = toVP(xv[0], sig[i]); } : undefined);
    if (p === 0) nfe = k;
    if (path) paths.push(path);
    ends[p] = x[0];
  }
  return { taus, sigmas: sig, paths, ends, nfe };
}

/** Deterministic inversion of a data point x₀: the path to noise, and the noise value (VP) at τ = 1. */
export function invert(m: Mix1, x0: number, steps: number, spacing: Spacing): { path: Float64Array; z: number; back: number } {
  const sig = sigmaGrid(steps, spacing, "cosine"), D = denoiser(m), x = new Float64Array([x0]);
  const path = new Float64Array(sig.length);
  invertOde(x, D, sig, (i, xv) => { path[i] = toVP(xv[0], sig[i]); });
  const z = path[0];
  // and back: the deterministic sampler from that noise
  const y = new Float64Array([x[0]]);
  sampleRun(y, D, { sigmas: sig, eta: 0 }, rng(1));
  return { path, z, back: y[0] };
}

/** True samples of the data (for the distances). */
export function truth(m: Mix1, n: number, seed = 99): Float64Array {
  const r = rng(seed), out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = r.unif(), k = 0;
    while (k < m.w.length - 1 && u > m.w[k]) { u -= m.w[k]; k++; }
    out[i] = m.mu[k] + Math.sqrt(m.v[k]) * r.norm();
  }
  return out;
}

export function mixMeanVar(m: Mix1): { mean: number; variance: number } {
  let mu = 0, e2 = 0;
  m.w.forEach((w, k) => { mu += w * m.mu[k]; e2 += w * (m.v[k] + m.mu[k] ** 2); });
  return { mean: mu, variance: e2 - mu * mu };
}

/** Summary of the endpoints against the exact data distribution. */
export function summary(m: Mix1, ends: Float64Array, tr: Float64Array): { mean: number; sd: number; off: number; frechet: number; w1: number } {
  const n = ends.length;
  let mu = 0; for (const v of ends) mu += v / n;
  let vv = 0; for (const v of ends) vv += (v - mu) ** 2 / n;
  let off = 0;
  for (const v of ends) { let near = false; for (let k = 0; k < m.w.length; k++) if (Math.abs(v - m.mu[k]) < 3 * Math.sqrt(m.v[k])) near = true; if (!near) off++; }
  const t = mixMeanVar(m);
  return { mean: mu, sd: Math.sqrt(vv), off: off / n, frechet: frechetGaussian([mu], [vv], [t.mean], [t.variance], 1), w1: wasserstein1d(ends, tr) };
}
