// Model of the scorematch1d demo (no DOM; used by train_dsm.ts and demos/test/diffusion-test.ts).
//
// 1D data = a Gaussian mixture, so the best denoiser E[x0 | x] for x = x0 + σ ε is exact.
// Training pairs (x, x0, ε) at one noise level σ; their binned average; and a tiny learned
// denoiser: an MLP (x/√(σ² + 2), ln σ) → ε̂, 2 → 32 → 32 → 1, tanh, trained by denoising
// score matching (squared error on the noise we added) with σ log-uniform in [0.05, 3].
import { makeDiag, type DiagMixture } from "../lib/mixture";
import { mixtureDenoiser } from "../lib/diffusion";
import { rng } from "../lib/ui";
import { mlp, gradients, applyGradients, Adam, type Mlp } from "../lib/nn";

export type DataId = "two" | "widenarrow" | "rare";
export const DATA_IDS: DataId[] = ["two", "widenarrow", "rare"];
export const DATA_LABEL: Record<DataId, string> = { two: "two narrow modes", widenarrow: "wide + narrow", rare: "rare mode (5 %)" };

export interface Data1D { id: DataId; w: number[]; mu: number[]; v: number[]; m: DiagMixture }

export function data1d(id: DataId): Data1D {
  const p = id === "two" ? { w: [0.5, 0.5], mu: [-1.5, 1.5], v: [0.0625, 0.0625] }
    : id === "widenarrow" ? { w: [0.6, 0.4], mu: [-1, 1.8], v: [0.64, 0.0225] }
      : { w: [0.95, 0.05], mu: [-0.5, 2.5], v: [0.25, 0.0225] };
  return { id, ...p, m: makeDiag(p.w, p.mu.map(u => [u]), p.v) };
}

/** Exact noised density p_σ(x). */
export function density(d: Data1D, s: number, x: number): number {
  let p = 0;
  for (let k = 0; k < d.w.length; k++) { const c = d.v[k] + s * s; p += d.w[k] * Math.exp(-0.5 * (x - d.mu[k]) ** 2 / c) / Math.sqrt(2 * Math.PI * c); }
  return p;
}

/** Exact denoiser E[x0 | x] at noise level σ. */
export function exactDenoiser(d: Data1D): (x: number, s: number) => number {
  const D = mixtureDenoiser(d.m), a = new Float64Array(1), o = new Float64Array(1);
  return (x, s) => { a[0] = x; D(a, s, o); return o[0]; };
}

export interface Pairs { x: Float64Array; x0: Float64Array; eps: Float64Array }

/** n seeded training pairs at noise level σ (the same x0 and ε for every σ, so the slider moves them smoothly). */
export function pairs(d: Data1D, s: number, n = 400, seed = 7): Pairs {
  const r = rng(seed), x = new Float64Array(n), x0 = new Float64Array(n), eps = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = r.unif(), k = 0;
    while (k < d.w.length - 1 && u > d.w[k]) { u -= d.w[k]; k++; }
    x0[i] = d.mu[k] + Math.sqrt(d.v[k]) * r.norm();
    eps[i] = r.norm();
    x[i] = x0[i] + s * eps[i];
  }
  return { x, x0, eps };
}

/** Average of `val` in equal bins of x over [lo, hi]: centres, means (NaN if empty) and counts. */
export function binned(x: ArrayLike<number>, val: ArrayLike<number>, lo: number, hi: number, nb: number): { c: Float64Array; m: Float64Array; n: Int32Array } {
  const c = new Float64Array(nb), m = new Float64Array(nb), n = new Int32Array(nb), h = (hi - lo) / nb;
  for (let b = 0; b < nb; b++) c[b] = lo + (b + 0.5) * h;
  for (let i = 0; i < x.length; i++) { const b = Math.floor((x[i] - lo) / h); if (b >= 0 && b < nb) { m[b] += val[i]; n[b]++; } }
  for (let b = 0; b < nb; b++) m[b] = n[b] ? m[b] / n[b] : NaN;
  return { c, m, n };
}

// ---------------------------------------------------------------------------------------
// The learned denoiser

export const S_MIN = 0.05, S_MAX = 3;
export const netInput = (x: number, s: number): [number, number] => [x / Math.sqrt(s * s + 2), Math.log(s)];

export function newNet(): Mlp { return mlp([2, 32, 32, 1], ["tanh", "tanh", "id"], 5); }

/** Predicted noise ε̂(x, σ). */
export function epsHat(net: Mlp, x: number, s: number): number {
  const [a0, a1] = netInput(x, s);
  let h1 = new Float64Array(32), h2 = new Float64Array(32);
  const W0 = net.W[0], W1 = net.W[1], W2 = net.W[2];
  for (let o = 0; o < 32; o++) h1[o] = Math.tanh(net.b[0][o] + W0[2 * o] * a0 + W0[2 * o + 1] * a1);
  for (let o = 0; o < 32; o++) { let v = net.b[1][o]; for (let i = 0; i < 32; i++) v += W1[o * 32 + i] * h1[i]; h2[o] = Math.tanh(v); }
  let y = net.b[2][0]; for (let i = 0; i < 32; i++) y += W2[i] * h2[i];
  return y;
}

/** Learned denoiser x̂0 = x − σ ε̂. */
export const learnedDenoiser = (net: Mlp, x: number, s: number): number => x - s * epsHat(net, x, s);

export const BATCH = 128, LR = 0.002;
const DATA_SEED: Record<DataId, number> = { two: 11, widenarrow: 12, rare: 13 };

/**
 * Train steps [from, from + n) with Adam (state `opt`). The batch of step t is drawn from
 * rng(1000 · seed + t), so a run is reproducible step by step (σ log-uniform, fresh ε).
 */
export function trainDsm(net: Mlp, opt: Adam, d: Data1D, from: number, n: number): number {
  let loss = 0;
  const X: number[][] = [], Y: number[][] = [];
  for (let t = from; t < from + n; t++) {
    const r = rng(1000 * DATA_SEED[d.id] + t);
    X.length = 0; Y.length = 0;
    for (let b = 0; b < BATCH; b++) {
      let u = r.unif(), k = 0;
      while (k < d.w.length - 1 && u > d.w[k]) { u -= d.w[k]; k++; }
      const x0 = d.mu[k] + Math.sqrt(d.v[k]) * r.norm();
      const s = S_MIN * (S_MAX / S_MIN) ** r.unif(), e = r.norm();
      X.push(netInput(x0 + s * e, s)); Y.push([e]);
    }
    const g = gradients(net, X, Y);
    applyGradients(net, g, opt);
    loss = g.loss;
  }
  return loss;
}

/** Float32 base64 of all weights and biases (layer by layer: W then b). */
export function packNet(net: Mlp): string {
  const n = net.W.reduce((s, w, l) => s + w.length + net.b[l].length, 0), f = new Float32Array(n);
  let o = 0; net.W.forEach((w, l) => { f.set(w, o); o += w.length; f.set(net.b[l], o); o += net.b[l].length; });
  const u = new Uint8Array(f.buffer);
  let s = ""; for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]);
  return btoa(s);
}

export function unpackNet(b64: string): Mlp {
  const net = newNet(), bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  const f = new Float32Array(u.buffer);
  let o = 0; net.W.forEach((w, l) => { for (let i = 0; i < w.length; i++) w[i] = f[o++]; for (let i = 0; i < net.b[l].length; i++) net.b[l][i] = f[o++]; });
  return net;
}

export const SNAP_STEPS = [200, 2000, 20000];
