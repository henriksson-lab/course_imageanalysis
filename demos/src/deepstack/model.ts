// A deep stack of dense layers for the normalisation unit: a batch of random inputs goes up
// through `depth` layers (weights normal with sd gain/sqrt(fan-in), zero biases, optional batch
// or layer normalisation of the sums, a bend, optional residual connection whose blocks start
// small), and a random loss
// gradient comes back down. Records per layer the size (root mean square) of the activations and
// of the gradients, their histograms, and the share of dead or saturated units.
// No DOM, so the exercise numbers come from this code (demos/test/training-test.ts).
import { rng } from "../lib/ui";

export type Act = "linear" | "relu" | "tanh" | "sigmoid";
export type Norm = "none" | "batch" | "layer";

export interface StackConfig {
  depth: number; width: number; batch: number;
  gain: number;          // weight sd = gain / sqrt(fan-in)
  act: Act; norm: Norm;
  residual: boolean;     // a ← a + bend(norm(W a)) from layer 2 on, those W scaled by 1/sqrt(depth)
  seed: number;
}
export const DEFAULT_STACK: StackConfig = { depth: 30, width: 64, batch: 64, gain: Math.SQRT2, act: "relu", norm: "none", residual: false, seed: 1 };

export const bend = (act: Act, z: number): number => act === "relu" ? Math.max(0, z) : act === "tanh" ? Math.tanh(z) : act === "sigmoid" ? 1 / (1 + Math.exp(-z)) : z;
export const slope = (act: Act, z: number): number => {
  if (act === "relu") return z > 0 ? 1 : 0;
  if (act === "tanh") { const t = Math.tanh(z); return 1 - t * t; }
  if (act === "sigmoid") { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }
  return 1;
};

/** Normalise z (batch × width, row-major) in place: "batch" each unit (column) over the batch,
 *  "layer" each example (row) over its units. Returns the 1/sqrt(var + eps) used, per group. */
export function normalise(z: Float64Array, batch: number, width: number, norm: Norm, eps = 1e-5): Float64Array {
  if (norm === "none") return new Float64Array(0);
  const groups = norm === "batch" ? width : batch, n = norm === "batch" ? batch : width;
  const idx = (g: number, k: number) => (norm === "batch" ? k * width + g : g * width + k);
  const inv = new Float64Array(groups);
  for (let g = 0; g < groups; g++) {
    let m = 0;
    for (let k = 0; k < n; k++) m += z[idx(g, k)];
    m /= n;
    let v = 0;
    for (let k = 0; k < n; k++) v += (z[idx(g, k)] - m) ** 2;
    v /= n;
    inv[g] = 1 / Math.sqrt(v + eps);
    for (let k = 0; k < n; k++) z[idx(g, k)] = (z[idx(g, k)] - m) * inv[g];
  }
  return inv;
}

/** Backward through `normalise`: given dL/dy and y (the normalised values), dL/dz =
 *  inv · (dy − mean(dy) − y · mean(dy · y)), the means over each group. In place on dy. */
export function normaliseBack(dy: Float64Array, y: Float64Array, inv: Float64Array, batch: number, width: number, norm: Norm): void {
  if (norm === "none") return;
  const groups = norm === "batch" ? width : batch, n = norm === "batch" ? batch : width;
  const idx = (g: number, k: number) => (norm === "batch" ? k * width + g : g * width + k);
  for (let g = 0; g < groups; g++) {
    let md = 0, mdy = 0;
    for (let k = 0; k < n; k++) { md += dy[idx(g, k)]; mdy += dy[idx(g, k)] * y[idx(g, k)]; }
    md /= n; mdy /= n;
    for (let k = 0; k < n; k++) { const i = idx(g, k); dy[i] = inv[g] * (dy[i] - md - y[i] * mdy); }
  }
}

export interface LayerStats {
  rms: number;           // root mean square of the layer's output a
  gradRms: number;       // root mean square of dL/da at the layer's output
  dead: number;          // share of units that are off (ReLU 0, or |slope| < 0.01) for every example
  saturated: number;     // share of all (example, unit) pairs with slope < 0.01 (tanh, sigmoid)
  values: Float64Array;  // the outputs (batch × width), for histograms
}

export interface StackResult { layers: LayerStats[]; inputRms: number }

/** Forward a random batch up the stack, then a random loss gradient (normal, sd 1, at the top) down. */
export function runStack(c: StackConfig): StackResult {
  const r = rng(c.seed), B = c.batch, W = c.width;
  let a: Float64Array = Float64Array.from({ length: B * W }, () => r.norm());
  const inputRms = rmsOf(a);
  const Ws: Float64Array[] = [], zs: Float64Array[] = [], ins: Float64Array[] = [], invs: Float64Array[] = [], outs: Float64Array[] = [];
  for (let l = 0; l < c.depth; l++) {
    // in a residual stack every block starts as a small change: its weights × 1/sqrt(depth)
    const sd = c.gain / Math.sqrt(W) / (c.residual && l > 0 ? Math.sqrt(c.depth) : 1);
    const w = Float64Array.from({ length: W * W }, () => sd * r.norm());
    const z = new Float64Array(B * W);
    for (let n = 0; n < B; n++) for (let j = 0; j < W; j++) {
      let s = 0;
      for (let i = 0; i < W; i++) s += w[j * W + i] * a[n * W + i];
      z[n * W + j] = s;
    }
    const inv = normalise(z, B, W, c.norm);
    const out = new Float64Array(B * W);
    const res = c.residual && l > 0;
    for (let k = 0; k < B * W; k++) out[k] = bend(c.act, z[k]) + (res ? a[k] : 0);
    Ws.push(w); zs.push(z); ins.push(a); invs.push(inv); outs.push(out);
    a = out;
  }
  // backward
  const gradRms = new Array<number>(c.depth);
  let d: Float64Array = Float64Array.from({ length: B * W }, () => r.norm());
  for (let l = c.depth - 1; l >= 0; l--) {
    gradRms[l] = rmsOf(d);
    const z = zs[l], dz = new Float64Array(B * W);
    for (let k = 0; k < B * W; k++) dz[k] = d[k] * slope(c.act, z[k]);
    normaliseBack(dz, z, invs[l], B, W, c.norm);
    const w = Ws[l], dPrev = new Float64Array(B * W);
    for (let n = 0; n < B; n++) for (let j = 0; j < W; j++) {
      const g = dz[n * W + j];
      if (g === 0) continue;
      for (let i = 0; i < W; i++) dPrev[n * W + i] += w[j * W + i] * g;
    }
    if (c.residual && l > 0) for (let k = 0; k < B * W; k++) dPrev[k] += d[k];
    d = dPrev;
  }
  const layers: LayerStats[] = outs.map((out, l) => {
    const z = zs[l];
    let deadUnits = 0, sat = 0;
    for (let j = 0; j < W; j++) {
      let alive = false;
      for (let n = 0; n < B; n++) { const s = Math.abs(slope(c.act, z[n * W + j])); if (s >= 0.01) alive = true; else sat++; }
      if (!alive) deadUnits++;
    }
    return { rms: rmsOf(out), gradRms: gradRms[l], dead: deadUnits / W, saturated: sat / (B * W), values: out };
  });
  return { layers, inputRms };
}

export function rmsOf(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s / v.length);
}

/** Histogram of v over [lo, hi] in `bins` bins; values outside go into the end bins. */
export function histogram(v: ArrayLike<number>, lo: number, hi: number, bins: number): number[] {
  const h = new Array<number>(bins).fill(0);
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (!Number.isFinite(x)) { h[x > 0 ? bins - 1 : 0]++; continue; }
    const k = Math.floor((x - lo) / (hi - lo) * bins);
    h[Math.max(0, Math.min(bins - 1, k))]++;
  }
  return h;
}
