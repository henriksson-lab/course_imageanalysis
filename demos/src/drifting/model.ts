// The algorithm of the drifting demo (diffusion-models day, unit drifting/), no DOM, so the
// tests and the figure scripts run it in node.
//
// Drifting (Deng et al. 2026, arXiv 2602.04770; Cazenavette et al. 2026, arXiv 2611.16642):
// every generated point x is moved along the field
//   V(x) = V⁺(x) − V⁻(x),
//   V⁺(x) = Σ_j a_j (y_j − x),  a = softmax_j(−|x − y_j| / τ)   over the real points y_j,
//   V⁻(x) = Σ_k b_k (x_k − x),  b = softmax_k(−|x − x_k| / τ)   over the OTHER generated points,
// which is "the kernel-weighted mean of the nearby real points minus the kernel-weighted mean of
// the nearby generated points" (the −x terms cancel because both weights sum to 1).
// Free points ("dataset distillation"): x ← x + η V(x), optionally with augmented views (jittered
// copies of every point; the views of the other points, and the other views of the same point,
// are negatives). Generator: a small MLP f(ε) trained with the loss mean |f(ε) − stopgrad(f(ε) + V)|²,
// i.e. a regression on the fixed targets f(ε) + V, whose gradient with respect to f(ε) is −2V.
import { rng } from "../lib/ui";
import { mlp, forward, gradients, applyGradients, Adam, type Mlp } from "../lib/nn";

export type Rng = ReturnType<typeof rng>;
/** Points are flat arrays [x0, y0, x1, y1, ...]. */
export type Pts = Float64Array;
export const npts = (P: Pts) => P.length / 2;
const dist = (dx: number, dy: number) => Math.sqrt(dx * dx + dy * dy);

// ------------------------------------------------------------------------------- target data

export type DataId = "moons" | "ring" | "blobs";
export const DATA_IDS: DataId[] = ["moons", "ring", "blobs"];
export const DATA_LABEL: Record<DataId, string> = { moons: "two moons", ring: "ring", blobs: "two blobs (3 : 1)" };

/** The real points: about 2 units across, centred near 0. */
export function makeData(id: DataId, n = 400, seed = 1): Pts {
  const r = rng(seed), P = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    if (id === "moons") {
      const c = i % 2, t = Math.PI * r.unif();
      x = c === 0 ? Math.cos(t) : 1 - Math.cos(t);
      y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
      x = x - 0.5 + 0.08 * r.norm(); y = y - 0.25 + 0.08 * r.norm();
    } else if (id === "ring") {
      const t = 2 * Math.PI * r.unif(), q = 1.2 + 0.07 * r.norm();
      x = q * Math.cos(t); y = q * Math.sin(t);
    } else {
      const big = i % 4 !== 3;            // 3 of 4 points in the left blob
      x = (big ? -1 : 1.1) + 0.3 * r.norm(); y = (big ? 0.2 : -0.4) + 0.3 * r.norm();
    }
    P[2 * i] = x; P[2 * i + 1] = y;
  }
  return P;
}

export type InitId = "noise" | "far" | "collapsed" | "data";
export const INIT_IDS: InitId[] = ["noise", "far", "collapsed", "data"];
export const INIT_LABEL: Record<InitId, string> = { noise: "noise", far: "far away", collapsed: "on one spot", data: "a new draw of the data" };

/** Starting positions of n free points ("data": a fresh draw of the data set `data`). */
export function initPoints(id: InitId, n: number, seed = 2, data: DataId = "moons"): Pts {
  if (id === "data") return makeData(data, n, seed + 1000);
  const r = rng(seed), P = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    if (id === "noise") { P[2 * i] = r.norm(); P[2 * i + 1] = r.norm(); }
    else if (id === "far") { P[2 * i] = 2.2 + 0.25 * r.norm(); P[2 * i + 1] = 2.2 + 0.25 * r.norm(); }
    else { P[2 * i] = -1 + 0.1 * r.norm(); P[2 * i + 1] = 0.2 + 0.1 * r.norm(); }
  }
  return P;
}

// ------------------------------------------------------------------------------- the field

/**
 * Softmax-weighted mean of the points of S as seen from (x, y): weights exp(−|p − s|/τ),
 * normalised; points whose index `skip(k)` is true are left out. Returns [mx, my] (or the
 * point itself when nothing is left).
 */
function kernelMean(x: number, y: number, S: Pts, tau: number, skip?: (k: number) => boolean): [number, number] {
  const n = npts(S);
  // subtract the largest logit (smallest distance) for numerical stability
  let dmin = Infinity;
  for (let k = 0; k < n; k++) { if (skip && skip(k)) continue; const d = dist(S[2 * k] - x, S[2 * k + 1] - y); if (d < dmin) dmin = d; }
  if (dmin === Infinity) return [x, y];
  let w = 0, mx = 0, my = 0;
  for (let k = 0; k < n; k++) {
    if (skip && skip(k)) continue;
    const e = Math.exp(-(dist(S[2 * k] - x, S[2 * k + 1] - y) - dmin) / tau);
    w += e; mx += e * S[2 * k]; my += e * S[2 * k + 1];
  }
  return [mx / w, my / w];
}

/** V⁺ at (x, y): the pull towards the real points. */
export function vPlus(x: number, y: number, real: Pts, tau: number): [number, number] {
  const [mx, my] = kernelMean(x, y, real, tau);
  return [mx - x, my - y];
}

/**
 * The field at every generated point: V = V⁺ − repel · V⁻ (repel = 1 is drifting; 0 is
 * attraction only, i.e. mean shift towards the real points). `gen` are the generated points
 * (or views); a point is never its own negative. Returns { v, plus, minus } as flat arrays.
 */
export function field(gen: Pts, real: Pts, tau: number, repel = 1): { v: Pts; plus: Pts; minus: Pts } {
  const n = npts(gen), v = new Float64Array(2 * n), plus = new Float64Array(2 * n), minus = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const x = gen[2 * i], y = gen[2 * i + 1];
    const [px, py] = vPlus(x, y, real, tau);
    const [nx, ny] = kernelMean(x, y, gen, tau, k => k === i);
    plus[2 * i] = px; plus[2 * i + 1] = py;
    minus[2 * i] = nx - x; minus[2 * i + 1] = ny - y;
    v[2 * i] = px - repel * (nx - x); v[2 * i + 1] = py - repel * (ny - y);
  }
  return { v, plus, minus };
}

/** Mean length of the field vectors. */
export function meanNorm(v: Pts): number {
  let s = 0; const n = npts(v);
  for (let i = 0; i < n; i++) s += Math.hypot(v[2 * i], v[2 * i + 1]);
  return s / n;
}

// ------------------------------------------------------------------------------- coverage

/** Distance from every point of P to its k-th nearest other point of P. */
export function knnRadius(P: Pts, k: number): Float64Array {
  const n = npts(P), out = new Float64Array(n), buf = new Float64Array(n - 1);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = 0; j < n; j++) if (j !== i) buf[m++] = dist(P[2 * i] - P[2 * j], P[2 * i + 1] - P[2 * j + 1]);
    buf.sort();
    out[i] = buf[Math.min(k, n - 1) - 1];
  }
  return out;
}

/**
 * Coverage (Cazenavette et al.): the fraction of real points whose k-NN ball (radius = distance
 * to the k-th nearest other real point) contains at least one generated point. `radius` may be
 * passed precomputed (knnRadius(real, k)).
 */
export function coverage(real: Pts, gen: Pts, k = 5, radius?: Float64Array): number {
  const rad = radius ?? knnRadius(real, k), n = npts(real), m = npts(gen);
  let c = 0;
  for (let j = 0; j < n; j++) {
    const r2 = rad[j] * rad[j];
    for (let i = 0; i < m; i++) {
      const dx = gen[2 * i] - real[2 * j], dy = gen[2 * i + 1] - real[2 * j + 1];
      if (dx * dx + dy * dy <= r2) { c++; break; }
    }
  }
  return c / n;
}

// ------------------------------------------------------------------------------- free points

export interface DriftOptions {
  tau: number;
  /** Step: x ← x + eta · V. */
  eta: number;
  /** Augmented views per point (1 = no augmentation) and their jitter (standard deviation). */
  views: number;
  jitter: number;
  /** Weight of the push away from the other generated points (1 = drifting, 0 = attraction only). */
  repel?: number;
}

/** A run of free points drifting towards the real points ("dataset distillation"). */
export class Drift {
  step = 0;
  /** History of mean |V| and coverage, one entry per step (index 0 = start). */
  normHist: number[] = [];
  covHist: number[] = [];
  lastV: Pts;
  /** The views used for the last field (drawn once per step; showing them draws no random numbers). */
  lastViews: Pts = new Float64Array(0);
  private r: Rng;
  private rad: Float64Array;
  constructor(public real: Pts, public x: Pts, public o: DriftOptions, seed = 5, public k = 5) {
    this.r = rng(seed);
    this.rad = knnRadius(real, k);
    this.lastV = this.fieldNow();
    this.record();
  }
  /** The views of every point: point i owns views i*A … i*A + A − 1 (view 0 = the point itself if A = 1). */
  views(): Pts {
    const n = npts(this.x), A = Math.max(1, this.o.views), out = new Float64Array(2 * n * A);
    for (let i = 0; i < n; i++) for (let a = 0; a < A; a++) {
      const j = i * A + a, s = A === 1 ? 0 : this.o.jitter;
      out[2 * j] = this.x[2 * i] + s * this.r.norm(); out[2 * j + 1] = this.x[2 * i + 1] + s * this.r.norm();
    }
    return out;
  }
  /** The field per point: the mean of the field at its views. */
  fieldNow(): Pts {
    const n = npts(this.x), A = Math.max(1, this.o.views), W = this.views(), f = field(W, this.real, this.o.tau, this.o.repel ?? 1).v, v = new Float64Array(2 * n);
    this.lastViews = W;
    for (let i = 0; i < n; i++) for (let a = 0; a < A; a++) { v[2 * i] += f[2 * (i * A + a)] / A; v[2 * i + 1] += f[2 * (i * A + a) + 1] / A; }
    return v;
  }
  private record(): void {
    this.normHist.push(meanNorm(this.lastV));
    this.covHist.push(coverage(this.real, this.x, this.k, this.rad));
  }
  /** One step: move every point by eta · V. */
  advance(): void {
    const v = this.lastV;
    for (let i = 0; i < this.x.length; i++) this.x[i] += this.o.eta * v[i];
    this.step++;
    this.lastV = this.fieldNow();
    this.record();
  }
}

// ------------------------------------------------------------------------------- a generator

export interface GenOptions {
  tau: number;
  /** Generated points per step (the negatives) and real points per step (the positives). */
  batch: number;
  lr: number;
  hidden: number;
  repel?: number;
}

/** A small MLP generator 2 → H → H → 2 (tanh), trained with the drifting loss. */
export class Generator {
  net: Mlp;
  opt: Adam;
  step = 0;
  lossHist: number[] = [];
  covHist: number[] = [];
  private r: Rng;
  private rad: Float64Array;
  /** Fixed noise for showing and judging the samples (one forward pass each). */
  showNoise: number[][];
  constructor(public real: Pts, public o: GenOptions, seed = 3, public k = 5) {
    this.net = mlp([2, o.hidden, o.hidden, 2], ["tanh", "tanh", "id"], seed);
    this.opt = new Adam(this.net, o.lr);
    this.r = rng(seed + 100);
    this.rad = knnRadius(real, k);
    this.showNoise = Array.from({ length: 400 }, () => [this.r.norm(), this.r.norm()]);
  }
  /** One forward pass per noise vector: the samples. */
  sample(noise: number[][] = this.showNoise): Pts {
    const out = new Float64Array(2 * noise.length);
    noise.forEach((e, i) => { const f = forward(this.net, e).a; const y = f[f.length - 1]; out[2 * i] = y[0]; out[2 * i + 1] = y[1]; });
    return out;
  }
  /** One training step; returns the loss (= mean |V|² / 2, the value of the stop-gradient loss averaged over the 2 outputs). */
  train(): number {
    const B = this.o.batch, eps = Array.from({ length: B }, () => [this.r.norm(), this.r.norm()]);
    const gen = this.sample(eps);
    // positives: a random batch of real points
    const n = npts(this.real), pos = new Float64Array(2 * B);
    for (let i = 0; i < B; i++) { const j = Math.floor(this.r.unif() * n); pos[2 * i] = this.real[2 * j]; pos[2 * i + 1] = this.real[2 * j + 1]; }
    const V = field(gen, pos, this.o.tau, this.o.repel ?? 1).v;
    // stop-gradient: the targets f(ε) + V are fixed numbers, so this is a plain regression
    const targets = eps.map((_, i) => [gen[2 * i] + V[2 * i], gen[2 * i + 1] + V[2 * i + 1]]);
    const g = gradients(this.net, eps, targets, { loss: "mse" });
    applyGradients(this.net, g, this.opt);
    this.step++;
    this.lossHist.push(g.loss);
    return g.loss;
  }
  /** Coverage of the real points by the 400 shown samples. */
  coverageNow(): number { return coverage(this.real, this.sample(), this.k, this.rad); }
}

// ------------------------------------------------------------------------------- the demo's settings

export interface Config {
  mode: "points" | "generator";
  data: DataId;
  init: InitId;
  /** Free points. */
  n: number;
  tau: number;
  views: number;
  jitter: number;
  /** 1 = drifting, 0 = attraction only. */
  repel: number;
}
export const ETA = 0.5;            // step of the free points
export const POINT_STEPS = 300;    // "run" stops here for free points
export const GEN_STEPS = 1500;     // and here for the generator
export const GEN = { batch: 128, lr: 0.003, hidden: 32 };
export const BASE: Config = { mode: "points", data: "moons", init: "noise", n: 200, tau: 0.2, views: 1, jitter: 0, repel: 1 };

export interface Preset { id: string; label: string; c: Partial<Config> }
export const PRESETS: Preset[] = [
  { id: "drift", label: "noise drifts to the moons", c: {} },
  { id: "match", label: "start on the data: V ≈ 0", c: { init: "data" } },
  { id: "far", label: "start far away", c: { init: "far" } },
  { id: "small", label: "τ small", c: { tau: 0.03 } },
  { id: "large", label: "τ large", c: { tau: 3 } },
  { id: "attract", label: "attraction only (mean shift)", c: { repel: 0 } },
  { id: "few", label: "20 points, no augmentation", c: { n: 20 } },
  { id: "views", label: "20 points, 8 views each", c: { n: 20, views: 8, jitter: 0.1 } },
  { id: "gen", label: "a generator, one pass", c: { mode: "generator", tau: 0.1 } },
  { id: "genattract", label: "generator, attraction only", c: { mode: "generator", tau: 0.1, repel: 0 } },
];
export const presetConfig = (id: string): Config => ({ ...BASE, ...(PRESETS.find(p => p.id === id) ?? PRESETS[0]).c });

/** A free-point run for a configuration. */
export const newDrift = (c: Config): Drift =>
  new Drift(makeData(c.data), initPoints(c.init, c.n, 2, c.data), { tau: c.tau, eta: ETA, views: c.views, jitter: c.jitter, repel: c.repel });
/** A generator for a configuration. */
export const newGenerator = (c: Config): Generator => new Generator(makeData(c.data), { ...GEN, tau: c.tau, repel: c.repel });
