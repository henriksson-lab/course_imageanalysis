// Model of the contrastive demo (no DOM; tested in demos/test/contrastive-test.ts).
//
// A toy of contrastive learning (unit contrastive/). 64 synthetic "cells", 16 × 16 pixels:
// three shapes (round, elongated, a dividing pair) at any orientation, each with a staining
// brightness between 0.35 and 1 (dim below 0.65). The labels (shape, dim/bright) are never used
// for training; they only colour the plot and score the embedding.
// Each step: two augmented views of every cell (rotation and shift, brightness gain, noise,
// each optional, with a common strength), a tiny encoder (256 → 16 ReLU → 2, then normalised
// to length 1: a point on the unit circle), and the InfoNCE loss with temperature τ (both
// directions: view 1 must pick its partner among all views 2, and back). Without negatives the
// loss is only "pull the two views together": −mean cos(z, z'). Gradients by hand, Adam.
import { rng } from "../lib/ui";

export const S = 16, D = S * S, H = 16, N = 64;
export const SHAPES = ["round", "elongated", "pair"];

export interface Cell { img: Float64Array; shape: number; bright: number }

type R = ReturnType<typeof rng>;

/** Render one cell: shape at angle `ang`, brightness `b`, centre offset (ox, oy). */
function render(shape: number, ang: number, b: number, ox: number, oy: number): Float64Array {
  const img = new Float64Array(D), c = Math.cos(ang), s = Math.sin(ang);
  const soft = (d: number) => 1 / (1 + Math.exp(d / 0.35));      // d < 0 inside
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const px = x + 0.5 - S / 2 - ox, py = y + 0.5 - S / 2 - oy;
    const u = c * px + s * py, v = -s * px + c * py;
    let m: number;
    if (shape === 0) m = soft(Math.hypot(u, v) - 3.6);
    else if (shape === 1) m = soft((Math.hypot(u / 5.2, v / 2.1) - 1) * 2.1);
    else m = Math.max(soft(Math.hypot(u - 3.1, v) - 2.4), soft(Math.hypot(u + 3.1, v) - 2.4));
    img[y * S + x] = b * m;
  }
  return img;
}

/** The data set: N cells, seeded. */
export function makeCells(seed = 1): Cell[] {
  const r = rng(seed), out: Cell[] = [];
  for (let i = 0; i < N; i++) {
    const shape = i % 3, bright = 0.35 + 0.65 * r.unif();
    out.push({ img: render(shape, Math.PI * r.unif(), bright, r.unif() - 0.5, r.unif() - 0.5), shape, bright });
  }
  return out;
}

export interface Aug { strength: number; geom: boolean; bright: boolean; noise: boolean }

/** One augmented view: rotate about the centre and shift (bilinear, edge = background), gain, noise. */
export function augment(img: ArrayLike<number>, a: Aug, r: R): Float64Array {
  const t = a.strength;
  const ang = a.geom ? Math.PI * t * (2 * r.unif() - 1) : 0;
  const dx = a.geom ? 2 * t * (2 * r.unif() - 1) : 0, dy = a.geom ? 2 * t * (2 * r.unif() - 1) : 0;
  const gain = a.bright ? Math.exp(1.2 * t * (2 * r.unif() - 1)) : 1;
  const sd = a.noise ? 0.05 * t : 0;
  const c = Math.cos(ang), s = Math.sin(ang), out = new Float64Array(D);
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= S || y >= S ? 0 : img[y * S + x]);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const px = x + 0.5 - S / 2 - dx, py = y + 0.5 - S / 2 - dy;
    const sx = c * px - s * py + S / 2 - 0.5, sy = s * px + c * py + S / 2 - 0.5;
    const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
    const v = (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
    out[y * S + x] = gain * v + (sd > 0 ? sd * r.norm() : 0);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// The encoder: u = W2 relu(W1 x + b1) + b2, z = u / |u|.

export interface Net { W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: Float64Array }
export function newNet(seed = 1): Net {
  const r = rng(seed);
  return {
    W1: Float64Array.from({ length: H * D }, () => r.norm() * Math.sqrt(2 / D)), b1: new Float64Array(H),
    W2: Float64Array.from({ length: 2 * H }, () => r.norm() * Math.sqrt(1 / H)), b2: new Float64Array(2),
  };
}
interface Fwd { x: ArrayLike<number>; h: Float64Array; u: [number, number]; z: [number, number] }
function forward(n: Net, x: ArrayLike<number>): Fwd {
  const h = new Float64Array(H);
  for (let j = 0; j < H; j++) { let s = n.b1[j]; for (let i = 0; i < D; i++) s += n.W1[j * D + i] * x[i]; h[j] = Math.max(0, s); }
  const u: [number, number] = [n.b2[0], n.b2[1]];
  for (let k = 0; k < 2; k++) for (let j = 0; j < H; j++) u[k] += n.W2[k * H + j] * h[j];
  const L = Math.hypot(u[0], u[1]) || 1e-12;
  return { x, h, u, z: [u[0] / L, u[1] / L] };
}
export const embed = (n: Net, x: ArrayLike<number>): [number, number] => forward(n, x).z;

/** InfoNCE (both directions) on unit vectors a[i], b[i]; returns the loss and dL/da, dL/db. */
export function infoNce(a: [number, number][], b: [number, number][], tau: number): { loss: number; ga: [number, number][]; gb: [number, number][] } {
  const n = a.length, Sm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Sm[i * n + j] = (a[i][0] * b[j][0] + a[i][1] * b[j][1]) / tau;
  const G = new Float64Array(n * n);
  let loss = 0;
  for (let pass = 0; pass < 2; pass++) for (let i = 0; i < n; i++) {
    // pass 0: row i (view a_i picks among all b); pass 1: column i (b_i picks among all a)
    const at = (j: number) => (pass === 0 ? i * n + j : j * n + i);
    let m = -Infinity; for (let j = 0; j < n; j++) m = Math.max(m, Sm[at(j)]);
    let s = 0; for (let j = 0; j < n; j++) s += Math.exp(Sm[at(j)] - m);
    loss += (m + Math.log(s) - Sm[at(i)]) / (2 * n);
    for (let j = 0; j < n; j++) G[at(j)] += (Math.exp(Sm[at(j)] - m) / s - (j === i ? 1 : 0)) / (2 * n);
  }
  const ga = a.map(() => [0, 0] as [number, number]), gb = b.map(() => [0, 0] as [number, number]);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const g = G[i * n + j] / tau;
    ga[i][0] += g * b[j][0]; ga[i][1] += g * b[j][1];
    gb[j][0] += g * a[i][0]; gb[j][1] += g * a[i][1];
  }
  return { loss, ga, gb };
}

/** Only pulling the pairs together: −mean a_i · b_i. */
export function alignOnly(a: [number, number][], b: [number, number][]): { loss: number; ga: [number, number][]; gb: [number, number][] } {
  const n = a.length;
  let loss = 0;
  for (let i = 0; i < n; i++) loss -= (a[i][0] * b[i][0] + a[i][1] * b[i][1]) / n;
  return { loss, ga: b.map(v => [-v[0] / n, -v[1] / n] as [number, number]), gb: a.map(v => [-v[0] / n, -v[1] / n] as [number, number]) };
}

/** Backpropagate dL/dz of one forward pass into the gradient accumulators. */
function backward(n: Net, f: Fwd, gz: [number, number], g: Net): void {
  const L = Math.hypot(f.u[0], f.u[1]) || 1e-12, dot = f.z[0] * gz[0] + f.z[1] * gz[1];
  const gu = [(gz[0] - f.z[0] * dot) / L, (gz[1] - f.z[1] * dot) / L];
  const gh = new Float64Array(H);
  for (let k = 0; k < 2; k++) {
    g.b2[k] += gu[k];
    for (let j = 0; j < H; j++) { g.W2[k * H + j] += gu[k] * f.h[j]; gh[j] += gu[k] * n.W2[k * H + j]; }
  }
  for (let j = 0; j < H; j++) {
    if (f.h[j] <= 0) continue;
    g.b1[j] += gh[j];
    for (let i = 0; i < D; i++) g.W1[j * D + i] += gh[j] * f.x[i];
  }
}

export interface Settings { tau: number; aug: Aug; negatives: boolean }

export class Trainer {
  net: Net; step = 0; losses: number[] = [];
  private r: R; private m: Net; private v: Net;
  views: { a: [number, number][]; b: [number, number][] } = { a: [], b: [] };
  constructor(public cells: Cell[], seed = 1) {
    this.net = newNet(seed); this.r = rng(1000 + seed);
    const zero = (): Net => ({ W1: new Float64Array(H * D), b1: new Float64Array(H), W2: new Float64Array(2 * H), b2: new Float64Array(2) });
    this.m = zero(); this.v = zero();
  }
  /** One step on all cells; returns the loss before the step. */
  train(s: Settings, lr = 0.01): number {
    const fa = this.cells.map(c => forward(this.net, augment(c.img, s.aug, this.r)));
    const fb = this.cells.map(c => forward(this.net, augment(c.img, s.aug, this.r)));
    const a = fa.map(f => f.z), b = fb.map(f => f.z);
    this.views = { a, b };
    const L = s.negatives ? infoNce(a, b, s.tau) : alignOnly(a, b);
    const g: Net = { W1: new Float64Array(H * D), b1: new Float64Array(H), W2: new Float64Array(2 * H), b2: new Float64Array(2) };
    fa.forEach((f, i) => backward(this.net, f, L.ga[i], g));
    fb.forEach((f, i) => backward(this.net, f, L.gb[i], g));
    this.step++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.step, c2 = 1 - b2 ** this.step;
    for (const k of ["W1", "b1", "W2", "b2"] as const) {
      const p = this.net[k], gr = g[k], m = this.m[k], v = this.v[k];
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * gr[i]; v[i] = b2 * v[i] + (1 - b2) * gr[i] * gr[i];
        p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
      }
    }
    this.losses.push(L.loss);
    return L.loss;
  }
}

/** Scores of an embedding of the clean cells: leave-one-out 5-NN accuracy for shape and for
 *  dim/bright, and the spread (mean length of the mean vector: 1 = all in one point). */
export function scores(net: Net, cells: Cell[]): { shape: number; bright: number; collapse: number; z: [number, number][] } {
  const z = cells.map(c => embed(net, c.img));
  const knn = (lab: (c: Cell) => number, K: number) => {
    let ok = 0;
    for (let i = 0; i < z.length; i++) {
      const d = z.map((q, j) => [j === i ? Infinity : (q[0] - z[i][0]) ** 2 + (q[1] - z[i][1]) ** 2, j]).sort((p, q) => p[0] - q[0]).slice(0, 5);
      const votes = new Array(K).fill(0);
      d.forEach(([, j]) => votes[lab(cells[j])]++);
      if (votes.indexOf(Math.max(...votes)) === lab(cells[i])) ok++;
    }
    return ok / z.length;
  };
  let mx = 0, my = 0; for (const q of z) { mx += q[0] / z.length; my += q[1] / z.length; }
  return { shape: knn(c => c.shape, 3), bright: knn(c => (c.bright < 0.675 ? 0 : 1), 2), collapse: Math.hypot(mx, my), z };
}
