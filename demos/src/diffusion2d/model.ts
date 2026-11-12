// Model of the diffusion2d demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// 2D data sets that are Gaussian mixtures (lib/mixture.ts), so the score of the noised data
// is known exactly at every noise level. Forward process x_τ = √ā(τ) x_0 + √(1 − ā(τ)) ε.
// Reverse: start from N(0, I) and take N steps from τ = 1 to 0 with the predicted noise
// ε̂ = −√(1 − ā) · score(x, τ): DDPM (stochastic, ancestral) or DDIM (deterministic).
// Score sources: exact (analytic), learned (a tiny network from train_score.py, two
// snapshots), conditional on one group of components, with classifier-free guidance
// score = (1 + w) score_cond − w score_uncond.
import { makeDiag, noisedMixture, restrictMixture, scoreDiag, sampleDiag, alphaBarAt, responsibilities, type DiagMixture, type Schedule } from "../lib/mixture";
import { rng } from "../lib/ui";
import { deq, type Q8Layer } from "../lib/q8";

export type DataId = "two" | "ring" | "rare" | "moons" | "twenty";

export interface DataSet { id: DataId; m: DiagMixture; groups: number[][]; groupNames: string[] }

function moonsMeans(): number[][] {
  const out: number[][] = [];
  for (let k = 0; k < 12; k++) { const t = Math.PI * k / 11; out.push([1.2 * Math.cos(t) - 0.6, 1.2 * Math.sin(t) - 0.3]); }
  for (let k = 0; k < 12; k++) { const t = Math.PI * k / 11; out.push([0.6 - 1.2 * Math.cos(t), 0.3 - 1.2 * Math.sin(t)]); }
  return out;
}

export function dataSet(id: DataId): DataSet {
  if (id === "two") return { id, m: makeDiag([1, 1], [[-1.2, 0.6], [1.2, -0.6]], 0.04), groups: [[0], [1]], groupNames: ["left blob", "right blob"] };
  if (id === "ring") {
    const means = Array.from({ length: 8 }, (_, k) => [1.6 * Math.cos(2 * Math.PI * k / 8), 1.6 * Math.sin(2 * Math.PI * k / 8)]);
    return { id, m: makeDiag(new Array(8).fill(1), means, 0.012), groups: [[0], [0, 1, 2, 3]], groupNames: ["one blob", "upper half"] };
  }
  if (id === "rare") return { id, m: makeDiag([0.95, 0.05], [[-0.7, -0.2], [1.6, 1.0]], [0.25, 0.008]), groups: [[0], [1]], groupNames: ["big blob", "rare blob"] };
  const mm = moonsMeans();
  if (id === "moons") return { id, m: makeDiag(new Array(24).fill(1), mm, 0.006), groups: [Array.from({ length: 12 }, (_, k) => k), Array.from({ length: 12 }, (_, k) => 12 + k)], groupNames: ["upper moon", "lower moon"] };
  // twenty training points drawn from the moons; the "perfect" score of this finite set: tiny width
  const s = sampleDiag(makeDiag(new Array(24).fill(1), mm, 0.006), 20, rng(20));
  const pts = Array.from({ length: 20 }, (_, i) => [s.X[2 * i], s.X[2 * i + 1]]);
  return { id, m: makeDiag(new Array(20).fill(1), pts, 1e-4), groups: [Array.from({ length: 10 }, (_, k) => k), Array.from({ length: 10 }, (_, k) => 10 + k)], groupNames: ["first ten", "last ten"] };
}

export type Sampler = "ddpm" | "ddim";

/** A score function s(x, τ) for 2D points. */
export type ScoreFn = (x: number, y: number, tau: number, ab: number, out: Float64Array) => void;

/** Exact score of the noised mixture (optionally conditioned on a group, with guidance weight w). */
export function exactScore(ds: DataSet, cond: number | null, w: number): ScoreFn {
  const cache = new Map<number, { u: DiagMixture; c: DiagMixture | null }>();
  const cm = cond === null ? null : restrictMixture(ds.m, ds.groups[cond]);
  const su = new Float64Array(2), sc = new Float64Array(2), ru = new Float64Array(ds.m.k), rc = new Float64Array(cm ? cm.k : 1), xv = new Float64Array(2);
  return (x, y, _tau, ab, out) => {
    let e = cache.get(ab);
    if (!e) { e = { u: noisedMixture(ds.m, ab), c: cm ? noisedMixture(cm, ab) : null }; if (cache.size > 2000) cache.clear(); cache.set(ab, e); }
    xv[0] = x; xv[1] = y;
    scoreDiag(e.u, xv, su, ru);
    if (!e.c) { out[0] = su[0]; out[1] = su[1]; return; }
    scoreDiag(e.c, xv, sc, rc);
    out[0] = (1 + w) * sc[0] - w * su[0]; out[1] = (1 + w) * sc[1] - w * su[1];
  };
}

// ---------------------------------------------------------------------------------------
// The learned score: an MLP (x, y, τ, √ā, √(1 − ā)) → ε̂ (2), tanh hidden layers.

export interface MlpQ8 { layers: Q8Layer[] }

export function learnedScore(net: MlpQ8): ScoreFn {
  const L = net.layers.map(deq);
  const buf = [new Float64Array(64), new Float64Array(64)];
  return (x, y, tau, ab, out) => {
    let a: ArrayLike<number> = [x, y, tau, Math.sqrt(ab), Math.sqrt(1 - ab)];
    L.forEach((l, k) => {
      const o = k === L.length - 1 ? out : buf[k % 2];
      for (let j = 0; j < l.nOut; j++) {
        let s = l.b[j];
        for (let i = 0; i < l.nIn; i++) s += l.W[j * l.nIn + i] * a[i];
        o[j] = k === L.length - 1 ? s : Math.tanh(s);
      }
      a = o;
    });
    // the network predicts the noise ε̂; score = −ε̂ / √(1 − ā)
    const k = -1 / Math.sqrt(Math.max(1e-6, 1 - ab));
    out[0] *= k; out[1] *= k;
  };
}

// ---------------------------------------------------------------------------------------
// Sampling

export interface Run {
  N: number;
  /** Positions after each step: frames[i] = points at step i (i = 0: pure noise at τ = 1). */
  frames: Float64Array[];
  taus: number[];
}

/**
 * Reverse process with N steps from τ = 1 to 0. Points start from seeded N(0, I).
 * DDIM (η = 0): x_{i−1} = √ā_{i−1} x̂_0 + √(1 − ā_{i−1}) ε̂. DDPM: the Gaussian posterior
 * q(x_{i−1} | x_i, x̂_0) with mean (√ā_{i−1} β_i x̂_0 + √α_i (1 − ā_{i−1}) x_i)/(1 − ā_i) and
 * variance β_i (1 − ā_{i−1})/(1 − ā_i), where α_i = ā_i/ā_{i−1}, β_i = 1 − α_i.
 */
export function sample(score: ScoreFn, n: number, N: number, sampler: Sampler, sched: Schedule, seed: number): Run {
  const r = rng(seed), noise = rng(seed + 1000);
  let X = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) X[i] = r.norm();
  const frames = [X.slice()], taus = [1];
  const s = new Float64Array(2);
  for (let i = N; i >= 1; i--) {
    const t = i / N, tp = (i - 1) / N, ab = alphaBarAt(t, sched), abp = i === 1 ? 1 : alphaBarAt(tp, sched);
    const Y = new Float64Array(2 * n);
    const al = ab / abp, be = 1 - al;
    for (let p = 0; p < n; p++) {
      const x = X[2 * p], y = X[2 * p + 1];
      score(x, y, t, ab, s);
      const ex = -Math.sqrt(1 - ab) * s[0], ey = -Math.sqrt(1 - ab) * s[1];
      const x0 = (x - Math.sqrt(1 - ab) * ex) / Math.sqrt(ab), y0 = (y - Math.sqrt(1 - ab) * ey) / Math.sqrt(ab);
      if (sampler === "ddim") {
        Y[2 * p] = Math.sqrt(abp) * x0 + Math.sqrt(1 - abp) * ex;
        Y[2 * p + 1] = Math.sqrt(abp) * y0 + Math.sqrt(1 - abp) * ey;
      } else {
        const c0 = Math.sqrt(abp) * be / (1 - ab), ct = Math.sqrt(al) * (1 - abp) / (1 - ab);
        const sd = i > 1 ? Math.sqrt(be * (1 - abp) / (1 - ab)) : 0;
        Y[2 * p] = c0 * x0 + ct * x + sd * noise.norm();
        Y[2 * p + 1] = c0 * y0 + ct * y + sd * noise.norm();
      }
    }
    X = Y; frames.push(X.slice()); taus.push(tp);
  }
  return { N, frames, taus };
}

/** Fixed data points and noise for the forward tab: x_τ = √ā x_0 + √(1 − ā) ε. */
export function forwardPoints(ds: DataSet, n: number, tau: number, sched: Schedule, seed = 4): Float64Array {
  const s = sampleDiag(ds.m, n, rng(seed)), r = rng(seed + 1), ab = alphaBarAt(tau, sched), out = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) out[i] = Math.sqrt(ab) * s.X[i] + Math.sqrt(1 - ab) * r.norm();
  return out;
}

/** Where samples ended: fraction per group (by responsibility of the clean mixture), fraction off the data (more than 3 sd from every component). */
export function summarise(ds: DataSet, X: Float64Array): { perGroup: number[]; off: number; nearTrain: number } {
  const n = X.length / 2, per = new Array(ds.groups.length).fill(0), r = new Float64Array(ds.m.k);
  let off = 0, near = 0;
  for (let p = 0; p < n; p++) {
    const x = [X[2 * p], X[2 * p + 1]];
    let minZ = Infinity, minD = Infinity;
    for (let j = 0; j < ds.m.k; j++) {
      const dx = x[0] - ds.m.mu[2 * j], dy = x[1] - ds.m.mu[2 * j + 1];
      minZ = Math.min(minZ, Math.hypot(dx, dy) / Math.sqrt(ds.m.v[2 * j]));
      minD = Math.min(minD, Math.hypot(dx, dy));
    }
    if (minZ > 3) off++;
    if (minD < 0.05) near++;
    responsibilities(ds.m, x, r);
    ds.groups.forEach((g, k) => { let s = 0; for (const j of g) s += r[j]; per[k] += s; });
  }
  return { perGroup: per.map(v => v / n), off: off / n, nearTrain: near / n };
}
