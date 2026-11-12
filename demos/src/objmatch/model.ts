// Model of the objmatch demo (no DOM): a synthetic nuclei image with a ground-truth label
// image, a second annotator, an "error generator" that turns the truth into predictions
// (boundary jitter, outline bias, merges of touching pairs, splits, missed dim nuclei,
// specks, a confidence score per object), and the evaluation (IoU table, matching, F1 vs τ,
// pixel Dice, AP, splits and merges).
import { Ellipse, nuclei, discs, addGaussianNoise, ellipseDistance } from "../lib/synth";
import { Grey, gaussianBlur } from "../lib/imageops";
import { rng, gauss } from "../lib/random";
import { diceMasks } from "../lib/metrics";
import * as EV from "../lib/eval";

export const W = 256;

/** Part of a predicted object: an ellipse, or one half of it (cut across the major axis, with a gap). */
export interface Part { e: Ellipse; half?: 1 | -1; gap?: number }
export type Kind = "ok" | "merge" | "split" | "speck";
export interface PredObj { parts: Part[]; score: number; kind: Kind }

export interface Scene {
  seed: number;
  image: Grey;
  ellipses: Ellipse[];
  truth: { labels: Int32Array; count: number };
  annotatorB: { labels: Int32Array; count: number };
  dim: boolean[];
  pairs: [number, number][];        // touching pairs, 0-based
  rand: { miss: number; split: number; merge: number[]; n: number[][]; specks: { x: number; y: number; r: number; u: number }[] };
}

const DIM = 3;

export function makeScene(seed = 1): Scene {
  const nu = nuclei({ width: W, height: W, count: 16, touching: 0.4, radius: [9, 14], seed: 40 + seed });
  const ellipses = nu.ellipses;
  const n = ellipses.length, r = rng(1000 + seed);
  const pairs = nu.touchingPairs.map(([a, b]) => [a - 1, b - 1] as [number, number]);
  const inPair = new Set(pairs.flat());
  // dim nuclei: the first DIM objects that are not in a touching pair
  const dim = new Array<boolean>(n).fill(false);
  let c = 0;
  for (let k = 0; k < n && c < DIM; k++) if (!inPair.has(k)) { dim[k] = true; c++; }
  const d = discs(W, W, ellipses, { background: 45, values: ellipses.map((_, k) => (dim[k] ? 88 : 175 + 15 * r.norm())) });
  const image = addGaussianNoise(gaussianBlur(d.image, 1.2), 7, 77 + seed);
  // annotator B: same nuclei, outlines a little different, and the dimmest one left out
  const rb = rng(500 + seed);
  const disputed = dim.indexOf(true);
  const eb = ellipses.filter((_, k) => k !== disputed).map(e => ({ ...e, x: e.x + 0.5 * rb.norm(), y: e.y + 0.5 * rb.norm(), a: e.a + 0.5 * rb.norm(), b: e.b + 0.5 * rb.norm() }));
  const lb = rasterise(eb.map(e => ({ parts: [{ e }], score: 1, kind: "ok" as Kind })));
  // fixed random numbers of the error generator: sliders act monotonically
  const rr = rng(2000 + seed);
  const rand = {
    miss: 0, split: 0,
    merge: pairs.map(() => rr.unif()),
    n: ellipses.map(() => [rr.norm(), rr.norm(), rr.norm(), rr.norm(), rr.norm(), rr.unif(), rr.unif()]),
    specks: [] as { x: number; y: number; r: number; u: number }[],
  };
  for (let t = 0; rand.specks.length < 30 && t < 2000; t++) {
    const x = 6 + (W - 12) * rr.unif(), y = 6 + (W - 12) * rr.unif();
    if (nu.labels[Math.round(y) * W + Math.round(x)]) continue;
    if (ellipses.some(e => ellipseDistance(e, x, y) < 1.5)) continue;
    rand.specks.push({ x, y, r: 1.6 + 1.4 * rr.unif(), u: rr.unif() });
  }
  return { seed, image, ellipses, truth: { labels: nu.labels, count: n }, annotatorB: { labels: lb, count: eb.length }, dim, pairs, rand };
}

// ---------------------------------------------------------------------------------------
// Rasterising predictions

function inPart(p: Part, x: number, y: number): number {
  const d = ellipseDistance(p.e, x, y);
  if (d > 1) return Infinity;
  if (p.half) {
    const c = Math.cos(p.e.angle), s = Math.sin(p.e.angle);
    const u = c * (x - p.e.x) + s * (y - p.e.y);
    if (u * p.half < (p.gap ?? 0) / 2) return Infinity;
  }
  return d;
}

/** Label image of predicted objects (1..n in list order); overlaps go to the object the pixel is deepest in. */
export function rasterise(objs: PredObj[], w = W, h = W): Int32Array {
  const lab = new Int32Array(w * h), best = new Float64Array(w * h).fill(Infinity);
  objs.forEach((o, k) => {
    for (const p of o.parts) {
      const R = Math.ceil(Math.max(p.e.a, p.e.b)) + 1;
      for (let y = Math.max(0, Math.floor(p.e.y - R)); y <= Math.min(h - 1, Math.ceil(p.e.y + R)); y++)
        for (let x = Math.max(0, Math.floor(p.e.x - R)); x <= Math.min(w - 1, Math.ceil(p.e.x + R)); x++) {
          const d = inPart(p, x, y), i = y * w + x;
          if (d < best[i]) { best[i] = d; lab[i] = k + 1; }
        }
    }
  });
  return lab;
}

// ---------------------------------------------------------------------------------------
// Error generator

export interface Settings {
  jitter: number;    // boundary jitter (pixels)
  dilate: number;    // outline bias (pixels, + = too large)
  merge: number;     // probability that a touching pair is merged
  split: number;     // probability that a nucleus is split in two
  miss: number;      // probability that a dim nucleus is missed
  specks: number;    // number of spurious specks
}

export const DEFAULT_SETTINGS: Settings = { jitter: 1, dilate: 0, merge: 0.3, split: 0, miss: 0.5, specks: 2 };

const clamp01 = (v: number) => Math.min(0.99, Math.max(0.01, v));

/** Predicted objects for the scene under the given error settings (deterministic). */
export function generate(sc: Scene, s: Settings): PredObj[] {
  const n = sc.ellipses.length;
  const jit = (k: number): Ellipse => {
    const e = sc.ellipses[k], q = sc.rand.n[k];
    return {
      x: e.x + s.jitter * q[0] / Math.SQRT2, y: e.y + s.jitter * q[1] / Math.SQRT2,
      a: Math.max(1.5, e.a + s.dilate + 0.6 * s.jitter * q[2]), b: Math.max(1.5, e.b + s.dilate + 0.6 * s.jitter * q[3]), angle: e.angle,
    };
  };
  // merge groups (union-find over the touching pairs that merge)
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  sc.pairs.forEach(([a, b], i) => { if (sc.rand.merge[i] < s.merge) parent[find(a)] = find(b); });
  const groups = new Map<number, number[]>();
  for (let k = 0; k < n; k++) { const g = find(k); if (!groups.has(g)) groups.set(g, []); groups.get(g)!.push(k); }
  const out: PredObj[] = [];
  for (let k = 0; k < n; k++) {
    const g = groups.get(find(k))!;
    if (g[0] !== k) continue;
    const q = sc.rand.n[k];
    if (g.length > 1) { out.push({ parts: g.map(j => ({ e: jit(j) })), score: clamp01(0.72 + 0.05 * q[4]), kind: "merge" }); continue; }
    if (sc.dim[k] && q[5] < s.miss) continue;
    const base = sc.dim[k] ? 0.5 : 0.9;
    if (q[6] < s.split) {
      const e = jit(k);
      out.push({ parts: [{ e, half: 1, gap: 2 }], score: clamp01(0.6 + 0.05 * q[4]), kind: "split" });
      out.push({ parts: [{ e, half: -1, gap: 2 }], score: clamp01(0.55 + 0.05 * q[4]), kind: "split" });
      continue;
    }
    out.push({ parts: [{ e: jit(k) }], score: clamp01(base + 0.04 * q[4]), kind: "ok" });
  }
  for (let i = 0; i < Math.min(s.specks, sc.rand.specks.length); i++) {
    const p = sc.rand.specks[i];
    out.push({ parts: [{ e: { x: p.x, y: p.y, a: p.r, b: p.r * 0.8, angle: 0 } }], score: 0.1 + 0.25 * p.u, kind: "speck" });
  }
  return out;
}

/** Predictions as a label image with their scores. */
export function predict(sc: Scene, s: Settings): { objs: PredObj[]; labels: Int32Array; count: number; scores: Float64Array } {
  const objs = generate(sc, s);
  return { objs, labels: rasterise(objs), count: objs.length, scores: Float64Array.from(objs, o => o.score) };
}

/**
 * Hand-made predictions where greedy matching at τ = 0.3 loses a match: every nucleus
 * predicted well, except one touching pair (G1, G2): a large P1 between them overlaps G1
 * most and G2 enough, and a small P2 sits inside the far side of G1.
 */
export function greedyTrap(sc: Scene): PredObj[] {
  const [i1, i2] = sc.pairs[0];
  const out: PredObj[] = [];
  const r = rng(9);
  sc.ellipses.forEach((e, k) => { if (k !== i1 && k !== i2) out.push({ parts: [{ e: { ...e, x: e.x + 0.5 * r.norm(), y: e.y + 0.5 * r.norm() } }], score: 0.9, kind: "ok" }); });
  const g1 = sc.ellipses[i1], g2 = sc.ellipses[i2];
  const dx = g2.x - g1.x, dy = g2.y - g1.y, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
  const m1 = Math.sqrt(g1.a * g1.b), m2 = Math.sqrt(g2.a * g2.b);
  out.push({ parts: [{ e: { x: g1.x + 0.35 * dx, y: g1.y + 0.35 * dy, a: 0.5 * L + 0.8 * (m1 + m2) / 2, b: Math.min(m1, m2), angle: Math.atan2(dy, dx) } }], score: 0.8, kind: "ok" });
  out.push({ parts: [{ e: { x: g1.x - 0.5 * m1 * ux, y: g1.y - 0.5 * m1 * uy, a: 0.7 * m1, b: 0.7 * m1, angle: 0 } }], score: 0.6, kind: "ok" });
  return out;
}

// ---------------------------------------------------------------------------------------
// Evaluation

export const TAUS: number[] = Array.from({ length: 18 }, (_, i) => Math.round((0.1 + 0.05 * i) * 100) / 100);

export interface Evaluation {
  table: EV.IouTable;
  match: EV.Matching;
  scores: EV.ObjectScores;
  dice: number;
  f1: number[];                // F1 at TAUS
  pr: EV.PRCurve;              // PR curve at τ
  apMean: number;              // AP averaged over 0.5 … 0.95
  splits: { t: number; parts: number[] }[];
  merges: { p: number; parts: number[] }[];
}

export function evaluate(truth: Int32Array, nt: number, pred: Int32Array, np: number, scores: ArrayLike<number>, tau: number, method: EV.MatchMethod): Evaluation {
  const table = EV.iouTable(truth, pred, nt, np);
  const match = EV.match(table, tau, method);
  const sm = EV.splitsMerges(table);
  return {
    table, match, scores: EV.objectScores(match),
    dice: diceMasks(pred, truth),
    f1: TAUS.map(t => EV.objectScores(EV.match(table, t, method)).f1),
    pr: EV.averagePrecision(table, scores, tau),
    apMean: EV.apOverTaus(table, scores),
    splits: sm.splits, merges: sm.merges,
  };
}

export interface Preset { id: string; label: string; settings: Settings; tau: number; method: "greedy" | "optimal"; annotator?: boolean; craft?: "greedy"; tab?: "f1" | "pr" }

export const PRESETS: Preset[] = [
  { id: "default", label: "a typical result", settings: { ...DEFAULT_SETTINGS }, tau: 0.5, method: "optimal" },
  { id: "dice", label: "Dice high, count wrong", settings: { jitter: 0.6, dilate: 0, merge: 1, split: 0, miss: 0, specks: 0 }, tau: 0.5, method: "optimal" },
  { id: "sloppy", label: "sloppy outlines", settings: { jitter: 3, dilate: 0, merge: 0, split: 0, miss: 0, specks: 0 }, tau: 0.5, method: "optimal" },
  { id: "split", label: "one split", settings: { jitter: 0.4, dilate: 0, merge: 0, split: 0.1, miss: 0, specks: 0 }, tau: 0.5, method: "optimal" },
  { id: "greedy", label: "greedy loses a match", settings: { jitter: 0, dilate: 0, merge: 0, split: 0, miss: 0, specks: 0 }, tau: 0.3, method: "greedy", craft: "greedy" },
  { id: "specks", label: "specks", settings: { jitter: 0.8, dilate: 0, merge: 0, split: 0, miss: 0, specks: 12 }, tau: 0.5, method: "optimal", tab: "pr" },
  { id: "annotator", label: "vs a second annotator", settings: { jitter: 0.8, dilate: 0, merge: 0, split: 0, miss: 0, specks: 0 }, tau: 0.5, method: "optimal", annotator: true },
];
