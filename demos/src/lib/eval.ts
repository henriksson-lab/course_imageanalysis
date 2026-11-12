// Object-level evaluation (owner: evaluation day; no DOM). Pixel-level metrics live in
// lib/metrics.ts; this file counts OBJECTS and measures BOUNDARIES.
//
//  - iouTable: overlap, union and IoU of every (true, predicted) object pair of two label
//    images, in one pass over the pixels
//  - matching at an IoU threshold τ: forced (τ > 0.5, every object has at most one partner),
//    greedy (largest IoU first), optimal (Hungarian algorithm: most matches, then the
//    largest total IoU; or the largest total IoU)
//  - object counts → precision / recall / F1 (the metrics.ts formulas), matching accuracy
//    TP/(TP+FP+FN), the F1-vs-τ curve
//  - splits and merges read from the overlap table
//  - precision–recall curve and average precision (AP) of scored predictions, AP over τ
//  - boundary distances between two masks or two point sets: directed distances,
//    Hausdorff, HD95, average surface distance (ASD)
//
// Label images: 0 = background, objects 1..n (any integer array).
import { Counts, precision, recall, f1 } from "./metrics";
import { edt } from "./distance";

type A = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Overlap table

/** Overlaps of nt true and np predicted objects; arrays are row-major nt × np. */
export interface IouTable {
  nt: number;
  np: number;
  inter: Int32Array;     // |T_i ∩ P_j|
  areaT: Int32Array;     // |T_i| (length nt)
  areaP: Int32Array;     // |P_j| (length np)
  iou: Float64Array;     // |T_i ∩ P_j| / |T_i ∪ P_j|
}

/** Largest label in a label image. */
export function maxLabel(lab: A): number { let m = 0; for (let i = 0; i < lab.length; i++) if (lab[i] > m) m = lab[i]; return m; }

/** IoU table of two label images of the same size (labels 1..nt and 1..np). */
export function iouTable(truth: A, pred: A, nt = maxLabel(truth), np = maxLabel(pred)): IouTable {
  const inter = new Int32Array(nt * np), areaT = new Int32Array(nt), areaP = new Int32Array(np);
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i], p = pred[i];
    if (t > 0) areaT[t - 1]++;
    if (p > 0) areaP[p - 1]++;
    if (t > 0 && p > 0) inter[(t - 1) * np + (p - 1)]++;
  }
  const iou = new Float64Array(nt * np);
  for (let i = 0; i < nt; i++) for (let j = 0; j < np; j++) {
    const k = i * np + j, c = inter[k];
    if (c) iou[k] = c / (areaT[i] + areaP[j] - c);
  }
  return { nt, np, inter, areaT, areaP, iou };
}

/** A table from given IoU values (rows = true objects), for worked examples and tests. */
export function tableFromIou(rows: number[][]): IouTable {
  const nt = rows.length, np = nt ? rows[0].length : 0;
  return { nt, np, inter: new Int32Array(nt * np), areaT: new Int32Array(nt), areaP: new Int32Array(np), iou: Float64Array.from(rows.flat()) };
}

// ---------------------------------------------------------------------------------------
// Matching

export type MatchMethod = "forced" | "greedy" | "optimal" | "optimal-sum";

/** A one-to-one matching: partner of each true (matchT) and predicted (matchP) object, or −1. */
export interface Matching {
  pairs: { t: number; p: number; iou: number }[];   // 0-based object indices
  matchT: Int32Array;
  matchP: Int32Array;
  tp: number; fp: number; fn: number;
}

function finish(T: IouTable, pairs: { t: number; p: number; iou: number }[]): Matching {
  const matchT = new Int32Array(T.nt).fill(-1), matchP = new Int32Array(T.np).fill(-1);
  for (const q of pairs) { matchT[q.t] = q.p; matchP[q.p] = q.t; }
  pairs.sort((a, b) => a.t - b.t);
  return { pairs, matchT, matchP, tp: pairs.length, fp: T.np - pairs.length, fn: T.nt - pairs.length };
}

/** Forced matching: every pair with IoU ≥ τ (unique when τ > 0.5; below 0.5 conflicts are resolved greedily). */
export function matchForced(T: IouTable, tau: number): Matching {
  if (tau <= 0.5) return matchGreedy(T, tau);
  const pairs: { t: number; p: number; iou: number }[] = [];
  for (let i = 0; i < T.nt; i++) for (let j = 0; j < T.np; j++) {
    const v = T.iou[i * T.np + j];
    if (v >= tau) pairs.push({ t: i, p: j, iou: v });
  }
  return finish(T, pairs);
}

/** Greedy matching: take pairs in order of decreasing IoU while both objects are free and IoU ≥ τ. */
export function matchGreedy(T: IouTable, tau: number): Matching {
  const cand: { t: number; p: number; iou: number }[] = [];
  for (let i = 0; i < T.nt; i++) for (let j = 0; j < T.np; j++) {
    const v = T.iou[i * T.np + j];
    if (v >= tau && v > 0) cand.push({ t: i, p: j, iou: v });
  }
  cand.sort((a, b) => b.iou - a.iou || a.t - b.t || a.p - b.p);
  const usedT = new Uint8Array(T.nt), usedP = new Uint8Array(T.np), pairs: typeof cand = [];
  for (const c of cand) if (!usedT[c.t] && !usedP[c.p]) { usedT[c.t] = 1; usedP[c.p] = 1; pairs.push(c); }
  return finish(T, pairs);
}

/**
 * Hungarian algorithm (shortest augmenting paths with potentials), O(n³): a minimum-cost
 * perfect assignment of an n × n cost matrix (row-major). Returns col[row].
 */
export function hungarian(cost: A, n: number): Int32Array {
  const INF = Infinity;
  const u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const col = new Int32Array(n).fill(-1);
  for (let j = 1; j <= n; j++) if (p[j]) col[p[j] - 1] = j - 1;
  return col;
}

/**
 * Optimal matching as a linear assignment problem. Pairs with IoU < τ are not allowed.
 * `maxCount` (default): most matches first, then the largest total IoU; otherwise only the
 * largest total IoU.
 */
export function matchOptimal(T: IouTable, tau: number, maxCount = true): Matching {
  const n = Math.max(T.nt, T.np);
  if (n === 0) return finish(T, []);
  const w = new Float64Array(n * n);
  for (let i = 0; i < T.nt; i++) for (let j = 0; j < T.np; j++) {
    const v = T.iou[i * T.np + j];
    if (v >= tau && v > 0) w[i * n + j] = maxCount ? 1 + v / (n + 1) : v;
  }
  let wmax = 0;
  for (let k = 0; k < w.length; k++) wmax = Math.max(wmax, w[k]);
  const cost = Float64Array.from(w, x => wmax - x);
  const col = hungarian(cost, n), pairs: { t: number; p: number; iou: number }[] = [];
  for (let i = 0; i < T.nt; i++) {
    const j = col[i];
    if (j >= 0 && j < T.np && w[i * n + j] > 0) pairs.push({ t: i, p: j, iou: T.iou[i * T.np + j] });
  }
  return finish(T, pairs);
}

export function match(T: IouTable, tau: number, method: MatchMethod = "optimal"): Matching {
  switch (method) {
    case "forced": return matchForced(T, tau);
    case "greedy": return matchGreedy(T, tau);
    case "optimal-sum": return matchOptimal(T, tau, false);
    default: return matchOptimal(T, tau, true);
  }
}

// ---------------------------------------------------------------------------------------
// Scores from counts

/** Object counts as metrics.ts Counts (tn = 0: there are no true-negative objects). */
export const objectCounts = (m: { tp: number; fp: number; fn: number }): Counts => ({ tp: m.tp, fp: m.fp, fn: m.fn, tn: 0 });

export interface ObjectScores { tp: number; fp: number; fn: number; precision: number; recall: number; f1: number; accuracy: number }

/** Precision, recall, F1 and matching accuracy TP/(TP+FP+FN) of a matching. */
export function objectScores(m: { tp: number; fp: number; fn: number }): ObjectScores {
  const c = objectCounts(m);
  return { tp: m.tp, fp: m.fp, fn: m.fn, precision: precision(c), recall: recall(c), f1: f1(c), accuracy: matchingAccuracy(m) };
}

/** Matching accuracy TP / (TP + FP + FN) at one τ (often called "AP at τ" in cell-segmentation papers; it is not an area under a PR curve). */
export function matchingAccuracy(m: { tp: number; fp: number; fn: number }): number {
  const d = m.tp + m.fp + m.fn;
  return d === 0 ? 1 : m.tp / d;
}

/** The thresholds 0.5, 0.55, …, 0.95. */
export const COCO_TAUS: number[] = Array.from({ length: 10 }, (_, i) => Math.round((0.5 + 0.05 * i) * 100) / 100);

/** F1 (and the other scores) at each τ. */
export function f1Curve(T: IouTable, taus: number[], method: MatchMethod = "optimal"): ObjectScores[] {
  return taus.map(t => objectScores(match(T, t, method)));
}

// ---------------------------------------------------------------------------------------
// Splits and merges

/**
 * Splits: true objects that contain two or more predictions (each prediction has at least
 * `contain` of its area inside the object and IoU ≥ minIou with it). Merges: predictions
 * that contain two or more true objects. Indices are 0-based.
 */
export function splitsMerges(T: IouTable, contain = 0.5, minIou = 0.1): { splits: { t: number; parts: number[] }[]; merges: { p: number; parts: number[] }[] } {
  const splits: { t: number; parts: number[] }[] = [], merges: { p: number; parts: number[] }[] = [];
  for (let i = 0; i < T.nt; i++) {
    const parts: number[] = [];
    for (let j = 0; j < T.np; j++) {
      const k = i * T.np + j;
      if (T.inter[k] >= contain * T.areaP[j] && T.iou[k] >= minIou) parts.push(j);
    }
    if (parts.length >= 2) splits.push({ t: i, parts });
  }
  for (let j = 0; j < T.np; j++) {
    const parts: number[] = [];
    for (let i = 0; i < T.nt; i++) {
      const k = i * T.np + j;
      if (T.inter[k] >= contain * T.areaT[i] && T.iou[k] >= minIou) parts.push(i);
    }
    if (parts.length >= 2) merges.push({ p: j, parts });
  }
  return { splits, merges };
}

// ---------------------------------------------------------------------------------------
// Precision–recall and average precision

export interface PRCurve {
  recall: Float64Array;      // after each prediction, in score order
  precision: Float64Array;
  interp: Float64Array;      // max precision at this or higher recall
  ap: number;                // interpolated AP (area under the never-decreasing-from-the-right curve)
  apStep: number;            // stepwise AP Σ (R_k − R_{k−1}) P_k
  order: Int32Array;         // prediction indices by decreasing score
}

/**
 * Walk down the predictions by decreasing score; each is a TP or an FP (`isTP`, decided by
 * the matching at τ). nTruth = number of true objects.
 */
export function prFromScores(scores: A, isTP: ArrayLike<boolean | number>, nTruth: number): PRCurve {
  const n = scores.length;
  const order = Int32Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
  const R = new Float64Array(n), P = new Float64Array(n);
  let tp = 0;
  for (let k = 0; k < n; k++) {
    if (isTP[order[k]]) tp++;
    R[k] = nTruth ? tp / nTruth : 1;
    P[k] = tp / (k + 1);
  }
  const I = new Float64Array(n);
  let run = 0;
  for (let k = n - 1; k >= 0; k--) { run = Math.max(run, P[k]); I[k] = run; }
  let ap = 0, apStep = 0, prev = 0;
  for (let k = 0; k < n; k++) { ap += (R[k] - prev) * I[k]; apStep += (R[k] - prev) * P[k]; prev = R[k]; }
  return { recall: R, precision: P, interp: I, ap, apStep, order };
}

/**
 * AP at one τ for scored predictions. Matching for AP is done greedily in score order (a
 * prediction takes the free true object with the largest IoU ≥ τ), as in detection benchmarks.
 */
export function averagePrecision(T: IouTable, scores: A, tau: number): PRCurve {
  const order = Array.from({ length: T.np }, (_, j) => j).sort((a, b) => scores[b] - scores[a] || a - b);
  const usedT = new Uint8Array(T.nt), isTP = new Uint8Array(T.np);
  for (const j of order) {
    let best = -1, bv = 0;
    for (let i = 0; i < T.nt; i++) {
      const v = T.iou[i * T.np + j];
      if (!usedT[i] && v >= tau && v > 0 && v > bv) { bv = v; best = i; }
    }
    if (best >= 0) { usedT[best] = 1; isTP[j] = 1; }
  }
  return prFromScores(scores, isTP, T.nt);
}

/** AP averaged over τ = 0.5, 0.55, …, 0.95 (or the given thresholds). */
export function apOverTaus(T: IouTable, scores: A, taus: number[] = COCO_TAUS): number {
  let s = 0;
  for (const t of taus) s += averagePrecision(T, scores, t).ap;
  return s / taus.length;
}

// ---------------------------------------------------------------------------------------
// Boundary distances

/** Boundary pixels of a mask: object pixels with a 4-neighbour outside the object (or on the image edge). */
export function boundaryMask(mask: A, w: number, h: number): Uint8Array {
  const b = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w]) b[i] = 1;
  }
  return b;
}

/** Summary of the two directed distance lists. */
export interface SurfaceDistances {
  dAB: Float64Array;   // for every boundary point of A, the distance to the nearest boundary point of B
  dBA: Float64Array;
  hAB: number; hBA: number;   // directed Hausdorff distances (max of each list)
  hausdorff: number;          // max(hAB, hBA)
  hd95: number;               // max of the two 95th percentiles
  asd: number;                // (Σ dAB + Σ dBA) / (|A| + |B|)
}

function pct95(d: Float64Array): number {
  if (!d.length) return 0;
  const s = Float64Array.from(d).sort();
  return s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)];
}

/** Hausdorff, HD95 and ASD from two directed distance lists. */
export function surfaceStats(dAB: Float64Array, dBA: Float64Array): SurfaceDistances {
  let hAB = 0, hBA = 0, s = 0;
  for (const v of dAB) { hAB = Math.max(hAB, v); s += v; }
  for (const v of dBA) { hBA = Math.max(hBA, v); s += v; }
  const n = dAB.length + dBA.length;
  return { dAB, dBA, hAB, hBA, hausdorff: Math.max(hAB, hBA), hd95: Math.max(pct95(dAB), pct95(dBA)), asd: n ? s / n : 0 };
}

/** Brute-force directed distances between two point sets (small sets, worked examples). */
export function pointSetDistances(A: [number, number][], B: [number, number][]): SurfaceDistances {
  const dir = (P: [number, number][], Q: [number, number][]) => Float64Array.from(P, ([x, y]) => {
    let m = Infinity;
    for (const [u, v] of Q) m = Math.min(m, Math.hypot(x - u, y - v));
    return m;
  });
  return surfaceStats(dir(A, B), dir(B, A));
}

/**
 * Boundary distances of two masks via the distance transform: the EDT of "not boundary of B"
 * gives, at every pixel, the distance to the nearest boundary pixel of B; read it on the
 * boundary of A (and the other way round). O(number of pixels).
 */
export function maskBoundaryDistances(a: A, b: A, w: number, h: number): SurfaceDistances {
  const ba = boundaryMask(a, w, h), bb = boundaryMask(b, w, h);
  const notA = Uint8Array.from(ba, v => (v ? 0 : 1)), notB = Uint8Array.from(bb, v => (v ? 0 : 1));
  const toA = edt(notA, w, h), toB = edt(notB, w, h);
  const dAB: number[] = [], dBA: number[] = [];
  for (let i = 0; i < w * h; i++) { if (ba[i]) dAB.push(toB[i]); if (bb[i]) dBA.push(toA[i]); }
  return surfaceStats(Float64Array.from(dAB), Float64Array.from(dBA));
}
