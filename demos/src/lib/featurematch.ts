// Matching descriptors (no DOM). Owner: the keypoints day (feature-matching/).
//
// matchDescriptors: for every descriptor of image 1, its nearest and second-nearest in image 2
// (brute force). ratioTest (d1 < t·d2), mutualCheck (cross-check), oneToOne (the assignment
// problem via lib/linking.ts assignDense), evaluateMatches against a known transform
// (correct = the transformed point lands within ε px of the matched keypoint), prCurve over
// the ratio threshold, and a k-d tree with exact and best-bin-first (leaf budget) search.
// These match in DESCRIPTOR space; lib/linking.ts matchPoints and lib/spotdetect.ts match
// positions.

import { Desc, distanceFn, Metric } from "./descriptors";
import { assignDense } from "./linking";

export interface Match {
  /** Index in image 1 and its nearest in image 2. */
  i: number; j: number;
  d1: number;
  /** Second nearest in image 2 (−1 if none) and its distance. */
  j2: number; d2: number;
}

/** For each descriptor of D1: the two nearest of D2 under the metric. */
export function matchDescriptors(D1: Desc[], D2: Desc[], metric: Metric = "l2"): Match[] {
  const f = distanceFn(metric);
  return D1.map((a, i) => {
    let j = -1, d1 = Infinity, j2 = -1, d2 = Infinity;
    for (let k = 0; k < D2.length; k++) {
      const d = f(a, D2[k]);
      if (d < d1) { j2 = j; d2 = d1; j = k; d1 = d; } else if (d < d2) { j2 = k; d2 = d; }
    }
    return { i, j, d1, j2, d2 };
  }).filter(m => m.j >= 0);
}

/** d1/d2 (1 when both are 0; Infinity-safe). */
export const ratioOf = (m: Match): number => (m.d2 > 0 && Number.isFinite(m.d2) ? m.d1 / m.d2 : m.d1 > 0 ? 1 : (Number.isFinite(m.d2) ? 1 : 0));

/** Keep matches whose best is clearly better than the second best: d1 < t·d2. */
export function ratioTest(ms: Match[], t: number): Match[] {
  return ms.filter(m => ratioOf(m) < t);
}

/** Keep (i, j) only if i is also j's nearest in the other direction. `back` = matches from image 2 to image 1. */
export function mutualCheck(ms: Match[], back: Match[]): Match[] {
  const nn = new Map<number, number>();
  for (const b of back) nn.set(b.i, b.j);
  return ms.filter(m => nn.get(m.j) === m.i);
}

/** One-to-one assignment minimising the total distance (pairs costing more than maxCost are left unmatched). */
export function oneToOne(D1: Desc[], D2: Desc[], metric: Metric = "l2", maxCost = Infinity): Match[] {
  const f = distanceFn(metric), m = D1.length, n = D2.length, C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { const d = f(D1[i], D2[j]); C[i * n + j] = d <= maxCost ? d : Infinity; }
  const alt = Number.isFinite(maxCost) ? maxCost / 2 : 1e6;
  const res = assignDense(C, m, n, alt, alt);
  return res.links.map(l => ({ i: l.i, j: l.j, d1: l.c, j2: -1, d2: Infinity }));
}

/** A known transform from image 1 to image 2. */
export type Map2 = (x: number, y: number) => [number, number];
export interface P { x: number; y: number }

/** Per match: does the transformed point of i land within eps of keypoint j? */
export function correctFlags(ms: Match[], k1: P[], k2: P[], T: Map2, eps: number): boolean[] {
  return ms.map(m => { const [x, y] = T(k1[m.i].x, k1[m.i].y); return Math.hypot(k2[m.j].x - x, k2[m.j].y - y) <= eps; });
}

/** Number of points of image 1 that have SOME keypoint of image 2 within eps of their transformed position. */
export function possibleCount(k1: P[], k2: P[], T: Map2, eps: number): number {
  let n = 0;
  for (const p of k1) {
    const [x, y] = T(p.x, p.y);
    for (const q of k2) if (Math.hypot(q.x - x, q.y - y) <= eps) { n++; break; }
  }
  return n;
}

export interface MatchQuality { correct: boolean[]; nCorrect: number; accepted: number; possible: number; precision: number; recall: number }

export function evaluateMatches(ms: Match[], k1: P[], k2: P[], T: Map2, eps: number, possible?: number): MatchQuality {
  const correct = correctFlags(ms, k1, k2, T, eps);
  const nCorrect = correct.filter(Boolean).length, poss = possible ?? possibleCount(k1, k2, T, eps);
  return { correct, nCorrect, accepted: ms.length, possible: poss, precision: ms.length ? nCorrect / ms.length : 1, recall: poss ? Math.min(1, nCorrect / poss) : 0 };
}

export interface PRPoint { t: number; accepted: number; correct: number; precision: number; recall: number }

/** Precision and recall as the ratio threshold sweeps (matches with their correctness flags). */
export function prCurve(ms: Match[], correct: boolean[], possible: number, thresholds: number[]): PRPoint[] {
  const r = ms.map(ratioOf);
  return thresholds.map(t => {
    let acc = 0, cor = 0;
    for (let k = 0; k < ms.length; k++) if (r[k] < t) { acc++; if (correct[k]) cor++; }
    return { t, accepted: acc, correct: cor, precision: acc ? cor / acc : 1, recall: possible ? cor / possible : 0 };
  });
}

/** Area under a precision–recall curve (points sorted by recall; trapezoids). */
export function prArea(pts: { precision: number; recall: number }[]): number {
  const s = pts.slice().sort((a, b) => a.recall - b.recall);
  let A = 0;
  for (let k = 1; k < s.length; k++) A += (s[k].recall - s[k - 1].recall) * (s[k].precision + s[k - 1].precision) / 2;
  return A;
}

// ---------------------------------------------------------------------------------------
// k-d tree (L2)

interface KdNode { lo: number; hi: number; dim: number; split: number; left: KdNode | null; right: KdNode | null }

export class KdTree {
  readonly pts: ArrayLike<number>[];
  readonly dim: number;
  private idx: Int32Array;
  private root: KdNode | null;
  readonly leafSize: number;

  constructor(pts: ArrayLike<number>[], leafSize = 8) {
    this.pts = pts; this.dim = pts.length ? pts[0].length : 0; this.leafSize = leafSize;
    this.idx = Int32Array.from(pts.map((_, i) => i));
    this.root = pts.length ? this.build(0, pts.length) : null;
  }

  private build(lo: number, hi: number): KdNode {
    if (hi - lo <= this.leafSize) return { lo, hi, dim: -1, split: 0, left: null, right: null };
    // split on the dimension of largest spread, at the median
    let dim = 0, best = -1;
    for (let d = 0; d < this.dim; d++) {
      let mn = Infinity, mx = -Infinity;
      for (let k = lo; k < hi; k++) { const v = this.pts[this.idx[k]][d]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx - mn > best) { best = mx - mn; dim = d; }
    }
    const sub = Array.from(this.idx.subarray(lo, hi)).sort((a, b) => this.pts[a][dim] - this.pts[b][dim]);
    this.idx.set(sub, lo);
    const mid = (lo + hi) >> 1, split = this.pts[this.idx[mid]][dim];
    return { lo, hi, dim, split, left: this.build(lo, mid), right: this.build(mid, hi) };
  }

  /**
   * Two nearest neighbours of q. maxLeaves = Infinity: exact (branch and bound). Otherwise
   * best-bin-first: visit leaves in order of their distance bound, stop after maxLeaves.
   * Returns the neighbours, distances and the number of leaves and distance evaluations.
   */
  nearest2(q: ArrayLike<number>, maxLeaves = Infinity): { j: number; d1: number; j2: number; d2: number; leaves: number; evals: number } {
    let j = -1, j2 = -1, b1 = Infinity, b2 = Infinity, leaves = 0, evals = 0;
    if (!this.root) return { j, d1: Infinity, j2, d2: Infinity, leaves, evals };
    // priority queue of (bound², node) as a simple binary heap
    const heap: [number, KdNode][] = [[0, this.root]];
    const push = (e: [number, KdNode]) => { heap.push(e); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const pop = (): [number, KdNode] => { const top = heap[0], last = heap.pop()!; if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
    while (heap.length && leaves < maxLeaves) {
      const [bound, node0] = pop();
      if (bound >= b2) break;
      let node = node0, bnd = bound;
      // descend to a leaf, pushing the far branches with their bounds
      while (node.dim >= 0) {
        const diff = q[node.dim] - node.split;
        const near = diff < 0 ? node.left! : node.right!, far = diff < 0 ? node.right! : node.left!;
        push([Math.max(bnd, diff * diff), far]);
        node = near;
      }
      leaves++;
      for (let k = node.lo; k < node.hi; k++) {
        const t = this.idx[k], p = this.pts[t];
        let s = 0; for (let d = 0; d < this.dim; d++) { const e = p[d] - q[d]; s += e * e; }
        evals++;
        if (s < b1) { b2 = b1; j2 = j; b1 = s; j = t; } else if (s < b2) { b2 = s; j2 = t; }
      }
      void bnd;
    }
    return { j, d1: Math.sqrt(b1), j2, d2: Math.sqrt(b2), leaves, evals };
  }
}

/** Matches by k-d tree search (exact or approximate) with the fraction of exact nearest neighbours found. */
export function matchKd(D1: Desc[], D2: Desc[], maxLeaves = Infinity): { matches: Match[]; evals: number } {
  const tree = new KdTree(D2 as ArrayLike<number>[]);
  let evals = 0;
  const matches = D1.map((a, i) => { const r = tree.nearest2(a, maxLeaves); evals += r.evals; return { i, j: r.j, d1: r.d1, j2: r.j2, d2: r.d2 }; }).filter(m => m.j >= 0);
  return { matches, evals };
}
