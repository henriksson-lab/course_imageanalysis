// Decision trees and random forests (CART), owner: random-forest day (day 14). No DOM.
//
// Data are feature-major tables: X[j * n + i] is feature j of sample i (n samples, d
// features). Before growing, a table is binned per feature (`binTable`): with at most
// `maxBins` distinct values every value is its own bin, so the split search is exact (the
// thresholds are midpoints between consecutive values, as in textbook CART); with more,
// values are grouped into quantile bins. One weighted split search (`bestSplit`) serves
// every tree, the forests and, later, the boosted stumps of lib/boost.ts (integral-images
// day): no second implementation.
//
//   const T = tableFromRows([[20, 2], [35, 3], ...]);
//   const B = binTable(T);
//   const tree = growTree(B, classTarget(y, 2), null, { criterion: "gini", maxDepth: 3 });
//   const forest = trainForest(B, classTarget(y, 2), null, { nTrees: 100, mtry: 1, seed: 1 });
//   const P = predictForest(forest, T.X, T.n);          // n × k class probabilities
//   oobError(forest, T, y);                             // out-of-bag error
//
// Options: criterion (Gini, entropy, misclassification; SSE for regression), maximum depth,
// minimum samples per leaf, features tried per split (m), random thresholds (extra trees),
// best-first growth with a split budget and forced splits (treeplay), sample weights (class
// balancing), bootstrap with in-bag counts (OOB error), impurity and permutation importance.
import { rng } from "./ui";
import type { Classifier, TrainingSet, Model } from "./classifier";

type Rng = ReturnType<typeof rng>;
type Num = ArrayLike<number>;

export type Criterion = "gini" | "entropy" | "misclass" | "sse";

// ---------------------------------------------------------------------------------------
// Tables and binning

/** A feature-major table: X[j * n + i] = feature j of sample i. */
export interface Table { n: number; d: number; X: Float32Array | Float64Array }

/** Table from rows (one array of d features per sample). */
export function tableFromRows(rows: Num[]): Table {
  const n = rows.length, d = n ? rows[0].length : 0, X = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[j * n + i] = rows[i][j];
  return { n, d, X };
}

/** Table from columns (one array of n values per feature). */
export function tableFromColumns(cols: Num[]): Table {
  const d = cols.length, n = d ? cols[0].length : 0, X = new Float64Array(n * d);
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) X[j * n + i] = cols[j][i];
  return { n, d, X };
}

/** Feature j of sample i. */
export const at = (t: Table, i: number, j: number): number => t.X[j * t.n + i];

/** A table with each feature replaced by its bin code. */
export interface Binned {
  n: number; d: number;
  /** codes[j * n + i] = bin of sample i in feature j (0 .. nb[j]-1, in increasing value order). */
  codes: Uint16Array;
  /** Number of bins per feature. */
  nb: Int32Array;
  /** Smallest and largest value of the samples in each bin, per feature. */
  lo: Float64Array[];
  hi: Float64Array[];
  /** The binned table itself. */
  table: Table;
}

/**
 * Bin every feature: exact (one bin per distinct value) when there are at most `maxBins`
 * distinct values, otherwise quantile bins of about equal counts. Bins never split a value.
 */
export function binTable(t: Table, maxBins = 256): Binned {
  const { n, d } = t;
  const codes = new Uint16Array(n * d), nb = new Int32Array(d), lo: Float64Array[] = [], hi: Float64Array[] = [];
  const col = new Float64Array(n);
  for (let j = 0; j < d; j++) {
    const off = j * n;
    for (let i = 0; i < n; i++) col[i] = t.X[off + i];
    const sorted = col.slice().sort();
    let distinct = 0;
    for (let r = 0; r < n; r++) if (r === 0 || sorted[r] !== sorted[r - 1]) distinct++;
    const L: number[] = [], H: number[] = [];
    let bin = -1, startRank = 0;
    const target = distinct <= maxBins ? 0 : n / maxBins;
    for (let r = 0; r < n; r++) {
      const v = sorted[r];
      const newValue = r === 0 || v !== sorted[r - 1];
      // open a new bin at a new value: always (exact), or once the bin holds its share (quantile)
      if (newValue && (bin < 0 || target === 0 || r - startRank >= target)) { bin++; L.push(v); H.push(v); startRank = r; }
      H[bin] = v;
    }
    // code of each sample: the last bin whose smallest value is ≤ the sample's value
    for (let i = 0; i < n; i++) {
      const v = col[i];
      let a = 0, b = bin;
      while (a < b) { const m = (a + b + 1) >> 1; if (L[m] <= v) a = m; else b = m - 1; }
      codes[off + i] = a;
    }
    nb[j] = Math.max(1, bin + 1);
    lo.push(Float64Array.from(L)); hi.push(Float64Array.from(H));
  }
  return { n, d, codes, nb, lo, hi, table: t };
}

// ---------------------------------------------------------------------------------------
// Targets and impurity

/** What a tree predicts: a class 0..k-1 per sample, or a number (regression). */
export type Target = { kind: "class"; y: Int32Array; k: number } | { kind: "reg"; y: Float64Array };

export function classTarget(y: Num, k?: number): Target {
  const yy = Int32Array.from(y as ArrayLike<number>);
  let kk = k ?? 0;
  if (k === undefined) for (const v of yy) kk = Math.max(kk, v + 1);
  return { kind: "class", y: yy, k: kk };
}
export function regTarget(y: Num): Target { return { kind: "reg", y: Float64Array.from(y as ArrayLike<number>) }; }

/** Impurity of a node with (weighted) class counts c[0..k-1] and total W. */
export function impurity(c: Num, W: number, crit: Criterion): number {
  if (W <= 0) return 0;
  const k = c.length;
  if (crit === "gini") { let s = 0; for (let i = 0; i < k; i++) { const p = c[i] / W; s += p * p; } return 1 - s; }
  if (crit === "entropy") { let s = 0; for (let i = 0; i < k; i++) { const p = c[i] / W; if (p > 0) s -= p * Math.log2(p); } return s; }
  let m = 0; for (let i = 0; i < k; i++) m = Math.max(m, c[i]);
  return 1 - m / W;   // misclassification (also used for "sse" on classes, which is not meaningful)
}

/** Impurity decrease of a split: I(parent) − W_L/W · I(L) − W_R/W · I(R). */
export function impurityDecrease(left: Num, right: Num, crit: Criterion): number {
  const k = left.length, P = new Float64Array(k);
  let WL = 0, WR = 0;
  for (let i = 0; i < k; i++) { P[i] = left[i] + right[i]; WL += left[i]; WR += right[i]; }
  const W = WL + WR;
  return impurity(P, W, crit) - (WL / W) * impurity(left, WL, crit) - (WR / W) * impurity(right, WR, crit);
}

// ---------------------------------------------------------------------------------------
// The split search (one implementation, weighted)

let scratchH = new Float64Array(1024), scratchC = new Int32Array(256);

export interface Split {
  feature: number;
  /** Samples with value ≤ threshold go left. */
  threshold: number;
  /** Last bin that goes left. */
  bin: number;
  /** Impurity decrease of this node (not multiplied by the node's share of the data). */
  gain: number;
  /** Weight and number of samples going left / right. */
  wL: number; wR: number; nL: number; nR: number;
}

export interface SplitScan {
  feature: number;
  thresholds: Float64Array;
  gains: Float64Array;
}

export interface SplitSearchOptions {
  criterion?: Criterion;
  /** Minimum number of samples in each child (default 1). */
  minLeaf?: number;
  /** Extra trees: one random threshold per feature instead of the best one. */
  random?: Rng | null;
  /** Record the gain of every threshold (for the split-scan plot). */
  scan?: SplitScan[];
}

/**
 * Best split of the samples idx[start..end) over the given features. Weights `w` (or 1)
 * multiply each sample's contribution (bootstrap counts, class weights, boosting weights).
 * Returns null when no split leaves at least `minLeaf` samples on each side.
 */
export function bestSplit(B: Binned, idx: Int32Array, start: number, end: number, target: Target,
  w: Num | null, features: Num, o: SplitSearchOptions = {}): Split | null {
  const crit = o.criterion ?? (target.kind === "reg" ? "sse" : "gini"), minLeaf = Math.max(1, o.minLeaf ?? 1);
  const n = B.n, cls = target.kind === "class", k = cls ? (target as { k: number }).k : 3;
  let best: Split | null = null;
  // parent statistics
  const P = new Float64Array(k);
  let W = 0;
  for (let r = start; r < end; r++) {
    const i = idx[r], wi = w ? w[i] : 1;
    W += wi;
    if (cls) P[target.y[i]] += wi;
    else { const v = target.y[i]; P[0] += wi; P[1] += wi * v; P[2] += wi * v * v; }
  }
  if (W <= 0) return null;
  const nodeImp = cls ? impurity(P, W, crit) : Math.max(0, P[2] / W - (P[1] / W) ** 2);
  const L = new Float64Array(k), R = new Float64Array(k);
  for (let f = 0; f < features.length; f++) {
    const j = features[f], nbj = B.nb[j], off = j * n;
    if (scratchH.length < nbj * k) scratchH = new Float64Array(nbj * k * 2);
    if (scratchC.length < nbj) scratchC = new Int32Array(nbj * 2);
    const hist = scratchH, cnt = scratchC;
    hist.fill(0, 0, nbj * k); cnt.fill(0, 0, nbj);
    for (let r = start; r < end; r++) {
      const i = idx[r], b = B.codes[off + i], wi = w ? w[i] : 1;
      cnt[b]++;
      if (cls) hist[b * k + target.y[i]] += wi;
      else { const v = target.y[i]; hist[b * k] += wi; hist[b * k + 1] += wi * v; hist[b * k + 2] += wi * v * v; }
    }
    // random threshold (extra trees): uniform between the node's smallest and largest value
    let tRand = NaN;
    if (o.random) {
      let bmin = -1, bmax = -1;
      for (let b = 0; b < nbj; b++) if (cnt[b]) { if (bmin < 0) bmin = b; bmax = b; }
      if (bmin === bmax) continue;
      const vmin = B.lo[j][bmin], vmax = B.hi[j][bmax];
      tRand = vmin + (vmax - vmin) * o.random.unif();
      if (!(tRand > vmin)) tRand = (vmin + vmax) / 2;
    }
    const thr: number[] = [], gs: number[] = [];
    const Hj = B.hi[j], Lj = B.lo[j];
    L.fill(0);
    let nLeft = 0, last = -1;
    for (let b = 0; b < nbj; b++) {
      if (!cnt[b]) continue;
      if (last >= 0) {
        // cut between bin `last` (and everything before) and bin b
        const t = 0.5 * (Hj[last] + Lj[b]);
        const nR = (end - start) - nLeft;
        const wanted = o.random ? (Hj[last] < tRand && tRand <= Lj[b]) : true;
        if (wanted && nLeft >= minLeaf && nR >= minLeaf) {
          let WL = 0;
          for (let c = 0; c < k; c++) R[c] = P[c] - L[c];
          let gain: number;
          if (cls) {
            for (let c = 0; c < k; c++) WL += L[c];
            const WR = W - WL;
            gain = nodeImp - (WL / W) * impurity(L, WL, crit) - (WR / W) * impurity(R, WR, crit);
          } else {
            WL = L[0];
            const WR = R[0];
            const vl = WL > 0 ? L[2] / WL - (L[1] / WL) ** 2 : 0, vr = WR > 0 ? R[2] / WR - (R[1] / WR) ** 2 : 0;
            gain = nodeImp - (WL / W) * Math.max(0, vl) - (WR / W) * Math.max(0, vr);
          }
          if (o.scan) { thr.push(o.random ? tRand : t); gs.push(gain); }
          if (!best || gain > best.gain + 1e-12) {
            best = { feature: j, threshold: o.random ? tRand : t, bin: last, gain, wL: WL, wR: W - WL, nL: nLeft, nR };
          }
        }
      }
      for (let c = 0; c < k; c++) L[c] += hist[b * k + c];
      nLeft += cnt[b];
      last = b;
    }
    if (o.scan) o.scan.push({ feature: j, thresholds: Float64Array.from(thr), gains: Float64Array.from(gs) });
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// Trees

export interface TreeOptions {
  criterion?: Criterion;
  /** Maximum depth (root = depth 0; default unlimited). */
  maxDepth?: number;
  /** Minimum number of samples in a leaf (default 1). */
  minLeaf?: number;
  /** Smallest impurity decrease worth a split (default 1e-12). */
  minGain?: number;
  /** Stop after this many splits, growing the most useful node first (default unlimited). */
  maxSplits?: number;
  /** Features tried at each split (m; default all). */
  mtry?: number;
  /** Extremely randomised trees: random thresholds. */
  extra?: boolean;
  seed?: number;
  /** Splits imposed on given nodes, by path from the root ("" = root, "L", "LR", ...). */
  forced?: Record<string, { feature: number; threshold: number }>;
}

/** A tree in flat arrays; node 0 is the root, feature[t] = -1 marks a leaf. */
export interface Tree {
  nNodes: number;
  /** Number of classes (1 for regression). */
  k: number;
  d: number;
  feature: Int32Array;
  threshold: Float64Array;
  left: Int32Array;
  right: Int32Array;
  /** Leaf value: class proportions (k per node) or the mean (regression). */
  value: Float64Array;
  /** Weighted size, number of samples, impurity, impurity decrease of the split, depth. */
  weight: Float64Array;
  count: Int32Array;
  impurity: Float64Array;
  gain: Float64Array;
  depth: Int32Array;
  /** Path of every node from the root ("", "L", "LR", ...). */
  path: string[];
  /** Order in which nodes were split (node ids). */
  splitOrder: number[];
  /** Impurity importance of each feature in this tree: Σ over its splits of W_t/W_root · ΔI_t. */
  importance: Float64Array;
}

interface Pending { node: number; start: number; end: number; depth: number; split: Split | null; forced: boolean }

/** Grow one tree on the samples `rows` (default all) with weights `w` (default 1). */
export function growTree(B: Binned, target: Target, w: Num | null, o: TreeOptions = {}, rows?: Int32Array, r?: Rng): Tree {
  const crit = o.criterion ?? (target.kind === "reg" ? "sse" : "gini");
  const maxDepth = o.maxDepth ?? Infinity, minLeaf = Math.max(1, o.minLeaf ?? 1), minGain = o.minGain ?? 1e-12;
  const maxSplits = o.maxSplits ?? Infinity, d = B.d, mtry = Math.min(d, Math.max(1, o.mtry ?? d));
  const R = r ?? rng(o.seed ?? 1);
  const cls = target.kind === "class", k = cls ? (target as { k: number }).k : 1;
  const idx = rows ? Int32Array.from(rows) : Int32Array.from({ length: B.n }, (_, i) => i);
  const N = idx.length;
  // growable node storage
  const F: number[] = [], T: number[] = [], Lc: number[] = [], Rc: number[] = [], V: number[] = [], Wt: number[] = [], Cn: number[] = [], Im: number[] = [], G: number[] = [], D: number[] = [], paths: string[] = [];
  const order: number[] = [];
  const imp = new Float64Array(d);
  const allFeatures = Int32Array.from({ length: d }, (_, j) => j);
  const featScratch = Int32Array.from(allFeatures);

  const newNode = (start: number, end: number, depth: number, path: string): number => {
    const id = F.length;
    let W = 0;
    const c = new Float64Array(cls ? k : 3);
    for (let q = start; q < end; q++) {
      const i = idx[q], wi = w ? w[i] : 1;
      W += wi;
      if (cls) c[target.y[i]] += wi; else { c[0] += wi; c[1] += wi * target.y[i]; c[2] += wi * target.y[i] ** 2; }
    }
    F.push(-1); T.push(0); Lc.push(-1); Rc.push(-1); G.push(0); D.push(depth); paths.push(path);
    Wt.push(W); Cn.push(end - start);
    if (cls) { for (let q = 0; q < k; q++) V.push(W > 0 ? c[q] / W : 1 / k); Im.push(impurity(c, W, crit)); }
    else { const m = W > 0 ? c[1] / W : 0; V.push(m); Im.push(W > 0 ? Math.max(0, c[2] / W - m * m) : 0); }
    return id;
  };

  const chooseFeatures = (): Num => {
    if (mtry >= d) return allFeatures;
    // partial Fisher–Yates: the first mtry entries are a random subset
    for (let a = 0; a < mtry; a++) {
      const b = a + Math.floor(R.unif() * (d - a));
      const t = featScratch[a]; featScratch[a] = featScratch[b]; featScratch[b] = t;
    }
    return featScratch.subarray(0, mtry);
  };

  const candidate = (node: number, start: number, end: number, depth: number): Pending => {
    const forced = o.forced?.[paths[node]];
    if (forced) {
      // evaluate the imposed split exactly (its bin from the threshold)
      const j = forced.feature;
      let bin = -1;
      for (let b = 0; b < B.nb[j]; b++) if (B.hi[j][b] <= forced.threshold) bin = b;
      const s = evalSplit(B, idx, start, end, target, w, j, bin, crit);
      if (s && s.nL > 0 && s.nR > 0) { s.threshold = forced.threshold; return { node, start, end, depth, split: s, forced: true }; }
    }
    let split: Split | null = null;
    if (depth < maxDepth && end - start >= 2 * minLeaf && Im[node] > 1e-12) {
      split = bestSplit(B, idx, start, end, target, w, chooseFeatures(), { criterion: crit, minLeaf, random: o.extra ? R : null });
      if (split && split.gain <= minGain) split = null;
    }
    return { node, start, end, depth, split, forced: false };
  };

  const root = newNode(0, N, 0, "");
  const open: Pending[] = [candidate(root, 0, N, 0)];
  const Wroot = Wt[root] || 1;
  let splits = 0;
  const bestFirst = Number.isFinite(maxSplits);
  while (open.length && splits < maxSplits) {
    // pick: forced splits first, then (best-first) the largest weighted gain, else last in (depth first)
    let pick = open.length - 1;
    if (bestFirst) {
      let bestScore = -Infinity;
      open.forEach((p, q) => {
        if (!p.split) return;
        const s = (p.forced ? 1e9 : 0) + Wt[p.node] * p.split.gain;
        if (s > bestScore) { bestScore = s; pick = q; }
      });
    }
    const p = open.splice(pick, 1)[0];
    if (!p.split) { if (bestFirst && open.every(q => !q.split)) break; else continue; }
    const s = p.split, j = s.feature, off = j * B.n;
    // partition idx[start..end) by bin
    let a = p.start, b = p.end - 1;
    while (a <= b) {
      if (B.codes[off + idx[a]] <= s.bin) a++;
      else { const t = idx[a]; idx[a] = idx[b]; idx[b] = t; b--; }
    }
    const mid = a;
    F[p.node] = j; T[p.node] = s.threshold; G[p.node] = s.gain;
    imp[j] += (Wt[p.node] / Wroot) * s.gain;
    order.push(p.node);
    splits++;
    const l = newNode(p.start, mid, p.depth + 1, paths[p.node] + "L");
    const rr = newNode(mid, p.end, p.depth + 1, paths[p.node] + "R");
    Lc[p.node] = l; Rc[p.node] = rr;
    open.push(candidate(l, p.start, mid, p.depth + 1));
    open.push(candidate(rr, mid, p.end, p.depth + 1));
  }
  const nNodes = F.length;
  return {
    nNodes, k, d,
    feature: Int32Array.from(F), threshold: Float64Array.from(T), left: Int32Array.from(Lc), right: Int32Array.from(Rc),
    value: Float64Array.from(V), weight: Float64Array.from(Wt), count: Int32Array.from(Cn), impurity: Float64Array.from(Im),
    gain: Float64Array.from(G), depth: Int32Array.from(D), path: paths, splitOrder: order, importance: imp,
  };
}

/** Evaluate one given split (feature j, samples with bin ≤ bin go left). */
export function evalSplit(B: Binned, idx: Int32Array, start: number, end: number, target: Target, w: Num | null,
  j: number, bin: number, crit: Criterion): Split | null {
  const cls = target.kind === "class", k = cls ? (target as { k: number }).k : 3, off = j * B.n;
  const Lw = new Float64Array(k), Rw = new Float64Array(k);
  let nL = 0, nR = 0;
  for (let q = start; q < end; q++) {
    const i = idx[q], wi = w ? w[i] : 1, A = B.codes[off + i] <= bin ? Lw : Rw;
    if (A === Lw) nL++; else nR++;
    if (cls) A[target.y[i]] += wi; else { A[0] += wi; A[1] += wi * target.y[i]; A[2] += wi * target.y[i] ** 2; }
  }
  if (!nL || !nR) return null;
  let gain: number, WL: number, WR: number;
  if (cls) {
    WL = Lw.reduce((a, b) => a + b, 0); WR = Rw.reduce((a, b) => a + b, 0);
    gain = impurityDecrease(Lw, Rw, crit);
  } else {
    WL = Lw[0]; WR = Rw[0];
    const W = WL + WR, m = (Lw[1] + Rw[1]) / W, v = (Lw[2] + Rw[2]) / W - m * m;
    const vl = Lw[2] / WL - (Lw[1] / WL) ** 2, vr = Rw[2] / WR - (Rw[1] / WR) ** 2;
    gain = v - (WL / W) * vl - (WR / W) * vr;
  }
  const lastLeft = bin, t = bin >= 0 && bin < B.nb[j] - 1 ? 0.5 * (B.hi[j][bin] + B.lo[j][bin + 1]) : NaN;
  return { feature: j, threshold: t, bin: lastLeft, gain, wL: WL, wR: WR, nL, nR };
}

/** Leaf reached by the sample with feature values get(j). */
export function leafOf(t: Tree, get: (j: number) => number): number {
  let q = 0;
  while (t.feature[q] >= 0) q = get(t.feature[q]) <= t.threshold[q] ? t.left[q] : t.right[q];
  return q;
}

/** Leaf of sample i of a feature-major array X with n samples. */
export function leafOfRow(t: Tree, X: Num, n: number, i: number): number {
  let q = 0;
  while (t.feature[q] >= 0) q = X[t.feature[q] * n + i] <= t.threshold[q] ? t.left[q] : t.right[q];
  return q;
}

/** Class proportions (or [mean]) predicted by one tree for a point given as an array. */
export function predictPoint(t: Tree, x: Num): Float64Array {
  const q = leafOf(t, j => x[j]);
  return t.value.slice(q * t.k, q * t.k + t.k);
}

/** Number of leaves and maximum depth of a tree. */
export function treeShape(t: Tree): { leaves: number; depth: number } {
  let leaves = 0, depth = 0;
  for (let q = 0; q < t.nNodes; q++) { if (t.feature[q] < 0) leaves++; depth = Math.max(depth, t.depth[q]); }
  return { leaves, depth };
}

// ---------------------------------------------------------------------------------------
// Forests

export interface ForestOptions extends TreeOptions {
  /** Number of trees (default 100). */
  nTrees?: number;
  /** Bootstrap samples (default true); otherwise every tree sees all samples. */
  bootstrap?: boolean;
}

export interface Forest {
  trees: Tree[];
  /** inbag[b][i] = how often sample i was drawn for tree b (0 = out of bag). */
  inbag: Uint8Array[];
  n: number; d: number; k: number;
  regression: boolean;
}

/** Train a random forest (bagging + a random subset of mtry features at each split). */
export function trainForest(B: Binned, target: Target, w: Num | null, o: ForestOptions = {}): Forest {
  const nT = o.nTrees ?? 100, n = B.n, R = rng(o.seed ?? 1);
  const trees: Tree[] = [], inbag: Uint8Array[] = [];
  const wb = new Float64Array(n);
  for (let b = 0; b < nT; b++) {
    const cnt = new Uint8Array(n);
    if (o.bootstrap === false) cnt.fill(1);
    else for (let q = 0; q < n; q++) { const i = Math.floor(R.unif() * n); if (cnt[i] < 255) cnt[i]++; }
    const rows: number[] = [];
    for (let i = 0; i < n; i++) if (cnt[i]) { rows.push(i); wb[i] = cnt[i] * (w ? w[i] : 1); } else wb[i] = 0;
    trees.push(growTree(B, target, wb.slice(), o, Int32Array.from(rows), R));
    inbag.push(cnt);
  }
  const k = target.kind === "class" ? (target as { k: number }).k : 1;
  return { trees, inbag, n, d: B.d, k, regression: target.kind === "reg" };
}

/**
 * Forest prediction for the n samples of a feature-major X: n × k values (class
 * probabilities = average of the trees' leaf proportions; regression: the mean).
 * `nTrees` uses only the first trees.
 */
export function predictForest(f: { trees: Tree[]; k: number }, X: Num, n: number, nTrees?: number, out?: Float32Array, from = 0, to = n): Float32Array {
  const k = f.k, res = out ?? new Float32Array(n * k), T = Math.min(f.trees.length, nTrees ?? f.trees.length);
  for (let i = from; i < to; i++) for (let c = 0; c < k; c++) res[i * k + c] = 0;
  for (let b = 0; b < T; b++) {
    const t = f.trees[b], feat = t.feature, thr = t.threshold, L = t.left, Rr = t.right, val = t.value;
    for (let i = from; i < to; i++) {
      let q = 0;
      while (feat[q] >= 0) q = X[feat[q] * n + i] <= thr[q] ? L[q] : Rr[q];
      const o = q * k, r0 = i * k;
      for (let c = 0; c < k; c++) res[r0 + c] += val[o + c];
    }
  }
  if (T > 0) for (let i = from; i < to; i++) for (let c = 0; c < k; c++) res[i * k + c] /= T;
  return res;
}

/** Forest prediction for one point (array of d features). */
export function predictForestPoint(f: { trees: Tree[]; k: number }, x: Num, nTrees?: number): Float64Array {
  const k = f.k, p = new Float64Array(k), T = Math.min(f.trees.length, nTrees ?? f.trees.length);
  for (let b = 0; b < T; b++) { const t = f.trees[b], q = leafOf(t, j => x[j]); for (let c = 0; c < k; c++) p[c] += t.value[q * k + c]; }
  if (T > 0) for (let c = 0; c < k; c++) p[c] /= T;
  return p;
}

/** Index of the largest entry of row i of an n × k array. */
export function argmaxRow(P: Num, k: number, i: number): number {
  let best = 0;
  for (let c = 1; c < k; c++) if (P[i * k + c] > P[i * k + best]) best = c;
  return best;
}

/**
 * Out-of-bag prediction: for each training sample, the average over the trees that did not
 * see it. `votes[i]` = number of such trees (0: no OOB prediction). X is the training table.
 */
export function oobPredict(f: Forest, X: Num, nTrees?: number): { P: Float64Array; votes: Int32Array } {
  const n = f.n, k = f.k, P = new Float64Array(n * k), votes = new Int32Array(n);
  const T = Math.min(f.trees.length, nTrees ?? f.trees.length);
  for (let b = 0; b < T; b++) {
    const t = f.trees[b], inb = f.inbag[b];
    for (let i = 0; i < n; i++) {
      if (inb[i]) continue;
      const q = leafOfRow(t, X, n, i);
      for (let c = 0; c < k; c++) P[i * k + c] += t.value[q * k + c];
      votes[i]++;
    }
  }
  for (let i = 0; i < n; i++) if (votes[i]) for (let c = 0; c < k; c++) P[i * k + c] /= votes[i];
  return { P, votes };
}

/** Error of probabilities P (n × k) against labels: misclassification rate (classes) or MSE. */
export function errorOf(P: Num, k: number, y: Num, regression = false, use?: Num): number {
  let e = 0, m = 0;
  for (let i = 0; i < y.length; i++) {
    if (use && !use[i]) continue;
    m++;
    if (regression) e += (P[i] - y[i]) ** 2;
    else if (argmaxRow(P, k, i) !== y[i]) e++;
  }
  return m ? e / m : NaN;
}

/** Out-of-bag error (misclassification or MSE) over the samples with at least one OOB tree. */
export function oobError(f: Forest, X: Num, y: Num, nTrees?: number): number {
  const { P, votes } = oobPredict(f, X, nTrees);
  return errorOf(P, f.k, y, f.regression, votes);
}

/** OOB error after 1, 2, ..., T trees (incremental, one pass). */
export function oobErrorCurve(f: Forest, X: Num, y: Num, T = f.trees.length): Float64Array {
  const n = f.n, k = f.k, S = new Float64Array(n * k), votes = new Int32Array(n), out = new Float64Array(T);
  for (let b = 0; b < T; b++) {
    const t = f.trees[b], inb = f.inbag[b];
    for (let i = 0; i < n; i++) {
      if (inb[i]) continue;
      const q = leafOfRow(t, X, n, i);
      for (let c = 0; c < k; c++) S[i * k + c] += t.value[q * k + c];
      votes[i]++;
    }
    out[b] = errorOf(S, k, y, f.regression, votes);   // argmax of the sum = argmax of the mean
  }
  return out;
}

/** Test error after 1, 2, ..., T trees on a separate feature-major table (n samples). */
export function testErrorCurve(f: Forest, X: Num, n: number, y: Num, T = f.trees.length): Float64Array {
  const k = f.k, S = new Float64Array(n * k), out = new Float64Array(T), tmp = new Float64Array(n * k);
  for (let b = 0; b < T; b++) {
    const t = f.trees[b];
    for (let i = 0; i < n; i++) { const q = leafOfRow(t, X, n, i); for (let c = 0; c < k; c++) S[i * k + c] += t.value[q * k + c]; }
    if (f.regression) { for (let i = 0; i < n; i++) tmp[i] = S[i] / (b + 1); out[b] = errorOf(tmp, 1, y, true); }
    else out[b] = errorOf(S, k, y);
  }
  return out;
}

/** Impurity (mean decrease in impurity) importance: average over the trees. */
export function impurityImportance(f: { trees: Tree[]; d: number }): Float64Array {
  const imp = new Float64Array(f.d);
  for (const t of f.trees) for (let j = 0; j < f.d; j++) imp[j] += t.importance[j];
  if (f.trees.length) for (let j = 0; j < f.d; j++) imp[j] /= f.trees.length;
  return imp;
}

/**
 * Permutation importance on the out-of-bag samples: increase of the OOB error when a
 * feature (or a group of features, permuted together with the same shuffle) is shuffled
 * across the samples. Averaged over `repeats` shuffles.
 */
export function permutationImportance(f: Forest, T: Table, y: Num, o: { repeats?: number; seed?: number; groups?: number[][] } = {}): Float64Array {
  const groups = o.groups ?? Array.from({ length: T.d }, (_, j) => [j]);
  const base = oobError(f, T.X, y), R = rng(o.seed ?? 11), reps = o.repeats ?? 5, n = T.n;
  const out = new Float64Array(groups.length);
  const Xp = T.X.slice();
  const perm = new Int32Array(n);
  for (let g = 0; g < groups.length; g++) {
    let s = 0;
    for (let r = 0; r < reps; r++) {
      for (let i = 0; i < n; i++) perm[i] = i;
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(R.unif() * (i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
      for (const j of groups[g]) for (let i = 0; i < n; i++) Xp[j * n + i] = T.X[j * n + perm[i]];
      s += oobError(f, Xp, y) - base;
      for (const j of groups[g]) for (let i = 0; i < n; i++) Xp[j * n + i] = T.X[j * n + i];
    }
    out[g] = s / reps;
  }
  return out;
}

/**
 * Average pairwise correlation of the trees' outputs (class-`c` probability, or the value)
 * over a set of points (feature-major X with n points), using up to `maxTrees` trees.
 */
export function treeCorrelation(f: Forest, X: Num, n: number, c = 0, maxTrees = 40): number {
  const T = Math.min(f.trees.length, maxTrees), k = f.k;
  if (T < 2) return NaN;
  const out: Float64Array[] = [];
  for (let b = 0; b < T; b++) {
    const t = f.trees[b], v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = t.value[leafOfRow(t, X, n, i) * k + (f.regression ? 0 : c)];
    out.push(v);
  }
  let s = 0, m = 0;
  for (let a = 0; a < T; a++) for (let b = a + 1; b < T; b++) {
    const r = pearson(out[a], out[b]);
    if (Number.isFinite(r)) { s += r; m++; }
  }
  return m ? s / m : NaN;
}

function pearson(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN;
}

/** Class weights that balance the classes: w_c = n / (k · n_c) (0 for absent classes). */
export function balancedClassWeights(y: Num, k: number): Float64Array {
  const cnt = new Float64Array(k);
  for (let i = 0; i < y.length; i++) cnt[y[i]]++;
  const w = new Float64Array(k);
  for (let c = 0; c < k; c++) w[c] = cnt[c] ? y.length / (k * cnt[c]) : 0;
  return w;
}

// ---------------------------------------------------------------------------------------
// Adapters to the common classifier interface (lib/classifier.ts)


function trainingTarget(data: TrainingSet): { B: Binned; target: Target } {
  const T: Table = { n: data.n, d: data.d, X: data.X };
  return { B: binTable(T, 64), target: { kind: "class", y: data.y, k: data.k } };
}

function forestModel(f: Forest, X: Table, y: Int32Array): Model {
  let oob: number | null | undefined;
  return {
    k: f.k, d: f.d,
    predict: (Xn, n, out, from, to) => predictForest(f, Xn, n, undefined, out, from ?? 0, to ?? n),
    oobError: () => (oob === undefined ? (oob = f.trees.length > 1 && f.inbag.some(b => b.some(v => v === 0)) ? oobError(f, X.X, y) : null) : oob),
    importance: () => impurityImportance(f),
    summary: () => {
      const depths = f.trees.map(t => treeShape(t).depth);
      return `${f.trees.length} tree${f.trees.length > 1 ? "s" : ""}, mean depth ${(depths.reduce((a, b) => a + b, 0) / depths.length).toFixed(1)}`;
    },
  };
}

/** Random forest: trees, features per split m, maximum depth, minimum leaf size. */
export const FOREST_CLASSIFIER: Classifier = {
  id: "forest",
  name: "random forest",
  description: "Many deep trees on bootstrap samples, each split choosing among m random features; the trees vote.",
  params: [
    { key: "trees", label: "trees", min: 1, max: 100, step: 1, value: 30 },
    { key: "mtry", label: "features per split m", min: 1, max: 40, step: 1, value: 6, maxIsD: true },
    { key: "depth", label: "max depth", min: 1, max: 20, step: 1, value: 14 },
    { key: "minLeaf", label: "min leaf size", min: 1, max: 50, step: 1, value: 2 },
  ],
  train(data, p, seed) {
    const { B, target } = trainingTarget(data);
    const f = trainForest(B, target, data.w ?? null, {
      nTrees: Math.round(p.trees ?? 30), mtry: Math.round(p.mtry ?? 6), maxDepth: Math.round(p.depth ?? 14),
      minLeaf: Math.round(p.minLeaf ?? 2), seed,
    });
    return forestModel(f, B.table, data.y);
  },
};

/** One decision tree on all the data (no bootstrap, all features at each split). */
export const TREE_CLASSIFIER: Classifier = {
  id: "tree",
  name: "single tree",
  description: "One decision tree on all painted pixels, every feature tried at every split.",
  params: [
    { key: "depth", label: "max depth", min: 1, max: 20, step: 1, value: 14 },
    { key: "minLeaf", label: "min leaf size", min: 1, max: 50, step: 1, value: 2 },
  ],
  train(data, p, seed) {
    const { B, target } = trainingTarget(data);
    const f = trainForest(B, target, data.w ?? null, {
      nTrees: 1, bootstrap: false, maxDepth: Math.round(p.depth ?? 14), minLeaf: Math.round(p.minLeaf ?? 2), seed,
    });
    return forestModel(f, B.table, data.y);
  },
};
