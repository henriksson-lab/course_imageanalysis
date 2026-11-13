// Hierarchical segmentation, no DOM. Owner: the graph-based segmentation day (merge-trees/,
// superpixels/). Tests: demos/test/graphseg-test.ts.
//
// Leaves 0..n−1 are initial regions (pixels, basins, superpixels). A merge tree has 2n−1 nodes
// at most: node n + i is created by the i-th merge, at a level (altitude); parent[] links up.
//
//  - Rag / rag(): region adjacency graph of a label image (sizes, sums, minima, boundary length,
//    pass value = lowest altitude on the shared boundary, mean gradient along it).
//  - mergeTree(): a tree from an ordered list of merges (Kruskal order on RAG edges).
//  - altitudeTree(): single linkage on the pass values (the watershed hierarchy).
//  - attributeTree(): re-weight every merge of the altitude tree by the dynamics, area or volume
//    of the smaller of the two regions it joins (extinction values), then rebuild.
//  - meanDiffTree(): greedy RAG merging by |difference of means| (levels made non-decreasing).
//  - cutAt(), cutNodes(), ultrametric(), ucmImage(), dendrogramLayout().
import { UnionFind } from "./label";
import { Heap } from "./pqueue";

export interface Rag {
  n: number;
  size: Float64Array; sum: Float64Array; min: Float64Array;
  /** Edges a < b, boundary length (pixel pairs), pass = min over the boundary of max(alt_p, alt_q), gradient sum. */
  a: Int32Array; b: Int32Array; len: Float64Array; pass: Float64Array; gsum: Float64Array;
}

/** RAG of a label image with labels 1..n (0 is ignored); node i = label i + 1. alt = altitude
 *  image (e.g. the gradient) for pass values; img = the intensity for region means. 4-neighbours. */
export function rag(labels: ArrayLike<number>, w: number, h: number, img: ArrayLike<number>, alt: ArrayLike<number> = img): Rag {
  let n = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] > n) n = labels[i];
  const size = new Float64Array(n), sum = new Float64Array(n), min = new Float64Array(n).fill(Infinity);
  for (let p = 0; p < w * h; p++) {
    const l = labels[p] - 1;
    if (l < 0) continue;
    size[l]++; sum[l] += img[p];
    if (alt[p] < min[l]) min[l] = alt[p];
  }
  const map = new Map<number, number>();
  const A: number[] = [], B: number[] = [], len: number[] = [], pass: number[] = [], gs: number[] = [];
  const add = (p: number, q: number) => {
    let la = labels[p] - 1, lb = labels[q] - 1;
    if (la < 0 || lb < 0 || la === lb) return;
    if (la > lb) { const t = la; la = lb; lb = t; }
    const key = la * n + lb;
    let e = map.get(key);
    if (e === undefined) { e = A.length; map.set(key, e); A.push(la); B.push(lb); len.push(0); pass.push(Infinity); gs.push(0); }
    const v = Math.max(alt[p], alt[q]);
    len[e]++; gs[e] += v;
    if (v < pass[e]) pass[e] = v;
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (x + 1 < w) add(p, p + 1);
    if (y + 1 < h) add(p, p + w);
  }
  return { n, size, sum, min, a: Int32Array.from(A), b: Int32Array.from(B), len: Float64Array.from(len), pass: Float64Array.from(pass), gsum: Float64Array.from(gs) };
}

export interface MergeTree {
  /** Number of leaves. */
  n: number;
  /** parent[node] (−1 at the root(s)); nodes n.. are merges in order. */
  parent: Int32Array;
  /** level[node]: 0 at the leaves, the merge level at internal nodes. */
  level: Float64Array;
  /** Children of internal node n + i: left[i], right[i]. */
  left: Int32Array; right: Int32Array;
  /** Number of merges. */
  merges: number;
}

/** Build a tree from merges (leaf or region ids are leaves; merges join the current regions of a and b). */
export function mergeTree(n: number, merges: { a: number; b: number; level: number }[]): MergeTree {
  const total = n + merges.length;
  const parent = new Int32Array(total).fill(-1), level = new Float64Array(total);
  const left = new Int32Array(merges.length), right = new Int32Array(merges.length);
  const uf = new UnionFind(n), top = new Int32Array(n);
  for (let i = 0; i < n; i++) top[i] = i;
  let m = 0;
  for (const mg of merges) {
    const ra = uf.find(mg.a), rb = uf.find(mg.b);
    if (ra === rb) continue;
    const node = n + m, ta = top[ra], tb = top[rb];
    parent[ta] = node; parent[tb] = node; level[node] = mg.level;
    left[m] = ta; right[m] = tb;
    const r = uf.union(ra, rb);
    top[r] = node;
    m++;
  }
  return { n, parent: parent.slice(0, n + m), level: level.slice(0, n + m), left: left.slice(0, m), right: right.slice(0, m), merges: m };
}

/** RAG edge order by a key (stable). */
function orderBy(key: ArrayLike<number>): number[] {
  const idx = Array.from({ length: key.length }, (_, i) => i);
  idx.sort((x, y) => (key[x] - key[y]) || (x - y));
  return idx;
}

/** Single linkage on pass values: Kruskal order, level = pass. */
export function altitudeTree(R: Rag): { tree: MergeTree; edges: number[] } {
  const order = orderBy(R.pass), uf = new UnionFind(R.n), merges: { a: number; b: number; level: number }[] = [], edges: number[] = [];
  for (const e of order) {
    if (uf.same(R.a[e], R.b[e])) continue;
    uf.union(R.a[e], R.b[e]);
    merges.push({ a: R.a[e], b: R.b[e], level: R.pass[e] }); edges.push(e);
  }
  return { tree: mergeTree(R.n, merges), edges };
}

export type Attribute = "altitude" | "dynamics" | "area" | "volume";

/** Extinction re-weighting: flood in altitude order; when two regions meet at pass h, the merge
 *  gets the attribute of the smaller one: dynamics = h − its minimum, area = its size,
 *  volume = size × (h − minimum). Then the merges are re-sorted by this value (ties: altitude
 *  order). Returns the tree and, per RAG edge of the MST, its new value (NaN off the MST). */
export function attributeTree(R: Rag, attr: Attribute): { tree: MergeTree; value: Float64Array } {
  const value = new Float64Array(R.a.length).fill(NaN);
  if (attr === "altitude") {
    const { tree, edges } = altitudeTree(R);
    for (const e of edges) value[e] = R.pass[e];
    return { tree, value };
  }
  const order = orderBy(R.pass), uf = new UnionFind(R.n);
  const mn = Float64Array.from(R.min), sz = Float64Array.from(R.size);
  const mst: number[] = [];
  for (const e of order) {
    const a = uf.find(R.a[e]), b = uf.find(R.b[e]);
    if (a === b) continue;
    const h = R.pass[e];
    const att = (r: number) => attr === "dynamics" ? h - mn[r] : attr === "area" ? sz[r] : sz[r] * (h - mn[r]);
    value[e] = Math.min(att(a), att(b));
    const r = uf.union(a, b);
    mn[r] = Math.min(mn[a], mn[b]); sz[r] = sz[a] + sz[b];
    mst.push(e);
  }
  const rank = new Map<number, number>();
  mst.forEach((e, i) => rank.set(e, i));
  mst.sort((x, y) => (value[x] - value[y]) || (rank.get(x)! - rank.get(y)!));
  return { tree: mergeTree(R.n, mst.map(e => ({ a: R.a[e], b: R.b[e], level: value[e] }))), value };
}

/** Greedy RAG merging: always merge the adjacent pair with the smallest |mean difference|
 *  (lazy priority queue); stored levels are the running maximum, so the tree stays nested. */
export function meanDiffTree(R: Rag): MergeTree {
  const n = R.n, uf = new UnionFind(n), sum = Float64Array.from(R.sum), size = Float64Array.from(R.size);
  const nb: Map<number, number>[] = Array.from({ length: n }, () => new Map());
  for (let e = 0; e < R.a.length; e++) { nb[R.a[e]].set(R.b[e], 1); nb[R.b[e]].set(R.a[e], 1); }
  const ver = new Int32Array(n);
  type Item = { a: number; b: number; d: number; va: number; vb: number };
  const H = new Heap<Item>((x, y) => x.d < y.d || (x.d === y.d && (x.a < y.a || (x.a === y.a && x.b < y.b))));
  const diff = (a: number, b: number) => Math.abs(sum[a] / size[a] - sum[b] / size[b]);
  for (let e = 0; e < R.a.length; e++) H.push({ a: R.a[e], b: R.b[e], d: diff(R.a[e], R.b[e]), va: 0, vb: 0 });
  const merges: { a: number; b: number; level: number }[] = [];
  let lv = -Infinity;
  while (H.size > 0) {
    const it = H.pop()!;
    if (ver[it.a] !== it.va || ver[it.b] !== it.vb) continue;
    if (uf.find(it.a) !== it.a || uf.find(it.b) !== it.b) continue;
    lv = Math.max(lv, it.d);
    merges.push({ a: it.a, b: it.b, level: lv });
    const r = uf.union(it.a, it.b), o = r === it.a ? it.b : it.a;
    sum[r] += sum[o]; size[r] += size[o];
    for (const [k] of nb[o]) if (k !== r) { nb[r].set(k, 1); nb[k].delete(o); nb[k].set(r, 1); }
    nb[r].delete(o); nb[o].clear();
    ver[r]++;
    for (const [k] of nb[r]) {
      const a = Math.min(r, k), b = Math.max(r, k);
      H.push({ a, b, d: diff(a, b), va: ver[a], vb: ver[b] });
    }
  }
  return mergeTree(n, merges);
}

/** Leaf → region id (consecutive 1..count) after applying every merge with level ≤ t. */
export function cutAt(T: MergeTree, t: number): { labels: Int32Array; count: number } {
  const uf = new UnionFind(T.n + T.merges);
  for (let i = 0; i < T.merges; i++) if (T.level[T.n + i] <= t) { uf.union(T.n + i, T.left[i]); uf.union(T.n + i, T.right[i]); }
  const out = new Int32Array(T.n), map = new Map<number, number>();
  for (let v = 0; v < T.n; v++) { const r = uf.find(v); if (!map.has(r)) map.set(r, map.size + 1); out[v] = map.get(r)!; }
  return { labels: out, count: map.size };
}

/** The tree nodes that form the partition at level t (nodes with level ≤ t whose parent is above t or absent). */
export function cutNodes(T: MergeTree, t: number): number[] {
  const out: number[] = [];
  for (let v = 0; v < T.n + T.merges; v++) {
    if (T.level[v] > t) continue;
    const p = T.parent[v];
    if (p < 0 || T.level[p] > t) out.push(v);
  }
  return out;
}

/** Leaf → region id for an arbitrary set of nodes that covers every leaf exactly once. */
export function labelsFromNodes(T: MergeTree, nodes: number[]): Int32Array {
  const out = new Int32Array(T.n);
  nodes.forEach((v, i) => { for (const leaf of leavesOf(T, v)) out[leaf] = i + 1; });
  return out;
}

/** Leaves under a node. */
export function leavesOf(T: MergeTree, v: number): number[] {
  const out: number[] = [], st = [v];
  while (st.length) {
    const u = st.pop()!;
    if (u < T.n) out.push(u); else { st.push(T.left[u - T.n], T.right[u - T.n]); }
  }
  return out;
}

/** Ultrametric distance: the level of the lowest common ancestor (Infinity if in different trees). */
export function ultrametric(T: MergeTree, a: number, b: number): number {
  if (a === b) return 0;
  const anc = new Set<number>();
  for (let v = a; v >= 0; v = T.parent[v]) anc.add(v);
  for (let v = b; v >= 0; v = T.parent[v]) if (anc.has(v)) return T.level[v];
  return Infinity;
}

/** For every leaf pair given as RAG edges: the ultrametric (when the boundary disappears). */
export function ucmEdges(T: MergeTree, R: Rag): Float64Array {
  const out = new Float64Array(R.a.length);
  for (let e = 0; e < R.a.length; e++) out[e] = ultrametric(T, R.a[e], R.b[e]);
  return out;
}

/** UCM image: each pixel with a 4-neighbour of another leaf gets the largest ultrametric to such
 *  a neighbour (the boundary's saliency); 0 inside regions. labels are 1-based leaves. */
export function ucmImage(T: MergeTree, labels: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h), cache = new Map<number, number>();
  const u = (a: number, b: number) => {
    const key = a < b ? a * T.n + b : b * T.n + a;
    let v = cache.get(key);
    if (v === undefined) { v = ultrametric(T, a, b); cache.set(key, v); }
    return v;
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, a = labels[p] - 1;
    if (a < 0) continue;
    let m = 0;
    if (x + 1 < w && labels[p + 1] - 1 !== a && labels[p + 1] > 0) m = Math.max(m, u(a, labels[p + 1] - 1));
    if (y + 1 < h && labels[p + w] - 1 !== a && labels[p + w] > 0) m = Math.max(m, u(a, labels[p + w] - 1));
    if (x > 0 && labels[p - 1] - 1 !== a && labels[p - 1] > 0) m = Math.max(m, u(a, labels[p - 1] - 1));
    if (y > 0 && labels[p - w] - 1 !== a && labels[p - w] > 0) m = Math.max(m, u(a, labels[p - w] - 1));
    out[p] = Number.isFinite(m) ? m : 0;
  }
  return out;
}

/** Dendrogram coordinates: leaves in an order where branches do not cross (x = 0..n−1),
 *  internal nodes at the mean x of their children, y = level. Roots of a forest side by side. */
export function dendrogramLayout(T: MergeTree): { x: Float64Array; order: Int32Array } {
  const total = T.n + T.merges, x = new Float64Array(total), order: number[] = [];
  const roots: number[] = [];
  for (let v = 0; v < total; v++) if (T.parent[v] < 0) roots.push(v);
  roots.sort((a, b) => T.level[a] - T.level[b] || a - b);
  for (const r of roots) {
    const st: [number, boolean][] = [[r, false]];
    while (st.length) {
      const [v, post] = st.pop()!;
      if (v < T.n) { x[v] = order.length; order.push(v); continue; }
      const i = v - T.n;
      if (post) { x[v] = (x[T.left[i]] + x[T.right[i]]) / 2; continue; }
      st.push([v, true], [T.right[i], false], [T.left[i], false]);
    }
  }
  return { x, order: Int32Array.from(order) };
}
