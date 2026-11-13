// Minimum spanning trees and forests for segmentation, no DOM. Owner: the graph-based
// segmentation day (mst-seg/, ws-msf/, merge-trees/, superpixels/). Tests: demos/test/graphseg-test.ts.
//
// Edge weights here are DISSIMILARITIES d_pq (small = same region): |I_p − I_q|, a colour
// distance, or the altitude max(f_p, f_q) of a landscape f (watershed).
//
//  - DGraph, gridDissim(): pixel-grid edge lists (4/8 neighbours; grey, colour or altitude).
//  - sortEdges(): stable order of the edges by weight (ties: lower edge index first).
//  - kruskal(): minimum spanning forest, with a trace of every decision (small graphs).
//  - singleLinkage(): components of the edges with d < t (= MST minus its edges ≥ t).
//  - felzenszwalb(): Felzenszwalb–Huttenlocher (2004) with Int(C) + k/|C|, minimum-size pass;
//    fhTrace() records every test for the step-through demo.
//  - primForest(): seeded minimum spanning forest grown from markers (Meyer flooding on edges),
//    FIFO among equal weights; kruskalSeeded(): the same forest by Kruskal with a virtual root.
//  - treePath(): the path between two nodes in a forest and its heaviest edge (minimax path).
import { UnionFind } from "./label";
import { PriorityQueue } from "./pqueue";

/** Undirected graph with dissimilarities: edge k joins p[k] and q[k] with weight d[k]. */
export interface DGraph { n: number; p: Int32Array; q: Int32Array; d: Float64Array }

export type DissimMode = "absdiff" | "altitude";

/** Grid edges (right, down, and for 8 neighbours down-right, down-left) with weights from one
 *  or more channels: absdiff = Euclidean distance of the channel vectors (|ΔI| for grey),
 *  altitude = max(f_p, f_q) of the first channel. */
export function gridDissim(channels: ArrayLike<number>[], w: number, h: number, conn: 4 | 8 = 4, mode: DissimMode = "absdiff"): DGraph {
  const P: number[] = [], Q: number[] = [], D: number[] = [];
  const C = channels.length;
  const add = (a: number, b: number) => {
    P.push(a); Q.push(b);
    if (mode === "altitude") D.push(Math.max(channels[0][a], channels[0][b]));
    else if (C === 1) D.push(Math.abs(channels[0][a] - channels[0][b]));
    else { let s = 0; for (let c = 0; c < C; c++) s += (channels[c][a] - channels[c][b]) ** 2; D.push(Math.sqrt(s)); }
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (x + 1 < w) add(p, p + 1);
    if (y + 1 < h) add(p, p + w);
    if (conn === 8 && y + 1 < h) {
      if (x + 1 < w) add(p, p + w + 1);
      if (x > 0) add(p, p + w - 1);
    }
  }
  return { n: w * h, p: Int32Array.from(P), q: Int32Array.from(Q), d: Float64Array.from(D) };
}

/** Edge indices sorted by weight (stable: ties keep index order). */
export function sortEdges(G: DGraph): Uint32Array {
  const m = G.d.length, idx = new Uint32Array(m);
  for (let k = 0; k < m; k++) idx[k] = k;
  const d = G.d;
  idx.sort((a, b) => (d[a] - d[b]) || (a - b));
  return idx;
}

export interface KruskalStep { edge: number; join: boolean; /** sizes of the two components before */ sa: number; sb: number }
export interface KruskalResult { inTree: Uint8Array; total: number; count: number; steps: KruskalStep[] | null }

/** Kruskal's minimum spanning forest. trace = true records every edge decision. */
export function kruskal(G: DGraph, order: ArrayLike<number> = sortEdges(G), trace = false): KruskalResult {
  const uf = new UnionFind(G.n), inTree = new Uint8Array(G.d.length);
  let total = 0, count = 0;
  const steps: KruskalStep[] | null = trace ? [] : null;
  for (let i = 0; i < order.length; i++) {
    const k = order[i], a = uf.find(G.p[k]), b = uf.find(G.q[k]);
    const join = a !== b;
    if (steps) steps.push({ edge: k, join, sa: uf.sizeOf[a], sb: uf.sizeOf[b] });
    if (join) { uf.union(a, b); inTree[k] = 1; total += G.d[k]; count++; }
  }
  return { inTree, total, count, steps };
}

/** Consecutive labels 1..count from union–find roots (raster order of first node). */
export function ufLabels(uf: UnionFind, n: number): { labels: Int32Array; count: number } {
  const lab = new Int32Array(n), map = new Int32Array(n).fill(0);
  let next = 0;
  for (let v = 0; v < n; v++) {
    const r = uf.find(v);
    if (!map[r]) map[r] = ++next;
    lab[v] = map[r];
  }
  return { labels: lab, count: next };
}

/** Single linkage at threshold t: join across every edge with d < t. */
export function singleLinkage(G: DGraph, t: number): { labels: Int32Array; count: number } {
  const uf = new UnionFind(G.n);
  for (let k = 0; k < G.d.length; k++) if (G.d[k] < t) uf.union(G.p[k], G.q[k]);
  return ufLabels(uf, G.n);
}

export interface FHResult {
  labels: Int32Array; count: number;
  /** Per label (index = label): internal difference Int(C) and size |C|. */
  int: Float64Array; size: Int32Array;
}

/** Felzenszwalb–Huttenlocher: process edges in increasing order; merge C1, C2 across d if
 *  d ≤ min(Int(C1) + k/|C1|, Int(C2) + k/|C2|), Int = heaviest MST edge inside. Then merge
 *  components smaller than minSize with a neighbour across their cheapest edges (in edge order). */
export function felzenszwalb(G: DGraph, k: number, minSize = 0, order: ArrayLike<number> = sortEdges(G)): FHResult {
  const n = G.n, uf = new UnionFind(n), int = new Float64Array(n);
  for (let i = 0; i < order.length; i++) {
    const e = order[i];
    let a = uf.find(G.p[e]), b = uf.find(G.q[e]);
    if (a === b) continue;
    const d = G.d[e];
    if (d <= Math.min(int[a] + k / uf.sizeOf[a], int[b] + k / uf.sizeOf[b])) {
      const r = uf.union(a, b);
      int[r] = Math.max(d, int[a], int[b]);
    }
  }
  if (minSize > 1) {
    for (let i = 0; i < order.length; i++) {
      const e = order[i];
      const a = uf.find(G.p[e]), b = uf.find(G.q[e]);
      if (a !== b && (uf.sizeOf[a] < minSize || uf.sizeOf[b] < minSize)) {
        const r = uf.union(a, b);
        int[r] = Math.max(G.d[e], int[a], int[b]);
      }
    }
  }
  const { labels, count } = ufLabels(uf, n);
  const I = new Float64Array(count + 1), S = new Int32Array(count + 1);
  for (let v = 0; v < n; v++) { const r = uf.find(v); I[labels[v]] = int[r]; S[labels[v]]++; }
  return { labels, count, int: I, size: S };
}

export interface FHStep {
  edge: number; d: number; same: boolean; merge: boolean;
  intA: number; sizeA: number; thrA: number; intB: number; sizeB: number; thrB: number;
  /** Component representative of every node after this step. */
  comp: Int32Array;
}

/** FH with a record of every edge test (small graphs). */
export function fhTrace(G: DGraph, k: number, order: ArrayLike<number> = sortEdges(G)): FHStep[] {
  const n = G.n, uf = new UnionFind(n), int = new Float64Array(n), steps: FHStep[] = [];
  const snap = () => { const c = new Int32Array(n); for (let v = 0; v < n; v++) c[v] = uf.find(v); return c; };
  for (let i = 0; i < order.length; i++) {
    const e = order[i], d = G.d[e];
    const a = uf.find(G.p[e]), b = uf.find(G.q[e]);
    const ia = int[a], sa = uf.sizeOf[a], ib = int[b], sb = uf.sizeOf[b];
    const thrA = ia + k / sa, thrB = ib + k / sb;
    const same = a === b, merge = !same && d <= Math.min(thrA, thrB);
    if (merge) { const r = uf.union(a, b); int[r] = Math.max(d, ia, ib); }
    steps.push({ edge: e, d, same, merge, intA: ia, sizeA: sa, thrA, intB: ib, sizeB: sb, thrB, comp: snap() });
  }
  return steps;
}

/** Edges at each node: lists of edge ids (for Prim and paths). */
export function incidence(G: DGraph): { start: Int32Array; edges: Int32Array } {
  const n = G.n, m = G.p.length, cnt = new Int32Array(n + 1);
  for (let k = 0; k < m; k++) { cnt[G.p[k] + 1]++; cnt[G.q[k] + 1]++; }
  for (let v = 0; v < n; v++) cnt[v + 1] += cnt[v];
  const start = Int32Array.from(cnt), fill = Int32Array.from(cnt), edges = new Int32Array(2 * m);
  for (let k = 0; k < m; k++) { edges[fill[G.p[k]]++] = k; edges[fill[G.q[k]]++] = k; }
  return { start, edges };
}

export interface PrimStep { edge: number; from: number; to: number; d: number; kind: "grow" | "cut" | "skip"; label: number }
export interface PrimResult { labels: Int32Array; inForest: Uint8Array; cut: Uint8Array; steps: PrimStep[] | null }

/** Seeded minimum spanning forest by Prim from all markers at once (markers[v] = 0 or a label):
 *  repeatedly take the cheapest edge leaving the labelled set (FIFO among ties; markers queue
 *  their edges in raster order). An edge whose both ends are labelled differently when it is
 *  popped is a cut edge. Unreached nodes keep label 0. */
export function primForest(G: DGraph, markers: ArrayLike<number>, trace = false): PrimResult {
  const n = G.n, m = G.p.length, inc = incidence(G);
  const lab = new Int32Array(n), inForest = new Uint8Array(m), cut = new Uint8Array(m), queued = new Uint8Array(m);
  const q = new PriorityQueue(Math.max(64, m >> 2));
  const steps: PrimStep[] | null = trace ? [] : null;
  const pushEdges = (v: number) => {
    for (let j = inc.start[v]; j < inc.start[v + 1]; j++) {
      const e = inc.edges[j];
      if (queued[e]) continue;
      queued[e] = 1; q.push(e, G.d[e]);
    }
  };
  for (let v = 0; v < n; v++) if (markers[v] > 0) lab[v] = markers[v];
  for (let v = 0; v < n; v++) if (lab[v] > 0) pushEdges(v);
  while (q.size > 0) {
    const e = q.pop(), a = G.p[e], b = G.q[e];
    const la = lab[a], lb = lab[b];
    if (la > 0 && lb > 0) {
      const isCut = la !== lb;
      if (isCut) cut[e] = 1;
      steps?.push({ edge: e, from: a, to: b, d: G.d[e], kind: isCut ? "cut" : "skip", label: 0 });
      continue;
    }
    const from = la > 0 ? a : b, to = la > 0 ? b : a, L = lab[from];
    lab[to] = L; inForest[e] = 1;
    steps?.push({ edge: e, from, to, d: G.d[e], kind: "grow", label: L });
    pushEdges(to);
  }
  return { labels: lab, inForest, cut, steps };
}

/** The same seeded forest by Kruskal with a virtual root: sort all edges, join unless both
 *  components already carry (different) marker labels or are the same component. */
export function kruskalSeeded(G: DGraph, markers: ArrayLike<number>, order: ArrayLike<number> = sortEdges(G)): { labels: Int32Array; inForest: Uint8Array } {
  const n = G.n, uf = new UnionFind(n), mark = new Int32Array(n), inForest = new Uint8Array(G.d.length);
  for (let v = 0; v < n; v++) mark[v] = markers[v] > 0 ? markers[v] : 0;
  // markers with the same label are joined through the virtual root
  const first = new Map<number, number>();
  for (let v = 0; v < n; v++) if (mark[v]) { const f = first.get(mark[v]); if (f === undefined) first.set(mark[v], v); else { const r = uf.union(f, v); mark[r] = mark[v]; } }
  for (let i = 0; i < order.length; i++) {
    const e = order[i], a = uf.find(G.p[e]), b = uf.find(G.q[e]);
    if (a === b) continue;
    if (mark[a] && mark[b]) continue;
    const L = mark[a] || mark[b];
    const r = uf.union(a, b);
    mark[r] = L; inForest[e] = 1;
  }
  const labels = new Int32Array(n);
  for (let v = 0; v < n; v++) labels[v] = mark[uf.find(v)];
  return { labels, inForest };
}

/** Path between a and b along the edges with inTree[e] = 1 (BFS); null if not connected.
 *  Returns the edge ids in order from a to b and the heaviest one. */
export function treePath(G: DGraph, inTree: ArrayLike<number>, a: number, b: number): { edges: number[]; nodes: number[]; maxEdge: number } | null {
  const inc = incidence(G), prevE = new Int32Array(G.n).fill(-2);
  prevE[a] = -1;
  const queue = [a];
  for (let h = 0; h < queue.length; h++) {
    const v = queue[h];
    if (v === b) break;
    for (let j = inc.start[v]; j < inc.start[v + 1]; j++) {
      const e = inc.edges[j];
      if (!inTree[e]) continue;
      const u = G.p[e] === v ? G.q[e] : G.p[e];
      if (prevE[u] !== -2) continue;
      prevE[u] = e; queue.push(u);
    }
  }
  if (prevE[b] === -2) return null;
  const edges: number[] = [], nodes: number[] = [b];
  let v = b;
  while (v !== a) { const e = prevE[v]; edges.push(e); v = G.p[e] === v ? G.q[e] : G.p[e]; nodes.push(v); }
  edges.reverse(); nodes.reverse();
  let maxEdge = -1;
  for (const e of edges) if (maxEdge < 0 || G.d[e] > G.d[maxEdge]) maxEdge = e;
  return { edges, nodes, maxEdge };
}

/** Minimax distance between all pairs from node s: the smallest possible largest edge on a path
 *  (a Dijkstra with max instead of +). */
export function minimaxFrom(G: DGraph, s: number): Float64Array {
  const inc = incidence(G), dist = new Float64Array(G.n).fill(Infinity), done = new Uint8Array(G.n);
  const q = new PriorityQueue(64);
  dist[s] = -Infinity; q.push(s, -Infinity);
  while (q.size > 0) {
    const v = q.pop();
    if (done[v]) continue;
    done[v] = 1;
    for (let j = inc.start[v]; j < inc.start[v + 1]; j++) {
      const e = inc.edges[j], u = G.p[e] === v ? G.q[e] : G.p[e];
      const nd = Math.max(dist[v], G.d[e]);
      if (nd < dist[u]) { dist[u] = nd; q.push(u, nd); }
    }
  }
  return dist;
}
