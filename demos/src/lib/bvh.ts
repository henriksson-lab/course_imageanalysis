// Acceleration structures for ray tracing (owner: ray-tracing day, unit bvh/; no DOM).
// Works in 2D and in 3D (the dimension is the length of the box corners).
//
//  - `slab(o, d, min, max, t0, t1)`: the slab test (Kay & Kajiya). Per axis the ray is inside
//    the slab for t between (min − o)/d and (max − o)/d; the box is the intersection of the
//    slabs, so intersect the intervals. d = 0 gives ±∞ (IEEE), 0·∞ = NaN is ignored by the
//    comparisons (a NaN never wins a max/min). Returns [tnear, tfar] or null.
//  - `buildBVH(boxes, method)`: a binary tree of boxes over the primitives' bounding boxes,
//    split at the median of the longest axis ("median") or by the binned surface area
//    heuristic ("sah"); `boxArea` = surface area (3D) or perimeter (2D).
//  - `traverseBVH`: nearest child first, skip a box farther than the best hit; optional trace
//    of the visited nodes and tested primitives, counters. `refitBVH` after primitives move.
//  - `buildGrid` / `traverseGrid`: a uniform grid walked cell by cell along the ray (3D/2D DDA,
//    Amanatides & Woo), stopping in the first cell with a hit inside that cell.
//  - 2D primitives for the bvhstep demo: `rayDisc2`, `raySegment2`.
export interface AABB { min: number[]; max: number[] }

export interface BVHNode {
  min: number[]; max: number[];
  /** Children (−1 for a leaf). */
  left: number; right: number;
  /** Leaf: the primitives order[start .. start + count). */
  start: number; count: number;
  depth: number;
}
export interface BVH { nodes: BVHNode[]; order: Int32Array; dim: number }

export type TraceEvent =
  | { kind: "node"; id: number; hit: boolean; tnear: number }
  | { kind: "cell"; cell: number[]; prims: number }
  | { kind: "prim"; id: number; t: number };

export interface TraceResult { t: number; prim: number; boxTests: number; primTests: number; trace: TraceEvent[] }

/** Slab test. Returns [tnear, tfar] ∩ [t0, t1] or null. */
export function slab(o: ArrayLike<number>, d: ArrayLike<number>, min: ArrayLike<number>, max: ArrayLike<number>, t0 = 0, t1 = Infinity): [number, number] | null {
  let tn = t0, tf = t1;
  for (let k = 0; k < min.length; k++) {
    const inv = 1 / d[k];
    let a = (min[k] - o[k]) * inv, b = (max[k] - o[k]) * inv;
    if (a > b) { const s = a; a = b; b = s; }
    if (a > tn) tn = a;     // NaN never wins: (NaN > x) is false
    if (b < tf) tf = b;
    if (tn > tf) return null;
  }
  return [tn, tf];
}

export function unionBox(boxes: AABB[], idx: ArrayLike<number>, from = 0, to = idx.length): AABB {
  const dim = boxes[idx[from]].min.length;
  const min = Array(dim).fill(Infinity), max = Array(dim).fill(-Infinity);
  for (let i = from; i < to; i++) { const b = boxes[idx[i]]; for (let k = 0; k < dim; k++) { if (b.min[k] < min[k]) min[k] = b.min[k]; if (b.max[k] > max[k]) max[k] = b.max[k]; } }
  return { min, max };
}

/** Surface area of a box (3D) or its perimeter (2D): proportional to the chance that a random
 *  ray (line) hits it (Cauchy–Crofton). */
export function boxArea(min: ArrayLike<number>, max: ArrayLike<number>): number {
  const e = Array.from(min, (m, k) => Math.max(0, max[k] - m));
  if (e.length === 2) return 2 * (e[0] + e[1]);
  return 2 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]);
}

export interface BuildOptions {
  method: "median" | "sah";
  /** Largest leaf for the median split (default 2). */
  leafSize?: number;
  /** Number of SAH bins per axis (default 16). */
  bins?: number;
  /** SAH costs: one box test and one primitive test. */
  cTrav?: number; cIsect?: number;
  maxDepth?: number;
}

/** SAH cost of a split: C_trav + A_L/A · N_L · C_isect + A_R/A · N_R · C_isect. */
export const sahSplitCost = (A: number, AL: number, NL: number, AR: number, NR: number, cTrav = 1, cIsect = 1): number =>
  cTrav + (AL / A) * NL * cIsect + (AR / A) * NR * cIsect;

/** Build a BVH over the primitives' bounding boxes. */
export function buildBVH(boxes: AABB[], opt: BuildOptions = { method: "median" }): BVH {
  const n = boxes.length, dim = n ? boxes[0].min.length : 2;
  const order = new Int32Array(n).map((_, i) => i);
  const cen = boxes.map(b => b.min.map((m, k) => 0.5 * (m + b.max[k])));
  const nodes: BVHNode[] = [];
  const leafSize = opt.leafSize ?? 2, bins = opt.bins ?? 16, cT = opt.cTrav ?? 1, cI = opt.cIsect ?? 1, maxD = opt.maxDepth ?? 64;
  if (n === 0) return { nodes: [{ min: Array(dim).fill(0), max: Array(dim).fill(0), left: -1, right: -1, start: 0, count: 0, depth: 0 }], order, dim };
  const build = (start: number, end: number, depth: number): number => {
    const bb = unionBox(boxes, order, start, end), id = nodes.length, cnt = end - start;
    nodes.push({ min: bb.min, max: bb.max, left: -1, right: -1, start, count: cnt, depth });
    if (cnt <= 1 || depth >= maxD) return id;
    // centroid bounds
    const cmin = Array(dim).fill(Infinity), cmax = Array(dim).fill(-Infinity);
    for (let i = start; i < end; i++) { const c = cen[order[i]]; for (let k = 0; k < dim; k++) { cmin[k] = Math.min(cmin[k], c[k]); cmax[k] = Math.max(cmax[k], c[k]); } }
    let mid = -1;
    if (opt.method === "median") {
      if (cnt <= leafSize) return id;
      let ax = 0;
      for (let k = 1; k < dim; k++) if (cmax[k] - cmin[k] > cmax[ax] - cmin[ax]) ax = k;
      const sub = Array.from(order.subarray(start, end)).sort((a, b) => cen[a][ax] - cen[b][ax] || a - b);
      order.set(sub, start);
      mid = start + (cnt >> 1);
    } else {
      const A = boxArea(bb.min, bb.max);
      let best = cnt * cI, bestAx = -1, bestB = -1;   // cost of a leaf
      for (let ax = 0; ax < dim; ax++) {
        const ext = cmax[ax] - cmin[ax];
        if (!(ext > 0)) continue;
        const bc = new Int32Array(bins), bmin = Array.from({ length: bins }, () => Array(dim).fill(Infinity)), bmax = Array.from({ length: bins }, () => Array(dim).fill(-Infinity));
        for (let i = start; i < end; i++) {
          const p = order[i], b = Math.min(bins - 1, Math.floor((cen[p][ax] - cmin[ax]) / ext * bins));
          bc[b]++;
          for (let k = 0; k < dim; k++) { bmin[b][k] = Math.min(bmin[b][k], boxes[p].min[k]); bmax[b][k] = Math.max(bmax[b][k], boxes[p].max[k]); }
        }
        // sweep: areas and counts left of each bin boundary
        const lA = new Float64Array(bins), lN = new Int32Array(bins);
        { const mn = Array(dim).fill(Infinity), mx = Array(dim).fill(-Infinity); let c = 0;
          for (let b = 0; b < bins - 1; b++) { c += bc[b]; for (let k = 0; k < dim; k++) { mn[k] = Math.min(mn[k], bmin[b][k]); mx[k] = Math.max(mx[k], bmax[b][k]); } lA[b] = c ? boxArea(mn, mx) : 0; lN[b] = c; } }
        { const mn = Array(dim).fill(Infinity), mx = Array(dim).fill(-Infinity); let c = 0;
          for (let b = bins - 1; b >= 1; b--) { c += bc[b]; for (let k = 0; k < dim; k++) { mn[k] = Math.min(mn[k], bmin[b][k]); mx[k] = Math.max(mx[k], bmax[b][k]); }
            const nl = lN[b - 1];
            if (nl === 0 || c === 0) continue;
            const cost = sahSplitCost(A, lA[b - 1], nl, c ? boxArea(mn, mx) : 0, c, cT, cI);
            if (cost < best) { best = cost; bestAx = ax; bestB = b; } } }
      }
      if (bestAx < 0) {
        if (cnt <= leafSize * 4) return id;   // no useful split: leaf
        // all centroids coincide: split in half
        mid = start + (cnt >> 1);
      } else {
        const ext = cmax[bestAx] - cmin[bestAx];
        const left: number[] = [], right: number[] = [];
        for (let i = start; i < end; i++) { const p = order[i], b = Math.min(bins - 1, Math.floor((cen[p][bestAx] - cmin[bestAx]) / ext * bins)); (b < bestB ? left : right).push(p); }
        order.set(left.concat(right), start);
        mid = start + left.length;
      }
    }
    const l = build(start, mid, depth + 1), r = build(mid, end, depth + 1);
    nodes[id].left = l; nodes[id].right = r; nodes[id].count = 0;
    return id;
  };
  build(0, n, 0);
  return { nodes, order, dim };
}

/** Recompute every box bottom-up after the primitives moved (same tree). */
export function refitBVH(bvh: BVH, boxes: AABB[]): void {
  const fit = (id: number): void => {
    const nd = bvh.nodes[id];
    if (nd.left < 0) { if (nd.count) { const b = unionBox(boxes, bvh.order, nd.start, nd.start + nd.count); nd.min = b.min; nd.max = b.max; } return; }
    fit(nd.left); fit(nd.right);
    const L = bvh.nodes[nd.left], R = bvh.nodes[nd.right];
    nd.min = L.min.map((v, k) => Math.min(v, R.min[k])); nd.max = L.max.map((v, k) => Math.max(v, R.max[k]));
  };
  fit(0);
}

/** Depth of the tree (root = 0) and number of leaves. */
export function bvhStats(bvh: BVH): { depth: number; leaves: number; nodes: number } {
  let depth = 0, leaves = 0;
  for (const n of bvh.nodes) { depth = Math.max(depth, n.depth); if (n.left < 0) leaves++; }
  return { depth, leaves, nodes: bvh.nodes.length };
}

/** SAH cost of the whole tree (expected tests per random ray hitting the root). */
export function bvhCost(bvh: BVH, cTrav = 1, cIsect = 1): number {
  const root = bvh.nodes[0], A = boxArea(root.min, root.max);
  let c = 0;
  for (const n of bvh.nodes) {
    const p = boxArea(n.min, n.max) / A;
    c += n.left < 0 ? p * n.count * cIsect : p * 2 * cTrav;
  }
  return c;
}

/** Nearest hit through the BVH. `hitPrim(i, tmax)` returns the primitive's t in (0, tmax) or Infinity. */
export function traverseBVH(bvh: BVH, o: ArrayLike<number>, d: ArrayLike<number>, hitPrim: (i: number, tmax: number) => number, tmax = Infinity, keepTrace = false): TraceResult {
  const res: TraceResult = { t: tmax, prim: -1, boxTests: 0, primTests: 0, trace: [] };
  const nodes = bvh.nodes;
  res.boxTests++;
  const r0 = slab(o, d, nodes[0].min, nodes[0].max, 0, res.t);
  if (keepTrace) res.trace.push({ kind: "node", id: 0, hit: !!r0, tnear: r0 ? r0[0] : Infinity });
  if (!r0) return res;
  const stack: [number, number][] = [[0, r0[0]]];
  while (stack.length) {
    const [id, tn] = stack.pop()!;
    if (tn > res.t) continue;     // farther than the best hit so far
    const nd = nodes[id];
    if (nd.left < 0) {
      for (let i = nd.start; i < nd.start + nd.count; i++) {
        const p = bvh.order[i];
        res.primTests++;
        const t = hitPrim(p, res.t);
        if (keepTrace) res.trace.push({ kind: "prim", id: p, t });
        if (t < res.t) { res.t = t; res.prim = p; }
      }
      continue;
    }
    const kids: [number, number][] = [];
    for (const c of [nd.left, nd.right]) {
      res.boxTests++;
      const r = slab(o, d, nodes[c].min, nodes[c].max, 0, res.t);
      if (keepTrace) res.trace.push({ kind: "node", id: c, hit: !!r, tnear: r ? r[0] : Infinity });
      if (r) kids.push([c, r[0]]);
    }
    kids.sort((a, b) => b[1] - a[1]);   // push the far child first: the near one is popped next
    for (const k of kids) stack.push(k);
  }
  return res;
}

/** Brute force: test every primitive. */
export function bruteForce(n: number, hitPrim: (i: number, tmax: number) => number, tmax = Infinity): TraceResult {
  const res: TraceResult = { t: tmax, prim: -1, boxTests: 0, primTests: 0, trace: [] };
  for (let i = 0; i < n; i++) { res.primTests++; const t = hitPrim(i, res.t); if (t < res.t) { res.t = t; res.prim = i; } }
  return res;
}

// ------------------------------------------------------------------------------------------
// Uniform grid

export interface Grid { dim: number; n: number[]; min: number[]; max: number[]; cell: number[]; cells: number[][] }

export function buildGrid(boxes: AABB[], min: number[], max: number[], n: number[]): Grid {
  const dim = min.length, cell = min.map((m, k) => (max[k] - m) / n[k]);
  const total = n.reduce((a, b) => a * b, 1), cells: number[][] = Array.from({ length: total }, () => []);
  const clampI = (v: number, k: number) => Math.max(0, Math.min(n[k] - 1, Math.floor((v - min[k]) / cell[k])));
  boxes.forEach((b, p) => {
    const lo = b.min.map((v, k) => clampI(v, k)), hi = b.max.map((v, k) => clampI(v, k));
    const idx = lo.slice();
    for (;;) {
      let lin = 0;
      for (let k = dim - 1; k >= 0; k--) lin = lin * n[k] + idx[k];
      cells[lin].push(p);
      let k = 0;
      while (k < dim) { idx[k]++; if (idx[k] <= hi[k]) break; idx[k] = lo[k]; k++; }
      if (k === dim) break;
    }
  });
  return { dim, n, min, max, cell, cells };
}

export const gridIndex = (g: Grid, idx: number[]): number => { let lin = 0; for (let k = g.dim - 1; k >= 0; k--) lin = lin * g.n[k] + idx[k]; return lin; };

/** Walk the grid cells along the ray in order (DDA); test the primitives listed in each cell
 *  (each at most once: "mailboxing"); stop in the first cell that contains the nearest hit. */
export function traverseGrid(g: Grid, o: ArrayLike<number>, d: ArrayLike<number>, hitPrim: (i: number, tmax: number) => number, tmax = Infinity, keepTrace = false): TraceResult & { cells: number } {
  const res = { t: tmax, prim: -1, boxTests: 0, primTests: 0, trace: [] as TraceEvent[], cells: 0 };
  const r = slab(o, d, g.min, g.max, 0, tmax);
  if (!r) return res;
  const dim = g.dim, t0 = r[0];
  const idx: number[] = [], step: number[] = [], tNext: number[] = [], tDelta: number[] = [];
  for (let k = 0; k < dim; k++) {
    const p = o[k] + t0 * d[k];
    idx[k] = Math.max(0, Math.min(g.n[k] - 1, Math.floor((p - g.min[k]) / g.cell[k])));
    if (d[k] > 0) { step[k] = 1; tNext[k] = t0 + (g.min[k] + (idx[k] + 1) * g.cell[k] - p) / d[k]; tDelta[k] = g.cell[k] / d[k]; }
    else if (d[k] < 0) { step[k] = -1; tNext[k] = t0 + (g.min[k] + idx[k] * g.cell[k] - p) / d[k]; tDelta[k] = -g.cell[k] / d[k]; }
    else { step[k] = 0; tNext[k] = Infinity; tDelta[k] = Infinity; }
  }
  const tested = new Set<number>();
  for (;;) {
    res.cells++;
    const list = g.cells[gridIndex(g, idx)];
    if (keepTrace) res.trace.push({ kind: "cell", cell: idx.slice(), prims: list.length });
    let ax = 0;
    for (let k = 1; k < dim; k++) if (tNext[k] < tNext[ax]) ax = k;
    const tExit = tNext[ax];
    for (const p of list) {
      if (tested.has(p)) continue;
      tested.add(p);
      res.primTests++;
      const t = hitPrim(p, res.t);
      if (keepTrace) res.trace.push({ kind: "prim", id: p, t });
      if (t < res.t) { res.t = t; res.prim = p; }
    }
    if (res.t <= tExit || tExit > r[1]) break;   // nearest hit lies in this cell, or the ray left the grid
    idx[ax] += step[ax];
    if (idx[ax] < 0 || idx[ax] >= g.n[ax]) break;
    tNext[ax] += tDelta[ax];
  }
  return res;
}

// ------------------------------------------------------------------------------------------
// 2D primitives

/** Ray (o, d) against the disc (circle) centre c, radius r: smallest t in (0, tmax), or Infinity. */
export function rayDisc2(o: ArrayLike<number>, d: ArrayLike<number>, cx: number, cy: number, r: number, tmax = Infinity): number {
  const ox = o[0] - cx, oy = o[1] - cy, a = d[0] * d[0] + d[1] * d[1], b = ox * d[0] + oy * d[1], k = ox * ox + oy * oy - r * r;
  const disc = b * b - a * k;
  if (disc < 0) return Infinity;
  const s = Math.sqrt(disc), q = -(b + (b >= 0 ? s : -s));
  let t1 = q / a, t2 = q !== 0 ? k / q : -b / a;
  if (t1 > t2) [t1, t2] = [t2, t1];
  if (t1 > 0 && t1 < tmax) return t1;
  if (t2 > 0 && t2 < tmax) return t2;
  return Infinity;
}

/** Ray against the segment a–b: t in (0, tmax) or Infinity. */
export function raySegment2(o: ArrayLike<number>, d: ArrayLike<number>, ax: number, ay: number, bx: number, by: number, tmax = Infinity): number {
  const ex = bx - ax, ey = by - ay, den = d[0] * ey - d[1] * ex;
  if (Math.abs(den) < 1e-15) return Infinity;
  const wx = ax - o[0], wy = ay - o[1];
  const t = (wx * ey - wy * ex) / den, s = (wx * d[1] - wy * d[0]) / den;
  return s >= 0 && s <= 1 && t > 0 && t < tmax ? t : Infinity;
}
