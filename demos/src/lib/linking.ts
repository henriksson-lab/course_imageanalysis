// Linking detections into tracks (tracking day). No DOM.
//
// Frame to frame: costs (squared distance by default), gating, nearest neighbour, greedy and
// the global linear assignment with "end" (death) and "start" (birth) alternatives: as one
// square augmented matrix with lib/eval.ts `hungarian` (assignDense, the form taught), and as a
// sparse shortest-augmenting-path search (assignSparse/assignGlobal, used by the demos). Second stage: segment linking (gap closing, splitting,
// merging) as one more augmented assignment. Track graph helpers (segments, tracks between
// divisions, start/end reasons) and evaluation (link P/R/F1, division triplets, complete
// tracks). Also `matchPoints`, a gated optimal point matching for evaluation.
//
// Augmented matrix (m rows = objects at t, n columns = objects at t+1), N = m + n:
//
//            n targets          m death dummies
//   m      [ C (∞ if gated)  |  diag(d), ∞ off-diagonal ]
//   n      [ diag(b), ∞ off  |  0                       ]
//
// A link i→j costs c_ij; an unlinked row pays its death cost d_i, an unlinked column its
// birth cost b_j; the zero block lets the leftover dummies pair up. So a link is taken only
// if c_ij < d_i + b_j (with d = b: c < 2b).
import { hungarian } from "./eval";

export interface Pt { x: number; y: number }
export type CostKind = "sq" | "dist";

/** Cost of a candidate link between two points: squared distance (default) or distance. */
export function pairCost(a: Pt, b: Pt, kind: CostKind = "sq"): number {
  const d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  return kind === "sq" ? d2 : Math.sqrt(d2);
}

/** m × n cost matrix (row-major); pairs farther apart than `gate` are Infinity (not allowed). */
export function costMatrix(A: Pt[], B: Pt[], kind: CostKind = "sq", gate = Infinity): Float64Array {
  const m = A.length, n = B.length, C = new Float64Array(m * n), g2 = gate * gate;
  for (let i = 0; i < m; i++) {
    const ax = A[i].x, ay = A[i].y;
    for (let j = 0; j < n; j++) {
      const dx = ax - B[j].x, dy = ay - B[j].y, d2 = dx * dx + dy * dy;
      C[i * n + j] = d2 <= g2 ? (kind === "sq" ? d2 : Math.sqrt(d2)) : Infinity;
    }
  }
  return C;
}

/** Nearest neighbour: every row picks its cheapest allowed column (conflicts possible). -1 = none. */
export function nearestNeighbour(C: ArrayLike<number>, m: number, n: number): Int32Array {
  const succ = new Int32Array(m).fill(-1);
  for (let i = 0; i < m; i++) {
    let best = Infinity;
    for (let j = 0; j < n; j++) if (C[i * n + j] < best) { best = C[i * n + j]; succ[i] = j; }
  }
  return succ;
}

/** Columns chosen by more than one row under nearest neighbour. */
export function conflicts(succ: Int32Array): Set<number> {
  const seen = new Map<number, number>(), out = new Set<number>();
  for (const j of succ) if (j >= 0) { const c = (seen.get(j) ?? 0) + 1; seen.set(j, c); if (c > 1) out.add(j); }
  return out;
}

export interface Link { i: number; j: number; c: number }

/**
 * Greedy: take the cheapest remaining allowed pair, remove its row and column, repeat.
 * Pairs costing `maxCost` or more are not taken (with alternatives b: maxCost = 2b).
 * Returns the links in the order they were taken.
 */
export function greedy(C: ArrayLike<number>, m: number, n: number, maxCost = Infinity): Link[] {
  const cand: Link[] = [];
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    const c = C[i * n + j];
    if (Number.isFinite(c) && c < maxCost) cand.push({ i, j, c });
  }
  cand.sort((a, b) => a.c - b.c || a.i - b.i || a.j - b.j);
  const ur = new Uint8Array(m), uc = new Uint8Array(n), out: Link[] = [];
  for (const l of cand) if (!ur[l.i] && !uc[l.j]) { ur[l.i] = 1; uc[l.j] = 1; out.push(l); }
  return out;
}

const vec = (v: number | ArrayLike<number>, k: number): Float64Array => {
  const out = new Float64Array(k);
  for (let i = 0; i < k; i++) out[i] = typeof v === "number" ? v : v[i];
  return out;
};

/** The (m+n) × (m+n) augmented matrix (Infinity = not allowed). `death` per row, `birth` per column. */
export function augmentedMatrix(C: ArrayLike<number>, m: number, n: number,
  death: number | ArrayLike<number>, birth: number | ArrayLike<number>): Float64Array {
  const N = m + n, M = new Float64Array(N * N).fill(Infinity), d = vec(death, m), b = vec(birth, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) M[i * N + j] = C[i * n + j];
    M[i * N + n + i] = d[i];
  }
  for (let j = 0; j < n; j++) {
    M[(m + j) * N + j] = b[j];
    for (let i = 0; i < m; i++) M[(m + j) * N + n + i] = 0;
  }
  return M;
}

export interface Assignment {
  links: Link[];
  /** rows (objects at t) whose track ends */
  ends: number[];
  /** columns (objects at t+1) that start a new track */
  starts: number[];
  total: number;
}

/** Solve a square cost matrix with Infinity entries: replaced by a value larger than any feasible total. */
export function solveSquare(M: Float64Array, N: number): Int32Array {
  let s = 0;
  for (let k = 0; k < M.length; k++) if (Number.isFinite(M[k])) s += Math.abs(M[k]);
  const big = 2 * s + 1, W = new Float64Array(M.length);
  for (let k = 0; k < M.length; k++) W[k] = Number.isFinite(M[k]) ? M[k] : big;
  return hungarian(W, N);
}

/** Global assignment with end/start alternatives on one dense augmented matrix. */
export function assignDense(C: ArrayLike<number>, m: number, n: number,
  death: number | ArrayLike<number>, birth: number | ArrayLike<number>): Assignment {
  const N = m + n, d = vec(death, m), b = vec(birth, n);
  const out: Assignment = { links: [], ends: [], starts: [], total: 0 };
  if (N === 0) return out;
  const col = solveSquare(augmentedMatrix(C, m, n, d, b), N);
  const linkedCol = new Uint8Array(n);
  for (let i = 0; i < m; i++) {
    const j = col[i];
    if (j < n && Number.isFinite(C[i * n + j])) { out.links.push({ i, j, c: C[i * n + j] }); linkedCol[j] = 1; out.total += C[i * n + j]; }
    else { out.ends.push(i); out.total += d[i]; }
  }
  for (let j = 0; j < n; j++) if (!linkedCol[j]) { out.starts.push(j); out.total += b[j]; }
  return out;
}

/**
 * Sparse solver for the same problem. With c'_ij = c_ij − d_i − b_j the total is
 * Σd + Σb + Σ_links c'_ij, so only pairs with c' < 0 matter. Rows are inserted one at a time
 * (shortest augmenting path with column potentials, Dijkstra over a sparse row → column graph)
 * into a rectangular problem where every row also has its own "end" column of cost 0. Each
 * search stays local, so a frame of a few hundred objects takes about a millisecond.
 */
export function assignSparse(C: ArrayLike<number>, m: number, n: number,
  death: number | ArrayLike<number>, birth: number | ArrayLike<number>): Assignment {
  const d = vec(death, m), b = vec(birth, n);
  const nc = n + m;                                   // real columns, then one private column per row
  const start = new Int32Array(m + 1), cols: number[] = [], cst: number[] = [];
  for (let i = 0; i < m; i++) {
    start[i] = cols.length;
    for (let j = 0; j < n; j++) {
      const c = C[i * n + j];
      if (Number.isFinite(c) && c - d[i] - b[j] < 0) { cols.push(j); cst.push(c - d[i] - b[j]); }
    }
    cols.push(n + i); cst.push(0);
  }
  start[m] = cols.length;
  const v = new Float64Array(nc), rowOf = new Int32Array(nc).fill(-1), colOf = new Int32Array(m).fill(-1);
  const dist = new Float64Array(nc).fill(Infinity), pred = new Int32Array(nc), scanned = new Uint8Array(nc);
  const touched: number[] = [], scannedList: number[] = [];
  const hk: number[] = [], hv: number[] = [];
  const push = (k: number, x: number) => {
    let i = hk.length; hk.push(k); hv.push(x);
    while (i > 0) { const p = (i - 1) >> 1; if (hk[p] <= k) break; hk[i] = hk[p]; hv[i] = hv[p]; i = p; }
    hk[i] = k; hv[i] = x;
  };
  const pop = (): number => {
    const top = hv[0], k = hk.pop()!, x = hv.pop()!;
    if (hk.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= hk.length) break;
        if (c + 1 < hk.length && hk[c + 1] < hk[c]) c++;
        if (hk[c] >= k) break;
        hk[i] = hk[c]; hv[i] = hv[c]; i = c;
      }
      hk[i] = k; hv[i] = x;
    }
    return top;
  };
  const relax = (r: number, base: number) => {
    for (let e = start[r]; e < start[r + 1]; e++) {
      const c = cols[e];
      if (scanned[c]) continue;
      const nd = base + cst[e] - v[c];
      if (nd < dist[c]) { if (dist[c] === Infinity) touched.push(c); dist[c] = nd; pred[c] = r; push(nd, c); }
    }
  };
  for (let i = 0; i < m; i++) {
    hk.length = 0; hv.length = 0;
    relax(i, 0);
    let sink = -1, last = 0;
    while (hk.length) {
      const key = hk[0], c = pop();
      if (scanned[c] || key > dist[c]) continue;
      scanned[c] = 1; scannedList.push(c);
      if (rowOf[c] < 0) { sink = c; last = dist[c]; break; }
      const r = rowOf[c];
      // reduced cost of the matched edge (r, c) is 0: u_r = cost(r, c) − v_c
      let cm = 0;
      for (let e = start[r]; e < start[r + 1]; e++) if (cols[e] === c) { cm = cst[e]; break; }
      relax(r, dist[c] - (cm - v[c]));
    }
    // potentials of the columns scanned before the sink
    for (const c of scannedList) if (c !== sink) v[c] += dist[c] - last;
    // augment along the path
    for (let c = sink; ;) {
      const r = pred[c], prevC = colOf[r];
      rowOf[c] = r; colOf[r] = c;
      if (r === i) break;
      c = prevC;
    }
    for (const c of touched) { dist[c] = Infinity; scanned[c] = 0; }
    for (const c of scannedList) scanned[c] = 0;
    touched.length = 0; scannedList.length = 0;
  }
  const out: Assignment = { links: [], ends: [], starts: [], total: 0 };
  const linkedC = new Uint8Array(n);
  for (let i = 0; i < m; i++) {
    const j = colOf[i];
    if (j >= 0 && j < n) { out.links.push({ i, j, c: C[i * n + j] }); linkedC[j] = 1; out.total += C[i * n + j]; }
    else { out.ends.push(i); out.total += d[i]; }
  }
  for (let j = 0; j < n; j++) if (!linkedC[j]) { out.starts.push(j); out.total += b[j]; }
  return out;
}

/**
 * Global assignment with alternatives: the minimum of (sum of link costs + d per row not
 * linked + b per column not linked). Same total as `assignDense` on the augmented matrix
 * (checked against brute force in the tests), solved with the sparse row-by-row search.
 */
export function assignGlobal(C: ArrayLike<number>, m: number, n: number,
  death: number | ArrayLike<number>, birth: number | ArrayLike<number>): Assignment {
  const out = assignSparse(C, m, n, death, birth);
  out.links.sort((a, b) => a.i - b.i);
  return out;
}

/** Total cost of a set of links with alternatives (rows not linked pay d, columns not linked pay b). */
export function totalCost(links: Link[], m: number, n: number, d: number, b: number): number {
  const rows = new Set(links.map(l => l.i)), cols = new Set(links.map(l => l.j));
  return links.reduce((s, l) => s + l.c, 0) + (m - rows.size) * d + (n - cols.size) * b;
}

/**
 * Optimal one-to-one matching of points within `maxDist` (evaluation of detections against a
 * ground truth): the largest number of pairs, and among those the smallest sum of squared
 * distances. Returns pairs [i in a, j in b].
 */
export function matchPoints(a: Pt[], b: Pt[], maxDist: number): [number, number][] {
  const C = costMatrix(a, b, "sq", maxDist);
  const alt = (maxDist * maxDist + 1) * (a.length + b.length + 1);
  return assignGlobal(C, a.length, b.length, alt, alt).links.map(l => [l.i, l.j] as [number, number]);
}

// ---------------------------------------------------------------------------------------
// Detections over time and the track graph

export interface Det { t: number; x: number; y: number; area?: number }
export type EdgeKind = "link" | "gap" | "split" | "merge";
/** An edge of the track graph between detection indices a (earlier) and b (later). */
export interface Edge { a: number; b: number; kind: EdgeKind; cost?: number }
export type LinkMethod = "nn" | "greedy" | "global";

/** Detection indices per frame (frames 0..T-1). */
export function byFrame(dets: Det[], T?: number): number[][] {
  const nT = T ?? (dets.reduce((m, d) => Math.max(m, d.t), -1) + 1);
  const out: number[][] = Array.from({ length: nT }, () => []);
  dets.forEach((d, k) => { if (d.t >= 0 && d.t < nT) out[d.t].push(k); });
  return out;
}

export interface FrameLinkOptions {
  method: LinkMethod;
  /** maximum link distance */
  gate: number;
  /** end/start cost (global; greedy uses it as maxCost = 2b) */
  b: number;
  cost?: CostKind;
  /** position used for the earlier detection (e.g. drift-corrected, or a prediction) */
  posA?: (k: number) => Pt;
  /** position used for the later detection */
  posB?: (k: number) => Pt;
}

/** Link every frame to the next. Nearest neighbour may give two links into one detection. */
export function linkFrames(dets: Det[], frames: number[][], o: FrameLinkOptions): Edge[] {
  const edges: Edge[] = [];
  const pa = o.posA ?? ((k: number) => dets[k]), pb = o.posB ?? ((k: number) => dets[k]);
  for (let t = 0; t + 1 < frames.length; t++) {
    const A = frames[t], B = frames[t + 1];
    const C = costMatrix(A.map(pa), B.map(pb), o.cost ?? "sq", o.gate);
    const m = A.length, n = B.length;
    if (o.method === "nn") {
      nearestNeighbour(C, m, n).forEach((j, i) => { if (j >= 0) edges.push({ a: A[i], b: B[j], kind: "link", cost: C[i * n + j] }); });
    } else {
      const links = o.method === "greedy" ? greedy(C, m, n, 2 * o.b) : assignGlobal(C, m, n, o.b, o.b).links;
      for (const l of links) edges.push({ a: A[l.i], b: B[l.j], kind: "link", cost: l.c });
    }
  }
  return edges;
}

export interface Adjacency { out: number[][]; inn: number[][] }
export function adjacency(nDets: number, edges: Edge[]): Adjacency {
  const out: number[][] = Array.from({ length: nDets }, () => []), inn: number[][] = Array.from({ length: nDets }, () => []);
  edges.forEach((e, k) => { out[e.a].push(k); inn[e.b].push(k); });
  return { out, inn };
}

/**
 * Tracks: maximal unbranched chains of the graph. A chain continues from a to b only if a has
 * one outgoing and b one incoming edge (of any kind in `follow`). Returns the track id of every
 * detection and the node lists (in time order).
 */
export function chains(nDets: number, edges: Edge[], follow: EdgeKind[] = ["link", "gap"]): { id: Int32Array; tracks: number[][] } {
  const ok = edges.map(e => follow.includes(e.kind));
  const adj = adjacency(nDets, edges);
  const next = new Int32Array(nDets).fill(-1), prev = new Int32Array(nDets).fill(-1);
  for (let k = 0; k < nDets; k++) {
    if (adj.out[k].length === 1) {
      const e = edges[adj.out[k][0]];
      if (ok[adj.out[k][0]] && adj.inn[e.b].length === 1) { next[k] = e.b; prev[e.b] = k; }
    }
  }
  const id = new Int32Array(nDets).fill(-1), tracks: number[][] = [];
  for (let k = 0; k < nDets; k++) if (prev[k] < 0) {
    const tr: number[] = [];
    for (let c = k; c >= 0; c = next[c]) { id[c] = tracks.length; tr.push(c); }
    tracks.push(tr);
  }
  return { id, tracks };
}

// ---------------------------------------------------------------------------------------
// Second stage: segment linking (gap closing, splitting, merging)

export interface SegmentOptions {
  /** largest number of missed frames a gap may bridge (0 = no gap closing) */
  maxGap: number;
  /** maximum distance per √(frames elapsed) for gap closing; for splits and merges per frame */
  gate: number;
  /** alternative cost of every segment end, start and split/merge candidate */
  b: number;
  splitting: boolean;
  merging: boolean;
  /** daughters (and pieces) must last at least this many frames, else the split is undone */
  minDaughter?: number;
}

export interface Candidate { a: number; b: number; kind: "gap" | "split" | "merge"; cost: number }

export interface SegmentResult {
  /** the frame-to-frame edges plus the chosen gap/split/merge edges */
  edges: Edge[];
  candidates: Candidate[];
  chosen: Candidate[];
  /** split edges removed by the minDaughter rule (and the merges of the same pieces) */
  removed: Candidate[];
}

const areaFactor = (whole: number | undefined, p1: number | undefined, p2: number | undefined): number => {
  if (!whole || !p1 || !p2) return 1;
  const r = whole / (p1 + p2);
  return Math.max(r, 1 / r) ** 2;
};

/**
 * Join frame-to-frame segments with one global assignment over segment ends (rows) and starts
 * (columns), plus split candidates (rows: a detection in the middle of a segment can send a
 * second link to a start in the next frame) and merge candidates (columns: an end can join the
 * middle of another segment in the next frame). Gap cost: d² / frames elapsed; split and merge
 * cost: d² × an area-mismatch factor (1 if the areas add up).
 */
export function linkSegments(dets: Det[], frameEdges: Edge[], o: SegmentOptions): SegmentResult {
  const nD = dets.length;
  const { tracks } = chains(nD, frameEdges, ["link"]);
  const adj = adjacency(nD, frameEdges);
  const nextOf = (k: number) => (adj.out[k].length ? frameEdges[adj.out[k][0]].b : -1);
  const prevOf = (k: number) => (adj.inn[k].length ? frameEdges[adj.inn[k][0]].a : -1);
  const ends = tracks.map(tr => tr[tr.length - 1]), starts = tracks.map(tr => tr[0]);
  const segOf = new Int32Array(nD).fill(-1);
  tracks.forEach((tr, s) => tr.forEach(k => { segOf[k] = s; }));
  const frames = byFrame(dets);
  const d2 = (a: number, b: number) => (dets[a].x - dets[b].x) ** 2 + (dets[a].y - dets[b].y) ** 2;
  const cands: Candidate[] = [];
  // gap closing
  if (o.maxGap > 0) for (const e of ends) for (const s of starts) {
    if (segOf[e] === segOf[s]) continue;
    const g = dets[s].t - dets[e].t;
    if (g < 2 || g > o.maxGap + 1) continue;
    const dd = d2(e, s);
    if (dd <= o.gate * o.gate * g) cands.push({ a: e, b: s, kind: "gap", cost: dd / g });
  }
  // splitting: middle detection k at t-1 → start s at t
  if (o.splitting) for (const s of starts) {
    const t = dets[s].t;
    if (t < 1) continue;
    for (const k of frames[t - 1] ?? []) {
      const nk = nextOf(k);
      if (nk < 0 || segOf[k] === segOf[s]) continue;
      const dd = d2(k, s);
      if (dd <= o.gate * o.gate) cands.push({ a: k, b: s, kind: "split", cost: dd * areaFactor(dets[k].area, dets[s].area, dets[nk].area) });
    }
  }
  // merging: end e at t → middle detection k at t+1
  if (o.merging) for (const e of ends) {
    const t = dets[e].t;
    for (const k of frames[t + 1] ?? []) {
      const pk = prevOf(k);
      if (pk < 0 || segOf[k] === segOf[e]) continue;
      const dd = d2(e, k);
      if (dd <= o.gate * o.gate) cands.push({ a: e, b: k, kind: "merge", cost: dd * areaFactor(dets[k].area, dets[e].area, dets[pk].area) });
    }
  }
  // rows: ends, then split sources; columns: starts, then merge targets
  const rowIdx = new Map<string, number>(), colIdx = new Map<string, number>();
  const rowKey = (c: Candidate) => (c.kind === "split" ? "m" + c.a : "e" + c.a);
  const colKey = (c: Candidate) => (c.kind === "merge" ? "m" + c.b : "s" + c.b);
  for (const c of cands) {
    if (!rowIdx.has(rowKey(c))) rowIdx.set(rowKey(c), rowIdx.size);
    if (!colIdx.has(colKey(c))) colIdx.set(colKey(c), colIdx.size);
  }
  const m = rowIdx.size, n = colIdx.size, C = new Float64Array(m * n).fill(Infinity);
  for (const c of cands) {
    const k = rowIdx.get(rowKey(c))! * n + colIdx.get(colKey(c))!;
    C[k] = Math.min(C[k], c.cost);
  }
  const res = assignGlobal(C, m, n, o.b, o.b);
  let chosen: Candidate[] = [];
  for (const l of res.links) {
    const c = cands.find(c => rowIdx.get(rowKey(c)) === l.i && colIdx.get(colKey(c)) === l.j && c.cost === l.c);
    if (c) chosen.push(c);
  }
  const removed: Candidate[] = [];
  if (o.minDaughter && o.minDaughter > 1) {
    // a split is kept only if the new branch lasts minDaughter frames before it ends or merges
    // back (a branch cut short by the end of the movie is kept: it cannot be judged)
    const all: Edge[] = [...frameEdges, ...chosen.map(c => ({ a: c.a, b: c.b, kind: c.kind } as Edge))];
    const { id, tracks: tr2 } = chains(nD, all, ["link", "gap"]);
    const tLast = dets.reduce((m, d) => Math.max(m, d.t), 0);
    for (const c of chosen) if (c.kind === "split") {
      const branch = tr2[id[c.b]];
      const end = dets[branch[branch.length - 1]].t, len = end - dets[branch[0]].t + 1;
      if (len < o.minDaughter && end < tLast) {
        removed.push(c);
        const last = branch[branch.length - 1];
        for (const c2 of chosen) if (c2.kind === "merge" && c2.a === last) removed.push(c2);
      }
    }
    chosen = chosen.filter(c => !removed.includes(c));
  }
  const edges: Edge[] = [...frameEdges, ...chosen.map(c => ({ a: c.a, b: c.b, kind: c.kind, cost: c.cost } as Edge))];
  return { edges, candidates: cands, chosen, removed };
}

// ---------------------------------------------------------------------------------------
// Evaluation

export interface Scores { tp: number; fp: number; fn: number; precision: number; recall: number; f1: number }
export function scores(tp: number, fp: number, fn: number): Scores {
  const precision = tp + fp ? tp / (tp + fp) : 1, recall = tp + fn ? tp / (tp + fn) : 1;
  return { tp, fp, fn, precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 };
}

const key = (a: number, b: number) => a * 1048576 + b;

/** Links compared as detection pairs (the kind is ignored: a gap link counts like any link). */
export function evalLinks(result: Edge[], truth: Edge[]): Scores & { wrong: boolean[] } {
  const T = new Set(truth.map(e => key(e.a, e.b))), R = new Set<number>();
  let tp = 0, fp = 0;
  const wrong = result.map(e => {
    const k = key(e.a, e.b);
    if (R.has(k)) return false;
    R.add(k);
    if (T.has(k)) { tp++; return false; }
    fp++; return true;
  });
  return { ...scores(tp, fp, T.size - tp), wrong };
}

export interface Division { mother: number; daughters: [number, number] }
/** Divisions: detections with two outgoing edges (daughters sorted). */
export function divisions(nDets: number, edges: Edge[]): Division[] {
  const adj = adjacency(nDets, edges), out: Division[] = [];
  for (let k = 0; k < nDets; k++) if (adj.out[k].length === 2) {
    const d = adj.out[k].map(e => edges[e].b).sort((a, b) => a - b);
    out.push({ mother: k, daughters: [d[0], d[1]] });
  }
  return out;
}

/** Division triplets: correct only if the mother and both daughters match. */
export function evalDivisions(result: Division[], truth: Division[]): Scores {
  const k3 = (d: Division) => `${d.mother}:${d.daughters[0]}:${d.daughters[1]}`;
  const T = new Set(truth.map(k3));
  const tp = result.filter(d => T.has(k3(d))).length;
  return scores(tp, result.length - tp, truth.length - tp);
}

/** Fraction of reference tracks (unbranched chains of the truth graph) whose every link is in the result. */
export function completeTracks(nDets: number, result: Edge[], truth: Edge[]): { complete: number; total: number } {
  const R = new Set(result.map(e => key(e.a, e.b)));
  const { tracks } = chains(nDets, truth, ["link", "gap", "split", "merge"]);
  let complete = 0, total = 0;
  for (const tr of tracks) {
    if (tr.length < 2) continue;
    total++;
    let ok = true;
    for (let i = 0; i + 1 < tr.length && ok; i++) ok = R.has(key(tr[i], tr[i + 1]));
    if (ok) complete++;
  }
  return { complete, total };
}

export type EndReason = "first frame" | "last frame" | "border" | "gap" | "division" | "merge" | "unexplained";

/** Why each track (unbranched chain) starts and ends, from the graph and the field size. */
export function reasons(dets: Det[], edges: Edge[], T: number, W: number, H: number, border: number, axes: "xy" | "x" = "xy"):
  { tracks: number[][]; start: EndReason[]; end: EndReason[] } {
  const { tracks } = chains(dets.length, edges, ["link", "gap"]);
  const adj = adjacency(dets.length, edges);
  const nearBorder = (k: number) => dets[k].x < border || dets[k].x > W - border ||
    (axes === "xy" && (dets[k].y < border || dets[k].y > H - border));
  const start = tracks.map(tr => {
    const k = tr[0];
    const inn = adj.inn[k].map(e => edges[e]);
    if (inn.some(e => e.kind === "split")) return "division";
    if (inn.length > 1 || inn.some(e => e.kind === "merge")) return "merge";
    if (inn.length === 1) return adj.out[inn[0].a].length > 1 ? "division" : "gap";
    if (dets[k].t === 0) return "first frame";
    if (nearBorder(k)) return "border";
    return "unexplained";
  }) as EndReason[];
  const end = tracks.map(tr => {
    const k = tr[tr.length - 1];
    const out = adj.out[k].map(e => edges[e]);
    if (out.length > 1 || out.some(e => e.kind === "split")) return "division";
    if (out.some(e => e.kind === "merge") || (out.length === 1 && adj.inn[out[0].b].length > 1)) return "merge";
    if (out.length === 1) return "gap";
    if (dets[k].t === T - 1) return "last frame";
    if (nearBorder(k)) return "border";
    return "unexplained";
  }) as EndReason[];
  return { tracks, start, end };
}
