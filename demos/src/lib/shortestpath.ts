// Shortest paths on the pixel grid (no DOM). Owner: the dynamic-programming day
// (livewire/, fastmarching/); users: skeleton (length along a centreline), watershed
// (comparison), level sets (arrival times).
//
//  - dijkstraGrid: Dijkstra on the implicit 4- or 8-neighbour grid graph with any
//    non-negative edge cost w(p, q, len), from one or more sources with start values.
//    Returns the distances, the predecessor of every pixel (a shortest-path tree) and the
//    order in which pixels became final. Uses lib/pqueue.ts with lazy deletion (a pixel
//    is pushed again when its distance improves; stale entries are skipped).
//  - dijkstraTrace: the same, recording every pop with the queue contents (step-through).
//  - fastMarching: Sethian's fast marching for |∇T| = P (4 neighbours, first-order upwind
//    update): Dijkstra's order with the update from the quadratic
//    (T − a)² + (T − b)² = P² instead of min(d(u) + w).
//  - descendPath: minimal path by gradient descent on an arrival-time map T.
import { PriorityQueue } from "./pqueue";

export type Conn = 4 | 8;

/** Edge cost of the move p → q (flat indices) of length `len` (1, or √2 for diagonals). */
export type EdgeCost = (p: number, q: number, len: number) => number;

export interface DijkstraOptions {
  width: number;
  height: number;
  conn: Conn;
  edge: EdgeCost;
  /** Source pixels, with their start value (default 0). */
  sources: (number | { index: number; value: number })[];
  /** Stop as soon as this pixel is final. */
  target?: number;
  /** Stop when the smallest tentative distance exceeds this. */
  maxDist?: number;
}

export interface ShortestPathResult {
  /** Distance of every pixel (∞ if not reached / not final when stopped early). */
  dist: Float64Array;
  /** Predecessor in the shortest-path tree (−1 for sources and unreached pixels). */
  pred: Int32Array;
  /** Pixels in the order they became final. */
  order: Int32Array;
  /** Edge relaxations performed. */
  relaxations: number;
}

/** Neighbour offsets [dx, dy, length]: 4-neighbours first (left, right, up, down), then diagonals. */
export function gridNeighbours(conn: Conn): [number, number, number][] {
  const n4: [number, number, number][] = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1]];
  if (conn === 4) return n4;
  const r = Math.SQRT2;
  return [...n4, [-1, -1, r], [1, -1, r], [-1, 1, r], [1, 1, r]];
}

/** Edge cost "pay the cost of the pixel you enter" (vertex weights), optionally times the step length. */
export function vertexCost(C: ArrayLike<number>, withLength = false): EdgeCost {
  return withLength ? (_p, q, len) => C[q] * len : (_p, q) => C[q];
}

const srcOf = (s: number | { index: number; value: number }) => (typeof s === "number" ? { index: s, value: 0 } : s);

export function dijkstraGrid(o: DijkstraOptions): ShortestPathResult {
  const { width: w, height: h } = o, n = w * h;
  const dist = new Float64Array(n).fill(Infinity), pred = new Int32Array(n).fill(-1);
  const done = new Uint8Array(n), order = new Int32Array(n), nb = gridNeighbours(o.conn);
  const pq = new PriorityQueue(1024);
  for (const s0 of o.sources) {
    const s = srcOf(s0);
    if (s.value < dist[s.index]) { dist[s.index] = s.value; pq.push(s.index, s.value); }
  }
  let k = 0, relax = 0;
  const m = nb.length, NX = Int32Array.from(nb, v => v[0]), NY = Int32Array.from(nb, v => v[1]), NL = Float64Array.from(nb, v => v[2]);
  const edge = o.edge, maxDist = o.maxDist ?? Infinity, target = o.target ?? -1;
  while (!pq.isEmpty()) {
    const d = pq.peekPriority(), p = pq.pop();
    if (done[p] || d > dist[p]) continue;
    if (d > maxDist) break;
    done[p] = 1; order[k++] = p;
    if (p === target) break;
    const x = p % w, y = (p - x) / w;
    for (let e = 0; e < m; e++) {
      const xx = x + NX[e], yy = y + NY[e];
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const q = yy * w + xx;
      if (done[q]) continue;
      relax++;
      const nd = d + edge(p, q, NL[e]);
      if (nd < dist[q]) { dist[q] = nd; pred[q] = p; pq.push(q, nd); }
    }
  }
  for (let q = 0; q < n; q++) if (!done[q]) { dist[q] = Infinity; pred[q] = -1; }
  return { dist, pred, order: order.slice(0, k), relaxations: relax };
}

export interface DijkstraEvent {
  /** Pixel made final. */
  pop: number;
  /** Its distance. */
  d: number;
  /** Neighbours whose tentative distance improved: [pixel, new distance]. */
  updated: [number, number][];
  /** Queue after this step, in the order it would be popped: [pixel, tentative distance]. */
  queue: [number, number][];
}

/** Dijkstra with a record of every step (small grids only: the queue is copied each time). */
export function dijkstraTrace(o: DijkstraOptions): { result: ShortestPathResult; events: DijkstraEvent[]; initialQueue: [number, number][] } {
  const { width: w, height: h } = o, n = w * h;
  const dist = new Float64Array(n).fill(Infinity), pred = new Int32Array(n).fill(-1);
  const done = new Uint8Array(n), pushOrder = new Float64Array(n), nb = gridNeighbours(o.conn);
  let counter = 0;
  const pq = new PriorityQueue(64);
  const push = (q: number, d: number) => { pushOrder[q] = counter++; pq.push(q, d); };
  for (const s0 of o.sources) { const s = srcOf(s0); if (s.value < dist[s.index]) { dist[s.index] = s.value; push(s.index, s.value); } }
  const snapshot = (): [number, number][] => {
    const q: number[] = [];
    for (let p = 0; p < n; p++) if (!done[p] && dist[p] < Infinity) q.push(p);
    q.sort((a, b) => dist[a] - dist[b] || pushOrder[a] - pushOrder[b]);
    return q.map(p => [p, dist[p]]);
  };
  const initialQueue = snapshot();
  const events: DijkstraEvent[] = [], order: number[] = [];
  let relax = 0;
  while (!pq.isEmpty()) {
    const d = pq.peekPriority(), p = pq.pop();
    if (done[p] || d > dist[p]) continue;
    done[p] = 1; order.push(p);
    const updated: [number, number][] = [];
    if (p !== o.target) {
      const x = p % w, y = (p - x) / w;
      for (const [dx, dy, len] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (done[q]) continue;
        relax++;
        const nd = d + o.edge(p, q, len);
        if (nd < dist[q]) { dist[q] = nd; pred[q] = p; push(q, nd); updated.push([q, nd]); }
      }
    }
    events.push({ pop: p, d, updated, queue: snapshot() });
    if (p === o.target) break;
  }
  return { result: { dist, pred, order: Int32Array.from(order), relaxations: relax }, events, initialQueue };
}

/** Path from the tree's root to `target` (flat indices, root first); empty if unreached. */
export function tracePath(pred: ArrayLike<number>, target: number, dist?: ArrayLike<number>): number[] {
  if (target < 0 || (dist && !(dist[target] < Infinity))) return [];
  const out: number[] = [];
  for (let p = target, guard = 0; p >= 0 && guard <= pred.length; p = pred[p], guard++) out.push(p);
  return out.reverse();
}

/** Euclidean length of a pixel path (steps of 1 and √2). */
export function pathLength(path: number[], width: number): number {
  let L = 0;
  for (let k = 1; k < path.length; k++) {
    const dx = Math.abs((path[k] % width) - (path[k - 1] % width)), dy = Math.abs(Math.floor(path[k] / width) - Math.floor(path[k - 1] / width));
    L += Math.hypot(dx, dy);
  }
  return L;
}

// ---------------------------------------------------------------------------------------
// Fast marching

/**
 * Upwind update of the Eikonal equation |∇T| = P on a grid with spacing h, from the
 * smaller known neighbour along x (a) and along y (b); ∞ means "no known neighbour".
 * Solves (T − a)² + (T − b)² = (Ph)² with T ≥ max(a, b), or falls back to the
 * one-neighbour update min(a, b) + Ph when the two values are too far apart.
 */
export function eikonalUpdate(a: number, b: number, P: number, h = 1): number {
  const f = P * h;
  if (!(Math.abs(a - b) < f)) return Math.min(a, b) + f;
  return (a + b + Math.sqrt(2 * f * f - (a - b) * (a - b))) / 2;
}

export interface FastMarchingOptions {
  /** Grid spacing (default 1). */
  h?: number;
  /** Stop when the smallest trial value exceeds this. */
  maxT?: number;
  /** Stop when this pixel is final. */
  target?: number;
}

/**
 * Arrival times T for |∇T| = P from the sources, 4-neighbour upwind scheme. A source is a
 * pixel (T = 0) or a pixel with a start value. `seedNeighbourhood(…)` gives start values
 * for a small disc around each seed (the usual trick that removes most of the error the
 * first-order scheme makes next to a point source).
 */
export function fastMarching(width: number, height: number, P: ArrayLike<number>, sources: (number | { index: number; value: number })[], o: FastMarchingOptions = {}):
  { T: Float64Array; order: Int32Array } {
  const w = width, hh = height, n = w * hh, hs = o.h ?? 1;
  const T = new Float64Array(n).fill(Infinity), done = new Uint8Array(n), order = new Int32Array(n);
  const pq = new PriorityQueue(1024);
  for (const s0 of sources) { const s = srcOf(s0); if (s.value < T[s.index]) { T[s.index] = s.value; pq.push(s.index, s.value); } }
  let k = 0;
  const known = (q: number) => (done[q] ? T[q] : Infinity);
  while (!pq.isEmpty()) {
    const t = pq.peekPriority(), p = pq.pop();
    if (done[p] || t > T[p]) continue;
    if (o.maxT !== undefined && t > o.maxT) break;
    done[p] = 1; order[k++] = p;
    if (p === o.target) break;
    const x = p % w, y = (p - x) / w;
    for (let e = 0; e < 4; e++) {
      const xx = x + (e === 0 ? -1 : e === 1 ? 1 : 0), yy = y + (e === 2 ? -1 : e === 3 ? 1 : 0);
      if (xx < 0 || yy < 0 || xx >= w || yy >= hh) continue;
      const q = yy * w + xx;
      if (done[q]) continue;
      const a = Math.min(xx > 0 ? known(q - 1) : Infinity, xx < w - 1 ? known(q + 1) : Infinity);
      const b = Math.min(yy > 0 ? known(q - w) : Infinity, yy < hh - 1 ? known(q + w) : Infinity);
      const nt = eikonalUpdate(a, b, P[q], hs);
      if (nt < T[q]) { T[q] = nt; pq.push(q, nt); }
    }
  }
  for (let q = 0; q < n; q++) if (!done[q]) T[q] = Infinity;
  return { T, order: order.slice(0, k) };
}

/** Start values P(seed)·(Euclidean distance) for the pixels within `radius` of each seed. */
export function seedNeighbourhood(width: number, height: number, P: ArrayLike<number>, seeds: number[], radius = 3): { index: number; value: number }[] {
  const out: { index: number; value: number }[] = [], R = Math.floor(radius);
  for (const s of seeds) {
    const x = s % width, y = (s - x) / width;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const xx = x + dx, yy = y + dy, d = Math.hypot(dx, dy);
      if (xx < 0 || yy < 0 || xx >= width || yy >= height || d > radius) continue;
      out.push({ index: yy * width + xx, value: P[s] * d });
    }
  }
  return out;
}

/**
 * Minimal path from `start` = [x, y] down the arrival-time map T to its minimum (a source),
 * by small steps against the (bilinearly interpolated) gradient. Where the gradient is
 * unusable (next to unreached pixels, or on a plateau) it falls back to the lowest
 * 8-neighbour. Returns the points, start first.
 */
export function descendPath(T: ArrayLike<number>, width: number, height: number, start: [number, number],
  o: { step?: number; maxSteps?: number; stopT?: number } = {}): [number, number][] {
  const w = width, h = height, step = o.step ?? 0.5, maxSteps = o.maxSteps ?? 8 * (w + h);
  const at = (x: number, y: number) => T[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  const grad = (x: number, y: number): [number, number] | null => {
    const xm = at(x - 1, y), xp = at(x + 1, y), ym = at(x, y - 1), yp = at(x, y + 1), c = at(x, y);
    if (![xm, xp, ym, yp, c].every(Number.isFinite)) return null;
    // one-sided at the image border
    const gx = x === 0 ? xp - c : x === w - 1 ? c - xm : (xp - xm) / 2;
    const gy = y === 0 ? yp - c : y === h - 1 ? c - ym : (yp - ym) / 2;
    return [gx, gy];
  };
  const sampleT = (x: number, y: number) => {
    const x0 = Math.floor(x), y0 = Math.floor(y), ax = x - x0, ay = y - y0;
    return (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x0 + 1, y0)) + ay * ((1 - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1));
  };
  const pts: [number, number][] = [[start[0], start[1]]];
  let x = start[0], y = start[1];
  const stopT = o.stopT ?? 1;
  for (let s = 0; s < maxSteps; s++) {
    const xi = Math.round(x), yi = Math.round(y);
    const tHere = at(xi, yi);
    if (!Number.isFinite(tHere)) break;
    if (tHere <= stopT) {
      // finish on the discrete minimum nearby
      let bx = xi, by = yi;
      for (let guard = 0; guard < 8 && at(bx, by) > 0; guard++) {
        let nb: [number, number] | null = null, nv = at(bx, by);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const v = at(bx + dx, by + dy);
          if (bx + dx >= 0 && by + dy >= 0 && bx + dx < w && by + dy < h && v < nv) { nv = v; nb = [bx + dx, by + dy]; }
        }
        if (!nb) break;
        [bx, by] = nb;
      }
      pts.push([bx, by]);
      break;
    }
    // bilinear interpolation of the gradient at (x, y)
    const x0 = Math.min(w - 2, Math.max(0, Math.floor(x))), y0 = Math.min(h - 2, Math.max(0, Math.floor(y)));
    const ax = x - x0, ay = y - y0;
    const g00 = grad(x0, y0), g10 = grad(x0 + 1, y0), g01 = grad(x0, y0 + 1), g11 = grad(x0 + 1, y0 + 1);
    let moved = false;
    if (g00 && g10 && g01 && g11) {
      const gx = (1 - ay) * ((1 - ax) * g00[0] + ax * g10[0]) + ay * ((1 - ax) * g01[0] + ax * g11[0]);
      const gy = (1 - ay) * ((1 - ax) * g00[1] + ax * g10[1]) + ay * ((1 - ax) * g01[1] + ax * g11[1]);
      const g = Math.hypot(gx, gy);
      if (g > 1e-9) {
        const nx = x - step * gx / g, ny = y - step * gy / g;
        if (nx >= 0 && ny >= 0 && nx <= w - 1 && ny <= h - 1 && sampleT(nx, ny) < sampleT(x, y)) { x = nx; y = ny; moved = true; }
      }
    }
    if (!moved) {
      // discrete fallback: lowest 8-neighbour of the nearest pixel
      let nv = tHere, nb: [number, number] | null = null;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = xi + dx, yy = yi + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const v = at(xx, yy);
        if (v < nv) { nv = v; nb = [xx, yy]; }
      }
      if (!nb) break;
      [x, y] = nb;
    }
    pts.push([x, y]);
  }
  return pts;
}

/** Length of a polyline. */
export function polylineLength(pts: [number, number][]): number {
  let L = 0;
  for (let k = 1; k < pts.length; k++) L += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
  return L;
}
