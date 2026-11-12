// Watershed by priority-queue flooding (Meyer 1994), no DOM. Owner: the watershed day
// (ws-landscape, ws-flooding, ws-gradient, ws-distance). Tests: demos/test/watershed-test.ts.
//
// Images are flat arrays (row-major, y down) with explicit w, h; a 1D signal is a 1 × n image.
// Markers are a label image (Int32Array-like): 0 = no marker, 1..k = marker labels.
//
//  - watershed(f, w, h, markers, opts): the fast version (binary heap, FIFO among equal
//    heights). Output labels 1..k; 0 = watershed line (opts.lines), outside opts.mask, or
//    never reached.
//  - floodWatershed(...): the same, plus the water level at which every pixel was flooded
//    (running maximum of the popped heights), for replaying the flooding.
//  - watershedTrace(...): an instrumented version for small grids that records every pop
//    (pixel, height, arrival number, labelled neighbours, decision, queue after the step),
//    with FIFO or LIFO tie-breaking. With FIFO it gives exactly the labels of watershed().
//  - minimaMarkers, hMinimaMarkers: regional minima / h-minima as labelled markers
//    (lib/morph.ts + lib/label.ts), boundaries(): pixels next to a different label.
//
// Algorithm (both versions):
//  1. every unlabelled neighbour of a marker pixel enters the queue with priority f(p)
//     (markers in raster order, neighbours in reading order: up-left, up, up-right, left,
//     right, down-left, down, down-right; 4-connectivity keeps up, left, right, down);
//  2. pop the pixel with the lowest f (ties: first in, first out);
//  3. lines = true: if all labelled neighbours carry the same label, take it, otherwise the
//     pixel is on the watershed line (label 0, it does not spread);
//     lines = false: take the label of the neighbour that queued it (reached it first);
//  4. a newly labelled pixel queues its unlabelled, not yet queued neighbours. Repeat.

import { PriorityQueue } from "./pqueue";
import { regionalMinima, hMinima } from "./morph";
import { label2d, Labels } from "./label";

export type WsConn = 4 | 8;
type A = ArrayLike<number>;

/** Neighbour offsets [dx, dy] in reading order (the order in which they are queued). */
export function wsOffsets(conn: WsConn): [number, number][] {
  return conn === 4
    ? [[0, -1], [-1, 0], [1, 0], [0, 1]]
    : [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
}

export interface WsOptions {
  /** Neighbourhood (default 4 for 1-row images, else 8). */
  conn?: WsConn;
  /** Leave a one-pixel watershed line (label 0) where basins meet (default false). */
  lines?: boolean;
  /** Only pixels with mask != 0 are flooded (e.g. the object mask for the distance transform). */
  mask?: A;
}

/** Internal state: label −1 marks a line pixel while flooding (returned as 0). */
const LINE = -1;

function neighbourLists(w: number, h: number, conn: WsConn): (p: number, out: number[]) => number {
  const off = wsOffsets(conn);
  return (p, out) => {
    const x = p % w, y = (p - x) / w;
    let k = 0;
    for (const [dx, dy] of off) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      out[k++] = yy * w + xx;
    }
    return k;
  };
}

export interface FloodResult {
  /** Labels 1..k; 0 = line / outside the mask / not reached. */
  labels: Int32Array;
  /** Water level when the pixel was flooded (running max of popped heights); −Infinity on markers, NaN if never flooded. */
  level: Float64Array;
  /** Number of pops (= flooded non-marker pixels). */
  pops: number;
}

/** Fast marker watershed with the flooding level of every pixel. */
export function floodWatershed(f: A, w: number, h: number, markers: A, opts: WsOptions = {}): FloodResult {
  const n = w * h, conn = opts.conn ?? (h === 1 ? 4 : 8), lines = !!opts.lines, mask = opts.mask;
  const lab = new Int32Array(n), queued = new Uint8Array(n), from = new Int32Array(n);
  const level = new Float64Array(n).fill(NaN);
  const nb = neighbourLists(w, h, conn), buf: number[] = new Array(8).fill(0);
  const q = new PriorityQueue(Math.max(64, n >> 2));
  const ok = (p: number) => !mask || mask[p] !== 0;
  for (let p = 0; p < n; p++) if (markers[p] > 0 && ok(p)) { lab[p] = markers[p]; queued[p] = 1; level[p] = -Infinity; }
  for (let p = 0; p < n; p++) {
    if (lab[p] <= 0) continue;
    const k = nb(p, buf);
    for (let j = 0; j < k; j++) {
      const r = buf[j];
      if (queued[r] || !ok(r)) continue;
      queued[r] = 1; from[r] = lab[p]; q.push(r, f[r]);
    }
  }
  let water = -Infinity, pops = 0;
  while (q.size > 0) {
    const pr = q.peekPriority(), p = q.pop();
    pops++;
    if (pr > water) water = pr;
    level[p] = water;
    let L = from[p];
    const k = nb(p, buf);
    if (lines) {
      L = 0;
      for (let j = 0; j < k; j++) {
        const l = lab[buf[j]];
        if (l <= 0) continue;
        if (L === 0) L = l; else if (l !== L) { L = LINE; break; }
      }
      if (L === 0) L = from[p];   // cannot happen (the pusher is labelled); kept for safety
    }
    lab[p] = L;
    if (L === LINE) continue;
    for (let j = 0; j < k; j++) {
      const r = buf[j];
      if (queued[r] || !ok(r)) continue;
      queued[r] = 1; from[r] = L; q.push(r, f[r]);
    }
  }
  for (let p = 0; p < n; p++) if (lab[p] < 0) lab[p] = 0;
  return { labels: lab, level, pops };
}

/** Fast marker watershed: labels 1..k, 0 = line / outside the mask / unreached. */
export function watershed(f: A, w: number, h: number, markers: A, opts: WsOptions = {}): Int32Array {
  return floodWatershed(f, w, h, markers, opts).labels;
}

// ---------------------------------------------------------------------------------------
// Instrumented version (small grids)

export interface QItem {
  /** Pixel index. */
  p: number;
  /** Its height f(p) = the priority. */
  value: number;
  /** Arrival number (0, 1, 2, ... in the order pixels entered the queue). */
  arrival: number;
  /** Label of the neighbour that queued it. */
  from: number;
}

export interface TraceStep {
  /** The popped item. */
  item: QItem;
  /** Distinct labels of its labelled neighbours at the moment of the pop (sorted). */
  nbLabels: number[];
  /** Labelled neighbour pixels (for highlighting). */
  nbPixels: number[];
  /** Assigned label; 0 = watershed line. */
  label: number;
  /** Pixels queued by this step. */
  pushed: number[];
  /** Water level after the step (running max of popped heights). */
  level: number;
  /** Queue after the step, in pop order. */
  queue: QItem[];
  /** Labels after the step (0 = unlabelled, −1 = line). */
  labels: Int32Array;
}

export interface Trace {
  /** Labels and queue before the first pop (markers only). */
  init: { labels: Int32Array; queue: QItem[] };
  steps: TraceStep[];
  /** Final labels (0 = line or unreached), as watershed(). */
  labels: Int32Array;
}

export interface TraceOptions extends WsOptions {
  /** Among equal heights pop the newest first (LIFO) instead of the oldest (FIFO, default). */
  lifo?: boolean;
}

/** Every pop of the flooding, for step-through demos. O(n²): keep n small (≤ a few thousand). */
export function watershedTrace(f: A, w: number, h: number, markers: A, opts: TraceOptions = {}): Trace {
  const n = w * h, conn = opts.conn ?? (h === 1 ? 4 : 8), lines = !!opts.lines, mask = opts.mask, lifo = !!opts.lifo;
  const lab = new Int32Array(n), queued = new Uint8Array(n);
  const nb = neighbourLists(w, h, conn), buf: number[] = new Array(8).fill(0);
  const ok = (p: number) => !mask || mask[p] !== 0;
  let queue: QItem[] = [], arrival = 0;
  const before = (a: QItem, b: QItem) => a.value < b.value || (a.value === b.value && (lifo ? a.arrival > b.arrival : a.arrival < b.arrival));
  const sorted = () => queue.slice().sort((a, b) => (before(a, b) ? -1 : before(b, a) ? 1 : 0));
  const push = (r: number, L: number, out?: number[]) => { queued[r] = 1; queue.push({ p: r, value: f[r], arrival: arrival++, from: L }); out?.push(r); };
  for (let p = 0; p < n; p++) if (markers[p] > 0 && ok(p)) { lab[p] = markers[p]; queued[p] = 1; }
  for (let p = 0; p < n; p++) {
    if (lab[p] <= 0) continue;
    const k = nb(p, buf);
    for (let j = 0; j < k; j++) { const r = buf[j]; if (!queued[r] && ok(r)) push(r, lab[p]); }
  }
  const init = { labels: lab.slice(), queue: sorted() };
  const steps: TraceStep[] = [];
  let water = -Infinity;
  while (queue.length) {
    let bi = 0;
    for (let i = 1; i < queue.length; i++) if (before(queue[i], queue[bi])) bi = i;
    const it = queue[bi];
    queue.splice(bi, 1);
    const p = it.p, k = nb(p, buf);
    if (it.value > water) water = it.value;
    const labs = new Set<number>(), nbPixels: number[] = [];
    for (let j = 0; j < k; j++) { const l = lab[buf[j]]; if (l > 0) { labs.add(l); nbPixels.push(buf[j]); } }
    let L = it.from;
    if (lines) L = labs.size > 1 ? LINE : labs.size === 1 ? [...labs][0] : it.from;
    lab[p] = L;
    const pushed: number[] = [];
    if (L !== LINE) for (let j = 0; j < k; j++) { const r = buf[j]; if (!queued[r] && ok(r)) push(r, L, pushed); }
    steps.push({ item: it, nbLabels: [...labs].sort((a, b) => a - b), nbPixels, label: L === LINE ? 0 : L, pushed, level: water, queue: sorted(), labels: lab.slice() });
  }
  return { init, steps, labels: Int32Array.from(lab, v => (v < 0 ? 0 : v)) };
}

// ---------------------------------------------------------------------------------------
// Markers and display helpers

/** Every regional minimum (plateau with only higher neighbours) as its own marker label. */
export function minimaMarkers(f: A, w: number, h: number, conn: WsConn = 8): Labels {
  return label2d(regionalMinima(f, w, h, conn), w, h, conn);
}

/** The h-minima (minima of depth > h, i.e. regional minima of HMIN_h) as marker labels; h ≤ 0 gives all minima. */
export function hMinimaMarkers(f: A, w: number, h: number, hh: number, conn: WsConn = 8): Labels {
  if (hh <= 0) return minimaMarkers(f, w, h, conn);
  return label2d(hMinima(f, w, h, hh, conn), w, h, conn);
}

/** 0/1 mask of labelled pixels with a 4-neighbour of a different label (region outlines, for drawing). */
export function boundaries(labels: A, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, l = labels[i];
    if (l <= 0) continue;
    if ((x + 1 < w && labels[i + 1] !== l) || (y + 1 < h && labels[i + w] !== l) ||
        (x > 0 && labels[i - 1] !== l) || (y > 0 && labels[i - w] !== l)) out[i] = 1;
  }
  return out;
}

/** Number of distinct positive labels. */
export function countLabels(labels: A): number {
  const s = new Set<number>();
  for (let i = 0; i < labels.length; i++) if (labels[i] > 0) s.add(labels[i]);
  return s.size;
}
