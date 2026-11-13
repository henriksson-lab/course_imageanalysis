// Instrumented data-parallel patterns (owner: GPU day, gpu-patterns/ and gpu-fit/). No DOM.
// Each function returns its result plus per-step frames and work/depth counters, so a demo can
// step through it and a test can check the counts against the formulas.
//
//  - reduceTree: pairwise tree reduction (optional rounding, e.g. float16, and a permutation
//    of the inputs = a different split of the work).
//  - scanHillisSteele (inclusive), scanBlelloch (exclusive, up-sweep + down-sweep).
//  - compact: stream compaction with an exclusive scan of a 0/1 mask.
//  - atomicHistogram: the serialisation of atomic increments, global or privatised per block.
//  - labelPropagation (optionally with pointer jumping): parallel connected components.
//  - wavefront: the anti-diagonal schedule of a DP table.
//  - levelFlood / queueFlood: watershed flooding level-synchronously vs one pixel at a time.

export interface Step {
  /** Values after this step (length n). */
  values: number[];
  /** Additions done in this step: [target, source] index pairs (target += source). */
  ops: [number, number][];
  label: string;
}

const id = (x: number) => x;

// ---------------------------------------------------------------------------------------
// Reduction

export interface ReduceResult { sum: number; steps: Step[]; work: number; depth: number }

/** Tree reduction: in step d, element i (a multiple of 2^(d+1)) adds element i + 2^d.
 *  `round` is applied after every addition. `order` permutes the inputs first. */
export function reduceTree(xs: ArrayLike<number>, round: (x: number) => number = id, order?: ArrayLike<number>): ReduceResult {
  const n = xs.length;
  const v = Array.from({ length: n }, (_, i) => round(xs[order ? order[i] : i]));
  const steps: Step[] = [{ values: v.slice(), ops: [], label: "input" }];
  let work = 0, depth = 0;
  for (let s = 1; s < n; s *= 2) {
    const ops: [number, number][] = [];
    for (let i = 0; i + s < n; i += 2 * s) { v[i] = round(v[i] + v[i + s]); ops.push([i, i + s]); work++; }
    depth++;
    steps.push({ values: v.slice(), ops, label: `step ${depth}: add the element ${s} to the right` });
  }
  return { sum: n ? v[0] : 0, steps, work, depth };
}

/** Left-to-right sum, rounding after every addition (the sequential loop). */
export function sumSequential(xs: ArrayLike<number>, round: (x: number) => number = id): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s = i === 0 ? round(xs[0]) : round(s + round(xs[i]));
  return s;
}

// ---------------------------------------------------------------------------------------
// Scan

export interface ScanResult { out: number[]; steps: Step[]; work: number; depth: number }

/** Hillis–Steele inclusive scan: in step d every element i >= 2^d adds element i - 2^d
 *  (all reads from the previous step). */
export function scanHillisSteele(xs: ArrayLike<number>): ScanResult {
  const n = xs.length;
  let v = Array.from(xs);
  const steps: Step[] = [{ values: v.slice(), ops: [], label: "input" }];
  let work = 0, depth = 0;
  for (let s = 1; s < n; s *= 2) {
    const nv = v.slice(), ops: [number, number][] = [];
    for (let i = s; i < n; i++) { nv[i] = v[i] + v[i - s]; ops.push([i, i - s]); work++; }
    v = nv; depth++;
    steps.push({ values: v.slice(), ops, label: `step ${depth}: every element adds the one ${s} to its left` });
  }
  return { out: v, steps, work, depth };
}

/** Blelloch work-efficient exclusive scan (n a power of two): up-sweep (a reduction tree
 *  towards the right end), clear the last element, down-sweep (swap and add). */
export function scanBlelloch(xs: ArrayLike<number>): ScanResult & { upSteps: number } {
  const n = xs.length;
  const v = Array.from(xs);
  const steps: Step[] = [{ values: v.slice(), ops: [], label: "input" }];
  let work = 0, depth = 0;
  for (let s = 1; s < n; s *= 2) {
    const ops: [number, number][] = [];
    for (let i = 2 * s - 1; i < n; i += 2 * s) { v[i] += v[i - s]; ops.push([i, i - s]); work++; }
    depth++;
    steps.push({ values: v.slice(), ops, label: `up-sweep ${depth}: partial sums of blocks of ${2 * s}` });
  }
  const upSteps = depth;
  if (n > 0) { v[n - 1] = 0; steps.push({ values: v.slice(), ops: [], label: "set the last element to 0" }); }
  for (let s = n >> 1; s >= 1; s >>= 1) {
    const ops: [number, number][] = [];
    for (let i = 2 * s - 1; i < n; i += 2 * s) {
      const t = v[i - s]; v[i - s] = v[i]; v[i] += t; ops.push([i, i - s]); work++;
    }
    depth++;
    steps.push({ values: v.slice(), ops, label: `down-sweep: pass the sum to the left (${s}), add the left part to the right` });
  }
  return { out: v, steps, work, depth, upSteps };
}

/** Sequential exclusive and inclusive prefix sums (the reference). */
export function prefixSums(xs: ArrayLike<number>): { inclusive: number[]; exclusive: number[] } {
  const inc: number[] = [], exc: number[] = [];
  let s = 0;
  for (let i = 0; i < xs.length; i++) { exc.push(s); s += xs[i]; inc.push(s); }
  return { inclusive: inc, exclusive: exc };
}

// ---------------------------------------------------------------------------------------
// Stream compaction

export interface CompactResult { positions: number[]; out: number[]; count: number }

/** Keep values[i] where mask[i] != 0: the exclusive scan of the mask is each kept element's
 *  output position. */
export function compact(values: ArrayLike<number>, mask: ArrayLike<number>): CompactResult {
  const m: number[] = Array.from(mask, (v): number => (v ? 1 : 0));
  const positions = prefixSums(m).exclusive;
  const count = m.reduce((a, b) => a + b, 0), out = new Array<number>(count);
  for (let i = 0; i < m.length; i++) if (m[i]) out[positions[i]] = values[i];
  return { positions, out, count };
}

// ---------------------------------------------------------------------------------------
// Histogram with atomics

export interface AtomicHistogramResult {
  counts: number[];
  /** Cycles: global = the busiest bin's queue; privatised = busiest private bin + merge. */
  cycles: number;
  /** Updates per bin (global), or per block and bin (privatised: blocks x bins). */
  queues: number[][];
  mergeCycles: number;
}

/** n threads each add 1 to bin[i]. Atomic updates to the same address are serialised, one
 *  per cycle; different addresses proceed in parallel. With `blocks` > 1 each block of
 *  consecutive threads has a private histogram (in fast memory), and the block histograms
 *  are then added into the global one: every global bin receives one update per block. */
export function atomicHistogram(bins: ArrayLike<number>, nBins: number, blocks = 1): AtomicHistogramResult {
  const n = bins.length, per = Math.ceil(n / blocks);
  const counts = new Array<number>(nBins).fill(0);
  const queues: number[][] = Array.from({ length: blocks }, () => new Array<number>(nBins).fill(0));
  for (let i = 0; i < n; i++) { counts[bins[i]]++; queues[Math.floor(i / per)][bins[i]]++; }
  let busiest = 0;
  for (const q of queues) for (const c of q) busiest = Math.max(busiest, c);
  const mergeCycles = blocks > 1 ? blocks : 0;
  return { counts, cycles: busiest + mergeCycles, queues, mergeCycles };
}

// ---------------------------------------------------------------------------------------
// Connected components by label propagation

export interface PropagationResult {
  /** Labels after each pass (pass 0 = initial: each pixel its own index + 1; background 0). */
  frames: Int32Array[];
  /** Passes that changed something (the last frame is stable; one more pass detects that). */
  passes: number;
  /** Pixel updates: foreground pixels visited per pass, summed over the passes run (incl. the check). */
  work: number;
  final: Int32Array;
}

/** Synchronous min-label propagation (every pass reads the previous pass), with `jump`:
 *  after each propagation, pointer jumping L[p] = L[L[p]] (labels are pixel indices + 1). */
export function labelPropagation(mask: ArrayLike<number>, w: number, h: number, conn: 4 | 8 = 4, jump = false, maxPasses = 10000): PropagationResult {
  const n = w * h, fg: number[] = [];
  let L = new Int32Array(n);
  for (let i = 0; i < n; i++) if (mask[i]) { L[i] = i + 1; fg.push(i); }
  const nb = conn === 4 ? [[-1, 0], [1, 0], [0, -1], [0, 1]] : [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];
  const frames = [L.slice()];
  let passes = 0, work = 0;
  for (let it = 0; it < maxPasses; it++) {
    const N = L.slice();
    let changed = false;
    for (const p of fg) {
      const x = p % w, y = (p - x) / w;
      let m = L[p];
      for (const [dx, dy] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (L[q] && L[q] < m) m = L[q];
      }
      N[p] = m;
    }
    if (jump) {
      const J = N.slice();
      for (const p of fg) J[p] = N[N[p] - 1];
      for (const p of fg) N[p] = J[p];
    }
    work += fg.length;
    for (const p of fg) if (N[p] !== L[p]) { changed = true; break; }
    if (!changed) break;
    L = N; passes++;
    frames.push(L.slice());
  }
  return { frames, passes, work, final: L };
}

// ---------------------------------------------------------------------------------------
// DP wavefront

/** Cells (i, j) of a rows x cols table grouped by anti-diagonal d = i + j: each group only
 *  depends on earlier groups when cell (i, j) needs (i-1, j), (i, j-1) and (i-1, j-1). */
export function wavefront(rows: number, cols: number): number[][] {
  const steps: number[][] = [];
  for (let d = 0; d < rows + cols - 1; d++) {
    const cells: number[] = [];
    for (let i = Math.max(0, d - cols + 1); i <= Math.min(rows - 1, d); i++) cells.push(i * cols + (d - i));
    steps.push(cells);
  }
  return steps;
}

// ---------------------------------------------------------------------------------------
// Flooding: priority queue (one pixel at a time) vs level-synchronous

export interface FloodResult {
  labels: Int32Array;
  /** Pixels labelled in each parallel step. */
  steps: number[][];
}

const N4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];

/** Level-synchronous flooding from markers (labels > 0) on integer levels: for each level
 *  in increasing order, repeated parallel rounds give every unlabelled pixel at or below that
 *  level the smallest label among its labelled 4-neighbours, until nothing changes. */
export function levelFlood(f: ArrayLike<number>, w: number, h: number, markers: ArrayLike<number>): FloodResult {
  const n = w * h, lab = Int32Array.from(markers);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { lo = Math.min(lo, f[i]); hi = Math.max(hi, f[i]); }
  const steps: number[][] = [];
  for (let lev = lo; lev <= hi; lev++) {
    for (;;) {
      const add: [number, number][] = [];
      for (let p = 0; p < n; p++) {
        if (lab[p] || f[p] > lev) continue;
        const x = p % w, y = (p - x) / w;
        let best = 0;
        for (const [dx, dy] of N4) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (lab[q] && (!best || lab[q] < best)) best = lab[q];
        }
        if (best) add.push([p, best]);
      }
      if (!add.length) break;
      for (const [p, l] of add) lab[p] = l;
      steps.push(add.map(a => a[0]));
    }
  }
  return { labels: lab, steps };
}

/** Priority-queue flooding (lowest level first, FIFO among ties): one pixel per step. */
export function queueFlood(f: ArrayLike<number>, w: number, h: number, markers: ArrayLike<number>): FloodResult {
  const n = w * h, lab = Int32Array.from(markers), inQ = new Uint8Array(n);
  const q: { p: number; k: number; t: number }[] = [];
  let tick = 0;
  const push = (p: number) => { inQ[p] = 1; q.push({ p, k: f[p], t: tick++ }); };
  for (let p = 0; p < n; p++) if (lab[p]) {
    const x = p % w, y = (p - x) / w;
    for (const [dx, dy] of N4) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const r = yy * w + xx;
      if (!lab[r] && !inQ[r]) push(r);
    }
  }
  const steps: number[][] = [];
  while (q.length) {
    let b = 0;
    for (let i = 1; i < q.length; i++) if (q[i].k < q[b].k || (q[i].k === q[b].k && q[i].t < q[b].t)) b = i;
    const { p } = q.splice(b, 1)[0];
    const x = p % w, y = (p - x) / w;
    let best = 0;
    for (const [dx, dy] of N4) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const r = yy * w + xx;
      if (lab[r] && (!best || lab[r] < best)) best = lab[r];
    }
    lab[p] = best;
    steps.push([p]);
    for (const [dx, dy] of N4) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const r = yy * w + xx;
      if (!lab[r] && !inQ[r]) push(r);
    }
  }
  return { labels: lab, steps };
}
