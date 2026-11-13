// A simulated GPU, for teaching (owner: GPU day, gpu-parallel/ ... gpu-fit/). No DOM.
//
// Everything here is a *model* with the rules of a real GPU and order-of-magnitude numbers:
//  - SIMT warps: a warp of 32 lanes issues one instruction for all lanes; lanes that do not
//    take a branch are masked. Cost of a branch = sum of the paths taken by any lane.
//  - Latency hiding: one multiprocessor issues one instruction per cycle from any ready warp;
//    a warp that loads from memory waits L cycles.
//  - Memory transactions: a warp's 32 addresses are served in 32-byte sectors of 128-byte
//    lines; bytes moved = touched sectors x 32.
//  - Tiling: global reads of a k x k stencil with T x T output tiles and a halo.
//  - Roofline: attainable rate min(peak, bandwidth x intensity); kernel time max(W/peak, Q/bw).
//  - End to end: upload + launches + kernels + download.
//  - Amdahl's law.
// Device presets are generic and labelled "order of magnitude"; one table so all demos agree.

export const WARP = 32;

// ---------------------------------------------------------------------------------------
// Devices

export interface Device {
  id: string;
  label: string;
  /** Peak float32 arithmetic, operations per second (a fused multiply-add counts 2). */
  peak: number;
  /** Peak of the low-precision matrix units (float16/bfloat16 matrix products), op/s; 0 = none. */
  matrix: number;
  /** float64 rate as a fraction of the float32 rate. */
  f64: number;
  /** Memory bandwidth, bytes per second. */
  bw: number;
  /** Link to the CPU (bytes/s); 0 = no transfer needed (the CPU itself). */
  link: number;
  /** Fixed cost per kernel launch, seconds. */
  launch: number;
  /** Number of arithmetic lanes (for the label only). */
  lanes: number;
}

export const DEVICES: Device[] = [
  { id: "cpu", label: "laptop CPU", peak: 0.4e12, matrix: 0, f64: 0.5, bw: 50e9, link: 0, launch: 0, lanes: 8 * 16 },
  { id: "gpu", label: "desktop GPU", peak: 30e12, matrix: 240e12, f64: 1 / 64, bw: 1e12, link: 25e9, launch: 10e-6, lanes: 10000 },
  { id: "dc", label: "data-centre GPU", peak: 60e12, matrix: 1000e12, f64: 0.5, bw: 3e12, link: 50e9, launch: 10e-6, lanes: 16000 },
];

export function device(id: string): Device {
  return DEVICES.find(d => d.id === id) ?? DEVICES[1];
}

// ---------------------------------------------------------------------------------------
// Latency and throughput

/** Time for n items: start-up t0 plus n / rate. */
export const linearTime = (t0: number, rate: number, n: number) => t0 + n / rate;

/** Break-even n where t0a + n/ra = t0b + n/rb (Infinity if b is never faster). */
export function breakEven(t0a: number, ra: number, t0b: number, rb: number): number {
  const d = 1 / ra - 1 / rb;
  if (d <= 0) return Infinity;
  return (t0b - t0a) / d;
}

/** Little's law: items (bytes) in flight = rate x latency. */
export const inFlight = (rate: number, latency: number) => rate * latency;

// ---------------------------------------------------------------------------------------
// SIMT: branch divergence

export interface WarpBranchResult {
  /** Cycles the warp spends (sum of the paths any lane takes). */
  cycles: number;
  /** Cycles if the warp did not diverge: the average per lane, i.e. useful lane-cycles / 32. */
  ideal: number;
  /** Lane-cycles of useful work. */
  useful: number;
  /** Lane-cycles issued (lanes x cycles). */
  issued: number;
  nA: number;
  nB: number;
  state: "A" | "B" | "mixed" | "empty";
}

/** One warp running `if (takeA) costA else costB`; `takeA` per active lane. */
export function warpBranch(takeA: ArrayLike<boolean | number>, costA: number, costB: number): WarpBranchResult {
  let nA = 0, nB = 0;
  for (let i = 0; i < takeA.length; i++) if (takeA[i]) nA++; else nB++;
  const cycles = (nA > 0 ? costA : 0) + (nB > 0 ? costB : 0);
  const useful = nA * costA + nB * costB, lanes = takeA.length;
  return {
    cycles, useful, issued: lanes * cycles, ideal: lanes ? useful / lanes : 0, nA, nB,
    state: lanes === 0 ? "empty" : nA === 0 ? "B" : nB === 0 ? "A" : "mixed",
  };
}

/** One warp running a loop with a per-lane trip count: it runs until the slowest lane is done. */
export function warpLoop(trips: ArrayLike<number>, costPerTrip: number): { cycles: number; useful: number; issued: number; maxTrips: number; meanTrips: number } {
  let mx = 0, s = 0;
  for (let i = 0; i < trips.length; i++) { mx = Math.max(mx, trips[i]); s += trips[i]; }
  const cycles = mx * costPerTrip;
  return { cycles, useful: s * costPerTrip, issued: trips.length * cycles, maxTrips: mx, meanTrips: trips.length ? s / trips.length : 0 };
}

/** Lane indices (pixel indices) of the warps that cover a w x h image, for warps shaped as
 *  rows of 32 (tw = 32, th = 1) or tiles (e.g. 8 x 4). Warps are numbered tile by tile in
 *  raster order; lanes inside a tile in raster order. */
export function warpsOf(w: number, h: number, tw: number, th: number): number[][] {
  const warps: number[][] = [];
  for (let ty = 0; ty < h; ty += th) for (let tx = 0; tx < w; tx += tw) {
    const lanes: number[] = [];
    for (let y = ty; y < Math.min(h, ty + th); y++) for (let x = tx; x < Math.min(w, tx + tw); x++) lanes.push(y * w + x);
    warps.push(lanes);
  }
  return warps;
}

export interface ImageBranchResult {
  warps: WarpBranchResult[];
  cycles: number;       // sum over warps (as if run one after another on one multiprocessor)
  ideal: number;        // sum of the per-warp ideal times
  efficiency: number;   // useful / issued
  mixed: number;        // number of divergent warps
}

/** All warps of an image: lane takes branch A where `inside[p]`. */
export function imageBranch(inside: ArrayLike<boolean | number>, w: number, h: number, tw: number, th: number, costA: number, costB: number): ImageBranchResult {
  const ws = warpsOf(w, h, tw, th).map(lanes => warpBranch(lanes.map(p => !!inside[p]), costA, costB));
  let cycles = 0, ideal = 0, useful = 0, issued = 0, mixed = 0;
  for (const r of ws) { cycles += r.cycles; ideal += r.ideal; useful += r.useful; issued += r.issued; if (r.state === "mixed") mixed++; }
  return { warps: ws, cycles, ideal, efficiency: issued ? useful / issued : 1, mixed };
}

/** Probability that a warp of n independent lanes, each taking the branch with probability
 *  p, is mixed: 1 - p^n - (1-p)^n. */
export const pMixed = (p: number, n = WARP) => 1 - p ** n - (1 - p) ** n;

/** Expected warp time p^n cA + (1-p)^n cB + P(mixed)(cA + cB). */
export const expectedWarpTime = (p: number, cA: number, cB: number, n = WARP) =>
  p ** n * cA + (1 - p) ** n * cB + pMixed(p, n) * (cA + cB);

// ---------------------------------------------------------------------------------------
// Latency hiding: N resident warps on one multiprocessor

export interface SchedulerResult {
  /** issue[t] = warp that issued in cycle t, or -1 (idle). */
  issue: Int16Array;
  /** Per warp: list of [start, end) intervals spent waiting for memory. */
  waits: [number, number][][];
  total: number;
  busy: number;
  utilisation: number;
  /** Little's-law estimate min(1, N k / (k + L)). */
  predicted: number;
}

/** N warps, each doing `iters` times: k compute instructions (one per cycle, one warp may
 *  issue per cycle; greedy: the same warp until it waits, then the next ready one in turn),
 *  then a load that returns after L cycles. */
export function schedule(N: number, k: number, L: number, iters: number): SchedulerResult {
  const left = new Int32Array(N).fill(k), done = new Int32Array(N), readyAt = new Int32Array(N);
  const issue: number[] = [], waits: [number, number][][] = Array.from({ length: N }, () => []);
  let t = 0, last = -1, busy = 0, finished = 0;
  const cap = (iters + 1) * (k + L) * Math.max(1, N) + 10;
  while (finished < N && t < cap) {
    let pick = -1;
    // greedy: keep issuing from the same warp until it waits, then the next ready one
    if (last >= 0 && done[last] < iters && readyAt[last] <= t) pick = last;
    else for (let j = 1; j <= N; j++) {
      const w = (last + j) % N;
      if (done[w] < iters && readyAt[w] <= t) { pick = w; break; }
    }
    issue.push(pick);
    if (pick >= 0) {
      busy++; last = pick;
      if (--left[pick] === 0) {
        done[pick]++;
        left[pick] = k;
        if (done[pick] < iters) { readyAt[pick] = t + 1 + L; waits[pick].push([t + 1, t + 1 + L]); }
        else finished++;
      }
    }
    t++;
  }
  return { issue: Int16Array.from(issue), waits, total: t, busy, utilisation: t ? busy / t : 0, predicted: Math.min(1, N * k / (k + L)) };
}

/** Resident warps allowed by the register file: 65 536 registers per multiprocessor, at
 *  most `maxWarps` warps. */
export function residentWarps(regsPerThread: number, maxWarps = 32, regFile = 65536): number {
  return Math.max(1, Math.min(maxWarps, Math.floor(regFile / (WARP * regsPerThread))));
}

// ---------------------------------------------------------------------------------------
// Memory transactions

export interface Transactions {
  /** Distinct 128-byte lines touched. */
  lines: number;
  /** Distinct 32-byte sectors touched. */
  sectors: number;
  /** Bytes moved (sectors x sector size). */
  moved: number;
  /** Bytes the lanes asked for (distinct addresses x element size). */
  used: number;
  efficiency: number;
  /** The touched sector indices (sorted). */
  sectorList: number[];
}

/** Memory traffic of one warp access: byte addresses of the lanes, each reading `elem` bytes. */
export function transactions(addrs: ArrayLike<number>, elem = 4, line = 128, sector = 32): Transactions {
  const L = new Set<number>(), S = new Set<number>(), A = new Set<number>();
  for (let i = 0; i < addrs.length; i++) {
    const a = addrs[i];
    A.add(a);
    for (let b = a; b < a + elem; b += Math.min(elem, sector)) { L.add(Math.floor(b / line)); S.add(Math.floor(b / sector)); }
    L.add(Math.floor((a + elem - 1) / line)); S.add(Math.floor((a + elem - 1) / sector));
  }
  const moved = S.size * sector, used = A.size * elem;
  return { lines: L.size, sectors: S.size, moved, used, efficiency: moved ? used / moved : 1, sectorList: [...S].sort((x, y) => x - y) };
}

export type AccessPattern = "row" | "column" | "stride" | "rgb" | "planar" | "random";

/** Byte addresses of the 32 lanes of a warp for a float32 access pattern.
 *  row: lane i reads element i; column: element i*W (a column of a W-wide row-major image);
 *  stride: element i*s; rgb: the R of interleaved RGB pixels (element 3i); planar: the R
 *  plane (element i); random: seeded random elements in [0, 32*W). */
export function accessAddresses(p: AccessPattern, opts: { W?: number; s?: number; seed?: number } = {}): number[] {
  const W = opts.W ?? 1024, s = opts.s ?? 2;
  let seed = (opts.seed ?? 7) >>> 0;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const out: number[] = [];
  for (let i = 0; i < WARP; i++) {
    let e: number;
    switch (p) {
      case "row": case "planar": e = i; break;
      case "column": e = i * W; break;
      case "stride": e = i * s; break;
      case "rgb": e = 3 * i; break;
      case "random": e = Math.floor(rnd() * 32 * W); break;
    }
    out.push(4 * e);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Tiling a stencil

/** Global reads per output pixel of a k x k stencil: direct (every tap from global memory)
 *  or with T x T output tiles that load a (T+k-1)^2 input tile once. */
export const readsDirect = (k: number) => k * k;
export const readsTiled = (k: number, T: number) => (T + k - 1) ** 2 / (T * T);

/** How often each input pixel of an n x n image (plus a halo of r = (k-1)/2 outside) is
 *  read from global memory, for n x n outputs. Returns counts on the (n+2r)^2 padded grid. */
export function readCounts(n: number, k: number, T: number | null): { counts: Int32Array; size: number; total: number } {
  const r = (k - 1) >> 1, m = n + 2 * r, counts = new Int32Array(m * m);
  let total = 0;
  if (T === null) {
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) { counts[(y + dy) * m + x + dx]++; total++; }
  } else {
    for (let ty = 0; ty < n; ty += T) for (let tx = 0; tx < n; tx += T) {
      const th = Math.min(T, n - ty), tw = Math.min(T, n - tx);
      for (let y = ty; y < ty + th + k - 1; y++) for (let x = tx; x < tx + tw + k - 1; x++) { counts[y * m + x]++; total++; }
    }
  }
  return { counts, size: m, total };
}

// ---------------------------------------------------------------------------------------
// Roofline and kernel time

export const attainable = (I: number, peak: number, bw: number) => Math.min(peak, bw * I);
export const ridge = (peak: number, bw: number) => peak / bw;
/** Time of a kernel with work W (operations) and traffic Q (bytes). */
export const kernelTime = (W: number, Q: number, peak: number, bw: number) => Math.max(W / peak, Q / bw);

/** An operation on an image of P pixels: work and traffic per pixel (float32), plus a
 *  fixed part (e.g. weights). `matrix`: can run on the matrix units in low precision. */
export interface OpCost { W: number; Q: number; matrix?: boolean }

export type OpId = "add" | "scale" | "threshold" | "pipe4" | "fused4" | "conv" | "convnaive" | "sep" | "fft" | "matmul" | "convlayer";

export interface OpSpec { id: OpId; label: string; short: string }
export const OPS: OpSpec[] = [
  { id: "add", label: "add two images", short: "add" },
  { id: "scale", label: "multiply by a constant", short: "scale" },
  { id: "threshold", label: "threshold", short: "thresh" },
  { id: "pipe4", label: "4-step pipeline, 4 passes", short: "4 passes" },
  { id: "fused4", label: "4-step pipeline, fused", short: "fused" },
  { id: "conv", label: "k×k filter (tiled)", short: "k×k" },
  { id: "convnaive", label: "k×k filter, no reuse", short: "k×k direct" },
  { id: "sep", label: "separable filter (2 passes)", short: "separable" },
  { id: "fft", label: "2D FFT (2 passes)", short: "FFT" },
  { id: "matmul", label: "matrix product N×N", short: "matmul" },
  { id: "convlayer", label: "network layer 3×3, C channels", short: "layer" },
];

export interface OpParams { k?: number; N?: number; C?: number; pixels?: number; bytes?: number }

/** Total work W and traffic Q of an operation (bytes per value `bytes`, default 4). For image
 *  operations `pixels` is the image size; for matmul, N is the matrix size. */
export function opCost(id: OpId, p: OpParams = {}): OpCost {
  const k = p.k ?? 3, N = p.N ?? 1024, C = p.C ?? 64, P = p.pixels ?? 4096 * 4096, b = p.bytes ?? 4;
  switch (id) {
    case "add": return { W: P, Q: 3 * b * P };
    case "scale": return { W: P, Q: 2 * b * P };
    case "threshold": return { W: P, Q: 2 * b * P };
    case "pipe4": return { W: 4 * P, Q: 4 * 2 * b * P };
    case "fused4": return { W: 4 * P, Q: 2 * b * P };
    case "conv": return { W: 2 * k * k * P, Q: 2 * b * P };
    case "convnaive": return { W: 2 * k * k * P, Q: b * (k * k + 1) * P };
    case "sep": return { W: 4 * k * P, Q: 4 * b * P };
    case "fft": { const n = Math.max(2, P); return { W: 5 * n * Math.log2(n), Q: 2 * 2 * 2 * b * n }; }
    case "matmul": return { W: 2 * N ** 3, Q: 3 * b * N * N, matrix: true };
    case "convlayer": return { W: 2 * 9 * C * C * P, Q: b * (2 * C * P + 9 * C * C), matrix: true };
  }
}

export const intensity = (c: OpCost) => c.W / c.Q;

// ---------------------------------------------------------------------------------------
// End to end

export interface EndToEnd { upload: number; launch: number; compute: number; download: number; total: number }

/** m kernels, each with work W and traffic Q, on an image of B bytes. If `stay`, the image is
 *  uploaded once and downloaded once; otherwise every kernel uploads and downloads it. */
export function endToEnd(dev: Device, B: number, m: number, W: number, Q: number, stay: boolean): EndToEnd {
  const xfer = dev.link > 0 ? B / dev.link : 0, trips = stay ? 1 : m;
  const upload = xfer * trips, download = xfer * trips, launch = dev.launch * m;
  const compute = m * kernelTime(W, Q, dev.peak, dev.bw);
  return { upload, launch, compute, download, total: upload + launch + compute + download };
}

// ---------------------------------------------------------------------------------------
// Amdahl

/** Speed-up when a fraction s stays serial and the rest gets k times faster. */
export const amdahl = (s: number, k: number) => 1 / (s + (1 - s) / k);

/** Speed-up of a pipeline of stages with times t[i], where stage i becomes f[i] times faster. */
export function pipelineSpeedup(t: number[], f: number[]): number {
  let a = 0, b = 0;
  t.forEach((ti, i) => { a += ti; b += ti / (f[i] ?? 1); });
  return a / b;
}
