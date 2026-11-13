// The model behind the graphcut demo: test images, seed strokes, data costs from seeds,
// and one call that segments. No DOM: also used to compute the numbers quoted in the
// exercise solutions.
//
// Images are square, side n (N = 128 by default; the demo also offers 256 and 512). Presets
// and the synthetic image are defined at 128 x 128 and scaled to other sizes.
// `Segmenter` caches what does not change between updates (the neighbour graph per size and
// neighbourhood, the contrast weights per image and sigma, the max-flow graph of the solver),
// so painting a seed recomputes only the intensity model, the t-links and the max flow.
import { rng } from "../lib/ui";
import { gridEdges, contrastWeights, solveSeg, thresholdLabels, segEnergy, SegProblem, SegResult, EnergySplit, Nbhd, GridEdges } from "../lib/graph";

export const N = 128;
export const SIZES = [128, 256, 512];
export const NOISE_SEED = 11;

/** Side of a square image stored row by row. */
export const sideOf = (I: ArrayLike<number>): number => Math.round(Math.sqrt(I.length));

/** Synthetic test image: a bright disk with a thin arm (3 px wide at 128 x 128) on a dark
 *  background; at other sizes the same shapes scaled by n / 128. */
export function synthImage(n = N): Float64Array {
  const I = new Float64Array(n * n), f = N / n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    // pixel centres in 128-pixel units (at n = 128 exactly the integer coordinates)
    const X = n === N ? x : (x + 0.5) * f - 0.5, Y = n === N ? y : (y + 0.5) * f - 0.5;
    const disk = (X - 52) ** 2 + (Y - 64) ** 2 <= 30 * 30;
    const arm = X >= 79.5 && X <= 116.5 && Y >= 62.5 && Y <= 65.5;
    I[y * n + x] = disk || arm ? 170 : 80;
  }
  return I;
}

/** 2 x 2 block average of a 256 x 256 image (the micrograph) to 128 x 128. */
export function downsample(big: ArrayLike<number>): Float64Array {
  const I = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = 2 * y * 256 + 2 * x;
    I[y * N + x] = (big[a] + big[a + 1] + big[a + 256] + big[a + 257]) / 4;
  }
  return I;
}

/** Seeded Gaussian noise, clipped to 0..255. */
export function addNoise(I: ArrayLike<number>, sd: number): Float64Array {
  const r = rng(NOISE_SEED);
  const out = new Float64Array(I.length);
  for (let k = 0; k < I.length; k++) out[k] = Math.min(255, Math.max(0, I[k] + (sd > 0 ? sd * r.norm() : 0)));
  return out;
}

/** A seed stroke: a polyline in pixel coordinates, label 1 (object) or -1 (background). */
export interface Stroke { label: 1 | -1; pts: [number, number][]; r?: number }

/** Paint a disk of radius r at (x, y) into the n x n seed map (label 0 erases). */
export function paintDisk(seeds: Int8Array, x: number, y: number, r: number, label: number): void {
  const n = sideOf(seeds);
  for (let yy = Math.max(0, Math.floor(y - r)); yy <= Math.min(n - 1, Math.ceil(y + r)); yy++)
    for (let xx = Math.max(0, Math.floor(x - r)); xx <= Math.min(n - 1, Math.ceil(x + r)); xx++)
      if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) seeds[yy * n + xx] = label;
}

/** Paint a line segment with a round brush. */
export function paintLine(seeds: Int8Array, x0: number, y0: number, x1: number, y1: number, r: number, label: number): void {
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
  for (let k = 0; k <= steps; k++) paintDisk(seeds, x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps, r, label);
}

/** Seed map of side n from strokes given in 128-pixel coordinates (scaled, with the brush). */
export function rasterize(strokes: Stroke[], n = N): Int8Array {
  const seeds = new Int8Array(n * n), f = n / N;
  const sc = (p: [number, number]): [number, number] => (n === N ? p : [(p[0] + 0.5) * f - 0.5, (p[1] + 0.5) * f - 0.5]);
  for (const s of strokes) {
    const r = (s.r ?? 2) * f;
    if (s.pts.length === 1) paintDisk(seeds, ...sc(s.pts[0]), r, s.label);
    for (let k = 1; k < s.pts.length; k++) paintLine(seeds, ...sc(s.pts[k - 1]), ...sc(s.pts[k]), r, s.label);
  }
  return seeds;
}

/** Nearest-neighbour resampling of a seed map to side n (when the image size changes). */
export function resampleSeeds(seeds: Int8Array, n: number): Int8Array {
  const m = sideOf(seeds);
  if (m === n) return seeds.slice();
  const out = new Int8Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
    out[y * n + x] = seeds[Math.min(m - 1, Math.floor((y + 0.5) * m / n)) * m + Math.min(m - 1, Math.floor((x + 0.5) * m / n))];
  return out;
}

/** Area-average (n < m) or bilinear (n > m) resampling of a square grey image to side n. */
export function resampleImage(I: ArrayLike<number>, n: number): Float64Array {
  const m = sideOf(I);
  const out = new Float64Array(n * n);
  if (m === n) { out.set(I as ArrayLike<number>); return out; }
  if (m % n === 0) {
    const k = m / n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let s = 0;
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) s += I[(y * k + dy) * m + x * k + dx];
      out[y * n + x] = s / (k * k);
    }
    return out;
  }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const fy = Math.min(m - 1, Math.max(0, (y + 0.5) * m / n - 0.5)), fx = Math.min(m - 1, Math.max(0, (x + 0.5) * m / n - 0.5));
    const y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(m - 1, y0 + 1), x1 = Math.min(m - 1, x0 + 1);
    const ay = fy - y0, ax = fx - x0;
    out[y * n + x] = (1 - ay) * ((1 - ax) * I[y0 * m + x0] + ax * I[y0 * m + x1]) + ay * ((1 - ax) * I[y1 * m + x0] + ax * I[y1 * m + x1]);
  }
  return out;
}

export type DataModel = "gauss" | "hist";
export const BINS = 16;

export interface Costs { D0: Float64Array; D1: Float64Array; info: string; ok: boolean }

/** Data costs -log P(I | label), with P estimated from the seed pixels of each label:
 *  a Gaussian (mean, SD at least 4), or a histogram of 16 bins (pseudocount 1, smoothed 1-2-1). */
export function seedCosts(I: ArrayLike<number>, seeds: Int8Array, model: DataModel): Costs {
  const n = I.length;
  const D0 = new Float64Array(n), D1 = new Float64Array(n);
  const vals: number[][] = [[], []];
  for (let p = 0; p < n; p++) if (seeds[p]) vals[seeds[p] === 1 ? 1 : 0].push(I[p]);
  if (!vals[0].length || !vals[1].length) return { D0, D1, info: "needs seeds of both kinds", ok: false };
  if (model === "gauss") {
    const fit = vals.map(v => {
      const m = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.max(4, Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length));
      return { m, sd };
    });
    for (let p = 0; p < n; p++) {
      D0[p] = (I[p] - fit[0].m) ** 2 / (2 * fit[0].sd ** 2) + Math.log(fit[0].sd);
      D1[p] = (I[p] - fit[1].m) ** 2 / (2 * fit[1].sd ** 2) + Math.log(fit[1].sd);
    }
    return { D0, D1, ok: true, info: `object: mean ${fit[1].m.toFixed(0)}, SD ${fit[1].sd.toFixed(0)}; background: mean ${fit[0].m.toFixed(0)}, SD ${fit[0].sd.toFixed(0)}` };
  }
  const logp = vals.map(v => {
    const h = new Float64Array(BINS).fill(1);
    for (const x of v) h[Math.min(BINS - 1, Math.floor(x / (256 / BINS)))]++;
    const sm = h.map((_, b) => (h[Math.max(0, b - 1)] + 2 * h[b] + h[Math.min(BINS - 1, b + 1)]) / 4);
    const tot = sm.reduce((a, b) => a + b, 0);
    return sm.map(c => -Math.log(c / tot));
  });
  for (let p = 0; p < n; p++) {
    const b = Math.min(BINS - 1, Math.floor(I[p] / (256 / BINS)));
    D0[p] = logp[0][b]; D1[p] = logp[1][b];
  }
  return { D0, D1, ok: true, info: `histograms of ${vals[1].length} object and ${vals[0].length} background seed pixels` };
}

export interface Settings { lambda: number; sigma: number; nbhd: Nbhd; model: DataModel }

export interface Segmentation {
  ok: boolean; info: string;
  cut: SegResult; thr: Uint8Array; thrEnergy: EnergySplit;
  ms: number;        // the whole update
  solveMs: number;   // of which the graph and the max flow
}

/** A max-flow solver for SegProblem: lib/graph.ts solveSeg (TypeScript), or the Wasm GridCut. */
export type Solver = (P: SegProblem) => { L: Uint8Array; flow: number };

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Segmentation with caches: the neighbour graph per (size, neighbourhood), the contrast
 *  weights per (image, size, neighbourhood, sigma). The solver keeps its own graph. */
export class Segmenter {
  private edges = new Map<string, GridEdges>();
  private wKey = ""; private wImg: ArrayLike<number> | null = null; private w: Float64Array | null = null;
  constructor(public solver: Solver = solveSeg) {}

  gridEdges(n: number, nbhd: Nbhd): GridEdges {
    const k = `${n}/${nbhd}`;
    let E = this.edges.get(k);
    if (!E) { E = gridEdges(n, n, nbhd); this.edges.set(k, E); }
    return E;
  }

  weights(I: ArrayLike<number>, E: GridEdges, n: number, nbhd: Nbhd, sigma: number): Float64Array {
    const k = `${n}/${nbhd}/${sigma}`;
    if (this.wImg !== I || this.wKey !== k || !this.w) { this.w = contrastWeights(I, E, sigma); this.wImg = I; this.wKey = k; }
    return this.w;
  }

  segment(I: ArrayLike<number>, seeds: Int8Array, s: Settings): Segmentation {
    const t0 = now();
    const n = sideOf(I);
    const c = seedCosts(I, seeds, s.model);
    const E = this.gridEdges(n, s.nbhd);
    const P: SegProblem = { n: n * n, D0: c.D0, D1: c.D1, E, w: this.weights(I, E, n, s.nbhd, s.sigma), lambda: s.lambda, seeds };
    const t1 = now();
    let cut: SegResult;
    if (c.ok) { const r = this.solver(P); cut = { L: r.L, flow: r.flow, energy: segEnergy(P, r.L) }; }
    else cut = { L: new Uint8Array(n * n), flow: 0, energy: segEnergy(P, new Uint8Array(n * n)) };
    const t2 = now();
    const thr = thresholdLabels(P);
    const thrEnergy = segEnergy(P, thr);
    return { ok: c.ok, info: c.info, cut, thr, thrEnergy, ms: now() - t0, solveMs: t2 - t1 };
  }
}

/** Graph-cut segmentation of a square image with seeds (TypeScript max flow, no caching), and
 *  the thresholded (lambda = 0) result. */
export function segment(I: ArrayLike<number>, seeds: Int8Array, s: Settings): Segmentation {
  return new Segmenter().segment(I, seeds, s);
}

export interface Preset {
  id: string; label: string; image: "synth" | "cell"; noise: number;
  settings: Settings; strokes: Stroke[];
}

const SYNTH_SEEDS: Stroke[] = [
  { label: 1, pts: [[40, 56], [60, 72]] },
  { label: -1, pts: [[10, 10], [118, 10]] },
  { label: -1, pts: [[10, 118], [118, 118]] },
];

export const PRESETS: Preset[] = [
  { id: "noise", label: "noise", image: "synth", noise: 50, settings: { lambda: 1.5, sigma: Infinity, nbhd: 8, model: "gauss" }, strokes: SYNTH_SEEDS },
  { id: "shrink", label: "thin arm lost", image: "synth", noise: 50, settings: { lambda: 6, sigma: Infinity, nbhd: 8, model: "gauss" }, strokes: SYNTH_SEEDS },
  { id: "blocky", label: "blocky (4 neighbours)", image: "synth", noise: 90, settings: { lambda: 6, sigma: Infinity, nbhd: 4, model: "gauss" }, strokes: SYNTH_SEEDS },
  {
    id: "nucleus", label: "nucleus", image: "cell", noise: 0, settings: { lambda: 10, sigma: 25, nbhd: 8, model: "hist" },
    strokes: [
      { label: 1, pts: [[62, 40], [76, 56], [66, 74]] },
      { label: -1, pts: [[110, 70], [120, 90]] }, { label: -1, pts: [[6, 60], [20, 70]] },
      { label: -1, pts: [[40, 6], [46, 20]] }, { label: -1, pts: [[60, 98], [74, 104]] }, { label: -1, pts: [[44, 46], [45, 66]] },
    ],
  },
  {
    id: "badmodel", label: "overlapping intensities", image: "cell", noise: 0, settings: { lambda: 10, sigma: 25, nbhd: 8, model: "hist" },
    strokes: [
      { label: 1, pts: [[62, 40], [76, 56], [66, 74]] },
      { label: -1, pts: [[110, 70], [120, 90]] }, { label: -1, pts: [[6, 60], [20, 70]] },
    ],
  },
];

/** True labels of the synthetic image (for counting wrong pixels). */
export function synthTruth(n = N): Uint8Array {
  return Uint8Array.from(synthImage(n), v => (v > 125 ? 1 : 0));
}
