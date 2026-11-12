// The model behind the graphcut demo: test images, seed strokes, data costs from seeds,
// and one call that segments. No DOM: also used to compute the numbers quoted in the
// exercise solutions.
import { rng } from "../lib/ui";
import { gridEdges, contrastWeights, solveSeg, thresholdLabels, segEnergy, SegProblem, SegResult, EnergySplit, Nbhd } from "../lib/graph";

export const N = 128;
export const NOISE_SEED = 11;

/** Synthetic test image: a bright disk with a thin arm (3 px wide) on a dark background. */
export function synthImage(): Float64Array {
  const I = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const disk = (x - 52) ** 2 + (y - 64) ** 2 <= 30 * 30;
    const arm = x >= 80 && x <= 116 && y >= 63 && y <= 65;
    I[y * N + x] = disk || arm ? 170 : 80;
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

/** Paint a disk of radius r at (x, y) into the seed map (label 0 erases). */
export function paintDisk(seeds: Int8Array, x: number, y: number, r: number, label: number): void {
  for (let yy = Math.max(0, Math.floor(y - r)); yy <= Math.min(N - 1, Math.ceil(y + r)); yy++)
    for (let xx = Math.max(0, Math.floor(x - r)); xx <= Math.min(N - 1, Math.ceil(x + r)); xx++)
      if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) seeds[yy * N + xx] = label;
}

/** Paint a line segment with a round brush. */
export function paintLine(seeds: Int8Array, x0: number, y0: number, x1: number, y1: number, r: number, label: number): void {
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
  for (let k = 0; k <= steps; k++) paintDisk(seeds, x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps, r, label);
}

export function rasterize(strokes: Stroke[]): Int8Array {
  const seeds = new Int8Array(N * N);
  for (const s of strokes) {
    const r = s.r ?? 2;
    if (s.pts.length === 1) paintDisk(seeds, s.pts[0][0], s.pts[0][1], r, s.label);
    for (let k = 1; k < s.pts.length; k++) paintLine(seeds, ...s.pts[k - 1], ...s.pts[k], r, s.label);
  }
  return seeds;
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
  cut: SegResult; thr: Uint8Array; thrEnergy: EnergySplit; ms: number;
}

/** Graph-cut segmentation of an N x N image with seeds, and the thresholded (lambda = 0) result. */
export function segment(I: ArrayLike<number>, seeds: Int8Array, s: Settings): Segmentation {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const c = seedCosts(I, seeds, s.model);
  const E = gridEdges(N, N, s.nbhd);
  const P: SegProblem = { n: N * N, D0: c.D0, D1: c.D1, E, w: contrastWeights(I, E, s.sigma), lambda: s.lambda, seeds };
  const cut = c.ok ? solveSeg(P) : { L: new Uint8Array(N * N), flow: 0, energy: segEnergy(P, new Uint8Array(N * N)) };
  const thr = thresholdLabels(P);
  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  return { ok: c.ok, info: c.info, cut, thr, thrEnergy: segEnergy(P, thr), ms: t1 - t0 };
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
export function synthTruth(): Uint8Array {
  return Uint8Array.from(synthImage(), v => (v > 125 ? 1 : 0));
}
