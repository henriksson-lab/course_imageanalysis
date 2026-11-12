// Seeded 2D point-cloud datasets with class labels (owner: random-forest day, the owner of
// feature spaces). Used by the point playground (lib/playground.ts) and the SVM, GMM,
// neural-network and ncut demos, so all days draw the same clouds.
//
// Coordinates are in the unit square [0, 1]^2 (x right, y up as in a graph); labels are
// 0..k-1. Every generator takes a seed, so a recording is reproducible. No DOM: safe for tests.
import { rng } from "./ui";

export interface LPoint {
  x: number;
  y: number;
  /** Class label 0..k-1. */
  c: number;
}

export interface Dataset2D {
  points: LPoint[];
  /** Number of classes. */
  k: number;
}

type Rng = ReturnType<typeof rng>;

export interface GenOptions {
  /** Total number of points (default depends on the set, about 100–200). */
  n?: number;
  /** Noise level (standard deviation in unit-square units, default per set). */
  noise?: number;
  seed?: number;
}

const clampPts = (pts: LPoint[]) => pts.map(p => ({ x: Math.min(1.05, Math.max(-0.05, p.x)), y: Math.min(1.05, Math.max(-0.05, p.y)), c: p.c }));

/** Gaussian blobs: one per centre, class = centre index (or `classes[i]`). */
export function blobs(centres: [number, number][], o: GenOptions & { sd?: number | number[]; classes?: number[]; sizes?: number[] } = {}): Dataset2D {
  const r = rng(o.seed ?? 1), n = o.n ?? 50 * centres.length, pts: LPoint[] = [];
  centres.forEach(([cx, cy], i) => {
    const m = o.sizes?.[i] ?? Math.round(n / centres.length);
    const sd = Array.isArray(o.sd) ? o.sd[i] : o.sd ?? o.noise ?? 0.07;
    for (let j = 0; j < m; j++) pts.push({ x: cx + sd * r.norm(), y: cy + sd * r.norm(), c: o.classes?.[i] ?? i });
  });
  const k = Math.max(...pts.map(p => p.c)) + 1;
  return { points: clampPts(pts), k };
}

/** Anisotropic Gaussians: centre, standard deviations along the principal axes, angle (radians). */
export function anisotropic(specs: { centre: [number, number]; sd: [number, number]; angle: number; n?: number; c?: number }[], o: GenOptions = {}): Dataset2D {
  const r = rng(o.seed ?? 2), pts: LPoint[] = [];
  specs.forEach((s, i) => {
    const m = s.n ?? Math.round((o.n ?? 160) / specs.length), co = Math.cos(s.angle), si = Math.sin(s.angle);
    for (let j = 0; j < m; j++) {
      const u = s.sd[0] * r.norm(), v = s.sd[1] * r.norm();
      pts.push({ x: s.centre[0] + co * u - si * v, y: s.centre[1] + si * u + co * v, c: s.c ?? i });
    }
  });
  return { points: clampPts(pts), k: Math.max(...pts.map(p => p.c)) + 1 };
}

/** Two interleaving half circles ("two moons"). */
export function moons(o: GenOptions = {}): Dataset2D {
  const r = rng(o.seed ?? 3), n = o.n ?? 160, sd = o.noise ?? 0.04, pts: LPoint[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, t = Math.PI * r.unif();
    const x = c === 0 ? Math.cos(t) : 1 - Math.cos(t), y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
    // the moons span x in [-1, 2], y in [-0.5, 1]: map into the unit square
    pts.push({ x: 0.1 + 0.8 * (x + 1) / 3 + sd * r.norm(), y: 0.2 + 0.6 * (y + 0.5) / 1.5 + sd * r.norm(), c });
  }
  return { points: clampPts(pts), k: 2 };
}

/** Concentric rings (class = ring index), centred in the square; radii default [0.12, 0.35]. */
export function rings(o: GenOptions & { radii?: number[] } = {}): Dataset2D {
  const r = rng(o.seed ?? 4), radii = o.radii ?? [0.12, 0.35], sd = o.noise ?? 0.025, pts: LPoint[] = [];
  const total = radii.reduce((a, b) => a + b, 0), n = o.n ?? 160;
  radii.forEach((rad, c) => {
    const m = Math.max(8, Math.round(n * rad / total));
    for (let j = 0; j < m; j++) {
      const a = 2 * Math.PI * r.unif(), q = rad + sd * r.norm();
      pts.push({ x: 0.5 + q * Math.cos(a), y: 0.5 + q * Math.sin(a), c });
    }
  });
  return { points: clampPts(pts), k: radii.length };
}

/** XOR: four quadrant clusters, opposite quadrants share a class. */
export function xor(o: GenOptions = {}): Dataset2D {
  const r = rng(o.seed ?? 5), n = o.n ?? 160, sd = o.noise ?? 0.0, pts: LPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.05 + 0.9 * r.unif(), y = 0.05 + 0.9 * r.unif();
    if (Math.abs(x - 0.5) < 0.03 || Math.abs(y - 0.5) < 0.03) { i--; continue; }
    pts.push({ x: x + sd * r.norm(), y: y + sd * r.norm(), c: (x > 0.5) !== (y > 0.5) ? 1 : 0 });
  }
  return { points: clampPts(pts), k: 2 };
}

/** Interleaved spirals, one per class (default 2). */
export function spirals(o: GenOptions & { arms?: number; turns?: number } = {}): Dataset2D {
  const r = rng(o.seed ?? 6), arms = o.arms ?? 2, n = o.n ?? 200, sd = o.noise ?? 0.015, turns = o.turns ?? 1.6, pts: LPoint[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % arms, t = 0.08 + 0.92 * (Math.floor(i / arms) / Math.ceil(n / arms));
    const a = 2 * Math.PI * (turns * t + c / arms), q = 0.42 * t;
    pts.push({ x: 0.5 + q * Math.cos(a) + sd * r.norm(), y: 0.5 + q * Math.sin(a) + sd * r.norm(), c });
  }
  return { points: clampPts(pts), k: arms };
}

/** Add `m` outliers: uniform points far from the data (or at given positions), with class `c`
 *  (default: random class, i.e. label noise far from the bulk). */
export function withOutliers(d: Dataset2D, m = 4, o: { seed?: number; at?: [number, number][]; c?: number } = {}): Dataset2D {
  const r = rng(o.seed ?? 7), pts = [...d.points];
  const at = o.at ?? Array.from({ length: m }, () => {
    for (let tries = 0; ; tries++) {
      const p: [number, number] = [0.03 + 0.94 * r.unif(), 0.03 + 0.94 * r.unif()];
      const dmin = Math.min(...d.points.map(q => Math.hypot(q.x - p[0], q.y - p[1])));
      if (dmin > 0.15 || tries > 200) return p;
    }
  });
  at.forEach(([x, y]) => pts.push({ x, y, c: o.c ?? Math.floor(r.unif() * d.k) }));
  return { points: pts, k: d.k };
}

/** Flip the label of a fraction of points to another random class (label noise). */
export function withLabelNoise(d: Dataset2D, fraction: number, seed = 8): Dataset2D {
  const r = rng(seed);
  return { k: d.k, points: d.points.map(p => (r.unif() < fraction && d.k > 1 ? { ...p, c: (p.c + 1 + Math.floor(r.unif() * (d.k - 1))) % d.k } : p)) };
}

export interface DatasetPreset {
  id: string;
  label: string;
  make: (o?: GenOptions) => Dataset2D;
}

/** Named presets for buttons and data-dataset attributes. */
export const DATASETS: DatasetPreset[] = [
  { id: "blobs", label: "two blobs", make: o => blobs([[0.32, 0.4], [0.68, 0.62]], { sd: 0.09, ...o }) },
  { id: "blobs3", label: "three blobs", make: o => blobs([[0.25, 0.3], [0.75, 0.3], [0.5, 0.75]], { sd: 0.08, ...o }) },
  { id: "overlap", label: "overlapping", make: o => blobs([[0.42, 0.45], [0.58, 0.55]], { sd: 0.11, ...o }) },
  { id: "unequal", label: "unequal sizes", make: o => blobs([[0.3, 0.5], [0.72, 0.5]], { sd: [0.13, 0.05], sizes: [120, 30], ...o }) },
  { id: "aniso", label: "elongated", make: o => anisotropic([{ centre: [0.45, 0.4], sd: [0.2, 0.04], angle: 0.6 }, { centre: [0.55, 0.62], sd: [0.2, 0.04], angle: 0.6 }], o) },
  { id: "moons", label: "two moons", make: o => moons(o) },
  { id: "rings", label: "circle in ring", make: o => rings(o) },
  { id: "xor", label: "XOR", make: o => xor(o) },
  { id: "spirals", label: "spirals", make: o => spirals(o) },
  { id: "outliers", label: "blobs + outliers", make: o => withOutliers(blobs([[0.32, 0.4], [0.68, 0.62]], { sd: 0.07, ...o }), 4, { seed: o?.seed }) },
];

/** Preset by id (falls back to the first). */
export function dataset(id: string | undefined, o?: GenOptions): Dataset2D {
  return (DATASETS.find(d => d.id === id) ?? DATASETS[0]).make(o);
}

/** Points as separate arrays (for plot.ts scatter, or numeric code). */
export function toArrays(d: Dataset2D): { x: Float64Array; y: Float64Array; c: Int32Array } {
  return { x: Float64Array.from(d.points, p => p.x), y: Float64Array.from(d.points, p => p.y), c: Int32Array.from(d.points, p => p.c) };
}

/** Mean and 2x2 covariance of the points of one class (or all, c = -1). */
export function classStats(d: Dataset2D, c = -1): { n: number; mean: [number, number]; cov: [[number, number], [number, number]] } {
  const P = d.points.filter(p => c < 0 || p.c === c), n = P.length;
  const mx = P.reduce((a, p) => a + p.x, 0) / (n || 1), my = P.reduce((a, p) => a + p.y, 0) / (n || 1);
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of P) { sxx += (p.x - mx) ** 2; sxy += (p.x - mx) * (p.y - my); syy += (p.y - my) ** 2; }
  const k = n > 1 ? 1 / (n - 1) : 0;
  return { n, mean: [mx, my], cov: [[sxx * k, sxy * k], [sxy * k, syy * k]] };
}

// The interactive playground lives in playground.ts (it needs the DOM); re-exported here for
// convenience.
export { pointPlayground } from "./playground";
export type { Playground, PlaygroundOptions } from "./playground";
