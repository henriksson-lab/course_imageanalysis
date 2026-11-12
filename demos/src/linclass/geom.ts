// Geometry of a linear classifier f(x) = w·x + b in 2D, and the presets of the linclass demo
// (no DOM; also used by the tests). Class 1 is the "+1" side (f > 0), class 0 the "−1" side.
import type { LPoint } from "../lib/datasets2d";
import { blobs, anisotropic, xor, rings } from "../lib/datasets2d";

export interface Line { w: [number, number]; b: number }

export const f = (L: Line, x: number, y: number): number => L.w[0] * x + L.w[1] * y + L.b;
export const norm = (L: Line): number => Math.hypot(L.w[0], L.w[1]);
/** Signed distance f/‖w‖ (positive on the +1 side). */
export const signedDistance = (L: Line, x: number, y: number): number => f(L, x, y) / (norm(L) || 1);
export const ySign = (c: number): number => (c === 1 ? 1 : -1);

/** Number of points on the wrong side (a point exactly on the line counts as wrong). */
export function errors(L: Line, pts: LPoint[]): number {
  return pts.filter(p => ySign(p.c) * f(L, p.x, p.y) <= 0).length;
}

/** Smallest y·f/‖w‖ over the points (negative if some point is on the wrong side). */
export function minMargin(L: Line, pts: LPoint[]): number {
  return pts.length ? Math.min(...pts.map(p => ySign(p.c) * signedDistance(L, p.x, p.y))) : NaN;
}

/** Nearest class mean as a line: w = μ₊ − μ₋, through the midpoint: b = −w·(μ₊ + μ₋)/2. */
export function nearestMeanLine(pts: LPoint[]): Line & { mPlus: [number, number]; mMinus: [number, number] } {
  const mean = (c: number): [number, number] => {
    const P = pts.filter(p => p.c === c);
    return P.length ? [P.reduce((s, p) => s + p.x, 0) / P.length, P.reduce((s, p) => s + p.y, 0) / P.length] : [0, 0];
  };
  const mp = mean(1), mm = mean(0);
  const w: [number, number] = [mp[0] - mm[0], mp[1] - mm[1]];
  const b = -(w[0] * (mp[0] + mm[0]) + w[1] * (mp[1] + mm[1])) / 2;
  return { w, b, mPlus: mp, mMinus: mm };
}

/** Class of the nearest training point (1-NN). */
export function nnClass(pts: LPoint[], x: number, y: number): number {
  let best = -1, bd = Infinity;
  for (const p of pts) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = p.c; } }
  return best;
}

/** One perceptron update: the first misclassified point (in order) moves the line: w += y x, b += y. */
export function perceptronStep(L: Line, pts: LPoint[], start = 0): { line: Line; index: number } {
  for (let q = 0; q < pts.length; q++) {
    const i = (start + q) % pts.length, p = pts[i], y = ySign(p.c);
    if (y * f(L, p.x, p.y) <= 0) return { line: { w: [L.w[0] + y * p.x, L.w[1] + y * p.y], b: L.b + y }, index: i };
  }
  return { line: L, index: -1 };
}

/** The by-hand example of the classify unit: w = (1, 2), b = −4 and four points. */
export const BY_HAND: Line & { points: LPoint[] } = {
  w: [1, 2], b: -4,
  points: [{ x: 1, y: 1, c: 0 }, { x: 3, y: 2, c: 1 }, { x: 0, y: 3, c: 1 }, { x: 2, y: 0.5, c: 0 }],
};

export interface LinPreset {
  id: string;
  label: string;
  range: [number, number];
  line: Line;
  points: () => LPoint[];
  axes?: [string, string];
  note: string;
}

const scaleTo = (pts: LPoint[], lo: number, hi: number): LPoint[] => pts.map(p => ({ x: lo + (hi - lo) * p.x, y: lo + (hi - lo) * p.y, c: p.c }));

export const LIN_PRESETS: LinPreset[] = [
  { id: "byhand", label: "by hand: w = (1, 2), b = −4", range: [-0.5, 4], line: { w: [1, 2], b: -4 }, points: () => BY_HAND.points.map(p => ({ ...p })), note: "The exercise example: read off f and the signed distance of each point." },
  {
    id: "many", label: "many lines are perfect", range: [0, 10], line: { w: [1, 0.2], b: -5.5 },
    points: () => scaleTo(blobs([[0.3, 0.35], [0.7, 0.65]], { sd: 0.07, n: 40, seed: 11 }).points, 0, 10),
    note: "Every line in the gap has zero errors. Which one would you trust on new points?",
  },
  {
    id: "mean", label: "nearest mean is not the best line", range: [0, 10], line: { w: [1, 0], b: -5 },
    points: () => scaleTo(anisotropic([{ centre: [0.42, 0.5], sd: [0.22, 0.025], angle: 1.1 }, { centre: [0.58, 0.5], sd: [0.22, 0.025], angle: 1.1 }], { n: 60, seed: 12 }).points, 0, 10),
    note: "Two long, slanted clouds: the nearest-mean line is vertical and cuts both.",
  },
  {
    id: "outlier", label: "one outlier", range: [0, 10], line: { w: [1, 1], b: -10 },
    points: () => [...scaleTo(blobs([[0.3, 0.35], [0.7, 0.65]], { sd: 0.07, n: 40, seed: 13 }).points, 0, 10), { x: 1.5, y: 1.5, c: 1 }],
    note: "One point of the + class sits deep among the − class. No line gets it right without many other errors.",
  },
  {
    id: "xor", label: "XOR: no line works", range: [0, 10], line: { w: [1, 0], b: -5 },
    points: () => scaleTo(xor({ n: 60, seed: 14 }).points, 0, 10),
    note: "Opposite corners share a class: every line makes at least about a quarter of errors.",
  },
  {
    id: "ring", label: "ring around a blob", range: [0, 10], line: { w: [1, 0], b: -5 },
    points: () => scaleTo(rings({ n: 70, seed: 15, radii: [0.1, 0.32] }).points, 0, 10).map(p => ({ ...p, c: 1 - p.c })),
    note: "A nucleus (inside) and its cytoplasm (around): no straight line separates them.",
  },
  {
    id: "objects", label: "objects: area and intensity", range: [0, 10], line: { w: [0, 1], b: -5 }, axes: ["area (100 px)", "mean intensity (×25)"],
    points: () => scaleTo(blobs([[0.35, 0.3], [0.6, 0.7]], { sd: [0.08, 0.1], n: 40, seed: 16 }).points, 0, 10),
    note: "Each point is a segmented object with two measured features: dim debris (−) and nuclei (+).",
  },
];
