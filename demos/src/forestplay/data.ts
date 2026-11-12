// Data of the forest playground (no DOM; also used by the solutions scripts): presets,
// point sets and test sets, and the feature table with derived and noise features.
import { dataset, Dataset2D } from "../lib/datasets2d";
import * as FO from "../lib/forest";

/** A fixed pseudo-random number in [0, 1) for (index, feature, salt): noise features stay put. */
function hashUnif(i: number, f: number, salt: number): number {
  let h = Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(f + 7, 0x85ebca6b) ^ Math.imul(salt + 3, 0xc2b2ae35);
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d); h ^= h >>> 15; h = Math.imul(h, 0x846ca68b); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export interface FeatureSet { extra: boolean; noise: number }

/** Feature names for the derived features. */
export const featureNames = (fs: FeatureSet): string[] =>
  ["x", "y", ...(fs.extra ? ["x + y", "x − y"] : []), ...Array.from({ length: fs.noise }, (_, j) => `noise ${j + 1}`)];

/** Feature-major table of points (x, y) with derived features; `salt` picks the noise draw. */
export function featureTable(xs: ArrayLike<number>, ys: ArrayLike<number>, fs: FeatureSet, salt: number): FO.Table {
  const n = xs.length, cols: Float64Array[] = [Float64Array.from(xs), Float64Array.from(ys)];
  if (fs.extra) { cols.push(Float64Array.from({ length: n }, (_, i) => xs[i] + ys[i])); cols.push(Float64Array.from({ length: n }, (_, i) => xs[i] - ys[i])); }
  for (let j = 0; j < fs.noise; j++) cols.push(Float64Array.from({ length: n }, (_, i) => hashUnif(i, j, salt)));
  return FO.tableFromColumns(cols);
}

export interface Preset { id: string; label: string; data: string; B: number; depth: number; minLeaf: number; m: number; extra: boolean; noise: number; bootstrap: boolean }

export const PRESETS: Preset[] = [
  { id: "single", label: "one deep tree overfits", data: "overlap", B: 1, depth: 20, minLeaf: 1, m: 2, extra: false, noise: 0, bootstrap: false },
  { id: "bagged", label: "100 bagged trees", data: "overlap", B: 100, depth: 20, minLeaf: 1, m: 2, extra: false, noise: 0, bootstrap: true },
  { id: "malike", label: "m = d: trees look alike", data: "moons", B: 100, depth: 20, minLeaf: 1, m: 4, extra: true, noise: 0, bootstrap: true },
  { id: "mone", label: "m = 1: diverse trees", data: "moons", B: 100, depth: 20, minLeaf: 1, m: 1, extra: true, noise: 0, bootstrap: true },
  { id: "noise", label: "noise features", data: "blobs", B: 100, depth: 20, minLeaf: 1, m: 2, extra: false, noise: 4, bootstrap: true },
  { id: "far", label: "confident far from the data", data: "blobs3", B: 100, depth: 20, minLeaf: 1, m: 2, extra: false, noise: 0, bootstrap: true },
  { id: "boundary", label: "uncertainty on the boundary", data: "moons", B: 100, depth: 20, minLeaf: 5, m: 2, extra: false, noise: 0, bootstrap: true },
];


export function makeData(id: string, test: boolean): Dataset2D {
  const n = id === "overlap" ? 200 : id === "moons" ? 160 : id === "blobs3" ? 120 : 100;
  const seed = id === "overlap" ? 4 : id === "moons" ? 3 : id === "blobs3" ? 6 : 1;
  const opts = id === "moons" ? { noise: 0.09 } : {};
  return dataset(id, { seed: test ? seed + 1000 : seed, n: test ? 1000 : n, ...opts });
}

