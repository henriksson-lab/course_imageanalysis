// Data of the tree playground (no DOM; also used by the solutions scripts): the presets'
// point sets and their hidden test sets, and the intensity profile for regression mode.
import { rng as FO_rng } from "../lib/ui";
import { dataset, LPoint, Dataset2D, blobs } from "../lib/datasets2d";


/** Uniform points, class 1 above the diagonal y = x (a boundary trees can only approximate by steps). */
export function diagonal(n: number, seed: number): Dataset2D {
  const r = FO_rng(seed), pts: LPoint[] = [];
  for (let i = 0; i < n; i++) { const x = 0.03 + 0.94 * r.unif(), y = 0.03 + 0.94 * r.unif(); pts.push({ x, y, c: y > x ? 1 : 0 }); }
  return { points: pts, k: 2 };
}

/** Two clean blobs plus one point of class 1 deep inside class 0. */
export function island(seed: number): Dataset2D {
  const d = blobs([[0.3, 0.35], [0.7, 0.68]], { sd: 0.08, n: 80, seed });
  d.points.push({ x: 0.27, y: 0.4, c: 1 });
  return d;
}

/** Profile across a cell edge: dark outside, bright inside, blurred and noisy (60 samples). */
export function edgeProfile(seed: number, n = 60): { x: number[]; y: number[] } {
  const r = FO_rng(seed), x: number[] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    x.push(t);
    y.push(40 + 120 / (1 + Math.exp(-(t - 0.45) / 0.035)) + 12 * r.norm());
  }
  return { x, y };
}

export interface Preset { id: string; label: string; mode: "class" | "reg"; data: () => Dataset2D; test: () => Dataset2D; splits: number; depth: number; view: "scan" | "depth"; showTest: boolean; select?: string }

export const PRESETS: Preset[] = [
  { id: "blobs", label: "two blobs: one stump", mode: "class", data: () => dataset("blobs", { seed: 1, n: 100 }), test: () => dataset("blobs", { seed: 1001, n: 400 }), splits: 1, depth: 6, view: "scan", showTest: false, select: "" },
  { id: "diagonal", label: "diagonal: staircase", mode: "class", data: () => diagonal(150, 2), test: () => diagonal(600, 1002), splits: Infinity, depth: 12, view: "scan", showTest: false },
  { id: "xor", label: "XOR: first split looks useless", mode: "class", data: () => dataset("xor", { seed: 5, n: 120 }), test: () => dataset("xor", { seed: 1005, n: 400 }), splits: 0, depth: 6, view: "scan", showTest: false, select: "" },
  { id: "island", label: "one mislabelled point", mode: "class", data: () => island(3), test: () => blobs([[0.3, 0.35], [0.7, 0.68]], { sd: 0.08, n: 400, seed: 1003 }), splits: Infinity, depth: 12, view: "scan", showTest: false },
  { id: "overlap", label: "noisy overlap: error vs depth", mode: "class", data: () => dataset("overlap", { seed: 4, n: 200 }), test: () => dataset("overlap", { seed: 1004, n: 1000 }), splits: Infinity, depth: 12, view: "depth", showTest: true },
  { id: "three", label: "three classes", mode: "class", data: () => dataset("blobs3", { seed: 6, n: 150 }), test: () => dataset("blobs3", { seed: 1006, n: 600 }), splits: Infinity, depth: 4, view: "scan", showTest: false },
  { id: "regression", label: "regression: a cell edge", mode: "reg", data: () => ({ points: [], k: 1 }), test: () => ({ points: [], k: 1 }), splits: Infinity, depth: 2, view: "depth", showTest: true },
];

