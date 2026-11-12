// Loss functions and presets of the hingeloss demo (no DOM; also used by the solution scripts).
import { QUALITATIVE } from "../lib/colour";
import { zeroOneLoss, hingeLoss, logisticLoss, squaredLoss } from "../lib/svm";

export interface Pt { x: number; y: 1 | -1 }
export interface Preset { id: string; label: string; pts: Pt[]; t: number; a: number; on: boolean[]; show: number; note: string }

const neg = (xs: number[]): Pt[] => xs.map(x => ({ x, y: -1 }));
const pos = (xs: number[]): Pt[] => xs.map(x => ({ x, y: 1 }));

export const LOSSES = [
  { name: "0–1", f: zeroOneLoss, color: "#555" },
  { name: "hinge", f: hingeLoss, color: QUALITATIVE[2] },
  { name: "logistic", f: logisticLoss, color: QUALITATIVE[4] },
  { name: "squared", f: squaredLoss, color: QUALITATIVE[3] },
];

export const PRESETS: Preset[] = [
  { id: "outlier", label: "an outlier on the wrong side: hinge vs squared", pts: [...neg([2, 3, 3.5, 4]), ...pos([6, 6.5, 7, 8]), ...pos([0.5])], t: 5, a: 1, on: [true, true, false, true], show: 3, note: "The red point at x = 0.5 is far on the wrong side. Its squared loss is larger than the losses of all other points together; its hinge loss grows only linearly with the distance. Move t to the minimum of each loss and compare." },
  { id: "far", label: "points far on the right side: logistic still pulls", pts: [...neg([1, 2, 3, 3.8]), ...pos([6, 7, 8, 9, 9.5])], t: 5, a: 1, on: [false, true, true, true], show: 2, note: "Points with y f ≥ 1 cost nothing under the hinge loss. The logistic loss is never exactly 0: every point has a say. The squared loss even punishes points that are 'too correct'." },
  { id: "steps", label: "0–1 has flat steps", pts: [...neg([1, 2.2, 3, 4.4, 5.2]), ...pos([3.8, 5.8, 6.5, 7.4, 8.6])], t: 5, a: 1, on: [true, true, false, false], show: 0, note: "The average 0–1 loss (right) is flat between data points and jumps at them: its slope is 0 or undefined, so walking downhill does not work. The hinge loss slopes towards a good t." },
  { id: "table", label: "by hand: y f = 2, 1, 0.5, 0, −1", pts: pos([7, 6, 5.5, 5, 4]), t: 5, a: 1, on: [true, true, true, true], show: 1, note: "Five red points with f(x) = x − 5: y f = 2, 1, 0.5, 0, −1. Read off each loss and compare with your table." },
];


/** Margin values y·a(x − t) of the points. */
export const margins = (pts: Pt[], t: number, a: number): number[] => pts.map(p => p.y * a * (p.x - t));

/** Average loss i over the points for threshold t and slope a. */
export const avgLoss = (pts: Pt[], i: number, t: number, a: number): number => margins(pts, t, a).reduce((s, m) => s + LOSSES[i].f(m), 0) / pts.length;

/** Threshold in [0, 10] (steps of 0.01) with the smallest average loss i; flat minima: the middle of the first minimal run. */
export function argminT(pts: Pt[], i: number, a: number): number {
  let best = 0, bv = Infinity;
  const vals: number[] = [];
  for (let k = 0; k <= 1000; k++) { const t = k / 100, v = avgLoss(pts, i, t, a); vals.push(v); if (v < bv - 1e-12) { bv = v; best = t; } }
  const k0 = Math.round(best * 100);
  let k1 = k0;
  while (k1 + 1 <= 1000 && Math.abs(vals[k1 + 1] - bv) < 1e-12) k1++;
  return Math.round((k0 + k1) / 2) / 100;
}
