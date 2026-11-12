// The bleedthrough demo's data, without DOM (numbers in the exercise come from here).
import * as U from "../lib/unmix";
import { poisson, rng } from "../lib/random";

export interface Data { x: Float64Array; y: Float64Array; g: Uint8Array; w: number; h: number; inCell?: Uint8Array }

export function imageData(s21: number, s12: number, photons: number, seed: number, vesicles: boolean): Data {
  const sc = U.cellScene(128, 3), P = 128 * 128, r = rng(seed);
  const x = new Float64Array(P), y = new Float64Array(P), g = new Uint8Array(P);
  // smooth brightness variation (staining is never uniform): a wide spread of intensities
  // along each population, so the scatter shows a line and its slope is well defined
  const field = (sd: number) => {
    const q = rng(sd), f = U.blurMap(Float32Array.from({ length: P }, () => q.unif()), 128, 128, 4);
    let lo = Infinity, hi = -Infinity;
    for (const v of f) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return f.map(v => (v - lo) / (hi - lo));
  };
  const f1 = field(21), f2 = field(22);
  for (let p = 0; p < P; p++) {
    const v = vesicles ? sc.vesicles[p] : 0;
    const a1 = 0.01 + sc.nucleus[p] * (0.25 + 1.1 * f1[p]) + 0.8 * v, a2 = 0.01 + sc.membrane[p] * (0.3 + 1.1 * f2[p]) + 0.8 * v;
    x[p] = poisson(r, photons * (a1 + s12 * a2));
    y[p] = poisson(r, photons * (s21 * a1 + a2));
    g[p] = v > 0.3 ? 3 : sc.nucleus[p] > 0.5 && sc.membrane[p] < 0.02 ? 1 : sc.membrane[p] > 0.5 && sc.nucleus[p] < 0.02 ? 2
      : sc.nucleus[p] < 0.01 && sc.membrane[p] < 0.01 && v < 0.01 ? 0 : 4;
  }
  return { x, y, g, w: 128, h: 128, inCell: Uint8Array.from(sc.cell) };
}

export function cellsData(s21: number, s12: number, seed: number): Data {
  const c = U.cytometryCells(s21, s12, seed);
  return { x: c.mx, y: c.my, g: Uint8Array.from(c.group), w: 0, h: 0 };
}


/** Compensate with a guessed S = [[1, s12], [s21, 1]]: S⁻¹ × measured for every point. */
export function compensate(D: Data, s21: number, s12: number): { cx: Float64Array; cy: Float64Array } {
  const k = 1 - s21 * s12, n = D.x.length, cx = new Float64Array(n), cy = new Float64Array(n);
  for (let i = 0; i < n; i++) { cx[i] = (D.x[i] - s12 * D.y[i]) / k; cy[i] = (-s21 * D.x[i] + D.y[i]) / k; }
  return { cx, cy };
}

/** The demo's readouts: slopes of the single-labelled populations (LS lines, as the old
 *  demo), medians, background medians, Pearson r inside the cell. */
export function summary(D: Data, cx: Float64Array, cy: Float64Array) {
  const pick = (v: Float64Array, g: number) => Array.from(v).filter((_, i) => D.g[i] === g);
  const l1 = U.lsLine(pick(cx, 1), pick(cy, 1)), l2 = U.lsLine(pick(cy, 2), pick(cx, 2));
  return {
    slope1: l1.b, slope2: l2.b, line1: l1, line2: l2,
    med1: U.median(pick(cy, 1)), med2: U.median(pick(cx, 2)),
    bgX: U.median(pick(cx, 0)), bgY: U.median(pick(cy, 0)),
    pearson: D.inCell ? U.pearson(cx, cy, D.inCell) : NaN,
    controlSlope: U.slopeThroughOrigin(pick(D.x, 1), pick(D.y, 1)),
    sdBg: (() => { const b = pick(cy, 0), m = b.reduce((s, v) => s + v, 0) / b.length; return Math.sqrt(b.reduce((s, v) => s + (v - m) ** 2, 0) / b.length); })(),
  };
}
