// Model of the intensityfeat demo (no DOM): one synthetic cell with a chosen intensity
// pattern (all patterns have the same mask and the same total signal), the acquisition
// (gain, background offset, noise, saturation), mask errors, a population of two cell
// types, and a two-plate experiment for aggregation and normalisation.
import { edt } from "../lib/distance";
import { gaussianBlur, fromArray } from "../lib/imageops";
import { rng, gauss, Rng } from "../lib/random";
import { median } from "../lib/stats";
import { robustZ, intensityRow, pixelsOf, IntensityRow } from "../lib/features";

export const N = 160;

export type PatternId = "uniform" | "rim" | "spot" | "punctate" | "gradient" | "nuclear";
export const PATTERNS: { id: PatternId; label: string }[] = [
  { id: "uniform", label: "uniform" },
  { id: "rim", label: "bright rim" },
  { id: "spot", label: "off-centre spot" },
  { id: "punctate", label: "punctate" },
  { id: "gradient", label: "gradient" },
  { id: "nuclear", label: "nuclear" },
];

export interface CellOptions {
  pattern: PatternId;
  size?: number;
  cx?: number; cy?: number; a?: number; b?: number; angle?: number;
  /** Mean intensity inside the mask (the total is mean × area). */
  mean?: number;
  seed?: number;
}

export interface Cell {
  w: number; h: number;
  mask: Uint8Array;
  clean: Float64Array;
  /** Total signal inside the mask. */
  target: number;
  area: number;
}

/** Ellipse mask (pixel centres inside). */
export function ellipseMask(n: number, cx: number, cy: number, a: number, b: number, angleDeg: number): Uint8Array {
  const m = new Uint8Array(n * n), t = angleDeg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = (x - cx) * c + (y - cy) * s, v = -(x - cx) * s + (y - cy) * c;
    if ((u / a) ** 2 + (v / b) ** 2 <= 1) m[y * n + x] = 1;
  }
  return m;
}

/** A synthetic cell: the pattern inside an elliptic mask, blurred a little, scaled so Σ_mask = mean · area. */
export function makeCell(o: CellOptions): Cell {
  const n = o.size ?? N, cx = o.cx ?? n / 2, cy = o.cy ?? n / 2, a = o.a ?? 52, b = o.b ?? 40, ang = o.angle ?? 20;
  const mask = ellipseMask(n, cx, cy, a, b, ang);
  const de = edt(mask, n, n);
  const r = rng(o.seed ?? 5);
  const t = ang * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const raw = new Float64Array(n * n);
  const spots: [number, number][] = [];
  if (o.pattern === "punctate") {
    let guard = 0;
    while (spots.length < 14 && guard++ < 2000) {
      const x = cx + (2 * r.unif() - 1) * a, y = cy + (2 * r.unif() - 1) * a;
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= n || yi >= n || de[yi * n + xi] < 5) continue;
      if (spots.some(([p, q]) => Math.hypot(p - x, q - y) < 9)) continue;
      spots.push([x, y]);
    }
  }
  const sx = cx + 26 * c - (-8) * s, sy = cy + 26 * s + (-8) * c;   // spot position in the cell frame (26, −8)
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    if (!mask[i]) continue;
    const u = (x - cx) * c + (y - cy) * s, v = -(x - cx) * s + (y - cy) * c;
    let f = 1;
    switch (o.pattern) {
      case "uniform": f = 1; break;
      case "rim": f = 0.15 + Math.exp(-((de[i] - 1) ** 2) / (2 * 2.5 * 2.5)); break;
      case "spot": f = 0.4 + 2.5 * Math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 9 * 9)); break;
      case "punctate": f = 0.3; for (const [p, q] of spots) f += 3 * Math.exp(-((x - p) ** 2 + (y - q) ** 2) / (2 * 1.8 * 1.8)); break;
      case "gradient": f = 0.1 + (u + a) / (2 * a); break;
      case "nuclear": f = 0.25 + ((u / (0.45 * a)) ** 2 + (v / (0.45 * b)) ** 2 <= 1 ? 1.6 : 0); break;
    }
    raw[i] = f;
  }
  const bl = gaussianBlur(fromArray(n, n, raw), 1.2).data;
  let sIn = 0, area = 0;
  for (let i = 0; i < n * n; i++) if (mask[i]) { sIn += bl[i]; area++; }
  const target = (o.mean ?? 100) * area;
  const clean = Float64Array.from(bl, v => v * target / sIn);
  return { w: n, h: n, mask, clean, target, area };
}

export interface Acquisition { offset: number; gain: number; noise: number; sat: number; seed?: number }

/** Observed image: min(sat, max(0, gain · clean + offset + noise)). */
export function observe(clean: ArrayLike<number>, q: Acquisition): Float64Array {
  const r = rng(q.seed ?? 17);
  return Float64Array.from(clean, v => Math.min(q.sat, Math.max(0, q.gain * v + q.offset + (q.noise ? gauss(r, 0, q.noise) : 0))));
}

/** Mask grown (k > 0) or shrunk (k < 0) by |k| pixels (exact Euclidean). */
export function adjustMask(mask: Uint8Array, w: number, h: number, k: number): Uint8Array {
  if (k === 0) return mask;
  if (k > 0) {
    const inv = mask.map(v => (v ? 0 : 1));
    const d = edt(inv, w, h);
    return Uint8Array.from(d, (v, i) => (mask[i] || v <= k ? 1 : 0));
  }
  const d = edt(mask, w, h);
  return Uint8Array.from(d, (v, i) => (mask[i] && v > -k ? 1 : 0));
}

/** Pixel of maximum (lightly smoothed) intensity inside the mask. */
export function intensityMax(img: ArrayLike<number>, mask: Uint8Array, w: number, h: number): [number, number] {
  const sm = gaussianBlur(fromArray(w, h, img), 2).data;
  let best = -Infinity, bi = 0;
  for (let i = 0; i < w * h; i++) if (mask[i] && sm[i] > best) { best = sm[i]; bi = i; }
  return [bi % w, Math.floor(bi / w)];
}

// ---------------------------------------------------------------------------------------
// Population of two cell types

export interface PopCell { type: 0 | 1; cell: Cell; img: Float64Array; row: IntensityRow & { area: number } }

/** 20 cells: type A (uniform, larger, dimmer) and type B (punctate, smaller), seeded. */
export function makePopulation(seed = 3, n = 20, size = 112): PopCell[] {
  const r: Rng = rng(seed);
  const out: PopCell[] = [];
  for (let k = 0; k < n; k++) {
    const type = (k % 2) as 0 | 1;
    const a = type ? 26 + 8 * r.unif() : 34 + 10 * r.unif(), b = a * (0.65 + 0.3 * r.unif());
    const cell = makeCell({ pattern: type ? "punctate" : "uniform", size, a, b, angle: 180 * r.unif(), mean: 70 + 50 * r.unif(), seed: 100 + k });
    const img = observe(cell.clean, { offset: 10, gain: 1, noise: 6, sat: 4095, seed: 200 + k });
    const row = { ...intensityRow(img, cell.mask, size, size), area: cell.area };
    out.push({ type, cell, img, row });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Plates: aggregation and normalisation

export const ROWS = 4, COLS = 6;
/** Control wells: columns 1 and 6. Treated wells (phenotype +delta): four fixed positions. */
export const CONTROL = Uint8Array.from({ length: ROWS * COLS }, (_, i) => (i % COLS === 0 || i % COLS === COLS - 1 ? 1 : 0));
export const TREATED = Uint8Array.from({ length: ROWS * COLS }, (_, i) => ([2, 10, 13, 21].includes(i) ? 1 : 0));

export interface PlateOptions { gradient?: number; delta?: number; cells?: number; seed?: number; gainB?: number; offsetB?: number }
export interface PlateData { values: Float64Array; valuesB: Float64Array }

/** Two plates of per-well medians (40 cells per well); plate B has another gain and offset (batch). */
export function makePlate(o: PlateOptions): PlateData {
  const r = rng(o.seed ?? 9);
  const g = o.gradient ?? 0, delta = o.delta ?? 40, nc = o.cells ?? 40;
  const plate = (gain: number, off: number) => Float64Array.from({ length: ROWS * COLS }, (_, i) => {
    const c = i % COLS, row = Math.floor(i / COLS);
    const mu = 100 + delta * TREATED[i] + g * (c - (COLS - 1) / 2) + 2 * (row - 1.5) + gauss(r, 0, 3);
    const cells = Array.from({ length: nc }, () => Math.max(0, gauss(r, mu, 15)));
    return gain * median(cells) + off;
  });
  return { values: plate(1, 0), valuesB: plate(o.gainB ?? 1.4, o.offsetB ?? 20) };
}

/** Median polish: subtract row and column medians (a few sweeps); returns the residuals + overall median. */
export function medianPolish(v: ArrayLike<number>, rows = ROWS, cols = COLS, sweeps = 6): Float64Array {
  const x = Float64Array.from(v);
  const overall = median(x);
  for (let s = 0; s < sweeps; s++) {
    for (let i = 0; i < rows; i++) { const m = median(x.subarray(i * cols, (i + 1) * cols)); for (let j = 0; j < cols; j++) x[i * cols + j] -= m; }
    for (let j = 0; j < cols; j++) { const col = Array.from({ length: rows }, (_, i) => x[i * cols + j]); const m = median(col); for (let i = 0; i < rows; i++) x[i * cols + j] -= m; }
  }
  return x.map(u => u + overall);
}

export type Norm = "none" | "robustz" | "polish";

/**
 * Normalise one plate: none; robust z against its control wells; or median polish followed by
 * a robust z of the residuals against all wells of the plate (the control columns lose their
 * spread in the polish, so they cannot set the scale any more).
 */
export function normalisePlate(v: Float64Array, how: Norm): Float64Array {
  if (how === "none") return v;
  if (how === "polish") return robustZ(medianPolish(v), medianPolish(v));
  return robustZ(v, Array.from(v).filter((_, i) => CONTROL[i]));
}

export { pixelsOf };
