// Model of the virtualstain demo (no DOM; tested in demos/test/diffusion-test.ts).
//
// Two-channel synthetic cells, 32 × 32: an INPUT channel like a transmitted-light image (cell
// outline, nuclear envelope, cytoplasm texture) and a TARGET channel like a fluorescence stain
// (the nucleus, plus a "marker"). 300 training pairs define the prior: one narrow Gaussian
// (sd 0.05 per input pixel, 0.02 per target pixel) around each 2 048-pixel pair. The input is measured with noise, the
// target not at all (a = 0 for its pixels), so the posterior over the target is exact
// (lib/mixture.ts posteriorDiag): a weighted set of training targets.
// Markers: "nucleus" (the target is the nucleus: visible in the input), "spots" (random spots
// whose positions the input does not show), "treated" (the truth comes from treated cells whose
// nucleus stains twice as bright; the training set has only untreated cells).
import { makeDiag, posteriorDiag, posteriorMean, posteriorSample, type DiagPosterior } from "../lib/mixture";
import { rng } from "../lib/ui";
import type { Rng } from "../lib/random";

export const W = 32, P = W * W, NTRAIN = 300, PRIOR_SD = 0.05, TARGET_SD = 0.02;
export type Marker = "nucleus" | "spots" | "treated";

export interface Cell { input: Float64Array; target: Float64Array; nucleus: Uint8Array; spots: [number, number][] }

let TEX: Float64Array | null = null;
function smoothNoise(r: Rng): Float64Array {
  // a few random low-frequency cosines
  const out = new Float64Array(P);
  for (let k = 0; k < 4; k++) {
    const fx = 1 + 3 * r.unif(), fy = 1 + 3 * r.unif(), ph = 2 * Math.PI * r.unif(), a = 0.015 * r.norm();
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) out[y * W + x] += a * Math.cos(2 * Math.PI * (fx * x + fy * y) / W + ph);
  }
  return out;
}

/**
 * One cell: geometry and input from `r` (outline, nucleus, texture), the marker from `rm`.
 * `treated`: the nucleus stains twice as bright.
 */
export function makeCell(r: Rng, marker: Marker, treated = false, rm: Rng = r): Cell {
  // the outline and the texture are the same for every cell; the nucleus varies (position, size)
  const cx = 15.5, cy = 15.5, a = 12, b = 10, th = 0.3;
  const nx = cx + 2.5 * (2 * r.unif() - 1), ny = cy + 2 * (2 * r.unif() - 1), na = 3.8 + 1.7 * r.unif(), nb = na * (0.8 + 0.2 * r.unif()), nth = 0;
  const tex = TEX ??= smoothNoise(rng(77));
  const input = new Float64Array(P), target = new Float64Array(P), nucleus = new Uint8Array(P);
  const ell = (x: number, y: number, x0: number, y0: number, A: number, B: number, t: number) => {
    const dx = x - x0, dy = y - y0, u = dx * Math.cos(t) + dy * Math.sin(t), v = -dx * Math.sin(t) + dy * Math.cos(t);
    return Math.sqrt((u / A) ** 2 + (v / B) ** 2);
  };
  const spots: [number, number][] = [];
  if (marker === "spots") {
    const n = 2 + Math.floor(4 * rm.unif());
    for (let k = 0; k < n; k++) { const ang = 2 * Math.PI * rm.unif(), rad = 0.75 * rm.unif(); spots.push([cx + rad * a * Math.cos(ang), cy + rad * b * Math.sin(ang)]); }
  }
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, e = ell(x, y, cx, cy, a, b, th), ne = ell(x, y, nx, ny, na, nb, nth);
    const inCell = e < 1, inNuc = ne < 1 && inCell;
    // input: background 0.2, cytoplasm 0.45 + texture, bright outline, darker nuclear envelope
    let v = inCell ? 0.45 + tex[i] : 0.2;
    v += 0.35 * Math.exp(-0.5 * ((e - 1) * Math.min(a, b) / 0.8) ** 2);
    v -= 0.25 * Math.exp(-0.5 * ((ne - 1) * Math.min(na, nb) / 0.7) ** 2) * (inCell ? 1 : 0);
    input[i] = v;
    nucleus[i] = inNuc ? 1 : 0;
    let t = 0;
    if (marker !== "spots") t = inNuc ? (treated ? 0.8 : 0.4) : 0;
    else { t = inNuc ? 0.15 : 0; for (const [sx, sy] of spots) t += 0.7 * Math.exp(-0.5 * ((x - sx) ** 2 + (y - sy) ** 2) / 1.1); }
    target[i] = t;
  }
  return { input, target, nucleus, spots };
}

/** 75 cell geometries (nucleus position and size), each with 4 independent draws of the marker (cells that look the same in the input). */
export const NGEOM = 75, NDRAW = 4;
export function trainingSet(marker: Marker): Cell[] {
  const out: Cell[] = [], rm = rng(marker === "spots" ? 52 : 51);
  for (let g = 0; g < NGEOM; g++) for (let d = 0; d < NDRAW; d++) out.push(makeCell(rng(500 + g), marker, false, rm));
  return out;
}

export function truthCell(marker: Marker, seed = 3): Cell {
  return makeCell(rng(1000 + seed), marker, marker === "treated", rng(2000 + seed));
}

/** Noisy input measurement. */
export function measureInput(c: Cell, noise: number, seed = 9): Float64Array {
  const r = rng(seed);
  return Float64Array.from(c.input, v => v + noise * r.norm());
}

/** Exact posterior over the 2 048-pixel pair given the noisy input. */
export function posterior(train: Cell[], y: Float64Array, noise: number): DiagPosterior {
  const means = train.map(c => { const v = new Float64Array(2 * P); v.set(c.input, 0); v.set(c.target, P); return v; });
  const vars = new Float64Array(2 * P); vars.fill(PRIOR_SD ** 2, 0, P); vars.fill(TARGET_SD ** 2, P);
  const m = makeDiag(new Array(train.length).fill(1), means, train.map(() => vars));
  const yy = new Float64Array(2 * P); yy.set(y, 0);
  const a = new Float64Array(2 * P); a.fill(1, 0, P);
  return posteriorDiag(m, yy, a, noise ** 2);
}

export const targetOf = (v: Float64Array): Float64Array => v.slice(P);
export const meanTarget = (p: DiagPosterior): Float64Array => targetOf(posteriorMean(p));
export function sampleTargets(p: DiagPosterior, n: number, seed: number): Float64Array[] {
  const r = rng(seed), out: Float64Array[] = [];
  for (let i = 0; i < n; i++) out.push(targetOf(posteriorSample(p, r).x));
  return out;
}

/** Pixel standard deviation over samples. */
export function disagreement(S: Float64Array[]): Float64Array {
  const out = new Float64Array(P);
  for (let i = 0; i < P; i++) { let m = 0; for (const s of S) m += s[i] / S.length; let v = 0; for (const s of S) v += (s[i] - m) ** 2 / S.length; out[i] = Math.sqrt(v); }
  return out;
}

/** Readouts of a target image: stained area (pixels above 0.2), spot count (strict local maxima above 0.45), mean intensity of the stained area. */
export function readouts(t: Float64Array): { area: number; spots: number; intensity: number } {
  let area = 0, spots = 0, s = 0, n = 0;
  for (let i = 0; i < P; i++) if (t[i] > 0.2) { area++; s += t[i]; n++; }
  for (let y = 1; y < W - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x, v = t[i];
    if (v < 0.45) continue;
    let max = true;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && t[i + dy * W + dx] >= v) max = false;
    if (max) spots++;
  }
  return { area, spots, intensity: n ? s / n : 0 };
}

/** The largest posterior weights (descending). */
export function topWeights(p: DiagPosterior, k = 4): { idx: number; w: number }[] {
  return Array.from(p.w, (w, idx) => ({ idx, w })).sort((a, b) => b.w - a.w).slice(0, k);
}
