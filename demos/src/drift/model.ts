// Model of the drift demo (no DOM).
// Frames mode: 128² frames of a 256² image whose content drifts by a seeded random walk
// d_k (frame k(x) = scene(x − d_k)), with noise, bleaching (falling brightness) and an
// optional slow change of the sample. Drift is estimated by phase correlation (Hann taper,
// upsampled DFT) of each frame to the first, to the previous one (summed), or to key frames.
// SMLM mode: point localisations of a blinking structure, drifting during the acquisition;
// localisations are rendered per time block, all block pairs are correlated, and the block
// drifts follow from least squares on the graph of pairs (as for a mosaic).
import { rng } from "../lib/random";
import { correlate2d, findPeaks, upsampledPeak2d, solveGlobal, TileEdge } from "../lib/registration";

export const S = 128, IMG = 256;

export interface DriftSettings { seed: number; n: number; noise: number; bleach: number; step: number; change: number }
export const DEFAULTS: DriftSettings = { seed: 5, n: 40, noise: 10, bleach: 0, step: 0.6, change: 0 };

export interface Frames { frames: Float64Array[]; dx: number[]; dy: number[]; n: number }

function bilinear(img: ArrayLike<number>, w: number, x: number, y: number): number {
  x = Math.max(0, Math.min(w - 1.0001, x)); y = Math.max(0, Math.min(w - 1.0001, y));
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (1 - fy) * ((1 - fx) * img[i] + fx * img[i + 1]) + fy * ((1 - fx) * img[i + w] + fx * img[i + w + 1]);
}

/** Synthetic stand-in for the micrograph (node tests). */
export function syntheticScene(seed = 11): Float64Array {
  const r = rng(seed), out = new Float64Array(IMG * IMG).fill(120);
  for (let k = 0; k < 300; k++) {
    const cx = IMG * r.unif(), cy = IMG * r.unif(), s = 1.5 + 4 * r.unif(), h = 40 + 80 * r.unif();
    for (let y = Math.max(0, Math.floor(cy - 3 * s)); y < Math.min(IMG, cy + 3 * s); y++)
      for (let x = Math.max(0, Math.floor(cx - 3 * s)); x < Math.min(IMG, cx + 3 * s); x++)
        out[y * IMG + x] += h * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
  }
  return out;
}

/** The frames. `scene`: 256² image (null: synthetic). */
export function makeFrames(s: DriftSettings, scene0: Float64Array | null): Frames {
  const scene = scene0 ?? syntheticScene(), r = rng(s.seed), rn = rng(s.seed + 99);
  const dx = [0], dy = [0];
  for (let k = 1; k < s.n; k++) { dx.push(dx[k - 1] + s.step * r.norm()); dy.push(dy[k - 1] + s.step * r.norm()); }
  const x0 = 64, y0 = 64, frames: Float64Array[] = [];
  for (let k = 0; k < s.n; k++) {
    const b = 1 - s.bleach * k / Math.max(1, s.n - 1), c = s.change * k / Math.max(1, s.n - 1), f = new Float64Array(S * S);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const X = x0 + x - dx[k], Y = y0 + y - dy[k];
      // the "changed" sample: the same field mirrored left-right
      const v = (1 - c) * bilinear(scene, IMG, X, Y) + (c ? c * bilinear(scene, IMG, IMG - 1 - X, Y) : 0);
      f[y * S + x] = b * v + s.noise * rn.norm();
    }
    frames.push(f);
  }
  return { frames, dx, dy, n: s.n };
}

/** Content shift d of frame b relative to frame a (b(x) = a(x − d)), by phase correlation. */
export function frameShift(a: Float64Array, b: Float64Array, w = S): [number, number] {
  const c = correlate2d(a, b, w, w, { p: 1, taper: true, eps: 1e-3, lowpass: 0.2 });
  const pk = findPeaks(c, 1, 3)[0];
  const u = upsampledPeak2d(c.R, w, w, pk.tx, pk.ty, 10);
  return [-u.tx, -u.ty];
}

export type DriftMode = "first" | "previous" | "key";

/** Estimated drift per frame (frame 0 = 0). Key frames every `every` frames, chained. */
export function registerFrames(f: Frames, mode: DriftMode, every = 10): { x: number[]; y: number[] } {
  const x = [0], y = [0];
  for (let k = 1; k < f.n; k++) {
    let ref = 0;
    if (mode === "previous") ref = k - 1;
    else if (mode === "key") ref = k % every === 0 ? k - every : k - (k % every);
    const [ux, uy] = frameShift(f.frames[ref], f.frames[k]);
    x.push(x[ref] + ux); y.push(y[ref] + uy);
  }
  return { x, y };
}

/** Shift an image back by (−dx, −dy) (bilinear), to undo a drift d. */
export function undrift(img: Float64Array, dx: number, dy: number, w = S): Float64Array {
  const out = new Float64Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) out[y * w + x] = bilinear(img, w, x + dx, y + dy);
  return out;
}

// ---------------------------------------------------------------------------------------
// SMLM mode

export interface SmlmSettings { seed: number; frames: number; blocks: number; precision: number; step: number }
export const SMLM_DEFAULTS: SmlmSettings = { seed: 3, frames: 4000, blocks: 10, precision: 0.25, step: 0.05 };
export const FIELD = 64;   // field of view in "pixels" of the camera; rendered at 4 px per camera pixel

export interface Smlm { x: Float64Array; y: Float64Array; t: Int32Array; dx: number[]; dy: number[]; s: SmlmSettings }

/** Emitters on three rings and two lines; each frame a few of them blink on and are localised. */
export function makeSmlm(s: SmlmSettings): Smlm {
  const r = rng(s.seed), ex: number[] = [], ey: number[] = [];
  for (const [cx, cy, R] of [[20, 22, 7], [42, 40, 9], [24, 46, 5]]) for (let k = 0; k < 60; k++) { const a = 2 * Math.PI * k / 60; ex.push(cx + R * Math.cos(a)); ey.push(cy + R * Math.sin(a)); }
  for (let k = 0; k < 80; k++) { ex.push(8 + 48 * k / 80); ey.push(10 + 6 * k / 80); }
  for (let k = 0; k < 60; k++) { ex.push(50 + 4 * k / 60); ey.push(12 + 44 * k / 60); }
  const dx = [0], dy = [0];
  // smooth drift: a random walk in velocity
  let vx = 0, vy = 0;
  for (let k = 1; k < s.frames; k++) {
    vx = 0.995 * vx + 0.0005 * r.norm() + s.step * 0.0002; vy = 0.995 * vy + 0.0005 * r.norm() - s.step * 0.0001;
    dx.push(dx[k - 1] + vx * s.step / 0.05); dy.push(dy[k - 1] + vy * s.step / 0.05);
  }
  const X: number[] = [], Y: number[] = [], T: number[] = [];
  for (let k = 0; k < s.frames; k++) {
    const m = 1 + Math.floor(3 * r.unif());
    for (let j = 0; j < m; j++) {
      const e = Math.floor(r.unif() * ex.length);
      X.push(ex[e] + dx[k] + s.precision * r.norm()); Y.push(ey[e] + dy[k] + s.precision * r.norm()); T.push(k);
    }
  }
  return { x: Float64Array.from(X), y: Float64Array.from(Y), t: Int32Array.from(T), dx, dy, s };
}

/** Histogram of localisations (frames in [t0, t1)) at `scale` px per camera pixel, blurred with σ = 1 px. */
export function render(m: Smlm, t0: number, t1: number, scale: number, shiftX?: (t: number) => number, shiftY?: (t: number) => number): Float64Array {
  const w = FIELD * scale, img = new Float64Array(w * w);
  for (let i = 0; i < m.t.length; i++) {
    if (m.t[i] < t0 || m.t[i] >= t1) continue;
    const x = (m.x[i] - (shiftX ? shiftX(m.t[i]) : 0)) * scale, y = (m.y[i] - (shiftY ? shiftY(m.t[i]) : 0)) * scale;
    const ix = Math.round(x), iy = Math.round(y);
    if (ix >= 0 && iy >= 0 && ix < w && iy < w) img[iy * w + ix] += 1;
  }
  return blur1(img, w);
}

function blur1(img: Float64Array, w: number): Float64Array {
  const k = [0.054, 0.244, 0.403, 0.244, 0.054], tmp = new Float64Array(w * w), out = new Float64Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) { let s = 0; for (let j = -2; j <= 2; j++) { const xx = x + j; if (xx >= 0 && xx < w) s += k[j + 2] * img[y * w + xx]; } tmp[y * w + x] = s; }
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) { let s = 0; for (let j = -2; j <= 2; j++) { const yy = y + j; if (yy >= 0 && yy < w) s += k[j + 2] * tmp[yy * w + x]; } out[y * w + x] = s; }
  return out;
}

/**
 * Redundant cross-correlation: render each time block (2 px per camera pixel, 128²), register
 * every pair of blocks (cross-correlation + upsampled DFT), and solve for one drift per block by
 * least squares (block 0 fixed at 0). Returns block centres (frames) and drifts (camera px).
 */
export function smlmDrift(m: Smlm): { centres: number[]; x: number[]; y: number[]; pairs: number } {
  const B = m.s.blocks, len = m.s.frames / B, scale = 2, w = FIELD * scale;
  const imgs = Array.from({ length: B }, (_, b) => render(m, b * len, (b + 1) * len, scale));
  const edges: TileEdge[] = [];
  for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
    const c = correlate2d(imgs[i], imgs[j], w, w, { p: 0, taper: false });
    const pk = findPeaks(c, 1, 2)[0];
    const u = upsampledPeak2d(c.R, w, w, pk.tx, pk.ty, 10);
    edges.push({ i, j, dx: -u.tx / scale, dy: -u.ty / scale, w: 1 });
  }
  const g = solveGlobal(B, edges)!;
  return { centres: Array.from({ length: B }, (_, b) => (b + 0.5) * len), x: Array.from(g.x), y: Array.from(g.y), pairs: edges.length };
}

/** Linear interpolation of the block drifts to every frame (constant beyond the ends); relative to frame 0's estimate. */
export function interpDrift(centres: number[], v: number[], t: number): number {
  if (t <= centres[0]) return v[0];
  for (let k = 1; k < centres.length; k++) if (t <= centres[k]) { const a = (t - centres[k - 1]) / (centres[k] - centres[k - 1]); return v[k - 1] + a * (v[k] - v[k - 1]); }
  return v[v.length - 1];
}
