// Models for the rlsteps demo (no DOM): the 2D photon-count forward model, snapshots of
// Richardson–Lucy at log-spaced iterations, blind RL, the 3D test volume, and the
// measurement field with its readouts (peak and integrated ratios, FWHM, separation, count).
import * as P from "../lib/psf";
import * as D from "../lib/deconv";
import { rng } from "../lib/ui";
import { poisson } from "../lib/random";
import { render, SceneFn, WAVELENGTH, PIXEL } from "../deconvlab/scenes";

/** Iteration counts at which snapshots are kept: 0 and about 10 per decade up to `max`. */
export function snapshotIters(max: number, perDecade = 10): number[] {
  const out = [0];
  for (let i = 0; ; i++) { const k = Math.round(10 ** (i / perDecade)); if (k > max) break; if (k !== out[out.length - 1]) out.push(k); }
  if (out[out.length - 1] !== max) out.push(max);
  return out;
}

/** Photon image: Poisson(scale·clean + b); `scale` = photons at value 1 (Infinity: no noise). */
export function photonData(clean: ArrayLike<number>, scale: number, background: number, seed = 5, noiseFree = false): Float64Array {
  const r = rng(seed), out = new Float64Array(clean.length);
  for (let i = 0; i < out.length; i++) {
    const m = Math.max(0, clean[i]) * scale + background;
    out[i] = noiseFree ? m : poisson(r, m);
  }
  return out;
}

/** Poisson deviance per pixel, 2/n Σ [y log(y/m) − (y − m)]: about 1 when the model fits to the noise level. */
export function deviance(y: ArrayLike<number>, m: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const mm = Math.max(1e-12, m[i]);
    s += (y[i] > 0 ? y[i] * Math.log(y[i] / mm) : 0) - (y[i] - mm);
  }
  return 2 * s / y.length;
}

// ---------------------------------------------------------------------------------------
// 3D test volume (x × y × z = 64 × 32 × 32, lateral 100 nm, axial zstep)

export const NX = 64, NY = 32, NZ = 32, PX3 = 100;
export interface Obj3 { beads: [number, number, number][]; filamentX: number; sheet: { x0: number; x1: number; z: number }; sphere: { x: number; z: number; r: number } }
/** Objects in nm, origin at the volume centre. */
export const OBJ3: Obj3 = {
  beads: [[-2200, 0, 0], [-1500, 0, 1400]],
  filamentX: -600,
  sheet: { x0: 100, x1: 1500, z: 0 },
  sphere: { x: 2350, z: -1500, r: 650 },
};
/** Voxelise the objects for a z-step (nm), with 2 × 2 × 8 sub-samples per voxel (x, y, z). */
export function volume3d(zstep: number): Float64Array {
  const v = new Float64Array(NX * NY * NZ), o = OBJ3;
  const cx = NX / 2, cy = NY / 2, cz = NZ / 2;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    let s = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 8; c++) {
      const X = (x - cx + (a - 0.5) / 2) * PX3, Y = (y - cy + (b - 0.5) / 2) * PX3, Z = (z - cz + (c - 3.5) / 8) * zstep;
      let val = 0;
      if (Math.abs(X - o.filamentX) < 75 && Math.abs(Y) < 75) val += 1;                       // vertical filament (along z)
      if (Math.abs(Z - o.sheet.z) < 100 && X > o.sheet.x0 && X < o.sheet.x1) val += 1;         // horizontal sheet, 200 nm thick
      const r = Math.hypot(X - o.sphere.x, Y, Z - o.sphere.z);
      if (Math.abs(r - o.sphere.r) < 120) val += 1;                                           // hollow sphere
      s += val;
    }
    v[(z * NY + y) * NX + x] = s / 32;
  }
  for (const [bx, by, bz] of o.beads) {   // beads: one bright voxel each
    const x = Math.round(bx / PX3 + cx), y = Math.round(by / PX3 + cy), z = Math.round(bz / zstep + cz);
    if (z >= 0 && z < NZ) v[(z * NY + y) * NX + x] += 8;
  }
  return v;
}

export const OPT3 = { NA: 0.8, wavelength: WAVELENGTH, pixel: PX3, ri: 1.0 };

/** Centred 3D PSF (widefield or confocal) of size NX × NY × NZ (computed on NX × NX and folded periodically in y). */
export function psf3d(kind: "widefield" | "confocal", zstep: number): Float64Array {
  const o = { ...OPT3, zstep };
  const full = kind === "widefield" ? P.widefieldPSF3d(NX, NZ, o) : P.confocalPSF3d(NX, NZ, o);
  const out = new Float64Array(NX * NY * NZ), cyF = NX / 2, cy = NY / 2;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NX; y++) for (let x = 0; x < NX; x++) {
    const yy = (((y - cyF) % NY) + NY + cy) % NY;
    out[(z * NY + yy) * NX + x] += full[(z * NX + y) * NX + x];
  }
  return out;
}

/** xz slice through y (z rows, top = first plane) and xy slice at plane z. */
export const xz = (v: ArrayLike<number>, y = NY / 2) => P.xzSlice(v, NX, NY, NZ, y);
export function xy(v: ArrayLike<number>, z: number): Float64Array {
  const out = new Float64Array(NX * NY);
  for (let i = 0; i < NX * NY; i++) out[i] = v[z * NX * NY + i];
  return out;
}

// ---------------------------------------------------------------------------------------
// Measurement field (128 × 128): bead pairs, spots of flux 1 : 2 : 4, disks

export const RAYLEIGH = 0.6098 * WAVELENGTH / 1.0 / PIXEL;   // px, Airy NA 1.0: 4.88 px
export const PAIR_SEPS = [0.8, 1.0, 1.2, 1.5];
export const SPOTS: [number, number, number][] = [[24, 64, 1], [64, 64, 2], [104, 64, 4]];
export const DISKS: [number, number, number][] = [[24, 104, 3], [60, 104, 5], [100, 104, 8]];   // radius px
export const PAIR_Y = 24, PAIR_X = [20, 48, 76, 106];
const SPOT_SD = 0.6;
export function measureScene(): SceneFn {
  return (x, y) => {
    let v = 0.05;
    const g = (cx: number, cy: number, a: number) => a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * SPOT_SD * SPOT_SD));
    PAIR_SEPS.forEach((s, i) => { const d = s * RAYLEIGH / 2; v += g(PAIR_X[i] - d, PAIR_Y, 1.5) + g(PAIR_X[i] + d, PAIR_Y, 1.5); });
    for (const [sx, sy, f] of SPOTS) v += g(sx, sy, 0.75 * f);
    for (const [dx, dy, r] of DISKS) if ((x - dx) ** 2 + (y - dy) ** 2 <= r * r) v += 0.4;
    return v;
  };
}
export const BASE = 0.05;
/** Spots are counted in rows 1 .. COUNT_ROWS − 1 (pairs and spots, not the disks), above this fixed threshold. */
export const COUNT_ROWS = 84, COUNT_THR = 0.06;
/** True number of point objects in the counted rows: 4 pairs + 3 spots. */
export const TRUE_COUNT = 11;

/** Readouts of an image (values in the same units as the truth; base = flat background level). */
export interface Readout { peak: number; integral: number; fwhm: number; sep: number; count: number }
export function measure(img: ArrayLike<number>, n: number, base: number, thr: number): Readout {
  const at = (x: number, y: number) => img[y * n + x];
  const peakNear = (cx: number, cy: number) => { let m = -Infinity; for (let y = cy - 2; y <= cy + 2; y++) for (let x = cx - 2; x <= cx + 2; x++) m = Math.max(m, at(x, y)); return m - base; };
  const integ = (cx: number, cy: number) => { let s = 0; for (let y = cy - 5; y <= cy + 5; y++) for (let x = cx - 5; x <= cx + 5; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= 25) s += at(x, y) - base; return s; };
  const [s1, s4] = [SPOTS[0], SPOTS[2]];
  const peak = peakNear(s4[0], s4[1]) / peakNear(s1[0], s1[1]);
  const integral = integ(s4[0], s4[1]) / integ(s1[0], s1[1]);
  // FWHM of the brightest spot along x (sub-pixel), above the base
  const row = Array.from({ length: 21 }, (_, i) => at(s4[0] - 10 + i, s4[1]) - base);
  const fwhm = P.fwhm(row);
  // separation of the 1.2 × Rayleigh pair: distance between the two maxima of the row profile (NaN if merged)
  const px = PAIR_X[2], pr = Array.from({ length: 17 }, (_, i) => at(px - 8 + i, PAIR_Y));
  const maxima: number[] = [];
  for (let i = 1; i < pr.length - 1; i++) if (pr[i] > pr[i - 1] && pr[i] >= pr[i + 1]) {
    const a = pr[i - 1], b = pr[i], c = pr[i + 1], d = (a - c) / (2 * (a - 2 * b + c) || 1);   // parabolic peak
    maxima.push(i + d);
  }
  maxima.sort((a, b) => pr[Math.round(b)] - pr[Math.round(a)]);
  const sep = maxima.length >= 2 ? Math.abs(maxima[0] - maxima[1]) : NaN;
  // detected spots: strict local maxima (8 neighbours) above base + thr, in the rows of the pairs and spots
  let count = 0;
  for (let y = 1; y < COUNT_ROWS; y++) for (let x = 1; x < n - 1; x++) {
    const v = at(x, y);
    if (v - base < thr) continue;
    let ok = true;
    for (let j = -1; j <= 1 && ok; j++) for (let i = -1; i <= 1; i++) if ((i || j) && at(x + i, y + j) >= v) { ok = false; break; }
    if (ok) count++;
  }
  return { peak, integral, fwhm, sep, count };
}

/** RL snapshots for an operator, computed lazily: call step() until done. */
export class RLRun {
  iters: number[];
  snaps: Float64Array[] = [];
  k = 0;
  x: Float64Array;
  constructor(readonly y: Float64Array, readonly A: D.Operator, readonly b: number, max: number, perDecade = 10) {
    this.iters = snapshotIters(max, perDecade);
    this.x = D.flatStart(y, b);
    this.snaps.push(this.x.slice());
  }
  get done(): boolean { return this.snaps.length === this.iters.length; }
  step(): void {
    if (this.done) return;
    this.x = D.rlStep(this.x, this.y, this.A, this.b);
    this.k++;
    if (this.k === this.iters[this.snaps.length]) this.snaps.push(this.x.slice());
  }
  /** Snapshot index for an iteration (the largest computed one ≤ k). */
  indexFor(k: number): number {
    let i = 0;
    while (i + 1 < this.snaps.length && this.iters[i + 1] <= k) i++;
    return i;
  }
}

export { render };
