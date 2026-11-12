// Model of the phasecorr demo (no DOM): two crops of a 256 × 256 image, degradations on B,
// and their correlation surface by cross-correlation (CC), partial normalisation |·|^p,
// phase correlation or overlap NCC. Convention: crop B's frame sits t = (tx, ty) from crop
// A's, so B(x) = A(x + t) and the peak should be at t.
import { rng } from "../lib/random";
import { correlate2d, maskedNcc, findPeaks, peakRatio, Surface, Peak, Complex, upsampledPeak2d, at } from "../lib/registration";

export const IMG = 256;
export type Source = "cell" | "texture" | "nuclei";
export type Method = "cc" | "phase" | "ncc";

export interface PCSettings {
  source: Source;
  size: number;           // crop side (64 or 128)
  ax: number; ay: number; // top-left of frame A
  tx: number; ty: number; // true shift (frame B = frame A + t)
  circular: boolean;      // B = A rolled by t (an exact circular shift)
  noise: number;          // sd, independent on A and B
  gain: number; offset: number; ramp: number;  // on B: gain · B + offset + ramp · x / size
  grad: number;           // uneven illumination of the whole field (both crops): + grad · (X + Y) / 510
  method: Method;
  p: number;              // exponent for "phase" (1 = phase correlation, 0 = CC)
  lowpass: number;        // Gaussian weight sd in cycles/px (0: off)
  taper: boolean;
  pad: boolean;
}

export const DEFAULTS: PCSettings = {
  source: "cell", size: 128, ax: 40, ay: 60, tx: 21, ty: 13, circular: false, noise: 0, gain: 1, offset: 0, ramp: 0, grad: 0,
  method: "phase", p: 1, lowpass: 0, taper: true, pad: false,
};

/** Synthetic periodic texture: a lattice of blobs, 18 px apart (noise is added per crop). */
export function texture(): Float64Array {
  const out = new Float64Array(IMG * IMG);
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
    const u = (x % 18) - 9, v = (y % 18) - 9;
    out[y * IMG + x] = 60 + 140 * Math.exp(-(u * u + v * v) / 18);
  }
  return out;
}

export interface Crops { a: Float64Array; b: Float64Array }

/** Crop A and B (with degradations) from a 256² image. */
export function makeCrops(img: ArrayLike<number>, s: PCSettings): Crops {
  const n = s.size, a = new Float64Array(n * n), b = new Float64Array(n * n);
  const px = (x: number, y: number) => {
    const X = Math.max(0, Math.min(IMG - 1, x)), Y = Math.max(0, Math.min(IMG - 1, y));
    return img[Y * IMG + X] + s.grad * (X + Y) / 510;
  };
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    a[y * n + x] = px(s.ax + x, s.ay + y);
    b[y * n + x] = s.circular
      ? px(s.ax + (((x + s.tx) % n) + n) % n, s.ay + (((y + s.ty) % n) + n) % n)
      : px(s.ax + s.tx + x, s.ay + s.ty + y);
  }
  const ra = rng(101), rb = rng(202);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    b[i] = s.gain * b[i] + s.offset + s.ramp * x / (n - 1);
    if (s.noise > 0) { a[i] += s.noise * ra.norm(); b[i] += s.noise * rb.norm(); }
  }
  return { a, b };
}

export interface Result {
  surface: Surface;          // circular index, shift 0 at index 0
  R: Complex | null;         // the (normalised) cross-power spectrum (null for NCC)
  peaks: Peak[];
  tx: number; ty: number;    // detected (integer) shift
  sx: number; sy: number;    // sub-pixel (upsampled DFT; = integer for NCC)
  height: number;            // peak value
  ratio: number;             // highest / second-highest peak
  ms: number;                // time of the FFT route
  taperedA: Float64Array | null;
}

/** Register crop B to crop A with the chosen method. */
export function register(c: Crops, s: PCSettings): Result {
  const n = s.size, t0 = now();
  let surface: Surface, R: Complex | null = null, taperedA: Float64Array | null = null;
  if (s.method === "ncc") {
    surface = maskedNcc(c.a, n, n, c.b, n, n, Math.round(0.25 * n * n));
  } else {
    const r = correlate2d(c.a, c.b, n, n, { p: s.method === "cc" ? 0 : s.p, lowpass: s.lowpass, taper: s.taper, pad: s.pad, eps: 1e-6 });
    surface = r; R = r.R; taperedA = r.a;
  }
  const ms = now() - t0;
  const peaks = findPeaks(surface, 5, 3);
  const pk = peaks[0] ?? { tx: 0, ty: 0, value: NaN, ix: 0, iy: 0 };
  let sx = pk.tx, sy = pk.ty;
  if (R && Number.isFinite(pk.value)) { const u = upsampledPeak2d(R, surface.w, surface.h, pk.tx, pk.ty, 10); sx = u.tx; sy = u.ty; }
  return { surface, R, peaks, tx: pk.tx, ty: pk.ty, sx, sy, height: pk.value, ratio: peakRatio(surface, 3), ms, taperedA };
}

export { at };

declare const performance: { now(): number } | undefined;
function now(): number { return typeof performance !== "undefined" ? performance.now() : Date.now(); }
