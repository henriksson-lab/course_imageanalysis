// Parallel-beam tomography (no DOM): projector and back-projector, filters, filtered
// back-projection, ART/SIRT (+ non-negativity, + TV), analytic ellipse projections and a
// few test objects. Owner: the Radon day (radon/, backprojection/, fbp/, algebraic-recon/);
// the Hough day may import `project` for its one Radon slide.
//
// Conventions (shared with the Hough day):
//  - an n x n image, row-major (index = row * n + col), pixel spacing 1;
//  - world coordinates with the origin at the image centre cen = (n - 1) / 2:
//    x = col - cen (right), y = cen - row (UP), so angles turn counter-clockwise on screen;
//  - a line is {x cos(theta) + y sin(theta) = rho}: theta in [0, pi) is the direction of the
//    normal, rho the signed distance from the centre;
//  - a sinogram has one row per angle: sino[a * nBins + b], detector bin b at
//    rho_b = (b - (nBins - 1) / 2) * dRho. Displayed with theta across and rho up.
//  - theta = 0 integrates along y (vertical rays): the projection is the column sums;
//    theta = 90 deg integrates along x: the row sums, top row at positive rho.
//
// Projector: Joseph's ray-driven method. A ray is walked one image row (or column, for
// rays closer to horizontal) at a time; in each row it passes between two pixels and takes
// their linear interpolation, times the path length per row 1/|cos theta| (or 1/|sin|).
// The back-projector scatters exactly the same weights, so it is the exact transpose:
// <A x, y> = <x, A^T y> to rounding. FBP, SIRT and the "back-projection = A^T" slide rely
// on that. Outside the image the object is zero.

import { fft1d, nextPow2 } from "./fft";
import { rng, poisson } from "./random";
import { tv2d } from "./smooth";
import { SHEPP_LOGAN } from "./synth";

export type Arr = Float32Array | Float64Array;

/** The scan geometry: image size, projection angles (radians), detector bins and spacing. */
export interface Geometry {
  n: number;
  angles: Float64Array;
  nBins: number;
  dRho: number;
}

export function geometry(n: number, angles: Float64Array, nBins = n, dRho = 1): Geometry {
  return { n, angles, nBins, dRho };
}

/** `count` angles from startDeg in steps of rangeDeg / count (end excluded): the full set is angleSet(count). */
export function angleSet(count: number, rangeDeg = 180, startDeg = 0): Float64Array {
  const a = new Float64Array(count);
  for (let k = 0; k < count; k++) a[k] = (startDeg + k * rangeDeg / count) * Math.PI / 180;
  return a;
}

/** `count` angles from loDeg to hiDeg, both included (a tilt series: angleRange(-60, 60, 61)). */
export function angleRange(loDeg: number, hiDeg: number, count: number): Float64Array {
  const a = new Float64Array(count);
  for (let k = 0; k < count; k++) a[k] = (count === 1 ? loDeg : loDeg + k * (hiDeg - loDeg) / (count - 1)) * Math.PI / 180;
  return a;
}

/** Number of angles over 180 deg needed for n detector bins: pi/2 * n (the angle rule). */
export const angleRule = (n: number): number => Math.ceil(Math.PI / 2 * n);

export const centre = (n: number): number => (n - 1) / 2;
export const rhoOf = (b: number, nBins: number, dRho = 1): number => (b - (nBins - 1) / 2) * dRho;
export const binOf = (rho: number, nBins: number, dRho = 1): number => rho / dRho + (nBins - 1) / 2;
/** World (x right, y up, origin at the centre) of a pixel position (col, row). */
export const toWorld = (col: number, row: number, n: number): [number, number] => [col - centre(n), centre(n) - row];
export const toPixel = (x: number, y: number, n: number): [number, number] => [x + centre(n), centre(n) - y];
/** Where a point (x, y) lands on the detector at angle theta: rho = x cos + y sin (its sinusoid). */
export const pointRho = (x: number, y: number, theta: number): number => x * Math.cos(theta) + y * Math.sin(theta);

// ---------------------------------------------------------------------------------------
// One ray

// Scratch buffers for one ray: pixel indices and weights (at most 2 per row or column).
let RIDX = new Int32Array(0), RW = new Float64Array(0);

/**
 * List the pixels and weights of the ray (c = cos theta, s = sin theta, rho) in RIDX / RW;
 * returns their number.
 */
function walk(n: number, c: number, s: number, rho: number): number {
  if (RIDX.length < 2 * n) { RIDX = new Int32Array(2 * n); RW = new Float64Array(2 * n); }
  const cen = (n - 1) / 2, idx = RIDX, wt = RW;
  let k = 0;
  if (Math.abs(c) >= Math.abs(s)) {
    // one sample per row r: col = col0 + r d
    const w = 1 / Math.abs(c), d = s / c, col0 = cen + (rho - cen * s) / c;
    // rows with -1 < col < n
    let r0 = 0, r1 = n - 1;
    if (d > 1e-12) { r0 = Math.max(0, Math.floor((-1 - col0) / d)); r1 = Math.min(n - 1, Math.ceil((n - col0) / d)); }
    else if (d < -1e-12) { r0 = Math.max(0, Math.floor((n - col0) / d)); r1 = Math.min(n - 1, Math.ceil((-1 - col0) / d)); }
    for (let r = r0; r <= r1; r++) {
      const col = col0 + r * d;
      if (col <= -1 || col >= n) continue;
      const i0 = Math.floor(col), f = col - i0, o = r * n;
      if (i0 >= 0 && f < 1) { idx[k] = o + i0; wt[k++] = (1 - f) * w; }
      if (i0 + 1 < n && f > 0) { idx[k] = o + i0 + 1; wt[k++] = f * w; }
    }
  } else {
    // one sample per column q: row = row0 + q d
    const w = 1 / Math.abs(s), d = c / s, row0 = cen - (rho + cen * c) / s;
    let q0 = 0, q1 = n - 1;
    if (d > 1e-12) { q0 = Math.max(0, Math.floor((-1 - row0) / d)); q1 = Math.min(n - 1, Math.ceil((n - row0) / d)); }
    else if (d < -1e-12) { q0 = Math.max(0, Math.floor((n - row0) / d)); q1 = Math.min(n - 1, Math.ceil((-1 - row0) / d)); }
    for (let q = q0; q <= q1; q++) {
      const row = row0 + q * d;
      if (row <= -1 || row >= n) continue;
      const i0 = Math.floor(row), f = row - i0;
      if (i0 >= 0 && f < 1) { idx[k] = i0 * n + q; wt[k++] = (1 - f) * w; }
      if (i0 + 1 < n && f > 0) { idx[k] = (i0 + 1) * n + q; wt[k++] = f * w; }
    }
  }
  return k;
}

/** a_i . x along the ray. */
function rayDot(x: Float64Array, n: number, c: number, s: number, rho: number): number {
  const k = walk(n, c, s, rho), idx = RIDX, wt = RW;
  let acc = 0;
  for (let j = 0; j < k; j++) acc += wt[j] * x[idx[j]];
  return acc;
}

/** x += val a_i. */
function rayAdd(x: Float64Array, n: number, c: number, s: number, rho: number, val: number): void {
  const k = walk(n, c, s, rho), idx = RIDX, wt = RW;
  for (let j = 0; j < k; j++) x[idx[j]] += wt[j] * val;
}

function rayN2(n: number, c: number, s: number, rho: number): number {
  const k = walk(n, c, s, rho), wt = RW;
  let acc = 0;
  for (let j = 0; j < k; j++) acc += wt[j] * wt[j];
  return acc;
}

const f64 = (a: Arr): Float64Array => (a instanceof Float64Array ? a : Float64Array.from(a));

/** Line integral of the image along the ray (theta, rho): one entry of A x. */
export const rayIntegral = (img: Arr, n: number, theta: number, rho: number): number => rayDot(f64(img), n, Math.cos(theta), Math.sin(theta), rho);
/** img += value * (row of A for this ray): smear one number back along its ray. */
export const raySmear = (img: Float64Array, n: number, theta: number, rho: number, value: number): void => rayAdd(img, n, Math.cos(theta), Math.sin(theta), rho, value);
/** Squared norm of the ray's row of A. */
export const rayNorm2 = (n: number, theta: number, rho: number): number => rayN2(n, Math.cos(theta), Math.sin(theta), rho);

/** Visit the pixels and weights of one ray (for drawing a ray or building A explicitly). */
export function rayWeights(n: number, theta: number, rho: number): { idx: number[]; w: number[] } {
  const probe = new Float64Array(n * n);
  rayAdd(probe, n, Math.cos(theta), Math.sin(theta), rho, 1);
  const idx: number[] = [], w: number[] = [];
  for (let p = 0; p < n * n; p++) if (probe[p] !== 0) { idx.push(p); w.push(probe[p]); }
  return { idx, w };
}

// ---------------------------------------------------------------------------------------
// Whole projections

/** Projection of the image at one angle into out[offset .. offset + nBins). */
export function projectAngle(img: Arr, n: number, theta: number, nBins = n, dRho = 1, out?: Float64Array, offset = 0): Float64Array {
  const o = out ?? new Float64Array(nBins), x = f64(img);
  const c = Math.cos(theta), s = Math.sin(theta);
  for (let b = 0; b < nBins; b++) o[offset + b] = rayDot(x, n, c, s, rhoOf(b, nBins, dRho));
  return o;
}

/** Add the back-projection of one projection (sino[offset .. offset + nBins)) to img, times `scale`. */
export function backprojectAngle(proj: Arr, n: number, theta: number, img: Float64Array, nBins = n, dRho = 1, offset = 0, scale = 1): Float64Array {
  const c = Math.cos(theta), s = Math.sin(theta);
  for (let b = 0; b < nBins; b++) {
    const v = proj[offset + b] * scale;
    if (v !== 0) rayAdd(img, n, c, s, rhoOf(b, nBins, dRho), v);
  }
  return img;
}

/** Sinogram A x: one row of nBins per angle. */
export function project(img: Arr, g: Geometry, out?: Float64Array): Float64Array {
  const o = out ?? new Float64Array(g.angles.length * g.nBins), x = f64(img);
  for (let a = 0; a < g.angles.length; a++) projectAngle(x, g.n, g.angles[a], g.nBins, g.dRho, o, a * g.nBins);
  return o;
}

/** Back-projection A^T y (exact transpose of `project`), times `scale`. */
export function backproject(sino: Arr, g: Geometry, scale = 1, out?: Float64Array): Float64Array {
  const o = out ?? new Float64Array(g.n * g.n);
  if (out) o.fill(0);
  for (let a = 0; a < g.angles.length; a++) backprojectAngle(sino, g.n, g.angles[a], o, g.nBins, g.dRho, a * g.nBins, scale);
  return o;
}

/** Angular step used as the weight of each back-projected angle (pi / count for a full set). */
export function angleStep(angles: Float64Array): number {
  const m = angles.length;
  if (m < 2) return Math.PI;
  const span = Math.abs(angles[m - 1] - angles[0]);
  // a full half-turn set [0, pi) has span pi - pi/m: step pi/m; a tilt series includes both ends
  return span / (m - 1);
}

/** Plain (unfiltered) back-projection, scaled like FBP: dtheta * A^T y. Blurred by 1/r. */
export function plainBackprojection(sino: Arr, g: Geometry): Float64Array {
  return backproject(sino, g, angleStep(g.angles));
}

// ---------------------------------------------------------------------------------------
// Ramp filters

export type FilterKind = "none" | "ramp" | "shepp-logan" | "cosine" | "hann";
export const FILTERS: FilterKind[] = ["none", "ramp", "shepp-logan", "cosine", "hann"];

/** Spatial Ram–Lak kernel for bin spacing 1: h[0] = 1/4, h[k] = -1/(pi k)^2 for odd k, 0 for even k != 0. */
export function ramLak(k: number): number {
  if (k === 0) return 0.25;
  return k % 2 === 0 ? 0 : -1 / (Math.PI * Math.PI * k * k);
}

/** Taper multiplying the ramp at frequency f (cycles per bin, |f| <= 0.5) with cutoff fc. */
export function taper(kind: FilterKind, f: number, fc = 0.5): number {
  const a = Math.abs(f);
  if (kind === "none") return 1;
  if (a > fc) return 0;
  const u = a / fc;
  switch (kind) {
    case "ramp": return 1;
    case "shepp-logan": return u === 0 ? 1 : Math.sin(Math.PI * u / 2) / (Math.PI * u / 2);
    case "cosine": return Math.cos(Math.PI * u / 2);
    case "hann": return 0.5 + 0.5 * Math.cos(Math.PI * u);
  }
}

/**
 * Frequency response (real, length L = FFT length) of the filter applied to one projection:
 * the DFT of the spatial Ram–Lak kernel (exact ramp |f| sampled the right way, nonzero at DC
 * because the projection is finite) times the taper. kind "none": all ones.
 */
export function filterResponse(kind: FilterKind, L: number, fc = 0.5): Float64Array {
  const H = new Float64Array(L);
  if (kind === "none") { H.fill(1); return H; }
  const re = new Float64Array(L), im = new Float64Array(L);
  for (let k = 0; k < L; k++) { const j = k < L / 2 ? k : k - L; re[k] = ramLak(j); }
  fft1d(re, im);
  for (let k = 0; k < L; k++) { const f = (k < L / 2 ? k : k - L) / L; H[k] = re[k] * taper(kind, f, fc); }
  return H;
}

/** The ideal ramp |f| times the taper (for plotting the response next to the sampled one). */
export const idealResponse = (kind: FilterKind, f: number, fc = 0.5): number => (kind === "none" ? 1 : Math.abs(f) * taper(kind, f, fc));

export interface FilterOptions {
  kind?: FilterKind;
  /** cutoff in cycles per bin (0.5 = Nyquist) */
  cutoff?: number;
  /** zero-pad to at least 2 nBins (default true); false: circular convolution (wrap-around) */
  pad?: boolean;
}

/** FFT length used by `filterSinogram`. */
export const filterLength = (nBins: number, pad = true): number => (pad ? nextPow2(2 * nBins) : nextPow2(nBins));

/** Filter every projection (row) of a sinogram with the (tapered) ramp, via the FFT. */
export function filterSinogram(sino: Arr, nAngles: number, nBins: number, o: FilterOptions = {}, dRho = 1): Float64Array {
  const kind = o.kind ?? "ramp", out = new Float64Array(nAngles * nBins);
  if (kind === "none") { out.set(sino); return out; }
  const L = filterLength(nBins, o.pad ?? true), H = filterResponse(kind, L, o.cutoff ?? 0.5);
  const re = new Float64Array(L), im = new Float64Array(L);
  for (let a = 0; a < nAngles; a++) {
    re.fill(0); im.fill(0);
    for (let b = 0; b < nBins; b++) re[b] = sino[a * nBins + b];
    fft1d(re, im);
    for (let k = 0; k < L; k++) { re[k] *= H[k]; im[k] *= H[k]; }
    fft1d(re, im, true);
    for (let b = 0; b < nBins; b++) out[a * nBins + b] = re[b] / (L * dRho);
  }
  return out;
}

/** Direct (non-FFT) convolution of a projection with the truncated Ram–Lak kernel |k| <= half. */
export function filterDirect(p: ArrayLike<number>, half = p.length): Float64Array {
  const n = p.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) if (Math.abs(i - j) <= half) s += ramLak(i - j) * p[j];
    out[i] = s;
  }
  return out;
}

/** Filtered back-projection: dtheta * A^T (h * p). kind "none" gives plain back-projection. */
export function fbp(sino: Arr, g: Geometry, o: FilterOptions = {}): Float64Array {
  const q = filterSinogram(sino, g.angles.length, g.nBins, o, g.dRho);
  return backproject(q, g, angleStep(g.angles));
}

// ---------------------------------------------------------------------------------------
// Noise: photon counts

export interface NoisySinogram { sino: Float64Array; starved: number; scale: number }

/**
 * Low-dose measurement. The sinogram is first scaled so that its largest line integral is
 * `maxAtten` (default 3: e^-3 = 5 % of the photons get through the thickest ray); each ray
 * gets counts ~ Poisson(I0 exp(-scale p)), and the measurement is -ln(count / I0) / scale
 * (a count of 0 is floored to 1: "photon starvation"). I0 = Infinity: no noise.
 */
export function noisySinogram(sino: Arr, I0: number, seed = 1, maxAtten = 3): NoisySinogram {
  let mx = 0;
  for (let i = 0; i < sino.length; i++) if (sino[i] > mx) mx = sino[i];
  const scale = mx > 0 ? maxAtten / mx : 1, out = new Float64Array(sino.length);
  if (!Number.isFinite(I0)) { out.set(sino); return { sino: out, starved: 0, scale }; }
  const r = rng(seed);
  let starved = 0;
  for (let i = 0; i < sino.length; i++) {
    let k = poisson(r, I0 * Math.exp(-scale * Math.max(0, sino[i])));
    if (k < 1) { k = 1; starved++; }
    out[i] = -Math.log(k / I0) / scale;
  }
  return { sino: out, starved, scale };
}

// ---------------------------------------------------------------------------------------
// Iterative reconstruction

/** Row sums A 1 and column sums A^T 1 (the SIRT weights are their inverses). */
export function rowColSums(g: Geometry): { row: Float64Array; col: Float64Array } {
  const one = new Float64Array(g.n * g.n).fill(1);
  const row = project(one, g);
  const col = backproject(new Float64Array(g.angles.length * g.nBins).fill(1), g);
  return { row, col };
}

export type Method = "sirt" | "art";

export interface IterOptions {
  method?: Method;
  /** relaxation omega (default 1) */
  omega?: number;
  /** clip negative values after each step */
  nonneg?: boolean;
  /** TV weight lambda (> 0: a TV denoising step after each data step) */
  tv?: number;
  tvIters?: number;
  /** ART: order of the angles (default: a fixed spread-out order) */
  order?: Int32Array;
}

/** A spread-out ordering of the angles (multiplicative stride close to m / golden ratio). */
export function spreadOrder(m: number): Int32Array {
  const o = new Int32Array(m);
  if (m === 0) return o;
  let stride = Math.max(1, Math.round(m / 1.618));
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
  while (gcd(stride, m) !== 1) stride++;
  for (let k = 0; k < m; k++) o[k] = (k * stride) % m;
  return o;
}

/**
 * Iterative solver of A x = b, one call to `step()` per iteration (SIRT: one simultaneous
 * update; ART: one sweep over all rays). Keeps the current image `x`.
 */
export class IterativeRecon {
  readonly g: Geometry;
  readonly b: Float64Array;
  readonly x: Float64Array;
  readonly o: IterOptions;
  iter = 0;
  private W: { row: Float64Array; col: Float64Array } | null = null;
  private norms: Float64Array | null = null;
  private order: Int32Array;
  private ax: Float64Array;
  private tmp: Float64Array;

  constructor(b: Arr, g: Geometry, o: IterOptions = {}, x0?: Arr) {
    this.g = g; this.o = o;
    this.b = Float64Array.from(b);
    this.x = new Float64Array(g.n * g.n);
    if (x0) this.x.set(x0);
    this.order = o.order ?? spreadOrder(g.angles.length);
    this.ax = new Float64Array(g.angles.length * g.nBins);
    this.tmp = new Float64Array(g.n * g.n);
  }

  step(): void {
    const g = this.g, x = this.x, om = this.o.omega ?? 1;
    if ((this.o.method ?? "sirt") === "sirt") {
      if (!this.W) this.W = rowColSums(g);
      const { row, col } = this.W;
      project(x, g, this.ax);
      for (let i = 0; i < this.ax.length; i++) this.ax[i] = row[i] > 1e-9 ? (this.b[i] - this.ax[i]) / row[i] : 0;
      backproject(this.ax, g, 1, this.tmp);
      for (let p = 0; p < x.length; p++) if (col[p] > 1e-9) x[p] += om * this.tmp[p] / col[p];
    } else {
      const nB = g.nBins;
      if (!this.norms) {
        this.norms = new Float64Array(g.angles.length * nB);
        for (let a = 0; a < g.angles.length; a++) for (let k = 0; k < nB; k++)
          this.norms[a * nB + k] = rayNorm2(g.n, g.angles[a], rhoOf(k, nB, g.dRho));
      }
      for (const a of this.order) {
        const th = g.angles[a], c = Math.cos(th), s = Math.sin(th);
        for (let k = 0; k < nB; k++) {
          const i = a * nB + k, nn = this.norms[i];
          if (nn < 1e-9) continue;
          const rho = rhoOf(k, nB, g.dRho);
          const r = this.b[i] - rayDot(x, g.n, c, s, rho);
          rayAdd(x, g.n, c, s, rho, om * r / nn);
        }
        if (this.o.nonneg) for (let p = 0; p < x.length; p++) if (x[p] < 0) x[p] = 0;
      }
    }
    if (this.o.nonneg || (this.o.tv ?? 0) > 0) for (let p = 0; p < x.length; p++) if (x[p] < 0) x[p] = 0;
    if ((this.o.tv ?? 0) > 0) {
      const y = tv2d(x, g.n, g.n, this.o.tv!, this.o.tvIters ?? 40);
      for (let p = 0; p < x.length; p++) x[p] = Math.max(0, y[p]);
    }
    this.iter++;
  }

  /** Relative data residual ||A x - b|| / ||b||. */
  residual(): number {
    project(this.x, this.g, this.ax);
    let num = 0, den = 0;
    for (let i = 0; i < this.ax.length; i++) { num += (this.ax[i] - this.b[i]) ** 2; den += this.b[i] ** 2; }
    return Math.sqrt(num / Math.max(den, 1e-30));
  }
}

// ---------------------------------------------------------------------------------------
// Explicit matrix (small images)

export interface Csr { rows: number; cols: number; rowPtr: Int32Array; col: Int32Array; val: Float64Array }

/** The projector as an explicit sparse matrix (one row per ray, angle-major), for small n. */
export function systemMatrix(g: Geometry): Csr {
  const rows = g.angles.length * g.nBins, rowPtr = new Int32Array(rows + 1);
  const cols: number[] = [], vals: number[] = [];
  for (let a = 0; a < g.angles.length; a++) for (let b = 0; b < g.nBins; b++) {
    const { idx, w } = rayWeights(g.n, g.angles[a], rhoOf(b, g.nBins, g.dRho));
    cols.push(...idx); vals.push(...w);
    rowPtr[a * g.nBins + b + 1] = cols.length;
  }
  return { rows, cols: g.n * g.n, rowPtr, col: Int32Array.from(cols), val: Float64Array.from(vals) };
}

export function csrMul(A: Csr, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(A.rows);
  for (let i = 0; i < A.rows; i++) { let s = 0; for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) s += A.val[k] * x[A.col[k]]; y[i] = s; }
  return y;
}

export function csrMulT(A: Csr, y: ArrayLike<number>): Float64Array {
  const x = new Float64Array(A.cols);
  for (let i = 0; i < A.rows; i++) for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) x[A.col[k]] += A.val[k] * y[i];
  return x;
}

// ---------------------------------------------------------------------------------------
// Analytic projections and test objects (pixel units: world coordinates as above)

/** An ellipse in world pixel units: centre (x, y), semi-axes a (along its own x) and b, angle in degrees CCW, added value. */
export interface WEllipse { x: number; y: number; a: number; b: number; angle: number; value: number }

/** Exact line integral of an ellipse along (theta, rho): 2 v a b sqrt(s^2 - t^2) / s^2. */
export function ellipseProjection(e: WEllipse, theta: number, rho: number): number {
  const phi = e.angle * Math.PI / 180, d = theta - phi;
  const s2 = (e.a * Math.cos(d)) ** 2 + (e.b * Math.sin(d)) ** 2;
  const t = rho - (e.x * Math.cos(theta) + e.y * Math.sin(theta));
  return t * t >= s2 ? 0 : 2 * e.value * e.a * e.b * Math.sqrt(s2 - t * t) / s2;
}

/** The (modified) Shepp–Logan ellipses of lib/synth.ts in world pixel units of an n x n image. */
export function sheppLoganEllipses(n: number): WEllipse[] {
  const k = n / 2;
  return SHEPP_LOGAN.map(e => ({ x: e.x * k, y: e.y * k, a: e.a * k, b: e.b * k, angle: e.angle, value: e.value }));
}

/** Exact sinogram of a set of ellipses. */
export function ellipseSinogram(el: WEllipse[], g: Geometry): Float64Array {
  const out = new Float64Array(g.angles.length * g.nBins);
  for (let a = 0; a < g.angles.length; a++) for (let b = 0; b < g.nBins; b++) {
    let s = 0;
    for (const e of el) s += ellipseProjection(e, g.angles[a], rhoOf(b, g.nBins, g.dRho));
    out[a * g.nBins + b] = s;
  }
  return out;
}

/** Rasterise ellipses (world pixel units) with ss x ss supersampling per pixel. */
export function ellipseImage(n: number, el: WEllipse[], ss = 4): Float64Array {
  const img = new Float64Array(n * n), cen = centre(n);
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
    let v = 0;
    for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++) {
      const x = q - cen - 0.5 + (i + 0.5) / ss, y = cen - r + 0.5 - (j + 0.5) / ss;
      for (const e of el) {
        const t = e.angle * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
        const u = c * (x - e.x) + s * (y - e.y), w = -s * (x - e.x) + c * (y - e.y);
        if ((u / e.a) ** 2 + (w / e.b) ** 2 <= 1) v += e.value;
      }
    }
    img[r * n + q] = v / (ss * ss);
  }
  return img;
}

/** A disc of radius R (world pixel units) as an ellipse. */
export const disc = (x: number, y: number, R: number, value = 1): WEllipse => ({ x, y, a: R, b: R, angle: 0, value });

/** Axis-aligned square of half-side h centred at (x, y), supersampled. */
export function squareImage(n: number, h: number, x = 0, y = 0, value = 1, ss = 4): Float64Array {
  const img = new Float64Array(n * n), cen = centre(n);
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
    let v = 0;
    for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++) {
      const xx = q - cen - 0.5 + (i + 0.5) / ss, yy = cen - r + 0.5 - (j + 0.5) / ss;
      if (Math.abs(xx - x) <= h && Math.abs(yy - y) <= h) v += value;
    }
    img[r * n + q] = v / (ss * ss);
  }
  return img;
}

/** Add a point mass m at world (x, y), spread over the 4 nearest pixels with bilinear weights. */
export function addPoint(img: Arr, n: number, x: number, y: number, m: number): void {
  const [col, row] = toPixel(x, y, n);
  const i0 = Math.floor(col), j0 = Math.floor(row), fx = col - i0, fy = row - j0;
  const put = (i: number, j: number, w: number) => { if (i >= 0 && i < n && j >= 0 && j < n) img[j * n + i] += m * w; };
  put(i0, j0, (1 - fx) * (1 - fy)); put(i0 + 1, j0, fx * (1 - fy));
  put(i0, j0 + 1, (1 - fx) * fy); put(i0 + 1, j0 + 1, fx * fy);
}

/** 1 inside the inscribed circle (radius n/2 - 1, the field of view every angle sees), soft 2-pixel edge. */
export function fovMask(n: number): Float64Array {
  const m = new Float64Array(n * n), cen = centre(n), R = n / 2 - 1;
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
    const d = Math.hypot(q - cen, r - cen);
    m[r * n + q] = d <= R - 2 ? 1 : d >= R ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * (d - R + 2) / 2);
  }
  return m;
}

// ---------------------------------------------------------------------------------------
// Small helpers

export function rmseOf(a: ArrayLike<number>, b: ArrayLike<number>, mask?: ArrayLike<number>): number {
  let s = 0, k = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i] > 0.5) { s += (a[i] - b[i]) ** 2; k++; }
  return Math.sqrt(s / Math.max(k, 1));
}

export function dotOf(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
