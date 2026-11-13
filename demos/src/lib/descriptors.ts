// Local descriptors (no DOM). Owner: the keypoints day (descriptors/).
//
// A keypoint (x, y, σ, θ) fixes a small coordinate frame; the window is resampled IN that
// frame (sampleFrame), so translation, zoom and rotation are removed before anything is
// computed. Then:
// - patchDescriptor: the normalised patch (mean 0, unit length: the dot product is the NCC);
// - gradHistDescriptor: 4 × 4 cells × 8 direction bins = 128 numbers, Gaussian-weighted,
//   soft (trilinear) binning, unit length, clipped at 0.2 and renormalised; optional root
//   descriptor (L1-normalise, square root: Euclidean distance = Hellinger distance);
// - binaryPattern / binaryDescriptor: 256 seeded point pairs on a smoothed patch, one bit per
//   pair (I(p) < I(q)), packed into 8 32-bit words, steered by θ;
// - intensityCentroidAngle; distances l2, l1, chi2, hamming (XOR + popcount), distanceFn.
//
// Frame convention: θ counter-clockwise on the screen (lib/keypoints.ts). In image
// coordinates (y down) the frame axes are u = (cos θ, −sin θ) and v = (sin θ, cos θ) (u turned
// 90° clockwise on the screen, like x → y). Patch sample (i, j) (column i, row j) sits at
// centre + spacing ((i − c) u + (j − c) v), c = (n − 1)/2.

import { Grey, createGrey, sampleBilinear } from "./imageops";
import { DoGPyramid, levelImage, Keypoint } from "./keypoints";
import { rng } from "./random";

export type Desc = Float32Array | Uint32Array;

/** n × n patch sampled bilinearly at centre + spacing·((i − c) u + (j − c) v). */
export function sampleFrame(img: Grey, cx: number, cy: number, spacing: number, theta: number, n: number): Grey {
  const out = createGrey(n, n), c = (n - 1) / 2;
  const ux = Math.cos(theta), uy = -Math.sin(theta), vx = Math.sin(theta), vy = Math.cos(theta);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const a = (i - c) * spacing, b = (j - c) * spacing;
    out.data[j * n + i] = sampleBilinear(img, cx + a * ux + b * vx, cy + a * uy + b * vy, "replicate");
  }
  return out;
}

/** Map a frame coordinate (a, b) (in samples from the centre) to image-1 coordinates (for drawing). */
export function frameToImage(kp: { x: number; y: number; theta: number }, spacingOrig: number, a: number, b: number, rotate = true): [number, number] {
  const t = rotate ? kp.theta : 0;
  const ux = Math.cos(t), uy = -Math.sin(t), vx = Math.sin(t), vy = Math.cos(t);
  return [kp.x + spacingOrig * (a * ux + b * vx), kp.y + spacingOrig * (a * uy + b * vy)];
}

export interface FrameOptions {
  /** Window width in units of σ (default 12: 4 cells of 3σ). */
  width?: number;
  /** Samples across (default 16). */
  n?: number;
  /** Sample in the keypoint's orientation (false: θ = 0, rotation not removed). */
  rotate?: boolean;
}

/** The keypoint window resampled in its frame from the pyramid level nearest σ. Extra `pad` samples on each side. */
export function keypointPatch(p: DoGPyramid, kp: Keypoint, o: FrameOptions = {}, pad = 0): { patch: Grey; spacing: number; spacingOrig: number } {
  const n = o.n ?? 16, width = o.width ?? 12, L = levelImage(p, kp);
  const spacing = width * L.s / n;
  const patch = sampleFrame(L.img, L.cx, L.cy, spacing, o.rotate === false ? 0 : kp.theta, n + 2 * pad);
  return { patch, spacing, spacingOrig: spacing * L.step };
}

// ---------------------------------------------------------------------------------------
// The normalised patch

/** Patch minus its mean, divided by its length: the dot product of two such vectors is their NCC. */
export function normalisePatch(v: ArrayLike<number>): Float32Array {
  const n = v.length; let m = 0;
  for (let i = 0; i < n; i++) m += v[i] / n;
  const out = new Float32Array(n); let s = 0;
  for (let i = 0; i < n; i++) { out[i] = v[i] - m; s += out[i] * out[i]; }
  const k = s > 0 ? 1 / Math.sqrt(s) : 0;
  for (let i = 0; i < n; i++) out[i] *= k;
  return out;
}

export function patchDescriptor(p: DoGPyramid, kp: Keypoint, o: FrameOptions = {}): Float32Array {
  return normalisePatch(keypointPatch(p, kp, { n: 16, ...o }).patch.data);
}

// ---------------------------------------------------------------------------------------
// Gradient histograms

export interface GradHistOptions extends FrameOptions {
  cells?: number;
  bins?: number;
  /** Trilinear vote sharing between neighbouring cells and bins (false: nearest cell and bin). */
  soft?: boolean;
  /** Gaussian weight over the window (σ = half the width). */
  weight?: boolean;
  /** Normalise to unit length. */
  normalise?: boolean;
  /** Clip entries at this value and renormalise (0 = no clipping). */
  clip?: number;
  /** Root descriptor: L1-normalise, then square roots. */
  root?: boolean;
}

export interface GradHistDetail {
  desc: Float32Array;
  /** Gradient of every patch sample in the frame (n × n), angle counter-clockwise in the frame. */
  gx: Float32Array; gy: Float32Array; mag: Float32Array; angle: Float32Array; w: Float32Array;
  patch: Grey;
  n: number;
}

/** The 128-number gradient-histogram descriptor of an (n + 2)² padded patch (the core, for the demo and the tests). */
export function gradHistFromPatch(padded: Grey, o: GradHistOptions = {}): GradHistDetail {
  const cells = o.cells ?? 4, nb = o.bins ?? 8, soft = o.soft !== false;
  const N = padded.width, n = N - 2, d = padded.data;
  const gx = new Float32Array(n * n), gy = new Float32Array(n * n), mag = new Float32Array(n * n), ang = new Float32Array(n * n), wt = new Float32Array(n * n);
  const desc = new Float32Array(cells * cells * nb);
  const c = (n - 1) / 2, sw = n / 2, cw = n / cells;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = i + 1, y = j + 1;
    const dx = (d[y * N + x + 1] - d[y * N + x - 1]) / 2, dy = (d[(y + 1) * N + x] - d[(y - 1) * N + x]) / 2;
    const m = Math.hypot(dx, dy);
    let a = Math.atan2(-dy, dx); if (a < 0) a += 2 * Math.PI;
    const k = j * n + i;
    gx[k] = dx; gy[k] = dy; mag[k] = m; ang[k] = a;
    const g = o.weight === false ? 1 : Math.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sw * sw));
    wt[k] = g;
    const vote = m * g;
    if (vote === 0) continue;
    // continuous cell coordinates (cell centres at integers) and bin coordinate (bin k centred at k·360°/nb)
    const cu = (i + 0.5) / cw - 0.5, cv = (j + 0.5) / cw - 0.5, ob = a / (2 * Math.PI) * nb;
    if (!soft) {
      const ci = Math.min(cells - 1, Math.floor((i) / cw)), cj = Math.min(cells - 1, Math.floor(j / cw)), bi = Math.round(ob) % nb;
      desc[(cj * cells + ci) * nb + bi] += vote;
      continue;
    }
    const u0 = Math.floor(cu), v0 = Math.floor(cv), o0 = Math.floor(ob);
    const fu = cu - u0, fv = cv - v0, fo = ob - o0;
    for (let a2 = 0; a2 < 2; a2++) {
      const cj = v0 + a2; if (cj < 0 || cj >= cells) continue;
      const wv = a2 ? fv : 1 - fv;
      for (let b2 = 0; b2 < 2; b2++) {
        const ci = u0 + b2; if (ci < 0 || ci >= cells) continue;
        const wu = b2 ? fu : 1 - fu;
        for (let c2 = 0; c2 < 2; c2++) {
          const bi = (o0 + c2) % nb, wo = c2 ? fo : 1 - fo;
          desc[(cj * cells + ci) * nb + bi] += vote * wu * wv * wo;
        }
      }
    }
  }
  finishDescriptor(desc, o);
  const patch = createGrey(n, n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) patch.data[j * n + i] = d[(j + 1) * N + i + 1];
  return { desc, gx, gy, mag, angle: ang, w: wt, patch, n };
}

/** Normalise, clip, renormalise; root descriptor if asked (in place). */
export function finishDescriptor(desc: Float32Array, o: GradHistOptions = {}): void {
  const norm = () => { let s = 0; for (const v of desc) s += v * v; const k = s > 0 ? 1 / Math.sqrt(s) : 0; for (let i = 0; i < desc.length; i++) desc[i] *= k; };
  if (o.normalise === false) return;
  norm();
  const clip = o.clip ?? 0.2;
  if (clip > 0) { for (let i = 0; i < desc.length; i++) desc[i] = Math.min(desc[i], clip); norm(); }
  if (o.root) {
    let s = 0; for (const v of desc) s += Math.abs(v);
    for (let i = 0; i < desc.length; i++) desc[i] = s > 0 ? Math.sqrt(Math.abs(desc[i]) / s) : 0;
  }
}

/** Gradient-histogram descriptor of a keypoint (128 numbers for the defaults). */
export function gradHistDescriptor(p: DoGPyramid, kp: Keypoint, o: GradHistOptions = {}): Float32Array {
  return gradHistDetail(p, kp, o).desc;
}

export function gradHistDetail(p: DoGPyramid, kp: Keypoint, o: GradHistOptions = {}): GradHistDetail & { spacingOrig: number } {
  const { patch, spacingOrig } = keypointPatch(p, kp, o, 1);
  return { ...gradHistFromPatch(patch, o), spacingOrig };
}

/** Histogram of one cell by nearest-bin voting: gradients (magnitude, angle in degrees), bins centred at k·360°/nb. */
export function cellHistogram(grads: [number, number][], nb = 8): number[] {
  const h = new Array(nb).fill(0);
  for (const [m, deg] of grads) {
    const b = ((Math.round(deg / (360 / nb)) % nb) + nb) % nb;
    h[b] += m;
  }
  return h;
}

// ---------------------------------------------------------------------------------------
// Binary descriptors

export interface BinaryPattern {
  /** Pairs (p, q) in patch samples relative to the centre: [px, py, qx, qy] × n. */
  pairs: Float32Array;
  n: number;
  /** Patch size in samples. */
  size: number;
}

/** n seeded point pairs, isotropic Gaussian (σ = size/5) clipped to the patch. */
export function binaryPattern(seed = 7, n = 256, size = 32): BinaryPattern {
  const r = rng(seed), pairs = new Float32Array(4 * n), lim = size / 2 - 1.5, sd = size / 5;
  for (let k = 0; k < 4 * n; k++) {
    let v: number;
    do v = Math.round(sd * r.norm()); while (Math.abs(v) > lim);
    pairs[k] = v;
  }
  return { pairs, n, size };
}

export interface BinaryOptions { /** Rotate the pattern by θ (steered). */ steer?: boolean; /** Window width in σ (default 12). */ width?: number }

/** Smoothed patch for the binary tests (sampled in the frame from the level image, which is already blurred by σ). */
export function binaryPatch(p: DoGPyramid, kp: Keypoint, pat: BinaryPattern, o: BinaryOptions = {}): { patch: Grey; spacingOrig: number } {
  const L = levelImage(p, kp), width = o.width ?? 12, spacing = width * L.s / pat.size;
  return { patch: sampleFrame(L.img, L.cx, L.cy, spacing, o.steer === false ? 0 : kp.theta, pat.size), spacingOrig: spacing * L.step };
}

/** Bits b_k = [I(p_k) < I(q_k)] of a patch, packed 32 per word (bit k in word k >> 5, position k & 31). */
export function binaryFromPatch(patch: Grey, pat: BinaryPattern): Uint32Array {
  const out = new Uint32Array(Math.ceil(pat.n / 32)), S = patch.width, c = (S - 1) / 2;
  for (let k = 0; k < pat.n; k++) {
    const px = pat.pairs[4 * k] + c, py = pat.pairs[4 * k + 1] + c, qx = pat.pairs[4 * k + 2] + c, qy = pat.pairs[4 * k + 3] + c;
    const a = sampleBilinear(patch, px, py), b = sampleBilinear(patch, qx, qy);
    if (a < b) out[k >> 5] |= (1 << (k & 31)) >>> 0;
  }
  return out;
}

export function binaryDescriptor(p: DoGPyramid, kp: Keypoint, pat: BinaryPattern, o: BinaryOptions = {}): Uint32Array {
  return binaryFromPatch(binaryPatch(p, kp, pat, o).patch, pat);
}

export const getBit = (b: Uint32Array, k: number): number => (b[k >> 5] >>> (k & 31)) & 1;

/**
 * Orientation from the intensity centroid: θ = atan2(−m01, m10) (counter-clockwise on the
 * screen) with m_pq = Σ x^p y^q I over a disc of radius r around (cx, cy), x, y relative.
 */
export function intensityCentroidAngle(img: Grey, cx: number, cy: number, r: number): number {
  let m10 = 0, m01 = 0;
  const R = Math.ceil(r);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    if (dx * dx + dy * dy > r * r) continue;
    const v = sampleBilinear(img, cx + dx, cy + dy, "replicate");
    m10 += dx * v; m01 += dy * v;
  }
  return Math.atan2(-m01, m10);
}

/** Keypoints with θ from the intensity centroid of a disc of radius 6σ (at the level image). */
export function centroidOrientations(p: DoGPyramid, kps: Keypoint[]): Keypoint[] {
  return kps.map(kp => { const L = levelImage(p, kp); let t = intensityCentroidAngle(L.img, L.cx, L.cy, 6 * L.s); if (t < 0) t += 2 * Math.PI; return { ...kp, theta: t }; });
}

// ---------------------------------------------------------------------------------------
// Distances

export type Metric = "l2" | "l1" | "chi2" | "hamming";

export function l2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
export function l1(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}
/** χ² distance ½ Σ (a − b)² / (a + b) (non-negative entries). */
export function chi2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < a.length; i++) { const t = a[i] + b[i]; if (t > 0) s += (a[i] - b[i]) ** 2 / t; }
  return s / 2;
}
/** Number of set bits in a 32-bit word. */
export function popcount(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}
/** Hamming distance: popcount(a XOR b), word by word. */
export function hamming(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += popcount((a[i] ^ b[i]) >>> 0);
  return s;
}
export function distanceFn(kind: Metric): (a: Desc, b: Desc) => number {
  return kind === "l1" ? l1 : kind === "chi2" ? chi2 : kind === "hamming" ? hamming : l2;
}
