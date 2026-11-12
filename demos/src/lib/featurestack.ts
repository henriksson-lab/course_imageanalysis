// The multiscale pixel feature stack (owner: random-forest day, day 14). No DOM.
//
// Each filter family at each scale σ gives one channel; a pixel's feature vector is its value
// in every channel. Used by the pixel-classification shell (pixelclass) for the forest, and
// by the SVM, integral-images and neural-network days.
//
//   const S = featureStack(img, ["intensity", "edges", "blobs", "ridges", "texture"], [0.7, 1, 1.6, 3.5, 5, 10]);
//   S.data[c * S.w * S.h + p]   // channel c at pixel p  (feature-major, as lib/classifier.ts expects)
//
// Families (all from the Gaussian-smoothed image L_σ = G_σ * I; the filters themselves are
// taught on day 1 (kernels/) and on the edges day):
//   intensity  L_σ                                     (1 channel per σ)
//   edges      gradient magnitude |∇L_σ|               (1)
//   blobs      scale-normalised Laplacian σ² ∇²L_σ      (1)
//   ridges     Hessian eigenvalues σ² λ1 ≥ σ² λ2        (2)
//   texture    structure-tensor eigenvalues μ1 ≥ μ2 of G_{σ/2} * (∇L_σ ∇L_σᵀ), times σ²   (2)
// Derivatives are central differences of L_σ (close to Gaussian derivatives for σ ≥ 1).
// Smoothing is incremental: L_σ2 = G_√(σ2² − σ1²) * L_σ1, so the stack costs about one blur
// per scale plus the structure-tensor smoothing.
//
// More families (e.g. box mean / local variance / Haar from the integral-images day) are
// added with `registerFeatureGroup` without editing this file.
import { Grey, gaussianBlur } from "./imageops";

export const DEFAULT_SIGMAS = [0.7, 1, 1.6, 3.5, 5, 10];

export interface FeatureStack {
  w: number;
  h: number;
  /** Number of channels. */
  d: number;
  /** Feature-major: data[c * w * h + p]. */
  data: Float32Array;
  names: string[];
  /** Family id and σ of every channel. */
  groups: string[];
  sigmas: number[];
}

/** What a family computes at one scale. */
export interface ScaleContext {
  img: Grey;
  sigma: number;
  /** L_σ (cached). */
  L(): Float32Array;
  /** Central-difference derivatives of L_σ (cached). */
  Lx(): Float32Array; Ly(): Float32Array;
  Lxx(): Float32Array; Lyy(): Float32Array; Lxy(): Float32Array;
}

export interface FeatureGroup {
  id: string;
  label: string;
  /** Channel names at scale σ, e.g. ["Hessian λ1 σ=1", "Hessian λ2 σ=1"]. */
  channels(sigma: number): string[];
  /** One Float32Array (w·h) per channel. */
  compute(ctx: ScaleContext): Float32Array[];
}

const fmtS = (s: number) => `σ=${s}`;

function eig2(a: Float32Array, b: Float32Array, c: Float32Array, scale: number): [Float32Array, Float32Array] {
  const n = a.length, l1 = new Float32Array(n), l2 = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    const m = 0.5 * (a[p] + c[p]), r = Math.hypot(0.5 * (a[p] - c[p]), b[p]);
    l1[p] = scale * (m + r); l2[p] = scale * (m - r);
  }
  return [l1, l2];
}

const GROUPS: FeatureGroup[] = [
  { id: "intensity", label: "intensity", channels: s => [`smoothed ${fmtS(s)}`], compute: c => [c.L()] },
  {
    id: "edges", label: "edges", channels: s => [`gradient ${fmtS(s)}`],
    compute: c => { const gx = c.Lx(), gy = c.Ly(), m = new Float32Array(gx.length); for (let p = 0; p < m.length; p++) m[p] = Math.hypot(gx[p], gy[p]); return [m]; },
  },
  {
    id: "blobs", label: "blobs", channels: s => [`LoG ${fmtS(s)}`],
    compute: c => { const a = c.Lxx(), b = c.Lyy(), s2 = c.sigma * c.sigma, m = new Float32Array(a.length); for (let p = 0; p < m.length; p++) m[p] = s2 * (a[p] + b[p]); return [m]; },
  },
  {
    id: "ridges", label: "ridges", channels: s => [`Hessian λ1 ${fmtS(s)}`, `Hessian λ2 ${fmtS(s)}`],
    compute: c => eig2(c.Lxx(), c.Lxy(), c.Lyy(), c.sigma * c.sigma),
  },
  {
    id: "texture", label: "texture", channels: s => [`struct. tensor μ1 ${fmtS(s)}`, `struct. tensor μ2 ${fmtS(s)}`],
    compute: c => {
      const gx = c.Lx(), gy = c.Ly(), n = gx.length, w = c.img.width, h = c.img.height;
      const xx = new Float32Array(n), xy = new Float32Array(n), yy = new Float32Array(n);
      for (let p = 0; p < n; p++) { xx[p] = gx[p] * gx[p]; xy[p] = gx[p] * gy[p]; yy[p] = gy[p] * gy[p]; }
      const so = c.sigma / 2;
      const sm = (a: Float32Array) => (so >= 0.3 ? gaussianBlur({ width: w, height: h, data: a }, so).data : a);
      return eig2(sm(xx), sm(xy), sm(yy), c.sigma * c.sigma);
    },
  },
];

/** Add (or replace) a feature family. */
export function registerFeatureGroup(g: FeatureGroup): void {
  const i = GROUPS.findIndex(q => q.id === g.id);
  if (i >= 0) GROUPS[i] = g; else GROUPS.push(g);
}

/** The registered families. */
export function featureGroups(): FeatureGroup[] { return [...GROUPS]; }

function diffs(L: Float32Array, w: number, h: number) {
  const cache: Record<string, Float32Array> = {};
  const idx = (x: number, y: number) => Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x));
  const make = (f: (x: number, y: number) => number) => { const o = new Float32Array(w * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) o[y * w + x] = f(x, y); return o; };
  return {
    Lx: () => cache.x ??= make((x, y) => 0.5 * (L[idx(x + 1, y)] - L[idx(x - 1, y)])),
    Ly: () => cache.y ??= make((x, y) => 0.5 * (L[idx(x, y + 1)] - L[idx(x, y - 1)])),
    Lxx: () => cache.xx ??= make((x, y) => L[idx(x + 1, y)] - 2 * L[idx(x, y)] + L[idx(x - 1, y)]),
    Lyy: () => cache.yy ??= make((x, y) => L[idx(x, y + 1)] - 2 * L[idx(x, y)] + L[idx(x, y - 1)]),
    Lxy: () => cache.xy ??= make((x, y) => 0.25 * (L[idx(x + 1, y + 1)] - L[idx(x - 1, y + 1)] - L[idx(x + 1, y - 1)] + L[idx(x - 1, y - 1)])),
  };
}

/**
 * Build the stack: for every σ (ascending order is fastest) and every requested family, its
 * channels. Channel order: σ outer, family inner (as listed in `groups`).
 */
export function featureStack(img: Grey, groups: string[], sigmas: number[] = DEFAULT_SIGMAS): FeatureStack {
  const w = img.width, h = img.height, N = w * h;
  const fams = groups.map(id => GROUPS.find(g => g.id === id)).filter((g): g is FeatureGroup => !!g);
  const names: string[] = [], gs: string[] = [], ss: number[] = [], chans: Float32Array[] = [];
  const sorted = [...sigmas].sort((a, b) => a - b);
  let prevS = 0, prevL: Grey = img;
  for (const s of sorted) {
    const inc = Math.sqrt(Math.max(0, s * s - prevS * prevS));
    const Lg = inc > 0.05 ? gaussianBlur(prevL, inc) : prevL;
    prevS = s; prevL = Lg;
    const D = diffs(Lg.data, w, h);
    const ctx: ScaleContext = { img, sigma: s, L: () => Lg.data, ...D };
    for (const f of fams) {
      const out = f.compute(ctx), nm = f.channels(s);
      out.forEach((a, q) => { chans.push(a); names.push(nm[q] ?? `${f.id} ${fmtS(s)}`); gs.push(f.id); ss.push(s); });
    }
  }
  const d = chans.length, data = new Float32Array(d * N);
  chans.forEach((a, c) => data.set(a, c * N));
  return { w, h, d, data, names, groups: gs, sigmas: ss };
}

/** Keep only the channels whose family and σ pass the filter (a view-free copy). */
export function selectChannels(S: FeatureStack, keep: (group: string, sigma: number, c: number) => boolean): FeatureStack {
  const N = S.w * S.h, idx: number[] = [];
  for (let c = 0; c < S.d; c++) if (keep(S.groups[c], S.sigmas[c], c)) idx.push(c);
  const data = new Float32Array(idx.length * N);
  idx.forEach((c, q) => data.set(S.data.subarray(c * N, (c + 1) * N), q * N));
  return { w: S.w, h: S.h, d: idx.length, data, names: idx.map(c => S.names[c]), groups: idx.map(c => S.groups[c]), sigmas: idx.map(c => S.sigmas[c]) };
}

/** Feature vector of pixel p. */
export function featureVector(S: FeatureStack, p: number): Float32Array {
  const N = S.w * S.h, v = new Float32Array(S.d);
  for (let c = 0; c < S.d; c++) v[c] = S.data[c * N + p];
  return v;
}

/** Channel c as a Grey image (a view on the stack's memory). */
export function channelImage(S: FeatureStack, c: number): Grey {
  const N = S.w * S.h;
  return { width: S.w, height: S.h, data: S.data.subarray(c * N, (c + 1) * N) };
}

/** Training table from chosen pixels: feature-major X (d × n) for the pixels `pix`. */
export function gatherPixels(S: FeatureStack, pix: ArrayLike<number>): Float32Array {
  const N = S.w * S.h, n = pix.length, X = new Float32Array(S.d * n);
  for (let c = 0; c < S.d; c++) { const off = c * N, o2 = c * n; for (let i = 0; i < n; i++) X[o2 + i] = S.data[off + pix[i]]; }
  return X;
}
