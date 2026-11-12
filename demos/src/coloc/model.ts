// Model of the coloc demo (no DOM): two seeded synthetic fluorescence channels A and B,
// 128 × 128, inside two cells. A = structures + diffuse cell signal; B = a fraction f of the
// same structures + (1 − f) independent ones (or, in "inside" mode, large structures that
// contain A's), + diffuse cell signal, + bleed-through of A, shifted sideways; both get a
// background offset and Gaussian noise, and a blur of about the size of the PSF.
import { gaussianBlur, fromArray } from "../lib/imageops";
import { rng, gauss } from "../lib/random";
import { pearsonMasked, overlapCoefficient, manders, orthoRegression, blockScramble, costes, CostesResult } from "../lib/neighbours";
import { quantile } from "../lib/stats";

export const S = 128;

export interface ChannelOptions {
  /** Fraction of B's structures shared with A (0..1). */
  f?: number;
  /** "mix": B = f · shared + (1 − f) · independent; "inside": A's spots sit inside B's large blobs. */
  layout?: "mix" | "inside";
  offset?: number;
  noise?: number;
  /** Diffuse signal inside the cells, both channels. */
  diffuse?: number;
  /** Fraction of A that leaks into B. */
  bleed?: number;
  /** Sideways shift of B in pixels. */
  shift?: number;
  seed?: number;
}

export interface Channels { w: number; h: number; a: Float64Array; b: Float64Array; cells: Uint8Array }

function blobs(r: ReturnType<typeof rng>, inside: Uint8Array, n: number, rMin: number, rMax: number, centres?: [number, number, number][]): { img: Float64Array; list: [number, number, number][] } {
  const img = new Float64Array(S * S), list: [number, number, number][] = centres ? [...centres] : [];
  let guard = 0;
  while (!centres && list.length < n && guard++ < 5000) {
    const x = 4 + (S - 8) * r.unif(), y = 4 + (S - 8) * r.unif(), rad = rMin + (rMax - rMin) * r.unif();
    if (!inside[Math.round(y) * S + Math.round(x)]) continue;
    list.push([x, y, rad]);
  }
  for (const [x, y, rad] of list) {
    for (let yy = Math.max(0, Math.floor(y - 3 * rad)); yy <= Math.min(S - 1, Math.ceil(y + 3 * rad)); yy++)
      for (let xx = Math.max(0, Math.floor(x - 3 * rad)); xx <= Math.min(S - 1, Math.ceil(x + 3 * rad)); xx++)
        img[yy * S + xx] += Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * rad * rad));
  }
  return { img, list };
}

export function cellMask(): Uint8Array {
  const m = new Uint8Array(S * S);
  const cells = [[42, 50, 34, 26, 0.5], [90, 82, 30, 36, -0.3]];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) for (const [cx, cy, a, b, t] of cells) {
    const u = (x - cx) * Math.cos(t) + (y - cy) * Math.sin(t), v = -(x - cx) * Math.sin(t) + (y - cy) * Math.cos(t);
    if ((u / a) ** 2 + (v / b) ** 2 <= 1) m[y * S + x] = 1;
  }
  return m;
}

export function makeChannels(o: ChannelOptions): Channels {
  const r = rng(o.seed ?? 4);
  const cells = cellMask();
  const f = o.f ?? 0.5, off = o.offset ?? 10, sd = o.noise ?? 4, diff = o.diffuse ?? 0, bleed = o.bleed ?? 0, shift = Math.round(o.shift ?? 0);
  let A: Float64Array, Bs: Float64Array;
  if ((o.layout ?? "mix") === "inside") {
    const big = blobs(r, cells, 7, 7, 9);
    const small: [number, number, number][] = [];
    big.list.slice(0, 4).forEach(([x, y]) => { for (let k = 0; k < 3; k++) small.push([x + 3 * r.norm(), y + 3 * r.norm(), 1.4]); });
    A = blobs(r, cells, 0, 0, 0, small).img.map(v => 160 * v);
    Bs = big.img.map(v => 110 * Math.min(1, v));
  } else {
    const shared = blobs(r, cells, 40, 1.3, 2.5);
    const indep = blobs(r, cells, 40, 1.3, 2.5);
    A = shared.img.map(v => 150 * v);
    Bs = Float64Array.from(shared.img, (v, i) => 150 * (f * v + (1 - f) * indep.img[i]));
  }
  const blurA = gaussianBlur(fromArray(S, S, A), 1).data, blurB = gaussianBlur(fromArray(S, S, Bs), 1).data;
  const a = new Float64Array(S * S), b = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) a[i] = blurA[i] + diff * cells[i];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const xs = Math.min(S - 1, Math.max(0, x - shift)), j = y * S + xs;
    b[y * S + x] = blurB[j] + diff * cells[j] + bleed * a[j];
  }
  const r2 = rng((o.seed ?? 4) + 101);
  for (let i = 0; i < S * S; i++) {
    a[i] = Math.max(0, a[i] + off + gauss(r2, 0, sd));
    b[i] = Math.max(0, b[i] + off + gauss(r2, 0, sd));
  }
  return { w: S, h: S, a, b, cells };
}

// ---------------------------------------------------------------------------------------
// Readouts (used by the demo and by the exercise numbers)

export interface ColocStats { r: number; rAbove: number; nAbove: number; overlap: number; m1: number; m2: number; slope: number; intercept: number }

/** All readouts for a region (mask, or all pixels) and thresholds tA, tB. */
export function colocStats(ch: Channels, roi: Uint8Array | undefined, tA: number, tB: number): ColocStats {
  const r = pearsonMasked(ch.a, ch.b, roi);
  const above = Uint8Array.from(ch.a, (v, i) => ((!roi || roi[i]) && v > tA && ch.b[i] > tB ? 1 : 0));
  const nAbove = above.reduce((x, y) => x + y, 0);
  const md = manders(ch.a, ch.b, tA, tB, roi);
  const reg = orthoRegression(ch.a, ch.b, roi);
  return { r, rAbove: nAbove > 2 ? pearsonMasked(ch.a, ch.b, above) : NaN, nAbove, overlap: overlapCoefficient(ch.a, ch.b, roi), m1: md.m1, m2: md.m2, slope: reg.slope, intercept: reg.intercept };
}

/** Pearson r of A against B with B's 8 × 8 blocks shuffled, `n` times (seeded). */
export function randomisation(ch: Channels, roi: Uint8Array | undefined, n = 50, seed = 77): number[] {
  const r = rng(seed), out: number[] = [];
  for (let k = 0; k < n; k++) out.push(pearsonMasked(ch.a, blockScramble(ch.b, S, S, 8, r), roi));
  return out;
}

/** Costes' thresholds, or (when Costes is not defined) the 75th percentile of each channel in the region. */
export function startThresholds(ch: Channels, roi: Uint8Array | undefined): { tA: number; tB: number; costes: CostesResult } {
  const c = costes(ch.a, ch.b, roi, 100);
  if (c.ok) return { tA: c.tA, tB: c.tB, costes: c };
  const a = Array.from(ch.a).filter((_, i) => !roi || roi[i]), b = Array.from(ch.b).filter((_, i) => !roi || roi[i]);
  return { tA: quantile(a, 0.75), tB: quantile(b, 0.75), costes: c };
}
