// Model of the guidance1d demo (no DOM; tested in demos/test/diffusion-test.ts).
// Two classes, each a 1D Gaussian mixture; the unconditional density is their weighted sum.
// Sharpened density p(x | y) p(y | x)^w (normalised on a grid), and guided diffusion: the
// classifier-free combination (1 + w) D(x, σ | y) − w D(x, σ) of the exact conditional and
// unconditional denoisers at every noise level (lib/diffusion.ts).
import { makeDiag } from "../lib/mixture";
import { mixtureDenoiser, guidedDenoiser, sigmaGrid, sampleRun } from "../lib/diffusion";
import { rng } from "../lib/ui";
import { normCdf } from "../lib/stats";

export type DataId = "two" | "submode" | "rare";
export const DATA_IDS: DataId[] = ["two", "submode", "rare"];
export const DATA_LABEL: Record<DataId, string> = { two: "two overlapping classes", submode: "class A with two sub-modes", rare: "rare class A (10 %)" };

/** Components: weight (within the whole data), mean, variance, class (0 = A, 1 = B). */
export interface Classes { w: number[]; mu: number[]; v: number[]; cls: number[] }

export function classes(id: DataId): Classes {
  if (id === "two") return { w: [0.5, 0.5], mu: [-1, 1], v: [1, 1], cls: [0, 1] };
  if (id === "submode") return { w: [0.3, 0.2, 0.5], mu: [-2, 0.6, 1.5], v: [0.16, 0.09, 0.25], cls: [0, 0, 1] };
  return { w: [0.1, 0.9], mu: [1, -0.5], v: [0.36, 1], cls: [0, 1] };
}

const comps = (c: Classes, y: number) => c.cls.map((k, j) => k === y ? j : -1).filter(j => j >= 0);
const npdf = (x: number, m: number, v: number) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);

/** p(x | y) and p(x). */
export function classDensity(c: Classes, y: number | null, x: number): number {
  let p = 0, z = 0;
  c.w.forEach((w, j) => { if (y === null || c.cls[j] === y) { p += w * npdf(x, c.mu[j], c.v[j]); z += w; } });
  return p / z;
}

/** p(y | x). */
export function classProb(c: Classes, y: number, x: number): number {
  let a = 0, b = 0;
  c.w.forEach((w, j) => { const p = w * npdf(x, c.mu[j], c.v[j]); b += p; if (c.cls[j] === y) a += p; });
  return a / b;
}

/** Sharpened density p(x | y) p(y | x)^w on a grid (normalised). */
export function sharpened(c: Classes, y: number, w: number, xs: ArrayLike<number>): Float64Array {
  const v = Float64Array.from(xs, x => classDensity(c, y, x) * classProb(c, y, x) ** w);
  const h = xs[1] - xs[0];
  let s = 0; for (const u of v) s += u * h;
  return v.map(u => u / s);
}

/** Guided samples of class y with weight w (exact denoisers, cosine schedule uniform in time). */
export function guidedSamples(c: Classes, y: number, w: number, steps: number, eta: number, n = 2000, seed = 21): Float64Array {
  const m = makeDiag(c.w, c.mu.map(u => [u]), c.v);
  const D = guidedDenoiser(mixtureDenoiser(m, comps(c, y)), mixtureDenoiser(m), w);
  const sig = sigmaGrid(steps, "time", "cosine"), r = rng(seed), rn = rng(seed + 1), out = new Float64Array(n), x = new Float64Array(1);
  for (let p = 0; p < n; p++) {
    x[0] = Math.sqrt(sig[0] ** 2 + 1) * r.norm();
    sampleRun(x, D, { sigmas: sig, eta }, rn);
    out[p] = x[0];
  }
  return out;
}

/** Mean, variance and direction (+1/−1: away from the other class) of class y. */
export function classStats(c: Classes, y: number): { mean: number; sd: number; dir: number; q99: number } {
  let z = 0, m = 0, e2 = 0, zo = 0, mo = 0;
  c.w.forEach((w, j) => { if (c.cls[j] === y) { z += w; m += w * c.mu[j]; e2 += w * (c.v[j] + c.mu[j] ** 2); } else { zo += w; mo += w * c.mu[j]; } });
  m /= z; e2 /= z; mo /= zo;
  const dir = m >= mo ? 1 : -1;
  // the 1 % most extreme real members in that direction: quantile by bisection of the class CDF
  const cdf = (x: number) => { let s = 0; c.w.forEach((w, j) => { if (c.cls[j] === y) s += w / z * normCdf((x - c.mu[j]) / Math.sqrt(c.v[j])); }); return s; };
  const target = dir > 0 ? 0.99 : 0.01;
  let lo = -20, hi = 20;
  for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (cdf(mid) < target) lo = mid; else hi = mid; }
  return { mean: m, sd: Math.sqrt(e2 - m * m), dir, q99: (lo + hi) / 2 };
}

/** Summary of guided samples against the class. */
export function summary(c: Classes, y: number, X: Float64Array): { mean: number; sd: number; beyondMean: number; beyondQ99: number } {
  const st = classStats(c, y), n = X.length;
  let m = 0; for (const v of X) m += v / n;
  let vv = 0; for (const v of X) vv += (v - m) ** 2 / n;
  let bm = 0, bq = 0;
  for (const v of X) { if ((v - st.mean) * st.dir > 0) bm++; if ((v - st.q99) * st.dir > 0) bq++; }
  return { mean: m, sd: Math.sqrt(vv), beyondMean: bm / n, beyondQ99: bq / n };
}
