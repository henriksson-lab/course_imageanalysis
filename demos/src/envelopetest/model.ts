// Model of the envelopetest demo (no DOM): curves, simulated null curves, the rejection-rate
// experiment and the two-level "conditions" simulation. Shared by the demo, the figure scripts
// and the test.
import { rng, gauss } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { tTestWelch } from "../lib/stats";

export const RMAX = 50;
export const RADII = PP.linspace(0, RMAX, 51);
export type Fn = "L" | "g";

/** L − r (translation correction) or g (bandwidth 2 µm) of a pattern. */
export function curve(p: PP.Points, win: PP.Window, fn: Fn): Float64Array {
  const pd = PP.pairData(p, win, RMAX + 3, "translation");
  return fn === "L" ? PP.LminusR(PP.KfromPairs(pd, RADII), RADII) : PP.pcfFromPairs(pd, RADII, 2);
}

/** The reference curve under CSR: 0 for L − r, 1 for g. */
export const reference = (fn: Fn): Float64Array => new Float64Array(RADII.length).fill(fn === "L" ? 0 : 1);

/** s CSR patterns with the same n in the same window, seeded. */
export function nullCurves(n: number, win: PP.Window, s: number, fn: Fn, seed: number): Float64Array[] {
  const r = rng(seed);
  return Array.from({ length: s }, () => curve(PP.binomialProcess(r, n, win), win, fn));
}

/** One trial of the rejection-rate experiment: a CSR pattern of n points in a square of side L,
 *  L − r with translation correction at m radii up to L/4, 39 simulations. */
export interface Trial { pw39: boolean; pw19: boolean; mad: boolean }
export function rejectionTrial(r: ReturnType<typeof rng>, n: number, side: number, m: number): Trial {
  const win = PP.rectWindow(side, side), radii = PP.linspace(side / 4 / m, side / 4, m);
  const L = (p: PP.Points) => PP.LminusR(PP.KfromPairs(PP.pairData(p, win, side / 4, "translation"), radii), radii);
  const obs = L(PP.binomialProcess(r, n, win));
  const sims = Array.from({ length: 39 }, () => L(PP.binomialProcess(r, n, win)));
  const idx = Array.from(radii, (_, k) => k), ref = new Float64Array(m);
  return {
    pw39: PP.escapes(obs, PP.pointwiseEnvelope(sims), idx).length > 0,
    pw19: PP.escapes(obs, PP.pointwiseEnvelope(sims.slice(0, 19)), idx).length > 0,
    mad: PP.madTest(obs, sims, ref, idx).p <= 0.05,
  };
}

/** Precomputed (tools/figs/envelopes/rates.ts: 1000 trials per m, n = 100 in 200 × 200 µm, seed 7):
 *  the share of CSR patterns that leave a pointwise envelope (39 or 19 simulations) somewhere, and
 *  that the MAD test rejects at 5 %, against the number m of radii (up to a quarter of the side). */
export const PRECOMP_M = [1, 2, 5, 10, 25, 50];
export const PRECOMP_PW39 = [0.041, 0.082, 0.164, 0.249, 0.371, 0.456];
export const PRECOMP_PW19 = [0.102, 0.172, 0.302, 0.432, 0.594, 0.692];
export const PRECOMP_MAD = [0.045, 0.045, 0.048, 0.047, 0.052, 0.052];

// ---------------------------------------------------------------------------------------
// Conditions: 2 conditions × E experiments × I images; per image one summary L(r0) − r0

export interface CondOptions { experiments: number; images: number; effect: number; dishSd: number; n: number; seed: number; r0: number }
export const DEFAULT_COND: CondOptions = { experiments: 4, images: 3, effect: 0, dishSd: 0.15, n: 120, seed: 1, r0: 10 };

export interface ImageSummary { cond: 0 | 1; exp: number; value: number }

/** A pattern with a fraction f of its points in small clusters (σ = 3 µm, 6 per cluster), the rest CSR. */
export function mixedPattern(r: ReturnType<typeof rng>, n: number, f: number, win: PP.Window): PP.Points {
  const nc = Math.round(n * Math.max(0, Math.min(1, f))), xs: number[] = [], ys: number[] = [];
  const parents = PP.binomialProcess(r, Math.max(1, Math.round(nc / 6)), win);
  for (let i = 0; xs.length < nc && i < 100000; i++) {
    const k = i % parents.x.length, x = parents.x[k] + gauss(r, 0, 3), y = parents.y[k] + gauss(r, 0, 3);
    if (PP.inside(win, x, y)) { xs.push(x); ys.push(y); }
  }
  const rest = PP.binomialProcess(r, n - xs.length, win);
  return PP.makePoints([...xs, ...rest.x], [...ys, ...rest.y]);
}

export function simulateConditions(o: CondOptions): { images: ImageSummary[]; patterns: PP.Points[] } {
  const r = rng(o.seed), win = PP.rectWindow(200, 200), images: ImageSummary[] = [], patterns: PP.Points[] = [];
  for (let e = 0; e < o.experiments; e++) for (const c of [0, 1] as const) {
    const f = 0.25 + (c === 1 ? o.effect : 0) + gauss(r, 0, o.dishSd);   // one dish per experiment and condition
    for (let i = 0; i < o.images; i++) {
      const p = mixedPattern(r, o.n, f, win), pd = PP.pairData(p, win, o.r0, "translation");
      const K = PP.KfromPairs(pd, [o.r0])[0];
      images.push({ cond: c, exp: e, value: Math.sqrt(K / Math.PI) - o.r0 });
      patterns.push(p);
    }
  }
  return { images, patterns };
}

export function conditionTests(images: ImageSummary[], experiments: number): { imagesP: number; expP: number; means: number[][] } {
  const a = images.filter(s => s.cond === 0).map(s => s.value), b = images.filter(s => s.cond === 1).map(s => s.value);
  const means: number[][] = [[], []];
  for (let e = 0; e < experiments; e++) for (const c of [0, 1]) {
    const v = images.filter(s => s.cond === c && s.exp === e).map(s => s.value);
    means[c].push(v.reduce((x, y) => x + y, 0) / v.length);
  }
  return { imagesP: tTestWelch(a, b).p, expP: tTestWelch(means[0], means[1]).p, means };
}
