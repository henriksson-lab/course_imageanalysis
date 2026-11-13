// Model of the coloc demo (no DOM): readout glue. The two-channel scene (two cells, 128 × 128,
// A = structures + diffuse cell signal; B = a fraction f of the same structures + independent
// ones, or large blobs containing A's; bleed-through, shift, offset, noise, blur) lives in
// lib/twochannel.ts (pixelScene, moved from here; bit-identical), the coefficients in
// lib/coloc.ts.
import { rng } from "../lib/random";
import { pixelScene, S, ChannelOptions, Channels } from "../lib/twochannel";
import { pearsonMasked, overlapCoefficient, manders, mandersNoOwnThreshold, orthoRegression, blockScramble, costes, CostesResult, spearmanMasked } from "../lib/coloc";
import { quantile } from "../lib/stats";

export { S };
export type { ChannelOptions, Channels };
/** The scene of the built demo (kept under its old name for the measurements test). */
export const makeChannels = pixelScene;

// ---------------------------------------------------------------------------------------
// Readouts (used by the demo and by the exercise numbers)

export interface ColocStats { r: number; rho: number; rAbove: number; nAbove: number; overlap: number; m1: number; m2: number; slope: number; intercept: number }

/** All readouts for a region (mask, or all pixels) and thresholds tA, tB. */
export function colocStats(ch: Channels, roi: Uint8Array | undefined, tA: number, tB: number, ownThreshold = true): ColocStats {
  const r = pearsonMasked(ch.a, ch.b, roi);
  const above = Uint8Array.from(ch.a, (v, i) => ((!roi || roi[i]) && v > tA && ch.b[i] > tB ? 1 : 0));
  const nAbove = above.reduce((x, y) => x + y, 0);
  const md = ownThreshold ? manders(ch.a, ch.b, tA, tB, roi) : mandersNoOwnThreshold(ch.a, ch.b, tA, tB, roi);
  const reg = orthoRegression(ch.a, ch.b, roi);
  return { r, rho: spearmanMasked(ch.a, ch.b, roi), rAbove: nAbove > 2 ? pearsonMasked(ch.a, ch.b, above) : NaN, nAbove, overlap: overlapCoefficient(ch.a, ch.b, roi), m1: md.m1, m2: md.m2, slope: reg.slope, intercept: reg.intercept };
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
