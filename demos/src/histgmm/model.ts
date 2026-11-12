// Model of the histgmm demo (no DOM, so the test can import it): a synthetic three-class
// image with known truth, EM of a 1D Gaussian mixture on the grey-level histogram (bins are
// points weighted by their counts, lib/gmm.ts), multi-level Otsu thresholds for comparison,
// Bayes boundaries, and the per-pixel label / posterior / log-density images.
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { rng, type Rng } from "../lib/random";
import { ellipseDistance, type Ellipse } from "../lib/synth";
import { type Mixture, eStep, mStep, initMixture, logJoint, logSumExp, prepare, dataMoments } from "../lib/gmm";

export const N = 128;
/** Grey levels of the synthetic classes: background, cytoplasm, organelles. */
export const LEVELS = [70, 130, 185];

/** Truth labels 0/1/2 of the synthetic image: a large cell (1) with a few small bright
 *  organelles (2) on a background (0). `small`: the bright class covers very few pixels. */
export function synthTruth(small = false): Int32Array {
  const lab = new Int32Array(N * N);
  const cell: Ellipse = { x: 64, y: 64, a: 58, b: 46, angle: 0.35 };
  const orgs: Ellipse[] = small
    ? [{ x: 50, y: 56, a: 9, b: 8, angle: 0 }, { x: 84, y: 80, a: 8, b: 7, angle: 0.5 }, { x: 78, y: 40, a: 10, b: 8, angle: 0.2 }]
    : [{ x: 46, y: 54, a: 19, b: 14, angle: 0.4 }, { x: 84, y: 80, a: 16, b: 12, angle: -0.3 }, { x: 78, y: 40, a: 10, b: 8, angle: 0 }, { x: 38, y: 88, a: 11, b: 7, angle: 1 }];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let c = ellipseDistance(cell, x, y) <= 1 ? 1 : 0;
    if (c && orgs.some(e => ellipseDistance(e, x, y) <= 1)) c = 2;
    lab[y * N + x] = c;
  }
  return lab;
}

/** The synthetic image: class grey levels, Gaussian blur (σ in pixels), Gaussian noise, rounded
 *  to integers 0..255. */
export function synthImage(truth: Int32Array, noise: number, blur: number, seed = 7): Grey {
  let g = createGrey(N, N);
  for (let p = 0; p < N * N; p++) g.data[p] = LEVELS[truth[p]];
  if (blur > 0) g = gaussianBlur(g, blur);
  const r = rng(seed);
  for (let p = 0; p < N * N; p++) g.data[p] = Math.max(0, Math.min(255, Math.round(g.data[p] + noise * r.norm())));
  return g;
}

/** Counts of the integer grey levels 0..255. */
export function greyCounts(img: Grey): Float64Array {
  const c = new Float64Array(256);
  for (const v of img.data) c[Math.max(0, Math.min(255, Math.round(v)))]++;
  return c;
}

const GREYS = Float64Array.from({ length: 256 }, (_, i) => i);

export type HistInit = "quantiles" | "kmeans" | "random";

/** Starting mixture for K components on a histogram. quantiles: means at the (j + ½)/K
 *  quantiles, equal weights, variance = total variance / K². */
export function initHist(counts: Float64Array, K: number, how: HistInit, r: Rng, equalVar: boolean): Mixture {
  const covType = equalVar ? "tied" : "full";
  if (how !== "quantiles") return initMixture(GREYS, 256, 1, K, how === "kmeans" ? "kmeans" : "random", r, { covType, weights: counts });
  const mom = dataMoments(GREYS, 256, 1, counts), tot = mom.total;
  const means = new Float64Array(K);
  let acc = 0, j = 0;
  for (let i = 0; i < 256 && j < K; i++) {
    acc += counts[i];
    while (j < K && acc >= (j + 0.5) / K * tot) { means[j] = i; j++; }
  }
  const v = Math.max(1, mom.cov[0] / (K * K));
  return { k: K, d: 1, weights: new Float64Array(K).fill(1 / K), means, covs: new Float64Array(K).fill(v) };
}

/** One EM iteration on the histogram: returns the new model and log p(D) of the old one. */
export function histStep(counts: Float64Array, m: Mixture, equalVar: boolean): { model: Mixture; logLik: number } {
  const e = eStep(m, GREYS, 256, counts);
  return { model: mStep(GREYS, 256, 1, e.resp, m.k, { covType: equalVar ? "tied" : "full", minVar: 0.25 }, counts, m), logLik: e.logLik };
}

export function histLogLik(counts: Float64Array, m: Mixture): number {
  return eStep(m, GREYS, 256, counts).logLik;
}

/** EM to convergence on the histogram. trace = log p(D) per iteration. */
export function histEM(counts: Float64Array, m0: Mixture, equalVar: boolean, maxIter = 400, tol = 1e-4): { model: Mixture; trace: number[] } {
  let m = m0;
  const trace = [histLogLik(counts, m)];
  for (let it = 0; it < maxIter; it++) {
    m = histStep(counts, m, equalVar).model;
    trace.push(histLogLik(counts, m));
    if (Math.abs(trace[trace.length - 1] - trace[trace.length - 2]) < tol) break;
  }
  return { model: m, trace };
}

/** Multi-level Otsu: K classes of consecutive bins that maximise the between-class variance
 *  (dynamic programming). Returns the K − 1 bin indices where classes 2..K start. */
export function multiOtsu(counts: ArrayLike<number>, K: number): number[] {
  const B = counts.length, W = new Float64Array(B + 1), M = new Float64Array(B + 1);
  for (let i = 0; i < B; i++) { W[i + 1] = W[i] + counts[i]; M[i + 1] = M[i] + counts[i] * i; }
  const S = (a: number, b: number) => { const w = W[b] - W[a]; return w > 0 ? (M[b] - M[a]) ** 2 / w : 0; };
  // best[j][b]: best score of j classes covering bins [0, b)
  const best = Array.from({ length: K + 1 }, () => new Float64Array(B + 1).fill(-Infinity));
  const arg = Array.from({ length: K + 1 }, () => new Int32Array(B + 1));
  best[0][0] = 0;
  for (let j = 1; j <= K; j++) for (let b = j; b <= B; b++) {
    for (let a = j - 1; a < b; a++) {
      if (best[j - 1][a] === -Infinity) continue;
      const v = best[j - 1][a] + S(a, b);
      if (v > best[j][b] + 1e-9) { best[j][b] = v; arg[j][b] = a; }
    }
  }
  const t: number[] = [];
  let b = B;
  for (let j = K; j > 1; j--) { b = arg[j][b]; t.unshift(b); }
  return t;
}

/** Grey values (0..255) where the most probable component changes. */
export function bayesBoundaries(m: Mixture): number[] {
  const prep = prepare(m), out: number[] = [];
  let prev = -1;
  for (let v = 0; v <= 255; v += 0.25) {
    const lj = logJoint(m, [v], 1, prep);
    let b = 0;
    for (let j = 1; j < m.k; j++) if (lj[j] > lj[b]) b = j;
    if (prev >= 0 && b !== prev) out.push(v - 0.125);
    prev = b;
  }
  return out;
}

/** Order components by mean (so labels run dark → bright). */
export function sortByMean(m: Mixture): Mixture {
  const idx = Array.from({ length: m.k }, (_, j) => j).sort((a, b) => m.means[a] - m.means[b]);
  return { k: m.k, d: 1, weights: Float64Array.from(idx, j => m.weights[j]), means: Float64Array.from(idx, j => m.means[j]), covs: Float64Array.from(idx, j => m.covs[j]) };
}

/** Per-grey-level lookup: label, posteriors (256·K) and log p for each grey level. */
export function greyTables(m: Mixture): { label: Int32Array; post: Float64Array; logp: Float64Array } {
  const e = eStep(m, GREYS, 256);
  const label = new Int32Array(256);
  for (let v = 0; v < 256; v++) { let b = 0; for (let j = 1; j < m.k; j++) if (e.resp[v * m.k + j] > e.resp[v * m.k + b]) b = j; label[v] = b; }
  return { label, post: e.resp, logp: e.logp };
}

/** Mixture density p(v) at grey value v and each weighted component π_k N_k(v). */
export function densities(m: Mixture, v: number): { total: number; parts: number[] } {
  const lj = logJoint(m, [v], 1);
  const parts = Array.from(lj, x => Math.exp(x));
  return { total: Math.exp(logSumExp(lj)), parts };
}
