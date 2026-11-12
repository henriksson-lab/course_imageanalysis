// Boosting (discrete AdaBoost with decision stumps) and the attentional cascade, over
// rectangle features computed from integral images. Owner: the integral-images day
// (unit viola-jones/). No DOM.
//
//  - A feature-value matrix (features × examples) is sorted once per feature; each
//    boosting round then finds the best stump of every feature in one pass over the
//    sorted values with running (prefix) sums of the weights.
//  - Examples with weight 0 are ignored, so a cascade stage trains on a subset by
//    zeroing the weights of the examples that earlier stages already rejected.
//  - Labels are +1 (target) and −1 (background).
//
// Note: the random forests day owns decision trees and their split search; a stump is a
// depth-1 tree. No shared lib/forest.ts existed when this file was written, so the
// weighted stump search lives here (one pass over sorted values, weighted error).

import { Table, Template, Corner, templateCorners, TEMPLATES } from "./integral";

// ---------------------------------------------------------------------------------------
// Feature matrix

export interface FeatureMatrix {
  nFeat: number;
  nEx: number;
  /** values[f * nEx + i]: feature f on example i. */
  values: Float32Array;
  /** order[f * nEx + k]: index of the example with the k-th smallest value of feature f. */
  order: Uint32Array;
}

/**
 * Sort every feature's values once (the expensive part; done before training). Each value
 * is turned into an order-preserving integer (the float32 bit pattern, sign-flipped) and
 * packed with its example index into one float64 key, so a plain numeric sort suffices.
 */
export function prepareFeatures(values: Float32Array, nFeat: number, nEx: number): FeatureMatrix {
  const order = new Uint32Array(nFeat * nEx);
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  let mult = 1;
  while (mult < nEx) mult *= 2;
  if (mult > 2 ** 21) throw new Error("too many examples");
  const keys = new Float64Array(nEx);
  for (let f = 0; f < nFeat; f++) {
    const off = f * nEx;
    for (let i = 0; i < nEx; i++) {
      const b = bits[off + i];
      const u = b & 0x80000000 ? (~b) >>> 0 : (b | 0x80000000) >>> 0;
      keys[i] = u * mult + i;
    }
    keys.sort();
    for (let k = 0; k < nEx; k++) order[off + k] = keys[k] % mult;
  }
  return { nFeat, nEx, values, order };
}

// ---------------------------------------------------------------------------------------
// Decision stumps

/** h(x) = +1 if polarity · (x_feature − threshold) > 0, else −1. */
export interface Stump {
  feature: number;
  threshold: number;
  polarity: 1 | -1;
  /** Weighted error when it was chosen. */
  error: number;
  /** Vote weight α = ½ ln((1 − ε) / ε). */
  alpha: number;
}

export function stumpPredict(s: Stump, v: number): 1 | -1 {
  return s.polarity * (v - s.threshold) > 0 ? 1 : -1;
}

/**
 * Best stump for one feature: one pass over the sorted values. With the threshold below
 * all values and polarity +1 every example is called positive, so the error is the weight
 * of the negatives; moving the threshold past example i flips its prediction to −1.
 * Polarity −1 has error (total − e). Only thresholds between distinct values are tried.
 */
export function bestStumpForFeature(M: FeatureMatrix, f: number, y: ArrayLike<number>, w: ArrayLike<number>,
  totals?: { total: number; neg: number }): { error: number; threshold: number; polarity: 1 | -1 } {
  const off = f * M.nEx, V = M.values, O = M.order;
  let total = 0, e = 0;
  if (totals) { total = totals.total; e = totals.neg; }
  else for (let i = 0; i < M.nEx; i++) { total += w[i]; if (y[i] < 0) e += w[i]; }
  let best = Math.min(e, total - e), bestK = -1, bestPol: 1 | -1 = e <= total - e ? 1 : -1;
  for (let k = 0; k < M.nEx; k++) {
    const i = O[off + k];
    e += y[i] > 0 ? w[i] : -w[i];
    if (k + 1 < M.nEx && V[off + O[off + k + 1]] === V[off + i]) continue; // not a boundary
    if (e < best) { best = e; bestK = k; bestPol = 1; }
    if (total - e < best) { best = total - e; bestK = k; bestPol = -1; }
  }
  let thr: number;
  if (bestK < 0) thr = V[off + O[off]] - 1;
  else if (bestK + 1 >= M.nEx) thr = V[off + O[off + bestK]] + 1;
  else thr = 0.5 * (V[off + O[off + bestK]] + V[off + O[off + bestK + 1]]);
  return { error: best / (total || 1), threshold: thr, polarity: bestPol };
}

/** Best stump over all features (or a subset). */
export function bestStump(M: FeatureMatrix, y: ArrayLike<number>, w: ArrayLike<number>, features?: ArrayLike<number>): Stump {
  let best: Stump = { feature: 0, threshold: 0, polarity: 1, error: Infinity, alpha: 0 };
  const n = features ? features.length : M.nFeat;
  // signed weights: the error changes by +w (positive) or −w (negative) as the threshold passes an example
  let total = 0, neg = 0;
  for (let i = 0; i < M.nEx; i++) { total += w[i]; if (y[i] < 0) neg += w[i]; }
  const sw = new Float64Array(M.nEx);
  for (let i = 0; i < M.nEx; i++) sw[i] = y[i] > 0 ? w[i] : -w[i];
  for (let j = 0; j < n; j++) {
    const f = features ? features[j] : j;
    const s = bestStumpSigned(M, f, sw, total, neg);
    if (s.error < best.error) best = { feature: f, threshold: s.threshold, polarity: s.polarity, error: s.error, alpha: 0 };
  }
  return best;
}

/** The inner loop of bestStump with precomputed signed weights (same result as bestStumpForFeature). */
function bestStumpSigned(M: FeatureMatrix, f: number, sw: Float64Array, total: number, neg: number): { error: number; threshold: number; polarity: 1 | -1 } {
  const n = M.nEx, off = f * n, V = M.values, O = M.order;
  let e = neg, best = Math.min(e, total - e), bestK = -1, bestPol: 1 | -1 = e <= total - e ? 1 : -1;
  let i = O[off], v = V[off + i];
  for (let k = 0; k < n; k++) {
    e += sw[i];
    const i2 = k + 1 < n ? O[off + k + 1] : -1;
    const v2 = i2 >= 0 ? V[off + i2] : NaN;
    if (v2 !== v) {
      if (e < best) { best = e; bestK = k; bestPol = 1; }
      if (total - e < best) { best = total - e; bestK = k; bestPol = -1; }
    }
    i = i2; v = v2;
  }
  let thr: number;
  if (bestK < 0) thr = V[off + O[off]] - 1;
  else if (bestK + 1 >= n) thr = V[off + O[off + bestK]] + 1;
  else thr = 0.5 * (V[off + O[off + bestK]] + V[off + O[off + bestK + 1]]);
  return { error: best / (total || 1), threshold: thr, polarity: bestPol };
}

/** Brute-force best stump of one feature (reference for tests): every threshold, both polarities. */
export function bestStumpBrute(vals: ArrayLike<number>, y: ArrayLike<number>, w: ArrayLike<number>): number {
  const cand = [...new Set(Array.from(vals))].sort((a, b) => a - b);
  const ths = [cand[0] - 1, ...cand.map((v, i) => (i + 1 < cand.length ? 0.5 * (v + cand[i + 1]) : v + 1))];
  let total = 0;
  for (let i = 0; i < vals.length; i++) total += w[i];
  let best = Infinity;
  for (const t of ths) for (const p of [1, -1]) {
    let e = 0;
    for (let i = 0; i < vals.length; i++) if ((p * (vals[i] - t) > 0 ? 1 : -1) !== y[i]) e += w[i];
    best = Math.min(best, e / total);
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// AdaBoost

export const alphaOf = (eps: number): number => 0.5 * Math.log((1 - eps) / Math.max(eps, 1e-12));

/** Discrete AdaBoost. Weights start equal over the active examples (weight > 0 in `init`). */
export class AdaBoost {
  readonly M: FeatureMatrix;
  readonly y: Int8Array;
  w: Float64Array;
  stumps: Stump[] = [];
  /** Running strong-classifier score Σ α h on every training example. */
  score: Float64Array;
  /** Weights before each round (for the "focus on hard examples" picture). */
  history: Float64Array[] = [];

  constructor(M: FeatureMatrix, y: ArrayLike<number>, init?: ArrayLike<number>) {
    this.M = M;
    this.y = Int8Array.from(y);
    this.w = new Float64Array(M.nEx);
    if (init) for (let i = 0; i < M.nEx; i++) this.w[i] = init[i];
    else this.w.fill(1);
    normalise(this.w);
    this.score = new Float64Array(M.nEx);
  }

  /** One round: best weighted stump, its vote, reweighting. Returns the stump. */
  round(features?: ArrayLike<number>): Stump {
    this.history.push(this.w.slice());
    const s = bestStump(this.M, this.y, this.w, features);
    s.alpha = alphaOf(s.error);
    const off = s.feature * this.M.nEx;
    for (let i = 0; i < this.M.nEx; i++) {
      const h = stumpPredict(s, this.M.values[off + i]);
      this.score[i] += s.alpha * h;
      this.w[i] *= Math.exp(-s.alpha * this.y[i] * h);
    }
    normalise(this.w);
    this.stumps.push(s);
    return s;
  }

  /** Training error of sign(score) over the active examples (weight in `mask` > 0, default all). */
  trainingError(mask?: ArrayLike<number>): number {
    let wrong = 0, n = 0;
    for (let i = 0; i < this.M.nEx; i++) {
      if (mask && !(mask[i] > 0)) continue;
      n++;
      if ((this.score[i] >= 0 ? 1 : -1) !== this.y[i]) wrong++;
    }
    return n ? wrong / n : 0;
  }

  /** The bound Π 2 √(ε_t (1 − ε_t)) on the training error. */
  bound(): number {
    return this.stumps.reduce((p, s) => p * 2 * Math.sqrt(s.error * (1 - s.error)), 1);
  }
}

function normalise(w: Float64Array): void {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i];
  if (s > 0) for (let i = 0; i < w.length; i++) w[i] /= s;
}

/** Strong-classifier score Σ α_t h_t for one example, given its feature values. */
export function strongScore(stumps: Stump[], value: (feature: number) => number, upTo = stumps.length): number {
  let s = 0;
  for (let t = 0; t < upTo; t++) s += stumps[t].alpha * stumpPredict(stumps[t], value(stumps[t].feature));
  return s;
}

// ---------------------------------------------------------------------------------------
// Attentional cascade

/** A stage passes a window if Σ α h ≥ threshold (lowered below 0 to keep most targets). */
export interface Stage { stumps: Stump[]; threshold: number }

export interface StageReport { stumps: number; detection: number; falsePositive: number }

/**
 * Train one stage on the active examples (mask > 0): add stumps until the stage reaches
 * detection ≥ dTarget with false-positive rate ≤ fTarget on them (the threshold is set at
 * the dTarget quantile of the positives' scores), or until maxStumps.
 */
export function trainStage(M: FeatureMatrix, y: ArrayLike<number>, mask: ArrayLike<number>, dTarget: number, fTarget: number,
  maxStumps = 40, features?: ArrayLike<number>): { stage: Stage; report: StageReport } {
  const init = Float64Array.from({ length: M.nEx }, (_, i) => (mask[i] > 0 ? 1 : 0));
  // balance: half the weight on targets, half on background (as Viola and Jones)
  let np = 0, nn = 0;
  for (let i = 0; i < M.nEx; i++) if (init[i]) { if (y[i] > 0) np++; else nn++; }
  for (let i = 0; i < M.nEx; i++) if (init[i]) init[i] = y[i] > 0 ? 0.5 / Math.max(1, np) : 0.5 / Math.max(1, nn);
  const ab = new AdaBoost(M, y, init);
  let thr = 0, d = 1, fp = 1;
  while (ab.stumps.length < maxStumps) {
    ab.round(features);
    const pos: number[] = [];
    for (let i = 0; i < M.nEx; i++) if (init[i] && y[i] > 0) pos.push(ab.score[i]);
    pos.sort((a, b) => a - b);
    const k = Math.min(pos.length - 1, Math.max(0, Math.floor((1 - dTarget) * pos.length)));
    thr = pos.length ? pos[k] - 1e-9 : 0;
    let pass = 0, fpass = 0;
    for (let i = 0; i < M.nEx; i++) if (init[i]) {
      if (ab.score[i] >= thr) { if (y[i] > 0) pass++; else fpass++; }
    }
    d = np ? pass / np : 1; fp = nn ? fpass / nn : 0;
    if (fp <= fTarget) break;
  }
  return { stage: { stumps: ab.stumps, threshold: thr }, report: { stumps: ab.stumps.length, detection: d, falsePositive: fp } };
}

/**
 * Run a cascade on one window. Returns the index of the rejecting stage (or −1 if every
 * stage passed) and the number of features evaluated.
 */
export function runCascade(stages: Stage[], value: (feature: number) => number): { rejectedAt: number; features: number } {
  let feats = 0;
  for (let k = 0; k < stages.length; k++) {
    const st = stages[k];
    const s = strongScore(st.stumps, value);
    feats += st.stumps.length;
    if (s < st.threshold) return { rejectedAt: k, features: feats };
  }
  return { rejectedAt: -1, features: feats };
}

/** Overall detection and false-positive rate (products) and expected features per window. */
export function cascadeRates(d: number[], f: number[], n: number[], passRate: number[] = f): { D: number; F: number; features: number } {
  let D = 1, F = 1, feats = 0, reach = 1;
  for (let i = 0; i < d.length; i++) {
    feats += n[i] * reach;
    reach *= passRate[i];
    D *= d[i]; F *= f[i];
  }
  return { D, F, features: feats };
}

// ---------------------------------------------------------------------------------------
// Rectangle features in a detection window, with feature scaling

/** One rectangle feature in a base window: a template at (x, y) with cell size sx × sy. */
export interface FeatureSpec { t: number; x: number; y: number; sx: number; sy: number }

/**
 * A pool of rectangle features in a base window of `win` × `win` pixels: every template,
 * positions on a `stride` grid, cell sizes from `sizes` (in both directions).
 */
export function featurePool(win: number, stride: number, sizes: number[], templates: Template[] = TEMPLATES): FeatureSpec[] {
  const pool: FeatureSpec[] = [];
  templates.forEach((t, ti) => {
    for (const sx of sizes) for (const sy of sizes) {
      const fw = t.cols * sx, fh = t.rows * sy;
      if (fw > win || fh > win) continue;
      for (let y = 0; y + fh <= win; y += stride)
        for (let x = 0; x + fw <= win; x += stride) pool.push({ t: ti, x, y, sx, sy });
    }
  });
  return pool;
}

/** A feature prepared for evaluation at one scale: its corners and its normaliser (cell area). */
export interface ScaledFeature { corners: Corner[]; area: number }

/**
 * Scale a feature by `scale` (rectangles grow, cost unchanged): positions and sizes are
 * rounded to whole pixels; the response is divided by the cell area, so values are
 * comparable across scales.
 */
export function scaleFeature(f: FeatureSpec, scale: number, templates: Template[] = TEMPLATES): ScaledFeature {
  const sx = Math.max(1, Math.round(f.sx * scale)), sy = Math.max(1, Math.round(f.sy * scale));
  const x = Math.round(f.x * scale), y = Math.round(f.y * scale);
  return { corners: templateCorners(templates[f.t], x, y, sx, sy), area: sx * sy };
}

/** Window statistics (mean, SD) of [x0, x0+size)² from the tables of I and I². */
export function windowStats(T1: Table, T2: Table, x0: number, y0: number, size: number): { mean: number; sd: number } {
  const W = T1.w + 1;
  const g = (T: Table) => T.data[(y0 + size) * W + x0 + size] - T.data[y0 * W + x0 + size] - T.data[(y0 + size) * W + x0] + T.data[y0 * W + x0];
  const n = size * size, m = g(T1) / n;
  return { mean: m, sd: Math.sqrt(Math.max(0, g(T2) / n - m * m)) };
}

/** Feature value in the window at (x0, y0): Σ c·S / (area · sd). */
export function featureValue(T: Table, sf: ScaledFeature, x0: number, y0: number, sd: number): number {
  const W = T.w + 1, D = T.data;
  let s = 0;
  for (const c of sf.corners) s += c.c * D[(c.y + y0) * W + c.x + x0];
  return s / (sf.area * Math.max(sd, 1));
}
