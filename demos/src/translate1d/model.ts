// Model of the translate1d demo (no DOM; tested in demos/test/gans-test.ts). Image-to-image
// translation reduced to one pixel: an input value x (a label-free image, e.g. transmitted
// light) and a target value y (a fluorescence stain), measured in pairs.
//
//   y = 0.2 + 0.4·x + m·gap + noise (sd 0.03),   x ~ U(0, 1),
//   m = 1 ("marker on") with probability q for cells with x > 0.5, else m = 0.
// The marker is not visible in x: for x > 0.5 the input does not decide the target.
//
// Nothing is trained here: the demo shows what each kind of translator converges to when
// trained perfectly on the training pairs, estimated with a kernel in x (bandwidth H):
//   squared-error loss  → the conditional mean   E[y | x]   (a "blurry" value between branches),
//   absolute-error loss → the conditional median (one branch, or anywhere between),
//   a conditional GAN   → samples from p(y | x)   (a random branch: realistic, but a coin flip).
// Unpaired translation (only the two distributions, as in cycle-consistent training) can at
// best match the distributions: the monotone quantile maps (increasing or decreasing), which
// are exactly invertible (cycle error 0) and ignore the pairs.
import { rng } from "../lib/ui";

export const NOISE = 0.03, H = 0.04, N_TRAIN = 400, N_TEST = 400;
export const base = (x: number) => 0.2 + 0.4 * x;

export interface Pair { x: number; y: number; m: number }

export function pairs(n: number, gap: number, q: number, seed: number): Pair[] {
  const r = rng(seed), out: Pair[] = [];
  for (let i = 0; i < n; i++) {
    const x = r.unif(), m = x > 0.5 && r.unif() < q ? 1 : 0;
    out.push({ x, y: base(x) + m * gap + NOISE * r.norm(), m });
  }
  return out;
}

/** Kernel weights of the training pairs at input x (Gaussian, bandwidth H). */
function weights(train: Pair[], x: number): Float64Array {
  return Float64Array.from(train, p => Math.exp(-0.5 * ((p.x - x) / H) ** 2));
}

/** What a squared-error translator learns: the weighted mean of y near x. */
export function condMean(train: Pair[], x: number): number {
  const w = weights(train, x);
  let s = 0, sw = 0;
  train.forEach((p, i) => { s += w[i] * p.y; sw += w[i]; });
  return s / sw;
}

/** What an absolute-error translator learns: the weighted median of y near x. */
export function condMedian(train: Pair[], x: number): number {
  const w = weights(train, x), idx = train.map((_, i) => i).sort((a, b) => train[a].y - train[b].y);
  let tot = 0;
  for (const v of w) tot += v;
  let acc = 0;
  for (const i of idx) { acc += w[i]; if (acc >= tot / 2) return train[i].y; }
  return train[idx[idx.length - 1]].y;
}

/** What a perfect conditional GAN does: draw a y from the training pairs near x. */
export function condSample(train: Pair[], x: number, r: ReturnType<typeof rng>): number {
  const w = weights(train, x);
  let tot = 0;
  for (const v of w) tot += v;
  let u = r.unif() * tot;
  for (let i = 0; i < w.length; i++) { u -= w[i]; if (u <= 0) return train[i].y; }
  return train[train.length - 1].y;
}

/** The unpaired translators: x ↦ the y with the same quantile (increasing) or the opposite
 *  one (decreasing), from the two separate sets of training values. */
export function quantileMap(xs: number[], ys: number[], x: number, decreasing = false): number {
  const sx = [...xs].sort((a, b) => a - b), sy = [...ys].sort((a, b) => a - b);
  let k = 0;
  while (k < sx.length && sx[k] < x) k++;
  const u = Math.min(1, k / sx.length), j = Math.min(sy.length - 1, Math.floor((decreasing ? 1 - u : u) * sy.length));
  return sy[Math.max(0, j)];
}

export interface Scores {
  /** Root mean squared error against the held-out truth. */
  rmseMean: number; rmseMedian: number; rmseSample: number; rmseUnpaired: number;
  /** Share of predictions that a real cell could have (within 3 noise sd of a branch). */
  realMean: number; realMedian: number; realSample: number;
  /** Share of "marker on" among test cells with x > 0.5: truth, and read from the samples. */
  onTruth: number; onSample: number;
  /** Mean spread (sd) of 16 samples per test cell, for x ≤ 0.5 and x > 0.5. */
  spreadLow: number; spreadHigh: number;
}

/** Could a real cell at x have the value y? */
export const isReal = (x: number, y: number, gap: number) =>
  Math.abs(y - base(x)) <= 3 * NOISE || (x > 0.5 && Math.abs(y - base(x) - gap) <= 3 * NOISE);

export function scores(train: Pair[], test: Pair[], gap: number, seed = 5): Scores {
  const r = rng(seed), xs = train.map(p => p.x), ys = train.map(p => p.y);
  let em = 0, ed = 0, es = 0, eu = 0, rm = 0, rd = 0, rs = 0, nOn = 0, nHigh = 0, sOn = 0, spL = 0, nL = 0, spH = 0;
  const S = 16;
  for (const p of test) {
    const mu = condMean(train, p.x), md = condMedian(train, p.x);
    em += (mu - p.y) ** 2; ed += (md - p.y) ** 2;
    eu += (quantileMap(xs, ys, p.x) - p.y) ** 2;
    if (isReal(p.x, mu, gap)) rm++;
    if (isReal(p.x, md, gap)) rd++;
    let s1 = 0, s2 = 0;
    for (let k = 0; k < S; k++) {
      const y = condSample(train, p.x, r);
      es += (y - p.y) ** 2 / S; s1 += y; s2 += y * y;
      if (isReal(p.x, y, gap)) rs += 1 / S;
      if (p.x > 0.5 && y - base(p.x) > gap / 2) sOn += 1 / S;
    }
    const sd = Math.sqrt(Math.max(0, s2 / S - (s1 / S) ** 2));
    if (p.x > 0.5) { nHigh++; nOn += p.m; spH += sd; } else { nL++; spL += sd; }
  }
  const n = test.length;
  return {
    rmseMean: Math.sqrt(em / n), rmseMedian: Math.sqrt(ed / n), rmseSample: Math.sqrt(es / n), rmseUnpaired: Math.sqrt(eu / n),
    realMean: rm / n, realMedian: rd / n, realSample: rs / n,
    onTruth: nOn / nHigh, onSample: sOn / nHigh, spreadLow: spL / nL, spreadHigh: spH / nHigh,
  };
}
