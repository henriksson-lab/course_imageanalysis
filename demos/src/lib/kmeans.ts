// k-means (Lloyd's algorithm) on weighted points in d dimensions (owner: Gaussian-mixture day;
// the thresholding day's isodata and the colour day's k-means in Lab can call it). No DOM.
//
// Points are a flat row-major Float64Array X of n·d numbers (point i = X[i*d .. i*d+d-1]);
// optional weights w[i] ≥ 0 let histogram bins act as points (a bin = its centre, weight = count).
// Centres are a flat k·d array. Every function that draws random numbers takes an Rng, so a
// run is reproducible.
//
//   const r = rng(1);
//   const c0 = kmeansPlusPlus(X, n, d, 3, r);
//   const res = kmeans(X, n, d, 3, { init: c0 });   // res.trace: centres and J per iteration
//
// The objective is J = Σ_i w_i ‖x_i − m_{c(i)}‖². The assignment step and the update step can
// each only lower J, so J never increases (tested in demos/test/gmm-test.ts).
import type { Rng } from "./random";
import { choice, randInt, sampleIndices, rng } from "./random";

export type KMeansInit = "random" | "kmeans++" | "partition";

/** Squared Euclidean distance between point i of X and centre j of C (both dimension d). */
export function dist2(X: ArrayLike<number>, i: number, C: ArrayLike<number>, j: number, d: number): number {
  let s = 0;
  for (let t = 0; t < d; t++) { const u = X[i * d + t] - C[j * d + t]; s += u * u; }
  return s;
}

/** Assignment step: nearest centre for every point (ties: the lower index) and J. */
export function assignStep(X: ArrayLike<number>, n: number, d: number, C: ArrayLike<number>, k: number,
  w?: ArrayLike<number>, out?: Int32Array): { assign: Int32Array; J: number } {
  const a = out ?? new Int32Array(n);
  let J = 0;
  for (let i = 0; i < n; i++) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < k; j++) { const q = dist2(X, i, C, j, d); if (q < bd) { bd = q; best = j; } }
    a[i] = best;
    J += (w ? w[i] : 1) * bd;
  }
  return { assign: a, J };
}

/** Update step: every centre moves to the (weighted) mean of its points. An empty cluster keeps
 *  its old centre (from `prev`), or stays at 0 if there is none. */
export function updateStep(X: ArrayLike<number>, n: number, d: number, assign: ArrayLike<number>, k: number,
  w?: ArrayLike<number>, prev?: ArrayLike<number>): { centres: Float64Array; sizes: Float64Array } {
  const C = new Float64Array(k * d), N = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    const j = assign[i], wi = w ? w[i] : 1;
    N[j] += wi;
    for (let t = 0; t < d; t++) C[j * d + t] += wi * X[i * d + t];
  }
  for (let j = 0; j < k; j++) for (let t = 0; t < d; t++) {
    C[j * d + t] = N[j] > 0 ? C[j * d + t] / N[j] : prev ? prev[j * d + t] : 0;
  }
  return { centres: C, sizes: N };
}

/** J = Σ w_i ‖x_i − m_{assign(i)}‖² for a given assignment and centres. */
export function objective(X: ArrayLike<number>, n: number, d: number, C: ArrayLike<number>, assign: ArrayLike<number>, w?: ArrayLike<number>): number {
  let J = 0;
  for (let i = 0; i < n; i++) J += (w ? w[i] : 1) * dist2(X, i, C, assign[i], d);
  return J;
}

/** k distinct data points (among those with positive weight) as centres. */
export function randomCentres(X: ArrayLike<number>, n: number, d: number, k: number, r: Rng, w?: ArrayLike<number>): Float64Array {
  const idx = w ? Array.from({ length: n }, (_, i) => i).filter(i => w[i] > 0) : null;
  const m = idx ? idx.length : n;
  const pick = sampleIndices(r, m, k);
  const C = new Float64Array(k * d);
  for (let j = 0; j < k; j++) {
    const i = idx ? idx[pick[j % pick.length]] : pick[j % pick.length];
    for (let t = 0; t < d; t++) C[j * d + t] = X[i * d + t];
  }
  return C;
}

/** Random partition: every point gets a random cluster; centres = the cluster means. */
export function randomPartition(X: ArrayLike<number>, n: number, d: number, k: number, r: Rng, w?: ArrayLike<number>): Float64Array {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = randInt(r, k);
  return updateStep(X, n, d, a, k, w).centres;
}

/** k-means++ seeding (Arthur & Vassilvitskii 2007): the first centre is a random point (weighted),
 *  each next centre a point drawn with probability ∝ w_i · D(x_i)², the squared distance to the
 *  nearest centre chosen so far. `first` fixes the index of the first centre. */
export function kmeansPlusPlus(X: ArrayLike<number>, n: number, d: number, k: number, r: Rng, w?: ArrayLike<number>, first?: number): Float64Array {
  const C = new Float64Array(k * d);
  const ones = w ?? new Float64Array(n).fill(1);
  const i0 = first ?? choice(r, ones);
  for (let t = 0; t < d; t++) C[t] = X[i0 * d + t];
  const D = new Float64Array(n);
  for (let i = 0; i < n; i++) D[i] = dist2(X, i, C, 0, d);
  for (let j = 1; j < k; j++) {
    const p = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { p[i] = ones[i] * D[i]; s += p[i]; }
    const i1 = s > 0 ? choice(r, p) : choice(r, ones);
    for (let t = 0; t < d; t++) C[j * d + t] = X[i1 * d + t];
    for (let i = 0; i < n; i++) D[i] = Math.min(D[i], dist2(X, i, C, j, d));
  }
  return C;
}

/** Initial centres by method name. */
export function initCentres(X: ArrayLike<number>, n: number, d: number, k: number, how: KMeansInit, r: Rng, w?: ArrayLike<number>): Float64Array {
  return how === "kmeans++" ? kmeansPlusPlus(X, n, d, k, r, w) : how === "partition" ? randomPartition(X, n, d, k, r, w) : randomCentres(X, n, d, k, r, w);
}

export interface KMeansStep {
  /** Centres used for this assignment (k·d). */
  centres: Float64Array;
  assign: Int32Array;
  /** J of this assignment with these centres. */
  J: number;
}

export interface KMeansResult {
  centres: Float64Array;
  assign: Int32Array;
  J: number;
  iterations: number;
  converged: boolean;
  /** One entry per assignment step: the centres before the update, the assignment, J. The last
   *  entry is the final (converged) state. */
  trace: KMeansStep[];
}

export interface KMeansOptions {
  /** Initial centres (k·d), or a method (default k-means++). */
  init?: Float64Array | KMeansInit;
  seed?: Rng;
  weights?: ArrayLike<number>;
  maxIter?: number;
}

/** Lloyd's algorithm: assign, update, until no assignment changes (or maxIter). */
export function kmeans(X: ArrayLike<number>, n: number, d: number, k: number, o: KMeansOptions = {}): KMeansResult {
  const w = o.weights;
  let C = o.init instanceof Float64Array ? Float64Array.from(o.init)
    : initCentres(X, n, d, k, o.init ?? "kmeans++", o.seed ?? defaultRng(), w);
  const trace: KMeansStep[] = [];
  let { assign, J } = assignStep(X, n, d, C, k, w);
  trace.push({ centres: Float64Array.from(C), assign: Int32Array.from(assign), J });
  const maxIter = o.maxIter ?? 300;
  let it = 0, converged = false;
  while (it < maxIter) {
    it++;
    C = updateStep(X, n, d, assign, k, w, C).centres;
    const nxt = assignStep(X, n, d, C, k, w);
    const same = nxt.assign.every((v, i) => v === assign[i]);
    assign = nxt.assign; J = nxt.J;
    trace.push({ centres: Float64Array.from(C), assign: Int32Array.from(assign), J });
    if (same) { converged = true; break; }
  }
  return { centres: C, assign, J, iterations: it, converged, trace };
}

/** Several runs from different seeds; returns the best (lowest J) and all final J values. */
export function kmeansRestarts(X: ArrayLike<number>, n: number, d: number, k: number, runs: number, r: Rng,
  o: { init?: KMeansInit; weights?: ArrayLike<number>; maxIter?: number } = {}): { best: KMeansResult; Js: number[] } {
  let best: KMeansResult | null = null;
  const Js: number[] = [];
  for (let t = 0; t < runs; t++) {
    const C = initCentres(X, n, d, k, o.init ?? "kmeans++", r, o.weights);
    const res = kmeans(X, n, d, k, { init: C, weights: o.weights, maxIter: o.maxIter });
    Js.push(res.J);
    if (!best || res.J < best.J) best = res;
  }
  return { best: best!, Js };
}

function defaultRng(): Rng { return rng(1); }
