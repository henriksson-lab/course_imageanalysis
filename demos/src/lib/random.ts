// Seeded random numbers for the demos (no DOM). Every demo seeds its generator, so a
// recording or a test run always looks the same. Never use Math.random in a demo.
//
//   const r = rng(42);          // { unif, norm }: uniform [0,1) and standard normal
//   gauss(r, 10, 2);            // normal with mean 10, sd 2
//   poisson(r, 3.5);            // Poisson count
//   shuffle(r, array);          // in-place Fisher–Yates
//
// All helpers take the generator first. `rng` itself lives in ui.ts (mulberry32) and is
// re-exported here; new code should import it from this file.

import { rng } from "./ui";
export { rng };

/** A seeded generator as returned by `rng(seed)`. */
export type Rng = ReturnType<typeof rng>;

/** Uniform real number in [a, b). */
export function uniform(r: Rng, a = 0, b = 1): number {
  return a + (b - a) * r.unif();
}

/** Uniform integer in 0 .. n-1. */
export function randInt(r: Rng, n: number): number {
  return Math.min(n - 1, Math.floor(r.unif() * n));
}

/** Normal (Gaussian) number with the given mean and standard deviation. */
export function gauss(r: Rng, mean = 0, sd = 1): number {
  return mean + sd * r.norm();
}

/** Exponential waiting time with the given rate (mean 1/rate). */
export function exponential(r: Rng, rate = 1): number {
  return -Math.log(1 - r.unif()) / rate;
}

/** true with probability p. */
export function bernoulli(r: Rng, p: number): boolean {
  return r.unif() < p;
}

/**
 * Poisson count with mean `lambda`. Exact for every lambda: Knuth's product method below
 * 30, above that the PTRS transformed-rejection method (Hörmann 1993). Same seed, same
 * sequence. (camerabits/sensor.ts keeps its own, slightly different, approximation.)
 */
export function poisson(r: Rng, lambda: number): number {
  if (!(lambda > 0)) return 0;
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= r.unif(); } while (p > L);
    return k - 1;
  }
  const slam = Math.sqrt(lambda), loglam = Math.log(lambda);
  const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
  const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    const U = r.unif() - 0.5, V = r.unif();
    const us = 0.5 - Math.abs(U);
    const k = Math.floor((2 * a / us + b) * U + lambda + 0.43);
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b)
      <= -lambda + k * loglam - logFactorial(k)) return k;
  }
}

/** log(k!) (exact sum below 20, Stirling series above). */
export function logFactorial(k: number): number {
  if (k < 20) { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }
  const x = k + 1;
  return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * x) - 1 / (360 * x ** 3);
}

/** Binomial count of successes in n trials with probability p (direct; n up to ~1e4). */
export function binomial(r: Rng, n: number, p: number): number {
  let k = 0;
  for (let i = 0; i < n; i++) if (r.unif() < p) k++;
  return k;
}

/** Shuffle an array in place (Fisher–Yates) and return it. */
export function shuffle<T, A extends { length: number; [i: number]: T }>(r: Rng, a: A): A {
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(r, i + 1);
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}

/** A random permutation of 0 .. n-1. */
export function permutation(r: Rng, n: number): Int32Array {
  const p = new Int32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  return shuffle(r, p);
}

/** k distinct indices from 0 .. n-1, in random order (k <= n). */
export function sampleIndices(r: Rng, n: number, k: number): Int32Array {
  return permutation(r, n).slice(0, Math.min(k, n));
}

/** Index drawn with probability proportional to weights[i] (non-negative). */
export function choice(r: Rng, weights: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < weights.length; i++) s += weights[i];
  let u = r.unif() * s;
  for (let i = 0; i < weights.length; i++) { u -= weights[i]; if (u < 0) return i; }
  return weights.length - 1;
}

/** Array of n normal numbers. */
export function normalArray(r: Rng, n: number, mean = 0, sd = 1): Float64Array {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = mean + sd * r.norm();
  return a;
}
