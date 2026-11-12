// Numerics of the bootstrap demo (no DOM): bootstrap counts, the out-of-bag fraction, and
// the variance of an average of B correlated predictors, by formula and by simulation.
import { rng } from "../lib/ui";

type Rng = ReturnType<typeof rng>;

/** How often each of n points is drawn in one bootstrap sample of size n. */
export function bootstrapCounts(n: number, r: Rng): Int32Array {
  const c = new Int32Array(n);
  for (let q = 0; q < n; q++) c[Math.floor(r.unif() * n)]++;
  return c;
}

/** Fraction of points left out (count 0) in `reps` bootstrap samples of size n. */
export function oobFractionSamples(n: number, reps: number, seed: number): Float64Array {
  const r = rng(seed), out = new Float64Array(reps);
  for (let t = 0; t < reps; t++) {
    const c = bootstrapCounts(n, r);
    let z = 0; for (const v of c) if (!v) z++;
    out[t] = z / n;
  }
  return out;
}

/** P(a given point is not drawn) = (1 − 1/n)^n. */
export const pNotDrawn = (n: number): number => Math.pow(1 - 1 / n, n);

/** Variance of the average of B predictors with variance σ² and pairwise correlation ρ. */
export const varianceOfAverage = (sigma2: number, rho: number, B: number): number => rho * sigma2 + (1 - rho) * sigma2 / B;

/**
 * Monte Carlo: f_b = σ(√ρ Z_0 + √(1−ρ) Z_b) has variance σ² and pairwise correlation ρ;
 * the variance of their average over `reps` repetitions, for each B in Bs.
 */
export function simulateAverageVariance(sigma2: number, rho: number, Bs: number[], reps: number, seed: number): Float64Array {
  const r = rng(seed), s = Math.sqrt(sigma2), a = Math.sqrt(Math.max(0, rho)), b = Math.sqrt(Math.max(0, 1 - rho));
  const out = new Float64Array(Bs.length);
  Bs.forEach((B, q) => {
    let s1 = 0, s2 = 0;
    for (let t = 0; t < reps; t++) {
      const z0 = r.norm();
      let m = 0;
      for (let k = 0; k < B; k++) m += s * (a * z0 + b * r.norm());
      m /= B;
      s1 += m; s2 += m * m;
    }
    const mean = s1 / reps;
    out[q] = s2 / reps - mean * mean;
  });
  return out;
}
