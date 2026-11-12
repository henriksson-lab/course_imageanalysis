// Exact minimisation of a two-label chain energy by dynamic programming (Viterbi).
// No DOM: also used by the test (demos/test/).
//   E(L) = sum_p D_p(L_p) + lambda * #{p : L_p != L_{p+1}}

export interface ChainEnergy { data: number; boundaries: number; total: number }

export function chainEnergy(D0: ArrayLike<number>, D1: ArrayLike<number>, lambda: number, L: ArrayLike<number>): ChainEnergy {
  let data = 0, boundaries = 0;
  for (let p = 0; p < L.length; p++) {
    data += L[p] ? D1[p] : D0[p];
    if (p > 0 && L[p] !== L[p - 1]) boundaries++;
  }
  return { data, boundaries, total: data + lambda * boundaries };
}

/** The labelling with the smallest energy. Ties are broken towards label 0, and towards
 *  not changing label, so the answer is deterministic. */
export function chainDP(D0: ArrayLike<number>, D1: ArrayLike<number>, lambda: number): Uint8Array {
  const n = D0.length;
  const L = new Uint8Array(n);
  if (n === 0) return L;
  // cost[l] = cheapest energy of p_0..p_k with L_k = l; from[k][l] = best label of p_{k-1}
  const from = new Uint8Array(2 * n);
  let c0 = D0[0], c1 = D1[0];
  for (let k = 1; k < n; k++) {
    const a0 = c0 <= c1 + lambda ? 0 : 1;          // best predecessor if L_k = 0
    const a1 = c1 <= c0 + lambda ? 1 : 0;          // best predecessor if L_k = 1
    const n0 = (a0 === 0 ? c0 : c1 + lambda) + D0[k];
    const n1 = (a1 === 1 ? c1 : c0 + lambda) + D1[k];
    from[2 * k] = a0; from[2 * k + 1] = a1;
    c0 = n0; c1 = n1;
  }
  L[n - 1] = c0 <= c1 ? 0 : 1;
  for (let k = n - 1; k > 0; k--) L[k - 1] = from[2 * k + L[k]];
  return L;
}

/** Thresholding = the optimum for lambda = 0: each sample takes its cheaper label. */
export function chainThreshold(D0: ArrayLike<number>, D1: ArrayLike<number>): Uint8Array {
  return Uint8Array.from({ length: D0.length }, (_, p) => (D1[p] < D0[p] ? 1 : 0));
}

// ---- the profile shown by the segchain demo ----
export const PROFILE_N = 80;
export const PROFILE_SEED = 3;
/** True labels: a cell (samples 12-49) and a thin structure (samples 64-65). */
export const PROFILE_TRUTH = Array.from({ length: PROFILE_N }, (_, p) => ((p >= 12 && p < 50) || (p >= 64 && p < 66) ? 1 : 0));
export const PROFILE_LEVELS = [60, 160];
/** Model noise s of the data term D(l) = (I - mu_l)^2 / (2 s^2). */
export const S_MODEL = 30;

/** The noisy profile: true levels plus seeded Gaussian noise, rounded and clipped to 0..255.
 *  `norm` is a seeded standard normal generator (rng(PROFILE_SEED).norm from lib/ui). */
export function makeProfile(noise: number, norm: () => number): number[] {
  return PROFILE_TRUTH.map(t => Math.round(Math.min(255, Math.max(0, PROFILE_LEVELS[t] + noise * norm()))));
}

export function chainCosts(I: number[], mu0: number, mu1: number): { D0: number[]; D1: number[] } {
  const s2 = 2 * S_MODEL * S_MODEL;
  return { D0: I.map(v => (v - mu0) ** 2 / s2), D1: I.map(v => (v - mu1) ** 2 / s2) };
}
