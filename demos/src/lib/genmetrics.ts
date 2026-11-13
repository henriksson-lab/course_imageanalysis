// Distribution-level metrics for generated samples (owner: the diffusion-models day, unit
// diffusion-sampling/). No DOM. Kept apart from metrics.ts (per image) and eval.ts (per object).
// Point sets are flat row-major arrays of n × d numbers. Tests: demos/test/diffusion-test.ts.
import { jacobiEigen } from "./linalg";

/** Mean and covariance (d × d, divided by n) of a point set. */
export function meanCov(X: ArrayLike<number>, d: number): { mu: Float64Array; S: Float64Array } {
  const n = X.length / d, mu = new Float64Array(d), S = new Float64Array(d * d);
  for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) mu[i] += X[p * d + i] / n;
  for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) S[i * d + j] += (X[p * d + i] - mu[i]) * (X[p * d + j] - mu[j]) / n;
  return { mu, S };
}

/** Symmetric PSD square root via eigen-decomposition. */
export function sqrtmSym(A: ArrayLike<number>, d: number): Float64Array {
  const { values, vectors } = jacobiEigen(Float64Array.from(A), d);
  const out = new Float64Array(d * d);
  for (let k = 0; k < d; k++) {
    const s = Math.sqrt(Math.max(0, values[k]));
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) out[i * d + j] += s * vectors[i * d + k] * vectors[j * d + k];
  }
  return out;
}

function matMulSq(A: ArrayLike<number>, B: ArrayLike<number>, d: number): Float64Array {
  const C = new Float64Array(d * d);
  for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) { const a = A[i * d + k]; for (let j = 0; j < d; j++) C[i * d + j] += a * B[k * d + j]; }
  return C;
}

/**
 * Fréchet distance between N(μ1, S1) and N(μ2, S2):
 * ‖μ1 − μ2‖² + tr(S1 + S2 − 2 (S1^½ S2 S1^½)^½). Closed form in 1D and 2D
 * (tr √M = √(tr M + 2√det M) for a 2 × 2 PSD M, whose eigenvalues are those of S1 S2),
 * eigen-decompositions otherwise.
 */
export function frechetGaussian(mu1: ArrayLike<number>, S1: ArrayLike<number>, mu2: ArrayLike<number>, S2: ArrayLike<number>, d: number): number {
  let m = 0; for (let i = 0; i < d; i++) m += (mu1[i] - mu2[i]) ** 2;
  let tr = 0; for (let i = 0; i < d; i++) tr += S1[i * d + i] + S2[i * d + i];
  let cross: number;
  if (d === 1) cross = Math.sqrt(Math.max(0, S1[0] * S2[0]));
  else if (d === 2) {
    const P = matMulSq(S1, S2, 2), trP = P[0] + P[3];
    const det1 = S1[0] * S1[3] - S1[1] * S1[2], det2 = S2[0] * S2[3] - S2[1] * S2[2];
    cross = Math.sqrt(Math.max(0, trP + 2 * Math.sqrt(Math.max(0, det1 * det2))));
  } else {
    const r = sqrtmSym(S1, d), M = matMulSq(matMulSq(r, S2, d), r, d);
    const sq = sqrtmSym(M, d);
    cross = 0; for (let i = 0; i < d; i++) cross += sq[i * d + i];
  }
  return m + tr - 2 * cross;
}

/** Fréchet distance between the Gaussians fitted to two point sets. */
export function frechetOfSets(A: ArrayLike<number>, B: ArrayLike<number>, d: number): number {
  const a = meanCov(A, d), b = meanCov(B, d);
  return frechetGaussian(a.mu, a.S, b.mu, b.S, d);
}

function dist2(X: ArrayLike<number>, p: number, Y: ArrayLike<number>, q: number, d: number): number {
  let s = 0; for (let i = 0; i < d; i++) { const e = X[p * d + i] - Y[q * d + i]; s += e * e; } return s;
}

/** Distance from every point of X to its k-th nearest OTHER point of X (brute force). */
export function knnRadii(X: ArrayLike<number>, d: number, k: number): Float64Array {
  const n = X.length / d, out = new Float64Array(n), buf = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    let m = 0;
    for (let q = 0; q < n; q++) if (q !== p) buf[m++] = dist2(X, p, X, q, d);
    const s = buf.subarray(0, m); s.sort();
    out[p] = Math.sqrt(s[Math.min(k, m) - 1]);
  }
  return out;
}

/** True where a point lies inside at least one ball (centre R[q], radius rad[q]). */
function inBalls(P: ArrayLike<number>, R: ArrayLike<number>, rad: ArrayLike<number>, d: number): Uint8Array {
  const n = P.length / d, m = R.length / d, out = new Uint8Array(n);
  for (let p = 0; p < n; p++) for (let q = 0; q < m; q++) if (dist2(P, p, R, q, d) <= rad[q] * rad[q]) { out[p] = 1; break; }
  return out;
}

/**
 * k-NN precision and recall (Kynkäänniemi et al. 2019): the real manifold is the union of balls
 * around real points with radius = distance to the k-th nearest real neighbour; precision = the
 * fraction of generated points inside it; recall = the fraction of real points inside the
 * generated manifold (same construction on the generated points).
 */
export function precisionRecall(real: ArrayLike<number>, fake: ArrayLike<number>, d: number, k = 3): { precision: number; recall: number; inReal: Uint8Array; inFake: Uint8Array; rReal: Float64Array; rFake: Float64Array } {
  const rReal = knnRadii(real, d, k), rFake = knnRadii(fake, d, k);
  const inReal = inBalls(fake, real, rReal, d), inFake = inBalls(real, fake, rFake, d);
  const mean = (u: Uint8Array) => { let s = 0; for (const v of u) s += v; return s / u.length; };
  return { precision: mean(inReal), recall: mean(inFake), inReal, inFake, rReal, rFake };
}

/** Distance from each sample to its nearest training point. */
export function nearestDistances(samples: ArrayLike<number>, train: ArrayLike<number>, d: number): Float64Array {
  const n = samples.length / d, m = train.length / d, out = new Float64Array(n);
  for (let p = 0; p < n; p++) { let b = Infinity; for (let q = 0; q < m; q++) b = Math.min(b, dist2(samples, p, train, q, d)); out[p] = Math.sqrt(b); }
  return out;
}

/** 1D Wasserstein-1 distance between two samples (same size: mean |sorted difference|; otherwise via quantile functions). */
export function wasserstein1d(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const A = Float64Array.from(a).sort(), B = Float64Array.from(b).sort();
  if (A.length === B.length) { let s = 0; for (let i = 0; i < A.length; i++) s += Math.abs(A[i] - B[i]); return s / A.length; }
  // integrate |F_A^{-1}(u) − F_B^{-1}(u)| over u on a merged grid
  const n = 2000; let s = 0;
  for (let i = 0; i < n; i++) { const u = (i + 0.5) / n; s += Math.abs(A[Math.floor(u * A.length)] - B[Math.floor(u * B.length)]); }
  return s / n;
}

/** Fraction of positions where lo[i] ≤ truth[i] ≤ hi[i]. */
export function intervalCoverage(truth: ArrayLike<number>, lo: ArrayLike<number>, hi: ArrayLike<number>): number {
  let c = 0; for (let i = 0; i < truth.length; i++) if (truth[i] >= lo[i] && truth[i] <= hi[i]) c++;
  return c / truth.length;
}
