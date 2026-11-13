// Two-view geometry (no DOM). Owner: the geometry-from-images day (two-view/, stereo/).
// Users: the epipolar demo, tests. Conventions as lib/camera.ts (vision axes, pixels with
// y down, row-major 3 × 3 Float64Arrays). The epipolar constraint is x2ᵀ F x1 = 0: F maps a
// point of image 1 to its epipolar line in image 2, a line (a, b, c) meaning a u + b v + c = 0.
//
// Contents: [t]ₓ, E from (R, t), F from E · eight-point (normalised or not) with the rank-2
// projection · epipolar lines, epipoles, distances (point–line, symmetric, Sampson) ·
// decomposing E (four candidates, cheirality) · triangulation (DLT, midpoint) · a RANSAC loop
// for F (the generic lib/ransac.ts works on 2D points only) · calibrated rectification.

import { svd } from "./linalg";
import { rng, Rng, sampleIndices } from "./random";
import { Camera, cameraCentre, cameraMatrix, mul3, mulVec3, transpose3, normalise2d, V2, V3, cross, sub3, dot3, add3, scale3, unit3, len3, invK } from "./camera";

export type M3 = Float64Array;

/** The cross-product matrix [t]ₓ: [t]ₓ v = t × v. */
export function skew(t: ArrayLike<number>): M3 {
  return Float64Array.from([0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]);
}

/** Essential matrix of a second camera with x2 = R x1 + t (camera 1 at the origin): E = [t]ₓ R. */
export const essentialFromRt = (R: ArrayLike<number>, t: ArrayLike<number>): M3 => mul3(skew(t), R);

/** F = K2⁻ᵀ E K1⁻¹. */
export function fundamentalFromE(E: ArrayLike<number>, K1: ArrayLike<number>, K2: ArrayLike<number>): M3 {
  return mul3(mul3(transpose3(inv3s(K2)), E), inv3s(K1));
}

function inv3s(K: ArrayLike<number>): M3 {
  // K is upper triangular with K[8] = 1
  const [a, b, c, , d, e] = [K[0], K[1], K[2], K[3], K[4], K[5]];
  return Float64Array.from([1 / a, -b / (a * d), (b * e - c * d) / (a * d), 0, 1 / d, -e / d, 0, 0, 1]);
}

/** Relative pose of camera 2 with respect to camera 1: x2 = R x1 + t (camera coordinates). */
export function relativePose(c1: Camera, c2: Camera): { R: M3; t: V3 } {
  const R = mul3(c2.R, transpose3(c1.R));
  return { R, t: sub3(c2.t, mulVec3(R, c1.t)) };
}

/** The true F between two cameras. */
export function fundamentalFromCameras(c1: Camera, c2: Camera): M3 {
  const { R, t } = relativePose(c1, c2);
  return normaliseF(fundamentalFromE(essentialFromRt(R, t), c1.K, c2.K));
}

/** Scale F to unit Frobenius norm with a fixed sign (largest |entry| positive). */
export function normaliseF(F: ArrayLike<number>): M3 {
  let n = 0, big = 0;
  for (let k = 0; k < 9; k++) { n += F[k] * F[k]; if (Math.abs(F[k]) > Math.abs(big)) big = F[k]; }
  const s = (big < 0 ? -1 : 1) / (Math.sqrt(n) || 1);
  return Float64Array.from(F, v => v * s);
}

/** Closest rank-2 matrix (Frobenius norm): zero the smallest singular value. */
export function enforceRank2(F: ArrayLike<number>): { F: M3; S: Float64Array } {
  const { U, S, V } = svd(F, 3, 3);
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 2; k++) s += U[i * 3 + k] * S[k] * V[j * 3 + k];
    out[i * 3 + j] = s;
  }
  return { F: out, S };
}

export interface EightPointResult {
  /** Rank-2, unit Frobenius norm. */
  F: M3;
  /** Before the rank-2 projection. */
  Ffull: M3;
  /** Singular values of the n × 9 system (descending). */
  SA: Float64Array;
  /** Singular values of Ffull (the third is set to 0 by the projection). */
  SF: Float64Array;
}

/** Eight-point algorithm on n ≥ 8 matches (x1[i] ↔ x2[i]), with or without Hartley normalisation. */
export function eightPoint(x1: ArrayLike<number>[], x2: ArrayLike<number>[], normalise = true): EightPointResult | null {
  const n = x1.length;
  if (n < 8) return null;
  const I3 = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const N1 = normalise ? normalise2d(x1) : { T: I3, pts: x1.map(p => [p[0], p[1]] as V2) };
  const N2 = normalise ? normalise2d(x2) : { T: I3, pts: x2.map(p => [p[0], p[1]] as V2) };
  const A = new Float64Array(n * 9);
  for (let i = 0; i < n; i++) {
    const [u, v] = N1.pts[i], [u2, v2] = N2.pts[i];
    A.set([u2 * u, u2 * v, u2, v2 * u, v2 * v, v2, u, v, 1], i * 9);
  }
  const { S: SA, V } = svd(A, n, 9);
  const f = Float64Array.from({ length: 9 }, (_, k) => V[k * 9 + 8]);
  const r2 = enforceRank2(f);
  const back = (G: ArrayLike<number>) => normaliseF(mul3(mul3(transpose3(N2.T), G), N1.T));
  return { F: back(r2.F), Ffull: back(f), SA, SF: r2.S };
}

/** Epipolar line in image 2 of a point in image 1: l = F x1 (a, b, c), scaled so a² + b² = 1. */
export function epipolarLine(F: ArrayLike<number>, x: ArrayLike<number>): V3 {
  const l = mulVec3(F, [x[0], x[1], 1]), s = Math.hypot(l[0], l[1]) || 1;
  return [l[0] / s, l[1] / s, l[2] / s];
}

/** Epipolar line in image 1 of a point in image 2: l = Fᵀ x2. */
export const epipolarLineT = (F: ArrayLike<number>, x: ArrayLike<number>): V3 => epipolarLine(transpose3(F), x);

/** Distance (pixels) of a point from a line normalised by epipolarLine. */
export const pointLineDistance = (l: ArrayLike<number>, x: ArrayLike<number>): number => Math.abs(l[0] * x[0] + l[1] * x[1] + l[2]);

/** Epipoles: e1 (image 1, F e1 = 0) and e2 (image 2, Fᵀ e2 = 0) as homogeneous 3-vectors. */
export function epipoles(F: ArrayLike<number>): { e1: V3; e2: V3 } {
  const V = svd(F, 3, 3).V, W = svd(transpose3(F), 3, 3).V;
  return { e1: [V[2], V[5], V[8]], e2: [W[2], W[5], W[8]] };
}

/** Sampson distance (first-order geometric error, pixels²) of a match under F. */
export function sampsonDistance(F: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>): number {
  const a = mulVec3(F, [x1[0], x1[1], 1]), b = mulVec3(transpose3(F), [x2[0], x2[1], 1]);
  const e = x2[0] * a[0] + x2[1] * a[1] + a[2];
  return (e * e) / (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]);
}

/** Symmetric epipolar distance: mean of the two point-to-line distances (pixels). */
export function symmetricDistance(F: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>): number {
  return 0.5 * (pointLineDistance(epipolarLine(F, x1), x2) + pointLineDistance(epipolarLineT(F, x2), x1));
}

// ---------------------------------------------------------------------------------------
// Decomposing E

/** The four (R, t) candidates of an essential matrix (t of unit length). */
export function decomposeE(E: ArrayLike<number>): { R: M3; t: V3 }[] {
  let { U, V } = svd(E, 3, 3);
  const det = (A: Float64Array) => A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  // the SVD leaves the left vector of a zero singular value as 0: complete U by u3 = u1 × u2
  const u3c = cross([U[0], U[3], U[6]], [U[1], U[4], U[7]]);
  U[2] = u3c[0]; U[5] = u3c[1]; U[8] = u3c[2];
  if (det(U) < 0) U = U.map((v, k) => (k % 3 === 2 ? -v : v));
  if (det(V) < 0) V = V.map((v, k) => (k % 3 === 2 ? -v : v));
  const W = Float64Array.from([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const Vt = transpose3(V);
  const R1 = mul3(mul3(U, W), Vt), R2 = mul3(mul3(U, transpose3(W)), Vt);
  const u3: V3 = [U[2], U[5], U[8]];
  return [{ R: R1, t: u3 }, { R: R1, t: scale3(u3, -1) }, { R: R2, t: u3 }, { R: R2, t: scale3(u3, -1) }];
}

/** Pick the candidate that puts the most triangulated points in front of both cameras. x are normalised coordinates (K⁻¹ x). */
export function selectPose(cands: { R: M3; t: V3 }[], n1: ArrayLike<number>[], n2: ArrayLike<number>[]): { R: M3; t: V3; inFront: number; index: number } {
  let best = { ...cands[0], inFront: -1, index: 0 };
  cands.forEach((c, k) => {
    const P1 = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    const P2 = Float64Array.from([c.R[0], c.R[1], c.R[2], c.t[0], c.R[3], c.R[4], c.R[5], c.t[1], c.R[6], c.R[7], c.R[8], c.t[2]]);
    let nf = 0;
    for (let i = 0; i < n1.length; i++) {
      const X = triangulateDLT(P1, P2, n1[i], n2[i]);
      const z2 = dot3([c.R[6], c.R[7], c.R[8]], X) + c.t[2];
      if (X[2] > 0 && z2 > 0) nf++;
    }
    if (nf > best.inFront) best = { ...c, inFront: nf, index: k };
  });
  return best;
}

// ---------------------------------------------------------------------------------------
// Triangulation

/** Linear (DLT) triangulation from two 3 × 4 matrices and the two image points. */
export function triangulateDLT(P1: ArrayLike<number>, P2: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>): V3 {
  const A = new Float64Array(16);
  const rows = (P: ArrayLike<number>, x: ArrayLike<number>, o: number) => {
    for (let k = 0; k < 4; k++) {
      A[o * 4 + k] = x[0] * P[8 + k] - P[k];
      A[(o + 1) * 4 + k] = x[1] * P[8 + k] - P[4 + k];
    }
  };
  rows(P1, x1, 0); rows(P2, x2, 2);
  const { V } = svd(A, 4, 4);
  const w = V[15];
  return [V[3] / w, V[7] / w, V[11] / w];
}

/** Midpoint of the shortest segment between two rays (origin + s · dir). Also returns the gap. */
export function triangulateMidpoint(o1: V3, d1: V3, o2: V3, d2: V3): { X: V3; gap: number } {
  const w0 = sub3(o1, o2), a = dot3(d1, d1), b = dot3(d1, d2), c = dot3(d2, d2), d = dot3(d1, w0), e = dot3(d2, w0);
  const den = a * c - b * b;
  const s = den > 1e-15 ? (b * e - c * d) / den : 0, t = den > 1e-15 ? (a * e - b * d) / den : e / c;
  const p1 = add3(o1, scale3(d1, s)), p2 = add3(o2, scale3(d2, t));
  return { X: scale3(add3(p1, p2), 0.5), gap: len3(sub3(p1, p2)) };
}

/** Triangulate a match between two Camera objects (DLT on the camera matrices; no distortion). */
export function triangulateCams(c1: Camera, c2: Camera, x1: ArrayLike<number>, x2: ArrayLike<number>): V3 {
  return triangulateDLT(cameraMatrix(c1), cameraMatrix(c2), x1, x2);
}

// ---------------------------------------------------------------------------------------
// RANSAC for F

export interface RansacFResult {
  F: M3 | null;
  inliers: number[];
  /** Iteration at which the best sample was found (1-based). */
  foundAt: number;
  iterations: number;
}

/** RANSAC + eight-point (normalised): sample 8 matches, fit, count symmetric epipolar distances ≤ eps; refit on the inliers. */
export function ransacF(x1: ArrayLike<number>[], x2: ArrayLike<number>[], eps: number, iterations: number, seed = 1): RansacFResult {
  const r: Rng = rng(seed), n = x1.length;
  let best: number[] = [], foundAt = 0;
  for (let it = 1; it <= iterations && n >= 8; it++) {
    const s = Array.from(sampleIndices(r, n, 8));
    const fit = eightPoint(s.map(i => x1[i]), s.map(i => x2[i]), true);
    if (!fit) continue;
    const inl: number[] = [];
    for (let i = 0; i < n; i++) if (symmetricDistance(fit.F, x1[i], x2[i]) <= eps) inl.push(i);
    if (inl.length > best.length) { best = inl; foundAt = it; }
  }
  if (best.length < 8) return { F: null, inliers: best, foundAt, iterations };
  const ref = eightPoint(best.map(i => x1[i]), best.map(i => x2[i]), true)!;
  const inl: number[] = [];
  for (let i = 0; i < n; i++) if (symmetricDistance(ref.F, x1[i], x2[i]) <= eps) inl.push(i);
  return { F: ref.F, inliers: inl, foundAt, iterations };
}

// ---------------------------------------------------------------------------------------
// Rectification (calibrated, same K for both cameras)

/**
 * Rotate both cameras (about their centres) to a common orientation whose x axis is the
 * baseline; returns the two image homographies H1, H2 (new pixel ≃ H old pixel) and the new
 * rotation. For the same K: H = K R_rect Rᵀ K⁻¹.
 */
export function rectifyCalibrated(c1: Camera, c2: Camera): { H1: M3; H2: M3; Rrect: M3 } {
  const C1 = cameraCentre(c1), C2 = cameraCentre(c2);
  const ex = unit3(sub3(C2, C1));
  const oldZ: V3 = add3([c1.R[6], c1.R[7], c1.R[8]], [c2.R[6], c2.R[7], c2.R[8]]);
  const ey = unit3(cross(oldZ, ex));
  const ez = cross(ex, ey);
  const Rrect = Float64Array.from([...ex, ...ey, ...ez]);
  const H = (c: Camera) => mul3(mul3(c.K, mul3(Rrect, transpose3(c.R))), inv3s(c.K));
  return { H1: H(c1), H2: H(c2), Rrect };
}

/** Normalised coordinates of a pixel (K⁻¹ x) as a 2-vector. */
export const normalisedPoint = (K: ArrayLike<number>, x: ArrayLike<number>): V2 => invK(K, x[0], x[1]);
