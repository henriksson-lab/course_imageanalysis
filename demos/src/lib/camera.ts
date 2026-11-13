// The pinhole camera of computer vision (no DOM). Owner: the geometry-from-images day
// (cameras/, two-view/, stereo/, sfm/). Users: pinhole, dltcalib, epipolar demos; twoview.ts.
//
// Conventions (vision): camera coordinates x right, y DOWN, z FORWARD (the graphics day's view
// space looks down −z with y up: the two differ by diag(1, −1, −1)). Pixels: u right, v down,
// pixel centres at integers. A camera maps a world point X to x ≃ K [R | t] X:
//   Xc = R X + t,  (xn, yn) = (Xc/Zc, Yc/Zc),  radial distortion (xd, yd) = (1 + k1 r² + k2 r⁴)(xn, yn),
//   u = fx xd + s yd + cx,  v = fy yd + cy.
// Matrices are row-major Float64Arrays (3 × 3: 9 entries, 3 × 4: 12 entries). Vectors are
// plain number arrays.
//
// Contents: building cameras (K, look-at, Rodrigues) · projection, pixel rays, distortion ·
// Hartley normalisation · DLT for P and for homographies · RQ decomposition of P · Zhang's
// planar calibration (K from homographies) · the graphics projection matrix for the same K.

import { svd, inv3, det3 } from "./linalg";

export type V3 = number[];
export type V2 = [number, number];

export interface Camera {
  /** Intrinsics, 3 × 3 row-major. */
  K: Float64Array;
  /** Rotation world → camera, 3 × 3 row-major. */
  R: Float64Array;
  /** Translation world → camera (Xc = R X + t). */
  t: V3;
  k1: number;
  k2: number;
}

// ---------------------------------------------------------------------------------------
// Small 3 × 3 helpers

export function mul3(A: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
    C[i * 3 + j] = s;
  }
  return C;
}
export function mulVec3(A: ArrayLike<number>, x: ArrayLike<number>): V3 {
  return [0, 1, 2].map(i => A[i * 3] * x[0] + A[i * 3 + 1] * x[1] + A[i * 3 + 2] * x[2]);
}
export function transpose3(A: ArrayLike<number>): Float64Array {
  return Float64Array.from([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]);
}
export const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len3 = (a: ArrayLike<number>): number => Math.hypot(a[0], a[1], a[2]);
export const unit3 = (a: V3): V3 => scale3(a, 1 / (len3(a) || 1));

/** K from focal lengths in pixels, principal point and skew. */
export function makeK(fx: number, cx: number, cy: number, fy = fx, s = 0): Float64Array {
  return Float64Array.from([fx, s, cx, 0, fy, cy, 0, 0, 1]);
}

/**
 * Rotation for a camera at `eye` looking at `target` (vision axes: z forward, y down).
 * `up` is the world's up direction; `roll` turns the camera about its viewing axis (radians,
 * clockwise in the image).
 */
export function lookAtR(eye: V3, target: V3, up: V3 = [0, 1, 0], roll = 0): Float64Array {
  const z = unit3(sub3(target, eye));
  let x = cross(z, up);                // right = forward × up (y down convention)
  if (len3(x) < 1e-9) x = cross(z, [1, 0, 0]);
  x = unit3(x);
  const y = cross(z, x);               // down
  const c = Math.cos(roll), s = Math.sin(roll);
  const xr = add3(scale3(x, c), scale3(y, s)), yr = add3(scale3(y, c), scale3(x, -s));
  return Float64Array.from([...xr, ...yr, ...z]);
}

/** A camera at `eye` looking at `target`. */
export function lookAtCamera(K: Float64Array, eye: V3, target: V3, roll = 0, k1 = 0, k2 = 0): Camera {
  const R = lookAtR(eye, target, [0, 1, 0], roll);
  return { K, R, t: scale3(mulVec3(R, eye), -1), k1, k2 };
}

/** Camera centre c = −Rᵀ t. */
export function cameraCentre(cam: { R: ArrayLike<number>; t: V3 }): V3 {
  return scale3(mulVec3(transpose3(cam.R), cam.t), -1);
}

/** The 3 × 4 camera matrix P = K [R | t] (distortion not included). */
export function cameraMatrix(cam: Camera): Float64Array {
  const KR = mul3(cam.K, cam.R), Kt = mulVec3(cam.K, cam.t), P = new Float64Array(12);
  for (let i = 0; i < 3; i++) { P[i * 4] = KR[i * 3]; P[i * 4 + 1] = KR[i * 3 + 1]; P[i * 4 + 2] = KR[i * 3 + 2]; P[i * 4 + 3] = Kt[i]; }
  return P;
}

/** World → camera coordinates. */
export const toCamera = (cam: { R: ArrayLike<number>; t: V3 }, X: ArrayLike<number>): V3 => add3(mulVec3(cam.R, X), cam.t);

/** Radial distortion of normalised coordinates. */
export function distort(xn: number, yn: number, k1: number, k2: number): V2 {
  const r2 = xn * xn + yn * yn, f = 1 + k1 * r2 + k2 * r2 * r2;
  return [xn * f, yn * f];
}

/** Invert the radial distortion by fixed-point iteration (enough for moderate k1, k2). */
export function undistortNormalised(xd: number, yd: number, k1: number, k2: number, iters = 20): V2 {
  let x = xd, y = yd;
  for (let i = 0; i < iters; i++) {
    const r2 = x * x + y * y, f = 1 + k1 * r2 + k2 * r2 * r2;
    x = xd / f; y = yd / f;
  }
  return [x, y];
}

/** Pixel from normalised (distorted) coordinates through K. */
export function applyK(K: ArrayLike<number>, x: number, y: number): V2 {
  return [K[0] * x + K[1] * y + K[2], K[4] * y + K[5]];
}

/** Normalised coordinates from a pixel: K⁻¹ (u, v, 1). */
export function invK(K: ArrayLike<number>, u: number, v: number): V2 {
  const y = (v - K[5]) / K[4];
  return [(u - K[2] - K[1] * y) / K[0], y];
}

export interface Projection { u: number; v: number; Z: number }

/** Project a world point (with distortion). Z is the depth along the viewing axis (≤ 0: behind). */
export function project(cam: Camera, X: ArrayLike<number>): Projection {
  const Xc = toCamera(cam, X), Z = Xc[2];
  const [xd, yd] = distort(Xc[0] / Z, Xc[1] / Z, cam.k1, cam.k2);
  const [u, v] = applyK(cam.K, xd, yd);
  return { u, v, Z };
}

/** Projection by a 3 × 4 matrix P (no distortion): x ≃ P X. */
export function projectP(P: ArrayLike<number>, X: ArrayLike<number>): V2 {
  const w = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
  return [(P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3]) / w, (P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7]) / w];
}

/** The ray of a pixel: origin = camera centre, unit direction Rᵀ K⁻¹ (u, v, 1) (distortion removed). */
export function pixelRay(cam: Camera, u: number, v: number): { origin: V3; dir: V3 } {
  let [x, y] = invK(cam.K, u, v);
  if (cam.k1 || cam.k2) [x, y] = undistortNormalised(x, y, cam.k1, cam.k2);
  return { origin: cameraCentre(cam), dir: unit3(mulVec3(transpose3(cam.R), [x, y, 1])) };
}

// ---------------------------------------------------------------------------------------
// Rodrigues

/** Rotation matrix from an axis-angle vector w (angle = |w|). */
export function axisAngleToR(w: ArrayLike<number>): Float64Array {
  const th = len3(w);
  if (th < 1e-12) return Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const k = [w[0] / th, w[1] / th, w[2] / th], c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return Float64Array.from([
    c + k[0] * k[0] * C, k[0] * k[1] * C - k[2] * s, k[0] * k[2] * C + k[1] * s,
    k[1] * k[0] * C + k[2] * s, c + k[1] * k[1] * C, k[1] * k[2] * C - k[0] * s,
    k[2] * k[0] * C - k[1] * s, k[2] * k[1] * C + k[0] * s, c + k[2] * k[2] * C,
  ]);
}

/** Axis-angle vector of a rotation matrix (angle in [0, π]). */
export function RToAxisAngle(R: ArrayLike<number>): V3 {
  const c = Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2)), th = Math.acos(c);
  if (th < 1e-12) return [0, 0, 0];
  if (Math.PI - th < 1e-6) {
    // near π: axis from the diagonal of (R + I)/2
    const xx = Math.sqrt(Math.max(0, (R[0] + 1) / 2)), yy = Math.sqrt(Math.max(0, (R[4] + 1) / 2)), zz = Math.sqrt(Math.max(0, (R[8] + 1) / 2));
    let k: V3 = [xx, Math.sign(R[1] || 1) * yy, Math.sign(R[2] || 1) * zz];
    k = unit3(k);
    return scale3(k, th);
  }
  const f = th / (2 * Math.sin(th));
  return [(R[7] - R[5]) * f, (R[2] - R[6]) * f, (R[3] - R[1]) * f];
}

// ---------------------------------------------------------------------------------------
// Hartley normalisation

/** Similarity T moving the points to mean 0 and RMS distance √2; returns T (3 × 3) and the moved points. */
export function normalise2d(pts: ArrayLike<number>[]): { T: Float64Array; pts: V2[] } {
  const n = pts.length;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p[0] / n; my += p[1] / n; }
  let ms = 0;
  for (const p of pts) ms += ((p[0] - mx) ** 2 + (p[1] - my) ** 2) / n;
  const s = Math.SQRT2 / (Math.sqrt(ms) || 1);
  return { T: Float64Array.from([s, 0, -s * mx, 0, s, -s * my, 0, 0, 1]), pts: pts.map(p => [s * (p[0] - mx), s * (p[1] - my)] as V2) };
}

/** 3D version: mean 0, RMS distance √3; T is 4 × 4. */
export function normalise3d(pts: ArrayLike<number>[]): { T: Float64Array; pts: V3[] } {
  const n = pts.length, m = [0, 0, 0];
  for (const p of pts) for (let k = 0; k < 3; k++) m[k] += p[k] / n;
  let ms = 0;
  for (const p of pts) ms += ((p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2 + (p[2] - m[2]) ** 2) / n;
  const s = Math.sqrt(3) / (Math.sqrt(ms) || 1);
  return {
    T: Float64Array.from([s, 0, 0, -s * m[0], 0, s, 0, -s * m[1], 0, 0, s, -s * m[2], 0, 0, 0, 1]),
    pts: pts.map(p => [s * (p[0] - m[0]), s * (p[1] - m[1]), s * (p[2] - m[2])]),
  };
}

// ---------------------------------------------------------------------------------------
// DLT

export interface DLTResult {
  /** 3 × 4, scaled so that the last row's rotation part has norm 1 and points are in front (det > 0). */
  P: Float64Array;
  /** The 2n × 12 system actually solved (normalised coordinates when normalise = true). */
  A: Float64Array;
  /** Its 12 singular values, descending (the last one ≈ 0 for exact data). */
  S: Float64Array;
  /** s_max / s_11 of A (the conditioning of the problem; s_12 is the null direction). */
  cond: number;
}

/** Camera matrix from n ≥ 6 world–pixel pairs: null vector of A (two rows per point). */
export function dltCamera(X: ArrayLike<number>[], x: ArrayLike<number>[], normalise = true): DLTResult {
  const n = X.length;
  const N2 = normalise ? normalise2d(x) : { T: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), pts: x.map(p => [p[0], p[1]] as V2) };
  const N3 = normalise ? normalise3d(X) : { T: Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), pts: X.map(p => [p[0], p[1], p[2]]) };
  const A = new Float64Array(2 * n * 12);
  for (let i = 0; i < n; i++) {
    const [Xx, Xy, Xz] = N3.pts[i], [u, v] = N2.pts[i], Xh = [Xx, Xy, Xz, 1];
    for (let k = 0; k < 4; k++) {
      // row 1: [0ᵀ, −Xᵀ, v Xᵀ]; row 2: [Xᵀ, 0ᵀ, −u Xᵀ]
      A[(2 * i) * 12 + 4 + k] = -Xh[k];
      A[(2 * i) * 12 + 8 + k] = v * Xh[k];
      A[(2 * i + 1) * 12 + k] = Xh[k];
      A[(2 * i + 1) * 12 + 8 + k] = -u * Xh[k];
    }
  }
  const { S, V } = svd(A, 2 * n, 12);
  const p = Float64Array.from({ length: 12 }, (_, k) => V[k * 12 + 11]);
  // denormalise: P = T2⁻¹ P̂ T3
  const T2i = inv3(N2.T)!;
  const tmp = new Float64Array(12);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += p[i * 4 + k] * N3.T[k * 4 + j]; tmp[i * 4 + j] = s; }
  const P = new Float64Array(12);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 3; k++) s += T2i[i * 3 + k] * tmp[k * 4 + j]; P[i * 4 + j] = s; }
  fixScaleP(P);
  return { P, A, S, cond: S[0] / (S[10] || 1e-300) };
}

/** Scale P so that ‖(p31, p32, p33)‖ = 1 and det(M) > 0 (points in front have positive depth). */
export function fixScaleP(P: Float64Array): void {
  const r = Math.hypot(P[8], P[9], P[10]) || 1;
  const M = [P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]];
  const sg = det3(M) < 0 ? -1 : 1;
  for (let k = 0; k < 12; k++) P[k] *= sg / r;
}

/** RQ decomposition of P's left 3 × 3: P = K [R | t] with positive diagonal of K, K[8] = 1. */
export function decomposeP(P: ArrayLike<number>): { K: Float64Array; R: Float64Array; t: V3 } {
  const M = [P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]];
  // RQ via Gram–Schmidt from the last row up: M = K R with K upper triangular.
  const m3 = [M[6], M[7], M[8]], m2 = [M[3], M[4], M[5]], m1 = [M[0], M[1], M[2]];
  const k33 = len3(m3), r3 = scale3(m3, 1 / k33);
  const k23 = dot3(m2, r3), q2 = sub3(m2, scale3(r3, k23)), k22 = len3(q2), r2 = scale3(q2, 1 / k22);
  const k13 = dot3(m1, r3), k12 = dot3(m1, r2), q1 = sub3(sub3(m1, scale3(r3, k13)), scale3(r2, k12)), k11 = len3(q1), r1 = scale3(q1, 1 / k11);
  let K = Float64Array.from([k11, k12, k13, 0, k22, k23, 0, 0, k33]);
  let R = Float64Array.from([...r1, ...r2, ...r3]);
  if (det3(R) < 0) { for (let k = 0; k < 9; k++) R[k] = -R[k]; for (let k = 0; k < 9; k++) K[k] = -K[k]; }
  const s = K[8];
  const Ki = inv3(K)!;
  const t = mulVec3(Ki, [P[3], P[7], P[11]]);
  K = K.map(v => v / s);
  return { K, R, t };
}

/** Homography H (3 × 3, H[8] = 1 when possible) with dst ≃ H src, from n ≥ 4 pairs. */
export function homographyDLT(src: ArrayLike<number>[], dst: ArrayLike<number>[], normalise = true): { H: Float64Array; S: Float64Array } {
  const n = src.length;
  const I3 = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const Ns = normalise ? normalise2d(src) : { T: I3, pts: src.map(p => [p[0], p[1]] as V2) };
  const Nd = normalise ? normalise2d(dst) : { T: I3, pts: dst.map(p => [p[0], p[1]] as V2) };
  const A = new Float64Array(2 * n * 9);
  for (let i = 0; i < n; i++) {
    const [x, y] = Ns.pts[i], [u, v] = Nd.pts[i], X = [x, y, 1];
    for (let k = 0; k < 3; k++) {
      A[(2 * i) * 9 + 3 + k] = -X[k]; A[(2 * i) * 9 + 6 + k] = v * X[k];
      A[(2 * i + 1) * 9 + k] = X[k]; A[(2 * i + 1) * 9 + 6 + k] = -u * X[k];
    }
  }
  const { S, V } = svd(A, 2 * n, 9);
  const h = Float64Array.from({ length: 9 }, (_, k) => V[k * 9 + 8]);
  const H = mul3(mul3(inv3(Nd.T)!, h), Ns.T);
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : Math.hypot(...H);
  return { H: H.map(v => v / s), S };
}

/** Apply a homography to a 2D point. */
export function applyH(H: ArrayLike<number>, x: number, y: number): V2 {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/**
 * Zhang's planar calibration: K from the homographies of a plane (Z = 0) in several views.
 * Each view gives h1ᵀ ω h2 = 0 and h1ᵀ ω h1 = h2ᵀ ω h2 with ω = K⁻ᵀ K⁻¹ (6 unknowns up to
 * scale). With fewer than 3 views, zero skew is added as a constraint (2 views), and 1 view
 * is not enough. Returns null when the system is degenerate (ω not positive definite).
 */
export function zhangK(Hs: ArrayLike<number>[], zeroSkew = false): { K: Float64Array | null; S: Float64Array; rows: number } {
  const v = (H: ArrayLike<number>, i: number, j: number): number[] => {
    const hi = [H[i], H[3 + i], H[6 + i]], hj = [H[j], H[3 + j], H[6 + j]];
    return [hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1], hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]];
  };
  const rows: number[][] = [];
  for (const H of Hs) {
    rows.push(v(H, 0, 1));
    const a = v(H, 0, 0), b = v(H, 1, 1);
    rows.push(a.map((x, k) => x - b[k]));
  }
  if (zeroSkew || Hs.length < 3) rows.push([0, 1, 0, 0, 0, 0]);
  const m = rows.length;
  const A = Float64Array.from(rows.flat());
  const { S, V } = svd(A, m, 6);
  if (m < 5) return { K: null, S, rows: m };
  let b = Array.from({ length: 6 }, (_, k) => V[k * 6 + 5]);
  if (b[0] < 0) b = b.map(x => -x);
  const [B11, B12, B22, B13, B23, B33] = b;
  const den = B11 * B22 - B12 * B12;
  if (!(den > 0) || !(B11 > 0)) return { K: null, S, rows: m };
  const v0 = (B12 * B13 - B11 * B23) / den;
  const lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
  if (!(lam / B11 > 0) || !(lam * B11 / den > 0)) return { K: null, S, rows: m };
  const alpha = Math.sqrt(lam / B11), beta = Math.sqrt(lam * B11 / den);
  const gamma = -B12 * alpha * alpha * beta / lam, u0 = gamma * v0 / beta - B13 * alpha * alpha / lam;
  return { K: Float64Array.from([alpha, gamma, u0, 0, beta, v0, 0, 0, 1]), S, rows: m };
}

/** Pose of a plane (Z = 0) view from its homography and K: H ≃ K [r1 r2 t]. */
export function poseFromHomography(K: ArrayLike<number>, H: ArrayLike<number>): { R: Float64Array; t: V3 } {
  const Ki = inv3(K)!;
  const h1 = mulVec3(Ki, [H[0], H[3], H[6]]), h2 = mulVec3(Ki, [H[1], H[4], H[7]]), h3 = mulVec3(Ki, [H[2], H[5], H[8]]);
  let s = 1 / len3(h1);
  if (h3[2] * s < 0) s = -s;
  const r1 = scale3(h1, s), r2 = scale3(h2, s), r3 = cross(r1, r2);
  // re-orthonormalise via SVD
  const Q = Float64Array.from([r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]]);
  const { U, V } = svd(Q, 3, 3);
  const Rc = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let a = 0; for (let k = 0; k < 3; k++) a += U[i * 3 + k] * V[j * 3 + k]; Rc[i * 3 + j] = a; }
  return { R: Rc, t: scale3(h3, s) };
}

/**
 * The graphics day's 4 × 4 projection for the same intrinsics (row-major), for an image of
 * w × h pixels spanning [0, w] × [0, h], near/far planes n, f. It acts on GL view coordinates
 * (= diag(1, −1, −1) times vision camera coordinates).
 */
export function glProjectionFromK(K: ArrayLike<number>, w: number, h: number, n: number, f: number): Float64Array {
  return Float64Array.from([
    2 * K[0] / w, -2 * K[1] / w, 1 - 2 * K[2] / w, 0,
    0, 2 * K[4] / h, 2 * K[5] / h - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0,
  ]);
}

/** RMS reprojection error (pixels) of a camera on world–pixel pairs. */
export function reprojectionRMS(cam: Camera, X: ArrayLike<number>[], x: ArrayLike<number>[]): number {
  let s = 0;
  for (let i = 0; i < X.length; i++) { const p = project(cam, X[i]); s += (p.u - x[i][0]) ** 2 + (p.v - x[i][1]) ** 2; }
  return Math.sqrt(s / Math.max(1, X.length));
}

// ---------------------------------------------------------------------------------------
// Calibration-object scenes

/** Corners of the calibration object: two orthogonal checkerboard planes meeting along the y axis
 *  (plane x = 0 and plane z = 0), `n` × `n` corners each, spacing `sq`; world y is up. */
export function twoPlaneTarget(n = 6, sq = 0.04): V3[] {
  const out: V3[] = [];
  for (let i = 1; i <= n; i++) for (let j = 1; j <= n; j++) out.push([0, j * sq, i * sq]);   // left wall (x = 0)
  for (let i = 1; i <= n; i++) for (let j = 1; j <= n; j++) out.push([i * sq, j * sq, 0]);   // right wall (z = 0)
  return out;
}

/** Corners of a flat checkerboard (Z = 0 in its own frame), nx × ny, spacing sq, centred. */
export function planarTarget(nx = 8, ny = 6, sq = 0.03): V3[] {
  const out: V3[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out.push([(i - (nx - 1) / 2) * sq, (j - (ny - 1) / 2) * sq, 0]);
  return out;
}
