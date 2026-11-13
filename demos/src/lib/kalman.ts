// A small dense linear Kalman filter (tracking day). No DOM. States of at most a few
// dimensions (random walk or constant velocity, 1D or 2D), so plain number[][] matrices.
//
//   model:        x_{t+1} = F x_t + w,  w ~ N(0, Q)       measurement: z = H x + v,  v ~ N(0, R)
//   predict:      x⁻ = F x,  P⁻ = F P Fᵀ + Q
//   update:       y = z − H x⁻,  S = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹,  x = x⁻ + K y,  P = (I − K H) P⁻
//   gate / cost:  d² = yᵀ S⁻¹ y (Mahalanobis),  cost = d² + log det S
export type Mat = number[][];

export const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array(c).fill(0));
export const eye = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
export const mul = (A: Mat, B: Mat): Mat => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
export const tr = (A: Mat): Mat => A[0].map((_, j) => A.map(row => row[j]));
export const add = (A: Mat, B: Mat): Mat => A.map((row, i) => row.map((a, j) => a + B[i][j]));
export const sub = (A: Mat, B: Mat): Mat => A.map((row, i) => row.map((a, j) => a - B[i][j]));
export const mv = (A: Mat, x: number[]): number[] => A.map(row => row.reduce((s, a, k) => s + a * x[k], 0));

/** Inverse of a small symmetric positive-definite (or any invertible) matrix, Gauss–Jordan. */
export function inv(A: Mat): Mat {
  const n = A.length, M = A.map((row, i) => [...row, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = M[r][c];
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map(row => row.slice(n));
}

export function det(A: Mat): number {
  const n = A.length;
  if (n === 1) return A[0][0];
  if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  let s = 0;
  for (let j = 0; j < n; j++) s += (j % 2 ? -1 : 1) * A[0][j] * det(A.slice(1).map(r => r.filter((_, k) => k !== j)));
  return s;
}

export interface KModel { F: Mat; H: Mat; Q: Mat; R: Mat }
export interface KState { x: number[]; P: Mat }

/** Random walk in `dims` dimensions: best guess = last position; Q = q I, R = r I. */
export function randomWalk(dims: number, q: number, r: number): KModel {
  const I = eye(dims);
  return { F: I, H: I, Q: I.map(row => row.map(v => v * q)), R: I.map(row => row.map(v => v * r)) };
}

/**
 * Constant velocity: per dimension the state (position, velocity), F = [[1, Δt], [0, 1]],
 * H picks the positions; process noise q Δt on the velocity only (Q = diag(0, qΔt)).
 * State order: (x, vx, y, vy, ...).
 */
export function constantVelocity(dims: number, dt: number, q: number, r: number): KModel {
  const n = 2 * dims, F = eye(n), H = zeros(dims, n), Q = zeros(n, n), R = zeros(dims, dims);
  for (let d = 0; d < dims; d++) {
    F[2 * d][2 * d + 1] = dt;
    H[d][2 * d] = 1;
    Q[2 * d + 1][2 * d + 1] = q * dt;
    R[d][d] = r;
  }
  return { F, H, Q, R };
}

export function predict(s: KState, m: KModel): KState {
  return { x: mv(m.F, s.x), P: add(mul(mul(m.F, s.P), tr(m.F)), m.Q) };
}

export interface Innovation { y: number[]; S: Mat; Sinv: Mat; d2: number; logdetS: number }
export function innovation(s: KState, z: number[], m: KModel): Innovation {
  const hx = mv(m.H, s.x), y = z.map((v, i) => v - hx[i]);
  const S = add(mul(mul(m.H, s.P), tr(m.H)), m.R), Sinv = inv(S);
  const d2 = y.reduce((acc, yi, i) => acc + yi * Sinv[i].reduce((a, sij, j) => a + sij * y[j], 0), 0);
  return { y, S, Sinv, d2, logdetS: Math.log(det(S)) };
}

export interface Update extends Innovation { state: KState; K: Mat }
export function update(s: KState, z: number[], m: KModel): Update {
  const inn = innovation(s, z, m);
  const K = mul(mul(s.P, tr(m.H)), inn.Sinv);
  const x = s.x.map((v, i) => v + K[i].reduce((a, k, j) => a + k * inn.y[j], 0));
  const P = mul(sub(eye(s.x.length), mul(K, m.H)), s.P);
  // keep P symmetric against rounding
  for (let i = 0; i < P.length; i++) for (let j = 0; j < i; j++) { const v = (P[i][j] + P[j][i]) / 2; P[i][j] = v; P[j][i] = v; }
  return { ...inn, state: { x, P }, K };
}

/** 1D update in closed form: gain K = P⁻/(P⁻ + R), x = x⁻ + K (z − x⁻), P = (1 − K) P⁻. */
export function update1(xPred: number, pPred: number, z: number, r: number): { x: number; P: number; K: number } {
  const K = pPred / (pPred + r);
  return { x: xPred + K * (z - xPred), P: (1 - K) * pPred, K };
}

/** Steady state of the 1D random-walk filter: P⁻∞ = (Q + √(Q² + 4QR))/2, K∞ = P⁻∞/(P⁻∞ + R). */
export function steadyRW(q: number, r: number): { Pminus: number; K: number; P: number } {
  const Pminus = (q + Math.sqrt(q * q + 4 * q * r)) / 2, K = Pminus / (Pminus + r);
  return { Pminus, K, P: (1 - K) * Pminus };
}

/** χ² quantile with 2 degrees of freedom: d² below which a fraction p of true measurements fall (2D). */
export const chi2Quantile2 = (p: number): number => -2 * Math.log(1 - p);
