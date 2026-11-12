// Small dense linear algebra (no DOM). Matrices are row-major Float64Arrays (or any
// ArrayLike<number> on input) with their sizes passed explicitly: an m × n matrix A has
// A[i*n + j] in row i, column j. Meant for the sizes of teaching demos (up to ~100 × 100):
// clarity over speed, but every routine is numerically sound (pivoting, Jacobi rotations).
//
// Sections: vectors and products · 2×2 and 3×3 closed forms · symmetric eigen (Jacobi) ·
// SVD (one-sided Jacobi) · LU solve/inverse/determinant · Cholesky · least squares,
// pseudo-inverse, condition number, rank and null space.

type M = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Vectors and products

export function dot(a: M, b: M): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const norm = (a: M): number => Math.sqrt(dot(a, a));

/** n × n identity. */
export function identity(n: number): Float64Array {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return I;
}

/** C = A B with A m × k and B k × n. */
export function matMul(A: M, B: M, m: number, k: number, n: number): Float64Array {
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let l = 0; l < k; l++) {
    const a = A[i * k + l];
    if (a === 0) continue;
    for (let j = 0; j < n; j++) C[i * n + j] += a * B[l * n + j];
  }
  return C;
}

/** Aᵀ of an m × n matrix (n × m). */
export function transpose(A: M, m: number, n: number): Float64Array {
  const T = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = A[i * n + j];
  return T;
}

/** y = A x with A m × n. */
export function matVec(A: M, x: M, m: number, n: number): Float64Array {
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < n; j++) s += A[i * n + j] * x[j]; y[i] = s; }
  return y;
}

/** y = Aᵀ x with A m × n (y has n entries). */
export function matTVec(A: M, x: M, m: number, n: number): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < m; i++) { const xi = x[i]; for (let j = 0; j < n; j++) y[j] += A[i * n + j] * xi; }
  return y;
}

/** Outer product a bᵀ (a.length × b.length). */
export function outer(a: M, b: M): Float64Array {
  const O = new Float64Array(a.length * b.length);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) O[i * b.length + j] = a[i] * b[j];
  return O;
}

/** Covariance matrix (d × d) of N points stored row-major as N × d; divides by N − 1 (or N with `population`). */
export function covariance(X: M, N: number, d: number, population = false): { mean: Float64Array; cov: Float64Array } {
  const mean = new Float64Array(d), cov = new Float64Array(d * d);
  for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / N;
  for (let i = 0; i < N; i++) for (let a = 0; a < d; a++) {
    const xa = X[i * d + a] - mean[a];
    for (let b = a; b < d; b++) cov[a * d + b] += xa * (X[i * d + b] - mean[b]);
  }
  const div = population ? N : Math.max(1, N - 1);
  for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { cov[a * d + b] /= div; cov[b * d + a] = cov[a * d + b]; }
  return { mean, cov };
}

// ---------------------------------------------------------------------------------------
// 2 × 2 and 3 × 3 closed forms

export interface SymEig2 {
  l1: number; l2: number;         // eigenvalues, l1 ≥ l2
  v1: [number, number];           // unit eigenvector of l1, (cos θ, sin θ)
  v2: [number, number];           // unit eigenvector of l2, perpendicular
  angle: number;                  // θ of v1 in (-π/2, π/2]
}

/** Eigen-decomposition of the symmetric matrix [[a, b], [b, c]] (closed form). */
export function eigSym2(a: number, b: number, c: number): SymEig2 {
  const m = (a + c) / 2, d = Math.hypot((a - c) / 2, b);
  const angle = 0.5 * Math.atan2(2 * b, a - c);
  const cs = Math.cos(angle), sn = Math.sin(angle);
  return { l1: m + d, l2: m - d, v1: [cs, sn], v2: [-sn, cs], angle };
}

/** Eigenvalues of a general 2 × 2 [[a, b], [c, d]]: real (im = 0, with unit eigenvectors) or a complex pair. */
export function eig2(a: number, b: number, c: number, d: number):
  { re: [number, number]; im: [number, number]; vectors: [[number, number], [number, number]] | null } {
  const tr = a + d, det = a * d - b * c, disc = tr * tr / 4 - det;
  if (disc < 0) { const s = Math.sqrt(-disc); return { re: [tr / 2, tr / 2], im: [s, -s], vectors: null }; }
  const s = Math.sqrt(disc), l1 = tr / 2 + s, l2 = tr / 2 - s;
  const vec = (l: number): [number, number] => {
    let x = b, y = l - a;                                  // from row 1: (a−l)x + b y = 0
    if (Math.abs(x) + Math.abs(y) < 1e-14 * (Math.abs(a) + Math.abs(b) + Math.abs(c) + Math.abs(d) + 1)) { x = l - d; y = c; }
    if (Math.abs(x) + Math.abs(y) === 0) return l === l1 ? [1, 0] : [0, 1];   // A = l·I
    const n = Math.hypot(x, y); return [x / n, y / n];
  };
  return { re: [l1, l2], im: [0, 0], vectors: [vec(l1), vec(l2)] };
}

/**
 * SVD of [[a, b], [c, d]] = U · diag(s1, s2) · Vᵀ in closed form, s1 ≥ s2 ≥ 0.
 * U and V are row-major 2 × 2 orthogonal matrices; for a 2D linear map, Vᵀ rotates,
 * diag stretches along the axes, U rotates (or reflects, when det < 0).
 */
export function svd2(a: number, b: number, c: number, d: number): { s1: number; s2: number; U: Float64Array; V: Float64Array } {
  const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
  const Q = Math.hypot(E, H), R = Math.hypot(F, G);
  let s1 = Q + R, s2 = Q - R;
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E);
  const th = (a2 - a1) / 2, ph = (a2 + a1) / 2;
  const U = new Float64Array([Math.cos(ph), -Math.sin(ph), Math.sin(ph), Math.cos(ph)]);
  // A = R(ph) diag(s1, s2) R(th)  ⇒  Vᵀ = R(th), V = R(−th)
  const V = new Float64Array([Math.cos(th), Math.sin(th), -Math.sin(th), Math.cos(th)]);
  if (s2 < 0) { s2 = -s2; U[1] = -U[1]; U[3] = -U[3]; }
  if (s1 < s2) { const t = s1; s1 = s2; s2 = t; }   // only when both ~0
  return { s1, s2, U, V };
}

export const det2 = (a: number, b: number, c: number, d: number): number => a * d - b * c;

/** Inverse of [[a, b], [c, d]] as [a', b', c', d'], or null when singular. */
export function inv2(a: number, b: number, c: number, d: number): Float64Array | null {
  const D = a * d - b * c;
  return D === 0 ? null : new Float64Array([d / D, -b / D, -c / D, a / D]);
}

export function det3(A: M): number {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

/** Inverse of a 3 × 3 matrix (adjugate / determinant), or null when singular. */
export function inv3(A: M): Float64Array | null {
  const D = det3(A);
  if (D === 0) return null;
  const c = (r0: number, c0: number, r1: number, c1: number) => A[r0 * 3 + c0] * A[r1 * 3 + c1] - A[r0 * 3 + c1] * A[r1 * 3 + c0];
  return new Float64Array([
    c(1, 1, 2, 2) / D, -c(0, 1, 2, 2) / D, c(0, 1, 1, 2) / D,
    -c(1, 0, 2, 2) / D, c(0, 0, 2, 2) / D, -c(0, 0, 1, 2) / D,
    c(1, 0, 2, 1) / D, -c(0, 0, 2, 1) / D, c(0, 0, 1, 1) / D,
  ]);
}

export function cross3(a: M, b: M): Float64Array {
  return new Float64Array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
}

// ---------------------------------------------------------------------------------------
// Symmetric eigen-decomposition

/**
 * Eigen-decomposition of a dense symmetric n × n matrix by cyclic Jacobi rotations.
 * Eigenvalues ascending (or descending with `descending`, as PCA wants), eigenvectors as
 * the COLUMNS of `vectors` (row-major n × n), so column k belongs to values[k].
 * Same algorithm and output as ncut/spectral.ts jacobiEigen.
 */
export function jacobiEigen(Ain: M, n: number, descending = false): { values: Float64Array; vectors: Float64Array } {
  const A = Float64Array.from(Ain);
  const V = identity(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = A[i * n + j] ** 2; tot += v; if (i !== j) off += v; }
    if (off <= 1e-26 * Math.max(tot, 1e-300)) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => descending ? A[j * n + j] - A[i * n + i] : A[i * n + i] - A[j * n + j]);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  order.forEach((src, k) => {
    values[k] = A[src * n + src];
    for (let r = 0; r < n; r++) vectors[r * n + k] = V[r * n + src];
  });
  return { values, vectors };
}

/** Column k of a row-major matrix with `cols` columns and `rows` rows. */
export function column(A: M, rows: number, cols: number, k: number): Float64Array {
  const v = new Float64Array(rows);
  for (let r = 0; r < rows; r++) v[r] = A[r * cols + k];
  return v;
}

// ---------------------------------------------------------------------------------------
// Singular value decomposition

export interface SVD {
  /** m × n, columns are the left singular vectors (zero columns where s = 0). */
  U: Float64Array;
  /** n singular values, descending (for m < n the last n − m are 0). */
  S: Float64Array;
  /** n × n orthogonal, columns are the right singular vectors (incl. a null-space basis). */
  V: Float64Array;
}

/** A = U diag(S) Vᵀ for an m × n matrix, by one-sided Jacobi (Hestenes): accurate for small matrices. */
export function svd(A: M, m: number, n: number): SVD {
  const U = Float64Array.from(A), V = identity(n);
  for (let sweep = 0; sweep < 80; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let al = 0, be = 0, ga = 0;
      for (let i = 0; i < m; i++) { const up = U[i * n + p], uq = U[i * n + q]; al += up * up; be += uq * uq; ga += up * uq; }
      if (Math.abs(ga) <= 1e-15 * Math.sqrt(al * be) || ga === 0) continue;
      rotated = true;
      const zeta = (be - al) / (2 * ga);
      const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), s = c * t;
      for (let i = 0; i < m; i++) {
        const up = U[i * n + p], uq = U[i * n + q];
        U[i * n + p] = c * up - s * uq; U[i * n + q] = s * up + c * uq;
      }
      for (let i = 0; i < n; i++) {
        const vp = V[i * n + p], vq = V[i * n + q];
        V[i * n + p] = c * vp - s * vq; V[i * n + q] = s * vp + c * vq;
      }
    }
    if (!rotated) break;
  }
  const sv = new Float64Array(n);
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += U[i * n + j] ** 2; sv[j] = Math.sqrt(s); }
  const order = [...Array(n).keys()].sort((a, b) => sv[b] - sv[a]);
  const smax = sv[order[0]] || 0;
  const Uo = new Float64Array(m * n), Vo = new Float64Array(n * n), S = new Float64Array(n);
  order.forEach((src, k) => {
    const s = sv[src] > 1e-300 && sv[src] > 1e-15 * smax ? sv[src] : 0;
    S[k] = s;
    for (let i = 0; i < m; i++) Uo[i * n + k] = s > 0 ? U[i * n + src] / sv[src] : 0;
    for (let i = 0; i < n; i++) Vo[i * n + k] = V[i * n + src];
  });
  return { U: Uo, S, V: Vo };
}

/** Default tolerance: singular values below rcond · s_max count as zero. */
const RCOND = 1e-12;

/** Moore–Penrose pseudo-inverse A⁺ (n × m) of an m × n matrix. */
export function pinv(A: M, m: number, n: number, rcond = RCOND): Float64Array {
  const { U, S, V } = svd(A, m, n), P = new Float64Array(n * m), cut = rcond * S[0];
  for (let k = 0; k < n; k++) {
    if (!(S[k] > cut)) continue;
    const inv = 1 / S[k];
    for (let i = 0; i < n; i++) { const v = V[i * n + k] * inv; if (v === 0) continue; for (let j = 0; j < m; j++) P[i * m + j] += v * U[j * n + k]; }
  }
  return P;
}

/** Least-squares solution of A x ≈ b (minimum-norm when rank deficient), via the SVD. */
export function lstsq(A: M, b: M, m: number, n: number, rcond = RCOND): Float64Array {
  return matVec(pinv(A, m, n, rcond), b, n, m);
}

/** 2-norm condition number s_max / s_min (Infinity if singular). For non-square: over min(m, n) values. */
export function cond(A: M, m: number, n: number): number {
  const S = svd(A, m, n).S, smin = S[Math.min(m, n) - 1];
  return smin > 0 ? S[0] / smin : Infinity;
}

/** Numerical rank: singular values above rcond · s_max. */
export function rank(A: M, m: number, n: number, rcond = 1e-10): number {
  const S = svd(A, m, n).S;
  let r = 0;
  for (let k = 0; k < S.length; k++) if (S[k] > rcond * S[0]) r++;
  return r;
}

/** Orthonormal basis of the null space {x : A x = 0} as a list of vectors of length n. */
export function nullSpace(A: M, m: number, n: number, rcond = 1e-10): Float64Array[] {
  const { S, V } = svd(A, m, n), out: Float64Array[] = [];
  for (let k = 0; k < n; k++) if (!(S[k] > rcond * S[0])) out.push(column(V, n, n, k));
  return out;
}

// ---------------------------------------------------------------------------------------
// LU with partial pivoting

export interface LU { LU: Float64Array; perm: Int32Array; sign: number; n: number }

/** PA = LU (L unit lower, U upper, packed in one matrix), or null when singular. */
export function lu(A: M, n: number): LU | null {
  const a = Float64Array.from(A), perm = new Int32Array(n);
  let sign = 1;
  for (let i = 0; i < n; i++) perm[i] = i;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(a[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(a[i * n + k]); if (v > best) { best = v; p = i; } }
    if (best === 0) return null;
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
      const t = perm[k]; perm[k] = perm[p]; perm[p] = t; sign = -sign;
    }
    const piv = a[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = (a[i * n + k] /= piv);
      if (f !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j];
    }
  }
  return { LU: a, perm, sign, n };
}

/** Solve with a factorisation from `lu`. */
export function luSolve(f: LU, b: M): Float64Array {
  const { LU: a, perm, n } = f, x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[perm[i]]; for (let j = 0; j < i; j++) s -= a[i * n + j] * x[j]; x[i] = s; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let j = i + 1; j < n; j++) s -= a[i * n + j] * x[j]; x[i] = s / a[i * n + i]; }
  return x;
}

/** Solve A x = b for square A, or null when A is singular. */
export function solve(A: M, b: M, n: number): Float64Array | null {
  const f = lu(A, n);
  return f ? luSolve(f, b) : null;
}

/** A⁻¹, or null when singular. */
export function inverse(A: M, n: number): Float64Array | null {
  const f = lu(A, n);
  if (!f) return null;
  const inv = new Float64Array(n * n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    const x = luSolve(f, e);
    for (let i = 0; i < n; i++) inv[i * n + j] = x[i];
  }
  return inv;
}

export function det(A: M, n: number): number {
  const f = lu(A, n);
  if (!f) return 0;
  let d = f.sign;
  for (let i = 0; i < n; i++) d *= f.LU[i * n + i];
  return d;
}

// ---------------------------------------------------------------------------------------
// Cholesky

/** A = L Lᵀ for symmetric positive-definite A; L lower triangular (row-major), or null if not SPD. */
export function cholesky(A: M, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (!(s > 0)) return null; L[i * n + i] = Math.sqrt(s); }
    else L[i * n + j] = s / L[j * n + j];
  }
  return L;
}

/** Solve A x = b given the Cholesky factor L of A. */
export function choleskySolve(L: M, b: M, n: number): Float64Array {
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

/** log det A from its Cholesky factor (for Gaussian log-likelihoods). */
export function choleskyLogDet(L: M, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.log(L[i * n + i]);
  return 2 * s;
}
