// Pure computations of the polybasis demo (no DOM), shared with the scratch scripts that
// compute the numbers quoted in the orthogonal-moments exercise.
// A signal on [−1, 1] sampled at NS midpoints; fits of degree ≤ N with monomials (normal
// equations = the Gram system, solved by Gaussian elimination in 64- or 32-bit arithmetic)
// and with Legendre polynomials (coefficients by quadrature, no system to solve).
import { legendre, unitCentres } from "../lib/moments";
import { rng } from "../lib/random";

export const NS = 400;
export const XS = unitCentres(NS);
export const DX = 2 / NS;

export type SignalId = "step" | "abs" | "bump" | "profile";
export const SIGNALS: { id: SignalId; label: string }[] = [
  { id: "step", label: "step" },
  { id: "abs", label: "|x|" },
  { id: "bump", label: "bump" },
  { id: "profile", label: "nucleus profile" },
];

export function signal(id: SignalId): Float64Array {
  const r = rng(3);
  return XS.map(x => {
    if (id === "step") return x > 0.2 ? 1 : 0;
    if (id === "abs") return Math.abs(x);
    if (id === "bump") return Math.exp(-((x - 0.3) ** 2) / 0.02);
    // a line across a nucleus: background 0.15, nucleus 0.55 between −0.5 and 0.6 with a
    // brighter envelope at both edges, mild noise
    const inside = 1 / (1 + Math.exp(-(x + 0.5) / 0.02)) * (1 / (1 + Math.exp((x - 0.6) / 0.02)));
    const rim = 0.35 * (Math.exp(-((x + 0.47) ** 2) / 0.0015) + Math.exp(-((x - 0.57) ** 2) / 0.0015));
    return 0.15 + 0.4 * inside + rim + 0.02 * r.norm();
  });
}

const f32 = Math.fround;

/** Gaussian elimination with partial pivoting; with `single`, every operation is rounded to
 *  32-bit floats (Math.fround). */
export function solveGauss(Ain: ArrayLike<number>, bin: ArrayLike<number>, n: number, single: boolean): Float64Array {
  const r = single ? f32 : (v: number) => v;
  const A = Float64Array.from(Ain, r), b = Float64Array.from(bin, r);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let k = c + 1; k < n; k++) if (Math.abs(A[k * n + c]) > Math.abs(A[piv * n + c])) piv = k;
    if (piv !== c) { for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; } const t = b[c]; b[c] = b[piv]; b[piv] = t; }
    const d = A[c * n + c];
    if (d === 0) continue;
    for (let k = c + 1; k < n; k++) {
      const fct = r(A[k * n + c] / d);
      for (let j = c; j < n; j++) A[k * n + j] = r(A[k * n + j] - r(fct * A[c * n + j]));
      b[k] = r(b[k] - r(fct * b[c]));
    }
  }
  const x = new Float64Array(n);
  for (let k = n - 1; k >= 0; k--) {
    let s = b[k];
    for (let j = k + 1; j < n; j++) s = r(s - r(A[k * n + j] * x[j]));
    x[k] = A[k * n + k] ? r(s / A[k * n + k]) : 0;
  }
  return x;
}

/** Monomials x^k at the samples. */
export function monomials(N: number): Float64Array[] {
  return Array.from({ length: N + 1 }, (_, k) => XS.map(x => x ** k));
}

/** Gram matrix ⟨b_j, b_k⟩ = Σ b_j b_k Δx of a set of sampled functions (row-major). */
export function gram(B: Float64Array[]): Float64Array {
  const n = B.length, G = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let k = j; k < n; k++) {
    let s = 0;
    for (let i = 0; i < NS; i++) s += B[j][i] * B[k][i];
    G[j * n + k] = G[k * n + j] = s * DX;
  }
  return G;
}

/** Least-squares fit by monomials of degree ≤ N via the Gram system G c = (⟨f, x^k⟩). */
export function fitMonomial(f: ArrayLike<number>, N: number, single = false): Float64Array {
  const B = monomials(N), G = gram(B);
  const rhs = B.map(b => { let s = 0; for (let i = 0; i < NS; i++) s += b[i] * f[i]; return s * DX; });
  const c = solveGauss(G, rhs, N + 1, single);
  const r = single ? f32 : (v: number) => v;
  return XS.map(x => { let s = 0, v = 1; for (let k = 0; k <= N; k++) { s = r(s + r(c[k] * v)); v = r(v * x); } return s; });
}

/** Fit by Legendre polynomials: a_k = (2k+1)/2 ⟨f, P_k⟩, f̂ = Σ a_k P_k (no system to solve). */
export function fitLegendre(f: ArrayLike<number>, N: number, single = false): Float64Array {
  const P = legendre(N, XS), r = single ? f32 : (v: number) => v;
  const a = P.map((p, k) => { let s = 0; for (let i = 0; i < NS; i++) s = r(s + r(p[i] * f[i])); return r(s * DX * (2 * k + 1) / 2); });
  return XS.map((_, i) => { let s = 0; for (let k = 0; k <= N; k++) s = r(s + r(a[k] * r(P[k][i]))); return s; });
}

/** Relative RMS error. */
export function relErr(a: ArrayLike<number>, f: ArrayLike<number>): number {
  let s = 0, t = 0;
  for (let i = 0; i < f.length; i++) { const d = a[i] - f[i]; s += Number.isFinite(d) ? d * d : 1e300; t += f[i] * f[i]; }
  return Math.sqrt(s / t);
}

/** Gram–Schmidt on 1, x, x², … with ⟨g, h⟩ = Σ g h Δx: the first `k` functions orthogonalised
 *  (each scaled so that its value at x = 1 is 1, as P_n(1) = 1), the rest still monomials.
 *  Also returns, for step k, the monomial x^k, its projections onto q_0..q_{k−1} and the residual. */
export function gramSchmidt(N: number, k: number): { set: Float64Array[]; step: { mono: Float64Array; proj: Float64Array[]; resid: Float64Array } | null } {
  const mon = monomials(N), Q: Float64Array[] = [];
  const ip = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < NS; i++) s += a[i] * b[i]; return s * DX; };
  let step: { mono: Float64Array; proj: Float64Array[]; resid: Float64Array } | null = null;
  for (let j = 0; j < Math.min(k, N + 1); j++) {
    const v = mon[j].slice(), proj: Float64Array[] = [];
    for (const q of Q) { const c = ip(mon[j], q) / ip(q, q); proj.push(q.map(t => c * t)); for (let i = 0; i < NS; i++) v[i] -= c * q[i]; }
    // value at x = 1 by extrapolating the last two samples (exact for the polynomial: evaluate
    // the same combination at x = 1)
    Q.push(v);
    if (j === k - 1) step = { mono: mon[j], proj, resid: v };
  }
  // scale each q so that q(1) = 1: recompute at x = 1 through the same projections
  const at1: number[] = [];
  for (let j = 0; j < Q.length; j++) {
    let v = 1;   // x^j at 1
    for (let l = 0; l < j; l++) v -= (ip(mon[j], Q[l]) / ip(Q[l], Q[l])) * at1[l];
    at1.push(v);
  }
  const set: Float64Array[] = Q.map((q, j) => q.map(t => t / at1[j]));
  if (step) { const s = at1[k - 1]; step = { mono: step.mono, proj: step.proj, resid: step.resid.map(t => t / s) }; }
  for (let j = Q.length; j <= N; j++) set.push(mon[j]);
  return { set, step };
}

/** Condition number of a symmetric positive (semi)definite matrix from its eigenvalues. */
export function condSym(G: Float64Array, n: number, jacobi: (A: Float64Array, n: number) => { values: Float64Array }): number {
  const ev = Array.from(jacobi(G, n).values).map(Math.abs);
  const mx = Math.max(...ev), mn = Math.min(...ev);
  return mn > 0 ? mx / mn : Infinity;
}
