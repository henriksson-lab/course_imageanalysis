// The computations of the lowrank demo (no DOM): weight changes ΔW, their singular value
// decomposition, the best rank-r approximation, its error and parameter count, and the real
// change of the fine-tuned toy network of the transfercurves demo.
import { svd, type SVD } from "../lib/linalg";
import { rng } from "../lib/ui";
import { cloneMlp, type Mlp } from "../lib/nn";
import { pretrained, makeSet, Run, subset, evaluate, oldTaskAccuracy, LOOK_A, type DataSet } from "../transfercurves/toy";

export interface Change { m: number; n: number; D: Float64Array; W0?: Float64Array }

/** A random m × n matrix with orthonormal-ish factors: Σ s_k u_k v_kᵀ (+ noise · Gaussian). */
export function syntheticChange(m: number, n: number, s: number[], noise: number, seed: number): Change {
  const r = rng(seed);
  const D = new Float64Array(m * n);
  // random orthonormal u_k, v_k by Gram–Schmidt
  const ortho = (len: number, count: number): Float64Array[] => {
    const out: Float64Array[] = [];
    for (let k = 0; k < count; k++) {
      const v = Float64Array.from({ length: len }, () => r.norm());
      for (const q of out) { let d = 0; for (let i = 0; i < len; i++) d += v[i] * q[i]; for (let i = 0; i < len; i++) v[i] -= d * q[i]; }
      let nn = 0; for (let i = 0; i < len; i++) nn += v[i] * v[i];
      nn = Math.sqrt(nn); for (let i = 0; i < len; i++) v[i] /= nn;
      out.push(v);
    }
    return out;
  };
  const U = ortho(m, s.length), V = ortho(n, s.length);
  for (let k = 0; k < s.length; k++) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) D[i * n + j] += s[k] * U[k][i] * V[k][j];
  if (noise > 0) for (let i = 0; i < m * n; i++) D[i] += noise * r.norm() / Math.sqrt(Math.max(m, n));
  return { m, n, D };
}

/** The best rank-r approximation U_r diag(S_r) V_rᵀ (Eckart–Young) from an SVD. */
export function truncate(S: SVD, m: number, n: number, r: number): Float64Array {
  const out = new Float64Array(m * n);
  const k = Math.min(r, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let v = 0;
    for (let q = 0; q < k; q++) v += S.U[i * n + q] * S.S[q] * S.V[j * n + q];
    out[i * n + j] = v;
  }
  return out;
}

/** Relative error ‖ΔW − ΔW_r‖_F / ‖ΔW‖_F from the singular values: sqrt(Σ_{k>r} s_k² / Σ s_k²). */
export function relError(S: ArrayLike<number>, r: number): number {
  let tot = 0, rest = 0;
  for (let k = 0; k < S.length; k++) { tot += S[k] * S[k]; if (k >= r) rest += S[k] * S[k]; }
  return tot > 0 ? Math.sqrt(rest / tot) : 0;
}

/** Numbers to store: full update m·n, low-rank update r·(m + n) (B is m × r, A is r × n). */
export const fullParams = (m: number, n: number) => m * n;
export const lowRankParams = (m: number, n: number, r: number) => r * (m + n);

/** The singular value decomposition of a change (for m < n, of the transpose's shape as is). */
export const decompose = (c: Change): SVD => svd(c.D, c.m, c.n);

// ---------------------------------------------------------------------------------------
// The real change: full fine-tuning of the pretrained toy network on 300 target labels.

export interface RealChange {
  net: Mlp;                 // the fine-tuned network
  layers: Change[];         // ΔW of layer 1 (32 × 100) and layer 2 (16 × 32), with W0
  svds: SVD[];
  val: DataSet;
  accFull: number;
  accByRank: number[];      // validation accuracy with both changes cut to rank r, r = 0..16
  oldByRank: number[];      // old-task accuracy with both changes cut to rank r
}

let real: RealChange | null = null;
export const REAL = { n: 300, lr: 0.003, steps: 300, maxRank: 16 };

export function realChange(): RealChange {
  if (real) return real;
  const pre = pretrained();
  const pool = makeSet("target", 300, 303, LOOK_A), val = makeSet("target", 300, 404, LOOK_A);
  const srcTest = makeSet("source", 400, 202);
  const run = new Run("full", subset(pool, REAL.n), val, REAL.lr);
  run.run(REAL.steps, 50);
  const dims: [number, number][] = [[32, 100], [16, 32]];
  const layers = dims.map(([m, n], l) => {
    const D = new Float64Array(m * n);
    for (let i = 0; i < m * n; i++) D[i] = run.net.W[l][i] - pre.W[l][i];
    return { m, n, D, W0: pre.W[l] };
  });
  const svds = layers.map(decompose);
  const accByRank: number[] = [], oldByRank: number[] = [];
  for (let r = 0; r <= REAL.maxRank; r++) {
    const net = withRank(run.net, layers, svds, r);
    accByRank.push(evaluate(net, val).acc);
    oldByRank.push(oldTaskAccuracy(net, srcTest));
  }
  real = { net: run.net, layers, svds, val, accFull: evaluate(run.net, val).acc, accByRank, oldByRank };
  return real;
}

/** The fine-tuned network with W0 + (ΔW cut to rank r) in layers 1 and 2 (the new head kept). */
export function withRank(net: Mlp, layers: Change[], svds: SVD[], r: number): Mlp {
  const out = cloneMlp(net);
  layers.forEach((c, l) => {
    const Dr = truncate(svds[l], c.m, c.n, r);
    for (let i = 0; i < c.m * c.n; i++) out.W[l][i] = c.W0![i] + Dr[i];
  });
  return out;
}
