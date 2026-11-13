// Nonlinear least squares and bundle adjustment (no DOM). Owner: the geometry-from-images day
// (sfm/; cameras/ uses the same solver to refine a calibration). Users: bundleadjust demo,
// dltcalib demo (refinement), tests.
//
//  - A least-squares problem gives residuals r(θ) and their Jacobian J (dense, row-major;
//    small sizes only). Gauss–Newton solves JᵀWJ δ = −JᵀW r; Levenberg–Marquardt adds μ·diag
//    (Marquardt's scaling): (JᵀWJ + μ D) δ = −JᵀW r, accepts the step if the cost drops
//    (then μ ← μ/3) and otherwise rejects it (μ ← 4μ). W are robust (Huber) weights by
//    iteratively reweighted least squares. Parameters can be held fixed (the gauge).
//  - The Schur complement: with parameters ordered [cameras | points] and the point blocks
//    of JᵀJ block-diagonal, (U − W V⁻¹ Wᵀ) δc = gc − W V⁻¹ gp, then δp = V⁻¹ (gp − Wᵀ δc).
//  - Flatland bundle adjustment: cameras (x, y, θ) with known f, c; points (X, Y); one
//    residual per observation u_ij − π(cam_j, X_i), analytic Jacobian.

import { solve, inverse, jacobiEigen, lstsq } from "./linalg";
import { Cam1D, P, project1d, duDp, toCam } from "./flatland";

export interface LSProblem {
  nParams: number;
  /** Residuals and Jacobian (m × nParams, row-major) at θ. */
  evaluate(theta: Float64Array, wantJ: boolean): { r: Float64Array; J?: Float64Array };
}

export type Loss = "squared" | "huber";

/** Robust weights: 1 for squared loss; Huber: 1 inside ±delta, delta/|r| outside. */
export function weights(r: ArrayLike<number>, loss: Loss, delta: number): Float64Array {
  return Float64Array.from(r, v => (loss === "huber" && Math.abs(v) > delta ? delta / Math.abs(v) : 1));
}

/** Cost Σ ρ(r): squared r², Huber r² inside ±delta and 2 delta |r| − delta² outside. */
export function robustCost(r: ArrayLike<number>, loss: Loss, delta: number): number {
  let c = 0;
  for (let i = 0; i < r.length; i++) {
    const a = Math.abs(r[i]);
    c += loss === "huber" && a > delta ? 2 * delta * a - delta * delta : a * a;
  }
  return c;
}

/** Normal equations A = JᵀWJ and g = −JᵀW r over the free parameters. */
export function normalEquations(J: ArrayLike<number>, r: ArrayLike<number>, w: ArrayLike<number>, m: number, n: number, free: number[]): { A: Float64Array; g: Float64Array } {
  const k = free.length, A = new Float64Array(k * k), g = new Float64Array(k);
  for (let i = 0; i < m; i++) {
    const wi = w[i], row = i * n;
    for (let a = 0; a < k; a++) {
      const ja = J[row + free[a]];
      if (ja === 0) continue;
      g[a] -= wi * ja * r[i];
      for (let b = a; b < k; b++) { const jb = J[row + free[b]]; if (jb !== 0) A[a * k + b] += wi * ja * jb; }
    }
  }
  for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) A[a * k + b] = A[b * k + a];
  return { A, g };
}

export interface StepInfo {
  theta: Float64Array;
  cost: number;
  costBefore: number;
  accepted: boolean;
  mu: number;
  stepNorm: number;
}

/**
 * One Gauss–Newton (mu = 0) or Levenberg–Marquardt step. GN uses the minimum-norm solution
 * (so it also runs with a free gauge); LM solves the damped system. Returns the new θ (equal
 * to the old one if an LM step is rejected) and the updated μ.
 */
export function lmStep(prob: LSProblem, theta: Float64Array, o: { mu: number; lm: boolean; fixed?: boolean[]; loss?: Loss; delta?: number }): StepInfo {
  const loss = o.loss ?? "squared", delta = o.delta ?? 1;
  const { r, J } = prob.evaluate(theta, true);
  const m = r.length, n = prob.nParams;
  const free = Array.from({ length: n }, (_, i) => i).filter(i => !o.fixed?.[i]);
  const w = weights(r, loss, delta);
  const costBefore = robustCost(r, loss, delta);
  const { A, g } = normalEquations(J!, r, w, m, n, free);
  const k = free.length;
  let delta_: Float64Array | null;
  if (o.lm) {
    const Ad = Float64Array.from(A);
    for (let a = 0; a < k; a++) Ad[a * k + a] += o.mu * (A[a * k + a] + 1e-9);
    delta_ = solve(Ad, g, k);
  } else delta_ = lstsq(A, g, k, k, 1e-12);
  if (!delta_) return { theta, cost: costBefore, costBefore, accepted: false, mu: o.mu * 4, stepNorm: 0 };
  const next = Float64Array.from(theta);
  free.forEach((p, a) => { next[p] += delta_![a]; });
  const cost = robustCost(prob.evaluate(next, false).r, loss, delta);
  const stepNorm = Math.hypot(...delta_);
  if (!o.lm) return { theta: next, cost, costBefore, accepted: true, mu: 0, stepNorm };
  if (Number.isFinite(cost) && cost < costBefore) return { theta: next, cost, costBefore, accepted: true, mu: Math.max(1e-7, o.mu / 3), stepNorm };
  return { theta, cost: costBefore, costBefore, accepted: false, mu: Math.min(1e8, o.mu * 4), stepNorm };
}

/** Run LM until the relative cost change is below tol or maxIter steps; returns the cost history. */
export function levenbergMarquardt(prob: LSProblem, theta0: Float64Array, o: { mu?: number; maxIter?: number; tol?: number; fixed?: boolean[]; loss?: Loss; delta?: number } = {}): { theta: Float64Array; costs: number[]; mu: number } {
  let theta: Float64Array = Float64Array.from(theta0), mu = o.mu ?? 1e-3;
  const costs = [robustCost(prob.evaluate(theta, false).r, o.loss ?? "squared", o.delta ?? 1)];
  for (let it = 0; it < (o.maxIter ?? 50); it++) {
    const s = lmStep(prob, theta, { mu, lm: true, fixed: o.fixed, loss: o.loss, delta: o.delta });
    mu = s.mu;
    if (s.accepted) {
      theta = s.theta;
      costs.push(s.cost);
      if (s.costBefore - s.cost < (o.tol ?? 1e-12) * Math.max(1, s.costBefore)) break;
    } else if (mu >= 1e8) break;
  }
  return { theta, costs, mu };
}

/**
 * Solve A δ = g with A = [[U, W], [Wᵀ, V]] where the first nc unknowns are camera
 * parameters and V (the rest) is block-diagonal with blocks of size pb (points).
 * Returns δ and the reduced camera matrix S = U − W V⁻¹ Wᵀ.
 */
export function schurSolve(A: ArrayLike<number>, g: ArrayLike<number>, n: number, nc: number, pb: number): { delta: Float64Array; S: Float64Array } | null {
  const np = n - nc, nb = np / pb;
  const at = (i: number, j: number) => A[i * n + j];
  // V⁻¹ block by block
  const Vinv: Float64Array[] = [];
  for (let b = 0; b < nb; b++) {
    const o = nc + b * pb, B = new Float64Array(pb * pb);
    for (let i = 0; i < pb; i++) for (let j = 0; j < pb; j++) B[i * pb + j] = at(o + i, o + j);
    const Bi = inverse(B, pb);
    if (!Bi) return null;
    Vinv.push(Bi);
  }
  // S = U − Σ_b W_b V_b⁻¹ W_bᵀ, rhs = gc − Σ_b W_b V_b⁻¹ g_b
  const S = new Float64Array(nc * nc), rhs = new Float64Array(nc);
  for (let i = 0; i < nc; i++) { rhs[i] = g[i]; for (let j = 0; j < nc; j++) S[i * nc + j] = at(i, j); }
  for (let b = 0; b < nb; b++) {
    const o = nc + b * pb, Vi = Vinv[b];
    // WV = W_b V_b⁻¹ (nc × pb)
    const WV = new Float64Array(nc * pb);
    for (let i = 0; i < nc; i++) for (let k = 0; k < pb; k++) { let s = 0; for (let l = 0; l < pb; l++) s += at(i, o + l) * Vi[l * pb + k]; WV[i * pb + k] = s; }
    for (let i = 0; i < nc; i++) {
      for (let j = 0; j < nc; j++) { let s = 0; for (let k = 0; k < pb; k++) s += WV[i * pb + k] * at(j, o + k); S[i * nc + j] -= s; }
      let s = 0; for (let k = 0; k < pb; k++) s += WV[i * pb + k] * g[o + k]; rhs[i] -= s;
    }
  }
  const dc = solve(S, rhs, nc);
  if (!dc) return null;
  const delta = new Float64Array(n);
  delta.set(dc);
  for (let b = 0; b < nb; b++) {
    const o = nc + b * pb, Vi = Vinv[b], tmp = new Float64Array(pb);
    for (let k = 0; k < pb; k++) { let s = g[o + k]; for (let i = 0; i < nc; i++) s -= at(i, o + k) * dc[i]; tmp[k] = s; }
    for (let k = 0; k < pb; k++) { let s = 0; for (let l = 0; l < pb; l++) s += Vi[k * pb + l] * tmp[l]; delta[o + k] = s; }
  }
  return { delta, S };
}

/** Eigenvalues of JᵀJ (ascending) over the free parameters. */
export function normalEigenvalues(prob: LSProblem, theta: Float64Array, fixed?: boolean[]): Float64Array {
  const { r, J } = prob.evaluate(theta, true), n = prob.nParams;
  const free = Array.from({ length: n }, (_, i) => i).filter(i => !fixed?.[i]);
  const { A } = normalEquations(J!, r, new Float64Array(r.length).fill(1), r.length, n, free);
  return jacobiEigen(A, free.length).values.slice().sort();
}

// ---------------------------------------------------------------------------------------
// Flatland bundle adjustment

export interface Obs { cam: number; pt: number; u: number }

/** Parameters: [x, y, θ] per camera, then [X, Y] per point. f, c, n come from `base`. */
export class FlatBA implements LSProblem {
  readonly nParams: number;
  constructor(readonly base: Cam1D[], readonly nPts: number, readonly obs: Obs[]) {
    this.nParams = 3 * base.length + 2 * nPts;
  }
  static pack(cams: Cam1D[], pts: P[]): Float64Array {
    const t = new Float64Array(3 * cams.length + 2 * pts.length);
    cams.forEach((c, j) => t.set([c.x, c.y, c.theta], 3 * j));
    pts.forEach((p, i) => t.set(p, 3 * cams.length + 2 * i));
    return t;
  }
  cams(theta: ArrayLike<number>): Cam1D[] {
    return this.base.map((c, j) => ({ ...c, x: theta[3 * j], y: theta[3 * j + 1], theta: theta[3 * j + 2] }));
  }
  pts(theta: ArrayLike<number>): P[] {
    const o = 3 * this.base.length;
    return Array.from({ length: this.nPts }, (_, i) => [theta[o + 2 * i], theta[o + 2 * i + 1]] as P);
  }
  evaluate(theta: Float64Array, wantJ: boolean): { r: Float64Array; J?: Float64Array } {
    const cams = this.cams(theta), pts = this.pts(theta), m = this.obs.length, n = this.nParams, nc = 3 * this.base.length;
    const r = new Float64Array(m), J = wantJ ? new Float64Array(m * n) : undefined;
    this.obs.forEach((o, i) => {
      const cam = cams[o.cam], p = pts[o.pt], pr = project1d(cam, p);
      r[i] = pr.Z > 1e-9 ? o.u - pr.u : 1e3;
      if (!J) return;
      const g = duDp(cam, p), [X, Z] = toCam(cam, p);
      // residual = u_obs − π: derivatives carry a minus sign
      J[i * n + 3 * o.cam] = g[0]; J[i * n + 3 * o.cam + 1] = g[1];
      J[i * n + 3 * o.cam + 2] = -cam.f * (Z * Z + X * X) / (Z * Z);
      J[i * n + nc + 2 * o.pt] = -g[0]; J[i * n + nc + 2 * o.pt + 1] = -g[1];
    });
    return { r, J };
  }
}

/**
 * Gauge fixing for flatland: hold camera `a` completely, and coordinate `coord` (0 = x,
 * 1 = y) of camera `b` (fixes the scale). Choose the coordinate along which the two cameras
 * are far apart: a coordinate nearly perpendicular to the baseline pins the scale only weakly.
 */
export function gaugeMask(nCams: number, nPts: number, a = 0, b = 1, fix = true, coord: 0 | 1 = 0): boolean[] {
  const m = new Array(3 * nCams + 2 * nPts).fill(false);
  if (!fix) return m;
  m[3 * a] = m[3 * a + 1] = m[3 * a + 2] = true;
  m[3 * b + coord] = true;
  return m;
}

/** Covariances (2 × 2, as [sxx, sxy, syy]) of the points: σ² (JᵀJ)⁻¹ over the free parameters. */
export function pointCovariances(prob: FlatBA, theta: Float64Array, fixed: boolean[], sigma: number): ([number, number, number] | null)[] {
  const { r, J } = prob.evaluate(theta, true), n = prob.nParams;
  const free = Array.from({ length: n }, (_, i) => i).filter(i => !fixed[i]);
  const { A } = normalEquations(J!, r, new Float64Array(r.length).fill(1), r.length, n, free);
  const Ai = inverse(A, free.length);
  const nc = 3 * prob.base.length;
  return Array.from({ length: prob.nPts }, (_, i) => {
    if (!Ai) return null;
    const a = free.indexOf(nc + 2 * i), b = free.indexOf(nc + 2 * i + 1), k = free.length;
    if (a < 0 || b < 0) return null;
    const s2 = sigma * sigma;
    return [s2 * Ai[a * k + a], s2 * Ai[a * k + b], s2 * Ai[b * k + b]];
  });
}
