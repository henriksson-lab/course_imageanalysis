// Nonlinear fitting of small parameter vectors (no DOM): Levenberg–Marquardt with a choice of
// cost, written for decay fitting on the FLIM day and meant for reuse (Gaussian spot fitting).
//
// Costs (y data, μ model):
//   "ls"        Σ (y − μ)²                          plain least squares
//   "weighted"  Σ w (y − μ)²                         fixed weights
//   "neyman"    Σ (y − μ)² / max(y, 1)               classic χ² with data weights (empty bins weight 1)
//   "poisson"   2 Σ [y ln(y/μ) − (y − μ)]            Poisson deviance (ML); IRLS weights 1/μ
// One LM step solves (JᵀWJ + λ diag JᵀWJ) δ = JᵀW (y − μ). Parameters can be held fixed,
// kept positive (fitted as log p), and clamped to bounds. The trace records every accepted
// step (for drawing the path on a cost landscape). The covariance at the optimum is
// (JᵀWJ)⁻¹, scaled by the residual variance for "ls" and "weighted".
// Variable projection helper: linearAmplitudes() solves for non-negative linear amplitudes
// of fixed basis curves under any of the costs (global analysis: shared lifetimes, per-pixel
// amplitudes).

import { solve, inverse } from "./linalg";
import { nnlsGram } from "./unmix";

type A = ArrayLike<number>;
export type Cost = "ls" | "weighted" | "neyman" | "poisson";

/** Value of a cost for data y and model mu. */
export function costValue(y: A, mu: A, cost: Cost, w?: A): number {
  let s = 0;
  for (let j = 0; j < y.length; j++) {
    const r = y[j] - mu[j];
    switch (cost) {
      case "ls": s += r * r; break;
      case "weighted": s += (w ? w[j] : 1) * r * r; break;
      case "neyman": s += r * r / Math.max(y[j], 1); break;
      case "poisson":
        if (!(mu[j] > 0)) { if (y[j] > 0 || mu[j] < 0) return Infinity; break; }
        s += 2 * ((y[j] > 0 ? y[j] * Math.log(y[j] / mu[j]) : 0) - r);
        break;
    }
  }
  return s;
}

/** The weights of one Gauss–Newton step for a cost. */
export function costWeights(y: A, mu: A, cost: Cost, w?: A): Float64Array {
  return Float64Array.from({ length: y.length }, (_, j) => {
    switch (cost) {
      case "ls": return 1;
      case "weighted": return w ? w[j] : 1;
      case "neyman": return 1 / Math.max(y[j], 1);
      case "poisson": return 1 / Math.max(mu[j], 1e-9);
    }
  });
}

export interface FitOptions {
  cost?: Cost;
  weights?: A;
  fixed?: boolean[];
  /** fitted on a log scale (stays > 0) */
  positive?: boolean[];
  lower?: number[];
  upper?: number[];
  maxIter?: number;
  /** initial damping (default 1e-3) */
  lambda?: number;
  /** stop when the relative decrease of the cost is below this (default 1e-10) */
  tol?: number;
}

export interface FitStep { p: Float64Array; cost: number; lambda: number }

export interface FitResult {
  p: Float64Array;
  cost: number;
  iterations: number;
  converged: boolean;
  trace: FitStep[];
  /** covariance of all parameters (fixed ones: rows/columns of 0) */
  cov: Float64Array | null;
  /** standard errors (sqrt of the diagonal) */
  se: Float64Array;
}

/** Levenberg–Marquardt, one step at a time (for demos), or run() to the end. */
export class LMFitter {
  readonly n: number;
  readonly m: number;
  p: Float64Array;
  cost: number;
  lambda: number;
  iterations = 0;
  converged = false;
  trace: FitStep[] = [];
  private free: number[];
  private o: FitOptions;
  private c: Cost;

  constructor(readonly model: (p: Float64Array) => Float64Array, readonly y: A, p0: A, opts: FitOptions = {}) {
    this.o = opts; this.c = opts.cost ?? "ls";
    this.n = y.length; this.m = p0.length;
    this.p = this.clamp(Float64Array.from(p0));
    this.free = [];
    for (let i = 0; i < this.m; i++) if (!opts.fixed?.[i]) this.free.push(i);
    this.lambda = opts.lambda ?? 1e-3;
    this.cost = costValue(y, model(this.p), this.c, opts.weights);
    this.trace.push({ p: this.p.slice(), cost: this.cost, lambda: this.lambda });
  }

  private clamp(p: Float64Array): Float64Array {
    const o = this.o;
    for (let i = 0; i < p.length; i++) {
      if (o.lower?.[i] !== undefined) p[i] = Math.max(o.lower[i], p[i]);
      if (o.upper?.[i] !== undefined) p[i] = Math.min(o.upper[i], p[i]);
      if (o.positive?.[i]) p[i] = Math.max(p[i], 1e-12);
    }
    return p;
  }

  /** Internal coordinate of parameter i (log for positive ones) and back. */
  private toQ(i: number, v: number): number { return this.o.positive?.[i] ? Math.log(v) : v; }
  private fromQ(i: number, q: number): number { return this.o.positive?.[i] ? Math.exp(q) : q; }

  /** Jacobian of the model in internal coordinates of the free parameters (n × k, row-major). */
  private jacobian(p: Float64Array, mu: Float64Array, internal = true): Float64Array {
    const k = this.free.length, J = new Float64Array(this.n * k);
    this.free.forEach((i, c) => {
      const q = internal ? this.toQ(i, p[i]) : p[i];
      const h = 1e-6 * Math.max(1, Math.abs(q));
      const pp = p.slice();
      pp[i] = internal ? this.fromQ(i, q + h) : q + h;
      const m2 = this.model(pp);
      for (let j = 0; j < this.n; j++) J[j * k + c] = (m2[j] - mu[j]) / h;
    });
    return J;
  }

  private normal(J: Float64Array, W: Float64Array, r: Float64Array): { H: Float64Array; b: Float64Array } {
    const k = this.free.length, H = new Float64Array(k * k), b = new Float64Array(k);
    for (let j = 0; j < this.n; j++) {
      const wj = W[j], rj = r[j], row = j * k;
      for (let a = 0; a < k; a++) {
        const ja = J[row + a] * wj;
        b[a] += ja * rj;
        for (let c = a; c < k; c++) H[a * k + c] += ja * J[row + c];
      }
    }
    for (let a = 0; a < k; a++) for (let c = 0; c < a; c++) H[a * k + c] = H[c * k + a];
    return { H, b };
  }

  /** One accepted LM step (with up to 12 damping increases). Returns false when done. */
  step(): boolean {
    if (this.converged || this.free.length === 0) { this.converged = true; return false; }
    const mu = this.model(this.p), k = this.free.length;
    const W = costWeights(this.y, mu, this.c, this.o.weights);
    const r = Float64Array.from({ length: this.n }, (_, j) => this.y[j] - mu[j]);
    const J = this.jacobian(this.p, mu);
    const { H, b } = this.normal(J, W, r);
    // active set: a parameter sitting on a bound whose descent direction points outwards
    // is held for this step (otherwise clamping wastes the step)
    const act: number[] = [];
    this.free.forEach((i, c) => {
      const lo = this.o.lower?.[i], hi = this.o.upper?.[i];
      const atLo = lo !== undefined && this.p[i] <= lo + 1e-12 && b[c] < 0;
      const atHi = hi !== undefined && this.p[i] >= hi - 1e-12 && b[c] > 0;
      if (!atLo && !atHi) act.push(c);
    });
    const ka = act.length;
    if (ka === 0) { this.converged = true; return false; }
    for (let tries = 0; tries < 12; tries++) {
      const Hd = new Float64Array(ka * ka), ba = new Float64Array(ka);
      act.forEach((a, u) => { ba[u] = b[a]; act.forEach((c, v) => { Hd[u * ka + v] = H[a * k + c]; }); Hd[u * ka + u] += this.lambda * Math.max(H[a * k + a], 1e-12); });
      const d = solve(Hd, ba, ka);
      if (!d || d.some(v => !Number.isFinite(v))) { this.lambda *= 4; continue; }
      const pn = this.p.slice();
      act.forEach((c, u) => { const i = this.free[c]; pn[i] = this.fromQ(i, this.toQ(i, this.p[i]) + d[u]); });
      this.clamp(pn);
      const cn = costValue(this.y, this.model(pn), this.c, this.o.weights);
      if (cn <= this.cost) {
        const rel = (this.cost - cn) / Math.max(1e-300, Math.abs(this.cost));
        this.p = pn; this.cost = cn; this.lambda = Math.max(1e-9, this.lambda / 3);
        this.iterations++;
        this.trace.push({ p: pn.slice(), cost: cn, lambda: this.lambda });
        if (rel < (this.o.tol ?? 1e-10)) this.converged = true;
        return !this.converged;
      }
      this.lambda *= 4;
    }
    this.converged = true;
    return false;
  }

  run(maxIter = this.o.maxIter ?? 100): FitResult {
    while (this.iterations < maxIter && this.step()) { /* iterate */ }
    return this.result();
  }

  /** Covariance (JᵀWJ)⁻¹ in the original parameters at the current point. */
  result(): FitResult {
    const mu = this.model(this.p), k = this.free.length, m = this.m;
    const W = costWeights(this.y, mu, this.c, this.o.weights);
    let cov: Float64Array | null = null;
    const se = new Float64Array(m).fill(0);
    if (k > 0) {
      const J = this.jacobian(this.p, mu, false);
      const { H } = this.normal(J, W, new Float64Array(this.n));
      const Hi = inverse(H, k);
      if (Hi) {
        const scale = this.c === "ls" || this.c === "weighted" ? this.cost / Math.max(1, this.n - k) : 1;
        cov = new Float64Array(m * m);
        this.free.forEach((i, a) => this.free.forEach((j, c) => { cov![i * m + j] = scale * Hi[a * k + c]; }));
        for (let i = 0; i < m; i++) se[i] = Math.sqrt(Math.max(0, cov[i * m + i]));
      }
    }
    return { p: this.p.slice(), cost: this.cost, iterations: this.iterations, converged: this.converged, trace: this.trace, cov, se };
  }
}

/** Fit to the end. */
export function lmFit(model: (p: Float64Array) => Float64Array, y: A, p0: A, opts: FitOptions = {}): FitResult {
  return new LMFitter(model, y, p0, opts).run();
}

/**
 * Non-negative amplitudes a of K fixed basis curves (B: n × K row-major) plus a fixed offset,
 * μ = B a + offset, minimising the chosen cost: weighted NNLS on the normal equations,
 * re-weighted (1/μ) a few times for "poisson". Returns a and the model μ.
 */
export function linearAmplitudes(B: A, n: number, K: number, y: A, cost: Cost = "ls", offset: A | number = 0, iters = 6): { a: Float64Array; mu: Float64Array } {
  const off = (j: number) => (typeof offset === "number" ? offset : offset[j]);
  let W = costWeights(y, y, cost === "poisson" ? "neyman" : cost);
  let a: Float64Array = new Float64Array(K);
  const mu = new Float64Array(n);
  const rounds = cost === "poisson" ? iters : 1;
  for (let it = 0; it < rounds; it++) {
    const G = new Float64Array(K * K), b = new Float64Array(K);
    for (let j = 0; j < n; j++) {
      const wj = W[j], yj = y[j] - off(j);
      for (let p = 0; p < K; p++) {
        const bp = B[j * K + p] * wj;
        b[p] += bp * yj;
        for (let q = 0; q < K; q++) G[p * K + q] += bp * B[j * K + q];
      }
    }
    a = nnlsGram(G, b, K);
    for (let j = 0; j < n; j++) { let s = off(j); for (let p = 0; p < K; p++) s += B[j * K + p] * a[p]; mu[j] = s; }
    if (cost === "poisson") W = costWeights(y, mu, "poisson");
  }
  return { a, mu };
}
