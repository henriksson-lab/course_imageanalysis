// Support vector machines (owner: SVM day). No DOM; runs under node.
//
//  - trainSVM: the soft-margin dual  min ½ αᵀQα − Σα,  0 ≤ α_i ≤ C_i,  Σ α_i y_i = 0,
//    Q_ij = y_i y_j k(x_i, x_j), solved by SMO with the second-order working-set selection of
//    Fan, Chen & Lin (2005) (the method of the standard solvers). Per-sample C_i allows class
//    weights. Returns α, the offset b, the support vectors, w for the linear kernel function,
//    and the dual objective. Decision value f(x) = Σ α_i y_i k(x_i, x) + b; class = sign f.
//  - primalObjective / dualObjective: for checks (duality gap ≈ 0 at the optimum).
//  - trainLinearSubgradient: plain subgradient descent on ½‖w‖² + C Σ hinge (the "descent" view).
//  - Multi-class: one-vs-rest and one-vs-one wrappers (trainMulti).
//  - Standardisation (z-scores with training statistics), folds by group (by stroke) or at
//    random, cross-validated error and a (C, γ) grid search.
//  - SVM_CLASSIFIER: the lib/classifier.ts plug-in used by the pixelclass shell.
//
// Data are ROW-MAJOR here (X[i * d + j], sample i, feature j) except in SVM_CLASSIFIER, which
// receives the feature-major tables of lib/classifier.ts and converts them.
import { KernelSpec, kernelFn, KernelFn } from "./svmkernels";
import type { Classifier, Model, TrainingSet } from "./classifier";
import { paramValue } from "./classifier";
import { rng } from "./random";

export interface SVMOptions {
  kernel?: KernelSpec;
  /** Slack penalty C (default 1). */
  C?: number;
  /** Per-sample C_i (overrides C); e.g. C · weight_i for class weights. */
  Ci?: ArrayLike<number>;
  /** Stopping tolerance on the KKT violation (default 1e-3). */
  tol?: number;
  /** Maximum number of SMO steps (default 100 000 + 100 n). */
  maxIter?: number;
}

export interface SVMModel {
  kernel: KernelSpec;
  d: number;
  n: number;
  /** Dual variables of all n training samples. */
  alpha: Float64Array;
  /** Labels ±1 of the training samples. */
  y: Float64Array;
  /** Per-sample upper bounds C_i. */
  Ci: Float64Array;
  b: number;
  /** Indices of the support vectors (α_i > 0). */
  sv: Int32Array;
  /** α_i y_i of each support vector. */
  coef: Float64Array;
  /** Support vectors, row-major (sv.length × d). */
  svX: Float64Array;
  /** Normal vector (linear kernel function only, else null). */
  w: Float64Array | null;
  /** Dual objective Σα − ½ αᵀQα (to be maximised). */
  dual: number;
  iterations: number;
  converged: boolean;
}

const TAU = 1e-12;

/** Train a two-class SVM. y holds +1 / −1 (anything > 0 counts as +1). */
export function trainSVM(X: ArrayLike<number>, n: number, d: number, yIn: ArrayLike<number>, o: SVMOptions = {}): SVMModel {
  const spec = o.kernel ?? { type: "linear" };
  const kf = kernelFn(spec);
  const y = Float64Array.from({ length: n }, (_, i) => (yIn[i] > 0 ? 1 : -1));
  const Ci = Float64Array.from({ length: n }, (_, i) => (o.Ci ? o.Ci[i] : o.C ?? 1));
  const alpha = new Float64Array(n);
  const tol = o.tol ?? 1e-3, maxIter = o.maxIter ?? 100000 + 100 * n;
  const nPos = y.reduce((s, v) => s + (v > 0 ? 1 : 0), 0);

  // degenerate: one class only -> constant decision
  if (nPos === 0 || nPos === n) {
    return finish(X, n, d, spec, alpha, y, Ci, nPos === n ? 1 : -1, 0, 0, true);
  }

  // kernel matrix (full for n ≤ 1500, otherwise rows computed on demand with a small cache)
  const full = n <= 1500;
  const K = full ? new Float64Array(n * n) : null;
  if (K) for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = kf(X, i * d, X, j * d, d); K[i * n + j] = v; K[j * n + i] = v; }
  const cache = new Map<number, Float64Array>();
  const row = (i: number): ArrayLike<number> => {
    if (K) return K.subarray(i * n, (i + 1) * n);
    let r = cache.get(i);
    if (!r) {
      r = new Float64Array(n);
      for (let j = 0; j < n; j++) r[j] = kf(X, i * d, X, j * d, d);
      if (cache.size > 256) cache.delete(cache.keys().next().value as number);
      cache.set(i, r);
    }
    return r;
  };
  const QD = Float64Array.from({ length: n }, (_, i) => (K ? K[i * n + i] : kf(X, i * d, X, i * d, d)));
  const G = new Float64Array(n).fill(-1);
  const upper = (t: number) => alpha[t] >= Ci[t];
  const lower = (t: number) => alpha[t] <= 0;

  let iter = 0, converged = false;
  while (iter < maxIter) {
    // ---- working set (i, j), second-order selection
    let Gmax = -Infinity, Gmax2 = -Infinity, gi = -1, gj = -1, objMin = Infinity;
    for (let t = 0; t < n; t++) {
      if (y[t] > 0) { if (!upper(t) && -G[t] >= Gmax) { Gmax = -G[t]; gi = t; } }
      else if (!lower(t) && G[t] >= Gmax) { Gmax = G[t]; gi = t; }
    }
    if (gi < 0) { converged = true; break; }
    const Ki = row(gi);
    for (let t = 0; t < n; t++) {
      if (y[t] > 0) {
        if (!lower(t)) {
          const gd = Gmax + G[t];
          if (G[t] >= Gmax2) Gmax2 = G[t];
          if (gd > 0) {
            const quad = QD[gi] + QD[t] - 2 * Ki[t];
            const od = -(gd * gd) / (quad > 0 ? quad : TAU);
            if (od <= objMin) { gj = t; objMin = od; }
          }
        }
      } else if (!upper(t)) {
        const gd = Gmax - G[t];
        if (-G[t] >= Gmax2) Gmax2 = -G[t];
        if (gd > 0) {
          const quad = QD[gi] + QD[t] - 2 * Ki[t];
          const od = -(gd * gd) / (quad > 0 ? quad : TAU);
          if (od <= objMin) { gj = t; objMin = od; }
        }
      }
    }
    if (Gmax + Gmax2 < tol || gj < 0) { converged = true; break; }
    iter++;
    const i = gi, j = gj, Kj = row(j);
    const Qij = y[i] * y[j] * Ki[j];
    const ai0 = alpha[i], aj0 = alpha[j], Ci_ = Ci[i], Cj_ = Ci[j];
    let ai = ai0, aj = aj0;
    if (y[i] !== y[j]) {
      let quad = QD[i] + QD[j] + 2 * Qij; if (quad <= 0) quad = TAU;
      const delta = (-G[i] - G[j]) / quad, diff = ai - aj;
      ai += delta; aj += delta;
      if (diff > 0) { if (aj < 0) { aj = 0; ai = diff; } }
      else if (ai < 0) { ai = 0; aj = -diff; }
      if (diff > Ci_ - Cj_) { if (ai > Ci_) { ai = Ci_; aj = Ci_ - diff; } }
      else if (aj > Cj_) { aj = Cj_; ai = Cj_ + diff; }
    } else {
      let quad = QD[i] + QD[j] - 2 * Qij; if (quad <= 0) quad = TAU;
      const delta = (G[i] - G[j]) / quad, sum = ai + aj;
      ai -= delta; aj += delta;
      if (sum > Ci_) { if (ai > Ci_) { ai = Ci_; aj = sum - Ci_; } }
      else if (aj < 0) { aj = 0; ai = sum; }
      if (sum > Cj_) { if (aj > Cj_) { aj = Cj_; ai = sum - Cj_; } }
      else if (ai < 0) { ai = 0; aj = sum; }
    }
    alpha[i] = ai; alpha[j] = aj;
    const dai = ai - ai0, daj = aj - aj0;
    // G_t += Q_ti Δα_i + Q_tj Δα_j
    for (let t = 0; t < n; t++) G[t] += y[t] * (y[i] * Ki[t] * dai + y[j] * Kj[t] * daj);
  }

  // ---- offset: b = −ρ
  let ub = Infinity, lb = -Infinity, sumFree = 0, nFree = 0;
  for (let t = 0; t < n; t++) {
    const yG = y[t] * G[t];
    if (upper(t)) { if (y[t] < 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else if (lower(t)) { if (y[t] > 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else { nFree++; sumFree += yG; }
  }
  const rho = nFree > 0 ? sumFree / nFree : (ub + lb) / 2;
  let v = 0;
  for (let t = 0; t < n; t++) v += alpha[t] * (G[t] - 1);
  return finish(X, n, d, spec, alpha, y, Ci, -rho, -v / 2, iter, converged);
}

function finish(X: ArrayLike<number>, n: number, d: number, spec: KernelSpec, alpha: Float64Array, y: Float64Array, Ci: Float64Array,
  b: number, dual: number, iterations: number, converged: boolean): SVMModel {
  const svIdx: number[] = [];
  for (let i = 0; i < n; i++) if (alpha[i] > 0) svIdx.push(i);
  const sv = Int32Array.from(svIdx), coef = Float64Array.from(svIdx, i => alpha[i] * y[i]);
  const svX = new Float64Array(sv.length * d);
  svIdx.forEach((i, s) => { for (let j = 0; j < d; j++) svX[s * d + j] = X[i * d + j]; });
  let w: Float64Array | null = null;
  if (spec.type === "linear") {
    w = new Float64Array(d);
    for (let s = 0; s < sv.length; s++) for (let j = 0; j < d; j++) w[j] += coef[s] * svX[s * d + j];
  }
  return { kernel: spec, d, n, alpha, y, Ci, b, sv, coef, svX, w, dual, iterations, converged };
}

/** Decision value f(x) for a vector stored at offset `ix` of `x`. */
export function decision(m: SVMModel, x: ArrayLike<number>, ix = 0, kf?: KernelFn): number {
  const d = m.d;
  if (m.w) { let s = m.b; for (let j = 0; j < d; j++) s += m.w[j] * x[ix + j]; return s; }
  const f = kf ?? kernelFn(m.kernel);
  let s = m.b;
  for (let q = 0; q < m.coef.length; q++) s += m.coef[q] * f(m.svX, q * d, x, ix, d);
  return s;
}

/** Decision values of n row-major samples. */
export function decisionBatch(m: SVMModel, X: ArrayLike<number>, n: number): Float64Array {
  const f = kernelFn(m.kernel), out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = decision(m, X, i * m.d, f);
  return out;
}

/** Margin width 2/‖w‖ of a linear model (Infinity if w = 0). */
export function marginWidth(m: SVMModel): number {
  if (!m.w) return NaN;
  const nw = Math.hypot(...m.w);
  return nw > 0 ? 2 / nw : Infinity;
}

/** ½ αᵀQα of a model (= ½‖w‖² in feature space). */
export function halfNormSq(m: SVMModel): number {
  const f = kernelFn(m.kernel), d = m.d;
  let s = 0;
  for (let p = 0; p < m.coef.length; p++) for (let q = 0; q < m.coef.length; q++) s += m.coef[p] * m.coef[q] * f(m.svX, p * d, m.svX, q * d, d);
  return s / 2;
}

/** Primal objective ½‖w‖² + Σ C_i max(0, 1 − y_i f(x_i)) on the training data. */
export function primalObjective(m: SVMModel, X: ArrayLike<number>): number {
  const f = kernelFn(m.kernel);
  let s = halfNormSq(m);
  for (let i = 0; i < m.n; i++) s += m.Ci[i] * Math.max(0, 1 - m.y[i] * decision(m, X, i * m.d, f));
  return s;
}

/** Dual objective Σα − ½ αᵀQα of a model. */
export function dualObjective(m: SVMModel): number {
  let s = 0;
  for (let i = 0; i < m.n; i++) s += m.alpha[i];
  return s - halfNormSq(m);
}

/** Largest violation of the KKT conditions (0 at the exact optimum, in units of y f). */
export function kktViolation(m: SVMModel, X: ArrayLike<number>, eps = 1e-8): number {
  const f = kernelFn(m.kernel);
  let worst = 0, sumAy = 0;
  for (let i = 0; i < m.n; i++) {
    const yf = m.y[i] * decision(m, X, i * m.d, f), a = m.alpha[i];
    sumAy += a * m.y[i];
    if (a <= eps) worst = Math.max(worst, 1 - yf);                       // α = 0: y f ≥ 1
    else if (a >= m.Ci[i] - eps) worst = Math.max(worst, yf - 1);        // α = C: y f ≤ 1
    else worst = Math.max(worst, Math.abs(yf - 1));                      // free: y f = 1
  }
  return Math.max(worst, Math.abs(sumAy));
}

// ---------------------------------------------------------------------------------------
// Linear SVM by subgradient descent on the primal (the "walk downhill" view)

export interface DescentResult { w: Float64Array; b: number; objective: number[] }

/**
 * Minimise ½‖w‖² + C Σ max(0, 1 − y_i (w·x_i + b)) by full-batch subgradient descent with
 * step size lr / √t. Returns the final (w, b) and the objective after every step.
 */
export function trainLinearSubgradient(X: ArrayLike<number>, n: number, d: number, yIn: ArrayLike<number>, C: number, steps: number,
  lr = 0.5, init?: { w: ArrayLike<number>; b: number }): DescentResult {
  const w = Float64Array.from({ length: d }, (_, j) => init?.w[j] ?? 0);
  let b = init?.b ?? 0;
  const obj: number[] = [];
  const y = Float64Array.from({ length: n }, (_, i) => (yIn[i] > 0 ? 1 : -1));
  const objective = () => {
    let s = 0;
    for (let j = 0; j < d; j++) s += w[j] * w[j] / 2;
    for (let i = 0; i < n; i++) { let f = b; for (let j = 0; j < d; j++) f += w[j] * X[i * d + j]; s += C * Math.max(0, 1 - y[i] * f); }
    return s;
  };
  let best = { w: Float64Array.from(w), b, o: objective() };
  for (let t = 1; t <= steps; t++) {
    const gw = Float64Array.from(w), eta = lr / Math.sqrt(t);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      let f = b; for (let j = 0; j < d; j++) f += w[j] * X[i * d + j];
      if (y[i] * f < 1) { for (let j = 0; j < d; j++) gw[j] -= C * y[i] * X[i * d + j]; gb -= C * y[i]; }
    }
    for (let j = 0; j < d; j++) w[j] -= eta * gw[j];
    b -= eta * gb;
    const o = objective();
    obj.push(o);
    if (o < best.o) best = { w: Float64Array.from(w), b, o };
  }
  return { w: best.w, b: best.b, objective: obj };
}

// ---------------------------------------------------------------------------------------
// Losses (for the loss demo and the by-hand tables), as functions of the margin value m = y f

export const zeroOneLoss = (m: number): number => (m > 0 ? 0 : 1);
export const hingeLoss = (m: number): number => Math.max(0, 1 - m);
/** Logistic (cross-entropy) loss in bits: log2(1 + e^{−m}) (1 at m = 0, like the others). */
export const logisticLoss = (m: number): number => Math.log2(1 + Math.exp(-m));
export const squaredLoss = (m: number): number => (1 - m) * (1 - m);

// ---------------------------------------------------------------------------------------
// Multi-class

export type MultiMode = "ovr" | "ovo";

export interface MultiModel {
  k: number;
  d: number;
  mode: MultiMode;
  /** OvR: k models (class c vs rest); OvO: k(k−1)/2 models for pairs (a < b), class a = +1. */
  models: (SVMModel | null)[];
  pairs: [number, number][];
}

/** Train a k-class SVM by one-vs-rest or one-vs-one. `cls` holds classes 0..k−1; `w` optional sample weights. */
export function trainMulti(X: ArrayLike<number>, n: number, d: number, cls: ArrayLike<number>, k: number, mode: MultiMode, o: SVMOptions = {}, w?: ArrayLike<number>): MultiModel {
  const C = o.C ?? 1;
  const models: (SVMModel | null)[] = [], pairs: [number, number][] = [];
  if (mode === "ovr" || k === 2) {
    const list = k === 2 ? [1] : Array.from({ length: k }, (_, c) => c);
    for (const c of list) {
      const y = Float64Array.from({ length: n }, (_, i) => (cls[i] === c ? 1 : -1));
      const has = y.some(v => v > 0);
      models.push(has ? trainSVM(X, n, d, y, { ...o, Ci: Float64Array.from({ length: n }, (_, i) => C * (w ? w[i] : 1)) }) : null);
      pairs.push([c, -1]);
    }
    return { k, d, mode: k === 2 ? "ovr" : mode, models, pairs };
  }
  for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) if (cls[i] === a || cls[i] === b) idx.push(i);
    const hasA = idx.some(i => cls[i] === a), hasB = idx.some(i => cls[i] === b);
    pairs.push([a, b]);
    if (!hasA || !hasB) { models.push(null); continue; }
    const Xs = new Float64Array(idx.length * d), y = new Float64Array(idx.length), Ci = new Float64Array(idx.length);
    idx.forEach((i, r) => { for (let j = 0; j < d; j++) Xs[r * d + j] = X[i * d + j]; y[r] = cls[i] === a ? 1 : -1; Ci[r] = C * (w ? w[i] : 1); });
    models.push(trainSVM(Xs, idx.length, d, y, { ...o, Ci }));
  }
  return { k, d, mode, models, pairs };
}

/**
 * Class scores of one sample (length k, summing to 1) from the decision values `f` of the
 * sub-models: k = 2: [1 − σ(2f), σ(2f)]; OvR: softmax of 2 f_c (a missing class gets 0);
 * OvO: vote fractions, ties broken by the summed pairwise σ(2f).
 */
export function multiScores(mm: MultiModel, f: ArrayLike<number>, out: Float32Array | Float64Array, o = 0): void {
  const k = mm.k, sig = (v: number) => 1 / (1 + Math.exp(-2 * v));
  if (k === 2) {
    const m = mm.models[0];
    const p = m ? sig(f[0]) : 0.5;
    out[o] = 1 - p; out[o + 1] = p;
    return;
  }
  if (mm.mode === "ovr") {
    let mx = -Infinity;
    for (let c = 0; c < k; c++) if (mm.models[c]) mx = Math.max(mx, 2 * f[c]);
    let s = 0;
    for (let c = 0; c < k; c++) { const e = mm.models[c] ? Math.exp(2 * f[c] - mx) : 0; out[o + c] = e; s += e; }
    for (let c = 0; c < k; c++) out[o + c] = s > 0 ? out[o + c] / s : 1 / k;
    return;
  }
  for (let c = 0; c < k; c++) out[o + c] = 0;
  let total = 0;
  mm.pairs.forEach(([a, b], q) => {
    const m = mm.models[q];
    if (!m) return;
    const p = sig(f[q]);
    out[o + (f[q] > 0 ? a : b)] += 1;
    out[o + a] += 1e-3 * p; out[o + b] += 1e-3 * (1 - p);
    total += 1 + 1e-3;
  });
  for (let c = 0; c < k; c++) out[o + c] = total > 0 ? out[o + c] / total : 1 / k;
}

/** Decision values of every sub-model for one sample at offset ix of x. */
export function multiDecisions(mm: MultiModel, x: ArrayLike<number>, ix = 0, out?: Float64Array): Float64Array {
  const f = out ?? new Float64Array(mm.models.length);
  mm.models.forEach((m, q) => { f[q] = m ? decision(m, x, ix) : (mm.mode === "ovr" ? -Infinity : 0); });
  return f;
}

/** Predicted class of one sample (argmax of the scores). */
export function multiPredict(mm: MultiModel, x: ArrayLike<number>, ix = 0): number {
  const s = new Float64Array(mm.k);
  multiScores(mm, multiDecisions(mm, x, ix), s);
  let best = 0;
  for (let c = 1; c < mm.k; c++) if (s[c] > s[best]) best = c;
  return best;
}

// ---------------------------------------------------------------------------------------
// Standardisation and cross-validation

export interface Standardiser { mean: Float64Array; sd: Float64Array }

/** Mean and standard deviation of each feature over the (row-major) training samples. */
export function fitStandardiser(X: ArrayLike<number>, n: number, d: number): Standardiser {
  const mean = new Float64Array(d), sd = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) sd[j] += (X[i * d + j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return { mean, sd };
}

/** z-scores in place (row-major). */
export function applyStandardiser(s: Standardiser, X: Float64Array, n: number, d: number): Float64Array {
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = (X[i * d + j] - s.mean[j]) / s.sd[j];
  return X;
}

/**
 * Fold of each sample. With `groups`, whole groups (strokes) go to one fold: groups are
 * ordered by their class and dealt out in turn, so each fold gets strokes of several classes.
 * Without groups, samples are dealt out at random (seeded).
 */
export function makeFolds(n: number, nFolds: number, cls: ArrayLike<number>, groups?: ArrayLike<number>, seed = 1): Int32Array {
  const fold = new Int32Array(n);
  if (groups) {
    const firstClass = new Map<number, number>();
    for (let i = 0; i < n; i++) if (!firstClass.has(groups[i])) firstClass.set(groups[i], cls[i]);
    const ids = [...firstClass.keys()].sort((a, b) => firstClass.get(a)! - firstClass.get(b)! || a - b);
    const f = new Map<number, number>();
    ids.forEach((g, q) => f.set(g, q % nFolds));
    for (let i = 0; i < n; i++) fold[i] = f.get(groups[i])!;
    return fold;
  }
  const r = rng(seed), order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  order.forEach((i, q) => { fold[i] = q % nFolds; });
  return fold;
}

/**
 * Cross-validated error rate: for each fold, `fit` gets the training indices and returns a
 * predictor of the class of a sample index; errors are counted on the held-out fold.
 */
export function cvError(n: number, cls: ArrayLike<number>, fold: Int32Array, nFolds: number,
  fit: (train: number[]) => (i: number) => number): number {
  let wrong = 0, total = 0;
  for (let f = 0; f < nFolds; f++) {
    const tr: number[] = [], te: number[] = [];
    for (let i = 0; i < n; i++) (fold[i] === f ? te : tr).push(i);
    if (!te.length || !tr.length) continue;
    const pred = fit(tr);
    for (const i of te) { total++; if (pred(i) !== cls[i]) wrong++; }
  }
  return total ? wrong / total : NaN;
}

/** Rows `idx` of a row-major table. */
export function takeRows(X: ArrayLike<number>, d: number, idx: ArrayLike<number>): Float64Array {
  const out = new Float64Array(idx.length * d);
  for (let r = 0; r < idx.length; r++) for (let j = 0; j < d; j++) out[r * d + j] = X[idx[r] * d + j];
  return out;
}

export interface GridResult { Cs: number[]; gammas: number[]; error: Float64Array; best: { C: number; gamma: number; error: number } }

/**
 * Cross-validated error of an RBF SVM on a (C, γ) grid: error[gi * Cs.length + ci].
 * Standardisation (if asked) uses the statistics of each training fold only.
 */
export function gridSearch(X: ArrayLike<number>, n: number, d: number, cls: ArrayLike<number>, k: number, Cs: number[], gammas: number[],
  fold: Int32Array, nFolds: number, o: { standardise?: boolean; mode?: MultiMode; w?: ArrayLike<number> } = {}): GridResult {
  const error = new Float64Array(Cs.length * gammas.length);
  let best = { C: Cs[0], gamma: gammas[0], error: Infinity };
  gammas.forEach((g, gi) => Cs.forEach((C, ci) => {
    const e = cvError(n, cls, fold, nFolds, tr => fitPredictor(X, d, cls, k, tr, { kernel: { type: "rbf", gamma: g }, C }, o));
    error[gi * Cs.length + ci] = e;
    if (e < best.error - 1e-12) best = { C, gamma: g, error: e };
  }));
  return { Cs, gammas, error, best };
}

/** Train on rows `tr` (standardised with their own statistics if asked) and return a predictor of any row index. */
export function fitPredictor(X: ArrayLike<number>, d: number, cls: ArrayLike<number>, k: number, tr: number[], so: SVMOptions,
  o: { standardise?: boolean; mode?: MultiMode; w?: ArrayLike<number> } = {}): (i: number) => number {
  const Xt = takeRows(X, d, tr);
  const st = o.standardise ? fitStandardiser(Xt, tr.length, d) : null;
  if (st) applyStandardiser(st, Xt, tr.length, d);
  const mm = trainMulti(Xt, tr.length, d, tr.map(i => cls[i]), k, o.mode ?? "ovo", so, o.w ? tr.map(i => o.w![i]) : undefined);
  const x = new Float64Array(d);
  return (i: number) => {
    for (let j = 0; j < d; j++) x[j] = st ? (X[i * d + j] - st.mean[j]) / st.sd[j] : X[i * d + j];
    return multiPredict(mm, x);
  };
}

/** Log-spaced grid lo..hi (powers of ten) with `per` points per decade. */
export function logGrid(lo: number, hi: number, per = 1): number[] {
  const out: number[] = [];
  for (let e = lo; e <= hi + 1e-9; e += 1 / per) out.push(Math.pow(10, Math.round(e * 1000) / 1000));
  return out;
}

// ---------------------------------------------------------------------------------------
// The classifier plug-in for the pixelclass shell (lib/classifier.ts)

/** Largest number of training samples the plug-in keeps (a random, class-stratified subset). */
export const SVM_MAX_TRAIN = 800;

export interface SVMPixelModel extends Model {
  multi: MultiModel;
  std: Standardiser | null;
  nTrain: number;
  nSV: number;
}

/** Feature-major table -> row-major rows `idx`. */
function rowsFromFeatureMajor(X: ArrayLike<number>, n: number, d: number, idx: ArrayLike<number>): Float64Array {
  const out = new Float64Array(idx.length * d);
  for (let r = 0; r < idx.length; r++) for (let j = 0; j < d; j++) out[r * d + j] = X[j * n + idx[r]];
  return out;
}

/** Train the SVM of the plug-in on a TrainingSet (feature-major), with the plug-in's parameter values. */
export function trainPixelSVM(data: TrainingSet, p: { kernel: "linear" | "rbf"; C: number; gamma: number; standardise: boolean; mode: MultiMode; maxTrain?: number }, seed = 1): SVMPixelModel {
  const { n, d, k } = data;
  // stratified random subset (the same fraction of every class)
  const cap = p.maxTrain ?? SVM_MAX_TRAIN;
  let idx = Array.from({ length: n }, (_, i) => i);
  if (n > cap) {
    const r = rng(seed), keep = cap / n;
    idx = idx.filter(() => r.unif() < keep);
  }
  const m = idx.length;
  const X = rowsFromFeatureMajor(data.X, n, d, idx);
  const std = p.standardise ? fitStandardiser(X, m, d) : null;
  if (std) applyStandardiser(std, X, m, d);
  const cls = Int32Array.from(idx, i => data.y[i]);
  const w = data.w ? Float64Array.from(idx, i => data.w![i]) : undefined;
  const kernel: KernelSpec = p.kernel === "linear" ? { type: "linear" } : { type: "rbf", gamma: p.gamma };
  const multi = trainMulti(X, m, d, cls, k, p.mode, { kernel, C: p.C, tol: 1e-3 }, w);
  // union of the support vectors of all sub-models: one kernel row per pixel, shared
  const svSet = new Map<string, number>();
  const svRows: Float64Array[] = [];
  const subIdx = multi.models.map(mod => {
    if (!mod) return new Int32Array(0);
    return Int32Array.from({ length: mod.coef.length }, (_, s) => {
      const key = Array.from(mod.svX.subarray(s * d, (s + 1) * d)).join(",");
      let q = svSet.get(key);
      if (q === undefined) { q = svRows.length; svSet.set(key, q); svRows.push(mod.svX.slice(s * d, (s + 1) * d)); }
      return q;
    });
  });
  const nSV = svRows.length;
  const SV = new Float64Array(nSV * d);
  svRows.forEach((r, q) => SV.set(r, q * d));
  const nm = multi.models.length, f = new Float64Array(nm), kv = new Float64Array(nSV), x = new Float64Array(d);
  const g = p.gamma, isLin = p.kernel === "linear";
  const names = ["linear", "RBF"];
  return {
    k, d, multi, std, nTrain: m, nSV,
    predict(Xf, N, out, from = 0, to = N) {
      const o = out ?? new Float32Array(N * k);
      for (let i = from; i < to; i++) {
        for (let j = 0; j < d; j++) { const v = Xf[j * N + i]; x[j] = std ? (v - std.mean[j]) / std.sd[j] : v; }
        if (isLin) {
          for (let q = 0; q < nm; q++) { const mod = multi.models[q]; f[q] = mod ? decision(mod, x) : (multi.mode === "ovr" ? -Infinity : 0); }
        } else {
          for (let q = 0; q < nSV; q++) {
            let s = 0;
            for (let j = 0; j < d; j++) { const t = SV[q * d + j] - x[j]; s += t * t; }
            kv[q] = Math.exp(-g * s);
          }
          for (let q = 0; q < nm; q++) {
            const mod = multi.models[q];
            if (!mod) { f[q] = multi.mode === "ovr" ? -Infinity : 0; continue; }
            let s = mod.b;
            const si = subIdx[q];
            for (let t = 0; t < si.length; t++) s += mod.coef[t] * kv[si[t]];
            f[q] = s;
          }
        }
        multiScores(multi, f, o, i * k);
      }
      return o;
    },
    summary: () => `${names[isLin ? 0 : 1]}, ${k > 2 ? (multi.mode === "ovr" ? "one-vs-rest, " : "one-vs-one, ") : ""}${nSV} support vectors of ${m} px${std ? "" : " (not standardised)"}`,
  };
}

export const SVM_CLASSIFIER: Classifier = {
  id: "svm",
  name: "SVM",
  description: "Support vector machine with a linear or RBF kernel function; features standardised with the training statistics unless switched off.",
  params: [
    { key: "kernel", label: "kernel function: 0 linear, 1 RBF", min: 0, max: 1, step: 1, value: 1 },
    { key: "C", label: "C", min: -2, max: 3, step: 0.25, value: 0, log: true },
    { key: "gamma", label: "γ", min: -3, max: 1, step: 0.25, value: -1.5, log: true },
    { key: "standardise", label: "standardise: 0 no, 1 yes", min: 0, max: 1, step: 1, value: 1 },
    { key: "ovo", label: "classes: 0 one-vs-rest, 1 one-vs-one", min: 0, max: 1, step: 1, value: 1 },
  ],
  train(data, params, seed) {
    return trainPixelSVM(data, {
      kernel: params.kernel >= 0.5 ? "rbf" : "linear",
      C: paramValue(SVM_CLASSIFIER, params, "C"),
      gamma: paramValue(SVM_CLASSIFIER, params, "gamma"),
      standardise: params.standardise >= 0.5,
      mode: params.ovo >= 0.5 ? "ovo" : "ovr",
    }, seed);
  },
};
