// Dynamic programming on chains and lattices (no DOM). Owner: the dynamic-programming day
// (dp-paths/, polar-boundary/); users: seam carving, DTW, polar boundaries, the Ising
// elective (chain Viterbi / forward–backward), graph cuts (segchain keeps its own file).
//
//  - pathDP: minimal monotone path through a cost matrix C (rows × cols, row-major) on a
//    lattice: each cell's predecessors are given as offsets (di, dj) with di < 0, or di = 0
//    and dj < 0, so a raster scan is a valid order. Per-step penalties, an optional band
//    |i − j| ≤ w, start in the first row (seam / minimal path) or at the corner (DTW), end
//    in the last row or at the far corner. Returns the cumulative table M, the pointer of
//    every cell, the path and the number of candidate evaluations. With `trace`, it also
//    records each cell's candidates and choice (for step-through demos).
//      M(start) = C(start),  M(i, j) = C(i, j) + min_s ( M(i + di_s, j + dj_s) + penalty_s )
//  - circularDP: closed path with one state per stage (one radius per angle), steps
//    |d| ≤ δ with penalty α|d| per step, closure "open" | "unroll" | "exact".
//  - viterbi / forwardBackward: K-label chain energies E = Σ U_p(ℓ_p) + Σ V(ℓ_p, ℓ_{p+1}).
//
// Ties are broken by the order of the steps (the first listed wins) and, at the end, by
// the smallest column, so every result is deterministic.

export interface LatticeStep {
  /** Predecessor offset: the cell (i + di, j + dj). Needs di < 0, or di = 0 and dj < 0. */
  di: number;
  dj: number;
  /** Added to the path cost when this step is taken (default 0). */
  penalty?: number;
}

/** Steps of a vertical minimal path / seam: one row down, at most `delta` columns sideways,
 *  penalty alpha·|d|. Listed straight first, then −1, +1, −2, +2, … (tie order). */
export function verticalSteps(delta = 1, alpha = 0): LatticeStep[] {
  const s: LatticeStep[] = [{ di: -1, dj: 0, penalty: 0 }];
  for (let d = 1; d <= delta; d++) {
    s.push({ di: -1, dj: -d, penalty: alpha * d });
    s.push({ di: -1, dj: d, penalty: alpha * d });
  }
  return s;
}

/** Steps right and down (top-left to bottom-right), optionally the diagonal too (DTW). */
export function cornerSteps(diagonal: boolean, diagonalPenalty = 0): LatticeStep[] {
  const s: LatticeStep[] = [];
  if (diagonal) s.push({ di: -1, dj: -1, penalty: diagonalPenalty });
  s.push({ di: -1, dj: 0, penalty: 0 }, { di: 0, dj: -1, penalty: 0 });
  return s;
}

export interface PathDPOptions {
  steps: LatticeStep[];
  /** "row": any cell of row 0 may start (default); "corner": only (0, 0). */
  start?: "row" | "corner";
  /** "row": the best cell of the last row ends (default); "corner": (rows−1, cols−1). */
  end?: "row" | "corner";
  /** Only cells with |i − j| ≤ band are allowed (DTW's Sakoe–Chiba band). */
  band?: number;
  /** Extra per-cell restriction (e.g. an annulus); disallowed cells get M = ∞. */
  allowed?: (i: number, j: number) => boolean;
  /** Record candidates and choices of every cell in fill order. */
  trace?: boolean;
}

export interface CellTrace {
  i: number;
  j: number;
  /** Candidate predecessors (flat index; −1 outside the grid / not allowed) and their values M + penalty. */
  cand: { from: number; step: number; value: number }[];
  /** Index into `cand` of the chosen minimum (−1 for a start cell or an unreachable one). */
  chosen: number;
  M: number;
}

export interface PathDPResult {
  rows: number;
  cols: number;
  /** Cumulative cost table (∞ where unreachable). */
  M: Float64Array;
  /** Predecessor (flat index) of each cell; −1 for start cells and unreachable cells. */
  from: Int32Array;
  /** Step index used to reach each cell (−1 if none). */
  stepOf: Int32Array;
  /** The optimal path as flat indices, first to last. Empty if nothing is reachable. */
  path: number[];
  /** Its cost (∞ if none). */
  cost: number;
  /** Number of candidate evaluations (one addition and one comparison each). */
  ops: number;
  trace?: CellTrace[];
}

/** Minimal monotone path through C (rows × cols, row-major). */
export function pathDP(C: ArrayLike<number>, rows: number, cols: number, o: PathDPOptions): PathDPResult {
  const n = rows * cols, M = new Float64Array(n).fill(Infinity), from = new Int32Array(n).fill(-1);
  const stepOf = new Int32Array(n).fill(-1);
  const start = o.start ?? "row", end = o.end ?? "row";
  for (const s of o.steps) if (!(s.di < 0 || (s.di === 0 && s.dj < 0))) throw new Error("pathDP: steps must point back in raster order");
  const ok = (i: number, j: number) => i >= 0 && j >= 0 && i < rows && j < cols &&
    (o.band === undefined || Math.abs(i - j) <= o.band) && (!o.allowed || o.allowed(i, j));
  const trace: CellTrace[] | undefined = o.trace ? [] : undefined;
  let ops = 0;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const p = i * cols + j;
    if (!ok(i, j)) continue;
    const isStart = start === "row" ? i === 0 : i === 0 && j === 0;
    if (isStart) {
      M[p] = C[p];
      trace?.push({ i, j, cand: [], chosen: -1, M: M[p] });
      continue;
    }
    let best = Infinity, bs = -1, bq = -1;
    const cand: CellTrace["cand"] = [];
    for (let s = 0; s < o.steps.length; s++) {
      const st = o.steps[s], ii = i + st.di, jj = j + st.dj;
      if (!ok(ii, jj)) { if (trace) cand.push({ from: -1, step: s, value: Infinity }); continue; }
      const q = ii * cols + jj, v = M[q] + (st.penalty ?? 0);
      ops++;
      if (trace) cand.push({ from: q, step: s, value: v });
      if (v < best) { best = v; bs = s; bq = q; }
    }
    if (bq >= 0 && best < Infinity) { M[p] = C[p] + best; from[p] = bq; stepOf[p] = bs; }
    if (trace) trace.push({ i, j, cand, chosen: bq >= 0 && best < Infinity ? cand.findIndex(c => c.step === bs) : -1, M: M[p] });
  }
  let last = -1;
  if (end === "corner") last = M[n - 1] < Infinity ? n - 1 : -1;
  else {
    let b = Infinity;
    for (let j = 0; j < cols; j++) { const p = (rows - 1) * cols + j; if (M[p] < b) { b = M[p]; last = p; } }
  }
  const path: number[] = [];
  for (let p = last; p >= 0; p = from[p]) path.push(p);
  path.reverse();
  return { rows, cols, M, from, stepOf, path, cost: last >= 0 ? M[last] : Infinity, ops, trace };
}

/** Cost of a given path (flat indices) under C and the steps' penalties; ∞ if a move is not a step. */
export function pathCost(C: ArrayLike<number>, cols: number, path: number[], steps: LatticeStep[]): number {
  let c = 0;
  for (let k = 0; k < path.length; k++) {
    c += C[path[k]];
    if (k === 0) continue;
    const di = Math.floor(path[k - 1] / cols) - Math.floor(path[k] / cols), dj = (path[k - 1] % cols) - (path[k] % cols);
    const s = steps.find(st => st.di === di && st.dj === dj);
    if (!s) return Infinity;
    c += s.penalty ?? 0;
  }
  return c;
}

/** Number of monotone paths from the start set to the end set (exact, BigInt). */
export function countPaths(rows: number, cols: number, o: PathDPOptions): bigint {
  const N = new Array<bigint>(rows * cols).fill(0n), start = o.start ?? "row", end = o.end ?? "row";
  const ok = (i: number, j: number) => i >= 0 && j >= 0 && i < rows && j < cols &&
    (o.band === undefined || Math.abs(i - j) <= o.band) && (!o.allowed || o.allowed(i, j));
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    if (!ok(i, j)) continue;
    const p = i * cols + j;
    if (start === "row" ? i === 0 : i === 0 && j === 0) { N[p] = 1n; continue; }
    let s = 0n;
    for (const st of o.steps) { const ii = i + st.di, jj = j + st.dj; if (ok(ii, jj)) s += N[ii * cols + jj]; }
    N[p] = s;
  }
  if (end === "corner") return N[rows * cols - 1];
  let t = 0n;
  for (let j = 0; j < cols; j++) t += N[(rows - 1) * cols + j];
  return t;
}

// ---------------------------------------------------------------------------------------
// Closed paths with one state per stage (polar boundaries)

export type Closure = "open" | "unroll" | "exact";

export interface CircularOptions {
  /** Largest change of state between neighbouring stages (hard constraint). */
  delta: number;
  /** Soft penalty per unit change of state (default 0). */
  alpha?: number;
  closure: Closure;
  /** Copies of the stages for "unroll" (default 3; the middle copy is kept). */
  copies?: number;
  /** Allowed states [lo, hi] (inclusive; default all). */
  lo?: number;
  hi?: number;
}

export interface CircularResult {
  /** State per stage (length K). */
  path: Int32Array;
  /** Cost of the closed curve: Σ C + α Σ|d| over all K transitions, including the closing one. */
  closedCost: number;
  /** Cost without the closing transition. */
  openCost: number;
  /** |s_0 − s_{K−1}|: the jump at the seam. */
  gap: number;
  /** True if the gap satisfies the hard constraint (gap ≤ δ). */
  closes: boolean;
  /** Cumulative table of the (single, or best-start) run, K × S, for display. ∞ = unreachable. */
  M: Float64Array;
  /** Candidate evaluations. */
  ops: number;
}

/**
 * Closed minimal path through C (K stages × S states, row-major: stage k is row k), one
 * state per stage, |s_{k+1} − s_k| ≤ δ with penalty α|d|, stage K−1 joined to stage 0.
 *  - "open": ignore the closure;
 *  - "unroll": run on `copies` stacked copies and keep the middle one (heuristic);
 *  - "exact": fix the start state, run once per start, keep the best closed path.
 */
export function circularDP(C: ArrayLike<number>, K: number, S: number, o: CircularOptions): CircularResult {
  const alpha = o.alpha ?? 0, lo = Math.max(0, o.lo ?? 0), hi = Math.min(S - 1, o.hi ?? S - 1), delta = o.delta;
  const evalClosed = (path: Int32Array) => {
    let c = 0, sm = 0;
    for (let k = 0; k < K; k++) c += C[k * S + path[k]];
    for (let k = 1; k < K; k++) sm += alpha * Math.abs(path[k] - path[k - 1]);
    const gap = Math.abs(path[0] - path[K - 1]);
    return { openCost: c + sm, closedCost: c + sm + alpha * gap, gap, closes: gap <= delta };
  };
  // one run over `stages` stages (stage k uses the costs of stage k mod K); s0 ≥ 0 fixes the first state
  const run = (stages: number, s0: number) => {
    const M = new Float64Array(stages * S).fill(Infinity), from = new Int32Array(stages * S).fill(-1);
    let ops = 0;
    for (let i = lo; i <= hi; i++) if (s0 < 0 || i === s0) M[i] = C[i];
    for (let k = 1; k < stages; k++) {
      const row = k * S, prev = row - S, crow = (k % K) * S;
      for (let i = lo; i <= hi; i++) {
        let best = M[prev + i], bi = i;            // straight first (tie order: 0, −1, +1, −2, …)
        for (let d = 1; d <= delta; d++) {
          const a = i - d, b = i + d, pen = alpha * d;
          if (a >= lo) { const v = M[prev + a] + pen; if (v < best) { best = v; bi = a; } }
          if (b <= hi) { const v = M[prev + b] + pen; if (v < best) { best = v; bi = b; } }
        }
        ops += 1 + Math.min(delta, i - lo) + Math.min(delta, hi - i);
        if (best < Infinity) { M[row + i] = C[crow + i] + best; from[row + i] = prev + bi; }
      }
    }
    return { M, from, ops };
  };
  const backtrack = (from: Int32Array, last: number) => {
    const p: number[] = [];
    for (let q = last; q >= 0; q = from[q]) p.push(q % S);
    return Int32Array.from(p.reverse());
  };
  const bestLast = (M: Float64Array, stages: number) => {
    let bj = -1, b = Infinity;
    for (let j = lo; j <= hi; j++) { const v = M[(stages - 1) * S + j]; if (v < b) { b = v; bj = j; } }
    return bj;
  };
  if (o.closure === "open") {
    const r = run(K, -1), bj = bestLast(r.M, K);
    const path = bj >= 0 ? backtrack(r.from, (K - 1) * S + bj) : new Int32Array(K);
    return { path, M: r.M, ops: r.ops, ...evalClosed(path) };
  }
  if (o.closure === "unroll") {
    const m = Math.max(2, o.copies ?? 3), r = run(m * K, -1), bj = bestLast(r.M, m * K);
    const full = bj >= 0 ? backtrack(r.from, (m * K - 1) * S + bj) : new Int32Array(m * K);
    const mid = Math.floor(m / 2), path = full.slice(mid * K, mid * K + K);
    return { path, M: r.M.slice(mid * K * S, (mid + 1) * K * S), ops: r.ops, ...evalClosed(path) };
  }
  // exact: one run per start state; the end must be within δ of the start (the closing step)
  let best: CircularResult | null = null, ops = 0;
  for (let s0 = lo; s0 <= hi; s0++) {
    const r = run(K, s0);
    ops += r.ops;
    let bj = -1, bc = Infinity;
    for (let j = Math.max(lo, s0 - delta); j <= Math.min(hi, s0 + delta); j++) {
      const v = r.M[(K - 1) * S + j] + alpha * Math.abs(j - s0);
      ops++;
      if (v < bc) { bc = v; bj = j; }
    }
    if (bj < 0 || !(bc < (best?.closedCost ?? Infinity))) continue;
    const path = backtrack(r.from, (K - 1) * S + bj);
    best = { path, M: r.M, ops: 0, ...evalClosed(path) };
  }
  if (!best) {
    const path = new Int32Array(K);
    return { path, M: new Float64Array(K * S).fill(Infinity), ops, openCost: Infinity, closedCost: Infinity, gap: 0, closes: false };
  }
  best.ops = ops;
  return best;
}

// ---------------------------------------------------------------------------------------
// K-label chains: Viterbi (exact minimum) and forward–backward (marginals at temperature T)

/** Pairwise term: a K × K table V[a * K + b] = V(a, b), or a function. */
export type Pairwise = ArrayLike<number> | ((a: number, b: number, p: number) => number);

const pairFn = (V: Pairwise, K: number) => (typeof V === "function" ? V : (a: number, b: number) => (V as ArrayLike<number>)[a * K + b]);

/** Potts pairwise table: 0 on the diagonal, λ elsewhere. */
export function potts(K: number, lambda: number): Float64Array {
  return Float64Array.from({ length: K * K }, (_, i) => (Math.floor(i / K) === i % K ? 0 : lambda));
}

/** Energy of a labelling: U is n × K (U[p * K + ℓ]). */
export function chainEnergyK(U: ArrayLike<number>, K: number, V: Pairwise, L: ArrayLike<number>): number {
  const f = pairFn(V, K);
  let e = 0;
  for (let p = 0; p < L.length; p++) { e += U[p * K + L[p]]; if (p > 0) e += f(L[p - 1], L[p], p - 1); }
  return e;
}

/** Exact minimum of Σ U_p(ℓ_p) + Σ V(ℓ_p, ℓ_{p+1}) by DP. Ties go to the smaller label. */
export function viterbi(U: ArrayLike<number>, K: number, V: Pairwise): { labels: Int32Array; energy: number } {
  const n = Math.floor(U.length / K), f = pairFn(V, K);
  const labels = new Int32Array(n);
  if (n === 0) return { labels, energy: 0 };
  let M = Float64Array.from({ length: K }, (_, l) => U[l]);
  const back = new Int32Array(n * K);
  for (let p = 1; p < n; p++) {
    const Mn = new Float64Array(K);
    for (let b = 0; b < K; b++) {
      let best = Infinity, ba = 0;
      for (let a = 0; a < K; a++) { const v = M[a] + f(a, b, p - 1); if (v < best) { best = v; ba = a; } }
      Mn[b] = best + U[p * K + b]; back[p * K + b] = ba;
    }
    M = Mn;
  }
  let bl = 0;
  for (let l = 1; l < K; l++) if (M[l] < M[bl]) bl = l;
  labels[n - 1] = bl;
  for (let p = n - 1; p > 0; p--) labels[p - 1] = back[p * K + labels[p]];
  return { labels, energy: M[bl] };
}

const logSumExp = (a: ArrayLike<number>): number => {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - m);
  return m + Math.log(s);
};

/**
 * Marginals of the Gibbs distribution P(L) ∝ exp(−E(L)/T) on a chain, by the
 * forward–backward recursions (in log space). Returns the n × K marginals and log Z.
 */
export function forwardBackward(U: ArrayLike<number>, K: number, V: Pairwise, T = 1): { marginals: Float64Array; logZ: number } {
  const n = Math.floor(U.length / K), f = pairFn(V, K);
  const A = new Float64Array(n * K), B = new Float64Array(n * K), tmp = new Float64Array(K);
  for (let l = 0; l < K; l++) A[l] = -U[l] / T;
  for (let p = 1; p < n; p++) for (let b = 0; b < K; b++) {
    for (let a = 0; a < K; a++) tmp[a] = A[(p - 1) * K + a] - f(a, b, p - 1) / T;
    A[p * K + b] = logSumExp(tmp) - U[p * K + b] / T;
  }
  for (let l = 0; l < K; l++) B[(n - 1) * K + l] = 0;
  for (let p = n - 2; p >= 0; p--) for (let a = 0; a < K; a++) {
    for (let b = 0; b < K; b++) tmp[b] = B[(p + 1) * K + b] - f(a, b, p) / T - U[(p + 1) * K + b] / T;
    B[p * K + a] = logSumExp(tmp);
  }
  const logZ = n ? logSumExp(A.subarray((n - 1) * K, n * K)) : 0;
  const marginals = new Float64Array(n * K);
  for (let i = 0; i < n * K; i++) marginals[i] = Math.exp(A[i] + B[i] - logZ);
  return { marginals, logZ };
}
