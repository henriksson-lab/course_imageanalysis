// The random walker (Grady 2006) on a weighted graph, no DOM. Owner: the graph-based
// segmentation day (random-walker/, ws-msf/, superpixels/). Tests: demos/test/graphseg-test.ts.
//
// Affinities w_pq (large = similar). Seeds clamp x = 1 (this label) / 0 (other labels); the
// free nodes solve the Dirichlet problem  L_U x_U = -B^T x_M,  i.e. every free node is the
// w-weighted average of its neighbours. Same numbers as: the probability that a random walk
// started at p (step p -> q with probability w_pq / d_p) reaches a seed of this label first,
// and the voltage when every edge is a conductance w_pq.
//
//  - WGraph / csr(): an undirected weighted graph as an edge list and its adjacency (CSR).
//  - gridGraph(): pixel grid with contrast weights exp(-(I_p - I_q)^2 / 2 sigma^2), floored at eps.
//  - cgSolve(): Jacobi-preconditioned conjugate gradients for a symmetric positive definite
//    operator given as a function (matrix-free), with warm start.
//  - walkerGraph(): K-label random walker on any graph (+ optional priors, Grady 2005).
//  - randomWalker(): the same on a grey image (grid graph).
//  - harmonicDense(): dense reference solver for small graphs (lib/linalg solve).
//  - gaussSeidelSweep(), simulateWalks(), chainVoltages(): for the 1D demo and the tests.
import { gridEdges, Nbhd } from "./graph";
import { solve } from "./linalg";
import { Rng } from "./random";

/** Undirected weighted graph: each edge once (p, q, w). */
export interface WGraph { n: number; p: Int32Array; q: Int32Array; w: Float64Array }

/** Compressed adjacency: neighbours of v are nbr[start[v] .. start[v+1]) with weights wt and edge ids eid. */
export interface Adjacency { start: Int32Array; nbr: Int32Array; wt: Float64Array; eid: Int32Array; deg: Float64Array }

export function csr(G: WGraph): Adjacency {
  const { n } = G, m = G.p.length;
  const cnt = new Int32Array(n + 1);
  for (let k = 0; k < m; k++) { cnt[G.p[k] + 1]++; cnt[G.q[k] + 1]++; }
  for (let v = 0; v < n; v++) cnt[v + 1] += cnt[v];
  const start = Int32Array.from(cnt), fill = Int32Array.from(cnt);
  const nbr = new Int32Array(2 * m), wt = new Float64Array(2 * m), eid = new Int32Array(2 * m), deg = new Float64Array(n);
  for (let k = 0; k < m; k++) {
    const a = G.p[k], b = G.q[k], w = G.w[k];
    let i = fill[a]++; nbr[i] = b; wt[i] = w; eid[i] = k;
    i = fill[b]++; nbr[i] = a; wt[i] = w; eid[i] = k;
    deg[a] += w; deg[b] += w;
  }
  return { start, nbr, wt, eid, deg };
}

/** Pixel grid graph with contrast weights w = exp(-(ΔI)^2 / 2σ^2) (σ = ∞: w = 1), floored at eps.
 *  8-neighbour diagonals get the same weight formula (no geometric factor: the walker needs no
 *  length metric). */
export function gridGraph(I: ArrayLike<number>, w: number, h: number, sigma: number, conn: Nbhd = 4, eps = 1e-4): WGraph {
  const E = gridEdges(h, w, conn);
  const m = E.p.length, W = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const d = I[E.p[k]] - I[E.q[k]];
    const v = Number.isFinite(sigma) ? Math.exp(-d * d / (2 * sigma * sigma)) : 1;
    W[k] = Math.max(eps, v);
  }
  return { n: w * h, p: E.p, q: E.q, w: W };
}

export interface CgOptions { tol?: number; maxIter?: number; onIter?: (it: number, res: number) => void }
export interface CgResult { x: Float64Array; iterations: number; residual: number }

/** Preconditioned conjugate gradients for A x = b (A symmetric positive definite, given as
 *  apply(x, out)); diag = the diagonal of A (Jacobi preconditioner). Stops when
 *  ‖r‖ ≤ tol ‖b‖ (or ‖r‖ ≤ tol if b = 0). x0 is the warm start (copied). */
export function cgSolve(apply: (x: Float64Array, out: Float64Array) => void, b: Float64Array, diag: Float64Array,
  x0?: ArrayLike<number>, o: CgOptions = {}): CgResult {
  const n = b.length, tol = o.tol ?? 1e-6, maxIter = o.maxIter ?? 5000;
  const x = new Float64Array(n);
  if (x0) for (let i = 0; i < n; i++) x[i] = x0[i];
  const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
  apply(x, Ap);
  let bn = 0;
  for (let i = 0; i < n; i++) { r[i] = b[i] - Ap[i]; bn += b[i] * b[i]; }
  bn = Math.sqrt(bn) || 1;
  let rz = 0, rn = 0;
  for (let i = 0; i < n; i++) { z[i] = r[i] / diag[i]; p[i] = z[i]; rz += r[i] * z[i]; rn += r[i] * r[i]; }
  let it = 0;
  rn = Math.sqrt(rn);
  while (rn > tol * bn && it < maxIter) {
    apply(p, Ap);
    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (!(pAp > 0)) break;
    const a = rz / pAp;
    let rz2 = 0; rn = 0;
    for (let i = 0; i < n; i++) {
      x[i] += a * p[i]; r[i] -= a * Ap[i];
      z[i] = r[i] / diag[i]; rz2 += r[i] * z[i]; rn += r[i] * r[i];
    }
    rn = Math.sqrt(rn);
    const beta = rz2 / rz; rz = rz2;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    it++;
    o.onIter?.(it, rn / bn);
  }
  return { x, iterations: it, residual: rn / bn };
}

export interface WalkerOptions {
  /** Prior weight λ (0 = no priors). */
  lambda?: number;
  /** Prior probabilities per label (K arrays of length n, summing to 1 per node). */
  priors?: Float64Array[];
  /** Warm start: previous probabilities per label. */
  x0?: Float64Array[];
  tol?: number;
  maxIter?: number;
}

export interface WalkerResult {
  /** prob[k][v]: probability of label k+1 at node v (seeds are exactly 0 or 1). */
  prob: Float64Array[];
  /** Label 1..K with the highest probability (0 if no seeds of any label). */
  labels: Int32Array;
  iterations: number;
  residual: number;
}

/** K-label random walker on a graph. seeds[v] = 0 (free) or 1..K. With priors (λ > 0):
 *  (L_U + λ I) x^k = λ prior^k − B^T m^k (Grady 2005 with normalised class probabilities).
 *  Solves K − 1 systems (the last label is 1 − the sum) when there are no priors, K with. */
export function walkerGraph(G: WGraph, seeds: ArrayLike<number>, K: number, o: WalkerOptions = {}): WalkerResult {
  const A = csr(G), n = G.n, lam = o.lambda ?? 0, pri = lam > 0 ? o.priors : undefined;
  const idx = new Int32Array(n).fill(-1), free: number[] = [];
  for (let v = 0; v < n; v++) if (!seeds[v]) { idx[v] = free.length; free.push(v); }
  const nu = free.length;
  const diag = new Float64Array(nu);
  for (let i = 0; i < nu; i++) diag[i] = A.deg[free[i]] + (pri ? lam : 0);
  for (let i = 0; i < nu; i++) if (diag[i] <= 0) diag[i] = 1e-12;
  // the free-free part of the Laplacian as a compact CSR (off-diagonal entries, negated)
  const rs = new Int32Array(nu + 1);
  for (let i = 0; i < nu; i++) {
    const v = free[i];
    let c = 0;
    for (let j = A.start[v]; j < A.start[v + 1]; j++) if (idx[A.nbr[j]] >= 0) c++;
    rs[i + 1] = rs[i] + c;
  }
  const cols = new Int32Array(rs[nu]), vals = new Float64Array(rs[nu]);
  for (let i = 0, t = 0; i < nu; i++) {
    const v = free[i];
    for (let j = A.start[v]; j < A.start[v + 1]; j++) { const u = idx[A.nbr[j]]; if (u >= 0) { cols[t] = u; vals[t] = A.wt[j]; t++; } }
  }
  const apply = (x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < nu; i++) {
      let s = diag[i] * x[i];
      for (let j = rs[i], e = rs[i + 1]; j < e; j++) s -= vals[j] * x[cols[j]];
      out[i] = s;
    }
  };
  const prob: Float64Array[] = [];
  let iterations = 0, residual = 0;
  const solveK = pri ? K : K - 1;
  for (let k = 1; k <= solveK; k++) {
    const b = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      const v = free[i];
      let s = pri ? lam * pri[k - 1][v] : 0;
      for (let j = A.start[v]; j < A.start[v + 1]; j++) if (seeds[A.nbr[j]] === k) s += A.wt[j];
      b[i] = s;
    }
    let x0: Float64Array | undefined;
    if (o.x0 && o.x0[k - 1]) { x0 = new Float64Array(nu); for (let i = 0; i < nu; i++) x0[i] = o.x0[k - 1][free[i]]; }
    const r = nu > 0 ? cgSolve(apply, b, diag, x0, { tol: o.tol ?? 1e-6, maxIter: o.maxIter ?? 5000 }) : { x: new Float64Array(0), iterations: 0, residual: 0 };
    iterations += r.iterations; residual = Math.max(residual, r.residual);
    const P = new Float64Array(n);
    for (let v = 0; v < n; v++) if (seeds[v]) P[v] = seeds[v] === k ? 1 : 0;
    for (let i = 0; i < nu; i++) P[free[i]] = Math.min(1, Math.max(0, r.x[i]));
    prob.push(P);
  }
  if (!pri) {
    const P = new Float64Array(n);
    for (let v = 0; v < n; v++) { let s = 1; for (let k = 0; k < K - 1; k++) s -= prob[k][v]; P[v] = Math.min(1, Math.max(0, s)); }
    prob.push(P);
  }
  const labels = new Int32Array(n);
  for (let v = 0; v < n; v++) {
    let best = 0, bp = -1;
    for (let k = 0; k < K; k++) if (prob[k][v] > bp) { bp = prob[k][v]; best = k + 1; }
    labels[v] = best;
  }
  return { prob, labels, iterations, residual };
}

export interface ImageWalkerOptions extends WalkerOptions { sigma?: number; conn?: Nbhd; eps?: number }

/** Random walker on a grey image (w × h, any range; σ in the same units). seeds: 0 free, 1..K. */
export function randomWalker(I: ArrayLike<number>, w: number, h: number, seeds: ArrayLike<number>, K: number, o: ImageWalkerOptions = {}): WalkerResult & { graph: WGraph } {
  const G = gridGraph(I, w, h, o.sigma ?? 20, o.conn ?? 4, o.eps ?? 1e-4);
  return { ...walkerGraph(G, seeds, K, o), graph: G };
}

/** Class probabilities from the seeds' intensity histograms (bins over [lo, hi], Gaussian-smoothed),
 *  normalised to sum to 1 per pixel: the intensity model of the random walker with priors. */
export function seedPriors(I: ArrayLike<number>, seeds: ArrayLike<number>, K: number, lo = 0, hi = 255, bins = 32, smooth = 1.5): Float64Array[] {
  const H: Float64Array[] = [];
  for (let k = 0; k < K; k++) H.push(new Float64Array(bins));
  const bin = (v: number) => Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * bins)));
  for (let v = 0; v < I.length; v++) if (seeds[v] > 0 && seeds[v] <= K) H[seeds[v] - 1][bin(I[v])]++;
  const S: Float64Array[] = H.map(h => {
    const out = new Float64Array(bins);
    let tot = 0;
    for (let i = 0; i < bins; i++) {
      let s = 0;
      for (let j = 0; j < bins; j++) s += h[j] * Math.exp(-((i - j) ** 2) / (2 * smooth * smooth));
      out[i] = s; tot += s;
    }
    for (let i = 0; i < bins; i++) out[i] = tot > 0 ? out[i] / tot + 1e-6 : 1 / bins;
    return out;
  });
  const P: Float64Array[] = S.map(() => new Float64Array(I.length));
  for (let v = 0; v < I.length; v++) {
    const b = bin(I[v]);
    let tot = 0;
    for (let k = 0; k < K; k++) tot += S[k][b];
    for (let k = 0; k < K; k++) P[k][v] = S[k][b] / tot;
  }
  return P;
}

/** Dense reference: harmonic values with fixed[v] = value at seeds, NaN at free nodes. */
export function harmonicDense(G: WGraph, fixed: ArrayLike<number>): Float64Array {
  const n = G.n, idx = new Int32Array(n).fill(-1), free: number[] = [];
  for (let v = 0; v < n; v++) if (Number.isNaN(fixed[v])) { idx[v] = free.length; free.push(v); }
  const m = free.length, L = new Float64Array(m * m), b = new Float64Array(m);
  for (let k = 0; k < G.p.length; k++) {
    const a = G.p[k], c = G.q[k], w = G.w[k], ia = idx[a], ic = idx[c];
    if (ia >= 0) { L[ia * m + ia] += w; if (ic >= 0) L[ia * m + ic] -= w; else b[ia] += w * fixed[c]; }
    if (ic >= 0) { L[ic * m + ic] += w; if (ia >= 0) L[ic * m + ia] -= w; else b[ic] += w * fixed[a]; }
  }
  const x = (m > 0 ? solve(L, b, m) : null) ?? new Float64Array(m).fill(NaN);
  const out = new Float64Array(n);
  for (let v = 0; v < n; v++) out[v] = idx[v] >= 0 ? x[idx[v]] : fixed[v];
  return out;
}

/** One Gauss–Seidel sweep in place: every free node (fixed[v] NaN) takes the w-weighted mean of
 *  its neighbours, in index order. Returns the largest change. */
export function gaussSeidelSweep(A: Adjacency, x: Float64Array, fixed: ArrayLike<number>): number {
  let change = 0;
  for (let v = 0; v < x.length; v++) {
    if (!Number.isNaN(fixed[v])) { x[v] = fixed[v]; continue; }
    let s = 0, d = 0;
    for (let j = A.start[v]; j < A.start[v + 1]; j++) { s += A.wt[j] * x[A.nbr[j]]; d += A.wt[j]; }
    if (d <= 0) continue;
    const nv = s / d;
    change = Math.max(change, Math.abs(nv - x[v]));
    x[v] = nv;
  }
  return change;
}

/** One Jacobi sweep (all free nodes updated from the old values at once). Returns the largest change. */
export function jacobiSweep(A: Adjacency, x: Float64Array, fixed: ArrayLike<number>): number {
  const old = Float64Array.from(x);
  let change = 0;
  for (let v = 0; v < x.length; v++) {
    if (!Number.isNaN(fixed[v])) { x[v] = fixed[v]; continue; }
    let s = 0, d = 0;
    for (let j = A.start[v]; j < A.start[v + 1]; j++) { s += A.wt[j] * old[A.nbr[j]]; d += A.wt[j]; }
    if (d <= 0) continue;
    x[v] = s / d;
    change = Math.max(change, Math.abs(x[v] - old[v]));
  }
  return change;
}

export interface Walks { /** Number of walks ending at each seed label (index = label). */ counts: number[]; /** The first few paths. */ paths: number[][]; unfinished: number }

/** n random walks from `start` (step to a neighbour with probability w/d) until they hit a seed
 *  (seeds[v] > 0) or maxSteps; records the first `keep` paths. */
export function simulateWalks(A: Adjacency, seeds: ArrayLike<number>, start: number, n: number, r: Rng, maxSteps = 100000, keep = 30): Walks {
  let K = 0;
  for (let v = 0; v < seeds.length; v++) K = Math.max(K, seeds[v]);
  const counts = new Array(K + 1).fill(0), paths: number[][] = [];
  let unfinished = 0;
  for (let t = 0; t < n; t++) {
    let v = start;
    const path = t < keep ? [v] : null;
    let steps = 0;
    while (!seeds[v] && steps < maxSteps) {
      const d = A.deg[v];
      let u = r.unif() * d, j = A.start[v];
      for (; j < A.start[v + 1] - 1; j++) { u -= A.wt[j]; if (u < 0) break; }
      v = A.nbr[j];
      steps++;
      if (path) path.push(v);
    }
    if (seeds[v]) counts[seeds[v]]++; else unfinished++;
    if (path) paths.push(path);
  }
  return { counts, paths, unfinished };
}

/** Chain graph 0–1–…–(n−1) with the given edge weights. */
export function chainGraph(weights: ArrayLike<number>): WGraph {
  const m = weights.length;
  const p = new Int32Array(m), q = new Int32Array(m), w = new Float64Array(m);
  for (let k = 0; k < m; k++) { p[k] = k; q[k] = k + 1; w[k] = weights[k]; }
  return { n: m + 1, p, q, w };
}

/** Voltages on a chain with fixed values (NaN = free): between two fixed nodes the voltage
 *  drops in proportion to resistance 1/w; free ends hang at the value of their nearest fixed node. */
export function chainVoltages(weights: ArrayLike<number>, fixed: ArrayLike<number>): Float64Array {
  const n = weights.length + 1, x = new Float64Array(n).fill(NaN);
  const F: number[] = [];
  for (let v = 0; v < n; v++) if (!Number.isNaN(fixed[v])) F.push(v);
  if (F.length === 0) return x;
  for (let v = 0; v <= F[0]; v++) x[v] = fixed[F[0]];
  for (let v = F[F.length - 1]; v < n; v++) x[v] = fixed[F[F.length - 1]];
  for (let s = 0; s + 1 < F.length; s++) {
    const a = F[s], b = F[s + 1];
    let R = 0;
    for (let k = a; k < b; k++) R += 1 / weights[k];
    let acc = 0;
    x[a] = fixed[a];
    for (let k = a; k < b; k++) { acc += 1 / weights[k]; x[k + 1] = fixed[a] + (fixed[b] - fixed[a]) * acc / R; }
  }
  return x;
}

/** Resumable preconditioned CG (same algorithm as cgSolve), for solving a little per animation frame. */
export class CgRun {
  readonly x: Float64Array;
  iterations = 0;
  residual = 1;
  private r: Float64Array; private z: Float64Array; private p: Float64Array; private Ap: Float64Array;
  private rz = 0; private bn = 1;
  constructor(private apply: (x: Float64Array, out: Float64Array) => void, b: Float64Array, private diag: Float64Array, x0?: ArrayLike<number>, public tol = 1e-5) {
    const n = b.length;
    this.x = new Float64Array(n);
    if (x0) for (let i = 0; i < n; i++) this.x[i] = x0[i];
    this.r = new Float64Array(n); this.z = new Float64Array(n); this.p = new Float64Array(n); this.Ap = new Float64Array(n);
    apply(this.x, this.Ap);
    let bn = 0, rn = 0;
    for (let i = 0; i < n; i++) { this.r[i] = b[i] - this.Ap[i]; bn += b[i] * b[i]; }
    this.bn = Math.sqrt(bn) || 1;
    for (let i = 0; i < n; i++) { this.z[i] = this.r[i] / diag[i]; this.p[i] = this.z[i]; this.rz += this.r[i] * this.z[i]; rn += this.r[i] * this.r[i]; }
    this.residual = Math.sqrt(rn) / this.bn;
  }
  get done(): boolean { return this.residual <= this.tol || this.x.length === 0; }
  /** Run up to `count` iterations (fewer if converged). */
  step(count: number): void {
    const { x, r, z, p, Ap, diag } = this, n = x.length;
    for (let c = 0; c < count && !this.done; c++) {
      this.apply(p, Ap);
      let pAp = 0;
      for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
      if (!(pAp > 0)) { this.residual = 0; break; }
      const a = this.rz / pAp;
      let rz2 = 0, rn = 0;
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; z[i] = r[i] / diag[i]; rz2 += r[i] * z[i]; rn += r[i] * r[i]; }
      const beta = rz2 / this.rz; this.rz = rz2;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
      this.iterations++;
      this.residual = Math.sqrt(rn) / this.bn;
    }
  }
}

/** A K-label random walker that is solved a little at a time (for live demos): call step()
 *  per animation frame until done, read prob()/labels() at any time. Same equations as walkerGraph. */
export class WalkerSolver {
  readonly K: number;
  readonly n: number;
  private free: Int32Array;
  private runs: CgRun[] = [];
  private seeds: ArrayLike<number>;
  private pri: boolean;
  constructor(G: WGraph, seeds: ArrayLike<number>, K: number, o: WalkerOptions = {}) {
    this.K = K; this.n = G.n; this.seeds = seeds;
    const A = csr(G), n = G.n, lam = o.lambda ?? 0, pri = lam > 0 ? o.priors : undefined;
    this.pri = !!pri;
    const idx = new Int32Array(n).fill(-1), fr: number[] = [];
    for (let v = 0; v < n; v++) if (!seeds[v]) { idx[v] = fr.length; fr.push(v); }
    const free = Int32Array.from(fr), nu = free.length;
    this.free = free;
    const diag = new Float64Array(nu);
    for (let i = 0; i < nu; i++) diag[i] = Math.max(1e-12, A.deg[free[i]] + (pri ? lam : 0));
    const rs = new Int32Array(nu + 1);
    for (let i = 0; i < nu; i++) { const v = free[i]; let c = 0; for (let j = A.start[v]; j < A.start[v + 1]; j++) if (idx[A.nbr[j]] >= 0) c++; rs[i + 1] = rs[i] + c; }
    const cols = new Int32Array(rs[nu]), vals = new Float64Array(rs[nu]);
    for (let i = 0, t = 0; i < nu; i++) { const v = free[i]; for (let j = A.start[v]; j < A.start[v + 1]; j++) { const u = idx[A.nbr[j]]; if (u >= 0) { cols[t] = u; vals[t] = A.wt[j]; t++; } } }
    const apply = (x: Float64Array, out: Float64Array) => {
      for (let i = 0; i < nu; i++) { let s = diag[i] * x[i]; for (let j = rs[i], e = rs[i + 1]; j < e; j++) s -= vals[j] * x[cols[j]]; out[i] = s; }
    };
    const solveK = pri ? K : K - 1;
    for (let k = 1; k <= solveK; k++) {
      const b = new Float64Array(nu);
      for (let i = 0; i < nu; i++) {
        const v = free[i];
        let s = pri ? lam * pri[k - 1][v] : 0;
        for (let j = A.start[v]; j < A.start[v + 1]; j++) if (seeds[A.nbr[j]] === k) s += A.wt[j];
        b[i] = s;
      }
      let x0: Float64Array | undefined;
      if (o.x0 && o.x0[k - 1]) { x0 = new Float64Array(nu); for (let i = 0; i < nu; i++) x0[i] = o.x0[k - 1][free[i]]; }
      this.runs.push(new CgRun(apply, b, diag, x0, o.tol ?? 1e-5));
    }
  }
  get done(): boolean { return this.runs.every(r => r.done); }
  get iterations(): number { return this.runs.reduce((a, r) => a + r.iterations, 0); }
  get residual(): number { return this.runs.reduce((a, r) => Math.max(a, r.residual), 0); }
  /** Run CG for at most `ms` milliseconds (checked every `chunk` iterations). */
  step(ms: number, chunk = 10): void {
    const t0 = performance.now();
    while (!this.done && performance.now() - t0 < ms) for (const r of this.runs) r.step(chunk);
  }
  /** Current probabilities per label (clamped to [0, 1]). */
  prob(): Float64Array[] {
    const { n, free, seeds, K } = this, out: Float64Array[] = [];
    for (let k = 1; k <= this.runs.length; k++) {
      const P = new Float64Array(n), x = this.runs[k - 1].x;
      for (let v = 0; v < n; v++) if (seeds[v]) P[v] = seeds[v] === k ? 1 : 0;
      for (let i = 0; i < free.length; i++) P[free[i]] = Math.min(1, Math.max(0, x[i]));
      out.push(P);
    }
    if (!this.pri) {
      const P = new Float64Array(n);
      for (let v = 0; v < n; v++) { let s = 1; for (let k = 0; k < K - 1; k++) s -= out[k][v]; P[v] = Math.min(1, Math.max(0, s)); }
      out.push(P);
    }
    return out;
  }
}
