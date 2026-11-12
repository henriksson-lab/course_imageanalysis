// Ising model and binary Markov random fields on a 4-neighbour lattice (owner: the Ising /
// MRF day). No DOM: also used by the tests (demos/test/ising-test.ts).
//
// Spins s_p in {-1, +1} (Int8Array, row-major, p = y * w + x). Energy
//     E(s) = -J sum_{(p,q)} s_p s_q  -  sum_p h_p s_p
// over the neighbour pairs of the lattice (free boundary, or periodic = a torus), with a
// uniform field h (a number) or a per-pixel field (an array). Gibbs distribution
// p(s) = exp(-E(s)/T) / Z.
//
//   const L = lattice(256, 256, true);
//   const s = randomSpins(L.n, rng(1));
//   metropolisSweep(L, s, 1, 0, 2.0, r);     // one sweep: every site visited once
//   magnetisation(s); energy(L, s, 1, 0).total / L.n;
//
// Also: Gibbs (heat-bath) sweeps, ICM, mean field, annealing schedules, exact enumeration for
// small lattices (Z, energy levels, magnetisation, marginals), the 1D transfer matrix, Onsager's
// exact results, and the bridge to segmentation: an Ising posterior is the binary segmentation
// energy with lambda = 2J and data costs D_p(1) - D_p(0) = -2 h_p, so the exact MAP is one
// minimum cut (lib/graph.ts solveSeg; Greig, Porteous & Seheult 1989).
import { Rng, randInt } from "./random";
import { gridEdges, solveSeg, SegProblem } from "./graph";

/** Critical temperature of the 2D Ising model with J = 1 (Onsager 1944): 2 / ln(1 + sqrt 2) = 2.269. */
export const TC = 2 / Math.log(1 + Math.SQRT2);

/** A field: one number for every pixel, or one per pixel. */
export type Field = number | ArrayLike<number>;

const fieldAt = (h: Field, p: number): number => (typeof h === "number" ? h : h[p]);

// ---------------------------------------------------------------------------------------
// Lattice

export interface Lattice {
  w: number; h: number; n: number;
  periodic: boolean;
  /** 4 neighbour indices per site (left, right, up, down); -1 where there is none (free boundary). */
  nb: Int32Array;
  /** Each undirected neighbour pair once: right and down neighbour of every site. */
  ep: Int32Array; eq: Int32Array;
  /** Sites of colour 0 ((x + y) even) then colour 1, for checkerboard sweeps; `half` = count of colour 0. */
  checker: Int32Array; half: number;
}

/** A w x h lattice with 4-neighbours. A periodic lattice of width 2 has the same pair twice
 *  (left and right neighbour coincide); the energy then counts it twice, as the torus does. */
export function lattice(w: number, h: number, periodic = false): Lattice {
  const n = w * h, nb = new Int32Array(4 * n).fill(-1);
  const P: number[] = [], Q: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    const l = x > 0 ? p - 1 : periodic && w > 1 ? p + w - 1 : -1;
    const r = x < w - 1 ? p + 1 : periodic && w > 1 ? p - w + 1 : -1;
    const u = y > 0 ? p - w : periodic && h > 1 ? p + (h - 1) * w : -1;
    const d = y < h - 1 ? p + w : periodic && h > 1 ? p - (h - 1) * w : -1;
    nb[4 * p] = l; nb[4 * p + 1] = r; nb[4 * p + 2] = u; nb[4 * p + 3] = d;
    if (r >= 0) { P.push(p); Q.push(r); }
    if (d >= 0) { P.push(p); Q.push(d); }
  }
  const checker = new Int32Array(n);
  let k = 0;
  for (let c = 0; c < 2; c++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (((x + y) & 1) === c) checker[k++] = y * w + x;
  let half = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (((x + y) & 1) === 0) half++;
  return { w, h, n, periodic, nb, ep: Int32Array.from(P), eq: Int32Array.from(Q), checker, half };
}

/** Number of neighbour pairs (edges) of the lattice. */
export const edgeCount = (L: Lattice): number => L.ep.length;

// ---------------------------------------------------------------------------------------
// States and observables

/** Independent random spins (each +1 with probability pPlus). */
export function randomSpins(n: number, r: Rng, pPlus = 0.5): Int8Array {
  const s = new Int8Array(n);
  for (let p = 0; p < n; p++) s[p] = r.unif() < pPlus ? 1 : -1;
  return s;
}

/** All spins equal to v. */
export const constantSpins = (n: number, v: 1 | -1 = 1): Int8Array => new Int8Array(n).fill(v);

/** Sum of the neighbouring spins of site p. */
export function neighbourSum(L: Lattice, s: ArrayLike<number>, p: number): number {
  let t = 0;
  for (let k = 4 * p; k < 4 * p + 4; k++) { const q = L.nb[k]; if (q >= 0) t += s[q]; }
  return t;
}

/** Local field u_p = J * (sum of neighbours) + h_p: the "vote" for s_p = +1. */
export function localField(L: Lattice, s: ArrayLike<number>, p: number, J: number, h: Field): number {
  return J * neighbourSum(L, s, p) + fieldAt(h, p);
}

export interface IsingEnergy { pair: number; field: number; total: number }

/** E(s) = -J sum_pairs s_p s_q - sum_p h_p s_p, split into the pair (coupling) and field parts. */
export function energy(L: Lattice, s: ArrayLike<number>, J: number, h: Field): IsingEnergy {
  let ss = 0, hs = 0;
  for (let k = 0; k < L.ep.length; k++) ss += s[L.ep[k]] * s[L.eq[k]];
  for (let p = 0; p < L.n; p++) hs += fieldAt(h, p) * s[p];
  const pair = -J * ss, field = -hs;
  return { pair, field, total: pair + field };
}

/** Mean spin m = (1/n) sum s_p (between -1 and +1). */
export function magnetisation(s: ArrayLike<number>): number {
  let t = 0;
  for (let p = 0; p < s.length; p++) t += s[p];
  return t / s.length;
}

/** Number of neighbour pairs with different spins (the length of all domain walls). */
export function disagreements(L: Lattice, s: ArrayLike<number>): number {
  let c = 0;
  for (let k = 0; k < L.ep.length; k++) if (s[L.ep[k]] !== s[L.eq[k]]) c++;
  return c;
}

/** Energy change of flipping site p: dE = 2 s_p u_p. */
export function flipDelta(L: Lattice, s: ArrayLike<number>, p: number, J: number, h: Field): number {
  return 2 * s[p] * localField(L, s, p, J, h);
}

/** p(s_p = +1 | neighbours) = 1 / (1 + exp(-2 u_p / T)); at T = 0 the sign of u_p (1/2 at u_p = 0). */
export function conditionalPlus(u: number, T: number): number {
  if (T <= 0) return u > 0 ? 1 : u < 0 ? 0 : 0.5;
  return 1 / (1 + Math.exp(-2 * u / T));
}

/** Metropolis acceptance probability min(1, exp(-dE / T)). */
export function acceptance(dE: number, T: number): number {
  if (dE <= 0) return 1;
  return T <= 0 ? 0 : Math.exp(-dE / T);
}

// ---------------------------------------------------------------------------------------
// Sweeps

/** Order in which a sweep visits the sites. "checkerboard": all (x + y) even, then all odd
 *  (each half only sees the other, so the result does not depend on the order within it);
 *  "raster": row by row; "random": n sites drawn at random (with repetition). */
export type Order = "checkerboard" | "raster" | "random";

function visit(L: Lattice, order: Order, r: Rng | undefined, f: (p: number) => void): void {
  const n = L.n;
  if (order === "checkerboard") { const c = L.checker; for (let k = 0; k < n; k++) f(c[k]); }
  else if (order === "raster") { for (let p = 0; p < n; p++) f(p); }
  else { if (!r) throw new Error("random order needs a generator"); for (let k = 0; k < n; k++) f(randInt(r, n)); }
}

/** Table of a probability per (s_p, neighbour sum) for a uniform field: index (s_p > 0 ? 9 : 0) + nsum + 4. */
function table(J: number, h: number, f: (s: number, u: number) => number): Float64Array {
  const t = new Float64Array(18);
  for (const s of [-1, 1]) for (let ns = -4; ns <= 4; ns++) t[(s > 0 ? 9 : 0) + ns + 4] = f(s, J * ns + h);
  return t;
}

/** One Metropolis sweep: at each visited site propose the flip, accept with min(1, e^{-dE/T}).
 *  Returns the number of accepted flips. A uniform field uses a lookup table. */
export function metropolisSweep(L: Lattice, s: Int8Array, J: number, h: Field, T: number, r: Rng, order: Order = "checkerboard"): number {
  let acc = 0;
  const nb = L.nb;
  if (typeof h === "number") {
    const tab = table(J, h, (sp, u) => acceptance(2 * sp * u, T));
    visit(L, order, r, p => {
      const k = 4 * p, a = nb[k], b = nb[k + 1], c = nb[k + 2], d = nb[k + 3];
      const ns = (a >= 0 ? s[a] : 0) + (b >= 0 ? s[b] : 0) + (c >= 0 ? s[c] : 0) + (d >= 0 ? s[d] : 0);
      const pa = tab[(s[p] > 0 ? 9 : 0) + ns + 4];
      if (pa >= 1 || r.unif() < pa) { s[p] = -s[p] as -1 | 1; acc++; }
    });
  } else {
    visit(L, order, r, p => {
      const pa = acceptance(2 * s[p] * localField(L, s, p, J, h), T);
      if (pa >= 1 || r.unif() < pa) { s[p] = -s[p] as -1 | 1; acc++; }
    });
  }
  return acc;
}

/** One Gibbs (heat-bath) sweep: redraw each visited spin from p(s_p | neighbours).
 *  Returns the number of spins that changed. T = 0 is ICM with random tie-breaking. */
export function gibbsSweep(L: Lattice, s: Int8Array, J: number, h: Field, T: number, r: Rng, order: Order = "checkerboard"): number {
  let ch = 0;
  const nb = L.nb;
  const set = (p: number, pp: number) => {
    const v = r.unif() < pp ? 1 : -1;
    if (v !== s[p]) { s[p] = v; ch++; }
  };
  if (typeof h === "number") {
    const tab = table(J, h, (_s, u) => conditionalPlus(u, T));
    visit(L, order, r, p => {
      const k = 4 * p, a = nb[k], b = nb[k + 1], c = nb[k + 2], d = nb[k + 3];
      const ns = (a >= 0 ? s[a] : 0) + (b >= 0 ? s[b] : 0) + (c >= 0 ? s[c] : 0) + (d >= 0 ? s[d] : 0);
      set(p, tab[ns + 4]);
    });
  } else {
    visit(L, order, r, p => set(p, conditionalPlus(localField(L, s, p, J, h), T)));
  }
  return ch;
}

/** One ICM sweep (Besag 1986): set each visited spin to the sign of its local field, i.e. the
 *  value with the lower energy given its neighbours; ties keep the current value. Never raises
 *  the energy. Returns the number of changes (0 = a local minimum). */
export function icmSweep(L: Lattice, s: Int8Array, J: number, h: Field, order: Order = "raster", r?: Rng): number {
  let ch = 0;
  visit(L, order, r, p => {
    const u = localField(L, s, p, J, h);
    const v = u > 0 ? 1 : u < 0 ? -1 : s[p];
    if (v !== s[p]) { s[p] = v; ch++; }
  });
  return ch;
}

/** ICM until no spin changes (or maxSweeps). Returns the number of sweeps made. */
export function icm(L: Lattice, s: Int8Array, J: number, h: Field, maxSweeps = 100, order: Order = "raster", r?: Rng): number {
  for (let k = 1; k <= maxSweeps; k++) if (icmSweep(L, s, J, h, order, r) === 0) return k;
  return maxSweeps;
}

/** One mean-field sweep, in place: m_p = tanh((J sum_q m_q + h_p) / T). Returns the largest change.
 *  Iterated, it approaches a fixed point; m_p approximates the posterior mean of s_p. */
export function meanFieldSweep(L: Lattice, m: Float64Array, J: number, h: Field, T: number, damping = 0): number {
  let dmax = 0;
  for (let p = 0; p < L.n; p++) {
    const u = J * neighbourSum(L, m, p) + fieldAt(h, p);
    const v = T > 0 ? Math.tanh(u / T) : Math.sign(u);
    const nv = damping * m[p] + (1 - damping) * v;
    dmax = Math.max(dmax, Math.abs(nv - m[p]));
    m[p] = nv;
  }
  return dmax;
}

/** Signs of a real array (0 counts as +1). */
export function signs(m: ArrayLike<number>): Int8Array {
  const s = new Int8Array(m.length);
  for (let p = 0; p < m.length; p++) s[p] = m[p] >= 0 ? 1 : -1;
  return s;
}

// ---------------------------------------------------------------------------------------
// Annealing schedules

/** Geometric cooling: T_k = T0 * rate^k (what people use). */
export const geometricT = (T0: number, rate: number, k: number): number => T0 * rate ** k;

/** Logarithmic cooling T_k = c / ln(k + 2) (Geman & Geman 1984: guaranteed for c large enough, impractically slow). */
export const logT = (c: number, k: number): number => c / Math.log(k + 2);

/** Number of sweeps for geometric cooling from T0 down to T1. */
export const geometricSweeps = (T0: number, T1: number, rate: number): number =>
  Math.max(0, Math.ceil(Math.log(T1 / T0) / Math.log(rate)));

// ---------------------------------------------------------------------------------------
// Exact enumeration (n <= 20)

export interface Level { E: number; count: number; weight: number; p: number }
export interface Enumeration {
  n: number;
  /** Energy and probability of every state; state k has s_i = +1 exactly where bit i of k is 1. */
  E: Float64Array; p: Float64Array;
  Z: number;
  /** log Z (safe for large |E|/T). */
  logZ: number;
  /** Distinct energies, ascending, with the number of states, weight count * e^{-E/T} (relative to the minimum energy when T is small) and probability. */
  levels: Level[];
  /** Distinct magnetisations m = (1/n) sum s, ascending, with count and probability. */
  mags: { m: number; count: number; p: number }[];
  /** p(s_i = +1) for every site. */
  marginals: Float64Array;
  /** Mean energy and mean |m|. */
  meanE: number; meanAbsM: number;
}

/** Spins of state k (bit i = 1 means s_i = +1). */
export function stateSpins(k: number, n: number): Int8Array {
  const s = new Int8Array(n);
  for (let i = 0; i < n; i++) s[i] = (k >> i) & 1 ? 1 : -1;
  return s;
}

/** Index of a spin configuration (inverse of stateSpins). */
export function stateIndex(s: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < s.length; i++) if (s[i] > 0) k |= 1 << i;
  return k;
}

const roundE = (e: number) => Math.round(e * 1e9) / 1e9;

/** Every state of a small lattice: Z, probabilities, energy levels, magnetisation, marginals. T > 0. */
export function enumerate(L: Lattice, J: number, h: Field, T: number): Enumeration {
  const n = L.n;
  if (n > 20) throw new Error("enumerate: at most 20 sites");
  const N = 1 << n, E = new Float64Array(N), p = new Float64Array(N);
  let Emin = Infinity;
  for (let k = 0; k < N; k++) { E[k] = energy(L, stateSpins(k, n), J, h).total; if (E[k] < Emin) Emin = E[k]; }
  let z = 0;  // Z * e^{Emin/T}
  for (let k = 0; k < N; k++) { p[k] = Math.exp(-(E[k] - Emin) / T); z += p[k]; }
  for (let k = 0; k < N; k++) p[k] /= z;
  const logZ = Math.log(z) - Emin / T;
  const lev = new Map<number, { count: number; p: number }>(), mg = new Map<number, { count: number; p: number }>();
  const marg = new Float64Array(n);
  let meanE = 0, meanAbsM = 0;
  for (let k = 0; k < N; k++) {
    const e = roundE(E[k]);
    const a = lev.get(e) ?? { count: 0, p: 0 }; a.count++; a.p += p[k]; lev.set(e, a);
    let sum = 0;
    for (let i = 0; i < n; i++) if ((k >> i) & 1) { sum++; marg[i] += p[k]; }
    const m = (2 * sum - n) / n;
    const b = mg.get(m) ?? { count: 0, p: 0 }; b.count++; b.p += p[k]; mg.set(m, b);
    meanE += p[k] * E[k]; meanAbsM += p[k] * Math.abs(m);
  }
  const Z = Math.exp(logZ);
  const levels = [...lev.entries()].sort((a, b) => a[0] - b[0])
    .map(([e, v]) => ({ E: e, count: v.count, weight: v.count * Math.exp(-e / T), p: v.p }));
  const mags = [...mg.entries()].sort((a, b) => a[0] - b[0]).map(([m, v]) => ({ m, count: v.count, p: v.p }));
  return { n, E, p, Z, logZ, levels, mags, marginals: marg, meanE, meanAbsM };
}

// ---------------------------------------------------------------------------------------
// 1D chain: transfer matrix

/** Transfer matrix of a 1D Ising chain, [[e^{(J+h)/T}, e^{-J/T}], [e^{-J/T}, e^{(J-h)/T}]] (rows: s = +1, -1). */
export function transferMatrix(J: number, h: number, T: number): [[number, number], [number, number]] {
  return [[Math.exp((J + h) / T), Math.exp(-J / T)], [Math.exp(-J / T), Math.exp((J - h) / T)]];
}

/** Eigenvalues of the transfer matrix, largest first. */
export function transferEigenvalues(J: number, h: number, T: number): [number, number] {
  const a = Math.exp(J / T) * Math.cosh(h / T);
  const d = Math.sqrt(Math.exp(2 * J / T) * Math.sinh(h / T) ** 2 + Math.exp(-2 * J / T));
  return [a + d, a - d];
}

/** Z of a ring of n spins: tr(M^n) = l1^n + l2^n. */
export function ringZ(n: number, J: number, h: number, T: number): number {
  const [l1, l2] = transferEigenvalues(J, h, T);
  return l1 ** n + l2 ** n;
}

// ---------------------------------------------------------------------------------------
// Onsager's exact results for the 2D square lattice (h = 0, infinite lattice)

/** Critical temperature for coupling J: 2J / ln(1 + sqrt 2). */
export const criticalT = (J = 1): number => 2 * J / Math.log(1 + Math.SQRT2);

/** Spontaneous magnetisation |m| = (1 - sinh(2J/T)^-4)^(1/8) below T_c, 0 above (Onsager, Yang 1952). */
export function onsagerM(T: number, J = 1): number {
  if (T <= 0) return 1;
  const x = Math.sinh(2 * J / T);
  const v = 1 - x ** -4;
  return v > 0 ? v ** 0.125 : 0;
}

// ---------------------------------------------------------------------------------------
// Bayesian denoising: data costs, Ising posterior and its exact MAP by min-cut

/** Field from data costs of the two labels: h_p = (D_p(0) - D_p(1)) / 2 (label 1 = spin +1).
 *  Up to a constant, sum_p D_p(L_p) = -sum_p h_p s_p. */
export function fieldFromCosts(D0: ArrayLike<number>, D1: ArrayLike<number>): Float64Array {
  const h = new Float64Array(D0.length);
  for (let p = 0; p < h.length; p++) h[p] = (D0[p] - D1[p]) / 2;
  return h;
}

/** Data costs of the flip-noise model: each pixel of the binary image y (0/1) was flipped with
 *  probability eps. Up to a constant: 0 if the label agrees with y, log((1 - eps)/eps) if not. */
export function flipCosts(y: ArrayLike<number>, eps: number): { D0: Float64Array; D1: Float64Array } {
  const c = Math.log((1 - eps) / eps), n = y.length;
  const D0 = new Float64Array(n), D1 = new Float64Array(n);
  for (let p = 0; p < n; p++) { if (y[p]) D0[p] = c; else D1[p] = c; }
  return { D0, D1 };
}

/** Data costs of Gaussian noise on two grey levels mu0, mu1: D_p(l) = (y_p - mu_l)^2 / (2 sd^2). */
export function gaussLevelCosts(y: ArrayLike<number>, mu0: number, mu1: number, sd: number): { D0: Float64Array; D1: Float64Array } {
  const n = y.length, D0 = new Float64Array(n), D1 = new Float64Array(n), k = 1 / (2 * sd * sd);
  for (let p = 0; p < n; p++) { D0[p] = (y[p] - mu0) ** 2 * k; D1[p] = (y[p] - mu1) ** 2 * k; }
  return { D0, D1 };
}

/** Segmentation energy sum_p D_p(L_p) + lambda * #(neighbour pairs with L_p != L_q), labels from spins (L = 1 for s = +1). */
export function segmentationEnergy(L: Lattice, s: ArrayLike<number>, D0: ArrayLike<number>, D1: ArrayLike<number>, lambda: number): { data: number; smooth: number; total: number } {
  let data = 0;
  for (let p = 0; p < L.n; p++) data += s[p] > 0 ? D1[p] : D0[p];
  const smooth = lambda * disagreements(L, s);
  return { data, smooth, total: data + smooth };
}

/** The Ising posterior (J, field h_p) on a free-boundary w x h grid as a segmentation problem
 *  for lib/graph.ts: lambda = 2J, D_p(1) = -h_p, D_p(0) = +h_p, w_pq = 1, 4-neighbours.
 *  E_ising(s) = E_seg(L) - J * #edges. Needs J >= 0 (submodular). */
export function toSegProblem(w: number, h: number, J: number, field: Field): SegProblem {
  const n = w * h, D0 = new Float64Array(n), D1 = new Float64Array(n);
  for (let p = 0; p < n; p++) { const f = fieldAt(field, p); D0[p] = f; D1[p] = -f; }
  const E = gridEdges(h, w, 4);
  return { n, D0, D1, E, w: E.geo, lambda: 2 * J };
}

/** Exact MAP (lowest-energy state) of an Ising posterior with J >= 0 on a free-boundary grid,
 *  by one minimum cut. Returns the spins and their Ising energy. */
export function exactMAP(w: number, h: number, J: number, field: Field): { s: Int8Array; energy: number } {
  if (J < 0) throw new Error("exactMAP: needs J >= 0");
  const res = solveSeg(toSegProblem(w, h, J, field));
  const s = new Int8Array(w * h);
  for (let p = 0; p < s.length; p++) s[p] = res.L[p] ? 1 : -1;
  return { s, energy: energy(lattice(w, h, false), s, J, field).total };
}

/** Spins from 0/1 labels and back. */
export const spinsFromLabels = (L: ArrayLike<number>): Int8Array => Int8Array.from({ length: L.length }, (_, p) => (L[p] ? 1 : -1));
export const labelsFromSpins = (s: ArrayLike<number>): Uint8Array => Uint8Array.from({ length: s.length }, (_, p) => (s[p] > 0 ? 1 : 0));
