// Grid graphs for images, and binary segmentation by graph cut. No DOM: also used by
// the test (demos/test/). Shared by the pixelgraph, graphcut and ncut demos.
//
// Pixels are numbered row by row, p = r * cols + c. Labels: 1 = object, 0 = background.
// Energy  E(L) = sum_p D_p(L_p) + lambda * sum_{(p,q)} w_pq [L_p != L_q].
// Graph (Boykov & Jolly 2001): node per pixel, source s = object, sink t = background.
//   s -> p with capacity D_p(0)   (cut when p is background)
//   p -> t with capacity D_p(1)   (cut when p is object)
//   p <-> q with capacity lambda * w_pq in both directions
// so the cost of the cut equals E(L) with L_p = 1 exactly for the pixels on the source side.
import { FlowGraph } from "./maxflow";

export type Nbhd = 4 | 8;

export interface GridEdges {
  p: Int32Array; q: Int32Array;   // each undirected neighbour pair once, p < q
  geo: Float64Array;              // geometric weight: 1 (4-nbhd); 1/2 sides, 1/(2 sqrt 2) diagonals (8-nbhd)
}

/** Neighbour pairs of a rows x cols grid. With 8 neighbours the weights follow
 *  Cauchy-Crofton (Boykov & Kolmogorov 2003), scaled so that a 4-neighbourhood has weight 1:
 *  a boundary then costs about the same in every direction. */
export function gridEdges(rows: number, cols: number, nbhd: Nbhd): GridEdges {
  const P: number[] = [], Q: number[] = [], G: number[] = [];
  const diag = 1 / (2 * Math.SQRT2);
  const side = nbhd === 4 ? 1 : 0.5;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const p = r * cols + c;
    if (c + 1 < cols) { P.push(p); Q.push(p + 1); G.push(side); }
    if (r + 1 < rows) { P.push(p); Q.push(p + cols); G.push(side); }
    if (nbhd === 8 && r + 1 < rows) {
      if (c + 1 < cols) { P.push(p); Q.push(p + cols + 1); G.push(diag); }
      if (c > 0) { P.push(p); Q.push(p + cols - 1); G.push(diag); }
    }
  }
  return { p: Int32Array.from(P), q: Int32Array.from(Q), geo: Float64Array.from(G) };
}

/** Contrast-sensitive weights w_pq = geo_pq * exp(-(I_p - I_q)^2 / (2 sigma^2)); sigma = Infinity gives w = geo. */
export function contrastWeights(I: ArrayLike<number>, E: GridEdges, sigma: number): Float64Array {
  const w = new Float64Array(E.p.length);
  for (let k = 0; k < w.length; k++) {
    const d = I[E.p[k]] - I[E.q[k]];
    w[k] = E.geo[k] * (Number.isFinite(sigma) ? Math.exp(-d * d / (2 * sigma * sigma)) : 1);
  }
  return w;
}

export interface SegProblem {
  n: number;
  D0: ArrayLike<number>;   // cost of labelling p background
  D1: ArrayLike<number>;   // cost of labelling p object
  E: GridEdges;
  w: ArrayLike<number>;
  lambda: number;
  seeds?: Int8Array;       // 1 = object seed, -1 = background seed, 0 = free
}

export interface EnergySplit { data: number; boundary: number; smooth: number; total: number }

/** Energy of a labelling; seeds are not part of the energy (the solver enforces them). */
export function segEnergy(P: SegProblem, L: ArrayLike<number>): EnergySplit {
  let data = 0, boundary = 0;
  for (let p = 0; p < P.n; p++) data += L[p] ? P.D1[p] : P.D0[p];
  for (let k = 0; k < P.E.p.length; k++) if (L[P.E.p[k]] !== L[P.E.q[k]]) boundary += P.w[k];
  const smooth = P.lambda * boundary;
  return { data, boundary, smooth, total: data + smooth };
}

export interface SegResult { L: Uint8Array; flow: number; energy: EnergySplit }

/** Minimise the energy exactly with one max-flow. Seeds become t-links that are never worth cutting. */
export function solveSeg(P: SegProblem): SegResult {
  const { n } = P;
  const g = new FlowGraph(n + 2, 2 * n + P.E.p.length);
  const s = n, t = n + 1;
  // "infinite" capacity: more than the cost of any labelling that respects the seeds
  let big = 1;
  for (let p = 0; p < n; p++) big += Math.max(P.D0[p], P.D1[p]);
  for (let k = 0; k < P.E.p.length; k++) big += P.lambda * P.w[k];
  let constant = 0;
  for (let p = 0; p < n; p++) {
    const sd = P.seeds ? P.seeds[p] : 0;
    let a = sd === 1 ? big : sd === -1 ? 0 : P.D0[p];   // s -> p
    let b = sd === -1 ? big : sd === 1 ? 0 : P.D1[p];   // p -> t
    if (sd === 1) constant += P.D1[p];
    if (sd === -1) constant += P.D0[p];
    // only the difference matters: send the common part straight through
    const m = Math.min(a, b);
    constant += m; a -= m; b -= m;
    if (a > 0) g.addEdge(s, p, a);
    if (b > 0) g.addEdge(p, t, b);
  }
  for (let k = 0; k < P.E.p.length; k++) {
    const c = P.lambda * P.w[k];
    if (c > 0) g.addEdge(P.E.p[k], P.E.q[k], c, c);
  }
  const f = g.maxflow(s, t);
  const S = g.sourceSide(s);
  const L = new Uint8Array(n);
  for (let p = 0; p < n; p++) L[p] = S[p];
  return { L, flow: f + constant, energy: segEnergy(P, L) };
}

/** Labelling with lambda = 0 (each pixel takes its cheaper label; ties go to background), seeds respected. */
export function thresholdLabels(P: SegProblem): Uint8Array {
  const L = new Uint8Array(P.n);
  for (let p = 0; p < P.n; p++) {
    const sd = P.seeds ? P.seeds[p] : 0;
    L[p] = sd === 1 ? 1 : sd === -1 ? 0 : (P.D1[p] < P.D0[p] ? 1 : 0);
  }
  return L;
}

/** Gaussian data costs -log P(I | label) up to a shared constant: (I - mu)^2 / (2 s^2). */
export function gaussCosts(I: ArrayLike<number>, mu0: number, mu1: number, s: number): { D0: Float64Array; D1: Float64Array } {
  const n = I.length, D0 = new Float64Array(n), D1 = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    D0[p] = (I[p] - mu0) ** 2 / (2 * s * s);
    D1[p] = (I[p] - mu1) ** 2 / (2 * s * s);
  }
  return { D0, D1 };
}
