// RANSAC and randomised Hough (no DOM). A generic hypothesise-and-verify loop over a model
// interface, line and circle models with exact minimal fits and least-squares refits
// (total least squares for a line, the algebraic circle fit), the iteration formula, a
// step recorder for the demo's step-through, sequential RANSAC for several models, and
// randomised Hough voting into a sparse map. Seeded through lib/random.ts.
// Owner: the Hough day (ransac/).
//
// Points are {x, y} in any Cartesian frame. A line is {theta, rho}: x cos(theta) +
// y sin(theta) = rho (normal form, as in lib/hough.ts); a circle is {a, b, r}.

import { rng, Rng, sampleIndices } from "./random";
import { eigSym2 } from "./linalg";

export interface P2 { x: number; y: number }

export interface Model<M> {
  name: string;
  /** Minimal sample size. */
  s: number;
  /** Exact fit through a minimal sample; null if degenerate. */
  fit(pts: P2[]): M | null;
  /** Distance of a point to the model (geometric). */
  distance(m: M, p: P2): number;
  /** Least-squares fit to many points; null if degenerate. */
  refit(pts: P2[]): M | null;
}

export interface Line { theta: number; rho: number }
export interface Circle { a: number; b: number; r: number }

/** Normal form of the line through two points, theta in [0, pi). */
export function lineFrom2(p: P2, q: P2): Line | null {
  const dx = q.x - p.x, dy = q.y - p.y;
  if (Math.hypot(dx, dy) < 1e-9) return null;
  let theta = Math.atan2(dx, -dy), rho = p.x * Math.cos(theta) + p.y * Math.sin(theta);
  if (theta < 0) { theta += Math.PI; rho = -rho; }
  if (theta >= Math.PI) { theta -= Math.PI; rho = -rho; }
  return { theta, rho };
}

/**
 * Total least squares: the normal is the eigenvector of the scatter matrix with the smaller
 * eigenvalue, and the line passes through the centroid (PCA of the points).
 */
export function lineTLS(pts: P2[]): Line | null {
  const n = pts.length;
  if (n < 2) return null;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) { const u = p.x - mx, v = p.y - my; sxx += u * u; sxy += u * v; syy += v * v; }
  const e = eigSym2(sxx, sxy, syy);
  // eigSym2 returns the angle of the eigenvector of the larger eigenvalue; the normal is perpendicular
  let theta = e.angle + Math.PI / 2;
  let rho = mx * Math.cos(theta) + my * Math.sin(theta);
  theta %= 2 * Math.PI; if (theta < 0) theta += 2 * Math.PI;
  if (theta >= Math.PI) { theta -= Math.PI; rho = -rho; }
  return { theta, rho };
}

/** Ordinary least squares y = m x + c (vertical residuals), as a normal-form line (the baseline the demo compares with). */
export function lineOLS(pts: P2[]): Line | null {
  const n = pts.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of pts) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
  const den = n * sxx - sx * sx;
  if (Math.abs(den) < 1e-12) return null;
  const m = (n * sxy - sx * sy) / den, c = (sy - m * sx) / n;
  return lineFrom2({ x: 0, y: c }, { x: 1, y: m + c });
}

export const lineDistance = (m: Line, p: P2): number => Math.abs(p.x * Math.cos(m.theta) + p.y * Math.sin(m.theta) - m.rho);

/** Circle through three points (circumcircle); null if (nearly) collinear. */
export function circleFrom3(p: P2, q: P2, t: P2): Circle | null {
  const ax = q.x - p.x, ay = q.y - p.y, bx = t.x - p.x, by = t.y - p.y;
  const d = 2 * (ax * by - ay * bx);
  if (Math.abs(d) < 1e-9) return null;
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
  const ux = (by * a2 - ay * b2) / d, uy = (ax * b2 - bx * a2) / d;
  return { a: p.x + ux, b: p.y + uy, r: Math.hypot(ux, uy) };
}

/**
 * Algebraic circle fit: x^2 + y^2 = 2 a x + 2 b y + c with c = r^2 - a^2 - b^2 is linear in
 * (a, b, c); solve the 3 x 3 normal equations.
 */
export function circleAlgebraic(pts: P2[]): Circle | null {
  if (pts.length < 3) return null;
  const M = [0, 0, 0, 0, 0, 0, 0, 0, 0], v = [0, 0, 0];
  for (const p of pts) {
    const row = [2 * p.x, 2 * p.y, 1], z = p.x * p.x + p.y * p.y;
    for (let i = 0; i < 3; i++) { v[i] += row[i] * z; for (let j = 0; j < 3; j++) M[3 * i + j] += row[i] * row[j]; }
  }
  const sol = solve3(M, v);
  if (!sol) return null;
  const [a, b, c] = sol, r2 = c + a * a + b * b;
  if (!(r2 > 0)) return null;
  return { a, b, r: Math.sqrt(r2) };
}

function solve3(M: number[], v: number[]): number[] | null {
  const A = M.slice(), b = v.slice();
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(A[3 * r + c]) > Math.abs(A[3 * piv + c])) piv = r;
    if (Math.abs(A[3 * piv + c]) < 1e-12) return null;
    if (piv !== c) { for (let k = 0; k < 3; k++) [A[3 * c + k], A[3 * piv + k]] = [A[3 * piv + k], A[3 * c + k]]; [b[c], b[piv]] = [b[piv], b[c]]; }
    for (let r = 0; r < 3; r++) if (r !== c) {
      const f = A[3 * r + c] / A[3 * c + c];
      for (let k = 0; k < 3; k++) A[3 * r + k] -= f * A[3 * c + k];
      b[r] -= f * b[c];
    }
  }
  return [b[0] / A[0], b[1] / A[4], b[2] / A[8]];
}

export const circleDistance = (m: Circle, p: P2): number => Math.abs(Math.hypot(p.x - m.a, p.y - m.b) - m.r);

export const LINE_MODEL: Model<Line> = { name: "line", s: 2, fit: ([p, q]) => lineFrom2(p, q), distance: lineDistance, refit: lineTLS };
export const CIRCLE_MODEL: Model<Circle> = { name: "circle", s: 3, fit: ([p, q, t]) => circleFrom3(p, q, t), distance: circleDistance, refit: circleAlgebraic };

// ---------------------------------------------------------------------------------------
// The loop

export interface RansacStep<M> {
  iter: number;
  sample: number[];
  model: M | null;
  inliers: number;
  /** Best inlier count so far (after this step). */
  best: number;
  improved: boolean;
}

export interface RansacResult<M> {
  best: M | null;
  bestSample: number[];
  inliers: number[];
  refit: M | null;
  /** Inliers of the refitted model (with the same epsilon). */
  refitInliers: number[];
  steps: RansacStep<M>[];
  /** Iteration at which the final best model was found (1-based; 0 if none). */
  foundAt: number;
}

export interface RansacOptions {
  eps: number;
  iterations: number;
  seed?: number;
  /** Indices to use (default all): sequential RANSAC passes the remaining points. */
  active?: number[];
}

/** Indices of the points within eps of the model. */
export function inliersOf<M>(model: Model<M>, m: M, pts: P2[], eps: number, active?: number[]): number[] {
  const out: number[] = [];
  const idx = active ?? pts.map((_, i) => i);
  for (const i of idx) if (model.distance(m, pts[i]) <= eps) out.push(i);
  return out;
}

/**
 * RANSAC: `iterations` times, draw a minimal sample (without replacement, seeded), fit it
 * exactly, count the inliers within eps; keep the model with the most (first one wins
 * ties). Then refit by least squares on its inliers. Every step is recorded.
 */
export function ransac<M>(model: Model<M>, pts: P2[], o: RansacOptions): RansacResult<M> {
  const r = rng(o.seed ?? 1), idx = o.active ?? pts.map((_, i) => i);
  const steps: RansacStep<M>[] = [];
  let best: M | null = null, bestN = -1, bestSample: number[] = [], foundAt = 0;
  for (let it = 1; it <= o.iterations; it++) {
    if (idx.length < model.s) break;
    const sample = drawSample(r, idx, model.s);
    const m = model.fit(sample.map(i => pts[i]));
    const n = m ? inliersOf(model, m, pts, o.eps, idx).length : 0;
    const improved = m !== null && n > bestN;
    if (improved) { best = m; bestN = n; bestSample = sample; foundAt = it; }
    steps.push({ iter: it, sample, model: m, inliers: n, best: Math.max(bestN, 0), improved });
  }
  const inl = best ? inliersOf(model, best, pts, o.eps, idx) : [];
  const ref = inl.length >= model.s ? model.refit(inl.map(i => pts[i])) : null;
  const refIn = ref ? inliersOf(model, ref, pts, o.eps, idx) : [];
  return { best, bestSample, inliers: inl, refit: ref, refitInliers: refIn, steps, foundAt };
}

/** A minimal sample of k distinct indices from idx. */
export function drawSample(r: Rng, idx: number[], k: number): number[] {
  return Array.from(sampleIndices(r, idx.length, k), i => idx[i]);
}

/** Sequential RANSAC: find a model, remove its inliers, repeat (up to maxModels, while >= minInliers). */
export function sequentialRansac<M>(model: Model<M>, pts: P2[], o: RansacOptions & { maxModels: number; minInliers: number }): RansacResult<M>[] {
  let active = o.active ?? pts.map((_, i) => i);
  const out: RansacResult<M>[] = [];
  for (let k = 0; k < o.maxModels; k++) {
    const res = ransac(model, pts, { ...o, active, seed: (o.seed ?? 1) + 7919 * k });
    if (!res.best || res.inliers.length < o.minInliers) break;
    out.push(res);
    const used = new Set(res.refitInliers.length >= res.inliers.length ? res.refitInliers : res.inliers);
    active = active.filter(i => !used.has(i));
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// How many iterations?

/** Probability that at least one of N samples of size s is all inliers, with inlier fraction w. */
export const successProbability = (w: number, s: number, N: number): number => 1 - Math.pow(1 - Math.pow(w, s), N);

/** Iterations needed for success probability p: N = log(1 - p) / log(1 - w^s), rounded up. */
export function iterationsNeeded(w: number, s: number, p = 0.99): number {
  const q = Math.pow(w, s);
  if (q >= 1) return 1;
  if (q <= 0) return Infinity;
  return Math.ceil(Math.log(1 - p) / Math.log(1 - q));
}

// ---------------------------------------------------------------------------------------
// Randomised Hough (Xu, Oja & Kultanen 1990): one vote per minimal sample, sparse accumulator

export interface SparseVotes { counts: Map<string, number>; best: { key: string; count: number; theta: number; rho: number } | null; samples: number }

/**
 * Randomised Hough for lines: draw pairs, compute (theta, rho), quantise with (dTheta, dRho)
 * and add one vote to that cell of a sparse map. Returns the map and the strongest cell.
 */
export function randomisedHoughLines(pts: P2[], samples: number, dTheta: number, dRho: number, seed = 1): SparseVotes {
  const r = rng(seed), idx = pts.map((_, i) => i), counts = new Map<string, number>();
  let best: SparseVotes["best"] = null;
  for (let t = 0; t < samples; t++) {
    const [i, j] = drawSample(r, idx, 2);
    const L = lineFrom2(pts[i], pts[j]);
    if (!L) continue;
    const a = Math.round(L.theta / dTheta) % Math.round(Math.PI / dTheta), b = Math.round(L.rho / dRho);
    const key = a + "," + b, c = (counts.get(key) ?? 0) + 1;
    counts.set(key, c);
    if (!best || c > best.count) best = { key, count: c, theta: a * dTheta, rho: b * dRho };
  }
  return { counts, best, samples };
}

// ---------------------------------------------------------------------------------------
// Synthetic point sets (seeded)

export interface PointSet { pts: P2[]; inlier: boolean[]; truth: (Line | Circle)[] }

/**
 * n points in the box [0, size]^2: a fraction (1 - outlierFrac) on the given lines/circles
 * (split evenly, Gaussian noise sd perpendicular), the rest uniform outliers.
 */
export function makePointSet(n: number, outlierFrac: number, models: ({ kind: "line" } & Line | { kind: "circle" } & Circle)[], sd: number, size: number, seed = 1): PointSet {
  const r = rng(seed), pts: P2[] = [], inlier: boolean[] = [];
  const nIn = Math.round(n * (1 - outlierFrac));
  for (let i = 0; i < nIn; i++) {
    const m = models[i % models.length];
    let p: P2;
    if (m.kind === "line") {
      // a point on the visible part of the line, plus noise along the normal
      const c = Math.cos(m.theta), s = Math.sin(m.theta);
      let tries = 0;
      do {
        const t = (r.unif() - 0.5) * size * 1.5, e = sd * r.norm();
        p = { x: (m.rho + e) * c - t * s, y: (m.rho + e) * s + t * c };
      } while ((p.x < 0 || p.y < 0 || p.x > size || p.y > size) && ++tries < 200);
    } else {
      const ang = 2 * Math.PI * r.unif(), rr = m.r + sd * r.norm();
      p = { x: m.a + rr * Math.cos(ang), y: m.b + rr * Math.sin(ang) };
    }
    pts.push(p); inlier.push(true);
  }
  for (let i = nIn; i < n; i++) { pts.push({ x: size * r.unif(), y: size * r.unif() }); inlier.push(false); }
  return { pts, inlier, truth: models.map(m => (m.kind === "line" ? { theta: m.theta, rho: m.rho } : { a: m.a, b: m.b, r: m.r })) };
}
