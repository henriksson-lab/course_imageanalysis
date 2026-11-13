// Spline curves and spline fitting (owner: the splines day). No DOM.
//
// - Polynomial interpolation: equispaced and Chebyshev nodes, barycentric Lagrange form.
// - Bézier curves: Bernstein basis, de Casteljau with all levels, split, derivative, rational.
// - B-splines: knot vectors (open uniform, clamped, closed/periodic), Cox–de Boor (single
//   function, the full tower of intermediate values, all non-zero values + derivatives),
//   curve evaluation with optional weights (NURBS: lift to homogeneous coordinates, evaluate,
//   divide), affected spans, curvature, arc length and total curvature (Gauss–Legendre).
// - Uniform B-spline kernels β⁰…β³ (centred) and β³'; `bspline3` is the cubic used by image
//   interpolation and free-form deformations (lib/splineimage.ts).
// - Fitting: natural cubic interpolating spline, least-squares regression spline, cubic
//   smoothing spline (Reinsch: data + λ∫f''²) with hat matrix, leave-one-out and GCV, the
//   discrete second-difference smoother, and the closed (periodic) P-spline used for outlines.
// Convention: points are [x, y]; parameters t increase along the curve.
import { thomas } from "./smooth";
import { inverse, solve } from "./linalg";

export type Pt = [number, number];

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const lerpPt = (a: Pt, b: Pt, t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

// ---------------------------------------------------------------------------------------
// Polynomial interpolation

/** n equispaced nodes on [a, b] (end points included). */
export function equispacedNodes(n: number, a = -1, b = 1): number[] {
  if (n === 1) return [(a + b) / 2];
  return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
}

/** n Chebyshev nodes on [a, b] (roots of T_n, bunched towards the ends), ascending. */
export function chebyshevNodes(n: number, a = -1, b = 1): number[] {
  return Array.from({ length: n }, (_, i) => (a + b) / 2 - (b - a) / 2 * Math.cos((2 * i + 1) * Math.PI / (2 * n)));
}

/** The interpolating polynomial through (xs, ys), evaluated stably in barycentric form. */
export function polyInterp(xs: ArrayLike<number>, ys: ArrayLike<number>): (x: number) => number {
  const n = xs.length, w = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let p = 1;
    for (let k = 0; k < n; k++) if (k !== j) p *= xs[j] - xs[k];
    w[j] = 1 / p;
  }
  return (x: number) => {
    let num = 0, den = 0;
    for (let j = 0; j < n; j++) {
      const d = x - xs[j];
      if (d === 0) return ys[j];
      const q = w[j] / d;
      num += q * ys[j]; den += q;
    }
    return num / den;
  };
}

/** Lagrange basis polynomial L_i at x. */
export function lagrangeBasis(xs: ArrayLike<number>, i: number, x: number): number {
  let p = 1;
  for (let j = 0; j < xs.length; j++) if (j !== i) p *= (x - xs[j]) / (xs[i] - xs[j]);
  return p;
}

/** Piecewise-linear interpolation through sorted xs. */
export function linearInterp(xs: ArrayLike<number>, ys: ArrayLike<number>, x: number): number {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (i < n - 2 && x > xs[i + 1]) i++;
  const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
  return ys[i] + (ys[i + 1] - ys[i]) * t;
}

// ---------------------------------------------------------------------------------------
// Bézier curves

export function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 1; i <= k; i++) r = r * (n - k + i) / i;
  return r;
}

/** Bernstein polynomial b_{i,n}(t) = C(n, i) tⁱ (1 − t)ⁿ⁻ⁱ. */
export function bernstein(n: number, i: number, t: number): number {
  return binom(n, i) * Math.pow(t, i) * Math.pow(1 - t, n - i);
}

/** De Casteljau at t: levels[0] = control points, …, levels[n] = [the curve point]. */
export function deCasteljau(P: Pt[], t: number): Pt[][] {
  const levels: Pt[][] = [P.map(p => [p[0], p[1]] as Pt)];
  while (levels[levels.length - 1].length > 1) {
    const L = levels[levels.length - 1], nx: Pt[] = [];
    for (let i = 0; i + 1 < L.length; i++) nx.push(lerpPt(L[i], L[i + 1], t));
    levels.push(nx);
  }
  return levels;
}

export function bezierPoint(P: Pt[], t: number): Pt {
  const L = deCasteljau(P, t);
  return L[L.length - 1][0];
}

/** Bernstein sum Σ b_{i,n}(t) P_i (reference for de Casteljau). */
export function bezierBernstein(P: Pt[], t: number): Pt {
  const n = P.length - 1;
  let x = 0, y = 0;
  for (let i = 0; i <= n; i++) { const b = bernstein(n, i, t); x += b * P[i][0]; y += b * P[i][1]; }
  return [x, y];
}

/** Control points of the derivative curve: n (P_{i+1} − P_i), degree n − 1. */
export function bezierDerivPts(P: Pt[]): Pt[] {
  const n = P.length - 1, out: Pt[] = [];
  for (let i = 0; i < n; i++) out.push([n * (P[i + 1][0] - P[i][0]), n * (P[i + 1][1] - P[i][1])]);
  return out;
}

/** First and second derivative of a Bézier curve at t. */
export function bezierDerivs(P: Pt[], t: number): [Pt, Pt] {
  const D1 = bezierDerivPts(P);
  const d1 = D1.length ? bezierPoint(D1, t) : [0, 0] as Pt;
  const D2 = D1.length > 1 ? bezierDerivPts(D1) : [];
  const d2 = D2.length ? bezierPoint(D2, t) : [0, 0] as Pt;
  return [d1, d2];
}

/** Split at t into two Bézier curves (the outer points of the de Casteljau construction). */
export function bezierSplit(P: Pt[], t: number): [Pt[], Pt[]] {
  const L = deCasteljau(P, t);
  return [L.map(l => l[0]), L.map(l => l[l.length - 1]).reverse()];
}

/** Rational Bézier point: Σ w_i b_i P_i / Σ w_i b_i. */
export function rationalBezier(P: Pt[], w: number[], t: number): Pt {
  const n = P.length - 1;
  let x = 0, y = 0, s = 0;
  for (let i = 0; i <= n; i++) { const b = bernstein(n, i, t) * w[i]; x += b * P[i][0]; y += b * P[i][1]; s += b; }
  return [x / s, y / s];
}

// ---------------------------------------------------------------------------------------
// Uniform B-spline kernels (centred at 0, support [−(p+1)/2, (p+1)/2])

export function bspline0(x: number): number { const a = Math.abs(x); return a < 0.5 ? 1 : a === 0.5 ? 0.5 : 0; }
export function bspline1(x: number): number { const a = Math.abs(x); return a < 1 ? 1 - a : 0; }
export function bspline2(x: number): number {
  const a = Math.abs(x);
  if (a < 0.5) return 0.75 - a * a;
  if (a < 1.5) return 0.5 * (1.5 - a) * (1.5 - a);
  return 0;
}
/** Cubic B-spline β³: 2/3 − x² + |x|³/2 on |x| ≤ 1, (2 − |x|)³/6 on 1 ≤ |x| ≤ 2. */
export function bspline3(x: number): number {
  const a = Math.abs(x);
  if (a < 1) return 2 / 3 - a * a + a * a * a / 2;
  if (a < 2) { const b = 2 - a; return b * b * b / 6; }
  return 0;
}
/** Derivative of β³. */
export function bspline3d(x: number): number {
  const a = Math.abs(x), s = x < 0 ? -1 : 1;
  if (a < 1) return s * (-2 * a + 1.5 * a * a);
  if (a < 2) return s * (-0.5 * (2 - a) * (2 - a));
  return 0;
}
/** Second derivative of β³. */
export function bspline3dd(x: number): number {
  const a = Math.abs(x);
  if (a < 1) return -2 + 3 * a;
  if (a < 2) return 2 - a;
  return 0;
}
export function bsplineKernel(p: number, x: number): number {
  return p === 0 ? bspline0(x) : p === 1 ? bspline1(x) : p === 2 ? bspline2(x) : bspline3(x);
}

// ---------------------------------------------------------------------------------------
// Knot vectors and the B-spline basis

export type KnotKind = "open" | "clamped" | "closed";

/** Knot vector for n control points of degree p. Open: 0, 1, …, n + p (domain [p, n]).
 *  Clamped: p + 1 equal end knots (domain [0, n − p]). Closed: the control polygon is wrapped
 *  (p points repeated, see `curveFrom`), knots 0 … n + 2p, domain [p, n + p]. */
export function makeKnots(n: number, p: number, kind: KnotKind): number[] {
  if (kind === "open") return Array.from({ length: n + p + 1 }, (_, i) => i);
  if (kind === "closed") return Array.from({ length: n + 2 * p + 1 }, (_, i) => i);
  const k: number[] = [];
  for (let i = 0; i <= p; i++) k.push(0);
  for (let i = 1; i < n - p; i++) k.push(i);
  for (let i = 0; i <= p; i++) k.push(Math.max(n - p, 0));
  return k;
}

/** Valid parameter range [t_p, t_{m−p}] for a knot vector of length m + 1. */
export function knotDomain(knots: ArrayLike<number>, p: number): [number, number] {
  return [knots[p], knots[knots.length - 1 - p]];
}

/** N_{i,p}(t) by the Cox–de Boor recursion (0/0 := 0). The last non-empty interval is closed
 *  on the right, so the basis sums to 1 on the whole domain including its end. */
export function basisN(i: number, p: number, knots: ArrayLike<number>, t: number): number {
  return coxDeBoorTower(i, p, knots, t)[p][0];
}

/** The Cox–de Boor tower for N_{i,p}: tower[q][j] = N_{i+j,q}(t), q = 0…p, j = 0…p − q. */
export function coxDeBoorTower(i: number, p: number, knots: ArrayLike<number>, t: number): number[][] {
  const m = knots.length - 1;
  const tEnd = knots[m];
  // index of the last non-empty interval (to close it on the right)
  let last = m - 1;
  while (last > 0 && knots[last] === knots[last + 1]) last--;
  const tower: number[][] = [];
  const base: number[] = [];
  for (let j = 0; j <= p; j++) {
    const k = i + j;
    if (k >= m) { base.push(0); continue; }
    const a = knots[k], b = knots[k + 1];
    base.push((t >= a && t < b) || (t === tEnd && k === last && a < b) ? 1 : 0);
  }
  tower.push(base);
  for (let q = 1; q <= p; q++) {
    const prev = tower[q - 1], row: number[] = [];
    for (let j = 0; j <= p - q; j++) {
      const k = i + j;
      const d1 = knots[k + q] - knots[k], d2 = knots[k + q + 1] - knots[k + 1];
      const left = d1 > 0 ? (t - knots[k]) / d1 * prev[j] : 0;
      const right = d2 > 0 ? (knots[k + q + 1] - t) / d2 * prev[j + 1] : 0;
      row.push(left + right);
    }
    tower.push(row);
  }
  return tower;
}

/** Knot span index s with t_s ≤ t < t_{s+1}, p ≤ s ≤ nCtrl − 1. */
export function findSpan(knots: ArrayLike<number>, p: number, nCtrl: number, t: number): number {
  const hi = nCtrl - 1;
  if (t >= knots[hi + 1]) {
    let s = hi;
    while (s > p && knots[s] === knots[s + 1]) s--;
    return s;
  }
  if (t <= knots[p]) {
    let s = p;
    while (s < hi && knots[s + 1] <= t) s++;
    return s;
  }
  let lo = p, up = hi + 1, mid = (lo + up) >> 1;
  while (t < knots[mid] || t >= knots[mid + 1]) {
    if (t < knots[mid]) up = mid; else lo = mid;
    mid = (lo + up) >> 1;
  }
  return mid;
}

/** All non-zero basis functions and their derivatives up to order nd at t (The NURBS Book,
 *  A2.3): ders[k][j] = d^k/dt^k N_{span−p+j, p}(t). */
export function dersBasisFuns(span: number, t: number, p: number, nd: number, knots: ArrayLike<number>): number[][] {
  const ndu: number[][] = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
  const left = new Array(p + 1).fill(0), right = new Array(p + 1).fill(0);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = t - knots[span + 1 - j];
    right[j] = knots[span + j] - t;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  const ders: number[][] = Array.from({ length: nd + 1 }, () => new Array(p + 1).fill(0));
  for (let j = 0; j <= p; j++) ders[0][j] = ndu[j][p];
  const a: number[][] = [new Array(p + 1).fill(0), new Array(p + 1).fill(0)];
  for (let r = 0; r <= p; r++) {
    let s1 = 0, s2 = 1;
    a[0][0] = 1;
    for (let k = 1; k <= nd; k++) {
      let d = 0;
      const rk = r - k, pk = p - k;
      if (r >= k) { a[s2][0] = a[s1][0] / ndu[pk + 1][rk]; d = a[s2][0] * ndu[rk][pk]; }
      const j1 = rk >= -1 ? 1 : -rk, j2 = r - 1 <= pk ? k - 1 : p - r;
      for (let j = j1; j <= j2; j++) { a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]; d += a[s2][j] * ndu[rk + j][pk]; }
      if (r <= pk) { a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]; d += a[s2][k] * ndu[r][pk]; }
      ders[k][r] = d;
      const tmp = s1; s1 = s2; s2 = tmp;
    }
  }
  let r = p;
  for (let k = 1; k <= nd; k++) { for (let j = 0; j <= p; j++) ders[k][j] *= r; r *= p - k; }
  return ders;
}

// ---------------------------------------------------------------------------------------
// B-spline and NURBS curves

export interface SplineCurve {
  /** Control points actually used (for a closed curve: the wrapped list). */
  P: Pt[];
  /** Weights (NURBS); all 1 when omitted. */
  w?: number[];
  p: number;
  knots: number[];
  /** Number of distinct (user) control points; for a closed curve P.length − p. */
  nUser: number;
  kind: KnotKind;
}

/** A B-spline (or NURBS when w is given) curve of degree p through the user control points. */
export function curveFrom(P: Pt[], p: number, kind: KnotKind, w?: number[]): SplineCurve {
  const n = P.length;
  p = Math.max(0, Math.min(p, kind === "closed" ? n - 1 : n - 1));
  if (kind === "closed") {
    const Pw = [...P, ...P.slice(0, p)];
    const ww = w ? [...w, ...w.slice(0, p)] : undefined;
    return { P: Pw, w: ww, p, knots: makeKnots(n, p, "closed"), nUser: n, kind };
  }
  return { P: P.slice(), w: w?.slice(), p, knots: makeKnots(n, p, kind), nUser: n, kind };
}

export function curveDomain(c: SplineCurve): [number, number] { return knotDomain(c.knots, c.p); }

/** Point and derivatives up to order nd (≤ 2) at t. Rational curves: evaluate the lifted
 *  (homogeneous) curve (w x, w y, w) and its derivatives, then apply the quotient rule. */
export function curveDerivs(c: SplineCurve, t: number, nd = 2): Pt[] {
  const { P, p, knots } = c, n = P.length;
  const s = findSpan(knots, p, n, t);
  const D = dersBasisFuns(s, t, p, Math.min(nd, p), knots);
  const A: [number, number, number][] = [];
  for (let k = 0; k <= nd; k++) {
    let x = 0, y = 0, W = 0;
    if (k <= p) for (let j = 0; j <= p; j++) {
      const i = s - p + j, wi = c.w ? c.w[i] : 1, b = D[k][j] * wi;
      x += b * P[i][0]; y += b * P[i][1]; W += b;
    }
    A.push([x, y, W]);
  }
  const out: Pt[] = [];
  const w0 = A[0][2];
  const C0: Pt = [A[0][0] / w0, A[0][1] / w0];
  out.push(C0);
  if (nd >= 1) {
    const w1 = A[1][2];
    const C1: Pt = [(A[1][0] - w1 * C0[0]) / w0, (A[1][1] - w1 * C0[1]) / w0];
    out.push(C1);
    if (nd >= 2) {
      const w2 = A[2][2];
      out.push([(A[2][0] - 2 * w1 * C1[0] - w2 * C0[0]) / w0, (A[2][1] - 2 * w1 * C1[1] - w2 * C0[1]) / w0]);
    }
  }
  return out;
}

export function curvePoint(c: SplineCurve, t: number): Pt { return curveDerivs(c, t, 0)[0]; }

/** Point by direct summation Σ N_i(t) w_i P_i / Σ N_i(t) w_i with Cox–de Boor (reference). */
export function curvePointRef(c: SplineCurve, t: number): Pt {
  let x = 0, y = 0, W = 0;
  for (let i = 0; i < c.P.length; i++) {
    const b = basisN(i, c.p, c.knots, t) * (c.w ? c.w[i] : 1);
    x += b * c.P[i][0]; y += b * c.P[i][1]; W += b;
  }
  return [x / W, y / W];
}

/** De Boor's algorithm (polynomial curves): the triangular scheme of lerps at t. */
export function deBoor(c: SplineCurve, t: number): Pt {
  const { P, p, knots } = c;
  const s = findSpan(knots, p, P.length, t);
  const d: Pt[] = [];
  for (let j = 0; j <= p; j++) d.push([...P[j + s - p]] as Pt);
  for (let r = 1; r <= p; r++) for (let j = p; j >= r; j--) {
    const den = knots[j + 1 + s - r] - knots[j + s - p];
    const a = den > 0 ? (t - knots[j + s - p]) / den : 0;
    d[j] = lerpPt(d[j - 1], d[j], a);
  }
  return d[p];
}

/** Signed curvature of a parametric curve from its first two derivatives. */
export function curvature(d1: Pt, d2: Pt): number {
  const sp2 = d1[0] * d1[0] + d1[1] * d1[1];
  if (sp2 === 0) return 0;
  return (d1[0] * d2[1] - d1[1] * d2[0]) / Math.pow(sp2, 1.5);
}

export interface CurveSample { t: number; pt: Pt; d1: Pt; d2: Pt; kappa: number }

/** n + 1 samples per knot span (spans of zero length skipped). */
export function sampleCurve(c: SplineCurve, perSpan = 24): CurveSample[] {
  const [a, b] = curveDomain(c), out: CurveSample[] = [];
  const ks = c.knots.filter((v, i) => v >= a && v <= b && (i === 0 || v !== c.knots[i - 1]));
  for (let s = 0; s + 1 < ks.length; s++) {
    for (let j = 0; j <= perSpan; j++) {
      if (j === 0 && s > 0) continue;
      const t = ks[s] + (ks[s + 1] - ks[s]) * j / perSpan;
      const [pt, d1, d2] = curveDerivs(c, t, 2);
      out.push({ t, pt, d1, d2, kappa: curvature(d1, d2) });
    }
  }
  return out;
}

const GL_X = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640];
const GL_W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];

/** ∫ f over the curve's domain, 5-point Gauss–Legendre per knot span split in `sub` parts. */
function integrate(c: SplineCurve, f: (d1: Pt, d2: Pt) => number, sub = 2): number {
  const [a, b] = curveDomain(c);
  const ks = c.knots.filter((v, i) => v >= a && v <= b && (i === 0 || v !== c.knots[i - 1]));
  let sum = 0;
  for (let s = 0; s + 1 < ks.length; s++) for (let q = 0; q < sub; q++) {
    const lo = ks[s] + (ks[s + 1] - ks[s]) * q / sub, hi = ks[s] + (ks[s + 1] - ks[s]) * (q + 1) / sub;
    const h = (hi - lo) / 2, m = (hi + lo) / 2;
    for (let g = 0; g < 5; g++) {
      const [, d1, d2] = curveDerivs(c, m + h * GL_X[g], 2);
      sum += GL_W[g] * h * f(d1, d2);
    }
  }
  return sum;
}

/** Arc length ∫‖C'(t)‖ dt. */
export function arcLength(c: SplineCurve, sub = 2): number {
  return integrate(c, d1 => Math.hypot(d1[0], d1[1]), sub);
}

/** Total signed curvature ∮ κ ds = ∫ (x'y'' − y'x'')/‖C'‖² dt (2π for a simple closed curve
 *  traversed counter-clockwise in a y-up frame). */
export function totalCurvature(c: SplineCurve, sub = 4): number {
  return integrate(c, (d1, d2) => {
    const s2 = d1[0] * d1[0] + d1[1] * d1[1];
    return s2 > 0 ? (d1[0] * d2[1] - d1[1] * d2[0]) / s2 : 0;
  }, sub);
}

/** Parameter interval in which user control point i has influence (closed curves: may be
 *  two intervals because of the wrap). */
export function influence(c: SplineCurve, i: number): [number, number][] {
  const [a, b] = curveDomain(c), out: [number, number][] = [];
  const idx = c.kind === "closed" ? [i, i + c.nUser].filter(k => k < c.P.length) : [i];
  for (const k of idx) {
    const lo = Math.max(a, c.knots[k]), hi = Math.min(b, c.knots[k + c.p + 1]);
    if (hi > lo) out.push([lo, hi]);
  }
  return out;
}

/** The standard NURBS circle: 9 control points (square corners and edge midpoints), weights
 *  1 and 1/√2 alternating, degree 2, knots 0,0,0,¼,¼,½,½,¾,¾,1,1,1 (scaled to 0…4). */
export function nurbsCircle(cx = 0, cy = 0, r = 1): SplineCurve {
  const s = Math.SQRT1_2;
  const P: Pt[] = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]].map(([x, y]) => [cx + r * x, cy + r * y] as Pt);
  const w = [1, s, 1, s, 1, s, 1, s, 1];
  return { P, w, p: 2, knots: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], nUser: 9, kind: "clamped" };
}

/** A quarter circle as a rational quadratic Bézier: (1,0), (1,1), (0,1) with weights 1, w, 1. */
export function quarterCircle(w = Math.SQRT1_2): SplineCurve {
  return curveFrom([[1, 0], [1, 1], [0, 1]], 2, "clamped", [1, w, 1]);
}

// ---------------------------------------------------------------------------------------
// Natural cubic interpolating spline

export interface CubicSpline {
  xs: Float64Array; ys: Float64Array;
  /** Second derivatives at the nodes (M_0 = M_{n−1} = 0 for the natural spline). */
  M: Float64Array;
  at(x: number): number;
  d1(x: number): number;
  d2(x: number): number;
}

/** A cubic spline from node values and node second derivatives (piecewise cubic, C²). */
export function cubicFromMoments(xs: ArrayLike<number>, ys: ArrayLike<number>, M: ArrayLike<number>): CubicSpline {
  const X = Float64Array.from(xs), Y = Float64Array.from(ys), MM = Float64Array.from(M), n = X.length;
  const seg = (x: number) => {
    let i = 0;
    if (x >= X[n - 1]) i = n - 2;
    else if (x > X[0]) { let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (X[m] <= x) lo = m; else hi = m; } i = lo; }
    return Math.max(0, Math.min(n - 2, i));
  };
  const parts = (x: number) => {
    const i = seg(x), h = X[i + 1] - X[i], a = (X[i + 1] - x) / h, b = (x - X[i]) / h;
    return { i, h, a, b };
  };
  const d1 = (x: number): number => {
    if (n === 1) return 0;
    x = Math.min(Math.max(x, X[0]), X[n - 1]);
    const { i, h, a, b } = parts(x);
    return (Y[i + 1] - Y[i]) / h - (3 * a * a - 1) * h / 6 * MM[i] + (3 * b * b - 1) * h / 6 * MM[i + 1];
  };
  const d2 = (x: number): number => {
    if (n === 1 || x < X[0] || x > X[n - 1]) return 0;
    const { i, a, b } = parts(x);
    return a * MM[i] + b * MM[i + 1];
  };
  const at = (x: number): number => {
    if (n === 1) return Y[0];
    // outside the nodes: linear continuation (a natural spline is linear there)
    if (x < X[0]) return Y[0] + (x - X[0]) * d1(X[0]);
    if (x > X[n - 1]) return Y[n - 1] + (x - X[n - 1]) * d1(X[n - 1]);
    const { i, h, a, b } = parts(x);
    return a * Y[i] + b * Y[i + 1] + ((a * a * a - a) * MM[i] + (b * b * b - b) * MM[i + 1]) * h * h / 6;
  };
  return { xs: X, ys: Y, M: MM, at, d1, d2 };
}

/** Natural cubic spline through (xs, ys) (xs strictly increasing): a tridiagonal system for
 *  the interior second derivatives, M_0 = M_{n−1} = 0. */
export function naturalCubic(xs: ArrayLike<number>, ys: ArrayLike<number>): CubicSpline {
  const n = xs.length, M = new Float64Array(n);
  if (n > 2) {
    const m = n - 2, a = new Float64Array(m), b = new Float64Array(m), c = new Float64Array(m), d = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const i = k + 1, h0 = xs[i] - xs[i - 1], h1 = xs[i + 1] - xs[i];
      a[k] = h0 / 6; b[k] = (h0 + h1) / 3; c[k] = h1 / 6;
      d[k] = (ys[i + 1] - ys[i]) / h1 - (ys[i] - ys[i - 1]) / h0;
    }
    const s = thomas(a, b, c, d);
    for (let k = 0; k < m; k++) M[k + 1] = s[k];
  }
  return cubicFromMoments(xs, ys, M);
}

/** ∫ s''² dx of a cubic spline (exact: s'' is linear on each piece). */
export function bendingEnergy(s: CubicSpline): number {
  let e = 0;
  for (let i = 0; i + 1 < s.xs.length; i++) {
    const h = s.xs[i + 1] - s.xs[i], a = s.M[i], b = s.M[i + 1];
    e += h * (a * a + a * b + b * b) / 3;
  }
  return e;
}

// ---------------------------------------------------------------------------------------
// Linear smoothers: fits ŷ = H y, with leave-one-out and GCV scores

export interface LinearFit {
  /** fitted values at the data */
  fitted: Float64Array;
  /** diagonal of the hat matrix H */
  hdiag: Float64Array;
  /** effective degrees of freedom tr H */
  df: number;
  /** leave-one-out CV score (1/n) Σ ((y − ŷ)/(1 − H_ii))² */
  loo: number;
  /** generalised CV n Σ(y − ŷ)² / (n − tr H)² */
  gcv: number;
  /** residual sum of squares */
  rss: number;
  /** the fitted function */
  at(x: number): number;
}

function scores(y: ArrayLike<number>, fitted: Float64Array, hdiag: Float64Array): { df: number; loo: number; gcv: number; rss: number } {
  const n = y.length;
  let df = 0, loo = 0, rss = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i] - fitted[i];
    df += hdiag[i]; rss += r * r;
    const d = 1 - hdiag[i];
    loo += d > 1e-12 ? (r / d) ** 2 : 0;
  }
  return { df, loo: loo / n, gcv: n * rss / Math.max(1e-12, (n - df) ** 2), rss };
}

/** Cubic smoothing spline (Reinsch 1967; Green & Silverman's form): minimise
 *  Σ (y_i − f(x_i))² + λ ∫ f''². The minimiser is a natural cubic spline with knots at the
 *  data; its values g solve (I + λ Q R⁻¹ Qᵀ) g = y. Dense O(n³): fine for n ≤ ~100. */
export function smoothingSpline(xs: ArrayLike<number>, ys: ArrayLike<number>, lambda: number): LinearFit & { spline: CubicSpline } {
  const n = xs.length;
  if (n < 3) {
    const s = naturalCubic(xs, ys), f = Float64Array.from(ys), h = new Float64Array(n).fill(1);
    return { ...scores(ys, f, h), fitted: f, hdiag: h, at: s.at, spline: s };
  }
  const m = n - 2, h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i + 1] - xs[i];
  // Q: n × m (column j for interior node j + 1), R: m × m tridiagonal
  const Q = new Float64Array(n * m);
  for (let j = 0; j < m; j++) {
    const i = j + 1;
    Q[(i - 1) * m + j] = 1 / h[i - 1];
    Q[i * m + j] = -1 / h[i - 1] - 1 / h[i];
    Q[(i + 1) * m + j] = 1 / h[i];
  }
  const ra = new Float64Array(m), rb = new Float64Array(m), rc = new Float64Array(m);
  for (let j = 0; j < m; j++) { const i = j + 1; rb[j] = (h[i - 1] + h[i]) / 3; ra[j] = h[i - 1] / 6; rc[j] = h[i] / 6; }
  // X = R⁻¹ Qᵀ (m × n), column by column
  const X = new Float64Array(m * n), col = new Float64Array(m);
  for (let c = 0; c < n; c++) {
    for (let j = 0; j < m; j++) col[j] = Q[c * m + j];
    const s = thomas(ra, rb, rc, col);
    for (let j = 0; j < m; j++) X[j * n + c] = s[j];
  }
  const A = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += Q[r * m + j] * X[j * n + c];
    A[r * n + c] = (r === c ? 1 : 0) + lambda * s;
  }
  const H = inverse(A, n)!;
  const g = new Float64Array(n), hd = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let c = 0; c < n; c++) s += H[r * n + c] * ys[c];
    g[r] = s; hd[r] = H[r * n + r];
  }
  const M = new Float64Array(n);
  for (let j = 0; j < m; j++) { let s = 0; for (let c = 0; c < n; c++) s += X[j * n + c] * g[c]; M[j + 1] = s; }
  const spline = cubicFromMoments(xs, g, M);
  return { ...scores(ys, g, hd), fitted: g, hdiag: hd, at: spline.at, spline };
}

/** Clamped cubic B-spline basis with m coefficients on uniform knots over [a, b]. */
export function regressionBasis(m: number, a: number, b: number, p = 3): { knots: number[]; eval(x: number): number[] } {
  const k = makeKnots(m, p, "clamped"), span = m - p;
  const knots = k.map(v => a + (b - a) * v / span);
  return {
    knots,
    eval(x: number) {
      const out = new Array(m).fill(0);
      const xx = Math.min(Math.max(x, a), b);
      const s = findSpan(knots, p, m, xx);
      const D = dersBasisFuns(s, xx, p, 0, knots);
      for (let j = 0; j <= p; j++) out[s - p + j] = D[0][j];
      return out;
    },
  };
}

/** Least-squares regression spline: m clamped cubic B-spline coefficients (m ≤ n). */
export function regressionSpline(xs: ArrayLike<number>, ys: ArrayLike<number>, m: number): LinearFit & { coef: Float64Array; knots: number[] } {
  const n = xs.length;
  m = Math.max(4, Math.min(m, n));
  const a = xs[0], b = xs[n - 1], basis = regressionBasis(m, a, b);
  const B = new Float64Array(n * m);
  for (let i = 0; i < n; i++) { const r = basis.eval(xs[i]); for (let j = 0; j < m; j++) B[i * m + j] = r[j]; }
  const BtB = new Float64Array(m * m), Bty = new Float64Array(m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const bij = B[i * m + j];
    if (!bij) continue;
    Bty[j] += bij * ys[i];
    for (let k = 0; k < m; k++) BtB[j * m + k] += bij * B[i * m + k];
  }
  for (let j = 0; j < m; j++) BtB[j * m + j] += 1e-10;
  const inv = inverse(BtB, m)!;
  const coef = new Float64Array(m);
  for (let j = 0; j < m; j++) { let s = 0; for (let k = 0; k < m; k++) s += inv[j * m + k] * Bty[k]; coef[j] = s; }
  const fitted = new Float64Array(n), hd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let f = 0, h = 0;
    for (let j = 0; j < m; j++) {
      f += B[i * m + j] * coef[j];
      let s = 0;
      for (let k = 0; k < m; k++) s += inv[j * m + k] * B[i * m + k];
      h += B[i * m + j] * s;
    }
    fitted[i] = f; hd[i] = h;
  }
  const at = (x: number) => { const r = basis.eval(x); let s = 0; for (let j = 0; j < m; j++) s += r[j] * coef[j]; return s; };
  return { ...scores(ys, fitted, hd), fitted, hdiag: hd, at, coef, knots: basis.knots };
}

/** Discrete second-difference smoother on uniform samples: (I + λ DᵀD) f = y. */
export function discreteSmooth(y: ArrayLike<number>, lambda: number): Float64Array {
  const n = y.length, A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) A[i * n + i] = 1;
  for (let r = 0; r + 2 < n; r++) {
    const d = [1, -2, 1];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) A[(r + a) * n + (r + b)] += lambda * d[a] * d[b];
  }
  return solve(A, Float64Array.from(y), n)!;
}

/** Frequency response of the discrete second-difference smoother on an infinite grid. */
export const discreteSmoothResponse = (omega: number, lambda: number): number => 1 / (1 + lambda * (2 - 2 * Math.cos(omega)) ** 2);

// ---------------------------------------------------------------------------------------
// Closed (periodic) P-spline for outlines

export interface ClosedFit {
  /** control points (m) of a closed uniform cubic B-spline */
  ctrl: Pt[];
  curve: SplineCurve;
  /** parameters of the data in [0, m) */
  params: Float64Array;
}

/** Fit a closed uniform cubic B-spline with m control points to an ordered closed point list
 *  (e.g. boundary pixel positions): parameters by chord length scaled to [0, m), penalised
 *  least squares (BᵀB + λ DᵀD) c = Bᵀ x with periodic second differences D (P-spline). */
export function fitClosedSpline(pts: Pt[], m: number, lambda: number): ClosedFit {
  const n = pts.length;
  m = Math.max(4, Math.min(m, n));
  const L = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const a = pts[i], b = pts[(i + 1) % n]; L[i + 1] = L[i] + Math.hypot(b[0] - a[0], b[1] - a[1]); }
  const params = new Float64Array(n);
  for (let i = 0; i < n; i++) params[i] = L[i] / L[n] * m;
  const A = new Float64Array(m * m), bx = new Float64Array(m), by = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    const t = params[i], j0 = Math.floor(t);
    const w: number[] = [], idx: number[] = [];
    // closed uniform cubic: point at t is Σ_j c_j β³(t − j) with indices mod m
    for (let d = -1; d <= 2; d++) { const j = j0 + d; w.push(bspline3(t - j)); idx.push(((j % m) + m) % m); }
    for (let a = 0; a < 4; a++) {
      bx[idx[a]] += w[a] * pts[i][0]; by[idx[a]] += w[a] * pts[i][1];
      for (let b = 0; b < 4; b++) A[idx[a] * m + idx[b]] += w[a] * w[b];
    }
  }
  for (let r = 0; r < m; r++) {
    const ii = [(r - 1 + m) % m, r, (r + 1) % m], d = [1, -2, 1];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) A[ii[a] * m + ii[b]] += lambda * d[a] * d[b];
  }
  for (let j = 0; j < m; j++) A[j * m + j] += 1e-12;
  const cx = solve(A, bx, m)!, cy = solve(A, by, m)!;
  // c_j sits at parameter j; as a curve with our closed knots (control point k ↔ centre k + 2
  // in knot units, domain [3, m + 3)) the list is rotated by one so the parameters agree.
  const ctrl: Pt[] = [];
  for (let k = 0; k < m; k++) { const j = (k - 1 + m) % m; ctrl.push([cx[j], cy[j]]); }
  return { ctrl, curve: curveFrom(ctrl, 3, "closed"), params };
}

/** Evaluate a closed fit at data parameter t ∈ [0, m) (curve parameter t + 3). */
export function closedFitPoint(f: ClosedFit, t: number): Pt { return curvePoint(f.curve, t + 3); }
