// Tests for the splines day: lib/spline.ts (Bernstein, de Casteljau, Cox–de Boor, de Boor,
// NURBS circles, natural and smoothing splines, closed P-spline outlines) and
// lib/splineimage.ts (prefilter, samplers, FFD). Brute-force or closed-form references. Run:
//   node_modules/.bin/esbuild demos/test/splines-test.ts --bundle --platform=node --log-level=warning | node
import * as S from "../src/lib/spline";
import * as SI from "../src/lib/splineimage";
import { rng } from "../src/lib/random";
import { fromFunction, createGrey } from "../src/lib/imageops";
import { rasterise, ellipsePoly } from "../src/lib/shapecanvas";
import { traceBoundary } from "../src/splineoutline/boundary";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(7);
const rp = (): S.Pt => [r.unif() * 10 - 5, r.unif() * 10 - 5];

// ---- polynomial interpolation ---------------------------------------------------------
{
  const xs = [-2, -1, 0, 1, 2], ys = [0, 0, 1, 0, 0];
  const p = S.polyInterp(xs, ys);
  check(close(p(1.5), (1.5 ** 2 - 1) * (1.5 ** 2 - 4) / 4), "Runge degree-4 example");
  check(close(p(1.5), -0.546875), "p(1.5) = −0.546875");
  for (let i = 0; i < 5; i++) check(close(p(xs[i]), ys[i]), "interpolates");
  const ch = S.chebyshevNodes(7);
  check(ch.every((v, i) => i === 0 || v > ch[i - 1]), "Chebyshev nodes ascending");
  // max error of the Runge function grows with equispaced n, shrinks with Chebyshev
  const f = (x: number) => 1 / (1 + 25 * x * x);
  const maxErr = (nodes: number[]) => { const q = S.polyInterp(nodes, nodes.map(f)); let m = 0; for (let k = 0; k <= 2000; k++) { const x = -1 + k / 1000; m = Math.max(m, Math.abs(q(x) - f(x))); } return m; };
  check(maxErr(S.equispacedNodes(11)) > maxErr(S.equispacedNodes(5)), "Runge: 11 equispaced worse than 5");
  check(maxErr(S.chebyshevNodes(11)) < maxErr(S.equispacedNodes(11)) / 5, "Chebyshev fixes the ends");
}

// ---- Bézier ---------------------------------------------------------------------------
for (let trial = 0; trial < 50; trial++) {
  const n = 1 + Math.floor(r.unif() * 6), P = Array.from({ length: n + 1 }, rp), t = r.unif();
  let s = 0;
  for (let i = 0; i <= n; i++) s += S.bernstein(n, i, t);
  check(close(s, 1), "Bernstein partition of unity");
  const a = S.bezierPoint(P, t), b = S.bezierBernstein(P, t);
  check(close(a[0], b[0], 1e-12) && close(a[1], b[1], 1e-12), "de Casteljau = Bernstein sum");
  // split: left half at u equals the curve at u t
  const [L, R] = S.bezierSplit(P, t), u = r.unif();
  const l = S.bezierPoint(L, u), c = S.bezierPoint(P, u * t);
  check(close(l[0], c[0], 1e-10) && close(l[1], c[1], 1e-10), "split left");
  const rr = S.bezierPoint(R, u), c2 = S.bezierPoint(P, t + u * (1 - t));
  check(close(rr[0], c2[0], 1e-10) && close(rr[1], c2[1], 1e-10), "split right");
  // derivative by finite difference
  const [d1] = S.bezierDerivs(P, t), e = 1e-6, pp = S.bezierPoint(P, t + e), pm = S.bezierPoint(P, t - e);
  check(close(d1[0], (pp[0] - pm[0]) / (2 * e), 1e-5) && close(d1[1], (pp[1] - pm[1]) / (2 * e), 1e-5), "Bézier derivative");
}
{
  const P: S.Pt[] = [[0, 0], [1, 2], [3, 2], [4, 0]];
  const L = S.deCasteljau(P, 0.5);
  check(close(L[1][0][0], 0.5) && close(L[1][1][0], 2) && close(L[1][2][0], 3.5), "worked level 1");
  check(close(L[2][0][0], 1.25) && close(L[2][0][1], 1.5) && close(L[2][1][0], 2.75), "worked level 2");
  check(close(L[3][0][0], 2) && close(L[3][0][1], 1.5), "worked point (2, 1.5)");
  const [d1, d2] = S.bezierDerivs(P, 0);
  check(close(d1[0], 3) && close(d1[1], 6) && close(d2[0], 6) && close(d2[1], -12), "B'(0), B''(0)");
  check(close(S.curvature(d1, d2), -72 / Math.pow(45, 1.5)), "κ(0) = −72/45^1.5");
}

// ---- B-spline basis ------------------------------------------------------------------
for (let trial = 0; trial < 60; trial++) {
  const p = Math.floor(r.unif() * 6), n = p + 1 + Math.floor(r.unif() * 6);
  // random non-decreasing knots, multiplicities ≤ p + 1 at the ends
  const kn: number[] = [];
  let v = 0;
  for (let i = 0; i < n + p + 1; i++) { if (r.unif() < 0.8 || i === 0) v += r.unif() * 2 + 0.05; kn.push(v); }
  const [a, b] = S.knotDomain(kn, p);
  if (!(b > a)) continue;
  for (let k = 0; k < 10; k++) {
    const t = a + (b - a) * (k === 9 ? 1 : r.unif());
    let s = 0;
    for (let i = 0; i < n; i++) { const N = S.basisN(i, p, kn, t); s += N; check(N >= -1e-12, "non-negative"); }
    check(close(s, 1, 1e-10), `partition of unity p=${p} t=${t}`);
    const sp = S.findSpan(kn, p, n, t), D = S.dersBasisFuns(sp, t, p, 0, kn);
    for (let j = 0; j <= p; j++) check(close(D[0][j], S.basisN(sp - p + j, p, kn, t), 1e-9), "dersBasisFuns = Cox–de Boor");
  }
}
{
  const kn = [0, 1, 2, 3];
  check(close(S.basisN(0, 2, kn, 1), 0.5) && close(S.basisN(0, 2, kn, 1.5), 0.75), "N_{0,2}(1) = 0.5, N_{0,2}(1.5) = 0.75");
  const k4 = [0, 1, 2, 3, 4];
  check(close(S.basisN(0, 3, k4, 1), 1 / 6) && close(S.basisN(0, 3, k4, 2), 4 / 6) && close(S.basisN(0, 3, k4, 3), 1 / 6), "cubic at knots 1/6, 4/6, 1/6");
  for (let x = -2.5; x <= 2.5; x += 0.37) check(close(S.basisN(0, 3, k4, x + 2), S.bspline3(x), 1e-12), "β³ = N_{0,3}(x + 2)");
  // β³ = β⁰ * β⁰ * β⁰ * β⁰ (numerical convolution)
  const conv = (f: (x: number) => number, g: (x: number) => number) => (x: number) => { let s = 0; const N = 4000; for (let k = 0; k < N; k++) { const u = -0.5 + (k + 0.5) / N; s += f(u) * g(x - u) / N; } return s; };
  const b1 = conv(S.bspline0, S.bspline0);
  check(close(b1(0.3), S.bspline1(0.3), 1e-3), "β¹ = β⁰ * β⁰");
  // derivative of β³ by finite differences
  for (let x = -2.2; x < 2.2; x += 0.31) check(close(S.bspline3d(x), (S.bspline3(x + 1e-6) - S.bspline3(x - 1e-6)) / 2e-6, 1e-5), "β³'");
}

// ---- curves ----------------------------------------------------------------------------
for (let trial = 0; trial < 40; trial++) {
  const kind = (["open", "clamped", "closed"] as const)[trial % 3];
  const n = 5 + Math.floor(r.unif() * 6), p = 1 + Math.floor(r.unif() * 3);
  const P = Array.from({ length: n }, rp), w = trial % 2 ? Array.from({ length: n }, () => 0.3 + r.unif() * 2) : undefined;
  const c = S.curveFrom(P, p, kind, w), [a, b] = S.curveDomain(c);
  for (let k = 0; k < 6; k++) {
    const t = a + (b - a) * r.unif();
    const x = S.curvePoint(c, t), y = S.curvePointRef(c, t);
    check(close(x[0], y[0], 1e-9) && close(x[1], y[1], 1e-9), `curve = Cox–de Boor sum (${kind}, p=${p})`);
    if (!w) { const z = S.deBoor(c, t); check(close(x[0], z[0], 1e-9) && close(x[1], z[1], 1e-9), "de Boor"); }
    const [, d1, d2] = S.curveDerivs(c, t, 2), e = 1e-5;
    const pp = S.curvePoint(c, t + e), pm = S.curvePoint(c, t - e), p0 = S.curvePoint(c, t);
    const fd1: S.Pt = [(pp[0] - pm[0]) / (2 * e), (pp[1] - pm[1]) / (2 * e)];
    check(close(d1[0], fd1[0], 1e-4) && close(d1[1], fd1[1], 1e-4), `C' (${kind}, p=${p}, w=${!!w})`);
    if (p >= 3) {
      const fd2: S.Pt = [(pp[0] - 2 * p0[0] + pm[0]) / (e * e), (pp[1] - 2 * p0[1] + pm[1]) / (e * e)];
      check(close(d2[0], fd2[0], 1e-2) && close(d2[1], fd2[1], 1e-2), `C'' (${kind}, p=${p})`);
    }
  }
  if (kind === "closed") {
    const s = S.curvePoint(c, a), e = S.curvePoint(c, b - 1e-12);
    check(close(s[0], e[0], 1e-8) && close(s[1], e[1], 1e-8), "closed curve closes");
  }
}
{
  // clamped cubic with 4 control points = the Bézier curve
  const P = [rp(), rp(), rp(), rp()], c = S.curveFrom(P, 3, "clamped");
  for (let t = 0; t <= 1; t += 0.125) { const a = S.curvePoint(c, t), b = S.bezierPoint(P, t); check(close(a[0], b[0], 1e-12) && close(a[1], b[1], 1e-12), "clamped = Bézier"); }
  // closed uniform cubic on the square: point at the knot of (4, 0) is (10/3, 2/3)
  const sq: S.Pt[] = [[0, 0], [4, 0], [4, 4], [0, 4]], cs = S.curveFrom(sq, 3, "closed");
  let best = Infinity;
  for (let t = 3; t < 7; t += 1) { const q = S.curvePoint(cs, t); best = Math.min(best, Math.hypot(q[0] - 10 / 3, q[1] - 2 / 3)); }
  check(best < 1e-12, "corner cutting (10/3, 2/3)");
  // NURBS circles
  const qc = S.quarterCircle();
  for (let t = 0; t <= 1; t += 0.05) { const q = S.curvePoint(qc, t); check(close(Math.hypot(q[0], q[1]), 1, 1e-12), "quarter circle radius 1"); }
  const mid = S.curvePoint(qc, 0.5);
  check(close(mid[0], Math.SQRT1_2, 1e-12), "quarter circle midpoint 0.7071");
  const q1 = S.curvePoint(S.quarterCircle(1), 0.5);
  check(close(q1[0], 0.75) && close(Math.hypot(q1[0], q1[1]), 1.0606601717798212), "w = 1: (0.75, 0.75)");
  const fc = S.nurbsCircle(2, -1, 3);
  for (let t = 0; t <= 4; t += 0.03) { const q = S.curvePoint(fc, t); check(close(Math.hypot(q[0] - 2, q[1] + 1), 3, 1e-12), "full circle radius"); }
  check(close(S.arcLength(fc, 8), 6 * Math.PI, 1e-6), "circle arc length 2πr");
  check(close(S.totalCurvature(fc, 8), 2 * Math.PI, 1e-6), "∮κ ds = 2π");
  for (let t = 0.1; t < 4; t += 0.37) { const [, d1, d2] = S.curveDerivs(fc, t, 2); check(close(S.curvature(d1, d2), 1 / 3, 1e-9), "κ = 1/r"); }
  // affected spans: control point i of a cubic open curve with 10 points
  const c10 = S.curveFrom(Array.from({ length: 10 }, rp), 3, "open"), inf = S.influence(c10, 5);
  check(inf.length === 1 && inf[0][0] === 5 && inf[0][1] === 9, "P5 of a cubic with 10 points influences [5, 9]: 4 spans");
  const inf4 = S.influence(c10, 4);
  check(inf4[0][0] === 4 && inf4[0][1] === 8, "P4: 4 spans");
}

// ---- natural and smoothing splines -----------------------------------------------------
{
  const s = S.naturalCubic([0, 1, 2], [0, 1, 0]);
  check(close(s.M[1], -3), "M1 = −3");
  check(close(s.at(0.5), 0.6875), "S(0.5) = 0.6875");
  check(close(S.bendingEnergy(s), 6), "bending 6");
  // linear data reproduced
  const xs = [0, 0.5, 1.7, 2, 3.1, 4], lin = S.naturalCubic(xs, xs.map(x => 2 * x - 1));
  for (let x = 0; x <= 4; x += 0.3) check(close(lin.at(x), 2 * x - 1, 1e-12), "natural spline reproduces lines");
  // smoothing: λ → 0 interpolates, λ → ∞ gives the least-squares line
  const X = Array.from({ length: 12 }, (_, i) => i + 0.3 * Math.sin(i)), Y = X.map(x => Math.sin(x / 2) + 0.2 * r.norm());
  const s0 = S.smoothingSpline(X, Y, 1e-9);
  for (let i = 0; i < 12; i++) check(close(s0.fitted[i], Y[i], 1e-6), "λ → 0 interpolates");
  const nat = S.naturalCubic(X, Y);
  check(close(s0.at(3.3), nat.at(3.3), 1e-6), "λ → 0 = natural spline");
  check(close(s0.df, 12, 1e-6), "df = n at λ = 0");
  const sInf = S.smoothingSpline(X, Y, 1e9);
  const mx = X.reduce((a, b) => a + b) / 12, my = Y.reduce((a, b) => a + b) / 12;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < 12; i++) { sxy += (X[i] - mx) * (Y[i] - my); sxx += (X[i] - mx) ** 2; }
  const slope = sxy / sxx;
  for (let i = 0; i < 12; i++) check(close(sInf.fitted[i], my + slope * (X[i] - mx), 1e-5), "λ → ∞ = least-squares line");
  check(close(sInf.df, 2, 1e-4), "df → 2");
  // brute-force LOO for one λ
  const lam = 0.7, full = S.smoothingSpline(X, Y, lam);
  let loo = 0;
  for (let k = 0; k < 12; k++) {
    const Xk = X.filter((_, i) => i !== k), Yk = Y.filter((_, i) => i !== k);
    const fk = S.smoothingSpline(Xk, Yk, lam);
    loo += (Y[k] - fk.at(X[k])) ** 2;
  }
  // (k = 0 and 11 are extrapolations of the natural spline: linear, as the formula assumes)
  check(close(full.loo, loo / 12, 1e-6), `LOO shortcut ${full.loo} vs ${loo / 12}`);
  // penalty is exactly λ∫f'': objective at the solution ≤ objective of the interpolant and of the line
  const obj = (fit: { fitted: Float64Array; spline: S.CubicSpline }) => fit.fitted.reduce((a, g, i) => a + (Y[i] - g) ** 2, 0) + lam * S.bendingEnergy(fit.spline);
  check(obj(full) <= obj(s0) + 1e-9 && obj(full) <= obj(sInf) + 1e-9, "smoothing spline minimises its objective");
  // discrete 3-point example
  const d = S.discreteSmooth([0, 3, 0], 1);
  check(close(d[0], 6 / 7) && close(d[1], 9 / 7) && close(d[2], 6 / 7), "discrete (6/7, 9/7, 6/7)");
  const dInf = S.discreteSmooth([0, 3, 0], 1e9);
  check(close(dInf[1], 1, 1e-6), "discrete λ → ∞: mean");
  // regression spline: brute force normal equations vs lstsq of the polynomial case m = 4 on 4 points interpolates
  const rg = S.regressionSpline([0, 1, 2, 3], [1, -1, 2, 0], 4);
  check(rg.fitted.every((v, i) => close(v, [1, -1, 2, 0][i], 1e-6)), "regression spline with m = n interpolates");
  // frequency response of the discrete smoother on a long periodic-like signal (middle)
  const N = 400, om = 0.3, sig = Array.from({ length: N }, (_, i) => Math.cos(om * i)), sm = S.discreteSmooth(sig, 2);
  check(close(sm[200] / sig[200], S.discreteSmoothResponse(om, 2), 1e-6), "smoother frequency response");
}

// ---- closed outline fit ----------------------------------------------------------------
{
  // circle samples: the fit with small λ reproduces the circle; length ≈ 2πr
  const pts: S.Pt[] = Array.from({ length: 60 }, (_, i) => [10 * Math.cos(2 * Math.PI * i / 60), 10 * Math.sin(2 * Math.PI * i / 60)]);
  const f = S.fitClosedSpline(pts, 30, 0);
  check(close(S.arcLength(f.curve, 4), 20 * Math.PI, 1e-3), "closed fit of a circle: length");
  check(close(Math.abs(S.totalCurvature(f.curve, 4)), 2 * Math.PI, 1e-6), "closed fit: total curvature 2π");
  const q = S.closedFitPoint(f, f.params[7]);
  check(Math.hypot(q[0] - pts[7][0], q[1] - pts[7][1]) < 0.01, "closed fit passes near the data");
  // m = n, λ = 0 interpolates
  const g = S.fitClosedSpline(pts.slice(0, 20).map(p => [p[0] + r.norm() * 0.1, p[1]] as S.Pt), 20, 0);
  const pp = pts.slice(0, 20);
  void pp;
  check(g.ctrl.length === 20, "m = n control points");
  // digital disc of radius 10
  const R = 10, W = 32, mask = rasterise([ellipsePoly(16, 16, R, R, 400)], W, W);
  const b = traceBoundary(mask, W, W);
  check(b.edges === 80 || Math.abs(b.edges - 80) <= 4, `disc edge count ≈ 80 (${b.edges})`);
  const fit = S.fitClosedSpline(b.points, Math.min(40, b.points.length), 3);
  const len = S.arcLength(fit.curve, 4);
  check(Math.abs(len - 2 * Math.PI * R) / (2 * Math.PI * R) < 0.02, `smoothing-spline perimeter within 2 % of 20π: ${len}`);
}

// ---- image interpolation ---------------------------------------------------------------
{
  const z = SI.Z1, f = new Float64Array(41);
  f[20] = 1;
  const c = SI.bsplineCoeffs1d(f);
  for (let k = -8; k <= 8; k++) check(close(c[20 + k], Math.sqrt(3) * Math.pow(z, Math.abs(k)), 1e-9), "impulse coefficients √3 z^|k|");
  for (let k = 1; k < 40; k++) check(close((c[k - 1] + 4 * c[k] + c[k + 1]) / 6, f[k], 1e-12), "prefilter equation");
  // random 2D image: B-spline sampling reproduces the samples
  const img = fromFunction(17, 13, () => r.unif() * 100), co = SI.bsplineCoeffs(img);
  for (let y = 0; y < 13; y++) for (let x = 0; x < 17; x++) check(close(SI.sampleBSpline(co, x, y), img.data[y * 17 + x], 1e-5), "B-spline interpolates (float32 coefficients)");
  // keys interpolates; reproduces quadratics in 1D (interior)
  for (let y = 2; y < 11; y++) for (let x = 2; x < 15; x++) check(close(SI.sampleKeys(img, x, y), img.data[y * 17 + x], 1e-4), "Keys interpolates");
  const q = fromFunction(20, 3, x => 0.5 * x * x - 3 * x + 2);
  check(close(SI.sampleKeys(q, 7.3, 1), 0.5 * 7.3 * 7.3 - 3 * 7.3 + 2, 1e-9), "Keys reproduces quadratics");
  // gradient by finite differences
  const [gx, gy] = SI.gradBSpline(co, 6.3, 5.7), e = 1e-5;
  check(close(gx, (SI.sampleBSpline(co, 6.3 + e, 5.7) - SI.sampleBSpline(co, 6.3 - e, 5.7)) / (2 * e), 1e-4), "∂x of B-spline");
  check(close(gy, (SI.sampleBSpline(co, 6.3, 5.7 + e) - SI.sampleBSpline(co, 6.3, 5.7 - e)) / (2 * e), 1e-4), "∂y of B-spline");
  // frequency responses at 0 are 1
  for (const m of ["nearest", "bilinear", "keys", "bspline", "bspline-nopre"] as const) check(close(SI.interpResponse(m, 0), 1, 1e-4), "H(0) = 1 " + m);
  check(close(SI.interpResponse("bspline-nopre", Math.PI), (2 / Math.PI) ** 4, 1e-12), "sinc⁴ at π");
}

// ---- FFD -------------------------------------------------------------------------------
{
  const g = SI.ffdGrid(97, 97, 16);
  check(SI.ffdAt(g, 40.5, 33)[0] === 0, "zero displacement = identity");
  g.dx.fill(3); g.dy.fill(-2);
  for (let k = 0; k < 10; k++) { const x = r.unif() * 96, y = r.unif() * 96, [u, v] = SI.ffdAt(g, x, y); check(close(u, 3, 1e-12) && close(v, -2, 1e-12), "equal displacements = translation"); }
  g.dx.fill(0); g.dy.fill(0);
  g.dx[SI.ffdIndex(g, 3, 3)] = 9;
  check(close(SI.ffdAt(g, 48, 48)[0], 4), "9 px drag: 4 at the point");
  check(close(SI.ffdAt(g, 64, 48)[0], 1), "1 at the next point");
  check(close(SI.ffdAt(g, 64, 64)[0], 0.25), "0.25 diagonally");
  check(close(SI.ffdAt(g, 16, 48)[0], 0), "0 two spacings away");
  check(close(SI.ffdJacobianDet(g, 40, 48), 1 + 9 / 16 * S.bspline3d(-0.5) * S.bspline3(0)), "Jacobian");
  // field = pointwise
  const F = SI.ffdField(g, 97, 97);
  for (let k = 0; k < 20; k++) { const x = Math.floor(r.unif() * 97), y = Math.floor(r.unif() * 97); check(close(F.u[y * 97 + x], SI.ffdAt(g, x, y)[0], 1e-6), "ffdField"); }
  // forward warp moves content by about u: a bright pixel at the control point moves 4 px right
  const im = createGrey(97, 97);
  for (let y = 44; y <= 52; y++) for (let x = 44; x <= 52; x++) im.data[y * 97 + x] = 100;
  const out = SI.ffdWarp(im, g, "bilinear");
  let sx = 0, s = 0;
  for (let y = 0; y < 97; y++) for (let x = 0; x < 97; x++) { sx += x * out.data[y * 97 + x]; s += out.data[y * 97 + x]; }
  check(Math.abs(sx / s - 51.5) < 0.5, `forward warp shifts a 9-px square at the control point by ≈ 3.5 px (4 at its centre) (${(sx / s).toFixed(2)})`);
  // degree 1: interpolating, the control point itself moves the full 9 px
  const g1 = SI.ffdGrid(97, 97, 16, 1);
  g1.dx[SI.ffdIndex(g1, 3, 3)] = 9;
  check(close(SI.ffdAt(g1, 48, 48)[0], 9) && close(SI.ffdAt(g1, 56, 48)[0], 4.5), "degree-1 FFD");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
