// Tests for the snakes day: lib/snake.ts (the snake matrix, its eigenvalues, the explicit and
// semi-implicit steps, forces, GVF), lib/shapes.ts (Procrustes, GPA, PDM, rasterising) and
// the ASM search (demos/src/asm/model.ts). Brute-force or closed-form references. Run:
//   node_modules/.bin/esbuild demos/test/snakes-test.ts --bundle --platform=node --log-level=warning | node
import * as SN from "../src/lib/snake";
import * as SH from "../src/lib/shapes";
import { jacobiEigen, matVec } from "../src/lib/linalg";
import { rng } from "../src/lib/random";
import { createGrey, fromFunction } from "../src/lib/imageops";
import { train, Search, SEARCH_DEFAULTS, makeScene, poseOf, poseScale, poseAngle, rmsError } from "../src/asm/model";
import { buildModel } from "../src/shapepca/model";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(2024);

// ---- the snake matrix -------------------------------------------------------------------
for (let n = 3; n <= 14; n++) {
  const alpha = r.unif() * 2, beta = r.unif() * 2;
  const A = SN.snakeMatrix(n, alpha, beta);
  // A 1 = 0 and symmetry
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) { s += A[i * n + j]; check(close(A[i * n + j], A[j * n + i]), `n=${n}: A symmetric`); }
    check(Math.abs(s) < 1e-12, `n=${n}: row sum 0`);
  }
  // eigenvalues = closed form (sorted)
  const ev = Array.from(jacobiEigen(A, n).values).sort((a, b) => a - b);
  const cf = Array.from(SN.snakeEigenvalues(n, alpha, beta)).sort((a, b) => a - b);
  for (let k = 0; k < n; k++) check(close(ev[k], cf[k], 1e-8), `n=${n}: eigenvalue ${k}: ${ev[k]} vs ${cf[k]}`);
  check(ev[0] > -1e-10, `n=${n}: positive semidefinite`);
  if (n % 2 === 0) check(close(SN.lambdaMax(n, alpha, beta), 4 * alpha + 16 * beta), `n=${n}: λmax = 4α + 16β`);
  // the sampled cosine of mode k is an eigenvector
  for (let k = 0; k < n; k++) {
    const v = Float64Array.from({ length: n }, (_, i) => Math.cos(2 * Math.PI * k * i / n)), Av = matVec(A, v, n, n), l = SN.snakeEigenvalue(k, n, alpha, beta);
    for (let i = 0; i < n; i++) check(Math.abs(Av[i] - l * v[i]) < 1e-9, `n=${n}: cos mode ${k} eigenvector`);
  }
  // applyA = A x
  const x = Float64Array.from({ length: n }, () => r.norm()), Ax = matVec(A, x, n, n), Bx = SN.applyA(x, alpha, beta);
  for (let i = 0; i < n; i++) check(close(Ax[i], Bx[i]), `n=${n}: applyA`);
}
// the n = 6, α = 1, β = 0 by-hand matrix: (1, −1, 1, −1, 1, −1) has eigenvalue 4
{
  const A = SN.snakeMatrix(6, 1, 0), v = [1, -1, 1, -1, 1, -1], Av = matVec(A, v, 6, 6);
  for (let i = 0; i < 6; i++) check(close(Av[i], 4 * v[i]), "n=6: alternating vector, eigenvalue 4");
  check(A[0] === 2 && A[1] === -1 && A[5] === -1 && A[2] === 0, "n=6: first row 2 −1 0 0 0 −1");
}

// ---- internal force = −gradient of the internal energy -----------------------------------
for (let trial = 0; trial < 5; trial++) {
  const n = 5 + trial * 3, alpha = r.unif(), beta = r.unif();
  const c = SN.circleContour(0, 0, 10, n);
  for (let i = 0; i < n; i++) { c.x[i] += r.norm(); c.y[i] += r.norm(); }
  const F = SN.internalForce(c, alpha, beta), h = 1e-5;
  for (let i = 0; i < n; i++) for (const ax of ["x", "y"] as const) {
    const cp = SN.cloneContour(c), cm = SN.cloneContour(c);
    cp[ax][i] += h; cm[ax][i] -= h;
    const g = (SN.internalEnergy(cp, alpha, beta).total - SN.internalEnergy(cm, alpha, beta).total) / (2 * h);
    check(Math.abs(-g - F[ax][i]) < 1e-5 * Math.max(1, Math.abs(g)), `force = −∂E/∂x (n=${n}, i=${i})`);
  }
}
// the by-hand square: 4 points, α = 1, β = 0: force at (1, 1) is (−2, −2); one step τ = 0.1 shrinks to 0.8
{
  const sq = SN.contour([1, -1, -1, 1], [1, 1, -1, -1]);
  const F = SN.internalForce(sq, 1, 0);
  check(close(F.x[0], -2) && close(F.y[0], -2), "square: elastic force (−2, −2)");
  const s1 = SN.explicitStep(sq, 1, 0, 0.1, null);
  check(close(s1.x[0], 0.8) && close(s1.y[0], 0.8) && close(s1.x[2], -0.8), "square: one explicit step to ±0.8");
  // with β = 1 the rigid part at a corner of the 4-point square: −(x_{i−2} − 4x_{i−1} + 6x_i − 4x_{i+1} + x_{i+2}), x_{i−2} = x_{i+2}
  const Fb = SN.internalForce(sq, 0, 1);
  check(close(Fb.x[0], -4) && close(Fb.y[0], -4), `square: rigid force (−4, −4), got ${Fb.x[0]}`);
}

// ---- steps: semi-implicit never increases E_int; explicit stable below 2/λmax ------------
for (const tau of [0.1, 1, 10, 1000]) {
  const n = 24, alpha = 0.3, beta = 0.2, S = SN.implicitSolver(n, alpha, beta, 1 / tau);
  let c = SN.circleContour(0, 0, 20, n);
  const rr = rng(3);
  for (let i = 0; i < n; i++) { c.x[i] += 2 * rr.norm(); c.y[i] += 2 * rr.norm(); }
  let E = SN.internalEnergy(c, alpha, beta).total, A = Math.abs(SN.signedArea(c)), ok = true, shrink = true;
  for (let it = 0; it < 50; it++) {
    c = SN.implicitStep(c, S, null);
    const E2 = SN.internalEnergy(c, alpha, beta).total, A2 = Math.abs(SN.signedArea(c));
    if (E2 > E + 1e-9) ok = false;
    if (A2 > A + 1e-9) shrink = false;
    E = E2; A = A2;
  }
  check(ok, `semi-implicit τ=${tau}: internal energy never increases`);
  void shrink;
  // a perfect circle stays a circle and shrinks
  let cc = SN.circleContour(3, 4, 20, n), rad = 20, okc = true;
  for (let it = 0; it < 10; it++) {
    cc = SN.implicitStep(cc, S, null);
    const rr2 = Array.from(cc.x, (x, i) => Math.hypot(x - 3, cc.y[i] - 4));
    if (!(rr2[0] < rad) || Math.max(...rr2) - Math.min(...rr2) > 1e-9) okc = false;
    rad = rr2[0];
  }
  check(okc, `semi-implicit τ=${tau}: a circle shrinks and stays a circle`);
}
{
  const n = 16, alpha = 0.25, beta = 0, lm = SN.lambdaMax(n, alpha, beta);   // = 1
  const zig = (c: SN.Contour) => SN.modeAmplitudes(c)[n / 2];
  const start = SN.circleContour(0, 0, 1, n);
  for (let i = 0; i < n; i++) start.x[i] += (i % 2 ? 1 : -1) * 0.01;
  for (const [tau, grows] of [[1.95, false], [2.1, true]] as const) {
    let c = start;
    for (let it = 0; it < 40; it++) c = SN.explicitStep(c, alpha, beta, tau, null);
    check((zig(c) > zig(start)) === grows, `explicit τλmax=${(tau * lm).toFixed(2)}: zig-zag ${grows ? "grows" : "decays"}`);
    // the zig-zag mode is multiplied by exactly 1 − τλmax per step
    const c1 = SN.explicitStep(start, alpha, beta, tau, null);
    check(close(zig(c1) / zig(start), Math.abs(1 - tau * lm), 1e-6), "zig-zag factor |1 − τλmax|");
  }
}
// mode amplitudes of a pure cosine wiggle
{
  const n = 12, x = Float64Array.from({ length: n }, (_, i) => 0.3 * Math.cos(2 * Math.PI * 3 * i / n)), y = new Float64Array(n);
  const a = SN.modeAmplitudes({ x, y });
  check(close(a[3], 0.3) && Math.abs(a[1]) < 1e-12 && Math.abs(a[2]) < 1e-12, "mode amplitude of a pure mode");
}
// resampling gives equal spacing and keeps the perimeter of a polygon with corners on the samples
{
  const sq = SN.contour([0, 4, 4, 0], [0, 0, 4, 4]), rs = SN.resample(sq, 16);
  let ok = true;
  for (let i = 0; i < 16; i++) { const j = (i + 1) % 16; if (Math.abs(Math.hypot(rs.x[j] - rs.x[i], rs.y[j] - rs.y[i]) - 1) > 1e-9) ok = false; }
  check(ok, "resample: equal spacing 1 on a 4 × 4 square");
}
// outward normals and balloon
{
  for (const dir of [1, -1]) {
    const c = SN.circleContour(5, 5, 3, 20);
    if (dir < 0) { c.x.reverse(); c.y.reverse(); }
    const N = SN.normals(c);
    let ok = true;
    for (let i = 0; i < 20; i++) if ((c.x[i] - 5) * N.x[i] + (c.y[i] - 5) * N.y[i] <= 0) ok = false;
    check(ok, `normals point outward (orientation ${dir})`);
  }
}
// 1D GVF by hand: g = (0, 0, 0, 1, 0, 0, 0), μ = 0.25, Δt = 1
{
  const g = [0, 0, 0, 1, 0, 0, 0];
  const v1 = SN.gvf1dStep(g, g, 0.25, 1);
  check(close(v1[2], 0.25) && close(v1[3], 0.5) && close(v1[4], 0.25) && v1[1] === 0, `1D GVF step 1: ${Array.from(v1)}`);
  const v2 = SN.gvf1dStep(v1, g, 0.25, 1);
  check(v2[1] > 0 && v2[0] >= 0, "1D GVF step 2 spreads further");
}
// GVF far from a disk points towards it; edge force there is ~0
{
  const img = fromFunction(64, 64, (x, y) => (Math.hypot(x - 32, y - 32) < 10 ? 1 : 0));
  const f = SN.edgeMap(img, 1.5), E = SN.gridGradient(f), G = SN.gvfFast(E, 0.2, 200, 20);
  const p = { x: Float64Array.from([32 + 25]), y: Float64Array.from([32]) };
  const fe = SN.sampleField(E, p), fg = SN.sampleField(G, p);
  check(Math.abs(fe.x[0]) < 1e-6, "edge force vanishes far from the edge");
  check(fg.x[0] < 0 && Math.abs(fg.y[0]) < Math.abs(fg.x[0]), "GVF far from the disk points towards it");
}

// ---- Procrustes ---------------------------------------------------------------------------
for (let trial = 0; trial < 20; trial++) {
  const n = 3 + trial % 7, p = Float64Array.from({ length: 2 * n }, () => r.norm());
  const s = 0.3 + 2 * r.unif(), th = (r.unif() - 0.5) * 6, tx = r.norm() * 3, ty = r.norm() * 3;
  const q = SH.transform(p, { s, theta: th, tx, ty });
  const f = SH.procrustes(p, q);
  check(f.rss < 1e-18 * n + 1e-20 * s && close(f.s, s, 1e-9) && close(Math.cos(f.theta), Math.cos(th), 1e-9) && close(Math.sin(f.theta), Math.sin(th), 1e-9), `Procrustes recovers s, θ (trial ${trial})`);
  check(close(f.tx, tx, 1e-9) && close(f.ty, ty, 1e-9), "Procrustes recovers t");
  const back = SH.invertFit(f, q);
  check(SH.sumSq(back, p) < 1e-18, "invertFit undoes the fit");
  // reflection guard
  const m = SH.transform(p, { s, theta: th, tx, ty, reflect: true });
  const f0 = SH.procrustes(p, m), f1 = SH.procrustes(p, m, { reflection: true });
  check(!f0.reflect && f0.rss > 1e-6, "no reflection: a mirror image is not matched exactly");
  check(f1.reflect && f1.rss < 1e-18, "with reflection: matched exactly");
  // brute force: no rotation/scale does better
  let best = Infinity;
  for (let k = 0; k < 360; k++) {
    const t = k * Math.PI / 180, a = f.s * Math.cos(t), b = f.s * Math.sin(t);
    const g = SH.procrustes(p, m);
    void g;
    const cand = SH.applySim(SH.centre(p), a, b), [mx, my] = SH.centroid(m);
    best = Math.min(best, SH.sumSq(SH.translateShape(cand, mx, my), m));
  }
  check(f0.rss <= best + 1e-9, "Procrustes beats a brute-force search over angles");
}
// the by-hand triangle: a = 1.2, b = 0.9
{
  const p = Float64Array.from([0, 0, 2, 0, 0, 1]), q = SH.applySim(p, 1.2, 0.9, -1, -1.5), f = SH.procrustes(p, q);
  check(close(f.a, 1.2) && close(f.b, 0.9) && close(f.s, 1.5), "triangle: a = 1.2, b = 0.9, s = 1.5");
}
// rigid (no scale): s = 1
{
  const p = Float64Array.from([0, 0, 2, 0, 0, 1]), q = SH.applySim(p, 1.2, 0.9), f = SH.procrustes(p, q, { scale: false });
  check(close(f.s, 1) && close(f.theta, Math.atan2(0.9, 1.2)), "rigid Procrustes keeps s = 1");
}

// ---- GPA and the PDM -----------------------------------------------------------------------
{
  // similarity copies of one shape: after GPA all equal the mean
  const base = SH.family("leaf").make([0.36, 0, 0]);
  const shapes = Array.from({ length: 8 }, () => SH.transform(base, { s: 0.5 + r.unif(), theta: r.norm(), tx: r.norm(), ty: r.norm() }));
  const g = SH.gpa(shapes);
  check(g.aligned.every(s => SH.sumSq(s, g.mean) < 1e-16), "GPA of similar shapes: all equal the mean");
  check(close(SH.centroidSize(g.mean), 1, 1e-9), "GPA mean has size 1");
}
{
  // a linear family with 2 hidden parameters: 2 dominant eigenvalues
  const n = 10, base = SH.family("nucleus").make([0, 0, 0]).subarray(0, 2 * n);
  const u = Float64Array.from({ length: 2 * n }, (_, i) => Math.sin(i)), v = Float64Array.from({ length: 2 * n }, (_, i) => Math.cos(3 * i));
  const shapes = Array.from({ length: 30 }, () => { const a = 0.05 * r.norm(), b = 0.03 * r.norm(); return base.map((x, i) => x + a * u[i] + b * v[i]); });
  const m = SH.pdm(shapes);
  check(m.values[1] > 1e3 * m.values[2] && m.values[2] < 1e-12 * m.total + 1e-20, `2 hidden parameters → 2 non-zero eigenvalues (${m.values[0]}, ${m.values[1]}, ${m.values[2]})`);
  check(close(m.values.reduce((a, b) => a + b, 0), m.total, 1e-8), "Σλ = tr S");
  // the model reproduces a training shape from its b exactly (all modes)
  const b = SH.projectB(m, shapes[4], 2), rec = SH.synthesise(m, b);
  check(SH.sumSq(rec, shapes[4]) < 1e-20, "x = x̄ + Φb with 2 modes reproduces a training shape");
  const cl = SH.clampB(m, [10, -10], 3);
  check(close(cl[0], 3 * Math.sqrt(m.values[0])) && close(cl[1], -3 * Math.sqrt(m.values[1])), "clamp to ±3√λ");
  // fewer shapes than dimensions: at most s − 1 non-zero eigenvalues
  const m3 = SH.pdm(shapes.slice(0, 2));
  check(m3.values[1] < 1e-12 * m3.total + 1e-20, "2 shapes: 1 non-zero eigenvalue");
}
{
  // the shapepca demo: hand with 3 hidden parameters: 3 modes > 93 % of the variance; 3 shapes: 2 modes
  const h = buildModel({ family: "hand", count: 40, noise: 0.01, align: true, corrupt: 0 });
  const v = h.model.values, t = h.model.total;
  check((v[0] + v[1] + v[2]) / t > 0.93, "hand PDM: 3 modes carry > 93 %");
  const h3 = buildModel({ family: "hand", count: 3, noise: 0.01, align: true, corrupt: 0 });
  check(h3.model.values[2] < 1e-10 * h3.model.total, "3 training shapes: only 2 modes");
}

// ---- rasterising ---------------------------------------------------------------------------
{
  const sq = Float64Array.from([10.2, 10.7, 30.6, 10.7, 30.6, 25.1, 10.2, 25.1]);
  const cov = SH.polygonCoverage(sq, 40, 40);
  let s = 0;
  for (const v of cov) s += v;
  check(Math.abs(s - 20.4 * 14.4) < 0.015 * 20.4 * 14.4, `coverage of a rectangle = its area (${s} vs ${20.4 * 14.4})`);
  const tri = Float64Array.from([5, 5, 35, 8, 12, 33]), ct = SH.polygonCoverage(tri, 40, 40, 16);
  let st = 0;
  for (const v of ct) st += v;
  const area = Math.abs((35 - 5) * (33 - 5) - (12 - 5) * (8 - 5)) / 2;
  check(Math.abs(st - area) < 0.5, `coverage of a triangle = its area (${st} vs ${area})`);
  void createGrey;
}

// ---- ASM ------------------------------------------------------------------------------------
{
  const M = train({ family: "nucleus", levels: 3 });
  const sc = makeScene(M.fam, { noise: 0 });
  const P = sc.pose;
  const start = poseOf(poseScale(P), poseAngle(P) + 10 * Math.PI / 180, P.tx + 10, P.ty);
  const S = new Search(M, sc.img, start, { ...SEARCH_DEFAULTS, levels: 3 });
  const e0 = rmsError(S.X, sc.truth);
  while (!S.done()) S.iterate();
  const e1 = rmsError(S.X, sc.truth);
  check(e0 > 8 && e1 < 1, `ASM converges from a 10 px / 10° offset: ${e0.toFixed(2)} → ${e1.toFixed(2)} px`);
  // without the pyramid from a far start it does not
  const far = poseOf(poseScale(P) * 0.9, poseAngle(P) - 20 * Math.PI / 180, P.tx - 22, P.ty + 12);
  const S1 = new Search(M, sc.img, far, { ...SEARCH_DEFAULTS, levels: 1 });
  while (!S1.done()) S1.iterate();
  const S3 = new Search(M, sc.img, far, { ...SEARCH_DEFAULTS, levels: 3 });
  while (!S3.done()) S3.iterate();
  check(rmsError(S1.X, sc.truth) > 10 && rmsError(S3.X, sc.truth) < 1, "far start: one level fails, three levels converge");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
