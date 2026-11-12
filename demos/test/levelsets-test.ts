// Tests for the level-set day: lib/levelset.ts (signed distances, upwind and curvature
// derivatives, the level set step, the CFL limit, reinitialisation, Chan–Vese) and the demo
// models (curveflow shapes, gac edge flow, chanvese scenes). Closed-form or brute-force
// references. Run:
//   node_modules/.bin/esbuild demos/test/levelsets-test.ts --bundle --platform=node --log-level=warning | node
import * as LS from "../src/lib/levelset";
import { rng } from "../src/lib/random";
import { shapePhi } from "../src/curveflow/shapes";
import * as GAC from "../src/gac/model";
import * as CV from "../src/chanvese/model";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const N = 128;
const radius = (phi: Float64Array) => Math.sqrt(LS.insideArea(phi, N, N) / Math.PI);
const r = rng(31);

// ---- signed distances against brute force ------------------------------------------------
{
  const P = LS.radialPoly(20, 18, t => 9 + 4 * Math.cos(3 * t) + 1.5 * Math.sin(5 * t), 37);
  const w = 40, h = 36, d = LS.polygonSDF([P], w, h);
  let maxErr = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let best = Infinity;
    for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length]; best = Math.min(best, LS.pointSegDist(x, y, a[0], a[1], b[0], b[1])); }
    maxErr = Math.max(maxErr, Math.abs(Math.abs(d[y * w + x]) - best));
    check((d[y * w + x] < 0) === LS.insideEvenOdd(x, y, [P]), `polygonSDF sign at ${x},${y}`);
  }
  check(maxErr < 1e-12, `polygonSDF = brute-force distance (max error ${maxErr})`);
  // random segments: bucketed distance = brute force
  const segs: number[] = [];
  for (let k = 0; k < 25; k++) segs.push(r.unif() * 60 - 5, r.unif() * 50 - 5, r.unif() * 60 - 5, r.unif() * 50 - 5);
  const D = LS.distanceToSegments(segs, 50, 40);
  let e2 = 0;
  for (let y = 0; y < 40; y++) for (let x = 0; x < 50; x++) {
    let best = Infinity;
    for (let s = 0; s < 25; s++) best = Math.min(best, LS.pointSegDist(x, y, segs[4 * s], segs[4 * s + 1], segs[4 * s + 2], segs[4 * s + 3]));
    e2 = Math.max(e2, Math.abs(D[y * 50 + x] - best));
  }
  check(e2 < 1e-12, `distanceToSegments = brute force (max error ${e2})`);
  // box SDF: inside depth and corner distance
  check(Math.abs(LS.sdfBox(0, 0, 0, 0, 1.5, 1.5) + 1.5) < 1e-12, "box SDF at the centre = −1.5");
  check(Math.abs(LS.sdfBox(2.5, 2.5, 0, 0, 1.5, 1.5) - Math.SQRT2) < 1e-12, "box SDF outside a corner = distance to the corner");
}

// ---- derivatives ----------------------------------------------------------------------------
{
  const phi = LS.sdfCircle(N, N, 64, 64, 30);
  for (const [x, y] of [[100, 64], [64, 20], [90, 90], [40, 80]]) {
    const rho = Math.hypot(x - 64, y - 64), k = LS.curvatureAt(phi, N, N, x, y);
    check(Math.abs(k - 1 / rho) < 0.02 / rho, `κ of a circle's level line at (${x},${y}) = 1/ρ: ${k} vs ${1 / rho}`);
    const [gx, gy] = LS.gradAt(phi, N, N, x, y);
    check(Math.abs(Math.hypot(gx, gy) - 1) < 0.01, `|∇φ| = 1 for a signed distance at (${x},${y})`);
  }
  // the 1D by-hand example: φ = |x| − 1 on x = −3..3, F = 1, Δt = 0.5
  const p1 = Float64Array.from([-3, -2, -1, 0, 1, 2, 3], v => Math.abs(v) - 1);
  const after = Array.from(p1, (v, i) => v - 0.5 * LS.upwindGrad(p1, 7, 1, i, 0, 1));
  const want = [1.5, 0.5, -0.5, -1, -0.5, 0.5, 1.5];
  check(after.every((v, i) => Math.abs(v - want[i]) < 1e-12), `1D upwind step: ${after}`);
  // central differences see no slope at the kink
  check(LS.centralGrad(p1, 7, 1, 3, 0) === 0, "central |∇φ| = 0 at the kink of |x|");
}

// ---- the level set step ---------------------------------------------------------------------
{
  check(Math.abs(LS.cflLimit({ F: 1 }) - Math.SQRT1_2) < 1e-12, "CFL limit for F = 1 is 1/√2");
  check(Math.abs(LS.cflLimit({ F: 0, eps: 1 }) - 0.25) < 1e-12, "curvature limit 1/(4ε)");
  // constant speed: a circle grows by F t
  for (const F of [1, -1]) {
    const phi = LS.sdfCircle(N, N, 63.5, 63.5, 30), dt = 0.5 * LS.cflLimit({ F });
    let t = 0;
    for (let s = 0; s < 40; s++) { LS.lsStep(phi, N, N, { F }, dt); t += dt; }
    check(Math.abs(radius(phi) - (30 + F * t)) < 0.5, `F = ${F}: radius ${radius(phi).toFixed(2)} vs ${(30 + F * t).toFixed(2)}`);
  }
  // stability: Δt at the limit stays bounded, 1.5 × the limit explodes
  for (const [f, stable] of [[1, true], [1.5, false]] as [number, boolean][]) {
    const phi = LS.sdfCircle(N, N, 63.5, 63.5, 20), dt = f * LS.cflLimit({ F: 1 });
    for (let s = 0; s < 60; s++) LS.lsStep(phi, N, N, { F: 1 }, dt);
    let mx = 0; for (const v of phi) mx = Math.max(mx, Math.abs(v));
    check(stable ? mx < 200 : mx > 1e4, `Δt = ${f} × limit: max|φ| = ${mx.toExponential(2)}`);
  }
  // curvature flow: r² = r0² − 2t (within 1 pixel)
  {
    const phi = LS.sdfCircle(N, N, 63.5, 63.5, 40), dt = 0.2;
    let t = 0;
    for (let s = 1; s <= 2500; s++) {
      LS.lsStep(phi, N, N, { eps: 1 }, dt); t += dt;
      if (s % 20 === 0) LS.reinitExact(phi, N, N);
      if (s % 500 === 0) check(Math.abs(radius(phi) - Math.sqrt(1600 - 2 * t)) < 1, `curvature flow at t = ${t}: r = ${radius(phi).toFixed(2)} vs ${Math.sqrt(1600 - 2 * t).toFixed(2)}`);
    }
  }
  // area under curvature flow drops by 2π per unit time (any closed curve)
  {
    const phi = shapePhi("star"), dt = 0.9 * LS.cflLimit({ eps: 1 });
    const a0 = LS.insideArea(phi, N, N);
    let t = 0;
    for (let s = 0; s < 200; s++) { LS.lsStep(phi, N, N, { eps: 1 }, dt); t += dt; }
    const rate = (a0 - LS.insideArea(phi, N, N)) / t;
    check(Math.abs(rate - 2 * Math.PI) < 0.3, `star: dA/dt = ${rate.toFixed(3)} vs 2π`);
  }
  // topology: two blobs merge under F = 1 after the gap closes; dumbbell splits under F = −1
  {
    const two = shapePhi("two");
    for (let s = 0; s < 34; s++) LS.lsStep(two, N, N, { F: 1 }, 0.25);   // t = 8.5 > gap/2 = 6
    check(LS.zeroContour(two, N, N).lines.length === 1, "two blobs merge");
    const db = shapePhi("dumbbell");
    for (let s = 0; s < 34; s++) LS.lsStep(db, N, N, { F: -1 }, 0.25);    // t = 8.5 > half-width 6
    check(LS.zeroContour(db, N, N).lines.length === 2, "dumbbell splits under shrinking");
  }
  // explicit polygon: a circle of vertices grows by F dt
  {
    const P = LS.radialPoly(0, 0, () => 10, 400), Q = LS.explicitStep(P, 1, 0, 0.5);
    check(Q.every(q => Math.abs(Math.hypot(q[0], q[1]) - 10.5) < 1e-3), "explicit step moves a circle outward by F Δt");
    const Qk = LS.explicitStep(P, 0, 1, 0.5);
    check(Qk.every(q => Math.abs(Math.hypot(q[0], q[1]) - (10 - 0.05)) < 1e-3), "explicit curvature step shrinks a circle by ε Δt / r");
  }
}

// ---- reinitialisation ---------------------------------------------------------------------
{
  for (const method of ["exact", "pde"]) {
    const phi = LS.fromFn(N, N, (x, y) => ((x - 63.5) ** 2 + (y - 63.5) ** 2 - 900) / 60);
    const r0 = radius(phi);
    if (method === "exact") LS.reinitExact(phi, N, N); else LS.reinitPDE(phi, N, N, 40);
    check(Math.abs(radius(phi) - r0) < 0.5, `${method} reinitialisation keeps the contour: ${r0.toFixed(3)} → ${radius(phi).toFixed(3)}`);
    check(Math.abs(LS.meanGradNearFront(phi, N, N) - 1) < 0.05, `${method} reinitialisation gives |∇φ| ≈ 1: ${LS.meanGradNearFront(phi, N, N).toFixed(3)}`);
  }
}

// ---- smoothed Heaviside and delta ---------------------------------------------------------
{
  for (const eps of [0.5, 1, 3]) {
    let s = 0;
    for (let x = -2000; x <= 2000; x += 0.01) s += LS.delta(x, eps) * 0.01;
    check(Math.abs(s - 1) < 1e-3, `∫δ_ε = 1 for ε = ${eps}: ${s}`);
    check(LS.heaviside(0, eps) === 0.5, "H_ε(0) = 1/2");
    const h = 1e-6, num = (LS.heaviside(0.7 + h, eps) - LS.heaviside(0.7 - h, eps)) / (2 * h);
    check(Math.abs(num - LS.delta(0.7, eps)) < 1e-6, "δ_ε = H_ε'");
  }
}

// ---- Chan–Vese -----------------------------------------------------------------------------
{
  // means and energy on a tiny image, by hand: I = 1 1 2 6 7 7 6 1 (one row), inside = pixels 3..6
  const I = [1, 1, 2, 6, 7, 7, 6, 1], phi = [1, 1, 1, -1, -1, -1, -1, 1];
  const m = LS.cvMeans(I, phi);
  check(m.c1 === 6.5 && m.c2 === 1.25, `cvMeans: ${m.c1}, ${m.c2}`);
  // energy non-increasing (in blocks of 5 steps) on the noisy test image from the grid start
  const sc = CV.scene("shapes"), J = CV.addNoise(sc.img, 0.15), p = CV.initPhi("grid"), P = { lambda: 0.1, nu: 0, eps: 1 };
  let prev = LS.cvEnergy(J, p, N, N, P).total, ups = 0, worst = 0;
  const e0 = prev;
  for (let s = 1; s <= 200; s++) {
    LS.cvStep(J, p, N, N, P, 10);
    if (s % 5 === 0) {
      const e = LS.cvEnergy(J, p, N, N, P).total;
      if (e > prev + 1e-9) { ups++; worst = Math.max(worst, (e - prev) / prev); }
      prev = e;
    }
  }
  check(prev < 0.6 * e0, `Chan–Vese lowers the energy: ${e0.toFixed(1)} → ${prev.toFixed(1)}`);
  check(worst < 0.005, `Chan–Vese energy non-increasing up to 0.5% per 5 steps (${ups} small rises, worst ${(100 * worst).toFixed(2)}%)`);
  check(CV.wrongPixels(p, sc.mask) < 300, `Chan–Vese on the noisy test image: ${CV.wrongPixels(p, sc.mask)} wrong pixels`);
  // λ = 0 converges to thresholding at (c1 + c2)/2 on the noise-free image
  const p0 = CV.initPhi("grid");
  for (let s = 0; s < 300; s++) LS.cvStep(sc.img, p0, N, N, { lambda: 0, nu: 0, eps: 1 }, 10);
  const mm = LS.cvMeans(sc.img, p0), th = (mm.c1 + mm.c2) / 2;
  let dif = 0;
  for (let q = 0; q < N * N; q++) if ((p0[q] < 0) !== (mm.c1 >= mm.c2 ? sc.img[q] > th : sc.img[q] <= th)) dif++;
  check(dif === 0, `λ = 0: Chan–Vese = threshold at (c1 + c2)/2 (${dif} pixels differ)`);
}

// ---- edge-stopping function and the gac shrink-wrap -----------------------------------------
{
  const e = LS.edgeStop([0, 2, 4], 3, 1, 2);
  check(Math.abs(e.g[0] - 1) < 1e-12 && Math.abs(e.g[1] - 0.5) < 1e-12 && Math.abs(e.g[2] - 0.2) < 1e-12, `g = 1/(1 + m²/k²): ${Array.from(e.g)}`);
  const { img, mask } = GAC.synthClean(), I = GAC.addNoise(img, 12), E = GAC.edgeMap(I, 1.5, 8);
  const T = GAC.gacTerms(E, { c: -0.3, eps: 0.3, a: 5 }), dt = 0.8 * LS.cflLimit(T, N * N), phi = GAC.boxPhi();
  for (let s = 1; s <= 1300; s++) { LS.lsStep(phi, N, N, T, dt); if (s % 20 === 0) LS.reinitExact(phi, N, N); }
  let wrong = 0;
  for (let q = 0; q < N * N; q++) if ((phi[q] < 0 ? 1 : 0) !== mask[q]) wrong++;
  check(LS.zeroContour(phi, N, N).lines.length === 4, "shrink-wrap splits into the 4 objects");
  check(wrong < 150, `shrink-wrap: ${wrong} wrong pixels`);
}

console.log(`levelsets: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
