// Tests for the edge-preserving smoothing day: lib/pde.ts and lib/inpaint.ts against closed
// forms, other course libraries and the by-hand values of the units heat-equation/,
// perona-malik/, tv-flow/, bilateral-nlm/, inpainting/. Run alone:
//   node_modules/.bin/esbuild demos/test/pdesmooth-test.ts --bundle --platform=node --log-level=warning | node
import {
  heatStep, heatSteps, implicitStep1d, implicitStep2d, amplification, stabilityLimit, nonlinearStep, diffusivity, flux,
  tvFlowStep, dirichletEnergy, pmEnergy, penaltyOf, edgeWidth1090, staircaseIndex, testCard, addNoise, effectiveN, gaussianHeat,
  bilateralWeights, jointBilateral, nlmWeights, nlm1dAt, sigmaOfTime,
} from "../src/lib/pde";
import { harmonicJacobi, harmonicGaussSeidel, harmonicSolve, harmonicResidual, TvInpaint, exemplarInpaint, initHole, fillFront } from "../src/lib/inpaint";
import { gmrf1d, tv1d, tv2d } from "../src/lib/smooth";
import { bilateral } from "../src/lib/denoise";
import { fromArray, gaussianBlur } from "../src/lib/imageops";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const arrNear = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => a.length === b.length && Array.from(a).every((v, i) => near(v, b[i], tol));
const fmt = (a: ArrayLike<number>) => Array.from(a).map(v => v.toFixed(3)).join(", ");

// ---- unit 1: the explicit step by hand ----
{
  const row = [0, 0, 4, 0, 0];
  check(arrNear(heatStep(row, 5, 1, 0.25), [0, 1, 2, 1, 0], 1e-12), "r = 1/4");
  check(arrNear(heatStep(row, 5, 1, 0.5), [0, 2, 0, 2, 0], 1e-12), "r = 1/2");
  check(arrNear(heatStep(row, 5, 1, 1), [0, 4, -4, 4, 0], 1e-12), "r = 1");
  for (const dt of [0.25, 0.5, 1]) check(near(heatStep(row, 5, 1, dt).reduce((s, v) => s + v, 0), 4, 1e-12), "mass kept " + dt);
  // 2D: centre 8, neighbours 0
  const p = [0, 0, 0, 0, 8, 0, 0, 0, 0];
  check(near(heatStep(p, 3, 3, 0.2, "zero")[4], 1.6, 1e-12) && near(heatStep(p, 3, 3, 0.3, "zero")[4], -1.6, 1e-12), "2D centre step");
  check(near(stabilityLimit(1), 0.5, 0) && near(stabilityLimit(2), 0.25, 0) && near(stabilityLimit(3), 1 / 6, 1e-15), "limits");
  check(near(amplification(0.25, Math.PI), 0, 1e-12) && near(amplification(0.5, Math.PI), -1, 1e-12) && near(amplification(0.6, Math.PI), -1.4, 1e-12), "A(π)");
  check(amplification(5, Math.PI, "implicit") > 0 && amplification(5, Math.PI, "implicit") <= 1, "implicit A in (0, 1]");
}

// ---- heat = Gaussian, mirror keeps the mean ----
{
  const n = 64, u = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) u[i] = Math.sin(i * 0.37) * 40 + ((i % n) > 30 ? 100 : 0) + (Math.floor(i / n) > 25 ? 50 : 0);
  const dt = 0.1, steps = 80, t = dt * steps, s = sigmaOfTime(t);
  const H = heatSteps(u, n, n, dt, steps, "mirror");
  const G = gaussianBlur(fromArray(n, n, u), s, "mirror").data;
  let maxRel = 0, range = 150;
  for (let y = 12; y < n - 12; y++) for (let x = 12; x < n - 12; x++) maxRel = Math.max(maxRel, Math.abs(H[y * n + x] - G[y * n + x]) / range);
  check(maxRel < 0.01, `heat for t = ${t} vs Gaussian σ = ${s.toFixed(3)}: max rel diff ${maxRel}`);
  // with the matching (half-sample symmetric) border the agreement holds up to the border too
  const GH = gaussianHeat(u, n, n, s, "mirror");
  let maxAll = 0; for (let i = 0; i < n * n; i++) maxAll = Math.max(maxAll, Math.abs(H[i] - GH[i]) / range);
  check(maxAll < 0.01, "heat vs gaussianHeat incl. borders: " + maxAll);
  const m0 = u.reduce((a, b) => a + b, 0) / u.length, m1 = H.reduce((a, b) => a + b, 0) / H.length;
  check(near(m0, m1, 1e-9), "mirror keeps mean");
  const Z = heatSteps(u.map(v => v + 100), n, n, 0.2, 50, "zero");
  check(Z.reduce((a, b) => a + b, 0) < u.reduce((a, b) => a + b + 100, 0), "zero border leaks");
}

// ---- stability: checkerboard grows above r = 1/2 ----
{
  const n = 64, u = new Float64Array(n);
  for (let i = 0; i < n; i++) u[i] = (i % 2 ? 1 : -1) * 1e-3 + Math.sin(i / 5);
  let a: Float64Array = Float64Array.from(u), b: Float64Array = Float64Array.from(u);
  for (let k = 0; k < 200; k++) { a = heatStep(a, n, 1, 0.5); b = heatStep(b, n, 1, 0.52); }
  check(Math.max(...a.map(Math.abs)) < 1.01, "r = 0.5 bounded");
  check(Math.max(...b.map(Math.abs)) > 10, "r = 0.52 explodes: " + Math.max(...b.map(Math.abs)));
}

// ---- implicit steps ----
{
  check(arrNear(implicitStep1d([0, 3, 0], 1), [0.75, 1.5, 0.75], 1e-12), "implicit (0,3,0) dt 1");
  const y = [1, 5, 2, 8, 3, 3, 9, 0];
  check(arrNear(implicitStep1d(y, 2.5), gmrf1d(y, 2.5), 1e-12), "implicit 1D = gmrf1d");
  // 2D DCT step vs CG-free check: (I + dt L) v = u holds
  const w = 16, h = 8, u = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) u[i] = (i * 7919) % 31;
  const dt = 3, v = implicitStep2d(u, w, h, dt);
  let maxr = 0;
  for (let yy = 0; yy < h; yy++) for (let x = 0; x < w; x++) {
    const p = yy * w + x;
    let lap = 0;
    if (x > 0) lap += v[p] - v[p - 1]; if (x < w - 1) lap += v[p] - v[p + 1];
    if (yy > 0) lap += v[p] - v[p - w]; if (yy < h - 1) lap += v[p] - v[p + w];
    maxr = Math.max(maxr, Math.abs(v[p] + dt * lap - u[p]));
  }
  check(maxr < 1e-9, "implicit 2D solves (I + dt L) v = u: " + maxr);
}

// ---- unit 2: Perona–Malik ----
{
  const row = [10, 12, 10, 50, 50];
  const heat = heatStep(row, 5, 1, 0.25), pm = nonlinearStep(row, 5, 1, 0.25, "g2", 5);
  check(arrNear(heat, [10.5, 11, 20.5, 40, 50], 1e-12), "heat row: " + fmt(heat));
  check(arrNear(pm, [10.43, 11.14, 10.58, 49.85, 50], 0.006), "PM row: " + fmt(pm));
  const sh = nonlinearStep([0, 2, 6, 10, 12], 5, 1, 0.25, "g2", 2);
  check(arrNear(sh, [0.25, 1.95, 6, 10.05, 11.75], 1e-12), "sharpening row: " + fmt(sh));
  check(near(diffusivity("g2", 1, 5), 1 / 1.04, 1e-12) && near(diffusivity("g2", 5, 5), 0.5, 1e-12) && near(diffusivity("g2", 20, 5), 1 / 17, 1e-12), "g2 values");
  // flux peaks
  check(flux("g2", 5, 5) > flux("g2", 4.9, 5) && flux("g2", 5, 5) > flux("g2", 5.1, 5), "g2 flux peak at K");
  check(flux("g1", 5 / Math.SQRT2, 5) > flux("g1", 3.4, 5) && flux("g1", 5 / Math.SQRT2, 5) > flux("g1", 3.7, 5), "g1 flux peak at K/√2");
  // K -> infinity = heat
  const n = 32, u = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) u[i] = (i * 2654435761 % 255);
  check(arrNear(nonlinearStep(u, n, n, 0.2, "g2", 1e9), heatStep(u, n, n, 0.2), 1e-6), "PM K→∞ = heat");
  // no new extrema with dt ≤ 1/4
  let v: Float64Array = Float64Array.from(u), lo = Math.min(...u), hi = Math.max(...u), ok = true;
  for (let k = 0; k < 50; k++) { v = nonlinearStep(v, n, n, 0.25, "g1", 20); if (Math.min(...v) < lo - 1e-9 || Math.max(...v) > hi + 1e-9) ok = false; }
  check(ok, "PM max principle");
  // pre-smoothing removes the spike: with g1, K = 5, a dot of 30 survives without, decays with σ = 1
  const m = 21, d = new Float64Array(m * m); d[10 * m + 10] = 30;
  let a: Float64Array = Float64Array.from(d), b: Float64Array = Float64Array.from(d);
  for (let k = 0; k < 200; k++) { a = nonlinearStep(a, m, m, 0.25, "g1", 5); b = nonlinearStep(b, m, m, 0.25, "g1", 5, 1); }
  check(a[10 * m + 10] > 29.9 && b[10 * m + 10] < 3, `spike kept ${a[10 * m + 10].toFixed(2)}, removed with σ = 1: ${b[10 * m + 10].toFixed(2)}`);
}

// ---- unit 3: TV ----
{
  const y = [0, 0, 0, 6, 6, 0, 0, 0];
  check(arrNear(tv1d(y, 2), [2 / 3, 2 / 3, 2 / 3, 4, 4, 2 / 3, 2 / 3, 2 / 3], 1e-9), "ROF λ = 2: " + fmt(tv1d(y, 2)));
  check(arrNear(tv1d(y, 6), new Array(8).fill(1.5), 1e-9), "ROF λ = 6: all 1.5");
  check(arrNear(tv1d(y, 4.5), new Array(8).fill(1.5), 1e-9), "ROF λ = 4.5: all 1.5");
  // TV flow in 1D: plateau drops at 2/width per unit time (small eps)
  let u: Float64Array = Float64Array.from([0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0]);
  const eps = 0.01, dt = eps / 2.5;
  for (let t = 0; t < 1; t += dt) u = tvFlowStep(u, 16, 1, dt, eps);
  check(near(u[7], 6 - 2 / 4, 0.05), "1D TV flow plateau drop 2/width: " + u[7].toFixed(3));
  // disc contrast under ROF falls by ≈ 2λ/r
  for (const r of [6, 10]) {
    const n = 64, img = new Float64Array(n * n);
    for (let yy = 0; yy < n; yy++) for (let x = 0; x < n; x++) if ((x - 31.5) ** 2 + (yy - 31.5) ** 2 <= r * r) img[yy * n + x] = 50;
    const lam = 3, out = tv2d(img, n, n, lam, 800);
    const drop = 50 - out[32 * n + 32], pred = 2 * lam / r;
    check(Math.abs(drop - pred) / pred < 0.15, `disc r = ${r}: drop ${drop.toFixed(3)} vs 2λ/r = ${pred.toFixed(3)}`);
  }
  check(near(penaltyOf("pm", 10, 2), 2 * Math.log(26), 1e-12) && near(10 * penaltyOf("pm", 1, 2), 20 * Math.log(1.25), 1e-12), "PM penalties");
  check(near(dirichletEnergy([0, 1, 3], 3, 1), 0.5 * (1 + 4), 1e-12) && near(pmEnergy([0, 10], 2, 1, 2), 2 * Math.log(26), 1e-12), "energies");
}

// ---- scores ----
{
  const ramp = Array.from({ length: 11 }, (_, i) => i * 10);
  check(near(staircaseIndex(ramp, 0, 10), 1, 1e-12), "staircase ramp = 1");
  const step = [0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100];
  check(near(staircaseIndex(step, 0, 10), 10, 1e-12), "staircase step = n");
  check(near(edgeWidth1090(ramp, 0, 10), 8, 1e-9), "10–90 width of a ramp");
}

// ---- unit 4: bilateral and NLM ----
{
  // bilateral at the centre of (12, 10, 50), σs large, σr = 10
  const bw = bilateralWeights([12, 10, 50], 3, 1, 1, 0, 1e6, 10, 1);
  check(near(bw.estimate, (Math.exp(-0.02) * 12 + 10 + Math.exp(-8) * 50) / (Math.exp(-0.02) + 1 + Math.exp(-8)), 1e-9), "bilateral by hand " + bw.estimate);
  check(near(effectiveN([1, 1, 1, 1]), 4, 1e-12) && near(effectiveN([1, 0.1, 0.1, 0.1]), 1.69 / 1.03, 1e-12), "N_eff");
  // 1D NLM by hand
  const x = [11, 9, 30, 10, 12, 29, 9, 11];
  const r = nlm1dAt(x, 3, 1, 10);
  check(near(r.estimate, 10.93, 0.005), "NLM 1D estimate " + r.estimate.toFixed(4));
  // σr → ∞: bilateral = Gaussian (window-normalised)
  const n = 24, u = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) u[i] = (i * 40503) % 97;
  const b = bilateral(u, n, n, 1.5, 1e9, 3), j = jointBilateral(u, u, n, n, 1.5, 1e9, 3);
  check(arrNear(b, j, 1e-9), "joint with guide = self equals bilateral");
  const ws: number[] = []; for (let k = -3; k <= 3; k++) ws.push(Math.exp(-k * k / (2 * 1.5 * 1.5)));
  const wsum = ws.reduce((s, v) => s + v, 0);
  const g = gaussianBlur(fromArray(n, n, u), 1.5, "mirror");
  void g;
  // direct Gaussian with the same truncated kernel and mirror border
  let maxd = 0;
  for (let yy = 4; yy < n - 4; yy++) for (let xx = 4; xx < n - 4; xx++) {
    let s = 0;
    for (let a = -3; a <= 3; a++) for (let c = -3; c <= 3; c++) s += ws[a + 3] * ws[c + 3] * u[(yy + a) * n + xx + c];
    maxd = Math.max(maxd, Math.abs(s / (wsum * wsum) - b[yy * n + xx]));
  }
  check(maxd < 1e-6, "bilateral σr→∞ = Gaussian: " + maxd);
  // NLM h → ∞: box mean over the search window
  const nm = nlmWeights(u, n, n, 10, 10, 1, 3, 1e9);
  let box = 0; for (let a = -3; a <= 3; a++) for (let c = -3; c <= 3; c++) box += u[(10 + a) * n + 10 + c];
  check(near(nm.estimate, box / 49, 1e-6), "NLM h→∞ = box mean");
  check(near(nm.neff, 49, 1e-6), "NLM h→∞ N_eff = 49");
  // self-fix: on a unique spot the own weight dominates without the fix
  const card = testCard(), noisy = addNoise(card.truth, 10, 3);
  const a1 = nlmWeights(noisy, 128, 128, 46, 76, 2, 7, 15, false), a2 = nlmWeights(noisy, 128, 128, 46, 76, 2, 7, 15, true);
  check(a1.selfWeight > 0.5 && a2.selfWeight < a1.selfWeight, `unique spot self weight ${a1.selfWeight.toFixed(2)} → ${a2.selfWeight.toFixed(2)}`);
  const lat = nlmWeights(noisy, 128, 128, 82, 20, 2, 7, 15, false);
  check(lat.neff > 5 * a1.neff, `lattice N_eff ${lat.neff.toFixed(1)} vs unique ${a1.neff.toFixed(1)}`);
}

// ---- unit 5: inpainting ----
{
  // 1D: 2 _ _ _ 10
  const f = [2, 0, 0, 0, 10], m = [0, 1, 1, 1, 0];
  check(arrNear(harmonicSolve(f, m, 5, 1).u, [2, 4, 6, 8, 10], 1e-9), "harmonic 1D");
  let j: Float64Array = Float64Array.from(f);
  j = harmonicJacobi(j, m, 5, 1); check(arrNear(j, [2, 1, 0, 5, 10], 1e-12), "Jacobi sweep 1 " + fmt(j));
  j = harmonicJacobi(j, m, 5, 1); check(arrNear(j, [2, 1, 3, 5, 10], 1e-12), "Jacobi sweep 2 " + fmt(j));
  // 2D two-pixel hole: a's other neighbours 10, b's 30 -> 14, 26
  const w = 4, h = 3, g = [10, 10, 30, 30, 10, 0, 0, 30, 10, 10, 30, 30];
  const mm = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0];
  const sol = harmonicSolve(g, mm, w, h, 1e-14);
  check(near(sol.u[5], 14, 1e-8) && near(sol.u[6], 26, 1e-8), `two-pixel hole ${sol.u[5]}, ${sol.u[6]}`);
  check(sol.residual < 1e-8, "solve residual " + sol.residual);
  // Jacobi and Gauss-Seidel converge to the solve
  const card = testCard(), W = 128, mask = new Uint8Array(W * W);
  for (let y = 40; y < 60; y++) for (let x = 20; x < 36; x++) mask[y * W + x] = 1;
  const S = harmonicSolve(card.truth, mask, W, W);
  let J: Float64Array = initHole(card.truth, mask), GS: Float64Array = initHole(card.truth, mask);
  for (let k = 0; k < 3000; k++) J = harmonicJacobi(J, mask, W, W);
  for (let k = 0; k < 1500; k++) harmonicGaussSeidel(GS, mask, W, W);
  check(arrNear(J, S.u, 1e-3) && arrNear(GS, S.u, 1e-3), "Jacobi / GS converge to the solve");
  check(harmonicResidual(S.u, mask, W, W) < 1e-6, "solve residual small");
  // TV connectivity: dark bar of width 6 with a gap of 4 connects, 10 breaks
  const bar = (gap: number) => {
    const bw = 64, bh = 32, u = new Float64Array(bw * bh).fill(200), mk = new Uint8Array(bw * bh);
    for (let y = 13; y < 19; y++) for (let x = 0; x < bw; x++) u[y * bw + x] = 50;
    const x0 = 32 - gap / 2;
    for (let y = 8; y < 24; y++) for (let x = x0; x < x0 + gap; x++) mk[y * bw + x] = 1;
    const t = new TvInpaint(u, mk, bw, bh);
    t.run(4000);
    return t.u[16 * bw + 32];
  };
  const c4 = bar(4), c10 = bar(10);
  check(c4 < 90 && c10 > 160, `TV bar: gap 4 centre ${c4.toFixed(1)}, gap 10 centre ${c10.toFixed(1)}`);
  // exemplar: a hole in an exactly periodic texture is reproduced
  const tw = 48, th = 48, tex = new Float64Array(tw * th);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) tex[y * tw + x] = ((x % 6) < 3 ? 40 : 160) + ((y % 4) < 2 ? 0 : 30);
  const hm = new Uint8Array(tw * th);
  for (let y = 18; y < 28; y++) for (let x = 20; x < 30; x++) hm[y * tw + x] = 1;
  const ex = exemplarInpaint(tex, hm, tw, th, { patch: 3, search: 20 });
  check(arrNear(ex.u, tex, 1e-9), "exemplar reproduces a periodic texture");
  check(ex.steps.length > 0 && fillFront(hm, tw, th).length === 36, "fill front of a 10×10 hole has 36 pixels");
}

console.log(`pdesmooth: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
