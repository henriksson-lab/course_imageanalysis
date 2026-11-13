// Tests for lib/flow.ts and lib/deform.ts (optical flow day): the by-hand numbers of the
// exercises, Lucas–Kanade against exact sub-pixel shifts, the pyramid on a large shift,
// Horn–Schunck updates and energy, error measures, B-spline FFD, Jacobian, bending energy
// (gradient against finite differences), similarity gradients, mutual information, and a
// registration run. Run alone:
//   node_modules/.bin/esbuild demos/test/opticalflow-test.ts --bundle --platform=node --log-level=warning | node
import {
  flow1d, windowLS1d, solveLK, lucasKanade, lkIterative, lkPyramid, HornSchunck, endpointError, angularErrorAt,
  makePair, texture, sampleField, fieldFn, divergence, curlScreen, warpByFlow, createFlow, interiorMean, upsampleFlow,
} from "../src/lib/flow";
import {
  bspline3, bspline3d, makeGrid, ffdDisplacement, ffdAdjoint, refineGrid, jacobianDeterminant, bendingEnergy,
  similarityCost, mutualInformation, randomGrid, deformForTruth, FFDRegistration, displacementError, remap, warpMoving,
} from "../src/lib/deform";
import { Grey, fromFunction, gaussianBlur, cropGrey } from "../src/lib/imageops";
import { fourierShift2d } from "../src/lib/registration";

declare const require: (m: string) => any;
declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol;

/** Decode the course micrograph (8-bit grey PNG, non-interlaced) in node. */
function loadCellNode(): Grey {
  const fs = require("fs"), zlib = require("zlib"), path = require("path");
  const buf: Uint8Array = fs.readFileSync(path.join(process_cwd(), "demos/src/fourier2d/cell256.png"));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 8, w = 0, h = 0; const idat: Uint8Array[] = [];
  while (p < buf.length) {
    const len = dv.getUint32(p), type = String.fromCharCode(...buf.subarray(p + 4, p + 8));
    if (type === "IHDR") { w = dv.getUint32(p + 8); h = dv.getUint32(p + 12); }
    if (type === "IDAT") idat.push(buf.subarray(p + 8, p + 8 + len));
    p += 12 + len;
  }
  const raw: Uint8Array = zlib.inflateSync(Buffer_concat(idat));
  const out = new Float32Array(w * h), prev = new Uint8Array(w), cur = new Uint8Array(w);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (w + 1)], row = raw.subarray(y * (w + 1) + 1, (y + 1) * (w + 1));
    for (let x = 0; x < w; x++) {
      const a = x > 0 ? cur[x - 1] : 0, b = prev[x], c = x > 0 ? prev[x - 1] : 0;
      let v = row[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    for (let x = 0; x < w; x++) out[y * w + x] = cur[x];
    prev.set(cur);
  }
  return { width: w, height: h, data: out };
}
function process_cwd(): string { return (globalThis as any).process.cwd(); }
function Buffer_concat(parts: Uint8Array[]): Uint8Array { return (globalThis as any).Buffer.concat(parts); }

// ---- unit 1: the 1D bump by hand ----
{
  const f1 = [0, 0, 1, 4, 1, 0, 0], f2 = [0, 0, 0, 1, 4, 1, 0];
  const r = flow1d(f1, f2);
  [0.5, 2, 0, -2, -0.5].forEach((v, k) => check(near(r.ix[k + 1], v), `bump Ix(${k + 1})`));
  [0, -1, -3, 3, 1].forEach((v, k) => check(near(r.it[k + 1], v), `bump It(${k + 1})`));
  [0, 0.5, NaN, 1.5, 2].forEach((v, k) => check(Number.isNaN(v) ? Number.isNaN(r.u[k + 1]) : near(r.u[k + 1], v), `bump u(${k + 1})`));
  const ls = windowLS1d(r.ix, r.it, 2);
  check(near(ls[3], 1), `bump window LS = ${ls[3]}`);
}
// ---- unit 1: two constraint lines (1,0,-2), (0,1,-1) → (2, 1) ----
{
  const [u, v] = solveLK(1, 0, 1, -2, -1);
  check(near(u, 2) && near(v, 1), "two constraint lines");
}
// ---- unit 2: four-pixel LS ----
{
  const sums = (px: [number, number, number][]) => {
    let a = 0, b = 0, c = 0, p = 0, q = 0;
    for (const [x, y, t] of px) { a += x * x; b += x * y; c += y * y; p += x * t; q += y * t; }
    return [a, b, c, p, q] as const;
  };
  let s = sums([[1, 0, -2], [1, 0, -2], [0, 1, -1], [0, 1, -1]]);
  let r = solveLK(...s);
  check(near(r[0], 2) && near(r[1], 1) && near(r[2], 2) && near(r[3], 2), "four pixels (2,1)");
  s = sums([[1, 0, -2], [1, 0, -2], [1, 0, -2], [1, 0, -2]]);
  r = solveLK(...s);
  check(near(r[0], 2) && near(r[1], 0) && near(r[3], 0) && near(r[2], 4), "edge: u = 2, v undetermined (min norm 0)");
  s = sums([[1, 1, -3], [1, 1, -3], [1, 1, -3], [1, 1, -3]]);
  r = solveLK(...s);
  check(near(r[0], 1.5, 1e-12) && near(r[1], 1.5, 1e-12) && near(r[2], 8, 1e-12) && near(r[3], 0, 1e-12), "45° edge: normal flow (1.5, 1.5)");
}
// ---- LK on the micrograph: exact sub-pixel shift (0.3, −0.6) ----
const cell = loadCellNode();
check(cell.width === 256 && cell.height === 256, "cell decoded");
const crop = cropGrey(cell, 64, 64, 128, 128);
{
  const sm = gaussianBlur(crop, 1.5);
  const I2: Grey = { width: 128, height: 128, data: Float32Array.from(fourierShift2d(sm.data, 128, 128, 0.3, -0.6)) };
  const r = lucasKanade(sm, I2, { sigma: 1, rho: 3 });
  const truth = sampleField(() => [0.3, -0.6], 128, 128);
  const e = endpointError(r, truth, 16);
  check(e.mean < 0.05, `LK sub-pixel shift: mean EPE ${e.mean.toFixed(4)} < 0.05`);
  // a 2.5 px shift: one linearised solve is biased, warping iterations fix it
  const J2: Grey = { width: 128, height: 128, data: Float32Array.from(fourierShift2d(sm.data, 128, 128, 2.5, 0)) };
  const t2 = sampleField(() => [2.5, 0], 128, 128);
  const one = endpointError(lucasKanade(sm, J2, { sigma: 1, rho: 3 }), t2, 16).mean;
  const three = endpointError(lkIterative(sm, J2, { sigma: 1, rho: 3, iters: 3 }), t2, 16).mean;
  check(three < 0.5 * one, `warping iterations improve a 2.5 px shift (${one.toFixed(3)} → ${three.toFixed(3)})`);
  console.log(`  2.5 px shift: one LK solve EPE ${one.toFixed(3)}, 3 iterations ${three.toFixed(3)}`);
}
// ---- pyramid rescues an 8 px shift ----
{
  const pair = makePair(crop, "translation", { s: 8, angle: 0 }, 0, 1);
  const single = lkIterative(pair.I1, pair.I2, { sigma: 1, rho: 4, iters: 3 });
  const pyr = lkPyramid(pair.I1, pair.I2, { sigma: 1, rho: 4, levels: 4, iters: 3 });
  const es = endpointError(single, pair.truth, 16).mean, ep = endpointError(pyr, pair.truth, 16).mean;
  check(es > 2, `single-level LK fails on 8 px (EPE ${es.toFixed(2)})`);
  check(ep < 0.5, `pyramid LK succeeds on 8 px (EPE ${ep.toFixed(3)})`);
  console.log(`  8 px shift: single-level EPE ${es.toFixed(2)}, 4-level pyramid EPE ${ep.toFixed(3)}`);
}
// ---- upsampleFlow doubles ----
{
  const f = createFlow(8, 8); f.u.fill(1); f.v.fill(-0.5);
  const g = upsampleFlow(f, 16, 16);
  check(near(g.u[100], 2) && near(g.v[100], -1), "upsampleFlow doubles");
}
// ---- Horn–Schunck: the by-hand update ----
{
  const mk = (ix: number, it: number) => {
    const I1: Grey = { width: 1, height: 1, data: new Float32Array([0]) };
    const hs = new HornSchunck(I1, I1, { lambda: 1, sigma: 0 });
    return hs;
  };
  // 3×1 image: the middle pixel has Ix = 2, It = −4; neighbours fixed at given values by hand formula
  const upd = (Ix: number, Iy: number, It: number, ub: number, vb: number, lam: number) => {
    const k = (Ix * ub + Iy * vb + It) / (4 * lam + Ix * Ix + Iy * Iy);
    return [ub - Ix * k, vb - Iy * k];
  };
  void mk;
  let r = upd(2, 0, -4, 0, 0, 1); check(near(r[0], 1) && near(r[1], 0), "HS update λ=1");
  r = upd(2, 0, -4, 0, 0, 0); check(near(r[0], 2), "HS update λ=0");
  r = upd(2, 0, -4, 2, 1, 1); check(near(r[0], 2) && near(r[1], 1), "HS update on the line: unchanged");
  // the class reproduces the formula at an interior pixel of a 3×3 image with neighbours set
  const I1 = fromFunction(3, 3, () => 0), I2 = fromFunction(3, 3, () => 0);
  const hs = new HornSchunck(I1, I2, { lambda: 1, sigma: 0 });
  hs.ix.fill(0); hs.it.fill(0); hs.ix[4] = 2; hs.it[4] = -4;
  (hs as any).updatePixel(1, 1);
  check(near(hs.u[4], 1) && near(hs.v[4], 0), "HornSchunck.updatePixel by hand");
}
// ---- Horn–Schunck on a textured rotation: energy decreases, exact solution is fixed ----
{
  const tex = texture(64, 64, 4, 2, 40);
  const pair = makePair(tex, "rotation", { s: 0.02 }, 0, 1);
  const hs = new HornSchunck(pair.I1, pair.I2, { lambda: 50, sigma: 1 });
  let prev = hs.energy().total, mono = true;
  for (let k = 0; k < 60; k++) { hs.step(1); const e = hs.energy().total; if (e > prev + 1e-6 * Math.abs(prev)) mono = false; prev = e; }
  check(mono, "HS energy never increases (red–black Gauss–Seidel)");
  const rob = new HornSchunck(pair.I1, pair.I2, { lambda: 50, sigma: 1, robust: true });
  prev = rob.energy().total; mono = true;
  for (let k = 0; k < 40; k++) { rob.step(1); const e = rob.energy().total; if (e > prev + 1e-6 * Math.abs(prev)) mono = false; prev = e; }
  check(mono, "robust HS energy never increases (reweighting = majorise–minimise)");
  // a uniform flow on the constraint lines everywhere is a fixed point
  const I1 = fromFunction(10, 10, (x, y) => 3 * x + y), hs2 = new HornSchunck(I1, I1, { lambda: 2, sigma: 0 });
  hs2.u.fill(1); hs2.v.fill(-1);
  for (let i = 0; i < 100; i++) hs2.it[i] = -(hs2.ix[i] * 1 + hs2.iy[i] * -1);
  hs2.step(5);
  check(hs2.u.every(x => near(x, 1, 1e-6)) && hs2.v.every(x => near(x, -1, 1e-6)), "HS leaves an exact solution unchanged");
  const ep = endpointError(hs, pair.truth, 8).mean;
  hs.step(500);
  check(endpointError(hs, pair.truth, 8).mean < ep, "more HS iterations reduce the EPE on the rotation");
}
// ---- error measures by hand ----
{
  check(near(Math.hypot(2 - 2.5, 1 - 1), 0.5), "EPE 1");
  const a1 = angularErrorAt(2, 1, 2.5, 1), a2 = angularErrorAt(0.1, 0, 0.6, 0);
  check(a2 > a1, `AE larger at slow pixel (${a1.toFixed(2)}° vs ${a2.toFixed(2)}°)`);
  console.log(`  AE: (2,1) vs (2.5,1): ${a1.toFixed(2)}°, (0.1,0) vs (0.6,0): ${a2.toFixed(2)}°`);
}
// ---- divergence and curl ----
{
  const rot = sampleField(fieldFn("rotation", 33, 33, { s: 0.01 }), 33, 33);
  const exp = sampleField(fieldFn("zoom", 33, 33, { s: 0.05 }), 33, 33);
  const i = 16 * 33 + 16;
  check(near(divergence(rot)[i], 0, 1e-4) && near(curlScreen(rot)[i], 2 * Math.sin(0.01), 1e-4), "rotation: div 0, curl 2Ω");
  check(near(divergence(exp)[i], 0.1, 1e-6) && near(curlScreen(exp)[i], 0, 1e-6), "expansion: div 2k, curl 0");
}
// ---- warpByFlow inverts makePair for a translation ----
{
  const tex = texture(64, 64, 2, 3, 30);
  const pair = makePair(tex, "translation", { s: 2, angle: 90 }, 0, 1);
  const back = warpByFlow(pair.I2, pair.truth.u, pair.truth.v);
  let err = 0; for (let y = 8; y < 56; y++) for (let x = 8; x < 56; x++) err = Math.max(err, Math.abs(back.data[y * 64 + x] - tex.data[y * 64 + x]));
  check(err < 1e-3, `warp back of a translation (max err ${err})`);
}

// ---- unit 4: B-splines ----
check(near(bspline3(0), 4 / 6, 1e-15) && near(bspline3(1), 1 / 6, 1e-15) && near(bspline3(-1), 1 / 6, 1e-15) && bspline3(2) === 0, "β³ at the knots");
check(near(bspline3(0.5), 23 / 48, 1e-15), "β³(0.5) = 23/48");
for (const t of [0.1, 0.37, 0.5, 0.93]) {
  let s = 0; for (let k = -3; k <= 3; k++) s += bspline3(t - k);
  check(near(s, 1, 1e-12), `partition of unity at ${t}`);
  const h = 1e-6;
  check(near(bspline3d(t + 0.6), (bspline3(t + 0.6 + h) - bspline3(t + 0.6 - h)) / (2 * h), 1e-6), `β³' at ${t + 0.6}`);
}
// 1D FFD by hand: knots every 10 px at x = 0..40, coefficients (0,0,6,0,0)
{
  const d = (x: number) => [0, 10, 20, 30, 40].reduce((s, k, i) => s + [0, 0, 6, 0, 0][i] * bspline3((x - k) / 10), 0);
  check(near(d(20), 4) && near(d(10), 1) && near(d(15), 2.875), "1D FFD by hand: 4, 1, 2.875");
  let maxSlope = 0; for (let x = 0; x <= 40; x += 0.01) maxSlope = Math.max(maxSlope, Math.abs((d(x + 1e-4) - d(x - 1e-4)) / 2e-4));
  check(near(maxSlope, 0.4, 1e-3), `steepest slope 6/10 · 2/3 = 0.4: ${maxSlope}`);
  console.log(`  1D FFD: steepest |d'| = ${maxSlope.toFixed(4)} (fold when coefficient > ${(10 / (maxSlope / 6 * 10)).toFixed(3)}·… → c = ${(1 / (maxSlope / 6)).toFixed(3)})`);
}
// 2D grid: one control point, constant field, adjoint, refinement, Jacobian
{
  const W = 48, H = 40, g = makeGrid(W, H, 8);
  const [k, l] = [3, 3], idx = l * g.nx + k;   // knot at (16, 16)
  g.cx[idx] = 9;
  const { u } = ffdDisplacement(g, W, H);
  check(near(u[16 * W + 16], 9 * 4 / 6 * 4 / 6, 1e-9) && near(u[16 * W + 24], 9 * 1 / 6 * 4 / 6, 1e-9) && near(u[24 * W + 24], 9 / 36, 1e-9), "2D FFD: 4, 1, 0.25 for 9 px");
  const c = makeGrid(W, H, 8); c.cx.fill(2.5); c.cy.fill(-1);
  const f = ffdDisplacement(c, W, H);
  check(f.u.every(x => near(x, 2.5, 1e-9)) && f.v.every(x => near(x, -1, 1e-9)), "constant coefficients = translation");
  const jac = jacobianDeterminant(f.u, f.v, W, H);
  check(jac.every(x => near(x, 1, 1e-6)), "Jacobian of a translation = 1");
  const zero = makeGrid(W, H, 8), z = ffdDisplacement(zero, W, H);
  check(jacobianDeterminant(z.u, z.v, W, H).every(x => near(x, 1)), "Jacobian of the identity = 1");
  // adjoint: <A c, g> = <c, A^T g>
  const r = randomGrid(W, H, 8, 2, 3), A = ffdDisplacement(r, W, H);
  const gu = Float64Array.from({ length: W * H }, (_, i) => Math.sin(i)), gv = Float64Array.from({ length: W * H }, (_, i) => Math.cos(0.3 * i));
  let lhs = 0; for (let i = 0; i < W * H; i++) lhs += A.u[i] * gu[i] + A.v[i] * gv[i];
  const At = ffdAdjoint(r, W, H, gu, gv);
  let rhs = 0; for (let i = 0; i < At.cx.length; i++) rhs += r.cx[i] * At.cx[i] + r.cy[i] * At.cy[i];
  check(near(lhs, rhs, 1e-6 * Math.abs(lhs)), "ffdAdjoint is the transpose");
  // refinement keeps the field
  const rf = refineGrid(r, W, H), B = ffdDisplacement(rf, W, H);
  let me = 0; for (let i = 0; i < W * H; i++) me = Math.max(me, Math.abs(A.u[i] - B.u[i]), Math.abs(A.v[i] - B.v[i]));
  check(me < 1e-9, `refineGrid keeps the deformation (max diff ${me})`);
  // bending gradient vs finite differences
  const be = bendingEnergy(r, W, H), hh = 1e-5, i0 = 2 * r.nx + 3;
  const rp = { ...r, cx: Float64Array.from(r.cx) }; rp.cx[i0] += hh;
  const rm = { ...r, cx: Float64Array.from(r.cx) }; rm.cx[i0] -= hh;
  const fd = (bendingEnergy(rp, W, H).value - bendingEnergy(rm, W, H).value) / (2 * hh);
  check(near(be.gx[i0], fd, 1e-6 * Math.max(1, Math.abs(fd))), "bending gradient");
  check(near(bendingEnergy(c, W, H).value, 0, 1e-15), "translation has no bending");
}
// similarity gradients vs finite differences
for (const m of ["ssd", "ncc", "mi"] as const) {
  const n = 400, F = Float64Array.from({ length: n }, (_, i) => 0.5 + 0.45 * Math.sin(0.07 * i * i));
  const Wv = Float64Array.from({ length: n }, (_, i) => 0.5 + 0.4 * Math.sin(0.07 * i * i + 0.3) + 0.03 * Math.cos(i));
  const base = similarityCost(F, Wv, m, 12);
  let ok = true;
  for (const i of [3, 50, 177, 301]) {
    const h = 1e-6, a = Float64Array.from(Wv), b = Float64Array.from(Wv); a[i] += h; b[i] -= h;
    const fd = (similarityCost(F, a, m, 12).cost - similarityCost(F, b, m, 12).cost) / (2 * h);
    if (!near(base.grad[i], fd, 1e-4 * Math.max(1e-3, Math.abs(fd)) + 1e-9)) { ok = false; console.log(`  ${m} grad ${i}: ${base.grad[i]} vs ${fd}`); }
  }
  check(ok, `${m} gradient matches finite differences`);
}
// mutual information by hand
{
  const A = [0, 0, 0, 0, 1, 1, 1, 1], B = A.map(a => 1 - a);
  const r = mutualInformation(A, B, 2);
  check(near(r.mi, 1) && near(r.ha, 1) && near(r.hb, 1) && near(r.hab, 1), "MI(A, 1 − A) = 1 bit");
  const C = [0, 0, 1, 1, 0, 0, 1, 1];
  const r2 = mutualInformation(A, C, 2);
  check(near(r2.mi, 0) && near(r2.hab, 2) && [2, 2, 2, 2].every((v, i) => r2.counts[i] === v), "independent: counts 2,2,2,2, H(A,B) = 2, MI = 0");
  // invariance to a one-to-one remap (exact histograms): permute B's grey levels
  const X = Array.from({ length: 64 }, (_, i) => ((i * 7) % 4) / 4 + 0.01), Y = X.map(x => ((Math.floor(x * 4) * 3) % 4) / 4 + 0.01);
  const m1 = mutualInformation(X, Y, 4).mi, Yp = Y.map(y => [3, 0, 2, 1][Math.floor(y * 4)] / 4 + 0.01);
  check(near(m1, mutualInformation(X, Yp, 4).mi, 1e-12), "MI invariant to a permutation of grey levels");
}
// registration with SSD reduces the displacement error on a seeded field
{
  const Fimg = cropGrey(cell, 64, 64, 128, 128), F01: Grey = { width: 128, height: 128, data: Fimg.data.map(v => v / 255) };
  const tg = randomGrid(128, 128, 32, 3, 7), T = ffdDisplacement(tg, 128, 128);
  const M = deformForTruth(F01, T.u, T.v);
  const Fs = gaussianBlur(F01, 1), back = warpMoving(deformForTruth(Fs, T.u, T.v), T.u, T.v);
  let be = 0; for (let y = 10; y < 118; y++) for (let x = 10; x < 118; x++) be += Math.abs(back.data[y * 128 + x] - Fs.data[y * 128 + x]) / (108 * 108);
  check(be < 0.01, `deformForTruth: warping back by the truth recovers F (mean err ${be.toFixed(4)}, bilinear twice)`);
  const zeroErr = displacementError(new Float32Array(128 * 128), new Float32Array(128 * 128), T.u, T.v, 128, 128);
  for (const m of ["ssd", "mi"] as const) {
    const reg = new FFDRegistration(F01, m === "mi" ? { ...M, data: M.data.map(v => remap(v, "inverted")) } : M, { measure: m, lambda: 0, spacing: 16, coarse: 2 });
    for (let k = 0; k < 150; k++) reg.iterate();
    const cur = reg.current!;
    const err = displacementError(cur.u, cur.v, T.u, T.v, 128, 128);
    check(err < 0.5 * zeroErr, `registration (${m}${m === "mi" ? ", inverted" : ""}) reduces the displacement error ${zeroErr.toFixed(2)} → ${err.toFixed(2)}`);
    console.log(`  FFD ${m}: error ${zeroErr.toFixed(2)} → ${err.toFixed(2)} px in ${reg.iterations} steps`);
  }
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
