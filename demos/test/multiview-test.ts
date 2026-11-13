// Tests for the geometry-from-images day: lib/camera.ts, lib/twoview.ts, lib/stereo.ts,
// lib/flatland.ts, lib/bundle.ts. Reference checks against brute force or known answers.
// Run: node_modules/.bin/esbuild demos/test/multiview-test.ts --bundle --platform=node --log-level=warning | node
import * as C from "../src/lib/camera";
import * as T from "../src/lib/twoview";
import * as S from "../src/lib/stereo";
import * as FL from "../src/lib/flatland";
import * as BA from "../src/lib/bundle";
import { svd, cond, rank, jacobiEigen, solve } from "../src/lib/linalg";
declare const process: { exit(code: number): never };
import { viterbi } from "../src/lib/dp";
import { rng } from "../src/lib/random";

let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- camera ----
const K = C.makeK(800, 320, 240);
const cam = C.lookAtCamera(K, [0.6, 0.5, 1.2], [0.1, 0.1, 0.1], 0.1);
{
  // project then back-project: the ray passes through the point
  const X = [0.15, 0.2, 0.05], p = C.project(cam, X), ray = C.pixelRay(cam, p.u, p.v);
  const d = C.sub3(X, ray.origin), along = C.dot3(d, ray.dir), perp = C.len3(C.sub3(d, C.scale3(ray.dir, along)));
  check("pixel ray passes through the point", perp < 1e-9, String(perp));
  // with distortion
  const camD = { ...cam, k1: -0.2, k2: 0.05 }, pd = C.project(camD, X), rd = C.pixelRay(camD, pd.u, pd.v);
  const dd = C.sub3(X, rd.origin), al = C.dot3(dd, rd.dir);
  check("pixel ray with distortion", C.len3(C.sub3(dd, C.scale3(rd.dir, al))) < 1e-7);
  // camera centre is the null vector of P
  const P = C.cameraMatrix(cam), c = C.cameraCentre(cam);
  const Pc = [0, 1, 2].map(i => P[i * 4] * c[0] + P[i * 4 + 1] * c[1] + P[i * 4 + 2] * c[2] + P[i * 4 + 3]);
  check("P c = 0", Math.hypot(...Pc) < 1e-9);
  // by-hand numbers of the exercise
  const cam0: C.Camera = { K, R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: [0, 0, 0], k1: 0, k2: 0 };
  const a = C.project(cam0, [0.5, -0.25, 5]), b = C.project(cam0, [0.5, -0.25, 10]);
  check("by hand (400,200),(360,220)", close(a.u, 400, 1e-9) && close(a.v, 200, 1e-9) && close(b.u, 360, 1e-9) && close(b.v, 220, 1e-9));
  // Rodrigues round trip
  const w = [0.3, -0.5, 0.2], R = C.axisAngleToR(w), w2 = C.RToAxisAngle(R);
  check("Rodrigues round trip", Math.hypot(w[0] - w2[0], w[1] - w2[1], w[2] - w2[2]) < 1e-12);
}
{
  // DLT recovers P from 6 noiseless non-coplanar points; fails (rank) for coplanar ones
  const target = C.twoPlaneTarget(6, 0.04);
  const pick = [0, 10, 26, 36, 59, 67];
  const X6 = pick.slice(0, 6).map(i => target[i]);
  const x6 = X6.map(X => { const p = C.project(cam, X); return [p.u, p.v]; });
  const dlt = C.dltCamera(X6, x6, true), Ptrue = C.cameraMatrix(cam);
  C.fixScaleP(Ptrue);
  let err = 0; for (let k = 0; k < 12; k++) err = Math.max(err, Math.abs(dlt.P[k] - Ptrue[k]));
  check("DLT exact from 6 points", err < 1e-6, String(err));
  const dec = C.decomposeP(dlt.P);
  check("RQ gives K back", Math.abs(dec.K[0] - 800) < 1e-4 && Math.abs(dec.K[2] - 320) < 1e-4 && Math.abs(dec.K[5] - 240) < 1e-4 && Math.abs(dec.K[1]) < 1e-4, Array.from(dec.K).join(","));
  let rerr = 0; for (let k = 0; k < 9; k++) rerr = Math.max(rerr, Math.abs(dec.R[k] - cam.R[k]));
  check("RQ gives R back", rerr < 1e-6);
  const Xc = target.slice(36, 48), xc = Xc.map(X => { const p = C.project(cam, X); return [p.u, p.v]; });
  const dc = C.dltCamera(Xc, xc, true);
  check("coplanar points: more than one null vector", dc.S[10] / dc.S[0] < 1e-8, String(dc.S[10] / dc.S[0]));
  // normalised DLT is much better conditioned
  const all = target.map(X => { const p = C.project(cam, X); return [p.u, p.v]; });
  const cu = C.dltCamera(target, all, false).cond, cn = C.dltCamera(target, all, true).cond;
  check("normalisation improves conditioning", cn * 100 < cu, `${cn} vs ${cu}`);
  console.log(`DLT condition (s1/s11): unnormalised ${cu.toExponential(2)}, normalised ${cn.toExponential(2)}`);
}
{
  // zhangK recovers K from 3 synthetic views of a plane
  const plane = C.planarTarget(8, 6, 0.03);
  const Ktrue = C.makeK(820, 330, 250);
  const poses = [[0.3, 0.1, 0.05], [-0.25, 0.3, -0.1], [0.1, -0.35, 0.2]];
  const Hs = poses.map((w, k) => {
    const R = C.axisAngleToR(w), t = [0.02 * k, -0.01, 0.6 + 0.05 * k];
    const c: C.Camera = { K: Ktrue, R, t, k1: 0, k2: 0 };
    const img = plane.map(X => { const p = C.project(c, X); return [p.u, p.v]; });
    return C.homographyDLT(plane.map(p => [p[0], p[1]]), img, true).H;
  });
  const z = C.zhangK(Hs);
  check("zhangK recovers K (3 views)", !!z.K && Math.abs(z.K[0] - 820) < 1e-3 && Math.abs(z.K[4] - 820) < 1e-3 && Math.abs(z.K[2] - 330) < 1e-3, z.K ? Array.from(z.K).join(",") : "null");
  const z1 = C.zhangK(Hs.slice(0, 1));
  check("zhangK with 1 view is not enough", z1.K === null);
}

// ---- two views ----
const camA = C.lookAtCamera(K, [-0.3, 0.4, 2], [0, 0.3, 0]);
const camB = C.lookAtCamera(K, [0.5, 0.5, 1.8], [0, 0.3, 0]);
const r0 = rng(5);
const pts3: C.V3[] = Array.from({ length: 40 }, () => [r0.unif() - 0.5, r0.unif() * 0.8, r0.unif() - 0.5]);
const xA = pts3.map(X => { const p = C.project(camA, X); return [p.u, p.v] as C.V2; });
const xB = pts3.map(X => { const p = C.project(camB, X); return [p.u, p.v] as C.V2; });
{
  const Ft = T.fundamentalFromCameras(camA, camB);
  let worst = 0;
  for (let i = 0; i < pts3.length; i++) worst = Math.max(worst, T.pointLineDistance(T.epipolarLine(Ft, xA[i]), xB[i]));
  check("true F: matches lie on epipolar lines", worst < 1e-8, String(worst));
  const ep = T.eightPoint(xA, xB, true)!;
  let w2 = 0;
  for (let i = 0; i < pts3.length; i++) w2 = Math.max(w2, T.pointLineDistance(T.epipolarLine(ep.F, xA[i]), xB[i]));
  check("eight-point: x2ᵀ F x1 ≈ 0 on noiseless data", w2 < 1e-6, String(w2));
  check("eight-point: rank 2", svd(ep.F, 3, 3).S[2] < 1e-10 * svd(ep.F, 3, 3).S[0]);
  // epipole: image of the other camera centre
  const { e1, e2 } = T.epipoles(ep.F), cB = C.project(camA, C.cameraCentre(camB)), cA = C.project(camB, C.cameraCentre(camA));
  check("epipole = image of the other centre", Math.hypot(e1[0] / e1[2] - cB.u, e1[1] / e1[2] - cB.v) < 1e-5);
  check("epipole 2", Math.hypot(e2[0] / e2[2] - cA.u, e2[1] / e2[2] - cA.v) < 1e-5);
  // decomposeE + cheirality returns the generating R, t̂
  const { R, t } = T.relativePose(camA, camB), E = T.essentialFromRt(R, t);
  const nA = xA.map(x => C.invK(K, x[0], x[1])), nB = xB.map(x => C.invK(K, x[0], x[1]));
  const sel = T.selectPose(T.decomposeE(E), nA, nB), th = C.unit3(t);
  let er = 0; for (let k = 0; k < 9; k++) er = Math.max(er, Math.abs(sel.R[k] - R[k]));
  check("decomposeE: R", er < 1e-8, String(er));
  check("decomposeE: t direction", C.len3(C.sub3(sel.t, th)) < 1e-8 && sel.inFront === pts3.length);
  // triangulation exact without noise
  const X = T.triangulateCams(camA, camB, xA[3], xB[3]);
  check("triangulation exact", C.len3(C.sub3(X, pts3[3])) < 1e-9);
  const rA = C.pixelRay(camA, xA[3][0], xA[3][1]), rB = C.pixelRay(camB, xB[3][0], xB[3][1]);
  check("midpoint exact", C.len3(C.sub3(T.triangulateMidpoint(rA.origin, rA.dir, rB.origin, rB.dir).X, pts3[3])) < 1e-9);
  // rectified pairs have equal y for matches
  const { H1, H2 } = T.rectifyCalibrated(camA, camB);
  let dy = 0;
  for (let i = 0; i < pts3.length; i++) dy = Math.max(dy, Math.abs(C.applyH(H1, xA[i][0], xA[i][1])[1] - C.applyH(H2, xB[i][0], xB[i][1])[1]));
  check("rectification: equal rows", dy < 1e-8, String(dy));
  // RANSAC with 30 % outliers finds the inliers
  const r1 = rng(9), xBo = xB.map((x, i) => (i % 10 < 3 ? [r1.unif() * 640, r1.unif() * 480] as C.V2 : [x[0] + 0.3 * r1.norm(), x[1] + 0.3 * r1.norm()] as C.V2));
  const rf = T.ransacF(xA, xBo, 1.5, 300, 1);
  check("RANSAC F keeps the inliers", rf.inliers.length >= 27 && rf.inliers.length <= 29, String(rf.inliers.length));
  // Sampson ≈ squared symmetric distance order of magnitude
  check("Sampson zero for exact match", T.sampsonDistance(Ft, xA[0], xB[0]) < 1e-16);
}

// ---- stereo ----
{
  const sc = S.makeScene("dots", 96, 72, 0, 3);
  const C1 = S.costVolume(sc.left, sc.right, sc.w, sc.h, 24, "sad", 2);
  const d = S.wta(C1, sc.w * sc.h, 24);
  const bp = S.badPixelRate(d, sc, 0.5, 24);
  check("SAD WTA finds the shift on random dots", bp.rate < 0.1, String(bp.rate));
  // census invariant under 2I + 10
  const L2 = sc.left.map(v => 2 * v + 10), R2 = sc.right.map(v => 2 * v + 10);
  const c1 = S.census(sc.left, sc.w, sc.h), c2 = S.census(L2, sc.w, sc.h);
  check("census invariant under 2I + 10", c1.every((v, i) => v === c2[i]));
  const Cc1 = S.costVolume(sc.left, sc.right, sc.w, sc.h, 24, "census", 1), Cc2 = S.costVolume(L2, R2, sc.w, sc.h, 24, "census", 1);
  check("census cost volume invariant", Cc1.every((v, i) => v === Cc2[i]));
  // NCC invariant under gain/offset
  const N1 = S.costVolume(sc.left, sc.right, sc.w, sc.h, 8, "ncc", 2), N2 = S.costVolume(sc.left, sc.right.map(v => 0.5 * v + 30), sc.w, sc.h, 8, "ncc", 2);
  let nd = 0; for (let i = 0; i < N1.length; i++) nd = Math.max(nd, Math.abs(N1[i] - N2[i]));
  check("NCC invariant under gain and offset", nd < 1e-4, String(nd));
  // scanline DP = Viterbi from lib/dp.ts (energy) on one row
  const D = 24, w = sc.w, P1 = 2, P2 = 10;
  const sd = S.scanlineDP(C1, w, 1, D, P1, P2);
  const vt = viterbi(C1.subarray(0, w * D), D, (a, b) => S.jumpCost(a, b, P1, P2));
  check("scanline DP energy = Viterbi energy", close(sd.energy, vt.energy, 1e-3), `${sd.energy} vs ${vt.energy}`);
  // SGM with one direction: at the last pixel of each row WTA equals the DP's end label
  const sg = S.sgm(C1, w, sc.h, D, P1, P2, 1), dS = S.wta(sg, w * sc.h, D), dp = S.scanlineDP(C1, w, sc.h, D, P1, P2).disp;
  let agree = 0; for (let y = 0; y < sc.h; y++) if (dS[y * w + w - 1] === dp[y * w + w - 1]) agree++;
  check("SGM (1 direction) ends where scanline DP ends", agree === sc.h, `${agree}/${sc.h}`);
  // SGM with P1 = P2 = 0 is WTA on the summed costs
  const s0 = S.sgm(C1, w, sc.h, D, 0, 0, 4), d0 = S.wta(s0, w * sc.h, D);
  const Csum = Float32Array.from(C1, v => 4 * v);   // each direction adds C(p, d) + const
  check("SGM with P1 = P2 = 0 → WTA", d0.every((v, i) => v === S.wta(Csum, w * sc.h, D)[i]));
  // left–right check marks the occluded band
  const CR = S.costVolumeRight(sc.left, sc.right, sc.w, sc.h, 24, "sad", 2), dR = S.wta(CR, sc.w * sc.h, 24);
  const lr = S.leftRightCheck(d, dR, sc.w, sc.h, 1);
  let occInvalid = 0, occ = 0;
  for (let i = 0; i < lr.length; i++) if (sc.occluded[i] && i % sc.w > 24) { occ++; if (lr[i] < 0) occInvalid++; }
  check("left–right check rejects most occluded pixels", occInvalid / occ > 0.6, `${occInvalid}/${occ}`);
  check("parabola vertex", close(S.parabolaVertex(4, 1, 2), 0.25, 1e-12));
}

// ---- flatland and bundle adjustment ----
{
  const c1 = FL.lookAt1d([-1, -3], [0, 0], 200, 160), c2 = FL.lookAt1d([1, -3], [0, 0], 200, 160);
  const p: FL.P = [0.2, 0.3];
  const u1 = FL.project1d(c1, p).u, u2 = FL.project1d(c2, p).u;
  const q = FL.triangulateRays([c1, c2], [u1, u2]), qo = FL.triangulateOptimal([c1, c2], [u1 + 0.5, u2 - 0.5]);
  check("flatland triangulation exact", Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9);
  check("optimal triangulation reprojects better than midpoint", (() => {
    const m = FL.triangulateRays([c1, c2], [u1 + 0.5, u2 - 0.5]);
    const e = (x: FL.P) => (FL.project1d(c1, x).u - u1 - 0.5) ** 2 + (FL.project1d(c2, x).u - u2 + 0.5) ** 2;
    return e(qo) <= e(m) + 1e-12;
  })());
  // finite-difference check of the BA Jacobian
  const r = rng(11);
  const cams = FL.ring(5, 3, 200, 160);
  const pts: FL.P[] = Array.from({ length: 12 }, () => [r.unif() - 0.5, r.unif() - 0.5]);
  const obs: BA.Obs[] = [];
  cams.forEach((c, j) => pts.forEach((pp, i) => { const pr = FL.project1d(c, pp); if (pr.Z > 0 && pr.u > 0 && pr.u < 159) obs.push({ cam: j, pt: i, u: pr.u + 0.3 * r.norm() }); }));
  const prob = new BA.FlatBA(cams, pts.length, obs);
  const th = BA.FlatBA.pack(cams, pts);
  const { J } = prob.evaluate(th, true);
  let jerr = 0;
  for (let k = 0; k < prob.nParams; k++) {
    const hp = Float64Array.from(th), hm = Float64Array.from(th), h = 1e-6;
    hp[k] += h; hm[k] -= h;
    const rp = prob.evaluate(hp, false).r, rm = prob.evaluate(hm, false).r;
    for (let i = 0; i < obs.length; i++) jerr = Math.max(jerr, Math.abs((rp[i] - rm[i]) / (2 * h) - J![i * prob.nParams + k]));
  }
  check("BA Jacobian = finite differences", jerr < 1e-4, String(jerr));
  // 4 zero eigenvalues of JᵀJ without gauge fixing (2D similarity), none with it
  const ev = BA.normalEigenvalues(prob, th);
  const big = ev[ev.length - 1];
  const zeros = Array.from(ev).filter(v => v < 1e-9 * big).length;
  check("4 gauge directions in 2D", zeros === 4, Array.from(ev.slice(0, 6)).map(v => v.toExponential(1)).join(" "));
  const fixed = BA.gaugeMask(cams.length, pts.length);
  const evf = BA.normalEigenvalues(prob, th, fixed);
  check("gauge fixed: no zero eigenvalue", evf[0] > 1e-9 * evf[evf.length - 1]);
  // LM converges from a perturbation
  const th0 = Float64Array.from(th);
  for (let k = 0; k < th0.length; k++) if (!fixed[k]) th0[k] += 0.05 * r.norm();
  const res = BA.levenbergMarquardt(prob, th0, { fixed, maxIter: 100 });
  const final = res.costs[res.costs.length - 1];
  check("LM converges", final < 0.09 * 2 * obs.length, `${res.costs[0]} → ${final}`);
  // Schur complement = dense solve
  const { r: rr, J: JJ } = prob.evaluate(th0, true);
  const free = Array.from({ length: prob.nParams }, (_, i) => i);
  const ne = BA.normalEquations(JJ!, rr, new Float64Array(rr.length).fill(1), rr.length, prob.nParams, free);
  for (let a = 0; a < prob.nParams; a++) ne.A[a * prob.nParams + a] += 1e-3;
  const sch = BA.schurSolve(ne.A, ne.g, prob.nParams, 3 * cams.length, 2);
  const dense = solve(ne.A, ne.g, prob.nParams)!;
  let se = 0; for (let k = 0; k < prob.nParams; k++) se = Math.max(se, Math.abs(sch!.delta[k] - dense[k]));
  check("Schur complement = dense solve", se < 1e-6, String(se));
  // Gauss-Newton by hand (exercise): r(Z) = 0.25 − 1/Z from Z0 = 3
  let Z = 3; const hist = [Z];
  for (let i = 0; i < 2; i++) { const rz = 0.25 - 1 / Z, jz = 1 / (Z * Z); Z = Z - rz / jz; hist.push(Z); }
  check("GN by hand: 3 → 3.75 → 3.984", close(hist[1], 3.75, 1e-12) && close(hist[2], 3.984375, 1e-9));
  // sweep finds the depth of a textured object
  const scene = FL.makeScene([[[-0.6, -0.4], [0.6, -0.4], [0.6, 0.4], [-0.6, 0.4]]]);
  const sc = FL.ring(4, 3, 200, 160, [0, 0], -Math.PI / 2 - 0.5, 1.0);
  const imgs = sc.map(c => FL.render1d(scene, c).values);
  const ref = 1, rd = FL.render1d(scene, sc[ref]);
  const depths = Array.from({ length: 120 }, (_, k) => 1.5 + k * 0.025);
  let good = 0, tot = 0;
  for (let u = 40; u < 120; u += 8) { const s = FL.sweepPixel(sc, imgs, ref, u, depths, 2); tot++; if (Math.abs(depths[s.best] - rd.depth[u]) < 0.06) good++; }
  check("plane sweep finds the depth", good >= tot - 1, `${good}/${tot}`);
}
void cond; void rank; void jacobiEigen;

console.log(`multiview: ${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
