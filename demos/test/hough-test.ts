// Reference checks for lib/hough.ts and lib/ransac.ts (the Hough day). Run:
//   node_modules/.bin/esbuild demos/test/hough-test.ts --bundle --platform=node --log-level=warning | node
import * as H from "../src/lib/hough";
import * as R from "../src/lib/ransac";
import { canny, edgeList } from "../src/lib/edges";
import { addGaussianNoise } from "../src/lib/synth";
import { rng } from "../src/lib/random";
import { project, geometry, angleSet } from "../src/lib/radon";
import { makeScene, groundTruthLines, N as LN } from "../src/houghlines/scenes";
import { makeCircleScene } from "../src/houghcircle/scenes";
import { TEMPLATES, makeGhtScene } from "../src/ght/scenes";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const angDiff = (a: number, b: number) => { const d = Math.abs(a - b) % Math.PI; return Math.min(d, Math.PI - d); };
/** Distance between two normal-form lines: angle difference in degrees or rho difference in pixels, whichever is larger (handles theta near 0 / 180). */
function lineDist(t1: number, r1: number, t2: number, r2: number): number {
  const d = Math.abs(t1 - t2), flip = d > Math.PI / 2;
  const dTheta = (flip ? Math.PI - d : d) * 180 / Math.PI, dRho = flip ? Math.abs(r1 + r2) : Math.abs(r1 - r2);
  return Math.max(dTheta, dRho);
}

// ---- lines: brute force accumulator, peaks, the worked example ----
function testLineAccumulator(): void {
  current = "line accumulator";
  const g = H.lineGeometry(21, 15, 36, 1.5), r = rng(5);
  const pts = Array.from({ length: 30 }, () => ({ x: Math.floor(21 * r.unif()), y: Math.floor(15 * r.unif()), theta: 2 * Math.PI * r.unif() }));
  const A = H.houghLines(pts, g);
  // brute force: bin of every (point, angle)
  const B = new Float32Array(g.nTheta * g.nRho);
  for (const p of pts) for (let j = 0; j < g.nTheta; j++) {
    const th = j * Math.PI / g.nTheta, rho = (p.x - 10) * Math.cos(th) + (7 - p.y) * Math.sin(th);
    const k = Math.round(rho / 1.5 + (g.nRho - 1) / 2);
    if (k >= 0 && k < g.nRho) B[j * g.nRho + k]++;
  }
  let same = true;
  for (let i = 0; i < B.length; i++) if (A.acc[i] !== B[i]) same = false;
  check(same, "houghLines equals the brute-force accumulator");
  check(A.votes === pts.length * g.nTheta, `votes ${A.votes} = points x angles`);
  // gradient window: total votes = points x (2 half + 1)
  const Gw = H.houghLines(pts, g, { gradWindow: 2 * Math.PI / 36 });
  check(Gw.votes === pts.length * 5, `gradient voting casts 5 votes per point (${Gw.votes})`);
  // every gradient vote is also a full vote
  let sub = true;
  for (let i = 0; i < Gw.acc.length; i++) if (Gw.acc[i] > A.acc[i]) sub = false;
  check(sub, "gradient-window votes are a subset of the full votes");
}

function testWorkedExample(): void {
  current = "worked example";
  // points (2,0), (1,1), (0,2): at 45 degrees all have rho = sqrt 2
  const P = [[2, 0], [1, 1], [0, 2]];
  for (const deg of [0, 45, 90, 135]) {
    const th = deg * Math.PI / 180, rh = P.map(([x, y]) => H.sinusoid(x, y, th));
    const spread = Math.max(...rh) - Math.min(...rh);
    if (deg === 45) check(spread < 1e-12 && close(rh[0], Math.SQRT2), "45 deg: all rho = sqrt 2");
    else check(spread > 1, `${deg} deg: rho differ`);
  }
  const L = H.lineThrough(2, 0, 0, 2);
  check(close(L.theta, Math.PI / 4) && close(L.rho, Math.SQRT2), "lineThrough (2,0),(0,2)");
  // the intersection of two sinusoids is the normal of the line through the points
  const r = rng(9);
  for (let t = 0; t < 50; t++) {
    const x1 = 10 * r.unif() - 5, y1 = 10 * r.unif() - 5, x2 = 10 * r.unif() - 5, y2 = 10 * r.unif() - 5;
    const l = H.lineThrough(x1, y1, x2, y2);
    check(close(H.sinusoid(x1, y1, l.theta), l.rho, 1e-9) && close(H.sinusoid(x2, y2, l.theta), l.rho, 1e-9), "both sinusoids pass through (theta, rho)");
    check(l.theta >= 0 && l.theta < Math.PI, "theta in [0, pi)");
    // amplitude-phase form
    const A = Math.hypot(x1, y1), ph = Math.atan2(y1, x1), th = Math.PI * r.unif();
    check(close(H.sinusoid(x1, y1, th), A * Math.cos(th - ph), 1e-9), "rho = A cos(theta - phi)");
  }
}

function testPeaks(): void {
  current = "peaks";
  // a synthetic accumulator with two isolated maxima and a plateau
  const g = H.lineGeometry(9, 9, 12, 1), A = { g, acc: new Float32Array(g.nTheta * g.nRho), votes: 0 };
  A.acc[3 * g.nRho + 5] = 10; A.acc[3 * g.nRho + 6] = 9; A.acc[8 * g.nRho + 2] = 7;
  A.acc[10 * g.nRho + 8] = 4; A.acc[10 * g.nRho + 9] = 4;
  const p = H.linePeaks(A, 3, 1);
  check(p.length === 3, `three peaks with NMS (got ${p.length})`);
  check(p[0].j === 3 && p[0].k === 5 && p[0].value === 10, "strongest first");
  check(p.some(q => q.j === 10 && q.k === 8) && !p.some(q => q.j === 10 && q.k === 9), "a plateau gives one peak");
  const p0 = H.linePeaks(A, 3, 0);
  check(p0.length === 5, `no NMS: every bin above threshold (${p0.length})`);
  // wrap: theta_0 neighbours theta_{n-1} with mirrored rho
  const B = { g, acc: new Float32Array(g.nTheta * g.nRho), votes: 0 };
  B.acc[0 * g.nRho + 3] = 5; B.acc[(g.nTheta - 1) * g.nRho + (g.nRho - 1 - 3)] = 6;
  const pw = H.linePeaks(B, 1, 1);
  check(pw.length === 1 && pw[0].value === 6, "NMS wraps across theta = 0 / 180 with -rho");
}

function testLineDetection(): void {
  current = "line detection";
  // recover the sides of the "shapes" scene within one bin
  const s = makeScene("shapes"), c = canny(s.img, 1.5, 4, 10);
  const pts = edgeList(c.hyst.edges, c.gx, c.gy, c.mag);
  const g = H.lineGeometry(LN, LN, 180, 1);
  const A = H.houghLines(pts, g);
  const peaks = H.linePeaks(A, 30, 4);
  const truth = groundTruthLines(s).filter(t => t.length > 40);
  for (const t of truth) {
    const hit = peaks.some(p => lineDist(H.thetaOf(g, p.j), H.rhoOf(g, p.k), t.theta, t.rho) <= 1.5);
    check(hit, `side theta=${(t.theta * 180 / Math.PI).toFixed(1)} rho=${t.rho.toFixed(1)} found`);
  }
  // gradient voting: far fewer votes, same peaks
  const Gv = H.houghLines(pts, g, { gradWindow: 3 * Math.PI / 180 });
  check(Gv.votes < A.votes / 20, `gradient voting: ${Gv.votes} vs ${A.votes} votes`);
  const pg = H.linePeaks(Gv, 20, 4);
  check(pg.length >= truth.length - 1, `gradient voting finds the sides (${pg.length} peaks, ${truth.length} sides)`);
  // deskew: dominant angle of the page = 90 deg + tilt (text lines have normals near vertical)
  const pg2 = makeScene("page"), c2 = canny(pg2.img, 1.5, 4, 10);
  const A2 = H.houghLines(edgeList(c2.hyst.edges, c2.gx, c2.gy, c2.mag), g);
  const dom = H.dominantAngle(A2);
  check(Math.abs(dom.theta - (Math.PI / 2 + pg2.tilt!)) <= 1.01 * Math.PI / 180, `deskew angle ${(dom.theta * 180 / Math.PI).toFixed(1)} vs ${(90 + pg2.tilt! * 180 / Math.PI).toFixed(1)}`);
}

function testSegments(): void {
  current = "segments";
  // a dashed horizontal line y = 0 (world) through the centre: pieces of 10 with gaps of 6
  const g = { cx: 50, cy: 50 }, pts: H.Pt[] = [];
  for (let x = 10; x < 90; x++) if ((x - 10) % 16 < 10) pts.push({ x, y: 50 });
  const segs = H.lineSegments(pts, g, Math.PI / 2, 0, 1, 3, 5);
  check(segs.length === 5, `5 dashes (got ${segs.length})`);
  const joined = H.lineSegments(pts, g, Math.PI / 2, 0, 1, 8, 5);
  check(joined.length === 1 && close(joined[0].length, 73), `gap 8 joins them into one of length 73 (first point 10, last 83) (${joined[0]?.length})`);
  const clip = H.clipLine({ w: 101, h: 101, cx: 50, cy: 50 }, 0, 10);
  check(!!clip && close(clip[0], 60) && close(clip[2], 60), "clipLine: vertical line x = 10");
}

// ---- Radon of a point is a sinusoid ----
function testRadonPoint(): void {
  current = "radon point";
  const n = 33, img = new Float64Array(n * n), col = 22, row = 9;
  img[row * n + col] = 1;
  const angles = angleSet(24), gm = geometry(n, angles), sino = project(img, gm);
  const x = col - 16, y = 16 - row;
  for (let a = 0; a < 24; a++) {
    let best = 0;
    for (let b = 1; b < n; b++) if (sino[a * n + b] > sino[a * n + best]) best = b;
    const rhoPeak = best - (n - 1) / 2, rho = H.sinusoid(x, y, angles[a]);
    check(Math.abs(rhoPeak - rho) <= 0.51, `angle ${a}: peak at ${rhoPeak}, sinusoid ${rho.toFixed(2)}`);
  }
}

// ---- circles ----
function testCircles(): void {
  current = "circles";
  // the full circle accumulator is the correlation of the edge map with a ring
  const w = 24, h = 20, r = rng(2), pts: H.OrientedPt[] = [];
  const map = new Uint8Array(w * h);
  for (let k = 0; k < 25; k++) { const x = Math.floor(w * r.unif()), y = Math.floor(h * r.unif()); if (!map[y * w + x]) { map[y * w + x] = 1; pts.push({ x, y }); } }
  const C = H.houghCircles(pts, w, h, { rMin: 3, rMax: 5 });
  let same = true;
  for (let ri = 0; ri < 3; ri++) {
    const ring = H.ringOffsets(3 + ri);
    for (let b = 0; b < h; b++) for (let a = 0; a < w; a++) {
      let s = 0;
      for (let q = 0; q < ring.length; q += 2) { const x = a - ring[q], y = b - ring[q + 1]; if (x >= 0 && y >= 0 && x < w && y < h) s += map[y * w + x]; }
      if (s !== C.acc[ri * w * h + b * w + a]) same = false;
    }
  }
  check(same, "known-radius accumulator = edge map correlated with a ring");
  const ring10 = H.ringOffsets(10).length / 2;
  check(ring10 > 2 * Math.PI * 10 * 0.85 && ring10 < 2 * Math.PI * 10 * 1.2, `ring of radius 10 has ${ring10} pixels (about 2 pi r)`);
  // the worked example: radius 5, point (3, 4), unit gradient (0.6, 0.8) in pixel coordinates
  const theta = Math.atan2(-0.8, 0.6), v = H.circleVotesOf({ x: 3, y: 4, theta }, 5, true, "either");
  check(close(v[0][0], 6) && close(v[0][1], 8) && close(v[1][0], 0) && close(v[1][1], 0), `two candidate centres (6,8) and (0,0): ${JSON.stringify(v)}`);
  // detection on the synthetic discs scene
  const sc = makeCircleScene("discs"), cn = canny(sc.img, 1.5, 4, 10);
  const ep = edgeList(cn.hyst.edges, cn.gx, cn.gy, cn.mag);
  for (const grad of [false, true]) {
    const A = H.houghCircles(ep, sc.img.width, sc.img.height, { rMin: 6, rMax: 24, grad, polarity: "bright" });
    const nrm = H.normaliseByCircumference(A);
    const pk = H.peaks3(nrm, A.w, A.h, A.nR, grad ? 0.35 : 0.5, 4, 2);
    for (const d of sc.truth.filter(t => t.clean)) {
      const hit = pk.some(p => Math.hypot(p.x - d.x, p.y - d.y) <= 1.5 && Math.abs(p.s + A.rMin - d.r) <= 1.01);
      check(hit, `${grad ? "gradient" : "full"} voting finds disc (${d.x},${d.y}) r=${d.r}`);
    }
  }
}

// ---- GHT ----
function testGHT(): void {
  current = "GHT";
  const tpl = TEMPLATES[0];
  // template drawn about (0, 0), reference point there; shifted by 100 for the self-vote
  const pts = H.polygonEdgePoints(tpl.poly, 1), ref = { x: 0, y: 0 };
  const T = H.buildRTable(pts, ref, 36);
  check(T.size === pts.length, "every template point is in the table");
  // the template itself votes all its entries into the reference point
  const A = H.ghtVote(pts.map(p => ({ ...p, x: p.x + 100, y: p.y + 100 })), T, 200, 200);
  const at = A.acc[100 * 200 + 100];
  check(at >= pts.length * 0.9, `self-vote at the reference: ${at} of ${pts.length}`);
  // a translated and rotated copy in a scene, through Canny
  for (const rotDeg of [0, 40]) {
    const sc = makeGhtScene(tpl, { copies: [{ x: 120, y: 90, phi: rotDeg * Math.PI / 180, s: 1 }], clutter: 0, noise: 4, occlusion: 0 });
    const cn = canny(sc.img, 1.2, 4, 10), ep = edgeList(cn.hyst.edges, cn.gx, cn.gy, cn.mag);
    const phis = Array.from({ length: 36 }, (_, i) => i * 10 * Math.PI / 180);
    const G = H.ghtVote(ep, T, sc.img.width, sc.img.height, { phis });
    const pk = H.peaks3(G.acc, G.w, G.h, phis.length, 1, 4, 1, 1)[0];
    check(Math.hypot(pk.x - 120, pk.y - 90) <= 2 && Math.abs(pk.s * 10 - rotDeg) <= 10, `copy at 120,90 rotated ${rotDeg}: found at ${pk.x},${pk.y}, ${pk.s * 10} deg`);
  }
  // rotation and scale of a vector
  const [x, y] = H.rotScale([1, 0], Math.PI / 2, 2);
  check(close(x, 0, 1e-12) && close(y, -2, 1e-12), "rotating (1,0) by 90 deg CCW on screen gives (0,-1) (y down)");
  // a rotated polygon's directions are shifted by phi
  const sq: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]];
  const d0 = H.polygonEdgePoints(sq, 5)[0].theta!, d1 = H.polygonEdgePoints(H.placePolygon(sq, { x: 5, y: 5 }, 0.3, 1, { x: 5, y: 5 }), 5)[0].theta!;
  check(close(((d1 - d0) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI), 0.3, 1e-9), "rotating a polygon by phi turns its gradient directions by phi");
  check(close(d0, -Math.PI / 2), "top side of a bright square: gradient points down (-90 deg)");
}

// ---- RANSAC ----
function testRansac(): void {
  current = "RANSAC";
  check(R.iterationsNeeded(0.5, 2) === 17 && R.iterationsNeeded(0.5, 3) === 35 && R.iterationsNeeded(0.2, 2) === 113, "N = 17, 35, 113");
  // the formula against simulation
  const r = rng(4), w = 0.4, s = 2, Nit = 10, trials = 20000;
  let succ = 0;
  for (let t = 0; t < trials; t++) {
    let ok = false;
    for (let i = 0; i < Nit && !ok; i++) { let all = true; for (let k = 0; k < s; k++) if (r.unif() >= w) all = false; ok = all; }
    if (ok) succ++;
  }
  const pSim = succ / trials, pTh = R.successProbability(w, s, Nit);
  check(Math.abs(pSim - pTh) < 0.015, `simulated ${pSim.toFixed(3)} vs formula ${pTh.toFixed(3)}`);
  // exact fits
  const L = R.lineFrom2({ x: 2, y: 0 }, { x: 0, y: 2 })!;
  check(close(L.theta, Math.PI / 4) && close(L.rho, Math.SQRT2), "line through (2,0),(0,2)");
  const Ci = R.circleFrom3({ x: 5, y: 0 }, { x: 0, y: 5 }, { x: -5, y: 0 })!;
  check(close(Ci.a, 0, 1e-12) && close(Ci.b, 0, 1e-12) && close(Ci.r, 5), "circle through three points");
  // TLS = smallest eigenvector; check against brute-force search over angles
  const rr = rng(8), pts: R.P2[] = Array.from({ length: 40 }, () => { const t = 10 * rr.unif(); return { x: t + 0.3 * rr.norm(), y: 0.5 * t + 1 + 0.3 * rr.norm() }; });
  const tls = R.lineTLS(pts)!;
  const cost = (th: number) => { const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length, my = pts.reduce((a, p) => a + p.y, 0) / pts.length; return pts.reduce((a, p) => a + ((p.x - mx) * Math.cos(th) + (p.y - my) * Math.sin(th)) ** 2, 0); };
  let bestTh = 0;
  for (let k = 0; k < 18000; k++) { const th = k * Math.PI / 18000; if (cost(th) < cost(bestTh)) bestTh = th; }
  check(angDiff(tls.theta, bestTh) < 2e-4, `TLS normal ${tls.theta} vs brute force ${bestTh}`);
  // algebraic circle fit on exact points
  const cp = Array.from({ length: 12 }, (_, i) => ({ x: 3 + 4 * Math.cos(i), y: -2 + 4 * Math.sin(i) }));
  const cf = R.circleAlgebraic(cp)!;
  check(close(cf.a, 3, 1e-9) && close(cf.b, -2, 1e-9) && close(cf.r, 4, 1e-9), "algebraic circle fit on exact points");
  // RANSAC with a fixed seed finds the inlier set
  const ps = R.makePointSet(60, 0.5, [{ kind: "line", theta: 1.1, rho: 40 }], 0.3, 100, 6);
  const res = R.ransac(R.LINE_MODEL, ps.pts, { eps: 1.5, iterations: 40, seed: 1 });
  const truthIn = ps.inlier.map((v, i) => (v ? i : -1)).filter(i => i >= 0);
  const found = new Set(res.refitInliers);
  const missed = truthIn.filter(i => !found.has(i)).length, extra = res.refitInliers.filter(i => !ps.inlier[i]).length;
  check(missed <= 1 && extra <= 2, `RANSAC inliers: missed ${missed}, extra ${extra}`);
  check(angDiff(res.refit!.theta, 1.1) < 0.02 && Math.abs(res.refit!.rho - 40) < 0.5, "refit close to the truth");
  const res2 = R.ransac(R.LINE_MODEL, ps.pts, { eps: 1.5, iterations: 40, seed: 1 });
  check(res2.foundAt === res.foundAt && res2.inliers.length === res.inliers.length, "same seed, same run");
  // least squares on all points is pulled away
  const ols = R.lineTLS(ps.pts)!;
  check(angDiff(ols.theta, 1.1) > 0.05 || Math.abs(ols.rho - 40) > 3, "TLS on all points is pulled away by outliers");
  // sequential RANSAC finds two lines
  const two = R.makePointSet(80, 0.25, [{ kind: "line", theta: 0.3, rho: 30 }, { kind: "line", theta: 2.0, rho: -20 }], 0.3, 100, 3);
  const seq = R.sequentialRansac(R.LINE_MODEL, two.pts, { eps: 1.5, iterations: 60, seed: 2, maxModels: 2, minInliers: 10 });
  check(seq.length === 2, "sequential RANSAC: two models");
  // randomised Hough finds the line
  const rh = R.randomisedHoughLines(ps.pts, 400, Math.PI / 90, 2, 1);
  check(!!rh.best && angDiff(rh.best.theta, 1.1) < 0.04 && Math.abs(rh.best.rho - 40) <= 2, `randomised Hough ${rh.best?.theta.toFixed(3)} ${rh.best?.rho}`);
}

testLineAccumulator();
testWorkedExample();
testPeaks();
testLineDetection();
testSegments();
testRadonPoint();
testCircles();
testGHT();
testRansac();
console.log(`hough-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
