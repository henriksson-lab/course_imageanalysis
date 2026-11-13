// Checks for the keypoints day: lib/keypoints.ts, lib/descriptors.ts, lib/featurematch.ts,
// lib/transforms2d.ts. Run:
//   node_modules/.bin/esbuild demos/test/keypoints-test.ts --bundle --platform=node --log-level=warning | node
import * as K from "../src/lib/keypoints";
import * as D from "../src/lib/descriptors";
import * as M from "../src/lib/featurematch";
import * as T from "../src/lib/transforms2d";
import { Grey, fromFunction, gaussianBlur } from "../src/lib/imageops";
import { scaleSpaceStack, extrema3D } from "../src/lib/edges";
import { homographyDLT } from "../src/lib/camera";
import { iterationsNeeded } from "../src/lib/ransac";
import { rng } from "../src/lib/random";
import { makeScene } from "../src/dogpyramid/scene";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const angDiff = (a: number, b: number) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));

// ---- the DoG pyramid ----
function testPyramid(): void {
  current = "pyramid";
  const img = makeScene(192, 0);
  const p = K.dogPyramid(img, { S: 3 });
  check(p.octaves.length >= 4, `octaves ${p.octaves.length}`);
  for (const oc of p.octaves) {
    check(oc.gauss.length === 6 && oc.dog.length === 5, "S + 3 blurred, S + 2 DoGs");
    oc.sigmas.forEach((s, i) => check(close(s, 1.6 * Math.pow(2, i / 3), 1e-12), `σ_${i} = σ0 k^i`));
  }
  check(p.octaves[1].dog[0].width === 96 && p.octaves[1].step === 2, "second octave halves");
  // the octave's blurred images equal one direct blur of the input (semigroup) away from the border
  const direct = gaussianBlur(img, Math.sqrt(1.6 * 1.6 * 4 - 0.25));
  const oc = p.octaves[0];
  let maxd = 0;
  for (let y = 30; y < 160; y++) for (let x = 30; x < 160; x++) maxd = Math.max(maxd, Math.abs(direct.data[y * 192 + x] - oc.gauss[3].data[y * 192 + x]));
  check(maxd < 0.5, `incremental blur matches a direct blur (max diff ${maxd.toFixed(3)})`);
  // DoG extrema of octave 0 agree with extrema3D on the same stack (as maxima of −D and of D)
  const cands = K.dogCandidates(p, 1).filter(c => c.octave === 0);
  const neg = oc.dog.map(g => ({ ...g, data: g.data.map(v => -v) }));
  const ex = [...extrema3D(neg, 1, 1, true), ...extrema3D(oc.dog, 1, 1, true)].filter(e => e.s >= 1 && e.s <= 3);
  const key = (x: number, y: number, s: number) => `${x},${y},${s}`;
  const A = new Set(cands.map(c => key(c.ix, c.iy, c.level))), B = new Set(ex.map(e => key(e.x, e.y, e.s)));
  let common = 0; for (const k of A) if (B.has(k)) common++;
  check(common === A.size && A.size === B.size, `candidates = extrema3D (${A.size} vs ${B.size}, ${common} common)`);
}

// ---- a Gaussian blob: scale and sub-pixel position ----
function testBlob(): void {
  current = "blob";
  for (const [s, cx, cy] of [[3, 60.3, 59.6], [5, 63.7, 64.2], [8, 64.4, 63.3]] as [number, number, number][]) {
    const img = fromFunction(128, 128, (x, y) => 40 + 160 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)));
    const r = K.detectDoG(img, { contrast: 1, inBlur: 0 });
    let best: K.Keypoint | null = null;
    for (const k of r.kps) if (!best || k.response > best.response) best = k;
    check(!!best, `blob of width ${s} found`);
    if (!best) continue;
    const lev = Math.abs(Math.log2(best.sigma / s) * 3);
    check(lev < 1, `σ ${best.sigma.toFixed(2)} within one level of ${s}`);
    check(Math.hypot(best.x - cx, best.y - cy) < 0.1, `position error ${Math.hypot(best.x - cx, best.y - cy).toFixed(3)} px < 0.1 after refinement (s = ${s})`);
  }
}

// ---- refinement, edge test, the worked numbers ----
function testRefineAndEdge(): void {
  current = "refine/edge";
  // 1D three-point rule: samples 0.6, 1.0, 0.8 → offset +1/6, peak value 1.0083
  const off = (0.6 - 0.8) / (2 * (0.6 - 2 * 1.0 + 0.8));
  check(close(off, 1 / 6, 1e-12), `offset ${off}`);
  const peak = 1.0 + 0.5 * ((0.8 - 0.6) / 2) * off;
  check(close(peak, 1.0083333, 1e-6), `peak ${peak}`);
  // edge test on the slide's Hessians
  const a = K.edgeRatioTest(-4, 0, -3), b = K.edgeRatioTest(-10, 0, -0.5), c = K.edgeRatioTest(3, 0, -2);
  check(a.pass && close(a.ratio, 49 / 12, 1e-12), `diag(−4, −3): ${a.ratio}`);
  check(!b.pass && close(b.ratio, 110.25 / 5, 1e-12), `diag(−10, −0.5): ${b.ratio}`);
  check(!c.pass, "saddle rejected");
  check(close(a.limit, 12.1, 1e-12), "limit 12.1 for r = 10");
  // the bar of the dogpyramid scene gives edge candidates, which the test removes; the corners of a square stay
  const img = makeScene(192, 2);
  const r = K.detectDoG(img, { S: 3, contrast: 3, edgeR: 10 });
  const onBar = r.records.filter(x => x.cand.octave === 0 && Math.abs(x.refined.y - 62) < 3 && x.refined.x > 30 && x.refined.x < 165);
  check(onBar.length >= 3, `bar candidates ${onBar.length}`);
  check(onBar.every(x => x.stage !== "kept"), "no bar candidate kept");
  const sq = fromFunction(96, 96, (x, y) => (x >= 28 && x <= 68 && y >= 28 && y <= 68 ? 200 : 60));
  const rs = K.detectDoG(gaussianBlur(sq, 0.7), { contrast: 2 });
  // a large square: the candidates in the middle of its sides fail the edge test; what is kept is the
  // square as one blob, at its centre and at a large scale (a 90° corner of a large square is not a DoG extremum)
  const edgeRej = rs.records.filter(x => x.stage === "edge");
  check(edgeRej.length >= 4 && edgeRej.every(x => Math.min(Math.abs(x.refined.x - 48), Math.abs(x.refined.y - 48)) < 1), `side candidates rejected by the edge test (${edgeRej.length})`);
  check(rs.kps.length === 1 && Math.hypot(rs.kps[0].x - 48, rs.kps[0].y - 48) < 0.5 && rs.kps[0].sigma > 10, `the square kept once, as a blob at its centre`);
}

// ---- repeatability ----
function testRepeatability(): void {
  current = "repeatability";
  const scene = T.featureScene({ size: 512, seed: 11 });
  const N = 256;
  const run = (spec: T.ViewPairSpec, det: "dog" | "harris") => {
    const { V1, V2, T12 } = T.viewPair(512, N, spec);
    const v1 = T.viewOf(scene, V1, { size: N }), v2 = T.viewOf(scene, V2, { size: N });
    const k = (img: Grey) => det === "dog" ? K.detectDoG(img, { contrast: 3, maxPoints: 200 }).kps : K.detectHarris(img, 1.5, { maxPoints: 200, rel: 0.002 });
    return K.repeatability(k(v1), k(v2), T.mapOf(T12), T.mapOf(T.invert(T12)), { w1: N, h1: N, w2: N, h2: N, eps: 2.5, scale: spec.zoom ?? 1, scaleTol: det === "dog" ? Math.log(1.5) : Infinity }).rep;
  };
  const d2 = run({ zoom: 2 }, "dog"), h2 = run({ zoom: 2 }, "harris");
  console.log(`  repeatability at zoom 2: DoG ${d2.toFixed(2)}, Harris (one σ) ${h2.toFixed(2)}`);
  check(d2 > 0.6, `DoG at zoom 2: ${d2}`);
  check(d2 > h2, `DoG ${d2} > Harris one σ ${h2} at zoom 2`);
  const r90 = run({ angle: 90 }, "dog");
  check(r90 > 0.9, `rotation by 90°: ${r90}`);
}

// ---- orientation ----
function testOrientation(): void {
  current = "orientation";
  const scene = T.featureScene({ size: 512, seed: 11 });
  for (const ang of [30, 120]) {
    const { V1, V2, T12 } = T.viewPair(512, 256, { angle: ang });
    const v1 = T.viewOf(scene, V1), v2 = T.viewOf(scene, V2);
    const r1 = K.detectDoG(v1, { contrast: 3, maxPoints: 80 }), r2 = K.detectDoG(v2, { contrast: 3, maxPoints: 150 });
    const o1 = K.assignOrientations(r1.pyr, r1.kps), o2 = K.assignOrientations(r2.pyr, r2.kps);
    const errs: number[] = [];
    for (const a of o1) {
      const [x, y] = T.apply(T12, a.x, a.y);
      const cand = o2.filter(b => Math.hypot(b.x - x, b.y - y) < 1.5);
      if (!cand.length) continue;
      errs.push(Math.min(...cand.map(b => angDiff(b.theta, a.theta + ang * Math.PI / 180))) * 180 / Math.PI);
    }
    errs.sort((a, b) => a - b);
    const med = errs[Math.floor(errs.length / 2)];
    check(errs.length > 20 && med < 3, `median orientation error after ${ang}°: ${med?.toFixed(2)}° (${errs.length} points)`);
  }
}

// ---- gradient histograms ----
function testGradHist(): void {
  current = "gradient histogram";
  // A1 of the descriptors exercise
  const h = D.cellHistogram([[2, 10], [1, 50], [3, 100], [2, 355]]);
  check(h.join() === "4,1,3,0,0,0,0,0", `cell histogram ${h}`);
  const hr = D.cellHistogram([[2, 100], [1, 140], [3, 190], [2, 85]]);
  check(hr.join() === "0,0,4,1,3,0,0,0", `rotated by 90°: ${hr}`);
  const scene = T.featureScene({ size: 512, seed: 11 });
  const { V1, V2, T12 } = T.viewPair(512, 256, { angle: 35, zoom: 1.15 });
  const v1 = T.viewOf(scene, V1), v2 = T.viewOf(scene, V2);
  const r1 = K.detectDoG(v1, { contrast: 3, maxPoints: 120 });
  const k1 = K.assignOrientations(r1.pyr, r1.kps);
  const d = D.gradHistDescriptor(r1.pyr, k1[0]);
  let n = 0; for (const v of d) n += v * v;
  check(d.length === 128 && close(Math.sqrt(n), 1, 1e-5), "128 numbers, unit length");
  // invariance to I → 3I + 20
  const v1b: Grey = { width: 256, height: 256, data: v1.data.map(v => 3 * v + 20) };
  const pb = K.dogPyramid(v1b);
  let maxd = 0;
  for (const k of k1.slice(0, 30)) { const a = D.gradHistDescriptor(r1.pyr, k), b = D.gradHistDescriptor(pb, k); maxd = Math.max(maxd, D.l2(a, b)); }
  check(maxd < 1e-4, `unchanged under 3I + 20 (max distance ${maxd})`);
  // descriptors of the rotated view (with θ): the true partner is nearer than any other
  const r2 = K.detectDoG(v2, { contrast: 3, maxPoints: 200 }), k2 = K.assignOrientations(r2.pyr, r2.kps);
  const D2 = k2.map(k => D.gradHistDescriptor(r2.pyr, k));
  let good = 0, tot = 0;
  for (const a of k1) {
    if (a.x > 180 && a.y > 180) continue;            // the scene's repetitive patch (feature-matching/ is about it)
    const [x, y] = T.apply(T12, a.x, a.y);
    const partners = k2.map((b, j) => ({ b, j })).filter(({ b }) => Math.hypot(b.x - x, b.y - y) < 1.5 && angDiff(b.theta, a.theta + 35 * Math.PI / 180) < 0.2);
    if (!partners.length) continue;
    tot++;
    const da = D.gradHistDescriptor(r1.pyr, a);
    let jn = -1, dn = Infinity;
    D2.forEach((b, j) => { const e = D.l2(da, b); if (e < dn) { dn = e; jn = j; } });
    if (partners.some(p => p.j === jn)) good++;
  }
  check(tot > 30 && good / tot > 0.9, `partner nearest for ${good}/${tot}`);
  console.log(`  rotated view: the true partner is the nearest descriptor for ${good} of ${tot} keypoints`);
}

// ---- binary descriptors and Hamming ----
function testBinary(): void {
  current = "binary";
  const bits = (s: string) => { const a = new Uint32Array(1); [...s.replace(/ /g, "")].forEach((c, k) => { if (c === "1") a[0] |= (1 << k) >>> 0; }); return a; };
  check(D.hamming(bits("1011 0010 1110 0001"), bits("1001 0110 1100 0011")) === 4, "worked example: 4");
  const r = rng(3);
  for (let t = 0; t < 50; t++) {
    const a = Uint32Array.from({ length: 8 }, () => Math.floor(r.unif() * 2 ** 32)), b = Uint32Array.from({ length: 8 }, () => Math.floor(r.unif() * 2 ** 32));
    let ref = 0; for (let k = 0; k < 256; k++) ref += ((a[k >> 5] >>> (k & 31)) & 1) !== ((b[k >> 5] >>> (k & 31)) & 1) ? 1 : 0;
    if (D.hamming(a, b) !== ref) { check(false, "hamming vs bit by bit"); break; }
  }
  const scene = T.featureScene({ size: 512, seed: 11 });
  const v1 = T.viewOf(scene, T.viewPair(512, 256, {}).V1);
  const p = K.dogPyramid(v1), kps = K.detectDoG(v1, { contrast: 3, maxPoints: 40 }, p).kps;
  const pat = D.binaryPattern(7);
  const p2 = K.dogPyramid({ width: 256, height: 256, data: v1.data.map(v => 255 * Math.pow(v / 255, 0.5) + 10) });
  let same = 0;
  // a strictly increasing map of the image: every pixel comparison keeps its sign. The pyramid blurs (a linear
  // operation), so the bits of the blurred patch are kept for almost all pairs, not exactly all.
  for (const k of kps) { const a = D.binaryDescriptor(p, k, pat), b = D.binaryDescriptor(p2, k, pat); same += 256 - D.hamming(a, b); }
  check(same / (256 * kps.length) > 0.9, `bits kept under a gamma map: ${(same / (256 * kps.length)).toFixed(3)}`);
  // on the unblurred patch the bits are exactly invariant to an increasing map
  const patch = T.viewOf(scene, T.viewPair(512, 32, {}).V1, { size: 32 });
  const patch2: Grey = { width: 32, height: 32, data: patch.data.map(v => v * v / 255 + 3) };
  const pat0: D.BinaryPattern = { ...pat, pairs: pat.pairs.map(Math.round) };
  check(D.hamming(D.binaryFromPatch(patch, pat0), D.binaryFromPatch(patch2, pat0)) === 0, "bits unchanged under I → I²/255 + 3 at pixel positions");
}

// ---- matching ----
function testMatching(): void {
  current = "matching";
  // the unit-3 exercise table: 4 points of image 1 × 5 of image 2
  const C = [[0.20, 0.60, 0.75, 0.90, 0.70], [0.55, 0.30, 0.32, 0.80, 0.85], [0.70, 0.65, 0.28, 0.90, 0.95], [0.21, 0.70, 0.90, 0.85, 0.60]];
  const D1 = C.map((_, i) => Float32Array.from([i])), D2 = C[0].map((_, j) => Float32Array.from([j]));
  // build Match lists from the table directly
  const m12: M.Match[] = C.map((row, i) => { const s = row.map((d, j) => ({ d, j })).sort((a, b) => a.d - b.d); return { i, j: s[0].j, d1: s[0].d, j2: s[1].j, d2: s[1].d }; });
  const m21: M.Match[] = C[0].map((_, j) => { const s = C.map((row, i) => ({ d: row[j], i })).sort((a, b) => a.d - b.d); return { i: j, j: s[0].i, d1: s[0].d, j2: s[1].i, d2: s[1].d }; });
  void D1; void D2;
  const nn = m12.map(m => `${m.i + 1}→${m.j + 1}`).join(" ");
  check(nn === "1→1 2→2 3→3 4→1", `nearest neighbours ${nn}`);
  const rt = M.ratioTest(m12, 0.8).map(m => m.i + 1).join();
  check(rt === "1,3,4", `ratio 0.8 keeps ${rt}`);
  const mu = M.mutualCheck(M.ratioTest(m12, 0.8), m21).map(m => m.i + 1).join();
  check(mu === "1,3", `ratio + mutual keeps ${mu}`);
  // ratio pairs of the slide
  const acc = (a: number, b: number, t: number) => a / b < t;
  check(acc(0.2, 0.6, 0.8) && !acc(0.3, 0.32, 0.8), "0.20/0.60 accepted, 0.30/0.32 rejected");
  // k-d tree exact = brute force; evaluateMatches against a known shift
  const r = rng(5), P = Array.from({ length: 300 }, () => Float32Array.from({ length: 8 }, () => r.unif()));
  const Q = Array.from({ length: 50 }, () => Float32Array.from({ length: 8 }, () => r.unif()));
  const tree = new M.KdTree(P);
  let agree = 0;
  for (const q of Q) {
    const e = tree.nearest2(q);
    let j = -1, d = Infinity; P.forEach((p, k) => { const x = D.l2(p, q); if (x < d) { d = x; j = k; } });
    if (e.j === j && close(e.d1, d, 1e-6)) agree++;
  }
  check(agree === Q.length, `k-d tree exact search = brute force (${agree}/${Q.length})`);
  const k1 = [{ x: 10, y: 10 }, { x: 50, y: 20 }, { x: 30, y: 70 }], k2 = [{ x: 15, y: 13 }, { x: 55, y: 23 }, { x: 90, y: 90 }];
  const ms: M.Match[] = [{ i: 0, j: 0, d1: 0, j2: -1, d2: 1 }, { i: 1, j: 1, d1: 0, j2: -1, d2: 1 }, { i: 2, j: 2, d1: 0, j2: -1, d2: 1 }];
  const q = M.evaluateMatches(ms, k1, k2, (x, y) => [x + 5, y + 3], 2);
  check(q.nCorrect === 2 && q.possible === 2 && close(q.precision, 2 / 3, 1e-12) && q.recall === 1, `evaluateMatches ${JSON.stringify({ c: q.nCorrect, p: q.possible })}`);
}

// ---- transforms and RANSAC ----
function testTransforms(): void {
  current = "transforms";
  const r = rng(8);
  const pts = Array.from({ length: 12 }, () => ({ x: 300 * r.unif(), y: 300 * r.unif() }));
  const gens: Record<T.Kind, T.Transform2D> = {
    translation: T.translation(12.5, -7),
    rigid: T.similarity(1, 0.4, 100, 80, 5, -3),
    similarity: T.similarity(1.3, -0.7, 120, 90, -8, 4),
    affine: Float64Array.from([1.1, 0.25, 7, -0.15, 0.9, -12, 0, 0, 1]),
    homography: T.rotatingCamera(300, 150, 150, 0.2, 0.15),
  };
  for (const kind of T.KINDS) {
    const s = T.minimalSize(kind), src = pts.slice(0, s), dst = src.map(p => { const [x, y] = T.apply(gens[kind], p.x, p.y); return { x, y }; });
    const F = T.fitTransform(kind, src, dst);
    const err = F ? Math.max(...pts.map(p => Math.hypot(...[0, 1].map(k => T.apply(F, p.x, p.y)[k] - T.apply(gens[kind], p.x, p.y)[k]) as [number, number]))) : Infinity;
    check(err < 1e-6, `${kind} recovered from ${s} pairs (err ${err})`);
  }
  // homographyFit agrees with camera.ts homographyDLT
  const src = pts.slice(0, 6), dst = src.map(p => { const [x, y] = T.apply(gens.homography, p.x, p.y); return { x: x + 0.3 * r.norm(), y: y + 0.3 * r.norm() }; });
  const F = T.fitTransform("homography", src, dst)!, G = homographyDLT(src.map(p => [p.x, p.y]), dst.map(p => [p.x, p.y])).H;
  check(F.every((v, k) => close(v, G[k], 1e-9)), "homography fit = homographyDLT");
  // RANSAC with 50 % outliers recovers a similarity to < 0.5 px
  const Tsim = gens.similarity, S: T.P2[] = [], Dd: T.P2[] = [];
  for (let k = 0; k < 60; k++) {
    const p = { x: 256 * r.unif(), y: 256 * r.unif() };
    const [x, y] = T.apply(Tsim, p.x, p.y);
    S.push(p);
    Dd.push(k % 2 === 0 ? { x: x + 0.4 * r.norm(), y: y + 0.4 * r.norm() } : { x: 300 * r.unif(), y: 300 * r.unif() });
  }
  const R = T.ransacTransform("similarity", S, Dd, { eps: 2, iterations: T.iterationsNeeded(0.5, 2), seed: 1 });
  const e = R.refit ? Math.max(...[[0, 0], [256, 0], [256, 256], [0, 256]].map(([x, y]) => { const a = T.apply(R.refit!, x, y), b = T.apply(Tsim, x, y); return Math.hypot(a[0] - b[0], a[1] - b[1]); })) : Infinity;
  check(e < 0.5, `RANSAC similarity, 50 % outliers: corner error ${e.toFixed(3)} px`);
  // the slide table
  const tab = [0.5, 0.3, 0.7].map(w => [1, 2, 3, 4].map(s => iterationsNeeded(w, s, 0.99)).join(","));
  check(tab[0] === "7,17,35,72" && tab[1] === "13,49,169,567" && tab[2] === "4,7,11,17", `iterations table ${tab.join(" | ")}`);
  // global similarities: exact data recovers the tiles
  const truth = [T.identity(), T.similarity(1.1, 0.1, 0, 0, 200, 10), T.similarity(0.95, -0.08, 0, 0, 10, 190)];
  const links: T.TileMatch[] = [];
  for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
    const p: T.P2[] = [], q: T.P2[] = [];
    for (let k = 0; k < 10; k++) {
      const w = { x: 100 + 150 * r.unif(), y: 100 + 150 * r.unif() };
      const a = T.apply(T.invert(truth[i]), w.x, w.y), b = T.apply(T.invert(truth[j]), w.x, w.y);
      p.push({ x: a[0], y: a[1] }); q.push({ x: b[0], y: b[1] });
    }
    links.push({ i, j, p, q });
  }
  const G2 = T.globalSimilarities(3, links);
  check(G2.every((Sx, k) => Sx.every((v, m) => close(v, truth[k][m], 1e-6))), "global similarities recover the tiles");
}

testPyramid(); testBlob(); testRefineAndEdge(); testRepeatability(); testOrientation(); testGradHist(); testBinary(); testMatching(); testTransforms();
console.log(failures ? `${failures} of ${checks} checks FAILED` : `all ${checks} checks passed`);
if (failures) process.exit(1);
