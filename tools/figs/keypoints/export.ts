// Data for the static figures of the keypoints day (keypoints/, descriptors/,
// feature-matching/, feature-alignment/), computed with the course libraries (the same code
// as the demos). Writes JSON files into the directory given as the first argument; the
// figs.py script of each unit reads them.
//
// Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/keypoints/export.ts --bundle --platform=node --log-level=warning > /tmp/kpexport.js
//   node /tmp/kpexport.js <data-dir>
// (any scratch location for the bundle; then python3 -I tools/figs/<unit>/figs.py <data-dir>)
import * as K from "../../../demos/src/lib/keypoints";
import * as D from "../../../demos/src/lib/descriptors";
import * as M from "../../../demos/src/lib/featurematch";
import * as T from "../../../demos/src/lib/transforms2d";
import { Grey } from "../../../demos/src/lib/imageops";
import { rng } from "../../../demos/src/lib/random";
import { makeScene } from "../../../demos/src/dogpyramid/scene";

declare const require: (m: string) => { writeFileSync(p: string, s: string): void; mkdirSync(p: string, o: object): void };
declare const process: { argv: string[] };
const fs = require("fs");
const out = process.argv[2] ?? "kpdata";
fs.mkdirSync(out, { recursive: true });
const save = (name: string, obj: unknown) => fs.writeFileSync(`${out}/${name}.json`, JSON.stringify(obj));
const img = (g: Grey) => ({ w: g.width, h: g.height, data: Array.from(g.data, v => Math.round(v * 10) / 10) });
const kp = (k: K.Keypoint) => ({ x: k.x, y: k.y, s: k.sigma, t: k.theta, r: k.response });

const scene = T.featureScene({ size: 512, seed: 11 });
const N = 256;
function pairOf(spec: T.ViewPairSpec, light: T.ViewOptions = { noise: 2 }, sc = scene) {
  const { V1, V2, T12 } = T.viewPair(512, N, spec);
  return { v1: T.viewOf(sc, V1, { size: N }), v2: T.viewOf(sc, V2, { size: N, seed: 5, ...light }), T12 };
}
const dog = (g: Grey, n = 200) => { const p = K.dogPyramid(g, { S: 3 }); return { pyr: p, kps: K.detectDoG(g, { contrast: 3, maxPoints: n }, p).kps }; };
const hist = (f: { pyr: K.DoGPyramid; kps: K.Keypoint[] }) => { const kps = K.assignOrientations(f.pyr, f.kps); return { kps, desc: kps.map(k => D.gradHistDescriptor(f.pyr, k)) }; };

// ---- keypoints/ ----
{
  // the task: two views, correct matches
  const P = pairOf({ angle: 30, zoom: 1.3 });
  const a = hist(dog(P.v1)), b = hist(dog(P.v2));
  const m = M.mutualCheck(M.ratioTest(M.matchDescriptors(a.desc, b.desc), 0.7), M.matchDescriptors(b.desc, a.desc));
  const ok = M.correctFlags(m, a.kps, b.kps, T.mapOf(P.T12), 3);
  save("task", { v1: img(P.v1), v2: img(P.v2), pairs: m.filter((_, k) => ok[k]).map(x => [kp(a.kps[x.i]), kp(b.kps[x.j])]) });
  // Harris at one σ on a view and its 2.2× zoom
  const Z = pairOf({ zoom: 2.2 });
  const h1 = K.detectHarris(Z.v1, 1.5, { maxPoints: 200, rel: 0.002 }), h2 = K.detectHarris(Z.v2, 1.5, { maxPoints: 200, rel: 0.002 });
  const rh = K.repeatability(h1, h2, T.mapOf(Z.T12), T.mapOf(T.invert(Z.T12)), { w1: N, h1: N, w2: N, h2: N, eps: 2.5, scaleTol: Infinity });
  const d1 = dog(Z.v1), d2 = dog(Z.v2);
  const rd = K.repeatability(d1.kps, d2.kps, T.mapOf(Z.T12), T.mapOf(T.invert(Z.T12)), { w1: N, h1: N, w2: N, h2: N, eps: 2.5, scale: 2.2 });
  save("harriszoom", { v1: img(Z.v1), v2: img(Z.v2), T12: Array.from(Z.T12), h1: h1.map(kp), h2: h2.map(kp), hp: Array.from(rh.partner), hrep: rh.rep, d1: d1.kps.map(kp), d2: d2.kps.map(kp), dp: Array.from(rd.partner), drep: rd.rep });
  // DoG pyramid of the dogpyramid scene: octave 0 DoGs and the candidate records
  const sc = makeScene(192, 2), r = K.detectDoG(sc, { S: 3, contrast: 3, edgeR: 10 });
  save("dogscene", {
    img: img(sc), dog: r.pyr.octaves[0].dog.map(img), gauss: r.pyr.octaves[0].gauss.map(img), sizes: r.pyr.octaves.map(o => o.dog[0].width),
    records: r.records.map(x => ({ o: x.cand.octave, l: x.cand.level, ix: x.cand.ix, iy: x.cand.iy, x: x.refined.x, y: x.refined.y, s: x.refined.sigma, stage: x.stage, ratio: Number.isFinite(x.edge.ratio) ? x.edge.ratio : 1e9 })),
  });
  // an orientation histogram with two peaks
  const V = pairOf({});
  const f = dog(V.v1, 200);
  let pick: { k: K.Keypoint; h: Float64Array; peaks: number[] } | null = null;
  for (const k of f.kps) { const o = K.orientationHistogram(f.pyr, k); if (o.peaks.length === 2 && k.sigma > 2.5 && (!pick || k.response > pick.k.response)) pick = { k, h: o.hist, peaks: o.peaks }; }
  const L = K.levelImage(f.pyr, pick!.k);
  const R = Math.round(3 * 1.5 * L.s), patch: number[][] = [];
  for (let y = -R; y <= R; y++) { const row: number[] = []; for (let x = -R; x <= R; x++) { const xx = Math.round(L.cx) + x, yy = Math.round(L.cy) + y; row.push(L.img.data[Math.max(0, Math.min(L.img.height - 1, yy)) * L.img.width + Math.max(0, Math.min(L.img.width - 1, xx))]); } patch.push(row); }
  save("orientation", { kp: kp(pick!.k), hist: Array.from(pick!.h), peaks: pick!.peaks, patch, s: L.s, step: L.step });
  // frames on both views
  const F = pairOf({ angle: 40, zoom: 1.4 });
  const fa = hist(dog(F.v1, 60)), fb = hist(dog(F.v2, 90));
  save("frames", { v1: img(F.v1), v2: img(F.v2), T12: Array.from(F.T12), k1: fa.kps.map(kp), k2: fb.kps.map(kp) });
  // repeatability curves (one-to-one, ε 2.5)
  const zooms = [0.4, 0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3], angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180];
  const curves: Record<string, { zoom: number[]; angle: number[] }> = {};
  for (const det of ["harris", "harrisms", "dog", "fast"]) {
    const run = (spec: T.ViewPairSpec) => {
      const P2 = pairOf(spec);
      const k = (g: Grey) => det === "dog" ? dog(g).kps : det === "harris" ? K.detectHarris(g, 1.5, { maxPoints: 200, rel: 0.002 }) : det === "harrisms" ? K.detectHarrisMultiScale(g, undefined, { maxPoints: 200, rel: 0.002 }) : K.detectFAST(g, 20, 9, { maxPoints: 200 });
      return K.repeatability(k(P2.v1), k(P2.v2), T.mapOf(P2.T12), T.mapOf(T.invert(P2.T12)), { w1: N, h1: N, w2: N, h2: N, eps: 2.5, scale: spec.zoom ?? 1, scaleTol: det === "harris" || det === "fast" ? Infinity : Math.log(1.5) }).rep;
    };
    curves[det] = { zoom: zooms.map(z => run({ zoom: z })), angle: angles.map(a => run({ angle: a })) };
  }
  save("repcurves", { zooms, angles, curves });
}

// ---- descriptors/ ----
{
  const P = pairOf({ angle: 25, zoom: 1.2 });
  const fa = dog(P.v1, 150), fb = dog(P.v2, 150), a = hist(fa), b = hist(fb);
  // the keypoint used by descriptorlab (strongest distinctive one at (53, 54))
  const i = a.kps.reduce((best, k, q) => Math.hypot(k.x - 53, k.y - 54) < Math.hypot(a.kps[best].x - 53, a.kps[best].y - 54) ? q : best, 0);
  const [x2, y2] = T.apply(P.T12, a.kps[i].x, a.kps[i].y);
  const ds = b.desc.map((d, j) => ({ j, d: D.l2(a.desc[i], d) })).sort((p, q) => p.d - q.d);
  const partner = b.kps.reduce((best, k, q) => Math.hypot(k.x - x2, k.y - y2) < Math.hypot(b.kps[best].x - x2, b.kps[best].y - y2) ? q : best, 0);
  const imp = ds.filter(e => e.j !== partner && Math.hypot(b.kps[e.j].x - x2, b.kps[e.j].y - y2) > 10).slice(0, 2).map(e => e.j);
  const det = D.gradHistDetail(fa.pyr, a.kps[i]);
  save("job", {
    v1: img(P.v1), v2: img(P.v2), k1: kp(a.kps[i]), d1: Array.from(a.desc[i]), partner: { k: kp(b.kps[partner]), d: Array.from(b.desc[partner]), dist: D.l2(a.desc[i], b.desc[partner]) },
    impostors: imp.map(j => ({ k: kp(b.kps[j]), d: Array.from(b.desc[j]), dist: D.l2(a.desc[i], b.desc[j]) })),
    detail: { patch: img(det.patch), gx: Array.from(det.gx), gy: Array.from(det.gy), mag: Array.from(det.mag), angle: Array.from(det.angle), spacing: det.spacingOrig },
    binary: (() => { const pat = D.binaryPattern(7); const bp = D.binaryPatch(fa.pyr, a.kps[i], pat); return { patch: img(bp.patch), pairs: Array.from(pat.pairs).slice(0, 4 * 30), bits: Array.from({ length: 30 }, (_, k) => D.getBit(D.binaryFromPatch(bp.patch, pat), k)) }; })(),
  });
  // NCC of a window with itself shifted by 0..4 px, textured and smooth windows (raw pixels of view 1)
  const ncc = (cx: number, cy: number, r: number, sh: number) => {
    const A: number[] = [], B: number[] = [];
    for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { A.push(P.v1.data[(cy + y) * N + cx + x]); B.push(P.v1.data[(cy + y) * N + cx + x + sh]); }
    const na = D.normalisePatch(A), nb = D.normalisePatch(B); let s = 0; for (let k = 0; k < na.length; k++) s += na[k] * nb[k]; return s;
  };
  const shifts = [0, 0.5, 1, 1.5, 2, 3, 4];
  // sub-pixel shifts by bilinear sampling
  const nccF = (cx: number, cy: number, r: number, sh: number) => {
    const A: number[] = [], B: number[] = [];
    for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { A.push(P.v1.data[(cy + y) * N + cx + x]); const xx = cx + x + sh, x0 = Math.floor(xx), f = xx - x0; B.push((1 - f) * P.v1.data[(cy + y) * N + x0] + f * P.v1.data[(cy + y) * N + x0 + 1]); }
    const na = D.normalisePatch(A), nb = D.normalisePatch(B); let s = 0; for (let k = 0; k < na.length; k++) s += na[k] * nb[k]; return s;
  };
  void ncc;
  save("nccshift", { shifts, texture: shifts.map(s => nccF(60, 200, 8, s)), edge: shifts.map(s => nccF(110, 100, 8, s)), patchT: img(T.viewOf(scene, T.translation(-(60 + 128 - 8), -(200 + 128 - 8)), { size: 17 })), patchE: img(T.viewOf(scene, T.translation(-(110 + 128 - 8), -(100 + 128 - 8)), { size: 17 })) });
  // concentration of distances: random unit vectors in 2 and 128 dimensions
  const r = rng(4), conc: Record<string, { nearest: number[]; farthest: number[] }> = {};
  for (const dim of [2, 128]) {
    const pts = Array.from({ length: 500 }, () => { const v = Array.from({ length: dim }, () => r.norm()); const n = Math.hypot(...v); return v.map(x => x / n); });
    const nearest: number[] = [], farthest: number[] = [];
    for (let q = 0; q < 100; q++) { let mn = Infinity, mx = 0; for (let k = 100; k < 500; k++) { const d = D.l2(pts[q], pts[k]); mn = Math.min(mn, d); mx = Math.max(mx, d); } nearest.push(mn); farthest.push(mx); }
    conc[dim] = { nearest, farthest };
  }
  save("concentration", conc);
  // failures: repetitive scene, strong tilt, contrast inversion
  const grid = T.gridScene({ size: 512 });
  const G = pairOf({ angle: 10 }, { noise: 2 }, grid);
  const Tl = pairOf({ tilt: 60 });
  const inv = pairOf({ angle: 15 }, { noise: 2, gain: -1, offset: 255 });
  save("failures", { grid: img(G.v1), tilt: img(Tl.v2), inv: img(inv.v2), orig: img(inv.v1) });
}

// ---- feature-matching/ ----
{
  const P = pairOf({ angle: 30, zoom: 1.3 }, { gain: 1.2, noise: 3 });
  const a = hist(dog(P.v1)), b = hist(dog(P.v2));
  const m12 = M.matchDescriptors(a.desc, b.desc), m21 = M.matchDescriptors(b.desc, a.desc);
  const ok = M.correctFlags(m12, a.kps, b.kps, T.mapOf(P.T12), 3), possible = M.possibleCount(a.kps, b.kps, T.mapOf(P.T12), 3);
  const ratio = M.ratioTest(m12, 0.8), mut = M.mutualCheck(ratio, m21);
  const flag = (ms: M.Match[]) => M.correctFlags(ms, a.kps, b.kps, T.mapOf(P.T12), 3);
  const ts = Array.from({ length: 36 }, (_, k) => 0.3 + 0.02 * k);
  const mutAll = M.mutualCheck(m12, m21);
  save("matching", {
    v1: img(P.v1), v2: img(P.v2), k1: a.kps.map(kp), k2: b.kps.map(kp), possible,
    nn: m12.map((m, k) => [m.i, m.j, m.d1, m.d2, ok[k] ? 1 : 0]),
    ratio: ratio.map((m, k) => [m.i, m.j, flag(ratio)[k] ? 1 : 0]), mutual: mut.map((m, k) => [m.i, m.j, flag(mut)[k] ? 1 : 0]),
    pr: M.prCurve(m12, ok, possible, ts), prMut: M.prCurve(mutAll, flag(mutAll), possible, ts),
  });
  const grid = T.gridScene({ size: 512 });
  const G = pairOf({ angle: 10 }, { noise: 2 }, grid);
  const ga = hist(dog(G.v1)), gb = hist(dog(G.v2));
  const gm = M.ratioTest(M.matchDescriptors(ga.desc, gb.desc), 0.8);
  const gok = M.correctFlags(gm, ga.kps, gb.kps, T.mapOf(G.T12), 3);
  save("gridmatch", { v1: img(G.v1), v2: img(G.v2), k1: ga.kps.map(kp), k2: gb.kps.map(kp), m: gm.map((m, k) => [m.i, m.j, gok[k] ? 1 : 0]) });
}

// ---- feature-alignment/ ----
{
  const P = pairOf({ angle: 25, zoom: 1.2 }, { noise: 3 });
  const a = hist(dog(P.v1)), b = hist(dog(P.v2));
  const acc = M.mutualCheck(M.ratioTest(M.matchDescriptors(a.desc, b.desc), 0.8), M.matchDescriptors(b.desc, a.desc));
  const src = acc.map(m => ({ x: a.kps[m.i].x, y: a.kps[m.i].y })), dst = acc.map(m => ({ x: b.kps[m.j].x, y: b.kps[m.j].y }));
  const r = rng(77), extra = Math.round(0.3 / 0.7 * acc.length);
  for (let e = 0; e < extra; e++) { const p = a.kps[Math.floor(r.unif() * a.kps.length)], q = b.kps[Math.floor(r.unif() * b.kps.length)]; src.push({ x: p.x, y: p.y }); dst.push({ x: q.x, y: q.y }); }
  const R = T.ransacTransform("similarity", src, dst, { eps: 3, iterations: 50, seed: 1 }), ls = T.fitTransform("similarity", src, dst)!;
  const warp = (Tr: T.Transform2D) => img(T.warpToFrame(P.v2, Tr, N, N));
  save("align", { v1: img(P.v1), v2: img(P.v2), src, dst, inl: R.refitInliers, ransac: Array.from(R.refit!), ls: Array.from(ls), w2r: warp(R.refit!), w2l: warp(ls), T12: Array.from(P.T12) });
  // rotating camera pair: affine vs homography
  const C = pairOf({ angle: 10, tilt: 30, pan: 25 }, { noise: 3 });
  const ca = hist(dog(C.v1)), cb = hist(dog(C.v2));
  const cacc = M.mutualCheck(M.ratioTest(M.matchDescriptors(ca.desc, cb.desc), 0.8), M.matchDescriptors(cb.desc, ca.desc));
  const cs = cacc.map(m => ({ x: ca.kps[m.i].x, y: ca.kps[m.i].y })), cd = cacc.map(m => ({ x: cb.kps[m.j].x, y: cb.kps[m.j].y }));
  const Ra = T.ransacTransform("affine", cs, cd, { eps: 2, iterations: 200, seed: 1 }), Rh = T.ransacTransform("homography", cs, cd, { eps: 2, iterations: 200, seed: 1 });
  save("camera", { v1: img(C.v1), v2: img(C.v2), wa: img(T.warpToFrame(C.v2, Ra.refit!, N, N)), wh: img(T.warpToFrame(C.v2, Rh.refit!, N, N)), na: Ra.refitInliers.length, nh: Rh.refitInliers.length, n: cs.length });
  // mosaic: four tiles, chain vs global, three blends
  const TILES = [{ c: [195, 190], a: 0, z: 1, g: 1 }, { c: [330, 200], a: 7, z: 1.08, g: 1.22 }, { c: [190, 325], a: -5, z: 0.94, g: 0.85 }, { c: [325, 330], a: 11, z: 1.04, g: 1.12 }];
  const V = TILES.map(t => T.compose(T.similarity(t.z, t.a * Math.PI / 180, 127.5, 127.5), T.translation(127.5 - t.c[0], 127.5 - t.c[1])));
  const tiles = V.map((v, k) => T.viewOf(scene, v, { size: N, gain: TILES[k].g, noise: 2, seed: 30 + k }));
  const truth = V.map(v => T.compose(V[0], T.invert(v)));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const S of truth) for (const [u, v] of [[0, 0], [N, 0], [N, N], [0, N]]) { const [p, q] = T.apply(S, u, v); x0 = Math.min(x0, p); y0 = Math.min(y0, q); x1 = Math.max(x1, p); y1 = Math.max(y1, q); }
  const cw = Math.ceil(x1 - x0) + 4, ch = Math.ceil(y1 - y0) + 4, ox = 2 - x0, oy = 2 - y0;
  const blends: Record<string, unknown> = {};
  for (const mode of ["none", "feather", "multiband"] as T.BlendMode[]) blends[mode] = img(T.blendTiles(tiles, truth, cw, ch, ox, oy, mode, 5).img);
  save("mosaic", { blends, tiles: truth.map(S => [[0, 0], [N, 0], [N, N], [0, N]].map(([u, v]) => { const [p, q] = T.apply(S, u - 0.5, v - 0.5); return [p + ox, q + oy]; })) });
}
console.log("wrote", out);
