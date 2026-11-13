// From matches to a transform. Pair mode: two views (known transform), DoG keypoints with
// gradient-histogram descriptors, matches after the ratio test and the mutual check, plus
// optional seeded wrong matches. RANSAC step by step for the chosen model (translation …
// homography): the minimal sample (yellow), the exact fit (outline of view 2 drawn on
// view 1), the pairs within ε (green); the best inlier count against the iteration with the
// predicted N(w, s, 0.99); the refit result as a checkerboard overlay or difference image,
// with the RMS residual and the true error against the known transform. "Least squares on
// all" for comparison.
// Mosaic mode: four overlapping tiles of a larger scene with rotations, zooms and different
// exposures; pairwise similarity by RANSAC; tile positions by chaining along a path or by a
// global least-squares fit over all inlier matches; blending none / feather / multi-band.
//
// <div data-demo="featurealign" data-mode="pair|mosaic"
//      data-preset="ls|sim|homog|tight|eps|camera|texture|chain|multiband"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { Grey, createGrey } from "../lib/imageops";
import { rng } from "../lib/random";
import * as K from "../lib/keypoints";
import * as M from "../lib/featurematch";
import * as T2 from "../lib/transforms2d";
import { CELL_CREDIT } from "../lib/cell";
import { ImgView, injectStyle } from "../canny/imgview";
import { getScene, makeViews, detect, describe, scheduler, SceneKind, PairView, GREEN, RED, YELLOW, BLUE, COMMON_CSS, ViewPairResult, VIEW } from "../repeatability/kpcommon";

const NAME = "featurealign";
const STYLE = COMMON_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .fa-main { flex: 1 1 520px; min-width: 300px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .fa-row { display: flex; gap: .5em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .fa-row > .ev-fig { flex: 0 1 240px; min-width: 160px; }
.qb-demo[data-demo="${NAME}"] .fa-row > svg { flex: 1 1 260px; min-width: 200px; max-width: 360px; }
.qb-demo[data-demo="${NAME}"] .fa-mosaic { flex: 1 1 460px; max-width: 560px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .fa-key { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .fa-main { flex: 0 0 630px; }
.reveal .qb-demo[data-demo="${NAME}"] .fa-row > .ev-fig { flex: 0 0 170px; }
.reveal .qb-demo[data-demo="${NAME}"] .fa-row > svg { max-width: 330px; }
.reveal .qb-demo[data-demo="${NAME}"] .fa-key { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .fa-mosaic { flex: 0 0 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 420px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .ev-credit { display: none; }
`;

const KIND_LABEL: Record<T2.Kind, string> = { translation: "translation", rigid: "rigid", similarity: "similarity", affine: "affine", homography: "homography" };

// ---------------------------------------------------------------------------------------
// Pair mode

interface PairState {
  scene: SceneKind; kind: T2.Kind; eps: number; iters: number; seed: number; outFrac: number; ratio: number; ls: boolean;
  angle: number; zoom: number; tilt: number; pan: number; blur: number; noise: number; show: "checker" | "diff"; step: number;
}
const PDEF: PairState = { scene: "scene", kind: "similarity", eps: 3, iters: 50, seed: 1, outFrac: 0.3, ratio: 0.8, ls: false, angle: 25, zoom: 1.2, tilt: 0, pan: 0, blur: 0, noise: 3, show: "checker", step: -1 };
interface PPreset { id: string; label: string; s: Partial<PairState> }
const PPRESETS: PPreset[] = [
  { id: "ls", label: "least squares on all matches fails", s: { ls: true } },
  { id: "sim", label: "RANSAC similarity: a few iterations", s: { kind: "similarity", iters: 20 } },
  { id: "homog", label: "homography needs hundreds at w = 0.3", s: { kind: "homography", outFrac: 0.65, ratio: 0.9, iters: 100 } },
  { id: "tight", label: "a tighter ratio test → fewer iterations", s: { kind: "homography", outFrac: 0, ratio: 0.7, iters: 100 } },
  { id: "eps", label: "ε too large", s: { eps: 25, outFrac: 0.5 } },
  { id: "camera", label: "affine on a rotating-camera pair; homography fits", s: { kind: "affine", angle: 10, zoom: 1, tilt: 14, pan: 12, outFrac: 0, iters: 200, eps: 2, show: "diff" } },
  { id: "texture", label: "no texture: too few inliers, report failure", s: { blur: 8, noise: 25, outFrac: 0 } },
];

function mountPair(host: HTMLElement, p0: PPreset | undefined, switchMode: () => void): void {
  const start = (): PairState => ({ ...PDEF, ...(host.dataset.scene ? { scene: host.dataset.scene as SceneKind } : {}), ...(p0?.s ?? {}) });
  const st: PairState = start();

  const main = htmlEl("div", "fa-main", "", host);
  const pv = new PairView(main, "");
  const key = htmlEl("div", "fa-key", "", main);
  key.innerHTML = `<b style="color:${YELLOW}">━</b> sample &nbsp; <b style="color:${GREEN}">━</b> inlier (within ε) &nbsp; <b style="color:${RED}">━</b> outlier &nbsp; <b style="color:${BLUE}">▭</b> best fit: view 2's outline`;
  const row = htmlEl("div", "fa-row", "", main);
  const over = new ImgView(row, "");
  const plot = new Plot(row, { width: 330, height: 200, x: [0, 50], y: [0, 1], font: 17, xlabel: "iteration", ylabel: "best inliers", margin: { l: 58, r: 10, t: 10, b: 46 }, xticks: 4, yticks: 3 });
  const curve = plot.step([], [], { color: "#1f77b4", width: 3 });
  const nLine = plot.vline(NaN, { color: "#d62728", width: 2.5, dash: "6 4", text: "N" });
  const tLine = plot.hline(NaN, { color: GREEN, width: 2, dash: "3 3" });
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const busy = htmlEl("div", "kp-busy", "", main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PPRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, PDEF, p.s); sync(); load(); } })));
  const kindBtns = presets(panel, T2.KINDS.map(k => ({ label: KIND_LABEL[k], apply: () => { st.kind = k; st.step = -1; fit(); } })));
  const runRow = htmlEl("div", "qb-presets", "", panel);
  button(runRow, "one sample", () => { st.step = st.step < 0 ? 1 : Math.min(st.iters, st.step + 1); fit(); });
  button(runRow, "run N", () => { st.step = -1; fit(); });
  button(runRow, "restart", () => { st.step = 0; fit(); });
  const lsC = checkbox(runRow, "least squares on all", st.ls, v => { st.ls = v; fit(); });
  const eS = slider(panel, "ε (px)", 0.5, 30, 0.5, st.eps, v => { st.eps = v; fit(); });
  const iS = slider(panel, "iterations N", 1, 1000, 1, st.iters, v => { st.iters = v; st.step = -1; fit(); });
  const sS = slider(panel, "seed", 1, 20, 1, st.seed, v => { st.seed = v; fit(); });
  const oS = slider(panel, "added wrong matches", 0, 0.8, 0.05, st.outFrac, v => { st.outFrac = v; build(); }, v => `${Math.round(100 * v)} %`);
  const rS = slider(panel, "ratio test t", 0.5, 1, 0.01, st.ratio, v => { st.ratio = v; build(); });
  const viewRow = htmlEl("div", "qb-presets", "", panel);
  const showBtns = presets(viewRow, [
    { label: "checkerboard", apply: () => { st.show = "checker"; draw(); } },
    { label: "difference", apply: () => { st.show = "diff"; draw(); } },
  ]);
  button(viewRow, "mosaic mode", switchMode);
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); load(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { lsC.checked = st.ls; eS.set(st.eps); iS.set(st.iters); sS.set(st.seed); oS.set(st.outFrac); rS.set(st.ratio); }

  let scene: Grey | null = null, views: ViewPairResult | null = null;
  let k1: K.Keypoint[] = [], k2: K.Keypoint[] = [], m12: M.Match[] = [], m21: M.Match[] = [];
  let src: T2.P2[] = [], dst: T2.P2[] = [], truth: boolean[] = [];
  let R: T2.PairRansac | null = null, Rall: T2.PairRansac | null = null, lsT: T2.Transform2D | null = null;
  const run = scheduler(on => { busy.textContent = on ? "computing…" : ""; });
  function load(): void { getScene(st.scene).then(g => { scene = g; recompute(); }); }

  function recompute(): void {
    run(() => {
      if (!scene) return;
      views = makeViews(st.scene, scene, { zoom: st.zoom, angle: st.angle, tilt: st.tilt, pan: st.pan }, { noise: st.noise, blur: st.blur });
      const a = detect(views.v1, "dog", 200), b = detect(views.v2, "dog", 200);
      const r1 = describe(a, "hist"), r2 = describe(b, "hist");
      k1 = r1.kps; k2 = r2.kps;
      m12 = M.matchDescriptors(r1.desc, r2.desc, "l2"); m21 = M.matchDescriptors(r2.desc, r1.desc, "l2");
      build();
    });
  }
  /** Accepted matches (ratio + mutual) and the added wrong ones. */
  function build(): void {
    if (!views) return;
    const acc = M.mutualCheck(M.ratioTest(m12, st.ratio < 1 ? st.ratio : 1.0001), m21);
    src = acc.map(m => ({ x: k1[m.i].x, y: k1[m.i].y })); dst = acc.map(m => ({ x: k2[m.j].x, y: k2[m.j].y }));
    truth = M.correctFlags(acc, k1, k2, views.map12, 3);
    const extra = Math.round(st.outFrac / (1 - st.outFrac) * acc.length), r = rng(77);
    for (let e = 0; e < extra && k1.length && k2.length; e++) {
      const p = k1[Math.floor(r.unif() * k1.length)], q = k2[Math.floor(r.unif() * k2.length)];
      src.push({ x: p.x, y: p.y }); dst.push({ x: q.x, y: q.y });
      const [x, y] = views.map12(p.x, p.y); truth.push(Math.hypot(q.x - x, q.y - y) <= 3);
    }
    fit();
  }
  function fit(): void {
    if (!views) return;
    lsT = T2.fitTransform(st.kind, src, dst);
    Rall = T2.ransacTransform(st.kind, src, dst, { eps: st.eps, iterations: st.iters, seed: st.seed });
    R = st.step >= 0 ? T2.ransacTransform(st.kind, src, dst, { eps: st.eps, iterations: st.step, seed: st.seed }) : Rall;
    draw();
  }

  function outline(T: T2.Transform2D, color: string, width: number, dash = ""): void {
    // view 2's border mapped into view 1 (through T⁻¹), and view 1's border into view 2 (through T)
    let Ti: T2.Transform2D; try { Ti = T2.invert(T); } catch { return; }
    const c = [[0, 0], [VIEW - 1, 0], [VIEW - 1, VIEW - 1], [0, VIEW - 1]];
    const pts = [] as string[];
    for (const [x, y] of c) for (let k = 0; k < 8; k++) {
      const [x2, y2] = c[(c.findIndex(q => q[0] === x && q[1] === y) + 1) % 4];
      const u = x + (x2 - x) * k / 8, v = y + (y2 - y) * k / 8, [a, b] = T2.apply(Ti, u, v);
      if (Number.isFinite(a) && Number.isFinite(b)) pts.push(`${Math.max(-50, Math.min(VIEW + 50, a)) + 0.5},${Math.max(-50, Math.min(VIEW + 50, b)) + 0.5}`);
    }
    const p = pv.add("polygon", { points: pts.join(" "), fill: "none", "stroke-width": width, "stroke-dasharray": dash, "clip-path": pv.clip(1) });
    p.style.stroke = color;
  }

  function trueError(T: T2.Transform2D | null): number {
    if (!T || !views) return NaN;
    let s = 0;
    const pts = [[0, 0], [VIEW - 1, 0], [VIEW - 1, VIEW - 1], [0, VIEW - 1], [127.5, 127.5]];
    for (const [x, y] of pts) { const [a, b] = T2.apply(T, x, y), [c, d] = views.map12(x, y); s += Math.hypot(a - c, b - d); }
    return s / pts.length;
  }

  function draw(): void {
    if (!views || !R || !Rall) return;
    kindBtns.forEach((b, i) => b.classList.toggle("active", T2.KINDS[i] === st.kind));
    showBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.show === "checker")));
    credit.style.display = st.scene === "cell" ? "" : "none";
    pv.images(views.v1, views.v2);
    pv.clear();
    const best = st.ls ? lsT : (R.refit ?? R.best);
    const inl = new Set(st.ls ? (lsT ? T2.inliersOfPairs(lsT, src, dst, st.eps) : []) : (st.step >= 0 ? R.inliers : R.refitInliers));
    src.forEach((p, k) => pv.line(pv.at(1, p.x, p.y), pv.at(2, dst[k].x, dst[k].y), inl.has(k) ? GREEN : RED, inl.has(k) ? 1.6 : 1.1, inl.has(k) ? 0.85 : 0.55));
    // the current sample and its exact fit
    const last = st.step > 0 ? R.steps[R.steps.length - 1] : null;
    if (last && !st.ls) {
      for (const k of last.sample) pv.line(pv.at(1, src[k].x, src[k].y), pv.at(2, dst[k].x, dst[k].y), YELLOW, 4, 1);
      if (last.T) outline(last.T, YELLOW, 2.5, "7 5");
    }
    if (best) outline(best, BLUE, 3);
    // overlay
    if (best) {
      const w2 = T2.warpToFrame(views.v2, best, VIEW, VIEW), px = new Uint8ClampedArray(4 * VIEW * VIEW);
      for (let y = 0; y < VIEW; y++) for (let x = 0; x < VIEW; x++) {
        const i = y * VIEW + x, a = views.v1.data[i], b = w2.data[i];
        let v: number;
        if (st.show === "checker") v = ((x >> 5) + (y >> 5)) % 2 === 0 || !Number.isFinite(b) ? a : b;
        else v = Number.isFinite(b) ? 128 + 1.5 * (a - b) : 128;
        v = Math.max(0, Math.min(255, v)); px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255;
      }
      over.rgba(px, VIEW, VIEW);
      over.setCaption(st.show === "checker" ? "checkerboard" : "difference");
    } else { over.rgba(new Uint8ClampedArray(4 * VIEW * VIEW).fill(200), VIEW, VIEW); over.setCaption("no fit"); }
    // inlier curve
    const xs = Rall.steps.map(s => s.iter), ys = Rall.steps.map(s => s.best);
    const shown = st.step >= 0 ? st.step : st.iters;
    const nTrue = truth.filter(Boolean).length, w = src.length ? nTrue / src.length : 0, s = T2.minimalSize(st.kind), N = T2.iterationsNeeded(w, s, 0.99);
    plot.setDomain([0, Math.max(st.iters, 5)], [0, Math.max(5, src.length) * 1.05]);
    curve.set(xs.slice(0, shown), ys.slice(0, shown));
    nLine.set(Number.isFinite(N) && N <= st.iters ? N : NaN); tLine.set(nTrue);
    const nIn = inl.size, fail = nIn < Math.max(10, 0.04 * src.length);
    const rms = best ? T2.rmsError(best, src, dst, [...inl]) : NaN;
    ro.textContent = `matches: ${src.length} (truly correct ${nTrue}, w = ${w.toFixed(2)})\n${KIND_LABEL[st.kind]}: s = ${s}, N(w, s, 0.99) = ${Number.isFinite(N) ? N : "∞"}\n` +
      (st.ls ? "least squares on all pairs" : `iteration ${shown} of ${st.iters}; best found at ${R.foundAt}`) +
      `\ninliers: ${nIn}, RMS ${Number.isFinite(rms) ? rms.toFixed(2) : "–"} px\ntrue error: ${trueError(best).toFixed(2)} px` + (fail ? "\n→ too few inliers: NO reliable alignment" : "");
  }

  sync();
  if (p0) presetBtns[PPRESETS.indexOf(p0)].classList.add("active");
  load();
}

// ---------------------------------------------------------------------------------------
// Mosaic mode

interface MosState { blend: T2.BlendMode; method: "chain" | "global"; jitter: number; showTruth: boolean }
const MDEF: MosState = { blend: "none", method: "chain", jitter: 1.5, showTruth: false };
interface MPreset { id: string; label: string; s: Partial<MosState> }
const MPRESETS: MPreset[] = [
  { id: "chain", label: "mosaic: chain drifts, global fixes", s: { method: "chain", jitter: 1.5 } },
  { id: "global", label: "global least squares", s: { method: "global", jitter: 1.5 } },
  { id: "feather", label: "feathering", s: { method: "global", blend: "feather" } },
  { id: "multiband", label: "multi-band blending", s: { method: "global", blend: "multiband" } },
];
const TILES = [
  { c: [195, 190], a: 0, z: 1, g: 1 }, { c: [330, 200], a: 7, z: 1.08, g: 1.22 },
  { c: [190, 325], a: -5, z: 0.94, g: 0.85 }, { c: [325, 330], a: 11, z: 1.04, g: 1.12 },
];

function mountMosaic(host: HTMLElement, p0: MPreset | undefined, switchMode: () => void): void {
  const st: MosState = { ...MDEF, ...(p0?.s ?? {}) };
  const main = htmlEl("div", "fa-mosaic", "", host);
  const view = new ImgView(main, "");
  const busy = htmlEl("div", "kp-busy", "", main);
  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, MPRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, MDEF, p.s); sync(); solve(); } })));
  const methBtns = presets(panel, [
    { label: "pairwise chain", apply: () => { st.method = "chain"; solve(); } },
    { label: "global refinement", apply: () => { st.method = "global"; solve(); } },
  ]);
  const blendBtns = presets(panel, (["none", "feather", "multiband"] as T2.BlendMode[]).map(b => ({ label: b === "multiband" ? "multi-band" : b, apply: () => { st.blend = b; compose(); } })));
  const jS = slider(panel, "match noise (px)", 0, 3, 0.25, st.jitter, v => { st.jitter = v; solve(); });
  const tC = checkbox(panel, "outline the true tile positions", st.showTruth, v => { st.showTruth = v; compose(); });
  const r2 = htmlEl("div", "qb-presets", "", panel);
  button(r2, "pair mode", switchMode);
  button(r2, "reset", () => { Object.assign(st, MDEF, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); sync(); solve(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { jS.set(st.jitter); tC.checked = st.showTruth; }

  let tiles: Grey[] = [], truthS: T2.Transform2D[] = [], estS: T2.Transform2D[] = [];
  let pairs: { i: number; j: number; src: T2.P2[]; dst: T2.P2[] }[] = [];
  const run = scheduler(on => { busy.textContent = on ? "computing…" : ""; });

  getScene("scene").then(scene => run(() => {
    const V = TILES.map(t => T2.compose(T2.similarity(t.z, t.a * Math.PI / 180, 127.5, 127.5), T2.translation(127.5 - t.c[0], 127.5 - t.c[1])));
    tiles = V.map((v, k) => T2.viewOf(scene, v, { size: VIEW, gain: TILES[k].g, noise: 2, seed: 30 + k }));
    truthS = V.map(v => T2.compose(V[0], T2.invert(v)));
    const feats = tiles.map(t => describe(detect(t, "dog", 250), "hist"));
    pairs = [];
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
      const a = M.matchDescriptors(feats[i].desc, feats[j].desc), b = M.matchDescriptors(feats[j].desc, feats[i].desc);
      const acc = M.mutualCheck(M.ratioTest(a, 0.8), b);
      const src = acc.map(m => ({ x: feats[i].kps[m.i].x, y: feats[i].kps[m.i].y })), dst = acc.map(m => ({ x: feats[j].kps[m.j].x, y: feats[j].kps[m.j].y }));
      const R = T2.ransacTransform("similarity", src, dst, { eps: 3, iterations: 200, seed: 1 });
      if (R.refitInliers.length >= 10) pairs.push({ i, j, src: R.refitInliers.map(k => src[k]), dst: R.refitInliers.map(k => dst[k]) });
    }
    solve();
  }));

  function solve(): void {
    if (!tiles.length) return;
    methBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.method === "chain")));
    const r = rng(12);
    const noisy = pairs.map(p => ({ ...p, src: p.src.map(q => ({ x: q.x + st.jitter * r.norm(), y: q.y + st.jitter * r.norm() })), dst: p.dst.map(q => ({ x: q.x + st.jitter * r.norm(), y: q.y + st.jitter * r.norm() })) }));
    if (st.method === "chain") {
      // a path through the tiles: 0 → 1 → 3 → 2 (each step fitted from its own pair only)
      const order = [[0, 1], [1, 3], [2, 3], [0, 2], [0, 3], [1, 2]];
      const edges = order.map(([i, j]) => noisy.find(p => p.i === i && p.j === j)).filter((p): p is typeof noisy[0] => !!p)
        .map(p => ({ i: p.i, j: p.j, T: T2.fitTransform("similarity", p.src, p.dst)! }));
      estS = T2.chainTransforms(4, edges).map(S => S ?? T2.identity());
    } else estS = T2.globalSimilarities(4, noisy.map(p => ({ i: p.i, j: p.j, p: p.src, q: p.dst })));
    compose();
  }

  function compose(): void {
    blendBtns.forEach((b, i) => b.classList.toggle("active", (["none", "feather", "multiband"] as T2.BlendMode[])[i] === st.blend));
    run(() => {
      // canvas from the true tile corners
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const S of truthS) for (const [u, v] of [[0, 0], [VIEW, 0], [VIEW, VIEW], [0, VIEW]]) { const [a, b] = T2.apply(S, u, v); x0 = Math.min(x0, a); y0 = Math.min(y0, b); x1 = Math.max(x1, a); y1 = Math.max(y1, b); }
      const cw = Math.ceil(x1 - x0) + 4, ch = Math.ceil(y1 - y0) + 4, ox = 2 - x0, oy = 2 - y0;
      const { img } = T2.blendTiles(tiles, estS, cw, ch, ox, oy, st.blend, 5);
      const px = new Uint8ClampedArray(4 * cw * ch);
      for (let i = 0; i < cw * ch; i++) { const v = img.data[i], c = Number.isFinite(v) ? Math.max(0, Math.min(255, v)) : 255; px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = c; px[4 * i + 3] = 255; }
      view.rgba(px, cw, ch);
      view.clear();
      const corners = [[0, 0], [VIEW, 0], [VIEW, VIEW], [0, VIEW]];
      const errs: string[] = [];
      estS.forEach((S, k) => {
        const pts = corners.map(([u, v]) => T2.apply(S, u - 0.5, v - 0.5));
        view.add("polygon", { points: pts.map(([a, b]) => `${a + ox},${b + oy}`).join(" "), fill: "none", stroke: ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"][k], "stroke-width": 2 });
        if (st.showTruth) view.add("polygon", { points: corners.map(([u, v]) => T2.apply(truthS[k], u - 0.5, v - 0.5)).map(([a, b]) => `${a + ox},${b + oy}`).join(" "), fill: "none", stroke: "#111", "stroke-width": 1.5, "stroke-dasharray": "5 4" });
        let e = 0; for (const [u, v] of corners) { const [a, b] = T2.apply(S, u, v), [c, d] = T2.apply(truthS[k], u, v); e += Math.hypot(a - c, b - d) / 4; }
        errs.push(`tile ${k + 1}: ${e.toFixed(2)} px`);
      });
      view.setCaption(`${st.method === "chain" ? "pairwise chain 1 → 2 → 4 → 3" : "global least squares"}, blend: ${st.blend}`);
      ro.textContent = `tile pairs with ≥ 10 inliers: ${pairs.length} of 6\n${pairs.map(p => `${p.i + 1}–${p.j + 1}: ${p.src.length}`).join(", ")}\nmean corner error against the truth:\n${errs.join("\n")}`;
    });
  }

  sync();
  if (p0) presetBtns[MPRESETS.indexOf(p0)].classList.add("active");
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  let mode: "pair" | "mosaic" = host.dataset.mode === "mosaic" ? "mosaic" : "pair";
  const go = () => {
    host.replaceChildren();
    if (mode === "pair") mountPair(host, PPRESETS.find(p => p.id === host.dataset.preset), () => { mode = "mosaic"; go(); });
    else mountMosaic(host, MPRESETS.find(p => p.id === host.dataset.preset), () => { mode = "pair"; go(); });
  };
  go();
}

autoMount(NAME, mount);
void createGrey; void svgEl;
