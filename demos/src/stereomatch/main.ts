// Stereo matching on a rectified pair (lib/stereo.ts). Synthetic scenes with ground truth:
// random dots with a raised square; a slanted textured ground with a textureless wall, a
// striped fence and a post in front; the dots with a gain/offset change in the right camera.
// Panels: left image (click a pixel: its window), right image (the matched window and the
// epipolar row), disparity map (invalid in grey), errors against the truth (> 1 px in red,
// occluded pixels in blue) with the bad-pixel rate. Costs SAD / NCC / census in a window;
// aggregation: none (winner-take-all), scanline DP, SGM with 4 or 8 directions (P1, P2);
// left–right check; sub-pixel parabola. The clicked pixel's cost curve C(d). Second tab
// (data-tab="slice"): one row's x × d cost slice with the winner-take-all choices, the DP path
// and the truth.
//
// <div data-demo="stereomatch" data-preset="window|textureless|fence|occlusion|gain|scanline|sgm|subpixel"
//      data-tab="images|slice"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { makeScene, StereoScene, SceneId, costVolume, costVolumeRight, CostKind, wta, scanlineDP, sgm, leftRightCheck, subpixel, badPixelRate, parabolaVertex } from "../lib/stereo";
import { Plot } from "../lib/plot";
import { viridis } from "../lib/colour";
import { ImgView, injectStyle } from "../canny/imgview";

const NAME = "stereomatch";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .8em; }
${SEL} .sm-main { flex: 0 1 540px; min-width: 280px; }
${SEL} .sm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .3em .6em; }
${SEL} .sm-side { flex: 0 1 290px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .qb-panel { flex: 1 1 260px; max-width: 320px; gap: .3em; }
${SEL} .qb-readout { font-size: 15px; }
${SEL} .sm-tabs { display: flex; gap: .3em; margin-bottom: .3em; }
${SEL} .ev-wrap { aspect-ratio: 4 / 3; }
${SEL} .sm-row { display: flex; flex-wrap: wrap; gap: .25em; align-items: center; }
${SEL} .sm-lab { font-size: 15px; color: var(--qb-muted); margin-right: .2em; }
.reveal ${SEL} { font-size: 18px; }
.reveal ${SEL} .sm-main { flex: 0 0 520px; }
.reveal ${SEL} .sm-side { flex: 0 0 280px; }
.reveal ${SEL} .qb-panel { gap: .15em; }
.reveal ${SEL} .qb-btn { font-size: 16px; padding: .06em .35em; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${SEL} .ev-fig figcaption { font-size: 18px; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

type Agg = "wta" | "dp" | "sgm4" | "sgm8";
interface State { scene: SceneId; noise: number; cost: CostKind; r: number; agg: Agg; P1: number; P2: number; lr: boolean; sub: boolean; px: number; py: number; tab: "images" | "slice" }
const DEFAULT: State = { scene: "mixed", noise: 2, cost: "sad", r: 1, agg: "wta", P1: 0.1, P2: 1.0, lr: false, sub: false, px: 58, py: 45, tab: "images" };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "window", label: "window 3 × 3 vs 15 × 15", s: { scene: "dots", r: 1, px: 50, py: 70 } },
  { id: "textureless", label: "textureless wall", s: { r: 3, px: 58, py: 45 } },
  { id: "fence", label: "striped fence: repeated", s: { r: 3, px: 140, py: 40 } },
  { id: "occlusion", label: "occlusion: left–right check", s: { agg: "sgm8", lr: true, px: 74, py: 110 } },
  { id: "gain", label: "gain change: SAD fails, census copes", s: { scene: "gain", r: 3, px: 100, py: 70 } },
  { id: "scanline", label: "scanline DP: streaks", s: { agg: "dp", px: 58, py: 45 } },
  { id: "sgm", label: "semi-global matching", s: { agg: "sgm8", px: 58, py: 45 } },
  { id: "subpixel", label: "sub-pixel on the slanted ground", s: { agg: "sgm4", sub: true, px: 160, py: 120 } },
];
/** Bring the three costs to similar scales (a typical mismatch ≈ 1–2), so P1 and P2 mean the same for each. */
const COST_SCALE: Record<CostKind, number> = { sad: 1 / 64, ssd: 1 / 4096, ncc: 1, census: 1 / 8 };
const GREY_INVALID: [number, number, number] = [150, 150, 150];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}), ...(host.dataset.tab === "slice" ? { tab: "slice" as const } : {}) });
  const st: State = start();

  // ---- layout ----
  const main = htmlEl("div", "sm-main", "", host);
  const tabs = htmlEl("div", "sm-tabs", "", main);
  const tabI = button(tabs, "images and disparity", () => { st.tab = "images"; layout(); draw(); });
  const tabS = button(tabs, "cost slice of one row", () => { st.tab = "slice"; layout(); draw(); });
  const grid = htmlEl("div", "sm-grid", "", main);
  const vL = new ImgView(grid, "left image"), vR = new ImgView(grid, "right image");
  const vD = new ImgView(grid, "disparity"), vE = new ImgView(grid, "errors");
  const sliceDiv = htmlEl("div", "", "", main);
  const slicePlot = new Plot(sliceDiv, { width: 540, height: 330, x: [0, 192], y: [0, 40], xlabel: "x (left image column)", ylabel: "disparity d", font: 17, grid: false, margin: { l: 60, r: 10, t: 8, b: 52 } });
  const sliceNote = htmlEl("div", "qb-note", "", sliceDiv);

  const side = htmlEl("div", "sm-side", "", host);
  const cPlot = new Plot(side, { width: 290, height: 220, x: [0, 40], y: [0, 3], xlabel: "disparity d", ylabel: "cost C(d)", font: 17, margin: { l: 56, r: 10, t: 8, b: 50 } });
  const cLine = cPlot.line([], [], { color: "var(--qb-c1)", width: 2.5 });
  const cAgg = cPlot.line([], [], { color: "var(--qb-c4)", width: 2, dash: "6 3" });
  const cPar = cPlot.line([], [], { color: "var(--qb-c2)", width: 2 });
  const cTrue = cPlot.vline(0, { color: "var(--qb-c3)", width: 2.5, text: "true" });
  const cMin = cPlot.scatter([], [], { color: "var(--qb-c2)", r: 6 });
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s, { tab: st.tab }); sync(); recompute("scene"); } })));
  const sceneRow = htmlEl("div", "sm-row", "", panel);
  htmlEl("span", "sm-lab", "scene", sceneRow);
  const scBtns = (["dots", "mixed", "gain"] as SceneId[]).map(s => button(sceneRow, s === "dots" ? "random dots" : s === "mixed" ? "wall, fence, post" : "gain change", () => { st.scene = s; recompute("scene"); }));
  const costRow = htmlEl("div", "sm-row", "", panel);
  htmlEl("span", "sm-lab", "cost", costRow);
  const cBtns = (["sad", "ncc", "census"] as CostKind[]).map(c => button(costRow, c.toUpperCase().replace("CENSUS", "census"), () => { st.cost = c; recompute("cost"); }));
  const aggRow = htmlEl("div", "sm-row", "", panel);
  htmlEl("span", "sm-lab", "aggregation", aggRow);
  const aBtns = (["wta", "dp", "sgm4", "sgm8"] as Agg[]).map(a => button(aggRow, a === "wta" ? "none" : a === "dp" ? "scanline DP" : a === "sgm4" ? "SGM 4" : "SGM 8", () => { st.agg = a; recompute("agg"); }));
  const rS = slider(panel, "window", 0, 7, 1, st.r, v => { st.r = v; recompute("cost"); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const p1S = slider(panel, "P1 (±1 step)", 0, 1, 0.05, st.P1, v => { st.P1 = v; recompute("agg"); });
  const p2S = slider(panel, "P2 (jump)", 0, 4, 0.1, st.P2, v => { st.P2 = v; recompute("agg"); });
  const yS = slider(panel, "row y", 0, 143, 1, st.py, v => { st.py = v; draw(); });
  const nS = slider(panel, "noise", 0, 15, 1, st.noise, v => { st.noise = v; recompute("scene"); });
  const lrC = checkbox(panel, "left–right check", st.lr, v => { st.lr = v; recompute("agg"); });
  const subC = checkbox(panel, "sub-pixel (parabola)", st.sub, v => { st.sub = v; recompute("final"); });
  button(panel, "reset", () => { Object.assign(st, start()); pBtns.forEach(b => b.classList.remove("active")); sync(); layout(); recompute("scene"); });
  function sync(): void { yS.set(st.py); rS.set(st.r); p1S.set(st.P1); p2S.set(st.P2); nS.set(st.noise); lrC.checked = st.lr; subC.checked = st.sub; }
  function layout(): void {
    grid.style.display = st.tab === "images" ? "" : "none";
    sliceDiv.style.display = st.tab === "slice" ? "" : "none";
    tabI.classList.toggle("active", st.tab === "images"); tabS.classList.toggle("active", st.tab === "slice");
  }
  const pick = (x: number, y: number) => { st.px = x; st.py = y; yS.set(y); draw(); };
  vL.onPick(pick); vD.onPick(pick); vE.onPick(pick);

  // ---- computation (cached stages; coalesced into one animation frame) ----
  let sc: StereoScene, C: Float32Array, CR: Float32Array | null = null, Agg: Float32Array | null = null, dInt: Int32Array, dFinal: Float32Array, ms = 0;
  let pending: "scene" | "cost" | "agg" | "final" | null = null, raf = 0;
  const order = { scene: 0, cost: 1, agg: 2, final: 3 };
  function recompute(stage: "scene" | "cost" | "agg" | "final"): void {
    if (!pending || order[stage] < order[pending]) pending = stage;
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; const s = pending!; pending = null; compute(s); draw(); });
  }
  function aggregate(V: Float32Array): { d: Int32Array; A: Float32Array | null } {
    const n = sc.w * sc.h, D = sc.maxD;
    if (st.agg === "wta") return { d: wta(V, n, D), A: null };
    if (st.agg === "dp") return { d: scanlineDP(V, sc.w, sc.h, D, st.P1, st.P2).disp, A: null };
    const A = sgm(V, sc.w, sc.h, D, st.P1, st.P2, st.agg === "sgm4" ? 4 : 8);
    return { d: wta(A, n, D), A };
  }
  function compute(stage: "scene" | "cost" | "agg" | "final"): void {
    const t0 = performance.now();
    if (stage === "scene") sc = makeScene(st.scene, 192, 144, st.noise, 7);
    const n = sc.w * sc.h, D = sc.maxD, k = COST_SCALE[st.cost];
    if (order[stage] <= 1) {
      C = costVolume(sc.left, sc.right, sc.w, sc.h, D, st.cost, st.r);
      for (let i = 0; i < C.length; i++) C[i] *= k;
      CR = null;
    }
    if (order[stage] <= 2) {
      const a = aggregate(C);
      dInt = a.d; Agg = a.A;
      if (st.lr) {
        if (!CR) { CR = costVolumeRight(sc.left, sc.right, sc.w, sc.h, D, st.cost, st.r); for (let i = 0; i < CR.length; i++) CR[i] *= k; }
        const dr = aggregate(CR).d;
        const chk = leftRightCheck(dInt, dr, sc.w, sc.h, 1);
        dInt = Int32Array.from(chk);
      }
    }
    dFinal = st.sub ? subpixel(Agg ?? C, dInt, n, D) : Float32Array.from(dInt);
    ms = performance.now() - t0;
  }

  // ---- drawing ----
  function draw(): void {
    layout();
    scBtns.forEach((b, i) => b.classList.toggle("active", ["dots", "mixed", "gain"][i] === st.scene));
    cBtns.forEach((b, i) => b.classList.toggle("active", ["sad", "ncc", "census"][i] === st.cost));
    aBtns.forEach((b, i) => b.classList.toggle("active", ["wta", "dp", "sgm4", "sgm8"][i] === st.agg));
    const { w, h, maxD: D } = sc, n = w * h;
    const greyOpts = { mode: "clip" as const, lo: 0, hi: 255 };
    vL.grey({ width: w, height: h, data: sc.left }, greyOpts);
    vR.grey({ width: w, height: h, data: sc.right }, greyOpts);
    // disparity map
    const px = new Uint8ClampedArray(4 * n), ex = new Uint8ClampedArray(4 * n);
    for (let i = 0; i < n; i++) {
      const d = dFinal[i];
      const c = d < 0 ? GREY_INVALID : viridis(Math.max(0, Math.min(1, d / D))).map(v => 255 * v);
      px.set([c[0], c[1], c[2], 255], 4 * i);
      const x = i % w, g = sc.left[i] * 0.55 + 60;
      let e: number[];
      if (x < D) e = [g * 0.6, g * 0.6, g * 0.6];
      else if (sc.occluded[i]) e = d < 0 ? [90, 140, 250] : [40, 70, 180];
      else if (d < 0) e = [200, 200, 200];
      else e = Math.abs(d - sc.disp[i]) > 1 ? [232, 64, 42] : [g, g, g];
      ex.set([e[0], e[1], e[2], 255], 4 * i);
    }
    vD.rgba(px, w, h); vE.rgba(ex, w, h);
    const bp = badPixelRate(dFinal, sc);
    vD.setCaption("disparity (grey: invalid)");
    vE.setCaption(`errors > 1 px: ${(100 * bp.rate).toFixed(1)} %`);
    // overlays
    const X = st.px, Y = st.py, i0 = Y * w + X, dSel = dInt[i0];
    [vL, vR, vD, vE].forEach(v => v.clear());
    vL.box(X, Y, st.r, "#ffd400", 2.5);
    vR.add("line", { x1: 0, y1: Y + 0.5, x2: w, y2: Y + 0.5, stroke: "#22d3ee", "stroke-width": 1.5, "stroke-dasharray": "4 3" });
    if (dSel >= 0) vR.box(X - dSel, Y, st.r, "#ffd400", 2.5);
    vR.box(X - Math.round(sc.disp[i0]), Y, 0, "#2ca02c", 2.5);
    vD.box(X, Y, 0, "#ff4d4d", 2.5); vE.box(X, Y, 0, "#ffd400", 2.5);
    vL.add("line", { x1: 0, y1: Y + 0.5, x2: w, y2: Y + 0.5, stroke: "#22d3ee", "stroke-width": 1.5, "stroke-dasharray": "4 3" });
    // cost curve
    const ds = Array.from({ length: D }, (_, d) => d), cs = ds.map(d => C[i0 * D + d]);
    let as: number[] = [];
    if (Agg) { const nd = st.agg === "sgm4" ? 4 : 8; as = ds.map(d => Agg![i0 * D + d] / nd); }
    const top = Math.max(...cs, ...(as.length ? as : [0])) * 1.08 || 1;
    cPlot.setDomain([0, D - 1], [0, top]);
    cLine.set(ds, cs); cAgg.set(Agg ? ds : [], as);
    cTrue.set(sc.disp[i0]);
    const curve = Agg ? as : cs;
    cMin.set(dSel >= 0 ? [dSel] : [], dSel >= 0 ? [curve[dSel]] : []);
    if (st.sub && dSel > 0 && dSel < D - 1) {
      const cm = curve[dSel - 1], c0 = curve[dSel], cp = curve[dSel + 1], a = (cm + cp - 2 * c0) / 2, b = (cp - cm) / 2;
      const xs = Array.from({ length: 21 }, (_, k) => dSel - 1.5 + 3 * k / 20);
      cPar.set(xs, xs.map(x => c0 + b * (x - dSel) + a * (x - dSel) ** 2));
    } else cPar.set([], []);
    const truth = sc.disp[i0], dF = dFinal[i0];
    let gs = 0, gc = 0, occN = 0, occInv = 0, inv = 0;
    for (let i = 0; i < n; i++) {
      if (i % w < D) continue;
      if (dFinal[i] < 0) inv++;
      if (sc.occluded[i]) { occN++; if (dFinal[i] < 0) occInv++; continue; }
      if (sc.layer[i] === 0 && dFinal[i] >= 0) { const e = Math.abs(dFinal[i] - sc.disp[i]); if (e < 1) { gs += e; gc++; } }
    }
    ro.textContent = `pixel (${X}, ${Y}): true d = ${truth.toFixed(2)}${sc.occluded[i0] ? " (occluded in the right view)" : ""}\n` +
      `found d = ${dF < 0 ? "invalid" : st.sub ? dF.toFixed(2) : dF.toFixed(0)}` + (st.sub && dSel > 0 && dSel < D - 1 && dF >= 0 ? `  (parabola offset ${parabolaVertex(curve[dSel - 1], curve[dSel], curve[dSel + 1]).toFixed(2)})` : "") +
      `\nbad pixels: ${bp.bad} of ${bp.n} (${(100 * bp.rate).toFixed(1)} %)` +
      (st.lr ? `\ninvalid: ${inv}, of them occluded ${occInv} (of ${occN} occluded)` : "") +
      (st.scene === "mixed" ? `\nground: mean |error| of good pixels ${gc ? (gs / gc).toFixed(3) : "–"}` : "") +
      (Agg ? "\nsolid: raw cost, dashed: aggregated / directions" : "") + `\nupdate: ${ms.toFixed(0)} ms`;
    if (st.tab === "slice") drawSlice();
  }

  function drawSlice(): void {
    const { w, maxD: D } = sc, Y = st.py;
    const vals = new Float32Array(w * D);
    for (let d = 0; d < D; d++) for (let x = 0; x < w; x++) vals[d * w + x] = C[(Y * w + x) * D + d];
    slicePlot.setDomain([0, w], [0, D]);
    slicePlot.clear();
    slicePlot.heatmap(vals, w, D, { map: viridis, flipY: false });
    const sliceTrue = slicePlot.line([], [], { color: "#fff", width: 2.5, dash: "5 3", label: "truth" });
    const sliceWTA = slicePlot.scatter([], [], { color: "#ff4d4d", r: 3, label: "winner-take-all" });
    const sliceDP = slicePlot.line([], [], { color: "#ffd400", width: 2.5, label: "scanline DP" });
    slicePlot.legend(undefined, "tr");
    const xs = Array.from({ length: w }, (_, x) => x + 0.5);
    const rowC = C.subarray(Y * w * D, (Y + 1) * w * D);
    const wt = wta(rowC, w, D), dp = scanlineDP(rowC, w, 1, D, st.P1, st.P2).disp;
    sliceWTA.set(xs, Array.from(wt, d => d + 0.5));
    sliceDP.set(xs, Array.from(dp, d => d + 0.5));
    sliceTrue.set(xs, xs.map((_, x) => sc.disp[Y * w + x] + 0.5));
    sliceNote.textContent = `row y = ${Y} (row slider, or click a pixel in the first tab); dark = low cost. DP: P1 = ${st.P1}, P2 = ${st.P2}.`;
  }

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  compute("scene");
  draw();
}

autoMount(NAME, mount);
