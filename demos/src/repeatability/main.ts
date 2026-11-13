// Repeatability of keypoint detectors. View 1: a synthetic scene (or the micrograph); view 2:
// the same scene under a known similarity (rotation, zoom), optional perspective tilt and
// lighting (gain, offset, gamma), blur and noise. Detectors: Harris at one σ, multi-scale
// Harris, DoG, FAST. Keypoints of view 1 found again in view 2 (within ε px, at a matching
// scale) are green, the others red. The plot sweeps zoom or angle for the current detector
// (computed in the background, cached) with the current value marked. Click a keypoint of
// view 1: its predicted position in view 2 and the nearest detection.
//
// <div data-demo="repeatability" data-preset="rotation|zoom|survive|tilt|light|blur|cell|fast"
//      data-tab="zoom|angle" data-scene="scene|cell"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { repeatability, Keypoint } from "../lib/keypoints";
import { ImgView, injectStyle } from "../canny/imgview";
import { CELL_CREDIT } from "../lib/cell";
import { getScene, makeViews, detect, Detector, DETECTOR_NAMES, drawKp, scheduler, SceneKind, VIEW, GREEN, RED, YELLOW, BLUE, CYAN, COMMON_CSS, ViewPairResult } from "./kpcommon";
import { Grey } from "../lib/imageops";
import { apply } from "../lib/transforms2d";

const NAME = "repeatability";
const STYLE = COMMON_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .rp-views { flex: 0 1 540px; min-width: 300px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rp-pair { display: flex; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .rp-pair .ev-fig { flex: 1 1 0; }
.qb-demo[data-demo="${NAME}"] .rp-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 270px; max-width: 330px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .rp-key { display: flex; flex-wrap: wrap; gap: .1em .8em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .rp-tabs { display: flex; gap: .3em; flex-wrap: wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .rp-views { flex: 0 0 530px; }
.reveal .qb-demo[data-demo="${NAME}"] .rp-mid { flex: 0 0 265px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 315px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .rp-key { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 900px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

interface State {
  scene: SceneKind; det: Detector; angle: number; zoom: number; tilt: number;
  gain: number; offset: number; gamma: number; blur: number; noise: number;
  n: number; eps: number; tab: "zoom" | "angle"; sel: number;
}
const DEFAULT: State = { scene: "scene", det: "dog", angle: 0, zoom: 1.6, tilt: 0, gain: 1, offset: 0, gamma: 1, blur: 0, noise: 2, n: 200, eps: 2.5, tab: "zoom", sel: -1 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "rotation", label: "rotation: all detectors fine", s: { angle: 40, zoom: 1, tab: "angle" } },
  { id: "zoom", label: "zoom 2×: Harris (one σ) loses the points", s: { det: "harris", zoom: 2.2, tab: "zoom" } },
  { id: "survive", label: "multi-scale and DoG survive", s: { det: "dog", zoom: 2.2, tab: "zoom" } },
  { id: "tilt", label: "tilt 50°: DoG degrades", s: { zoom: 1, tilt: 50, tab: "angle" } },
  { id: "light", label: "gain and offset: no effect", s: { zoom: 1, angle: 20, gain: 1.5, offset: -30, tab: "angle" } },
  { id: "blur", label: "blur", s: { zoom: 1, angle: 20, blur: 2, tab: "angle" } },
  { id: "cell", label: "the micrograph", s: { scene: "cell", zoom: 1.6, angle: 30, noise: 0 } },
  { id: "fast", label: "FAST is fast", s: { det: "fast", zoom: 1, angle: 30, tab: "angle" } },
];
const ZOOMS = [0.4, 0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3];
const ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180];
const DET_COLOR: Record<Detector, string> = { harris: "#d62728", harrisms: "#9467bd", dog: "#1f77b4", fast: "#ff7f0e" };

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.scene === "cell" ? { scene: "cell" as SceneKind, noise: 0 } : {}), ...(host.dataset.tab === "angle" ? { tab: "angle" as const } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const left = htmlEl("div", "rp-views", "", host);
  const pair = htmlEl("div", "rp-pair", "", left);
  const view1 = new ImgView(pair, "view 1"), view2 = new ImgView(pair, "view 2");
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, left);
  const busy = htmlEl("div", "kp-busy", "", left);

  const mid = htmlEl("div", "rp-mid", "", host);
  const tabs = htmlEl("div", "rp-tabs", "", mid);
  const tabBtns = (["zoom", "angle"] as const).map(t => button(tabs, t === "zoom" ? "against zoom" : "against angle", () => { st.tab = t; draw(); sweep(); }));
  const plot = new Plot(mid, { width: 300, height: 250, x: [0.4, 3], y: [0, 1], xlog: true, font: 18, xlabel: "zoom of view 2", ylabel: "repeatability", margin: { l: 58, r: 10, t: 10, b: 50 }, xticks: [0.5, 1, 2, 3], xfmt: v => String(v) });
  const key = htmlEl("div", "rp-key", "", mid);
  for (const d of Object.keys(DETECTOR_NAMES) as Detector[]) { const s = htmlEl("span", "", "", key); s.innerHTML = `<b style="color:${DET_COLOR[d]}">━</b> ${DETECTOR_NAMES[d]}`; }
  const curves = new Map<Detector, ReturnType<Plot["line"]>>();
  const cur = plot.vline(1, { color: "#888", width: 2, dash: "5 4" });
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); load(); } })));
  const detBtns = presets(panel, (Object.keys(DETECTOR_NAMES) as Detector[]).map(d => ({ label: DETECTOR_NAMES[d], apply: () => { st.det = d; st.sel = -1; recompute(); } })));
  const sceneBtns = presets(panel, [
    { label: "synthetic scene", apply: () => { st.scene = "scene"; load(); } },
    { label: "micrograph", apply: () => { st.scene = "cell"; load(); } },
  ]);
  const zS = slider(panel, "zoom", -100, 100, 1, 0, v => { st.zoom = zOf(v); recompute(); }, v => zOf(v).toFixed(2));
  const aS = slider(panel, "rotation°", -180, 180, 5, st.angle, v => { st.angle = v; recompute(); });
  const tS = slider(panel, "tilt°", 0, 70, 5, st.tilt, v => { st.tilt = v; recompute(); });
  const gS = slider(panel, "gain", 0.5, 2, 0.05, st.gain, v => { st.gain = v; recompute(); });
  const oS = slider(panel, "offset", -60, 60, 5, st.offset, v => { st.offset = v; recompute(); });
  const gmS = slider(panel, "gamma", 0.4, 2.5, 0.05, st.gamma, v => { st.gamma = v; recompute(); });
  const bS = slider(panel, "blur σ", 0, 4, 0.25, st.blur, v => { st.blur = v; recompute(); });
  const nS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; recompute(); });
  const kS = slider(panel, "keypoints", 50, 400, 25, st.n, v => { st.n = v; recompute(); });
  const eS = slider(panel, "ε (px)", 1, 6, 0.5, st.eps, v => { st.eps = v; draw(); sweep(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); load(); });

  function zOf(v: number): number { return Math.round(Math.pow(3, v / 100) * 100) / 100; }
  function vOf(z: number): number { return Math.round(100 * Math.log(z) / Math.log(3)); }
  function sync(): void {
    zS.set(vOf(st.zoom)); aS.set(st.angle); tS.set(st.tilt); gS.set(st.gain); oS.set(st.offset); gmS.set(st.gamma);
    bS.set(st.blur); nS.set(st.noise); kS.set(st.n); eS.set(st.eps);
  }

  let scene: Grey | null = null, views: ViewPairResult | null = null;
  let k1: Keypoint[] = [], k2: Keypoint[] = [], ms2 = 0;
  const run = scheduler(on => { busy.textContent = on ? "computing…" : ""; });

  function load(): void {
    getScene(st.scene).then(g => { scene = g; recompute(); });
  }
  const light = () => ({ gain: st.gain, offset: st.offset, gamma: st.gamma, blur: st.blur, noise: st.noise });
  function repOf(v: ViewPairResult, det: Detector, zoom: number): ReturnType<typeof repeatability> {
    const a = detect(v.v1, det, st.n), b = detect(v.v2, det, st.n);
    return repeatability(a.kps, b.kps, v.map12, v.map21, { w1: VIEW, h1: VIEW, w2: VIEW, h2: VIEW, eps: st.eps, scale: zoom, scaleTol: det === "harris" || det === "fast" ? Infinity : Math.log(1.5) });
  }
  function recompute(): void {
    run(() => {
      if (!scene) return;
      views = makeViews(st.scene, scene, { zoom: st.zoom, angle: st.angle, tilt: st.tilt }, light());
      const a = detect(views.v1, st.det, st.n), b = detect(views.v2, st.det, st.n);
      k1 = a.kps; k2 = b.kps; ms2 = b.ms;
      draw();
      sweep();
    });
  }

  // ---- the sweep (background, cached) ----
  const cache = new Map<string, (number | null)[]>();
  let token = 0;
  const keyOf = (det: Detector) => `${st.scene}|${det}|${st.tab}|${st.n}|${st.eps}|${st.tilt}|${st.gain}|${st.offset}|${st.gamma}|${st.blur}|${st.noise}|${st.tab === "zoom" ? st.angle : st.zoom}`;
  function sweep(): void {
    const my = ++token, det = st.det, key = keyOf(det), xs = st.tab === "zoom" ? ZOOMS : ANGLES;
    let ys = cache.get(key);
    if (!ys) { ys = xs.map(() => null); cache.set(key, ys); }
    drawCurves();
    const step = (i: number) => {
      if (my !== token || !scene) return;
      while (i < xs.length && ys![i] !== null) i++;
      if (i >= xs.length) { drawCurves(); return; }
      const spec = st.tab === "zoom" ? { zoom: xs[i], angle: st.angle, tilt: st.tilt } : { zoom: st.zoom, angle: xs[i], tilt: st.tilt };
      const v = makeViews(st.scene, scene, spec, light());
      ys![i] = repOf(v, det, spec.zoom).rep;
      drawCurves();
      setTimeout(() => step(i + 1), 0);
    };
    setTimeout(() => step(0), 30);
  }
  function drawCurves(): void {
    const xs = st.tab === "zoom" ? ZOOMS : ANGLES;
    const isZoom = st.tab === "zoom";
    if (plot.opts.xlog !== isZoom) {
      plot.opts.xlog = isZoom; plot.opts.xticks = isZoom ? [0.5, 1, 2, 3] : [0, 45, 90, 135, 180];
      plot.opts.xlabel = isZoom ? "zoom of view 2" : "rotation of view 2 (°)";
      plot.setDomain(isZoom ? [0.4, 3] : [0, 180]);
    }
    for (const d of Object.keys(DETECTOR_NAMES) as Detector[]) {
      const ys = cache.get(keyOf(d));
      const px: number[] = [], py: number[] = [];
      if (ys) ys.forEach((y, i) => { if (y !== null) { px.push(xs[i]); py.push(y); } });
      let c = curves.get(d);
      if (!c) { c = plot.line([], [], { color: DET_COLOR[d], width: 3, label: DETECTOR_NAMES[d] }); curves.set(d, c); }
      c.style.width = d === st.det ? 4 : 2; c.style.opacity = d === st.det ? 1 : 0.45;
      c.set(px, py);
    }
    cur.set(st.tab === "zoom" ? st.zoom : Math.abs(st.angle));
  }

  view1.onPick((x, y) => {
    let best = -1, bd = 12;
    k1.forEach((k, i) => { const d = Math.hypot(k.x - x, k.y - y); if (d < bd) { bd = d; best = i; } });
    st.sel = best; draw();
  }, false);

  function draw(): void {
    if (!views) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "zoom")));
    detBtns.forEach((b, i) => b.classList.toggle("active", (Object.keys(DETECTOR_NAMES) as Detector[])[i] === st.det));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.scene === "cell")));
    credit.style.display = st.scene === "cell" ? "" : "none";
    view1.grey(views.v1, { mode: "clip", lo: 0, hi: 255 }); view2.grey(views.v2, { mode: "clip", lo: 0, hi: 255 });
    view1.clear(); view2.clear();
    const r = repOf(views, st.det, st.zoom);
    // common region: view-2 border mapped into view 1
    const corners = [[0, 0], [VIEW, 0], [VIEW, VIEW], [0, VIEW]].map(([x, y]) => views!.map21(x - 0.5, y - 0.5));
    view1.add("polygon", { points: corners.map(p => `${p[0] + 0.5},${p[1] + 0.5}`).join(" "), fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "5 4" });
    k1.forEach((k, i) => {
      const part = r.partner[i];
      if (part === -1) drawKp(view1, k, "rgba(200,200,200,.7)", { tick: false, width: 1 });
      else drawKp(view1, k, part >= 0 ? GREEN : RED, { tick: false, fill: part >= 0 ? "rgba(44,160,44,.35)" : "rgba(232,64,42,.25)", width: 2 });
    });
    k2.forEach(k => drawKp(view2, k, BLUE, { tick: false, width: 1.5 }));
    let sel = "";
    if (st.sel >= 0 && st.sel < k1.length) {
      const k = k1[st.sel], [px, py] = views.map12(k.x, k.y);
      drawKp(view1, k, YELLOW, { tick: false, width: 3 });
      view2.add("path", { d: `M${px + 0.5 - 6},${py + 0.5}h12M${px + 0.5},${py + 0.5 - 6}v12`, stroke: YELLOW, "stroke-width": 3 });
      let nb = -1, nd = Infinity;
      k2.forEach((q, j) => { const d = Math.hypot(q.x - px, q.y - py); if (d < nd) { nd = d; nb = j; } });
      if (nb >= 0) drawKp(view2, k2[nb], CYAN, { tick: false, width: 3 });
      sel = `\nselected: (${k.x.toFixed(1)}, ${k.y.toFixed(1)}), σ ${k.sigma.toFixed(2)}\n → predicted (${px.toFixed(1)}, ${py.toFixed(1)}), σ×zoom ${(k.sigma * st.zoom).toFixed(2)}` +
        (nb >= 0 ? `\n nearest: ${nd.toFixed(1)} px away, σ ${k2[nb].sigma.toFixed(2)}` : "");
    }
    void apply;
    ro.textContent = `${DETECTOR_NAMES[st.det]}: ${k1.length} / ${k2.length} points\nin the common region: ${r.n1} / ${r.n2}\nrepeated: ${r.repeated} → repeatability ${(100 * r.rep).toFixed(0)} %\ndetection in view 2: ${ms2.toFixed(0)} ms` + sel;
    drawCurves();
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  load();
}

autoMount(NAME, mount);
