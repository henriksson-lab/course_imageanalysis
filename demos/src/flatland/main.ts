// Flatland multi-view stereo (lib/flatland.ts): a 2D world seen by 2–5 1D cameras whose
// "photographs" are pixel rows (shown on each camera and as strips). Plane (line) sweep from
// the reference camera: for a chosen pixel, depth hypotheses along its ray are projected into
// the other cameras; photo-consistency = variance of the sampled values across the views
// (mean over a window of ±w pixels). Plots: the consistency curve of the chosen pixel (true
// depth marked) and the depth of every reference pixel (estimate vs truth).
// Only data-mode="sweep" is built so far (the grid radiance field and the 2D splats of the
// optional radiance-fields unit are planned as further modes).
//
// <div data-demo="flatland" data-mode="sweep" data-preset="textured|textureless|occluded"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Cam1D, P, makeScene, ring, render1d, sweepPixel, pointAtDepth, project1d, Scene, noise1d } from "../lib/flatland";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import { FlatView, injectStyle, CAM_COLORS } from "../triangulate/flatview";

const NAME = "flatland";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .8em; }
${SEL} .fl-main { flex: 0 1 440px; min-width: 260px; }
${SEL} .fl-side { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
${SEL} .qb-readout { font-size: 15px; }
${SEL} .fl-strips canvas { width: 100%; height: 16px; display: block; image-rendering: pixelated; margin-bottom: 3px; cursor: crosshair; border-radius: 2px; }
${SEL} .fl-strips .lab { font-size: 15px; }
.reveal ${SEL} { font-size: 18px; }
.reveal ${SEL} .fl-main { flex: 0 0 430px; }
.reveal ${SEL} .fl-side { flex: 0 0 320px; }
.reveal ${SEL} .qb-panel { gap: .2em; }
.reveal ${SEL} .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .25em .45em; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

const F = 150, NPIX = 120, DEPTHS = Array.from({ length: 90 }, (_, k) => 1.6 + k * 0.035);
type SceneId = "textured" | "textureless" | "occluded";
interface State { scene: SceneId; views: number; win: number; noise: number; u: number; showAll: boolean }
const DEFAULT: State = { scene: "textured", views: 3, win: 3, noise: 0, u: 60, showAll: true };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "textured", label: "textured object", s: { scene: "textured", u: 60 } },
  { id: "textureless", label: "a textureless face", s: { scene: "textureless", u: 66 } },
  { id: "occluded", label: "occluded in one view", s: { scene: "occluded", u: 60 } },
];

const OBJECT: P[] = [[-0.9, -0.15], [-0.2, -0.65], [0.75, -0.35], [0.85, 0.5], [-0.6, 0.55]];
function buildScene(id: SceneId): Scene {
  const polys: P[][] = [OBJECT];
  if (id === "occluded") polys.push([[0.6, -1.8], [0.85, -1.8], [0.85, -1.55], [0.6, -1.55]]);
  const flat = () => 0.62;
  return makeScene(polys, (k, e) => (id === "textureless" && k === 0 && e === 1 ? flat : k === 1 ? (s => 0.2 + 0.6 * noise1d(77, 0.03)(s)) : null), 3);
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const main = htmlEl("div", "fl-main", "", host);
  const view = new FlatView(main, 440, 400, -2.6, 2.6, -3.5, 1.2);
  const strips = htmlEl("div", "fl-strips", "", main);
  const side = htmlEl("div", "fl-side", "", host);
  const cPlot = new Plot(side, { width: 330, height: 200, x: [DEPTHS[0], DEPTHS[DEPTHS.length - 1]], y: [0, 0.1], xlabel: "depth hypothesis Z", ylabel: "variance", font: 17, margin: { l: 66, r: 10, t: 8, b: 50 } });
  const cLine = cPlot.line([], [], { color: "var(--qb-c1)", width: 2.5 });
  const cTrue = cPlot.vline(2, { color: "var(--qb-c3)", width: 2.5, text: "true" });
  const cBest = cPlot.scatter([], [], { color: "var(--qb-c2)", r: 6 });
  const dPlot = new Plot(side, { width: 330, height: 200, x: [0, NPIX - 1], y: [DEPTHS[0], DEPTHS[DEPTHS.length - 1]], xlabel: "reference pixel u", ylabel: "depth Z", font: 17, margin: { l: 66, r: 10, t: 8, b: 50 } });
  const dTrue = dPlot.line([], [], { color: "var(--qb-c3)", width: 3, label: "true" });
  const dEst = dPlot.scatter([], [], { color: "var(--qb-c2)", r: 3, label: "sweep" });
  const dSel = dPlot.vline(60, { color: "#888", width: 1.5, dash: "4 3" });
  dPlot.legend(undefined, "tr");
  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  const uS = slider(panel, "pixel u (reference)", 0, NPIX - 1, 1, st.u, v => { st.u = v; draw(); });
  const vS = slider(panel, "views", 2, 5, 1, st.views, v => { st.views = v; rebuild(); });
  const wS = slider(panel, "window ±w", 0, 5, 1, st.win, v => { st.win = v; draw(true); });
  const nS = slider(panel, "noise", 0, 0.1, 0.01, st.noise, v => { st.noise = v; rebuild(); });
  const allC = checkbox(panel, "depth of every pixel", st.showAll, v => { st.showAll = v; draw(true); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); pBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { uS.set(st.u); vS.set(st.views); wS.set(st.win); nS.set(st.noise); allC.checked = st.showAll; }

  let scene: Scene, cams: Cam1D[] = [], imgs: Float32Array[] = [], truthDepth: Float32Array = new Float32Array(0), all: number[] = [];
  let REF = 1;
  function rebuild(): void {
    scene = buildScene(st.scene);
    // the reference camera (2) stays in place; extra views are added on both sides
    const arcs = [0, -0.42, 0.42, -0.84, 0.84].slice(0, st.views).sort((a, b) => a - b);
    REF = arcs.indexOf(0);
    cams = arcs.map(a => ring(1, 3.1, F, NPIX, [0, -0.05], -Math.PI / 2 + a, 0)[0]);
    const r = rng(17);
    imgs = cams.map(c => { const v = render1d(scene, c).values; return Float32Array.from(v, x => Math.max(0, Math.min(1, x + st.noise * r.norm()))); });
    truthDepth = render1d(scene, cams[REF]).depth;
    draw(true);
  }

  function drawStrips(): void {
    strips.replaceChildren();
    cams.forEach((c, j) => {
      const lab = htmlEl("div", "lab", `camera ${j + 1}${j === REF ? " (reference)" : ""}`, strips);
      lab.style.color = CAM_COLORS[j];
      const cv = htmlEl("canvas", "", "", strips);
      cv.width = NPIX; cv.height = 1;
      const ctx = cv.getContext("2d")!, id = ctx.createImageData(NPIX, 1);
      for (let u = 0; u < NPIX; u++) { const g = Math.round(255 * imgs[j][u]); id.data.set([g, g, g, 255], 4 * u); }
      ctx.putImageData(id, 0, 0);
      if (j === REF) cv.addEventListener("pointerdown", ev => { const b = cv.getBoundingClientRect(); st.u = Math.max(0, Math.min(NPIX - 1, Math.floor((ev.clientX - b.left) / b.width * NPIX))); uS.set(st.u); draw(); });
    });
  }

  function draw(recomputeAll = false): void {
    view.clear();
    view.poly(OBJECT, "#e8e2d0", "#555", 0.6, 1.5);
    scene.edges.forEach(e => {
      // textured edges drawn as their texture
      const L = Math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1]), n = Math.max(2, Math.round(L / 0.03));
      for (let k = 0; k < n; k++) {
        const t0 = k / n, t1 = (k + 1) / n, v = Math.round(255 * e.tex(e.s0 + L * (t0 + t1) / 2));
        view.line([e.a[0] + (e.b[0] - e.a[0]) * t0, e.a[1] + (e.b[1] - e.a[1]) * t0], [e.a[0] + (e.b[0] - e.a[0]) * t1, e.a[1] + (e.b[1] - e.a[1]) * t1], `rgb(${v},${v},${v})`, 6).setAttribute("stroke-linecap", "butt");
      }
    });
    if (st.scene === "occluded") view.poly(scene.polys[1], "#b9b9b9", "#555", 0.5, 1);
    const ref = cams[REF];
    const sw = sweepPixel(cams, imgs, REF, st.u, DEPTHS, st.win);
    // hypotheses along the ray
    const pA = pointAtDepth(ref, st.u, DEPTHS[0]), pB = pointAtDepth(ref, st.u, DEPTHS[DEPTHS.length - 1]);
    view.line([ref.x, ref.y], pB, CAM_COLORS[REF], 2);
    view.line(pA, pB, "#666", 5, "1 6").setAttribute("stroke-linecap", "round");
    const best = sw.best >= 0 ? pointAtDepth(ref, st.u, DEPTHS[sw.best]) : null;
    if (best) cams.forEach((c, j) => {
      if (j === REF) return;
      const q = project1d(c, best);
      if (q.Z > 0 && q.u >= 0 && q.u <= NPIX - 1) view.line([c.x, c.y], best, CAM_COLORS[j], 1.5, "5 4");
    });
    cams.forEach((c, j) => view.camera(c, CAM_COLORS[j], { len: 0.5, values: imgs[j], label: `${j + 1}` }));
    view.dot(view.imagePoint(ref, st.u, 0.5), 5, "#e8402a", "#fff");
    if (best) view.dot(best, 6, "#e8402a", "#fff");
    // consistency curve
    const vals = sw.cost.filter(Number.isFinite);
    cPlot.setDomain(undefined, [0, Math.max(0.005, ...vals) * 1.1]);
    cLine.set(DEPTHS, sw.cost);
    const tz = truthDepth[st.u];
    cTrue.set(Number.isFinite(tz) ? tz : -10);
    cBest.set(sw.best >= 0 ? [DEPTHS[sw.best]] : [], sw.best >= 0 ? [sw.cost[sw.best]] : []);
    // all pixels
    if (recomputeAll) {
      all = [];
      for (let u = 0; u < NPIX; u++) { const s = sweepPixel(cams, imgs, REF, u, DEPTHS, st.win); all.push(s.best >= 0 ? DEPTHS[s.best] : NaN); }
      drawStrips();
    }
    const us = Array.from({ length: NPIX }, (_, u) => u);
    dTrue.set(us, Array.from(truthDepth, z => (Number.isFinite(z) ? z : NaN)));
    dEst.set(st.showAll ? us : [], st.showAll ? all : []);
    dSel.set(st.u);
    let bad = 0, n = 0;
    all.forEach((z, u) => { if (Number.isFinite(truthDepth[u])) { n++; if (!(Math.abs(z - truthDepth[u]) <= 0.1)) bad++; } });
    ro.textContent = `pixel ${st.u}: best depth ${sw.best >= 0 ? DEPTHS[sw.best].toFixed(2) : "–"}, true ${Number.isFinite(tz) ? tz.toFixed(2) : "background"}\n` +
      `all pixels: ${bad} of ${n} off by more than 0.1`;
  }

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  rebuild();
}

autoMount(NAME, mount);
