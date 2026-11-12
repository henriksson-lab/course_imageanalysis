// Circle detection with the Hough transform. Left: the image with its edge points (Canny,
// lib/edges.ts) and the detected circles. Middle: one slice of the (a, b, r) accumulator,
// at the radius on the slider. Right: the maximum over all radii. Values are votes divided
// by 2 pi r: the fraction of a full circumference that voted. Click an edge point: its
// votes (a ring per radius, or one or two dots with gradient voting) appear on the slice.
// Click the slice: the votes at that centre are plotted against the radius (a radius
// histogram; for concentric rings with a known centre this is all one needs).
//
// <div data-demo="houghcircle" data-preset="one|wrong|scan|grad|false|ellipse|plate|rings|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, Slider } from "../lib/ui";
import { Grey, fromArray } from "../lib/imageops";
import { canny, edgeList, EdgePoint } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { Plot } from "../lib/plot";
import { houghCircles, normaliseByCircumference, maxOverSlices, peaks3, circleVotesOf, Peak3, Polarity } from "../lib/hough";
import { makeCircleScene, CircleSceneId, N } from "./scenes";
import { ImgPanel, injectStyle, PANEL_CSS, greyRGBA, heatRGBA, COL, fmtInt } from "../houghlines/view";

const NAME = "houghcircle";
const STYLE = PANEL_CSS(NAME) + `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .hc-left { flex: 1 1 800px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .hc-figs { display: flex; gap: .6em; }
[data-demo="${NAME}"] .hc-figs .hv-fig { flex: 1 1 0; }
[data-demo="${NAME}"] .hc-bottom { display: flex; gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .hc-bottom .qb-presets { flex: 1 1 36%; }
[data-demo="${NAME}"] .hc-prof { flex: 1 1 30%; min-width: 0; }
[data-demo="${NAME}"] .hc-bottom .qb-readout { flex: 1 1 34%; font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; min-width: 240px; gap: .3em; }
[data-demo="${NAME}"] .qb-slider { grid-template-columns: 6.2em 1fr 3.4em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .hc-left { flex: 0 0 820px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 310px; gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .06em .4em; }
.reveal [data-demo="${NAME}"] .hv-credit { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"], [data-demo="${NAME}"] .hc-bottom { flex-wrap: wrap; } }
`;

type Img = CircleSceneId | "cell";
interface State { image: Img; noise: number; r: number; rMin: number; rMax: number; thr: number; nms: number; grad: boolean; pol: Polarity }
interface Preset { id: string; label: string; s: Partial<State>; pick?: [number, number]; centre?: [number, number] }
const DEFAULT: State = { image: "discs", noise: 6, r: 12, rMin: 12, rMax: 12, thr: 0.5, nms: 5, grad: false, pol: "either" };
const PRESETS: Preset[] = [
  { id: "one", label: "one radius: a sharp peak", s: {}, pick: [35, 23] },
  { id: "wrong", label: "wrong radius: no peak", s: { r: 15, rMin: 15, rMax: 15 } },
  { id: "scan", label: "scan the radius", s: { rMin: 5, rMax: 24, r: 18 } },
  { id: "grad", label: "gradient voting", s: { rMin: 5, rMax: 24, r: 18, grad: true, pol: "bright", thr: 0.35 }, pick: [96, 14] },
  { id: "false", label: "low threshold: false circles", s: { rMin: 5, rMax: 24, r: 12, thr: 0.3 } },
  { id: "ellipse", label: "ellipse: smeared peak", s: { rMin: 5, rMax: 24, r: 13, grad: true, pol: "bright", thr: 0.35 }, centre: [108, 82] },
  { id: "plate", label: "plate wells", s: { image: "plate", r: 8, rMin: 8, rMax: 8, grad: true, pol: "dark", thr: 0.4 } },
  { id: "rings", label: "diffraction rings", s: { image: "rings", noise: 8, rMin: 5, rMax: 65, r: 29, grad: true, pol: "either", thr: 0.4 }, centre: [80, 80] },
  { id: "nuclei", label: "synthetic nuclei", s: { image: "nuclei", rMin: 6, rMax: 16, r: 10, grad: true, pol: "bright", thr: 0.25 } },
  { id: "cell", label: "micrograph", s: { image: "cell", noise: 0, rMin: 25, rMax: 50, r: 38, grad: true, pol: "dark", thr: 0.15, nms: 10 } },
];
const R_LIMIT = 70;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  // ---- layout ----
  const left = htmlEl("div", "hc-left", "", host);
  const figs = htmlEl("div", "hc-figs", "", left);
  const pI = new ImgPanel(figs, "edges and detections", N, N);
  const pS = new ImgPanel(figs, "", N, N);
  const pM = new ImgPanel(figs, "maximum over radii", N, N);
  const credit = htmlEl("div", "hv-credit", CELL_CREDIT, left);
  const bottom = htmlEl("div", "hc-bottom", "", left);
  const presetBtns = presets(bottom, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const profDiv = htmlEl("div", "hc-prof", "", bottom);
  const prof = new Plot(profDiv, { width: 300, height: 170, x: [0, 30], y: [0, 1], xlabel: "radius r", font: 16, margin: { l: 44, r: 8, t: 8, b: 44 } });
  const ro = htmlEl("div", "qb-readout", "", bottom);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hv-lab", "image", panel);
  const IMGS: [Img, string][] = [["discs", "discs"], ["plate", "plate"], ["rings", "rings"], ["nuclei", "nuclei"], ["cell", "micrograph"]];
  const imgBtns = presets(panel, IMGS.map(([id, l]) => ({ label: l, apply: () => { st.image = id; makeImage(); } })));
  const sl: Record<string, Slider> = {};
  sl.noise = slider(panel, "noise", 0, 40, 1, st.noise, v => { st.noise = v; makeImage(); });
  sl.rMin = slider(panel, "r min", 3, R_LIMIT, 1, st.rMin, v => { st.rMin = v; if (st.rMax < v) { st.rMax = v; sl.rMax.set(v); } clampR(); vote(); });
  sl.rMax = slider(panel, "r max", 3, R_LIMIT, 1, st.rMax, v => { st.rMax = v; if (st.rMin > v) { st.rMin = v; sl.rMin.set(v); } clampR(); vote(); });
  sl.r = slider(panel, "slice r", 3, R_LIMIT, 1, st.r, v => { st.r = Math.max(st.rMin, Math.min(st.rMax, v)); if (st.r !== v) sl.r.set(st.r); draw(); });
  sl.thr = slider(panel, "threshold", 0.05, 3, 0.05, st.thr, v => { st.thr = v; detect(); });
  sl.nms = slider(panel, "NMS window", 1, 15, 1, st.nms, v => { st.nms = v; detect(); }, v => `±${v}`);
  const gradC = checkbox(panel, "gradient voting", st.grad, v => { st.grad = v; vote(); });
  htmlEl("div", "hv-lab", "polarity (with gradient voting)", panel);
  const POLS: [Polarity, string][] = [["bright", "bright on dark"], ["dark", "dark on bright"], ["either", "either"]];
  const polBtns = presets(panel, POLS.map(([p, l]) => ({ label: l, apply: () => { st.pol = p; vote(); } })));
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); setPick(p0); sync(); makeImage(); });

  function clampR(): void { st.r = Math.max(st.rMin, Math.min(st.rMax, st.r)); sl.r.set(st.r); }
  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); setPick(p); sync(); makeImage(); }
  function setPick(p: Preset | undefined): void { pickXY = p?.pick ?? null; centre = p?.centre ?? null; }
  function sync(): void {
    sl.noise.set(st.noise); sl.rMin.set(st.rMin); sl.rMax.set(st.rMax); sl.r.set(st.r); sl.thr.set(st.thr); sl.nms.set(st.nms);
    gradC.checked = st.grad;
  }
  function syncButtons(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMGS[i][0] === st.image));
    polBtns.forEach((b, i) => b.classList.toggle("active", POLS[i][0] === st.pol));
  }

  // ---- computation ----
  let cell: Grey | null = null, img: Grey | null = null, pts: EdgePoint[] = [], edgeIdx: number[] = [];
  let acc: Float32Array | null = null, nR = 1, rMin0 = 1, votes = 0, ms = 0, maxMap: Float32Array | null = null, argMap: Int32Array | null = null;
  let found: Peak3[] = [], pickXY: [number, number] | null = null, centre: [number, number] | null = null;
  setPick(p0);
  function makeImage(): void {
    syncButtons();
    if (st.image === "cell") {
      if (!cell) { loadCell(N).then(c => { cell = c; makeImage(); }); return; }
      img = st.noise > 0 ? addGaussianNoise(cell, st.noise, 23) : cell;
    } else img = makeCircleScene(st.image, st.noise).img;
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    const c = st.image === "cell" ? canny(img, 2, 3, 8) : canny(img, 1.5, 4, 10);
    pts = edgeList(c.hyst.edges, c.gx, c.gy, c.mag);
    edgeIdx = pts.map(p => p.y * N + p.x);
    vote();
  }
  let pending = false;
  function vote(): void {
    syncButtons();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const t0 = performance.now();
      const C = houghCircles(pts, N, N, { rMin: st.rMin, rMax: st.rMax, grad: st.grad, polarity: st.pol });
      ms = performance.now() - t0;
      acc = normaliseByCircumference(C); nR = C.nR; rMin0 = C.rMin; votes = C.votes;
      const m = maxOverSlices(acc, N * N, nR); maxMap = m.max; argMap = m.arg;
      detect();
    });
  }
  function detect(): void {
    if (!acc) return;
    found = peaks3(acc, N, N, nR, st.thr, st.nms, 2, 60);
    draw();
  }

  // ---- interaction ----
  pI.onPick((x, y) => { pickXY = [x, y]; draw(); });
  const pickCentre = (x: number, y: number) => { centre = [Math.round(x), Math.round(y)]; draw(); };
  pS.onPick(pickCentre); pM.onPick(pickCentre);

  // ---- drawing ----
  function draw(): void {
    if (!img || !acc || !maxMap) return;
    const S = N * N, ri = st.r - rMin0, slice = acc.subarray(ri * S, (ri + 1) * S);
    let hi = 0;
    for (let i = 0; i < acc.length; i++) if (acc[i] > hi) hi = acc[i];
    hi = Math.max(hi, 0.05);
    // image + edges + detections
    const rgb = greyRGBA(img.data, 0, 255, 0.6);
    for (const i of edgeIdx) { rgb[4 * i] = 255; rgb[4 * i + 1] = 159; rgb[4 * i + 2] = 28; }
    pI.putRGBA(rgb);
    pI.clearOverlay();
    for (const f of found) pI.circle(f.x, f.y, f.s + rMin0, COL.cyan, 2.5);
    // nearest edge point to the pick
    let pk: EdgePoint | null = null;
    if (pickXY) { let bd = 16; for (const p of pts) { const d = (p.x - pickXY[0]) ** 2 + (p.y - pickXY[1]) ** 2; if (d < bd) { bd = d; pk = p; } } }
    if (pk) pI.circle(pk.x, pk.y, 3.5, COL.magenta, 3);
    // slice
    pS.putRGBA(heatRGBA(slice, hi));
    pS.cap.textContent = `accumulator slice r = ${st.r}`;
    pS.clearOverlay();
    for (const f of found) if (Math.abs(f.s - ri) <= 0) pS.circle(f.x, f.y, 5, COL.cyan, 2.5);
    if (pk) {
      pS.dot(pk.x, pk.y, 2, COL.magenta);
      const v = circleVotesOf(pk, st.r, st.grad, st.pol);
      if (!st.grad) pS.circle(pk.x, pk.y, st.r, COL.magenta, 2, "none", "4 3");
      else for (const [x, y] of v) pS.dot(x, y, 2.5, COL.magenta);
    }
    if (centre) pS.circle(centre[0], centre[1], 3, "#fff", 2);
    // max over radii
    pM.putRGBA(heatRGBA(maxMap, hi));
    pM.clearOverlay();
    for (const f of found) pM.circle(f.x, f.y, 5, COL.cyan, 2.5);
    if (centre) pM.circle(centre[0], centre[1], 3, "#fff", 2);
    // radius profile at the chosen centre
    prof.clear();
    if (centre) {
      const xs: number[] = [], ys: number[] = [];
      for (let k = 0; k < nR; k++) { xs.push(rMin0 + k); ys.push(acc[k * S + centre[1] * N + centre[0]]); }
      const ymax = Math.max(0.2, ...ys) * 1.1;
      prof.setDomain([rMin0 - 0.5, rMin0 + nR - 0.5], [0, ymax]);
      prof.bars(xs, ys, { color: "var(--qb-c1)", barWidth: 0.85 });
      prof.vline(st.r, { color: COL.magenta, width: 2 });
      prof.hline(st.thr, { color: COL.red, width: 2, dash: "5 4" });
      prof.setLabels({ xlabel: `radius r, at centre (${centre[0]}, ${centre[1]})` });
    } else {
      prof.setDomain([rMin0 - 0.5, rMin0 + nR - 0.5], [0, 1]);
      prof.setLabels({ xlabel: "click the accumulator: votes vs r" });
    }
    // readout
    const lines = [
      `edge points ${fmtInt(pts.length)}   radii ${st.rMin}–${st.rMax}`,
      `bins ${fmtInt(N * N * nR)}   votes ${fmtInt(votes)}`,
      `voting ${ms.toFixed(0)} ms   detected ${found.length}`,
    ];
    for (const f of found.slice(0, 4)) lines.push(`  (${f.x}, ${f.y}) r = ${f.s + rMin0}: ${f.value.toFixed(2)}`);
    if (found.length > 4) lines.push(`  … ${found.length - 4} more`);
    if (pk) lines.push(`point (${pk.x}, ${pk.y}), gradient ${(((pk.theta * 180 / Math.PI) + 360) % 360).toFixed(0)}°`);
    if (centre && argMap) lines.push(`cell (${centre[0]}, ${centre[1]}): best r = ${argMap[centre[1] * N + centre[0]] + rMin0}`);
    ro.textContent = lines.join("\n");
  }

  void fromArray;
  sync();
  makeImage();
}

autoMount(NAME, mount);
