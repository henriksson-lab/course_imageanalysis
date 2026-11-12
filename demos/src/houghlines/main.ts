// Line detection with the Hough transform. Three panels: the edge points (Canny, from
// lib/edges.ts, with its sigma and thresholds collapsed into one "edges" setting), the
// (theta, rho) accumulator (theta across, rho up; peaks ringed), and the detected lines or
// segments drawn on the image. Click an edge point: its sinusoid of votes appears in the
// accumulator. Click the accumulator: that line is drawn dashed on the image.
// Sliders: bin sizes, vote threshold, NMS window, noise, clutter; toggles: gradient voting
// (with its window) and segments (gap, minimum length).
//
// <div data-demo="houghlines" data-preset="clean|nonms|fine|coarse|clutter|grad|segments|deskew|filaments|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, Slider } from "../lib/ui";
import { Grey } from "../lib/imageops";
import { canny, edgeList, EdgePoint } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { lineGeometry, houghLines, linePeaks, LineAccumulator, Peak2, thetaOf, rhoOf, rhoBin, clipLine, lineSegments, lineVotesOf, dominantAngle, LineGeometry } from "../lib/hough";
import { makeScene, addClutter, SceneId, N } from "./scenes";
import { ImgPanel, injectStyle, PANEL_CSS, greyRGBA, heatRGBA, COL, fmtInt, maxOf } from "./view";

const NAME = "houghlines";
const STYLE = PANEL_CSS(NAME) + `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .hl-left { flex: 1 1 800px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .hl-figs { display: flex; gap: .6em; }
[data-demo="${NAME}"] .hl-figs .hv-fig { flex: 1 1 0; }
[data-demo="${NAME}"] .hl-bottom { display: flex; gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .hl-bottom .qb-presets { flex: 1 1 60%; }
[data-demo="${NAME}"] .hl-bottom .qb-readout { flex: 1 1 40%; font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; min-width: 240px; gap: .3em; }
[data-demo="${NAME}"] .qb-slider { grid-template-columns: 6.2em 1fr 3.4em; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .hl-left { flex: 0 0 820px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 310px; gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .06em .4em; }
.reveal [data-demo="${NAME}"] .hv-credit { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"], [data-demo="${NAME}"] .hl-bottom { flex-wrap: wrap; } }
`;

type Img = SceneId | "cell";
type Edges = "fine" | "medium" | "smooth";
interface State {
  image: Img; noise: number; clutter: number; edges: Edges;
  dTheta: number; dRho: number; thr: number; nms: number;
  grad: boolean; gradWin: number; seg: boolean; gap: number; minLen: number;
}
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { image: "shapes", noise: 4, clutter: 0, edges: "medium", dTheta: 1, dRho: 1, thr: 40, nms: 4, grad: false, gradWin: 3, seg: false, gap: 4, minLen: 15 };
const PRESETS: Preset[] = [
  { id: "clean", label: "clean lines", s: {} },
  { id: "nonms", label: "no NMS: clusters", s: { nms: 0, thr: 40 } },
  { id: "fine", label: "bins too fine", s: { noise: 18, dTheta: 0.25, dRho: 0.5, thr: 40 } },
  { id: "coarse", label: "bins too coarse", s: { image: "grid", dTheta: 5, dRho: 8, thr: 40, nms: 2 } },
  { id: "clutter", label: "clutter raises the floor", s: { clutter: 35, thr: 30 } },
  { id: "grad", label: "gradient voting", s: { clutter: 35, grad: true, gradWin: 3, thr: 30 } },
  { id: "segments", label: "segments with gaps", s: { image: "road", seg: true, thr: 30, gap: 6, minLen: 12, noise: 3 } },
  { id: "deskew", label: "deskew the page", s: { image: "page", grad: true, gradWin: 3, thr: 60, nms: 3, noise: 3 } },
  { id: "filaments", label: "filaments", s: { image: "filaments", clutter: 25, noise: 10, grad: true, gradWin: 4, seg: true, thr: 35, gap: 5, minLen: 25 } },
  { id: "cell", label: "micrograph", s: { image: "cell", noise: 0, edges: "medium", thr: 14, grad: true, gradWin: 5, seg: true, gap: 3, minLen: 12 } },
];
const EDGE_SET: Record<Edges, { sigma: number; lo: number; hi: number }> = {
  fine: { sigma: 1, lo: 6, hi: 14 }, medium: { sigma: 1.5, lo: 4, hi: 10 }, smooth: { sigma: 2.5, lo: 2.5, hi: 6 },
};
const MAX_DRAW = 60;
const LINE_COLS = [COL.red, COL.cyan, COL.yellow, COL.green, COL.magenta, COL.orange, COL.blue];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  // ---- layout ----
  const left = htmlEl("div", "hl-left", "", host);
  const figs = htmlEl("div", "hl-figs", "", left);
  const pE = new ImgPanel(figs, "edge points (click one)", N, N);
  const pA = new ImgPanel(figs, "accumulator: θ →, ρ ↑", 180, 273, 1);
  const pD = new ImgPanel(figs, "detected lines", N, N);
  const credit = htmlEl("div", "hv-credit", CELL_CREDIT, left);
  const bottom = htmlEl("div", "hl-bottom", "", left);
  const presetBtns = presets(bottom, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const ro = htmlEl("div", "qb-readout", "", bottom);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hv-lab", "image", panel);
  const IMGS: [Img, string][] = [["shapes", "shapes"], ["grid", "windows"], ["road", "road"], ["page", "page"], ["filaments", "filaments"], ["cell", "micrograph"]];
  const imgBtns = presets(panel, IMGS.map(([id, l]) => ({ label: l, apply: () => { st.image = id; makeImage(); } })));
  htmlEl("div", "hv-lab", "edges (Canny: smoothing and thresholds)", panel);
  const edgeBtns = presets(panel, (["fine", "medium", "smooth"] as Edges[]).map(e => ({ label: e, apply: () => { st.edges = e; runEdges(); } })));
  const sl: Record<string, Slider> = {};
  sl.noise = slider(panel, "noise", 0, 40, 1, st.noise, v => { st.noise = v; makeImage(); });
  sl.clutter = slider(panel, "clutter", 0, 60, 1, st.clutter, v => { st.clutter = v; makeImage(); });
  sl.dTheta = slider(panel, "Δθ", 0, 7, 1, 0, v => { st.dTheta = DTH[v]; runHough(); }, v => DTH[v] + "°");
  sl.dRho = slider(panel, "Δρ", 0, 6, 1, 0, v => { st.dRho = DRH[v]; runHough(); }, v => DRH[v] + " px");
  sl.thr = slider(panel, "threshold", 5, 150, 1, st.thr, v => { st.thr = v; runPeaks(); });
  sl.nms = slider(panel, "NMS window", 0, 10, 1, st.nms, v => { st.nms = v; runPeaks(); }, v => (v ? `±${v}` : "off"));
  const gradC = checkbox(panel, "gradient voting", st.grad, v => { st.grad = v; runHough(); });
  sl.gradWin = slider(panel, "  window", 0, 20, 0.5, st.gradWin, v => { st.gradWin = v; if (st.grad) runHough(); }, v => `±${v}°`);
  const segC = checkbox(panel, "segments", st.seg, v => { st.seg = v; draw(); });
  sl.gap = slider(panel, "  gap g", 1, 20, 1, st.gap, v => { st.gap = v; if (st.seg) draw(); }, v => v + " px");
  sl.minLen = slider(panel, "  L min", 3, 100, 1, st.minLen, v => { st.minLen = v; if (st.seg) draw(); }, v => v + " px");
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); pick = null; selBin = null; sync(); makeImage(); }
  function sync(): void {
    sl.noise.set(st.noise); sl.clutter.set(st.clutter); sl.dTheta.set(nearest(DTH, st.dTheta)); sl.dRho.set(nearest(DRH, st.dRho));
    sl.thr.set(st.thr); sl.nms.set(st.nms); sl.gradWin.set(st.gradWin); sl.gap.set(st.gap); sl.minLen.set(st.minLen);
    gradC.checked = st.grad; segC.checked = st.seg;
  }
  function syncButtons(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMGS[i][0] === st.image));
    edgeBtns.forEach((b, i) => b.classList.toggle("active", ["fine", "medium", "smooth"][i] === st.edges));
  }

  // ---- computation ----
  let cell: Grey | null = null, img: Grey | null = null, pts: EdgePoint[] = [], edgeIdx: number[] = [];
  let A: LineAccumulator | null = null, g: LineGeometry = lineGeometry(N, N), peaks: Peak2[] = [], ms = 0, msEdges = 0;
  let pick: EdgePoint | null = null, selBin: [number, number] | null = null;
  function makeImage(): void {
    syncButtons();
    if (st.image === "cell") {
      if (!cell) { loadCell(N).then(c => { cell = c; makeImage(); }); return; }
      img = cell;
    } else img = makeScene(st.image).img;
    if (st.clutter > 0) img = addClutter(img, st.clutter);
    if (st.noise > 0) img = addGaussianNoise(img, st.noise, 17);
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    runEdges();
  }
  function runEdges(): void {
    syncButtons();
    if (!img) return;
    const t0 = performance.now(), e = EDGE_SET[st.edges];
    const c = canny(img, e.sigma, e.lo, e.hi);
    pts = edgeList(c.hyst.edges, c.gx, c.gy, c.mag);
    edgeIdx = pts.map(p => p.y * N + p.x);
    msEdges = performance.now() - t0;
    if (pick && !pts.some(p => p.x === pick!.x && p.y === pick!.y)) pick = null;
    runHough();
  }
  function runHough(): void {
    g = lineGeometry(N, N, Math.round(180 / st.dTheta), st.dRho);
    const t0 = performance.now();
    A = houghLines(pts, g, st.grad ? { gradWindow: st.gradWin * Math.PI / 180 } : {});
    ms = performance.now() - t0;
    pA.resize(g.nTheta, g.nRho);
    selBin = null;
    runPeaks();
  }
  function runPeaks(): void {
    if (!A) return;
    peaks = linePeaks(A, st.thr, st.nms, 400);
    draw();
  }

  // ---- interaction ----
  pE.onPick((x, y) => {
    let best: EdgePoint | null = null, bd = 9;
    for (const p of pts) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = p; } }
    pick = best; draw();
  });
  pA.onPick((x, y) => {
    const j = Math.round(x), k = g.nRho - 1 - Math.round(y);
    if (j < 0 || j >= g.nTheta || k < 0 || k >= g.nRho) return;
    selBin = [j, k]; draw();
  });

  // ---- drawing ----
  function draw(): void {
    if (!img || !A) return;
    // panel 1: dimmed image with edge points
    const rgb = greyRGBA(img.data, 0, 255, 0.55);
    for (const i of edgeIdx) { rgb[4 * i] = 255; rgb[4 * i + 1] = 159; rgb[4 * i + 2] = 28; }
    pE.putRGBA(rgb);
    pE.clearOverlay();
    if (pick) pE.circle(pick.x, pick.y, 4, COL.cyan, 3);
    // panel 2: accumulator, theta across, rho up
    const { nTheta: nT, nRho: nR } = g, disp = new Float32Array(nT * nR);
    for (let j = 0; j < nT; j++) for (let k = 0; k < nR; k++) disp[(nR - 1 - k) * nT + j] = A.acc[j * nR + k];
    pA.putRGBA(heatRGBA(disp, Math.max(1, maxOf(A.acc)), undefined, 0.5));
    pA.clearOverlay();
    const rr = Math.max(nT, nR) / 40;
    const shown = peaks.slice(0, MAX_DRAW);
    shown.forEach((p, i) => { const c = pA.circle(p.j, nR - 1 - p.k, rr, LINE_COLS[i % LINE_COLS.length], 2.5); c.setAttribute("r", String(rr)); });
    if (pick) {
      const v = lineVotesOf(pick, g, st.grad ? { gradWindow: st.gradWin * Math.PI / 180 } : {});
      for (const [j, k] of v) pA.dot(j, nR - 1 - k, Math.max(0.5, rr / 4), COL.cyan).setAttribute("stroke-width", "0");
    }
    if (selBin) pA.circle(selBin[0], nR - 1 - selBin[1], rr * 1.4, "#fff", 2, "none", "3 3");
    // panel 3: detections
    pD.putRGBA(greyRGBA(img.data, 0, 255, 0.8));
    pD.clearOverlay();
    let nSeg = 0;
    shown.forEach((p, i) => {
      const th = thetaOf(g, p.j), rh = rhoOf(g, p.k), col = LINE_COLS[i % LINE_COLS.length];
      if (st.seg) {
        const segs = lineSegments(pts, g, th, rh, Math.max(1, st.dRho / 2), st.gap, st.minLen);
        for (const s of segs) { pD.line(s.x0, s.y0, s.x1, s.y1, col, 3); pD.dot(s.x0, s.y0, 2.2, col); pD.dot(s.x1, s.y1, 2.2, col); nSeg++; }
      } else {
        const c = clipLine(g, th, rh);
        if (c) pD.line(c[0], c[1], c[2], c[3], col, 2);
      }
    });
    if (selBin) { const c = clipLine(g, thetaOf(g, selBin[0]), rhoOf(g, selBin[1])); if (c) pD.line(c[0], c[1], c[2], c[3], "#fff", 2, "6 4"); }
    pD.cap.textContent = st.seg ? `detected segments (${nSeg})` : `detected lines (${peaks.length})`;

    // readout
    const lines = [
      `edge points ${fmtInt(pts.length)}   bins ${fmtInt(nT)} × ${fmtInt(nR)}`,
      `votes cast ${fmtInt(A.votes)} (${(A.votes / Math.max(1, pts.length)).toFixed(1)} per point)`,
      `peaks ≥ ${st.thr}: ${peaks.length}${peaks.length > MAX_DRAW ? ` (first ${MAX_DRAW} drawn)` : ""}   strongest ${peaks[0]?.value ?? 0}`,
      `time: edges ${msEdges.toFixed(0)} ms, voting ${ms.toFixed(1)} ms`,
    ];
    if (st.image === "page") {
      const d = dominantAngle(A);
      lines.push(`dominant θ = ${(d.theta * 180 / Math.PI).toFixed(1)}°: page tilted ${(d.theta * 180 / Math.PI - 90).toFixed(1)}°`);
    }
    if (pick) lines.push(`point (${pick.x}, ${pick.y}): gradient ${(((pick.theta * 180 / Math.PI) + 360) % 360).toFixed(0)}°`);
    if (selBin) lines.push(`bin θ = ${(thetaOf(g, selBin[0]) * 180 / Math.PI).toFixed(1)}°, ρ = ${rhoOf(g, selBin[1]).toFixed(1)}: ${A.acc[selBin[0] * nR + selBin[1]]} votes`);
    ro.textContent = lines.join("\n");
  }

  void rhoBin;
  sync();
  makeImage();
}

const DTH = [0.25, 0.5, 1, 2, 3, 4, 5, 6];
const DRH = [0.5, 1, 2, 3, 4, 6, 8];
const nearest = (arr: number[], v: number) => { let b = 0; arr.forEach((a, i) => { if (Math.abs(a - v) < Math.abs(arr[b] - v)) b = i; }); return b; };

autoMount(NAME, mount);
