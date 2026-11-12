// skelgraph: from a skeleton to a graph (skeleton day, unit skel-graph). A 256 × 256 mask
// (presets, or draw with the brush) is thinned to a one-pixel skeleton; its pixels are
// classified by their number of skeleton neighbours (1 = end point, ≥ 3 = junction; adjacent
// junction pixels form one node), and the pieces between nodes become edges. Overlay: nodes
// (blue = end, red = junction, purple = loop) and edges coloured by length, with their
// lengths. Pruning removes terminal edges shorter than k (or k × DT at the junction). The
// table lists every edge with its length by three estimators and its mean radius (DT along
// the edge). The "straight line" preset draws a digital line at angle θ and plots
// estimate / true length against θ for the three estimators.
//
// <div data-demo="skelgraph" data-preset="Y|spurs|prune|loop|line|vessel|draw"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { viridis, toCss } from "../lib/colour";
import { Plot, seriesColor } from "../lib/plot";
import {
  skeletonGraph, cyclomatic, topology, pathLength, meanAlong, chainCounts, lengthFromCounts, LengthEstimator, SkelGraph,
} from "../lib/skeleton";
import { ImgView, injectStyle } from "../canny/imgview";
import { MaskId, makeMask, digitalLine, analyseMask } from "./masks";
import { choice, CHOICE_CSS } from "../medialaxis/choice";

const NAME = "skelgraph";
const N = 256;
const STYLE = CHOICE_CSS + `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; align-items: flex-start; }
[data-demo="${NAME}"] .sg-main { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .sg-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .sg-legend { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] .sg-tablebox { max-height: 300px; overflow-y: auto; overflow-x: hidden; }
[data-demo="${NAME}"] table.sg-table { border-collapse: collapse; font-size: 14px; width: 100%; font-variant-numeric: tabular-nums; }
[data-demo="${NAME}"] table.sg-table th, [data-demo="${NAME}"] table.sg-table td { padding: .08em .2em; text-align: right; border-bottom: 1px solid var(--qb-grid, #ddd); }
[data-demo="${NAME}"] table.sg-table th { position: sticky; top: 0; background: var(--qb-panel, #f6f7f9); }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .sg-main { flex: 0 0 440px; }
.reveal [data-demo="${NAME}"] .sg-mid { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .sg-tablebox { max-height: 250px; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

interface State { mask: MaskId; k: number; relative: boolean; est: LengthEstimator; theta: number; brush: number; erase: boolean; labels: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { mask: "Y", k: 0, relative: false, est: "weighted", theta: 30, brush: 6, erase: false, labels: true };
const PRESETS: Preset[] = [
  { id: "Y", label: "a Y: 3 ends, 1 junction", s: { mask: "Y" } },
  { id: "spurs", label: "noise makes spurs", s: { mask: "spurs" } },
  { id: "prune", label: "prune them", s: { mask: "spurs", k: 24 } },
  { id: "loop", label: "a loop: cycles = holes", s: { mask: "loop" } },
  { id: "line", label: "diagonal lines and the √2", s: { mask: "line", theta: 45, est: "count" } },
  { id: "vessel", label: "a thresholded vessel crop", s: { mask: "vessel", k: 8 } },
  { id: "draw", label: "draw your own", s: { mask: "draw", labels: true } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...BASE, ...(p0?.s ?? {}) });
  const st: State = start();

  // ---- layout ----
  const main = htmlEl("div", "sg-main", "", host);
  const view = new ImgView(main, "mask, skeleton and graph");
  const legend = htmlEl("div", "sg-legend", "", main);
  legend.innerHTML = `<span style="color:#1f77b4">●</span> end point &nbsp;<span style="color:#d62728">●</span> junction &nbsp;<span style="color:#9467bd">●</span> loop node &nbsp;· edges coloured by length`;
  const mid = htmlEl("div", "sg-mid", "", host);
  const ro = htmlEl("div", "qb-readout", "", mid);
  const plotBox = htmlEl("div", "", "", mid);
  const plot = new Plot(plotBox, { width: 300, height: 250, x: [0, 90], y: [0.5, 1.12], xlabel: "angle θ (°)", ylabel: "estimate / true", font: 15, xticks: [0, 15, 30, 45, 60, 75, 90], margin: { l: 56, r: 8, t: 8, b: 44 } });
  const ests: LengthEstimator[] = ["count", "weighted", "corrected"];
  const estLabel: Record<LengthEstimator, string> = { count: "pixel steps", weighted: "1 / √2", corrected: "corrected" };
  {
    const th: number[] = [], ys: number[][] = [[], [], []];
    for (let a = 0; a <= 90; a += 1) {
      const { path, truth } = digitalLine(40, 216, 170, a), c = chainCounts(path, N);
      th.push(a); ests.forEach((e, k) => ys[k].push(lengthFromCounts(c, e) / truth));
    }
    ests.forEach((e, k) => plot.line(th, ys[k], { color: seriesColor(k), label: estLabel[e], width: 2.5 }));
    plot.hline(1, { color: "#888", dash: "4 4", width: 1.5 });
    plot.legend(undefined, "bl");
  }
  const thetaMark = plot.vline(st.theta, { color: "#444", width: 1.5 });
  const tableBox = htmlEl("div", "sg-tablebox", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "examples", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const estC = choice<LengthEstimator>(panel, "length", [["count", "pixel steps"], ["weighted", "1 / √2"], ["corrected", "corrected"]], st.est, v => { st.est = v; analyse(); });
  const kS = slider(panel, "prune k", 0, 30, 1, st.k, v => { st.k = v; analyse(); }, v => (st.relative ? `${v / 4} × DT` : `${v} px`));
  const relC = checkbox(panel, "k relative to the radius DT", st.relative, v => { st.relative = v; kS.set(st.k); analyse(); });
  const thetaBox = htmlEl("div", "", "", panel);
  const thS = slider(thetaBox, "line angle θ", 0, 90, 1, st.theta, v => { st.theta = v; thetaMark.set(v); newMask(); }, v => `${v}°`);
  const drawBox = htmlEl("div", "", "", panel);
  drawBox.style.display = "flex"; drawBox.style.flexDirection = "column"; drawBox.style.gap = "inherit";
  const brushC = choice<"draw" | "erase">(drawBox, "brush", [["draw", "draw"], ["erase", "erase"]], "draw", v => { st.erase = v === "erase"; });
  const bS = slider(drawBox, "brush size", 1, 14, 1, st.brush, v => { st.brush = v; });
  button(drawBox, "clear", () => { mask.fill(0); st.mask = "draw"; analyse(); });
  const labC = checkbox(panel, "write edge lengths", st.labels, v => { st.labels = v; overlay(); });
  button(panel, "⟲ reset", () => { Object.assign(st, start()); pbtns.forEach(b => b.classList.remove("active")); sync(); newMask(); });

  function sync(): void { estC.set(st.est); kS.set(st.k); relC.checked = st.relative; thS.set(st.theta); thetaMark.set(st.theta); bS.set(st.brush); brushC.set(st.erase ? "erase" : "draw"); labC.checked = st.labels; }
  function load(p: Preset): void { Object.assign(st, BASE, p.s); sync(); newMask(); }

  // ---- computation ----
  let mask: Uint8Array = new Uint8Array(N * N), dt: Float32Array = new Float32Array(N * N), skel0: Uint8Array = new Uint8Array(N * N), skel: Uint8Array = skel0, g: SkelGraph = skeletonGraph(skel, N, N);
  let removed = 0;
  function newMask(): void {
    mask = makeMask(st.mask, st.theta);
    thetaBox.style.display = st.mask === "line" ? "" : "none";
    plotBox.style.display = st.mask === "line" ? "" : "none";
    drawBox.style.display = st.mask === "draw" ? "flex" : "none";
    analyse();
  }
  function analyse(): void {
    for (let k = 0; k < N; k++) { mask[k] = 0; mask[(N - 1) * N + k] = 0; mask[k * N] = 0; mask[k * N + N - 1] = 0; }
    const r = analyseMask(mask, st.mask === "line", st.k, st.relative, st.est);
    dt = r.dt; skel0 = r.skel0; skel = r.skel; removed = r.removed; g = r.graph;
    draw();
  }

  // ---- drawing ----
  function draw(): void {
    const px = new Uint8ClampedArray(N * N * 4);
    for (let i = 0; i < N * N; i++) {
      const c = skel[i] ? [25, 25, 25] : skel0[i] ? [250, 160, 160] : mask[i] ? [200, 205, 215] : [255, 255, 255];
      px.set([c[0], c[1], c[2], 255], 4 * i);
    }
    view.rgba(px, N, N);
    const ends = g.nodes.filter(n => n.kind === "end").length, junc = g.nodes.filter(n => n.kind === "junction").length;
    const topo = topology(mask, N, N), lenTot = g.edges.reduce((s, e) => s + pathLength(e.path, N, st.est), 0);
    let text = `nodes V = ${g.nodes.length} (${ends} end, ${junc} junction)\nedges E = ${g.edges.length}, pieces C = ${g.components}\n` +
      `E − V + C = ${cyclomatic(g)}; holes in the mask: ${topo.holes}\ntotal length (${estLabel[st.est]}): ${lenTot.toFixed(1)}`;
    if (removed) text += `\npruned: ${removed} spur${removed === 1 ? "" : "s"} (pink)`;
    if (st.mask === "line" && g.edges.length) {
      const e = g.edges[0], { truth } = digitalLine(40, 216, 170, st.theta), c = chainCounts(e.path, N);
      text += `\nline at ${st.theta}°: true ${truth.toFixed(1)}\n  steps ${lengthFromCounts(c, "count").toFixed(1)}, 1/√2 ${lengthFromCounts(c, "weighted").toFixed(1)}, corrected ${lengthFromCounts(c, "corrected").toFixed(1)}\n  (n_e = ${c.ne} axial, n_o = ${c.no} diagonal, n_c = ${c.nc} corners)`;
    }
    ro.textContent = text;
    // edge table
    const rows = g.edges.slice(0, 60).map(e => {
      const c = chainCounts(e.path, N);
      return `<tr><td>${e.id + 1}</td><td>${e.a + 1}–${e.b + 1}</td><td>${lengthFromCounts(c, "count").toFixed(0)}</td><td>${lengthFromCounts(c, "weighted").toFixed(1)}</td><td>${lengthFromCounts(c, "corrected").toFixed(1)}</td><td>${meanAlong(e.path, dt).toFixed(1)}</td></tr>`;
    }).join("");
    tableBox.innerHTML = st.mask === "line" ? "" : `<table class="sg-table"><tr><th>#</th><th>nodes</th><th>steps</th><th>1/√2</th><th>corr.</th><th>r</th></tr>${rows}</table>` + (g.edges.length > 60 ? `<div class="sg-legend">… ${g.edges.length - 60} more</div>` : "");
    overlay();
  }

  function overlay(): void {
    view.clear();
    const lens = g.edges.map(e => pathLength(e.path, N, st.est)), mx = Math.max(1, ...lens);
    g.edges.forEach((e, k) => {
      const pts = e.path.map(i => `${(i % N) + 0.5},${Math.floor(i / N) + 0.5}`).join(" ");
      view.add("polyline", { points: pts, fill: "none", stroke: toCss(viridis(0.15 + 0.8 * lens[k] / mx)), "stroke-width": 4, "stroke-linejoin": "round", opacity: 0.85 });
    });
    if (st.labels && g.edges.length <= 40) g.edges.forEach((e, k) => {
      const i = e.path[Math.floor(e.path.length / 2)], x = (i % N) + 0.5, y = Math.floor(i / N) + 0.5;
      const t = svgEl("text", { x: x + 4, y: y - 4, class: "qb-halo", "font-size": 11, "font-weight": 700 }, view.over);
      t.textContent = lens[k].toFixed(lens[k] < 10 ? 1 : 0);
    });
    for (const n of g.nodes) {
      const col = n.kind === "end" ? "#1f77b4" : n.kind === "junction" ? "#d62728" : "#9467bd";
      view.add("circle", { cx: n.x + 0.5, cy: n.y + 0.5, r: n.kind === "junction" ? 4.5 : 3.5, fill: col, stroke: "#fff", "stroke-width": 1.5 });
    }
  }

  // brush
  let paintPending = false;
  view.onPick((x, y) => {
    if (st.mask === "line") return;
    if (st.mask !== "draw") { st.mask = "draw"; drawBox.style.display = "flex"; pbtns.forEach(b => b.classList.remove("active")); }
    const r = st.brush;
    for (let yy = y - r; yy <= y + r; yy++) for (let xx = x - r; xx <= x + r; xx++)
      if (xx > 0 && yy > 0 && xx < N - 1 && yy < N - 1 && (xx - x) ** 2 + (yy - y) ** 2 <= r * r) mask[yy * N + xx] = st.erase ? 0 : 1;
    if (paintPending) return;
    paintPending = true;
    requestAnimationFrame(() => { paintPending = false; analyse(); });
  });

  if (p0) pbtns[PRESETS.indexOf(p0)]?.classList.add("active");
  newMask();
}

autoMount(NAME, mount);
