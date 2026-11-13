// The random walker on a chain (graph-based segmentation day, unit 1). Nodes in a row joined
// by conductances w (drawn as thickness; resistance 1/w). Seeds: object (value 1) and
// background (value 0); free nodes get the harmonic value = voltage = probability that a
// random walk from there reaches an object seed first. Bars show the values with the 0.5 line;
// under each edge its share of the voltage drop. "averaging sweep" does one Gauss–Seidel sweep
// (every free node takes the weighted mean of its neighbours), "solve" the exact answer,
// "walkers" releases 200 seeded random walks from the selected node. Optional: the minimum
// graph cut on the same weights. Weights can also come from an intensity profile with σ.
// <div data-demo="walker1d" data-preset="hand|ramp|twoweak|moveseed|sweep|walks|profile"></div>
import { autoMount, htmlEl, svgEl, slider, button, presets, checkbox, Slider } from "../lib/ui";
import { chainGraph, csr, harmonicDense, gaussSeidelSweep, simulateWalks } from "../lib/randomwalker";
import { solveSeg, gridEdges } from "../lib/graph";
import { rng } from "../lib/random";

const NAME = "walker1d";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .w1-fig { flex: 1 1 640px; max-width: 760px; }
.qb-demo[data-demo="${NAME}"] .w1-fig svg { width: 100%; height: auto; display: block; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .w1-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 20px; }
.qb-demo[data-demo="${NAME}"] svg .w1-small { font-size: 18px; fill: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] svg .w1-edge { cursor: pointer; }
.qb-demo[data-demo="${NAME}"] svg .w1-node { cursor: pointer; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .w1-fig { flex: 0 0 720px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 390px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .w1-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type NodeState = 0 | 1 | 2;   // free, object, background
interface Preset { id: string; label: string; w?: number[]; profile?: number[]; sigma?: number; seeds: Record<number, NodeState>; cut?: boolean; solved?: boolean; sel?: number; walks?: boolean }
const PRESETS: Preset[] = [
  { id: "hand", label: "by hand (a): one weak edge", w: [1, 1, 0.25, 1], seeds: { 0: 1, 4: 2 }, solved: true, cut: false },
  { id: "ramp", label: "equal weights: a straight ramp", w: [1, 1, 1, 1, 1, 1, 1], seeds: { 0: 1, 7: 2 }, solved: true },
  { id: "twoweak", label: "two weak edges: walker and cut disagree", w: [0.2, 1, 1, 1, 1, 0.25], seeds: { 0: 1, 6: 2 }, solved: true, cut: true },
  { id: "moveseed", label: "move a seed inwards", w: [0.2, 1, 1, 1, 1, 0.25], seeds: { 2: 1, 6: 2 }, solved: true, cut: true },
  { id: "sweep", label: "averaging converges", w: [0.2, 1, 1, 1, 1, 0.25], seeds: { 0: 1, 6: 2 }, solved: false },
  { id: "walks", label: "walkers estimate the voltage", w: [1, 1, 0.25, 1], seeds: { 0: 1, 4: 2 }, solved: true, sel: 2, walks: true },
  { id: "profile", label: "weights from an intensity profile", profile: [20, 24, 22, 45, 48, 46, 100, 104, 102], sigma: 20, seeds: { 0: 1, 8: 2 }, solved: true, cut: true },
];
const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(3).replace(/0+$/, ""));

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let w: number[] = [], profile: number[] | null = null, sigma = 10;
  let state: NodeState[] = [], x: number[] = [], solved = true, showCut = false, sel = -1, selEdge = -1, sweeps = 0, lastChange = 0;
  let walkInfo = "";

  const fig = htmlEl("div", "w1-fig", "", host);
  const W = 760, Hh = 470;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${Hh}` }, fig);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "w1-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "w1-lab", "selected edge (click an edge)", panel);
  const wS: Slider = slider(panel, "w of edge", -3, 2, 0.05, 0, v => { if (selEdge < 0 || profile) return; w[selEdge] = Math.max(0.01, Math.round(Math.exp(v) * 100) / 100); changed(); }, v => (selEdge < 0 ? "–" : fmt(Math.round(Math.exp(v) * 100) / 100)));
  const nS = slider(panel, "nodes", 5, 12, 1, 5, v => setN(v));
  const sS = slider(panel, "σ (profile)", 2, 40, 1, 10, v => { sigma = v; if (profile) { weightsFromProfile(); changed(); } });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "averaging sweep", () => { sweep(); });
  button(row, "solve", () => { solved = true; draw(); });
  button(row, "walkers", () => walkers());
  button(row, "reset", () => load(p0));
  checkbox(panel, "show the minimum graph cut", false, v => { showCut = v; draw(); });
  const cutBox = panel.querySelectorAll<HTMLInputElement>("input[type=checkbox]")[0];
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click a node to cycle free → object seed (1) → background seed (0). Click an edge to select it and set its conductance w with the slider (resistance 1/w). The bars are the voltages = probabilities; the orange tick marks where they cross 0.5.", panel);

  function weightsFromProfile(): void {
    if (!profile) return;
    w = profile.slice(1).map((v, i) => Math.max(0.001, Math.round(Math.exp(-((v - profile![i]) ** 2) / (2 * sigma * sigma)) * 1000) / 1000));
  }
  function setN(n: number): void {
    profile = null;
    while (w.length < n - 1) w.push(1);
    w.length = n - 1;
    state = state.slice(0, n) as NodeState[];
    while (state.length < n) state.push(0);
    if (!state.some(s => s === 1)) state[0] = 1;
    if (!state.some(s => s === 2)) state[n - 1] = 2;
    x = new Array(n).fill(0.5);
    sel = -1; selEdge = -1;
    changed();
  }
  function load(p: Preset): void {
    profile = p.profile ? p.profile.slice() : null;
    sigma = p.sigma ?? 10; sS.set(sigma);
    if (profile) weightsFromProfile(); else w = p.w!.slice();
    const n = w.length + 1;
    state = new Array(n).fill(0);
    for (const [k, v] of Object.entries(p.seeds)) state[Number(k)] = v;
    solved = p.solved !== false; showCut = !!p.cut; cutBox.checked = showCut;
    sel = p.sel ?? -1; selEdge = -1; sweeps = 0; walkInfo = "";
    x = state.map(s => (s === 1 ? 1 : s === 2 ? 0 : 0.5));
    nS.set(n);
    pb.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
    if (p.walks) walkers();
  }
  function changed(): void { sweeps = 0; walkInfo = ""; x = state.map((s, i) => (s === 1 ? 1 : s === 2 ? 0 : x[i] ?? 0.5)); draw(); }

  const fixedOf = () => state.map(s => (s === 1 ? 1 : s === 2 ? 0 : NaN));
  function exact(): number[] {
    const st = state;
    if (!st.some(s => s === 1) || !st.some(s => s === 2)) return st.map(s => (s === 1 ? 1 : s === 2 ? 0 : NaN));
    return Array.from(harmonicDense(chainGraph(w), fixedOf()));
  }
  function sweep(): void {
    if (solved) { solved = false; x = state.map(s => (s === 1 ? 1 : s === 2 ? 0 : 0.5)); sweeps = 0; }
    const xa = Float64Array.from(x);
    lastChange = gaussSeidelSweep(csr(chainGraph(w)), xa, fixedOf());
    x = Array.from(xa); sweeps++;
    draw();
  }
  function walkers(): void {
    const n = state.length;
    if (sel < 0 || state[sel] !== 0) sel = state.findIndex(s => s === 0);
    if (sel < 0) return;
    const seeds = state.map(s => (s === 1 ? 1 : s === 2 ? 2 : 0));
    const wk = simulateWalks(csr(chainGraph(w)), seeds, sel, 200, rng(1000 + sel * 7 + n));
    const v = exact()[sel];
    walkInfo = `200 walks from node ${sel + 1}: ${wk.counts[1]} reach the object seed first → ${(wk.counts[1] / 200).toFixed(3)}  (voltage ${v.toFixed(3)})`;
    draw();
  }
  function minCut(): number[] {
    const n = state.length, seeds = new Int8Array(n);
    state.forEach((s, i) => { seeds[i] = s === 1 ? 1 : s === 2 ? -1 : 0; });
    if (!state.some(s => s === 1) || !state.some(s => s === 2)) return [];
    const E = gridEdges(1, n, 4);
    const L = solveSeg({ n, D0: new Float64Array(n), D1: new Float64Array(n), E, w, lambda: 1, seeds }).L;
    const cuts: number[] = [];
    for (let i = 0; i + 1 < n; i++) if (L[i] !== L[i + 1]) cuts.push(i);
    return cuts;
  }

  function draw(): void {
    svg.innerHTML = "";
    const n = state.length, x0 = 50, x1 = W - 50, dx = (x1 - x0) / (n - 1), ny = 84;
    const X = (i: number) => x0 + i * dx;
    const vals = solved ? exact() : x;
    const wmax = Math.max(...w, 1);
    // edges
    w.forEach((wi, i) => {
      const g = svgEl("g", { class: "w1-edge" }, svg);
      svgEl("line", { x1: X(i), y1: ny, x2: X(i + 1), y2: ny, stroke: i === selEdge ? "var(--qb-accent, #d62728)" : "var(--qb-fg, #333)", "stroke-width": 2 + 14 * wi / wmax, "stroke-linecap": "round" }, g);
      svgEl("rect", { x: X(i) + 18, y: ny - 22, width: dx - 36, height: 44, fill: "transparent" }, g);
      const t = svgEl("text", { x: (X(i) + X(i + 1)) / 2, y: ny - 24, "text-anchor": "middle" }, g);
      t.textContent = fmt(wi);
      g.addEventListener("pointerdown", () => { selEdge = i; if (!profile) wS.set(Math.log(w[i])); draw(); });
    });
    const tl = svgEl("text", { x: 4, y: ny - 24, class: "w1-small" }, svg); tl.textContent = "w";
    // nodes
    state.forEach((s, i) => {
      const g = svgEl("g", { class: "w1-node" }, svg);
      const fill = s === 1 ? "#e6550d" : s === 2 ? "#3182bd" : "var(--qb-bg, #fff)";
      svgEl("circle", { cx: X(i), cy: ny, r: 17, fill, stroke: i === sel ? "#2ca02c" : "var(--qb-fg, #333)", "stroke-width": i === sel ? 5 : 2 }, g);
      const t = svgEl("text", { x: X(i), y: ny + 7, "text-anchor": "middle", fill: s ? "#fff" : "var(--qb-fg, #333)" }, g);
      t.textContent = String(i + 1);
      if (profile) { const tp = svgEl("text", { x: X(i), y: ny + 44, "text-anchor": "middle", class: "w1-small" }, g); tp.textContent = "I=" + profile[i]; }
      g.addEventListener("pointerdown", () => {
        if (sel === i || s !== 0) { state[i] = ((s + 1) % 3) as NodeState; }
        sel = i; changed();
      });
    });
    // bars
    const by0 = 425, bh = 250;
    svgEl("line", { x1: x0 - 30, y1: by0, x2: x1 + 30, y2: by0, stroke: "var(--qb-muted, #888)" }, svg);
    svgEl("line", { x1: x0 - 30, y1: by0 - bh / 2, x2: x1 + 30, y2: by0 - bh / 2, stroke: "var(--qb-muted, #888)", "stroke-dasharray": "6 5" }, svg);
    const t05 = svgEl("text", { x: 2, y: by0 - bh / 2 - 6, class: "w1-small" }, svg); t05.textContent = "0.5";
    const t1 = svgEl("text", { x: 2, y: by0 - bh + 4, class: "w1-small" }, svg); t1.textContent = "1";
    svgEl("line", { x1: x0 - 30, y1: by0 - bh, x2: x1 + 30, y2: by0 - bh, stroke: "var(--qb-grid, #ddd)" }, svg);
    vals.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      const s = state[i], bw = Math.min(40, dx * 0.5);
      svgEl("rect", { x: X(i) - bw / 2, y: by0 - v * bh, width: bw, height: Math.max(0.5, v * bh), fill: s === 1 ? "#e6550d" : s === 2 ? "#3182bd" : v > 0.5 ? "#fdae6b" : "#9ecae1" }, svg);
      const t = svgEl("text", { x: X(i), y: by0 - v * bh - 8, "text-anchor": "middle" }, svg);
      t.textContent = v.toFixed(3);
    });
    // drop shares and the 0.5 crossings
    const tot = vals.reduce((a, v, i) => (i > 0 && Number.isFinite(v) && Number.isFinite(vals[i - 1]) ? a + Math.abs(v - vals[i - 1]) : a), 0);
    w.forEach((_, i) => {
      const a = vals[i], b = vals[i + 1];
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      if (tot > 1e-12) { const t = svgEl("text", { x: (X(i) + X(i + 1)) / 2, y: by0 + 30, "text-anchor": "middle", class: "w1-small" }, svg); t.textContent = Math.round(100 * Math.abs(b - a) / tot) + "%"; }
      if ((a - 0.5) * (b - 0.5) < 0) {
        const xc = X(i) + (0.5 - a) / (b - a) * dx;
        svgEl("line", { x1: xc, y1: by0 - bh - 10, x2: xc, y2: by0 + 8, stroke: "#ff7f0e", "stroke-width": 4 }, svg);
      }
    });
    const td = svgEl("text", { x: 2, y: by0 + 30, class: "w1-small" }, svg); td.textContent = "drop";
    if (showCut) for (const i of minCut()) {
      const xc = (X(i) + X(i + 1)) / 2;
      svgEl("line", { x1: xc, y1: ny - 14, x2: xc, y2: ny + 26, stroke: "#d62728", "stroke-width": 4, "stroke-dasharray": "6 4" }, svg);
      const t = svgEl("text", { x: xc, y: 22, "text-anchor": "middle", fill: "#d62728" }, svg); t.textContent = "min cut";
    }
    // readout
    const R = w.map(v => 1 / v), Rt = R.reduce((a, b) => a + b, 0);
    const lines = [
      solved ? "exact solution" : `${sweeps} averaging sweep${sweeps === 1 ? "" : "s"}, largest change ${lastChange.toFixed(4)}`,
      "resistances 1/w: " + R.map(fmt).join(", ") + `  (sum ${fmt(Rt)})`,
    ];
    if (showCut) { const c = minCut(); if (c.length) lines.push(`min cut: edge ${c.map(i => `${i + 1}–${i + 2}`).join(", ")} (w ${c.map(i => fmt(w[i])).join(", ")})`); }
    if (walkInfo) lines.push(walkInfo);
    ro.textContent = lines.join("\n");
  }

  load(p0);
}

autoMount(NAME, mount);
