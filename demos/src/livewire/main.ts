// Live-wire / intelligent scissors (Mortensen & Barrett 1995). Click to place a seed: one
// Dijkstra run (8 neighbours, lib/shortestpath.ts) gives the cheapest path from the seed to
// every pixel. Moving the pointer (or dragging the round free-end handle) only follows the
// stored predecessors back to the seed, so the wire follows in real time. Click again to fix
// an anchor (and start a new tree there); "close" joins the last anchor to the first seed.
// Link cost: length × (ε + wG·(1 − G/Gmax) + wZ·[no LoG zero crossing]) + wD·(direction
// term), at smoothing σ; or "bright line" / "dark line" costs (the smoothed intensity).
// Views: the image, the local cost, or the distance from the current seed (iso-lines).
// Auto-mounts on <div data-demo="livewire">. Options: data-preset="nucleus|weak|noise|neighbour|neurite",
// data-view="image|cost|distance".
import { htmlEl, slider, presets, button, autoMount, svgEl, svgPoint } from "../lib/ui";
import { Grey, drawGrey, gaussianBlur } from "../lib/imageops";
import { drawScalarImage, viridis } from "../lib/colour";
import { dijkstraGrid, tracePath, pathLength, ShortestPathResult } from "../lib/shortestpath";
import { liveWireFeatures, liveWireEdge, localCost, CostMode, CostWeights, LiveWireFeatures, SIZE } from "./model";
import { getImage, ImageId, IMAGE_LABELS, CELL_CREDIT, nucleusPoint, neuron } from "./images";

const NAME = "livewire";
type View = "image" | "cost" | "distance";

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 480px; min-width: 240px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .lw-box { position: relative; width: 100%; }
[data-demo="${NAME}"] .lw-box canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .lw-box svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .35em; }
[data-demo="${NAME}"] .lw-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .lw-credit { font-size: 12px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] figure { flex-basis: 500px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; max-width: 380px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .lw-lab, .reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Pt = [number, number];
interface Settings { image: ImageId; sigma: number; mode: CostMode; w: CostWeights; view: View }
interface Preset { label: string; s: Partial<Settings>; anchors: () => Promise<Pt[]>; free: () => Pt; close?: boolean }

const BASE: Settings = { image: "nucleus", sigma: 1.5, mode: "edges", w: { wG: 0.43, wZ: 0.43, wD: 0.14 }, view: "image" };

/** Four points on the nuclear envelope of the micrograph (read off the image). */
const CELL_ANCHORS: Pt[] = [[122, 48], [195, 123], [147, 181], [103, 110]];

const PRESETS: Record<string, Preset> = {
  nucleus: { label: "trace the nucleus (4 anchors)", s: { image: "cell", sigma: 2, mode: "valley" }, anchors: async () => CELL_ANCHORS, free: () => [0, 0], close: true },
  weak: { label: "weak edge: the wire short-cuts", s: { image: "nucleus", sigma: 1.5 }, anchors: async () => [nucleusPoint(95)], free: () => nucleusPoint(205) },
  noise: { label: "no smoothing: noise wins", s: { image: "nucleus", sigma: 0.5 }, anchors: async () => [nucleusPoint(250)], free: () => nucleusPoint(330) },
  neighbour: { label: "the neighbour steals the path", s: { image: "nucleus", sigma: 1.5 }, anchors: async () => [nucleusPoint(-50)], free: () => nucleusPoint(50) },
  neurite: { label: "measure along a neurite", s: { image: "neurite", sigma: 1, mode: "ridge" }, anchors: async () => [roundPt(neuron().tips[0])], free: () => roundPt(neuron().tips[neuron().tips.length - 1]) },
};
const roundPt = (p: Pt): Pt => [Math.round(p[0]), Math.round(p[1])];

function mount(host: HTMLElement): void {
  injectStyle();
  const startKey = host.dataset.preset && PRESETS[host.dataset.preset] ? host.dataset.preset : "";
  let st: Settings = { ...BASE, w: { ...BASE.w }, ...(host.dataset.view ? { view: host.dataset.view as View } : {}) };

  const fig = htmlEl("figure", "", "", host);
  htmlEl("figcaption", "", "click: seed / anchor · move or drag the handle: the wire follows", fig);
  const box = htmlEl("div", "lw-box", "", fig);
  const canvas = htmlEl("canvas", "", "", box);
  const svg = svgEl("svg", { viewBox: `0 0 ${SIZE} ${SIZE}`, preserveAspectRatio: "none" }, box);
  const fixedG = svgEl("g", {}, svg);
  const live = svgEl("polyline", { fill: "none", stroke: "#ffd23f", "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
  const anchorsG = svgEl("g", {}, svg);
  const handle = svgEl("circle", { r: 7, fill: "rgba(255,255,255,.8)", stroke: "#1f77b4", "stroke-width": 3 }, svg);
  handle.style.cursor = "grab";
  const credit = htmlEl("div", "lw-credit", CELL_CREDIT, fig);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "lw-lab", "image", panel);
  const imgBtns = presets(panel, IMAGE_LABELS.map(([id, l]) => ({ label: l, apply: () => { clearPresets(); setImage(id); } })));
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "close", () => closeContour());
  button(ctl, "undo anchor", () => undo());
  button(ctl, "clear", () => { clearPath(); draw(); });
  const sSigma = slider(panel, "smoothing σ", 0.5, 4, 0.5, st.sigma, v => { st.sigma = v; refeature(); });
  htmlEl("div", "lw-lab", "cost", panel);
  const MODES: [CostMode, string][] = [["edges", "edges"], ["ridge", "bright line"], ["valley", "dark line"]];
  const modeBtns = presets(panel, MODES.map(([m, l]) => ({ label: l, apply: () => { st.mode = m; recost(); } })));
  const sG = slider(panel, "w gradient", 0, 1, 0.01, st.w.wG, v => { st.w.wG = v; recost(); });
  const sZ = slider(panel, "w zero-cross.", 0, 1, 0.01, st.w.wZ, v => { st.w.wZ = v; recost(); });
  const sD = slider(panel, "w direction", 0, 1, 0.01, st.w.wD, v => { st.w.wD = v; recost(); });
  htmlEl("div", "lw-lab", "view", panel);
  const VIEWS: [View, string][] = [["image", "image"], ["cost", "cost"], ["distance", "distance from seed"]];
  const viewBtns = presets(panel, VIEWS.map(([v, l]) => ({ label: l, apply: () => { st.view = v; drawBase(); } })));
  htmlEl("div", "lw-lab", "presets", panel);
  const presetBtns = presets(panel, Object.entries(PRESETS).map(([k, p]) => ({ label: p.label, apply: () => applyPreset(k) })));
  button(panel, "reset", () => { clearPresets(); st = { ...BASE, w: { ...BASE.w }, view: host.dataset.view as View ?? "image" }; if (startKey) applyPreset(startKey); else { syncControls(); setImage(st.image); } });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Yellow: the live wire from the last anchor to the free end. Orange: fixed segments.", panel);
  const clearPresets = () => presetBtns.forEach(b => b.classList.remove("active"));

  // ---- state
  let img: Grey | null = null, feat: LiveWireFeatures | null = null, cost: Float32Array | null = null;
  let anchors: Pt[] = [], fixed: number[][] = [], closed = false, tree: ShortestPathResult | null = null, treeMs = 0;
  let free: Pt | null = null;
  let token = 0;

  function syncControls(): void {
    sSigma.set(st.sigma); sG.set(st.w.wG); sZ.set(st.w.wZ); sD.set(st.w.wD);
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMAGE_LABELS[i][0] === st.image));
    modeBtns.forEach((b, i) => b.classList.toggle("active", MODES[i][0] === st.mode));
    viewBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i][0] === st.view));
    credit.style.display = st.image === "cell" ? "" : "none";
  }
  function clearPath(): void { anchors = []; fixed = []; closed = false; tree = null; }

  async function setImage(id: ImageId, keepPath = false): Promise<void> {
    st.image = id;
    const my = ++token;
    const g = await getImage(id);
    if (my !== token) return;
    img = g;
    if (!keepPath) { clearPath(); free = [SIZE / 2, SIZE / 2]; }
    syncControls();
    feat = liveWireFeatures(img, st.sigma);
    cost = localCost(feat, st.mode, st.w);
    drawBase(); draw();
  }
  function refeature(): void { if (!img) return; feat = liveWireFeatures(img, st.sigma); recost(); }
  /** Costs changed: recompute every fixed segment from its anchors, then the tree. */
  function recost(): void {
    if (!feat) return;
    syncControls();
    cost = localCost(feat, st.mode, st.w);
    const a = anchors.slice(), wasClosed = closed;
    clearPath();
    for (const p of a) addAnchor(p, false);
    if (wasClosed) closeContour(false);
    drawBase(); draw();
  }

  const idx = (p: Pt) => p[1] * SIZE + p[0];
  function runTree(from: Pt): void {
    const t0 = performance.now();
    tree = dijkstraGrid({ width: SIZE, height: SIZE, conn: 8, edge: liveWireEdge(feat!, st.mode, st.w), sources: [idx(from)] });
    treeMs = performance.now() - t0;
  }
  function addAnchor(p: Pt, redraw = true): void {
    if (!feat || closed) return;
    if (anchors.length && tree) fixed.push(tracePath(tree.pred, idx(p), tree.dist));
    anchors.push(p);
    runTree(p);
    if (redraw) { if (st.view === "distance") drawBase(); draw(); }
  }
  function closeContour(redraw = true): void {
    if (!tree || anchors.length < 2 || closed) return;
    fixed.push(tracePath(tree.pred, idx(anchors[0]), tree.dist));
    closed = true;
    if (redraw) draw();
  }
  function undo(): void {
    if (!feat) return;
    if (closed) { closed = false; fixed.pop(); }
    else if (anchors.length > 1) { anchors.pop(); fixed.pop(); runTree(anchors[anchors.length - 1]); }
    else clearPath();
    drawBase(); draw();
  }

  async function applyPreset(k: string): Promise<void> {
    const p = PRESETS[k];
    st = { ...BASE, w: { ...BASE.w }, ...p.s, view: st.view };
    presetBtns.forEach((b, i) => b.classList.toggle("active", Object.keys(PRESETS)[i] === k));
    await setImage(st.image);
    const a = await p.anchors();
    clearPath();
    for (const q of a) addAnchor(q, false);
    if (p.close) closeContour(false);
    free = p.close ? null : p.free();
    drawBase(); draw();
  }

  // ---- pointer
  const toPx = (ev: PointerEvent): Pt => {
    const [x, y] = svgPoint(svg, ev);
    return [Math.max(0, Math.min(SIZE - 1, Math.round(x))), Math.max(0, Math.min(SIZE - 1, Math.round(y)))];
  };
  let dragging = false, down: Pt | null = null;
  svg.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    svg.setPointerCapture(ev.pointerId);
    if (ev.target === handle) { dragging = true; handle.style.cursor = "grabbing"; return; }
    down = toPx(ev);
  });
  svg.addEventListener("pointermove", ev => {
    if (!dragging && down && ev.pointerType !== "mouse") return;
    free = toPx(ev);
    draw();
  });
  svg.addEventListener("pointerup", ev => {
    if (dragging) { dragging = false; handle.style.cursor = "grab"; return; }
    const p = toPx(ev);
    if (down && Math.hypot(p[0] - down[0], p[1] - down[1]) <= 4 && !closed) { clearPresets(); addAnchor(p); }
    down = null;
  });
  svg.addEventListener("pointercancel", () => { dragging = false; down = null; });

  // ---- drawing
  function drawBase(): void {
    if (!img || !feat || !cost) return;
    viewBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i][0] === st.view));
    if (st.view === "image") { drawGrey(canvas, img, { mode: "clip", lo: 0, hi: 255 }); return; }
    if (st.view === "cost") { drawScalarImage(canvas, cost, SIZE, SIZE, {}); return; }   // dark = cheap
    if (!tree) { drawGrey(canvas, img, { mode: "clip", lo: 0, hi: 255 }); return; }
    const d = tree.dist;
    let mx = 0;
    for (let i = 0; i < d.length; i++) if (d[i] < Infinity && d[i] > mx) mx = d[i];
    const step = niceStep(mx / 14);
    drawScalarImage(canvas, d, SIZE, SIZE, { map: viridis, lo: 0, hi: mx });
    const c2 = canvas.getContext("2d")!, id = c2.getImageData(0, 0, SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x, b = Math.floor(d[i] / step);
      if ((x + 1 < SIZE && Math.floor(d[i + 1] / step) !== b) || (y + 1 < SIZE && Math.floor(d[i + SIZE] / step) !== b)) {
        id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = 255;
      }
    }
    c2.putImageData(id, 0, 0);
  }
  const niceStep = (v: number) => { const p = 10 ** Math.floor(Math.log10(Math.max(v, 1e-9))), m = v / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; };

  const pts = (path: number[]) => path.map(p => `${p % SIZE},${Math.floor(p / SIZE)}`).join(" ");
  function draw(): void {
    fixedG.replaceChildren();
    let fixedLen = 0, fixedCost = 0;
    for (let k = 0; k < fixed.length; k++) {
      svgEl("polyline", { points: pts(fixed[k]), fill: "none", stroke: "#ff7f0e", "stroke-width": 2.4, "stroke-linejoin": "round" }, fixedG);
      fixedLen += pathLength(fixed[k], SIZE);
    }
    anchorsG.replaceChildren();
    anchors.forEach((a, k) => svgEl("circle", { cx: a[0], cy: a[1], r: k === 0 ? 4.5 : 3.5, fill: k === 0 ? "#d62728" : "#ff7f0e", stroke: "#fff", "stroke-width": 1.2 }, anchorsG));
    // costs of the fixed segments: the tree distances at the time are not kept, so sum the edge costs
    if (feat) {
      const edge = liveWireEdge(feat, st.mode, st.w);
      for (const f of fixed) for (let k = 1; k < f.length; k++) fixedCost += edge(f[k - 1], f[k], Math.hypot((f[k] % SIZE) - (f[k - 1] % SIZE), Math.floor(f[k] / SIZE) - Math.floor(f[k - 1] / SIZE)));
    }
    let liveLen = 0, liveCost = 0;
    if (free && tree && !closed) {
      const path = tracePath(tree.pred, idx(free), tree.dist);
      live.setAttribute("points", pts(path));
      liveLen = pathLength(path, SIZE); liveCost = tree.dist[idx(free)];
    } else live.setAttribute("points", "");
    handle.style.display = free && !closed ? "" : "none";
    if (free) { handle.setAttribute("cx", String(free[0])); handle.setAttribute("cy", String(free[1])); }
    const lines: string[] = [];
    if (!anchors.length) lines.push("click on the image to place a seed");
    else {
      if (!closed) {
        lines.push(`live wire   length ${liveLen.toFixed(1)} px`);
        lines.push(`            cost   ${liveCost.toFixed(2)}`);
        if (free) {
          const a = anchors[anchors.length - 1];
          lines.push(`            straight line ${Math.hypot(free[0] - a[0], free[1] - a[1]).toFixed(1)} px`);
        }
      }
      lines.push(`fixed       ${fixed.length} segment${fixed.length === 1 ? "" : "s"}, ${fixedLen.toFixed(1)} px, cost ${fixedCost.toFixed(2)}${closed ? " (closed)" : ""}`);
      lines.push(`Dijkstra    ${treeMs.toFixed(0)} ms for ${SIZE * SIZE} pixels`);
    }
    ro.textContent = lines.join("\n");
  }

  syncControls();
  if (startKey) applyPreset(startKey).then(() => presetBtns.forEach((b, i) => b.classList.toggle("active", Object.keys(PRESETS)[i] === startKey)));
  else setImage(st.image);
}

autoMount(NAME, mount);
