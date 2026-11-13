// From grid cells to a list of boxes: the decoding of a single-shot detector, on a simulated
// network (Networks for cells and objects day, unit yolo/). Analytic: no trained network.
// A 512 × 512 synthetic scene (cells, debris, a large cell or a palisade of rods) with its true
// boxes; three grids (strides 8, 16, 32: 64², 32², 16² cells). Every cell outputs four 16-bin
// distance distributions (left, top, right, bottom, in strides) and two class scores
// ("cell", "debris"), simulated from the ground truth as the network is trained (model.ts).
//  - Click a cell (grid shown) or a box: its centre, its four distributions as bar charts
//    (drag a bar to change it), the expected distances, the decoded box, its IoU with the
//    nearest true box; class logits as sliders.
//  - Score threshold → candidates (coloured by stride), greedy NMS per class or across classes,
//    step by step; TP/FP/FN/F1 at IoU ≥ 0.5 with the evaluation day's matching (lib/eval.ts).
//  - "anchors" switches to the older head: corrections to one square template box per cell;
//    objects that no anchor overlaps with IoU ≥ 0.5 are never learned.
// Auto-mounts on <div data-demo="yolodecode">. Options: data-preset="b1|twopeaks|large|small|
// aware|agnostic|rods|anchors".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, type Slider } from "../lib/ui";
import { toRGBA } from "../lib/imageops";
import { scalarToRGBA, inferno } from "../lib/colour";
import { type Box, boxIou, boxW, boxH, type NmsResult } from "../lib/boxes";
import * as M from "./model";

const NAME = "yolodecode";
const SCALE_COL = ["#e69f00", "#56b4e9", "#009e73"];   // stride 8, 16, 32 (as in yolonet)
const TRUTH_COL = "#ff6ec7", DEL_COL = "#a0a0a0", OPEN_COL = "#ffd23f";
const SIDES = ["left", "top", "right", "bottom"];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .yd-left { flex: 0 1 470px; min-width: 260px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .yd-stage { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .yd-stage canvas, .qb-demo[data-demo="${NAME}"] .yd-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .yd-stage svg { touch-action: none; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .yd-stage svg text { font-size: 19px; paint-order: stroke; stroke: rgba(0,0,0,.8); stroke-width: 4px; fill: #fff; font-weight: 600; }
.qb-demo[data-demo="${NAME}"] .yd-maps { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .yd-maps figure { margin: 0; flex: 1 1 0; min-width: 0; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .yd-maps canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; border: 3px solid transparent; image-rendering: pixelated; background: #000; }
.qb-demo[data-demo="${NAME}"] .yd-maps figure.yd-off canvas { opacity: .3; }
.qb-demo[data-demo="${NAME}"] .yd-maps figcaption { font-size: 14px; text-align: center; line-height: 1.2; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 400px; max-width: 600px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .yd-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .yd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .yd-two { display: grid; grid-template-columns: 1fr 1fr; gap: .2em .9em; }
.qb-demo[data-demo="${NAME}"] .yd-insp { width: 100%; height: auto; display: block; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .yd-insp text { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .yd-insp .yd-chart { cursor: ns-resize; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .yd-hide { display: none !important; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .yd-left { flex: 0 0 400px; }
.reveal .qb-demo[data-demo="${NAME}"] .yd-maps figure { flex: 0 0 105px; }
.reveal .qb-demo[data-demo="${NAME}"] .yd-maps figcaption { font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 600px; max-width: 700px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .45em; line-height: 1.25; }
.reveal .qb-demo[data-demo="${NAME}"] .yd-lab, .reveal .qb-demo[data-demo="${NAME}"] .yd-noslide { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type View = "raw" | "nms";
interface State {
  scene: M.SceneId; head: M.Head; view: View; grid: number;   // grid: −1 none, else scale index
  conf: number; iou: number; agnostic: boolean; scales: boolean[]; truth: boolean;
  boxNoise: number; scoreNoise: number;
  sel: number; shown: number;
}
const BASE: State = {
  scene: "mixed", head: "dist", view: "nms", grid: -1, conf: 0.25, iou: 0.45, agnostic: false, scales: [true, true, true],
  truth: true, boxNoise: M.SIM_DEFAULTS.boxNoise, scoreNoise: M.SIM_DEFAULTS.scoreNoise, sel: -1, shown: Infinity,
};

interface Preset { id: string; label: string; s: Partial<State>; edit?: (sim: M.Sim) => void; sel?: () => number }
const B1_CELL = M.cellIndex(1, 5, 7);
const b1Edit = (sim: M.Sim, right: [number, number][] = [[1, 0.5], [2, 0.5]]) => {
  M.setSideProbs(sim, B1_CELL, 0, M.massAt([[2, 0.5], [3, 0.5]]));
  M.setSideProbs(sim, B1_CELL, 1, M.massAt([[1, 1]]));
  M.setSideProbs(sim, B1_CELL, 2, M.massAt(right));
  M.setSideProbs(sim, B1_CELL, 3, M.massAt([[2, 1]]));
  M.setClassLogit(sim, B1_CELL, 0, 1.2);
};
const PRESETS: Preset[] = [
  { id: "b1", label: "B1: decode one cell", s: { scene: "mixed", grid: 1 }, edit: (s) => b1Edit(s), sel: () => B1_CELL },
  { id: "twopeaks", label: "two peaks: an uncertain side", s: { scene: "mixed", grid: 1 }, edit: (s) => b1Edit(s, [[2, 0.6], [9, 0.4]]), sel: () => B1_CELL },
  { id: "large", label: "too large for stride 8", s: { scene: "large", grid: 0 }, sel: () => M.cellAt(0, 256, 260) },
  { id: "small", label: "small cell: only stride 8 sees it", s: { scene: "mixed", grid: 2 }, sel: () => M.cellAt(0, 32, 160) },
  { id: "aware", label: "class-aware NMS keeps the debris box", s: { scene: "mixed", agnostic: false }, sel: () => M.cellAt(0, 56, 318) },
  { id: "agnostic", label: "class-agnostic NMS removes it", s: { scene: "mixed", agnostic: true }, sel: () => M.cellAt(0, 56, 318) },
  { id: "rods", label: "crowded rods", s: { scene: "rods" } },
  { id: "anchors", label: "anchors: what no template fits", s: { scene: "mixed", head: "anchor" } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const st: State = { ...BASE, scales: [...BASE.scales] };
  const sceneCache = new Map<M.SceneId, M.Scene>();
  const getScene = (id: M.SceneId) => { if (!sceneCache.has(id)) sceneCache.set(id, M.makeScene(id)); return sceneCache.get(id)!; };
  let sim: M.Sim, dec: M.Decoded, cands: M.Candidate[] = [], res: NmsResult = { keep: [], steps: [], dropped: [] };

  // ---- DOM ------------------------------------------------------------------------------------
  const left = htmlEl("div", "yd-left", "", host);
  const stage = htmlEl("div", "yd-stage", "", left);
  const cv = htmlEl("canvas", "", "", stage); cv.width = cv.height = M.IMG;
  const svg = svgEl("svg", { viewBox: `0 0 ${M.SIZE} ${M.SIZE}` }, stage);
  const maps = htmlEl("div", "yd-maps", "", left);
  const mapFigs = M.GRIDS.map((g, k) => {
    const f = htmlEl("figure", "", "", maps);
    const c = htmlEl("canvas", "", "", f); c.width = c.height = g.g;
    c.style.borderColor = SCALE_COL[k];
    const fc = htmlEl("figcaption", "", "", f);
    c.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      const r = c.getBoundingClientRect();
      st.sel = M.cellAt(k, (ev.clientX - r.left) / r.width * M.SIZE, (ev.clientY - r.top) / r.height * M.SIZE);
      draw();
    });
    return { f, c, fc };
  });
  htmlEl("div", "qb-note yd-noslide", "Score maps: the best class probability of every cell (black 0, yellow 1); the selected cell in green. Click a cell.", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "yd-lab", "Presets", panel);
  const pbtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p) })));

  const row1 = htmlEl("div", "yd-row", "", panel);
  const sceneBtns = (["mixed", "large", "rods"] as M.SceneId[]).map((id) =>
    button(row1, id === "mixed" ? "cells + debris" : id === "large" ? "large cell" : "rods", () => { st.scene = id; st.sel = -1; resim(); }));
  htmlEl("span", "", " │ ", row1);
  const headBtns = (["dist", "anchor"] as M.Head[]).map((h) => button(row1, h === "dist" ? "distance head" : "anchors (older)", () => { st.head = h; recompute(); }));
  const row2 = htmlEl("div", "yd-row", "", panel);
  const viewBtns = (["raw", "nms"] as View[]).map((v) => button(row2, v === "raw" ? "candidates" : "after NMS", () => { st.view = v; st.shown = Infinity; draw(); }));
  htmlEl("span", "", " │ grid:", row2);
  const gridBtns = [-1, 0, 1, 2].map((g) => button(row2, g < 0 ? "off" : String(M.STRIDES[g]), () => { st.grid = g; draw(); }));
  const row3 = htmlEl("div", "yd-row", "", panel);
  const cbScale = M.STRIDES.map((s, k) => {
    const cb = checkbox(row3, `stride ${s}`, true, (v) => { st.scales[k] = v; recompute(); });
    const lab = cb.parentElement as HTMLElement; lab.style.color = SCALE_COL[k]; lab.style.fontWeight = "600";
    return cb;
  });
  const cbTruth = checkbox(row3, "true boxes", st.truth, (v) => { st.truth = v; draw(); });
  const two = htmlEl("div", "yd-two", "", panel);
  const sConf = slider(two, "score >", 0.05, 0.95, 0.05, st.conf, (v) => { st.conf = v; recompute(); });
  const sIou = slider(two, "NMS: IoU >", 0.05, 1, 0.05, st.iou, (v) => { st.iou = v; recompute(); });
  const row4 = htmlEl("div", "yd-row", "", panel);
  const cbAgn = checkbox(row4, "NMS across classes", st.agnostic, (v) => { st.agnostic = v; recompute(); });
  htmlEl("span", "", " │ NMS:", row4);
  button(row4, "restart", () => { st.view = "nms"; st.shown = 0; draw(); });
  button(row4, "step", () => { st.view = "nms"; st.shown = Math.min((isFinite(st.shown) ? st.shown : res.steps.length) + 1, res.steps.length); if (st.shown >= res.steps.length) st.shown = Infinity; draw(); });
  button(row4, "run", () => { st.view = "nms"; st.shown = Infinity; draw(); });
  const noise = htmlEl("div", "yd-two yd-noslide", "", panel);
  const sBox = slider(noise, "box noise", 0, 1, 0.05, st.boxNoise, (v) => { st.boxNoise = v; resim(); });
  const sScore = slider(noise, "score noise", 0, 2, 0.1, st.scoreNoise, (v) => { st.scoreNoise = v; resim(); });

  htmlEl("div", "yd-lab", "Selected cell: drag the bars to change a distribution", panel);
  const insp = svgEl("svg", { class: "yd-insp", viewBox: "0 0 560 118" }, panel);
  const clsRow = htmlEl("div", "yd-two", "", panel);
  const sCls: Slider[] = M.CLASSES.map((name, c) => slider(clsRow, `logit ${name}`, -8, 8, 0.1, 0, (v) => {
    if (st.sel < 0 || st.head !== "dist") return;
    M.setClassLogit(sim, st.sel, c, v); recompute(true);
  }));
  const readout = htmlEl("div", "qb-readout", "", panel);

  // ---- Model ------------------------------------------------------------------------------------
  function resim(edit?: (s: M.Sim) => void): void {
    sim = M.simulate(getScene(st.scene), { boxNoise: st.boxNoise, scoreNoise: st.scoreNoise, seed: M.SIM_DEFAULTS.seed });
    if (edit) edit(sim);
    const g = cv.getContext("2d")!;
    g.putImageData(new ImageData(toRGBA(sim.scene.image, { mode: "clip", lo: 0, hi: 255 }), M.IMG, M.IMG), 0, 0);
    recompute();
  }
  function recompute(keepStep = false): void {
    dec = M.decodeAll(sim, st.head);
    cands = M.candidates(dec, st.conf, st.scales);
    res = M.runNms(dec, cands, st.iou, st.agnostic);
    if (!keepStep) st.shown = Infinity;
    draw();
  }

  // ---- Drawing ----------------------------------------------------------------------------------
  const rect = (b: Box, attrs: Record<string, string | number>, parent: Element = svg) =>
    svgEl("rect", { x: b.x0, y: b.y0, width: Math.max(0, b.x1 - b.x0), height: Math.max(0, b.y1 - b.y0), fill: "none", ...attrs }, parent);

  function draw(): void {
    svg.replaceChildren();
    // grid
    if (st.grid >= 0) {
      const g = M.GRIDS[st.grid];
      let d = "";
      for (let k = 1; k < g.g; k++) d += `M${k * g.stride} 0V${M.SIZE}M0 ${k * g.stride}H${M.SIZE}`;
      svgEl("path", { d, stroke: SCALE_COL[st.grid], "stroke-width": 1.2, opacity: 0.55, fill: "none" }, svg);
    }
    if (st.truth) for (const o of sim.scene.objs) rect(o.box, { stroke: TRUTH_COL, "stroke-width": 2, "stroke-dasharray": "6 4" });
    const sc = (c: M.Candidate) => M.cellInfo(c.cell).scale;
    if (st.view === "raw") {
      for (let k = cands.length - 1; k >= 0; k--) {
        const c = cands[k];
        rect(dec.boxes[c.cell], { stroke: SCALE_COL[sc(c)], "stroke-width": 1.5, opacity: 0.2 + 0.8 * c.score });
      }
    } else {
      const stepping = isFinite(st.shown), n = Math.min(st.shown, res.steps.length), last = n - 1;
      if (stepping) {
        const decided = new Uint8Array(cands.length);
        for (let s = 0; s < n; s++) { decided[res.steps[s].kept] = 1; for (const r of res.steps[s].removed) decided[r.j] = 1; }
        for (let k = 0; k < cands.length; k++) if (!decided[k]) rect(dec.boxes[cands[k].cell], { stroke: OPEN_COL, "stroke-width": 1, opacity: 0.35 });
        if (last >= 0) res.steps[last].removed.forEach((r, q) => {
          const b = dec.boxes[cands[r.j].cell];
          rect(b, { stroke: DEL_COL, "stroke-width": 2, "stroke-dasharray": "5 4" });
          if (q < 6) svgEl("text", { x: b.x1 - 4, y: b.y1 - 6 - 21 * q, "text-anchor": "end" }, svg).textContent = r.iou.toFixed(2);
        });
      }
      for (let s = n - 1; s >= 0; s--) {
        const c = cands[res.steps[s].kept], b = dec.boxes[c.cell];
        rect(b, { stroke: SCALE_COL[sc(c)], "stroke-width": stepping && s === last ? 5 : 3.5 });
        const t = svgEl("text", { x: Math.max(2, b.x0 + 2), y: Math.max(18, b.y0 - 5) }, svg);
        t.textContent = `${M.CLASSES[c.cls]} ${c.score.toFixed(2)}`;
        t.setAttribute("style", `fill:${SCALE_COL[sc(c)]}`);
      }
    }
    drawSelection();
    drawMaps(); drawInspector(); showReadout(); syncControls();
  }

  function drawSelection(): void {
    if (st.sel < 0) return;
    const a = st.sel, c = M.cellInfo(a), s = c.stride;
    rect({ x0: c.j * s, y0: c.i * s, x1: (c.j + 1) * s, y1: (c.i + 1) * s }, { stroke: "#fff", "stroke-width": 2.5 });
    const b = dec.boxes[a];
    if (st.head === "anchor") rect(M.anchorBox(a), { stroke: "#fff", "stroke-width": 2, "stroke-dasharray": "3 5" });
    rect(b, { stroke: "#000", "stroke-width": 6, opacity: 0.6 });
    rect(b, { stroke: "#fff", "stroke-width": 3 });
    if (st.head === "dist") {
      const ends: [number, number][] = [[b.x0, c.cy], [c.cx, b.y0], [b.x1, c.cy], [c.cx, b.y1]];
      for (const [x, y] of ends) {
        svgEl("line", { x1: c.cx, y1: c.cy, x2: x, y2: y, stroke: "#000", "stroke-width": 5, opacity: 0.6 }, svg);
        svgEl("line", { x1: c.cx, y1: c.cy, x2: x, y2: y, stroke: "#fff", "stroke-width": 2.5 }, svg);
      }
    }
    svgEl("circle", { cx: c.cx, cy: c.cy, r: 5, fill: "#fff", stroke: "#000", "stroke-width": 2 }, svg);
  }

  function drawMaps(): void {
    M.GRIDS.forEach((g, k) => {
      const { f, c, fc } = mapFigs[k];
      const vals = new Float32Array(g.count);
      let above = 0;
      for (let q = 0; q < g.count; q++) {
        const a = g.start + q;
        let p = 0;
        for (let cl = 0; cl < M.NC; cl++) p = Math.max(p, dec.prob[a * M.NC + cl]);
        vals[q] = p;
        if (p > st.conf) above++;
      }
      const rgba = scalarToRGBA(vals, { map: inferno, lo: 0, hi: 1 });
      // outline cells above the threshold (white pixels at the cell borders are too fine: tint them)
      const ctx = c.getContext("2d")!;
      ctx.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, g.g, g.g), 0, 0);
      if (st.sel >= g.start && st.sel < g.start + g.count) {
        const q = st.sel - g.start;
        ctx.fillStyle = "#00ff66"; ctx.fillRect(q % g.g, Math.floor(q / g.g), 1, 1);
      }
      f.classList.toggle("yd-off", !st.scales[k]);
      fc.textContent = `stride ${g.stride}: ${g.g}², ${above} > ${st.conf.toFixed(2)}`;
    });
  }

  // inspector: four bar charts (distance head) or the anchor's corrections
  const IW = 132, IH = 66, ITOP = 24;
  function drawInspector(): void {
    insp.replaceChildren();
    if (st.sel < 0) { svgEl("text", { x: 4, y: 22 }, insp).textContent = "No cell selected: click a cell (grid on), a box, or a score map."; return; }
    const a = st.sel, ci = M.cellInfo(a);
    if (st.head === "anchor") {
      const A = M.anchorBox(a), t = sim.aT.subarray(a * 4, a * 4 + 4);
      const lines = [`Anchor head, stride ${ci.stride}: template box ${A.x1 - A.x0} × ${A.y1 - A.y0} px centred on the cell.`,
        `Predicted corrections t = (${Array.from(t).map((v) => v.toFixed(2)).join(", ")}):`,
        `centre shift (t_x, t_y) × anchor size, width and height × e^t_w, e^t_h.`,
        sim.aOwner[a] >= 0 ? `Trained on object ${sim.aOwner[a] + 1} (anchor IoU ${sim.aIou[a].toFixed(2)} ≥ 0.5).` : "No object overlaps this anchor with IoU ≥ 0.5: background."];
      lines.forEach((l, k) => { svgEl("text", { x: 4, y: 20 + 24 * k }, insp).textContent = l; });
      return;
    }
    for (let side = 0; side < 4; side++) {
      const p = M.sideProbs(sim, a, side), x0 = side * (IW + 8) + 2, bw = IW / M.BINS;
      const g = svgEl("g", { class: "yd-chart" }, insp);
      svgEl("rect", { x: x0, y: ITOP, width: IW, height: IH, fill: "var(--qb-panel)" }, g);
      p.forEach((v, k) => svgEl("rect", { x: x0 + k * bw + 0.5, y: ITOP + IH * (1 - v), width: bw - 1, height: IH * v, fill: SCALE_COL[ci.scale] }, g));
      const d = dec.dist[a * 4 + side];
      svgEl("line", { x1: x0 + (d + 0.5) * bw, x2: x0 + (d + 0.5) * bw, y1: ITOP - 3, y2: ITOP + IH, stroke: "var(--qb-fg)", "stroke-width": 2 }, insp);
      const tgt = sim.target[a * 4 + side];
      if (sim.owner[a] >= 0 && isFinite(tgt)) {
        const tx = x0 + (Math.min(tgt, 15.4) + 0.5) * bw;
        svgEl("path", { d: `M${tx - 5} ${ITOP + IH + 9}L${tx + 5} ${ITOP + IH + 9}L${tx} ${ITOP + IH + 2}Z`, fill: TRUTH_COL }, insp);
      }
      svgEl("text", { x: x0, y: ITOP - 7 }, insp).textContent = `${SIDES[side]}: ${d.toFixed(2)}`;
      svgEl("text", { x: x0, y: ITOP + IH + 22, fill: "var(--qb-muted)" }, insp).textContent = "0";
      svgEl("text", { x: x0 + IW, y: ITOP + IH + 22, "text-anchor": "end", fill: "var(--qb-muted)" }, insp).textContent = "15";
      // drag a bar
      const edit = (ev: PointerEvent) => {
        const [x, y] = svgPoint(insp, ev);
        const k = Math.min(M.BINS - 1, Math.max(0, Math.floor((x - x0) / bw)));
        const v = Math.min(1, Math.max(0, 1 - (y - ITOP) / IH));
        M.setSideProbs(sim, a, side, M.setBin(M.sideProbs(sim, a, side), k, v));
        recompute(true);
      };
      g.addEventListener("pointerdown", (ev) => {
        ev.preventDefault(); g.setPointerCapture(ev.pointerId); edit(ev);
        const move = (e: PointerEvent) => edit(e);
        const up = () => { g.removeEventListener("pointermove", move); g.removeEventListener("pointerup", up); };
        g.addEventListener("pointermove", move); g.addEventListener("pointerup", up);
      });
    }
  }

  function showReadout(): void {
    const L: string[] = [];
    const per = [0, 0, 0], kept = [0, 0, 0];
    cands.forEach((c) => per[M.cellInfo(c.cell).scale]++);
    res.keep.forEach((k) => kept[M.cellInfo(cands[k].cell).scale]++);
    L.push(st.head === "dist"
      ? `512 × 512 input: 64² + 32² + 16² = ${M.NCELLS} cells, each 4 × 16 distance bins + ${M.NC} class scores.`
      : `Anchor head: ${M.NCELLS} cells, each one square template (4 strides) + 4 corrections + ${M.NC} class scores.`);
    L.push(`Score > ${st.conf.toFixed(2)}: ${cands.length} candidates (stride 8/16/32: ${per.join("/")}) → after NMS (IoU > ${st.iou.toFixed(2)}, ` +
      `${st.agnostic ? "any class" : "same class"}): ${res.keep.length} (${kept.join("/")}).` +
      (st.view === "nms" && isFinite(st.shown) ? ` Step ${Math.min(st.shown, res.steps.length)} of ${res.steps.length}.` : ""));
    const ev = M.evaluate(sim.scene.objs, res.keep.map((k) => dec.boxes[cands[k].cell]), res.keep.map((k) => cands[k].cls));
    L.push(`${sim.scene.objs.length} true objects. At IoU ≥ 0.5 (class must match): TP ${ev.tp}, FP ${ev.fp}, FN ${ev.fn}, F1 ${ev.f1.toFixed(2)}.`);
    if (st.head === "anchor") {
      const never = sim.scene.objs.map((o, k) => ({ o, k, v: sim.bestAnchor[k] })).filter((x) => x.v < 0.5);
      L.push(never.length ? `No anchor fits (best IoU < 0.5), never learned: ${never.map((x) => `${x.o.name ?? M.CLASSES[x.o.cls]} ${Math.round(boxW(x.o.box))}×${Math.round(boxH(x.o.box))} (${x.v.toFixed(2)})`).join(", ")}.` : "Every object has an anchor with IoU ≥ 0.5.");
    }
    if (st.sel >= 0) {
      const a = st.sel, c = M.cellInfo(a), b = dec.boxes[a];
      let best = -1, bi = 0;
      sim.scene.objs.forEach((o, k) => { const v = boxIou(o.box, b); if (v > bi) { bi = v; best = k; } });
      const probs = M.CLASSES.map((n, cl) => `p(${n}) ${dec.prob[a * M.NC + cl].toFixed(2)}`).join(", ");
      L.push(`Cell: stride ${c.stride}, row ${c.i}, col ${c.j}, centre (${c.cx}, ${c.cy}).` +
        (st.head === "dist" ? ` Distances ${Array.from(dec.dist.subarray(a * 4, a * 4 + 4)).map((v) => v.toFixed(2)).join(", ")} strides` : "") +
        ` → box (${b.x0.toFixed(0)}, ${b.y0.toFixed(0)})–(${b.x1.toFixed(0)}, ${b.y1.toFixed(0)}). ${probs}.`);
      L.push(best >= 0 ? `Nearest true box (${M.CLASSES[sim.scene.objs[best].cls]}): IoU ${bi.toFixed(2)}, GIoU ${M.giou(sim.scene.objs[best].box, b).toFixed(2)}.` : "No true box overlaps it.");
      const own = st.head === "dist" ? sim.owner[a] : sim.aOwner[a];
      if (own >= 0) {
        const o = sim.scene.objs[own], ins = M.centresInside(o.box);
        let line = `Its object (${Math.round(boxW(o.box))} × ${Math.round(boxH(o.box))} px) contains ${ins.join(" / ")} cell centres of stride 8 / 16 / 32.`;
        if (st.head === "dist") {
          const far = [0, 1, 2, 3].filter((s) => sim.target[a * 4 + s] > 15);
          if (far.length) line += ` True ${far.map((s) => `${SIDES[s]} ${sim.target[a * 4 + s].toFixed(1)}`).join(", ")} strides: the bins stop at 15.`;
        }
        L.push(line);
      } else L.push("This cell is not responsible for any object (background in training).");
    }
    readout.textContent = L.join("\n");
  }

  function syncControls(): void {
    sConf.set(st.conf); sIou.set(st.iou); cbAgn.checked = st.agnostic; cbTruth.checked = st.truth;
    sBox.set(st.boxNoise); sScore.set(st.scoreNoise);
    cbScale.forEach((cb, k) => { cb.checked = st.scales[k]; });
    sceneBtns.forEach((b, k) => b.classList.toggle("active", (["mixed", "large", "rods"] as M.SceneId[])[k] === st.scene));
    headBtns.forEach((b, k) => b.classList.toggle("active", (["dist", "anchor"] as M.Head[])[k] === st.head));
    viewBtns.forEach((b, k) => b.classList.toggle("active", (["raw", "nms"] as View[])[k] === st.view));
    gridBtns.forEach((b, k) => b.classList.toggle("active", k - 1 === st.grid));
    const on = st.sel >= 0 && st.head === "dist";
    sCls.forEach((s, c) => { s.input.disabled = !on; if (on) s.set(Math.round(M.classLogit(sim, st.sel, c) * 10) / 10); });
  }

  // ---- Picking on the stage ----------------------------------------------------------------------
  svg.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    const [x, y] = svgPoint(svg, ev);
    if (st.grid >= 0) { st.sel = M.cellAt(st.grid, x, y); draw(); return; }
    const list = st.view === "raw" ? cands : res.keep.map((k) => cands[k]);
    let best = -1, area = Infinity;
    for (const c of list) {
      const b = dec.boxes[c.cell], ar = (b.x1 - b.x0) * (b.y1 - b.y0);
      if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1 && ar < area) { area = ar; best = c.cell; }
    }
    st.sel = best >= 0 ? best : M.cellAt(0, x, y);
    draw();
  });

  // ---- Presets ----------------------------------------------------------------------------------
  function applyPreset(p: Preset): void {
    Object.assign(st, { ...BASE, scales: [...BASE.scales] }, p.s);
    st.sel = p.sel ? p.sel() : -1;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    resim(p.edit);
  }

  const p0 = PRESETS.find((p) => p.id === host.dataset.preset) ?? PRESETS[0];
  applyPreset(p0);
}

autoMount(NAME, mount);
