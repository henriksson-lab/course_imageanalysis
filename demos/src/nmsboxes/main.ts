// Boxes, IoU, non-maximum suppression, anchors and centre heat maps (modern deep learning
// day, unit detection/). Four modes on one stage:
//   iou      two boxes on a grid (drag the body to move, the corner handle to resize):
//            intersection shaded, areas, union and IoU printed live;
//   nms      a simulated detector's candidate boxes on a synthetic cell image; greedy NMS with
//            IoU threshold t and score threshold, step by step (kept box outlined, removed
//            boxes greyed with their IoU), and the evaluation day's readouts (lib/eval.ts):
//            TP/FP/FN, F1 and matching accuracy at τ = 0.5, AP of the surviving boxes;
//            click a box to select it, then ± its score;
//   anchors  a grid of anchors (stride, size, 1–3 shapes): each true object's best anchor
//            and its IoU, objects without an anchor above 0.5 in red;
//   heatmap  a simulated centre heat map (bump width, noise), its local maxima in a window,
//            and the boxes they give.
// Everything is analytic (no trained network). Auto-mounts on <div data-demo="nmsboxes">.
// Options: data-preset="iou|dup|crowd|high|rod|anchor|heat|noisyheat", data-mode.
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { toRGBA } from "../lib/imageops";
import { scalarToRGBA, inferno } from "../lib/colour";
import { type Box, boxArea, boxIntersection, nms, boxIouTable, anchorGrid, bestAnchors, localMaxima, type NmsResult } from "../lib/boxes";
import { match, objectScores, averagePrecision } from "../lib/eval";
import { N, makeScene, simulateDetector, centreHeatmap, boxesFromPeaks, type Scene, type SceneId, type Candidates } from "./model";

const NAME = "nmsboxes";
type Mode = "iou" | "nms" | "anchors" | "heatmap";

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .nb-stage { position: relative; flex: 0 1 470px; min-width: 260px; aspect-ratio: 1; }
[data-demo="${NAME}"] .nb-stage canvas, [data-demo="${NAME}"] .nb-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
[data-demo="${NAME}"] .nb-stage canvas { image-rendering: pixelated; }
[data-demo="${NAME}"] .nb-stage svg { touch-action: none; overflow: visible; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 330px; max-width: 470px; gap: .4em; }
[data-demo="${NAME}"] .nb-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .nb-hide { display: none !important; }
[data-demo="${NAME}"] svg text { font-size: 4.6px; paint-order: stroke; stroke: rgba(0,0,0,.75); stroke-width: .9px; fill: #fff; }
.reveal [data-demo="${NAME}"] { font-size: 18px; gap: .8em; }
.reveal [data-demo="${NAME}"] .nb-stage { flex-basis: 470px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .nb-stage { flex: 0 0 430px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 560px; max-width: 680px; }
.reveal [data-demo="${NAME}"] .nb-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; line-height: 1.2; padding: .25em .5em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State {
  mode: Mode; scene: SceneId;
  t: number; minScore: number; shown: number; showTruth: boolean; sel: number;
  stride: number; size: number; shapes: number;
  sigma: number; noise: number; win: number; peakMin: number;
  A: Box; B: Box;
}

interface Preset { id: string; label: string; s: Partial<State>; run?: boolean }
const BASE: State = {
  mode: "nms", scene: "sparse", t: 0.5, minScore: 0.2, shown: 0, showTruth: false, sel: -1,
  stride: 16, size: 20, shapes: 1, sigma: 3, noise: 0, win: 1, peakMin: 0.3,
  A: { x0: 0, y0: 0, x1: 4, y1: 4 }, B: { x0: 2, y0: 2, x1: 6, y1: 6 },
};
const PRESETS: Preset[] = [
  { id: "iou", label: "IoU of two boxes", s: { mode: "iou" } },
  { id: "dup", label: "duplicates removed", s: { mode: "nms", scene: "sparse", t: 0.5 }, run: true },
  { id: "crowd", label: "touching cells: NMS deletes the neighbour", s: { mode: "nms", scene: "crowd", t: 0.2, showTruth: true }, run: true },
  { id: "high", label: "threshold too high: duplicates survive", s: { mode: "nms", scene: "crowd", t: 0.75, showTruth: true }, run: true },
  { id: "rod", label: "diagonal rod: a box is a poor shape", s: { mode: "nms", scene: "rod", t: 0.5, showTruth: true }, run: true },
  { id: "anchor", label: "small cell: no anchor matches", s: { mode: "anchors", scene: "rod", stride: 16, size: 20, shapes: 1 } },
  { id: "heat", label: "heat-map peaks = objects", s: { mode: "heatmap", scene: "crowd", sigma: 3, noise: 0, win: 1 } },
  { id: "noisyheat", label: "noisy heat map: two peaks in one cell", s: { mode: "heatmap", scene: "sparse", sigma: 5, noise: 0.2, win: 1, peakMin: 0.3 } },
];

const COL_KEPT = "#2ca02c", COL_DEL = "#9a9a9a", COL_OPEN = "#ffd23f", COL_TRUTH = "#56b4e9", COL_BAD = "#ff3b30";

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st: State = { ...BASE, A: { ...BASE.A }, B: { ...BASE.B } };
  let scene: Scene = makeScene(st.scene), cand: Candidates = simulateDetector(scene.truth, 7), res: NmsResult;
  const sceneCache = new Map<SceneId, { scene: Scene; cand: Candidates }>();

  const stage = htmlEl("div", "nb-stage", "", host);
  const cv = htmlEl("canvas", "", "", stage); cv.width = N; cv.height = N;
  const svg = svgEl("svg", { viewBox: `0 0 ${N} ${N}` }, stage);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "nb-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "nb-lab", "mode", panel);
  const modeRow = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = (["iou", "nms", "anchors", "heatmap"] as Mode[]).map(m => button(modeRow, m === "iou" ? "IoU" : m === "nms" ? "NMS" : m, () => { st.mode = m; refresh(); }));
  const sceneBox = htmlEl("div", "", "", panel);
  htmlEl("div", "nb-lab", "image", sceneBox);
  const sceneRow = htmlEl("div", "qb-presets", "", sceneBox);
  const sceneBtns = (["sparse", "crowd", "rod"] as SceneId[]).map(s => button(sceneRow, s === "rod" ? "rod + small cell" : s, () => { st.scene = s; loadScene(); st.shown = 0; refresh(); }));

  // NMS controls
  const nmsBox = htmlEl("div", "", "", panel);
  const tS = slider(nmsBox, "IoU threshold t", 0.05, 0.95, 0.05, st.t, v => { st.t = v; refresh(true); });
  const sS = slider(nmsBox, "score threshold", 0, 0.9, 0.05, st.minScore, v => { st.minScore = v; refresh(true); });
  const stepRow = htmlEl("div", "qb-presets", "", nmsBox);
  button(stepRow, "step", () => { st.shown = Math.min(st.shown + 1, res.steps.length); draw(); });
  button(stepRow, "run", () => { st.shown = res.steps.length; draw(); });
  button(stepRow, "reset", () => { st.shown = 0; draw(); });
  button(stepRow, "score +", () => bump(0.05));
  button(stepRow, "score −", () => bump(-0.05));
  const truthCb = checkbox(nmsBox, "show the true boxes", st.showTruth, v => { st.showTruth = v; draw(); });

  // anchor controls
  const ancBox = htmlEl("div", "", "", panel);
  const strS = slider(ancBox, "stride", 8, 32, 4, st.stride, v => { st.stride = v; draw(); });
  const sizS = slider(ancBox, "anchor size", 6, 40, 2, st.size, v => { st.size = v; draw(); });
  const shpS = slider(ancBox, "shapes", 1, 3, 1, st.shapes, v => { st.shapes = v; draw(); }, v => ["", "1:1", "1:1, 2:1", "1:1, 2:1, 1:2"][v]);

  // heat-map controls
  const heatBox = htmlEl("div", "", "", panel);
  const sigS = slider(heatBox, "bump width σ", 1, 8, 0.5, st.sigma, v => { st.sigma = v; draw(); });
  const noiS = slider(heatBox, "noise", 0, 0.5, 0.05, st.noise, v => { st.noise = v; draw(); });
  const winS = slider(heatBox, "window", 1, 6, 1, st.win, v => { st.win = v; draw(); }, v => `${2 * v + 1}×${2 * v + 1}`);
  const pkS = slider(heatBox, "min peak", 0.05, 0.9, 0.05, st.peakMin, v => { st.peakMin = v; draw(); });

  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  button(panel, "reset", () => applyPreset(p0));

  function loadScene(): void {
    let c = sceneCache.get(st.scene);
    if (!c) { const sc = makeScene(st.scene); c = { scene: sc, cand: simulateDetector(sc.truth, 7) }; sceneCache.set(st.scene, c); }
    scene = c.scene; cand = { boxes: c.cand.boxes, scores: [...c.cand.scores], source: c.cand.source };
    st.sel = -1;
  }

  function applyPreset(p: Preset): void {
    Object.assign(st, BASE, { A: { ...BASE.A }, B: { ...BASE.B } }, p.s);
    sceneCache.clear();
    loadScene();
    tS.set(st.t); sS.set(st.minScore); strS.set(st.stride); sizS.set(st.size); shpS.set(st.shapes);
    sigS.set(st.sigma); noiS.set(st.noise); winS.set(st.win); pkS.set(st.peakMin); truthCb.checked = st.showTruth;
    st.shown = 0;
    refresh();
    if (p.run) { st.shown = res.steps.length; draw(); }
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function bump(d: number): void {
    if (st.sel < 0) return;
    cand.scores[st.sel] = Math.round(Math.min(0.99, Math.max(0.01, cand.scores[st.sel] + d)) * 100) / 100;
    refresh(true);
  }

  /** Recompute NMS; keep showing all steps if they were all shown. */
  function refresh(keepRun = false): void {
    const wasAll = res && st.shown >= res.steps.length;
    res = nms(cand.boxes, cand.scores, st.t, st.minScore);
    if (keepRun && wasAll) st.shown = res.steps.length;
    st.shown = Math.min(st.shown, res.steps.length);
    modeBtns.forEach((b, i) => b.classList.toggle("active", ["iou", "nms", "anchors", "heatmap"][i] === st.mode));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["sparse", "crowd", "rod"][i] === st.scene));
    sceneBox.classList.toggle("nb-hide", st.mode === "iou");
    nmsBox.classList.toggle("nb-hide", st.mode !== "nms");
    ancBox.classList.toggle("nb-hide", st.mode !== "anchors");
    heatBox.classList.toggle("nb-hide", st.mode !== "heatmap");
    draw();
  }

  // ---- drawing helpers ----
  function rect(b: Box, attrs: Record<string, string | number>, parent: Element = svg): SVGRectElement {
    return svgEl("rect", { x: b.x0, y: b.y0, width: Math.max(0, b.x1 - b.x0), height: Math.max(0, b.y1 - b.y0), fill: "none", ...attrs }, parent);
  }
  function label(x: number, y: number, s: string, parent: Element = svg, anchor = "start"): void {
    const t = svgEl("text", { x, y, "text-anchor": anchor }, parent); t.textContent = s;
  }
  function drawImage(): void {
    const ctx = cv.getContext("2d")!;
    ctx.putImageData(new ImageData(toRGBA(scene.image, { lo: 0, hi: 255 }), N, N), 0, 0);
  }

  function draw(): void {
    svg.innerHTML = "";
    if (st.mode === "iou") return drawIou();
    drawImage();
    if (st.mode === "nms") drawNms();
    else if (st.mode === "anchors") drawAnchors();
    else drawHeat();
  }

  // ---- IoU mode: world [-2, 14]² on a grid ----
  const W0 = -2, W1 = 14, k = N / (W1 - W0);
  const toS = (v: number) => (v - W0) * k, toW = (v: number) => v / k + W0;
  function drawIou(): void {
    const ctx = cv.getContext("2d")!;
    ctx.fillStyle = "#fbfbfb"; ctx.fillRect(0, 0, N, N);
    for (let v = W0; v <= W1; v++) {
      const s = toS(v);
      svgEl("line", { x1: s, y1: 0, x2: s, y2: N, stroke: v === 0 ? "#888" : "#ddd", "stroke-width": v === 0 ? 0.5 : 0.3 }, svg);
      svgEl("line", { x1: 0, y1: s, x2: N, y2: s, stroke: v === 0 ? "#888" : "#ddd", "stroke-width": v === 0 ? 0.5 : 0.3 }, svg);
      if (v % 2 === 0 && v > W0) { const t = svgEl("text", { x: s, y: N - 1.5, "text-anchor": "middle", style: "fill:#555;stroke:none" }, svg); t.textContent = String(v); }
    }
    const fillA = rect(st.A, { fill: "#1f77b4", "fill-opacity": 0.18, stroke: "#1f77b4", "stroke-width": 1 });
    const fillB = rect(st.B, { fill: "#d62728", "fill-opacity": 0.18, stroke: "#d62728", "stroke-width": 1 });
    const inter = rect(st.A, { fill: "#9467bd", "fill-opacity": 0.55, stroke: "none" });
    const parts = ([[st.A, "A", fillA], [st.B, "B", fillB]] as [Box, string, SVGRectElement][]).map(([B, name, fr]) => {
      const body = rect(B, { fill: "transparent", stroke: "none", style: "cursor:move" });
      const hd = svgEl("rect", { width: 5, height: 5, fill: "#fff", stroke: name === "A" ? "#1f77b4" : "#d62728", "stroke-width": 0.8, style: "cursor:nwse-resize" }, svg);
      const t = svgEl("text", {}, svg); t.textContent = name;
      return { B, fr, body, hd, t };
    });
    const update = () => {
      const S = (b: Box) => ({ x: toS(b.x0), y: toS(b.y0), width: toS(b.x1) - toS(b.x0), height: toS(b.y1) - toS(b.y0) });
      for (const p of parts) {
        for (const e of [p.fr, p.body]) for (const [k, v] of Object.entries(S(p.B))) e.setAttribute(k, String(v));
        p.hd.setAttribute("x", String(toS(p.B.x1) - 2.5)); p.hd.setAttribute("y", String(toS(p.B.y1) - 2.5));
        p.t.setAttribute("x", String(toS(p.B.x0) + 1.5)); p.t.setAttribute("y", String(toS(p.B.y0) + 5.5));
      }
      const I: Box = { x0: Math.max(st.A.x0, st.B.x0), y0: Math.max(st.A.y0, st.B.y0), x1: Math.min(st.A.x1, st.B.x1), y1: Math.min(st.A.y1, st.B.y1) };
      const vis = I.x1 > I.x0 && I.y1 > I.y0;
      inter.style.display = vis ? "" : "none";
      if (vis) for (const [k, v] of Object.entries(S(I))) inter.setAttribute(k, String(v));
      const ia = boxIntersection(st.A, st.B), u = boxArea(st.A) + boxArea(st.B) - ia;
      ro.textContent = `A = [${st.A.x0}, ${st.A.x1}] × [${st.A.y0}, ${st.A.y1}]   B = [${st.B.x0}, ${st.B.x1}] × [${st.B.y0}, ${st.B.y1}]\n` +
        `area A = ${boxArea(st.A)}   area B = ${boxArea(st.B)}\nintersection = ${ia}\nunion = ${boxArea(st.A)} + ${boxArea(st.B)} − ${ia} = ${u}\nIoU = ${ia} / ${u} = ${(ia / u || 0).toFixed(3)}`;
    };
    for (const p of parts) { dragBox(p.body, p.B, "move", update); dragBox(p.hd, p.B, "size", update); }
    update();
    note.textContent = "Drag a box to move it, its white corner to resize it (whole grid units).";
  }
  function dragBox(el: SVGElement, B: Box, how: "move" | "size", update: () => void): void {
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault();
      el.setPointerCapture(ev.pointerId);
      const [sx, sy] = svgPoint(svg, ev), b0 = { ...B }, wx = toW(sx), wy = toW(sy);
      const move = (e: PointerEvent) => {
        const [x, y] = svgPoint(svg, e), dx = Math.round(toW(x) - wx), dy = Math.round(toW(y) - wy);
        if (how === "move") { B.x0 = b0.x0 + dx; B.x1 = b0.x1 + dx; B.y0 = b0.y0 + dy; B.y1 = b0.y1 + dy; }
        else { B.x1 = Math.max(B.x0 + 1, b0.x1 + dx); B.y1 = Math.max(B.y0 + 1, b0.y1 + dy); }
        update();
      };
      const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); };
      el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    });
  }

  // ---- NMS mode ----
  function drawNms(): void {
    const status = new Int8Array(cand.boxes.length);        // 0 open, 1 kept, 2 removed, 3 below score
    const iouOf = new Float64Array(cand.boxes.length);
    for (const j of res.dropped) status[j] = 3;
    res.steps.slice(0, st.shown).forEach(s => { status[s.kept] = 1; for (const r of s.removed) { status[r.j] = 2; iouOf[r.j] = r.iou; } });
    const last = st.shown > 0 ? res.steps[st.shown - 1] : null;
    if (st.showTruth) scene.truth.forEach(t => rect(t, { stroke: COL_TRUTH, "stroke-width": 0.9, "stroke-dasharray": "2 1.2" }));
    const order = cand.boxes.map((_, i) => i).sort((a, b) => status[b] === 1 ? -1 : status[a] === 1 ? 1 : 0);
    const g = svgEl("g", {}, svg);
    for (const i of order) {
      const b = cand.boxes[i], s = status[i];
      if (s === 3) { rect(b, { stroke: "#777", "stroke-width": 0.35, "stroke-dasharray": "1 1" }, g); continue; }
      const col = s === 1 ? COL_KEPT : s === 2 ? COL_DEL : COL_OPEN;
      const r = rect(b, { stroke: col, "stroke-width": s === 1 ? (last && last.kept === i ? 1.6 : 1.1) : 0.5, "stroke-dasharray": s === 2 ? "1.5 1" : "none", fill: "transparent", style: "cursor:pointer" }, g);
      if (i === st.sel) r.setAttribute("stroke-width", "1.6"), r.setAttribute("stroke", "#e377c2");
      r.addEventListener("pointerdown", ev => { ev.preventDefault(); st.sel = i; draw(); });
    }
    for (const i of order) {
      const b = cand.boxes[i], s = status[i];
      if (s === 1 || i === st.sel) label(b.x0 + 0.6, b.y0 + 4.4, cand.scores[i].toFixed(2), svg);
      if (last && last.removed.some(r => r.j === i)) label(b.x1 - 0.5, b.y1 - 0.8, "IoU " + iouOf[i].toFixed(2), svg, "end");
    }
    // evaluation of the kept boxes (all steps), and of all candidates for comparison
    const kept = res.keep, kb = kept.map(i => cand.boxes[i]), ks = kept.map(i => cand.scores[i]);
    const T = boxIouTable(scene.truth, kb), m = match(T, 0.5, "optimal"), sc = objectScores(m);
    const ap = averagePrecision(T, ks, 0.5).ap;
    const above = cand.boxes.map((_, i) => i).filter(i => cand.scores[i] >= st.minScore);
    const apAll = averagePrecision(boxIouTable(scene.truth, above.map(i => cand.boxes[i])), above.map(i => cand.scores[i]), 0.5).ap;
    let fill = "";
    if (st.scene === "rod") {
      const e = scene.ellipses[0], area = Math.PI * e.a * e.b;
      fill = `\nrod: the object fills ${(100 * area / boxArea(scene.truth[0])).toFixed(0)} % of its box`;
    }
    const lostNb = st.scene === "crowd" ? lostNeighbours(kept) : "";
    ro.textContent =
      `${cand.boxes.length} candidates, ${scene.truth.length} true cells\n` +
      `step ${st.shown} / ${res.steps.length}` + (last ? `: keep ${cand.scores[last.kept].toFixed(2)}, remove ${last.removed.length}` : "") + "\n" +
      `after NMS (all steps): ${kept.length} boxes\n` +
      `TP ${m.tp}  FP ${m.fp}  FN ${m.fn}   (τ = 0.5)\n` +
      `F1 ${sc.f1.toFixed(2)}   matching accuracy ${(m.tp / Math.max(1, m.tp + m.fp + m.fn)).toFixed(2)}\n` +
      `AP: ${ap.toFixed(2)} after NMS, ${apAll.toFixed(2)} without` + fill + lostNb;
    note.textContent = "Yellow: not yet decided; green: kept; grey dashed: removed (IoU with the kept box); thin dotted: below the score threshold. Click a box, then score ± to change its score.";
  }
  function lostNeighbours(kept: number[]): string {
    const hit = new Set(kept.map(i => cand.source[i]));
    const lost = scene.truth.filter((_, k) => !hit.has(k)).length;
    return lost ? `\ncells with no kept box: ${lost}` : "";
  }

  // ---- anchors mode ----
  function drawAnchors(): void {
    const s = st.size, shapes: [number, number][] = [[s, s], [s * Math.SQRT2, s / Math.SQRT2], [s / Math.SQRT2, s * Math.SQRT2]].slice(0, st.shapes) as [number, number][];
    const anchors = anchorGrid(N, N, st.stride, shapes);
    for (const a of anchors) if (a.shape === 0) svgEl("circle", { cx: (a.x0 + a.x1) / 2, cy: (a.y0 + a.y1) / 2, r: 0.7, fill: "#ffd23f" }, svg);
    const best = bestAnchors(scene.truth, anchors);
    let ok = 0;
    scene.truth.forEach((t, k) => {
      const b = best[k], good = b.iou >= 0.5;
      if (good) ok++;
      rect(t, { stroke: good ? COL_KEPT : COL_BAD, "stroke-width": 1.1 });
      if (b.index >= 0) rect(anchors[b.index], { stroke: "#ffd23f", "stroke-width": 0.6, "stroke-dasharray": "1.5 1" });
      label(t.x0 + 0.5, t.y0 - 1, b.iou.toFixed(2));
    });
    ro.textContent = `${anchors.length} anchors (${anchors.length / shapes.length} positions × ${shapes.length} shape${shapes.length > 1 ? "s" : ""})\n` +
      `objects whose best anchor has IoU ≥ 0.5: ${ok} / ${scene.truth.length}\n` +
      `(green: can be learned; red: no anchor matches)`;
    note.textContent = "Yellow dots: anchor centres. Dashed yellow: the best anchor of each object, with its IoU.";
  }

  // ---- heat-map mode ----
  function drawHeat(): void {
    const H = centreHeatmap(scene.truth, st.sigma, st.noise, 5);
    const rgba = scalarToRGBA(H, { map: inferno, lo: 0, hi: 1 });
    const base = toRGBA(scene.image, { lo: 0, hi: 255 });
    for (let i = 0; i < N * N; i++) for (let c = 0; c < 3; c++) rgba[4 * i + c] = 0.75 * rgba[4 * i + c] + 0.25 * base[4 * i + c];
    cv.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), N, N), 0, 0);
    const peaks = localMaxima(H, N, N, st.win, st.peakMin);
    const boxes = boxesFromPeaks(peaks, scene.truth);
    boxes.forEach(b => rect(b, { stroke: "#7fff7f", "stroke-width": 0.8 }));
    for (const p of peaks) {
      const x = (p % N) + 0.5, y = Math.floor(p / N) + 0.5;
      svgEl("path", { d: `M${x - 2},${y} h4 M${x},${y - 2} v4`, stroke: "#fff", "stroke-width": 0.8 }, svg);
    }
    const T = boxIouTable(scene.truth, boxes), m = match(T, 0.5, "optimal"), sc = objectScores(m);
    ro.textContent = `peaks (maxima of their ${2 * st.win + 1}×${2 * st.win + 1} window, ≥ ${st.peakMin.toFixed(2)}): ${peaks.length}\n` +
      `true cells: ${scene.truth.length}\nTP ${m.tp}  FP ${m.fp}  FN ${m.fn}   F1 ${sc.f1.toFixed(2)}  (τ = 0.5)`;
    note.textContent = "Colour: the simulated centre heat map. White crosses: local maxima; green: their boxes (size from the simulated size output).";
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
