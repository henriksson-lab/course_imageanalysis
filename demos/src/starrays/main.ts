// Star-convex polygons (networks for cells and objects day, unit stardist/). Analytic, no network.
// Two modes.
//  - Shapes: a shape on a 64 × 64 pixel grid (presets: disc, ellipse 2:1, rectangle 8:4, L, C,
//    ring, bent rod, five-pointed star; editable and drawable with lib/shapecanvas.ts), its
//    objectness map, a draggable centre with n rays (4–64) to their first exit, the polygon
//    through the ray tips, its area (triangle rule) against the shape's, IoU, missed pixels
//    (red hatch) and extra pixels (blue hatch); "best centre" searches every pixel for the
//    centre with the largest IoU.
//  - Decoding: a synthetic 128 × 128 nuclei scene with ground truth and a SIMULATED network
//    output (exact objectness and rays + noise, scene.ts); probability threshold → one polygon
//    per grid cell → polygon NMS (step through it: kept polygon yellow, the ones it deletes red
//    with their IoU) → labels, scored at τ = 0.5 with the evaluation day's matching.
// Auto-mounts on <div data-demo="starrays">. Options: data-mode="shape|decode",
// data-preset="corners8|rays32|cshape|lshape|star|touching|strong|weak|low|high|step".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, draggable, type Slider } from "../lib/ui";
import { shapeCanvas, type Pt } from "../lib/shapecanvas";
import { SHAPES } from "./shapes";
import { toRGBA } from "../lib/imageops";
import { scalarToRGBA, viridis, labelLUT } from "../lib/colour";
import * as G from "./geom";
import * as S from "./scene";

const NAME = "starrays";
const SZ = 64;          // shape grid
const SN = S.SN;        // decode scene
let uid = 0;

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-direction: column; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .sr-modes { display: flex; gap: .35em; align-items: center; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .sr-body { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; align-self: stretch; width: 100%; }
.qb-demo[data-demo="${NAME}"] .sr-hide { display: none !important; }
.qb-demo[data-demo="${NAME}"] .sr-shape-left { flex: 1 1 420px; max-width: 480px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .sr-dec-left { flex: 1 1 620px; min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sr-panels { display: flex; gap: .5em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] .sr-stage { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .sr-stage canvas, .qb-demo[data-demo="${NAME}"] .sr-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .sr-stage canvas { image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .sr-stage svg { overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .sr-sim { font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 330px; max-width: 470px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .sr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] details summary { cursor: pointer; font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] svg text.sr-t { font-weight: 700; paint-order: stroke; stroke: #000; stroke-linejoin: round; fill: #fff; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .2em; flex-basis: 400px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note, .reveal .qb-demo[data-demo="${NAME}"] .sr-lab, .reveal .qb-demo[data-demo="${NAME}"] details { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .sr-shape-left { flex: 0 0 470px; max-width: 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .sr-dec-left { flex: 0 1 700px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] figure figcaption { font-size: 18px; color: var(--qb-fg); }
.reveal .qb-demo[data-demo="${NAME}"] .sr-sim { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// ---------------------------------------------------------------------------------------
// Shapes on the 64 × 64 grid

interface ShapePreset { id: string; label: string; shape: string; n: number; centre?: Pt | "best" | "middle" }
const SHAPE_PRESETS: ShapePreset[] = [
  { id: "corners8", label: "corners cut (8 rays)", shape: "rect", n: 8, centre: [32, 32] },
  { id: "rays32", label: "32 rays", shape: "rect", n: 32, centre: [32, 32] },
  { id: "cshape", label: "C: no centre works", shape: "C", n: 32, centre: "best" },
  { id: "lshape", label: "L: centre matters", shape: "L", n: 32, centre: [19.5, 12.5] },
  { id: "star", label: "star: polygon sticks out", shape: "star", n: 5, centre: [32, 32] },
];

interface DecPreset { id: string; label: string; prob?: number; nms?: number; step?: boolean }
const DEC_PRESETS: DecPreset[] = [
  { id: "touching", label: "touching pair survives" },
  { id: "strong", label: "NMS too strong: neighbour deleted", nms: 0.02 },
  { id: "weak", label: "NMS too weak: duplicates", nms: 0.8 },
  { id: "low", label: "low threshold: specks", prob: 0.3 },
  { id: "high", label: "high threshold: dim nucleus lost", prob: 0.7 },
  { id: "step", label: "step through NMS", step: true },
];
const D0 = { prob: 0.5, nms: 0.3, n: 32, noise: 0.3, contact: 0.2 };
const RAY_CHOICES = [4, 8, 16, 32, 64];
const nearestRays = (v: number) => RAY_CHOICES.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));

function mount(host: HTMLElement): void {
  injectStyle();
  const id = ++uid;
  const modeRow = htmlEl("div", "sr-modes", "", host);
  htmlEl("span", "sr-lab", "mode:", modeRow).style.marginBottom = "0";
  const mShape = button(modeRow, "shapes: rays and polygon", () => setMode("shape"));
  const mDec = button(modeRow, "decoding: threshold + NMS", () => setMode("decode"));
  const shapeBody = htmlEl("div", "sr-body", "", host);
  const decBody = htmlEl("div", "sr-body", "", host);

  // =====================================================================================
  // Shape mode
  const sLeft = htmlEl("div", "sr-shape-left", "", shapeBody);
  const sPanel = htmlEl("div", "qb-panel", "", shapeBody);
  const sh = { n: 32, cx: 32, cy: 32, shape: "rect", showObj: false, best: null as number | null };
  htmlEl("div", "sr-lab", "presets", sPanel);
  const sPresetBtns = presets(sPanel, SHAPE_PRESETS.map(p => ({ label: p.label, apply: () => applyShapePreset(p) })));
  htmlEl("div", "sr-lab", "shape", sPanel);
  const shapeRow = htmlEl("div", "qb-presets", "", sPanel);
  const shapeBtns = SHAPES.map(s => button(shapeRow, s.label, () => { setShape(s.id); centreMiddle(); update(); }));
  const nS: Slider = slider(sPanel, "rays n", 4, 64, 1, 32, v => { sh.n = v; sh.best = null; update(); });
  const cRow = htmlEl("div", "qb-presets", "", sPanel);
  button(cRow, "best centre", () => { const f = G.bestCentre(canvas.mask(), SZ, SZ, sh.n); if (f) { sh.cx = f.cx; sh.cy = f.cy; sh.best = f.iou; } update(); });
  button(cRow, "most central pixel", () => { centreMiddle(); update(); });
  checkbox(cRow, "objectness", false, v => { sh.showObj = v; update(); });
  const sRead = htmlEl("div", "qb-readout", "", sPanel);
  htmlEl("div", "qb-note", "Drag the yellow centre. Red hatch: shape pixels the polygon misses; blue hatch: polygon pixels outside the shape.", sPanel);
  const det = htmlEl("details", "", "", sPanel);
  htmlEl("summary", "", "edit or draw the shape", det);
  const canvas = shapeCanvas(sLeft, { size: SZ, view: 480, controlsParent: det, ariaLabel: "a shape with the rays from one centre and the polygon through their tips",
    onChange: () => { sh.best = null; update(); } });
  canvas.selected = -1;

  // overlay (pixel units)
  const defs = svgEl("defs", {}, canvas.svg);
  const hatch = (hid: string, col: string, ang: number) => {
    const p = svgEl("pattern", { id: hid, patternUnits: "userSpaceOnUse", width: 1, height: 1, patternTransform: `rotate(${ang})` }, defs);
    svgEl("rect", { width: 1, height: 1, fill: col, opacity: 0.25 }, p);
    svgEl("line", { x1: 0, y1: 0.5, x2: 1, y2: 0.5, stroke: col, "stroke-width": 0.35 }, p);
  };
  const hidMiss = `sr-miss-${id}`, hidExtra = `sr-extra-${id}`;
  hatch(hidMiss, "#d62728", 45); hatch(hidExtra, "#1f77b4", -45);
  const ov = canvas.overlay;
  const objImg = svgEl("image", { x: 0, y: 0, width: SZ, height: SZ, preserveAspectRatio: "none", opacity: 0.85 }, ov);
  objImg.style.setProperty("image-rendering", "pixelated");
  const missPath = svgEl("path", { fill: `url(#${hidMiss})` }, ov);
  const extraPath = svgEl("path", { fill: `url(#${hidExtra})` }, ov);
  const rayPath = svgEl("path", { stroke: "#b8860b", "stroke-width": 0.18, fill: "none" }, ov);
  const polyPath = svgEl("path", { stroke: "#d62728", "stroke-width": 0.4, fill: "none", "stroke-linejoin": "round" }, ov);
  const tipsG = svgEl("g", { fill: "#d62728" }, ov);
  const handle = svgEl("circle", { r: 1.6, fill: "#ffd400", stroke: "#000", "stroke-width": 0.3 }, ov);
  handle.addEventListener("pointerdown", ev => ev.stopPropagation());
  draggable(handle, ev => { const [x, y] = svgPoint(canvas.svg, ev); const k = 480 / SZ; return [x / k, y / k]; }, (x, y) => {
    sh.cx = Math.max(0.01, Math.min(SZ - 0.01, x)); sh.cy = Math.max(0.01, Math.min(SZ - 0.01, y)); sh.best = null; update();
  });
  const objCanvas = document.createElement("canvas"); objCanvas.width = SZ; objCanvas.height = SZ;

  function setShape(sid: string): void {
    sh.shape = sid; sh.best = null;
    canvas.setPolys((SHAPES.find(s => s.id === sid) ?? SHAPES[0]).polys());
    canvas.selected = -1;
    shapeBtns.forEach((b, i) => b.classList.toggle("active", SHAPES[i].id === sid));
  }
  function centreMiddle(): void {
    const m = canvas.mask(), o = G.objectness(m, SZ, SZ);
    let bi = -1, bv = -1;
    for (let i = 0; i < o.length; i++) if (o[i] > bv + 1e-9) { bv = o[i]; bi = i; }
    if (bi >= 0) { sh.cx = (bi % SZ) + 0.5; sh.cy = Math.floor(bi / SZ) + 0.5; }
    sh.best = null;
  }
  function applyShapePreset(p: ShapePreset): void {
    setShape(p.shape); sh.n = p.n; nS.set(p.n);
    if (p.centre === "best") { const f = G.bestCentre(canvas.mask(), SZ, SZ, sh.n); if (f) { sh.cx = f.cx; sh.cy = f.cy; sh.best = f.iou; } }
    else if (p.centre === "middle" || !p.centre) centreMiddle();
    else { sh.cx = p.centre[0]; sh.cy = p.centre[1]; }
    update();
    sPresetBtns.forEach((b, i) => b.classList.toggle("active", SHAPE_PRESETS[i] === p));
  }
  const pixPath = (idx: number[]) => idx.map(p => `M${p % SZ},${Math.floor(p / SZ)}h1v1h-1z`).join("");

  function update(): void {
    const mask = canvas.mask();
    const inside = mask[Math.floor(sh.cy) * SZ + Math.floor(sh.cx)] === 1;
    const f = G.fitAt(mask, SZ, SZ, sh.cx, sh.cy, sh.n);
    const inPoly = new Uint8Array(SZ * SZ); for (const p of f.pix) inPoly[p] = 1;
    const miss: number[] = [], extra: number[] = [];
    let area = 0;
    for (let p = 0; p < SZ * SZ; p++) { if (mask[p]) area++; if (mask[p] && !inPoly[p]) miss.push(p); if (!mask[p] && inPoly[p]) extra.push(p); }
    missPath.setAttribute("d", pixPath(miss));
    extraPath.setAttribute("d", pixPath(extra));
    const tips = G.polyVertices(sh.cx, sh.cy, f.rays);
    rayPath.setAttribute("d", inside ? tips.map(([x, y]) => `M${sh.cx},${sh.cy}L${x},${y}`).join("") : "");
    polyPath.setAttribute("d", inside ? "M" + tips.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L") + "Z" : "");
    tipsG.replaceChildren();
    if (inside && sh.n <= 32) for (const [x, y] of tips) svgEl("circle", { cx: x, cy: y, r: 0.45 }, tipsG);
    handle.setAttribute("cx", String(sh.cx)); handle.setAttribute("cy", String(sh.cy));
    if (sh.showObj) {
      const o = G.objectness(mask, SZ, SZ);
      const rgba = scalarToRGBA(o, { map: viridis, lo: 0, hi: 1 });
      for (let p = 0; p < SZ * SZ; p++) if (!mask[p]) rgba[4 * p + 3] = 0;
      objCanvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), SZ, SZ), 0, 0);
      objImg.setAttribute("href", objCanvas.toDataURL());
    } else objImg.removeAttribute("href");
    const lines: string[] = [];
    lines.push(`centre (${sh.cx.toFixed(1)}, ${sh.cy.toFixed(1)}), ${sh.n} rays`);
    if (!inside) { lines.push("the centre is outside the shape: no rays"); sRead.textContent = lines.join("\n"); return; }
    if (sh.n <= 8) lines.push("ray lengths (px): " + Array.from(f.rays, r => r.toFixed(1)).join(", "));
    lines.push(`shape ${area} px · polygon ${f.pix.length} px (triangle rule ${G.polyArea(f.rays).toFixed(1)})`);
    lines.push(`IoU ${(f.iou).toFixed(3)} · missed ${f.missed} px · extra ${f.extra} px`);
    if (sh.best !== null) lines.push(`best centre of all pixels: IoU ${sh.best.toFixed(3)}`);
    sRead.textContent = lines.join("\n");
  }

  // =====================================================================================
  // Decode mode
  const dLeft = htmlEl("div", "sr-dec-left", "", decBody);
  const dPanels = htmlEl("div", "sr-panels", "", dLeft);
  const mk = (cap: string) => {
    const f = htmlEl("figure", "", "", dPanels);
    const c = htmlEl("figcaption", "", cap, f);
    const stg = htmlEl("div", "sr-stage", "", f);
    const cv = htmlEl("canvas", "", "", stg); cv.width = SN; cv.height = SN;
    const svg = svgEl("svg", { viewBox: `0 0 ${SN} ${SN}` }, stg);
    return { cap: c, cv, svg };
  };
  const pA = mk("objectness + polygons"), pB = mk("labels vs truth (τ = 0.5)");
  htmlEl("div", "sr-sim", "Simulated network output: the exact training targets plus the noise you set. Dashed white: true outlines.", dLeft);
  const dRead = htmlEl("div", "qb-readout", "", dLeft);
  const dPanel = htmlEl("div", "qb-panel", "", decBody);
  const dc = { ...D0, step: -1 };   // step -1 = NMS finished
  htmlEl("div", "sr-lab", "presets", dPanel);
  const dPresetBtns = presets(dPanel, DEC_PRESETS.map(p => ({ label: p.label, apply: () => applyDecPreset(p) })));
  const probS = slider(dPanel, "probability threshold", 0.05, 0.95, 0.01, D0.prob, v => { dc.prob = v; dc.step = -1; runDecode(); });
  const nmsS = slider(dPanel, "NMS overlap threshold", 0, 0.95, 0.01, D0.nms, v => { dc.nms = v; dc.step = -1; runDecode(); });
  const raysS = slider(dPanel, "rays n", 4, 64, 1, D0.n, v => { dc.n = nearestRays(v); dc.step = -1; runDecode(); }, v => String(nearestRays(v)));
  const noiseS = slider(dPanel, "network noise", 0, 1, 0.05, D0.noise, v => { dc.noise = v; dc.step = -1; runDecode(); });
  const contS = slider(dPanel, "contact blur", 0, 0.5, 0.05, D0.contact, v => { dc.contact = v; dc.step = -1; runDecode(); });
  const stepRow = htmlEl("div", "qb-presets", "", dPanel);
  button(stepRow, "NMS: restart", () => { dc.step = 0; drawDecode(); });
  button(stepRow, "step", () => { if (dc.step < 0) dc.step = 0; else dc.step = Math.min(dec.nms.steps.length, dc.step + 1); if (dc.step === dec.nms.steps.length) dc.step = -1; drawDecode(); });
  button(stepRow, "finish", () => { dc.step = -1; drawDecode(); });
  button(stepRow, "reset", () => applyDecPreset(DEC_PRESETS[0]));
  htmlEl("div", "qb-note", "Left: candidates above the threshold (thin white), the polygon just kept (yellow), the polygons it deletes (red, with their IoU). Right: kept polygons as labels; FP = no true nucleus matched, FN = nucleus not found.", dPanel);

  let sim: S.Sim, dec: S.Decoded;
  const sc = S.makeScene();

  function applyDecPreset(p: DecPreset): void {
    dc.prob = p.prob ?? D0.prob; dc.nms = p.nms ?? D0.nms; dc.n = D0.n; dc.noise = D0.noise; dc.contact = D0.contact;
    probS.set(dc.prob); nmsS.set(dc.nms); raysS.set(dc.n); noiseS.set(dc.noise); contS.set(dc.contact);
    dc.step = p.step ? 0 : -1;
    runDecode();
    dPresetBtns.forEach((b, i) => b.classList.toggle("active", DEC_PRESETS[i] === p));
  }
  function runDecode(): void {
    sim = S.simulate(dc.n, dc.noise, dc.contact);
    dec = S.decode(sim, dc.prob, dc.nms);
    drawDecode();
  }
  const polyD = (c: G.Candidate) => "M" + G.polyVertices(c.cx, c.cy, c.rays).map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L") + "Z";
  const lut = labelLUT(64, [0, 0, 0], 255);
  const labCss = (k: number) => `rgb(${lut[4 * (k % 64)]},${lut[4 * (k % 64) + 1]},${lut[4 * (k % 64) + 2]})`;
  function truthOutline(svg: SVGSVGElement): void {
    let d = "";
    for (let y = 0; y < SN; y++) for (let x = 0; x < SN; x++) {
      const p = y * SN + x, l = sc.truth[p];
      if (!l) continue;
      if (x === SN - 1 || sc.truth[p + 1] !== l) d += `M${x + 1},${y}v1`;
      if (x === 0 || sc.truth[p - 1] !== l) d += `M${x},${y}v1`;
      if (y === SN - 1 || sc.truth[p + SN] !== l) d += `M${x},${y + 1}h1`;
      if (y === 0 || sc.truth[p - SN] !== l) d += `M${x},${y}h1`;
    }
    svgEl("path", { d, stroke: "#fff", "stroke-width": 0.35, "stroke-dasharray": "1 1", fill: "none", opacity: 0.8 }, svg);
  }

  function drawDecode(): void {
    // ---- left: objectness + candidates / NMS state
    pA.cv.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(scalarToRGBA(sim.prob, { map: viridis, lo: 0, hi: 1 })), SN, SN), 0, 0);
    const svg = pA.svg; svg.replaceChildren();
    const steps = dec.nms.steps, nSteps = steps.length, s = dc.step < 0 ? nSteps : dc.step;
    const cand = (i: number) => sim.cands[dec.above[i]];
    const state = new Int8Array(dec.above.length);   // 0 alive, 1 kept, 2 removed earlier, 3 kept now, 4 removed now
    for (let k = 0; k < s; k++) {
      state[steps[k].kept] = k === s - 1 && dc.step >= 0 ? 3 : 1;
      for (const r of steps[k].removed) state[r.j] = k === s - 1 && dc.step >= 0 ? 4 : 2;
    }
    let alive = "";
    if (dc.step >= 0) for (let i = 0; i < state.length; i++) if (state[i] === 0) alive += polyD(cand(i));
    if (dc.step < 0 && dec.above.length <= 1500) for (let i = 0; i < state.length; i++) if (state[i] === 2) alive += polyD(cand(i));
    svgEl("path", { d: alive, stroke: "#fff", "stroke-width": 0.2, fill: "none", opacity: dc.step >= 0 ? 0.45 : 0.18 }, svg);
    for (let k = 0; k < s; k++) {
      const c = cand(steps[k].kept), now = k === s - 1 && dc.step >= 0;
      svgEl("path", { d: polyD(c), stroke: now ? "#ffd400" : labCss(k + 1), "stroke-width": now ? 1 : 0.7, fill: "none" }, svg);
    }
    if (dc.step > 0) {
      const st = steps[s - 1];
      let rd = "";
      for (const r of st.removed) rd += polyD(cand(r.j));
      svgEl("path", { d: rd, stroke: "#ff3b30", "stroke-width": 0.35, fill: "none", "stroke-dasharray": "1.2 0.8" }, svg);
      // IoU labels only on deleted polygons centred in ANOTHER object than the kept one (a deleted neighbour)
      const own = (c: G.Candidate) => sc.objects[Math.floor(c.cy) * SN + Math.floor(c.cx)];
      const ko = own(cand(st.kept));
      for (const r of st.removed.filter(r => own(cand(r.j)) !== ko)) { const c = cand(r.j); const t = svgEl("text", { x: c.cx, y: c.cy, "font-size": 4.5, class: "sr-t", "text-anchor": "middle", "stroke-width": 1 }, svg); t.textContent = r.iou.toFixed(2); }
    }
    pA.cap.textContent = dc.step >= 0 ? `NMS step ${s} of ${nSteps}` : "objectness + kept polygons";
    // ---- right: labels on the image, truth outlines, FP/FN marks
    const base = toRGBA({ width: SN, height: SN, data: sc.image }, { lo: 0, hi: 230 });
    const out = new Uint8ClampedArray(4 * SN * SN);
    for (let i = 0; i < SN * SN; i++) {
      const l = dec.labels[i];
      for (let c = 0; c < 3; c++) out[4 * i + c] = l ? 0.6 * lut[4 * (l % 64) + c] + 0.4 * base[4 * i + c] : 0.55 * base[4 * i + c];
      out[4 * i + 3] = 255;
    }
    pB.cv.getContext("2d")!.putImageData(new ImageData(out, SN, SN), 0, 0);
    const sv = pB.svg; sv.replaceChildren();
    truthOutline(sv); truthOutline(svg);
    dec.kept.forEach((ci, j) => {
      if (dec.matchOf[j]) return;
      const c = sim.cands[ci];
      svgEl("path", { d: polyD(c), stroke: "#ff9f1c", "stroke-width": 0.7, fill: "none", "stroke-dasharray": "1.5 1" }, sv);
      const t = svgEl("text", { x: c.cx, y: c.cy + 2, "font-size": 6, class: "sr-t", "text-anchor": "middle", "stroke-width": 1.2 }, sv); t.textContent = "FP";
    });
    for (const l of dec.missed) {
      let x = 0, y = 0, n = 0;
      for (let p = 0; p < SN * SN; p++) if (sc.truth[p] === l) { x += p % SN; y += Math.floor(p / SN); n++; }
      const t = svgEl("text", { x: x / n + 0.5, y: y / n + 2.5, "font-size": 6, class: "sr-t", "text-anchor": "middle", "stroke-width": 1.2 }, sv); t.textContent = "FN";
    }
    // ---- readout
    const sc_ = dec.scores, lines: string[] = [];
    lines.push(`grid cells above ${dc.prob.toFixed(2)}: ${dec.above.length} candidates (of ${(SN / S.GRID) ** 2})`);
    lines.push(`after NMS at ${dc.nms.toFixed(2)}: ${dec.kept.length} polygons · IoUs computed ${dec.nms.iouComputed}`);
    if (dc.step > 0) {
      const st = steps[s - 1], ious = st.removed.map(r => r.iou);
      lines.push(`step ${s}: kept prob ${cand(st.kept).score.toFixed(2)}, deleted ${st.removed.length}` + (ious.length ? ` (IoU ${Math.min(...ious).toFixed(2)}–${Math.max(...ious).toFixed(2)})` : ""));
    } else if (dc.step === 0) lines.push("step 0: press step");
    lines.push(`vs truth (${sc.nTrue} nuclei, τ = 0.5): TP ${sc_.tp} · FP ${sc_.fp} · FN ${sc_.fn} · F1 ${sc_.f1.toFixed(2)}`);
    dRead.textContent = lines.join("\n");
  }

  // =====================================================================================
  function setMode(m: "shape" | "decode"): void {
    shapeBody.classList.toggle("sr-hide", m !== "shape");
    decBody.classList.toggle("sr-hide", m !== "decode");
    mShape.classList.toggle("active", m === "shape");
    mDec.classList.toggle("active", m === "decode");
  }

  const pre = host.dataset.preset ?? "";
  const sp = SHAPE_PRESETS.find(p => p.id === pre), dp = DEC_PRESETS.find(p => p.id === pre);
  applyShapePreset(sp ?? SHAPE_PRESETS[0]);
  applyDecPreset(dp ?? DEC_PRESETS[0]);
  setMode(dp ? "decode" : sp ? "shape" : host.dataset.mode === "decode" ? "decode" : "shape");
}

autoMount(NAME, mount);
