// 2x2 matrix playground. Auto-mounts on every <div data-demo="matrix2x2">.
//
// Attributes (all optional):
//   data-mode="transform" | "order" | "eigen" | "image"   start mode (default transform)
//   data-modes="transform order"   show tabs to switch between these modes
//   data-a="1 0 0 1"   matrix A, row-major (a b c d)
//   data-b="0 -1 1 0"  matrix B (order mode)
//   data-x="1 1"       test vector x (transform mode)
//   data-v="20"        starting angle of v in degrees (eigen mode)
//
// transform: drag the column tips (where (1,0) and (0,1) land); the grid, unit square,
//            a house and a test vector x follow.
// order:     two matrices A and B; AB (B first) next to BA (A first).
// eigen:     a unit vector v you turn; v and Av drawn; Av on v's line => eigenvector.
// image:     A applied to a small synthetic cell image, around the image centre.

import {
  svgEl, htmlEl, slider, button, presets, checkbox, draggable, svgPoint, numList, autoMount, setAttrs,
  Slider,
} from "../lib/ui";
import {
  M2, ID, apply, mul, areaFactor, rotation, same, snap, fmt, eigenDirections, lineGap, EigDir,
} from "./geom";
import { H, makeCellImage, renderImage } from "./image";

type Mode = "transform" | "order" | "eigen" | "image";
const MODES: Mode[] = ["transform", "order", "eigen", "image"];
const TAB_LABEL: Record<Mode, string> = {
  transform: "One matrix", order: "Order: AB vs BA", eigen: "Eigen-directions", image: "Image",
};

const COL1 = "#d62728", COL2 = "#1f77b4", XCOL = "#555", AXCOL = "#2ca02c";
const VCOL = "#222", AVCOL = "#ff7f0e", EIGCOL = "#2ca02c";

// ---------------------------------------------------------------------------------------
// Scoped styles (injected once).
const CSS = `
.qb-demo[data-demo="matrix2x2"] .mx-plot { flex: 1 1 320px; min-width: 0; max-width: 520px; position: relative; }
.qb-demo[data-demo="matrix2x2"] .mx-plot > svg, .qb-demo[data-demo="matrix2x2"] .mx-plot > canvas { width: 100%; height: auto; display: block; }
.qb-demo[data-demo="matrix2x2"] .mx-plot > svg.mx-over { position: absolute; left: 0; top: 0; height: 100%; }
.qb-demo[data-demo="matrix2x2"] .mx-two { flex: 1 1 100%; max-width: none; display: flex; gap: .8em; }
.qb-demo[data-demo="matrix2x2"] .mx-two > svg { flex: 1 1 0; min-width: 0; max-width: 400px; height: auto; }
.qb-demo[data-demo="matrix2x2"] .qb-panel { flex: 1 1 280px; max-width: 560px; gap: .45em; }
.qb-demo[data-demo="matrix2x2"] .mx-tabs .qb-btn { font-weight: 600; }
.qb-demo[data-demo="matrix2x2"] .mx-sliders { display: grid; grid-template-columns: 1fr; gap: .25em .9em; }
.qb-demo[data-demo="matrix2x2"] .mx-sliders .qb-slider { grid-template-columns: 1em 1fr 3.2em; }
.qb-demo[data-demo="matrix2x2"] .mx-sliders .qb-slider-name { font-weight: 700; font-family: ui-monospace, monospace; }
.qb-demo[data-demo="matrix2x2"] .mx-row { display: flex; align-items: center; gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="matrix2x2"] .mx-mat { display: inline-flex; align-items: center; gap: .35em; font-family: ui-monospace, monospace; font-size: 1.1em; font-weight: 600; }
.qb-demo[data-demo="matrix2x2"] .qb-readout { font-size: 1em; color: #222; }
.qb-demo[data-demo="matrix2x2"] .mx-br { display: inline-grid; grid-template-columns: auto auto; column-gap: .9em;
  padding: .05em .4em; border-left: 2px solid currentColor; border-right: 2px solid currentColor;
  border-radius: 7px; text-align: right; line-height: 1.25; }
.qb-demo[data-demo="matrix2x2"] .mx-lab { font-size: .9em; font-weight: 600; }
.qb-demo[data-demo="matrix2x2"] .mx-status { font-weight: 600; min-height: 1.35em; }
.qb-demo[data-demo="matrix2x2"] .mx-ok { color: ${EIGCOL}; }
.qb-demo[data-demo="matrix2x2"] svg text.mx-t { font-size: 20px; font-weight: 600; paint-order: stroke;
  stroke: #fff; stroke-width: 5px; stroke-linejoin: round; }
.qb-demo[data-demo="matrix2x2"] svg .mx-gridT { stroke: #9ecae1; stroke-width: 1.2; }
.reveal .qb-demo[data-demo="matrix2x2"] { flex-wrap: nowrap; gap: 1.1em; }
.reveal .qb-demo[data-demo="matrix2x2"] .mx-plot { flex: 0 0 520px; width: 520px; max-width: 520px; }
.reveal .qb-demo[data-demo="matrix2x2"] .mx-two { flex: 0 0 750px; width: 750px; max-width: 750px; }
.reveal .qb-demo[data-demo="matrix2x2"] .mx-two > svg { max-width: 370px; }
.reveal .qb-demo[data-demo="matrix2x2"] .qb-panel { font-size: 20px; max-width: none; }
.reveal .qb-demo[data-demo="matrix2x2"] .mx-sliders { grid-template-columns: 1fr 1fr; }
.reveal .qb-demo[data-demo="matrix2x2"] .qb-presets .qb-btn { padding: .2em .55em; }
@media (max-width: 640px) { .qb-demo[data-demo="matrix2x2"] .mx-two { flex-direction: column; } }
`;
let styled = false;
function injectStyle(): void {
  if (styled) return;
  styled = true;
  const st = document.createElement("style");
  st.textContent = CSS;
  document.head.appendChild(st);
}

let uid = 0;

// ---------------------------------------------------------------------------------------
// A square SVG plot with world coordinates [-R, R]^2, y up.
interface Plot {
  svg: SVGSVGElement; k: number; size: number;
  X(x: number): number; Y(y: number): number;
  world(ev: PointerEvent): [number, number];
  layer(): SVGGElement;
  marker(color: string): string;
}

function makePlot(parent: Element, size: number, R: number, overlay = false): Plot {
  const svg = svgEl("svg", { viewBox: `0 0 ${size} ${size}`, class: overlay ? "mx-over" : "" }, parent);
  const k = size / (2 * R);
  const defs = svgEl("defs", {}, svg);
  const id = `mx${++uid}`;
  const markers = new Map<string, string>();
  const p: Plot = {
    svg, k, size,
    X: x => size / 2 + x * k,
    Y: y => size / 2 - y * k,
    world: ev => { const [px, py] = svgPoint(svg, ev); return [(px - size / 2) / k, (size / 2 - py) / k]; },
    layer: () => svgEl("g", {}, svg),
    marker: color => {
      let m = markers.get(color);
      if (m) return m;
      m = `${id}-m${markers.size}`;
      const mk = svgEl("marker", { id: m, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 4.2, markerHeight: 4.2, orient: "auto-start-reverse", markerUnits: "strokeWidth" }, defs);
      svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: color }, mk);
      markers.set(color, m);
      return m;
    },
  };
  return p;
}

function drawGrid(p: Plot, R: number, axes = true): void {
  const g = p.layer();
  const n = Math.floor(R);
  for (let i = -n; i <= n; i++) {
    svgEl("line", { x1: p.X(i), y1: p.Y(-R), x2: p.X(i), y2: p.Y(R), class: "grid" }, g);
    svgEl("line", { x1: p.X(-R), y1: p.Y(i), x2: p.X(R), y2: p.Y(i), class: "grid" }, g);
  }
  if (axes) {
    svgEl("line", { x1: p.X(-R), y1: p.Y(0), x2: p.X(R), y2: p.Y(0), class: "axis", "stroke-width": 1.5 }, g);
    svgEl("line", { x1: p.X(0), y1: p.Y(-R), x2: p.X(0), y2: p.Y(R), class: "axis", "stroke-width": 1.5 }, g);
  }
}

/** Grid lines x = i and y = i pushed through A. */
function transformedGrid(p: Plot, g: SVGGElement, n: number) {
  const lines: SVGLineElement[] = [];
  for (let i = -n; i <= n; i++) for (let dir = 0; dir < 2; dir++) {
    lines.push(svgEl("line", { class: i === 0 ? "" : "mx-gridT", stroke: i === 0 ? "#6baed6" : "", "stroke-width": i === 0 ? 2 : 1.2 }, g));
  }
  return (A: M2) => {
    let j = 0;
    for (let i = -n; i <= n; i++) for (let dir = 0; dir < 2; dir++) {
      const [x0, y0] = dir ? apply(A, i, -n * 3) : apply(A, -n * 3, i);
      const [x1, y1] = dir ? apply(A, i, n * 3) : apply(A, n * 3, i);
      setAttrs(lines[j++], { x1: p.X(x0), y1: p.Y(y0), x2: p.X(x1), y2: p.Y(y1) });
    }
  };
}

/** An arrow from (x0,y0) to (x1,y1) in world coordinates. */
function arrow(p: Plot, g: Element, color: string, width = 4, dash = "") {
  const l = svgEl("line", { stroke: color, "stroke-width": width, "stroke-linecap": "round", "marker-end": `url(#${p.marker(color)})` }, g);
  if (dash) l.setAttribute("stroke-dasharray", dash);
  const set = (x0: number, y0: number, x1: number, y1: number) => {
    const tiny = Math.hypot(x1 - x0, y1 - y0) * p.k < 4;
    setAttrs(l, { x1: p.X(x0), y1: p.Y(y0), x2: p.X(x1), y2: p.Y(y1), "marker-end": tiny ? "none" : `url(#${p.marker(color)})` });
  };
  return { el: l, set, color: (c: string) => { l.setAttribute("stroke", c); l.setAttribute("marker-end", `url(#${p.marker(c)})`); } };
}

/** A text label placed just beyond the tip of a vector, pushed away from the origin. */
function tipLabel(p: Plot, g: Element, text: string, color: string) {
  const t = svgEl("text", { class: "mx-t", fill: color, "text-anchor": "middle", "dominant-baseline": "central" }, g);
  t.textContent = text;
  t.style.fill = color;
  return (x: number, y: number, txt?: string) => {
    if (txt !== undefined) t.textContent = txt;
    const L = Math.hypot(x, y);
    const ux = L > 1e-9 ? x / L : 0.7, uy = L > 1e-9 ? y / L : 0.7;
    // beside the handle (radius 15), on the side away from the origin
    const anchor = ux > 0.35 ? "start" : ux < -0.35 ? "end" : "middle";
    let px = p.X(x) + (anchor === "start" ? 20 : anchor === "end" ? -20 : 0);
    let py = p.Y(y) - (anchor === "middle" ? Math.sign(uy || 1) * 30 : uy * 20);
    const w = t.getComputedTextLength ? t.getComputedTextLength() || 40 : 40;
    const lo = anchor === "start" ? 4 : anchor === "end" ? w + 4 : w / 2 + 4;
    const hi = anchor === "start" ? p.size - w - 4 : anchor === "end" ? p.size - 4 : p.size - w / 2 - 4;
    px = Math.min(Math.max(px, lo), Math.max(lo, hi));
    py = Math.min(Math.max(py, 16), p.size - 14);
    setAttrs(t, { x: px, y: py, "text-anchor": anchor });
  };
}

function handle(p: Plot, g: Element, color: string, r = 15): SVGCircleElement {
  const c = svgEl("circle", { r, class: "handle", stroke: color }, g);
  c.style.fill = color;
  c.style.fillOpacity = "0.22";
  return c;
}

/** A 2x2 matrix shown with brackets; column 1 red, column 2 blue. */
function matView(parent: Element, name: string, colored = true) {
  const box = htmlEl("span", "mx-mat", "", parent);
  htmlEl("span", "", name + " =", box);
  const br = htmlEl("span", "mx-br", "", box);
  const cells = [0, 1, 2, 3].map(i => {
    const c = htmlEl("span", "", "", br);
    if (colored) c.style.color = i % 2 === 0 ? COL1 : COL2;
    return c;
  });
  return (m: M2) => m.forEach((v, i) => (cells[i].textContent = fmt(v)));
}

function path(p: Plot, pts: [number, number][]): string {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${p.X(x).toFixed(1)},${p.Y(y).toFixed(1)}`).join(" ") + " Z";
}

// The house (fits in the unit square): walls, roof, chimney on the right, door on the left.
const HOUSE: [number, number][] = [
  [0.1, 0], [0.4, 0], [0.4, 0.35], [0.22, 0.35], [0.22, 0], [0.4, 0], [0.9, 0], [0.9, 0.6], [0.8, 0.7], [0.8, 0.92],
  [0.68, 0.92], [0.68, 0.81], [0.5, 0.97], [0.1, 0.6],
];
const SQUARE: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
const mapPts = (A: M2, pts: [number, number][]) => pts.map(([x, y]) => apply(A, x, y));

// ---------------------------------------------------------------------------------------
// Shared state across modes of one embed.
interface State {
  A: M2; x: [number, number]; vDeg: number;
  A0: M2; x0: [number, number]; v0: number;
  OA: M2; B: M2; OA0: M2; B0: M2;   // the two matrices of the order mode
}

type Cleanup = () => void;

/** Panel pieces shared by transform / image / eigen modes: matrix readout, a b c d sliders. */
function matrixControls(panel: Element, st: State, onChange: () => void) {
  const view = matView(panel, "A");
  const box = htmlEl("div", "mx-sliders", "", panel);
  const names = ["a", "b", "c", "d"];
  const sl: Slider[] = names.map((n, i) => {
    const s = slider(box, n, -3, 3, 0.1, st.A[i], v => { st.A[i] = v; onChange(); }, v => fmt(v));
    (s.input.parentElement!.firstElementChild as HTMLElement).style.color = i % 2 === 0 ? COL1 : COL2;
    s.input.style.accentColor = i % 2 === 0 ? COL1 : COL2;
    return s;
  });
  return {
    sync: () => { view(st.A); st.A.forEach((v, i) => sl[i].set(v)); },
  };
}

function setA(st: State, m: M2): void { for (let i = 0; i < 4; i++) st.A[i] = m[i]; }

// ---------------------------------------------------------------------------------------
// Mode: transform
function buildTransform(host: HTMLElement, panel: HTMLElement, st: State): Cleanup {
  const R = 3.5, size = 520;
  const wrap = htmlEl("div", "mx-plot", "");
  host.insertBefore(wrap, panel);
  const p = makePlot(wrap, size, R);
  drawGrid(p, R);
  const gT = p.layer();
  const gridT = transformedGrid(p, gT, 4);
  const gShapes = p.layer();
  svgEl("path", { d: path(p, SQUARE), fill: "none", stroke: "#999", "stroke-width": 2, "stroke-dasharray": "6 5" }, gShapes);
  const para = svgEl("path", { fill: "#9467bd", "fill-opacity": 0.16, stroke: "#9467bd", "stroke-width": 2 }, gShapes);
  const house = svgEl("path", { fill: "#ff7f0e", "fill-opacity": 0.35, stroke: "#b85c00", "stroke-width": 2.5, "stroke-linejoin": "round" }, gShapes);
  const gVec = p.layer();
  const cons1 = arrow(p, gVec, COL1, 2.5, "7 5"), cons2 = arrow(p, gVec, COL2, 2.5, "7 5");
  const xArr = arrow(p, gVec, XCOL, 3.5, "8 5"), axArr = arrow(p, gVec, AXCOL, 4.5);
  const c1 = arrow(p, gVec, COL1, 5), c2 = arrow(p, gVec, COL2, 5);
  const gLab = p.layer();
  const l1 = tipLabel(p, gLab, "col 1", COL1), l2 = tipLabel(p, gLab, "col 2", COL2);
  const lx = tipLabel(p, gLab, "x", XCOL), lax = tipLabel(p, gLab, "Ax", AXCOL);
  const gH = p.layer();
  const hx = handle(p, gH, XCOL, 12);
  const h1 = handle(p, gH, COL1), h2 = handle(p, gH, COL2);

  // panel
  const mc = matrixControls(panel, st, () => draw());
  const area = htmlEl("div", "", "", panel);
  const rot = slider(panel, "rotate by θ", -180, 180, 5, 0, v => { setA(st, rotation(v)); draw(); }, v => `${fmt(v)}°`);
  presets(panel, [
    { label: "Identity", apply: () => setA(st, ID) },
    { label: "Scale", apply: () => setA(st, [2, 0, 0, 0.5]) },
    { label: "Rotate 90°", apply: () => setA(st, [0, -1, 1, 0]) },
    { label: "Shear", apply: () => setA(st, [1, 0.5, 0, 1]) },
    { label: "Reflect", apply: () => setA(st, [-1, 0, 0, 1]) },
    { label: "Project", apply: () => setA(st, [1, 0, 0, 0]) },
    { label: "Collapse", apply: () => setA(st, [1, 1, 0.5, 0.5]) },
  ].map(it => ({ label: it.label, apply: () => { it.apply(); draw(); } })));
  const row = htmlEl("div", "mx-row", "", panel);
  let showX = true;
  checkbox(row, "test vector x", showX, v => { showX = v; draw(); });
  button(row, "Reset", () => { setA(st, st.A0); st.x = [...st.x0]; rot.set(0); draw(); });
  const out = htmlEl("div", "qb-readout", "", panel);

  draggable(h1, p.world, (x, y) => { st.A[0] = snap(x, 0.1); st.A[2] = snap(y, 0.1); draw(); });
  draggable(h2, p.world, (x, y) => { st.A[1] = snap(x, 0.1); st.A[3] = snap(y, 0.1); draw(); });
  draggable(hx, p.world, (x, y) => { st.x = [snap(x, 0.1), snap(y, 0.1)]; draw(); });

  function draw(): void {
    const A = st.A;
    gridT(A);
    para.setAttribute("d", path(p, mapPts(A, SQUARE)));
    house.setAttribute("d", path(p, mapPts(A, HOUSE)));
    c1.set(0, 0, A[0], A[2]); c2.set(0, 0, A[1], A[3]);
    setAttrs(h1, { cx: p.X(A[0]), cy: p.Y(A[2]) });
    setAttrs(h2, { cx: p.X(A[1]), cy: p.Y(A[3]) });
    l1(A[0], A[2]); l2(A[1], A[3]);
    const [x, y] = st.x;
    const [ax, ay] = apply(A, x, y);
    for (const e of [xArr.el, axArr.el, cons1.el, cons2.el, hx]) e.style.display = showX ? "" : "none";
    (gLab.children[2] as SVGElement).style.display = showX ? "" : "none";
    (gLab.children[3] as SVGElement).style.display = showX ? "" : "none";
    xArr.set(0, 0, x, y); axArr.set(0, 0, ax, ay);
    cons1.set(0, 0, x * A[0], x * A[2]);
    cons2.set(x * A[0], x * A[2], ax, ay);
    setAttrs(hx, { cx: p.X(x), cy: p.Y(y) });
    lx(x, y); lax(ax, ay);
    mc.sync();
    const k = areaFactor(A);
    area.innerHTML = "";
    htmlEl("span", "qb-big", Math.abs(k) < 1e-9 ? "area × 0: collapsed onto a line" :
      `area × ${fmt(Math.abs(k))}${k < 0 ? ", flipped" : ""}`, area);
    htmlEl("span", "qb-note", "  (bonus: this is det A)", area);
    out.textContent = showX
      ? `x  = (${fmt(x)}, ${fmt(y)})\nAx = ${fmt(x)}·col 1 + ${fmt(y)}·col 2 = (${fmt(ax)}, ${fmt(ay)})`
      : "Drag the red and blue handles:\nthey are the columns of A.";
  }
  draw();
  return () => { /* nothing running */ };
}

// ---------------------------------------------------------------------------------------
// Mode: order (AB vs BA)
const ORDER_PRESETS: { label: string; m: M2 }[] = [
  { label: "Identity", m: ID },
  { label: "Scale x ×2", m: [2, 0, 0, 1] },
  { label: "Rotate 90°", m: [0, -1, 1, 0] },
  { label: "Rotate 30°", m: rotation(30) },
  { label: "Shear", m: [1, 0.5, 0, 1] },
  { label: "Flip x", m: [-1, 0, 0, 1] },
];

function buildOrder(host: HTMLElement, panel: HTMLElement, st: State): Cleanup {
  const R = 2.5, size = 370;
  const wrap = htmlEl("div", "mx-plot mx-two", "");
  host.insertBefore(wrap, panel);
  const sides = [0, 1].map(side => {
    const p = makePlot(wrap, size, R);
    drawGrid(p, R);
    const g = p.layer();
    svgEl("path", { d: path(p, HOUSE), fill: "#bbb", "fill-opacity": 0.35, stroke: "#999", "stroke-width": 2 }, g);
    const mid = svgEl("path", { fill: "none", stroke: "#b85c00", "stroke-width": 2, "stroke-dasharray": "6 5" }, g);
    const fin = svgEl("path", { fill: "#ff7f0e", "fill-opacity": 0.4, stroke: "#b85c00", "stroke-width": 2.5, "stroke-linejoin": "round" }, g);
    const t1 = svgEl("text", { x: 12, y: 28, class: "mx-t", fill: "#222" }, p.svg);
    const t2 = svgEl("text", { x: 12, y: 54, class: "mx-t", fill: "#666" }, p.svg);
    t1.style.fill = "#222"; t2.style.fill = "#666"; t2.style.fontWeight = "400";
    t1.textContent = side === 0 ? "AB" : "BA";
    t2.textContent = side === 0 ? "first B, then A" : "first A, then B";
    return { p, mid, fin };
  });

  const ba = presets(panel, ORDER_PRESETS.map(it => ({ label: it.label, apply: () => { st.OA = [...it.m] as M2; draw(); } })));
  const bb = presets(panel, ORDER_PRESETS.map(it => ({ label: it.label, apply: () => { st.B = [...it.m] as M2; draw(); } })));
  for (const [btns, name] of [[ba, "A:"], [bb, "B:"]] as const) {
    const row = btns[0].parentElement!;
    row.insertBefore(htmlEl("span", "mx-lab", name), row.firstChild);
    row.style.alignItems = "center";
  }
  const r1 = htmlEl("div", "mx-row", "", panel);
  const vA = matView(r1, "A", false), vB = matView(r1, "B", false);
  const r2 = htmlEl("div", "mx-row", "", panel);
  const vAB = matView(r2, "AB", false), vBA = matView(r2, "BA", false);
  const verdict = htmlEl("div", "qb-big", "", panel);
  const r3 = htmlEl("div", "mx-row", "", panel);
  button(r3, "Swap A and B", () => { const t = st.OA; st.OA = st.B; st.B = t; draw(); });
  button(r3, "Reset", () => { st.OA = [...st.OA0] as M2; st.B = [...st.B0] as M2; draw(); });
  htmlEl("div", "qb-note", "Grey: the house before. Dashed: after the first step. Orange: after both.", panel);

  function draw(): void {
    const AB = mul(st.OA, st.B), BA = mul(st.B, st.OA);
    sides[0].mid.setAttribute("d", path(sides[0].p, mapPts(st.B, HOUSE)));
    sides[0].fin.setAttribute("d", path(sides[0].p, mapPts(AB, HOUSE)));
    sides[1].mid.setAttribute("d", path(sides[1].p, mapPts(st.OA, HOUSE)));
    sides[1].fin.setAttribute("d", path(sides[1].p, mapPts(BA, HOUSE)));
    vA(st.OA); vB(st.B); vAB(AB); vBA(BA);
    const eq = same(AB, BA);
    verdict.textContent = eq ? "AB = BA: here the order does not matter" : "AB ≠ BA: the order matters";
    verdict.style.color = eq ? EIGCOL : COL1;
    ba.forEach((b, i) => b.classList.toggle("active", same(ORDER_PRESETS[i].m, st.OA)));
    bb.forEach((b, i) => b.classList.toggle("active", same(ORDER_PRESETS[i].m, st.B)));
  }
  draw();
  return () => { /* nothing running */ };
}

// ---------------------------------------------------------------------------------------
// Mode: eigen
function buildEigen(host: HTMLElement, panel: HTMLElement, st: State): Cleanup {
  const R = 3.6, size = 520;
  const wrap = htmlEl("div", "mx-plot", "");
  host.insertBefore(wrap, panel);
  const p = makePlot(wrap, size, R);
  drawGrid(p, R);
  const gE = p.layer();      // eigen-lines found by the sweep
  const trace = svgEl("polyline", { fill: "none", stroke: AVCOL, "stroke-width": 2, "stroke-opacity": 0.55 }, p.layer());
  const gV = p.layer();
  const vLine = svgEl("line", { stroke: "#888", "stroke-width": 1.5, "stroke-dasharray": "3 6" }, gV);
  const avArr = arrow(p, gV, AVCOL, 5);
  const vArr = arrow(p, gV, VCOL, 4);
  const gLab = p.layer();
  const lv = tipLabel(p, gLab, "v", VCOL), lav = tipLabel(p, gLab, "Av", AVCOL);
  const hv = handle(p, p.layer(), VCOL);

  const mc = matrixControls(panel, st, () => matrixChanged());
  const vs = slider(panel, "direction of v", 0, 359, 1, st.vDeg, v => { stopSweep(); setV(v); }, v => `${fmt(v)}°`);
  presets(panel, [
    { label: "Symmetric", m: [2, 1, 1, 2] as M2 },
    { label: "Diagonal", m: [3, 0, 0, 1.5] as M2 },
    { label: "Triangular", m: [3, 1, 0, 2] as M2 },
    { label: "Projection", m: [1, 0, 0, 0] as M2 },
    { label: "Mirror", m: [0, 1, 1, 0] as M2 },
    { label: "Shear", m: [1, 0.5, 0, 1] as M2 },
    { label: "Rotate 90°", m: [0, -1, 1, 0] as M2 },
    { label: "Scale ×2", m: [2, 0, 0, 2] as M2 },
  ].map(it => ({ label: it.label, apply: () => { setA(st, it.m); matrixChanged(); } })));
  const row = htmlEl("div", "mx-row", "", panel);
  const sweepBtn = button(row, "Sweep v around", () => (sweeping ? stopSweep() : startSweep()));
  button(row, "Reset", () => { stopSweep(); setA(st, st.A0); matrixChanged(); setV(st.v0); });
  const out = htmlEl("div", "qb-readout", "", panel);
  const status = htmlEl("div", "mx-status", "", panel);
  const found = htmlEl("div", "qb-note", "", panel);

  let eig: EigDir[] | "all" = eigenDirections(st.A);
  let shown = new Set<number>();          // indices of eigen-lines already marked
  let tracePts: string[] = [];

  draggable(hv, p.world, (x, y) => {
    stopSweep();
    if (Math.hypot(x, y) < 0.15) return;
    setV(Math.atan2(y, x) * 180 / Math.PI);
  });

  /** Snap the angle onto an eigen-direction when it is within 3 degrees. */
  function setV(deg: number, doSnap = true): void {
    let t = deg * Math.PI / 180;
    if (doSnap && eig !== "all") {
      for (const e of eig) {
        if (lineGap(t, e.theta) < 3 * Math.PI / 180) {
          const cands = [e.theta, e.theta + Math.PI, e.theta - Math.PI, e.theta + 2 * Math.PI];
          t = cands.reduce((b, c) => (Math.abs(c - t) < Math.abs(b - t) ? c : b));
        }
      }
    }
    st.vDeg = ((t * 180 / Math.PI) % 360 + 360) % 360;
    vs.set(Math.round(st.vDeg) % 360);
    draw();
  }

  function matrixChanged(): void {
    stopSweep();
    eig = eigenDirections(st.A);
    shown = new Set();
    tracePts = [];
    gE.innerHTML = "";
    trace.setAttribute("points", "");
    found.textContent = "";
    setV(st.vDeg);
  }

  function markLine(i: number): void {
    if (eig === "all" || shown.has(i)) return;
    shown.add(i);
    const e = eig[i];
    const cx = Math.cos(e.theta), sy = Math.sin(e.theta), L = R * 1.5;
    svgEl("line", { x1: p.X(-L * cx), y1: p.Y(-L * sy), x2: p.X(L * cx), y2: p.Y(L * sy), stroke: EIGCOL, "stroke-width": 3, "stroke-opacity": 0.6, "stroke-dasharray": "10 6" }, gE);
    // label near the edge of the view, on the side with y >= 0
    const s = sy < -1e-9 || (Math.abs(sy) < 1e-9 && cx < 0) ? -1 : 1;
    const lx = Math.min(Math.max(p.X(s * cx * R * 0.78), 70), size - 70);
    const ly = Math.min(Math.max(p.Y(s * sy * R * 0.78) - 14, 24), size - 14);
    const t = svgEl("text", { x: lx, y: ly, class: "mx-t", "text-anchor": "middle" }, gE);
    t.style.fill = EIGCOL;
    t.textContent = `λ = ${fmt(e.lambda)}`;
  }

  function foundText(complete: boolean): void {
    if (eig === "all") { found.textContent = "Every direction stays on its line."; return; }
    const list = eig.filter((_, i) => shown.has(i))
      .map(e => {
        const sg = Math.cos(e.theta) < -1e-9 ? -1 : 1;     // show (0.71, -0.71) rather than (-0.71, 0.71)
        return `λ = ${fmt(e.lambda)} along (${fmt(sg * Math.cos(e.theta))}, ${fmt(sg * Math.sin(e.theta))})`;
      });
    if (complete && list.length === 0) found.textContent = "Swept all the way round: no direction stays on its line.";
    else if (list.length) found.textContent = "Found: " + list.join(";  ");
  }

  function draw(): void {
    const t = st.vDeg * Math.PI / 180;
    const vx = Math.cos(t), vy = Math.sin(t);
    const [ax, ay] = apply(st.A, vx, vy);
    const L = R * 1.5;
    setAttrs(vLine, { x1: p.X(-L * vx), y1: p.Y(-L * vy), x2: p.X(L * vx), y2: p.Y(L * vy) });
    vArr.set(0, 0, vx, vy);
    avArr.set(0, 0, ax, ay);
    setAttrs(hv, { cx: p.X(vx), cy: p.Y(vy) });
    lv(vx, vy);
    const cross = vx * ay - vy * ax, dot = vx * ax + vy * ay, len = Math.hypot(ax, ay);
    const parallel = len < 1e-9 || Math.abs(cross) < 1e-6 * Math.max(1, len);
    avArr.color(parallel ? EIGCOL : AVCOL);
    avArr.el.setAttribute("stroke-width", parallel ? "6" : "5");
    lav(ax, ay, parallel ? `Av = ${fmt(dot)}v` : "Av");
    (gLab.children[1] as SVGTextElement).style.fill = parallel ? EIGCOL : AVCOL;
    mc.sync();
    out.textContent = `v  = (${fmt(vx)}, ${fmt(vy)})\nAv = (${fmt(ax)}, ${fmt(ay)})`;
    if (parallel) {
      status.className = "mx-status mx-ok qb-big";
      status.textContent = len < 1e-9 ? "Av = 0 = 0·v: eigenvector, λ = 0"
        : `Av = ${fmt(dot)}·v: eigenvector, λ = ${fmt(dot)}`;
    } else {
      status.className = "mx-status";
      const turn = Math.atan2(cross, dot) * 180 / Math.PI;
      status.textContent = `Av is turned ${fmt(Math.abs(turn), 0)}° away from v`;
    }
  }

  // --- sweep: turn v once around, pausing on each eigen-direction ---
  let sweeping = false, raf = 0, last = 0, done = 0, pauseUntil = 0;
  const SPEED = 60;   // degrees per second
  function startSweep(): void {
    sweeping = true;
    sweepBtn.textContent = "Stop";
    done = 0; last = 0; pauseUntil = 0;
    tracePts = [];
    raf = requestAnimationFrame(step);
  }
  function stopSweep(): void {
    if (!sweeping) return;
    sweeping = false;
    cancelAnimationFrame(raf);
    sweepBtn.textContent = "Sweep v around";
  }
  function step(now: number): void {
    if (!sweeping) return;
    if (!last) last = now;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (now < pauseUntil) { raf = requestAnimationFrame(step); return; }
    const from = st.vDeg;
    let inc = SPEED * dt;
    if (done + inc >= 360) inc = 360 - done;
    // stop exactly on an eigen-direction crossed during this step
    let hit = -1;
    if (eig !== "all") {
      const t0 = from * Math.PI / 180, t1 = (from + inc) * Math.PI / 180;
      eig.forEach((e, i) => {
        for (let kk = -2; kk <= 4; kk++) {
          const te = e.theta + kk * Math.PI;
          if (te > t0 + 1e-9 && te <= t1 + 1e-12) {
            const d = (te - t0) * 180 / Math.PI;
            if (hit < 0 || d < inc) { inc = d; hit = i; }
          }
        }
      });
    }
    done += inc;
    st.vDeg = from + inc;
    const t = st.vDeg * Math.PI / 180;
    const [ax, ay] = apply(st.A, Math.cos(t), Math.sin(t));
    tracePts.push(`${p.X(ax).toFixed(1)},${p.Y(ay).toFixed(1)}`);
    trace.setAttribute("points", tracePts.join(" "));
    vs.set(Math.round(((st.vDeg % 360) + 360) % 360) % 360);
    draw();
    if (hit >= 0) { markLine(hit); foundText(false); pauseUntil = now + 900; }
    if (done >= 360 - 1e-9) {
      st.vDeg = ((st.vDeg % 360) + 360) % 360;
      if (eig !== "all") eig.forEach((_, i) => markLine(i));
      foundText(true);
      stopSweep();
      return;
    }
    raf = requestAnimationFrame(step);
  }

  setV(st.vDeg);
  return () => stopSweep();
}

// ---------------------------------------------------------------------------------------
// Mode: image
let cellImage: ImageData | null = null;

function buildImage(host: HTMLElement, panel: HTMLElement, st: State): Cleanup {
  const R = 3.5, size = 520;
  const wrap = htmlEl("div", "mx-plot", "");
  host.insertBefore(wrap, panel);
  const cv = htmlEl("canvas", "", "", wrap);
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d")!;
  const off = document.createElement("canvas");
  off.width = off.height = size;
  const octx = off.getContext("2d")!;
  const outImg = octx.createImageData(size, size);
  if (!cellImage) cellImage = makeCellImage(200);
  const src = cellImage;

  const p = makePlot(wrap, size, R, true);
  const g = p.layer();
  const frame: [number, number][] = [[-H, -H], [H, -H], [H, H], [-H, H]];
  svgEl("path", { d: path(p, frame), fill: "none", stroke: "#999", "stroke-width": 2, "stroke-dasharray": "6 5" }, g);
  const tFrame = svgEl("path", { fill: "none", stroke: "#333", "stroke-width": 2.5 }, g);
  const c1 = arrow(p, g, COL1, 5), c2 = arrow(p, g, COL2, 5);
  const gLab = p.layer();
  const l1 = tipLabel(p, gLab, "col 1", COL1), l2 = tipLabel(p, gLab, "col 2", COL2);
  const gH = p.layer();
  const h1 = handle(p, gH, COL1), h2 = handle(p, gH, COL2);

  const mc = matrixControls(panel, st, () => draw());
  const area = htmlEl("div", "", "", panel);
  const rot = slider(panel, "rotate by θ", -180, 180, 5, 0, v => { setA(st, rotation(v)); draw(); }, v => `${fmt(v)}°`);
  presets(panel, [
    { label: "Identity", m: ID },
    { label: "Rotate 30°", m: rotation(30) },
    { label: "Shear", m: [1, 0.5, 0, 1] as M2 },
    { label: "Scale", m: [2, 0, 0, 0.5] as M2 },
    { label: "Reflect", m: [-1, 0, 0, 1] as M2 },
    { label: "Collapse", m: [1, 1, 0.5, 0.5] as M2 },
    { label: "Misaligned", m: [0.94, 0.58, -0.34, 0.85] as M2 },
  ].map(it => ({ label: it.label, apply: () => { setA(st, it.m); draw(); } })));
  button(htmlEl("div", "mx-row", "", panel), "Reset", () => { setA(st, st.A0); rot.set(0); draw(); });
  htmlEl("div", "qb-note", "The picture turns about its centre (the origin). Dashed: the frame before.", panel);

  draggable(h1, p.world, (x, y) => { st.A[0] = snap(x, 0.1); st.A[2] = snap(y, 0.1); draw(); });
  draggable(h2, p.world, (x, y) => { st.A[1] = snap(x, 0.1); st.A[3] = snap(y, 0.1); draw(); });

  function background(): void {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = "#e3e3e3";
    ctx.lineWidth = 1;
    for (let i = -Math.floor(R); i <= Math.floor(R); i++) {
      ctx.beginPath(); ctx.moveTo(p.X(i), 0); ctx.lineTo(p.X(i), size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p.Y(i)); ctx.lineTo(size, p.Y(i)); ctx.stroke();
    }
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, p.Y(0)); ctx.lineTo(size, p.Y(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.X(0), 0); ctx.lineTo(p.X(0), size); ctx.stroke();
  }

  function draw(): void {
    const A = st.A;
    renderImage(A, src, outImg, R);
    octx.putImageData(outImg, 0, 0);
    background();
    ctx.drawImage(off, 0, 0);
    tFrame.setAttribute("d", path(p, mapPts(A, frame)));
    c1.set(0, 0, A[0], A[2]); c2.set(0, 0, A[1], A[3]);
    setAttrs(h1, { cx: p.X(A[0]), cy: p.Y(A[2]) });
    setAttrs(h2, { cx: p.X(A[1]), cy: p.Y(A[3]) });
    l1(A[0], A[2]); l2(A[1], A[3]);
    mc.sync();
    const k = areaFactor(A);
    area.innerHTML = "";
    htmlEl("span", "qb-big", Math.abs(k) < 1e-9 ? "area × 0: collapsed onto a line" :
      `area × ${fmt(Math.abs(k))}${k < 0 ? ", flipped" : ""}`, area);
    htmlEl("span", "qb-note", "  (bonus: this is det A)", area);
  }
  draw();
  return () => { /* nothing running */ };
}

// ---------------------------------------------------------------------------------------
const BUILD: Record<Mode, (h: HTMLElement, p: HTMLElement, s: State) => Cleanup> = {
  transform: buildTransform, order: buildOrder, eigen: buildEigen, image: buildImage,
};

const asMode = (s: string | undefined): Mode | null => (MODES as string[]).includes(s ?? "") ? (s as Mode) : null;

autoMount("matrix2x2", host => {
  injectStyle();
  const first = asMode(host.dataset.mode) ?? "transform";
  const tabs = (host.dataset.modes ?? "").split(/[\s,]+/).map(asMode).filter((m): m is Mode => m !== null);
  const defA: M2 = first === "eigen" ? [2, 1, 1, 2] : first === "order" ? [2, 0, 0, 1] : ID;
  const A = numList(host.dataset.a, defA).slice(0, 4) as M2;
  const OA: M2 = first === "order" ? A : [2, 0, 0, 1];
  const B = numList(host.dataset.b, [0, -1, 1, 0]).slice(0, 4) as M2;
  const x = numList(host.dataset.x, [1, 1]).slice(0, 2) as [number, number];
  const v = numList(host.dataset.v, [20])[0];
  const st: State = {
    A: [...A] as M2, x: [...x] as [number, number], vDeg: v, A0: A, x0: x, v0: v,
    OA: [...OA] as M2, B: [...B] as M2, OA0: OA, B0: B,
  };
  let cleanup: Cleanup | null = null;

  function show(mode: Mode): void {
    if (cleanup) cleanup();
    host.innerHTML = "";
    host.dataset.current = mode;
    const panel = htmlEl("div", "qb-panel", "", host);
    if (tabs.length > 1) {
      const btns = presets(panel, tabs.map(m => ({ label: TAB_LABEL[m], apply: () => show(m) })));
      btns[0].parentElement!.classList.add("mx-tabs");
      btns.forEach((b, i) => b.classList.toggle("active", tabs[i] === mode));
    }
    cleanup = BUILD[mode](host, panel, st);
  }
  show(first);
});
