// Library check page (not linked from the site): exercises lib/plot.ts, lib/colour.ts,
// lib/gridui.ts, lib/gl3d.ts, lib/paint.ts, lib/playground.ts and lib/shapecanvas.ts so a
// screenshot shows that they render. Auto-mounts on <div data-demo="libdemo">.
import { autoMount, htmlEl, rng } from "../lib/ui";
import { Plot, PolarPlot, heatmapGrid, colorbar, seriesColor } from "../lib/plot";
import { viridis, inferno, diverging, grey, COLORMAPS, drawScalarImage, drawLabelImage, labelColor, toCss, lightnessProfile } from "../lib/colour";
import { gridStepper, gridFromText, GridAlgorithm, GridFrame, CROSS3 } from "../lib/gridui";
import { GLView, OrbitCamera, sphereMesh, cubeMesh, axesLines, boxLines, heightFieldMesh, viewButtons } from "../lib/gl3d";
import * as M from "../lib/mat4";
import { LabelPainter } from "../lib/paint";
import { dataset, classStats, pointPlayground } from "../lib/datasets2d";
import { shapeCanvas } from "../lib/shapecanvas";

const NAME = "libdemo";
const STYLE = `
[data-demo="${NAME}"] { display: block; }
[data-demo="${NAME}"] .ld-section h3 { margin: .8em 0 .3em; font-size: 20px; }
[data-demo="${NAME}"] .ld-row { display: flex; flex-wrap: wrap; gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .ld-box { min-width: 0; }
[data-demo="${NAME}"] .ld-strip { display: grid; grid-template-columns: 6em 1fr; align-items: center; gap: .5em; margin: .2em 0; }
[data-demo="${NAME}"] .ld-strip canvas { width: 100%; height: 22px; image-rendering: pixelated; }
[data-demo="${NAME}"] .ld-swatches { display: flex; gap: 3px; margin-top: .3em; }
[data-demo="${NAME}"] .ld-swatches span { width: 30px; height: 26px; display: grid; place-items: center; font-size: 14px; border-radius: 3px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function section(host: HTMLElement, title: string): HTMLElement {
  const s = htmlEl("section", "ld-section", "", host);
  htmlEl("h3", "", title, s);
  return htmlEl("div", "ld-row", "", s);
}

function box(row: HTMLElement, w = 380): HTMLElement {
  const d = htmlEl("div", "ld-box", "", row);
  d.style.flex = `0 1 ${w}px`;
  return d;
}

function plots(host: HTMLElement): void {
  const row = section(host, "plot.ts and colour.ts");
  const r = rng(3);
  // line, step, band, scatter, ellipse, markers, legend
  const p = new Plot(box(row, 420), { x: [0, 10], y: [-1.5, 1.5], xlabel: "x", ylabel: "signal", title: "lines and markers" });
  const xs = Array.from({ length: 101 }, (_, i) => i / 10);
  p.band(xs, xs.map(x => Math.sin(x) - 0.3), xs.map(x => Math.sin(x) + 0.3), { color: seriesColor(0) });
  p.line(xs, xs.map(Math.sin), { color: seriesColor(0), label: "sin x" });
  const xk = xs.filter((_, i) => i % 10 === 0);
  p.step(xk, xk.map(x => Math.round(2 * Math.cos(x)) / 2), { color: seriesColor(1), label: "quantised cos" });
  p.vline(Math.PI, { text: "π" });
  p.hline(0, { width: 1.5, dash: "3 3" });
  p.legend(undefined, "br");
  // scatter + covariance ellipse
  const s = new Plot(box(row, 380), { x: [0, 1], y: [0, 1], equal: true, xlabel: "feature 1", ylabel: "feature 2", title: "scatter + 2-SD ellipse" });
  const d = dataset("aniso", { seed: 2 });
  s.scatter(d.points.map(q => q.x), d.points.map(q => q.y), { colors: d.points.map(q => seriesColor(q.c)), r: 5 });
  for (let c = 0; c < d.k; c++) { const st = classStats(d, c); s.ellipse(st.mean, st.cov, 2, { color: seriesColor(c) }); }
  // histogram with a threshold marker and log y
  const hv = Array.from({ length: 2000 }, () => (r.unif() < 0.6 ? 70 + 12 * r.norm() : 160 + 20 * r.norm()));
  const edges = Array.from({ length: 33 }, (_, i) => i * 8), counts = new Array(32).fill(0);
  hv.forEach(v => { const k = Math.floor(v / 8); if (k >= 0 && k < 32) counts[k]++; });
  const h = new Plot(box(row, 380), { x: [0, 256], y: [0, Math.max(...counts) * 1.1], xlabel: "grey value", ylabel: "count", title: "histogram" });
  h.histogram(edges, counts, { color: seriesColor(2) });
  h.vline(118, { text: "t = 118", color: seriesColor(1) });
  // bars on a log axis
  const b = new Plot(box(row, 380), { x: [0.3, 6.7], y: [0.01, 100], ylog: true, xticks: [1, 2, 3, 4, 5, 6], xlabel: "component", ylabel: "eigenvalue", title: "bars, log axis" });
  b.bars([1, 2, 3, 4, 5, 6], [40, 9, 2, 0.5, 0.12, 0.03], { colors: [0, 1, 2, 3, 4, 5].map(seriesColor) });
  // heat map with numbers (confusion matrix)
  const hm = box(row, 300);
  heatmapGrid(hm, [41, 3, 1, 5, 37, 8, 0, 6, 44], 3, 3, { map: t => viridis(1 - 0.85 * t), title: "confusion matrix", rowLabels: ["cat", "dog", "fox"], colLabels: ["cat", "dog", "fox"] });
  // heat map in a plot (image mode) + colour bar
  const im = new Plot(box(row, 380), { x: [-1, 1], y: [-1, 1], equal: true, title: "heat map (image mode)" });
  const N = 96, f = Float64Array.from({ length: N * N }, (_, i) => { const x = (i % N) / N * 2 - 1, y = 1 - Math.floor(i / N) / N * 2; return Math.sin(4 * x) * Math.cos(3 * y) * Math.exp(-(x * x + y * y)); });
  im.heatmap(f, N, N, { map: diverging, lo: -1, hi: 1 });
  colorbar(im.svg.parentElement!, diverging, -1, 1, { label: "signed value" });
  // polar plot
  const pp = new PolarPlot(box(row, 300), { rmax: 1, title: "polar plot" });
  const th = Array.from({ length: 181 }, (_, i) => i * Math.PI / 90);
  pp.line(th, th.map(t => 0.55 + 0.4 * Math.cos(3 * t)), { color: seriesColor(0), closed: true, fill: seriesColor(0) });
  pp.ray(Math.PI / 3, 0.9);
  const ph = new PolarPlot(box(row, 300), { rmax: 1, half: true, spokeStep: 45, title: "half polar (0–180°)" });
  ph.scatter([0.3, 1.1, 2.0, 2.8], [0.5, 0.9, 0.7, 0.3], { color: seriesColor(3) });
  // colour maps: strips and L* profiles, label palette
  const cm = box(row, 520);
  const lp = new Plot(cm, { x: [0, 1], y: [0, 100], height: 260, xlabel: "t", ylabel: "L*", title: "colour maps and their lightness" });
  (Object.keys(COLORMAPS) as (keyof typeof COLORMAPS)[]).forEach((k, i) => lp.line(Array.from({ length: 64 }, (_, j) => j / 63), lightnessProfile(COLORMAPS[k]), { color: seriesColor(i), label: k }));
  lp.legend(undefined, "tl");
  for (const [name, map] of [["grey", grey], ["viridis", viridis], ["inferno", inferno], ["diverging", diverging]] as const) {
    const line = htmlEl("div", "ld-strip", "", cm);
    htmlEl("span", "", name, line);
    const c = htmlEl("canvas", "", "", line);
    drawScalarImage(c, Float64Array.from({ length: 256 }, (_, i) => i), 256, 1, { map });
  }
  const lab = htmlEl("div", "ld-strip", "", cm);
  htmlEl("span", "", "labels", lab);
  const lc = htmlEl("canvas", "", "", lab);
  drawLabelImage(lc, Float64Array.from({ length: 24 }, (_, i) => i), 24, 1);
  const sw = htmlEl("div", "ld-swatches", "", cm);
  for (let k = 1; k <= 12; k++) { const e = htmlEl("span", "", String(k), sw); e.style.background = toCss(labelColor(k)); }
}

// ---- gridui ----
const SHAPE = gridFromText([
  "0000000000",
  "0111100000",
  "0111110000",
  "0111111100",
  "0011111110",
  "0001111110",
  "0000111100",
  "0000000000",
]);

const erosion: GridAlgorithm = {
  id: "erode", label: "erosion (cross)", auxTitle: "output",
  *run(inp, rows, cols) {
    const out = new Int32Array(inp.length);
    yield { values: inp, aux: out, log: "start: output all 0" };
    for (let i = 0; i < inp.length; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const hits = CROSS3.map(([dr, dc]) => { const rr = r + dr, cc = c + dc; return rr >= 0 && cc >= 0 && rr < rows && cc < cols && inp[rr * cols + cc] === 1; });
      out[i] = hits.every(Boolean) ? 1 : 0;
      const f: GridFrame = { values: inp, aux: out.slice(), focus: i, auxFocus: i, window: { at: i, offsets: CROSS3, hits }, stage: c === 0 ? `row ${r}` : undefined };
      if (out[i]) f.log = `(${r}, ${c}): SE fits -> 1`;
      yield f;
    }
  },
};

const cityBlock: GridAlgorithm = {
  id: "dt", label: "distance, 2 passes", auxTitle: "distance to background", auxDisplay: "int", auxText: v => (v >= 99 ? "∞" : String(v)),
  *run(inp, rows, cols) {
    const D = Int32Array.from(inp, v => (v ? 99 : 0));
    yield { values: inp, aux: D, log: "start: 0 on background, ∞ inside" };
    for (let i = 0; i < D.length; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      if (!D[i]) continue;
      const up = r > 0 ? D[i - cols] : 99, left = c > 0 ? D[i - 1] : 99;
      D[i] = Math.min(D[i], up + 1, left + 1);
      yield { values: inp, aux: D.slice(), auxFocus: i, auxArrows: [...(r > 0 ? [{ from: i - cols, to: i }] : []), ...(c > 0 ? [{ from: i - 1, to: i }] : [])], stage: i === 0 ? "forward pass" : undefined, log: `fwd (${r}, ${c}) = ${D[i]}` };
    }
    let first = true;
    for (let i = D.length - 1; i >= 0; i--) {
      const r = Math.floor(i / cols), c = i % cols;
      if (!D[i]) continue;
      const dn = r < rows - 1 ? D[i + cols] : 99, rt = c < cols - 1 ? D[i + 1] : 99;
      D[i] = Math.min(D[i], dn + 1, rt + 1);
      yield { values: inp, aux: D.slice(), auxFocus: i, stage: first ? "backward pass" : undefined, log: `bwd (${r}, ${c}) = ${D[i]}` };
      first = false;
    }
  },
};

function grids(host: HTMLElement): void {
  const row = section(host, "gridui.ts");
  const g = gridStepper(box(row, 1100), { rows: SHAPE.rows, cols: SHAPE.cols, initial: SHAPE.values, display: "binary", algorithms: [erosion, cityBlock], cell: 40, title: "input (click to edit)", indices: true });
  g.goTo(23);
}

// ---- gl3d ----
function gl(host: HTMLElement): void {
  const row = section(host, "gl3d.ts and mat4.ts");
  const left = box(row, 760);
  const cam = new OrbitCamera({ distance: 2.6 });
  const view = new GLView(left, {
    aspect: 1.6,
    viewports: [
      { x: 0, y: 0, w: 0.5, h: 0.5, camera: cam, label: "mesh, points, lines" },
      { x: 0.5, y: 0, w: 0.5, h: 0.5, camera: new OrbitCamera({ distance: 2.2, elevation: 0.7 }), label: "height field + slice" },
      { x: 0, y: 0.5, w: 0.5, h: 0.5, camera: new OrbitCamera({ distance: 2.2 }), label: "MIP", background: [0, 0, 0] },
      { x: 0.5, y: 0.5, w: 0.5, h: 0.5, camera: new OrbitCamera({ distance: 2.2 }), label: "isosurface / ray marching" },
    ],
  });
  const mesh = view.addMesh(sphereMesh(0.45, 20), { shading: "flat", lighting: "blinn-phong", wireframe: "overlay", color: [0.55, 0.72, 0.9] });
  mesh.viewports = [0];
  const cube = view.addMesh(cubeMesh([-0.15, -0.15, -0.15], [0.15, 0.15, 0.15]), { color: [0.95, 0.6, 0.3], lighting: "lambert" });
  cube.setModel(M.mul(M.translation(0.7, 0.3, 0), M.rotationY(0.5)));
  cube.viewports = [0];
  const r = rng(5), P: number[] = [], C: number[] = [];
  for (let i = 0; i < 400; i++) { const v: M.Vec3 = [r.norm(), r.norm(), r.norm()], n = M.normalize(v); P.push(...M.scale3(n, 0.75)); C.push(...viridis((n[1] + 1) / 2)); }
  view.addPoints({ positions: P, colors: C }, { size: 5 }).viewports = [0];
  view.addLines(axesLines(0.9), { width: 3 }).viewports = [0, 1];
  view.addLines(boxLines(), { width: 1.5, color: [0.4, 0.4, 0.45] }).viewports = [0];
  // height field and a slice through it
  const N = 48, hv = Float64Array.from({ length: N * N }, (_, i) => { const x = (i % N) / N - 0.5, y = Math.floor(i / N) / N - 0.5; return Math.exp(-((x - 0.15) ** 2 + y * y) / 0.02) + 0.6 * Math.exp(-((x + 0.2) ** 2 + (y + 0.2) ** 2) / 0.01); });
  view.addMesh(heightFieldMesh(hv, N, N, { scale: 0.35, map: viridis }), { shading: "smooth", lighting: "lambert" }).viewports = [1];
  const sl = view.addSlice({ values: hv, w: N, h: N, map: inferno }, { corners: [[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], opacity: 0.85, filter: "nearest" });
  sl.viewports = [1];
  // a volume: two blobs and a shell
  const n = 48, vol = new Float32Array(n * n * n);
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = x / n - 0.5, v = y / n - 0.5, w = z / n - 0.5;
    const b1 = Math.exp(-((u - 0.12) ** 2 + (v + 0.1) ** 2 + w * w) / 0.008), b2 = 0.7 * Math.exp(-((u + 0.15) ** 2 + (v - 0.12) ** 2 + (w - 0.1) ** 2) / 0.005);
    const shell = 0.35 * Math.exp(-((Math.hypot(u, v, w) - 0.4) ** 2) / 0.0008);
    vol[(z * n + y) * n + x] = b1 + b2 + shell;
  }
  const mip = view.addVolume({ data: vol, nx: n, ny: n, nz: n, spacing: [1, 1, 1.5] }, { mode: "mip", transfer: grey });
  mip.viewports = [2];
  const iso = view.addVolume({ data: vol, nx: n, ny: n, nz: n }, { mode: "iso", iso: 0.3, isoColor: [0.9, 0.55, 0.35] });
  iso.viewports = [3];
  // outlines of the volume boxes: the unit cube through each volume's model matrix
  for (const v of [mip, iso]) { const b = view.addLines(boxLines([0, 0, 0], [1, 1, 1]), { width: 1.5, color: [0.55, 0.55, 0.6] }); b.setModel(v.model); b.viewports = v.viewports; }
  const panel = box(row, 300);
  panel.classList.add("qb-panel");
  htmlEl("div", "qb-note", "Drag in a view to rotate it. Buttons act on the top-left view.", panel);
  viewButtons(panel, view, 0);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const show = () => { ro.textContent = "V (top-left view)\n" + M.formatMat(cam.view(), 2) + "\n\nWebGL2: " + (view.error ? "unavailable" : view.active ? "context live" : "waiting"); };
  view.opts.onCamera = show;
  view.opts.onRender = show;
  show();
}

// ---- paint, playground, shapes ----
function editors(host: HTMLElement): void {
  const row = section(host, "paint.ts, datasets2d.ts / playground.ts, shapecanvas.ts");
  const pb = box(row, 330);
  const painter = new LabelPainter(pb, { width: 64, height: 64, labels: [{ value: 1, name: "object", color: "#ff7f0e" }, { value: 2, name: "background", color: "#1f77b4" }], brush: 2, caption: "paint labels" });
  painter.setImage(Float64Array.from({ length: 64 * 64 }, (_, i) => { const x = i % 64, y = Math.floor(i / 64); return 60 + 150 * Math.exp(-((x - 30) ** 2 + (y - 34) ** 2) / 200); }));
  const lab = new Uint8Array(64 * 64);
  for (let x = 22; x < 38; x++) for (let y = 32; y < 35; y++) lab[y * 64 + x] = 1;
  for (let x = 4; x < 60; x++) for (let y = 4; y < 7; y++) lab[y * 64 + x] = 2;
  painter.setLabels(lab);

  const pg = box(row, 330);
  const play = pointPlayground(pg, { data: dataset("moons", { seed: 1 }), presets: ["blobs", "moons", "rings", "xor", "spirals", "outliers"] });
  play.setBackground((x, y) => { const p = 1 / (1 + Math.exp(-12 * (y - 0.5 + 0.25 * Math.sin(5 * x)))); return [p, 1 - p]; });

  const sc = box(row, 330);
  shapeCanvas(sc, { size: 48, presets: ["square", "L", "star", "ring", "two"], shapes: undefined });
}

function mount(host: HTMLElement): void {
  injectStyle();
  plots(host);
  grids(host);
  gl(host);
  editors(host);
}

autoMount(NAME, mount);
