// Intensity features of one object. A synthetic cell (same mask, same total signal) with a
// chosen intensity pattern; sliders for the acquisition (background offset, gain, noise,
// saturation) and for mask errors; the image shows the mask, the edge band, the radial rings
// and the geometric vs intensity-weighted centroid (mass displacement). Plots: the
// histogram inside the mask with mean, median and quartiles; FracAtD and MeanFrac per ring;
// a table of all features with the change against the ideal measurement of the same cell.
// Tabs: "population" (20 cells of two types, scatter of two chosen features) and "plate"
// (two plates of per-well medians: batch offset, plate gradient, robust z, median polish).
// Auto-mounts on <div data-demo="intensityfeat">. Options: data-mode="cell|population|plate",
// data-preset="<id>", data-rows="mean,sd,..." (subset of table rows, for slides).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure, heatmapGrid, seriesColor } from "../lib/plot";
import { diverging, viridis } from "../lib/colour";
import { histogram } from "../lib/histogram";
import { intensityRow, radialDistribution, edgeBand, pixelsOf, valuesAt, IntensityRow } from "../lib/features";
import {
  N, PATTERNS, PatternId, makeCell, observe, adjustMask, intensityMax, Cell,
  makePopulation, PopCell, makePlate, normalisePlate, Norm, ROWS, COLS, CONTROL, TREATED,
} from "./model";

const NAME = "intensityfeat";
const SAT_OFF = 1000;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .if-view { display: contents; }
[data-demo="${NAME}"] .if-img { flex: 0 1 330px; min-width: 220px; margin: 0; }
[data-demo="${NAME}"] .if-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .if-plots { flex: 1 1 280px; min-width: 200px; max-width: 360px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .if-table { flex: 1 1 250px; min-width: 210px; max-width: 320px; max-height: 520px; overflow-y: auto; }
[data-demo="${NAME}"] table { border-collapse: collapse; font-size: 14px; width: 100%; }
[data-demo="${NAME}"] td, [data-demo="${NAME}"] th { padding: 1px 4px; text-align: right; border-bottom: 1px solid var(--qb-grid); white-space: nowrap; }
[data-demo="${NAME}"] td:first-child, [data-demo="${NAME}"] th:first-child { text-align: left; }
[data-demo="${NAME}"] td.up { color: #b8312f; } [data-demo="${NAME}"] td.down { color: #1f5fa8; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 380px; gap: .35em; }
[data-demo="${NAME}"] .if-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .if-key { font-size: 14px; color: var(--qb-muted); line-height: 1.25; }
[data-demo="${NAME}"] .if-pop { flex: 0 1 420px; min-width: 240px; margin: 0; }
[data-demo="${NAME}"] .if-pop canvas { width: 100%; display: block; }
[data-demo="${NAME}"] .if-scatter { flex: 1 1 320px; min-width: 240px; max-width: 460px; }
[data-demo="${NAME}"] .if-plate { flex: 1 1 300px; min-width: 240px; max-width: 420px; }
[data-demo="${NAME}"] select { font: inherit; font-size: 15px; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 15px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .if-img { flex: 0 1 300px; }
.reveal .qb-demo[data-demo="${NAME}"] .if-plots { flex: 0 0 265px; max-width: 265px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .if-table { flex: 1 1 270px; max-width: 310px; max-height: 470px; }
.reveal .qb-demo[data-demo="${NAME}"] table { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 250px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .if-key { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface CellState { pattern: PatternId; offset: number; gain: number; noise: number; sat: number; maskErr: number; rings: number; centre: "centroid" | "max" }
const IDEAL: CellState = { pattern: "uniform", offset: 0, gain: 1, noise: 0, sat: SAT_OFF, maskErr: 0, rings: 4, centre: "centroid" };

interface PlateState { gradient: number; norm: Norm }

interface Preset { id: string; label: string; mode: "cell" | "population" | "plate"; cell?: Partial<CellState>; plate?: Partial<PlateState> }
const PRESETS: Preset[] = [
  { id: "offset", label: "background offset", mode: "cell", cell: { pattern: "uniform", offset: 40, noise: 4 } },
  { id: "same", label: "same integral, other pattern", mode: "cell", cell: { pattern: "punctate" } },
  { id: "rim", label: "rim vs filled", mode: "cell", cell: { pattern: "rim" } },
  { id: "polar", label: "polarised cell", mode: "cell", cell: { pattern: "spot" } },
  { id: "nuclear", label: "nuclear: radial bins", mode: "cell", cell: { pattern: "nuclear", rings: 4 } },
  { id: "dilate", label: "dilated mask", mode: "cell", cell: { pattern: "uniform", offset: 10, maskErr: 2 } },
  { id: "saturation", label: "saturation", mode: "cell", cell: { pattern: "spot", sat: 300 } },
  { id: "population", label: "two cell types", mode: "population" },
  { id: "batch", label: "plate offset: robust z", mode: "plate", plate: { gradient: 0, norm: "none" } },
  { id: "gradient", label: "plate gradient hides it", mode: "plate", plate: { gradient: 12, norm: "robustz" } },
  { id: "polish", label: "row/column correction", mode: "plate", plate: { gradient: 12, norm: "polish" } },
];

type RowKey = "area" | "integrated" | "mean" | "sd" | "min" | "max" | "median" | "q1" | "q3" | "mad" | "edgeMean" | "edgeRatio" | "massDisplacement";
const ROWDEF: { key: RowKey; label: string; dec: number }[] = [
  { key: "area", label: "area (px)", dec: 0 },
  { key: "integrated", label: "integrated", dec: 0 },
  { key: "mean", label: "mean", dec: 1 },
  { key: "sd", label: "std", dec: 1 },
  { key: "min", label: "min", dec: 1 },
  { key: "max", label: "max", dec: 1 },
  { key: "median", label: "median", dec: 1 },
  { key: "q1", label: "lower quartile", dec: 1 },
  { key: "q3", label: "upper quartile", dec: 1 },
  { key: "mad", label: "MAD", dec: 1 },
  { key: "edgeMean", label: "edge mean", dec: 1 },
  { key: "edgeRatio", label: "edge / mean", dec: 2 },
  { key: "massDisplacement", label: "mass displ. (px)", dec: 2 },
];

interface Measured { row: IntensityRow & { area: number }; mask: Uint8Array; img: Float64Array; centre: [number, number]; radial: ReturnType<typeof radialDistribution>; band: Uint8Array; values: Float64Array }

function measure(cell: Cell, s: CellState): Measured {
  const img = observe(cell.clean, { offset: s.offset, gain: s.gain, noise: s.noise, sat: s.sat >= SAT_OFF ? Infinity : s.sat });
  const mask = adjustMask(cell.mask, N, N, s.maskErr);
  const r = intensityRow(img, mask, N, N);
  const centre: [number, number] = s.centre === "max" ? intensityMax(img, mask, N, N) : [r.cx, r.cy];
  const radial = radialDistribution(img, mask, N, N, s.rings, centre);
  return { row: { ...r, area: r.n }, mask, img, centre, radial, band: edgeBand(mask, N, N), values: valuesAt(img, pixelsOf(mask)) };
}

function fmt(v: number, dec: number): string {
  if (!Number.isFinite(v)) return "–";
  return Math.abs(v) >= 1e7 ? v.toExponential(3) : v.toFixed(dec);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 24 : 18;

  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const startMode = (p0?.mode ?? host.dataset.mode ?? "cell") as "cell" | "population" | "plate";
  const rowFilter = host.dataset.rows ? host.dataset.rows.split(/[\s,]+/) : null;
  let mode = startMode;
  const cs: CellState = { ...IDEAL };
  const ps: PlateState = { gradient: 0, norm: "none" };

  const view = htmlEl("div", "if-view", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  // ---------------- cell view ----------------
  const cellView = htmlEl("div", "if-view", "", view);
  const imgFig = htmlEl("figure", "if-img", "", cellView);
  const imgCap = htmlEl("figcaption", "", "", imgFig);
  const cv = htmlEl("canvas", "", "", imgFig);
  const S = 3; cv.width = N * S; cv.height = N * S;
  const plots = htmlEl("div", "if-plots", "", cellView);
  const hFig = plotFigure(plots, "inside the mask: mean (blue), median and quartiles (red)");
  const hPlot = new Plot(hFig, { width: 340, height: 230, xlabel: "intensity", ylabel: "pixels", font: FS });
  const rFig = plotFigure(plots, "radial distribution (ring 1 = centre)");
  const rPlot = new Plot(rFig, { width: 340, height: 220, xlabel: "ring", font: FS, margin: { l: 2.8 * FS } });
  const tableBox = htmlEl("div", "if-table", "", cellView);

  // ---------------- population view ----------------
  const popView = htmlEl("div", "if-view", "", view);
  const popFig = htmlEl("figure", "if-pop", "", popView);
  htmlEl("figcaption", "", "20 cells: type A (blue) and type B (red)", popFig);
  const popCv = htmlEl("canvas", "", "", popFig);
  const scatFig = plotFigure(popView, "", "if-scatter qb-figure");
  let pop: PopCell[] | null = null;
  const FEATS: { key: RowKey; label: string }[] = ROWDEF.filter(r => r.key !== "min").map(r => ({ key: r.key, label: r.label }));
  let fx: RowKey = "mean", fy: RowKey = "sd";

  // ---------------- plate view ----------------
  const plateView = htmlEl("div", "if-view", "", view);
  const plA = htmlEl("div", "if-plate", "", plateView);
  const plB = htmlEl("div", "if-plate", "", plateView);
  const plRO = htmlEl("div", "qb-readout", "", plateView);

  // ---------------- panel ----------------
  htmlEl("div", "if-lab", "view", panel);
  const modeBtns = presets(panel, [
    { label: "one cell", apply: () => setMode("cell") },
    { label: "population", apply: () => setMode("population") },
    { label: "plate", apply: () => setMode("plate") },
  ]);
  htmlEl("div", "if-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  const cellCtl = htmlEl("div", "qb-panel", "", panel);
  cellCtl.style.flex = "none";
  htmlEl("div", "if-lab", "pattern (same mask, same total signal)", cellCtl);
  const patBtns = presets(cellCtl, PATTERNS.map(p => ({ label: p.label, apply: () => { cs.pattern = p.id; newCell(); } })));
  const offS = slider(cellCtl, "offset", 0, 100, 1, cs.offset, v => { cs.offset = v; runCell(); });
  const gainS = slider(cellCtl, "gain", 0.5, 2, 0.05, cs.gain, v => { cs.gain = v; runCell(); });
  const noiseS = slider(cellCtl, "noise σ", 0, 30, 1, cs.noise, v => { cs.noise = v; runCell(); });
  const satS = slider(cellCtl, "saturate", 100, SAT_OFF, 10, cs.sat, v => { cs.sat = v; runCell(); }, v => (v >= SAT_OFF ? "off" : String(v)));
  const maskS = slider(cellCtl, "mask ±px", -2, 2, 1, cs.maskErr, v => { cs.maskErr = v; runCell(); }, v => (v > 0 ? "+" : "") + v);
  const ringS = slider(cellCtl, "rings", 2, 8, 1, cs.rings, v => { cs.rings = v; runCell(); });
  const centreBox = checkbox(cellCtl, "rings around the intensity maximum", cs.centre === "max", v => { cs.centre = v ? "max" : "centroid"; runCell(); });
  button(cellCtl, "reset", () => { Object.assign(cs, IDEAL); syncCell(); newCell(); presetBtns.forEach(b => b.classList.remove("active")); });
  htmlEl("div", "if-key", "Yellow: mask outline; cyan: edge band; tints: rings; white +: centroid; red dot: intensity-weighted centroid; arrow: mass displacement. Δ: change against the ideal measurement (no offset, gain 1, no noise, no saturation, exact mask).", cellCtl);

  const popCtl = htmlEl("div", "qb-panel", "", panel);
  popCtl.style.flex = "none";
  const selX = htmlEl("select", "", "", htmlEl("label", "", "x: ", popCtl));
  const selY = htmlEl("select", "", "", htmlEl("label", "", "y: ", popCtl));
  for (const sel of [selX, selY]) for (const f of FEATS) { const o = htmlEl("option", "", f.label, sel); o.value = f.key; }
  selX.value = fx; selY.value = fy;
  selX.addEventListener("change", () => { fx = selX.value as RowKey; selX.blur(); drawPop(); });
  selY.addEventListener("change", () => { fy = selY.value as RowKey; selY.blur(); drawPop(); });
  htmlEl("div", "if-key", "Type A: uniform staining, larger. Type B: punctate, smaller. Brightness varies within both types. Offset 10, noise σ = 6.", popCtl);

  const plateCtl = htmlEl("div", "qb-panel", "", panel);
  plateCtl.style.flex = "none";
  const gradS = slider(plateCtl, "column gradient", 0, 20, 1, ps.gradient, v => { ps.gradient = v; drawPlate(); });
  htmlEl("div", "if-lab", "normalisation", plateCtl);
  const normBtns = presets(plateCtl, [
    { label: "none", apply: () => { ps.norm = "none"; drawPlate(); } },
    { label: "robust z", apply: () => { ps.norm = "robustz"; drawPlate(); } },
    { label: "row/col + robust z", apply: () => { ps.norm = "polish"; drawPlate(); } },
  ]);
  htmlEl("div", "if-key", "Each well: the median of 40 cells. C = control columns; * = treated wells (+40). Plate B was imaged with another gain and offset.", plateCtl);

  // ---------------- logic ----------------
  let cell: Cell = makeCell({ pattern: cs.pattern });
  let ideal: Measured = measure(cell, { ...IDEAL, pattern: cs.pattern });

  function setMode(m: typeof mode): void {
    mode = m;
    cellView.style.display = m === "cell" ? "" : "none";
    popView.style.display = m === "population" ? "" : "none";
    plateView.style.display = m === "plate" ? "" : "none";
    cellCtl.style.display = m === "cell" ? "" : "none";
    popCtl.style.display = m === "population" ? "" : "none";
    plateCtl.style.display = m === "plate" ? "" : "none";
    modeBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "population", "plate"][i] === m));
    presetBtns.forEach((b, i) => { b.style.display = PRESETS[i].mode === m ? "" : "none"; });
    if (m === "cell") runCell();
    if (m === "population") drawPop();
    if (m === "plate") drawPlate();
  }

  function syncCell(): void {
    offS.set(cs.offset); gainS.set(cs.gain); noiseS.set(cs.noise); satS.set(cs.sat); maskS.set(cs.maskErr); ringS.set(cs.rings);
    centreBox.checked = cs.centre === "max";
    patBtns.forEach((b, i) => b.classList.toggle("active", PATTERNS[i].id === cs.pattern));
  }

  function applyPreset(p: Preset): void {
    if (p.cell) { Object.assign(cs, IDEAL, p.cell); syncCell(); cell = makeCell({ pattern: cs.pattern }); }
    if (p.plate) { Object.assign(ps, p.plate); gradS.set(ps.gradient); }
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    setMode(p.mode);
  }

  function newCell(): void {
    cell = makeCell({ pattern: cs.pattern });
    syncCell();
    runCell();
  }

  let pending = false;
  function runCell(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      ideal = measure(cell, { ...IDEAL, pattern: cs.pattern, rings: cs.rings, centre: cs.centre });
      drawCell(measure(cell, cs));
    });
  }

  function drawCell(m: Measured): void {
    // image with overlays
    const ctx = cv.getContext("2d")!;
    const id = ctx.createImageData(N, N);
    const hi = Math.max(1, ...Array.from(ideal.img).filter((_, i) => cell.mask[i])) * 1.1;
    const tint = [[255, 170, 60], [90, 170, 255]];
    for (let i = 0; i < N * N; i++) {
      let g = Math.min(255, 255 * m.img[i] / hi), r = g, b = g;
      const k = m.radial.ring[i];
      if (k >= 0) { const t = tint[k % 2]; r = 0.88 * r + 0.12 * t[0]; g = 0.88 * g + 0.12 * t[1]; b = 0.88 * b + 0.12 * t[2]; }
      if (m.band[i]) { r = 0.4 * r; g = 0.5 * g + 0.5 * 220; b = 0.5 * b + 0.5 * 230; }
      id.data[4 * i] = r; id.data[4 * i + 1] = g; id.data[4 * i + 2] = b; id.data[4 * i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = N; tmp.height = N;
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, N * S, N * S);
    // mask outline
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      if (!m.mask[i]) continue;
      if (x + 1 >= N || !m.mask[i + 1]) { ctx.moveTo((x + 1) * S, y * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
      if (x === 0 || !m.mask[i - 1]) { ctx.moveTo(x * S, y * S); ctx.lineTo(x * S, (y + 1) * S); }
      if (y + 1 >= N || !m.mask[i + N]) { ctx.moveTo(x * S, (y + 1) * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
      if (y === 0 || !m.mask[i - N]) { ctx.moveTo(x * S, y * S); ctx.lineTo((x + 1) * S, y * S); }
    }
    ctx.stroke();
    const P = (v: number) => (v + 0.5) * S;
    const gx = P(m.row.cx), gy = P(m.row.cy), wx = P(m.row.wx), wy = P(m.row.wy);
    // arrow (mass displacement)
    const len = Math.hypot(wx - gx, wy - gy);
    if (len > 3) {
      ctx.strokeStyle = "#e8251f"; ctx.fillStyle = "#e8251f"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(wx, wy); ctx.stroke();
      const ux = (wx - gx) / len, uy = (wy - gy) / len;
      ctx.beginPath(); ctx.moveTo(wx + 4 * ux, wy + 4 * uy); ctx.lineTo(wx - 10 * ux - 6 * uy, wy - 10 * uy + 6 * ux); ctx.lineTo(wx - 10 * ux + 6 * uy, wy - 10 * uy - 6 * ux); ctx.fill();
    }
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(gx - 9, gy); ctx.lineTo(gx + 9, gy); ctx.moveTo(gx, gy - 9); ctx.lineTo(gx, gy + 9); ctx.stroke();
    ctx.fillStyle = "#e8251f"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(wx, wy, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    if (cs.centre === "max") {
      ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(P(m.centre[0]), P(m.centre[1]), 8, 0, 2 * Math.PI); ctx.stroke();
    }
    imgCap.textContent = `${PATTERNS.find(p => p.id === cs.pattern)!.label}: mass displacement ${m.row.massDisplacement.toFixed(1)} px`;

    // histogram
    const vmax = Math.max(10, m.row.max * 1.05);
    const hh = histogram(m.values, { bins: 40, min: 0, max: vmax });
    hPlot.clear();
    hPlot.setDomain([0, vmax], [0, Math.max(...hh.counts) * 1.1]);
    hPlot.histogram(Array.from({ length: hh.counts.length + 1 }, (_, k) => hh.lo + k * hh.binWidth), hh.counts, { color: seriesColor(0) });
    hPlot.vline(m.row.mean, { color: "var(--qb-c1)", width: 3 });
    hPlot.vline(m.row.median, { color: "var(--qb-c2)", width: 3 });
    hPlot.vline(m.row.q1, { color: "var(--qb-c2)", width: 2, dash: "5 4" });
    hPlot.vline(m.row.q3, { color: "var(--qb-c2)", width: 2, dash: "5 4" });

    // radial bars
    const n = cs.rings, xs = Array.from({ length: n }, (_, k) => k + 1);
    rPlot.clear();
    const ymax = Math.max(1.6, ...Array.from(m.radial.meanFrac).filter(Number.isFinite)) * 1.08;
    rPlot.setDomain([0.4, n + 0.6], [0, ymax]);
    rPlot.bars(xs.map(x => x - 0.19), m.radial.fracAtD, { barWidth: 0.36, color: "var(--qb-c1)", label: "FracAtD" });
    rPlot.bars(xs.map(x => x + 0.19), Array.from(m.radial.meanFrac, v => (Number.isFinite(v) ? v : 0)), { barWidth: 0.36, color: "var(--qb-c4)", label: "MeanFrac" });
    rPlot.hline(1, { color: "var(--qb-muted)", dash: "4 4", width: 1.5 });
    rPlot.legend(undefined, "tl");

    // table
    const rows = rowFilter ? ROWDEF.filter(r => rowFilter.includes(r.key)) : ROWDEF;
    let html = "<table><tr><th>feature</th><th>value</th><th>Δ</th></tr>";
    for (const r of rows) {
      const v = m.row[r.key], v0 = ideal.row[r.key], d = v - v0;
      const small = Math.abs(d) < 0.5 * 10 ** -r.dec;
      html += `<tr><td>${r.label}</td><td>${fmt(v, r.dec)}</td><td class="${small ? "" : d > 0 ? "up" : "down"}">${small ? "0" : (d > 0 ? "+" : "") + fmt(d, r.dec)}</td></tr>`;
    }
    for (let k = 0; k < n && !rowFilter; k++) html += `<tr><td>FracAtD ${k + 1}</td><td>${fmt(m.radial.fracAtD[k], 3)}</td><td></td></tr>`;
    for (let k = 0; k < n && !rowFilter; k++) html += `<tr><td>MeanFrac ${k + 1}</td><td>${fmt(m.radial.meanFrac[k], 2)}</td><td></td></tr>`;
    for (let k = 0; k < n && !rowFilter; k++) html += `<tr><td>RadialCV ${k + 1}</td><td>${fmt(m.radial.radialCV[k], 2)}</td><td></td></tr>`;
    tableBox.innerHTML = html + "</table>";
  }

  // ---------------- population ----------------
  function drawPop(): void {
    if (!pop) {
      pop = makePopulation();
      const T = 112, cols = 5, rws = 4, sc = 1;
      popCv.width = cols * T * sc; popCv.height = rws * T * sc;
      const ctx = popCv.getContext("2d")!;
      const id = ctx.createImageData(cols * T, rws * T);
      pop.forEach((p, k) => {
        const ox = (k % cols) * T, oy = Math.floor(k / cols) * T;
        for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
          const v = Math.min(255, p.img[y * T + x] * 1.1), q = 4 * ((oy + y) * cols * T + ox + x);
          id.data[q] = v; id.data[q + 1] = v; id.data[q + 2] = v; id.data[q + 3] = 255;
        }
      });
      ctx.putImageData(id, 0, 0);
      pop.forEach((p, k) => {
        ctx.strokeStyle = p.type ? "#d62728" : "#1f77b4"; ctx.lineWidth = 3;
        ctx.strokeRect((k % cols) * T + 1.5, Math.floor(k / cols) * T + 1.5, T - 3, T - 3);
      });
    }
    scatFig.replaceChildren();
    const lx = FEATS.find(f => f.key === fx)!.label, ly = FEATS.find(f => f.key === fy)!.label;
    const X = pop.map(p => p.row[fx]), Y = pop.map(p => p.row[fy]);
    const ext = (a: number[]): [number, number] => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) || 1; return [lo - 0.08 * d, hi + 0.08 * d]; };
    const pl = new Plot(scatFig, { width: 440, height: 400, x: ext(X), y: ext(Y), xlabel: lx, ylabel: ly, font: FS });
    pl.scatter(X, Y, { colors: pop.map(p => (p.type ? "#d62728" : "#1f77b4")), r: 7 });
    pl.legend([{ label: "type A", color: "#1f77b4", kind: "dot" }, { label: "type B", color: "#d62728", kind: "dot" }], "tl");
  }

  // ---------------- plate ----------------
  function drawPlate(): void {
    const d = makePlate({ gradient: ps.gradient });
    normBtns.forEach((b, i) => b.classList.toggle("active", ["none", "robustz", "polish"][i] === ps.norm));
    const colLabels = Array.from({ length: COLS }, (_, c) => (CONTROL[c] ? "C" : String(c + 1)));
    const rowLabels = ["A", "B", "C", "D"].slice(0, ROWS);
    const lines: string[] = [];
    [[plA, d.values, "plate A"], [plB, d.valuesB, "plate B"]].forEach(([box, v, title]) => {
      const el = box as HTMLElement, vals = normalisePlate(v as Float64Array, ps.norm);
      el.replaceChildren();
      const isZ = ps.norm !== "none";
      heatmapGrid(el, vals, ROWS, COLS, {
        cell: 58, rowLabels, colLabels, title: `${title}: ${ps.norm === "polish" ? "corrected z" : isZ ? "robust z" : "well median"}`,
        map: isZ ? diverging : viridis, lo: isZ ? -12 : 60, hi: isZ ? 12 : 240,
        text: (x, i) => (isZ ? x.toFixed(1) : x.toFixed(0)) + (TREATED[i] ? "*" : ""),
      });
      if (isZ) {
        let tp = 0, fp = 0;
        vals.forEach((z, i) => { if (Math.abs(z) > 3) { if (TREATED[i]) tp++; else fp++; } });
        lines.push(`${title}: |z| > 3 in ${tp} of 4 treated wells and ${fp} other wells`);
      } else {
        const c = Array.from(v as Float64Array).filter((_, i) => CONTROL[i]);
        lines.push(`${title}: control wells from ${Math.min(...c).toFixed(0)} to ${Math.max(...c).toFixed(0)}`);
      }
    });
    plRO.textContent = lines.join("\n");
  }

  syncCell();
  if (p0) applyPreset(p0);
  else setMode(startMode);
}

autoMount(NAME, mount);
