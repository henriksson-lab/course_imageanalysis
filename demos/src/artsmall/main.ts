// ART and SIRT on tiny systems, the by-hand exercises made interactive.
// Mode "2 pixels": every measurement of a two-pixel image is a line in the (x1, x2) plane;
// drag a line's square handle to move it and its round handle to turn it. ART projects the
// estimate onto one line after the other (zig-zag); SIRT averages the projections onto all
// lines. Inconsistent (noisy) lines do not meet in one point: ART circles, a small
// relaxation omega settles near the least-squares point (star).
// Mode "image": a 2 x 2 or 3 x 3 image measured by ray sums along rows, columns and
// diagonals; the matrix A, the data b, the current estimate x and the residual; "next ray"
// spreads one ray's error evenly over its pixels. Readout: rank of A and a null-space image.
// Auto-mounts on <div data-demo="artsmall">. Options: data-mode="pixels|image",
// data-preset="same|diag|sweep|three|plane|parallel|noisy".
import { htmlEl, slider, presets, button, checkbox, autoMount, draggable, svgEl } from "../lib/ui";
import { Plot, plotFigure, seriesColor } from "../lib/plot";
import { injectStyle } from "../radonproj/view";
import * as M from "./model";

const NAME = "artsmall";

interface Line { px: number; py: number; dir: number }   // a point on the line and its direction (degrees)
interface Preset {
  id: string; label: string; mode: "pixels" | "image";
  lines?: Line[]; sirt?: boolean; omega?: number;
  n?: number; truth?: number[]; sets?: M.RaySet[]; steps?: number;
}
const PRESETS: Preset[] = [
  { id: "same", label: "2×2: two images, same data", mode: "image", n: 2, truth: [1, 2, 3, 4], sets: ["rows", "cols"], steps: 0 },
  { id: "sweep", label: "ART: one sweep", mode: "image", n: 2, truth: [1, 2, 3, 4], sets: ["rows", "cols"], steps: 4 },
  { id: "diag", label: "add a diagonal ray", mode: "image", n: 2, truth: [2, 1, 2, 5], sets: ["rows", "cols", "diag"], steps: 0 },
  { id: "three", label: "3×3: rows and columns", mode: "image", n: 3, truth: [0, 1, 0, 1, 1, 1, 0, 1, 0], sets: ["rows", "cols"], steps: 0 },
  { id: "plane", label: "2 pixels: ART zig-zags", mode: "pixels", lines: [{ px: 2, py: 3, dir: 15 }, { px: 2, py: 3, dir: 105 }], omega: 1 },
  { id: "parallel", label: "nearly parallel rays: slow", mode: "pixels", lines: [{ px: 2, py: 3, dir: 30 }, { px: 2, py: 3, dir: 45 }], omega: 1 },
  { id: "noisy", label: "noisy: ART circles", mode: "pixels", lines: [{ px: 2, py: 3.8, dir: 0 }, { px: 2.9, py: 3, dir: 75 }, { px: 1.2, py: 3, dir: 130 }], omega: 1 },
  { id: "noisy3", label: "noisy, ω = 0.3 settles", mode: "pixels", lines: [{ px: 2, py: 3.8, dir: 0 }, { px: 2.9, py: 3, dir: 75 }, { px: 1.2, py: 3, dir: 130 }], omega: 0.3 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .as-fig { flex: 1 1 260px; max-width: 360px; margin: 0; }
[data-demo="${NAME}"] .as-fig figcaption { font-size: 17px; text-align: center; margin: 0 0 .1em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .35em; }
[data-demo="${NAME}"] .as-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .as-sub { display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .as-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] table.as-A { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 15px; }
[data-demo="${NAME}"] table.as-A td, [data-demo="${NAME}"] table.as-A th { padding: .1em .35em; text-align: right; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] table.as-A th { color: var(--qb-muted); font-weight: 500; }
[data-demo="${NAME}"] table.as-A tr.cur td { background: #fde7c4; }
[data-demo="${NAME}"] table.as-A td.one { color: var(--qb-c1); font-weight: 700; }
[data-demo="${NAME}"] table.as-A td.zero { color: var(--qb-grid); }
[data-demo="${NAME}"] .as-tab { flex: 1 1 300px; max-width: 420px; overflow-x: auto; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
`;

const LETTERS = "abcdefghi";

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (host.dataset.mode === "pixels" ? PRESETS[4] : PRESETS[0]);

  // ---------- state ----------
  let mode: "pixels" | "image" = p0.mode;
  // pixels mode
  let lines: Line[] = [];
  let omega = 1, showSirt = false, shown = 0;
  // image mode
  let n = 2, truth: number[] = [1, 2, 3, 4], sets: M.RaySet[] = ["rows", "cols"];
  let sys = M.smallSystem(n, sets), b: Float64Array = M.apply(sys, truth), x: Float64Array = new Float64Array(n * n), next = 0;

  // ---------- layout ----------
  // pixels mode: the plane
  const planeFig = plotFigure(host, "pixel 1 across, pixel 2 up", "as-fig");
  const plane = new Plot(planeFig, { width: 360, height: 360, x: [-0.5, 5.5], y: [-0.5, 5.5], equal: true, xlabel: "x₁", ylabel: "x₂", xticks: [0, 1, 2, 3, 4, 5], yticks: [0, 1, 2, 3, 4, 5] });
  // image mode: the true image with rays, and the estimate
  const truthFig = plotFigure(host, "true image (click: +1) and rays", "as-fig");
  const tPlot = new Plot(truthFig, { width: 300, height: 300, x: [-0.6, 3.6], y: [-0.6, 3.6], equal: true, grid: false, xticks: [], yticks: [], margin: { l: 4, r: 4, t: 4, b: 4 } });
  tPlot.svg.querySelector(".qb-axes")?.setAttribute("display", "none");
  const estFig = plotFigure(host, "estimate x", "as-fig");
  const ePlot = new Plot(estFig, { width: 300, height: 300, x: [-0.6, 3.6], y: [-0.6, 3.6], equal: true, grid: false, xticks: [], yticks: [], margin: { l: 4, r: 4, t: 4, b: 4 } });
  ePlot.svg.querySelector(".qb-axes")?.setAttribute("display", "none");
  const tab = htmlEl("div", "as-tab", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  const modeRow = htmlEl("div", "as-row", "", panel);
  const modeBtns = (["pixels", "image"] as const).map(m => button(modeRow, m === "pixels" ? "mode: 2 pixels" : "mode: image", () => { mode = m; layoutMode(); }));
  button(modeRow, "reset", () => load(p0));
  htmlEl("div", "as-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  // pixels controls
  const pixBox = htmlEl("div", "as-sub", "", panel);
  const omS = slider(pixBox, "relaxation ω", 0.1, 1, 0.05, 1, v => { omega = v; drawPlane(); }, v => v.toFixed(2));
  const pRow = htmlEl("div", "as-row", "", pixBox);
  button(pRow, "next step", () => { shown++; drawPlane(); });
  button(pRow, "+1 sweep", () => { shown += lines.length; drawPlane(); });
  button(pRow, "clear path", () => { shown = 0; drawPlane(); });
  const pRow2 = htmlEl("div", "as-row", "", pixBox);
  button(pRow2, "+ line", () => { if (lines.length < 4) { lines.push({ px: 2, py: 3, dir: 60 + 30 * lines.length }); rebuildHandles(); drawPlane(); } });
  button(pRow2, "− line", () => { if (lines.length > 2) { lines.pop(); rebuildHandles(); drawPlane(); } });
  const sirtCb = checkbox(pixBox, "show SIRT path", false, v => { showSirt = v; drawPlane(); });
  // image controls
  const imgBox = htmlEl("div", "as-sub", "", panel);
  htmlEl("div", "as-lab", "rays (angle of the normal)", imgBox);
  const setRow = htmlEl("div", "as-row", "", imgBox);
  const allSets: M.RaySet[] = ["cols", "rows", "diag", "anti"];
  const setBtns = allSets.map(s => button(setRow, `${M.SET_ANGLE[s]}: ${s === "cols" ? "columns" : s === "rows" ? "rows" : s === "diag" ? "↘ diagonals" : "↙ diagonals"}`, () => {
    sets = sets.includes(s) ? sets.filter(t => t !== s) : allSets.filter(t => t === s || sets.includes(t));
    if (sets.length === 0) sets = [s];
    rebuildSystem(); drawImage();
  }));
  const sizeRow = htmlEl("div", "as-row", "", imgBox);
  const sizeBtns = [2, 3].map(k => button(sizeRow, `${k}×${k}`, () => { n = k; truth = Array.from({ length: k * k }, (_, i) => (k === 2 ? [1, 2, 3, 4][i] : [0, 1, 0, 1, 1, 1, 0, 1, 0][i])); rebuildSystem(); drawImage(); }));
  button(sizeRow, "+ null image", () => { const v = M.nullImages(sys)[0]; if (v) { truth = truth.map((t, i) => t + v[i]); rebuildData(); drawImage(); } });
  const iRow = htmlEl("div", "as-row", "", imgBox);
  button(iRow, "next ray", () => { x = M.kaczmarzStep(sys, b, x, next, omega); next = (next + 1) % sys.rays.length; drawImage(); });
  button(iRow, "sweep", () => { for (let k = 0; k < sys.rays.length; k++) { x = M.kaczmarzStep(sys, b, x, next, omega); next = (next + 1) % sys.rays.length; } drawImage(); });
  button(iRow, "SIRT step", () => { x = M.sirtStep(sys, b, x, omega); drawImage(); });
  button(iRow, "x = 0", () => { x = new Float64Array(n * n); next = 0; drawImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  // ---------- pixels mode ----------
  const pathMark = plane.line([0], [0], { color: "var(--qb-c4)", width: 2.5 });
  const pathDots = plane.scatter([0], [0], { color: "var(--qb-c4)", r: 4.5 });
  const sirtMark = plane.line([0], [0], { color: "var(--qb-c5)", width: 2.5, dash: "5 3" });
  const lsMark = plane.text(0, 0, "★", { anchor: "middle", size: 26, color: "var(--qb-fg)" });
  let lineMarks: { set(...a: unknown[]): void; remove(): void }[] = [];
  let handleEls: SVGElement[] = [];

  const toLine = (l: Line): M.Line2 => {
    const t = l.dir * Math.PI / 180, a: [number, number] = [-Math.sin(t), Math.cos(t)];
    return { a, b: a[0] * l.px + a[1] * l.py };
  };

  function rebuildHandles(): void {
    lineMarks.forEach(m => m.remove()); handleEls.forEach(h => h.remove());
    lineMarks = lines.map((_, i) => plane.line([0], [0], { color: seriesColor(i), width: 3 }) as unknown as { set(...a: unknown[]): void; remove(): void });
    handleEls = [];
    lines.forEach((l, i) => {
      const sq = svgEl("rect", { width: 16, height: 16, fill: "#fff", stroke: seriesColor(i), "stroke-width": 3 }, plane.over);
      const ci = svgEl("circle", { r: 9, fill: "#fff", stroke: seriesColor(i), "stroke-width": 3 }, plane.over);
      draggable(sq, ev => plane.toData(ev), (px, py) => { l.px = clamp(px); l.py = clamp(py); unpreset(); drawPlane(); });
      draggable(ci, ev => plane.toData(ev), (px, py) => { l.dir = Math.atan2(py - l.py, px - l.px) * 180 / Math.PI; unpreset(); drawPlane(); });
      handleEls.push(sq, ci);
    });
  }
  const clamp = (v: number) => Math.max(-0.4, Math.min(5.4, v));

  function drawPlane(): void {
    omS.set(omega); sirtCb.checked = showSirt;
    const L2 = lines.map(toLine);
    lines.forEach((l, i) => {
      const t = l.dir * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
      lineMarks[i].set([l.px - 10 * c, l.px + 10 * c], [l.py - 10 * s, l.py + 10 * s]);
      const [sq, ci] = [handleEls[2 * i], handleEls[2 * i + 1]];
      sq.setAttribute("x", String(plane.x(l.px) - 8)); sq.setAttribute("y", String(plane.y(l.py) - 8));
      ci.setAttribute("cx", String(plane.x(l.px + 1.3 * c))); ci.setAttribute("cy", String(plane.y(l.py + 1.3 * s)));
    });
    const sweeps = Math.ceil(Math.max(shown, 1) / lines.length);
    const path = M.artPath2(L2, [0, 0], sweeps, omega).slice(0, shown + 1);
    pathMark.set(path.map(p => p[0]), path.map(p => p[1]));
    pathDots.set(path.map(p => p[0]), path.map(p => p[1]));
    const ls = M.lsq2(L2);
    lsMark.set(ls ? ls[0] : -9, ls ? ls[1] + 0.05 : -9, "★");
    const sp = M.sirtPath2(L2, [0, 0], Math.max(1, Math.ceil(shown / lines.length)), omega);
    sirtMark.set(showSirt ? sp.map(p => p[0]) : [0], showSirt ? sp.map(p => p[1]) : [0]);
    const last = path[path.length - 1], lastS = sp[sp.length - 1];
    let txt = `${lines.length} measurement lines\nlines:` + L2.map((l, i) => `\n  ${l.a[0].toFixed(2)} x₁ + ${l.a[1].toFixed(2)} x₂ = ${l.b.toFixed(2)}`).join("");
    txt += `\nART after ${shown} steps: (${last[0].toFixed(3)}, ${last[1].toFixed(3)})`;
    if (showSirt) txt += `\nSIRT after ${sp.length - 1} its: (${lastS[0].toFixed(3)}, ${lastS[1].toFixed(3)})`;
    txt += ls ? `\nleast squares ★: (${ls[0].toFixed(3)}, ${ls[1].toFixed(3)})` : "\nlines parallel: no unique solution";
    ro.textContent = txt;
  }

  // ---------- image mode ----------
  function rebuildSystem(): void { sys = M.smallSystem(n, sets); rebuildData(); }
  function rebuildData(): void { b = M.apply(sys, truth); x = new Float64Array(n * n); next = 0; }

  // a cell (col, row) occupies [col, col+1] x [n-1-row, n-row] in plot coordinates
  function drawGrid(p: Plot, vals: ArrayLike<number>, hi: number[] | null): void {
    p.clear();
    p.setDomain([-0.6, n + 0.6], [-0.6, n + 0.6]);
    p.heatmap(Array.from(vals), n, n, { x0: 0, x1: n, y0: 0, y1: n, lo: -1, hi: 6, numbers: true, gridLines: true, fmt: v => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2)) });
    for (let i = 0; i < n * n; i++) {
      const c = i % n, r = Math.floor(i / n);
      p.text(c + 0.12, n - r - 0.22, LETTERS[i], { size: 15, color: "var(--qb-muted)" });
    }
    if (hi) for (const i of hi) {
      const c = i % n, r = Math.floor(i / n);
      p.line([c, c + 1, c + 1, c, c], [n - r, n - r, n - r - 1, n - r - 1, n - r], { color: "var(--qb-c4)", width: 4 });
    }
  }

  function rayLine(p: Plot, ray: M.Ray, color: string, w: number): void {
    const pts = ray.pix.map(i => [i % n + 0.5, n - Math.floor(i / n) - 0.5]);
    const a = pts[0], z = pts[pts.length - 1];
    let dx = z[0] - a[0], dy = z[1] - a[1];
    if (pts.length === 1) { dx = ray.set === "diag" ? 1 : -1; dy = ray.set === "diag" ? -1 : -1; }
    const len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
    p.arrow(a[0] - 0.55 * ux, a[1] - 0.55 * uy, z[0] + 0.55 * ux, z[1] + 0.55 * uy, { color, width: w });
  }

  function drawImage(): void {
    sizeBtns.forEach((bt, i) => bt.classList.toggle("active", [2, 3][i] === n));
    setBtns.forEach((bt, i) => bt.classList.toggle("active", sets.includes(allSets[i])));
    omS.set(omega);
    const cur = sys.rays[next];
    drawGrid(tPlot, truth, null);
    sys.rays.forEach((r, i) => rayLine(tPlot, r, i === next ? "var(--qb-c4)" : "rgba(31,119,180,.45)", i === next ? 4 : 2));
    drawGrid(ePlot, x, cur.pix);
    (estFig.querySelector("figcaption") as HTMLElement).textContent = `estimate x (orange: next ray, ${cur.label})`;
    // table
    const ax = M.apply(sys, x);
    let h = `<table class="as-A"><tr><th>ray</th>${Array.from({ length: n * n }, (_, i) => `<th>${LETTERS[i]}</th>`).join("")}<th>data</th><th>sum of x</th><th>residual</th></tr>`;
    sys.rays.forEach((r, i) => {
      h += `<tr class="${i === next ? "cur" : ""}"><th>${r.label}</th>`;
      for (let p = 0; p < n * n; p++) h += r.pix.includes(p) ? `<td class="one">1</td>` : `<td class="zero">0</td>`;
      h += `<td>${fmt(b[i])}</td><td>${fmt(ax[i])}</td><td>${fmt(b[i] - ax[i])}</td></tr>`;
    });
    tab.innerHTML = h + "</table>";
    const rk = M.rankOf(sys), ns = M.nullImages(sys);
    let txt = `A: ${sys.rays.length} rays × ${n * n} pixels, rank ${rk}`;
    txt += ns.length ? `\nnull space: ${ns.length} image${ns.length > 1 ? "s" : ""}, e.g.\n` + gridText(ns[0]) : "\nnull space: only 0 (data decide the image)";
    txt += `\nx = (${Array.from(x).map(fmt).join(", ")})`;
    ro.textContent = txt;
  }
  const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
  const gridText = (v: ArrayLike<number>) => Array.from({ length: n }, (_, r) => "  " + Array.from({ length: n }, (_, c) => fmt(v[r * n + c]).padStart(5)).join(" ")).join("\n");

  function layoutMode(): void {
    const pix = mode === "pixels";
    planeFig.style.display = pix ? "" : "none";
    pixBox.style.display = pix ? "" : "none";
    truthFig.style.display = pix ? "none" : "";
    estFig.style.display = pix ? "none" : "";
    tab.style.display = pix ? "none" : "";
    imgBox.style.display = pix ? "none" : "";
    modeBtns.forEach((bt, i) => bt.classList.toggle("active", ["pixels", "image"][i] === mode));
    if (pix) drawPlane(); else drawImage();
  }

  function unpreset(): void { presetBtns.forEach(bt => bt.classList.remove("active")); }

  function load(p: Preset): void {
    mode = p.mode;
    omega = p.omega ?? 1;
    if (p.mode === "pixels") {
      lines = (p.lines ?? []).map(l => ({ ...l }));
      shown = 4 * lines.length; showSirt = !!p.sirt;
      rebuildHandles();
    } else {
      n = p.n ?? 2; truth = [...(p.truth ?? [1, 2, 3, 4])]; sets = [...(p.sets ?? ["rows", "cols"])];
      rebuildSystem();
      for (let k = 0; k < (p.steps ?? 0); k++) { x = M.kaczmarzStep(sys, b, x, next, omega); next = (next + 1) % sys.rays.length; }
    }
    presetBtns.forEach((bt, i) => bt.classList.toggle("active", PRESETS[i] === p));
    layoutMode();
  }

  // click on the true image: +1 (cycles 0..5)
  tPlot.svg.addEventListener("pointerdown", ev => {
    const [px, py] = tPlot.toData(ev);
    const c = Math.floor(px), r = n - 1 - Math.floor(py);
    if (c < 0 || c >= n || r < 0 || r >= n) return;
    truth[r * n + c] = (truth[r * n + c] + 1) % 6;
    unpreset(); rebuildData(); drawImage();
  });
  tPlot.svg.style.cursor = "pointer";

  load(p0);
}

autoMount(NAME, mount);
