// flowtrack: how a flow field becomes objects (the Cellpose family), on a 48 x 48 grid.
// Four panels: (1) the training target: heat spread inside each cell after k averaging steps
// (play), with its source; (2) the "predicted" arrows (the exact target plus the errors you set:
// noise, blur, one shared field for touching cells, debris with random arrows), colour wheel +
// arrows; (3) the particles: every foreground pixel follows the arrows (play / step); (4) end
// points, sinks (seeds +), masks, and the flow check (error per mask, dropped masks hatched).
// Click panel 2 or 3 to follow one pixel. Draw your own cells on panel 1. A grid mode shows the
// 5 x 5 field of exercise B2 (lib/gridui GridView): click a cell to follow it, or turn arrows.
// Auto-mounts on <div data-demo="flowtrack">. Options: data-preset = start | heat | bend |
// fewsteps | noise | check | shared | grid.
import { htmlEl, slider, presets, button, checkbox, autoMount, type Slider } from "../lib/ui";
import { labelColor, inferno } from "../lib/colour";
import { GridView } from "../lib/gridui";
import { iouTable, match, objectScores } from "../lib/eval";
import * as M from "./model";

const NAME = "flowtrack";
const N = M.SIZE;
const S = 6; // canvas pixels per grid pixel
const W = N * S;

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .ft-left { flex: 1 1 520px; min-width: 280px; max-width: 640px; }
.qb-demo[data-demo="${NAME}"] .ft-panels { display: grid; grid-template-columns: 1fr 1fr; gap: .4em .6em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .1em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qb-demo[data-demo="${NAME}"] canvas { display: block; width: 100%; aspect-ratio: 1; border-radius: 3px; touch-action: none; }
.qb-demo[data-demo="${NAME}"] canvas.ft-click { cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .ft-grid { display: flex; gap: 1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .ft-grid > div { flex: 0 1 380px; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 440px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .ft-lab { font-size: 14px; color: var(--qb-muted); margin-top: .15em; }
.qb-demo[data-demo="${NAME}"] .ft-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .ft-hide { display: none !important; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .ft-left { flex: 0 1 540px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 520px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .06em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .ft-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .2em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State {
  shape: M.ShapeId | "own";
  autoK: boolean; k: number;
  err: M.Errors;
  steps: number; eta: number; thr: number;
  click: number; // clicked pixel (-1 none)
}
interface Preset { id: string; label: string; grid?: boolean; s: Partial<Omit<State, "err">> & { err?: Partial<M.Errors> } }

const E0: M.Errors = { noise: 0, blur: 0, shared: false, debris: false };
const S0: State = { shape: "disc", autoK: true, k: 100, err: { ...E0 }, steps: 100, eta: 1, thr: 0.4, click: -1 };
const C_TIP = 12 * N + 30; // a pixel near the upper tip of the C
const PRESETS: Preset[] = [
  { id: "start", label: "start (disc)", s: {} },
  { id: "heat", label: "heat spreads", s: { shape: "C", autoK: false, k: 10 } },
  { id: "bend", label: "arrows follow the bend", s: { shape: "C", click: C_TIP } },
  { id: "fewsteps", label: "too few steps", s: { shape: "C", steps: 10, click: C_TIP } },
  { id: "noise", label: "noise: extra sink", s: { shape: "C", err: { noise: 0.35 }, thr: 0 } },
  { id: "check", label: "flow check drops a bad mask", s: { shape: "disc", err: { debris: true } } },
  { id: "shared", label: "shared field: merge", s: { shape: "touching", err: { shared: true } } },
  { id: "grid", label: "B2 grid", grid: true, s: {} },
];

const HUE = (vx: number, vy: number, m: number): [number, number, number] => {
  // the colour wheel of cellposenet (Cellpose's dx_to_circ): direction = hue, length = brightness
  const v = (255 * Math.min(1, Math.max(0, m))) / 2;
  const ang = Math.atan2(vx, vy) + Math.PI;
  return [v * (Math.cos(ang) + 1), v * (Math.cos(ang + (2 * Math.PI) / 3) + 1), v * (Math.cos(ang + (4 * Math.PI) / 3) + 1)];
};

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find((p) => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...S0, err: { ...E0 } };
  let own: Int32Array = new Int32Array(N * N);
  let lab = M.makeShape("disc");
  let pred: M.Prediction, dec: M.Decoded, frames: Float32Array[] = [], fgIdx: number[] = [];
  let showStep = st.steps;
  let timer = 0;

  // ---- layout
  const left = htmlEl("div", "ft-left", "", host);
  const panelsEl = htmlEl("div", "ft-panels", "", left);
  const mk = (cap: string, click = false) => {
    const f = htmlEl("figure", "", "", panelsEl);
    const c = htmlEl("figcaption", "", cap, f);
    const cv = htmlEl("canvas", click ? "ft-click" : "", "", f);
    cv.width = W; cv.height = W;
    return { cap: c, cv, ctx: cv.getContext("2d")! };
  };
  const pHeat = mk("1 · target: heat inside the cell", true);
  const pFlow = mk("2 · predicted arrows", true);
  const pPart = mk("3 · particles", true);
  const pRes = mk("4 · sinks and masks");
  const gridBox = htmlEl("div", "ft-grid ft-hide", "", left);
  const gridHost = htmlEl("div", "", "", gridBox);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "ft-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p) })));

  // field-mode controls
  const fieldCtl = htmlEl("div", "", "", panel);
  fieldCtl.style.display = "contents";
  htmlEl("div", "ft-lab", "cells (draw on panel 1)", fieldCtl);
  const shapeRow = htmlEl("div", "ft-row", "", fieldCtl);
  const shapeBtns = M.SHAPES.map((s) => button(shapeRow, s.label, () => { st.shape = s.id; st.click = -1; lab = M.makeShape(s.id); run(); }));
  let drawMode: "off" | "add" | "erase" = "off";
  const drawRow = htmlEl("div", "ft-row", "", fieldCtl);
  const addBtn = button(drawRow, "draw a cell", () => setDraw(drawMode === "add" ? "off" : "add"));
  const eraseBtn = button(drawRow, "erase", () => setDraw(drawMode === "erase" ? "off" : "erase"));
  button(drawRow, "clear", () => { own = new Int32Array(N * N); st.shape = "own"; lab = own; st.click = -1; setDraw("add"); run(); });
  function setDraw(m: typeof drawMode): void {
    drawMode = m;
    addBtn.classList.toggle("active", m === "add"); eraseBtn.classList.toggle("active", m === "erase");
  }

  htmlEl("div", "ft-lab", "training target", fieldCtl);
  const kRow = htmlEl("div", "ft-row", "", fieldCtl);
  const kS: Slider = slider(kRow, "heat steps k", 0, 200, 1, st.k, (v) => { st.k = v; st.autoK = false; autoBox.checked = false; run(); });
  (kS.input.parentElement as HTMLElement).style.flex = "1 1 200px";
  button(kRow, "play", () => playHeat());
  const autoBox = checkbox(fieldCtl, "k automatic: 2 × (width + height) of the cell", true, (v) => { st.autoK = v; run(); });
  htmlEl("div", "ft-lab", "network errors (simulated)", fieldCtl);
  const noiseS = slider(fieldCtl, "noise on arrows", 0, 1.5, 0.05, 0, (v) => { st.err.noise = v; run(); });
  const blurS = slider(fieldCtl, "blur", 0, 4, 0.25, 0, (v) => { st.err.blur = v; run(); });
  const errRow = htmlEl("div", "ft-row", "", fieldCtl);
  const sharedBox = checkbox(errRow, "shared field (touching cells)", false, (v) => { st.err.shared = v; run(); });
  const debrisBox = checkbox(errRow, "debris", false, (v) => { st.err.debris = v; run(); });
  htmlEl("div", "ft-lab", "decoding", fieldCtl);
  const stepsS = slider(fieldCtl, "Euler steps", 0, 200, 1, st.steps, (v) => { st.steps = v; showStep = v; run(); });
  const etaS = slider(fieldCtl, "step length", 0.25, 2, 0.25, st.eta, (v) => { st.eta = v; run(); });
  const thrS = slider(fieldCtl, "flow threshold", 0, 2, 0.05, st.thr, (v) => { st.thr = v; run(); }, (v) => (v === 0 ? "off" : v.toFixed(2)));
  const showRow = htmlEl("div", "ft-row", "", fieldCtl);
  const showS: Slider = slider(showRow, "show step", 0, 200, 1, st.steps, (v) => { stopAnim(); showStep = Math.min(v, st.steps); drawParticles(); });
  (showS.input.parentElement as HTMLElement).style.flex = "1 1 200px";
  button(showRow, "play", () => playParticles());

  // grid-mode controls
  const gridCtl = htmlEl("div", "ft-hide", "", panel);
  gridCtl.style.flexDirection = "column"; gridCtl.style.gap = ".3em";
  htmlEl("div", "qb-note", "Exercise B2: follow the arrows one cell per step (· = no movement). Click a cell to see its path.", gridCtl);
  let gridEdit = false, showGroups = false;
  let field = M.B2_FIELD.slice();
  let gridStart = -1;
  const gRow = htmlEl("div", "ft-row", "", gridCtl);
  const traceBtn = button(gRow, "click: follow", () => { gridEdit = false; syncGridBtns(); });
  const turnBtn = button(gRow, "click: turn arrow", () => { gridEdit = true; syncGridBtns(); });
  const gRow2 = htmlEl("div", "ft-row", "", gridCtl);
  const groupBtn = button(gRow2, "where does every cell end?", () => { showGroups = !showGroups; drawGrid(); });
  button(gRow2, "reset field", () => { field = M.B2_FIELD.slice(); gridStart = -1; showGroups = false; drawGrid(); });
  function syncGridBtns(): void { traceBtn.classList.toggle("active", !gridEdit); turnBtn.classList.toggle("active", gridEdit); groupBtn.classList.toggle("active", showGroups); }

  const readout = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "The arrows are simulated: the exact training target plus the errors you set. Panel 4: red dots = end points (size = number of particles), + = seeds, numbers = flow error, hatched = dropped by the flow check.", panel);

  // ---- grid view (built lazily)
  let gv: GridView | null = null;
  function ensureGrid(): GridView {
    if (gv) return gv;
    gv = new GridView(gridHost, 5, 5, field, {
      cell: 64, display: "int", lo: 0, hi: 8, numbers: true, indices: true, edit: false,
      map: () => [1, 1, 1], text: (v) => M.GLYPH[v], ariaLabel: "5 by 5 flow field",
    });
    gv.svg.addEventListener("pointerdown", (ev) => {
      const hit = gv!.cellAt(ev);
      if (!hit) return;
      if (gridEdit) field[hit.index] = (field[hit.index] + 1) % 9;
      else gridStart = hit.index;
      drawGrid();
    });
    return gv;
  }
  const GROUP_COL = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"];
  function drawGrid(): void {
    const g = ensureGrid();
    g.values = Int32Array.from(field);
    const marks: { index: number; color?: string; fill?: boolean; tag?: string }[] = [];
    const lines: string[] = [];
    const rc = (p: number) => `(${Math.floor(p / 5)}, ${p % 5})`;
    if (showGroups) {
      const groups = [...M.gridGroups(field, 5, 5).entries()].sort((a, b) => b[1].length - a[1].length);
      groups.forEach(([k, cells], i) => {
        const cyc = k.split(",").map(Number);
        for (const c of cells) marks.push({ index: c, color: GROUP_COL[i % GROUP_COL.length], fill: true });
        lines.push(`${cells.length} cells end ${cyc.length === 1 ? "at " + rc(cyc[0]) : "going back and forth between " + cyc.map(rc).join(" and ")}`);
      });
      lines.unshift(`${groups.length} groups = ${groups.length} objects`);
    }
    const arrows: { from: number; to: number; color?: string }[] = [];
    if (gridStart >= 0) {
      const f = M.gridFollow(field, 5, 5, gridStart);
      for (let i = 0; i + 1 < f.path.length; i++) arrows.push({ from: f.path[i], to: f.path[i + 1], color: "var(--qb-c2, #d62728)" });
      const last = f.path[f.path.length - 1];
      const [dr, dc] = M.DIRS[field[last]];
      const nxt = Math.max(0, Math.min(4, Math.floor(last / 5) + dr)) * 5 + Math.max(0, Math.min(4, (last % 5) + dc));
      if (nxt !== last) arrows.push({ from: last, to: nxt, color: "var(--qb-c2, #d62728)" });
      f.path.forEach((p, i) => marks.push({ index: p, color: "var(--qb-c2, #d62728)", tag: String(i) }));
      const path = f.path.map(rc).join(" → ");
      lines.unshift(`from ${rc(gridStart)}: ${path} → ` + (f.cycle.length === 1 ? "stays: a sink" : `back to ${rc(f.cycle[0])}: stuck between ${f.cycle.map(rc).join(" and ")}`));
    }
    g.show({ values: g.values, marks, arrows, focus: gridStart >= 0 ? gridStart : null });
    syncGridBtns();
    readout.textContent = lines.join("\n") || "Click a cell to follow its arrows.";
  }

  function setGridMode(on: boolean): void {
    panelsEl.classList.toggle("ft-hide", on);
    gridBox.classList.toggle("ft-hide", !on);
    fieldCtl.style.display = on ? "none" : "contents";
    gridCtl.classList.toggle("ft-hide", !on);
    gridCtl.style.display = on ? "flex" : "";
    note.classList.toggle("ft-hide", on);
  }

  // ---- presets
  function applyPreset(p: Preset): void {
    stopAnim();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    if (p.grid) {
      setGridMode(true);
      field = M.B2_FIELD.slice(); gridStart = -1; showGroups = false; gridEdit = false;
      drawGrid();
      return;
    }
    setGridMode(false);
    const s = { ...S0, ...p.s, err: { ...E0, ...(p.s.err ?? {}) } } as State;
    Object.assign(st, s);
    lab = M.makeShape(st.shape === "own" ? "disc" : st.shape);
    kS.set(st.k); autoBox.checked = st.autoK;
    noiseS.set(st.err.noise); blurS.set(st.err.blur); sharedBox.checked = st.err.shared; debrisBox.checked = st.err.debris;
    stepsS.set(st.steps); etaS.set(st.eta); thrS.set(st.thr);
    showStep = st.steps;
    setDraw("off");
    run();
  }

  // ---- computation
  function run(): void {
    stopAnim();
    const iters = st.autoK ? null : st.k;
    pred = M.predict(lab, N, N, st.err, iters);
    const f = M.followFlows(pred.fg, pred.vx, pred.vy, N, N, st.steps, st.eta, true);
    frames = f.frames!;
    fgIdx = []; for (let p = 0; p < N * N; p++) if (pred.fg[p]) fgIdx.push(p);
    dec = M.decode(pred.fg, pred.vx, pred.vy, N, N, { steps: st.steps, eta: st.eta, flowThreshold: st.thr, iters });
    showStep = Math.min(showStep, st.steps);
    showS.input.max = String(st.steps); showS.set(showStep);
    shapeBtns.forEach((b, i) => b.classList.toggle("active", M.SHAPES[i].id === st.shape));
    if (st.autoK) kS.set(Math.max(...pred.target.iters, 0));
    draw();
  }

  // ---- drawing
  const img = (ctx: CanvasRenderingContext2D, rgb: (p: number) => [number, number, number]) => {
    const id = ctx.createImageData(W, W), d = id.data;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const c = rgb(y * N + x);
      for (let dy = 0; dy < S; dy++) for (let dx = 0; dx < S; dx++) {
        const o = 4 * ((y * S + dy) * W + x * S + dx);
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
      }
    }
    ctx.putImageData(id, 0, 0);
  };
  const BG: [number, number, number] = [28, 28, 32];
  function outline(ctx: CanvasRenderingContext2D, L: ArrayLike<number>, col: string, lw = 1.5): void {
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const v = L[y * N + x];
      if (!v) continue;
      if (x === N - 1 || L[y * N + x + 1] !== v) { ctx.moveTo((x + 1) * S, y * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
      if (x === 0 || L[y * N + x - 1] !== v) { ctx.moveTo(x * S, y * S); ctx.lineTo(x * S, (y + 1) * S); }
      if (y === N - 1 || L[(y + 1) * N + x] !== v) { ctx.moveTo(x * S, (y + 1) * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
      if (y === 0 || L[(y - 1) * N + x] !== v) { ctx.moveTo(x * S, y * S); ctx.lineTo((x + 1) * S, y * S); }
    }
    ctx.stroke();
  }
  const cx = (x: number) => (x + 0.5) * S;
  function star(ctx: CanvasRenderingContext2D, p: number, col = "#ff3b3b"): void {
    const x = cx(p % N), y = cx(Math.floor(p / N)), r = 7;
    ctx.fillStyle = col; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }

  function drawHeat(): void {
    const H = pred.target.heat, tl = pred.targetLab;
    img(pHeat.ctx, (p) => {
      if (!tl[p]) return BG;
      const v = H[p];
      const t = v > 0 ? Math.max(0, 1 + Math.log10(v) / 2.5) : 0; // log scale, 2.5 decades
      const c = inferno(0.08 + 0.92 * t);
      return [c[0] * 255, c[1] * 255, c[2] * 255];
    });
    outline(pHeat.ctx, lab, "rgba(255,255,255,.8)");
    pred.target.sources.forEach((s) => { if (s >= 0) star(pHeat.ctx, s); });
    const ks = [...new Set(pred.target.iters)].join(", ");
    pHeat.cap.textContent = `1 · target heat, k = ${ks || 0}`;
  }

  function drawArrows(ctx: CanvasRenderingContext2D, vx: Float32Array, vy: Float32Array, fg: ArrayLike<number>, col = "rgba(255,255,255,.9)"): void {
    ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath();
    for (let y = 0; y < N; y += 2) for (let x = 0; x < N; x += 2) {
      const p = y * N + x;
      if (!fg[p]) continue;
      const u = vx[p], v = vy[p], m = Math.hypot(u, v);
      if (m < 1e-6) { ctx.moveTo(cx(x) + 1.5, cx(y)); ctx.arc(cx(x), cx(y), 1.5, 0, 2 * Math.PI); continue; }
      const L = 0.9 * S * Math.min(1, m), ux = u / m, uy = v / m;
      const x0 = cx(x) - ux * L / 2, y0 = cx(y) - uy * L / 2, x1 = cx(x) + ux * L / 2, y1 = cx(y) + uy * L / 2;
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - 3 * ux + 2.2 * uy, y1 - 3 * uy - 2.2 * ux);
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - 3 * ux - 2.2 * uy, y1 - 3 * uy + 2.2 * ux);
    }
    ctx.stroke();
  }

  function drawWheel(ctx: CanvasRenderingContext2D): void {
    const r = 18, X = W - r - 4, Y = r + 4;
    for (let a = 0; a < 360; a += 6) {
      const t = (a * Math.PI) / 180, c = HUE(Math.cos(t), Math.sin(t), 1);
      ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
      ctx.beginPath(); ctx.moveTo(X, Y); ctx.arc(X, Y, r, t - 0.06, t + 0.12); ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.stroke();
  }

  function clickedPath(): Float32Array | null {
    if (st.click < 0 || !pred.fg[st.click]) return null;
    return M.tracePath(pred.vx, pred.vy, N, N, st.click % N, Math.floor(st.click / N), st.steps, st.eta);
  }
  function drawPath(ctx: CanvasRenderingContext2D, path: Float32Array | null, upto: number): void {
    if (!path) return;
    const n = Math.min(path.length / 2 - 1, upto);
    ctx.lineJoin = "round";
    for (const [c, w] of [["#000", 5], ["#ffeb3b", 2.5]] as [string, number][]) {
      ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath();
      for (let j = 0; j <= n; j++) { const X = cx(path[2 * j]), Y = cx(path[2 * j + 1]); if (j === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
      ctx.stroke();
    }
    ctx.fillStyle = "#ffeb3b"; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx(path[0]), cx(path[1]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.rect(cx(path[2 * n]) - 4, cx(path[2 * n + 1]) - 4, 8, 8); ctx.fill(); ctx.stroke();
  }

  function drawFlow(): void {
    img(pFlow.ctx, (p) => (pred.fg[p] ? HUE(pred.vx[p], pred.vy[p], Math.hypot(pred.vx[p], pred.vy[p])) : BG));
    drawArrows(pFlow.ctx, pred.vx, pred.vy, pred.fg);
    outline(pFlow.ctx, lab, "rgba(255,255,255,.35)", 1);
    drawWheel(pFlow.ctx);
    drawPath(pFlow.ctx, clickedPath(), st.steps);
  }

  function drawParticles(): void {
    const ctx = pPart.ctx;
    img(ctx, (p) => (pred.fg[p] ? [70, 70, 78] : BG));
    outline(ctx, lab, "rgba(255,255,255,.35)", 1);
    const f = frames[Math.min(showStep, frames.length - 1)];
    ctx.fillStyle = "rgba(255,255,255,.85)";
    if (f) for (let k = 0; k < fgIdx.length; k++) ctx.fillRect(cx(f[2 * k]) - 1.5, cx(f[2 * k + 1]) - 1.5, 3, 3);
    drawPath(ctx, clickedPath(), showStep);
    pPart.cap.textContent = `3 · particles, step ${showStep} of ${st.steps}`;
  }

  function drawResult(): void {
    const ctx = pRes.ctx;
    img(ctx, (p) => {
      const l = dec.raw[p];
      if (!l) return pred.fg[p] ? [60, 60, 66] : BG;
      const c = labelColor(l);
      const a = dec.dropped[l - 1] ? 0.35 : 0.75;
      return [BG[0] * (1 - a) + 255 * c[0] * a, BG[1] * (1 - a) + 255 * c[1] * a, BG[2] * (1 - a) + 255 * c[2] * a];
    });
    // hatching of dropped masks
    ctx.save();
    ctx.strokeStyle = "rgba(255,60,60,.9)"; ctx.lineWidth = 1.2; ctx.beginPath();
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const l = dec.raw[y * N + x];
      if (l && dec.dropped[l - 1] && (x + y) % 2 === 0) { ctx.moveTo(x * S, (y + 1) * S); ctx.lineTo((x + 1) * S, y * S); }
    }
    ctx.stroke(); ctx.restore();
    outline(ctx, dec.raw, "rgba(255,255,255,.9)", 1.5);
    outline(ctx, lab, "rgba(255,255,0,.55)", 1);
    // end-point histogram
    ctx.fillStyle = "rgba(255,40,40,.9)";
    ctx.beginPath();
    for (let p = 0; p < N * N; p++) if (dec.hist[p] > 0) {
      const r = Math.min(9, 0.8 + Math.sqrt(dec.hist[p]) * 0.6);
      ctx.moveTo(cx(p % N) + r, cx(Math.floor(p / N))); ctx.arc(cx(p % N), cx(Math.floor(p / N)), r, 0, 2 * Math.PI);
    }
    ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath();
    for (const s of dec.seeds) { const X = cx(s % N), Y = cx(Math.floor(s / N)); ctx.moveTo(X - 7, Y); ctx.lineTo(X + 7, Y); ctx.moveTo(X, Y - 7); ctx.lineTo(X, Y + 7); }
    ctx.stroke();
    // flow error per mask
    const sx = new Float64Array(dec.nRaw + 1), sy = new Float64Array(dec.nRaw + 1), sn = new Float64Array(dec.nRaw + 1);
    for (let p = 0; p < N * N; p++) { const l = dec.raw[p]; if (l) { sx[l] += p % N; sy[l] += Math.floor(p / N); sn[l]++; } }
    ctx.font = "700 17px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let l = 1; l <= dec.nRaw; l++) {
      const X = cx(sx[l] / sn[l]), Y = cx(sy[l] / sn[l]) + 14, t = dec.errors[l - 1].toFixed(2);
      ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.strokeText(t, X, Y);
      ctx.fillStyle = dec.dropped[l - 1] ? "#ff6b6b" : "#fff"; ctx.fillText(t, X, Y);
    }
  }

  function draw(): void {
    drawHeat(); drawFlow(); drawParticles(); drawResult(); updateReadout();
  }

  function updateReadout(): void {
    let nTrue = 0; for (const v of lab) if (v > nTrue) nTrue = v;
    const lines: string[] = [];
    lines.push(`cells: ${nTrue} · pixels followed: ${fgIdx.length} for ${st.steps} steps`);
    lines.push(`sinks (seeds): ${dec.seeds.length} → masks: ${dec.nRaw}${dec.small ? ` (+${dec.small} too small)` : ""}`);
    if (dec.nRaw) {
      const e = dec.errors.map((x, i) => `${x.toFixed(2)}${dec.dropped[i] ? " ✗" : ""}`).join(", ");
      lines.push(`flow error per mask: ${e}${st.thr > 0 ? ` (threshold ${st.thr.toFixed(2)})` : " (check off)"}`);
    }
    const sc = objectScores(match(iouTable(lab, dec.masks), 0.5));
    let assigned = 0; for (const v of dec.masks) if (v) assigned++;
    lines.push(`final: ${dec.count} object${dec.count === 1 ? "" : "s"}, ${assigned} pixels · TP ${sc.tp} FP ${sc.fp} FN ${sc.fn}`);
    if (st.click >= 0 && pred.fg[st.click]) {
      const path = clickedPath()!, n = path.length / 2 - 1;
      lines.push(`clicked (${st.click % N}, ${Math.floor(st.click / N)}) → (${path[2 * n].toFixed(1)}, ${path[2 * n + 1].toFixed(1)})`);
    }
    readout.textContent = lines.join("\n");
  }

  // ---- animation
  function stopAnim(): void { if (timer) { clearInterval(timer); timer = 0; } }
  function playParticles(): void {
    stopAnim();
    showStep = 0;
    timer = window.setInterval(() => {
      showStep = Math.min(st.steps, showStep + Math.max(1, Math.round(st.steps / 60)));
      showS.set(showStep); drawParticles();
      if (showStep >= st.steps) stopAnim();
    }, 50);
  }
  function playHeat(): void {
    stopAnim();
    const target = st.autoK ? Math.max(...pred.target.iters, 1) : Math.max(st.k, 120);
    let k = 0;
    st.autoK = false; autoBox.checked = false;
    timer = window.setInterval(() => {
      k = Math.min(target, k + Math.max(1, Math.round(k * 0.08)));
      st.k = k; kS.set(k);
      pred = M.predict(lab, N, N, st.err, k);
      drawHeat();
      if (k >= target) { stopAnim(); run(); }
    }, 60);
  }

  // ---- pointer
  const pixelAt = (cv: HTMLCanvasElement, ev: PointerEvent): number => {
    const r = cv.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * N), y = Math.floor(((ev.clientY - r.top) / r.height) * N);
    return x < 0 || y < 0 || x >= N || y >= N ? -1 : y * N + x;
  };
  for (const pnl of [pFlow, pPart]) pnl.cv.addEventListener("pointerdown", (ev) => {
    const p = pixelAt(pnl.cv, ev);
    if (p < 0) return;
    st.click = p; drawFlow(); drawParticles(); updateReadout();
  });
  let painting = false, paintLabel = 0;
  function paintAt(ev: PointerEvent): void {
    const p = pixelAt(pHeat.cv, ev);
    if (p < 0) return;
    const x0 = p % N, y0 = Math.floor(p / N);
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      if (dx * dx + dy * dy > 5) continue;
      const x = x0 + dx, y = y0 + dy;
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      own[y * N + x] = drawMode === "erase" ? 0 : paintLabel;
    }
  }
  pHeat.cv.addEventListener("pointerdown", (ev) => {
    if (drawMode === "off") return;
    if (st.shape !== "own") { own = lab.slice(); st.shape = "own"; }
    lab = own;
    let mx = 0; for (const v of own) if (v > mx) mx = v;
    paintLabel = mx + 1;
    painting = true;
    pHeat.cv.setPointerCapture(ev.pointerId);
    paintAt(ev); run();
  });
  pHeat.cv.addEventListener("pointermove", (ev) => { if (painting) { paintAt(ev); run(); } });
  const end = () => { if (painting) { painting = false; own = M.relabel(own).labels; lab = own; run(); } };
  pHeat.cv.addEventListener("pointerup", end);
  pHeat.cv.addEventListener("pointercancel", end);

  applyPreset(p0);
}

autoMount(NAME, mount);
