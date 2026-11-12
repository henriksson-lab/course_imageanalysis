// Grid step-through UI (owner: morphology day; used by morphgrid and its label/hitmiss modes,
// floodgrid, dpgrid, satstep, raster, ...). Two parts:
//
//  - GridView: a small integer/binary grid drawn as SVG cells (>= 24 px on a slide) with
//    numbers, click/drag editing, a highlighted neighbourhood / structuring-element window,
//    marked cells, a focus cell and arrows between cells. Use it alone, e.g. as the 3x3
//    structuring-element editor next to the main grid.
//  - gridStepper(): a GridView (plus an optional second "aux" grid, e.g. the output or the
//    DP table) with step / back / play / end / reset controls, a step counter and a log.
//
// An algorithm plugs in as a generator of frames (snapshots), so step-back is free:
//
//   const erode: GridAlgorithm = {
//     id: "erode", label: "erosion",
//     *run(inp, rows, cols) {
//       const out = new Int32Array(inp.length);
//       yield { values: inp, aux: out, log: "start" };
//       for (let i = 0; i < inp.length; i++) {
//         out[i] = ...;
//         yield { values: inp, aux: out.slice(), focus: i, window: { at: i, offsets: CROSS }, log: `(${r}, ${c}) -> ${out[i]}` };
//       }
//     },
//   };
//   gridStepper(host, { rows: 8, cols: 8, initial, algorithms: [erode], edit: "toggle" });
//
// All frames are generated up front (grids are small; at most `maxFrames`). Editing the
// input regenerates them and goes back to the first frame. Pointer only: no keyboard.
import { svgEl, htmlEl, button, slider } from "./ui";
import { ColorMap, viridis, grey, labelColor, textColorOn, RGB, toCss } from "./colour";

/** How cell values are shown: binary (0 white / 1 dark), int / float (grey or colour-mapped
 *  with numbers), label (label palette, 0 = background). */
export type CellDisplay = "binary" | "int" | "float" | "label";

/** Editing rule for a click on a cell: toggle 0/1, cycle in [min,max], plus/minus (upper half
 *  of the cell +1, lower half -1), or any function old -> new. */
export type EditRule = "toggle" | { min: number; max: number } | "plusminus" | ((old: number, index: number) => number) | false;

export interface CellMark {
  index: number;
  /** CSS colour of the outline (default the accent colour). */
  color?: string;
  /** Fill the cell lightly instead of outlining it. */
  fill?: boolean;
  /** Small text in the cell corner (e.g. a queue position, a sign "+" / "−"). */
  tag?: string;
}

export interface CellArrow { from: number; to: number; color?: string }

/** A neighbourhood / structuring element drawn around cell `at`: offsets [dRow, dCol]. */
export interface GridWindow {
  at: number;
  offsets: [number, number][];
  /** Mark the origin cell of the window (default true). */
  origin?: boolean;
  color?: string;
  /** Per-offset state, e.g. hit (true) / miss (false); shown as a filled / empty dot. */
  hits?: boolean[];
}

export interface GridViewOptions {
  /** Cell size in SVG units (default 44; keep >= 24 rendered px on slides). */
  cell?: number;
  display?: CellDisplay;
  /** Write the value in each cell (default true except for binary). */
  numbers?: boolean;
  /** Value range for int/float colouring (default: from the data). */
  lo?: number;
  hi?: number;
  /** Colour map for int/float (default grey, light = low). */
  map?: ColorMap;
  edit?: EditRule;
  /** Called after each edit with the new values (on every cell during a drag). */
  onEdit?: (values: Int32Array, index: number) => void;
  /** Text override per cell (e.g. "∞" for a large number, "" to hide). */
  text?: (v: number, index: number) => string | undefined;
  title?: string;
  /** Row/column index labels around the grid. */
  indices?: boolean;
  ariaLabel?: string;
}

const ACCENT = "var(--qb-c2, #d62728)";

function css(el: Element, props: Record<string, string | number | undefined>): void {
  const s = (el as SVGElement).style;
  for (const [k, v] of Object.entries(props)) if (v !== undefined) s.setProperty(k, String(v));
}

/** A rows x cols grid of cells in SVG. Values are stored as Int32Array (row-major). */
export class GridView {
  readonly svg: SVGSVGElement;
  rows: number;
  cols: number;
  values: Int32Array;
  opts: GridViewOptions;
  private cellsG: SVGGElement;
  private overG: SVGGElement;
  private focus: number | null = null;
  private win: GridWindow | null = null;
  private marks: CellMark[] = [];
  private arrows: CellArrow[] = [];
  private ox = 4;
  private oy = 4;

  constructor(parent: Element, rows: number, cols: number, values?: ArrayLike<number>, opts: GridViewOptions = {}) {
    this.rows = rows; this.cols = cols; this.opts = opts;
    this.values = values ? Int32Array.from(values) : new Int32Array(rows * cols);
    this.svg = svgEl("svg", { class: "qb-grid-svg", role: "img", "aria-label": opts.ariaLabel ?? opts.title ?? "grid" }, parent);
    this.cellsG = svgEl("g", {}, this.svg);
    this.overG = svgEl("g", { "pointer-events": "none" }, this.svg);
    this.layout();
    this.installEditing();
    this.draw();
  }

  get cell(): number { return this.opts.cell ?? 44; }

  private layout(): void {
    const c = this.cell, f = Math.max(14, c * 0.4), idx = this.opts.indices;
    this.ox = 4 + (idx ? f * 1.3 : 0);
    this.oy = 4 + (this.opts.title ? f * 1.5 : 0) + (idx ? f * 1.1 : 0);
    const W = this.ox + this.cols * c + 4, H = this.oy + this.rows * c + 4;
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("width", "100%");
    this.svg.style.maxWidth = `${W * 1.6}px`;
  }

  /** Change the grid size (values are reset to `values` or zeros). */
  resize(rows: number, cols: number, values?: ArrayLike<number>): void {
    this.rows = rows; this.cols = cols;
    this.values = values ? Int32Array.from(values) : new Int32Array(rows * cols);
    this.focus = null; this.win = null; this.marks = []; this.arrows = [];
    this.layout(); this.draw();
  }

  setValues(v: ArrayLike<number>): void { this.values = Int32Array.from(v); this.draw(); }
  setFocus(i: number | null): void { this.focus = i; this.drawOverlay(); }
  setWindow(w: GridWindow | null): void { this.win = w; this.drawOverlay(); }
  setMarks(m: CellMark[]): void { this.marks = m; this.drawOverlay(); }
  setArrows(a: CellArrow[]): void { this.arrows = a; this.drawOverlay(); }
  /** Set everything shown at once (values + overlays). */
  show(s: { values?: ArrayLike<number>; focus?: number | null; window?: GridWindow | null; marks?: CellMark[]; arrows?: CellArrow[] }): void {
    if (s.values) this.values = Int32Array.from(s.values);
    this.focus = s.focus ?? null; this.win = s.window ?? null; this.marks = s.marks ?? []; this.arrows = s.arrows ?? [];
    this.draw();
  }

  /** Centre of cell i in SVG units. */
  centre(i: number): [number, number] {
    const r = Math.floor(i / this.cols), c = i % this.cols;
    return [this.ox + (c + 0.5) * this.cell, this.oy + (r + 0.5) * this.cell];
  }

  private colourOf(v: number): RGB {
    const o = this.opts, d = o.display ?? "int";
    if (d === "binary") return v ? [0.16, 0.2, 0.3] : [1, 1, 1];
    if (d === "label") return labelColor(v, [1, 1, 1]);
    let lo = o.lo, hi = o.hi;
    if (lo === undefined || hi === undefined) {
      let a = Infinity, b = -Infinity;
      for (const x of this.values) if (Math.abs(x) < 1e8) { if (x < a) a = x; if (x > b) b = x; }
      lo ??= Math.min(0, a); hi ??= b;
    }
    const t = hi > lo ? (v - lo) / (hi - lo) : 0;
    // default: light cells for low values, so black numbers stay readable
    return (o.map ?? ((u: number) => grey(1 - 0.75 * u)))(Math.max(0, Math.min(1, t)));
  }

  /** Redraw the cells and the overlay. */
  draw(): void {
    const g = this.cellsG, c = this.cell, o = this.opts, f = Math.max(14, c * 0.42);
    g.replaceChildren();
    if (o.title) { const t = svgEl("text", { x: this.ox, y: f * 1.05, "font-weight": 600 }, g); t.textContent = o.title; css(t, { "font-size": f + "px" }); }
    if (o.indices) {
      for (let k = 0; k < this.cols; k++) { const t = svgEl("text", { x: this.ox + (k + 0.5) * c, y: this.oy - 5, "text-anchor": "middle" }, g); t.textContent = String(k); css(t, { "font-size": f * 0.75 + "px", fill: "var(--qb-muted, #777)" }); }
      for (let k = 0; k < this.rows; k++) { const t = svgEl("text", { x: this.ox - 5, y: this.oy + (k + 0.5) * c + f * 0.27, "text-anchor": "end" }, g); t.textContent = String(k); css(t, { "font-size": f * 0.75 + "px", fill: "var(--qb-muted, #777)" }); }
    }
    const numbers = o.numbers ?? (o.display !== "binary");
    for (let i = 0; i < this.values.length; i++) {
      const r = Math.floor(i / this.cols), q = i % this.cols, v = this.values[i], col = this.colourOf(v);
      const rect = svgEl("rect", { x: this.ox + q * c, y: this.oy + r * c, width: c, height: c }, g);
      css(rect, { fill: toCss(col), stroke: "var(--qb-grid, #bbb)", "stroke-width": 1.2 });
      if (numbers) {
        const s = o.text?.(v, i) ?? String(v);
        if (!s) continue;
        const fs = Math.min(f, c * 1.5 / Math.max(1.5, s.length));
        const t = svgEl("text", { x: this.ox + (q + 0.5) * c, y: this.oy + (r + 0.5) * c + fs * 0.36, "text-anchor": "middle" }, g);
        t.textContent = s;
        css(t, { "font-size": fs.toFixed(1) + "px", fill: textColorOn(col) });
      }
    }
    css(svgEl("rect", { x: this.ox, y: this.oy, width: this.cols * c, height: this.rows * c, fill: "none" }, g), { stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 });
    this.drawOverlay();
  }

  private drawOverlay(): void {
    const g = this.overG, c = this.cell;
    g.replaceChildren();
    const box = (i: number, col: string, w: number, inset: number, fill = false) => {
      const r = Math.floor(i / this.cols), q = i % this.cols;
      const el = svgEl("rect", { x: this.ox + q * c + inset, y: this.oy + r * c + inset, width: c - 2 * inset, height: c - 2 * inset, rx: 3 }, g);
      css(el, fill ? { fill: col, "fill-opacity": 0.3, stroke: "none" } : { fill: "none", stroke: col, "stroke-width": w });
    };
    for (const m of this.marks) {
      if (m.index < 0 || m.index >= this.values.length) continue;
      box(m.index, m.color ?? "var(--qb-c3, #2ca02c)", 3.5, 2.5, !!m.fill);
      if (m.tag) {
        const r = Math.floor(m.index / this.cols), q = m.index % this.cols;
        const t = svgEl("text", { x: this.ox + q * c + 4, y: this.oy + r * c + c * 0.3 }, g);
        t.textContent = m.tag;
        css(t, { "font-size": (c * 0.27).toFixed(1) + "px", fill: m.color ?? "var(--qb-c3, #2ca02c)", "font-weight": 700 });
      }
    }
    const w = this.win;
    if (w) {
      const r0 = Math.floor(w.at / this.cols), c0 = w.at % this.cols, col = w.color ?? "var(--qb-c1, #1f77b4)";
      w.offsets.forEach(([dr, dc], k) => {
        const r = r0 + dr, q = c0 + dc;
        const x = this.ox + q * c, y = this.oy + r * c;
        const out = r < 0 || q < 0 || r >= this.rows || q >= this.cols;
        const el = svgEl("rect", { x: x + 1.5, y: y + 1.5, width: c - 3, height: c - 3, rx: 4 }, g);
        css(el, { fill: col, "fill-opacity": 0.18, stroke: col, "stroke-width": 3, "stroke-dasharray": out ? "4 3" : undefined });
        if (w.hits) {
          const d = svgEl("circle", { cx: x + c - 8, cy: y + 8, r: 5 }, g);
          css(d, { fill: w.hits[k] ? col : "var(--qb-bg, #fff)", stroke: col, "stroke-width": 2 });
        }
      });
      if (w.origin !== false) box(w.at, col, 2, 7);
    }
    if (this.focus !== null && this.focus >= 0) box(this.focus, ACCENT, 4.5, 1.5);
    for (const a of this.arrows) {
      const [x0, y0] = this.centre(a.from), [x1, y1] = this.centre(a.to);
      const L = Math.hypot(x1 - x0, y1 - y0); if (L < 1) continue;
      const ux = (x1 - x0) / L, uy = (y1 - y0) / L, s = c * 0.3;
      const ax = x0 + ux * s, ay = y0 + uy * s, bx = x1 - ux * s, by = y1 - uy * s, hw = 6, hl = 10;
      const col = a.color ?? "var(--qb-c4, #ff7f0e)";
      css(svgEl("line", { x1: ax, y1: ay, x2: bx - ux * hl, y2: by - uy * hl }, g), { stroke: col, "stroke-width": 3 });
      css(svgEl("polygon", { points: `${bx},${by} ${bx - ux * hl - uy * hw},${by - uy * hl + ux * hw} ${bx - ux * hl + uy * hw},${by - uy * hl - ux * hw}` }, g), { fill: col });
    }
  }

  /** Cell index under a pointer event (null outside). Uses getBoundingClientRect, so it
   *  works inside reveal.js's scaled slides. */
  cellAt(ev: { clientX: number; clientY: number }): { index: number; fy: number } | null {
    const rect = this.svg.getBoundingClientRect(), vb = this.svg.viewBox.baseVal;
    const k = Math.min(rect.width / vb.width, rect.height / vb.height);
    const offX = (rect.width - vb.width * k) / 2, offY = (rect.height - vb.height * k) / 2;
    const x = (ev.clientX - rect.left - offX) / k, y = (ev.clientY - rect.top - offY) / k;
    const q = Math.floor((x - this.ox) / this.cell), r = Math.floor((y - this.oy) / this.cell);
    if (q < 0 || r < 0 || q >= this.cols || r >= this.rows) return null;
    return { index: r * this.cols + q, fy: (y - this.oy) / this.cell - r };
  }

  private installEditing(): void {
    const svg = this.svg;
    let paintValue: number | null = null, last = -1;
    const apply = (hit: { index: number; fy: number }, first: boolean) => {
      const rule = this.opts.edit;
      if (!rule) return;
      const i = hit.index, old = this.values[i];
      let nv: number;
      if (!first && paintValue !== null) nv = paintValue;
      else if (rule === "toggle") nv = old ? 0 : 1;
      else if (rule === "plusminus") nv = old + (hit.fy < 0.5 ? 1 : -1);
      else if (typeof rule === "function") nv = rule(old, i);
      else nv = old >= rule.max ? rule.min : old + 1;
      if (typeof rule === "object" && rule) nv = Math.max(rule.min, Math.min(rule.max, nv));
      // drag-painting only makes sense for toggle / functions; plusminus edits one cell per click
      paintValue = rule === "plusminus" ? null : nv;
      if (nv === old) return;
      this.values[i] = nv;
      this.draw();
      this.opts.onEdit?.(this.values, i);
    };
    svg.addEventListener("pointerdown", ev => {
      if (!this.opts.edit) return;
      const hit = this.cellAt(ev);
      if (!hit) return;
      ev.preventDefault();
      svg.setPointerCapture(ev.pointerId);
      last = hit.index;
      apply(hit, true);
    });
    svg.addEventListener("pointermove", ev => {
      if (!svg.hasPointerCapture(ev.pointerId) || paintValue === null) return;
      const hit = this.cellAt(ev);
      if (!hit || hit.index === last) return;
      last = hit.index;
      apply(hit, false);
    });
    const end = () => { paintValue = null; last = -1; };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.style.touchAction = "none";
    if (this.opts.edit) svg.style.cursor = "pointer";
  }
}

// ---------------------------------------------------------------------------------------
// The stepper

/** One snapshot of an algorithm. Everything except `values` is optional. */
export interface GridFrame {
  /** The main grid as it should look in this frame. */
  values: ArrayLike<number>;
  /** The second grid (output, table, distances, labels), if the algorithm has one. */
  aux?: ArrayLike<number>;
  /** Cell being processed (outlined in red) on the main / aux grid. */
  focus?: number;
  auxFocus?: number;
  /** Neighbourhood or structuring element around a cell of the main grid. */
  window?: GridWindow;
  auxWindow?: GridWindow;
  marks?: CellMark[];
  auxMarks?: CellMark[];
  arrows?: CellArrow[];
  auxArrows?: CellArrow[];
  /** Line appended to the log at this frame. */
  log?: string;
  /** Status line shown under the controls while this frame is shown. */
  note?: string;
  /** Name of a stage that starts at this frame (e.g. "row 3", "pass 2"); "next stage" jumps between them. */
  stage?: string;
}

export interface GridAlgorithm {
  id: string;
  label: string;
  /** Generate the frames. The first frame is the start state. `input` is a fresh copy. */
  run(input: Int32Array, rows: number, cols: number): Iterable<GridFrame>;
  /** Display of the aux grid (default: like the main grid, or "int"). */
  auxDisplay?: CellDisplay;
  auxTitle?: string;
  auxText?: (v: number, index: number) => string | undefined;
  auxLo?: number;
  auxHi?: number;
}

export interface GridStepperOptions {
  rows: number;
  cols: number;
  initial: ArrayLike<number>;
  algorithms: GridAlgorithm[];
  /** Id of the algorithm to start with (default the first). */
  algorithm?: string;
  /** Edit rule for the main grid (default "toggle" for binary, "plusminus" otherwise). */
  edit?: EditRule;
  display?: CellDisplay;
  numbers?: boolean;
  text?: (v: number, index: number) => string | undefined;
  lo?: number;
  hi?: number;
  cell?: number;
  title?: string;
  indices?: boolean;
  /** Input presets (buttons). */
  presets?: { label: string; values: ArrayLike<number>; rows?: number; cols?: number }[];
  /** Extra buttons in the control row (e.g. "fill row"), given the stepper. */
  extraButtons?: { label: string; onClick: (s: GridStepper) => void }[];
  /** Upper limit on generated frames (default 5000). */
  maxFrames?: number;
  /** Delay between frames when playing, in ms (default 350; a speed slider changes it). */
  interval?: number;
  /** Show the speed slider (default true). */
  speed?: boolean;
  /** Called whenever a frame is shown. */
  onFrame?: (frame: GridFrame, index: number, s: GridStepper) => void;
  /** Called after the input is edited (before the frames are regenerated). */
  onInput?: (values: Int32Array) => void;
}

export interface GridStepper {
  root: HTMLElement;
  main: GridView;
  aux: GridView | null;
  /** Panel with the controls (append your own sliders/buttons here). */
  panel: HTMLElement;
  log: HTMLElement;
  frames: GridFrame[];
  readonly index: number;
  algorithm: GridAlgorithm;
  goTo(i: number): void;
  step(): void;
  back(): void;
  play(on?: boolean): void;
  toEnd(): void;
  reset(): void;
  /** Jump to the next frame with a `stage` (or the end). */
  nextStage(): void;
  setAlgorithm(id: string): void;
  /** Replace the input (optionally with a new size) and regenerate. */
  setInput(values: ArrayLike<number>, rows?: number, cols?: number): void;
  /** Regenerate the frames from the current input (after changing parameters). */
  rerun(): void;
  input(): Int32Array;
}

/** Build the step-through component into `parent` (adds the layout classes of demo.css). */
export function gridStepper(parent: HTMLElement, o: GridStepperOptions): GridStepper {
  let rows = o.rows, cols = o.cols, input = Int32Array.from(o.initial);
  let algo = o.algorithms.find(a => a.id === o.algorithm) ?? o.algorithms[0];
  let frames: GridFrame[] = [], idx = 0, timer: number | null = null, interval = o.interval ?? 350;

  const root = htmlEl("div", "qb-gridstep", "", parent);
  const grids = htmlEl("div", "qb-gridstep-grids", "", root);
  const mainBox = htmlEl("div", "qb-gridstep-cell", "", grids);
  const auxBox = htmlEl("div", "qb-gridstep-cell", "", grids);
  const editRule: EditRule = o.edit ?? (o.display === "binary" ? "toggle" : "plusminus");
  const main = new GridView(mainBox, rows, cols, input, {
    cell: o.cell, display: o.display ?? "int", numbers: o.numbers, text: o.text, lo: o.lo, hi: o.hi, title: o.title, indices: o.indices,
    edit: editRule, onEdit: v => { input = Int32Array.from(v); o.onInput?.(input); regenerate(); },
  });
  let aux: GridView | null = null;

  const panel = htmlEl("div", "qb-panel qb-gridstep-panel", "", root);
  let algoBtns: HTMLButtonElement[] | undefined;
  if (o.algorithms.length > 1) {
    const row = htmlEl("div", "qb-presets", "", panel);
    algoBtns = o.algorithms.map(a => button(row, a.label, () => api.setAlgorithm(a.id)));
  }
  const ctl = htmlEl("div", "qb-presets qb-gridstep-ctl", "", panel);
  button(ctl, "⏮ reset", () => api.reset());
  button(ctl, "◀ back", () => api.back());
  const stepBtn = button(ctl, "step ▶", () => api.step());
  const playBtn = button(ctl, "play", () => api.play());
  button(ctl, "end ⏭", () => api.toEnd());
  const stageBtn = button(ctl, "next stage", () => api.nextStage());
  for (const b of o.extraButtons ?? []) button(ctl, b.label, () => b.onClick(api));
  if (o.speed !== false) slider(panel, "speed", 1, 10, 1, Math.round(10 - Math.log2(interval / 50) * 2) || 5, v => { interval = 50 * 2 ** ((10 - v) / 2); if (timer !== null) { api.play(false); api.play(true); } }, v => `${v}`);
  const counter = htmlEl("div", "qb-readout qb-gridstep-count", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  const log = htmlEl("div", "qb-readout qb-gridstep-log", "", panel);
  if (o.presets?.length) {
    htmlEl("div", "qb-note", "input", panel);
    const row = htmlEl("div", "qb-presets", "", panel);
    for (const p of o.presets) button(row, p.label, () => api.setInput(p.values, p.rows, p.cols));
  }

  function makeAux(): void {
    const first = frames[0];
    const want = !!first?.aux;
    if (!want) { if (aux) { aux.svg.remove(); aux = null; } auxBox.style.display = "none"; return; }
    auxBox.style.display = "";
    const opts: GridViewOptions = { cell: o.cell, display: algo.auxDisplay ?? (o.display === "binary" ? "binary" : "int"), title: algo.auxTitle, text: algo.auxText, lo: algo.auxLo, hi: algo.auxHi, indices: o.indices, edit: false };
    if (aux) { aux.svg.remove(); }
    aux = new GridView(auxBox, rows, cols, first.aux, opts);
  }

  function regenerate(keepIndex = false): void {
    const old = idx;
    frames = [];
    const max = o.maxFrames ?? 5000;
    for (const f of algo.run(Int32Array.from(input), rows, cols)) {
      frames.push({ ...f, values: Int32Array.from(f.values), aux: f.aux ? Int32Array.from(f.aux) : undefined });
      if (frames.length >= max) { frames.push({ values: frames[frames.length - 1].values, aux: frames[frames.length - 1].aux, log: `… stopped after ${max} steps` }); break; }
    }
    if (!frames.length) frames.push({ values: input });
    makeAux();
    stageBtn.style.display = frames.some(f => f.stage) ? "" : "none";
    algoBtns?.forEach((b, i) => b.classList.toggle("active", o.algorithms[i] === algo));
    api.goTo(keepIndex ? Math.min(old, frames.length - 1) : 0);
  }

  function showFrame(): void {
    const f = frames[idx];
    main.show({ values: f.values, focus: f.focus, window: f.window, marks: f.marks, arrows: f.arrows });
    if (aux && f.aux) aux.show({ values: f.aux, focus: f.auxFocus, window: f.auxWindow, marks: f.auxMarks, arrows: f.auxArrows });
    const lines: string[] = [];
    for (let k = 0; k <= idx; k++) if (frames[k].log) lines.push(frames[k].log!);
    log.replaceChildren();
    const shown = lines.slice(-200);
    shown.forEach((s, k) => htmlEl("div", k === shown.length - 1 && frames[idx].log ? "qb-gridstep-cur" : "", s, log));
    log.scrollTop = log.scrollHeight;
    const stage = [...frames.slice(0, idx + 1)].reverse().find(fr => fr.stage)?.stage;
    counter.textContent = `step ${idx} / ${frames.length - 1}` + (stage ? ` · ${stage}` : "");
    note.textContent = f.note ?? "";
    stepBtn.disabled = idx >= frames.length - 1;
    o.onFrame?.(f, idx, api);
  }

  const api: GridStepper = {
    root, main, get aux() { return aux; }, panel, log,
    get frames() { return frames; },
    get index() { return idx; },
    get algorithm() { return algo; },
    set algorithm(a: GridAlgorithm) { algo = a; },
    goTo(i: number) { idx = Math.max(0, Math.min(frames.length - 1, i)); showFrame(); },
    step() { api.goTo(idx + 1); },
    back() { api.play(false); api.goTo(idx - 1); },
    play(on?: boolean) {
      const want = on ?? timer === null;
      if (timer !== null) { clearInterval(timer); timer = null; }
      if (want) {
        if (idx >= frames.length - 1) api.goTo(0);
        timer = window.setInterval(() => {
          if (!root.isConnected || idx >= frames.length - 1) { api.play(false); return; }
          api.step();
        }, interval);
      }
      playBtn.textContent = timer !== null ? "pause" : "play";
      playBtn.classList.toggle("active", timer !== null);
    },
    toEnd() { api.play(false); api.goTo(frames.length - 1); },
    reset() { api.play(false); api.goTo(0); },
    nextStage() {
      api.play(false);
      let k = idx + 1;
      while (k < frames.length - 1 && !frames[k].stage) k++;
      api.goTo(k);
    },
    setAlgorithm(id: string) { api.play(false); algo = o.algorithms.find(a => a.id === id) ?? algo; regenerate(); },
    setInput(values: ArrayLike<number>, r?: number, c?: number) {
      api.play(false);
      if (r && c && (r !== rows || c !== cols)) { rows = r; cols = c; main.resize(rows, cols, values); }
      input = Int32Array.from(values);
      main.setValues(input);
      o.onInput?.(input);
      regenerate();
    },
    rerun() { regenerate(true); },
    input: () => Int32Array.from(input),
  };
  regenerate();
  return api;
}

// ---------------------------------------------------------------------------------------
// Neighbourhood helpers (offsets as [dRow, dCol])

/** 4-neighbourhood (city block), without the centre. */
export const N4: [number, number][] = [[-1, 0], [0, -1], [0, 1], [1, 0]];
/** 8-neighbourhood (chessboard), without the centre. */
export const N8: [number, number][] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
/** 3x3 cross and 3x3 square structuring elements (with the centre). */
export const CROSS3: [number, number][] = [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]];
export const SQUARE3: [number, number][] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]];

/** Offsets of the set cells of a small k x k grid (e.g. an SE drawn in a 3x3 GridView), centred. */
export function offsetsFromGrid(values: ArrayLike<number>, k: number): [number, number][] {
  const h = Math.floor(k / 2), out: [number, number][] = [];
  for (let i = 0; i < k * k; i++) if (values[i]) out.push([Math.floor(i / k) - h, (i % k) - h]);
  return out;
}

/** Parse a grid from text rows, e.g. ["0110", "1111"] or ["3 1 2", "4 0 1"]. */
export function gridFromText(rows: string[]): { rows: number; cols: number; values: Int32Array } {
  const parsed = rows.map(r => (/\s/.test(r.trim()) ? r.trim().split(/\s+/) : r.trim().split("")).map(Number));
  const cols = Math.max(...parsed.map(r => r.length));
  const values = new Int32Array(rows.length * cols);
  parsed.forEach((r, i) => r.forEach((v, j) => { values[i * cols + j] = v; }));
  return { rows: rows.length, cols, values };
}

/** Viridis for int grids where colour (not grey) reads better, e.g. distance maps. */
export const GRID_VIRIDIS = (t: number) => viridis(0.15 + 0.85 * t);
