// 2D point playground (owner: random-forest day): a square of labelled points that the
// student edits (add points of the chosen class, drag them, remove them), with a hook that
// paints a decision / probability map in the background. Used by treeplay, forestplay,
// svm2d, linclass, kmeans2d, em2d, nnplayground and the ncut point mode.
//
//   const pg = pointPlayground(host, { data: dataset("moons"), presets: true,
//     onChange: pts => { model = train(pts); pg.setBackground((x, y) => model.probs(x, y)); } });
//
// Data coordinates: the unit square, x right, y up. Draw extra geometry (boundaries,
// margins, centres) into `pg.overlay` using `pg.sx` / `pg.sy` to map to SVG units.
import { svgEl, htmlEl, button, presets as presetRow, svgPoint, scale } from "./ui";
import { hexToRgb, QUALITATIVE } from "./colour";
import type { Dataset2D, LPoint } from "./datasets2d";
import { DATASETS } from "./datasets2d";

/** Class probabilities at (x, y) (length k, summing to about 1). */
export type ProbabilityFn = (x: number, y: number) => ArrayLike<number>;
/** Any colour at (x, y): [r, g, b, a] with r, g, b in 0..255 and a in 0..1. */
export type ColourFn = (x: number, y: number) => [number, number, number, number];

export interface PlaygroundOptions {
  data?: Dataset2D;
  /** Number of classes offered for adding points (default data.k or 2). */
  k?: number;
  /** viewBox size of the square (default 480). */
  size?: number;
  /** Class colours as hex (default the qualitative palette). */
  colors?: string[];
  /** Class names for the buttons (default "class 1", ...). */
  names?: string[];
  /** Dataset preset buttons: true = all of DATASETS, or a list of ids. */
  presets?: boolean | string[];
  /** Seed for presets (default 1). */
  seed?: number;
  /** Allow adding/moving/removing points (default true). */
  editable?: boolean;
  /** Build the class/tool/clear buttons (default true). */
  controls?: boolean;
  controlsParent?: HTMLElement;
  /** Grid resolution of the background map (default 64). */
  resolution?: number;
  /** Point radius in SVG units (default 8). */
  r?: number;
  /** Called after every change of the points (during a drag at most once per frame). */
  onChange?: (points: LPoint[], final: boolean) => void;
  ariaLabel?: string;
}

export interface Playground {
  svg: SVGSVGElement;
  /** Group above the background and below the points for demo drawings. */
  overlay: SVGGElement;
  /** Group above the points (labels, handles). */
  top: SVGGElement;
  /** Data (unit square) -> SVG units. */
  sx: ReturnType<typeof scale>;
  sy: ReturnType<typeof scale>;
  points: LPoint[];
  k: number;
  /** Class colour (hex). */
  color(c: number): string;
  setPoints(p: LPoint[] | Dataset2D): void;
  /** Probability map: each grid cell is the probability-weighted mix of the class colours. */
  setBackground(fn: ProbabilityFn | null): void;
  /** Free colouring of the background. */
  setBackgroundColour(fn: ColourFn | null): void;
  /** Pointer -> data coordinates. */
  toData(ev: { clientX: number; clientY: number }): [number, number];
  /** Class used for new points. */
  activeClass: number;
  /** "add" (click empty space adds, drag moves) or "remove" (click removes). */
  tool: "add" | "remove";
  redraw(): void;
}

export function pointPlayground(parent: HTMLElement, o: PlaygroundOptions = {}): Playground {
  const S = o.size ?? 480, M = 10, R = o.r ?? 8, res = o.resolution ?? 64;
  const cols = o.colors ?? QUALITATIVE;
  const sx = scale(0, 1, M, S - M), sy = scale(0, 1, S - M, M);
  const wrap = htmlEl("div", "qb-plot qb-playground", "", parent);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%", role: "img", "aria-label": o.ariaLabel ?? "2D points of several classes" }, wrap);
  const frame = svgEl("rect", { x: M, y: M, width: S - 2 * M, height: S - 2 * M }, svg);
  frame.style.setProperty("fill", "var(--qb-bg, #fff)");
  const bgImg = svgEl("image", { x: M, y: M, width: S - 2 * M, height: S - 2 * M, preserveAspectRatio: "none" }, svg);
  const gridG = svgEl("g", {}, svg);
  for (const v of [0.25, 0.5, 0.75]) {
    for (const l of [svgEl("line", { x1: sx(v), x2: sx(v), y1: sy(0), y2: sy(1) }, gridG), svgEl("line", { y1: sy(v), y2: sy(v), x1: sx(0), x2: sx(1) }, gridG)]) {
      l.style.setProperty("stroke", "var(--qb-grid, #e3e3e3)");
    }
  }
  const overlay = svgEl("g", {}, svg);
  const ptsG = svgEl("g", {}, svg);
  const top = svgEl("g", {}, svg);
  const border = svgEl("rect", { x: M, y: M, width: S - 2 * M, height: S - 2 * M, fill: "none" }, svg);
  border.style.setProperty("stroke", "var(--qb-axis, #444)");
  border.style.setProperty("stroke-width", "1.5");

  let points: LPoint[] = (o.data?.points ?? []).map(p => ({ ...p }));
  const initial = points.map(p => ({ ...p }));
  let k = o.k ?? o.data?.k ?? 2;
  let bgFn: ColourFn | null = null;
  const color = (c: number) => cols[c % cols.length];

  const pg: Playground = {
    svg, overlay, top, sx, sy,
    get points() { return points; },
    set points(p: LPoint[]) { points = p; },
    get k() { return k; },
    color,
    setPoints(p) {
      const d = Array.isArray(p) ? { points: p, k } : p;
      points = d.points.map(q => ({ ...q }));
      k = Math.max(k, d.k);
      syncButtons();
      draw(); changed(true);
    },
    setBackground(fn) {
      if (!fn) { pg.setBackgroundColour(null); return; }
      const rgb = Array.from({ length: 16 }, (_, c) => hexToRgb(color(c)).map(v => 255 * v));
      pg.setBackgroundColour((x, y) => {
        const p = fn(x, y);
        let r = 0, g = 0, b = 0, s = 0, mx = 0;
        for (let c = 0; c < p.length; c++) { const w = p[c]; r += w * rgb[c][0]; g += w * rgb[c][1]; b += w * rgb[c][2]; s += w; if (w > mx) mx = w; }
        if (s > 0) { r /= s; g /= s; b /= s; }
        // light tint, stronger where the model is confident
        return [r, g, b, 0.18 + 0.32 * Math.max(0, (mx - 1 / p.length) / (1 - 1 / p.length || 1))];
      });
    },
    setBackgroundColour(fn) { bgFn = fn; drawBackground(); },
    toData(ev) { const [x, y] = svgPoint(svg, ev); return [sx.invert(x), sy.invert(y)]; },
    activeClass: 0,
    tool: "add",
    redraw: () => { draw(); drawBackground(); },
  };

  // ---- drawing ----
  function draw(): void {
    ptsG.replaceChildren();
    for (const p of points) {
      const c = svgEl("circle", { cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r: R }, ptsG);
      c.style.setProperty("fill", color(p.c));
      c.style.setProperty("stroke", "var(--qb-bg, #fff)");
      c.style.setProperty("stroke-width", "1.5");
    }
  }

  const bgCanvas = document.createElement("canvas");
  bgCanvas.width = res; bgCanvas.height = res;
  function drawBackground(): void {
    if (!bgFn) { bgImg.removeAttribute("href"); return; }
    const ctx = bgCanvas.getContext("2d")!, id = ctx.createImageData(res, res);
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      const [r, g, b, a] = bgFn((i + 0.5) / res, 1 - (j + 0.5) / res), q = 4 * (j * res + i);
      id.data[q] = r; id.data[q + 1] = g; id.data[q + 2] = b; id.data[q + 3] = 255 * a;
    }
    ctx.putImageData(id, 0, 0);
    bgImg.setAttribute("href", bgCanvas.toDataURL());
  }

  let pending = false;
  function changed(final: boolean): void {
    if (final) { o.onChange?.(points, true); return; }
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; o.onChange?.(points, false); });
  }

  // ---- pointer ----
  const hit = (x: number, y: number): number => {
    let best = -1, bd = (R * 1.8) ** 2;
    points.forEach((p, i) => { const d = (sx(p.x) - x) ** 2 + (sy(p.y) - y) ** 2; if (d <= bd) { bd = d; best = i; } });
    return best;
  };
  let dragging = -1;
  svg.style.touchAction = "none";
  svg.addEventListener("pointerdown", ev => {
    if (o.editable === false) return;
    ev.preventDefault();
    const [x, y] = svgPoint(svg, ev), i = hit(x, y);
    if (pg.tool === "remove") {
      if (i >= 0) { points.splice(i, 1); draw(); changed(true); }
      return;
    }
    svg.setPointerCapture(ev.pointerId);
    if (i >= 0) dragging = i;
    else {
      const [dx, dy] = [sx.invert(x), sy.invert(y)];
      if (dx < 0 || dx > 1 || dy < 0 || dy > 1) return;
      points.push({ x: dx, y: dy, c: pg.activeClass });
      dragging = points.length - 1;
      draw(); changed(false);
    }
  });
  svg.addEventListener("pointermove", ev => {
    if (dragging < 0) return;
    const [x, y] = pg.toData(ev);
    points[dragging].x = Math.min(1, Math.max(0, x));
    points[dragging].y = Math.min(1, Math.max(0, y));
    draw(); changed(false);
  });
  const up = () => { if (dragging >= 0) { dragging = -1; changed(true); } };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);

  // ---- controls ----
  let classBtns: HTMLButtonElement[] = [], toolBtns: HTMLButtonElement[] = [];
  let classRow: HTMLElement | null = null;
  function syncButtons(): void {
    if (!classRow) return;
    if (classBtns.length !== k) {
      classRow.replaceChildren();
      classBtns = Array.from({ length: k }, (_, c) => {
        const b = button(classRow!, o.names?.[c] ?? `class ${c + 1}`, () => { pg.activeClass = c; pg.tool = "add"; syncButtons(); }, "qb-paint-label");
        b.style.setProperty("--qb-label", color(c));
        return b;
      });
    }
    classBtns.forEach((b, c) => b.classList.toggle("active", pg.tool === "add" && c === pg.activeClass));
    toolBtns.forEach((b, i) => b.classList.toggle("active", i === 0 && pg.tool === "remove"));
  }
  if (o.controls !== false && o.editable !== false) {
    const panel = o.controlsParent ?? htmlEl("div", "qb-panel", "", parent);
    classRow = htmlEl("div", "qb-presets", "", panel);
    const row = htmlEl("div", "qb-presets", "", panel);
    toolBtns = [
      button(row, "remove points", () => { pg.tool = pg.tool === "remove" ? "add" : "remove"; syncButtons(); }),
      button(row, "clear", () => { points = []; draw(); changed(true); }),
      button(row, "reset", () => { points = initial.map(p => ({ ...p })); draw(); changed(true); }),
    ];
    if (o.presets) {
      const ids = o.presets === true ? DATASETS.map(d => d.id) : o.presets;
      const items = DATASETS.filter(d => ids.includes(d.id));
      presetRow(panel, items.map(d => ({ label: d.label, apply: () => pg.setPoints(d.make({ seed: o.seed })) })));
    }
    htmlEl("div", "qb-note", "Click to add a point of the chosen class; drag points to move them.", panel);
    syncButtons();
  }

  draw();
  return pg;
}
