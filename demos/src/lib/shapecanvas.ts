// Polygon / shape editing canvas (owner: labelling day, for the `moments` shell; used by the
// moments, measurements and snakes demos). Closed polygons on a pixel grid: draw new shapes
// by clicking vertices, drag vertices, drag a shape to move it, and buttons to rotate,
// scale and mirror the selected shape (pointer only; the keyboard stays with reveal.js).
// The shapes are rasterised to a binary mask (pixel centres, even-odd rule), shown under
// the outlines, so every measurement demo works on the same pixels the student sees.
//
// Coordinates: pixel units of a size x size image, x right, y DOWN (image convention);
// pixel (i, j) = (row, column) covers [j, j+1] x [i, i+1]. Rotation buttons turn
// counter-clockwise as seen on screen.
import { svgEl, htmlEl, button, presets as presetRow, svgPoint } from "./ui";

export type Pt = [number, number];
/** A closed polygon (the last vertex connects to the first). */
export type Poly = Pt[];

// ---------------------------------------------------------------------------------------
// Geometry (no DOM)

/** Rasterise polygons into a w x h mask (1 inside) by testing pixel centres. Even-odd rule
 *  over all polygons together, so a polygon inside another makes a hole; "nonzero" unions. */
export function rasterise(polys: Poly[], w: number, h: number, rule: "evenodd" | "nonzero" = "evenodd"): Uint8Array {
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const py = y + 0.5;
    // crossings of the scanline with every edge: x position and winding direction
    const xs: [number, number][] = [];
    for (const p of polys) for (let i = 0; i < p.length; i++) {
      const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length];
      if ((y0 <= py) !== (y1 <= py)) xs.push([x0 + (py - y0) * (x1 - x0) / (y1 - y0), y1 > y0 ? 1 : -1]);
    }
    xs.sort((a, b) => a[0] - b[0]);
    let wind = 0, inside = false;
    for (let k = 0; k + 1 < xs.length; k++) {
      wind += xs[k][1];
      inside = rule === "evenodd" ? !inside : wind !== 0;
      if (!inside) continue;
      // pixel centres x + 0.5 in [xs[k], xs[k+1])
      const xa = Math.max(0, Math.ceil(xs[k][0] - 0.5)), xb = Math.min(w, Math.ceil(xs[k + 1][0] - 0.5));
      for (let x = xa; x < xb; x++) mask[y * w + x] = 1;
    }
  }
  return mask;
}

/** Signed area (shoelace). Positive when the vertices run counter-clockwise in a y-up frame,
 *  i.e. clockwise as seen on screen with y down. */
export function polygonArea(p: Poly): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; a += x0 * y1 - x1 * y0; }
  return a / 2;
}

/** Centroid of the polygon's area (vertex mean for degenerate polygons). */
export function polygonCentroid(p: Poly): Pt {
  const A = polygonArea(p);
  if (Math.abs(A) < 1e-12) return [p.reduce((s, q) => s + q[0], 0) / p.length, p.reduce((s, q) => s + q[1], 0) / p.length];
  let cx = 0, cy = 0;
  for (let i = 0; i < p.length; i++) {
    const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length], c = x0 * y1 - x1 * y0;
    cx += (x0 + x1) * c; cy += (y0 + y1) * c;
  }
  return [cx / (6 * A), cy / (6 * A)];
}

/** Rotate counter-clockwise AS SEEN ON SCREEN (y down) by `angle` radians about `about`. */
export function rotatePoly(p: Poly, angle: number, about: Pt = polygonCentroid(p)): Poly {
  const c = Math.cos(angle), s = Math.sin(angle);
  return p.map(([x, y]) => { const dx = x - about[0], dy = y - about[1]; return [about[0] + c * dx + s * dy, about[1] - s * dx + c * dy] as Pt; });
}

export function scalePoly(p: Poly, sx: number, sy = sx, about: Pt = polygonCentroid(p)): Poly {
  return p.map(([x, y]) => [about[0] + sx * (x - about[0]), about[1] + sy * (y - about[1])] as Pt);
}

export function translatePoly(p: Poly, dx: number, dy: number): Poly {
  return p.map(([x, y]) => [x + dx, y + dy] as Pt);
}

/** Mirror left-right ("x") or top-bottom ("y") through the centroid. Vertex order is
 *  reversed so the orientation (sign of the area) is kept. */
export function mirrorPoly(p: Poly, axis: "x" | "y", about: Pt = polygonCentroid(p)): Poly {
  const q = p.map(([x, y]) => (axis === "x" ? [2 * about[0] - x, y] : [x, 2 * about[1] - y]) as Pt);
  return q.reverse();
}

/** Regular polygon / ellipse approximation with n vertices. */
export function ellipsePoly(cx: number, cy: number, rx: number, ry: number, n = 48, angle = 0): Poly {
  const c = Math.cos(angle), s = Math.sin(angle);
  return Array.from({ length: n }, (_, i) => {
    const t = 2 * Math.PI * i / n, u = rx * Math.cos(t), v = ry * Math.sin(t);
    return [cx + c * u + s * v, cy - s * u + c * v] as Pt;
  });
}

/** Star with `k` points, outer radius r1 and inner radius r2. */
export function starPoly(cx: number, cy: number, r1: number, r2: number, k = 5): Poly {
  return Array.from({ length: 2 * k }, (_, i) => {
    const t = Math.PI * i / k - Math.PI / 2, r = i % 2 ? r2 : r1;
    return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as Pt;
  });
}

export interface ShapePreset { id: string; label: string; make: (size: number) => Poly[] }

/** Shape presets, scaled to a size x size grid. */
export const SHAPE_PRESETS: ShapePreset[] = [
  { id: "square", label: "square", make: s => [[[0.3 * s, 0.3 * s], [0.7 * s, 0.3 * s], [0.7 * s, 0.7 * s], [0.3 * s, 0.7 * s]]] },
  { id: "rect", label: "rectangle", make: s => [[[0.2 * s, 0.38 * s], [0.8 * s, 0.38 * s], [0.8 * s, 0.62 * s], [0.2 * s, 0.62 * s]]] },
  { id: "ellipse", label: "ellipse", make: s => [ellipsePoly(0.5 * s, 0.5 * s, 0.32 * s, 0.17 * s, 48, 0.5)] },
  { id: "triangle", label: "triangle", make: s => [[[0.5 * s, 0.18 * s], [0.82 * s, 0.78 * s], [0.18 * s, 0.78 * s]]] },
  { id: "L", label: "L shape", make: s => [[[0.25 * s, 0.2 * s], [0.45 * s, 0.2 * s], [0.45 * s, 0.6 * s], [0.75 * s, 0.6 * s], [0.75 * s, 0.8 * s], [0.25 * s, 0.8 * s]]] },
  { id: "star", label: "star", make: s => [starPoly(0.5 * s, 0.52 * s, 0.36 * s, 0.15 * s, 5)] },
  { id: "ring", label: "ring (hole)", make: s => [ellipsePoly(0.5 * s, 0.5 * s, 0.34 * s, 0.34 * s, 48), ellipsePoly(0.5 * s, 0.5 * s, 0.17 * s, 0.17 * s, 32)] },
  { id: "two", label: "two shapes", make: s => [ellipsePoly(0.3 * s, 0.35 * s, 0.15 * s, 0.15 * s, 32), [[0.55 * s, 0.55 * s], [0.85 * s, 0.6 * s], [0.75 * s, 0.85 * s], [0.5 * s, 0.8 * s]]] },
];

// ---------------------------------------------------------------------------------------
// The editor (DOM)

export interface ShapeCanvasOptions {
  /** Pixel grid size (default 64). */
  size?: number;
  /** Initial shapes (default: the first preset). */
  shapes?: Poly[];
  /** Show the rasterised mask under the outlines (default true). */
  showMask?: boolean;
  /** Fill rule for the mask (default even-odd: nested shapes make holes). */
  rule?: "evenodd" | "nonzero";
  /** Build the buttons (default true). */
  controls?: boolean;
  controlsParent?: HTMLElement;
  /** Preset buttons: true = all of SHAPE_PRESETS, or a list of ids. */
  presets?: boolean | string[];
  /** viewBox size of the square (default 480). */
  view?: number;
  /** Called after every change (during drags at most once per frame; final at the end). */
  onChange?: (polys: Poly[], mask: Uint8Array, final: boolean) => void;
  ariaLabel?: string;
}

export interface ShapeCanvas {
  svg: SVGSVGElement;
  /** Group above the shapes for demo drawings (in pixel units: the group is scaled). */
  overlay: SVGGElement;
  size: number;
  getPolys(): Poly[];
  setPolys(p: Poly[]): void;
  mask(): Uint8Array;
  /** Index of the selected shape (target of the buttons), or -1. */
  selected: number;
  /** Apply a transformation to the selected shape (all shapes if none is selected). */
  transform(f: (p: Poly) => Poly): void;
  setShowMask(on: boolean): void;
}

export function shapeCanvas(parent: HTMLElement, o: ShapeCanvasOptions = {}): ShapeCanvas {
  const N = o.size ?? 64, V = o.view ?? 480, k = V / N, rule = o.rule ?? "evenodd";
  let polys: Poly[] = (o.shapes ?? SHAPE_PRESETS[0].make(N)).map(p => p.map(q => [...q] as Pt));
  const initial = polys.map(p => p.map(q => [...q] as Pt));
  let selected = polys.length ? 0 : -1, showMask = o.showMask !== false;
  let drawing: Poly | null = null;
  let mask = rasterise(polys, N, N, rule);

  const wrap = htmlEl("div", "qb-plot qb-shapecanvas", "", parent);
  const svg = svgEl("svg", { viewBox: `0 0 ${V} ${V}`, width: "100%", role: "img", "aria-label": o.ariaLabel ?? "editable shapes on a pixel grid" }, wrap);
  svg.style.touchAction = "none";
  const bg = svgEl("rect", { x: 0, y: 0, width: V, height: V }, svg);
  bg.style.setProperty("fill", "var(--qb-bg, #fff)");
  const maskImg = svgEl("image", { x: 0, y: 0, width: V, height: V, preserveAspectRatio: "none" }, svg);
  maskImg.style.setProperty("image-rendering", "pixelated");
  const gridG = svgEl("g", {}, svg);
  for (let i = 0; i <= N; i += N >= 32 ? 8 : 1) {
    for (const l of [svgEl("line", { x1: i * k, x2: i * k, y1: 0, y2: V }, gridG), svgEl("line", { y1: i * k, y2: i * k, x1: 0, x2: V }, gridG)]) {
      l.style.setProperty("stroke", "var(--qb-grid, #e3e3e3)");
    }
  }
  const shapesG = svgEl("g", {}, svg);
  const overlay = svgEl("g", { transform: `scale(${k})` }, svg);
  const handlesG = svgEl("g", {}, svg);
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = N; maskCanvas.height = N;

  const toPx = (ev: { clientX: number; clientY: number }): Pt => { const [x, y] = svgPoint(svg, ev); return [x / k, y / k]; };

  function drawMask(): void {
    if (!showMask) { maskImg.removeAttribute("href"); return; }
    const ctx = maskCanvas.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) if (mask[p]) { id.data[4 * p] = 120; id.data[4 * p + 1] = 150; id.data[4 * p + 2] = 190; id.data[4 * p + 3] = 150; }
    ctx.putImageData(id, 0, 0);
    maskImg.setAttribute("href", maskCanvas.toDataURL());
  }

  function draw(): void {
    shapesG.replaceChildren(); handlesG.replaceChildren();
    polys.forEach((p, i) => {
      const path = svgEl("path", { d: "M" + p.map(([x, y]) => `${(x * k).toFixed(1)},${(y * k).toFixed(1)}`).join("L") + "Z", "data-i": i }, shapesG);
      const sel = i === selected;
      path.style.setProperty("fill", sel ? "var(--qb-c1, #1f77b4)" : "var(--qb-muted, #777)");
      path.style.setProperty("fill-opacity", "0.12");
      path.style.setProperty("stroke", sel ? "var(--qb-c1, #1f77b4)" : "var(--qb-muted, #777)");
      path.style.setProperty("stroke-width", sel ? "3" : "2");
      path.style.cursor = "move";
    });
    const hp = selected >= 0 ? polys[selected] : null;
    hp?.forEach(([x, y], j) => {
      const c = svgEl("circle", { cx: x * k, cy: y * k, r: 7, class: "handle", "data-v": j }, handlesG);
      c.style.setProperty("stroke", "var(--qb-c1, #1f77b4)");
    });
    if (drawing) {
      const d = svgEl("path", { d: "M" + drawing.map(([x, y]) => `${x * k},${y * k}`).join("L"), fill: "none" }, handlesG);
      d.style.setProperty("stroke", "var(--qb-c2, #d62728)");
      d.style.setProperty("stroke-width", "3");
      drawing.forEach(([x, y], j) => {
        const c = svgEl("circle", { cx: x * k, cy: y * k, r: j === 0 ? 10 : 5 }, handlesG);
        c.style.setProperty("fill", j === 0 ? "var(--qb-bg, #fff)" : "var(--qb-c2, #d62728)");
        c.style.setProperty("stroke", "var(--qb-c2, #d62728)");
        c.style.setProperty("stroke-width", "3");
      });
    }
  }

  let pending = false;
  function changed(final: boolean): void {
    mask = rasterise(polys, N, N, rule);
    drawMask(); draw();
    if (final) { o.onChange?.(polys, mask, true); return; }
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; o.onChange?.(polys, mask, false); });
  }

  // ---- pointer: drag vertex / drag shape / click to draw ----
  type Drag = { kind: "vertex"; i: number; j: number } | { kind: "shape"; i: number; last: Pt } | null;
  let drag: Drag = null;
  const inside = (p: Poly, [x, y]: Pt) => {
    let c = false;
    for (let a = 0, b = p.length - 1; a < p.length; b = a++) {
      const [xa, ya] = p[a], [xb, yb] = p[b];
      if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) c = !c;
    }
    return c;
  };
  svg.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const q = toPx(ev), rr = 12 / k;
    if (drawing) {
      if (drawing.length >= 3 && Math.hypot(q[0] - drawing[0][0], q[1] - drawing[0][1]) < 1.6 * rr) finishDrawing();
      else { drawing.push(q); draw(); }
      return;
    }
    svg.setPointerCapture(ev.pointerId);
    if (selected >= 0) {
      const j = polys[selected].findIndex(([x, y]) => Math.hypot(x - q[0], y - q[1]) < rr);
      if (j >= 0) { drag = { kind: "vertex", i: selected, j }; return; }
    }
    for (let i = polys.length - 1; i >= 0; i--) {
      if (inside(polys[i], q)) { selected = i; drag = { kind: "shape", i, last: q }; draw(); return; }
    }
    selected = -1; draw();
  });
  svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const q = toPx(ev);
    const cl = (v: number) => Math.max(0, Math.min(N, v));
    if (drag.kind === "vertex") polys[drag.i][drag.j] = [cl(q[0]), cl(q[1])];
    else { polys[drag.i] = translatePoly(polys[drag.i], q[0] - drag.last[0], q[1] - drag.last[1]); drag.last = q; }
    changed(false);
  });
  const up = () => { if (drag) { drag = null; changed(true); } };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);

  function finishDrawing(): void {
    if (drawing && drawing.length >= 3) { polys.push(drawing); selected = polys.length - 1; }
    drawing = null;
    syncDraw();
    changed(true);
  }

  // ---- controls ----
  let drawBtn: HTMLButtonElement | null = null;
  function syncDraw(): void { if (drawBtn) { drawBtn.textContent = drawing ? "close shape" : "draw new shape"; drawBtn.classList.toggle("active", !!drawing); } }

  const api: ShapeCanvas = {
    svg, overlay, size: N,
    getPolys: () => polys.map(p => p.map(q => [...q] as Pt)),
    setPolys(p) { polys = p.map(x => x.map(q => [...q] as Pt)); selected = polys.length ? 0 : -1; drawing = null; syncDraw(); changed(true); },
    mask: () => mask,
    get selected() { return selected; },
    set selected(i: number) { selected = i; draw(); },
    transform(f) {
      if (selected >= 0) polys[selected] = f(polys[selected]);
      else polys = polys.map(f);
      changed(true);
    },
    setShowMask(on) { showMask = on; drawMask(); },
  };

  if (o.controls !== false) {
    const panel = o.controlsParent ?? htmlEl("div", "qb-panel", "", parent);
    const r1 = htmlEl("div", "qb-presets", "", panel);
    drawBtn = button(r1, "draw new shape", () => { if (drawing) finishDrawing(); else { drawing = []; selected = -1; syncDraw(); draw(); } });
    button(r1, "delete shape", () => { if (selected >= 0) { polys.splice(selected, 1); selected = polys.length ? 0 : -1; changed(true); } });
    const r2 = htmlEl("div", "qb-presets", "", panel);
    button(r2, "⟲ 15°", () => api.transform(p => rotatePoly(p, Math.PI / 12)));
    button(r2, "⟳ 15°", () => api.transform(p => rotatePoly(p, -Math.PI / 12)));
    button(r2, "× 1.1", () => api.transform(p => scalePoly(p, 1.1)));
    button(r2, "× 0.9", () => api.transform(p => scalePoly(p, 1 / 1.1)));
    button(r2, "mirror ↔", () => api.transform(p => mirrorPoly(p, "x")));
    button(r2, "mirror ↕", () => api.transform(p => mirrorPoly(p, "y")));
    const r3 = htmlEl("div", "qb-presets", "", panel);
    button(r3, "←", () => api.transform(p => translatePoly(p, -2, 0)));
    button(r3, "→", () => api.transform(p => translatePoly(p, 2, 0)));
    button(r3, "↑", () => api.transform(p => translatePoly(p, 0, -2)));
    button(r3, "↓", () => api.transform(p => translatePoly(p, 0, 2)));
    button(r3, "reset", () => api.setPolys(initial));
    if (o.presets) {
      const ids = o.presets === true ? SHAPE_PRESETS.map(s => s.id) : o.presets;
      presetRow(panel, SHAPE_PRESETS.filter(s => ids.includes(s.id)).map(s => ({ label: s.label, apply: () => api.setPolys(s.make(N)) })));
    }
    htmlEl("div", "qb-note", "Drag a corner or a whole shape. Draw: click the corners, then click the first corner (or the button) to close.", panel);
  }

  drawMask(); draw();
  return api;
}
