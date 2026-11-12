// World coordinates, axes and contour drawing on top of the shared point playground
// (lib/playground.ts), for the SVM day's 2D demos (linclass, svm2d, svmlift).
// The playground works in the unit square; here a square world range [lo, hi]² is mapped onto
// it, the playground's quarter grid is replaced by a grid with labelled ticks in world units,
// and decision values can be drawn as filled sign regions plus contour lines (marching squares).
import { svgEl } from "../lib/ui";
import { Playground } from "../lib/playground";
import { marchingSquares } from "../lib/marching";
import { hexToRgb } from "../lib/colour";
import type { LPoint } from "../lib/datasets2d";

export interface World {
  lo: number;
  hi: number;
  /** World -> unit square. */
  toU(v: number): number;
  /** Unit square -> world. */
  fromU(u: number): number;
  /** World -> SVG x / y. */
  X(v: number): number;
  Y(v: number): number;
  /** Points of the playground in world coordinates. */
  points(): LPoint[];
  /** Replace the points (world coordinates). */
  set(points: LPoint[]): void;
  /** Change the range and redraw the axes. */
  setRange(lo: number, hi: number, xlabel?: string, ylabel?: string): void;
}

/** Nice tick step for a range: about 4–8 ticks. */
export function tickStep(span: number): number {
  const raw = span / 5, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export function worldOf(pg: Playground, lo: number, hi: number, xlabel = "x₁", ylabel = "x₂"): World {
  // hide the playground's own quarter grid (the first group of its svg)
  const g0 = pg.svg.querySelector(":scope > g");
  if (g0 && g0 !== pg.overlay) (g0 as SVGGElement).style.display = "none";
  const axes = svgEl("g", {}, pg.svg);
  pg.svg.insertBefore(axes, pg.overlay);
  const labels = svgEl("g", { class: "qb-halo" }, pg.top);
  const w: World = {
    lo, hi,
    toU: v => (v - w.lo) / (w.hi - w.lo),
    fromU: u => w.lo + u * (w.hi - w.lo),
    X: v => pg.sx(w.toU(v)),
    Y: v => pg.sy(w.toU(v)),
    points: () => pg.points.map(p => ({ x: w.fromU(p.x), y: w.fromU(p.y), c: p.c })),
    set: pts => pg.setPoints(pts.map(p => ({ x: w.toU(p.x), y: w.toU(p.y), c: p.c }))),
    setRange(l, h, xl, yl) { w.lo = l; w.hi = h; draw(xl ?? xlabel, yl ?? ylabel); },
  };
  function draw(xl: string, yl: string): void {
    xlabel = xl; ylabel = yl;
    axes.replaceChildren(); labels.replaceChildren();
    const step = tickStep(w.hi - w.lo), fs = 17;
    const dec = step >= 1 ? 0 : Math.ceil(-Math.log10(step) - 1e-9);
    for (let v = Math.ceil(w.lo / step - 1e-9) * step; v <= w.hi + 1e-9; v += step) {
      const strong = Math.abs(v) < 1e-9;
      for (const l of [svgEl("line", { x1: w.X(v), x2: w.X(v), y1: w.Y(w.lo), y2: w.Y(w.hi) }, axes), svgEl("line", { y1: w.Y(v), y2: w.Y(v), x1: w.X(w.lo), x2: w.X(w.hi) }, axes)]) {
        l.style.setProperty("stroke", strong ? "var(--qb-axis, #444)" : "var(--qb-grid, #e3e3e3)");
        l.style.setProperty("stroke-width", strong ? "1.2" : "1");
        if (strong) l.style.setProperty("opacity", "0.5");
      }
      if (v > w.lo + 1e-9 && v < w.hi - 1e-9) {
        const t1 = svgEl("text", { x: w.X(v), y: w.Y(w.lo) - 5, "text-anchor": "middle" }, labels);
        t1.textContent = v.toFixed(dec); t1.style.fontSize = fs + "px"; t1.style.setProperty("fill", "var(--qb-muted, #777)");
        const t2 = svgEl("text", { x: w.X(w.lo) + 5, y: w.Y(v) + fs * 0.35 }, labels);
        t2.textContent = v.toFixed(dec); t2.style.fontSize = fs + "px"; t2.style.setProperty("fill", "var(--qb-muted, #777)");
      }
    }
    const a = svgEl("text", { x: w.X(w.hi) - 6, y: w.Y(w.lo) - 26, "text-anchor": "end" }, labels);
    a.textContent = xlabel + " →"; a.style.fontSize = "19px";
    const b = svgEl("text", { x: w.X(w.lo) + 6, y: w.Y(w.hi) + 22 }, labels);
    b.textContent = "↑ " + ylabel; b.style.fontSize = "19px";
  }
  draw(xlabel, ylabel);
  return w;
}

/** Sample f on a res × res grid of world points (row 0 at the TOP, as images). */
export function sampleGrid(w: World, res: number, f: (x: number, y: number) => number): Float64Array {
  const v = new Float64Array(res * res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) v[j * res + i] = f(w.fromU(i / (res - 1)), w.fromU(1 - j / (res - 1)));
  return v;
}

/** SVG path of the level-c contour of a grid from sampleGrid. */
export function contourPath(w: World, pg: Playground, grid: ArrayLike<number>, res: number, c: number): string {
  const C = marchingSquares(grid, res, res, c);
  let d = "";
  for (const L of C.lines) {
    L.ids.forEach((id, q) => {
      const gx = C.points[2 * id], gy = C.points[2 * id + 1];
      d += (q ? "L" : "M") + pg.sx(gx / (res - 1)).toFixed(1) + "," + pg.sy(1 - gy / (res - 1)).toFixed(1);
    });
    if (L.closed) d += "Z";
  }
  return d;
}

/** Draw a path into a group with a stroke style. */
export function pathEl(g: SVGGElement, d: string, color: string, width: number, dash?: string, opacity = 1): SVGPathElement {
  const p = svgEl("path", { d, fill: "none" }, g);
  p.style.setProperty("stroke", color);
  p.style.setProperty("stroke-width", String(width));
  if (dash) p.style.setProperty("stroke-dasharray", dash);
  p.style.setProperty("opacity", String(opacity));
  p.style.setProperty("stroke-linejoin", "round");
  return p;
}

/**
 * Background colouring by a decision value f (world coordinates): the side's class colour
 * (class 1 where f > 0, class 0 where f < 0), stronger further from 0; the street |f| < 1
 * optionally lighter (grey).
 */
export function signColour(w: World, colors: string[], f: (x: number, y: number) => number, o: { street?: boolean; scale?: number } = {}) {
  const c0 = hexToRgb(colors[0]).map(v => 255 * v), c1 = hexToRgb(colors[1]).map(v => 255 * v), sc = o.scale ?? 1;
  return (u: number, v: number): [number, number, number, number] => {
    const val = f(w.fromU(u), w.fromU(v));
    if (o.street && Math.abs(val) < 1) return [200, 200, 200, 0.28];
    const c = val > 0 ? c1 : c0, a = 0.12 + 0.28 * Math.tanh(Math.abs(val) / (2 * sc));
    return [c[0], c[1], c[2], Number.isFinite(a) ? a : 0.4];
  };
}
