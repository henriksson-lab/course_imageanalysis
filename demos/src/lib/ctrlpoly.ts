// A control-polygon editor on top of a Plot (owner: the splines day; DOM, SVG).
// Large draggable handles (pointer only; the keyboard stays with reveal.js), optional
// labels P0, P1, …, open or closed polygon, selection, add/delete, per-point weight shown
// as the handle size, fixed (non-draggable) points, and a "constrained" style for handles
// that follow a rule (e.g. the C¹ handle of a joined Bézier piece).
//   const cp = new CtrlPoly(plot, { points, onChange: (pts, i) => redraw() });
import { svgEl } from "./ui";
import type { Plot } from "./plot";
import type { Pt } from "./spline";

export interface CtrlPolyOptions {
  points: Pt[];
  closed?: boolean;
  /** handle radius in SVG units (default 11) */
  r?: number;
  color?: string;
  /** label prefix ("P" → P0, P1, …); empty: no labels */
  label?: string;
  /** draw the polygon legs (default true) */
  legs?: boolean;
  weights?: number[];
  /** points that cannot be dragged (drawn hollow) */
  fixed?: (i: number) => boolean;
  /** points that are set by a rule (drawn dashed) */
  constrained?: (i: number) => boolean;
  /** clamp dragging to this data box [x0, x1, y0, y1] */
  box?: [number, number, number, number];
  onChange?: (pts: Pt[], i: number) => void;
  onSelect?: (i: number) => void;
  onRelease?: (i: number) => void;
}

export class CtrlPoly {
  pts: Pt[];
  selected = -1;
  dragging = -1;
  readonly g: SVGGElement;
  private o: CtrlPolyOptions;
  private plot: Plot;
  private els: { c: SVGCircleElement; t?: SVGTextElement; rr: number }[] = [];
  private leg: SVGPathElement | null = null;

  constructor(plot: Plot, o: CtrlPolyOptions) {
    this.plot = plot; this.o = o;
    this.pts = o.points.map(p => [p[0], p[1]] as Pt);
    this.g = svgEl("g", { class: "qb-ctrlpoly" }, plot.over);
    this.draw();
  }

  set(points: Pt[], weights?: number[]): void {
    this.pts = points.map(p => [p[0], p[1]] as Pt);
    if (weights) this.o.weights = weights.slice();
    if (this.selected >= this.pts.length) this.selected = -1;
    this.draw();
  }
  /** Current weights (kept in step with add/remove). */
  get weights(): number[] | undefined { return this.o.weights; }
  setWeights(w: number[] | undefined): void { this.o.weights = w?.slice(); this.draw(); }
  setClosed(c: boolean): void { this.o.closed = c; this.draw(); }
  setOptions(o: Partial<CtrlPolyOptions>): void { Object.assign(this.o, o); this.draw(); }
  select(i: number): void { this.selected = i; this.draw(); this.o.onSelect?.(i); }

  /** Insert a point after the selected one (or at the end), halfway to the next. */
  add(): number {
    const n = this.pts.length, i = this.selected >= 0 ? this.selected : n - 1;
    const a = this.pts[i], b = this.pts[(i + 1) % n];
    const p: Pt = i + 1 < n || this.o.closed ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] : [2 * a[0] - this.pts[i - 1][0], 2 * a[1] - this.pts[i - 1][1]];
    this.pts.splice(i + 1, 0, p);
    if (this.o.weights) this.o.weights.splice(i + 1, 0, 1);
    this.selected = i + 1;
    this.draw();
    this.o.onChange?.(this.pts, i + 1);
    return i + 1;
  }

  remove(i = this.selected): void {
    if (i < 0 || this.pts.length <= 2) return;
    this.pts.splice(i, 1);
    if (this.o.weights) this.o.weights.splice(i, 1);
    this.selected = Math.min(i, this.pts.length - 1);
    this.draw();
    this.o.onChange?.(this.pts, -1);
  }

  draw(): void {
    const { plot, o } = this, g = this.g, X = plot.x, Y = plot.y, col = o.color ?? "var(--qb-c1, #1f77b4)";
    g.replaceChildren();
    this.els = []; this.leg = null;
    if (o.legs !== false && this.pts.length > 1) {
      const leg = svgEl("path", { d: this.legPath(), fill: "none" }, g);
      leg.style.cssText = `stroke:${col};stroke-width:2;stroke-dasharray:7 5;opacity:.75`;
      this.leg = leg;
    }
    const r = o.r ?? 11;
    this.pts.forEach((p, i) => {
      const fixed = o.fixed?.(i) ?? false, cons = o.constrained?.(i) ?? false;
      const w = o.weights?.[i] ?? 1, rr = r * Math.min(1.8, Math.max(0.6, Math.sqrt(w)));
      const c = svgEl("circle", { cx: X(p[0]).toFixed(1), cy: Y(p[1]).toFixed(1), r: rr.toFixed(1) }, g);
      c.style.cssText = `fill:${fixed || cons ? "var(--qb-bg, #fff)" : col};stroke:${i === this.selected ? "var(--qb-c4, #ff7f0e)" : col};stroke-width:${i === this.selected ? 4 : 2.5};${cons ? "stroke-dasharray:4 3;" : ""}fill-opacity:${fixed || cons ? 1 : 0.85}`;
      let t: SVGTextElement | undefined;
      if (o.label) {
        t = svgEl("text", { x: (X(p[0]) + rr + 3).toFixed(1), y: (Y(p[1]) - rr - 2).toFixed(1), class: "qb-halo" }, g);
        t.textContent = o.label + i;
        t.style.fontSize = "18px";
      }
      this.els.push({ c, t, rr });
      if (fixed || cons) return;
      c.style.cursor = "grab";
      c.style.touchAction = "none";
      c.addEventListener("pointerdown", ev => {
        ev.preventDefault(); ev.stopPropagation();
        c.setPointerCapture(ev.pointerId);
        this.dragging = i;
        if (this.selected !== i) { this.selected = i; o.onSelect?.(i); }
        const move = (e: PointerEvent) => {
          let [x, y] = plot.toData(e);
          if (o.box) { x = Math.min(o.box[1], Math.max(o.box[0], x)); y = Math.min(o.box[3], Math.max(o.box[2], y)); }
          this.pts[i] = [x, y];
          o.onChange?.(this.pts, i);
          this.update();
        };
        const up = () => {
          c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); c.removeEventListener("pointercancel", up);
          this.dragging = -1;
          this.draw();
          o.onRelease?.(i);
        };
        c.addEventListener("pointermove", move); c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
        this.els.forEach((e, k) => { e.c.style.stroke = k === i ? "var(--qb-c4, #ff7f0e)" : col; e.c.style.strokeWidth = k === i ? "4" : "2.5"; });
      });
    });
  }

  private legPath(): string {
    const X = this.plot.x, Y = this.plot.y;
    return this.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("") + (this.o.closed ? "Z" : "");
  }

  /** Move the existing handles to the current points (no rebuild: keeps pointer capture).
   *  Rebuilds instead when the number of points changed. */
  update(): void {
    if (this.els.length !== this.pts.length) { if (this.dragging < 0) this.draw(); return; }
    const X = this.plot.x, Y = this.plot.y;
    this.pts.forEach((p, i) => {
      const e = this.els[i];
      e.c.setAttribute("cx", X(p[0]).toFixed(1)); e.c.setAttribute("cy", Y(p[1]).toFixed(1));
      if (e.t) { e.t.setAttribute("x", (X(p[0]) + e.rr + 3).toFixed(1)); e.t.setAttribute("y", (Y(p[1]) - e.rr - 2).toFixed(1)); }
    });
    if (this.leg) this.leg.setAttribute("d", this.legPath());
  }
}
