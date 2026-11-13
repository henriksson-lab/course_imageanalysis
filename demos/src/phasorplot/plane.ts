// Shared by the FLIM-day demos (phasorplot, flimcell, fretphasor, modulation): a phasor plane
// with the universal semicircle and lifetime ticks for the current frequency, plus a few
// helpers for draggable SVG handles in plot coordinates.
import { Plot, PlotOptions } from "../lib/plot";
import { svgEl } from "../lib/ui";
import { semicircle } from "../lib/phasor";

export const TAU_TICKS = [0.25, 0.5, 1, 2, 4, 8];

export interface PhasorPlane {
  plot: Plot;
  setOmega(omega: number): void;
}

/** A Plot over g ∈ [-0.05, 1.05], s ∈ [-0.04, 0.62] with the semicircle and τ ticks. */
export function phasorPlane(parent: Element, omega: number, opts: PlotOptions = {}): PhasorPlane {
  const plot = new Plot(parent, { width: 520, height: 360, x: [-0.06, 1.06], y: [-0.03, 0.62], equal: true, xlabel: "g", ylabel: "s", xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [0, 0.25, 0.5], xfmt: v => String(v), yfmt: v => String(v), margin: { l: 66, r: 10, t: 10, b: 62 }, ...opts });
  let om = omega;
  plot.mark(g => {
    // semicircle
    const pts: string[] = [];
    for (let i = 0; i <= 180; i++) { const a = Math.PI * i / 180; pts.push(`${plot.x(0.5 + 0.5 * Math.cos(a)).toFixed(2)},${plot.y(0.5 * Math.sin(a)).toFixed(2)}`); }
    const p = svgEl("path", { d: "M" + pts.join("L") }, g);
    p.style.cssText = "fill:none;stroke:var(--qb-axis,#444);stroke-width:2";
    for (const t of TAU_TICKS) {
      const q = semicircle(om * t);
      const cx = plot.x(q.g), cy = plot.y(q.s);
      const c = svgEl("circle", { cx, cy, r: 3.5 }, g);
      c.style.fill = "var(--qb-axis,#444)";
      // label outside the circle, along the radius from (0.5, 0)
      const dx = q.g - 0.5, dy = q.s, L = Math.hypot(dx, dy) || 1;
      const lx = plot.x(q.g + 0.075 * dx / L), ly = plot.y(q.s + 0.075 * dy / L);
      const tx = svgEl("text", { x: lx, y: ly + 6, "text-anchor": "middle", class: "qb-halo" }, g);
      tx.textContent = String(t) + (t === 2 ? " ns" : "");
      tx.style.fontSize = "16px";
    }
  }, [], {}, "line", plot.over);
  return { plot, setOmega(o: number) { om = o; plot.renderAll(); } };
}

/** A draggable circle handle in plot data coordinates; returns a setter for its position. */
export function handle(plot: Plot, color: string, r: number, onDrag: (x: number, y: number) => void, label = ""): { set(x: number, y: number): void; el: SVGGElement } {
  const g = svgEl("g", { class: "qb-handle" }, plot.over);
  const c = svgEl("circle", { r, cx: 0, cy: 0 }, g);
  c.style.cssText = `fill:${color};stroke:var(--qb-bg,#fff);stroke-width:2.5;cursor:grab;touch-action:none`;
  const t = svgEl("text", { x: r + 4, y: -r, class: "qb-halo" }, g);
  t.textContent = label;
  t.style.fontSize = "18px"; t.style.fontWeight = "600"; t.style.fill = color;
  c.addEventListener("pointerdown", ev => {
    ev.preventDefault(); c.setPointerCapture(ev.pointerId);
    const move = (e: PointerEvent) => { const [x, y] = plot.toData(e); onDrag(x, y); };
    const up = () => { c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); c.removeEventListener("pointercancel", up); };
    c.addEventListener("pointermove", move); c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
  });
  let lx = 0, ly = 0;
  const set = (x: number, y: number) => { lx = x; ly = y; g.setAttribute("transform", `translate(${plot.x(lx).toFixed(2)},${plot.y(ly).toFixed(2)})`); };
  return { set, el: g };
}

/** A log-scale slider value: u ∈ [0, 100] ↔ v ∈ [lo, hi]. */
export const logMap = (lo: number, hi: number) => ({
  v: (u: number) => lo * Math.pow(hi / lo, u / 100),
  u: (v: number) => 100 * Math.log(v / lo) / Math.log(hi / lo),
});

/** Format a count compactly: 1234 → "1 234", 1.2e6 → "1.2 M". */
export function fmtCount(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " M";
  return Math.round(n).toLocaleString("en-US").replace(/,/g, " ");
}

/** Short form for slider outputs: 950, 12 k, 1.2 M. */
export function fmtShort(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e4) return Math.round(n / 1e3) + "k";
  return String(Math.round(n));
}

/** Shared CSS of the FLIM demos (the per-demo selectors add to it). */
export const FLIM_CSS = `
.qb-demo .fl-col { display: flex; flex-direction: column; gap: .3em; min-width: 0; }
.qb-demo .fl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo .fl-ro { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 15px; line-height: 1.35; }
.reveal .qb-demo .fl-lab { font-size: 15px; }
.reveal .qb-demo .fl-ro { font-size: 16px; }
.reveal .qb-demo .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal .qb-demo .qb-slider { font-size: 17px; }
`;
