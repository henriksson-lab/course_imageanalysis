// Small SVG helpers for the moments overlays (canvas units; the overlay group is scaled so
// one unit = one canvas pixel, hence the thin stroke widths) and the readout table.
import { svgEl, htmlEl } from "../lib/ui";

export type P2 = [number, number];

export function sline(g: SVGGElement, a: P2, b: P2, color: string, width = 0.35, dash?: string): SVGLineElement {
  const l = svgEl("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, g);
  l.style.setProperty("stroke", color); l.style.setProperty("stroke-width", String(width));
  if (dash) l.style.setProperty("stroke-dasharray", dash);
  l.style.setProperty("stroke-linecap", "round");
  return l;
}

export function spoly(g: SVGGElement, pts: P2[], color: string, width = 0.35, fill = "none", dash?: string, closed = true): SVGPathElement {
  const p = svgEl("path", { d: pts.length ? "M" + pts.map(q => `${q[0].toFixed(3)},${q[1].toFixed(3)}`).join("L") + (closed ? "Z" : "") : "" }, g);
  p.style.setProperty("stroke", color); p.style.setProperty("stroke-width", String(width)); p.style.setProperty("fill", fill);
  p.style.setProperty("stroke-linejoin", "round");
  if (fill !== "none") p.style.setProperty("fill-opacity", "0.15");
  if (dash) p.style.setProperty("stroke-dasharray", dash);
  return p;
}

export function scircle(g: SVGGElement, c: P2, r: number, color: string, width = 0.35, dash?: string): SVGCircleElement {
  const e = svgEl("circle", { cx: c[0], cy: c[1], r: Math.max(0, r) }, g);
  e.style.setProperty("stroke", color); e.style.setProperty("stroke-width", String(width)); e.style.setProperty("fill", "none");
  if (dash) e.style.setProperty("stroke-dasharray", dash);
  return e;
}

export function sarrow(g: SVGGElement, a: P2, b: P2, color: string, width = 0.45): void {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
  if (L < 1e-6) return;
  const ux = dx / L, uy = dy / L, hl = Math.min(2.2, L * 0.4), hw = hl * 0.5;
  sline(g, a, [b[0] - ux * hl * 0.8, b[1] - uy * hl * 0.8], color, width);
  const h = svgEl("polygon", { points: `${b[0]},${b[1]} ${b[0] - ux * hl - uy * hw},${b[1] - uy * hl + ux * hw} ${b[0] - ux * hl + uy * hw},${b[1] - uy * hl - ux * hw}` }, g);
  h.style.setProperty("fill", color);
}

export function scross(g: SVGGElement, c: P2, s: number, color: string, width = 0.45): void {
  sline(g, [c[0] - s, c[1]], [c[0] + s, c[1]], color, width);
  sline(g, [c[0], c[1] - s], [c[0], c[1] + s], color, width);
}

/** A two- or three-column table; rows of [name, value, (extra)]; a row with a single string
 *  is a header. */
export function table(parent: HTMLElement, rows: (string[] | string)[]): HTMLTableElement {
  parent.replaceChildren();
  const t = htmlEl("table", "mo-tab", "", parent);
  for (const r of rows) {
    const tr = htmlEl("tr", typeof r === "string" ? "h" : "", "", t);
    if (typeof r === "string") { const td = htmlEl("td", "", r, tr); td.colSpan = 3; continue; }
    r.forEach((c, i) => htmlEl("td", i ? "v" : "", c, tr));
  }
  return t;
}

export const f2 = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "∞");
