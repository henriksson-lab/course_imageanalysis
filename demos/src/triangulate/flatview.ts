// SVG view of flatland (lib/flatland.ts): a top-down 2D world with y up, 1D cameras drawn as a
// centre, a frustum and an image line (optionally with its pixels), plus helpers for rays,
// wedges, polygons, ellipses and draggable handles. Shared by the triangulate, bundleadjust
// and flatland demos (geometry-from-images day).
import { svgEl, svgPoint, draggable } from "../lib/ui";
import { Cam1D, P, fwd, right } from "../lib/flatland";

export function injectStyle(name: string, css: string): void {
  const id = "qb-style-" + name;
  if (document.getElementById(id)) return;
  const s = document.createElement("style");
  s.id = id; s.textContent = css;
  document.head.appendChild(s);
}

export class FlatView {
  readonly svg: SVGSVGElement;
  readonly bg: SVGGElement;
  readonly layer: SVGGElement;
  readonly top: SVGGElement;
  /** Persistent handles (not cleared by clear()). */
  readonly handles: SVGGElement;
  /** World units → SVG units. */
  k: number;
  constructor(parent: Element, readonly W: number, readonly H: number, readonly x0: number, readonly x1: number, readonly y0: number, readonly y1: number) {
    this.svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", class: "fl-svg" }, parent);
    this.k = Math.min(W / (x1 - x0), H / (y1 - y0));
    this.bg = svgEl("g", {}, this.svg);
    this.layer = svgEl("g", {}, this.svg);
    this.top = svgEl("g", {}, this.svg);
    this.handles = svgEl("g", {}, this.svg);
    const fr = svgEl("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, rx: 4, fill: "none" }, this.bg);
    fr.style.stroke = "var(--qb-grid)";
  }
  sx(x: number): number { return (this.W - this.k * (this.x1 - this.x0)) / 2 + this.k * (x - this.x0); }
  sy(y: number): number { return (this.H - this.k * (this.y1 - this.y0)) / 2 + this.k * (this.y1 - y); }
  toWorld(ev: { clientX: number; clientY: number }): P {
    const [px, py] = svgPoint(this.svg, ev);
    const ox = (this.W - this.k * (this.x1 - this.x0)) / 2, oy = (this.H - this.k * (this.y1 - this.y0)) / 2;
    return [this.x0 + (px - ox) / this.k, this.y1 - (py - oy) / this.k];
  }
  clear(): void { this.layer.replaceChildren(); this.top.replaceChildren(); }

  line(a: P, b: P, color: string, width = 2, dash = "", g: SVGGElement = this.layer): SVGLineElement {
    const l = svgEl("line", { x1: this.sx(a[0]), y1: this.sy(a[1]), x2: this.sx(b[0]), y2: this.sy(b[1]), "stroke-width": width, "stroke-linecap": "round" }, g);
    l.style.stroke = color;
    if (dash) l.setAttribute("stroke-dasharray", dash);
    return l;
  }
  poly(pts: P[], fill: string, stroke = "none", opacity = 1, width = 1.5, g: SVGGElement = this.layer): SVGPolygonElement {
    const p = svgEl("polygon", { points: pts.map(q => `${this.sx(q[0]).toFixed(2)},${this.sy(q[1]).toFixed(2)}`).join(" "), "stroke-width": width }, g);
    p.style.fill = fill; p.style.stroke = stroke; p.style.fillOpacity = String(opacity);
    return p;
  }
  polyline(pts: P[], stroke: string, width = 2, g: SVGGElement = this.layer): SVGPolylineElement {
    const p = svgEl("polyline", { points: pts.map(q => `${this.sx(q[0]).toFixed(2)},${this.sy(q[1]).toFixed(2)}`).join(" "), "stroke-width": width, fill: "none" }, g);
    p.style.stroke = stroke;
    return p;
  }
  dot(p: P, r: number, fill: string, stroke = "", g: SVGGElement = this.layer): SVGCircleElement {
    const c = svgEl("circle", { cx: this.sx(p[0]), cy: this.sy(p[1]), r }, g);
    c.style.fill = fill;
    if (stroke) { c.style.stroke = stroke; c.style.strokeWidth = "2"; }
    return c;
  }
  text(p: P, s: string, size = 18, anchor = "middle", color = "", g: SVGGElement = this.top, dy = 0): SVGTextElement {
    const t = svgEl("text", { x: this.sx(p[0]), y: this.sy(p[1]) + dy, "text-anchor": anchor, "font-size": size, class: "qb-halo" }, g);
    t.textContent = s;
    if (color) t.style.fill = color;
    return t;
  }
  /** Ellipse from a 2 × 2 covariance [sxx, sxy, syy] at nsd standard deviations. */
  ellipse(c: P, cov: [number, number, number], nsd: number, stroke: string, g: SVGGElement = this.layer): void {
    const [a, b, d] = cov, tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc), ang = Math.atan2(l1 - a, b || 1e-30);
    const pts: P[] = [];
    for (let k = 0; k <= 48; k++) {
      const t = 2 * Math.PI * k / 48, u = nsd * Math.sqrt(l1) * Math.cos(t), v = nsd * Math.sqrt(l2) * Math.sin(t);
      pts.push([c[0] + u * Math.cos(ang) - v * Math.sin(ang), c[1] + u * Math.sin(ang) + v * Math.cos(ang)]);
    }
    this.polyline(pts, stroke, 2, g);
  }

  /**
   * A camera: centre, frustum to an image line at distance `len` (world units), optional pixel
   * values on the image line (grey 0..1), a label. Returns the image-line end points.
   */
  camera(cam: Cam1D, color: string, o: { len?: number; values?: ArrayLike<number>; label?: string; faded?: boolean; frustum?: number } = {}): { a: P; b: P } {
    const len = o.len ?? 0.6, F = fwd(cam), R = right(cam), half = (cam.n / 2) / cam.f * len;
    const ctr: P = [cam.x + F[0] * len, cam.y + F[1] * len];
    const a: P = [ctr[0] - R[0] * half, ctr[1] - R[1] * half], b: P = [ctr[0] + R[0] * half, ctr[1] + R[1] * half];
    const g = svgEl("g", {}, this.layer);
    if (o.faded) g.style.opacity = "0.35";
    if (o.frustum) {
      const far = o.frustum, hf = (cam.n / 2) / cam.f * far;
      const fc: P = [cam.x + F[0] * far, cam.y + F[1] * far];
      this.poly([[cam.x, cam.y], [fc[0] - R[0] * hf, fc[1] - R[1] * hf], [fc[0] + R[0] * hf, fc[1] + R[1] * hf]], color, "none", 0.06, 1, g);
    }
    this.poly([[cam.x, cam.y], a, b], color, color, 0.18, 1.5, g);
    if (o.values) {
      const n = o.values.length;
      for (let i = 0; i < n; i++) {
        const t0 = i / n, t1 = (i + 1) / n;
        const p0: P = [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0], p1: P = [a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1];
        const v = Math.round(255 * Math.max(0, Math.min(1, o.values[i])));
        this.line(p0, p1, `rgb(${v},${v},${v})`, 9, "", g).setAttribute("stroke-linecap", "butt");
      }
    } else this.line(a, b, color, 4, "", g);
    this.dot([cam.x, cam.y], 7, color, "#fff", g);
    if (o.label) this.text([cam.x - F[0] * 0.25, cam.y - F[1] * 0.25], o.label, 18, "middle", color, this.top, 6);
    return { a, b };
  }

  /** Point on the image line of a camera drawn with `len`, at pixel coordinate u. */
  imagePoint(cam: Cam1D, u: number, len = 0.6): P {
    const F = fwd(cam), R = right(cam), x = (u - cam.c) / cam.f * len;
    return [cam.x + F[0] * len + R[0] * x, cam.y + F[1] * len + R[1] * x];
  }

  /** A persistent draggable handle; move it with set(p), hide with show(false). */
  handle(r: number, color: string, onDrag: (p: P) => void, title = ""): { el: SVGCircleElement; set(p: P): void; show(v: boolean): void } {
    const c = svgEl("circle", { r }, this.handles);
    c.style.fill = "rgba(255,255,255,0.01)"; c.style.stroke = color; c.style.strokeWidth = "3";
    if (title) { const t = svgEl("title", {}, c); t.textContent = title; }
    draggable(c, ev => this.toWorld(ev), (x, y) => onDrag([x, y]));
    return {
      el: c,
      set: (p: P) => { c.setAttribute("cx", String(this.sx(p[0]))); c.setAttribute("cy", String(this.sy(p[1]))); },
      show: (v: boolean) => { c.style.display = v ? "" : "none"; },
    };
  }
}

export const CAM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];
