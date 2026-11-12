// DOM helpers shared by the Hough day's demos (houghlines, houghcircle, ght): an image
// panel = a pixel canvas with an SVG overlay in pixel units (lines, circles, markers), with
// pointer picking that works in scaled reveal.js slides; a heat-map painter for
// accumulators; and style injection.
import { htmlEl, svgEl } from "../lib/ui";
import { colormapLUT, ColorMap, inferno } from "../lib/colour";

export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}

/** Shared CSS for the panels (scoped to the demo by the caller's prefix). */
export const PANEL_CSS = (name: string) => `
[data-demo="${name}"] .hv-fig { margin: 0; min-width: 0; }
[data-demo="${name}"] .hv-fig figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; line-height: 1.2; }
[data-demo="${name}"] .hv-box { position: relative; width: 100%; }
[data-demo="${name}"] .hv-box canvas { width: 100%; height: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${name}"] .hv-box svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; overflow: hidden; }
[data-demo="${name}"] .hv-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${name}"] .hv-credit { font-size: 12px; color: var(--qb-muted); }
`;

export class ImgPanel {
  readonly fig: HTMLElement;
  readonly cap: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly svg: SVGSVGElement;
  /** Overlay group (cleared by clearOverlay). */
  readonly over: SVGGElement;
  w = 1; h = 1;

  constructor(parent: Element, caption: string, w: number, h: number, aspect?: number) {
    this.fig = htmlEl("figure", "hv-fig", "", parent);
    this.cap = htmlEl("figcaption", "", caption, this.fig);
    const box = htmlEl("div", "hv-box", "", this.fig);
    box.style.aspectRatio = String(aspect ?? w / h);
    this.canvas = htmlEl("canvas", "", "", box);
    this.svg = svgEl("svg", { preserveAspectRatio: "none" }, box);
    this.over = svgEl("g", {}, this.svg);
    this.resize(w, h);
  }

  resize(w: number, h: number): void {
    this.w = w; this.h = h;
    this.canvas.width = w; this.canvas.height = h;
    this.svg.setAttribute("viewBox", `-0.5 -0.5 ${w} ${h}`);
  }

  /** Pixel coordinates (continuous, pixel centres at integers) of a pointer event. */
  toPixel(ev: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * this.w - 0.5, (ev.clientY - r.top) / r.height * this.h - 0.5];
  }

  onPick(fn: (x: number, y: number) => void): void {
    this.svg.style.cursor = "crosshair";
    this.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); fn(...this.toPixel(ev)); });
  }

  putRGBA(rgba: Uint8ClampedArray): void {
    const ctx = this.canvas.getContext("2d")!;
    const id = ctx.createImageData(this.w, this.h);
    id.data.set(rgba);
    ctx.putImageData(id, 0, 0);
  }

  clearOverlay(): void { this.over.replaceChildren(); }

  line(x0: number, y0: number, x1: number, y1: number, color: string, width = 1.5, dash?: string): SVGLineElement {
    const l = svgEl("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: color, "stroke-width": width, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke" }, this.over);
    if (dash) l.setAttribute("stroke-dasharray", dash);
    return l;
  }

  circle(x: number, y: number, r: number, color: string, width = 2, fill = "none", dash?: string): SVGCircleElement {
    const c = svgEl("circle", { cx: x, cy: y, r, stroke: color, "stroke-width": width, fill, "vector-effect": "non-scaling-stroke" }, this.over);
    if (dash) c.setAttribute("stroke-dasharray", dash);
    return c;
  }

  dot(x: number, y: number, r: number, color: string): SVGCircleElement {
    return svgEl("circle", { cx: x, cy: y, r, fill: color, stroke: "#fff", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, this.over);
  }

  polyline(pts: [number, number][], color: string, width = 2, closed = false): SVGPolylineElement {
    return svgEl(closed ? "polygon" : "polyline", { points: pts.map(p => p.join(",")).join(" "), stroke: color, "stroke-width": width, fill: "none", "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }, this.over) as SVGPolylineElement;
  }
}

/** RGBA of a grey image (display range lo..hi), with optional per-pixel colour overrides. */
export function greyRGBA(data: ArrayLike<number>, lo: number, hi: number, dim = 1): Uint8ClampedArray {
  const out = new Uint8ClampedArray(data.length * 4), k = 255 / Math.max(1e-9, hi - lo);
  for (let i = 0; i < data.length; i++) {
    const v = Math.max(0, Math.min(255, (data[i] - lo) * k)) * dim;
    out[4 * i] = v; out[4 * i + 1] = v; out[4 * i + 2] = v; out[4 * i + 3] = 255;
  }
  return out;
}

/** Paint a set of pixels (indices) in a colour into an RGBA buffer. */
export function paintPixels(rgba: Uint8ClampedArray, idx: ArrayLike<number>, rgb: [number, number, number]): void {
  for (let n = 0; n < idx.length; n++) { const i = idx[n]; rgba[4 * i] = rgb[0]; rgba[4 * i + 1] = rgb[1]; rgba[4 * i + 2] = rgb[2]; }
}

/** RGBA heat map of values (0..hi mapped through a colour map; sqrt option for a wide range). */
export function heatRGBA(values: ArrayLike<number>, hi: number, map: ColorMap = inferno, gamma = 1): Uint8ClampedArray {
  const lut = colormapLUT(map, 256), out = new Uint8ClampedArray(values.length * 4), inv = 1 / Math.max(1e-9, hi);
  for (let i = 0; i < values.length; i++) {
    let t = Math.max(0, Math.min(1, values[i] * inv));
    if (gamma !== 1) t = Math.pow(t, gamma);
    const k = Math.round(255 * t);
    out[4 * i] = lut[4 * k]; out[4 * i + 1] = lut[4 * k + 1]; out[4 * i + 2] = lut[4 * k + 2]; out[4 * i + 3] = 255;
  }
  return out;
}

export const COL = { red: "#e8402a", orange: "#ff9f1c", blue: "#4ea3ff", green: "#3ddc84", yellow: "#ffe14d", cyan: "#29d3e0", magenta: "#ff5bd1" };

export const fmtInt = (n: number): string => Math.round(n).toLocaleString("en-US");

/** A max over a typed array. */
export function maxOf(a: ArrayLike<number>): number { let m = 0; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
