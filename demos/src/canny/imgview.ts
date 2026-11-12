// An image panel for the edges-day demos (canny, blobscale, hessian, structtensor): a figure
// with a caption, a pixelated canvas and an SVG overlay in image-pixel coordinates (pixel
// (x, y) covers [x, x+1] × [y, y+1], so its centre is (x + 0.5, y + 0.5)). Clicks report the
// integer pixel. Strokes in the overlay keep their screen width (non-scaling-stroke).
// Candidate for a shared lib (an "image panel with overlay" is needed by many days).
import { htmlEl, svgEl, SVGNS } from "../lib/ui";
import { Grey, displayRange, DisplayOptions } from "../lib/imageops";
import { ColorMap, scalarToRGBA, hsvToRgb } from "../lib/colour";

export class ImgView {
  readonly fig: HTMLElement;
  readonly cap: HTMLElement;
  readonly wrap: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly svg: SVGSVGElement;
  readonly over: SVGGElement;
  w = 1; h = 1;

  constructor(parent: Element, caption: string, cls = "") {
    this.fig = htmlEl("figure", "ev-fig " + cls, "", parent);
    this.cap = htmlEl("figcaption", "", caption, this.fig);
    this.wrap = htmlEl("div", "ev-wrap", "", this.fig);
    this.canvas = htmlEl("canvas", "ev-canvas", "", this.wrap);
    this.svg = svgEl("svg", { class: "ev-svg", preserveAspectRatio: "none" }, this.wrap);
    this.over = svgEl("g", {}, this.svg);
  }

  setCaption(s: string): void { this.cap.textContent = s; }

  private size(w: number, h: number): void {
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (w !== this.w || h !== this.h) { this.w = w; this.h = h; this.svg.setAttribute("viewBox", `0 0 ${w} ${h}`); this.wrap.style.aspectRatio = `${w} / ${h}`; }
  }

  /** Grey display of an image (see imageops DisplayOptions: clip/auto/abs/mid). */
  grey(img: Grey, opts: DisplayOptions = { mode: "auto" }): void {
    const { lo, hi } = displayRange(img, opts), abs = opts.mode === "abs";
    const v = abs ? Float32Array.from(img.data, Math.abs) : img.data;
    this.rgba(scalarToRGBA(v, { lo, hi }), img.width, img.height);
  }

  /** Colour-map display of values. */
  map(values: ArrayLike<number>, w: number, h: number, map: ColorMap, lo: number, hi: number): void {
    this.rgba(scalarToRGBA(values, { map, lo, hi }), w, h);
  }

  rgba(px: Uint8ClampedArray, w: number, h: number): void {
    this.size(w, h);
    const id = new ImageData(w, h);
    id.data.set(px);
    this.canvas.getContext("2d")!.putImageData(id, 0, 0);
  }

  /** Call cb with the integer pixel under a click or drag. */
  onPick(cb: (x: number, y: number) => void, drag = true): void {
    this.wrap.style.cursor = "crosshair";
    this.wrap.style.touchAction = "none";
    const pick = (ev: PointerEvent) => {
      const r = this.wrap.getBoundingClientRect();
      const x = Math.floor((ev.clientX - r.left) / r.width * this.w), y = Math.floor((ev.clientY - r.top) / r.height * this.h);
      if (x >= 0 && y >= 0 && x < this.w && y < this.h) cb(x, y);
    };
    let down = false;
    this.wrap.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; this.wrap.setPointerCapture(ev.pointerId); pick(ev); });
    this.wrap.addEventListener("pointermove", ev => { if (down && drag) pick(ev); });
    const up = () => { down = false; };
    this.wrap.addEventListener("pointerup", up);
    this.wrap.addEventListener("pointercancel", up);
  }

  clear(): void { this.over.replaceChildren(); }

  /** Add an SVG element to the overlay; strokes do not scale with the image. */
  add<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent: Element = this.over): SVGElementTagNameMap[K] {
    const e = svgEl(tag, { "vector-effect": "non-scaling-stroke", ...attrs }, parent);
    return e;
  }

  /** A square frame around pixel (x, y) (or a (2r+1)² window). */
  box(x: number, y: number, r = 0, color = "#ffd400", width = 2.5): SVGRectElement {
    return this.add("rect", { x: x - r, y: y - r, width: 2 * r + 1, height: 2 * r + 1, fill: "none", stroke: color, "stroke-width": width });
  }

  /** An arrow from (x0, y0) to (x1, y1) in image coordinates. */
  arrow(x0: number, y0: number, x1: number, y1: number, color: string, width = 3, head = 0.3): SVGGElement {
    const g = svgEl("g", {}, this.over);
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, hl = Math.max(0.6, head * L);
    this.add("line", { x1: x0, y1: y0, x2: x1 - ux * hl * 0.6, y2: y1 - uy * hl * 0.6, stroke: color, "stroke-width": width, "stroke-linecap": "round" }, g);
    const bx = x1 - ux * hl, by = y1 - uy * hl, hw = hl * 0.45;
    svgEl("polygon", { points: `${x1},${y1} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}`, fill: color }, g);
    return g;
  }
}

export { SVGNS };

/** RGBA of a direction/magnitude field: hue = angle (radians, on screen), value = magnitude / vmax. Angles mod π if `axial`. */
export function hueRGBA(angle: ArrayLike<number>, value: ArrayLike<number>, vmax: number, sat?: ArrayLike<number>, axial = false): Uint8ClampedArray {
  const n = angle.length, out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    let deg = angle[i] * 180 / Math.PI * (axial ? 2 : 1);
    deg = ((deg % 360) + 360) % 360;
    const v = Math.min(1, Math.max(0, value[i] / (vmax || 1)));
    const c = hsvToRgb([deg, sat ? Math.min(1, Math.max(0, sat[i])) : 1, v]);
    out[4 * i] = 255 * c[0]; out[4 * i + 1] = 255 * c[1]; out[4 * i + 2] = 255 * c[2]; out[4 * i + 3] = 255;
  }
  return out;
}

/** A small colour wheel (SVG) as a legend for hue = direction; `axial`: 0° and 180° share a colour. */
export function colourWheel(parent: Element, size = 90, axial = false, label = ""): SVGSVGElement {
  const svg = svgEl("svg", { viewBox: "-60 -60 120 120", width: size, height: size, class: "ev-wheel" }, parent);
  for (let a = 0; a < 360; a += 6) {
    const t0 = a * Math.PI / 180, t1 = (a + 6.5) * Math.PI / 180;
    const deg = axial ? (2 * a) % 360 : a;
    const c = hsvToRgb([deg, 1, 1]).map(v => Math.round(255 * v));
    // screen y is down: angle CCW on screen → (cos, −sin)
    const p = (r: number, t: number) => `${(r * Math.cos(t)).toFixed(2)},${(-r * Math.sin(t)).toFixed(2)}`;
    svgEl("path", { d: `M${p(22, t0)}L${p(46, t0)}L${p(46, t1)}L${p(22, t1)}Z`, fill: `rgb(${c.join(",")})` }, svg);
  }
  const t = svgEl("text", { x: 0, y: 7, "text-anchor": "middle", "font-size": 20 }, svg);
  t.textContent = label;
  return svg;
}

/** CSS shared by the edges-day demos (injected once). */
export const IMGVIEW_CSS = `
.qb-demo .ev-fig { margin: 0; min-width: 0; }
.qb-demo .ev-fig figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; line-height: 1.2; }
.qb-demo .ev-wrap { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo .ev-canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 3px; display: block; }
.qb-demo .ev-svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
.qb-demo .ev-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo .ev-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo .ev-tabs .qb-btn { padding: .15em .5em; }
.reveal .qb-demo .ev-fig figcaption { font-size: 18px; }
.reveal .qb-demo .ev-lab { font-size: 15px; }
`;

export function injectStyle(name: string, css: string): void {
  if (!document.getElementById("qb-style-ev")) {
    const s = document.createElement("style"); s.id = "qb-style-ev"; s.textContent = IMGVIEW_CSS; document.head.appendChild(s);
  }
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}
