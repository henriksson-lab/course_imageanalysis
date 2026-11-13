// Small DOM helpers shared by the FRET-day demos (threecube, apbleach, ratioimage,
// fretruler, rationoise): an image tile (caption, pixelated canvas, SVG overlay in image
// pixel coordinates, click/drag picking) and per-demo style injection. Candidate for a
// shared lib (the same "image panel with overlay" exists in canny/imgview.ts).
import { htmlEl, svgEl } from "../lib/ui";

export class Tile {
  readonly fig: HTMLElement;
  readonly cap: HTMLElement;
  readonly wrap: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly svg: SVGSVGElement;
  readonly over: SVGGElement;
  w = 1; h = 1;

  constructor(parent: Element, caption: string, cls = "") {
    this.fig = htmlEl("figure", "fr-fig" + (cls ? " " + cls : ""), "", parent);
    this.cap = htmlEl("figcaption", "", caption, this.fig);
    this.wrap = htmlEl("div", "fr-wrap", "", this.fig);
    this.canvas = htmlEl("canvas", "fr-canvas", "", this.wrap);
    this.svg = svgEl("svg", { class: "fr-svg", preserveAspectRatio: "none" }, this.wrap);
    this.over = svgEl("g", {}, this.svg);
  }

  setCaption(s: string): void { this.cap.textContent = s; }

  /** Put RGBA pixels (w × h) on the canvas. */
  rgba(px: Uint8ClampedArray, w: number, h: number): void {
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (w !== this.w || h !== this.h) { this.w = w; this.h = h; this.svg.setAttribute("viewBox", `0 0 ${w} ${h}`); this.wrap.style.aspectRatio = `${w} / ${h}`; }
    const id = new ImageData(w, h);
    id.data.set(px);
    this.canvas.getContext("2d")!.putImageData(id, 0, 0);
  }

  /** Grey display of values between lo and hi. */
  grey(v: ArrayLike<number>, w: number, h: number, lo: number, hi: number): void {
    const px = new Uint8ClampedArray(4 * w * h), k = 255 / ((hi - lo) || 1);
    for (let i = 0; i < w * h; i++) {
      const g = (v[i] - lo) * k;
      px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = g; px[4 * i + 3] = 255;
    }
    this.rgba(px, w, h);
  }

  /** Image-pixel position (fractional) of a pointer event. */
  toImage(ev: { clientX: number; clientY: number }): [number, number] {
    const r = this.wrap.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * this.w, (ev.clientY - r.top) / r.height * this.h];
  }

  /** Pointer down / move (while down) / up callbacks in image coordinates. */
  onDrag(down: (x: number, y: number) => void, move: (x: number, y: number) => void, up: () => void = () => {}): void {
    this.wrap.style.cursor = "crosshair";
    this.wrap.style.touchAction = "none";
    let active = false;
    this.wrap.addEventListener("pointerdown", ev => { ev.preventDefault(); active = true; this.wrap.setPointerCapture(ev.pointerId); down(...this.toImage(ev)); });
    this.wrap.addEventListener("pointermove", ev => { if (active) move(...this.toImage(ev)); });
    const end = () => { if (active) { active = false; up(); } };
    this.wrap.addEventListener("pointerup", end);
    this.wrap.addEventListener("pointercancel", end);
  }

  clear(): void { this.over.replaceChildren(); }

  add<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
    return svgEl(tag, { "vector-effect": "non-scaling-stroke", ...attrs }, this.over);
  }

  /** Text with a light halo, in image coordinates (size in image pixels). */
  label(x: number, y: number, s: string, size: number, color = "#111"): SVGTextElement {
    const t = svgEl("text", { x, y, "font-size": size, "text-anchor": "middle", "dominant-baseline": "central", "paint-order": "stroke", stroke: "rgba(255,255,255,.85)", "stroke-width": size * 0.25, "font-weight": 600 }, this.over);
    t.style.fill = color;
    t.textContent = s;
    return t;
  }
}

/** CSS shared by the FRET demos, scoped to each demo (demo.css loads late). */
export function tileCss(name: string): string {
  const q = `.qb-demo[data-demo="${name}"]`;
  return `
${q} .fr-fig { margin: 0; min-width: 0; }
${q} .fr-fig figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; line-height: 1.15; }
${q} .fr-wrap { position: relative; width: 100%; aspect-ratio: 1; }
${q} .fr-canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 3px; display: block; background: #000; }
${q} .fr-svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
${q} .fr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.3em; }
${q} .qb-readout { font-size: 14px; }
${q} .qb-panel { gap: .35em; }
${q} .qb-btn { padding: .12em .5em; }
.reveal ${q} { font-size: 18px; margin: 0; }
.reveal ${q} .fr-fig figcaption { font-size: 18px; }
.reveal ${q} .fr-lab { font-size: 18px; }
.reveal ${q} .qb-readout { font-size: 18px; padding: .2em .4em; line-height: 1.25; }
.reveal ${q} .qb-btn { font-size: 18px; padding: .05em .35em; }
.reveal ${q} .qb-panel { gap: .2em; }
.reveal ${q} .qb-slider { gap: .3em; }
`;
}

export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}

/** Number formatting with a fixed number of decimals and a proper minus sign. */
export const fmt = (v: number, d = 2): string => (Number.isFinite(v) ? v.toFixed(d).replace("-", "−") : v > 0 ? "∞" : "—");

/** Debounce: run fn after `ms` without further calls. */
export function debounce(fn: () => void, ms: number): () => void {
  let t = 0;
  return () => { if (t) clearTimeout(t); t = window.setTimeout(() => { t = 0; fn(); }, ms); };
}

/** Keep only the presets named in data-presets="id1 id2 …" (all if absent). The preset a
 *  demo starts with (data-preset) is always kept. */
export function filterPresets<T extends { id: string }>(host: HTMLElement, all: T[]): T[] {
  const want = (host.dataset.presets ?? "").trim().split(/[\s,]+/).filter(Boolean);
  if (!want.length) return all;
  return all.filter(p => want.includes(p.id) || p.id === host.dataset.preset);
}

/** Hide the row of a slider (or any control inside a label). */
export function hideControl(el: { input: HTMLElement } | HTMLElement): void {
  const inp = "input" in el && !(el instanceof HTMLElement) ? el.input : (el as HTMLElement);
  const row = inp.closest("label") ?? inp;
  (row as HTMLElement).style.display = "none";
}
