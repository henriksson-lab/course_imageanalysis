// DOM helpers shared by the unmixing day's demos (spectramix, bleedthrough, lsunmix,
// unmixcloud, endmembers, stainsep). Local to the day: small canvas and style helpers that
// lib/ has no exact equivalent for.
import { htmlEl } from "../lib/ui";
import { wavelengthToSrgb } from "../lib/spectra";
import { RGB, ColorMap, scalarToRGBA } from "../lib/colour";

/** Add a demo's CSS once per page. */
export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name; s.textContent = css;
  document.head.appendChild(s);
}

/** lib/plot.ts numbers its clip paths per bundle, so two demos on one page can clash:
 *  give every clip path of an SVG a page-unique id (same workaround as colourcloud/plotfix.ts;
 *  belongs in lib/plot.ts). */
let clipN = 0;
export function uniqueClip(svg: SVGSVGElement, prefix: string): void {
  for (const cp of Array.from(svg.querySelectorAll("clipPath"))) {
    const old = cp.id;
    let id: string;
    do { id = `${prefix}-clip-${++clipN}`; } while (document.getElementById(id));
    cp.id = id;
    for (const el of Array.from(svg.querySelectorAll(`[clip-path="url(#${old})"]`))) el.setAttribute("clip-path", `url(#${id})`);
  }
}

/** CSS colour of a wavelength (nm). */
export function wlCss(l: number, alpha = 1): string {
  const c = wavelengthToSrgb(l);
  return `rgba(${Math.round(255 * c[0])},${Math.round(255 * c[1])},${Math.round(255 * c[2])},${alpha})`;
}

/** A captioned canvas (figure) of w × h pixels, shown at CSS width `cssW`. */
export function canvasFigure(parent: Element, caption: string, w: number, h: number, cls = ""): { fig: HTMLElement; canvas: HTMLCanvasElement; cap: HTMLElement } {
  const fig = htmlEl("figure", "qb-figure um-fig " + cls, "", parent);
  const cap = htmlEl("figcaption", "", caption, fig);
  const canvas = htmlEl("canvas", "", "", fig);
  canvas.width = w; canvas.height = h;
  return { fig, canvas, cap };
}

/** Draw a scalar map with a colour map (lo..hi). */
export function drawMap(canvas: HTMLCanvasElement, v: ArrayLike<number>, w: number, h: number, map: ColorMap, lo: number, hi: number): void {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d")!, id = ctx.createImageData(w, h);
  scalarToRGBA(v, { map, lo, hi }, id.data);
  ctx.putImageData(id, 0, 0);
}

/** Additive pseudocolour merge: map k (offset `off` inside a channel-major array, or an
 *  array of maps) times colour k, each scaled by 1/hi[k]; display values clipped at 1. */
export function drawMerge(canvas: HTMLCanvasElement, maps: ArrayLike<number>[], cols: RGB[], hi: number[], w: number, h: number): void {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d")!, id = ctx.createImageData(w, h), d = id.data, P = w * h;
  for (let p = 0; p < P; p++) {
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < maps.length; k++) {
      const v = Math.max(0, maps[k][p] / (hi[k] || 1));
      r += v * cols[k][0]; g += v * cols[k][1]; b += v * cols[k][2];
    }
    d[4 * p] = 255 * Math.min(1, r); d[4 * p + 1] = 255 * Math.min(1, g); d[4 * p + 2] = 255 * Math.min(1, b); d[4 * p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

/** Pixel index under a pointer event on a canvas showing a w × h image (−1 outside). */
export function canvasPixel(canvas: HTMLCanvasElement, ev: { clientX: number; clientY: number }, w: number, h: number): number {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor((ev.clientX - r.left) / r.width * w), y = Math.floor((ev.clientY - r.top) / r.height * h);
  return x >= 0 && y >= 0 && x < w && y < h ? y * w + x : -1;
}

/** Slice of a channel-major stack: map k. */
export const plane = (S: Float32Array, k: number, P: number): Float32Array => S.subarray(k * P, (k + 1) * P);

/** Percentile of finite values (q in 0..1). */
export function quantile(v: ArrayLike<number>, q: number): number {
  const a = Array.from(v).filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))] : 0;
}

/** Number formatting with a real minus sign. */
export const fmt = (v: number, d = 2): string => (Number.isFinite(v) ? v.toFixed(d).replace(/^-/, "−") : v > 0 ? "∞" : "—");

/** Shared CSS of the day's demos (figures, small canvases). */
export const COMMON_CSS = `
.qb-demo .um-fig canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; touch-action: none; }
.qb-demo .um-fig figcaption { font-size: 15px; }
.qb-demo .um-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo .um-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo .um-readout { font-size: 14px; white-space: pre; }
.reveal .qb-demo .um-fig figcaption { font-size: 18px; }
.reveal .qb-demo .um-readout { font-size: 18px; }
.reveal .qb-demo .qb-note { display: none; }
`;
