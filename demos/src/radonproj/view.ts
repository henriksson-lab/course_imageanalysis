// Shared pieces of the Radon-day demos (radonproj, fourierslice, backproj, tomosampling,
// iterrecon): test objects, image and sinogram panels on top of lib/plot.ts, styles.
// Panels are Plots in world coordinates: x right, y UP, origin at the image centre, so a
// pointer event maps straight to (x, y) and rays are lines x cos(theta) + y sin(theta) = rho.
import { htmlEl } from "../lib/ui";
import { Plot, plotFigure, Mark } from "../lib/plot";
import { ColorMap, grey } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import * as R from "../lib/radon";

export { CELL_CREDIT };

export function injectStyle(name: string, css: string): void {
  const id = "qb-style-" + name;
  if (document.getElementById(id)) return;
  const s = document.createElement("style");
  s.id = id;
  s.textContent = css;
  document.head.appendChild(s);
}

/** Common CSS for the Radon demos (prefix = the demo's data-demo selector). */
export function baseCss(name: string): string {
  const d = `[data-demo="${name}"]`;
  return `
${d} { flex-direction: column; gap: .5em; align-items: stretch; }
${d} .rd-figs { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; width: 100%; }
${d} .rd-ctls { display: flex; flex-wrap: wrap; gap: .5em 1.2em; align-items: flex-start; width: 100%; }
${d} .rd-col { flex: 1 1 240px; min-width: 0; display: flex; flex-direction: column; gap: .35em; }
${d} .rd-sub { display: flex; flex-direction: column; gap: .35em; }
${d} .rd-fig { flex: 1 1 230px; min-width: 180px; max-width: 330px; margin: 0; }
${d} .rd-fig figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin: 0 0 .1em; }
${d} .rd-fig.wide { flex: 1.3 1 300px; max-width: 440px; }
${d} .qb-readout { font-size: 14px; }
${d} .rd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${d} .rd-credit { font-size: 12px; color: var(--qb-muted); }
${d} .rd-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
${d} .rd-click { cursor: crosshair; touch-action: none; }
.reveal ${d} { font-size: 17px; gap: .2em; }
.reveal ${d} .rd-figs { flex-wrap: nowrap; }
.reveal ${d} .rd-fig { max-width: 300px; }
.reveal ${d} .rd-fig.wide { max-width: 400px; }
.reveal ${d} .rd-col, .reveal ${d} .rd-sub { gap: .15em; }
.reveal ${d} .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal ${d} .qb-note { display: none; }
.reveal ${d} .qb-readout { font-size: 14px; padding: .25em .5em; }
`;
}

/** Figures row on top, a row of control columns below. */
export function layout(host: HTMLElement, cols = 3): { figs: HTMLElement; cols: HTMLElement[] } {
  const figs = htmlEl("div", "rd-figs", "", host);
  const ctls = htmlEl("div", "rd-ctls", "", host);
  return { figs, cols: Array.from({ length: cols }, () => htmlEl("div", "rd-col", "", ctls)) };
}

// ---------------------------------------------------------------------------------------
// Test objects

export type ObjectName = "points" | "point" | "disc" | "bigdisc" | "square" | "twodiscs" | "shepp" | "cell";
export const OBJECT_LABELS: Record<ObjectName, string> = {
  points: "points only", point: "one point", disc: "disc", bigdisc: "large disc", square: "square", twodiscs: "two discs", shepp: "Shepp–Logan", cell: "micrograph",
};

let cellCache: Float64Array | null = null;
let cellPromise: Promise<Float64Array> | null = null;

/** The course micrograph at 128 x 128, values 0..1 (dark = high attenuation), inside the field of view. */
export function cellObject(n = 128): Promise<Float64Array> {
  if (cellCache && n === 128) return Promise.resolve(cellCache);
  if (!cellPromise) cellPromise = loadCell(128).then(g => {
    const m = R.fovMask(128), out = new Float64Array(128 * 128);
    // the micrograph is dark where the specimen absorbs: use 1 - grey as "attenuation"
    for (let i = 0; i < out.length; i++) out[i] = (1 - g.data[i] / 255) * m[i];
    cellCache = out;
    return out;
  });
  return cellPromise;
}

/** Ellipse description of an analytic object (for exact sinograms), or null. */
export function objectEllipses(name: ObjectName, n: number): R.WEllipse[] | null {
  const k = n / 128;
  switch (name) {
    case "disc": return [R.disc(10 * k, 6 * k, 30 * k, 1)];
    case "bigdisc": return [R.disc(0, 0, 60 * k, 1)];
    case "twodiscs": return [R.disc(-26 * k, 14 * k, 16 * k, 1), R.disc(22 * k, -18 * k, 22 * k, 0.6)];
    case "shepp": return R.sheppLoganEllipses(n);
    default: return null;
  }
}

/** The object as an n x n image (synchronous; "cell" needs cellObject() resolved first). */
export function objectImage(name: ObjectName, n: number): Float64Array {
  const el = objectEllipses(name, n);
  if (el) return R.ellipseImage(n, el, 3);
  if (name === "square") return R.squareImage(n, 22 * n / 128, -8 * n / 128, 4 * n / 128, 1);
  if (name === "cell") return Float64Array.from(cellCache ?? new Float64Array(n * n));
  const img = new Float64Array(n * n);
  if (name === "point") R.addPoint(img, n, 30 * n / 128, 20 * n / 128, 1);
  return img;
}

/** Shift an image by (tx, ty) world units (bilinear, zero outside). */
export function shiftImage(img: ArrayLike<number>, n: number, tx: number, ty: number): Float64Array {
  const out = new Float64Array(n * n);
  if (tx === 0 && ty === 0) { out.set(img); return out; }
  // output pixel (col, row) reads input at (col - tx, row + ty)
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
    const x = q - tx, y = r + ty, i0 = Math.floor(x), j0 = Math.floor(y), fx = x - i0, fy = y - j0;
    const at = (i: number, j: number) => (i >= 0 && i < n && j >= 0 && j < n ? img[j * n + i] : 0);
    out[r * n + q] = (1 - fy) * ((1 - fx) * at(i0, j0) + fx * at(i0 + 1, j0)) + fy * ((1 - fx) * at(i0, j0 + 1) + fx * at(i0 + 1, j0 + 1));
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Panels

export interface ImagePanel {
  fig: HTMLElement;
  cap: HTMLElement;
  plot: Plot;
  img: Mark<[ArrayLike<number>]>;
  /** Replace the image (n x n, row-major) and its display range. */
  show(v: ArrayLike<number>, lo?: number, hi?: number, map?: ColorMap): void;
}

/** An n x n image in world coordinates [-n/2, n/2]^2 (y up), with room `pad` around it for drawing. */
export function imagePanel(parent: Element, caption: string, n: number, pad = 0, cls = "rd-fig"): ImagePanel {
  const fig = plotFigure(parent, caption, cls);
  const cap = fig.querySelector("figcaption") as HTMLElement;
  const h = n / 2 + pad;
  const plot = new Plot(fig, { width: 320, height: 320, x: [-h, h], y: [-h, h], equal: true, grid: false, xticks: [], yticks: [], margin: { l: 4, r: 4, t: 4, b: 4 } });
  plot.svg.querySelector(".qb-axes")?.setAttribute("display", "none");
  let cur = { lo: 0 as number | undefined, hi: 1 as number | undefined, map: grey as ColorMap };
  const img = plot.mark((g, V) => {
    g.replaceChildren();
    drawHeat(plot, g, V, n, n, -n / 2, n / 2, -n / 2, n / 2, cur.lo, cur.hi, cur.map);
  }, [new Float64Array(n * n) as ArrayLike<number>]);
  return {
    fig, cap, plot, img,
    show(v, lo, hi, map = grey) { cur = { lo, hi, map }; img.set(v); },
  };
}

/** Draw a heat map (row 0 at the top) into a mark group: one pixelated <image>. */
export function drawHeat(plot: Plot, g: SVGGElement, V: ArrayLike<number>, nx: number, ny: number,
  x0: number, x1: number, y0: number, y1: number, lo?: number, hi?: number, map: ColorMap = grey): void {
  if (lo === undefined || hi === undefined) {
    let a = Infinity, b = -Infinity;
    for (let i = 0; i < V.length; i++) { const v = V[i]; if (Number.isFinite(v)) { if (v < a) a = v; if (v > b) b = v; } }
    lo ??= a; hi ??= b;
  }
  const c = document.createElement("canvas");
  c.width = nx; c.height = ny;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(nx, ny);
  const lut: number[][] = [];
  for (let k = 0; k < 256; k++) lut.push(map(k / 255).map(v => Math.round(255 * v)));
  const sc = hi! > lo! ? 255 / (hi! - lo!) : 0;
  for (let i = 0; i < nx * ny; i++) {
    const v = V[i];
    if (!Number.isFinite(v)) { id.data[4 * i + 3] = 0; continue; }
    const k = Math.max(0, Math.min(255, Math.round((v - lo!) * sc))), col = lut[k];
    id.data[4 * i] = col[0]; id.data[4 * i + 1] = col[1]; id.data[4 * i + 2] = col[2]; id.data[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  const xa = plot.x(x0), xb = plot.x(x1), ya = plot.y(y1), yb = plot.y(y0);
  const im = document.createElementNS("http://www.w3.org/2000/svg", "image");
  im.setAttribute("x", String(Math.min(xa, xb))); im.setAttribute("y", String(Math.min(ya, yb)));
  im.setAttribute("width", String(Math.abs(xb - xa))); im.setAttribute("height", String(Math.abs(yb - ya)));
  im.setAttribute("preserveAspectRatio", "none");
  im.setAttribute("href", c.toDataURL());
  im.style.imageRendering = "pixelated";
  g.appendChild(im);
}

/** Sinogram as a display array: theta across (column = angle index), rho up (row 0 = largest rho). */
export function sinoDisplay(sino: ArrayLike<number>, nAngles: number, nBins: number, shown = nAngles): Float64Array {
  const V = new Float64Array(nAngles * nBins);
  for (let a = 0; a < nAngles; a++) for (let b = 0; b < nBins; b++)
    V[(nBins - 1 - b) * nAngles + a] = a < shown ? sino[a * nBins + b] : NaN;
  return V;
}

export interface SinoPanel {
  fig: HTMLElement;
  plot: Plot;
  /** Show a sinogram with angles from th0 to th1 degrees (column centres spread evenly). */
  show(sino: ArrayLike<number>, nAngles: number, nBins: number, th0: number, th1: number, shown?: number, lo?: number, hi?: number, map?: ColorMap): void;
}

/** A sinogram panel: theta (degrees) across, rho up. */
export function sinoPanel(parent: Element, caption: string, rhoMax: number, thetaRange: [number, number] = [0, 180], cls = "rd-fig wide", height = 290): SinoPanel {
  const fig = plotFigure(parent, caption, cls);
  const plot = new Plot(fig, { width: 420, height, x: thetaRange, y: [-rhoMax, rhoMax], grid: false, xlabel: "θ (degrees)", ylabel: "ρ", xticks: tick45(thetaRange) });
  let args: [ArrayLike<number>, number, number, number, number, number | undefined, number | undefined, number | undefined, ColorMap] = [new Float64Array(1), 1, 1, 0, 180, 0, 0, 1, grey];
  const m = plot.mark(g => {
    const [s, na, nb, t0, t1, shown, lo, hi, map] = args;
    const step = na > 1 ? (t1 - t0) / na : 1;
    drawHeat(plot, g, sinoDisplay(s, na, nb, shown ?? na), na, nb, t0, t0 + step * na, -nb / 2, nb / 2, lo, hi, map);
  }, []);
  return {
    fig, plot,
    show(sino, na, nb, t0, t1, shown, lo, hi, map = grey) {
      args = [sino, na, nb, t0, t1, shown, lo, hi, map];
      m.render();
    },
  };
}

export function tick45(r: [number, number]): number[] {
  const out: number[] = [];
  const step = r[1] - r[0] > 200 ? 90 : r[1] - r[0] > 100 ? 45 : 30;
  for (let t = Math.ceil(r[0] / step) * step; t <= r[1] + 1e-9; t += step) out.push(t);
  return out;
}

/** A small text block under a panel. */
export function note(parent: Element, text: string, cls = "qb-note"): HTMLElement {
  return htmlEl("div", cls, text, parent);
}

/** Pointer handling on a plot: down/move/up in data coordinates. */
export function onPointer(plot: Plot, h: { down?: (x: number, y: number, ev: PointerEvent) => void; move?: (x: number, y: number) => void; up?: () => void }): void {
  const svg = plot.svg;
  svg.classList.add("rd-click");
  let active = false;
  svg.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    active = true;
    svg.setPointerCapture(ev.pointerId);
    const [x, y] = plot.toData(ev);
    h.down?.(x, y, ev);
  });
  svg.addEventListener("pointermove", ev => { if (!active) return; const [x, y] = plot.toData(ev); h.move?.(x, y); });
  const end = () => { if (active) { active = false; h.up?.(); } };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
}

/** A segment of the line x cos(th) + y sin(th) = rho, from t = t0 to t1 along (-sin, cos). */
export function lineSeg(th: number, rho: number, t0: number, t1: number): [number[], number[]] {
  const c = Math.cos(th), s = Math.sin(th);
  return [[rho * c - t0 * s, rho * c - t1 * s], [rho * s + t0 * c, rho * s + t1 * c]];
}

export const fmt = (v: number, d = 2): string => (Number.isFinite(v) ? v.toFixed(d) : "–");
