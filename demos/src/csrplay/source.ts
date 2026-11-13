// Shared pattern source for the spatial statistics demos (csrplay, gfj, ripleyk, envelopetest):
// the same models, windows, seeds and presets in every unit, a pattern view (SVG, µm
// coordinates) and the generator controls. No demo-specific logic here.
import { htmlEl, svgEl, slider, button, Slider, svgPoint } from "../lib/ui";
import { rng } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { viridis } from "../lib/colour";

export const SIDE = 200;   // the default window is SIDE × SIDE µm

export type Model = "csr" | "thomas" | "hardcore" | "lattice" | "inhom" | "colonies" | "pairs";
export type WinKind = "square" | "small" | "band" | "tissue";
export type Artefact = "none" | "merge" | "split";

export const MODEL_LABELS: Record<Model, string> = {
  csr: "random (CSR)", thomas: "clusters (Thomas)", hardcore: "hard core", lattice: "lattice",
  inhom: "gradient", colonies: "colonies + hard core", pairs: "tight pairs",
};
export const WIN_LABELS: Record<WinKind, string> = { square: "200 × 200 µm", small: "60 × 60 µm", band: "tissue band", tissue: "tissue mask" };

export interface Spec {
  model: Model;
  /** points expected in a 200 × 200 µm window (scaled by the window area) */
  n: number;
  /** clusters per 200 × 200 µm (Thomas, colonies) */
  k: number;
  /** mean points per cluster */
  mu: number;
  /** Thomas offspring spread σ (µm) */
  sigma: number;
  /** colony radius (µm) */
  R: number;
  /** hard-core distance (µm) */
  h: number;
  /** lattice spacing and jitter (µm) */
  spacing: number; jitter: number;
  /** gradient strength 0..0.95: ρ(x) ∝ 1 + grad·(2x/w − 1) */
  grad: number;
  /** distance within tight pairs (µm) */
  pairD: number;
  artefact: Artefact;
  /** merge distance (µm) / split fraction */
  mergeD: number; splitFrac: number;
  win: WinKind;
  seed: number;
}

export const DEFAULT_SPEC: Spec = {
  model: "csr", n: 150, k: 18, mu: 8, sigma: 4, R: 10, h: 6, spacing: 16, jitter: 2, grad: 0.8, pairD: 3,
  artefact: "none", mergeD: 8, splitFrac: 0.3, win: "square", seed: 1,
};

// ---------------------------------------------------------------------------------------
// Windows (cached; masks are fixed shapes, seeded)

const winCache = new Map<WinKind, PP.Window>();
export function makeWindow(kind: WinKind): PP.Window {
  let w = winCache.get(kind);
  if (w) return w;
  if (kind === "square") w = PP.rectWindow(SIDE, SIDE);
  else if (kind === "small") w = PP.rectWindow(60, 60);
  else if (kind === "band") w = PP.maskWindow(PP.bandMask(SIDE, SIDE, 30, 0.22), SIDE, SIDE);
  else w = PP.maskWindow(PP.blobMask(rng(7), SIDE, SIDE, { radius: 88, wobble: 0.16 }), SIDE, SIDE);
  winCache.set(kind, w);
  return w;
}

/** ρ(x) of the gradient model, normalised to the expected count in the window. */
export function gradientRho(s: Spec, win: PP.Window): (x: number, y: number) => number {
  const rho0 = s.n / (SIDE * SIDE);
  return (x: number) => rho0 * (1 + s.grad * (2 * x / win.w - 1));
}

/** The pattern of a spec (deterministic for a seed). marks: cluster index where meaningful. */
export function makePattern(s: Spec): PP.Points {
  const win = makeWindow(s.win), r = rng(1000 * s.seed + 17 + Object.keys(MODEL_LABELS).indexOf(s.model));
  const scale = win.area / (SIDE * SIDE), n = Math.max(2, Math.round(s.n * scale)), kappa = s.k / (SIDE * SIDE);
  let p: PP.Points;
  switch (s.model) {
    case "csr": p = PP.binomialProcess(r, n, win); break;
    case "thomas": p = PP.thomas(r, kappa, s.mu, s.sigma, win); break;
    case "hardcore": p = PP.ssi(r, n, s.h, win); break;
    case "lattice": p = PP.jitteredLattice(r, s.spacing, s.jitter, win); break;
    case "inhom": p = PP.inhomBinomial(r, n, gradientRho(s, win), (s.n / (SIDE * SIDE)) * (1 + s.grad), win); break;
    case "colonies": p = PP.colonies(r, kappa, s.mu, s.R, s.h, win); break;
    case "pairs": p = PP.tightPairs(r, Math.round(n / 2), s.pairD, win); break;
  }
  if (s.artefact === "merge") p = PP.mergeClose(p, s.mergeD);
  else if (s.artefact === "split") p = PP.splitSome(rng(77 + s.seed), p, s.splitFrac, 3, win);
  return p;
}

/** A CSR pattern with the same number of points in the same window (for envelopes). */
export const csrLike = (p: PP.Points, win: PP.Window, r: ReturnType<typeof rng>): PP.Points => PP.binomialProcess(r, p.x.length, win);

/** One-line description of a pattern. */
export function describe(s: Spec, p: PP.Points): string {
  const win = makeWindow(s.win);
  return `${MODEL_LABELS[s.model]}${s.artefact !== "none" ? " · " + (s.artefact === "merge" ? "merged" : "split") : ""}, n = ${p.x.length}, window ${WIN_LABELS[s.win]} (${Math.round(win.area)} µm²)`;
}

/** A scalar map (nx × ny, row-major, NaN = transparent) as a PNG data URL in viridis. */
export function heatUrl(values: ArrayLike<number>, nx: number, ny: number, lo: number, hi: number, alpha = 255): string {
  const c = document.createElement("canvas");
  c.width = nx; c.height = ny;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(nx, ny);
  for (let i = 0; i < nx * ny; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const [r, g, b] = viridis((v - lo) / (hi - lo || 1));
    id.data[4 * i] = 255 * r; id.data[4 * i + 1] = 255 * g; id.data[4 * i + 2] = 255 * b; id.data[4 * i + 3] = alpha;
  }
  ctx.putImageData(id, 0, 0);
  return c.toDataURL();
}

// ---------------------------------------------------------------------------------------
// Pattern view

const maskUrlCache = new Map<PP.Window, string>();
function maskUrl(win: PP.Window): string {
  let u = maskUrlCache.get(win);
  if (u) return u;
  const c = document.createElement("canvas");
  c.width = win.w; c.height = win.h;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(win.w, win.h);
  for (let i = 0; i < win.w * win.h; i++) {
    const inn = win.mask![i] === 1;
    id.data[4 * i] = inn ? 214 : 120; id.data[4 * i + 1] = inn ? 222 : 120; id.data[4 * i + 2] = inn ? 232 : 120; id.data[4 * i + 3] = inn ? 140 : 70;
  }
  ctx.putImageData(id, 0, 0);
  u = c.toDataURL();
  maskUrlCache.set(win, u);
  return u;
}

/** An SVG view of a window (in µm) with layers: window, under (maps), mid (overlays below the points), points, overlay. */
export class PatternView {
  readonly svg: SVGSVGElement;
  readonly under: SVGGElement;
  readonly back: SVGGElement;
  readonly mid: SVGGElement;
  readonly dots: SVGGElement;
  readonly over: SVGGElement;
  win: PP.Window = makeWindow("square");
  dotR = 2.2;
  private pad = 6;

  constructor(parent: Element, cls = "ss-view") {
    this.svg = svgEl("svg", { class: cls, role: "img", "aria-label": "point pattern" }, parent);
    this.back = svgEl("g", {}, this.svg);
    this.under = svgEl("g", {}, this.svg);
    this.mid = svgEl("g", {}, this.svg);
    this.dots = svgEl("g", {}, this.svg);
    this.over = svgEl("g", {}, this.svg);
    this.setWindow(this.win);
  }

  setWindow(win: PP.Window): void {
    this.win = win;
    const p = this.pad * Math.max(win.w, win.h) / SIDE;
    this.svg.setAttribute("viewBox", `${-p} ${-p} ${win.w + 2 * p} ${win.h + 2 * p}`);
    this.dotR = 2.2 * Math.max(win.w, win.h) / SIDE;
    this.back.replaceChildren();
    if (win.mask) {
      svgEl("image", { x: 0, y: 0, width: win.w, height: win.h, href: maskUrl(win), preserveAspectRatio: "none" }, this.back);
      svgEl("rect", { x: 0, y: 0, width: win.w, height: win.h, fill: "none", stroke: "var(--qb-grid)", "stroke-width": 0.8 * this.dotR / 2.2 }, this.back);
    } else {
      svgEl("rect", { x: 0, y: 0, width: win.w, height: win.h, fill: "var(--qb-panel)", stroke: "var(--qb-axis)", "stroke-width": 0.8 * this.dotR / 2.2 }, this.back);
    }
  }

  /** Draw the points; colours per point (default series colour 1). */
  setPoints(p: PP.Points, colors?: (i: number) => string, r = this.dotR): void {
    this.dots.replaceChildren();
    for (let i = 0; i < p.x.length; i++)
      svgEl("circle", { cx: p.x[i].toFixed(2), cy: p.y[i].toFixed(2), r, fill: colors ? colors(i) : "var(--qb-c1)", stroke: "var(--qb-bg)", "stroke-width": r * 0.25 }, this.dots);
  }

  /** Pointer position in µm. */
  toWorld(ev: { clientX: number; clientY: number }): [number, number] { return svgPoint(this.svg, ev); }

  /** Index of the point nearest to (x, y) within `tol` µm, or −1. */
  static nearest(p: PP.Points, x: number, y: number, tol: number): number {
    let best = tol * tol, bi = -1;
    for (let i = 0; i < p.x.length; i++) { const q = (p.x[i] - x) ** 2 + (p.y[i] - y) ** 2; if (q < best) { best = q; bi = i; } }
    return bi;
  }
}

/** A <g> with text at µm coordinates (font size in µm units of the view). */
export function viewText(g: SVGGElement, x: number, y: number, s: string, size: number, fill = "var(--qb-fg)", anchor = "middle"): SVGTextElement {
  const t = svgEl("text", { x: x.toFixed(2), y: y.toFixed(2), "text-anchor": anchor, "font-size": size.toFixed(2), fill, "paint-order": "stroke", stroke: "var(--qb-bg)", "stroke-width": (size * 0.18).toFixed(2) }, g);
  t.textContent = s;
  return t;
}

// ---------------------------------------------------------------------------------------
// Generator controls

type NumKey = "n" | "k" | "mu" | "sigma" | "R" | "h" | "spacing" | "jitter" | "grad" | "pairD" | "mergeD" | "splitFrac";
const SLIDERS: { key: NumKey; label: string; min: number; max: number; step: number }[] = [
  { key: "n", label: "points", min: 20, max: 400, step: 10 },
  { key: "k", label: "clusters", min: 2, max: 60, step: 1 },
  { key: "mu", label: "per cluster", min: 2, max: 30, step: 1 },
  { key: "sigma", label: "spread σ µm", min: 1, max: 20, step: 0.5 },
  { key: "R", label: "colony R µm", min: 5, max: 30, step: 1 },
  { key: "h", label: "hard core µm", min: 1, max: 14, step: 0.5 },
  { key: "spacing", label: "spacing µm", min: 8, max: 40, step: 1 },
  { key: "jitter", label: "jitter µm", min: 0, max: 12, step: 0.5 },
  { key: "grad", label: "gradient", min: 0, max: 0.95, step: 0.05 },
  { key: "pairD", label: "pair distance", min: 0.5, max: 10, step: 0.5 },
  { key: "mergeD", label: "merge below µm", min: 2, max: 16, step: 1 },
  { key: "splitFrac", label: "split fraction", min: 0, max: 1, step: 0.05 },
];
const USES: Record<Model, NumKey[]> = {
  csr: ["n"], thomas: ["k", "mu", "sigma"], hardcore: ["n", "h"], lattice: ["spacing", "jitter"],
  inhom: ["n", "grad"], colonies: ["k", "mu", "R", "h"], pairs: ["n", "pairD"],
};

export interface SourceUI { sync(): void; el: HTMLElement }

/** Model buttons, the sliders of the current model, artefact buttons and seed / reseed.
 *  `opts.models` limits the model list; `opts.windows` shows window buttons; `opts.artefacts` shows the artefact row. */
export function sourceControls(parent: Element, s: Spec, onChange: () => void,
  opts: { models?: Model[]; windows?: WinKind[]; artefacts?: boolean; compact?: boolean } = {}): SourceUI {
  const el = htmlEl("div", "ss-source", "", parent);
  const models = opts.models ?? (Object.keys(MODEL_LABELS) as Model[]);
  const sel = htmlEl("select", "ss-select", "", el);
  for (const m of models) { const o = htmlEl("option", "", MODEL_LABELS[m], sel); o.value = m; }
  sel.addEventListener("change", () => { s.model = sel.value as Model; sel.blur(); sync(); onChange(); });
  let winSel: HTMLSelectElement | null = null;
  if (opts.windows && opts.windows.length > 1) {
    winSel = htmlEl("select", "ss-select", "", el);
    for (const w of opts.windows) { const o = htmlEl("option", "", "window: " + WIN_LABELS[w], winSel); o.value = w; }
    winSel.addEventListener("change", () => { s.win = winSel!.value as WinKind; winSel!.blur(); onChange(); });
  }
  let artSel: HTMLSelectElement | null = null;
  if (opts.artefacts) {
    artSel = htmlEl("select", "ss-select", "", el);
    for (const [v, l] of [["none", "detection: correct"], ["merge", "detection: close spots merge"], ["split", "detection: some objects split"]]) { const o = htmlEl("option", "", l, artSel); o.value = v; }
    artSel.addEventListener("change", () => { s.artefact = artSel!.value as Artefact; artSel!.blur(); sync(); onChange(); });
  }
  const sl = {} as Record<NumKey, Slider>;
  for (const d of SLIDERS) sl[d.key] = slider(el, d.label, d.min, d.max, d.step, s[d.key], v => { s[d.key] = v; onChange(); });
  const seedRow = htmlEl("div", "ss-seedrow", "", el);
  const seedOut = htmlEl("span", "ss-seed", "", seedRow);
  button(seedRow, "new seed", () => { s.seed++; sync(); onChange(); });
  function sync(): void {
    sel.value = s.model;
    if (winSel) winSel.value = s.win;
    if (artSel) artSel.value = s.artefact;
    for (const d of SLIDERS) {
      sl[d.key].set(s[d.key]);
      const show = USES[s.model].includes(d.key) || (d.key === "mergeD" && s.artefact === "merge") || (d.key === "splitFrac" && s.artefact === "split");
      sl[d.key].input.closest("label")!.style.display = show ? "" : "none";
    }
    seedOut.textContent = "seed " + s.seed;
  }
  sync();
  return { sync, el };
}

// ---------------------------------------------------------------------------------------
// Styles

export function injectStyle(name: string, css: string): void {
  const id = "qb-style-" + name;
  if (document.getElementById(id)) return;
  const st = document.createElement("style");
  st.id = id;
  st.textContent = css;
  document.head.appendChild(st);
}

/** CSS shared by the source controls of every demo (prefixed per demo). */
export function sourceCss(name: string): string {
  const p = `.qb-demo[data-demo="${name}"]`;
  return `
${p} .ss-view { display: block; width: 100%; height: auto; touch-action: none; }
${p} .ss-source { display: flex; flex-direction: column; gap: .25em; }
${p} .ss-select { max-width: 100%; font: inherit; font-size: .9em; padding: .1em .3em; border-radius: 5px; border: 1px solid #bbb; background: var(--qb-bg); color: var(--qb-fg); }
${p} .ss-seedrow { display: flex; gap: .5em; align-items: center; }
${p} .ss-seed { font-family: ui-monospace, monospace; font-size: .85em; color: var(--qb-muted); }
${p} .ss-tabs { display: flex; flex-wrap: wrap; gap: .25em; }
${p} .ss-lab { font-size: .85em; color: var(--qb-muted); }
.reveal ${p} { font-size: 18px; gap: .5em; flex-wrap: nowrap; }
.reveal ${p} .qb-btn { font-size: 17px; padding: .05em .35em; }
.reveal ${p} .qb-slider { gap: .3em; }
.reveal ${p} .qb-readout { font-size: 18px; line-height: 1.2; padding: .25em .45em; }
.reveal ${p} .qb-presets { gap: .2em; }
.reveal ${p} .ss-select { font-size: 18px; }
.reveal ${p} .ss-lab { font-size: 18px; }
`;
}
