// Small SVG plotting for the course demos (owner: morphology day; never in ui.ts).
// Readable at slide size (18 px labels in viewBox units, thick lines) and light/dark safe:
// every colour defaults to a CSS variable of demos/demo.css (--qb-fg, --qb-axis, --qb-grid,
// --qb-c1..), set through `style` so the variables resolve.
//
//   const p = new Plot(host, { x: [0, 10], y: [0, 1], xlabel: "iteration", ylabel: "energy" });
//   const curve = p.line(xs, ys, { color: seriesColor(0), label: "E" });
//   curve.set(xs2, ys2);          // update in place (no re-creation)
//   p.setDomain([0, 20]);         // axes and every mark are redrawn
//   p.legend();                   // from the marks' labels
//
// Also: step lines, scatter, bars, histograms, heat maps (with numbers in cells), covariance
// ellipses, vertical/horizontal markers, arrows, text, a colour bar and a polar plot.
// Data coordinates: x right, y up (as in a graph). For images use heatmap() with flipY.
import { svgEl, htmlEl, svgPoint } from "./ui";
import { ColorMap, grey, colormapLUT, textColorOn, finiteRange, RGB } from "./colour";

// ---------------------------------------------------------------------------------------
// Scales and ticks

/** A map from data to SVG units with an inverse. */
export interface Scale {
  (v: number): number;
  invert(px: number): number;
  domain: [number, number];
  range: [number, number];
  log: boolean;
}

/** Linear (or base-10 log) scale from domain [d0, d1] to range [r0, r1]. */
export function makeScale(domain: [number, number], range: [number, number], log = false): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const t = log ? Math.log10 : (v: number) => v;
  const it = log ? (v: number) => 10 ** v : (v: number) => v;
  const a = t(d0), b = t(d1), k = b !== a ? (r1 - r0) / (b - a) : 0;
  const f = ((v: number) => r0 + (t(v) - a) * k) as Scale;
  f.invert = (px: number) => it(a + (px - r0) / (k || 1));
  f.domain = [d0, d1]; f.range = [r0, r1]; f.log = log;
  return f;
}

/** About `count` "nice" tick values (steps of 1, 2 or 5 times a power of ten) in [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) return Number.isFinite(lo) ? [lo] : [];
  // A range below float resolution relative to the values cannot be stepped through.
  if (hi - lo <= 1e-12 * Math.max(Math.abs(lo), Math.abs(hi))) return [lo];
  const raw = (hi - lo) / Math.max(1, count), p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const step = (m >= 7.5 ? 10 : m >= 3.5 ? 5 : m >= 1.5 ? 2 : 1) * p;
  if (!(step > 0) || !Number.isFinite(step)) return [lo, hi];
  const out: number[] = [];
  const first = Math.ceil(lo / step - 1e-9);
  for (let k = 0; k < 1000; k++) {   // hard cap: never loop forever
    const v = (first + k) * step;
    if (v > hi + step * 1e-9) break;
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

/** Powers of ten in [lo, hi] (lo > 0), for log axes. */
export function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e);
  return out;
}

/** Format a tick: as few decimals as the spacing needs; powers of ten as 10^k on log axes. */
export function formatTick(v: number, ticks: number[], log = false): string {
  if (log) {
    const e = Math.round(Math.log10(v));
    return Math.abs(e) <= 3 ? String(10 ** e >= 1 ? 10 ** e : Number((10 ** e).toFixed(-e))) : `10${superscript(e)}`;
  }
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v) || 1;
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(d).replace(/^-0(\.0*)?$/, "0").replace("-", "−");
}

function superscript(n: number): string {
  const s = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  return (n < 0 ? "⁻" : "") + String(Math.abs(n)).split("").map(c => s[Number(c)]).join("");
}

/** [min, max] over several arrays (finite values only), padded by `pad` of the range. */
export function extent(arrays: ArrayLike<number>[], pad = 0.05): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) { const [l, h] = finiteRange(a); if (a.length) { lo = Math.min(lo, l); hi = Math.max(hi, h); } }
  if (!(hi >= lo)) return [0, 1];
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  const d = (hi - lo) * pad;
  return [lo - d, hi + d];
}

/** Colour of data series i: the demo.css variables --qb-c1..--qb-c5, then more fixed colours. */
export function seriesColor(i: number): string {
  const fb = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];
  const k = ((i % fb.length) + fb.length) % fb.length;
  return k < 5 ? `var(--qb-c${k + 1}, ${fb[k]})` : fb[k];
}

// ---------------------------------------------------------------------------------------
// Styles and marks

/** Style of a mark. Colours may be any CSS colour, including var(--qb-...). */
export interface MarkStyle {
  color?: string;        // stroke (lines) or fill (dots, bars); default series colour 0
  fill?: string;         // fill for areas, bars, ellipses ("none" for none)
  fillOpacity?: number;
  width?: number;        // stroke width in SVG units (default 3 for lines)
  dash?: string;         // stroke-dasharray, e.g. "6 4"
  opacity?: number;
  r?: number;            // dot radius (scatter), default 6
  label?: string;        // shown by legend()
  cls?: string;          // extra CSS class on the mark's <g>
}

/** A drawn element that can be updated in place with new data, or removed. */
export interface Mark<A extends unknown[]> {
  g: SVGGElement;
  style: MarkStyle;
  /** Replace the data and redraw. */
  set(...args: A): void;
  /** Redraw with the current data (after a domain change). */
  render(): void;
  remove(): void;
  /** Legend symbol for this mark. */
  kind: "line" | "dot" | "box";
}

function css(el: Element, props: Record<string, string | number | undefined>): void {
  const s = (el as SVGElement).style;
  for (const [k, v] of Object.entries(props)) if (v !== undefined) s.setProperty(k, String(v));
}

/** Common base: a set of marks drawn in a clipped data layer. */
abstract class MarkHost {
  readonly svg: SVGSVGElement;
  /** Clipped layer for data marks. */
  readonly layer: SVGGElement;
  /** Unclipped layer above the data, for annotations. */
  readonly over: SVGGElement;
  protected marks: Mark<any>[] = [];
  readonly font: number;

  protected constructor(svg: SVGSVGElement, font: number) {
    this.svg = svg; this.font = font;
    this.layer = svgEl("g", {}, svg);
    this.over = svgEl("g", {}, svg);
  }

  /** Add a custom mark: `draw(g, ...args)` fills the group from scratch on every render. */
  mark<A extends unknown[]>(draw: (g: SVGGElement, ...args: A) => void, args: A, style: MarkStyle = {}, kind: Mark<A>["kind"] = "line", parent?: SVGGElement): Mark<A> {
    const g = svgEl("g", style.cls ? { class: style.cls } : {}, parent ?? this.layer);
    let cur = args;
    const m: Mark<A> = {
      g, style, kind,
      set: (...a: A) => { cur = a; m.render(); },
      render: () => { g.replaceChildren(); draw(g, ...cur); },
      remove: () => { g.remove(); this.marks = this.marks.filter(x => x !== m); },
    };
    this.marks.push(m);
    m.render();
    return m;
  }

  /** Remove every mark. */
  clear(): void { for (const m of [...this.marks]) m.remove(); }

  /** Redraw every mark. */
  renderAll(): void { for (const m of this.marks) m.render(); }

  /** Legend from the marks that have a label (or from explicit items). Corner: tr, tl, br, bl. */
  legend(items?: { label: string; color: string; kind?: "line" | "dot" | "box"; dash?: string }[], corner: "tr" | "tl" | "br" | "bl" = "tr"): SVGGElement {
    this.svg.querySelector(":scope > g.qb-legend")?.remove();
    const its = items ?? this.marks.filter(m => m.style.label).map(m => ({ label: m.style.label!, color: m.style.color ?? seriesColor(0), kind: m.kind, dash: m.style.dash }));
    const g = svgEl("g", { class: "qb-legend" }, this.svg);
    const fs = this.font * 0.9, lh = fs * 1.3, sw = 28;
    const w = sw + 10 + Math.max(0, ...its.map(i => i.label.length)) * fs * 0.55 + 10, h = its.length * lh + 8;
    const box = this.legendBox();
    const x0 = corner[1] === "r" ? box.r - w - 6 : box.l + 6, y0 = corner[0] === "t" ? box.t + 6 : box.b - h - 6;
    const bg = svgEl("rect", { x: x0, y: y0, width: w, height: h, rx: 5 }, g);
    css(bg, { fill: "var(--qb-bg, #fff)", "fill-opacity": 0.85, stroke: "var(--qb-grid, #ddd)" });
    its.forEach((it, i) => {
      const y = y0 + 4 + lh * (i + 0.5);
      if (it.kind === "dot") css(svgEl("circle", { cx: x0 + 8 + sw / 2, cy: y, r: 6 }, g), { fill: it.color });
      else if (it.kind === "box") css(svgEl("rect", { x: x0 + 8, y: y - 7, width: sw, height: 14 }, g), { fill: it.color, "fill-opacity": 0.75 });
      else css(svgEl("line", { x1: x0 + 8, x2: x0 + 8 + sw, y1: y, y2: y }, g), { stroke: it.color, "stroke-width": 3.5, "stroke-dasharray": it.dash });
      const t = svgEl("text", { x: x0 + sw + 16, y: y + fs * 0.35 }, g);
      t.textContent = it.label;
      css(t, { "font-size": fs + "px" });
    });
    return g;
  }

  protected abstract legendBox(): { l: number; r: number; t: number; b: number };
}

// ---------------------------------------------------------------------------------------
// Cartesian plot

export interface PlotOptions {
  /** viewBox size in SVG units (the SVG is drawn at 100% of its container's width). */
  width?: number;
  height?: number;
  x?: [number, number];
  y?: [number, number];
  xlog?: boolean;
  ylog?: boolean;
  xlabel?: string;
  ylabel?: string;
  title?: string;
  /** Margins in SVG units (defaults leave room for 18-px tick labels). */
  margin?: { l?: number; r?: number; t?: number; b?: number };
  /** Background grid lines at the ticks (default true). */
  grid?: boolean;
  /** Explicit tick values, or an approximate tick count (default 5). */
  xticks?: number[] | number;
  yticks?: number[] | number;
  xfmt?: (v: number) => string;
  yfmt?: (v: number) => string;
  /** Label font size in SVG units (default 18). */
  font?: number;
  /** Same data-units-per-pixel on both axes (shrinks one domain's range to fit). */
  equal?: boolean;
  /** Accessible description of the figure. */
  ariaLabel?: string;
  /** CSS class on the <svg> (default "qb-plot-svg"). */
  cls?: string;
}

/** Clip-path ids must be unique on the page, and every demo bundle has its own copy of this
 *  module, so the counter lives on globalThis (shared by all bundles). */
function nextClipId(prefix: string): string {
  const g = globalThis as unknown as { __qbClipCounter?: number };
  g.__qbClipCounter = (g.__qbClipCounter ?? 0) + 1;
  return prefix + g.__qbClipCounter;
}

/** A 2D plot with axes, ticks and labels; data marks are added with the methods below. */
export class Plot extends MarkHost {
  x!: Scale;
  y!: Scale;
  readonly opts: PlotOptions;
  readonly W: number;
  readonly H: number;
  /** Plot area in SVG units. */
  box!: { l: number; r: number; t: number; b: number };
  private axes: SVGGElement;
  private clipRect: SVGRectElement;

  constructor(parent: Element, opts: PlotOptions = {}) {
    const W = opts.width ?? 560, H = opts.height ?? 360, font = opts.font ?? 18;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", class: opts.cls ?? "qb-plot-svg", role: "img", "aria-label": opts.ariaLabel ?? opts.title ?? "plot" }, parent);
    const axes = svgEl("g", { class: "qb-axes" }, svg);
    super(svg, font);
    this.axes = axes;
    this.opts = opts; this.W = W; this.H = H;
    const id = nextClipId("qb-clip-");
    const defs = svgEl("defs", {}, svg);
    const cp = svgEl("clipPath", { id }, defs);
    this.clipRect = svgEl("rect", {}, cp);
    this.layer.setAttribute("clip-path", `url(#${id})`);
    svg.insertBefore(defs, svg.firstChild);
    this.setDomain(opts.x ?? [0, 1], opts.y ?? [0, 1]);
  }

  protected legendBox() { return this.box; }

  /** Change the x and/or y domain; axes and all marks are redrawn. */
  setDomain(x?: [number, number], y?: [number, number]): this {
    const o = this.opts, f = this.font;
    const m = { l: o.margin?.l ?? (o.ylabel ? 2.0 * f + 44 : 56), r: o.margin?.r ?? 14, t: o.margin?.t ?? (o.title ? 1.8 * f + 6 : 12), b: o.margin?.b ?? (o.xlabel ? 2.9 * f + 8 : 1.7 * f + 12) };
    let box = { l: m.l, r: this.W - m.r, t: m.t, b: this.H - m.b };
    const xd = x ?? this.x.domain, yd = y ?? this.y.domain;
    if (o.equal) {
      const kx = (box.r - box.l) / (xd[1] - xd[0]), ky = (box.b - box.t) / (yd[1] - yd[0]), k = Math.min(kx, ky);
      const cw = (xd[1] - xd[0]) * k, ch = (yd[1] - yd[0]) * k;
      box = { l: box.l + (box.r - box.l - cw) / 2, r: box.l + (box.r - box.l + cw) / 2, t: box.t + (box.b - box.t - ch) / 2, b: box.t + (box.b - box.t + ch) / 2 };
    }
    this.box = box;
    this.x = makeScale(xd, [box.l, box.r], !!o.xlog);
    this.y = makeScale(yd, [box.b, box.t], !!o.ylog);
    setAttrsN(this.clipRect, { x: box.l, y: box.t, width: box.r - box.l, height: box.b - box.t });
    this.drawAxes();
    this.renderAll();
    return this;
  }

  /** Data coordinates of a pointer event (works with reveal.js scaling). */
  toData(ev: { clientX: number; clientY: number }): [number, number] {
    const [px, py] = svgPoint(this.svg, ev);
    return [this.x.invert(px), this.y.invert(py)];
  }

  private ticks(axis: "x" | "y"): number[] {
    const spec = axis === "x" ? this.opts.xticks : this.opts.yticks, s = axis === "x" ? this.x : this.y;
    const [lo, hi] = [Math.min(...s.domain), Math.max(...s.domain)];
    if (Array.isArray(spec)) return spec.filter(v => v >= lo - 1e-9 * (hi - lo) && v <= hi + 1e-9 * (hi - lo));
    // no more ticks than fit: about 3.2 label widths per x tick, 1.8 line heights per y tick
    const room = axis === "x" ? (this.box.r - this.box.l) / (this.font * 3.2) : (this.box.b - this.box.t) / (this.font * 1.8);
    return s.log ? logTicks(lo, hi) : niceTicks(lo, hi, Math.max(2, Math.min(spec ?? (axis === "x" ? 6 : 5), Math.floor(room))));
  }

  private drawAxes(): void {
    const g = this.axes, o = this.opts, f = this.font, b = this.box;
    g.replaceChildren();
    const xt = this.ticks("x"), yt = this.ticks("y");
    const fx = o.xfmt ?? ((v: number) => formatTick(v, xt, this.x.log)), fy = o.yfmt ?? ((v: number) => formatTick(v, yt, this.y.log));
    if (o.grid !== false) {
      for (const v of xt) css(svgEl("line", { x1: this.x(v), x2: this.x(v), y1: b.t, y2: b.b }, g), { stroke: "var(--qb-grid, #e3e3e3)", "stroke-width": 1 });
      for (const v of yt) css(svgEl("line", { y1: this.y(v), y2: this.y(v), x1: b.l, x2: b.r }, g), { stroke: "var(--qb-grid, #e3e3e3)", "stroke-width": 1 });
    }
    const ax = svgEl("path", { d: `M${b.l},${b.t}V${b.b}H${b.r}`, fill: "none" }, g);
    css(ax, { stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 });
    for (const v of xt) {
      css(svgEl("line", { x1: this.x(v), x2: this.x(v), y1: b.b, y2: b.b + 6 }, g), { stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 });
      const t = svgEl("text", { x: this.x(v), y: b.b + 8 + f * 0.85, "text-anchor": "middle" }, g);
      t.textContent = fx(v); css(t, { "font-size": f + "px" });
    }
    for (const v of yt) {
      css(svgEl("line", { x1: b.l - 6, x2: b.l, y1: this.y(v), y2: this.y(v) }, g), { stroke: "var(--qb-axis, #444)", "stroke-width": 1.5 });
      const t = svgEl("text", { x: b.l - 9, y: this.y(v) + f * 0.35, "text-anchor": "end" }, g);
      t.textContent = fy(v); css(t, { "font-size": f + "px" });
    }
    if (o.xlabel) {
      const t = svgEl("text", { x: (b.l + b.r) / 2, y: b.b + 10 + 2.0 * f, "text-anchor": "middle" }, g);
      t.textContent = o.xlabel; css(t, { "font-size": f + "px" });
    }
    if (o.ylabel) {
      const cy = (b.t + b.b) / 2, cx = f * 0.95;
      const t = svgEl("text", { x: cx, y: cy, "text-anchor": "middle", transform: `rotate(-90 ${cx} ${cy})` }, g);
      t.textContent = o.ylabel; css(t, { "font-size": f + "px" });
    }
    if (o.title) {
      const t = svgEl("text", { x: (b.l + b.r) / 2, y: f * 1.2, "text-anchor": "middle", "font-weight": 600 }, g);
      t.textContent = o.title; css(t, { "font-size": f + "px" });
    }
  }

  /** Change the axis labels / title and redraw the axes. */
  setLabels(l: { xlabel?: string; ylabel?: string; title?: string }): void {
    Object.assign(this.opts, l);
    this.setDomain();
  }

  // ---- marks ----

  private path(pts: [number, number][]): string {
    let d = "", pen = false;
    for (const [px, py] of pts) {
      if (!Number.isFinite(px) || !Number.isFinite(py)) { pen = false; continue; }
      d += (pen ? "L" : "M") + px.toFixed(2) + "," + py.toFixed(2);
      pen = true;
    }
    return d;
  }

  private strokeStyle(el: Element, s: MarkStyle, w = 3): void {
    css(el, { fill: "none", stroke: s.color ?? seriesColor(0), "stroke-width": s.width ?? w, "stroke-dasharray": s.dash, opacity: s.opacity, "stroke-linejoin": "round", "stroke-linecap": "round" });
  }

  /** Polyline through (xs[i], ys[i]); non-finite values break the line. */
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, style: MarkStyle = {}): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, X, Y) => {
      const pts: [number, number][] = [];
      for (let i = 0; i < Math.min(X.length, Y.length); i++) pts.push([this.x(X[i]), this.y(Y[i])]);
      this.strokeStyle(svgEl("path", { d: this.path(pts) }, g), style);
    }, [xs, ys], style, "line");
  }

  /** Line of a function sampled at n points across the current x domain. */
  fn(f: (x: number) => number, style: MarkStyle = {}, n = 200): Mark<[(x: number) => number]> {
    return this.mark((g, F) => {
      const [a, b] = this.x.domain, pts: [number, number][] = [];
      for (let i = 0; i < n; i++) {
        const xv = this.x.log ? a * (b / a) ** (i / (n - 1)) : a + (b - a) * i / (n - 1);
        pts.push([this.x(xv), this.y(F(xv))]);
      }
      this.strokeStyle(svgEl("path", { d: this.path(pts) }, g), style);
    }, [f], style, "line");
  }

  /** Step line: value ys[i] holds from xs[i] to xs[i+1] ("post"), or is centred on xs[i] ("mid"). */
  step(xs: ArrayLike<number>, ys: ArrayLike<number>, style: MarkStyle = {}, where: "post" | "mid" = "post"): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, X, Y) => {
      const n = Math.min(X.length, Y.length), pts: [number, number][] = [];
      for (let i = 0; i < n; i++) {
        let a: number, b: number;
        if (where === "post") { a = X[i]; b = i + 1 < X.length ? X[i + 1] : X[i] + (n > 1 ? X[i] - X[i - 1] : 1); }
        else {
          const dl = i > 0 ? (X[i] - X[i - 1]) / 2 : (n > 1 ? (X[1] - X[0]) / 2 : 0.5);
          const dr = i + 1 < n ? (X[i + 1] - X[i]) / 2 : dl;
          a = X[i] - dl; b = X[i] + dr;
        }
        pts.push([this.x(a), this.y(Y[i])], [this.x(b), this.y(Y[i])]);
      }
      this.strokeStyle(svgEl("path", { d: this.path(pts) }, g), style);
    }, [xs, ys], style, "line");
  }

  /** Filled band between y0s and y1s (e.g. ±1 SD, or under a curve with y0s = 0s). */
  band(xs: ArrayLike<number>, y0s: ArrayLike<number>, y1s: ArrayLike<number>, style: MarkStyle = {}): Mark<[ArrayLike<number>, ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, X, A, B) => {
      if (X.length === 0) return; // empty band (used to hide it): draw nothing
      const top: string[] = [], bot: string[] = [];
      for (let i = 0; i < X.length; i++) {
        top.push(`${this.x(X[i]).toFixed(2)},${this.y(B[i]).toFixed(2)}`);
        bot.push(`${this.x(X[i]).toFixed(2)},${this.y(A[i]).toFixed(2)}`);
      }
      const p = svgEl("path", { d: "M" + top.join("L") + "L" + bot.reverse().join("L") + "Z" }, g);
      css(p, { fill: style.fill ?? style.color ?? seriesColor(0), "fill-opacity": style.fillOpacity ?? 0.2, stroke: "none" });
    }, [xs, y0s, y1s], style, "box");
  }

  /** Dots at (xs[i], ys[i]); `colors[i]` overrides the colour per point. */
  scatter(xs: ArrayLike<number>, ys: ArrayLike<number>, style: MarkStyle & { colors?: string[]; stroke?: string } = {}): Mark<[ArrayLike<number>, ArrayLike<number>, string[]?]> {
    return this.mark((g, X, Y, C) => {
      const cols = C ?? style.colors;
      for (let i = 0; i < Math.min(X.length, Y.length); i++) {
        if (!Number.isFinite(X[i]) || !Number.isFinite(Y[i])) continue;
        const c = svgEl("circle", { cx: this.x(X[i]).toFixed(2), cy: this.y(Y[i]).toFixed(2), r: style.r ?? 6 }, g);
        css(c, { fill: cols?.[i] ?? style.color ?? seriesColor(0), "fill-opacity": style.fillOpacity, opacity: style.opacity, stroke: style.stroke ?? "var(--qb-bg, #fff)", "stroke-width": 1.2 });
      }
    }, [xs, ys, undefined], style, "dot");
  }

  /** Vertical bars at xs with heights hs (from `base`); width in data units (default: 80% of the spacing). */
  bars(xs: ArrayLike<number>, hs: ArrayLike<number>, style: MarkStyle & { barWidth?: number; base?: number; colors?: string[] } = {}): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, X, Hh) => {
      const bw = style.barWidth ?? (X.length > 1 ? 0.8 * Math.abs(X[1] - X[0]) : 0.8);
      const base = style.base ?? (this.y.log ? Math.min(...this.y.domain) : 0);
      for (let i = 0; i < Math.min(X.length, Hh.length); i++) {
        const x0 = this.x(X[i] - bw / 2), x1 = this.x(X[i] + bw / 2), y0 = this.y(base), y1 = this.y(Hh[i]);
        const r = svgEl("rect", { x: Math.min(x0, x1).toFixed(2), y: Math.min(y0, y1).toFixed(2), width: Math.abs(x1 - x0).toFixed(2), height: Math.abs(y1 - y0).toFixed(2) }, g);
        css(r, { fill: style.colors?.[i] ?? style.fill ?? style.color ?? seriesColor(0), "fill-opacity": style.fillOpacity ?? 0.85, opacity: style.opacity });
      }
    }, [xs, hs], style, "box");
  }

  /** Histogram: counts[i] over [edges[i], edges[i+1]] as touching bars with an outline. */
  histogram(edges: ArrayLike<number>, counts: ArrayLike<number>, style: MarkStyle = {}): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, E, C) => {
      let d = "";
      const y0 = this.y(this.y.log ? Math.max(this.y.domain[0], 1e-12) : Math.max(0, this.y.domain[0]));
      for (let i = 0; i < C.length && i + 1 < E.length; i++) {
        const a = this.x(E[i]), b = this.x(E[i + 1]), h = this.y(Math.max(C[i], this.y.log ? 1e-12 : 0));
        d += `M${a.toFixed(2)},${y0.toFixed(2)}V${h.toFixed(2)}H${b.toFixed(2)}V${y0.toFixed(2)}Z`;
      }
      const p = svgEl("path", { d }, g);
      css(p, { fill: style.fill ?? style.color ?? seriesColor(0), "fill-opacity": style.fillOpacity ?? 0.55, stroke: style.color ?? seriesColor(0), "stroke-width": style.width ?? 1, opacity: style.opacity });
    }, [edges, counts], style, "box");
  }

  /**
   * Heat map of an nx x ny grid (values[row * nx + col]) over the data rectangle
   * [x0,x1] x [y0,y1]. Row 0 is at the top when flipY (image convention, default) and at
   * the bottom otherwise. Small grids are drawn as rectangles (optionally with the numbers
   * in the cells); grids with more than `maxCells` cells as one pixelated image.
   */
  heatmap(values: ArrayLike<number>, nx: number, ny: number, opts: HeatmapOptions = {}): Mark<[ArrayLike<number>]> {
    return this.mark((g, V) => {
      const x0 = opts.x0 ?? this.x.domain[0], x1 = opts.x1 ?? this.x.domain[1];
      const y0 = opts.y0 ?? this.y.domain[0], y1 = opts.y1 ?? this.y.domain[1];
      const px = (c: number) => this.x(x0 + (x1 - x0) * c / nx);
      const py = (r: number) => this.y(opts.flipY === false ? y0 + (y1 - y0) * r / ny : y1 - (y1 - y0) * r / ny);
      drawCells(g, V, nx, ny, px, py, opts, this.font);
    }, [values], { cls: opts.cls }, "box");
  }

  /** Ellipse of a 2D Gaussian: mean, 2x2 covariance [[sxx, sxy], [sxy, syy]], drawn at `nsd` SDs. */
  ellipse(mean: [number, number], cov: [[number, number], [number, number]], nsd = 2, style: MarkStyle = {}): Mark<[[number, number], [[number, number], [number, number]]]> {
    return this.mark((g, mu, S) => {
      const pts: [number, number][] = [];
      for (const [px, py] of ellipsePoints(mu, S, nsd, 72)) pts.push([this.x(px), this.y(py)]);
      const p = svgEl("path", { d: this.path(pts) + "Z" }, g);
      css(p, { fill: style.fill ?? style.color ?? seriesColor(0), "fill-opacity": style.fillOpacity ?? 0.1, stroke: style.color ?? seriesColor(0), "stroke-width": style.width ?? 2.5, "stroke-dasharray": style.dash, opacity: style.opacity });
    }, [mean, cov], style, "line");
  }

  /** Vertical marker line at x (with an optional label at the top). */
  vline(x: number, style: MarkStyle & { text?: string } = {}): Mark<[number]> {
    return this.mark((g, X) => {
      const px = this.x(X);
      if (!Number.isFinite(px)) return; // NaN / off a log axis: hidden
      this.strokeStyle(svgEl("line", { x1: px, x2: px, y1: this.box.t, y2: this.box.b }, g), { dash: "6 4", width: 2.5, color: "var(--qb-fg, #222)", ...style });
      if (style.text) {
        const t = svgEl("text", { x: px + 5, y: this.box.t + this.font, class: "qb-halo" }, g);
        t.textContent = style.text;
        css(t, { "font-size": this.font + "px", fill: style.color ?? "var(--qb-fg, #222)" });
      }
    }, [x], style, "line");
  }

  /** Horizontal marker line at y (with an optional label at the right end). */
  hline(y: number, style: MarkStyle & { text?: string } = {}): Mark<[number]> {
    return this.mark((g, Y) => {
      const py = this.y(Y);
      if (!Number.isFinite(py)) return; // NaN / off a log axis: hidden
      this.strokeStyle(svgEl("line", { x1: this.box.l, x2: this.box.r, y1: py, y2: py }, g), { dash: "6 4", width: 2.5, color: "var(--qb-fg, #222)", ...style });
      if (style.text) {
        const t = svgEl("text", { x: this.box.r - 4, y: py - 6, "text-anchor": "end", class: "qb-halo" }, g);
        t.textContent = style.text;
        css(t, { "font-size": this.font + "px", fill: style.color ?? "var(--qb-fg, #222)" });
      }
    }, [y], style, "line");
  }

  /** Arrow from (x0, y0) to (x1, y1) in data coordinates. */
  arrow(x0: number, y0: number, x1: number, y1: number, style: MarkStyle = {}): Mark<[number, number, number, number]> {
    return this.mark((g, a, b, c, d) => drawArrow(g, this.x(a), this.y(b), this.x(c), this.y(d), style), [x0, y0, x1, y1], style, "line");
  }

  /** Text at (x, y) in data coordinates. */
  text(x: number, y: number, str: string, style: MarkStyle & { anchor?: "start" | "middle" | "end"; size?: number } = {}): Mark<[number, number, string]> {
    return this.mark((g, X, Y, S) => {
      const t = svgEl("text", { x: this.x(X), y: this.y(Y), "text-anchor": style.anchor ?? "start", class: "qb-halo" }, g);
      t.textContent = S;
      css(t, { "font-size": (style.size ?? this.font) + "px", fill: style.color ?? "var(--qb-fg, #222)" });
    }, [x, y, str], style, "line", this.over);
  }
}

function setAttrsN(e: Element, a: Record<string, number>): void { for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v)); }

/** Points on the `nsd`-SD ellipse of a 2D Gaussian (closed-form 2x2 eigen-decomposition). */
export function ellipsePoints(mean: [number, number], cov: [[number, number], [number, number]], nsd = 2, n = 64): [number, number][] {
  const a = cov[0][0], b = cov[0][1], d = cov[1][1];
  const tr = (a + d) / 2, disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
  const l1 = Math.max(0, tr + disc), l2 = Math.max(0, tr - disc);
  const th = 0.5 * Math.atan2(2 * b, a - d);
  const c = Math.cos(th), s = Math.sin(th), r1 = nsd * Math.sqrt(l1), r2 = nsd * Math.sqrt(l2);
  return Array.from({ length: n }, (_, i) => {
    const t = 2 * Math.PI * i / n, u = r1 * Math.cos(t), v = r2 * Math.sin(t);
    return [mean[0] + c * u - s * v, mean[1] + s * u + c * v] as [number, number];
  });
}

function drawArrow(g: SVGGElement, x0: number, y0: number, x1: number, y1: number, style: MarkStyle): void {
  const col = style.color ?? "var(--qb-fg, #222)", w = style.width ?? 3, L = Math.hypot(x1 - x0, y1 - y0);
  if (L < 1e-6) return;
  const ux = (x1 - x0) / L, uy = (y1 - y0) / L, hl = Math.min(L, 6 + 3 * w), hw = 3 + 1.8 * w;
  const bx = x1 - ux * hl, by = y1 - uy * hl;
  css(svgEl("line", { x1: x0, y1: y0, x2: bx, y2: by }, g), { stroke: col, "stroke-width": w, "stroke-dasharray": style.dash, opacity: style.opacity });
  css(svgEl("polygon", { points: `${x1},${y1} ${bx - uy * hw / 2},${by + ux * hw / 2} ${bx + uy * hw / 2},${by - ux * hw / 2}` }, g), { fill: col, opacity: style.opacity });
}

// ---------------------------------------------------------------------------------------
// Heat maps

export interface HeatmapOptions {
  map?: ColorMap;
  lo?: number;
  hi?: number;
  /** Write the value into each cell (rectangle mode only). */
  numbers?: boolean;
  fmt?: (v: number) => string;
  /** Data rectangle covered by the grid (Plot.heatmap; default: the whole domain). */
  x0?: number; x1?: number; y0?: number; y1?: number;
  /** Row 0 at the top (default true). */
  flipY?: boolean;
  /** Above this many cells, draw one image instead of rectangles (default 4096). */
  maxCells?: number;
  /** Thin lines between cells (rectangle mode). */
  gridLines?: boolean;
  /** Per-cell override of the text (e.g. "∞"); undefined = use fmt. */
  text?: (v: number, i: number) => string | undefined;
  cls?: string;
}

function drawCells(g: SVGGElement, V: ArrayLike<number>, nx: number, ny: number, px: (c: number) => number, py: (r: number) => number, o: HeatmapOptions, font: number): void {
  const map = o.map ?? grey;
  let lo = o.lo, hi = o.hi;
  if (lo === undefined || hi === undefined) { const r = finiteRange(V); lo ??= r[0]; hi ??= r[1]; }
  const t = (v: number) => (hi! > lo! ? (v - lo!) / (hi! - lo!) : 0.5);
  if (nx * ny > (o.maxCells ?? 4096)) {
    const c = document.createElement("canvas");
    c.width = nx; c.height = ny;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(nx, ny), lut = colormapLUT(map, 256);
    for (let r = 0; r < ny; r++) for (let q = 0; q < nx; q++) {
      const v = V[r * nx + q], k = Number.isFinite(v) ? Math.max(0, Math.min(255, Math.round(255 * t(v)))) : -1;
      const p = 4 * (r * nx + q);
      if (k < 0) { id.data[p + 3] = 0; continue; }
      id.data[p] = lut[4 * k]; id.data[p + 1] = lut[4 * k + 1]; id.data[p + 2] = lut[4 * k + 2]; id.data[p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const xa = px(0), xb = px(nx), ya = py(0), yb = py(ny);
    const img = svgEl("image", { x: Math.min(xa, xb), y: Math.min(ya, yb), width: Math.abs(xb - xa), height: Math.abs(yb - ya), preserveAspectRatio: "none", href: c.toDataURL() }, g);
    css(img, { "image-rendering": "pixelated" });
    if (ya > yb) img.setAttribute("transform", `scale(1,-1) translate(0,${-(ya + yb)})`);
    return;
  }
  const fmt = o.fmt ?? ((v: number) => (Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) >= 10 ? 1 : 2)));
  for (let r = 0; r < ny; r++) for (let q = 0; q < nx; q++) {
    const v = V[r * nx + q];
    const xa = px(q), xb = px(q + 1), ya = py(r), yb = py(r + 1);
    const col: RGB = Number.isFinite(v) ? map(t(v)) : [0.5, 0.5, 0.5];
    const rect = svgEl("rect", { x: Math.min(xa, xb).toFixed(2), y: Math.min(ya, yb).toFixed(2), width: (Math.abs(xb - xa) + (o.gridLines ? 0 : 0.5)).toFixed(2), height: (Math.abs(yb - ya) + (o.gridLines ? 0 : 0.5)).toFixed(2) }, g);
    css(rect, { fill: `rgb(${col.map(c => Math.round(255 * c)).join(",")})`, stroke: o.gridLines ? "var(--qb-bg, #fff)" : undefined, "stroke-width": o.gridLines ? 1 : undefined });
    if (o.numbers) {
      const s = o.text?.(v, r * nx + q) ?? (Number.isFinite(v) ? fmt(v) : "–");
      const cw = Math.abs(xb - xa), ch = Math.abs(yb - ya);
      const fs = Math.min(font, ch * 0.6, cw * 1.6 / Math.max(1, s.length));
      const tx = svgEl("text", { x: ((xa + xb) / 2).toFixed(2), y: ((ya + yb) / 2 + fs * 0.35).toFixed(2), "text-anchor": "middle" }, g);
      tx.textContent = s;
      css(tx, { "font-size": fs.toFixed(1) + "px", fill: textColorOn(col) });
    }
  }
}

export interface HeatmapGridOptions extends HeatmapOptions {
  /** Cell size in SVG units (default 44). */
  cell?: number;
  rowLabels?: string[];
  colLabels?: string[];
  title?: string;
  font?: number;
  ariaLabel?: string;
}

/**
 * A standalone heat map of a rows x cols matrix (values[r * cols + c]) with optional row and
 * column labels: confusion matrices, Gram matrices, accumulator tables, DP tables.
 * Returns the SVG and a `set` to update the values.
 */
export function heatmapGrid(parent: Element, values: ArrayLike<number>, rows: number, cols: number, opts: HeatmapGridOptions = {}): { svg: SVGSVGElement; set(v: ArrayLike<number>): void } {
  const cell = opts.cell ?? 44, f = opts.font ?? 18;
  const lw = opts.rowLabels ? Math.max(...opts.rowLabels.map(s => s.length)) * f * 0.6 + 10 : 4;
  const th = (opts.title ? f * 1.5 : 0) + (opts.colLabels ? f * 1.4 : 4);
  const W = lw + cols * cell + 4, H = th + rows * cell + 4;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", class: "qb-plot-svg", role: "img", "aria-label": opts.ariaLabel ?? opts.title ?? "heat map" }, parent);
  svg.style.maxWidth = `${Math.round(W * 1.3)}px`;
  if (opts.title) { const t = svgEl("text", { x: lw + cols * cell / 2, y: f * 1.1, "text-anchor": "middle", "font-weight": 600 }, svg); t.textContent = opts.title; css(t, { "font-size": f + "px" }); }
  opts.colLabels?.forEach((s, c) => { const t = svgEl("text", { x: lw + (c + 0.5) * cell, y: th - 6, "text-anchor": "middle" }, svg); t.textContent = s; css(t, { "font-size": f * 0.9 + "px" }); });
  opts.rowLabels?.forEach((s, r) => { const t = svgEl("text", { x: lw - 6, y: th + (r + 0.5) * cell + f * 0.32, "text-anchor": "end" }, svg); t.textContent = s; css(t, { "font-size": f * 0.9 + "px" }); });
  const g = svgEl("g", {}, svg);
  const set = (v: ArrayLike<number>) => { g.replaceChildren(); drawCells(g, v, cols, rows, c => lw + c * cell, r => th + r * cell, { numbers: true, gridLines: true, ...opts }, f); };
  set(values);
  return { svg, set };
}

/** A colour bar for a map from lo to hi, with ticks. Horizontal by default. */
export function colorbar(parent: Element, map: ColorMap, lo: number, hi: number, opts: { label?: string; width?: number; height?: number; vertical?: boolean; ticks?: number[]; font?: number; fmt?: (v: number) => string } = {}): SVGSVGElement {
  const f = opts.font ?? 18, vert = !!opts.vertical;
  const W = opts.width ?? (vert ? 90 : 400), H = opts.height ?? (vert ? 300 : 30 + 2 * f + (opts.label ? f * 1.2 : 0));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", class: "qb-plot-svg", role: "img", "aria-label": `colour bar ${opts.label ?? ""}` }, parent);
  const id = nextClipId("qb-cb-");
  const grad = svgEl("linearGradient", { id, x1: 0, y1: vert ? 1 : 0, x2: vert ? 0 : 1, y2: 0 }, svgEl("defs", {}, svg));
  for (let i = 0; i <= 16; i++) {
    const c = map(i / 16);
    svgEl("stop", { offset: i / 16, "stop-color": `rgb(${c.map(v => Math.round(255 * v)).join(",")})` }, grad);
  }
  const ticks = opts.ticks ?? niceTicks(Math.min(lo, hi), Math.max(lo, hi), 4);
  const fmt = opts.fmt ?? ((v: number) => formatTick(v, ticks));
  if (!vert) {
    const y0 = opts.label ? f * 1.3 : 2, bw = W - 2 * f, x0 = f;
    if (opts.label) { const t = svgEl("text", { x: W / 2, y: f, "text-anchor": "middle" }, svg); t.textContent = opts.label; css(t, { "font-size": f + "px" }); }
    css(svgEl("rect", { x: x0, y: y0, width: bw, height: 22, fill: `url(#${id})` }, svg), { stroke: "var(--qb-axis, #444)", "stroke-width": 1 });
    for (const v of ticks) {
      const x = x0 + bw * (v - lo) / (hi - lo || 1);
      css(svgEl("line", { x1: x, x2: x, y1: y0 + 22, y2: y0 + 28 }, svg), { stroke: "var(--qb-axis, #444)" });
      const t = svgEl("text", { x, y: y0 + 28 + f, "text-anchor": "middle" }, svg); t.textContent = fmt(v); css(t, { "font-size": f + "px" });
    }
  } else {
    const x0 = 4, bh = H - 2 * f, y0 = f;
    css(svgEl("rect", { x: x0, y: y0, width: 22, height: bh, fill: `url(#${id})` }, svg), { stroke: "var(--qb-axis, #444)", "stroke-width": 1 });
    for (const v of ticks) {
      const y = y0 + bh * (1 - (v - lo) / (hi - lo || 1));
      css(svgEl("line", { x1: x0 + 22, x2: x0 + 28, y1: y, y2: y }, svg), { stroke: "var(--qb-axis, #444)" });
      const t = svgEl("text", { x: x0 + 32, y: y + f * 0.35 }, svg); t.textContent = fmt(v); css(t, { "font-size": f + "px" });
    }
  }
  return svg;
}

// ---------------------------------------------------------------------------------------
// Polar plot

export interface PolarOptions {
  /** viewBox size (square), default 420. */
  size?: number;
  /** Radius of the outer ring in data units (default 1). */
  rmax?: number;
  /** Ring radii (default: nice ticks up to rmax). */
  rings?: number[];
  /** Spoke spacing in degrees (default 30). */
  spokeStep?: number;
  /** Show only the upper half, angles 0..180° (Hough/Radon angle range). */
  half?: boolean;
  font?: number;
  title?: string;
  ariaLabel?: string;
}

/**
 * Polar plot. Angles in radians, counter-clockwise from the +x axis as seen on screen (the
 * course convention); r >= 0 from the centre. Marks: line, scatter, ray, plus the generic
 * `mark`. Data -> SVG with `pos(theta, r)`.
 */
export class PolarPlot extends MarkHost {
  rmax: number;
  readonly opts: PolarOptions;
  readonly cx: number;
  readonly cy: number;
  readonly R: number;
  private grid: SVGGElement;

  constructor(parent: Element, opts: PolarOptions = {}) {
    const S = opts.size ?? 420, font = opts.font ?? 18, half = !!opts.half;
    const H = half ? S / 2 + 2.2 * font + (opts.title ? font * 1.5 : 0) : S;
    const svg = svgEl("svg", { viewBox: `0 0 ${S} ${H}`, width: "100%", class: "qb-plot-svg", role: "img", "aria-label": opts.ariaLabel ?? opts.title ?? "polar plot" }, parent);
    const grid = svgEl("g", { class: "qb-axes" }, svg);
    super(svg, font);
    this.grid = grid;
    this.opts = opts; this.rmax = opts.rmax ?? 1;
    const tt = opts.title ? font * 1.4 : 0;
    this.R = S / 2 - 1.9 * font - tt / 2;
    this.cx = S / 2;
    this.cy = half ? H - 1.4 * font : S / 2 + tt / 2;
    this.drawGrid();
  }

  protected legendBox() { const r = this.R; return { l: this.cx - r - this.font, r: this.cx + r + this.font, t: this.cy - r - this.font, b: this.opts.half ? this.cy : this.cy + r }; }

  /** SVG position of polar data (theta radians, CCW on screen). */
  pos(theta: number, r: number): [number, number] {
    const k = this.R * r / this.rmax;
    return [this.cx + k * Math.cos(theta), this.cy - k * Math.sin(theta)];
  }

  /** Change the outer radius; grid and marks are redrawn. */
  setRmax(rmax: number): void { this.rmax = rmax; this.drawGrid(); this.renderAll(); }

  private drawGrid(): void {
    const g = this.grid, o = this.opts, f = this.font, half = !!o.half;
    g.replaceChildren();
    const rings = o.rings ?? niceTicks(0, this.rmax, 4).filter(v => v > 0);
    for (const r of rings) {
      const k = this.R * r / this.rmax;
      const el = half ? svgEl("path", { d: `M${this.cx - k},${this.cy}A${k},${k} 0 0 1 ${this.cx + k},${this.cy}`, fill: "none" }, g) : svgEl("circle", { cx: this.cx, cy: this.cy, r: k, fill: "none" }, g);
      css(el, { stroke: "var(--qb-grid, #e3e3e3)", "stroke-width": 1.2 });
      const la = (o.spokeStep ?? 30) / 2 * Math.PI / 180, [tx, ty] = this.pos(la, r);
      const t = svgEl("text", { x: tx + 3, y: ty - 3 }, g); t.textContent = formatTick(r, rings.length > 1 ? rings : [0, r]); css(t, { "font-size": f * 0.8 + "px", fill: "var(--qb-muted, #777)" });
    }
    const step = o.spokeStep ?? 30;
    for (let a = 0; a < (half ? 181 : 360); a += step) {
      const [x, y] = this.pos(a * Math.PI / 180, this.rmax);
      css(svgEl("line", { x1: this.cx, y1: this.cy, x2: x, y2: y }, g), { stroke: "var(--qb-grid, #e3e3e3)", "stroke-width": 1.2 });
      const [lx, ly] = this.pos(a * Math.PI / 180, this.rmax * (1 + 1.0 * f / this.R));
      const t = svgEl("text", { x: lx, y: ly + f * 0.35, "text-anchor": "middle" }, g); t.textContent = `${a}°`; css(t, { "font-size": f * 0.85 + "px" });
    }
    if (o.title) { const t = svgEl("text", { x: this.cx, y: f * 1.1, "text-anchor": "middle", "font-weight": 600 }, g); t.textContent = o.title; css(t, { "font-size": f + "px" }); }
  }

  /** Curve r(theta) through the points (thetas[i], rs[i]). */
  line(thetas: ArrayLike<number>, rs: ArrayLike<number>, style: MarkStyle & { closed?: boolean } = {}): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, T, Rr) => {
      let d = "";
      for (let i = 0; i < Math.min(T.length, Rr.length); i++) { const [x, y] = this.pos(T[i], Rr[i]); d += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2); }
      if (style.closed) d += "Z";
      css(svgEl("path", { d }, g), { fill: style.fill ?? "none", "fill-opacity": style.fillOpacity ?? 0.15, stroke: style.color ?? seriesColor(0), "stroke-width": style.width ?? 3, "stroke-dasharray": style.dash, "stroke-linejoin": "round" });
    }, [thetas, rs], style, "line");
  }

  /** Dots at polar positions. */
  scatter(thetas: ArrayLike<number>, rs: ArrayLike<number>, style: MarkStyle & { colors?: string[] } = {}): Mark<[ArrayLike<number>, ArrayLike<number>]> {
    return this.mark((g, T, Rr) => {
      for (let i = 0; i < Math.min(T.length, Rr.length); i++) {
        const [x, y] = this.pos(T[i], Rr[i]);
        css(svgEl("circle", { cx: x, cy: y, r: style.r ?? 6 }, g), { fill: style.colors?.[i] ?? style.color ?? seriesColor(0), stroke: "var(--qb-bg, #fff)", "stroke-width": 1.2 });
      }
    }, [thetas, rs], style, "dot");
  }

  /** Arrow from the centre at angle theta with length r (default rmax). */
  ray(theta: number, r?: number, style: MarkStyle = {}): Mark<[number, number | undefined]> {
    return this.mark((g, T, Rr) => {
      const [x, y] = this.pos(T, Rr ?? this.rmax);
      drawArrow(g, this.cx, this.cy, x, y, { color: seriesColor(1), ...style });
    }, [theta, r], style, "line");
  }
}

// ---------------------------------------------------------------------------------------

/** A titled container (figure + caption) to put a plot in, styled by demo.css. */
export function plotFigure(parent: Element, caption = "", cls = "qb-plot"): HTMLElement {
  const f = htmlEl("figure", cls + " qb-figure", "", parent);
  if (caption) htmlEl("figcaption", "", caption, f);
  return f;
}
