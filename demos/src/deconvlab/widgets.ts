// Small DOM helpers shared by the deconvolution day's demos (deconvlab, regpath, rlsteps,
// denoisers): a drop-down that gives the keyboard back to reveal.js, image panels drawn
// from Float64Arrays, a log slider, and a time-sliced background job runner.
import { htmlEl, releaseFocus, slider, Slider } from "../lib/ui";
import { drawScalarImage, ColorMap, grey } from "../lib/colour";

export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}

export interface Select<T extends string> { el: HTMLSelectElement; get(): T; set(v: T): void; row: HTMLElement }

/** A labelled drop-down; blurs itself after a change so the arrow keys go back to reveal.js. */
export function select<T extends string>(parent: Element, label: string, options: [T, string][], value: T, onChange: (v: T) => void): Select<T> {
  const row = htmlEl("label", "dc-select", "", parent);
  htmlEl("span", "qb-slider-name", label, row);
  const el = htmlEl("select", "", "", row);
  for (const [v, t] of options) { const o = htmlEl("option", "", t, el); o.value = v; }
  el.value = value;
  el.addEventListener("change", () => { onChange(el.value as T); el.blur(); });
  releaseFocus(el);
  return { el, row, get: () => el.value as T, set: (v: T) => { el.value = v; } };
}

/** A slider over log10 of the value: shows the value, passes the value. */
export function logSlider(parent: Element, label: string, lo: number, hi: number, value: number, onInput: (v: number) => void, fmt: (v: number) => string = fmtSci): Slider & { getValue(): number; setValue(v: number): void } {
  const s = slider(parent, label, Math.log10(lo), Math.log10(hi), 0.01, Math.log10(value), u => onInput(10 ** u), u => fmt(10 ** u));
  return Object.assign(s, { getValue: () => 10 ** s.get(), setValue: (v: number) => s.set(Math.log10(v)) });
}

/** Compact number: 3 significant digits, scientific outside 1e-3..1e5. */
export function fmtSci(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) { const e = Math.floor(Math.log10(a)), m = v / 10 ** e; return `${m.toFixed(1)}e${e}`; }
  return Number(v.toPrecision(3)).toString();
}

export interface ImagePanel { fig: HTMLElement; canvas: HTMLCanvasElement; cap: HTMLElement; draw(v: ArrayLike<number>, w: number, h: number, opts?: { lo?: number; hi?: number; map?: ColorMap }): void }

/** A captioned canvas showing a scalar image (pixelated, fills its column). */
export function imagePanel(parent: Element, caption: string, cls = ""): ImagePanel {
  const fig = htmlEl("figure", "dc-img" + (cls ? " " + cls : ""), "", parent);
  const cap = htmlEl("figcaption", "", caption, fig);
  const canvas = htmlEl("canvas", "", "", fig);
  return {
    fig, canvas, cap,
    draw(v, w, h, o = {}) { drawScalarImage(canvas, v, w, h, { map: o.map ?? grey, lo: o.lo, hi: o.hi }); },
  };
}

/**
 * Runs a generator in time slices of about `budget` ms, yielding to the browser between
 * slices (via a MessageChannel, which background-tab timer throttling does not slow down),
 * so long computations (sweeps, many iterations) keep the page responsive. `onSlice` runs
 * after every slice and at the end (done = true). Starting a new job cancels the old one.
 */
export class Jobs {
  private token = 0;
  private ch = new MessageChannel();
  private queue: (() => void)[] = [];
  constructor() { this.ch.port1.onmessage = () => { const f = this.queue.shift(); if (f) f(); }; }
  private later(f: () => void): void { this.queue.push(f); this.ch.port2.postMessage(0); }
  run(gen: Generator<void, void, void>, onSlice: (done: boolean) => void = () => {}, budget = 14, firstBudget = budget): void {
    const my = ++this.token;
    let first = true;
    const step = () => {
      if (my !== this.token) return;
      const t0 = performance.now(), b = first ? firstBudget : budget;
      first = false;
      while (performance.now() - t0 < b) {
        const r = gen.next();
        if (my !== this.token) return;
        if (r.done) { onSlice(true); return; }
      }
      onSlice(false);
      this.later(step);
    };
    this.later(step);
  }
  cancel(): void { this.token++; }
}

/** An HTML legend row (line swatches + labels) below a plot. */
export function legendRow(parent: Element, items: { label: string; color: string; dash?: boolean }[]): HTMLElement {
  const row = htmlEl("div", "dc-legend", "", parent);
  for (const it of items) {
    const s = htmlEl("span", "", "", row);
    const sw = htmlEl("i", "", "", s);
    sw.style.borderTop = `3px ${it.dash ? "dashed" : "solid"} ${it.color}`;
    s.append(it.label);
  }
  return row;
}

/** Shared CSS for the day's demos (scoped by the demo name). */
export function baseStyle(name: string): string {
  const D = `[data-demo="${name}"]`;
  return `
${D} figure { margin: 0; }
${D} figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin: 0 0 .15em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
${D} .dc-img canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 2px; }
${D} .dc-select { display: grid; grid-template-columns: auto 1fr; gap: .5em; align-items: center; }
${D} .dc-select select { font: inherit; font-size: .9em; padding: .1em .2em; min-width: 0; }
${D} .dc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.3em; }
${D} .qb-readout { font-size: 14px; }
${D} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
${D} .qb-presets .qb-btn { font-size: 13px; padding: .15em .45em; }
${D} .dc-legend { display: flex; flex-wrap: wrap; gap: .1em .9em; font-size: 15px; justify-content: center; margin-top: -.2em; }
${D} .dc-legend span { white-space: nowrap; }
${D} .dc-legend i { display: inline-block; width: 22px; margin-right: .3em; vertical-align: middle; }
${D} .dc-credit { font-size: 12px; color: var(--qb-muted); }
.reveal ${D} { font-size: 16px; gap: .6em; }
.reveal ${D} .qb-panel { gap: .2em; }
.reveal ${D} .qb-note, .reveal ${D} .dc-legend { display: flex; flex-wrap: wrap; gap: .1em .9em; font-size: 15px; justify-content: center; margin-top: -.2em; }
${D} .dc-legend span { white-space: nowrap; }
${D} .dc-legend i { display: inline-block; width: 22px; margin-right: .3em; vertical-align: middle; }
${D} .dc-credit { display: none; }
.reveal ${D} .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal ${D} .qb-slider { gap: .3em; }
`;
}
