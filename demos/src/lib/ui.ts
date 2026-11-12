// Shared helpers for the course demos. Every demo:
//  - auto-mounts on <div data-demo="<name>">, reading options from data-* attributes;
//  - draws in SVG with a viewBox (or a canvas), so it scales inside reveal's slide;
//  - blurs its inputs after use, so the arrow keys go back to reveal.js;
//  - uses a fixed random seed, so a recording is reproducible.

export const SVGNS = "http://www.w3.org/2000/svg";

/** Create an SVG element with attributes. */
export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K, attrs: Record<string, string | number> = {}, parent?: Element,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

/** Set several attributes at once. */
export function setAttrs(e: Element, attrs: Record<string, string | number>): void {
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
}

/** Create an HTML element with a class and optional text. */
export function htmlEl<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls = "", text = "", parent?: Element,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

/** Give the keyboard back to reveal.js: it ignores keys while an input has focus. */
export function releaseFocus(inp: HTMLElement): void {
  inp.addEventListener("pointerup", () => inp.blur());
  inp.addEventListener("change", () => inp.blur());
}

export interface Slider {
  input: HTMLInputElement;
  get(): number;
  set(v: number): void;
}

/** A labelled range slider with a live value readout. */
export function slider(
  parent: Element, label: string, min: number, max: number, step: number, value: number,
  onInput: (v: number) => void, fmt: (v: number) => string = v => v.toFixed(decimals(step)),
): Slider {
  const lab = htmlEl("label", "qb-slider", "", parent);
  htmlEl("span", "qb-slider-name", label, lab);
  const input = htmlEl("input", "", "", lab);
  Object.assign(input, { type: "range", min: String(min), max: String(max), step: String(step), value: String(value) });
  const out = htmlEl("output", "", fmt(value), lab);
  input.addEventListener("input", () => { out.value = fmt(Number(input.value)); onInput(Number(input.value)); });
  releaseFocus(input);
  return {
    input,
    get: () => Number(input.value),
    set: (v: number) => { input.value = String(v); out.value = fmt(v); },
  };
}

/** A button that blurs itself after a click. */
export function button(parent: Element, text: string, onClick: () => void, cls = ""): HTMLButtonElement {
  const b = htmlEl("button", "qb-btn" + (cls ? " " + cls : ""), text, parent);
  b.type = "button";
  b.addEventListener("click", () => { onClick(); b.blur(); });
  return b;
}

/** A row of preset buttons; the last one clicked is highlighted. Returns the buttons. */
export function presets(parent: Element, items: { label: string; apply: () => void }[]): HTMLButtonElement[] {
  const row = htmlEl("div", "qb-presets", "", parent);
  const btns: HTMLButtonElement[] = [];
  for (const it of items) {
    const b = button(row, it.label, () => {
      btns.forEach(x => x.classList.toggle("active", x === b));
      it.apply();
    });
    btns.push(b);
  }
  return btns;
}

/** A labelled checkbox. */
export function checkbox(parent: Element, label: string, value: boolean, onChange: (v: boolean) => void): HTMLInputElement {
  const lab = htmlEl("label", "qb-check", "", parent);
  const input = htmlEl("input", "", "", lab);
  input.type = "checkbox";
  input.checked = value;
  input.addEventListener("change", () => { onChange(input.checked); input.blur(); });
  lab.append(" " + label);
  return input;
}

function decimals(step: number): number {
  const s = String(step);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

/**
 * Make an SVG point draggable. `toWorld` maps a pointer event to world coords;
 * `onDrag` receives them. Works with mouse, pen and touch.
 */
export function draggable(handle: SVGElement, toWorld: (ev: PointerEvent) => [number, number],
  onDrag: (x: number, y: number) => void): void {
  handle.style.cursor = "grab";
  handle.style.touchAction = "none";
  handle.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    const move = (e: PointerEvent) => { const [x, y] = toWorld(e); onDrag(x, y); };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
}

/** Pointer position in the user coordinates of an SVG (respects viewBox and CSS transforms). */
export function svgPoint(svg: SVGSVGElement, ev: { clientX: number; clientY: number }): [number, number] {
  const ctm = svg.getScreenCTM();
  if (!ctm) return [0, 0];
  const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(ctm.inverse());
  return [p.x, p.y];
}

/** Linear map from [d0,d1] to [r0,r1]. */
export function scale(d0: number, d1: number, r0: number, r1: number) {
  const f = (v: number) => r0 + (v - d0) * (r1 - r0) / (d1 - d0);
  f.invert = (v: number) => d0 + (v - r0) * (d1 - d0) / (r1 - r0);
  return f;
}

/** Seeded random numbers (mulberry32), so every run of a demo looks the same. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const unif = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const norm = () => {
    const u = 1 - unif(), v = unif();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { unif, norm };
}

/** Read a numeric list from a data-* attribute, e.g. data-a="1 0 0 1". */
export function numList(s: string | undefined, fallback: number[]): number[] {
  if (!s) return fallback;
  const v = s.trim().split(/[\s,]+/).map(Number);
  return v.every(Number.isFinite) ? v : fallback;
}

/** Mount `fn` on every <div data-demo="name"> once the page is ready. */
export function autoMount(name: string, fn: (host: HTMLElement) => void): void {
  const run = () => document.querySelectorAll<HTMLElement>(`[data-demo="${name}"]:not(.qb-demo)`)
    .forEach(host => { host.classList.add("qb-demo"); host.innerHTML = ""; fn(host); });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
  else run();
}
