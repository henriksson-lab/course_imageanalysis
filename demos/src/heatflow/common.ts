// Small UI helpers shared by the edge-preserving smoothing demos (heatflow, peronamalik,
// tvflow, patchweights, inpaint, smoothercompare). Image panels come from canny/imgview.ts.
import { htmlEl, button, slider, Slider } from "../lib/ui";
import { Grey, fromArray } from "../lib/imageops";
import { loadCell } from "../lib/cell";

/** A row of tab buttons; returns a function that highlights the active one. */
export function tabRow(parent: Element, items: [string, string][], onPick: (id: string) => void): (id: string) => void {
  const row = htmlEl("div", "ps-tabs", "", parent);
  const btns = items.map(([id, label]) => { const b = button(row, label, () => onPick(id)); b.dataset.id = id; return b; });
  return (id: string) => btns.forEach(b => b.classList.toggle("active", b.dataset.id === id));
}

/** A slider on a log scale between lo and hi (value shown with fmt). */
export function logSlider(parent: Element, label: string, lo: number, hi: number, value: number, onInput: (v: number) => void, fmt: (v: number) => string = v => v.toPrecision(2)): Slider & { val(): number; setVal(v: number): void } {
  const toV = (u: number) => lo * Math.pow(hi / lo, u / 1000), toU = (v: number) => 1000 * Math.log(v / lo) / Math.log(hi / lo);
  const s = slider(parent, label, 0, 1000, 1, toU(value), u => onInput(toV(u)), u => fmt(toV(u)));
  return Object.assign(s, { val: () => toV(s.get()), setVal: (v: number) => s.set(toU(v)) });
}

/** A labelled group of choice buttons (radio-like). Returns a setter for the highlight. */
export function choice(parent: Element, label: string, items: [string, string][], onPick: (id: string) => void): ((id: string) => void) & { el: HTMLElement } {
  const wrap = htmlEl("div", "ps-choice", "", parent);
  if (label) htmlEl("span", "ps-choice-lab", label, wrap);
  const btns = items.map(([id, l]) => { const b = button(wrap, l, () => { onPick(id); }); b.dataset.id = id; return b; });
  return Object.assign((id: string) => btns.forEach(b => b.classList.toggle("active", b.dataset.id === id)), { el: wrap });
}

let cellCache: Promise<Grey> | null = null;
/** The course micrograph at 128 × 128 (cached). */
export function cell128(): Promise<Grey> {
  if (!cellCache) cellCache = loadCell(128);
  return cellCache;
}

export const toGrey = (a: ArrayLike<number>, w: number, h: number): Grey => fromArray(w, h, a);

/** CSS shared by the day's demos (each demo adds its own, prefixed). */
export const PS_CSS = `
.qb-demo .ps-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .25em; }
.qb-demo .ps-tabs .qb-btn { padding: .1em .55em; }
.qb-demo .ps-choice { display: flex; flex-wrap: wrap; align-items: center; gap: .25em; }
.qb-demo .ps-choice .qb-btn { padding: .08em .45em; }
.qb-demo .ps-choice-lab { font-size: 15px; color: var(--qb-muted); margin-right: .2em; }
.qb-demo .ps-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.3em; }
.qb-demo .ps-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo .ps-choice-lab { font-size: 16px; }
.reveal .qb-demo .ps-credit { display: none; }
`;

/** Requestanimationframe loop that stops itself; returns {start, stop, running}. */
export function animator(tick: () => boolean): { start(): void; stop(): void; running(): boolean } {
  let id = 0;
  const loop = () => { if (!tick()) { id = 0; return; } id = requestAnimationFrame(loop); };
  return {
    start: () => { if (!id) id = requestAnimationFrame(loop); },
    stop: () => { if (id) cancelAnimationFrame(id); id = 0; },
    running: () => id !== 0,
  };
}
