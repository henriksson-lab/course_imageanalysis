// A labelled row of mutually exclusive buttons (a segmented control), shared by the
// skeleton-day demos (medialaxis, skelgraph, arbor). Candidate for lib/ui.ts.
import { htmlEl, button } from "../lib/ui";

export interface Choice<T extends string> {
  get(): T;
  set(v: T): void;
  row: HTMLElement;
}

export function choice<T extends string>(parent: Element, label: string, options: [T, string][], value: T, onChange: (v: T) => void): Choice<T> {
  const box = htmlEl("div", "sk-choice", "", parent);
  if (label) htmlEl("span", "sk-choice-lab", label, box);
  const row = htmlEl("span", "qb-presets", "", box);
  let cur = value;
  const btns = options.map(([v, l]) => button(row, l, () => { api.set(v); onChange(v); }));
  const api: Choice<T> = {
    get: () => cur,
    set(v: T) { cur = v; btns.forEach((b, k) => b.classList.toggle("active", options[k][0] === v)); },
    row: box,
  };
  api.set(value);
  return api;
}

export const CHOICE_CSS = `
.qb-demo .sk-choice { display: flex; flex-wrap: wrap; align-items: center; gap: .3em .45em; }
.qb-demo .sk-choice-lab { font-size: 15px; color: var(--qb-muted); min-width: 4.2em; }
.qb-demo .sk-choice .qb-btn { padding: .12em .45em; }
`;
