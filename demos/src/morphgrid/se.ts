// Structuring-element editor of the morphgrid demo: a k × k grid of cells (click toggles a
// cell), a marked origin ("set origin", then click a cell: no hover or modifier keys) and
// SE preset buttons. Exported so other modes (e.g. hit-or-miss) can reuse or wrap it.
import { htmlEl, button } from "../lib/ui";
import { GridView } from "../lib/gridui";
import { Off, seFromGrid } from "./ops";

export interface SEPreset { id: string; label: string; rows: string[]; origin?: [number, number] }

/** 5 × 5 presets; origin is [row, col] (default the centre). */
export const SE_PRESETS: SEPreset[] = [
  { id: "cross", label: "cross", rows: ["00000", "00100", "01110", "00100", "00000"] },
  { id: "square", label: "square", rows: ["00000", "01110", "01110", "01110", "00000"] },
  { id: "disc", label: "disc", rows: ["01110", "11111", "11111", "11111", "01110"] },
  { id: "line", label: "line", rows: ["00000", "00000", "01110", "00000", "00000"] },
  { id: "vline", label: "column", rows: ["00000", "00100", "00100", "00100", "00000"] },
  { id: "L", label: "L-shape", rows: ["00000", "00000", "00110", "00100", "00000"] },
];

export interface SEEditor {
  view: GridView;
  /** Offsets [dx, dy] of the set cells relative to the origin. */
  offsets(): Off[];
  set(id: string): void;
  setGrid(rows: string[], origin?: [number, number]): void;
  /** Short description, e.g. "cross (5 cells)". */
  name(): string;
}

export function seEditor(parent: HTMLElement, onChange: () => void, opts: { k?: number; cell?: number; presets?: SEPreset[] } = {}): SEEditor {
  const k = opts.k ?? 5, presets = opts.presets ?? SE_PRESETS;
  const box = htmlEl("div", "mg-se", "", parent);
  htmlEl("div", "mg-lab", "structuring element", box);
  let origin = (k >> 1) * k + (k >> 1), picking = false, current = "";
  const view = new GridView(box, k, k, undefined, {
    cell: opts.cell ?? 34, display: "binary", numbers: false, ariaLabel: "structuring element editor",
    edit: (old: number, i: number) => {
      if (picking) {
        picking = false; origin = i; originBtn.classList.remove("active");
        current = ""; refresh(); onChange();
        return old;
      }
      return old ? 0 : 1;
    },
    onEdit: () => { current = ""; refresh(); onChange(); },
  });
  const row = htmlEl("div", "qb-presets", "", box);
  const originBtn = button(row, "set origin", () => { picking = !picking; originBtn.classList.toggle("active", picking); });
  const prow = htmlEl("div", "qb-presets", "", box);
  const pbtns = presets.map(p => button(prow, p.label, () => { api.set(p.id); onChange(); }));

  function refresh(): void {
    view.setMarks([{ index: origin, color: "var(--qb-c4, #ff7f0e)", tag: "o" }]);
    pbtns.forEach((b, i) => b.classList.toggle("active", presets[i].id === current));
  }

  const api: SEEditor = {
    view,
    offsets: () => seFromGrid(view.values, k, origin),
    set(id: string) {
      const p = presets.find(q => q.id === id);
      if (!p) return;
      api.setGrid(p.rows, p.origin);
      current = p.id; refresh();
    },
    setGrid(rows: string[], o?: [number, number]) {
      const v = new Int32Array(k * k);
      rows.forEach((r, y) => r.split("").forEach((c, x) => { if (y < k && x < k) v[y * k + x] = c === "1" ? 1 : 0; }));
      view.setValues(v);
      origin = o ? o[0] * k + o[1] : (k >> 1) * k + (k >> 1);
      current = ""; refresh();
    },
    name: () => {
      const n = seFromGrid(view.values, k, origin).length;
      const p = presets.find(q => q.id === current);
      return `${p ? p.label : "your SE"} (${n} cell${n === 1 ? "" : "s"})`;
    },
  };
  refresh();
  return api;
}
