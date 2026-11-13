// Lineage trees (tracking day). Pure layout and statistics, plus one DOM draw function
// (marked). A lineage is a list of branches: one per cell, from its first to its last frame,
// with a parent (-1 for a founder) and its children (the daughters of its division).
import { svgEl } from "./ui";

export interface Branch {
  id: number;
  parent: number;
  t0: number;
  t1: number;
  children: number[];
  /** drawn red (e.g. a wrong division) */
  wrong?: boolean;
  label?: string;
}

/** Branches of the subtree rooted at `root`, in depth-first order. */
export function subtree(branches: Branch[], root: number): Branch[] {
  const byId = new Map(branches.map(b => [b.id, b]));
  const out: Branch[] = [], stack = [root];
  while (stack.length) {
    const b = byId.get(stack.pop()!);
    if (!b) continue;
    out.push(b);
    for (let i = b.children.length - 1; i >= 0; i--) stack.push(b.children[i]);
  }
  return out;
}

/** Tidy horizontal positions: leaves 0, 1, 2, … left to right; a parent sits midway between its children. */
export function layoutX(branches: Branch[], root: number): Map<number, number> {
  const byId = new Map(branches.map(b => [b.id, b])), x = new Map<number, number>();
  let next = 0;
  const visit = (id: number): number => {
    const b = byId.get(id)!;
    const kids = b.children.filter(c => byId.has(c));
    const v = kids.length ? kids.map(visit).reduce((s, q) => s + q, 0) / kids.length : next++;
    x.set(id, v);
    return v;
  };
  if (byId.has(root)) visit(root);
  return x;
}

export interface CycleTime { id: number; cycle: number; complete: boolean; generation: number }

/**
 * Cycle time of every branch = last frame − first frame + 1. Complete only if the cell was born
 * by a division (not a founder or an entering cell) and divided itself before the movie ended.
 */
export function cycleTimes(branches: Branch[], divided: (b: Branch) => boolean): CycleTime[] {
  const byId = new Map(branches.map(b => [b.id, b]));
  const gen = (b: Branch): number => { let g = 0; for (let p = b.parent; p >= 0; p = byId.get(p)?.parent ?? -1) g++; return g; };
  return branches.map(b => ({ id: b.id, cycle: b.t1 - b.t0 + 1, complete: b.parent >= 0 && divided(b), generation: gen(b) }));
}

/**
 * DOM: draw a lineage tree into an SVG group. Time runs downwards: `ty(t)` gives y, `bx(x)` the
 * x of a leaf position. Vertical lines are cells, horizontal bars are divisions.
 */
export function drawLineage(g: SVGGElement, branches: Branch[], root: number,
  bx: (x: number) => number, ty: (t: number) => number,
  style: { color?: string; wrong?: string; width?: number; labels?: boolean; font?: number } = {}): void {
  const x = layoutX(branches, root), byId = new Map(branches.map(b => [b.id, b]));
  const col = style.color ?? "var(--qb-c1, #1f77b4)", bad = style.wrong ?? "var(--qb-c2, #d62728)", w = style.width ?? 3;
  for (const b of subtree(branches, root)) {
    const X = bx(x.get(b.id)!);
    const c = b.wrong ? bad : col;
    const v = svgEl("line", { x1: X, x2: X, y1: ty(b.t0), y2: ty(b.t1 + 1) }, g);
    v.style.stroke = c; v.style.strokeWidth = String(w); v.style.strokeLinecap = "round";
    const kids = b.children.filter(k => x.has(k));
    if (kids.length) {
      const xs = kids.map(k => bx(x.get(k)!));
      const h = svgEl("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: ty(b.t1 + 1), y2: ty(b.t1 + 1) }, g);
      const anyWrong = kids.some(k => byId.get(k)?.wrong);
      h.style.stroke = anyWrong ? bad : c; h.style.strokeWidth = String(w);
    }
    if (style.labels && b.label) {
      const t = svgEl("text", { x: X + 5, y: (ty(b.t0) + ty(b.t1 + 1)) / 2 + 5 }, g);
      t.textContent = b.label;
      t.style.fontSize = (style.font ?? 16) + "px";
    }
  }
}
