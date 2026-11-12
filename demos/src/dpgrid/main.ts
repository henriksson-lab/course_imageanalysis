// The DP table by hand, as a step-through: a small grid of integer costs (left) and the
// cumulative table M (right). Step fills one cell: its candidates in the row above are
// outlined, the chosen minimum is shaded and an arrow points back to it. "next stage"
// fills a row; "fill all" the whole table; "backtrack" follows the arrows from the best
// cell of the last row. Readout: cells filled, operations so far, and the number of paths.
//
// data-mode="dp" (default, unit dp-paths/): monotone paths, moves ↙↓↘ (top row to bottom
//   row) or → ↓ (top-left to bottom-right; the DTW table).
// data-mode="dijkstra" (unit livewire/): 4 neighbours in every direction; step finalises the
//   smallest node of the queue (shown sorted below the controls) and updates its neighbours.
//   A second algorithm, "one raster pass", shows why a fixed order fails on a detour.
// Options: data-preset="exercise|greedy|equal|detour" (dp) or "detour|exercise|wall"
// (dijkstra), data-moves="3|rd", data-pointers="0|1", data-frame="fill|end|<n>" (start frame). Click the upper half of a cost cell
// to raise it by 1, the lower half to lower it (range 0–9).
import { htmlEl, button, checkbox, autoMount } from "../lib/ui";
import { gridStepper, GridAlgorithm, GridFrame, CellMark, CellArrow, GridStepper, N4 } from "../lib/gridui";
import { pathDP, verticalSteps, cornerSteps, countPaths, PathDPOptions } from "../lib/dp";
import { dijkstraTrace, vertexCost, tracePath } from "../lib/shortestpath";
import { DPGRID_PRESETS, LW_PRESETS, GridPreset, Source, Target, greedy } from "./grids";

const NAME = "dpgrid";
const EMPTY = -1, UNREACHED = -2, INF = 1_000_000_000;
const C_PATH = "var(--qb-c4, #ff7f0e)", C_CHOSEN = "var(--qb-c3, #2ca02c)", C_FINAL = "var(--qb-c1, #1f77b4)", C_WRONG = "var(--qb-c2, #d62728)";

const STYLE = `
[data-demo="${NAME}"] .qb-gridstep-grids { flex: 1 1 560px; }
[data-demo="${NAME}"] .qb-gridstep-cell { flex: 1 1 240px; max-width: 420px; }
[data-demo="${NAME}"] .qb-gridstep-panel { flex: 0 1 330px; gap: .4em; }
[data-demo="${NAME}"] .qb-gridstep-count + .qb-note { font-family: ui-monospace, monospace; font-size: 15px; color: var(--qb-fg); white-space: pre-wrap; }
[data-demo="${NAME}"] .dg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .dg-help { font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-gridstep-log { font-size: 16px; max-height: 5.2em; }
.reveal [data-demo="${NAME}"] .qb-gridstep-panel { flex-basis: 340px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-slider { display: none; }
.reveal [data-demo="${NAME}"] .dg-help { display: none; }
.reveal [data-demo="${NAME}"] .qb-gridstep-count + .qb-note { font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const rc = (p: number, cols: number) => `(${Math.floor(p / cols)},${p % cols})`;
const fmtBig = (b: bigint) => b.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const fmtV = (v: number) => (v === UNREACHED || v >= INF / 2 || !Number.isFinite(v) ? "∞" : String(v));

interface Settings { moves: "3" | "rd"; pointers: boolean; source: Source; target: Target }

function targetOf(t: Target, d: ArrayLike<number>, rows: number, cols: number): number {
  if (t === "br") return rows * cols - 1;
  if (t === "bl") return (rows - 1) * cols;
  let best = (rows - 1) * cols;
  for (let j = 1; j < cols; j++) if (d[(rows - 1) * cols + j] < d[best]) best = (rows - 1) * cols + j;
  return best;
}

function dpAlgorithm(id: "3" | "rd", st: Settings): GridAlgorithm {
  return {
    id: "dp" + id,
    label: id === "3" ? "moves ↙ ↓ ↘" : "moves → ↓",
    auxTitle: "table M",
    auxText: v => (v === EMPTY ? "" : fmtV(v)),
    auxLo: 0,
    *run(C, rows, cols): Iterable<GridFrame> {
      const n = rows * cols;
      const opts: PathDPOptions = id === "3" ? { steps: verticalSteps(1), trace: true } : { steps: cornerSteps(false), start: "corner", end: "corner", trace: true };
      const res = pathDP(C, rows, cols, opts), total = countPaths(rows, cols, opts);
      const M = new Int32Array(n).fill(EMPTY), ptr: CellArrow[] = [];
      let ops = 0, filled = 0;
      const note = () => `cells filled ${filled} / ${n}\noperations ${ops}\npaths ${fmtBig(total)}`;
      yield { values: C, aux: M, log: id === "3" ? "Start: every cell of the top row may start a path." : "Start: the path starts at the top-left cell.", note: note() };
      const tr = res.trace!;
      let k = 0;
      // the start cells of the first row in one frame (moves ↙↓↘)
      if (id === "3") {
        for (; k < tr.length && tr[k].i === 0; k++) { M[tr[k].i * cols + tr[k].j] = tr[k].M; filled++; }
        yield { values: C, aux: M, stage: "row 0", log: "Row 0: M = C (the path starts here).", note: note(), auxMarks: Array.from({ length: cols }, (_, j) => ({ index: j, color: C_CHOSEN })) };
      }
      for (; k < tr.length; k++) {
        const t = tr[k], p = t.i * cols + t.j;
        M[p] = Number.isFinite(t.M) ? t.M : INF;
        filled++;
        const valid = t.cand.filter(c => c.from >= 0);
        ops += valid.length;
        const stage = (t.j === 0 && id === "3") || (t.j === 0 && id === "rd") ? `row ${t.i}` : undefined;
        if (t.chosen < 0) {
          yield { values: C, aux: M, focus: p, auxFocus: p, stage, log: `M${rc(p, cols)} = C = ${t.M} (start)`, note: note() };
          continue;
        }
        const ch = t.cand[t.chosen];
        ptr.push({ from: p, to: ch.from });
        const offsets = valid.map(c => [Math.floor(c.from / cols) - t.i, (c.from % cols) - t.j] as [number, number]);
        yield {
          values: C, aux: M, focus: p, auxFocus: p, stage,
          auxWindow: { at: p, offsets, origin: false },
          auxMarks: [{ index: ch.from, color: C_CHOSEN, fill: true }],
          auxArrows: st.pointers ? ptr.slice() : [ptr[ptr.length - 1]],
          log: `M${rc(p, cols)} = ${C[p]} + min(${valid.map(c => fmtV(c.value)).join(", ")}) = ${fmtV(t.M)}`,
          note: note(),
        };
      }
      // backtracking
      const path = res.path, marks: CellMark[] = [], pm: CellMark[] = [];
      const g = id === "3" ? greedy(C, rows, cols) : null;
      for (let s = path.length - 1; s >= 0; s--) {
        marks.push({ index: path[s], color: C_PATH, fill: true });
        pm.push({ index: path[s], color: C_PATH, fill: true });
        const first = s === path.length - 1;
        yield {
          values: C, aux: M, stage: first ? "backtrack" : undefined,
          auxMarks: marks.slice(), marks: pm.slice(), auxArrows: st.pointers ? ptr.slice() : [],
          log: first ? (id === "3" ? `Best cell of the last row: ${rc(path[s], cols)}, M = ${res.cost}` : `End cell ${rc(path[s], cols)}, M = ${res.cost}`) : `back to ${rc(path[s], cols)}`,
          note: note() + `\noptimal cost ${res.cost}` + (s === 0 && g ? `\ngreedy cost ${g.cost}` : ""),
        };
      }
      if (g) {
        yield {
          values: C, aux: M, auxMarks: marks.slice(), auxArrows: st.pointers ? ptr.slice() : [],
          marks: [...pm, ...g.path.map(q => ({ index: q, color: C_WRONG, tag: "g" }))],
          log: `Greedy (red "g"): ${g.cost}; optimal: ${res.cost}`,
          note: note() + `\noptimal cost ${res.cost}\ngreedy cost ${g.cost}`,
        };
      }
    },
  };
}

function sourcesOf(C: ArrayLike<number>, cols: number, st: Settings) {
  return st.source === "row" ? Array.from({ length: cols }, (_, j) => ({ index: j, value: C[j] })) : [{ index: 0, value: C[0] }];
}

function dijkstraAlgorithm(st: Settings): GridAlgorithm {
  return {
    id: "dijkstra",
    label: "Dijkstra",
    auxTitle: "distance d",
    auxText: v => (v === EMPTY ? "" : fmtV(v)),
    auxLo: 0,
    *run(C, rows, cols): Iterable<GridFrame> {
      const n = rows * cols;
      const { result, events, initialQueue } = dijkstraTrace({ width: cols, height: rows, conn: 4, edge: vertexCost(C), sources: sourcesOf(C, cols, st) });
      const d = new Int32Array(n).fill(UNREACHED), pred = new Int32Array(n).fill(-1);   // ∞, drawn light
      for (const [q, v] of initialQueue) d[q] = v;
      const final: CellMark[] = [];
      const qtext = (q: [number, number][]) => q.length ? q.slice(0, 7).map(([p, v]) => `${v}${rc(p, cols)}`).join(" ") + (q.length > 7 ? ` … (+${q.length - 7})` : "") : "empty";
      const arrows = (): CellArrow[] => {
        const a: CellArrow[] = [];
        if (!st.pointers) return a;
        for (let q = 0; q < n; q++) if (pred[q] >= 0) a.push({ from: q, to: pred[q], color: final.some(m => m.index === q) ? "var(--qb-c4, #ff7f0e)" : "var(--qb-muted, #999)" });
        return a;
      };
      yield {
        values: C, aux: d, auxMarks: initialQueue.map(([q]) => ({ index: q, color: C_CHOSEN })),
        log: st.source === "row" ? "Start: every top-row cell is in the queue with d = its cost." : "Start: the top-left cell is in the queue with d = its cost.",
        note: `final 0 / ${n}\nqueue: ${qtext(initialQueue)}`,
      };
      for (let k = 0; k < events.length; k++) {
        const e = events[k];
        final.push({ index: e.pop, color: C_FINAL, fill: true, tag: String(k + 1) });
        for (const [q, v] of e.updated) { d[q] = v; pred[q] = e.pop; }
        const offsets = N4.filter(([dr, dc]) => { const r = Math.floor(e.pop / cols) + dr, c = (e.pop % cols) + dc; return r >= 0 && c >= 0 && r < rows && c < cols; });
        yield {
          values: C, aux: d.slice(), focus: e.pop, auxFocus: e.pop,
          auxWindow: { at: e.pop, offsets, origin: false },
          auxMarks: final.slice(), auxArrows: arrows(),
          stage: k === 0 ? "pops" : undefined,
          log: `final ${rc(e.pop, cols)} d = ${e.d}` + (e.updated.length ? `; update ${e.updated.map(([q, v]) => `${rc(q, cols)}→${v}`).join(", ")}` : ""),
          note: `final ${k + 1} / ${n}\nqueue: ${qtext(e.queue)}`,
        };
      }
      // backtracking from the target
      const t = targetOf(st.target, result.dist, rows, cols), path = tracePath(result.pred, t, result.dist);
      const pm: CellMark[] = [];
      for (let s = path.length - 1; s >= 0; s--) {
        pm.push({ index: path[s], color: C_PATH, fill: true });
        yield {
          values: C, aux: d, marks: pm.slice(), auxMarks: [...final.map(m => ({ ...m, fill: false })), ...pm], auxArrows: arrows(),
          stage: s === path.length - 1 ? "backtrack" : undefined,
          log: s === path.length - 1 ? `Target ${rc(t, cols)}: d = ${result.dist[t]}` : `back to ${rc(path[s], cols)}`,
          note: `final ${events.length} / ${n}\npath cost ${result.dist[t]}, ${path.length} cells`,
        };
      }
    },
  };
}

/** One forward raster pass: d = C + min(d above, d left), as the first pass of the distance transform. */
function onePassAlgorithm(st: Settings): GridAlgorithm {
  return {
    id: "onepass",
    label: "one raster pass",
    auxTitle: "d after one pass",
    auxText: v => (v === EMPTY ? "" : fmtV(v)),
    auxLo: 0,
    *run(C, rows, cols): Iterable<GridFrame> {
      const n = rows * cols, d = new Int32Array(n).fill(EMPTY), src = sourcesOf(C, cols, st);
      const isSrc = new Map(src.map(s => [s.index, s.value]));
      yield { values: C, aux: d, log: "One pass in raster order, from above and from the left only.", note: "" };
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const p = i * cols + j, cand: number[] = [], offs: [number, number][] = [];
        if (i > 0) { cand.push(d[p - cols]); offs.push([-1, 0]); }
        if (j > 0) { cand.push(d[p - 1]); offs.push([0, -1]); }
        const m = Math.min(...cand.map(v => (v === EMPTY || v >= INF ? INF : v)), INF);
        let v = m >= INF ? INF : m + C[p];
        if (isSrc.has(p)) v = Math.min(v, isSrc.get(p)!);
        d[p] = v;
        yield { values: C, aux: d, focus: p, auxFocus: p, auxWindow: offs.length ? { at: p, offsets: offs, origin: false } : undefined,
          stage: j === 0 ? `row ${i}` : undefined, log: `d${rc(p, cols)} = ${isSrc.has(p) ? "start " : ""}${fmtV(v)}`, note: "" };
      }
      const truth = dijkstraTrace({ width: cols, height: rows, conn: 4, edge: vertexCost(C), sources: src }).result.dist;
      const wrong: CellMark[] = [];
      for (let p = 0; p < n; p++) if (d[p] !== truth[p]) wrong.push({ index: p, color: C_WRONG, tag: String(truth[p]) });
      const t = targetOf(st.target, truth, rows, cols);
      yield { values: C, aux: d, auxMarks: wrong, stage: "check", marks: [{ index: t, color: C_PATH }],
        log: wrong.length ? `${wrong.length} cells too high (red; small number = true d)` : "Correct everywhere: no path needs to go up or left.",
        note: `target ${rc(t, cols)}: one pass ${fmtV(d[t])}, Dijkstra ${truth[t]}` };
    },
  };
}

function mount(host: HTMLElement): void {
  injectStyle();
  const mode = host.dataset.mode === "dijkstra" ? "dijkstra" : "dp";
  const list = mode === "dp" ? DPGRID_PRESETS : LW_PRESETS;
  const initPreset = list.find(p => p.id === host.dataset.preset) ?? list[0];
  const st: Settings = {
    moves: host.dataset.moves === "rd" ? "rd" : "3",
    pointers: host.dataset.pointers !== "0",
    source: initPreset.source ?? "row",
    target: initPreset.target ?? "row",
  };
  const algorithms = mode === "dp" ? [dpAlgorithm("3", st), dpAlgorithm("rd", st)] : [dijkstraAlgorithm(st), onePassAlgorithm(st)];
  let stepper: GridStepper | null = null;
  const lastFill = (s: GridStepper) => { const k = s.frames.findIndex(f => f.stage === "backtrack"); return k > 0 ? k - 1 : s.frames.length - 1; };
  stepper = gridStepper(host, {
    rows: initPreset.rows, cols: initPreset.cols, initial: initPreset.values,
    algorithms, algorithm: mode === "dp" ? "dp" + st.moves : "dijkstra",
    display: "int", lo: 0, hi: 9, title: "cost C", edit: "plusminus", indices: true,
    interval: 500,
    extraButtons: [
      { label: mode === "dp" ? "fill all" : "run all", onClick: s => { s.play(false); s.goTo(lastFill(s)); } },
      { label: "backtrack", onClick: s => { const k = s.frames.findIndex(f => f.stage === "backtrack"); if (k >= 0) { s.goTo(k); s.play(true); } } },
    ],
    onInput: v => {
      if (v.some(x => x < 0 || x > 9)) {
        const c = Int32Array.from(v, x => Math.max(0, Math.min(9, x)));
        queueMicrotask(() => stepper?.setInput(c));
      }
    },
  });
  // data-frame: start at a frame ("fill" = table filled, "end" = last frame, or a number)
  const fr = host.dataset.frame;
  if (fr === "fill") stepper.goTo(lastFill(stepper));
  else if (fr === "end") stepper.toEnd();
  else if (fr && Number.isFinite(Number(fr))) stepper.goTo(Number(fr));
  const panel = stepper.panel;
  checkbox(panel, "show pointers", st.pointers, v => { st.pointers = v; stepper!.rerun(); });
  htmlEl("div", "dg-lab", "presets", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  const btns: HTMLButtonElement[] = [];
  const apply = (p: GridPreset) => {
    st.source = p.source ?? "row"; st.target = p.target ?? "row";
    btns.forEach((b, k) => b.classList.toggle("active", list[k] === p));
    stepper!.setInput(p.values, p.rows, p.cols);
  };
  list.forEach(p => btns.push(button(row, p.label, () => apply(p))));
  button(row, "reset", () => {
    st.pointers = host.dataset.pointers !== "0";
    stepper!.setAlgorithm(mode === "dp" ? "dp" + (host.dataset.moves === "rd" ? "rd" : "3") : "dijkstra");
    apply(initPreset);
    btns.forEach(b => b.classList.remove("active"));
  });
  htmlEl("div", "dg-help", mode === "dp"
    ? "Click the upper half of a cost cell to raise it, the lower half to lower it. Blue outline: the candidates; green shading: the chosen minimum; arrows point to where the minimum came from."
    : "Blue cells are final (number = order). Outlined: the neighbours being updated. The queue lists d(row,col), smallest first.", panel);
}

autoMount(NAME, mount);
