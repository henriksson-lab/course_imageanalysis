// morphgrid mode "label" (labelling day, unit label-connectivity/): connected-component
// labelling step by step. Left: the binary mask (click to edit). Right: the labels.
//
//  - flood fill (queue) / flood fill (stack): raster scan for an unlabelled object pixel,
//    then grow its component; the queue or stack is marked on the mask (1 = next out).
//  - two-pass + union–find: first pass with the already-visited neighbours (4: up, left;
//    8: up-left, up, up-right, left), provisional labels and equivalences; then the classes
//    are numbered 1, 2, 3, … and the second pass replaces every provisional label.
//  - runs: the same with horizontal runs instead of pixels.
//
// The side column chooses the connectivity pair: objects 8 / background 4 (the course
// convention) or objects 4 / background 8. The generators below are pure (no DOM) and are
// checked against lib/label.ts in demos/test/labelling-test.ts.
import { htmlEl, button } from "../lib/ui";
import { GridAlgorithm, GridFrame, CellMark } from "../lib/gridui";
import { label2d } from "../lib/label";
import { registerMode, ModeContext, ModeSpec } from "./registry";
import { Forest } from "../unionfind/forest";

export type Conn = 4 | 8;
type Off = [number, number];   // [dRow, dCol]

const N4: Off[] = [[-1, 0], [0, -1], [0, 1], [1, 0]];
const N8: Off[] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
/** Already-visited neighbours in a raster scan. */
const BACK4: Off[] = [[-1, 0], [0, -1]];
const BACK8: Off[] = [[-1, -1], [-1, 0], [-1, 1], [0, -1]];

const QUEUE = "var(--qb-c4, #ff7f0e)", RUN = "var(--qb-c1, #1f77b4)", EQ = "var(--qb-c2, #d62728)";
const rc = (i: number, w: number) => `(${Math.floor(i / w)}, ${i % w})`;
const short = (xs: string[], n = 7) => (xs.length > n ? xs.slice(0, n).join(" ") + ` … (+${xs.length - n})` : xs.join(" "));

/** Flood-fill labelling: BFS with a queue or DFS with an explicit stack. */
export function* floodFrames(inp: ArrayLike<number>, w: number, h: number, conn: Conn, kind: "bfs" | "dfs"): Generator<GridFrame> {
  const lab = new Int32Array(w * h), nb = conn === 8 ? N8 : N4;
  const level = new Int32Array(w * h);
  const name = kind === "bfs" ? "queue" : "stack";
  yield { values: inp, aux: lab, log: `flood fill with a ${name}, ${conn}-connectivity`, note: "Scan for an object pixel without a label; give it a new label and grow its component." };
  let next = 0, maxLen = 0;
  for (let s = 0; s < w * h; s++) {
    if (!inp[s] || lab[s]) continue;
    const L = ++next;
    lab[s] = L; level[s] = 0;
    const list: number[] = [s];
    const marks = () => list.map((q, k): CellMark => ({ index: q, color: QUEUE, tag: String(kind === "bfs" ? k + 1 : list.length - k) }));
    yield { values: inp, aux: lab.slice(), focus: s, auxFocus: s, marks: marks(), stage: `object ${L}`, log: `new object ${L} at ${rc(s, w)}; ${name}: ${rc(s, w)}` };
    while (list.length) {
      maxLen = Math.max(maxLen, list.length);
      const p = kind === "bfs" ? list.shift()! : list.pop()!;
      const r = Math.floor(p / w), c = p % w;
      const added: string[] = [], offs: Off[] = [], hits: boolean[] = [];
      for (const [dr, dc] of nb) {
        const rr = r + dr, cc = c + dc;
        offs.push([dr, dc]);
        if (rr < 0 || cc < 0 || rr >= h || cc >= w) { hits.push(false); continue; }
        const q = rr * w + cc;
        const fresh = !!inp[q] && !lab[q];
        hits.push(!!inp[q]);
        if (fresh) { lab[q] = L; level[q] = level[p] + 1; list.push(q); added.push(rc(q, w)); }
      }
      const content = (kind === "bfs" ? list : [...list].reverse()).map(q => rc(q, w));
      yield {
        values: inp, aux: lab.slice(), focus: p, auxFocus: p, window: { at: p, offsets: offs, hits, origin: false }, marks: marks(),
        log: `take ${rc(p, w)}${kind === "bfs" ? ` (ring ${level[p]})` : ""}: ${added.length ? "add " + added.join(" ") : "nothing new"}`,
        note: `${name} (${kind === "bfs" ? "front → back" : "top → bottom"}): ${short(content) || "empty"}`,
      };
    }
  }
  const other = label2d(inp, w, h, conn === 8 ? 4 : 8).count;
  yield { values: inp, aux: lab, stage: "result", log: `done: ${next} objects (largest ${name}: ${maxLen})`, note: `${next} objects with ${conn}-connectivity (${other} with ${conn === 8 ? 4 : 8}). Largest ${name}: ${maxLen} pixels.` };
}

const classText = (f: Forest, n: number) => f.classes().filter(c => c[0] !== 0 && c[0] <= n).map(c => (c.length > 1 ? `{${c.join(", ")}}` : null)).filter(Boolean).join(" ") || "none";

/** Two-pass labelling with union–find (union by rank, full path compression). */
export function* twoPassFrames(inp: ArrayLike<number>, w: number, h: number, conn: Conn): Generator<GridFrame> {
  const prov = new Int32Array(w * h), back = conn === 8 ? BACK8 : BACK4;
  const f = new Forest(1, { byRank: true, compression: "full" });   // node 0 unused
  let n = 0, eqs = 0;
  yield { values: inp, aux: prov, log: `two-pass labelling, ${conn}-connectivity`, note: `First pass: look only at the already visited neighbours (${conn === 8 ? "up-left, up, up-right, left" : "up, left"}).` };
  let row = -1;
  for (let i = 0; i < w * h; i++) {
    if (!inp[i]) continue;
    const r = Math.floor(i / w), c = i % w;
    const seen: number[] = [], hits: boolean[] = [];
    for (const [dr, dc] of back) {
      const rr = r + dr, cc = c + dc;
      const v = rr >= 0 && cc >= 0 && cc < w ? prov[rr * w + cc] : 0;
      hits.push(v > 0);
      if (v > 0 && !seen.includes(v)) seen.push(v);
    }
    let msg: string;
    const marks: CellMark[] = [];
    if (!seen.length) { prov[i] = ++n; f.add(); msg = `new label ${n}`; }
    else {
      const m = Math.min(...seen);
      prov[i] = m;
      const newEq: string[] = [];
      for (const v of seen) if (v !== m && f.rootOf(v) !== f.rootOf(m)) { f.union(v, m); newEq.push(`${m} ≡ ${v}`); eqs++; }
      msg = seen.length === 1 ? `copy ${m}` : `labels ${seen.sort((a, b) => a - b).join(", ")} → ${m}` + (newEq.length ? `; record ${newEq.join(", ")}` : " (already equivalent)");
      if (newEq.length) marks.push({ index: i, color: EQ, tag: "≡" });
    }
    yield {
      values: inp, aux: prov.slice(), focus: i, auxFocus: i, auxWindow: { at: i, offsets: back, hits, origin: false }, auxMarks: marks,
      stage: r !== row ? `first pass, row ${r}` : undefined,
      log: `${rc(i, w)}: ${msg}`, note: `Equivalent: ${classText(f, n)}`,
    };
    row = r;
  }
  // number the classes in order of their smallest provisional label
  const final = new Int32Array(n + 1);
  let count = 0;
  for (let l = 1; l <= n; l++) { const r = f.find(l).root; if (!final[r]) final[r] = ++count; final[l] = final[r]; }
  const table = Array.from({ length: n }, (_, k) => `${k + 1}→${final[k + 1]}`).join("  ");
  yield { values: inp, aux: prov.slice(), stage: "equivalence table", log: `${n} provisional labels, ${eqs} equivalences → ${count} objects: ${table}`, note: `Classes: ${classText(f, n)}. Final numbers: ${table}` };
  const out = prov.slice();
  let firstRow = true;
  for (let r = 0; r < h; r++) {
    let changed = false;
    const marks: CellMark[] = [];
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      if (out[i]) { if (out[i] !== final[out[i]]) marks.push({ index: i, color: EQ }); out[i] = final[prov[i]]; changed = true; }
    }
    if (!changed) continue;
    const st = firstRow; firstRow = false;
    yield { values: inp, aux: out.slice(), auxMarks: marks, stage: st ? "second pass" : undefined, log: `second pass, row ${r}${marks.length ? `: ${marks.length} label(s) replaced` : ""}`, note: `Second pass: replace each provisional label by its class number. ${table}` };
  }
  const other = label2d(inp, w, h, conn === 8 ? 4 : 8).count;
  yield { values: inp, aux: out, stage: "result", log: `done: ${count} objects`, note: `${count} objects with ${conn}-connectivity (${other} with ${conn === 8 ? 4 : 8}); ${n} provisional labels, ${eqs} equivalences.` };
}

/** Run-based labelling: horizontal runs, each overlapping run of the previous row merged. */
export function* runFrames(inp: ArrayLike<number>, w: number, h: number, conn: Conn): Generator<GridFrame> {
  const prov = new Int32Array(w * h);
  const f = new Forest(1, { byRank: true, compression: "full" });
  let prev: [number, number, number][] = [], n = 0, eqs = 0, runs = 0;
  const ext = conn === 8 ? 1 : 0;
  yield { values: inp, aux: prov, log: `run-based labelling, ${conn}-connectivity`, note: `A run is a horizontal stretch of object pixels. Runs overlap ${conn === 8 ? "(diagonally too)" : "(sharing a column)"} with runs of the row above.` };
  for (let r = 0; r < h; r++) {
    const cur: [number, number, number][] = [];
    for (let c = 0; c < w; c++) {
      if (!inp[r * w + c]) continue;
      let e = c;
      while (e + 1 < w && inp[r * w + e + 1]) e++;
      runs++;
      const over = prev.filter(([a, b]) => a <= e + ext && b >= c - ext).map(q => q[2]);
      const uniq = [...new Set(over)].sort((a, b) => a - b);
      let L: number, msg: string;
      if (!uniq.length) { L = ++n; f.add(); msg = `new label ${L}`; }
      else {
        L = uniq[0];
        const ne: string[] = [];
        for (const v of uniq.slice(1)) if (f.rootOf(v) !== f.rootOf(L)) { f.union(v, L); ne.push(`${L} ≡ ${v}`); eqs++; }
        msg = uniq.length === 1 ? `overlaps run ${L}` : `overlaps runs ${uniq.join(", ")} → ${L}` + (ne.length ? `; record ${ne.join(", ")}` : "");
      }
      for (let k = c; k <= e; k++) prov[r * w + k] = L;
      cur.push([c, e, L]);
      const marks: CellMark[] = [];
      for (let k = c; k <= e; k++) marks.push({ index: r * w + k, color: RUN, fill: true });
      yield { values: inp, aux: prov.slice(), marks, auxMarks: marks, stage: `row ${r}`, log: `row ${r}, columns ${c}–${e}: ${msg}`, note: `Equivalent: ${classText(f, n)}` };
      c = e;
    }
    prev = cur;
  }
  const final = new Int32Array(n + 1);
  let count = 0;
  for (let l = 1; l <= n; l++) { const rt = f.find(l).root; if (!final[rt]) final[rt] = ++count; final[l] = final[rt]; }
  const out = prov.map(v => (v ? final[v] : 0));
  yield { values: inp, aux: out, stage: "result", log: `done: ${runs} runs, ${eqs} equivalences → ${count} objects`, note: `${count} objects; ${runs} runs instead of ${Array.from(inp).filter(Boolean).length} pixels, ${eqs} equivalences.` };
}

const G = {
  diag: ["000000", "011000", "011000", "000110", "000110", "000000"],
  rings: ["000000000", "011111110", "011111110", "011111110", "011111110", "011111110", "011111110", "011111110", "000000000"],
  u: ["0000000", "0110110", "0110110", "0110110", "0111110", "0111110", "0000000"],
  comb: ["00000000000", "01010101010", "01010101010", "01010101010", "01111111110", "00000000000"],
  count: ["11000010", "11000100", "00101010", "00010001", "01110000", "01010011", "01110011", "00000000"],
  hand: ["100010", "100100", "100101", "111101", "000001", "110111"],
  spiral: ["0000000000", "0111111110", "0100000010", "0101111010", "0101001010", "0101011010", "0100000010", "0111111110", "0000000000"],
};

registerMode({
  id: "label",
  build(ctx: ModeContext): ModeSpec {
    let conn: Conn = 8, ready = false;
    htmlEl("div", "mg-lab", "connectivity", ctx.side);
    const row = htmlEl("div", "qb-presets", "", ctx.side);
    const b8 = button(row, "objects 8 · background 4", () => setConn(8));
    const b4 = button(row, "objects 4 · background 8", () => setConn(4));
    const setConn = (c: Conn) => { conn = c; b8.classList.toggle("active", c === 8); b4.classList.toggle("active", c === 4); if (ready) ctx.restart(); };
    setConn(conn);
    htmlEl("div", "qb-note", "Orange tags: the queue/stack (1 = next out). Red ≡: an equivalence is recorded here.", ctx.side);
    const mk = (id: string, label: string, run: (inp: Int32Array, w: number, h: number) => Iterable<GridFrame>): GridAlgorithm => ({
      id, label, auxDisplay: "label", auxTitle: "labels", auxText: v => (v ? String(v) : ""),
      run: (input, rows, cols) => run(input, cols, rows),
    });
    return {
      algorithms: [
        mk("bfs", "flood fill (queue)", (i, w, h) => floodFrames(i, w, h, conn, "bfs")),
        mk("dfs", "flood fill (stack)", (i, w, h) => floodFrames(i, w, h, conn, "dfs")),
        mk("twopass", "two-pass + union–find", (i, w, h) => twoPassFrames(i, w, h, conn)),
        mk("runs", "runs", (i, w, h) => runFrames(i, w, h, conn)),
      ],
      display: "binary", edit: "toggle", cell: 36, title: "mask (click to edit)",
      presets: [
        { id: "diag", label: "diagonal neighbours: one or two objects?", grid: G.diag, algorithm: "bfs", apply: () => setConn(8), frame: "end" },
        { id: "rings", label: "queue rings vs stack arms", grid: G.rings, algorithm: "bfs", apply: () => setConn(4) },
        { id: "u", label: "a U needs one equivalence", grid: G.u, algorithm: "twopass", apply: () => setConn(8) },
        { id: "comb", label: "a comb: many", grid: G.comb, algorithm: "twopass", apply: () => setConn(8) },
        { id: "count", label: "count with 4 and with 8", grid: G.count, algorithm: "twopass", apply: () => setConn(8), frame: "end" },
        { id: "spiral", label: "a spiral: runs", grid: G.spiral, algorithm: "runs", apply: () => setConn(8) },
        { id: "hand", label: "6 × 6 (exercise)", grid: G.hand, algorithm: "twopass", apply: () => setConn(8) },
        { id: "blank", label: "empty 12 × 12", grid: { rows: 12, cols: 12, values: new Int32Array(144) }, algorithm: "twopass", apply: () => setConn(8) },
      ],
      ready() { ready = true; },
    };
  },
});

export const LABEL_GRIDS = G;
