// morphgrid mode "hitmiss" (skeleton day, unit skel-thinning): hit-or-miss with three-state
// templates, template thinning (sequential or parallel), thinning by simple points (with or
// without keeping end points), and a "simple?" check of every pixel.
//
// Side column: the template editor (click cycles 1 → 0 → · = object / background / don't
// care), template presets, and the toggles. Main grid: the image (click to edit). Second
// grid: the hit-or-miss result, or the pass in which each pixel was removed (dark = kept).
// The algorithms live in lib/skeleton.ts (tested in demos/test/skeleton-test.ts); this file
// only turns them into frames.
import { htmlEl, button, checkbox } from "../lib/ui";
import { GridView, GridAlgorithm, GridFrame, CellMark } from "../lib/gridui";
import { RGB } from "../lib/colour";
import {
  Template, templateRing, rotateTemplate, matchAt, neighbours, simpleTest, thin, thinSimple, RING, RING_NAMES,
  THIN_TEMPLATES, THIN_NAMES, T_END, T_ISOLATED, T_CORNER, T_TJUNCTION, T_YJUNCTION, L1, L2,
} from "../lib/skeleton";
import { registerMode, ModeContext, ModeSpec, ModePreset } from "./registry";

const GREEN = "var(--qb-c3, #2ca02c)", RED = "var(--qb-c2, #d62728)", ORANGE = "var(--qb-c4, #ff7f0e)", BLUE = "var(--qb-c1, #1f77b4)";
const COMP_COLOURS = [BLUE, ORANGE, "var(--qb-c5, #9467bd)", "#8c564b"];

/** Template cell value: 1 object, 0 background, 2 don't care. */
const tplToValues = (t: Template): Int32Array => Int32Array.from(t.join("").split(""), c => (c === "1" ? 1 : c === "0" ? 0 : 2));
const valuesToTpl = (v: ArrayLike<number>): Template => [0, 1, 2].map(r => [0, 1, 2].map(c => (v[r * 3 + c] === 1 ? "1" : v[r * 3 + c] === 0 ? "0" : ".")).join(""));
const TPL_MAP = (t: number): RGB => (t < 0.25 ? [1, 1, 1] : t < 0.75 ? [0.16, 0.2, 0.3] : [0.82, 0.82, 0.82]);

const TPL_PRESETS: { id: string; label: string; t: Template }[] = [
  { id: "end", label: "end point", t: T_END },
  { id: "isolated", label: "isolated", t: T_ISOLATED },
  { id: "corner", label: "corner", t: T_CORNER },
  { id: "tjunction", label: "T-junction", t: T_TJUNCTION },
  { id: "yjunction", label: "Y-junction", t: T_YJUNCTION },
  { id: "L1", label: "L1 (edge)", t: L1 },
  { id: "L2", label: "L2 (corner)", t: L2 },
];

const idx = (x: number, y: number, w: number) => y * w + x;
const inside = (x: number, y: number, w: number, h: number) => x >= 0 && y >= 0 && x < w && y < h;
const count = (a: ArrayLike<number>) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i]) n++; return n; };

/** Window offsets [dRow, dCol] of the 3 × 3 neighbourhood with the centre. */
const WIN9: [number, number][] = [];
for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) WIN9.push([dr, dc]);

/** Marks for a template placed at (x, y): green where a cell fits, red where it fails. */
function templateMarks(m: Uint8Array, w: number, h: number, x: number, y: number, t: Template): { marks: CellMark[]; fails: number; cells: number } {
  const marks: CellMark[] = [];
  let fails = 0, cells = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const ch = t[r][c];
    if (ch === ".") continue;
    cells++;
    const xx = x + c - 1, yy = y + r - 1;
    const v = inside(xx, yy, w, h) ? m[idx(xx, yy, w)] : 0;
    const ok = (ch === "1") === (v === 1);
    if (!ok) fails++;
    if (inside(xx, yy, w, h)) marks.push({ index: idx(xx, yy, w), color: ok ? GREEN : RED, fill: true, tag: ch });
  }
  return { marks, fails, cells };
}

function* runHitMiss(m: Uint8Array, w: number, h: number, base: Template, rot: boolean): Generator<GridFrame> {
  const ts = rot ? [0, 1, 2, 3, 4, 5, 6, 7].map(s => rotateTemplate(base, s)) : [base];
  const uniq = ts.filter((t, k) => ts.findIndex(u => u.join() === t.join()) === k);
  const out = new Uint8Array(w * h), centreCare = templateRing(base).centre;
  yield {
    values: m, aux: out, log: `template ${base.join(" / ")}${rot ? `, ${uniq.length} rotation${uniq.length > 1 ? "s" : ""}` : ""}`,
    note: "Hit-or-miss: the template must fit the object (1) and the background (0) at once; · = don't care.",
  };
  let lastRow = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = idx(x, y, w);
    if (centreCare === 1 && !m[i]) continue;   // the centre must be an object pixel: skip the background
    const stage = y !== lastRow ? `row ${y}` : undefined;
    lastRow = y;
    let hit = -1;
    for (let k = 0; k < uniq.length; k++) if (matchAt(m, w, h, x, y, uniq[k])) { hit = k; break; }
    const shown = uniq[hit >= 0 ? hit : 0], tm = templateMarks(m, w, h, x, y, shown);
    if (hit >= 0) out[i] = 1;
    yield {
      values: m, aux: out.slice(), focus: i, auxFocus: i, window: { at: i, offsets: WIN9, color: BLUE }, marks: tm.marks,
      log: `(${y}, ${x}): ` + (hit >= 0 ? `fits${uniq.length > 1 ? ` (rotation ${hit + 1})` : ""} → 1` : uniq.length > 1 ? "no rotation fits → 0" : `${tm.fails} of ${tm.cells} cells fail → 0`),
      stage,
      note: "Green: the template cell fits. Red: it does not.",
    };
  }
  yield { values: m, aux: out, stage: "result", log: `result: ${count(out)} pixel${count(out) === 1 ? "" : "s"} found`, note: `${count(out)} pixel${count(out) === 1 ? "" : "s"} where the template fits.` };
}

/** Aux values for thinning: pass of removal (1, 2, ...); kept pixels get KEPT (darkest). */
const KEPT = 99;
function* runThin(m: Uint8Array, w: number, h: number, parallel: boolean): Generator<GridFrame> {
  const res = thin(m, w, h, { parallel, record: true });
  const kept = KEPT, aux = new Int32Array(w * h), cur = m.slice();
  for (let i = 0; i < w * h; i++) aux[i] = m[i] ? kept : 0;
  yield { values: cur, aux, log: `start: ${count(m)} pixels`, note: parallel ? "Parallel: all 8 templates are tested on the same image, and every match is removed at once." : "Sequential: the 8 templates in turn; each removes its matches before the next one looks." };
  for (const st of res.steps) {
    for (const i of st.removed) { cur[i] = 0; aux[i] = st.pass; }
    const name = st.template < 0 ? "all 8 templates" : THIN_NAMES[st.template];
    yield {
      values: cur.slice(), aux: aux.slice(), marks: st.removed.map(i => ({ index: i, color: RED, fill: true })),
      log: `pass ${st.pass}, ${name}: ${st.removed.length} removed`,
      stage: st.template <= 0 ? `pass ${st.pass}` : undefined,
      note: st.removed.length ? `Red: removed by ${name} in pass ${st.pass}.` : `${name} matches nothing.`,
    };
  }
  const n = count(cur);
  yield {
    values: cur, aux, stage: "result", log: `done after ${res.passes} passes (the last removes nothing): ${n} pixels left`,
    note: n ? `Skeleton: ${n} pixels (dark on the right; the numbers are the pass in which a pixel went).` : "Nothing is left: parallel removal cut the shape away.",
  };
}

function* runSimpleThin(m: Uint8Array, w: number, h: number, keepEnds: boolean): Generator<GridFrame> {
  const res = thinSimple(m, w, h, { keepEnds, record: true });
  const cur = m.slice(), aux = new Int32Array(w * h), kept = KEPT;
  for (let i = 0; i < w * h; i++) aux[i] = m[i] ? kept : 0;
  yield { values: cur, aux, log: `start: ${count(m)} pixels; end points ${keepEnds ? "kept" : "not kept"}`, note: "Visit the border pixels (one side at a time) and remove each one that is simple." };
  let lastPass = 0, lastDir = -1;
  for (const st of res.steps) {
    if (st.removed) { cur[st.index] = 0; aux[st.index] = st.pass; }
    const x = st.index % w, y = (st.index - x) / w;
    const verdict = st.removed ? "removed" : st.test.simple && st.end ? "end point: kept" : "kept";
    const stage = st.pass !== lastPass || st.dir !== lastDir ? `pass ${st.pass}, ${RING_NAMES[st.dir]} side` : undefined;
    lastPass = st.pass; lastDir = st.dir;
    yield {
      values: cur.slice(), aux: aux.slice(), focus: st.index, window: { at: st.index, offsets: WIN9, color: st.removed ? RED : BLUE },
      log: `(${y}, ${x}): ${st.test.reason} → ${verdict}`, stage,
    };
  }
  yield { values: cur, aux, stage: "result", log: `done after ${res.passes} passes: ${count(cur)} pixels left`, note: keepEnds ? "End points kept: the lines keep their length." : "End points removed too: every tree shrinks to a single pixel; loops survive." };
}

function* runSimpleCheck(m: Uint8Array, w: number, h: number): Generator<GridFrame> {
  const all: CellMark[] = [];
  yield { values: m, log: "check every object pixel", note: "Simple = 1 object part and 1 background part among the 8 neighbours." };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = idx(x, y, w);
    if (!m[i]) continue;
    const st = simpleTest(neighbours(m, w, h, x, y)), marks: CellMark[] = [];
    st.fg.forEach((comp, k) => comp.forEach(r => { const xx = x + RING[r][0], yy = y + RING[r][1]; marks.push({ index: idx(xx, yy, w), color: COMP_COLOURS[k % 4], fill: true, tag: String.fromCharCode(97 + k) }); }));
    st.bg.forEach((comp, k) => comp.forEach(r => { const xx = x + RING[r][0], yy = y + RING[r][1]; if (inside(xx, yy, w, h)) marks.push({ index: idx(xx, yy, w), color: COMP_COLOURS[k % 4], tag: String(k + 1) }); }));
    const col = st.simple ? (st.degree === 1 ? ORANGE : GREEN) : RED;
    all.push({ index: i, color: col });
    yield {
      values: m, focus: i, window: { at: i, offsets: WIN9, color: BLUE }, marks,
      log: `(${y}, ${x}): ${st.fg.length} object part${st.fg.length === 1 ? "" : "s"}, ${st.bg.length} background part${st.bg.length === 1 ? "" : "s"} → ${st.simple ? "simple" : "not simple"}`,
      note: `${st.reason}. Object parts: filled, a, b, ...; background parts touching the pixel: outlined, 1, 2, ...`,
    };
  }
  yield { values: m, marks: all, stage: "result", log: `${all.filter(a => a.color === GREEN).length} simple, ${all.filter(a => a.color === ORANGE).length} end points, ${all.filter(a => a.color === RED).length} not simple`, note: "Green: simple (removable). Orange: end point (simple, but kept). Red: not simple." };
}

// ---------------------------------------------------------------------------------------
// presets (text rows: 1 = object)

const G = {
  ends: ["0000000", "0000000", "0111100", "0000010", "0000000", "0100000", "0000000"],
  specks: ["0000000000", "0100000000", "0000011100", "0001011100", "0000011100", "0000000000", "0110000010", "0000000000"],
  rect: ["000000000", "011111110", "011111110", "011111110", "000000000"],
  L: ["0000000000", "0111000000", "0111000000", "0111000000", "0111000000", "0111111110", "0111111110", "0111111110", "0000000000"],
  ring: ["000000000", "001111100", "011111110", "011000110", "011000110", "011000110", "011111110", "001111100", "000000000"],
  line2: ["000000000000", "011111111110", "011111111110", "000000000000"],
  // five 3 × 3 neighbourhoods (exercise B2), centres at (2, 2), (2, 7), (2, 12), (2, 17), (2, 22)
  simple: ["0000000000000000000000000", "0010001100011100010001010", "0010001100011100110000100", "0010000000011100000000000", "0000000000000000000000000"],
  bigL: ["0000000000000", "0111100000000", "0111100000000", "0111100000000", "0111100000000", "0111111111110", "0111111111110", "0111111111110", "0000000000000"],
};

registerMode({
  id: "hitmiss",
  build(ctx: ModeContext): ModeSpec {
    let rot = true, keepEnds = true, parallel = false, inspect = false, ready = false;
    const safeRestart = () => { if (ready) ctx.restart(); };
    htmlEl("div", "mg-lab", "template (click: 1 → 0 → ·)", ctx.side);
    const tpl = new GridView(ctx.side, 3, 3, tplToValues(T_END), {
      cell: 44, display: "int", lo: 0, hi: 2, map: TPL_MAP, text: v => (v === 2 ? "·" : String(v)), ariaLabel: "hit-or-miss template",
      edit: (old: number) => (old === 1 ? 0 : old === 0 ? 2 : 1),
      onEdit: () => { markTpl(""); safeRestart(); },
    });
    (tpl.svg as SVGSVGElement).style.maxWidth = "150px";
    const prow = htmlEl("div", "qb-presets", "", ctx.side);
    const pb = TPL_PRESETS.map(p => button(prow, p.label, () => { setTpl(p.id); safeRestart(); }));
    const markTpl = (id: string) => pb.forEach((b, k) => b.classList.toggle("active", TPL_PRESETS[k].id === id));
    const setTpl = (id: string) => { const p = TPL_PRESETS.find(q => q.id === id); if (p) { tpl.setValues(tplToValues(p.t)); markTpl(id); } };
    markTpl("end");
    const rotBox = checkbox(ctx.side, "all 8 rotations", rot, v => { rot = v; safeRestart(); });
    const parBox = checkbox(ctx.side, "parallel (no subiterations)", parallel, v => { parallel = v; safeRestart(); });
    const endBox = checkbox(ctx.side, "keep end points", keepEnds, v => { keepEnds = v; safeRestart(); });
    const inspBox = checkbox(ctx.side, "click inspects a pixel", inspect, v => { inspect = v; setInspect(); });
    const usedLab = htmlEl("div", "mg-lab", "", ctx.side);
    const used = new GridView(ctx.side, 3, 3, tplToValues(THIN_TEMPLATES[0]), { cell: 30, display: "int", lo: 0, hi: 2, map: TPL_MAP, text: v => (v === 2 ? "·" : String(v)), edit: false, ariaLabel: "template in use" });
    (used.svg as SVGSVGElement).style.maxWidth = "100px";
    const showUsed = (k: number) => {
      const on = k >= 0;
      used.svg.style.display = on ? "" : "none";
      usedLab.textContent = on ? `template in use: ${THIN_NAMES[k]}` : "";
      if (on) used.setValues(tplToValues(THIN_TEMPLATES[k]));
    };
    showUsed(-1);

    let editRule: unknown = null;
    function setInspect(): void {
      const s = ctx.stepper();
      if (editRule === null) editRule = s.main.opts.edit;
      s.main.opts.edit = inspect ? ((old: number, i: number) => {
        if (s.algorithm.id !== "simple") s.setAlgorithm("simple");
        const k = s.frames.findIndex(f => f.focus === i);
        if (k >= 0) s.goTo(k);
        return old;
      }) : (editRule as typeof s.main.opts.edit);
    }
    const set = (o: { t?: string; rot?: boolean; par?: boolean; ends?: boolean }) => {
      if (o.t) setTpl(o.t);
      rot = o.rot ?? true; rotBox.checked = rot;
      parallel = o.par ?? false; parBox.checked = parallel;
      keepEnds = o.ends ?? true; endBox.checked = keepEnds;
      inspect = false; inspBox.checked = false; if (ready) setInspect();
    };
    const bin = (v: Int32Array) => Uint8Array.from(v, x => (x ? 1 : 0));
    const algorithms: GridAlgorithm[] = [
      { id: "hitmiss", label: "hit-or-miss", auxDisplay: "binary", auxTitle: "result", run: (inp, rows, cols) => runHitMiss(bin(inp), cols, rows, valuesToTpl(tpl.values), rot) },
      {
        id: "thin", label: "thin: 8 templates", auxDisplay: "int", auxTitle: "removed in pass (dark: kept)", auxLo: 0, auxHi: 8,
        auxText: v => (v === 0 || v === KEPT ? "" : String(v)),
        run: (inp, rows, cols) => runThin(bin(inp), cols, rows, parallel),
      },
      {
        id: "simplethin", label: "thin: simple points", auxDisplay: "int", auxTitle: "removed in pass (dark: kept)", auxLo: 0, auxHi: 8,
        auxText: v => (v === 0 || v === KEPT ? "" : String(v)),
        run: (inp, rows, cols) => runSimpleThin(bin(inp), cols, rows, keepEnds),
      },
      { id: "simple", label: "simple?", run: (inp, rows, cols) => runSimpleCheck(bin(inp), cols, rows) },
    ];
    const presets: ModePreset[] = [
      { id: "ends", label: "find the end points", grid: G.ends, algorithm: "hitmiss", apply: () => set({ t: "end" }) },
      { id: "isolated", label: "isolated pixels", grid: G.specks, algorithm: "hitmiss", apply: () => set({ t: "isolated", rot: false }) },
      { id: "rect", label: "thin a rectangle", grid: G.rect, algorithm: "thin", apply: () => set({}) },
      { id: "L", label: "thin an L", grid: G.bigL, algorithm: "thin", apply: () => set({}) },
      { id: "ring", label: "a ring keeps its hole", grid: G.ring, algorithm: "simplethin", apply: () => set({ ends: false }) },
      { id: "parallel", label: "parallel removal cuts the line", grid: G.line2, algorithm: "thin", apply: () => set({ par: true }) },
      { id: "noends", label: "no end points: a line shrinks", grid: G.rect, algorithm: "simplethin", apply: () => set({ ends: false }) },
      { id: "simple", label: "simple or not?", grid: G.simple, algorithm: "simple", apply: () => set({}) },
      { id: "blank", label: "empty 12 × 12", grid: { rows: 12, cols: 12, values: new Int32Array(144) }, algorithm: "hitmiss", apply: () => set({ t: "end" }) },
    ];
    return {
      algorithms, display: "binary", edit: "toggle", cell: 40, title: "image (click to edit)", maxFrames: 4000,
      presets,
      onFrame: (f: GridFrame, _i: number, s) => {
        if (s.algorithm.id !== "thin") { showUsed(-1); return; }
        const m = /pass \d+, (.*?):/.exec(f.log ?? "");
        const k = m ? THIN_NAMES.indexOf(m[1]) : -1;
        showUsed(k);
      },
      ready() { ready = true; },
    };
  },
});

export const HITMISS_GRIDS = G;
