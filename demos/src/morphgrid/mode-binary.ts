// morphgrid modes "binary" (unit morph-binary: erosion, dilation, boundary, complement,
// duality, two lines = a square) and "openclose" (unit morph-openclose: opening, closing,
// their combinations, idempotence).
//
// Display: one grid. Cell value = I + 2·D, with I the input pixel and D the pixel of the set
// shown in this frame: 0 white (0 → 0), 1 red (removed), 2 green (added), 3 dark (kept).
// While a pass scans the image, the set being scanned is shown and the decisions so far are
// outlined (red: will be removed, green: will be added). Editing toggles the input bit.
import { htmlEl, button, checkbox } from "../lib/ui";
import { GridAlgorithm, GridFrame, CellMark } from "../lib/gridui";
import { RGB } from "../lib/colour";
import { registerMode, ModeContext, ModePreset, ModeSpec } from "./registry";
import { Off, probe, reflect, erodeGrid, dilateGrid, complement } from "./ops";
import { seEditor } from "./se";

const RED = "var(--qb-c2, #d62728)", GREEN = "var(--qb-c3, #2ca02c)";
const PAL: RGB[] = [[1, 1, 1], [0.93, 0.42, 0.40], [0.50, 0.82, 0.48], [0.16, 0.2, 0.3]];
const H3: Off[] = [[-1, 0], [0, 0], [1, 0]], V3: Off[] = [[0, -1], [0, 0], [0, 1]];
const SQ3: Off[] = [];
for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) SQ3.push([dx, dy]);

type Kind = "erode" | "dilate";
interface Stage { kind: Kind; name: string; se?: Off[] }
interface Op {
  id: string; label: string; stages: Stage[];
  /** After the passes: "boundary" shows A \ result, "complement" shows Aᶜ. */
  final?: "boundary" | "complement";
  /** Compare the result with another expression (shown in the last note). */
  compare?: { label: string; f: (a: Uint8Array, w: number, h: number, se: Off[], o: number) => Uint8Array };
  /** Expression of the dual (shown in the second grid with "show duality"). */
  dual?: string;
}

const E = (name = "erosion"): Stage => ({ kind: "erode", name });
const D = (name = "dilation"): Stage => ({ kind: "dilate", name });

const OPS: Record<string, Op> = {
  erode: { id: "erode", label: "erosion A ⊖ B", stages: [E()], dual: "(Aᶜ ⊕ B̌)ᶜ" },
  dilate: { id: "dilate", label: "dilation A ⊕ B", stages: [D()], dual: "(Aᶜ ⊖ B̌)ᶜ" },
  open: { id: "open", label: "opening", stages: [E("erosion (opening, 1/2)"), D("dilation (opening, 2/2)")], dual: "(Aᶜ • B̌)ᶜ" },
  close: { id: "close", label: "closing", stages: [D("dilation (closing, 1/2)"), E("erosion (closing, 2/2)")], dual: "(Aᶜ ∘ B̌)ᶜ" },
  boundary: { id: "boundary", label: "boundary A \\ (A ⊖ B)", stages: [E()], final: "boundary" },
  complement: { id: "complement", label: "complement Aᶜ", stages: [], final: "complement" },
  twolines: {
    id: "twolines", label: "⊕ row, then ⊕ column",
    stages: [{ kind: "dilate", name: "dilation by a row of 3", se: H3 }, { kind: "dilate", name: "dilation by a column of 3", se: V3 }],
    compare: { label: "A ⊕ (3 × 3 square)", f: (a, w, h, _se, o) => dilateGrid(a, w, h, SQ3, o) },
  },
  openclose: {
    id: "openclose", label: "open, then close",
    stages: [E("opening: erosion"), D("opening: dilation"), D("closing: dilation"), E("closing: erosion")],
    compare: { label: "close, then open", f: (a, w, h, se, o) => dilateGrid(erodeGrid(erodeGrid(dilateGrid(a, w, h, se, o), w, h, se, o), w, h, se, o), w, h, se, o) },
  },
  closeopen: {
    id: "closeopen", label: "close, then open",
    stages: [D("closing: dilation"), E("closing: erosion"), E("opening: erosion"), D("opening: dilation")],
    compare: { label: "open, then close", f: (a, w, h, se, o) => erodeGrid(dilateGrid(dilateGrid(erodeGrid(a, w, h, se, o), w, h, se, o), w, h, se, o), w, h, se, o) },
  },
  opentwice: {
    id: "opentwice", label: "opening, twice",
    stages: [E("1st opening: erosion"), D("1st opening: dilation"), E("2nd opening: erosion"), D("2nd opening: dilation")],
    compare: { label: "one opening", f: (a, w, h, se, o) => dilateGrid(erodeGrid(a, w, h, se, o), w, h, se, o) },
  },
};

/** Apply the stages of an op without frames (for the dual and for tests). */
export function applyStages(stages: { kind: Kind; se?: Off[] }[], a: Uint8Array, w: number, h: number, se: Off[], outside: number): Uint8Array {
  let s = a;
  for (const st of stages) s = (st.kind === "erode" ? erodeGrid : dilateGrid)(s, w, h, st.se ?? se, outside);
  return s;
}

/** The dual expression: complement, swap erosion/dilation, reflect the SE, swap the border value, complement. */
export function dualOf(stages: { kind: Kind; se?: Off[] }[], a: Uint8Array, w: number, h: number, se: Off[], outside: number): Uint8Array {
  const sw = stages.map(s => ({ kind: (s.kind === "erode" ? "dilate" : "erode") as Kind, se: reflect(s.se ?? se) }));
  return complement(applyStages(sw, complement(a), w, h, se, 1 - outside));
}

const codes = (I: ArrayLike<number>, S: ArrayLike<number>) => Int32Array.from(I, (v, i) => (v ? 1 : 0) + 2 * (S[i] ? 1 : 0));
const count = (a: ArrayLike<number>) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i]) n++; return n; };
const diff = (a: ArrayLike<number>, b: ArrayLike<number>) => { let n = 0; for (let i = 0; i < a.length; i++) if (!a[i] !== !b[i]) n++; return n; };

function* runOp(op: Op, inp: Uint8Array, w: number, h: number, se: Off[], outside: number, seName: string, dualOn: boolean): Generator<GridFrame> {
  const dual = dualOn && op.dual ? dualOf(op.stages, inp, w, h, se, outside) : undefined;
  const plain = codes(inp, inp);
  yield {
    values: plain, aux: dual, log: `A: ${count(inp)} pixels; B = ${seName}; outside the grid = ${outside}`,
    note: "dark: in the set · red: removed · green: added. Dots on the SE cells: filled = on a set pixel.",
  };
  let src = inp;
  for (const st of op.stages) {
    const b = st.se ?? se, out = new Uint8Array(w * h), base = codes(inp, src), marks: CellMark[] = [];
    const win: [number, number][] = (st.kind === "erode" ? b : reflect(b)).map(([dx, dy]) => [dy, dx]);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, vals = probe(src, w, h, x, y, b, st.kind, outside), on = count(vals);
      const v = st.kind === "erode" ? (on === vals.length ? 1 : 0) : (on > 0 ? 1 : 0);
      out[i] = v;
      if (v !== src[i]) marks.push({ index: i, color: v ? GREEN : RED });
      const why = st.kind === "erode"
        ? (v ? `all ${vals.length} on the set → 1` : `${vals.length - on} of ${vals.length} miss → 0`)
        : (v ? `${on} hit${on > 1 ? "s" : ""} → 1` : `no hit → 0`);
      yield {
        values: base, aux: dual, focus: i, window: { at: i, offsets: win, hits: vals.map(u => u === 1) }, marks: marks.slice(),
        log: `(${y}, ${x}): ${why}`, stage: x === 0 ? `${st.name}, row ${y}` : undefined,
        note: st.kind === "erode" ? "Erosion: does B, placed with its origin here, fit inside the set?" : "Dilation: does the reflected B̌, placed here, hit the set?",
      };
    }
    src = out;
    yield { values: codes(inp, src), aux: dual, log: `${st.name}: ${count(src)} pixels`, stage: `after ${st.name}` };
  }
  let res = src;
  if (op.final === "boundary") res = Uint8Array.from(inp, (v, i) => (v && !src[i] ? 1 : 0));
  if (op.final === "complement") res = complement(inp);
  const notes: string[] = [];
  if (op.final === "boundary") notes.push(`boundary: ${count(res)} pixels (dark); the eroded interior is red.`);
  if (op.final === "complement") notes.push("Aᶜ: every pixel flips.");
  if (op.compare) {
    const c = op.compare.f(inp, w, h, se, outside), d = diff(c, res);
    notes.push(d === 0 ? `Same as ${op.compare.label}.` : `${op.compare.label} differs in ${d} pixel${d > 1 ? "s" : ""}.`);
  }
  if (dual) {
    const d = diff(dual, res);
    notes.push(d === 0 ? `The dual ${op.dual} (right) is identical: duality.` : `The dual ${op.dual} differs in ${d} pixels.`);
  }
  yield {
    values: codes(inp, res), aux: dual,
    log: `result: ${count(res)} pixels (${diff(res, inp)} changed)`, stage: "result",
    note: notes.join(" ") || `${count(inp)} → ${count(res)} pixels.`,
  };
}

// ---------------------------------------------------------------------------------------
// presets (text rows: 1 = object)

const G = {
  fits: ["000000000000", "011111000000", "011111000000", "011111000000", "011111001110", "000000001110",
    "000000001110", "001000001110", "001000000000", "001111111100", "000000000000", "000000000000"],
  layer: ["000000000000", "000000000000", "000000000000", "000111110000", "000111110000", "000110110000",
    "000111110000", "000000000000", "000000001000", "000000000000", "000000000000", "000000000000"],
  single: ["00000000", "00000000", "00000000", "00010000", "00000000", "00000110", "00000100", "00000000"],
  duality: ["000000000000", "001111100000", "001111110000", "001111111000", "000111111000", "000011111000",
    "000001111000", "000000000000", "000000011100", "000000011100", "000000000000", "000000000000"],
  border: ["111110000000", "111110000000", "111110000000", "111110000000", "000000000000", "000000000000",
    "000000011110", "000000011110", "000000011110", "000000011110", "000000000000", "000000000000"],
  hand: ["0000000", "0111100", "0111100", "0110000", "0111110", "0000110", "0000000"],
  speck: ["000000000000", "011111100000", "011111100000", "011111111110", "011111100000", "011111100000",
    "000000000000", "000000000100", "000110000000", "000000000000", "000000000000", "000000000000"],
  gap: ["000000000000", "011111111110", "011111111110", "011101111110", "011111111110", "011111011110",
    "011111011110", "011111111110", "000000000000", "011110111110", "011110111110", "000000000000"],
  noisy: ["000000000000", "011111110000", "011101110010", "011111110000", "011111111100", "010111101100",
    "011111111100", "000000111100", "001000110100", "000000111100", "000000000000", "000100000000"],
  hand2: ["0000000", "0111110", "0111110", "0111011", "0111110", "0100000", "0000000"],
};

function binaryMode(id: string, opIds: string[], presets: (se: ReturnType<typeof seEditor>, setOutside: (o: number) => void, setDual: (v: boolean) => void) => ModePreset[]) {
  registerMode({
    id,
    build(ctx: ModeContext): ModeSpec {
      let outside = 0, dualOn = false;
      const se = seEditor(ctx.side, () => ctx.restart());
      htmlEl("div", "mg-lab", "outside the grid", ctx.side);
      const brow = htmlEl("div", "qb-presets", "", ctx.side);
      const bb = [0, 1].map(v => button(brow, `= ${v}`, () => setOutside(v)));
      const setOutside = (v: number) => { outside = v; bb.forEach((b, k) => b.classList.toggle("active", k === v)); safeRestart(); };
      const dualBox = checkbox(ctx.side, "show duality", false, v => { dualOn = v; safeRestart(); });
      const setDual = (v: boolean) => { dualOn = v; dualBox.checked = v; };
      let ready = false;
      function safeRestart(): void { if (ready) ctx.restart(); }
      bb[0].classList.add("active");
      const algorithms: GridAlgorithm[] = opIds.map(k => {
        const op = OPS[k];
        return {
          id: op.id, label: op.label, auxDisplay: "binary", auxTitle: op.dual,
          run: (input: Int32Array, rows: number, cols: number) =>
            runOp(op, Uint8Array.from(input, v => v & 1), cols, rows, se.offsets(), outside, se.name(), dualOn),
        };
      });
      return {
        algorithms, display: "int", numbers: false, lo: 0, hi: 3, cell: 36,
        map: (t: number) => PAL[Math.round(t * 3)],   // code / 3 → colour
        edit: (old: number) => ((old & 1) ? 0 : 3),
        presets: presets(se, setOutside, setDual),
        ready() { ready = true; },
      };
    },
  });
}

binaryMode("binary", ["erode", "dilate", "open", "close", "boundary", "complement", "twolines"], (se, setOutside, setDual) => [
  { id: "fits", label: "erosion: where it fits", grid: G.fits, algorithm: "erode", apply: () => { se.set("cross"); setOutside(0); setDual(false); } },
  { id: "layer", label: "dilation adds a layer", grid: G.layer, algorithm: "dilate", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "reflect", label: "the L-shape: why reflect", grid: G.single, algorithm: "dilate", apply: () => { se.set("L"); setOutside(0); setDual(false); } },
  { id: "duality", label: "duality", grid: G.duality, algorithm: "erode", apply: () => { se.set("L"); setOutside(0); setDual(true); } },
  { id: "twolines", label: "two lines make a square", grid: G.layer, algorithm: "twolines", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "border", label: "border convention", grid: G.border, algorithm: "erode", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "hand", label: "7 × 7 (exercise)", grid: G.hand, algorithm: "erode", apply: () => { se.set("L"); setOutside(0); setDual(false); } },
  { id: "blank", label: "empty 16 × 16", grid: { rows: 16, cols: 16, values: new Int32Array(256) }, algorithm: "dilate", apply: () => { se.set("cross"); setOutside(0); setDual(false); } },
]);

binaryMode("openclose", ["erode", "dilate", "open", "close", "openclose", "closeopen", "opentwice"], (se, setOutside, setDual) => [
  { id: "speck", label: "opening removes the speck", grid: G.speck, algorithm: "open", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "gap", label: "closing fills the gap", grid: G.gap, algorithm: "close", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "order", label: "open∘close ≠ close∘open", grid: G.noisy, algorithm: "openclose", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "twice", label: "apply twice: nothing changes", grid: G.noisy, algorithm: "opentwice", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "hand", label: "7 × 7 (exercise)", grid: G.hand2, algorithm: "open", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
  { id: "blank", label: "empty 16 × 16", grid: { rows: 16, cols: 16, values: new Int32Array(256) }, algorithm: "open", apply: () => { se.set("square"); setOutside(0); setDual(false); } },
]);

export const BINARY_PALETTE = PAL;
export const GRIDS = G;
