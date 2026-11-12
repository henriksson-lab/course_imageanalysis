// morphgrid mode "distance" (unit morph-openclose): the two-pass distance transform, step
// by step. Left: the mask (click to edit). Right: the distances; the forward pass reads the
// neighbours above and to the left, the backward pass those below and to the right, each
// "min over neighbours + 1". The slider r marks {DT > r} = the erosion by the diamond
// (city-block) or square (chessboard) of radius r.
import { htmlEl, slider } from "../lib/ui";
import { GridAlgorithm, GridFrame, CellMark } from "../lib/gridui";
import { registerMode, ModeContext, ModeSpec } from "./registry";
import { FORWARD, INF } from "./ops";

type Metric = "cityblock" | "chessboard";

function* runDT(inp: Uint8Array, w: number, h: number, metric: Metric, r: number): Generator<GridFrame> {
  const d = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = inp[i] ? INF : 0;
  yield { values: inp, aux: d, log: "start: 0 on the background, ∞ on the object", note: "DT(p) = number of steps from p to the nearest background pixel." };
  for (const s of [1, -1]) {
    const pass = s > 0 ? "forward pass" : "backward pass";
    const nb = FORWARD[metric].map(([dx, dy]) => [s * dx, s * dy] as [number, number]);
    let rowSeen = -1;
    for (let k = 0; k < w * h; k++) {
      const i = s > 0 ? k : w * h - 1 - k, x = i % w, y = (i - x) / w;
      if (!inp[i]) continue;
      const terms: string[] = [d[i] >= INF ? "∞" : String(d[i])];
      const offs: [number, number][] = [[0, 0]], hits: boolean[] = [true];
      let best = d[i];
      for (const [dx, dy] of nb) {
        const xx = x + dx, yy = y + dy;
        offs.push([dy, dx]);
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) { hits.push(false); continue; }
        const v = d[yy * w + xx];
        hits.push(true);
        terms.push(v >= INF ? "∞ + 1" : `${v} + 1`);
        best = Math.min(best, v + 1);
      }
      d[i] = best;
      const stage = y !== rowSeen ? `${pass}, row ${y}` : undefined;
      rowSeen = y;
      yield {
        values: inp, aux: d.slice(), focus: i, auxFocus: i, auxWindow: { at: i, offsets: offs, hits },
        log: `${pass} (${y}, ${x}): min(${terms.join(", ")}) = ${best >= INF ? "∞" : best}`, stage,
        note: s > 0 ? "Forward pass (raster order): neighbours above and to the left are already done." : "Backward pass (reverse order): neighbours below and to the right.",
      };
    }
  }
  let mx = 0;
  for (const v of d) if (v < INF) mx = Math.max(mx, v);
  const marks: CellMark[] = [];
  for (let i = 0; i < w * h; i++) if (inp[i] && d[i] > r) marks.push({ index: i, color: "var(--qb-c3, #2ca02c)" });
  const se = metric === "cityblock" ? "diamond" : "square";
  yield {
    values: inp, aux: d, marks, auxMarks: marks, stage: "result",
    log: `done: largest distance ${mx}; ${marks.length} pixels with DT > ${r}`,
    note: `Green: DT > ${r} = the erosion by the ${se} of radius ${r} (${marks.length} pixels).`,
  };
}

registerMode({
  id: "distance",
  build(ctx: ModeContext): ModeSpec {
    let r = 1, ready = false;
    htmlEl("div", "mg-lab", "erosion = threshold of DT", ctx.side);
    const rs = slider(ctx.side, "r", 0, 6, 1, r, v => { r = v; if (ready) { const s = ctx.stepper(); s.rerun(); } });
    const mk = (metric: Metric, label: string): GridAlgorithm => ({
      id: metric, label, auxDisplay: "int", auxTitle: "distance DT", auxLo: 0, auxHi: 6,
      auxText: v => (v >= INF ? "∞" : String(v)),
      run: (input, rows, cols) => runDT(Uint8Array.from(input, v => (v ? 1 : 0)), cols, rows, metric, r),
    });
    const set = (v: number) => { r = v; rs.set(v); };
    return {
      algorithms: [mk("cityblock", "city-block (4 steps)"), mk("chessboard", "chessboard (8 steps)")],
      display: "binary", edit: "toggle", cell: 40, title: "mask (click to edit)",
      presets: [
        { id: "hand", label: "8 × 8 (exercise)", algorithm: "cityblock", apply: () => set(2), grid: ["00000000", "01111110", "01111110", "01111111", "01111111", "00011111", "00011110", "00000000"] },
        { id: "rect", label: "rectangle", algorithm: "cityblock", apply: () => set(1), grid: ["000000000", "011111110", "011111110", "011111110", "011111110", "011111110", "000000000"] },
        { id: "blob", label: "blob", algorithm: "chessboard", apply: () => set(1), grid: ["0000000000", "0001111000", "0011111100", "0111111110", "0111111110", "0111111110", "0011111100", "0001111000", "0000000000", "0000000000"] },
        { id: "edge", label: "object at the edge", algorithm: "cityblock", apply: () => set(1), grid: ["11111000", "11111000", "11111100", "11111100", "11110000", "00000000"] },
      ],
      ready() { ready = true; },
    };
  },
});
