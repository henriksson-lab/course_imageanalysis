// Data for the label-connectivity figures, from the course's demo code (morphgrid "label").
//   run via figs.py (esbuild --platform=node), argument: output JSON path
import * as fs from "fs";
import { floodFrames } from "../../../demos/src/morphgrid/mode-label";

// the "queue rings vs stack arms" preset: a 7 x 7 block in a 9 x 9 grid, objects 4-connected
const rings = ["000000000", "011111110", "011111110", "011111110", "011111110", "011111110", "011111110", "011111110", "000000000"];
const w = 9, h = 9, inp = Uint8Array.from(rings.join(""), c => +c);
const out: Record<string, unknown> = { w, h, mask: Array.from(inp) };
for (const kind of ["bfs", "dfs"] as const) {
  const order: number[] = [];
  let note = "";
  let maxLen = 0;
  for (const f of floodFrames(inp, w, h, 4, kind)) {
    if ((f as any).window) order.push((f as any).focus);
    if ((f as any).marks) maxLen = Math.max(maxLen, (f as any).marks.length);
    note = (f as any).note ?? note;
  }
  out[kind] = { order, note };
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
