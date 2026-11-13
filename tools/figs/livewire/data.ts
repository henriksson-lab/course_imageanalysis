// Data for the livewire figures, computed with the demo's own code (dpgrid presets, lib/dp.ts,
// lib/shortestpath.ts). Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/livewire/data.ts --bundle --platform=node --outfile=<tmp>/lw-data.js
//   node <tmp>/lw-data.js <tmp>/lw-data.json
import { writeFileSync } from "fs";
import { DPGRID_PRESETS, LW_PRESETS } from "../../../demos/src/dpgrid/grids";
import { pathDP, verticalSteps } from "../../../demos/src/lib/dp";
import { dijkstraTrace, vertexCost, tracePath } from "../../../demos/src/lib/shortestpath";

const out: any = {};
const det = DPGRID_PRESETS.find(p => p.id === "detour")!;
const dp = pathDP(det.values, det.rows, det.cols, { steps: verticalSteps(1) });
const srcRow = Array.from({ length: det.cols }, (_, j) => ({ index: j, value: det.values[j] }));
const dj = dijkstraTrace({ width: det.cols, height: det.rows, conn: 4, edge: vertexCost(det.values), sources: srcRow }).result;
let best = -1;
for (let j = 0; j < det.cols; j++) { const q = (det.rows - 1) * det.cols + j; if (best < 0 || dj.dist[q] < dj.dist[best]) best = q; }
out.detour = { rows: det.rows, cols: det.cols, C: det.values, dpPath: Array.from(dp.path), dpCost: dp.cost,
  djPath: tracePath(dj.pred, best), djCost: dj.dist[best] };
for (const id of ["exercise", "wall"]) {
  const p = LW_PRESETS.find(q => q.id === id)!;
  const t = dijkstraTrace({ width: p.cols, height: p.rows, conn: 4, edge: vertexCost(p.values), sources: [{ index: 0, value: p.values[0] }] });
  const target = p.target === "bl" ? (p.rows - 1) * p.cols : p.rows * p.cols - 1;
  out[id] = { rows: p.rows, cols: p.cols, C: p.values, dist: Array.from(t.result.dist), pred: Array.from(t.result.pred),
    order: Array.from(t.result.order), events: t.events, target, path: tracePath(t.result.pred, target) };
}
writeFileSync(process.argv[2], JSON.stringify(out));
