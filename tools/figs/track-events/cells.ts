// Simulates the gapclose demo's "censor" preset (demos/src/gapclose/model.ts) and prints the true
// cells as JSON, for fig-censor.svg. Called by figs.py through esbuild + node.
import { simulate, trueCycles, PRESETS, DEFAULT, T } from "../../../demos/src/gapclose/model";
const p = PRESETS.find(q => q.id === "censor")!;
const mv = simulate({ ...DEFAULT, ...p.s });
console.log(JSON.stringify({ T, cells: mv.cells.map(c => ({ id: c.id, parent: c.parent, t0: c.t0, t1: c.t1, divides: c.divides })), cycles: trueCycles(mv) }));
