// Image data for tools/figs/heat-equation/figs.py (run by it; prints JSON). Uses lib/pde.ts.
import { heatSteps, gaussianHeat, testCard, addNoise, heatStep } from "../../../demos/src/lib/pde";
declare const require: (m: string) => any; declare const process: { argv: string[] };
const fs = require("fs");
const cell = Float64Array.from(JSON.parse(fs.readFileSync(process.argv[2], "utf8")) as number[]);
const r = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 100) / 100);
const out: Record<string, unknown> = {};
// heat = Gaussian: 10, 40, 160 steps of Δt = 0.2 (t = 2, 8, 32; σ = 2, 4, 8)
out.heat = [10, 40, 160].map(n => { const h = heatSteps(cell, 128, 128, 0.2, n), g = gaussianHeat(cell, 128, 128, Math.sqrt(2 * 0.2 * n)); return { n, h: r(h), g: r(g) }; });
out.cell = r(cell);
// checkerboard: noisy test card at 1.2 × the limit (Δt = 0.3)
const card = addNoise(testCard().truth, 12, 5);
let u: Float64Array = Float64Array.from(card); const cb: number[][] = [r(u)];
for (let k = 1; k <= 12; k++) { u = heatStep(u, 128, 128, 0.3); if (k % 4 === 0) cb.push(r(u)); }
out.checker = cb;
console.log(JSON.stringify(out));
