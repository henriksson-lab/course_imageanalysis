// Data for the figures of the unit flows, from the flow2d demo's code and its stored weights.
// Run from the repo root (after train.ts):
//   node_modules/.bin/esbuild tools/figs/flows/dump.ts --bundle --platform=node --outfile=$TMP/f.js && node $TMP/f.js > tools/figs/flows/data.json
//   python3 -I tools/figs/flows/figs.py
import { Flow, sampleData, importWeights, K_LAYERS, HIDDEN } from "../../../demos/src/flow2d/model";
import { rng } from "../../../demos/src/lib/ui";
import W from "../../../demos/src/flow2d/weights.json";

const ck = (W as any[])[3], f = new Flow(K_LAYERS, HIDDEN, 1);
importWeights(f, ck.w);
const data = sampleData("moons", 400, rng(5));
const out: any = { nll: (W as any[]).map(c => [c.step, c.nll]), data: [], layers: [], density: [], samples: [] };
for (let i = 0; i < 400; i++) out.data.push([data[2 * i], data[2 * i + 1]]);
for (let k = 0; k <= K_LAYERS; k++) out.layers.push(out.data.map(([x, y]: number[]) => f.forward(x, y, k).z));
const G = 120, R = 3;
for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) out.density.push(Math.exp(f.logDensity(-R + 2 * R * (c + 0.5) / G, R - 2 * R * (r + 0.5) / G)));
const rn = rng(6);
for (let i = 0; i < 400; i++) out.samples.push(f.inverse(rn.norm(), rn.norm()));
const probe = (x: number, y: number) => { const q = f.forward(x, y); return { x: [x, y], z: q.z, logdet: q.logdet, logp: f.logDensity(x, y) }; };
out.probes = [probe(0.5, 0.3), probe(0.5, 1.2), probe(-1.6, 0.6), probe(2.5, 2.5)];
console.log(JSON.stringify(out, (_k, v) => typeof v === "number" ? +v.toPrecision(4) : v));
