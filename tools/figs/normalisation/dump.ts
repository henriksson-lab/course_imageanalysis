// Data for the normalisation figures and the numbers on the unit's pages, computed with the demo's
// own code (demos/src/deepstack/model.ts). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/normalisation/dump.ts --bundle --platform=node --outfile=$TMP/no.js && node $TMP/no.js > tools/figs/normalisation/data.json
//   python3 -I tools/figs/normalisation/figs.py
// The plain-vs-residual training curves come from the course code instead:
//   cargo run --manifest-path code/Cargo.toml -p training --features solution -- normalisation
//   cp out/normalisation/plain-vs-residual.csv tools/figs/normalisation/
import { runStack, DEFAULT_STACK, type StackConfig } from "../../../demos/src/deepstack/model";

const S2 = Math.SQRT2;
const cases: [string, Partial<StackConfig>][] = [
  ["relu1", { act: "relu", gain: 1 }], ["relu_he", { act: "relu", gain: S2 }], ["relu2", { act: "relu", gain: 2 }],
  ["tanh05", { act: "tanh", gain: 0.5 }], ["tanh1", { act: "tanh", gain: 1 }], ["tanh3", { act: "tanh", gain: 3 }], ["sigmoid1", { act: "sigmoid", gain: 1 }],
  ["bn2", { act: "relu", gain: 2, norm: "batch" }], ["ln1", { act: "relu", gain: 1, norm: "layer" }],
  ["res_he", { act: "relu", gain: S2, residual: true }], ["ressig", { act: "sigmoid", gain: 1, residual: true }], ["restanh3", { act: "tanh", gain: 3, residual: true }],
  ["linear05", { act: "linear", gain: 0.5 }], ["linear1", { act: "linear", gain: 1 }], ["linear15", { act: "linear", gain: 1.5 }],
];
const out: any = {};
for (const [id, c] of cases) {
  const r = runStack({ ...DEFAULT_STACK, depth: 50, ...c });
  out[id] = { input: r.inputRms, rms: r.layers.map(l => l.rms), grad: r.layers.map(l => l.gradRms), dead: r.layers.map(l => l.dead), sat: r.layers.map(l => l.saturated) };
}
// presets of the demo at its default depth 30
out.presets30 = Object.fromEntries(cases.map(([id, c]) => {
  const r = runStack({ ...DEFAULT_STACK, ...c }), L = r.layers, n = L.length - 1;
  return [id, { rms1: L[0].rms, rms30: L[n].rms, grad1: L[0].gradRms, grad30: L[n].gradRms, dead30: L[n].dead, sat30: L[n].saturated }];
}));
console.log(JSON.stringify(out));
