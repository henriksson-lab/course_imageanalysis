// Trains the flow of the flow2d demo (6 affine coupling layers, two moons) with the demo's own
// code and writes the weights at a few checkpoints to demos/src/flow2d/weights.json, which the
// demo steps through (training it live takes about a minute). Also prints the held-out
// negative log-likelihood and the checks quoted on the pages. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/flows/train.ts --bundle --platform=node --outfile=$TMP/fl.js && node $TMP/fl.js
import { writeFileSync } from "fs";
import { Flow, sampleData, exportWeights, importWeights, K_LAYERS, HIDDEN } from "../../../demos/src/flow2d/model";
import { rng } from "../../../demos/src/lib/ui";

const CHECKPOINTS = [0, 300, 1000, 5000];
const f = new Flow(K_LAYERS, HIDDEN, 1, 3e-3);
const test = sampleData("moons", 2000, rng(5));
const nll = (g: Flow) => { let s = 0; for (let i = 0; i < 2000; i++) s -= g.logDensity(test[2 * i], test[2 * i + 1]); return s / 2000; };
const out: { step: number; nll: number; w: number[][][] }[] = [];
for (let s = 0; s <= CHECKPOINTS[CHECKPOINTS.length - 1]; s++) {
  if (CHECKPOINTS.includes(s)) {
    const w = exportWeights(f), g = new Flow(K_LAYERS, HIDDEN, 1);
    importWeights(g, w);   // the rounded weights, as the demo will see them
    out.push({ step: s, nll: +nll(g).toFixed(4), w });
    console.log(`step ${s}: held-out NLL ${nll(g).toFixed(3)} (unrounded ${nll(f).toFixed(3)})`);
  }
  if (s < CHECKPOINTS[CHECKPOINTS.length - 1]) f.train("moons");
}
writeFileSync("demos/src/flow2d/weights.json", JSON.stringify(out));
