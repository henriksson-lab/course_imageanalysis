// Data for the viola-jones figures, computed with the boostdetect demo code.
//   node_modules/.bin/esbuild tools/figs/viola-jones/data.ts --bundle --platform=node --outfile=<tmp>/data.js
//   node <tmp>/data.js <out.json>
import * as fs from "fs";
import { makeDataset, testScene } from "../../../demos/src/boostdetect/synth";
import { makePool, featureMatrix, detect, matches } from "../../../demos/src/boostdetect/model";
import { AdaBoost } from "../../../demos/src/lib/boost";
import { DEFAULTS, VARIANTS } from "../../../demos/src/boostdetect/defaults";
import { TRAINED } from "../../../demos/src/boostdetect/trained";

const out: Record<string, unknown> = {};
const pool = makePool();
const v = VARIANTS.find(x => x.id === "faces")!;
const train = makeDataset("faces", DEFAULTS.nPos, DEFAULTS.nNeg, v.seed, { glasses: "random" });
const FM = featureMatrix(pool, train).M;
const ab = new AdaBoost(FM, train.y);
const s1 = ab.round();
out.stump1 = { ...s1, spec: pool[s1.feature], values: Array.from(FM.values.slice(s1.feature * FM.nEx, (s1.feature + 1) * FM.nEx)) };
out.y = Array.from(train.y);
out.poolSize = pool.length;
// a few training patches of each class
out.posPatches = [0, 1, 2, 3, 4, 5].map(i => Array.from(train.patches[i]));
out.negPatches = [0, 1, 2, 3, 4, 5].map(i => Array.from(train.patches[train.nPos + i]));
// scenes and detections with the precomputed cascade
for (const rotated of [false, true]) {
  const sc = testScene("faces", 5, { rotated });
  const r = detect(sc.img, 256, 256, pool, TRAINED.faces.cascade, null);
  out[rotated ? "rotated" : "plain"] = {
    img: Array.from(sc.img), targets: sc.targets,
    merged: r.merged.map(m => ({ ...m, ok: sc.targets.some(t => matches(m, t)) })),
    found: sc.targets.map(t => r.merged.some(m => matches(m, t))),
    windows: r.windows, featuresPerWindow: r.totalFeatures / r.windows,
  };
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
