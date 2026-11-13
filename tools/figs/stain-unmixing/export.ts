// Exports the stainsep demo's synthetic H&E slide (same scene, seed 7, noise 2 grey levels,
// render seed 5, density 1) and its colour deconvolution with Ruifrok's and the true
// vectors, plus Macenko's estimate, for figs.py.
//   node_modules/.bin/esbuild tools/figs/stain-unmixing/export.ts --bundle --platform=node --outfile=<tmp>/x.js
//   node <tmp>/x.js <outdir>      (writes <outdir>/stain.json)
import { writeFileSync } from "fs";
import * as U from "../../../demos/src/lib/unmix";
import { N, I0_TRUE, TRUE_V, RUIF } from "../../../demos/src/stainsep/model";

const out = process.argv[2] ?? ".";
const sc = U.stainScene(N, 7, "he"), Strue = U.stainMatrix(TRUE_V[0], TRUE_V[1], TRUE_V[2]);
const img = U.renderBrightfield(sc, Strue, I0_TRUE, { noise: 2, seed: 5 });
const r = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1e4) / 1e4);
const ruif = U.colourDeconvolve(img.R, img.G, img.B, I0_TRUE, U.stainMatrix(RUIF[0], RUIF[1]))!;
const tru = U.colourDeconvolve(img.R, img.G, img.B, I0_TRUE, U.stainMatrix(TRUE_V[0], TRUE_V[1]))!;
const mac = U.macenko(ruif.od)!;
const ang = (a: number[], b: number[]) => Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;
console.log("Macenko vs true: H", ang(mac.v1, TRUE_V[0]).toFixed(1), "E", ang(mac.v2, TRUE_V[1]).toFixed(1),
  " Ruifrok vs true: H", ang(RUIF[0], TRUE_V[0]).toFixed(1), "E", ang(RUIF[1], TRUE_V[1]).toFixed(1));
writeFileSync(out + "/stain.json", JSON.stringify({
  N, R: Array.from(img.R), G: Array.from(img.G), B: Array.from(img.B),
  conc: sc.conc.map(r), ruif: ruif.c.map(r), tru: tru.c.map(r), od: ruif.od.map(r),
  mac: [mac.v1, mac.v2], trueV: TRUE_V, ruifV: RUIF,
}));
