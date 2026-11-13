// Computes the data behind the box-filters figures with the course's own demo code, so
// that the numbers match the localstats demo. Bundle and run (from the repo root):
//   node_modules/.bin/esbuild tools/figs/box-filters/data.ts --bundle --platform=node --outfile=<tmp>/data.js
//   node <tmp>/data.js <cell.json> <out.json>
// cell.json: the micrograph as a flat array of 256*256 grey values (written by figs.py).
import * as fs from "fs";
import { synthImage, N } from "../../../demos/src/localstats/model";
import { localStats, contrastNormalise } from "../../../demos/src/lib/integral";

const cell: number[] = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out: Record<string, unknown> = {};
const synth = synthImage();
out.synth = Array.from(synth);
// preset "equalise": r = 8, eps = 4, float64
const ls = localStats(synth, N, N, 8);
out.synthCn = Array.from(contrastNormalise(synth, ls, 4));
// preset "cell": r = 4
const lc = localStats(cell, N, N, 4);
out.cellMean = Array.from(lc.mean);
out.cellSd = Array.from(lc.sd);
// preset "negative" / "offset": synth + 200, r = 3, float32
const img = Float64Array.from(synth, v => v + 200);
let sum = 0;
for (const v of img) sum += v;
const offset = Math.round(sum / img.length);
const exact = localStats(img, N, N, 3, { offset });
for (const sub of [false, true]) {
  const got = localStats(img, N, N, 3, { kind: "f32", offset: sub ? offset : 0 });
  const err: number[] = [];
  let emax = 0;
  for (let i = 0; i < got.sd.length; i++) {
    const e = Math.abs(got.sd[i] - exact.sd[i]);
    emax = Math.max(emax, e);
    err.push(got.variance[i] < 0 ? -1 : e);
  }
  out[sub ? "offset" : "negative"] = { err, emax, negatives: got.negatives };
}
out.offsetValue = offset;
fs.writeFileSync(process.argv[3], JSON.stringify(out));
