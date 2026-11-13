// Data for the rect-features figures, computed with the course's demo code.
//   node_modules/.bin/esbuild tools/figs/rect-features/data.ts --bundle --platform=node --outfile=<tmp>/data.js
//   node <tmp>/data.js <cell.json> <out.json>
import * as fs from "fs";
import { boxFeature, poolSize, BoxFeature } from "../../../demos/src/lib/integral";

const N = 256;
const cell: number[] = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out: Record<string, unknown> = {};
const feats: BoxFeature[] = ["mean", "sd", "dob"];
for (const f of feats) for (const s of [1, 4, 16]) out[`${f}${s}`] = Array.from(boxFeature(cell, N, N, f, s));
// pool sizes in a 24 x 24 window: template = a x b cells (across x down)
out.pool = {
  "edge across": poolSize(2, 1, 24, 24), "edge down": poolSize(1, 2, 24, 24),
  "line across": poolSize(3, 1, 24, 24), "line down": poolSize(1, 3, 24, 24),
  "diagonal": poolSize(2, 2, 24, 24), "centre − surround": poolSize(3, 3, 24, 24),
};
fs.writeFileSync(process.argv[3], JSON.stringify(out));
