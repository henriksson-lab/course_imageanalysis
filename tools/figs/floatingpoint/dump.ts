// Dumps the floatbits demo's summation data (the exact demo code: pixels mean 1000, sd 50,
// seed 21, and the running float16 / float32 sums) for figs.py. Run from the repository root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/floatingpoint/dump.ts --outfile=$TMP/dump.js && node $TMP/dump.js $TMP
import { writeFileSync } from "fs";
import { pixels, sumTable } from "../../../demos/src/floatbits/sum";
import { F32, F16, rounder } from "../../../demos/src/floatbits/fp";
const out = process.argv[2] ?? ".";
const n = 2 ** 22;
const px = pixels(n, 1000, 50);
// running sums every 1024 pixels: exact, float32 naive, float16 naive (normalised, x 65535)
const r32 = rounder(F32), r16 = rounder(F16);
const rows: number[][] = [];
let ex = 0, s32 = 0, s16 = 0;
for (let i = 0; i < n; i++) {
  ex += px[i]; s32 = r32(s32 + px[i]); s16 = r16(s16 + r16(px[i] / 65535));
  if ((i + 1) % 1024 === 0) rows.push([i + 1, ex, s32, s16]);
}
writeFileSync(`${out}/runsum.json`, JSON.stringify({ rows, table: sumTable(px), first: Array.from(px.subarray(0, 32768)) }));
console.log("wrote", out);
