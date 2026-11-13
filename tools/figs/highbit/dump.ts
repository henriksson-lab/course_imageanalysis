// Dumps the window16 demo's synthetic images (the exact demo code, fixed seeds) as raw
// little-endian arrays for figs.py. Run from the repository root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/highbit/dump.ts --outfile=$TMP/dump.js && node $TMP/dump.js $TMP
import { writeFileSync } from "fs";
import { fluorescence, ctPhantom } from "../../../demos/src/window16/data";
const out = process.argv[2] ?? ".";
writeFileSync(`${out}/fluo_u16.bin`, Buffer.from(fluorescence().buffer));
writeFileSync(`${out}/ct_i16.bin`, Buffer.from(ctPhantom().buffer));
console.log("wrote", out);
