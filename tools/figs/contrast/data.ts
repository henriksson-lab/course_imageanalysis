// Dumps the contrast demos' synthetic images and CLAHE results (lib/contrast.ts, fixed
// seeds) as raw little-endian arrays for figs.py. Run from the repository root:
//   TMP=$(mktemp -d)
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/contrast/data.ts --outfile=$TMP/data.js && node $TMP/data.js $TMP
//   python3 -I tools/figs/contrast/figs.py $TMP
import { writeFileSync } from "fs";
import * as C from "../../../demos/src/lib/contrast";

const out = process.argv[2] ?? ".";
const fluo = C.fluorescence12();
writeFileSync(`${out}/fluo_u16.bin`, Buffer.from(fluo.buffer));
writeFileSync(`${out}/foggy_u8.bin`, Buffer.from(C.foggyScene().buffer));
writeFileSync(`${out}/faint_u8.bin`, Buffer.from(C.faintCellScene().buffer));
const img = { width: 256, height: 256, data: Float32Array.from(fluo) };
let lo = Infinity, hi = -Infinity;
for (const v of img.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
for (const [name, interp] of [["clahe_interp", true], ["clahe_seams", false]] as [string, boolean][]) {
  const r = C.clahe(img, { tilesX: 4, tilesY: 4, clip: 3, bins: 256, range: [lo, hi + 1e-6], interpolate: interp });
  writeFileSync(`${out}/${name}_f32.bin`, Buffer.from(r.out.data.buffer));
  if (interp) writeFileSync(`${out}/clahe_luts_f64.bin`, Buffer.from(Float64Array.from(r.tileLuts.flatMap(l => Array.from(l))).buffer));
}
writeFileSync(`${out}/range.json`, JSON.stringify({ lo, hi }));
console.log("wrote", out, lo, hi);
