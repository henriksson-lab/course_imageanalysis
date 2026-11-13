// Exports the bleedthrough demo's synthetic cell (the same seed and photons as the demo)
// for figs.py. Bundle and run from the repository root:
//   node_modules/.bin/esbuild tools/figs/crosstalk/export.ts --bundle --platform=node --outfile=<tmp>/x.js
//   node <tmp>/x.js <outdir>      (writes <outdir>/crosstalk.json)
import { writeFileSync } from "fs";
import { imageData, compensate, summary } from "../../../demos/src/bleedthrough/model";

const out = process.argv[2] ?? ".";
const D = imageData(0.25, 0.05, 1000, 1, true);   // demo defaults: true S hidden = (0.25, 0.05)
const res: Record<string, unknown> = { w: D.w, h: D.h, x: Array.from(D.x), y: Array.from(D.y), g: Array.from(D.g) };
for (const g21 of [0, 0.15, 0.25, 0.35]) {
  const g12 = g21 === 0 ? 0 : 0.05;
  const c = compensate(D, g21, g12), s = summary(D, c.cx, c.cy);
  res["g" + g21] = { cx: Array.from(c.cx), cy: Array.from(c.cy), med1: s.med1, bgY: s.bgY, slope1: s.slope1, pearson: s.pearson, sdBg: s.sdBg };
  console.log(g21, g12, "median dye1-pop in ch2:", s.med1.toFixed(1), "bg:", s.bgY.toFixed(1), "slope:", s.slope1.toFixed(3), "r:", s.pearson.toFixed(3), "sdBg:", s.sdBg.toFixed(1));
}
writeFileSync(out + "/crosstalk.json", JSON.stringify(res));
