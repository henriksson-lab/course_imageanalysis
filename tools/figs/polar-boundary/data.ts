// Data for the polar-boundary figures, computed with the demo's own code (demos/src/polarseg/model.ts).
// Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/polar-boundary/data.ts --bundle --platform=node --loader:.png=dataurl --outfile=<tmp>/polar-data.js
//   node <tmp>/polar-data.js <tmp>/polar-data.json
import { writeFileSync } from "fs";
import { nucleusImage, NUCLEUS, segmentPolar, PolarSettings } from "../../../demos/src/polarseg/model";
import { gaussianBlur } from "../../../demos/src/lib/imageops";

const raw = nucleusImage();
const smooth = gaussianBlur(raw, 1.5);
const base: PolarSettings = { cx: NUCLEUS.cx, cy: NUCLEUS.cy, K: 120, R: 110, rMin: 5, rMax: 105,
  delta: 1, alpha: 0.1, polarity: "bright-dark", closure: "exact", sigma: 1.5 };
const runs: Record<string, Partial<PolarSettings>> = {
  done: {},
  nosmooth: { delta: 8, alpha: 0 },
  open: { closure: "open", delta: 2, alpha: 0 },
  unroll: { closure: "unroll", delta: 2, alpha: 0 },
  exact: { closure: "exact", delta: 2, alpha: 0 },
};
const out: any = { size: raw.width, raw: Array.from(raw.data), smooth: Array.from(smooth.data), runs: {} };
for (const [k, s] of Object.entries(runs)) {
  const r = segmentPolar(smooth, { ...base, ...s });
  out.runs[k] = { settings: { ...base, ...s }, path: Array.from(r.dp.path), gap: r.dp.gap, closes: r.dp.closes,
    closedCost: r.dp.closedCost, openCost: r.dp.openCost, contour: r.contour,
    polar: Array.from(r.polar.data), polarW: r.polar.width, polarH: r.polar.height, cost: Array.from(r.cost) };
}
writeFileSync(process.argv[2], JSON.stringify(out));
