// Data for icp/lec.qmd, computed with the demo's own ICP code and settings (demos/src/icp/icp.ts):
// the close start (10°), the local minimum (40°), partial overlap with plain and trimmed ICP,
// the outline, and the sweeps over the rotation between the sets.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/icp/data.ts --bundle --platform=node --outfile=<tmp>/icp-data.js
//   node <tmp>/icp-data.js <tmp>/icp-data.json
// then python3 -I tools/figs/icp/figs.py <tmp>/icp-data.json
import { writeFileSync } from "fs";
import * as I from "../../../demos/src/icp/icp";

declare const process: { argv: string[] };

const BASE: I.DemoSettings = { scene: "spots", kind: "rigid", angle: 10, noise: 1, unpaired: 0, keep: 1, centroids: false, startShift: [0, 0] };
const r2 = (v: number) => Math.round(v * 100) / 100;
const pts = (P: I.Pt[]) => P.map(p => [r2(p.x), r2(p.y)]);

function dump(s: Partial<I.DemoSettings>) {
  const st = { ...BASE, ...s };
  const { sc, res, err } = I.demoRun(st);
  return {
    F: pts(sc.F), M: pts(sc.M), partner: Array.from(sc.partner), truth: pts(I.applyAll(sc.truth, sc.M)),
    steps: res.steps.map((k, j) => ({ moved: pts(I.applyAll(k.T, sc.M)), match: Array.from(k.match), kept: k.kept, rms: r2(k.rms), err: r2(err[j]), angle: r2(I.angleOf(k.T) * 180 / Math.PI) })),
    final: pts(I.applyAll(res.T, sc.M)),
  };
}
const SW = Array.from({ length: 73 }, (_, i) => -180 + 5 * i);
writeFileSync(process.argv[2], JSON.stringify({
  close: dump({}),
  far: dump({ angle: 40 }),
  overlap: dump({ unpaired: 0.3 }),
  trimmed: dump({ unpaired: 0.3, keep: 0.7 }),
  outline: dump({ scene: "outline", angle: 30 }),
  sweep: { angles: SW, spots: I.sweep(BASE, SW).map(r2), outline: I.sweep({ ...BASE, scene: "outline" }, SW).map(r2) },
}));
