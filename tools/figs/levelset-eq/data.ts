// Snapshots of moving fronts for levelset-eq/lec.qmd, computed with the curveflow demo's code
// (demos/src/lib/levelset.ts, demos/src/curveflow/shapes.ts), same time steps as its presets.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/levelset-eq/data.ts --bundle --platform=node --outfile=<tmp>/lse-data.js
//   node <tmp>/lse-data.js <tmp>/lse-data.json
// then python3 -I tools/figs/levelset-eq/figs.py <tmp>/lse-data.json
import { writeFileSync } from "fs";
import * as LS from "../../../demos/src/lib/levelset";
import { N, ShapeId, shapePhi } from "../../../demos/src/curveflow/shapes";

interface Run { shape: ShapeId; F: number; eps: number; factor: number; reinit: number; times: number[] }
const round = (a: Float64Array) => Array.from(a, v => Math.round(v * 1000) / 1000);

function run(R: Run): { t: number; steps: number; phi: number[]; pieces: number; grad: number }[] {
  const phi = shapePhi(R.shape), T: LS.SpeedTerms = { F: R.F, eps: R.eps };
  const dt = R.factor * LS.cflLimit(T);
  const out = [];
  let steps = 0, t = 0;
  const work = new Float64Array(N * N);
  for (const target of R.times) {
    while (t < target - 1e-9) {
      LS.lsStep(phi, N, N, T, dt, "upwind", work);
      steps++; t += dt;
      if (R.reinit > 0 && steps % R.reinit === 0) LS.reinitExact(phi, N, N);
    }
    const C = LS.zeroContour(phi, N, N);
    out.push({ t, steps, phi: round(phi), pieces: C.lines.length, grad: LS.meanGradNearFront(phi, N, N) });
    console.log(R.shape, `F=${R.F} eps=${R.eps} reinit=${R.reinit}`, "t", t.toFixed(2), "steps", steps, "pieces", C.lines.length,
      "area", LS.insideArea(phi, N, N).toFixed(1), "mean|grad| near front", LS.meanGradNearFront(phi, N, N).toFixed(2));
  }
  return out;
}

const data = {
  N,
  merge: run({ shape: "two", F: 1, eps: 0, factor: 0.5, reinit: 0, times: [0, 3, 5.6, 6.0, 10] }),
  split: run({ shape: "dumbbell", F: -1, eps: 0, factor: 0.5, reinit: 0, times: [0, 3, 6.0, 10] }),
  nosplit: run({ shape: "dumbbell", F: 0, eps: 1, factor: 0.9, reinit: 25, times: [0, 100, 250, 400] }),
  square: run({ shape: "square", F: 1, eps: 0, factor: 0.5, reinit: 0, times: [0, 8, 16] }),
  star: run({ shape: "star", F: 0, eps: 1, factor: 0.9, reinit: 0, times: [0, 30, 100, 250] }),
  starReinit: run({ shape: "star", F: 0, eps: 1, factor: 0.9, reinit: 10, times: [100] }),
};
writeFileSync(process.argv[2], JSON.stringify(data));
