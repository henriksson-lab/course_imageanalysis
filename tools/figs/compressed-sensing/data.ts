// Phase diagrams for compressed-sensing/fig-phase.png, computed with the phasetransition demo's
// code (demos/src/lib/sparse.ts: recoveryTrial, cellSeed). Run by figs.py: node data.js <outdir>
import { writeFileSync } from "fs";
import { recoveryTrial, cellSeed, DONOHO_TANNER_L1 } from "../../../demos/src/lib/sparse";

declare const process: { argv: string[] };
const out = process.argv[2];
const N = 64, G = 20, T = 4;
const res: Record<string, number[]> = {};
for (const solver of ["l1", "omp"] as const) {
  const v: number[] = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const d = (i + 0.5) / G, r = (j + 0.5) / G, M = Math.max(1, Math.round(d * N)), K = Math.max(1, Math.round(r * M));
    let ok = 0;
    for (let t = 0; t < T; t++) if (recoveryTrial(N, M, K, "gaussian", solver, cellSeed(1, d, r, t)).success) ok++;
    v.push(ok / T);
  }
  res[solver] = v;
}
writeFileSync(`${out}/phase.json`, JSON.stringify({ N, G, T, ...res, dt: DONOHO_TANNER_L1 }));
