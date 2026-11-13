// Exports the lsunmix demo's lambda stack and unmixing results (same model, seed 1) for
// figs.py. Bundle and run from the repository root:
//   node_modules/.bin/esbuild tools/figs/lsunmix/export.ts --bundle --platform=node --outfile=<tmp>/x.js
//   node <tmp>/x.js <outdir>      (writes <outdir>/lsunmix.json)
import { writeFileSync } from "fs";
import * as U from "../../../demos/src/lib/unmix";
import { dyesFor, truthMaps, simulate, unmix, residualChi, rmse, negativeShare, P, L0, L1 } from "../../../demos/src/lsunmix/model";

const out = process.argv[2] ?? ".";
const A = truthMaps(), arr = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1e4) / 1e4);
const res: Record<string, unknown> = { truth: arr(A) };
const wl: number[] = []; for (let l = L0 - 10; l <= L1 + 10; l += 1) wl.push(l);
res.wl = wl;
for (const sep of [45, 6]) res["em" + sep] = dyesFor(sep).map(d => wl.map(l => U.emission(d, l)));
// the default view: 16 channels, peaks 45 nm apart, 400 photons
{
  const { bands, M, Y } = simulate(A, 16, 45, 400, 1);
  res.bands = bands; res.M = arr(M); res.Y = arr(Y);
  const Pi = U.pseudoInverse(M, 16, 4)!; res.Pi = arr(Pi);
  const r4 = unmix(Y, M, 16, 4, "ls", 400), r3 = unmix(Y, M, 16, 3, "ls", 400);
  res.chi4 = arr(residualChi(r4.R, Y, 16)); res.chi3 = arr(residualChi(r3.R, Y, 16));
  res.Ah = arr(r4.Ah);
  console.log("16ch sep45 400ph LS: rmse", [0, 1, 2, 3].map(k => rmse(r4.Ah, A, k).toFixed(3)).join(" "), "neg share", negativeShare(r4.Ah, 4).toFixed(3));
  console.log("  without auto: rmse mito", rmse(r3.Ah, A, 1).toFixed(3));
}
for (const ph of [20]) {
  const { M, Y } = simulate(A, 16, 45, ph, 1), r = unmix(Y, M, 16, 4, "ls", ph);
  res.Ah20 = arr(r.Ah);
  console.log(`16ch sep45 ${ph}ph LS neg share`, negativeShare(r.Ah, 4).toFixed(3), "rmse", [0, 1, 2, 3].map(k => rmse(r.Ah, A, k).toFixed(3)).join(" "));
}
for (const sep of [6, 45]) for (const n of [8, 16, 32]) {
  const bands = U.contiguousBands(L0, L1, n), M = U.channelMatrix(dyesFor(sep), bands);
  console.log(`sep ${sep} n ${n}: cond ${U.condition(M, n, 4).toFixed(2)}`);
}
{ const bands = U.contiguousBands(L0, L1, 16); res.M6 = arr(U.channelMatrix(dyesFor(6), bands)); }
writeFileSync(out + "/lsunmix.json", JSON.stringify(res));
