// Lifetime maps of the flimcell demo's "quencher" scene (demos/src/lib/decay.ts flimScene) for
// lifetime/fig-display.png, with the demo's estimator (mean arrival minus the IRF's mean).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/lifetime/scene.ts --bundle --platform=node --log-level=warning --outfile=<tmp>/scene.js && node <tmp>/scene.js <tmp>/scene.json
// then  python3 -I tools/figs/lifetime/figs.py <tmp>/scene.json
import * as fs from "fs";
import * as D from "../../../demos/src/lib/decay";

const out: any = {};
const sc = D.flimScene("quencher", { photons: 60 });
const n = sc.n, P = n * n, nb = sc.ax.nBins, ts = D.binCentres(sc.ax), irfMean = D.meanArrival(sc.irf, sc.ax);
out.n = n;
for (const r of [0, 1]) {
  const st = D.binStack(sc.stack, n, nb, r);
  const I = new Float32Array(P), tau = new Float32Array(P);
  for (let p = 0; p < P; p++) { let a = 0; for (let j = 0; j < nb; j++) { I[p] += st[j * P + p]; a += st[j * P + p] * ts[j]; } tau[p] = I[p] > 0 ? a / I[p] - irfMean : NaN; }
  out["I" + r] = Array.from(I); out["tau" + r] = Array.from(tau, v => (Number.isFinite(v) ? +v.toFixed(4) : null));
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
console.log("photons: max", Math.max(...out.I0), "binned max", Math.max(...out.I1));
