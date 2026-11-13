// Rejection rates of the pointwise envelope and the MAD test under CSR, against the number of
// radii (the constants PRECOMP_* in demos/src/envelopetest/model.ts). Run from the repo root (~70 s):
//   node_modules/.bin/esbuild tools/figs/envelopes/rates.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../../../demos/src/lib/random";
import { rejectionTrial, PRECOMP_M } from "../../../demos/src/envelopetest/model";
const T = 1000;
console.log("m  pointwise39  pointwise19  MAD");
for (const m of PRECOMP_M) {
  const r = rng(7);
  let a = 0, b = 0, c = 0;
  for (let t = 0; t < T; t++) { const x = rejectionTrial(r, 100, 200, m); if (x.pw39) a++; if (x.pw19) b++; if (x.mad) c++; }
  console.log(m, (a / T).toFixed(3), (b / T).toFixed(3), (c / T).toFixed(3));
}
