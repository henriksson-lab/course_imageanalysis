// Data for the slide figures of diffusion-inverse/lec.qmd, from the posterior1d demo's own model
// (same settings and seeds as its "below" preset: separation 4, low-pass keeping 6 cosines, noise
// 0.05, matched prior, 32 exact posterior samples with seed 7) and its coverage of the 90 % band
// (200 repeats) for the three priors. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/diffusion-inverse/data.ts --bundle --platform=node --log-level=warning | node > <tmp>/di.json
import * as M from "../../../demos/src/posterior1d/model";

const m: M.Meas = { id: "lowpass", blur: 2, cutoff: 6, gap: [26, 38], noise: 0.05 };
const x = M.testSignal(4), y = M.measure(x, m, 3);
const tr = M.trainingSet("matched"), post = M.posterior(tr, y, m);
const S = M.postSamples(post, 32, 7), b = M.band(S);
const A = (v: ArrayLike<number>) => Array.from(v);
const cov: Record<string, number> = {};
for (const p of ["matched", "wide", "noclose"] as M.PriorId[]) cov[p] = M.coverage(p, m, 200);
console.log(JSON.stringify({
  truth: A(x), meas: A(M.measuredSignal(y, m)), mean: A(M.postMeanSignal(post)),
  samples: S.map(A), lo: A(b.lo), hi: A(b.hi),
  nullS: S.map(s => A(M.nullPart(s, m))), nullT: A(M.nullPart(x, m)),
  counts: S.map(s => M.countSpots(s)), coverage: cov,
}));
