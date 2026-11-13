// Data for the gmm-images slide figures, computed with the demos' own code (histgmm and
// colourgmm models, lib/gmm.ts, lib/kmeans.ts) so that the figures match the demos.
// Bundle and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/gmm-images/data.ts --bundle --platform=node --outfile=$TMP/gmmimg.js
//   node $TMP/gmmimg.js $TMP/cell128.json > $TMP/gmmimg.json
// cell128.json: the course micrograph shrunk to 128 × 128 by 2 × 2 area averaging (written
// by figs.py --cell128), as the histgmm demo does with loadCell(128).
// Then python3 -I tools/figs/gmm-images/figs.py $TMP/gmmimg.json draws the figures.
import { readFileSync } from "fs";
import { synthTruth, synthImage, greyCounts, initHist, histEM, sortByMean, bayesBoundaries, greyTables, multiOtsu } from "../../../demos/src/histgmm/model";
import { makeImage, features, score } from "../../../demos/src/colourgmm/model";
import { rng } from "../../../demos/src/lib/random";
import { eStep, mStep, fromCentres, hardLabels, type Mixture } from "../../../demos/src/lib/gmm";
import { kmeansPlusPlus, assignStep, updateStep } from "../../../demos/src/lib/kmeans";

const out: Record<string, unknown> = {};
const arr = (a: ArrayLike<number>) => Array.from(a);
const mix = (m: Mixture) => ({ weights: arr(m.weights), means: arr(m.means), vars: arr(m.covs) });

// ---- micrograph, K = 2 (preset k2) ----
const cellData: number[] = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cell = { width: 128, height: 128, data: Float32Array.from(cellData, v => Math.round(v)) };
{
  const counts = greyCounts(cell);
  const r = histEM(counts, initHist(counts, 2, "quantiles", rng(1), false), false);
  const m = sortByMean(r.model);
  out.cellK2 = { counts: arr(counts), model: mix(m), bounds: bayesBoundaries(m), otsu: multiOtsu(counts, 2) };
}

// ---- synthetic three-class image, noise 12 and 22 (presets equal/noise use synth) ----
for (const noise of [12, 22]) {
  const truth = synthTruth(false), img = synthImage(truth, noise, 1);
  const counts = greyCounts(img);
  const r = histEM(counts, initHist(counts, 3, "quantiles", rng(1), false), false);
  const m = sortByMean(r.model);
  const t = greyTables(m);
  const label = Array.from(img.data, v => t.label[v]);
  let ok = 0;
  for (let p = 0; p < label.length; p++) if (label[p] === truth[p]) ok++;
  const post = Array.from(img.data, v => arr(t.post.subarray(v * 3, v * 3 + 3)));
  out[`synth${noise}`] = { img: arr(img.data), truth: arr(truth), label, post, counts: arr(counts), model: mix(m), bounds: bayesBoundaries(m), correct: ok / label.length };
}

// ---- colour scene, RGB, K = 4, start 5 (preset shadow) ----
{
  const im = makeImage("scene");
  const F = features(im, "rgb"), n = 128 * 128, K = 4;
  const C0 = kmeansPlusPlus(F.X, n, F.d, K, rng(5));
  // k-means as the demo steps it
  let C = C0, J = NaN;
  for (let it = 0; it < 300; it++) {
    const a = assignStep(F.X, n, F.d, C, K);
    const u = updateStep(F.X, n, F.d, a.assign, K, undefined, C);
    const b = assignStep(F.X, n, F.d, u.centres, K);
    C = u.centres; J = b.J;
    if (b.assign.every((v, i) => v === a.assign[i])) break;
  }
  const km = assignStep(F.X, n, F.d, C, K).assign;
  // full-covariance GMM from the same seeds
  let m = fromCentres(F.X, n, F.d, K, C0, { covType: "full", floor: 0.01 }), L = NaN;
  for (let it = 0; it < 300; it++) {
    const e = eStep(m, F.X, n);
    m = mStep(F.X, n, F.d, e.resp, K, { covType: "full", floor: 0.01 }, undefined, m);
    const L2 = eStep(m, F.X, n).logLik;
    const done = Math.abs(L2 - L) < 1e-3;
    L = L2;
    if (done) break;
  }
  const gm = hardLabels(eStep(m, F.X, n).resp, n, K);
  out.scene = {
    rgb: arr(im.rgb), truth: arr(im.truth), km: arr(km), gm: arr(gm), J, logLik: L,
    diceKm: score(km, im.truth, K, 4).dice, diceGm: score(gm, im.truth, K, 4).dice,
    kmCentres: arr(C), gmMeans: arr(m.means), gmCovs: arr(m.covs),
  };
}
process.stdout.write(JSON.stringify(out));
