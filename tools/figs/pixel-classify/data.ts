// Data for the slide figures of pixel-classify/ (run with the pixelclass demo's own scene,
// strokes, feature stack and SVM plug-in, and the svmcv tab's cross-validation, so the numbers
// match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/pixel-classify/data.ts
// then python3 -I tools/figs/pixel-classify/figs.py <dir>. Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import { synthScene } from "../../../demos/src/pixelclass/scene";
import { PIXEL_PRESETS, strokesToMap, SMALL, LARGE, ALL_FAMILIES } from "../../../demos/src/pixelclass/presets";
import { trainingSet, regionScores } from "../../../demos/src/pixelclass/pipeline";
import { featureStack, selectChannels } from "../../../demos/src/lib/featurestack";
import { trainPixelSVM, makeFolds, cvError, fitPredictor, logGrid, trainSVM, decision } from "../../../demos/src/lib/svm";
import { labelsAndUncertainty } from "../../../demos/src/lib/classifier";
import { rng } from "../../../demos/src/lib/random";
import { blobs } from "../../../demos/src/lib/datasets2d";

const TMP = process.env.FIGTMP || os.tmpdir();
const size = 256, N = size * size;
const sc = synthScene(size, 1);
const full = featureStack(sc.image, ALL_FAMILIES, [...SMALL, ...LARGE]);
const p = PIXEL_PRESETS.find(q => q.id === "full")!;
const labels = strokesToMap(p.strokes(sc), size, size);
const S = selectChannels(full, (g, s) => p.families.includes(g) && (s <= 1.6 ? p.small : p.large));
const info = trainingSet(S, labels, { cap: 4000, balance: false, seed: 1 })!;
const data = info.data;
const out: Record<string, unknown> = { size, image: Array.from(sc.image.data, v => Math.round(v * 10) / 10), truth: Array.from(sc.truth), labels: Array.from(labels) };

function run(o: { standardise: boolean; gamma: number; kernel: "rbf" | "linear" }) {
  const m = trainPixelSVM(data, { kernel: o.kernel, C: 10, gamma: o.gamma, standardise: o.standardise, mode: "ovo" }, 1);
  const P = m.predict(S.data, N, new Float32Array(N * data.k), 0, N);
  const { label } = labelsAndUncertainty(P, N, data.k);
  const all = regionScores(label, labels, sc.truth, 3);
  console.log(`std=${o.standardise} gamma=${o.gamma} ${o.kernel}: acc ${(100 * all.truthAcc).toFixed(1)} Dice ${all.dice.map(v => v.toFixed(2)).join(" ")}; ${m.summary()}`);
  return { pred: Array.from(label), prob: Array.from({ length: N }, (_, q) => Math.round(P[q * data.k + 1] * 1000) / 1000), acc: all.truthAcc, dice: all.dice, nSV: m.nSV };
}
out.svm = run({ standardise: true, gamma: 0.01, kernel: "rbf" });
out.unscaled = run({ standardise: false, gamma: 0.01, kernel: "rbf" });

// spread of each channel over the training pixels
const sds = S.names.map((_, j) => {
  let s = 0, s2 = 0;
  for (let i = 0; i < data.n; i++) { const v = data.X[j * data.n + i]; s += v; s2 += v * v; }
  const mu = s / data.n; return Math.sqrt(Math.max(0, s2 / data.n - mu * mu));
});
const jMin = sds.indexOf(Math.min(...sds)), jMax = sds.indexOf(Math.max(...sds));
console.log("smallest sd", S.names[jMin], sds[jMin].toFixed(2), "largest sd", S.names[jMax], sds[jMax].toFixed(1));
const sdAll = (j: number) => { let s = 0, s2 = 0; for (let q = 0; q < N; q++) { const v = S.data[j * N + q]; s += v; s2 += v * v; } const mu = s / N; return Math.sqrt(s2 / N - mu * mu); };
const allSds = S.names.map((_, j) => sdAll(j));
console.log("whole image: smallest", S.names[allSds.indexOf(Math.min(...allSds))], Math.min(...allSds).toFixed(2), "largest", S.names[allSds.indexOf(Math.max(...allSds))], Math.max(...allSds).toFixed(1));
out.sdAll = allSds; out.names = S.names;
const jA = allSds.indexOf(Math.min(...allSds)), jB = allSds.indexOf(Math.max(...allSds));
out.chans = { names: [S.names[jA], S.names[jB]], sd: [allSds[jA], allSds[jB]], a: Array.from(S.data.subarray(jA * N, (jA + 1) * N), v => +v.toPrecision(4)), b: Array.from(S.data.subarray(jB * N, (jB + 1) * N), v => +v.toPrecision(5)) };
out.scatter = {
  names: [S.names[jMin], S.names[jMax]], sd: [sds[jMin], sds[jMax]],
  x: Array.from({ length: data.n }, (_, i) => data.X[jMin * data.n + i]), y: Array.from({ length: data.n }, (_, i) => data.X[jMax * data.n + i]),
  cls: Array.from(data.y),
};

// the svmcv tab's cross-validation (300-pixel subset with rng(7), 5 folds, RBF, standardised, one-vs-one)
const Cs = logGrid(-1, 3), gs = logGrid(-3, 1);
const r = rng(7), keep = Math.min(1, 300 / data.n), idx: number[] = [];
for (let i = 0; i < data.n; i++) if (r.unif() < keep) idx.push(i);
const n = idx.length, d = data.d, X = new Float64Array(n * d), cls = new Int32Array(n), grp = new Int32Array(n);
idx.forEach((i, q) => { for (let j = 0; j < d; j++) X[q * d + j] = data.X[j * data.n + i]; cls[q] = data.y[i]; grp[q] = data.groups ? data.groups[i] : i; });
const nGroups = new Set(grp).size;
const cv: Record<string, unknown> = { Cs, gs, n, nGroups, pix: idx.map(i => info.pix[i]) };
for (const mode of ["stroke", "pixel"] as const) {
  const nF = mode === "stroke" ? Math.min(5, nGroups) : 5;
  const folds = mode === "stroke" ? makeFolds(n, nF, cls, grp) : makeFolds(n, nF, cls, undefined, 3);
  const E: number[] = [];
  for (let gi = 0; gi < gs.length; gi++) for (let ci = 0; ci < Cs.length; ci++)
    E.push(cvError(n, cls, folds, nF, tr => fitPredictor(X, d, cls, data.k, tr, { kernel: { type: "rbf", gamma: gs[gi] }, C: Cs[ci], tol: 1e-3 }, { standardise: true, mode: "ovo" })));
  const best = E.indexOf(Math.min(...E));
  console.log(`CV by ${mode}: ${nF} folds, ${n} px, ${nGroups} strokes; best ${(100 * E[best]).toFixed(1)}% at C=${Cs[best % Cs.length]} gamma=${gs[Math.floor(best / Cs.length)]}; zero cells ${E.filter(v => v === 0).length}`);
  cv[mode] = { E, best, folds: Array.from(folds), nF };
}
out.cv = cv;

// class imbalance: a small class against a large one (synthetic 2D), equal vs balanced C_i
const imb = blobs([[0.38, 0.5], [0.62, 0.5]], { sd: 0.09, sizes: [400, 30], seed: 51 }).points.map(q => ({ x: 10 * q.x, y: 10 * q.y, c: q.c }));
const Xi = new Float64Array(2 * imb.length); imb.forEach((q, i) => { Xi[2 * i] = q.x; Xi[2 * i + 1] = q.y; });
const yi = imb.map(q => (q.c === 1 ? 1 : -1));
const nPos = yi.filter(v => v > 0).length, nNeg = yi.length - nPos;
const res: unknown[] = [];
for (const bal of [false, true]) {
  const Ci = yi.map(v => (bal ? (v > 0 ? imb.length / (2 * nPos) : imb.length / (2 * nNeg)) : 1));
  const m = trainSVM(Xi, imb.length, 2, yi, { kernel: { type: "linear" }, Ci, tol: 1e-4 });
  const f = imb.map((_, i) => decision(m, Xi, 2 * i));
  const tp = f.filter((v, i) => v > 0 && yi[i] > 0).length, fp = f.filter((v, i) => v > 0 && yi[i] < 0).length;
  console.log(`imbalance balanced=${bal}: n+ ${nPos} n- ${nNeg}; recall ${tp}/${nPos}, false positives ${fp}; w ${Array.from(m.w!).map(v => v.toFixed(2))} b ${m.b.toFixed(2)}`);
  res.push({ balanced: bal, w: Array.from(m.w!), b: m.b, recall: tp / nPos, tp, fp, f });
}
out.imbalance = { points: imb, nPos, nNeg, res };
fs.writeFileSync(TMP + "/pixel-classify.json", JSON.stringify(out));
