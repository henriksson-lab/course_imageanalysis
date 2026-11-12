// Reference checks for the random-forest day: lib/forest.ts (split search, trees, forests,
// OOB, importance), lib/featurestack.ts, lib/classifier.ts, and the demos' own models
// (bootstrap simulation, featimp population, pixelclass synthetic image). Run:
//   node_modules/.bin/esbuild demos/test/randomforest-test.ts --bundle --platform=node --log-level=warning | node
import * as FO from "../src/lib/forest";
import { featureStack, selectChannels, gatherPixels, featureVector } from "../src/lib/featurestack";
import { registerClassifier, getClassifier, listClassifiers, labelsAndUncertainty } from "../src/lib/classifier";
import { createGrey, fromFunction } from "../src/lib/imageops";
import { rng } from "../src/lib/random";
import { oobFractionSamples, varianceOfAverage, simulateAverageVariance } from "../src/bootstrap/sim";
import { population, FEATURES } from "../src/featimp/population";
import { synthScene } from "../src/pixelclass/scene";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- impurity
check(close(FO.impurity([4, 4], 8, "gini"), 0.5), "gini of 4/4");
check(close(FO.impurity([4, 4], 8, "entropy"), 1), "entropy of 4/4");
check(close(FO.impurity([3, 1], 4, "misclass"), 0.25), "misclass of 3/1");
check(close(FO.impurity([2, 3, 5], 10, "gini"), 2 * (0.2 * 0.3 + 0.2 * 0.5 + 0.3 * 0.5)), "gini = sum_{k≠l} p_k p_l");

// ---- the unit-1 table (dtree exercise B1)
{
  const I = [20, 35, 45, 50, 60, 80, 90, 110], V = [2, 3, 5, 9, 12, 4, 15, 11];
  const y = [0, 0, 1, 0, 1, 0, 1, 1];   // B = 0, N = 1
  const T = FO.tableFromColumns([I, V]), B = FO.binTable(T), tg = FO.classTarget(y, 2);
  const idx = Int32Array.from({ length: 8 }, (_, i) => i);
  const sI = FO.bestSplit(B, idx, 0, 8, tg, null, [0])!, sV = FO.bestSplit(B, idx, 0, 8, tg, null, [1])!;
  check(close(0.5 - sI.gain, 1 / 3), `best weighted Gini on I = 1/3 (got ${0.5 - sI.gain})`);
  check(close(0.5 - sV.gain, 0.2), `best weighted Gini on V = 0.2 (got ${0.5 - sV.gain})`);
  const both = FO.bestSplit(B, idx, 0, 8, tg, null, [0, 1])!;
  check(both.feature === 1, "root splits on V");
  const eI = FO.bestSplit(B, idx, 0, 8, tg, null, [0], { criterion: "entropy" })!, eV = FO.bestSplit(B, idx, 0, 8, tg, null, [1], { criterion: "entropy" })!;
  check(close(eI.gain, 0.31127812445913283, 1e-9) && close(eV.gain, 0.5487949406953986, 1e-9), `information gains ${eI.gain} ${eV.gain}`);
}

// ---- split search vs brute force on random tables (weighted, all criteria)
{
  const r = rng(3);
  for (let rep = 0; rep < 200; rep++) {
    const n = 3 + Math.floor(r.unif() * 25), d = 1 + Math.floor(r.unif() * 3), k = 2 + Math.floor(r.unif() * 2);
    const cols = Array.from({ length: d }, () => Array.from({ length: n }, () => Math.floor(r.unif() * 8)));
    const y = Array.from({ length: n }, () => Math.floor(r.unif() * k));
    const w = Array.from({ length: n }, () => (rep % 2 ? 0.2 + r.unif() : 1));
    const crit = (["gini", "entropy", "misclass"] as const)[rep % 3];
    const minLeaf = 1 + (rep % 3);
    const T = FO.tableFromColumns(cols), B = FO.binTable(T, rep % 4 === 0 ? 3 : 256);
    const idx = Int32Array.from({ length: n }, (_, i) => i);
    const s = FO.bestSplit(B, idx, 0, n, FO.classTarget(y, k), w, Array.from({ length: d }, (_, j) => j), { criterion: crit, minLeaf });
    // brute force over every threshold between distinct values (or bin boundary)
    let best = -Infinity;
    for (let j = 0; j < d; j++) {
      const vals = [...new Set(cols[j])].sort((a, b) => a - b);
      for (let q = 0; q + 1 < vals.length; q++) {
        const t = (vals[q] + vals[q + 1]) / 2;
        if (rep % 4 === 0) { // binned: thresholds only at bin boundaries
          const bq = B.codes[j * n + cols[j].indexOf(vals[q])], bq1 = B.codes[j * n + cols[j].indexOf(vals[q + 1])];
          if (bq === bq1) continue;
        }
        const L = new Array(k).fill(0), R = new Array(k).fill(0);
        let nL = 0, nR = 0;
        for (let i = 0; i < n; i++) if (cols[j][i] <= t) { L[y[i]] += w[i]; nL++; } else { R[y[i]] += w[i]; nR++; }
        if (nL < minLeaf || nR < minLeaf) continue;
        best = Math.max(best, FO.impurityDecrease(L, R, crit));
      }
    }
    if (best === -Infinity) check(s === null, `rep ${rep}: no valid split expected`);
    else check(s !== null && close(s.gain, best, 1e-9), `rep ${rep}: split gain ${s?.gain} vs brute force ${best}`);
  }
}

// ---- regression split vs brute force
{
  const r = rng(5);
  for (let rep = 0; rep < 50; rep++) {
    const n = 4 + Math.floor(r.unif() * 15);
    const x = Array.from({ length: n }, () => Math.floor(r.unif() * 10)), y = Array.from({ length: n }, () => r.norm());
    const T = FO.tableFromColumns([x]), B = FO.binTable(T);
    const s = FO.bestSplit(B, Int32Array.from({ length: n }, (_, i) => i), 0, n, FO.regTarget(y), null, [0]);
    const v = (a: number[]) => { const m = a.reduce((p, q) => p + q, 0) / a.length; return a.reduce((p, q) => p + (q - m) ** 2, 0) / a.length; };
    let best = -Infinity;
    const vals = [...new Set(x)].sort((a, b) => a - b);
    for (let q = 0; q + 1 < vals.length; q++) {
      const t = (vals[q] + vals[q + 1]) / 2, L = y.filter((_, i) => x[i] <= t), R = y.filter((_, i) => x[i] > t);
      best = Math.max(best, v(y) - L.length / n * v(L) - R.length / n * v(R));
    }
    if (best === -Infinity) check(s === null, "regression: none");
    else check(s !== null && close(s.gain, best, 1e-9), `regression gain ${s?.gain} vs ${best}`);
  }
}

// ---- trees: a full tree has zero training error on distinct points; depth limit respected
{
  const r = rng(7), n = 150;
  const xs = Array.from({ length: n }, () => [r.unif(), r.unif()]), y = xs.map(() => (r.unif() < 0.5 ? 1 : 0));
  const T = FO.tableFromRows(xs), B = FO.binTable(T, 1024);
  const t = FO.growTree(B, FO.classTarget(y, 2), null, {});
  let err = 0;
  xs.forEach((x, i) => { if (FO.predictPoint(t, x)[y[i]] < 1) err++; });
  check(err === 0, `full tree training error ${err}`);
  const t3 = FO.growTree(B, FO.classTarget(y, 2), null, { maxDepth: 3 });
  check(FO.treeShape(t3).depth <= 3, "depth limit");
  const t5 = FO.growTree(B, FO.classTarget(y, 2), null, { minLeaf: 5 });
  let minCount = Infinity;
  for (let q = 0; q < t5.nNodes; q++) if (t5.feature[q] < 0) minCount = Math.min(minCount, t5.count[q]);
  check(minCount >= 5, "min leaf size");
  const ts = FO.growTree(B, FO.classTarget(y, 2), null, { maxSplits: 4 });
  check(ts.splitOrder.length === 4, "split budget");
  // best-first with an unlimited budget gives the same tree as depth-first
  const tb = FO.growTree(B, FO.classTarget(y, 2), null, { maxSplits: 1e9 });
  check(tb.nNodes === t.nNodes, "best-first full tree = depth-first full tree");
  // importance sums to the root impurity minus the weighted leaf impurities
  let leafImp = 0;
  for (let q = 0; q < t.nNodes; q++) if (t.feature[q] < 0) leafImp += t.weight[q] / t.weight[0] * t.impurity[q];
  check(close(t.importance[0] + t.importance[1], t.impurity[0] - leafImp, 1e-9), "MDI telescopes");
  // forced split at the root
  const tf = FO.growTree(B, FO.classTarget(y, 2), null, { maxSplits: 1, forced: { "": { feature: 1, threshold: 0.5 } } });
  check(tf.feature[0] === 1 && tf.threshold[0] === 0.5, "forced root split");
}

// ---- forests: OOB fraction, predictions, OOB error, curves
{
  const r = rng(9), n = 300;
  const xs = Array.from({ length: n }, () => [r.unif(), r.unif()]);
  const y = xs.map(([a, b]) => (a + 0.3 * r.norm() > b ? 1 : 0));
  const T = FO.tableFromRows(xs), B = FO.binTable(T, 1024), tg = FO.classTarget(y, 2);
  const f = FO.trainForest(B, tg, null, { nTrees: 60, seed: 4 });
  let oob = 0, tot = 0;
  for (const ib of f.inbag) for (const v of ib) { tot++; if (!v) oob++; }
  check(Math.abs(oob / tot - Math.pow(1 - 1 / n, n)) < 0.01, `OOB fraction ${oob / tot}`);
  const P = FO.predictForest(f, T.X, n);
  let maxd = 0;
  for (let i = 0; i < 20; i++) { const p = FO.predictForestPoint(f, xs[i]); maxd = Math.max(maxd, Math.abs(p[1] - P[2 * i + 1])); }
  check(maxd < 1e-6, "predictForest = predictForestPoint");
  const curve = FO.oobErrorCurve(f, T.X, y);
  check(close(curve[59], FO.oobError(f, T.X, y), 1e-12), "OOB curve ends at the OOB error");
  check(FO.oobError(f, T.X, y) < 0.35, "OOB error sensible");
  // a forest of 1 tree without bootstrap = the single tree
  const f1 = FO.trainForest(B, tg, null, { nTrees: 1, bootstrap: false });
  const t1 = FO.growTree(B, tg, null, {});
  check(f1.trees[0].nNodes === t1.nNodes, "one tree without bootstrap = the full tree");
  const tc = FO.testErrorCurve(f, T.X, n, y);
  check(tc[59] <= 0.05, `training-set error of the forest ${tc[59]}`);
}

// ---- importance of an exact duplicate is split about in half
{
  const r = rng(12), n = 400;
  const a = Array.from({ length: n }, () => r.unif()), noise = Array.from({ length: n }, () => r.unif());
  const y = a.map(v => (v + 0.15 * r.norm() > 0.5 ? 1 : 0));
  const T1 = FO.tableFromColumns([a, noise]), T2 = FO.tableFromColumns([a, a.slice(), noise]);
  const f1 = FO.trainForest(FO.binTable(T1), FO.classTarget(y, 2), null, { nTrees: 200, mtry: 1, seed: 1 });
  const f2 = FO.trainForest(FO.binTable(T2), FO.classTarget(y, 2), null, { nTrees: 200, mtry: 1, seed: 1 });
  const i1 = FO.impurityImportance(f1), i2 = FO.impurityImportance(f2);
  check(Math.abs((i2[0] + i2[1]) - i1[0]) < 0.25 * i1[0] && Math.abs(i2[0] - i2[1]) < 0.25 * i1[0], `duplicate splits credit: ${i1[0].toFixed(3)} vs ${i2[0].toFixed(3)} + ${i2[1].toFixed(3)}`);
  const p2 = FO.permutationImportance(f2, T2, y, { repeats: 3 });
  const pg = FO.permutationImportance(f2, T2, y, { repeats: 3, groups: [[0, 1]] });
  check(pg[0] > 2 * Math.max(p2[0], p2[1]), `permuting the pair together (${pg[0].toFixed(3)}) beats each alone (${p2[0].toFixed(3)}, ${p2[1].toFixed(3)})`);
}

// ---- class weights = resampling (dtree/pixelclass math question)
{
  const w = FO.balancedClassWeights([0, 0, 0, 0, 1], 2);
  check(close(w[0] * 4, w[1] * 1), "balanced weights equalise the class totals");
  // weighted Gini with class weights equals Gini of the replicated counts
  const g1 = FO.impurity([4 * 2, 1 * 8], 16, "gini"), g2 = FO.impurity([8, 8], 16, "gini");
  check(close(g1, g2), "weighting = replicating");
}

// ---- bootstrap demo maths
{
  const f = oobFractionSamples(20, 2000, 1);
  const m = f.reduce((a, b) => a + b, 0) / f.length;
  check(Math.abs(m - Math.pow(0.95, 20)) < 0.01, `mean OOB fraction n=20: ${m}`);
  check(close(varianceOfAverage(1, 0.3, 10), 0.3 + 0.7 / 10), "variance formula");
  const sim = simulateAverageVariance(1, 0.5, [1, 10, 100], 4000, 3);
  check(Math.abs(sim[0] - 1) < 0.08 && Math.abs(sim[1] - 0.55) < 0.05 && Math.abs(sim[2] - 0.505) < 0.05, `Monte Carlo variance ${Array.from(sim).map(v => v.toFixed(3))}`);
}

// ---- feature stack
{
  const img = createGrey(40, 30, 77);
  const S = featureStack(img, ["intensity", "edges", "blobs", "ridges", "texture"], [1, 3]);
  check(S.d === 2 * 7, `channels ${S.d}`);
  let ok = true;
  for (let c = 0; c < S.d; c++) for (let p = 0; p < 40 * 30; p++) {
    const v = S.data[c * 1200 + p];
    if (S.groups[c] === "intensity" ? Math.abs(v - 77) > 1e-3 : Math.abs(v) > 1e-3) ok = false;
  }
  check(ok, "constant image: smoothed = constant, every derivative channel 0");
  // a bright square: LoG negative in the middle at a matching scale; gradient largest on the edge
  const sq = fromFunction(64, 64, (x, y) => (Math.abs(x - 32) < 6 && Math.abs(y - 32) < 6 ? 100 : 0));
  const S2 = featureStack(sq, ["edges", "blobs"], [4]);
  check(S2.data[64 * 64 + 32 * 64 + 32] < 0, "LoG < 0 at the centre of a bright blob");
  check(S2.data[32 * 64 + 26] > S2.data[32 * 64 + 32], "gradient larger at the edge than at the centre");
  const sel = selectChannels(S, g => g === "blobs");
  check(sel.d === 2 && sel.groups.every(g => g === "blobs"), "selectChannels");
  const X = gatherPixels(S, [5, 17]);
  check(X[3 * 2 + 1] === featureVector(S, 17)[3], "gatherPixels layout");
  // incremental smoothing = direct smoothing
  const ramp = fromFunction(48, 48, (x, y) => Math.sin(x / 3) * 50 + y);
  const A = featureStack(ramp, ["intensity"], [1, 3]), Bs = featureStack(ramp, ["intensity"], [3]);
  let md = 0;
  for (let p = 0; p < 48 * 48; p++) { const x = p % 48, y = (p / 48) | 0; if (x > 12 && x < 36 && y > 12 && y < 36) md = Math.max(md, Math.abs(A.data[48 * 48 + p] - Bs.data[p])); }
  check(md < 0.5, `incremental blur matches direct blur (max diff ${md.toFixed(3)})`);
}

// ---- classifier registry and adapters
{
  registerClassifier(FO.FOREST_CLASSIFIER);
  registerClassifier(FO.TREE_CLASSIFIER);
  check(listClassifiers().length >= 2 && getClassifier("forest") === FO.FOREST_CLASSIFIER, "registry");
  const r = rng(2), n = 200, d = 3, X = new Float32Array(n * d), y = new Int32Array(n);
  for (let i = 0; i < n; i++) { for (let j = 0; j < d; j++) X[j * n + i] = r.unif(); y[i] = X[i] > 0.5 ? 1 : 0; }
  const m = FO.FOREST_CLASSIFIER.train({ X, n, d, y, k: 2 }, { trees: 20, mtry: 2, depth: 10, minLeaf: 1 }, 1);
  const P = m.predict(X, n);
  const { label } = labelsAndUncertainty(P, n, 2);
  let err = 0; for (let i = 0; i < n; i++) if (label[i] !== y[i]) err++;
  check(err <= 2, `forest adapter training error ${err}`);
  let rowsum = 0; for (let i = 0; i < n; i++) rowsum = Math.max(rowsum, Math.abs(P[2 * i] + P[2 * i + 1] - 1));
  check(rowsum < 1e-5, "scores sum to 1");
  check((m.oobError?.() ?? 1) < 0.1, "forest adapter OOB error");
  const part = new Float32Array(n * 2);
  m.predict(X, n, part, 10, 20);
  check(Math.abs(part[2 * 15 + 1] - P[2 * 15 + 1]) < 1e-6 && part[2 * 5] === 0, "partial prediction");
}

// ---- demo data
{
  const pop = population({ seed: 1 });
  check(pop.n === 400 && pop.cols.length === FEATURES.length, "featimp population size");
  const area = pop.cols[0], eqd = pop.cols[2];
  let mono = true;
  const idx = Array.from({ length: pop.n }, (_, i) => i).sort((a, b) => area[a] - area[b]);
  for (let q = 1; q < idx.length; q++) if (eqd[idx[q]] < eqd[idx[q - 1]] - 1e-9) mono = false;
  check(mono, "equivalent diameter is a monotone function of area");
  const sc = synthScene(128, 1);
  let cnt = [0, 0, 0, 0];
  for (const v of sc.truth) cnt[v]++;
  check(cnt[1] > 0 && cnt[2] > 0 && cnt[3] > 0, `synthetic scene has all classes ${cnt}`);
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
