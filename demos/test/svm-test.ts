// Tests for the SVM day (lib/svm.ts, lib/svmkernels.ts and the demos' pure helpers).
// Run without the rest of the suite:
//   node_modules/.bin/esbuild demos/test/svm-test.ts --bundle --platform=node --log-level=warning | node
import {
  trainSVM, decision, marginWidth, primalObjective, dualObjective, kktViolation, trainLinearSubgradient,
  trainMulti, multiPredict, makeFolds, cvError, fitPredictor, fitStandardiser, applyStandardiser, gridSearch,
  hingeLoss, logisticLoss, zeroOneLoss, squaredLoss, trainPixelSVM, SVM_CLASSIFIER, logGrid,
} from "../src/lib/svm";
import { kernel, quadMap, quadMap6, kernelMatrix, isPSD, gaussianAffinity, polyFeatureCount, paraboloidMap } from "../src/lib/svmkernels";
import { defaultParams } from "../src/lib/classifier";
import { rng } from "../src/lib/random";
import { BY_HAND, nearestMeanLine } from "../src/linclass/geom";
import { SVM_PRESETS, presetPoints } from "../src/svm2d/presets";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

// ---- 1. the by-hand max-margin example of maxmargin/: −1 at (0,0), (0,2); +1 at (2,1), (3,0)
{
  const X = [0, 0, 0, 2, 2, 1, 3, 0], y = [-1, -1, 1, 1];
  const m = trainSVM(X, 4, 2, y, { C: 1e6, tol: 1e-9 });
  check(near(m.w![0], 1, 1e-6) && near(m.w![1], 0, 1e-6), `by-hand w = (1, 0), got ${Array.from(m.w!)}`);
  check(near(m.b, -1, 1e-6), `by-hand b = −1, got ${m.b}`);
  check(near(marginWidth(m), 2, 1e-6), "by-hand width 2");
  check(near(m.alpha[0], 0.25, 1e-6) && near(m.alpha[1], 0.25, 1e-6) && near(m.alpha[2], 0.5, 1e-6) && m.alpha[3] === 0, `by-hand α = 0.25, 0.25, 0.5, 0: ${Array.from(m.alpha)}`);
  const yf = [0, 1, 2, 3].map(i => y[i] * decision(m, X, 2 * i));
  check(yf.every((v, i) => near(v, [1, 1, 1, 2][i], 1e-6)), `by-hand y f = 1, 1, 1, 2: ${yf}`);
}

// ---- 2. two points: α = 2/‖x+ − x−‖², w = α (x+ − x−), street through the midpoint
{
  const xp = [3, 1], xm = [1, 2];
  const m = trainSVM([...xp, ...xm], 2, 2, [1, -1], { C: 1e6, tol: 1e-10 });
  const dd = (xp[0] - xm[0]) ** 2 + (xp[1] - xm[1]) ** 2, a = 2 / dd;
  check(near(m.alpha[0], a, 1e-6) && near(m.alpha[1], a, 1e-6), `two-point α = 2/‖Δ‖² = ${a}, got ${Array.from(m.alpha)}`);
  check(near(m.w![0], a * 2, 1e-6) && near(m.w![1], a * -1, 1e-6), "two-point w = α (x+ − x−)");
  check(near(decision(m, [2, 1.5]), 0, 1e-6), "two-point boundary through the midpoint");
  check(near(marginWidth(m), Math.sqrt(dd), 1e-6), "two-point width = distance between the points");
}

// ---- 3. random data: KKT, duality gap, all kernel functions, per-sample C
function cloud(n: number, seed: number, sep = 1): { X: Float64Array; y: Float64Array } {
  const r = rng(seed), X = new Float64Array(2 * n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { y[i] = i % 2 ? 1 : -1; X[2 * i] = r.norm() + sep * y[i]; X[2 * i + 1] = r.norm() - 0.5 * sep * y[i]; }
  return { X, y };
}
for (const [spec, C] of [[{ type: "linear" }, 1], [{ type: "rbf", gamma: 0.7 }, 3], [{ type: "poly", degree: 2, coef0: 1 }, 0.5], [{ type: "rbf", gamma: 5 }, 100]] as const) {
  const { X, y } = cloud(60, 3);
  const m = trainSVM(X, 60, 2, y, { kernel: spec, C, tol: 1e-7 });
  const P = primalObjective(m, X), D = dualObjective(m);
  check(m.converged, `${spec.type} converged`);
  check(kktViolation(m, X, 1e-9) < 1e-3, `${spec.type} C=${C}: KKT violation ${kktViolation(m, X, 1e-9)}`);
  check(Math.abs(P - D) <= 1e-4 * Math.max(1, Math.abs(P)), `${spec.type} C=${C}: duality gap ${P - D} (P ${P}, D ${D})`);
  check(near(D, m.dual, 1e-6), `${spec.type}: reported dual = recomputed dual`);
  check(Array.from(m.alpha).every((a, i) => a >= 0 && a <= m.Ci[i] + 1e-12), `${spec.type}: 0 ≤ α ≤ C`);
}
{
  const { X, y } = cloud(40, 5);
  const Ci = Float64Array.from(y, v => (v > 0 ? 5 : 0.5));
  const m = trainSVM(X, 40, 2, y, { Ci, tol: 1e-8 });
  check(Array.from(m.alpha).every((a, i) => a <= Ci[i] + 1e-12), "per-sample C respected");
  check(kktViolation(m, X, 1e-9) < 1e-3, "per-sample C: KKT");
}

// ---- 4. SMO against a brute-force projected-gradient solution of the dual (≤ 8 points)
function bruteDual(K: Float64Array, y: Float64Array, C: number, n: number): number {
  // projected gradient ascent on Σα − ½ αᵀQα over {0 ≤ α ≤ C, yᵀα = 0}; projection by bisection on ν
  const a = new Float64Array(n);
  const proj = (v: Float64Array) => {
    let lo = -1e6, hi = 1e6;
    const at = (nu: number) => { let s = 0; for (let i = 0; i < n; i++) s += y[i] * Math.min(C, Math.max(0, v[i] - nu * y[i])); return s; };
    for (let it = 0; it < 200; it++) { const mid = (lo + hi) / 2; if (at(mid) > 0) lo = mid; else hi = mid; }
    const nu = (lo + hi) / 2;
    for (let i = 0; i < n; i++) v[i] = Math.min(C, Math.max(0, v[i] - nu * y[i]));
  };
  let L = 0;
  for (let i = 0; i < n * n; i++) L = Math.max(L, Math.abs(K[i]));
  const step = 1 / (L * n);
  for (let it = 0; it < 60000; it++) {
    const g = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 1; for (let j = 0; j < n; j++) s -= y[i] * y[j] * K[i * n + j] * a[j]; g[i] = s; }
    for (let i = 0; i < n; i++) a[i] += step * g[i];
    proj(a);
  }
  let obj = 0;
  for (let i = 0; i < n; i++) { obj += a[i]; for (let j = 0; j < n; j++) obj -= 0.5 * a[i] * a[j] * y[i] * y[j] * K[i * n + j]; }
  return obj;
}
for (const seed of [11, 12, 13]) {
  const n = 8, { X, y } = cloud(n, seed, 0.6);
  for (const spec of [{ type: "linear" as const }, { type: "rbf" as const, gamma: 1 }]) {
    const C = 2, m = trainSVM(X, n, 2, y, { kernel: spec, C, tol: 1e-10 });
    const ref = bruteDual(kernelMatrix(spec, X, n, 2), y, C, n);
    check(Math.abs(m.dual - ref) < 1e-5 * Math.max(1, Math.abs(ref)), `SMO vs brute force (${spec.type}, seed ${seed}): ${m.dual} vs ${ref}`);
  }
}

// ---- 5. separable data with large C: zero training error
{
  const { X, y } = cloud(80, 7, 3);
  const m = trainSVM(X, 80, 2, y, { C: 1e5 });
  let wrong = 0;
  for (let i = 0; i < 80; i++) if (y[i] * decision(m, X, 2 * i) <= 0) wrong++;
  check(wrong === 0, `separable, large C: ${wrong} training errors`);
}

// ---- 6. kernel functions
{
  const x = [1, 2], z = [3, 1];
  const px = quadMap(1, 2), pz = quadMap(3, 1);
  const dot = px[0] * pz[0] + px[1] * pz[1] + px[2] * pz[2];
  check(near(dot, 25, 1e-12) && near(kernel({ type: "poly", degree: 2, coef0: 0 }, x, z), 25, 1e-12), "φ(x)·φ(z) = (x·z)² = 25");
  const p6x = quadMap6(1, 2), p6z = quadMap6(3, 1);
  check(near(p6x.reduce((s, v, i) => s + v * p6z[i], 0), 36, 1e-12), "6-coordinate map: (x·z + 1)² = 36");
  check(near(kernel({ type: "rbf", gamma: 1 }, [0, 0], [2, 0]), Math.exp(-4), 1e-12), "RBF value");
  check(near(gaussianAffinity(2, 1), Math.exp(-2), 1e-12), "Gaussian affinity = RBF with γ = 1/(2σ²)");
  check(polyFeatureCount(20, 2) === 231 && polyFeatureCount(2, 2) === 6, "number of quadratic features (incl. constant)");
  const r = rng(9), P = new Float64Array(2 * 15);
  for (let i = 0; i < 30; i++) P[i] = r.norm();
  check(isPSD(kernelMatrix({ type: "rbf", gamma: 2 }, P, 15, 2), 15), "RBF kernel matrix is PSD");
  check(!isPSD(Float64Array.from([1, 2, 2, 1]), 2), "[[1, 2], [2, 1]] is not a kernel matrix");
  const lift = paraboloidMap(0.6, 0.8);
  check(near(lift[2], 1, 1e-12), "paraboloid lift");
}

// ---- 7. subgradient descent approaches the SMO optimum (linear)
{
  const { X, y } = cloud(40, 21, 1.5);
  const m = trainSVM(X, 40, 2, y, { C: 1, tol: 1e-8 });
  const sg = trainLinearSubgradient(X, 40, 2, y, 1, 3000, 0.05);
  const best = Math.min(...sg.objective);
  const P = primalObjective(m, X);
  check(best >= P - 1e-6 && best <= P * 1.02 + 1e-3, `subgradient objective ${best} ≈ SMO primal ${P}`);
}

// ---- 8. multi-class, folds, CV, standardisation, grid search
{
  const r = rng(31), n = 90, X = new Float64Array(2 * n), cls = new Int32Array(n);
  const centres = [[0, 0], [3, 0], [1.5, 2.6]];
  for (let i = 0; i < n; i++) { const c = i % 3; cls[i] = c; X[2 * i] = centres[c][0] + 0.5 * r.norm(); X[2 * i + 1] = centres[c][1] + 0.5 * r.norm(); }
  for (const mode of ["ovr", "ovo"] as const) {
    const mm = trainMulti(X, n, 2, cls, 3, mode, { C: 10 });
    check(mm.models.length === 3, `${mode}: 3 sub-models for k = 3`);
    let wrong = 0;
    for (let i = 0; i < n; i++) if (multiPredict(mm, X, 2 * i) !== cls[i]) wrong++;
    check(wrong <= 3, `${mode}: ${wrong} training errors on three blobs`);
  }
  const mm5 = trainMulti(new Float64Array(10), 5, 2, [0, 1, 2, 3, 4], 5, "ovo");
  check(mm5.models.length === 10, "OvO with K = 5: 10 classifiers");
  const groups = Int32Array.from({ length: n }, (_, i) => Math.floor(i / 10));
  const f = makeFolds(n, 3, cls, groups);
  check(Array.from({ length: n }, (_, i) => i).every(i => f[i] === f[groups[i] * 10]), "group folds keep a group together");
  const e = cvError(n, cls, f, 3, tr => fitPredictor(X, 2, cls, 3, tr, { kernel: { type: "rbf", gamma: 1 }, C: 10 }, { standardise: true }));
  check(e >= 0 && e < 0.1, `CV error on three blobs ${e}`);
  const st = fitStandardiser(X, n, 2), Z = applyStandardiser(st, Float64Array.from(X), n, 2);
  let m0 = 0, v0 = 0;
  for (let i = 0; i < n; i++) m0 += Z[2 * i] / n;
  for (let i = 0; i < n; i++) v0 += (Z[2 * i] - m0) ** 2 / n;
  check(near(m0, 0, 1e-9) && near(v0, 1, 1e-9), "z-scores have mean 0 and variance 1");
  const gs = gridSearch(X, n, 2, cls, 3, logGrid(-1, 2), logGrid(-2, 1), makeFolds(n, 3, cls), 3, { standardise: true });
  check(gs.error.length === 16 && gs.best.error < 0.1, `grid search best ${JSON.stringify(gs.best)}`);
}

// ---- 9. losses
check(hingeLoss(2) === 0 && hingeLoss(0.5) === 0.5 && hingeLoss(-1) === 2, "hinge loss values");
check(zeroOneLoss(0.5) === 0 && zeroOneLoss(0) === 1 && zeroOneLoss(-1) === 1, "0–1 loss values");
check(near(logisticLoss(0), 1, 1e-12) && squaredLoss(2) === 1, "logistic and squared loss values");

// ---- 10. the classifier plug-in (feature-major tables, unscaled features)
{
  const r = rng(41), n = 400, d = 2, X = new Float32Array(n * d), y = new Int32Array(n);
  // feature 0: informative, range 0..1; feature 1: noise with range 0..4000
  for (let i = 0; i < n; i++) { y[i] = i % 2; X[i] = 0.3 + 0.4 * y[i] + 0.08 * r.norm(); X[n + i] = 4000 * r.unif(); }
  const data = { X, n, d, y, k: 2 };
  const Xt = new Float32Array(n * d), yt = new Int32Array(n);
  for (let i = 0; i < n; i++) { yt[i] = i % 2; Xt[i] = 0.3 + 0.4 * yt[i] + 0.08 * r.norm(); Xt[n + i] = 4000 * r.unif(); }
  const acc = (std: number) => {
    const p = { ...defaultParams(SVM_CLASSIFIER), standardise: std };
    const m = SVM_CLASSIFIER.train(data, p, 1), P = m.predict(Xt, n);
    let ok = 0;
    for (let i = 0; i < n; i++) if ((P[2 * i + 1] > P[2 * i] ? 1 : 0) === yt[i]) ok++;
    return ok / n;
  };
  const a1 = acc(1), a0 = acc(0);
  check(a1 > 0.95, `standardised RBF SVM accuracy ${a1}`);
  check(a0 < 0.8, `unstandardised RBF SVM fails on a 0..4000 noise feature: accuracy ${a0}`);
  const m3 = trainPixelSVM({ X, n, d, y, k: 2 }, { kernel: "linear", C: 1, gamma: 1, standardise: true, mode: "ovr" });
  const P3 = m3.predict(X, n);
  check(Array.from({ length: n }, (_, i) => near(P3[2 * i] + P3[2 * i + 1], 1, 1e-5)).every(Boolean), "plug-in scores sum to 1");
}

// ---- 11. demo helpers: by-hand numbers of the linear-classifier unit, svm2d presets
{
  const f = (x: number[]) => BY_HAND.w[0] * x[0] + BY_HAND.w[1] * x[1] + BY_HAND.b;
  check(BY_HAND.points.every(p => Math.sign(f([p.x, p.y])) === (p.c ? 1 : -1) || f([p.x, p.y]) === 0), "linclass by-hand points on their sides");
  const nm = nearestMeanLine([{ x: 0, y: 0, c: 0 }, { x: 2, y: 0, c: 1 }]);
  check(near(nm.w[0], 2, 1e-12) && near(nm.b, -2, 1e-12), "nearest mean line of (0,0) and (2,0): 2x − 2");
  for (const p of SVM_PRESETS) {
    const pts = presetPoints(p);
    check(pts.length > 0 && pts.every(q => q.x >= p.range[0] && q.x <= p.range[1] && q.y >= p.range[0] && q.y <= p.range[1]), `svm2d preset ${p.id}: points inside the range`);
  }
}

console.log(`svm: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
