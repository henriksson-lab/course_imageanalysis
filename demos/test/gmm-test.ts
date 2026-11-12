// Tests of lib/kmeans.ts and lib/gmm.ts (Gaussian-mixture day), and the by-hand numbers of
// the units kmeans/, gaussian-mixture/, em/ and gmm-images/.
// Run: node_modules/.bin/esbuild demos/test/gmm-test.ts --bundle --platform=node --log-level=warning | node
import { kmeans, kmeansPlusPlus, assignStep, updateStep, kmeansRestarts } from "../src/lib/kmeans";
import {
  makeMixture, eStep, mStep, fitGMM, nParams, bic, aic, mahalanobis2, logSumExp, sampleMixture,
  initMixture, logLikelihood, COV_TYPES, hardLabels, cloneMixture, type Mixture,
} from "../src/lib/gmm";
import { rng } from "../src/lib/random";
import { multiOtsu, histEM, initHist } from "../src/histgmm/model";

declare const process: { exit(code: number): never };
let failed = 0, passed = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passed++;
  else { failed++; console.log("FAIL", name, info); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- k-means by hand (unit kmeans/) ----
{
  const X = Float64Array.from([1, 2, 4, 7, 9, 10]);
  const r = kmeans(X, 6, 1, 2, { init: Float64Array.from([1, 2]) });
  const c = r.trace.map(t => Array.from(t.centres).map(v => v.toFixed(2)).join(","));
  check("kmeans trace", c.join(" | ") === "1.00,2.00 | 1.00,6.40 | 1.50,7.50 | 2.33,8.67", c.join(" | "));
  check("kmeans J", near(r.J, 28 / 3, 1e-9), String(r.J));
  check("kmeans J trace", r.trace.map(t => t.J.toFixed(2)).join(",") === "142.00,26.84,15.50,9.33", r.trace.map(t => t.J.toFixed(2)).join(","));
  const Y = Float64Array.from([0, 1, 5, 6, 10, 11]);
  const s = kmeans(Y, 6, 1, 3, { init: Float64Array.from([0, 1, 8]) });
  check("kmeans stuck fixed point", s.J === 26 && s.iterations === 1 && Array.from(s.centres).join() === "0,1,8", `${s.J} ${Array.from(s.centres)}`);
  const g = kmeans(Y, 6, 1, 3, { init: Float64Array.from([0, 5, 10]) });
  check("kmeans good minimum", near(g.J, 1.5, 1e-12), String(g.J));
  // k-means++ with first centre 0: P(second centre is 10 or 11) = 221/283
  const rr = rng(5);
  let hits = 0;
  const T = 40000;
  for (let t = 0; t < T; t++) { const C = kmeansPlusPlus(Y, 6, 1, 2, rr, undefined, 0); if (C[1] >= 10) hits++; }
  check("kmeans++ probability", near(hits / T, 221 / 283, 0.01), String(hits / T));
}

// ---- k-means: J never increases, on random 2D data ----
{
  const r = rng(11);
  for (let trial = 0; trial < 20; trial++) {
    const n = 150, d = 2, X = new Float64Array(n * d);
    for (let i = 0; i < n * d; i++) X[i] = r.unif();
    const w = trial % 2 ? Float64Array.from({ length: n }, () => 1 + Math.floor(5 * r.unif())) : undefined;
    const res = kmeans(X, n, d, 2 + (trial % 5), { init: "random", seed: r, weights: w });
    let ok = true;
    for (let t = 1; t < res.trace.length; t++) if (res.trace[t].J > res.trace[t - 1].J + 1e-9) ok = false;
    // also after the update step, before the next assignment
    const up = updateStep(X, n, d, res.trace[0].assign, res.centres.length / d, w, res.trace[0].centres);
    let Jup = 0;
    for (let i = 0; i < n; i++) { let q = 0; for (let a = 0; a < d; a++) q += (X[i * d + a] - up.centres[res.trace[0].assign[i] * d + a]) ** 2; Jup += (w ? w[i] : 1) * q; }
    if (Jup > res.trace[0].J + 1e-9 || res.trace.length > 1 && res.trace[1].J > Jup + 1e-9) ok = false;
    check("kmeans J non-increasing " + trial, ok && res.converged);
  }
  const X = new Float64Array(400);
  for (let i = 0; i < 400; i++) X[i] = r.unif();
  const rs = kmeansRestarts(X, 200, 2, 5, 10, rng(3));
  check("restarts best", rs.best.J === Math.min(...rs.Js));
  const a = assignStep(X, 200, 2, rs.best.centres, 5);
  check("restarts converged", near(a.J, rs.best.J, 1e-9));
}

// ---- Gaussian by hand (unit gaussian-mixture/) ----
{
  const S = [5, 2, 2, 5 - 3];
  check("mahalanobis (2,1)", near(mahalanobis2([2, 1], [0, 0], S, 2), 5 / 6, 1e-12));
  check("mahalanobis (-1,2)", near(mahalanobis2([-1, 2], [0, 0], S, 2), 5, 1e-12));
  check("params K=3 d=3", [nParams(3, 3, "spherical"), nParams(3, 3, "diagonal"), nParams(3, 3, "full")].join() === "14,20,29");
  check("params 1D", nParams(4, 1, "full") === 11 && nParams(4, 1, "spherical") === 11);
  check("logSumExp", near(logSumExp([1000, 1000]), 1000 + Math.log(2), 1e-12) && logSumExp([-Infinity, -Infinity]) === -Infinity);
  // sampling reproduces the parameters
  const m = makeMixture([0.3, 0.7], [[0, 0], [3, 1]], [[[1, 0.6], [0.6, 0.5]], [[0.2, 0], [0, 2]]]);
  const s = sampleMixture(m, 40000, rng(2));
  const f = fitGMM(s.X, 40000, 2, 2, { covType: "full", init: "kmeans", seed: rng(4) });
  const j0 = f.model.means[0] < 1.5 ? 0 : 1, M = f.model;
  check("EM recovers weights", near(M.weights[j0], 0.3, 0.01), String(M.weights[j0]));
  check("EM recovers means", near(M.means[2 * j0], 0, 0.03) && near(M.means[2 * (1 - j0) + 1], 1, 0.03));
  check("EM recovers covariances", near(M.covs[4 * j0 + 1], 0.6, 0.03) && near(M.covs[4 * (1 - j0) + 3], 2, 0.06), `${M.covs[4 * j0 + 1]} ${M.covs[4 * (1 - j0) + 3]}`);
}

// ---- EM by hand (unit em/) ----
{
  const X = Float64Array.from([1, 2, 4, 7, 9, 10]);
  let m = makeMixture([0.5, 0.5], [[2], [8]], [4, 4]);
  const e = eStep(m, X, 6);
  const r1 = Array.from({ length: 6 }, (_, i) => e.resp[2 * i]);
  check("EM r1(4)", near(r1[2], 0.8176, 1e-4) && near(r1[3], 0.0474, 1e-4), r1.map(v => v.toFixed(4)).join());
  const tr = [e.logLik];
  let cur = e;
  for (let it = 0; it < 3; it++) {
    m = mStep(X, 6, 1, cur.resp, 2, { covType: "full" }, undefined, m);
    if (it === 0) {
      check("EM N1 mu pi", near(m.weights[0] * 6, 2.855, 1e-3) && near(m.means[0], 2.314, 1e-3) && near(m.means[1], 8.391, 1e-3) && near(m.weights[0], 0.476, 1e-3),
        `${m.weights[0] * 6} ${m.means[0]} ${m.means[1]}`);
    }
    cur = eStep(m, X, 6);
    tr.push(cur.logLik);
  }
  check("EM logL trace", tr.map(v => v.toFixed(2)).join() === "-14.94,-14.22,-14.03,-13.99", tr.map(v => v.toFixed(2)).join());
  // BIC / AIC (given log-likelihoods)
  const L = [-4210, -4020, -4012, -4010];
  const B = L.map((l, i) => bic(l, nParams(i + 1, 1, "full"), 1000)), A = L.map((l, i) => aic(l, nParams(i + 1, 1, "full")));
  check("BIC picks 2", B.indexOf(Math.min(...B)) === 1 && near(B[1], 8074.54, 0.01), B.map(v => v.toFixed(1)).join());
  check("AIC picks 3", A.indexOf(Math.min(...A)) === 2 && A[2] === 8040, A.join());
}

// ---- EM: log-likelihood never decreases (all covariance types, weighted and not) ----
{
  const r = rng(21);
  for (const type of COV_TYPES) for (let trial = 0; trial < 6; trial++) {
    const d = 1 + (trial % 3), n = 120, X = new Float64Array(n * d);
    for (let i = 0; i < n; i++) { const c = i % 3; for (let a = 0; a < d; a++) X[i * d + a] = c * (a + 1) + 0.7 * r.norm(); }
    const w = trial >= 3 ? Float64Array.from({ length: n }, () => 1 + Math.floor(4 * r.unif())) : undefined;
    const f = fitGMM(X, n, d, 2 + (trial % 3), { covType: type, init: trial % 2 ? "random" : "kmeans++", seed: r, weights: w, floor: 1e-6, tol: 0, maxIter: 60 });
    let ok = true;
    for (let t = 1; t < f.trace.length; t++) if (f.trace[t] < f.trace[t - 1] - 1e-8 * Math.abs(f.trace[t - 1])) ok = false;
    check(`EM monotone ${type} ${trial}`, ok, f.trace.slice(0, 5).map(v => v.toFixed(4)).join());
  }
}

// ---- hard EM with small equal spherical variances = k-means ----
{
  const r = rng(31), n = 200, X = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) X[i] = r.unif();
  const C0 = kmeansPlusPlus(X, n, 2, 4, rng(8));
  const km = kmeans(X, n, 2, 4, { init: C0 });
  const s2 = 1e-5;
  let m: Mixture = { k: 4, d: 2, weights: new Float64Array(4).fill(0.25), means: Float64Array.from(C0), covs: new Float64Array(16) };
  for (let j = 0; j < 4; j++) { m.covs[4 * j] = s2; m.covs[4 * j + 3] = s2; }
  for (let it = 0; it < 100; it++) {
    const e = eStep(m, X, n);
    m = mStep(X, n, 2, e.resp, 4, { fixCovs: true, equalWeights: true }, undefined, m);
  }
  let dmax = 0;
  for (let q = 0; q < 8; q++) dmax = Math.max(dmax, Math.abs(m.means[q] - km.centres[q]));
  check("small-variance EM = k-means", dmax < 1e-6, String(dmax));
  const lab = hardLabels(eStep(m, X, n).resp, n, 4);
  check("small-variance EM labels = k-means", lab.every((v, i) => v === km.assign[i]));
}

// ---- weighted (histogram) EM equals per-pixel EM ----
{
  const r = rng(41), n = 3000, px = new Float64Array(n);
  for (let i = 0; i < n; i++) px[i] = Math.max(0, Math.min(255, Math.round(i % 3 === 0 ? 60 + 12 * r.norm() : i % 3 === 1 ? 130 + 20 * r.norm() : 200 + 8 * r.norm())));
  const counts = new Float64Array(256);
  for (const v of px) counts[v]++;
  const g = Float64Array.from({ length: 256 }, (_, i) => i);
  const m0 = makeMixture([1 / 3, 1 / 3, 1 / 3], [[50], [120], [210]], [400, 400, 400]);
  const a = fitGMM(px, n, 1, 3, { init: m0, tol: 0, maxIter: 30 });
  const b = fitGMM(g, 256, 1, 3, { init: m0, weights: counts, tol: 0, maxIter: 30 });
  let dmax = 0;
  for (let j = 0; j < 3; j++) dmax = Math.max(dmax, Math.abs(a.model.means[j] - b.model.means[j]), Math.abs(a.model.covs[j] - b.model.covs[j]), Math.abs(a.model.weights[j] - b.model.weights[j]));
  check("histogram EM = pixel EM", dmax < 1e-8 && near(a.logLik, b.logLik, 1e-6), String(dmax));
  // and the histgmm demo's own EM on bins
  const h = histEM(counts, initHist(counts, 3, "quantiles", rng(1), false), false);
  check("histgmm model runs", h.trace.length > 1 && h.trace.every((v, i) => i === 0 || v >= h.trace[i - 1] - 1e-6));
}

// ---- BIC picks the true K on well-separated data ----
{
  const m = makeMixture([0.3, 0.3, 0.4], [[0.2, 0.2], [0.8, 0.3], [0.5, 0.8]], [0.004, 0.004, 0.004]);
  const s = sampleMixture(m, 600, rng(51));
  const B: number[] = [];
  for (let K = 1; K <= 6; K++) {
    let best = -Infinity;
    for (let t = 0; t < 5; t++) best = Math.max(best, fitGMM(s.X, 600, 2, K, { covType: "full", init: "kmeans", seed: rng(100 + t), floor: 1e-6 }).logLik);
    B.push(bic(best, nParams(K, 2, "full"), 600));
  }
  check("BIC picks K = 3", B.indexOf(Math.min(...B)) === 2, B.map(v => v.toFixed(0)).join());
}

// ---- collapse: without a floor the likelihood of a component on one point grows without bound ----
{
  const X = Float64Array.from([0, 0.1, 0.2, 0.3, 5]);
  let m = makeMixture([0.5, 0.5], [[0.15], [5]], [0.5, 0.5]);
  const L0 = logLikelihood(m, X, 5);
  for (let it = 0; it < 30; it++) m = mStep(X, 5, 1, eStep(m, X, 5).resp, 2, {}, undefined, m);
  const L1 = logLikelihood(m, X, 5);
  check("collapse: variance to the minimum", m.covs[1] < 1e-8 && L1 > L0 + 5, `${m.covs[1]} ${L1}`);
  let mf = makeMixture([0.5, 0.5], [[0.15], [5]], [0.5, 0.5]);
  for (let it = 0; it < 30; it++) mf = mStep(X, 5, 1, eStep(mf, X, 5).resp, 2, { floor: 0.01 }, undefined, mf);
  check("floor keeps it finite", mf.covs[1] >= 0.01, String(mf.covs[1]));
}

// ---- multi-level Otsu (histgmm) against brute force ----
{
  const r = rng(61), counts = new Float64Array(32);
  for (let i = 0; i < 32; i++) counts[i] = 1 + Math.floor(20 * r.unif()) + (i > 8 && i < 13 ? 30 : 0);
  const t = multiOtsu(counts, 3);
  // brute force: between-class variance over all pairs of thresholds
  const tot = counts.reduce((a, b) => a + b, 0);
  let mu = 0;
  counts.forEach((c, i) => { mu += c * i; });
  mu /= tot;
  let best = -1, bt: number[] = [];
  for (let a = 1; a < 32; a++) for (let b = a + 1; b < 32; b++) {
    let s = 0;
    for (const [lo, hi] of [[0, a], [a, b], [b, 32]]) {
      let w = 0, m = 0;
      for (let i = lo; i < hi; i++) { w += counts[i]; m += counts[i] * i; }
      if (w > 0) s += w * (m / w - mu) ** 2;
    }
    if (s > best + 1e-9) { best = s; bt = [a, b]; }
  }
  check("multi-Otsu = brute force", t.join() === bt.join(), `${t} vs ${bt}`);
}

console.log(`gmm-test: ${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
// keep imports used
void initMixture; void cloneMixture;
