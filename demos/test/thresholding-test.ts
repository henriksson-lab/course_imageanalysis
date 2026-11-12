// Reference checks for the thresholding day: lib/threshold.ts (global methods, local
// thresholds, hysteresis, the two-Gaussian model) and threshroc/roc.ts (ROC from class
// histograms). Each routine is compared with a brute-force or textbook version. Run:
//   node_modules/.bin/esbuild demos/test/thresholding-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import * as TH from "../src/lib/threshold";
import { fromFunction, Grey } from "../src/lib/imageops";
import { intHistogram } from "../src/lib/histogram";
import { confusion } from "../src/lib/metrics";
import { sweep, auc, probBrighter } from "../src/threshroc/roc";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const randHist = (r: ReturnType<typeof rng>, L: number, max = 10) => Float64Array.from({ length: L }, () => Math.floor(r.unif() * (max + 1)));

/** Brute force: class statistics of a split, straight from the definitions. */
function brute(h: ArrayLike<number>, t: number) {
  let N = 0, s = 0;
  for (let i = 0; i < h.length; i++) { N += h[i]; s += i * h[i]; }
  const mu = s / N;
  let vT = 0;
  for (let i = 0; i < h.length; i++) vT += h[i] * (i - mu) ** 2;
  vT /= N;
  let n0 = 0, n1 = 0, s0 = 0, s1 = 0;
  for (let i = 0; i < h.length; i++) if (i <= t) { n0 += h[i]; s0 += i * h[i]; } else { n1 += h[i]; s1 += i * h[i]; }
  const m0 = s0 / n0, m1 = s1 / n1;
  let v0 = 0, v1 = 0;
  for (let i = 0; i < h.length; i++) if (i <= t) v0 += h[i] * (i - m0) ** 2; else v1 += h[i] * (i - m1) ** 2;
  v0 /= n0; v1 /= n1;
  const w0 = n0 / N, w1 = n1 / N;
  return { N, mu, vT, w0, w1, m0, m1, v0, v1, vW: w0 * v0 + w1 * v1, vB: w0 * (m0 - mu) ** 2 + w1 * (m1 - mu) ** 2, n0, n1 };
}

// ---------------------------------------------------------------------------------------
function testWorkedExample(): void {
  current = "worked example";
  const h = [2, 5, 3, 1, 1, 2, 4, 2];
  const o = TH.otsu(h);
  check(o.t === 3, `Otsu t* = 3 (got ${o.t})`);
  check(close(o.mu, 3.3) && close(o.varT, 5.81), `μ = 3.3, σ_T² = 5.81 (got ${o.mu}, ${o.varT})`);
  const expect = [1.21, 3.60, 4.84, 5.02, 4.86, 3.94, 1.52];
  for (let t = 0; t < 7; t++) check(Math.abs(o.curve[t] - expect[t]) < 0.005, `σ_B²(${t}) ≈ ${expect[t]} (got ${o.curve[t].toFixed(4)})`);
  const s = TH.splitStats(h, 3);
  check(close(s.w0, 0.55) && Math.abs(s.mu0 - 1.27) < 0.005 && Math.abs(s.mu1 - 5.78) < 0.005, "ω0, μ0, μ1 at t* = 3");
  check(close(s.varW + s.varB, s.varT), "σ_T² = σ_W² + σ_B² at t* = 3");
  const iso = TH.isodata(h, 5.5);
  const tr = iso.trace.map(v => Math.round(v * 100) / 100);
  check(tr.length === 4 && tr[1] === 4.17 && tr[2] === 3.75 && tr[3] === 3.53, `isodata trace 5.5 → 4.17 → 3.75 → 3.53 (got ${tr.join(", ")})`);
  check(iso.t === 3, "isodata ends at the same split as Otsu");
  // the "stuck" preset of otsubars: isodata from 6.5 stops at split 4, Otsu is at 0
  const st = [9, 0, 0, 8, 7, 0, 0, 5];
  check(TH.isodata(st, 6.5).t === 4 && TH.otsu(st).t === 0, "isodata stuck at a local fixed point");
}

function testOtsuBrute(): void {
  current = "Otsu vs brute force";
  const r = rng(3);
  for (let trial = 0; trial < 300; trial++) {
    const L = 2 + Math.floor(r.unif() * 30), h = randHist(r, L);
    if (h.reduce((a, b) => a + b, 0) === 0) continue;
    const o = TH.otsu(h);
    let best = -1, bestW = Infinity;
    for (let t = 0; t < L - 1; t++) {
      const b = brute(h, t);
      if (b.n0 === 0 || b.n1 === 0) { check(!Number.isFinite(o.curve[t]), "undefined split gives NaN"); continue; }
      check(close(b.vW + b.vB, b.vT, 1e-9), "σ_T² = σ_W² + σ_B²");
      check(close(o.curve[t], b.vB, 1e-9), `σ_B² from running sums (t=${t})`);
      check(close(b.vB, b.w0 * b.w1 * (b.m0 - b.m1) ** 2, 1e-9), "σ_B² = ω0ω1(μ0 − μ1)²");
      const s = TH.splitStats(h, t);
      check(close(s.varW, b.vW, 1e-9) && close(s.varB, b.vB, 1e-9), "splitStats");
      if (b.vW < bestW - 1e-12) { bestW = b.vW; best = t; }
    }
    if (best >= 0) check(o.t === best, `argmax σ_B² = argmin σ_W² (${o.t} vs ${best})`);
    // Otsu's threshold is an isodata fixed point (in 1D with bin values: the midpoint of the
    // class means lies in [t, t+1) up to the discreteness; check the continuous condition loosely)
    if (o.t >= 0) {
      const s = TH.splitStats(h, o.t), mid = (s.mu0 + s.mu1) / 2;
      check(mid >= o.t - 1 && mid <= o.t + 2, `Otsu split near an isodata fixed point (mid ${mid.toFixed(2)}, t ${o.t})`);
    }
  }
}

function testOtsuIsodataFixedPoint(): void {
  current = "Otsu is an isodata fixed point (fine bins)";
  // a smooth two-Gaussian histogram with many levels: the continuous condition holds
  for (const [m0, m1, s0, s1, w1] of [[60, 160, 15, 20, 0.3], [80, 150, 12, 30, 0.5], [50, 120, 10, 10, 0.1]]) {
    const h = Float64Array.from({ length: 256 }, (_, x) => 1e6 * ((1 - w1) * TH.normalPdf(x, m0, s0) + w1 * TH.normalPdf(x, m1, s1)));
    const o = TH.otsu(h), s = TH.splitStats(h, o.t), mid = (s.mu0 + s.mu1) / 2;
    check(Math.abs(mid - (o.t + 0.5)) <= 1, `(μ0+μ1)/2 = ${mid.toFixed(2)} within a level of t* + ½ = ${o.t + 0.5}`);
    const iso = TH.isodata(h, o.t + 0.5);
    check(Math.abs(iso.t - o.t) <= 1, "isodata started at Otsu stays there");
  }
}

function testMulti(): void {
  current = "multi-level Otsu";
  const r = rng(8);
  for (let trial = 0; trial < 60; trial++) {
    const L = 3 + Math.floor(r.unif() * 14), h = randHist(r, L);
    const N = h.reduce((a, b) => a + b, 0);
    if (N === 0) continue;
    let mu = 0;
    for (let i = 0; i < L; i++) mu += i * h[i] / N;
    let best = -Infinity;
    for (let t1 = 0; t1 < L - 2; t1++) for (let t2 = t1 + 1; t2 < L - 1; t2++) {
      let v = 0, ok = true;
      for (const [a, b] of [[0, t1], [t1 + 1, t2], [t2 + 1, L - 1]]) {
        let n = 0, s = 0;
        for (let i = a; i <= b; i++) { n += h[i]; s += i * h[i]; }
        if (n === 0) { ok = false; break; }
        v += (n / N) * (s / n - mu) ** 2;
      }
      if (ok && v > best) best = v;
    }
    const m = TH.otsuMulti3(h);
    if (best > -Infinity) check(close(m.varB, best, 1e-9), `3-class σ_B² (${m.varB} vs ${best})`);
  }
}

function testOtherGlobal(): void {
  current = "Kittler, Kapur, triangle, percentile";
  // Kittler–Illingworth on a densely sampled equal-variance mixture ≈ the Bayes threshold
  for (const [m0, m1, s, w1] of [[60, 160, 15, 0.3], [80, 150, 12, 0.5], [70, 130, 15, 0.1]]) {
    const h = Float64Array.from({ length: 256 }, (_, x) => 1e6 * ((1 - w1) * TH.normalPdf(x, m0, s) + w1 * TH.normalPdf(x, m1, s)));
    const k = TH.kittler(h), bayes = TH.gaussianCrossings(m0, s, m1, s, w1)[0];
    const analytic = (m0 + m1) / 2 + s * s / (m1 - m0) * Math.log((1 - w1) / w1);
    check(close(bayes, analytic, 1e-9), `equal-variance crossing = formula (${bayes} vs ${analytic})`);
    check(Math.abs(k.t + 0.5 - bayes) <= 3, `Kittler ${k.t + 0.5} near Bayes ${bayes.toFixed(2)}`);
  }
  // unequal variances: two crossings, the density ratio is 1 at both
  const cr = TH.gaussianCrossings(80, 12, 150, 40, 0.4);
  check(cr.length === 2, "two crossings for unequal variances");
  for (const c of cr) check(close(0.6 * TH.normalPdf(c, 80, 12), 0.4 * TH.normalPdf(c, 150, 40), 1e-9), "weighted densities equal at a crossing");
  // Kapur vs brute force
  const r = rng(12);
  for (let trial = 0; trial < 50; trial++) {
    const L = 3 + Math.floor(r.unif() * 20), h = randHist(r, L);
    const N = h.reduce((a, b) => a + b, 0);
    if (N === 0) continue;
    const k = TH.kapur(h);
    for (let t = 0; t < L - 1; t++) {
      let w0 = 0;
      for (let i = 0; i <= t; i++) w0 += h[i] / N;
      const w1 = 1 - w0;
      if (w0 <= 1e-12 || w1 <= 1e-12) continue;
      let H = 0;
      for (let i = 0; i < L; i++) { const p = h[i] / N, w = i <= t ? w0 : w1; if (p > 0) H -= (p / w) * Math.log(p / w); }
      check(close(k.curve[t], H, 1e-9), `Kapur entropy at t=${t}`);
    }
  }
  // triangle: a peak with a long linear tail to the right and a dip; the threshold lies in the tail
  const tri = Float64Array.from({ length: 100 }, (_, i) => (i < 20 ? 5 * i : Math.max(0, 100 - 1.1 * (i - 20)) * Math.exp(-(i - 20) / 15)));
  const tr = TH.triangle(tri);
  check(tr.tailRight && tr.t > 20 && tr.t < 80, `triangle threshold in the tail (got ${tr.t})`);
  // percentile
  const p = TH.percentileThreshold([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], 0.2);
  check(p.t === 7, `percentile: brightest 20 % of 10 equal levels → t = 7 (got ${p.t})`);
}

function testLocal(): void {
  current = "local mean/sd and local thresholds";
  const r = rng(5), w = 23, h = 17;
  const img = fromFunction(w, h, () => Math.round(255 * r.unif()));
  for (const rad of [1, 2, 4]) {
    const { mean, sd } = TH.localMeanSd(img, rad);
    let ok = true;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0, s2 = 0, n = 0;
      for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
        const xx = Math.min(w - 1, Math.max(0, x + dx)), yy = Math.min(h - 1, Math.max(0, y + dy));
        const v = img.data[yy * w + xx]; s += v; s2 += v * v; n++;
      }
      const m = s / n, sdv = Math.sqrt(Math.max(0, s2 / n - m * m));
      if (Math.abs(mean.data[y * w + x] - m) > 1e-3 || Math.abs(sd.data[y * w + x] - sdv) > 1e-2) ok = false;
    }
    check(ok, `box mean/sd vs naive window (r = ${rad}, replicated borders)`);
    const T = TH.localThreshold(img, "sauvola", { r: rad, k: 0.3 }), N = TH.localThreshold(img, "niblack", { r: rad, k: -0.2 }), M = TH.localThreshold(img, "meanC", { r: rad, C: 7 });
    const i = 5 * w + 6;
    check(close(T.data[i], mean.data[i] * (1 + 0.3 * (sd.data[i] / 128 - 1)), 1e-5), "Sauvola formula");
    check(close(N.data[i], mean.data[i] - 0.2 * sd.data[i], 1e-5), "Niblack formula");
    check(close(M.data[i], mean.data[i] - 7, 1e-5), "mean − C formula");
  }
  const mask = TH.applyThreshold(img, 100, true);
  check(mask.every((v, i) => v === (img.data[i] < 100 ? 1 : 0)), "applyThreshold dark");
}

function testHysteresis(): void {
  current = "hysteresis";
  const r = rng(9);
  for (let trial = 0; trial < 40; trial++) {
    const w = 3 + Math.floor(r.unif() * 12), h = 3 + Math.floor(r.unif() * 12);
    const d = Float64Array.from({ length: w * h }, () => Math.floor(10 * r.unif()));
    const lo = 3 + Math.floor(3 * r.unif()), hi = lo + 1 + Math.floor(3 * r.unif());
    for (const conn of [4, 8] as const) {
      const res = TH.hysteresis(d, w, h, lo, hi, conn);
      // reference: BFS flood from every seed inside the low mask
      const ref = new Uint8Array(w * h), q: number[] = [];
      for (let i = 0; i < w * h; i++) if (d[i] > hi) { ref[i] = 1; q.push(i); }
      while (q.length) {
        const i = q.pop()!, x = i % w, y = (i - x) / w;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if ((dx === 0 && dy === 0) || (conn === 4 && dx !== 0 && dy !== 0)) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const j = yy * w + xx;
          if (!ref[j] && d[j] > lo) { ref[j] = 1; q.push(j); }
        }
      }
      check(res.mask.every((v, i) => v === ref[i]), `hysteresis = flood from seeds (conn ${conn})`);
    }
  }
  // 1D
  const y = [0, 5, 9, 5, 0, 5, 5, 0, 9];
  const k = TH.hysteresis1d(y, 4, 8);
  check(k.join("") === "011100001", `1D hysteresis (got ${k.join("")})`);
}

function testRoc(): void {
  current = "ROC from class histograms";
  const r = rng(13), n = 4000;
  const v = Float64Array.from({ length: n }, () => 0), truth = new Uint8Array(n);
  for (let i = 0; i < n; i++) { truth[i] = r.unif() < 0.2 ? 1 : 0; v[i] = Math.max(0, Math.min(255, Math.round((truth[i] ? 140 : 90) + 30 * r.norm()))); }
  const hObj = intHistogram(v, 0, 255, truth).counts, hBg = intHistogram(v, 0, 255, Uint8Array.from(truth, x => 1 - x)).counts;
  const pts = sweep(hObj, hBg);
  for (const t of [-1, 0, 50, 100, 120, 200, 255]) {
    const p = pts[t + 1], c = confusion(Uint8Array.from(v, x => (x > t ? 1 : 0)), truth);
    check(p.t === t && p.tp === c.tp && p.fp === c.fp && p.fn === c.fn && p.tn === c.tn, `confusion from histograms = from pixels (t = ${t})`);
  }
  // AUC = P(object pixel brighter than background pixel), by brute force over pairs
  let s = 0, cnt = 0;
  const obj: number[] = [], bg: number[] = [];
  for (let i = 0; i < n; i++) (truth[i] ? obj : bg).push(v[i]);
  for (const a of obj) for (const b of bg) { s += a > b ? 1 : a === b ? 0.5 : 0; cnt++; }
  check(close(auc(pts), s / cnt, 1e-9), `AUC = P(brighter) (${auc(pts)} vs ${s / cnt})`);
  check(close(probBrighter(hObj, hBg), s / cnt, 1e-9), "probBrighter by histograms");
  // ROC from sorting the pixels: the sweep points are the steps of the sorted list
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[b] - v[a]);
  let tp = 0, fp = 0, k = 0;
  for (let t = 255; t >= -1; t--) {
    while (k < n && v[order[k]] > t) { if (truth[order[k]]) tp++; else fp++; k++; }
    const p = pts[t + 1];
    if (p.tp !== tp || p.fp !== fp) { check(false, `ROC by sorting at t = ${t}`); break; }
  }
  check(true, "ROC by sorting");
}

function testModel(): void {
  current = "two-Gaussian model";
  // error of a threshold by numerical integration
  const [m0, s0, m1, s1, w1] = [50, 20, 150, 20, 0.1];
  for (const t of [80, 100, 117.6, 130]) {
    let e = 0;
    for (let x = -300.005; x < 500; x += 0.01) e += 0.01 * (x > t ? (1 - w1) * TH.normalPdf(x, m0, s0) : w1 * TH.normalPdf(x, m1, s1));
    check(Math.abs(TH.thresholdError(t, m0, s0, m1, s1, w1) - e) < 1e-4, `error rate at t = ${t}`);
  }
  // the equal-variance Bayes threshold minimises the error
  const tb = TH.gaussianCrossings(m0, s0, m1, s1, w1)[0];
  check(TH.thresholdError(tb, m0, s0, m1, s1, w1) <= Math.min(TH.thresholdError(tb - 1, m0, s0, m1, s1, w1), TH.thresholdError(tb + 1, m0, s0, m1, s1, w1)), "Bayes threshold is the minimum");
  check(Math.abs(TH.normalCdf(1.96) - 0.975) < 1e-4 && Math.abs(TH.normalCdf(0) - 0.5) < 1e-7, "normal cdf");
}

testWorkedExample();
testOtsuBrute();
testOtsuIsodataFixedPoint();
testMulti();
testOtherGlobal();
testLocal();
testHysteresis();
testRoc();
testModel();
console.log(failures ? `thresholding: ${failures} of ${checks} checks FAILED` : `thresholding: all ${checks} checks passed`);
if (failures) process.exit(1);
