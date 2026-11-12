// Checks for the evaluation day: lib/eval.ts (matching, AP, boundary distances),
// lib/stats.ts (t distribution, tests, Bland–Altman, multiple testing, simulators) and the
// models of the demos objmatch, contourdist, blandaltman, leakage, testpeek, pseudorep and
// featurescan. Brute-force or textbook references, and the by-hand numbers of the units.
// Run:
//   node_modules/.bin/esbuild demos/test/evaluation-test.ts --bundle --platform=node --log-level=warning | node
import { rng, gauss } from "../src/lib/random";
import * as EV from "../src/lib/eval";
import * as ST from "../src/lib/stats";
import { rasteriseEllipses } from "../src/lib/synth";
import * as OM from "../src/objmatch/model";
import * as CD from "../src/contourdist/model";
import * as BA from "../src/blandaltman/model";
import * as LK from "../src/leakage/model";
import * as TP from "../src/testpeek/model";
import * as PR from "../src/pseudorep/model";
import * as FS from "../src/featurescan/model";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---------------------------------------------------------------------------------------
function bruteMatch(T: EV.IouTable, tau: number, maxCount: boolean): number {
  // best objective over all one-to-one partial matchings (rows choose a column or nothing)
  let best = -1;
  const usedP = new Uint8Array(T.np);
  const rec = (i: number, cnt: number, s: number) => {
    if (i === T.nt) { const v = maxCount ? cnt * 1000 + s : s; if (v > best) best = v; return; }
    rec(i + 1, cnt, s);
    for (let j = 0; j < T.np; j++) {
      const v = T.iou[i * T.np + j];
      if (!usedP[j] && v >= tau && v > 0) { usedP[j] = 1; rec(i + 1, cnt + 1, s + v); usedP[j] = 0; }
    }
  };
  rec(0, 0, 0);
  return best;
}

function testMatching(): void {
  current = "matching";
  const r = rng(7);
  for (let trial = 0; trial < 300; trial++) {
    const nt = 1 + Math.floor(r.unif() * 7), np = 1 + Math.floor(r.unif() * 7);
    const rows = Array.from({ length: nt }, () => Array.from({ length: np }, () => (r.unif() < 0.4 ? Math.round(r.unif() * 100) / 100 : 0)));
    const T = EV.tableFromIou(rows);
    const tau = [0.1, 0.3, 0.5][trial % 3];
    for (const maxCount of [true, false]) {
      const m = EV.matchOptimal(T, tau, maxCount);
      const val = maxCount ? m.tp * 1000 + m.pairs.reduce((s, q) => s + q.iou, 0) : m.pairs.reduce((s, q) => s + q.iou, 0);
      check(close(val, bruteMatch(T, tau, maxCount), 1e-9), `Hungarian = brute force (${nt}x${np}, τ ${tau}, maxCount ${maxCount})`);
      check(m.tp + m.fn === nt && m.tp + m.fp === np, "counts add up");
    }
    const g = EV.matchGreedy(T, tau);
    check(g.tp <= EV.matchOptimal(T, tau).tp, "greedy never has more matches than optimal");
  }
  // forced = optimal above 0.5 on random label images (shifted ellipses)
  for (let trial = 0; trial < 40; trial++) {
    const E = Array.from({ length: 8 }, () => ({ x: 8 + 48 * r.unif(), y: 8 + 48 * r.unif(), a: 3 + 6 * r.unif(), b: 3 + 3 * r.unif(), angle: 3 * r.unif() }));
    const P = E.flatMap(e => (r.unif() < 0.2 ? [] : [{ ...e, x: e.x + 3 * r.norm(), y: e.y + 3 * r.norm(), a: e.a * (0.8 + 0.4 * r.unif()) }]));
    const lt = rasteriseEllipses(64, 64, E), lp = rasteriseEllipses(64, 64, P);
    const T = EV.iouTable(lt, lp, E.length, P.length);
    // brute IoU check of one pair
    let inter = 0, a = 0, b = 0;
    for (let i = 0; i < lt.length; i++) { if (lt[i] === 1) a++; if (lp[i] === 1) b++; if (lt[i] === 1 && lp[i] === 1) inter++; }
    if (P.length) check(close(T.iou[0], inter ? inter / (a + b - inter) : 0), "IoU of pair (1,1)");
    for (const tau of [0.55, 0.7, 0.9]) {
      const f = EV.matchForced(T, tau), o = EV.matchOptimal(T, tau);
      check(f.tp === o.tp, `forced = optimal at τ ${tau}`);
      for (let i = 0; i < T.nt; i++) { let c = 0; for (let j = 0; j < T.np; j++) if (T.iou[i * T.np + j] > 0.5) c++; check(c <= 1, "at most one partner above 0.5"); }
    }
  }
  // by-hand table (b): 3 true × 4 predicted
  const Tb = EV.tableFromIou([[0.82, 0.05, 0, 0], [0.10, 0.45, 0.38, 0], [0, 0, 0, 0.62]]);
  const f = (tau: number) => EV.objectScores(EV.matchOptimal(Tb, tau));
  check(f(0.5).tp === 2 && f(0.5).fp === 2 && f(0.5).fn === 1 && close(f(0.5).f1, 4 / 7), "F1 = 4/7 at 0.5");
  check(f(0.3).tp === 3 && close(f(0.3).f1, 6 / 7), "F1 = 6/7 at 0.3");
  check(f(0.7).tp === 1 && close(f(0.7).f1, 2 / 7), "F1 = 2/7 at 0.7");
  // (c) greedy vs optimal
  const Tc = EV.tableFromIou([[0.60, 0.40], [0.35, 0]]);
  check(EV.matchGreedy(Tc, 0.3).tp === 1 && EV.matchOptimal(Tc, 0.3).tp === 2, "greedy 1 match, optimal 2");
  check(EV.matchOptimal(Tc, 0.3, false).tp === 2 && close(EV.matchOptimal(Tc, 0.3, false).pairs.reduce((s, q) => s + q.iou, 0), 0.75), "sum objective 0.75 > 0.60");
}

function testAP(): void {
  current = "average precision";
  const pr = EV.prFromScores([5, 4, 3, 2, 1], [1, 1, 0, 1, 0], 4);
  check(close(pr.ap, 0.6875), `AP = 0.6875 (${pr.ap})`);
  check(close(pr.precision[2], 2 / 3) && close(pr.recall[4], 0.75), "PR points");
  check(close(EV.matchingAccuracy({ tp: 3, fp: 2, fn: 1 }), 0.5), "matching accuracy 0.5");
  // AP ≤ 1, all TP → 1
  check(close(EV.prFromScores([3, 2, 1], [1, 1, 1], 3).ap, 1), "perfect AP");
  // brute: interpolated AP ≥ stepwise AP on random sequences
  const r = rng(3);
  for (let t = 0; t < 100; t++) {
    const n = 1 + Math.floor(10 * r.unif()), s = Array.from({ length: n }, () => r.unif()), tp = s.map(() => (r.unif() < 0.6 ? 1 : 0));
    const p = EV.prFromScores(s, tp, n + 1);
    check(p.ap >= p.apStep - 1e-12 && p.ap <= 1 + 1e-12, "interpolated ≥ stepwise");
  }
}

function testBoundary(): void {
  current = "boundary distances";
  const d = EV.pointSetDistances([[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1], [1, 4]]);
  check(close(d.hausdorff, 4) && close(d.asd, 10 / 7), `Hausdorff 4, ASD 10/7 (${d.hausdorff}, ${d.asd})`);
  // masks: compare the EDT route with brute force over boundary pixels
  const r = rng(5);
  for (let t = 0; t < 10; t++) {
    const w = 40, h = 32;
    const a = rasteriseEllipses(w, h, [{ x: 20 + 3 * r.norm(), y: 16, a: 10, b: 7, angle: r.unif() }]);
    const b = rasteriseEllipses(w, h, [{ x: 20, y: 16 + 2 * r.norm(), a: 9, b: 8, angle: r.unif() }, ...(t % 2 ? [{ x: 4, y: 4, a: 1.5, b: 1.5, angle: 0 }] : [])]);
    const m = EV.maskBoundaryDistances(a, b, w, h);
    const pts = (mk: Uint8Array) => { const o: [number, number][] = []; for (let i = 0; i < w * h; i++) if (mk[i]) o.push([i % w, Math.floor(i / w)]); return o; };
    const ref = EV.pointSetDistances(pts(EV.boundaryMask(a, w, h)), pts(EV.boundaryMask(b, w, h)));
    check(close(m.hausdorff, ref.hausdorff, 1e-6) && close(m.asd, ref.asd, 1e-6) && close(m.hd95, ref.hd95, 1e-6), "EDT route = brute force");
  }
  // contourdist model: uniform dilation by 1 px → ASD ≈ 1, Hausdorff ≈ 1
  const truth = CD.truthCurve(1), dil = CD.offsetCurve(truth, 1);
  const s = CD.curveDistances(truth, dil);
  check(Math.abs(s.asd - 1) < 0.05 && Math.abs(s.hausdorff - 1) < 0.1, `offset curve: ASD ${s.asd.toFixed(3)}, H ${s.hausdorff.toFixed(3)}`);
}

function testStats(): void {
  current = "stats";
  // tabulated t quantiles
  const tab: [number, number][] = [[1, 12.706], [2, 4.303], [3, 3.182], [4, 2.776], [5, 2.571], [10, 2.228], [22, 2.074], [30, 2.042]];
  for (const [df, q] of tab) check(Math.abs(ST.tQuantile(0.975, df) - q) < 1e-3, `t_0.975(${df}) = ${q}: ${ST.tQuantile(0.975, df)}`);
  check(Math.abs(ST.normQuantile(0.975) - 1.959964) < 1e-6, "z_0.975");
  check(Math.abs(ST.normCdf(-1.96) - 0.0249979) < 1e-6, "Φ(−1.96)");
  check(Math.abs(ST.tCdf(1, 1) - 0.75) < 1e-9, "Cauchy CDF(1) = 0.75");
  // by-hand (a) of the replicates unit
  const ctl = [[10, 11, 12, 11], [14, 15, 13, 14], [8, 9, 7, 8]], trt = [[12, 13, 14, 13], [15, 16, 14, 15], [10, 9, 11, 10]];
  const mc = ctl.map(ST.mean), mt = trt.map(ST.mean);
  const pd = ST.tTestPaired(mc, mt);
  check(close(pd.t, 5, 1e-9) && close(pd.estimate, 5 / 3), `paired t = 5 (${pd.t})`);
  const um = ST.tTestWelch(mc, mt);
  check(Math.abs(um.t - 0.74) < 0.01, `unpaired means t ≈ 0.74 (${um.t})`);
  const cn = ST.tTestPooled(ctl.flat(), trt.flat());
  check(Math.abs(cn.t - 1.65) < 0.01, `cells as n t ≈ 1.65 (${cn.t})`);
  // Bland–Altman by hand
  const ba = ST.blandAltman([20, 35, 50, 65, 80, 100], [19, 33, 46, 60, 72, 88]);
  check(Math.abs(ba.bias + 5.333) < 1e-3 && Math.abs(ba.sd - 4.08) < 0.005, `bias −5.33, SD 4.08 (${ba.bias}, ${ba.sd})`);
  // multiple testing by hand
  const p = [0.001, 0.004, 0.012, 0.02, 0.03, 0.2, 0.4, 0.5, 0.7, 0.9];
  check(p.filter(v => v <= 0.05).length === 5, "5 uncorrected");
  check(ST.bonferroni(p).filter(Boolean).length === 2, "Bonferroni 2");
  check(ST.benjaminiHochberg(p).k === 4, `BH 4 (${ST.benjaminiHochberg(p).k})`);
  // BH vs brute definition on random p
  const r = rng(11);
  for (let t = 0; t < 200; t++) {
    const m = 1 + Math.floor(30 * r.unif()), ps = Array.from({ length: m }, () => r.unif() ** 3);
    const s = [...ps].sort((a, b) => a - b);
    let k = 0; for (let j = 1; j <= m; j++) if (s[j - 1] <= j * 0.05 / m) k = j;
    check(ST.benjaminiHochberg(ps).k === k, "BH = brute force");
  }
  // correct analysis has 5 % false positives; cells-as-n inflates
  const rr = rng(2024);
  let fpMeans = 0, fpCells = 0, fpPaired = 0;
  const N = 4000;
  for (let i = 0; i < N; i++) {
    // independent design: cells-as-n inflates, experiment means do not; paired design: paired test at 5 %
    const a = ST.analyseTwoLevel(ST.simulateTwoLevel(rr, { experiments: 4, cells: 20, sdBetween: 1, sdWithin: 2, effect: 0, design: "independent" }));
    if (a.means.p < 0.05) fpMeans++;
    if (a.cells.p < 0.05) fpCells++;
    const b = ST.analyseTwoLevel(ST.simulateTwoLevel(rr, { experiments: 4, cells: 20, sdBetween: 1, sdWithin: 2, effect: 0, design: "paired" }));
    if (b.paired!.p < 0.05) fpPaired++;
  }
  check(Math.abs(fpPaired / N - 0.05) < 0.015, `paired FPR ≈ 0.05 (${fpPaired / N})`);
  check(fpMeans / N < 0.065, `means (unpaired) FPR ≤ 0.05 (${fpMeans / N})`);
  check(fpCells / N > 0.2, `cells-as-n FPR inflated (${fpCells / N})`);
  // cells-as-n formula vs simulation (z-test with known variances, large E)
  const m = 100, rho = 0.1, E = 50, sb = Math.sqrt(rho), sw = Math.sqrt(1 - rho);
  let rej = 0;
  const R = 3000, r2 = rng(99);
  for (let i = 0; i < R; i++) {
    let s0 = 0, s1 = 0;
    for (const c of [0, 1]) for (let e = 0; e < E; e++) { const u = gauss(r2, 0, sb); for (let k = 0; k < m; k++) { const v = u + gauss(r2, 0, sw); if (c) s1 += v; else s0 += v; } }
    const z = (s1 / (E * m) - s0 / (E * m)) / Math.sqrt(2 / (E * m));
    if (Math.abs(z) > 1.959964) rej++;
  }
  check(Math.abs(rej / R - ST.pseudoRepFPR(m, rho)) < 0.03, `FPR formula ${ST.pseudoRepFPR(m, rho).toFixed(3)} vs simulation ${(rej / R).toFixed(3)}`);
  check(Math.abs(ST.pseudoRepFPR(100, 0.1) - 0.55) < 0.01, "FPR(100, 0.1) ≈ 0.55");
  check(close(ST.median([3, 1, 2]), 2) && close(ST.mad([1, 2, 3, 4, 100]), 1), "median, MAD");
}

function testDemoModels(): void {
  current = "objmatch";
  const sc = OM.makeScene(1);
  check(sc.truth.count >= 12 && sc.truth.count <= 20, `scene has ~15 nuclei (${sc.truth.count})`);
  const perfect = OM.evaluate(sc.truth.labels, sc.truth.count, sc.truth.labels, sc.truth.count, new Float64Array(sc.truth.count).fill(1), 0.5, "optimal");
  check(perfect.scores.f1 === 1 && close(perfect.dice, 1), "truth vs itself: F1 = 1, Dice = 1");
  for (const p of OM.PRESETS) {
    const pred = OM.predict(sc, p.settings);
    check(pred.count >= 1, `preset ${p.id} gives predictions`);
  }
  const mergesOnly = OM.PRESETS.find(p => p.id === "dice")!;
  const pm = OM.predict(sc, mergesOnly.settings);
  const ev = OM.evaluate(sc.truth.labels, sc.truth.count, pm.labels, pm.count, pm.scores, 0.5, "optimal");
  check(ev.dice > 0.9 && ev.scores.fn > 0, `merges only: Dice ${ev.dice.toFixed(3)} > 0.9 with FN ${ev.scores.fn}`);

  current = "blandaltman";
  const d = BA.makeData(BA.DEFAULTS);
  check(d.manual.length === 40, "40 images");
  const prop = BA.makeData({ ...BA.DEFAULTS, ...BA.PRESETS.find(q => q.id === "corr")!.settings });
  const a = ST.blandAltman(prop.manual, prop.auto);
  check(a.r > 0.98 && Math.abs(a.bias) > 5, `r ${a.r.toFixed(3)} high, bias ${a.bias.toFixed(1)} large`);

  current = "leakage";
  const t0 = Date.now();
  const L = LK.makeSet({ ...LK.DEFAULTS, ...LK.PRESETS.find(q => q.id === "nothing")!.settings });
  const res = LK.evaluateAll(L, { ...LK.DEFAULTS, ...LK.PRESETS.find(q => q.id === "nothing")!.settings });
  const ms = Date.now() - t0;
  const byImage = res.find(q => q.strategy === "image")!, byPix = res.find(q => q.strategy === "pixels")!;
  check(Math.abs(byImage.mean - 0.5) < 0.05, `nothing to learn: by image ≈ 0.5 (${byImage.mean.toFixed(3)})`);
  check(byPix.mean > 0.75, `nothing to learn: random pixels high (${byPix.mean.toFixed(3)})`);
  console.log(`  leakage: all strategies in ${ms} ms; pixels ${byPix.mean.toFixed(3)}, image ${byImage.mean.toFixed(3)}`);

  current = "testpeek";
  const o = TP.runMany({ ...TP.DEFAULTS, k: 50, n: 100 }, 1000, 1);
  check(o.optimism > 0.05 && o.optimism < 0.1, `50 settings, 100 objects: optimism ${o.optimism.toFixed(3)}`);
  const v = TP.runMany({ ...TP.DEFAULTS, k: 50, n: 100, validation: true }, 1000, 1);
  check(Math.abs(v.optimism) < 0.01, `with validation: optimism ${v.optimism.toFixed(3)}`);

  current = "pseudorep";
  const fr = PR.runMany({ ...PR.DEFAULTS, effect: 0 }, 1000, 3);
  check(fr.cells > 0.2 && Math.abs(fr.means - 0.05) < 0.025, `null: cells ${fr.cells.toFixed(3)}, means ${fr.means.toFixed(3)}`);

  // summary-statistic repetitions agree with full simulations
  const sm = PR.runMany({ ...PR.DEFAULTS, m: 10 }, 3000, 5);
  let fc = 0;
  for (let i = 0; i < 3000; i++) if (PR.runOnce({ ...PR.DEFAULTS, m: 10 }, 1000 + i).res.cells.p < 0.05) fc++;
  check(Math.abs(sm.cells - fc / 3000) < 0.03, `summary vs full simulation, cells-as-n FPR ${sm.cells.toFixed(3)} vs ${(fc / 3000).toFixed(3)}`);

  current = "featurescan";
  const pl = FS.makePlate({ ...FS.DEFAULTS, fracReal: 0, rowGradient: 0, colGradient: 0, edge: 0, features: 300 }, 1);
  const an = FS.analyse(pl);
  check(an.raw >= 5 && an.raw <= 28, `300 null features: ${an.raw} raw hits (≈ 15)`);
  const conf = FS.analyse(FS.makePlate({ ...FS.DEFAULTS, ...FS.PRESETS.find(q => q.id === "confounded")!.settings }, 1));
  check(conf.falseRaw > 50, `plate gradient + left layout: many false hits (${conf.falseRaw})`);
}

testMatching();
testAP();
testBoundary();
testStats();
testDemoModels();
console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
