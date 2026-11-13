// Reference checks for the spatial statistics day: lib/pointpattern.ts. Run:
//   node_modules/.bin/esbuild demos/test/spatialstats-test.ts --bundle --platform=node --log-level=warning | node
// Checks: K̂ of CSR is unbiased with translation and isotropic corrections and biased low without
// one; Ĝ and F̂ of CSR match 1 − exp(−ρπr²); ĝ of a hard-core pattern vanishes below h; the
// Thomas K̂ matches its formula; random labelling gives K11 ≈ K22 ≈ K12; the inhomogeneous K of an
// inhomogeneous Poisson pattern (true ρ) is ≈ πr²; Monte Carlo ranks are uniform under CSR; the MAD
// test keeps its level while the pointwise envelope does not; χ² CDF and set covariance are exact.
import { rng } from "../src/lib/random";
import * as PP from "../src/lib/pointpattern";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const meanOf = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

function testBasics(): void {
  current = "basics";
  // χ² CDF against known values
  check(close(PP.chi2Cdf(3.841459, 1), 0.95, 1e-5), "chi2 1 df");
  check(close(PP.chi2Cdf(36.415, 24), 0.95, 1e-4), "chi2 24 df upper 5 % at 36.4: " + PP.chi2Cdf(36.415, 24));
  check(close(PP.chi2Upper(0.05, 24), 36.415, 1e-3), "chi2 quantile");
  // circle fraction in a rectangle: centre far inside = 1; on an edge = 1/2; in a corner = 1/4
  const win = PP.rectWindow(100, 50);
  check(close(PP.circleFraction(win, 50, 25, 10), 1, 1e-12), "circle inside");
  check(close(PP.circleFraction(win, 50, 0, 10), 0.5, 1e-9), "circle on edge");
  check(close(PP.circleFraction(win, 0, 0, 10), 0.25, 1e-9), "circle in corner");
  // a point 5 from the left edge, radius 10: the outside arc is 2·acos(0.5) = 120° → fraction 2/3
  check(close(PP.circleFraction(win, 5, 25, 10), 2 / 3, 1e-9), "circle crossing an edge");
  // mask set covariance (FFT) = rectangle formula for a full mask
  const m = PP.maskWindow(new Uint8Array(40 * 30).fill(1), 40, 30);
  for (const [dx, dy] of [[0, 0], [5, 3], [-12, 7], [20, -10]]) check(close(PP.setCovariance(m, dx, dy), (40 - Math.abs(dx)) * (30 - Math.abs(dy)), 1e-6), `set covariance ${dx},${dy}`);
  // translation weight from the plan: Δ = (20, 10) in 100 × 50 → 5000/(80·40) = 1.5625
  check(close(win.area / PP.setCovariance(win, 20, 10), 1.5625, 1e-12), "translation weight");
  // quadrat counts
  const pts = PP.makePoints([1, 2, 3, 6, 7, 8.5, 8.6, 8.7], [1, 1, 1, 6, 6, 8, 8, 8]);
  const q = PP.quadratCounts(pts, PP.rectWindow(10, 10), 2, 2);
  check(q.mean === 2 && q.counts[0] === 3 && q.counts[3] === 5, "quadrat counts");
  // nearest neighbours vs brute force
  const r = rng(3), p = PP.binomialProcess(r, 60, win), { d, idx } = PP.nnDistances(p);
  let ok = true;
  for (let i = 0; i < 60; i++) {
    let best = Infinity;
    for (let j = 0; j < 60; j++) if (j !== i) best = Math.min(best, Math.hypot(p.x[i] - p.x[j], p.y[i] - p.y[j]));
    if (!close(best, d[i], 1e-12) || !close(Math.hypot(p.x[i] - p.x[idx[i]], p.y[i] - p.y[idx[i]]), best, 1e-12)) ok = false;
  }
  check(ok, "nn distances");
  // pairData finds exactly the pairs within rmax
  const pd = PP.pairData(p, win, 15, "none");
  let brute = 0;
  for (let i = 0; i < 60; i++) for (let j = 0; j < 60; j++) if (i !== j && Math.hypot(p.x[i] - p.x[j], p.y[i] - p.y[j]) <= 15) brute++;
  check(pd.d.length === brute, `pair count ${pd.d.length} vs ${brute}`);
  // mergeClose leaves no pair closer than d
  const mg = PP.mergeClose(PP.binomialProcess(rng(4), 200, PP.rectWindow(100, 100)), 4);
  check(PP.nnDistances(mg).d.every(v => v >= 4), "merge leaves no close pairs");
}

function testKUnbiased(): void {
  current = "K of CSR";
  const L = 1, win = PP.rectWindow(L, L), radii = [0.1, 0.25], n = 100, trials = 400, r = rng(11);
  const acc: Record<string, number[][]> = { none: [[], []], translation: [[], []], isotropic: [[], []], border: [[], []] };
  for (let t = 0; t < trials; t++) {
    const p = PP.binomialProcess(r, n, win);
    for (const c of Object.keys(acc) as PP.KCorr[]) {
      const K = PP.Kest(p, win, radii, c);
      acc[c][0].push(K[0]); acc[c][1].push(K[1]);
    }
  }
  const rel = (c: string, k: number) => meanOf(acc[c][k]) / (Math.PI * radii[k] ** 2) - 1;
  for (const c of ["translation", "isotropic"]) for (const k of [0, 1]) check(Math.abs(rel(c, k)) < 0.02, `${c} r=${radii[k]}: rel bias ${rel(c, k).toFixed(4)}`);
  check(Math.abs(rel("border", 0)) < 0.03, `border r=0.1: ${rel("border", 0).toFixed(4)}`);
  check(rel("none", 0) < -0.06 && rel("none", 0) > -0.11, `none r=0.1 about −8 %: ${rel("none", 0).toFixed(4)}`);
  check(rel("none", 1) < -0.17 && rel("none", 1) > -0.24, `none r=0.25 about −20 %: ${rel("none", 1).toFixed(4)}`);
  console.log(`  K bias (n=100, unit square): none ${(100 * rel("none", 0)).toFixed(1)} % / ${(100 * rel("none", 1)).toFixed(1)} %, translation ${(100 * rel("translation", 0)).toFixed(1)} / ${(100 * rel("translation", 1)).toFixed(1)}, isotropic ${(100 * rel("isotropic", 0)).toFixed(1)} / ${(100 * rel("isotropic", 1)).toFixed(1)}, border ${(100 * rel("border", 0)).toFixed(1)} / ${(100 * rel("border", 1)).toFixed(1)}`);
  // a mask window: translation correction with the FFT set covariance is unbiased too
  const mwin = PP.maskWindow(PP.blobMask(rng(2), 120, 120), 120, 120), r2 = rng(12), acc2: number[] = [];
  for (let t = 0; t < 150; t++) acc2.push(PP.Kest(PP.binomialProcess(r2, 120, mwin), mwin, [15], "translation")[0]);
  const relm = meanOf(acc2) / (Math.PI * 225) - 1;
  check(Math.abs(relm) < 0.04, `mask translation rel bias ${relm.toFixed(4)}`);
}

function testGF(): void {
  current = "G and F";
  const win = PP.rectWindow(200, 200), n = 150, rho = n / win.area, radii = [3, 6, 9, 12], r = rng(21);
  const G: number[][] = radii.map(() => []), F: number[][] = radii.map(() => []);
  for (let t = 0; t < 200; t++) {
    const p = PP.binomialProcess(r, n, win);
    const g = PP.Gest(p, win, radii, "border"), f = PP.Fest(p, win, radii, "border", 5);
    radii.forEach((_, k) => { G[k].push(g[k]); F[k].push(f[k]); });
  }
  radii.forEach((rr, k) => {
    const th = PP.csrG(rr, rho), se = Math.sqrt(th * (1 - th) / n / 200) * 2;
    check(Math.abs(meanOf(G[k]) - th) < 2 * se + 0.01, `G(${rr}) ${meanOf(G[k]).toFixed(4)} vs ${th.toFixed(4)}`);
    check(Math.abs(meanOf(F[k]) - th) < 0.02, `F(${rr}) ${meanOf(F[k]).toFixed(4)} vs ${th.toFixed(4)}`);
  });
  // uncorrected G is biased low (looks regular)
  const r2 = rng(22), gn: number[] = [];
  for (let t = 0; t < 200; t++) gn.push(PP.Gest(PP.binomialProcess(r2, 30, PP.rectWindow(60, 60)), PP.rectWindow(60, 60), [12], "none")[0]);
  check(meanOf(gn) < PP.csrG(12, 30 / 3600), `uncorrected G low: ${meanOf(gn).toFixed(3)} < ${PP.csrG(12, 30 / 3600).toFixed(3)}`);
}

function testModels(): void {
  current = "models";
  // hard core: no pairs below h, g ≈ 0 there
  const win = PP.rectWindow(200, 200);
  const hc = PP.ssi(rng(31), 200, 8, win);
  check(hc.x.length === 200, "ssi placed 200");
  check(PP.nnDistances(hc).d.every(v => v >= 8), "ssi hard core");
  const g = PP.pcf(hc, win, [2, 4, 5], 1.5);
  check(g.every(v => v < 0.05), "g = 0 below h: " + Array.from(g).map(v => v.toFixed(3)).join(" "));
  // Thomas: averaged K̂ vs formula within 5 % (a 400 µm window: with only ~20 clusters per pattern
  // the ratio estimator |W|/n(n−1) is biased upwards by about 8 %)
  const kappa = 0.0005, mu = 8, sigma = 4, radii = [5, 10, 20], r = rng(32), acc: number[][] = radii.map(() => []), big = PP.rectWindow(400, 400);
  for (let t = 0; t < 80; t++) {
    const p = PP.thomas(r, kappa, mu, sigma, big), K = PP.Kest(p, big, radii, "translation");
    radii.forEach((_, k) => acc[k].push(K[k]));
  }
  radii.forEach((rr, k) => { const th = PP.thomasK(rr, kappa, sigma); check(Math.abs(meanOf(acc[k]) / th - 1) < 0.05, `Thomas K(${rr}) ${meanOf(acc[k]).toFixed(1)} vs ${th.toFixed(1)}`); });
  // Thomas intensity κμ
  let tot = 0;
  for (let t = 0; t < 100; t++) tot += PP.thomas(rng(100 + t), kappa, mu, sigma, win).x.length;
  check(Math.abs(tot / 100 / (kappa * mu * win.area) - 1) < 0.05, "Thomas intensity κμ: " + (tot / 100).toFixed(1));
  // random labelling: K11 ≈ K22 ≈ K12 (averaged)
  const r3 = rng(33), k11: number[] = [], k22: number[] = [], k12: number[] = [];
  for (let t = 0; t < 100; t++) {
    const base = PP.thomas(r3, 0.0008, 6, 5, win), p = PP.labelRandom(r3, base, 0.4);
    const pd = PP.pairData(p, win, 15, "translation");
    k11.push(PP.KfromPairs(pd, [15], { marks: p.marks, a: 1, b: 1 })[0]);
    k22.push(PP.KfromPairs(pd, [15], { marks: p.marks, a: 0, b: 0 })[0]);
    k12.push(PP.KfromPairs(pd, [15], { marks: p.marks, a: 1, b: 0 })[0]);
  }
  check(Math.abs(meanOf(k11) / meanOf(k12) - 1) < 0.06 && Math.abs(meanOf(k22) / meanOf(k12) - 1) < 0.06, `random labelling ${meanOf(k11).toFixed(0)} ${meanOf(k22).toFixed(0)} ${meanOf(k12).toFixed(0)}`);
  // mark connection under random labelling ≈ 2 p1 p2
  const p = PP.labelRandom(rng(34), PP.binomialProcess(rng(35), 600, win), 0.3);
  const mc = PP.markConnection(PP.pairData(p, win, 30, "none"), p.marks!, [10, 20], 3, 1, 0);
  check(Math.abs(mc[0] - 0.42) < 0.05 && Math.abs(mc[1] - 0.42) < 0.04, `mark connection ${Array.from(mc).map(v => v.toFixed(3))}`);
  // inhomogeneous Poisson with the true ρ: Kinhom ≈ πr²
  const rhoFn = (x: number, _y: number) => 0.001 + 0.006 * x / 200, r4 = rng(36), ki: number[] = [];
  for (let t = 0; t < 150; t++) {
    const q = PP.inhomPoisson(r4, rhoFn, 0.007, win);
    const rhoAt = Float64Array.from(q.x, (x, i) => rhoFn(x, q.y[i]));
    ki.push(PP.Kinhom(q, win, [10], rhoAt)[0]);
  }
  check(Math.abs(meanOf(ki) / (Math.PI * 100) - 1) < 0.05, `Kinhom ${meanOf(ki).toFixed(1)} vs ${(Math.PI * 100).toFixed(1)}`);
  // ... while the plain K says "clustered"
  const q = PP.inhomPoisson(rng(37), rhoFn, 0.007, win), Kp = PP.Kest(q, win, [10])[0];
  check(Kp > Math.PI * 100 * 1.1, `plain K of a gradient looks clustered: ${Kp.toFixed(0)}`);
}

function testMonteCarlo(): void {
  current = "Monte Carlo";
  // ranks uniform under CSR: test statistic L(5) − 5 with 19 sims, 500 trials; histogram of ranks
  const win = PP.rectWindow(1, 1), n = 100, radii = PP.linspace(0.01, 0.25, 25), ref = Float64Array.from(radii);
  const r = rng(41), idx = PP.rangeIdx(radii, [0.01, 0.25]);
  const L = (p: PP.Points) => PP.Lfun(PP.Kest(p, win, radii, "translation"));
  const ranks = new Array(20).fill(0);
  for (let t = 0; t < 500; t++) {
    const obs = L(PP.binomialProcess(r, n, win))[4];
    const sims = Array.from({ length: 19 }, () => L(PP.binomialProcess(r, n, win))[4]);
    ranks[PP.mcRank(obs, sims).rank - 1]++;
  }
  const chi = ranks.reduce((s, c) => s + (c - 25) ** 2 / 25, 0);
  check(chi < PP.chi2Upper(0.001, 19), `ranks uniform: X² = ${chi.toFixed(1)}`);
  // pointwise (39 sims) vs MAD rejection rates under CSR, 300 trials
  let pw = 0, mad = 0, pw19 = 0;
  const trials = 300;
  for (let t = 0; t < trials; t++) {
    const obs = L(PP.binomialProcess(r, n, win));
    const sims = Array.from({ length: 39 }, () => L(PP.binomialProcess(r, n, win)));
    if (PP.escapes(obs, PP.pointwiseEnvelope(sims), idx).length) pw++;
    if (PP.escapes(obs, PP.pointwiseEnvelope(sims.slice(0, 19)), idx).length) pw19++;
    if (PP.madTest(obs, sims, ref, idx).p <= 0.05) mad++;
  }
  console.log(`  CSR rejected (300 trials, 25 radii): pointwise 39 sims ${(100 * pw / trials).toFixed(1)} %, pointwise 19 sims ${(100 * pw19 / trials).toFixed(1)} %, MAD ${(100 * mad / trials).toFixed(1)} %`);
  check(pw / trials > 0.25 && pw / trials < 0.5, `pointwise 39: ${pw / trials}`);
  check(pw19 / trials > 0.45 && pw19 / trials < 0.75, `pointwise 19: ${pw19 / trials}`);
  check(mad / trials > 0.015 && mad / trials < 0.09, `MAD level: ${mad / trials}`);
}

testBasics();
testKUnbiased();
testGF();
testModels();
testMonteCarlo();
console.log(`spatialstats: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
