// Reference checks for lib/fret.ts (FRET day): efficiency curve and its inverse, κ² for the
// textbook geometries and its isotropic average, Förster radius scaling, the three-image
// forward model and its correction (exact E, zero for no FRET, coefficients from controls),
// the incomplete-bleaching correction, delta-method CV and bias of a ratio of Poisson counts
// against Monte Carlo, calibration inverses, gain invariance of a ratio, and the scenes.
// Run:
//   node_modules/.bin/esbuild demos/test/fret-test.ts --bundle --platform=node --log-level=warning | node
import * as F from "../src/lib/fret";
import { rng, poisson } from "../src/lib/random";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---------------------------------------------------------------------------------------
current = "efficiency";
{
  check(close(F.efficiency(5, 5), 0.5), "E(R0) = 0.5");
  check(close(F.efficiency(2.5, 5), 64 / 65), "E(R0/2) = 64/65");
  check(close(F.efficiency(10, 5), 1 / 65), "E(2R0) = 1/65");
  for (const r of [1, 2.5, 4, 5, 6.3, 9]) check(close(F.distanceFromE(F.efficiency(r, 5), 5), r, 1e-9), `distanceFromE inverts E at r = ${r}`);
  // dE/dr against a central difference; its minimum (steepest) at r = R0 with value −1.5/R0
  for (const r of [3, 5, 7]) {
    const h = 1e-5, num = (F.efficiency(r + h, 5) - F.efficiency(r - h, 5)) / (2 * h);
    check(close(F.dEdr(r, 5), num, 1e-6), `dE/dr at ${r}`);
  }
  check(close(F.dEdr(5, 5), -1.5 / 5), "slope at R0 = −1.5/R0");
  let best = 0, bestR = 0;
  for (let r = 1; r <= 10; r += 0.001) { const s = -F.dEdr(r, 5); if (s > best) { best = s; bestR = r; } }
  check(Math.abs(bestR - 5 * (5 / 7) ** (1 / 6)) < 0.002, `steepest at (5/7)^(1/6) R0 = 0.945 R0 (found ${bestR.toFixed(3)})`);
  // rates
  const q = F.efficiencyFromRates(4, 5, 2.5);
  check(close(q.E, F.efficiency(4, 5)), "rates give the same E");
  check(close(1 - q.tauDA / 2.5, q.E), "E = 1 − τDA/τD");
  check(close(F.lifetimeWithAcceptor(2.5, q.E), q.tauDA), "lifetimeWithAcceptor");
}

// ---------------------------------------------------------------------------------------
current = "kappa2";
{
  check(close(F.kappa2([1, 0, 0], [1, 0, 0], [1, 0, 0]), 4), "collinear along r: 4");
  check(close(F.kappa2([0, 1, 0], [0, 1, 0], [1, 0, 0]), 1), "parallel side by side: 1");
  check(close(F.kappa2([0, 1, 0], [0, 0, 1], [1, 0, 0]), 0), "perpendicular: 0");
  check(close(F.kappa2([1, 0, 0], [0, 1, 0], [1, 0, 0]), 0), "one along r, one across: 0");
  check(close(F.kappa2Planar(0, 0), 4) && close(F.kappa2Planar(Math.PI / 2, Math.PI / 2), 1), "planar helper");
  // isotropic average 2/3 (Monte Carlo over random unit vectors)
  const r = rng(3), N = 400000;
  const rv = (): F.V3 => { for (;;) { const v: F.V3 = [2 * r.unif() - 1, 2 * r.unif() - 1, 2 * r.unif() - 1]; const n = Math.hypot(...v); if (n > 0.05 && n <= 1) return v; } };
  let s = 0;
  for (let i = 0; i < N; i++) s += F.kappa2(rv(), rv(), [1, 0, 0]);
  check(Math.abs(s / N - 2 / 3) < 0.01, `isotropic mean κ² = ${(s / N).toFixed(4)} ≈ 2/3`);
}

// ---------------------------------------------------------------------------------------
current = "forster radius";
{
  const J = F.overlapIntegral(F.CYAN_DONOR.em, F.YELLOW_ACCEPTOR.ex, F.YELLOW_ACCEPTOR.eps);
  const R = F.forsterRadius(0.4, 2 / 3, 1.4, J);
  check(R > 4 && R < 6, `example pair R0 = ${R.toFixed(2)} nm in 4..6`);
  check(close(F.forsterRadius(0.4, 2 / 3, 1.4, 2 * J) / R, 2 ** (1 / 6), 1e-12), "double overlap: × 2^(1/6)");
  check(close(F.forsterRadius(0.4, 4, 1.4, J) / R, 6 ** (1 / 6), 1e-12), "κ² 4 vs 2/3: × 6^(1/6)");
  check(F.forsterRadius(0.4, 0, 1.4, J) === 0, "κ² = 0: R0 = 0");
  // J scales with ε and moving the acceptor away reduces it
  const far = F.overlapIntegral(F.CYAN_DONOR.em, { ...F.YELLOW_ACCEPTOR.ex, peak: 600 }, F.YELLOW_ACCEPTOR.eps);
  check(far < J, "less overlap, smaller J");
}

// ---------------------------------------------------------------------------------------
current = "three cube";
{
  const r = rng(5);
  for (let t = 0; t < 200; t++) {
    const p = { ND: 100 * r.unif(), NA: 100 * r.unif(), E: r.unif(), fPaired: r.unif(), d: r.unif(), a: 0.3 * r.unif(), G: 0.5 + 3 * r.unif(), cDD: 0.5 + r.unif(), cAA: 0.5 + r.unif() };
    const v = F.threeCube(p);
    const Fc = F.correctFc(v.IDA, v.IDD, v.IAA, p.d, p.a);
    check(close(Fc, v.Fc, 1e-9), "Fc recovered");
    if (p.ND > 1) check(close(F.eFret(Fc, v.IDD, p.G), p.fPaired * p.E, 1e-9), "E_app = f E exactly");
    // index dependence on stoichiometry
    const v2 = F.threeCube({ ...p, NA: 2 * p.NA });
    check(close(F.eFret(F.correctFc(v2.IDA, v2.IDD, v2.IAA, p.d, p.a), v2.IDD, p.G), F.eFret(Fc, v.IDD, p.G), 1e-9), "E unchanged with twice the acceptor");
    if (Fc > 1e-6) check(close(F.perAcceptorIndex(v2.Fc, v2.IAA) / F.perAcceptorIndex(v.Fc, v.IAA), 0.5, 1e-9), "Fc/IAA halves");
    if (Fc > 1e-6) check(close(F.nfretIndex(v2.Fc, v2.IDD, v2.IAA) / F.nfretIndex(v.Fc, v.IDD, v.IAA), Math.SQRT1_2, 1e-9), "index × 1/√2");
  }
  const z = F.threeCube({ ND: 50, NA: 70, E: 0.4, fPaired: 0, d: 0.5, a: 0.1, G: 2 });
  check(close(F.correctFc(z.IDA, z.IDD, z.IAA, 0.5, 0.1), 0), "no FRET: Fc = 0");
  // the matrix picture: L = [[1,0,0],[d,1,a],[0,0,1]] and forward substitution
  // by-hand numbers of the exercise
  check(close(F.correctFc(500, 400, 1000, 0.5, 0.1), 200), "B: Fc = 200");
  check(close(F.eFret(200, 400, 2), 0.2), "B: E = 0.2");
  check(close(F.fcVariance(500, 400, 1000, 0.5, 0.1), 610), "B: Var Fc = 610");
  // slopes from noise-free controls equal d and a
  const fld = F.fretField({ seed: 5 });
  const m = F.measureField(fld, { photons: 300, offset: 0, noise: false });
  const donorMask = Uint8Array.from(fld.labels, L => (L && fld.cls[L - 1] === "donor" ? 1 : 0));
  const accMask = Uint8Array.from(fld.labels, L => (L && fld.cls[L - 1] === "acceptor" ? 1 : 0));
  check(close(F.slopeThroughOriginMasked(m.IDD, m.IDA, donorMask), fld.d, 1e-6), "d from donor-only control");
  check(close(F.slopeThroughOriginMasked(m.IAA, m.IDA, accMask), fld.a, 1e-6), "a from acceptor-only control");
  // per-cell E_app equals f E for every cell with donor (noise-free)
  for (let k = 0; k < fld.count; k++) {
    if (fld.cls[k] === "acceptor") continue;
    let sFc = 0, sD = 0;
    for (let i = 0; i < fld.labels.length; i++) if (fld.labels[i] === k + 1) { sFc += F.correctFc(m.IDA[i], m.IDD[i], m.IAA[i], fld.d, fld.a); sD += m.IDD[i]; }
    check(close(F.eFret(sFc, sD, fld.G), fld.trueE[k] * fld.trueF[k], 1e-5), `cell ${k + 1} (${fld.cls[k]}) E_app`);
  }
  check(fld.cls.includes("donor") && fld.cls.includes("acceptor") && fld.cls.includes("free") && fld.cls.includes("fusion"), "all classes present");
}

// ---------------------------------------------------------------------------------------
current = "bleaching";
{
  check(close(F.eFromBleach(600, 800), 0.25), "600/800 → 0.25");
  check(Math.abs(F.correctIncomplete(0.25, 0.8) - 0.2941) < 1e-4, "incomplete 0.8 → 0.294");
  check(Math.abs(F.correctDonorBleach(600, 800, 1, 0.95) - 0.2875) < 1e-9, "donor bleaching 5 % → 0.2875");
  for (const E of [0.05, 0.3, 0.7]) for (const b of [0.2, 0.5, 0.9, 1]) {
    check(close(F.correctIncomplete(F.eObservedIncomplete(E, b), b), E, 1e-12), `incomplete inverts (E ${E}, b ${b})`);
    // forward model against the donor intensities
    const pre = F.donorRelative(E, 1, 1), post = F.donorRelative(E, 1, 1 - b);
    check(close(F.eFromBleach(pre, post), F.eObservedIncomplete(E, b), 1e-12), "eObservedIncomplete = 1 − pre/post");
  }
  // fraction paired: E_app = f E for complete bleaching
  check(close(F.eFromBleach(F.donorRelative(0.4, 0.5, 1), F.donorRelative(0.4, 0.5, 0)), 0.2), "half paired at 0.4 → 0.2");
  check(Math.abs(F.bleachSigmaE(600, 800) - 0.0405) < 1e-3, "σE at 600/800");
  // scene: noise-free region E equals the formula
  const s = F.bleachScene({ E: 0.3, f: 1, pulses: 30, noise: false });
  let pre = 0, post = 0;
  for (let i = 0; i < s.region.length; i++) if (s.region[i] && s.labels[i] !== s.accOnlyCell) { pre += s.donorPre[i]; post += s.donorPost[i]; }
  check(Math.abs(F.eFromBleach(pre, post) - F.eObservedIncomplete(0.3, 1 - 0.7 ** 30)) < 1e-6, "scene region E (noise-free)");
  const ser = s.series;
  check(close(1 - ser.regionDonor[0] / ser.regionDonor[30], F.eObservedIncomplete(0.3, 1 - 0.7 ** 30), 1e-6), "series consistent");
  // monte carlo of σE
  const r = rng(9);
  const es: number[] = [];
  for (let i = 0; i < 40000; i++) es.push(F.eFromBleach(poisson(r, 600), poisson(r, 800)));
  const m = es.reduce((a, b) => a + b, 0) / es.length, sd = Math.sqrt(es.reduce((a, b) => a + (b - m) ** 2, 0) / es.length);
  check(Math.abs(sd / F.bleachSigmaE(600, 800) - 1) < 0.05, `MC σE ${sd.toFixed(4)} vs delta ${F.bleachSigmaE(600, 800).toFixed(4)}`);
}

// ---------------------------------------------------------------------------------------
current = "ratios";
{
  for (const [muA, muD] of [[100, 400], [400, 100], [150, 150], [1000, 300]]) {
    const mc = F.ratioDraws({ muA, muD, n: 200000, seed: 11 });
    let s = 0, ss = 0, n = 0;
    for (const v of mc.ratios) if (Number.isFinite(v)) { s += v; ss += v * v; n++; }
    const mean = s / n, cv = Math.sqrt(ss / n - mean * mean) / mean;
    check(Math.abs(cv / F.ratioCVPredicted(muA, muD) - 1) < 0.05, `CV at ${muA}/${muD}: MC ${cv.toFixed(4)} vs ${F.ratioCVPredicted(muA, muD).toFixed(4)}`);
    const bias = mean / (muA / muD) - 1, pb = F.ratioMeanPredicted(muA, muD) / (muA / muD) - 1;
    check(Math.abs(bias - pb) < 0.05 * Math.max(pb, 0.002) + 0.0015, `bias at ${muA}/${muD}: MC ${bias.toFixed(5)} vs ${pb.toFixed(5)}`);
  }
  // shared flicker cancels: CV with flicker ≈ CV without (Poisson part only)
  const a = F.ratioDraws({ muA: 500, muD: 500, n: 100000, seed: 2 }), b = F.ratioDraws({ muA: 500, muD: 500, n: 100000, flicker: 0.2, seed: 2 });
  const cvOf = (x: Float64Array) => { let s = 0, ss = 0; for (const v of x) { s += v; ss += v * v; } const m = s / x.length; return Math.sqrt(ss / x.length - m * m) / m; };
  check(cvOf(a.D) < 0.06 && cvOf(b.D) > 0.18, "flicker makes each channel noisy");
  check(Math.abs(cvOf(b.ratios) / cvOf(a.ratios) - 1) < 0.1, `but not the ratio (${cvOf(a.ratios).toFixed(4)} → ${cvOf(b.ratios).toFixed(4)})`);
  // ratio of sums vs mean of ratios; gain invariance
  const A = [10, 50, 100, 200], D = [5, 50, 200, 400];
  check(close(F.ratioOfSums(A, D), 360 / 655), "ratio of sums");
  check(close(F.meanOfRatios(A, D), (2 + 1 + 0.5 + 0.5) / 4), "mean of ratios");
  check(close(F.ratioOfSums(A.map(v => 3.7 * v), D.map(v => 3.7 * v)), F.ratioOfSums(A, D)), "common gain cancels");
  check(Math.abs((100 + 50) / (400 + 50) - 1 / 3) < 1e-12, "offset does not cancel");
  // calibration
  for (const c of [10, 200, 1800, 3800]) check(close(F.calibrate(F.calibrateInverse(c, 0.5, 2.5, 200), 0.5, 2.5, 200), c, 1e-9), `calibrate inverts at ${c}`);
  check(close(F.calibrate(1.5, 0.5, 2.5, 200), 200) && close(F.calibrate(2.3, 0.5, 2.5, 200), 1800) && close(F.calibrate(2.4, 0.5, 2.5, 200), 3800), "plan's calibration numbers");
  const h = 1e-6;
  check(close(F.dcdR(2, 0.5, 2.5, 200), (F.calibrate(2 + h, 0.5, 2.5, 200) - F.calibrate(2 - h, 0.5, 2.5, 200)) / (2 * h), 1e-5), "dc/dR");
  // binning
  const bs = F.binSum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 4, 4, 2);
  check(bs.w === 2 && bs.data[0] === 1 + 2 + 5 + 6 && bs.data[3] === 11 + 12 + 15 + 16, "binSum");
  check(F.shiftMap([1, 2, 3, 4], 2, 2, 1, 0)[1] === 1 && F.shiftMap([1, 2, 3, 4], 2, 2, 1, 0)[0] === 0, "shiftMap");
  // sensor scene: noise-free ratio = calibrateInverse(conc) inside cells
  const s = F.sensorScene();
  const fr = F.sensorFrame(s, 12, { photons: 200, offset: 0, noise: false });
  let ok = true;
  for (let i = 0; i < fr.D.length; i++) if (s.labels[i]) ok = ok && close(fr.A[i] / fr.D[i], F.calibrateInverse(s.conc[12][i], s.Rmin, s.Rmax, s.Kd), 1e-5);
  check(ok, "sensor ratio follows calibration");
}

console.log(`fret-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
