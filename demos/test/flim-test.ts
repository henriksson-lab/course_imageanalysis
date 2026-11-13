// Reference checks for the FLIM day: lib/decay.ts, lib/phasor.ts, lib/nlfit.ts. Run:
//   node_modules/.bin/esbuild demos/test/flim-test.ts --bundle --platform=node --log-level=warning | node
import * as D from "../src/lib/decay";
import * as Ph from "../src/lib/phasor";
import * as F from "../src/lib/nlfit";
import { rng, exponential } from "../src/lib/random";
import { dft1d } from "../src/lib/fft";
import { lstsq } from "../src/lib/linalg";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
const ax: D.TimeAxis = { T: 12.5, nBins: 256 };
const omega = D.omegaOf(80);

// ---------------------------------------------------------------------------------------
current = "periodic model";
{
  for (const tau of [0.3, 1, 2.5, 8, 30]) {
    const b = D.expBins(tau, ax);
    check(close(sum(b), 1, 1e-12), `bins sum to 1 (τ = ${tau})`);
    // numeric integral of the density over bin 10
    const d = D.binWidth(ax);
    let s = 0;
    for (let i = 0; i < 1000; i++) s += D.periodicDensity(10 * d + (i + 0.5) * d / 1000, tau, ax.T) * d / 1000;
    check(close(s, b[10], 1e-6), `bin integral = integral of the density (τ = ${tau})`);
  }
  const mu = D.expectedCounts({ tau: [2], f: [1] }, { sigma: 0.1, delay: 0.5 }, ax, 5000, 0);
  check(close(sum(mu), 5000, 1e-9), "expected counts integrate to N per period");
  check(close(D.wrappedFraction(4, 12.5), Math.exp(-3.125)), "wrapped fraction");
}

current = "IRF convolution";
{
  const spike = D.gaussianIRF(0, 0, ax);
  check(spike[0] === 1 && sum(spike) === 1, "spike IRF");
  const m = D.expBins(1.7, ax), c = D.convolveCircular(spike, m);
  check(m.every((v, j) => close(v, c[j], 1e-14)), "spike at 0 returns the model");
  const g = D.gaussianIRF(0.15, 2, ax);
  check(close(sum(g), 1, 1e-12), "Gaussian IRF sums to 1");
  check(close(D.meanArrival(g, ax), 2, 1e-3), "Gaussian IRF mean = delay");
  // means add under convolution (exact model; no wrap: short τ, early IRF)
  const conv = D.decayBins({ tau: [1], f: [1] }, { sigma: 0.15, delay: 2 }, ax);
  check(close(sum(conv), 1, 1e-9), "exact model sums to 1");
  check(Math.abs(D.meanArrival(conv, ax) - D.meanArrival(g, ax) - 1) < 1e-3, `means add under convolution: ${D.meanArrival(conv, ax) - D.meanArrival(g, ax)}`);
  // the discrete convolution of bins shifts by half a bin
  const conv2 = D.convolveCircular(g, D.expBins(1, ax));
  check(Math.abs(D.meanArrival(conv2, ax) - D.meanArrival(g, ax) - 1 + D.binWidth(ax) / 2) < 2e-4, "discrete convolution: half-bin shift");
  // exact model with σ → 0 and delay 0 equals the plain periodic exponential; wrap-around
  const e0 = D.decayBins({ tau: [8], f: [1] }, { sigma: 0, delay: 0 }, ax), e1 = D.expBins(8, ax);
  check(e0.every((v, j) => close(v, e1[j], 1e-9)), "exact model, spike IRF = periodic exponential (τ = 8 wraps)");
  const e2 = D.decayBins({ tau: [8], f: [1] }, { sigma: 0.2, delay: 12 }, ax);
  check(close(sum(e2), 1, 1e-7), "IRF near the end of the period wraps, still sums to 1");
}

current = "quick estimators";
{
  // two gates exact on a noise-free exponential (gates aligned with bins)
  const b = D.expBins(2, ax), d = D.binWidth(ax);
  const r = D.twoGateTau(b, ax, 0, 64 * d);
  check(close(r.tau, 2, 1e-9), `two gates: ${r.tau}`);
  // periodic mean and its inverse
  check(close(D.tauFromPeriodicMean(D.periodicMean(2.5, 12.5), 12.5), 2.5, 1e-6), "periodic mean inverse");
  // pile-up shortens the mean arrival time
  const o: D.TcspcOptions = { tau: [3], f: [1], T: 12.5, irfSigma: 0, irfDelay: 0, bgFraction: 0, rate: 0.01 };
  const lo = D.simulateTcspc(rng(1), o, ax, 50000), hi = D.simulateTcspc(rng(1), { ...o, rate: 0.6 }, ax, 50000);
  check(D.meanArrival(hi, ax) < D.meanArrival(lo, ax) - 0.2, "pile-up shortens τ");
  check(Math.abs(D.meanArrival(lo, ax) - D.periodicMean(3, 12.5)) < 0.05, "low rate: mean arrival ≈ periodic mean");
}

current = "fractions";
{
  const f = D.photonFractions([0.5, 0.5], [0.5, 3]);
  check(close(f[0], 0.5 / 3.5) && close(f[1], 3 / 3.5), "photon fractions");
  const a = D.amplitudeFractions(f, [0.5, 3]);
  check(close(a[0], 0.5) && close(a[1], 0.5), "amplitude fractions invert photon fractions");
  check(close(D.tauAmplitude([0.5, 0.5], [0.5, 3]), 1.75), "τ_a");
  check(close(D.tauIntensity([0.5, 0.5], [0.5, 3]), (0.125 + 4.5) / 1.75), "τ_f");
  check(close(D.fretDistance(D.fretEfficiency(6, 5), 5), 6, 1e-12), "FRET distance inverts efficiency");
}

// ---------------------------------------------------------------------------------------
current = "phasor";
{
  for (const tau of [0.4, 1, 2.5, 4]) {
    const P = Ph.phasor(D.expBins(tau, ax)), S = Ph.semicircle(omega * tau);
    const k = Ph.binningCorrection(ax.nBins);
    check(Math.hypot(P.g - k * S.g, P.s - k * S.s) < 2e-3, `binned exponential on the semicircle × sinc (τ = ${tau})`);
    const q = D.decayPhasor({ tau: [tau], f: [1] }, omega);
    check(close(q.g, S.g) && close(q.s, S.s), "decayPhasor = semicircle");
    check(close(Math.hypot(S.g - 0.5, S.s), 0.5, 1e-12), "on the circle");
    const lt = Ph.lifetimeFromPhasor(S, omega);
    check(close(lt.tauPhi, tau, 1e-9) && close(lt.tauM, tau, 1e-9), "τ_φ = τ_m for one exponential");
  }
  // phasor = conj(DFT bin 1) / bin 0, with the half-bin phase of bin centres
  const y = D.expectedCounts({ tau: [1.3, 3.1], f: [0.4, 0.6] }, D.gaussianIRF(0.2, 1, ax), ax, 1000, 0.5);
  const F1 = dft1d(y, new Float64Array(y.length)), P = Ph.phasor(y);
  const ph = Math.PI / ax.nBins, c1 = { g: F1.re[1] / F1.re[0], s: -F1.im[1] / F1.re[0] };
  const rot = Ph.cmul(c1, { g: Math.cos(ph), s: Math.sin(ph) });
  check(close(rot.g, P.g, 1e-10) && close(rot.s, P.s, 1e-10), "phasor = conj(c1)/c0 (bin-centre phase)");
  // mixture on the segment at the photon fraction
  const a = Ph.phasor(D.expBins(0.5, ax)), b = Ph.phasor(D.expBins(3, ax));
  const m = Ph.phasor(D.decayBins({ tau: [0.5, 3], f: [0.3, 0.7] }, D.gaussianIRF(0, 0, ax), ax));
  check(close(Ph.fractionOnSegment(m, a, b), 0.7, 1e-9), "lever rule");
  // P(h ⊛ I) = P(h) P(I) for circular convolution
  const h = D.gaussianIRF(0.3, 1.2, ax), I = D.expBins(2, ax);
  const lhs = Ph.phasor(D.convolveCircular(h, I)), rhs = Ph.cmul(Ph.phasor(h), Ph.phasor(I));
  // bin-centre phases: one extra half-bin rotation on the right
  const rhs2 = Ph.cmul(rhs, { g: Math.cos(-ph), s: Math.sin(-ph) });
  check(close(lhs.g, rhs2.g, 1e-10) && close(lhs.s, rhs2.s, 1e-10), "convolution multiplies phasors");
  // calibration recovers a reference exactly, and fixes another sample
  const ref = Ph.phasor(D.convolveCircular(h, D.expBins(4, ax)));
  const K = Ph.calibrationFactor(ref, 4, omega);
  const cr = Ph.calibrate(ref, K), s4 = Ph.semicircle(omega * 4);
  check(close(cr.g, s4.g, 1e-12) && close(cr.s, s4.s, 1e-12), "calibrate recovers the reference");
  const smp = Ph.calibrate(Ph.phasor(D.convolveCircular(h, D.expBins(1.5, ax))), K), s15 = Ph.semicircle(omega * 1.5);
  check(Math.hypot(smp.g - s15.g, smp.s - s15.s) < 1e-4, "calibration fixes another lifetime");
  // τ_φ < τ_m for random mixtures
  const r = rng(5);
  let ok = true;
  for (let t = 0; t < 200; t++) {
    const t1 = 0.2 + 5 * r.unif(), t2 = 0.2 + 5 * r.unif(), f = 0.05 + 0.9 * r.unif();
    if (Math.abs(t1 - t2) < 0.05) continue;
    const lt = Ph.lifetimeFromPhasor(D.decayPhasor({ tau: [t1, t2], f: [f, 1 - f] }, omega), omega);
    if (!(lt.tauPhi < lt.tauM)) ok = false;
  }
  check(ok, "τ_φ < τ_m for mixtures");
  // barycentric sums to 1 and reproduces the point
  const A = Ph.semicircle(0.2), B = Ph.semicircle(1.2), C = Ph.semicircle(2.3);
  const p = { g: 0.2 * A.g + 0.5 * B.g + 0.3 * C.g, s: 0.2 * A.s + 0.5 * B.s + 0.3 * C.s };
  const w = Ph.barycentric3(p, A, B, C);
  check(close(w[0] + w[1] + w[2], 1) && close(w[0], 0.2) && close(w[1], 0.5) && close(w[2], 0.3), "barycentric3");
  // line through a two-species cloud meets the semicircle at both lifetimes
  const gs: number[] = [], ss: number[] = [];
  for (let i = 0; i <= 20; i++) { const q = D.decayPhasor({ tau: [0.6, 3.5], f: [1 - i / 20, i / 20] }, omega); gs.push(q.g); ss.push(q.s); }
  const L = Ph.lineThroughCloud(gs, ss, omega);
  check(L.taus.length === 2 && close(L.taus[0], 0.6, 1e-6) && close(L.taus[1], 3.5, 1e-6), `line through cloud: ${L.taus}`);
}

// ---------------------------------------------------------------------------------------
current = "nlfit";
{
  const ax2: D.TimeAxis = { T: 12.5, nBins: 128 }, irf = D.gaussianIRF(0.12, 1, ax2);
  const model = (p: Float64Array) => D.expectedCounts({ tau: [p[1]], f: [1] }, irf, ax2, p[0], p[2]);
  const y = model(Float64Array.from([5000, 2.3, 1.5]));
  for (const cost of ["ls", "neyman", "poisson"] as F.Cost[]) {
    const r = F.lmFit(model, y, [3000, 1.2, 0.5], { cost, positive: [true, true, true] });
    check(close(r.p[0], 5000, 1e-5) && close(r.p[1], 2.3, 1e-5) && close(r.p[2], 1.5, 1e-4), `${cost} recovers noise-free parameters: ${Array.from(r.p)}`);
  }
  // bi-exponential, noise-free
  const model2 = (p: Float64Array) => D.expectedCounts({ tau: [p[1], p[3]], f: [p[0] / (p[0] + p[2]), p[2] / (p[0] + p[2])] }, irf, ax2, p[0] + p[2], 0);
  const y2 = model2(Float64Array.from([3000, 0.6, 7000, 3.0]));
  const r2 = F.lmFit(model2, y2, [5000, 0.9, 5000, 2.0], { cost: "poisson", positive: [true, true, true, true], maxIter: 300 });
  check(close(r2.p[1], 0.6, 1e-3) && close(r2.p[3], 3.0, 1e-3), `bi-exponential: ${Array.from(r2.p)}`);
  // Poisson ML from arrival times without IRF over a long window = mean arrival time:
  // the binned ML with fine bins approaches it.
  const axL: D.TimeAxis = { T: 400, nBins: 4000 }, r = rng(3);
  const h = new Float64Array(axL.nBins);
  let tsum = 0;
  const N = 2000;
  for (let i = 0; i < N; i++) { const t = exponential(r, 1 / 2); tsum += t; h[Math.min(axL.nBins - 1, Math.floor(t / 0.1))]++; }
  const spike = D.gaussianIRF(0, 0, axL);
  const mL = (p: Float64Array) => D.expectedCounts({ tau: [p[1]], f: [1] }, spike, axL, p[0], 0);
  const rL = F.lmFit(mL, h, [N, 1], { cost: "poisson", positive: [true, true] });
  check(Math.abs(rL.p[1] - tsum / N) < 2e-3, `ML τ ${rL.p[1]} vs mean arrival ${tsum / N}`);
  check(close(rL.p[0], N, 1e-6), "ML amplitude = photon count");
  // Monte Carlo spread of the ML τ ≈ τ/√N at N = 1e4 (within 10 %)
  const ests: number[] = [];
  const rr = rng(9), muT = D.expectedCounts({ tau: [2], f: [1] }, D.gaussianIRF(0, 0, ax2), ax2, 1e4, 0);
  const mT = (p: Float64Array) => D.expectedCounts({ tau: [p[1]], f: [1] }, D.gaussianIRF(0, 0, ax2), ax2, p[0], 0);
  for (let k = 0; k < 300; k++) ests.push(F.lmFit(mT, D.poissonHistogram(rr, muT), [1e4, 2], { cost: "poisson", positive: [true, true] }).p[1]);
  const mean = ests.reduce((a, b) => a + b, 0) / ests.length;
  const sd = Math.sqrt(ests.reduce((a, b) => a + (b - mean) ** 2, 0) / (ests.length - 1));
  check(Math.abs(sd / mean * Math.sqrt(1e4) - 1) < 0.1, `relative spread × √N = ${(sd / mean * 100).toFixed(3)}`);
  // global analysis: amplitudes exact for known lifetimes (noise-free), and = lstsq
  const B1 = D.decayBins({ tau: [1], f: [1] }, irf, ax2), B3 = D.decayBins({ tau: [3], f: [1] }, irf, ax2);
  const B = new Float64Array(ax2.nBins * 2);
  for (let j = 0; j < ax2.nBins; j++) { B[2 * j] = B1[j]; B[2 * j + 1] = B3[j]; }
  const yy = Float64Array.from({ length: ax2.nBins }, (_, j) => 300 * B1[j] + 700 * B3[j]);
  for (const cost of ["ls", "poisson"] as F.Cost[]) {
    const g = F.linearAmplitudes(B, ax2.nBins, 2, yy, cost);
    check(close(g.a[0], 300, 1e-6) && close(g.a[1], 700, 1e-6), `global amplitudes (${cost}): ${Array.from(g.a)}`);
  }
  const ls = lstsq(B, yy, ax2.nBins, 2);
  check(close(ls[0], 300, 1e-6) && close(ls[1], 700, 1e-6), "lstsq agrees");
}

current = "scene";
{
  const sc = D.flimScene("twodyes", { photons: 300 });
  const P = sc.n * sc.n;
  check(sc.stack.length === P * sc.ax.nBins, "stack size");
  const ph = Ph.phasorStack(sc.stack, sc.ax.nBins, P);
  check(close(ph.I[1000], sum(D.pixelHistogram(sc.stack, P, sc.ax.nBins, 1000))), "phasorStack intensity");
  const b = D.binStack(sc.stack, sc.n, sc.ax.nBins, 1);
  let s = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += sc.stack[5 * P + (60 + dy) * sc.n + 60 + dx];
  check(b[5 * P + 60 * sc.n + 60] === s, "binStack 3×3");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
