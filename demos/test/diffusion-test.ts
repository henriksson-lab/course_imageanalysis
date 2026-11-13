// Tests for the diffusion-models day (score-matching/, diffusion-sampling/, guidance/,
// diffusion-inverse/): lib/diffusion.ts, lib/genmetrics.ts and the models of scorematch1d,
// noisespectrum, samplerpaths, genmetrics, guidance1d, latentdiff, posterior1d, virtualstain.
// Brute-force or reference checks, and every by-hand number of the exercises.
// Run: node_modules/.bin/esbuild demos/test/diffusion-test.ts --bundle --platform=node --log-level=warning | node
import { makeDiag, noisedMixture, scoreDiag, sampleDiag, posteriorDiag, alphaBarAt } from "../src/lib/mixture";
import * as DF from "../src/lib/diffusion";
import * as GM from "../src/lib/genmetrics";
import { rng } from "../src/lib/ui";
import { Adam } from "../src/lib/nn";
import { normCdf } from "../src/lib/stats";
import * as SM from "../src/scorematch1d/model";
import { DSM_WEIGHTS } from "../src/scorematch1d/dsm_weights";
import * as NS from "../src/noisespectrum/model";
import * as SP from "../src/samplerpaths/model";
import * as GN from "../src/genmetrics/model";
import * as G1 from "../src/guidance1d/model";
import * as LD from "../src/latentdiff/model";
import { CHECK_MASKS, CHECK_MU, CHECK_SD } from "../src/latentdiff/latent_check";
import { shapeVae } from "../src/vaeimages/imagevae";
import * as P1 from "../src/posterior1d/model";
import * as VS from "../src/virtualstain/model";

let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const meanVar = (x: ArrayLike<number>) => { let m = 0; for (let i = 0; i < x.length; i++) m += x[i] / x.length; let v = 0; for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2 / x.length; return { m, v }; };

// ---- noise levels and schedules ------------------------------------------------------------
{
  let ok = true;
  for (const k of ["cosine", "linear"] as const) for (const t of [0.05, 0.3, 0.5, 0.8, 0.97]) if (!close(DF.tauFromAbar(alphaBarAt(t, k), k), t, 1e-9)) ok = false;
  check("tauFromAbar inverts alphaBarAt", ok);
  check("SNR, σ for ā = 0.8, 0.5, 0.2", close(DF.sigmaFromAbar(0.8), 0.5) && close(DF.sigmaFromAbar(0.5), 1) && close(DF.sigmaFromAbar(0.2), 2) && close(DF.abarFromLogSnr(DF.logSnr(0.3)), 0.3));
  const g = DF.sigmaGrid(10, "data");
  let mono = true; for (let i = 0; i < 10; i++) if (!(g[i] > g[i + 1])) mono = false;
  check("sigma grids decrease to 0", mono && g[10] === 0 && close(DF.sigmaGrid(5, "time")[0], DF.sigmaFromAbar(alphaBarAt(1))));
}

// ---- mixture denoiser = Tweedie on the noised mixture (lib/mixture.ts) ----------------------
{
  const m = makeDiag([0.3, 0.7], [[-1, 0.5], [1.2, -0.4]], [[0.2, 0.1], [0.05, 0.3]]), D = DF.mixtureDenoiser(m), r = rng(3);
  let ok = true;
  for (let t = 0; t < 50; t++) {
    const s = 0.05 + 3 * r.unif(), x = new Float64Array([3 * r.norm(), 3 * r.norm()]), out = new Float64Array(2);
    D(x, s, out);
    const ab = DF.abarFromSigma(s), xv = Float64Array.from(x, v => v * Math.sqrt(ab)), sc = scoreDiag(noisedMixture(m, ab), xv);
    for (let i = 0; i < 2; i++) { const tw = (xv[i] + (1 - ab) * sc[i]) / Math.sqrt(ab); if (!close(out[i], tw, 1e-8)) ok = false; }
  }
  check("mixtureDenoiser equals Tweedie with the VP mixture score", ok);
}

// ---- samplers with the exact score reproduce the data (1D and 2D) --------------------------
{
  const m1 = SP.data("three"), t1 = SP.mixMeanVar(m1);
  for (const eta of [0, 1]) {
    const r = SP.run(m1, { steps: 1000, eta, spacing: "time", twoStage: false, langevin: 0, seed: 2, n: 3000, nPaths: 0 });
    const mv = meanVar(r.ends);
    check(`1D sampler η=${eta}, 1000 steps: mean and variance`, Math.abs(mv.m - t1.mean) < 0.06 && Math.abs(mv.v - t1.variance) < 0.08, JSON.stringify(mv));
  }
  const m2 = makeDiag([0.5, 0.5], [[-1, 0.5], [1, -0.5]], 0.05), D = DF.mixtureDenoiser(m2), sg = DF.sigmaGrid(1000), rr = rng(4), rn = rng(5);
  const X: number[] = [], Y: number[] = [];
  for (let p = 0; p < 1500; p++) { const x = new Float64Array([101 * rr.norm(), 101 * rr.norm()]); DF.sampleRun(x, D, { sigmas: sg, eta: 1 }, rn); X.push(x[0]); Y.push(x[1]); }
  const mx = meanVar(X), my = meanVar(Y);
  check("2D ancestral sampler: mean and variance", Math.abs(mx.m) < 0.08 && Math.abs(mx.v - 1.05) < 0.08 && Math.abs(my.v - 0.30) < 0.05, JSON.stringify({ mx, my }));
  // inversion round trip
  let ok = true;
  for (const x0 of [-1.6, -1.4, 1.45, 1.55]) { const iv = SP.invert(SP.data("two"), x0, 200, "time"); if (Math.abs(iv.back - x0) > 0.02) ok = false; }
  check("ODE inversion round trip (200 steps)", ok);
  // two-stage step beats Euler at equal cost on Gaussian data
  const g = SP.data("gauss"), tr = SP.truth(g, 2000);
  const e = SP.summary(g, SP.run(g, { steps: 10, eta: 0, spacing: "time", twoStage: false, langevin: 0, seed: 1, n: 2000, nPaths: 0 }).ends, tr);
  const h = SP.summary(g, SP.run(g, { steps: 5, eta: 0, spacing: "time", twoStage: true, langevin: 0, seed: 1, n: 2000, nPaths: 0 }).ends, tr);
  check("two-stage (9 evaluations) closer than Euler (10)", Math.abs(h.sd - 1) < Math.abs(e.sd - 1), `${h.sd} ${e.sd}`);
  // stochastic forgets a bad start
  const m3 = SP.data("three"), tr3 = SP.truth(m3, 2000);
  const bad0 = SP.summary(m3, SP.run(m3, { steps: 20, eta: 0, spacing: "time", twoStage: false, langevin: 0, seed: 1, n: 2000, nPaths: 0, startScale: 0.2 }).ends, tr3);
  const bad1 = SP.summary(m3, SP.run(m3, { steps: 20, eta: 1, spacing: "time", twoStage: false, langevin: 0, seed: 1, n: 2000, nPaths: 0, startScale: 0.2 }).ends, tr3);
  check("bad start: Fréchet 0.49 deterministic, 0.005 ancestral", close(bad0.frechet, 0.4877, 2e-3) && bad1.frechet < 0.01, `${bad0.frechet} ${bad1.frechet}`);
}

// ---- by-hand numbers: sampling steps --------------------------------------------------------
{
  const a = DF.ddimStepVP(1, 0.36, 0.64, 0.5), b = DF.ddpmStepVP(1, 0.36, 0.64, 0.5);
  check("DDIM step: x̂0 = 1.0, x = 1.1", close(a.x0, 1) && close(a.x, 1.1));
  check("DDPM step: mean 0.969, variance 0.246", close(b.mean, 0.96875) && close(b.variance, 0.24609375));
  const D: DF.Denoiser = (x, s, out) => { out[0] = x[0] / (1 + s * s); };
  const spread = (grid: number[]) => { const x = new Float64Array([Math.sqrt(10)]), t = new Float64Array(1); for (let i = 0; i + 1 < grid.length; i++) DF.eulerOdeStep(x, grid[i], grid[i + 1], D, t); return x[0]; };
  check("Euler spreads 0.316 / 0.535 / 0.632 / 0.890", close(spread([3, 0]), 0.3162, 1e-3) && close(spread([3, 1.5, 0]), 0.5351, 1e-3) && close(spread([3, 1, 0]), 0.6325, 1e-3) && close(spread(Array.from({ length: 11 }, (_, i) => 3 - 0.3 * i)), 0.890, 2e-3));
  // the VE ddimStep with η = 0 is the Euler step; with η = 1 the DDPM step scaled by √ā'
  const x = new Float64Array([1 / 0.6]), t = new Float64Array(1), Dc: DF.Denoiser = (_x, _s, out) => { out[0] = 1.0; };
  DF.ddimStep(x, Math.sqrt(0.64 / 0.36), Math.sqrt(0.36 / 0.64), 0, Dc, t, rng(1));
  check("VE deterministic step = VP DDIM step / √ā'", close(x[0], 1.1 / 0.8, 1e-12));
}

// ---- genmetrics: Fréchet and precision/recall -------------------------------------------------
{
  check("1D Fréchet 0.468 / 0.216 / 0", close(GM.frechetGaussian([0], [0.1], [0], [1], 1), (1 - Math.sqrt(0.1)) ** 2) && close(GM.frechetGaussian([0], [0.535 ** 2], [0], [1], 1), 0.216, 1e-2) && close(GM.frechetGaussian([0], [1], [0], [1], 1), 0));
  // 2D closed form against the general (eigen) route, and against a brute-force square root (Newton)
  const r = rng(8);
  let ok = true;
  for (let t = 0; t < 20; t++) {
    const A = [r.norm(), r.norm(), r.norm(), r.norm()], B = [r.norm(), r.norm(), r.norm(), r.norm()];
    const S1 = [A[0] * A[0] + A[1] * A[1] + 0.1, A[0] * A[2] + A[1] * A[3], A[0] * A[2] + A[1] * A[3], A[2] * A[2] + A[3] * A[3] + 0.1];
    const S2 = [B[0] * B[0] + B[1] * B[1] + 0.1, B[0] * B[2] + B[1] * B[3], B[0] * B[2] + B[1] * B[3], B[2] * B[2] + B[3] * B[3] + 0.1];
    const f2 = GM.frechetGaussian([0, 1], S1, [1, 0], S2, 2);
    const r1 = GM.sqrtmSym(S1, 2);
    const Mx = [0, 0, 0, 0]; // r1 S2 r1
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) for (let l = 0; l < 2; l++) Mx[i * 2 + j] += r1[i * 2 + k] * S2[k * 2 + l] * r1[l * 2 + j];
    // Denman–Beavers iteration for the square root of M: Y ← (Y + Z⁻¹)/2, Z ← (Z + Y⁻¹)/2
    const inv2 = (Q: number[]) => { const d = Q[0] * Q[3] - Q[1] * Q[2]; return [Q[3] / d, -Q[1] / d, -Q[2] / d, Q[0] / d]; };
    let Y = [...Mx], Zm = [1, 0, 0, 1];
    for (let it = 0; it < 50; it++) { const Yi = inv2(Y), Zi = inv2(Zm); Y = Y.map((v, i) => (v + Zi[i]) / 2); Zm = Zm.map((v, i) => (v + Yi[i]) / 2); }
    const brute = 2 + S1[0] + S1[3] + S2[0] + S2[3] - 2 * (Y[0] + Y[3]);
    if (!close(f2, brute, 1e-7)) ok = false;
    // 3D route with a decoupled third axis equals 2D + 1D
    const S1b = [S1[0], S1[1], 0, S1[2], S1[3], 0, 0, 0, 2], S2b = [S2[0], S2[1], 0, S2[2], S2[3], 0, 0, 0, 0.5];
    if (!close(GM.frechetGaussian([0, 1, 0], S1b, [1, 0, 0], S2b, 3), f2 + (Math.sqrt(2) - Math.sqrt(0.5)) ** 2, 1e-7)) ok = false;
  }
  check("Fréchet distance: closed form = Denman–Beavers square root = eigen route", ok);
  // ±2 data, generator N(0, 4.25): Fréchet 0, 37 % within ±1
  const dm = 4 + 0.25;
  check("between the modes: Fréchet 0, 37 % in [−1, 1]", close(GM.frechetGaussian([0], [dm], [0], [4.25], 1), 0) && close(2 * normCdf(1 / Math.sqrt(4.25)) - 1, 0.372, 2e-3));
  // k-NN precision/recall on hand-made sets: real = a 5 × 5 grid; fake = 10 grid points + 10 far points
  const real: number[] = [], fake: number[] = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) real.push(i, j);
  for (let k = 0; k < 10; k++) fake.push(k % 5, Math.floor(k / 5));
  for (let k = 0; k < 10; k++) fake.push(50 + k, 50);
  const pr = GM.precisionRecall(real, fake, 2, 1);
  // precision: 10 of 20 fake points inside unit balls of the grid; recall: grid points within the fake 1-NN radii (1) of the 10 copies: rows 0, 1 and the row-2 neighbours
  let rec = 0; for (let p = 0; p < 25; p++) { const x = real[2 * p], y = real[2 * p + 1]; let inside = false; for (let k = 0; k < 10; k++) if (Math.hypot(x - fake[2 * k], y - fake[2 * k + 1]) <= 1) inside = true; if (inside) rec++; }
  check("k-NN precision 0.5 and recall by brute force", close(pr.precision, 0.5) && close(pr.recall, rec / 25), `${pr.precision} ${pr.recall} ${rec / 25}`);
  check("nearest distances and Wasserstein", close(GM.nearestDistances([0, 0, 3, 4], [0, 1, 3, 0], 2)[1], 4) && close(GM.nearestDistances([0, 0, 3, 4], [0, 1, 3, 0], 2)[0], 1) && close(GM.wasserstein1d([0, 1, 2], [1, 2, 3]), 1) && close(GM.intervalCoverage([1, 2, 3], [0, 2.5, 2], [2, 3, 4]), 2 / 3));
}

// ---- genmetrics demo: the quoted numbers ----------------------------------------------------
{
  const s = GN.realSets("two", null), m = GN.metrics(s.real, GN.generate("two", "between", 100, 0, s.train), s.train);
  check("genmetrics between: precision 0.483, Fréchet 0.0136", close(m.precision, 0.483, 1e-9) && close(m.frechet, 0.0136, 2e-3), JSON.stringify({ p: m.precision, f: m.frechet }));
}

// ---- guidance: by-hand numbers ----------------------------------------------------------------
{
  check("CFG combination 1.4", close(DF.cfgCombine([0.5], [0.2], 3)[0], 1.4));
  const c = G1.classes("two");
  const ratio = (w: number) => { const g = (x: number) => G1.classDensity(c, 0, x) * G1.classProb(c, 0, x) ** w; return g(-2) / g(-1); };
  check("sharpened density ratios 0.61 / 0.94 / 1.45", close(ratio(0), 0.6065, 1e-3) && close(ratio(4), 0.9372, 1e-3) && close(ratio(8), 1.4481, 1e-3));
  // guided Gaussian m = 1, s² = 0.5, w = 1: mean 4/3, variance 1/3 (numerically, on a grid)
  const xs = Array.from({ length: 4001 }, (_, i) => -5 + 10 * i / 4000), h = xs[1] - xs[0];
  const dens = xs.map(x => Math.exp(-(2) * (x - 1) ** 2 / (2 * 0.5) + x * x / 2));
  let z = 0, mu = 0, e2 = 0; xs.forEach((x, i) => { z += dens[i] * h; mu += x * dens[i] * h; e2 += x * x * dens[i] * h; });
  mu /= z; e2 /= z;
  check("guided Gaussian: mean 4/3, variance 1/3", close(mu, 4 / 3, 1e-6) && close(e2 - mu * mu, 1 / 3, 1e-5));
  const X = G1.guidedSamples(c, 0, 0, 100, 1);
  const mv = meanVar(X);
  check("guidance1d w = 0 samples the class", Math.abs(mv.m + 1) < 0.08 && Math.abs(mv.v - 1) < 0.1);
}

// ---- scorematch1d: the learned network, reproducible ------------------------------------------
{
  const d = SM.data1d("two"), net = SM.newNet(), opt = new Adam(net, SM.LR);
  SM.trainDsm(net, opt, d, 0, 200);
  const snap = SM.unpackNet(DSM_WEIGHTS.two[0]);
  let ok = true;
  net.W.forEach((w, l) => { for (let i = 0; i < w.length; i++) if (Math.abs(w[i] - snap.W[l][i]) > 1e-5 * Math.max(1, Math.abs(w[i]))) ok = false; });
  check("DSM snapshot after 200 steps reproduced in node", ok);
  const ex = SM.exactDenoiser(d), P = SM.pairs(d, 0.6);
  let e = 0; for (let i = 0; i < P.x.length; i++) e += (ex(P.x[i], 0.6) - P.x0[i]) ** 2;
  check("exact denoising error 0.132 at σ = 0.6", close(e / P.x.length, 0.132, 3e-3));
  // the binned average converges to the exact denoiser with many pairs
  const Q = SM.pairs(d, 0.6, 200000, 3), bn = SM.binned(Q.x, Q.x0, -3, 3, 30);
  let worst = 0; for (let b = 0; b < 30; b++) if (bn.n[b] > 2000) { const xc = bn.c[b], dl = 0.1, avg = (ex(xc - dl / 2, 0.6) + 4 * ex(xc, 0.6) + ex(xc + dl / 2, 0.6)) / 6; worst = Math.max(worst, Math.abs(bn.m[b] - avg)); }
  check("binned average → exact denoiser (200 000 pairs)", worst < 0.03, String(worst));
}

// ---- noisespectrum: Wiener never worse than the noisy image -------------------------------------
{
  const r = rng(5), x = new Float64Array(NS.N * NS.N);
  for (let y = 0; y < NS.N; y++) for (let k = 0; k < NS.N; k++) x[y * NS.N + k] = Math.sin(k / 9) + Math.cos(y / 13) + 0.3 * r.norm();
  const xs = NS.standardise(x), m = NS.powerModel(xs), nz = NS.noiseField();
  let ok = true; for (const s of [0.1, 1, 10]) { const a = NS.analyse(xs, m, s, nz); if (!(a.mseWiener < a.mseNoisy)) ok = false; }
  check("Wiener MSE below the noisy MSE", ok);
}

// ---- posterior1d: exact posterior against importance sampling; coverage ------------------------
{
  const tr = P1.trainingSet("matched"), m: P1.Meas = { id: "lowpass", blur: 2, cutoff: 6, gap: [26, 38], noise: 0.05 };
  const x = P1.testSignal(4), y = P1.measure(x, m, 3), post = P1.posterior(tr, y, m), mu = P1.postMeanSignal(post);
  // importance sampling from the prior: weights = likelihood of y
  const { a } = P1.measDiag(m);
  const draw = (r: ReturnType<typeof rng>) => { const base = tr[Math.floor(r.unif() * tr.length)], xs = Float64Array.from(base, v => v + P1.PRIOR_SD * r.norm()), c = P1.toBasis(xs, false); let l = 0; for (let i = 0; i < P1.N; i++) if (a[i] !== 0) l -= (y[i] - a[i] * c[i]) ** 2 / (2 * m.noise ** 2); return { xs, l }; };
  const NIS = 200000;
  let lm = -Infinity; { const r1 = rng(11); for (let k = 0; k < NIS; k++) lm = Math.max(lm, draw(r1).l); }
  const acc = new Float64Array(P1.N); let wsum = 0, w2 = 0;
  { const r2 = rng(11); for (let k = 0; k < NIS; k++) { const d = draw(r2), w = Math.exp(d.l - lm); wsum += w; w2 += w * w; for (let i = 0; i < P1.N; i++) acc[i] += w * d.xs[i]; } }
  const r = rng(12);
  let err = 0; for (let i = 0; i < P1.N; i++) err = Math.max(err, Math.abs(acc[i] / wsum - mu[i]));
  const ess = wsum * wsum / w2;
  check("posterior mean: exact = importance sampling", err < 0.04 && ess > 100, `max error ${err}, effective sample size ${ess}`);
  // the fast isotropic denoiser equals the general mixture denoiser
  const coefs = tr.slice(0, 50).map(v => P1.toBasis(v, false)), Dfast = P1.isoDenoiser(coefs, P1.PRIOR_SD ** 2);
  const Dgen = DF.mixtureDenoiser(makeDiag(new Array(50).fill(1), coefs, P1.PRIOR_SD ** 2));
  let ok = true;
  for (const s of [0.01, 0.3, 3]) { const xv = Float64Array.from({ length: P1.N }, () => r.norm()), o1 = new Float64Array(P1.N), o2 = new Float64Array(P1.N); Dfast(xv, s, o1); Dgen(xv, s, o2); for (let i = 0; i < P1.N; i++) if (!close(o1[i], o2[i], 1e-8)) ok = false; }
  check("isoDenoiser = mixtureDenoiser", ok);
  const cov = P1.coverage("matched", m, 200), covW = P1.coverage("wide", m, 200);
  check("coverage ≈ 0.9 with the matched prior (0.884), lower with the wrong one (0.745)", cov > 0.85 && cov < 0.93 && close(cov, 0.884, 2e-3) && close(covW, 0.745, 2e-3), `${cov} ${covW}`);
  // null space: adding a null-space vector leaves the residual unchanged
  const zn = P1.nullPart(P1.testSignal(10), m), x2 = x.map((v, i) => v + zn[i]);
  check("residual blind to the null space", close(P1.residualRatio(x, y, m), P1.residualRatio(x2, y, m), 1e-9));
  // two-pixel camera of B1: weights unchanged, mean (0.5, 0.5)
  const two = makeDiag([0.4, 0.4, 0.2], [[1, 0], [0, 1], [0.5, 0.5]], 1e-6);
  // in the basis (mean, difference)/√2 the camera measures only the first coordinate
  const rot = (v: number[]) => [(v[0] + v[1]) / Math.SQRT2, (v[0] - v[1]) / Math.SQRT2];
  const twoR = makeDiag([0.4, 0.4, 0.2], [[1, 0], [0, 1], [0.5, 0.5]].map(rot), 1e-6);
  const pp = posteriorDiag(twoR, [0.5 * Math.SQRT2, 0], [1, 0], 0.01);
  check("two-pixel camera: posterior weights 0.4 / 0.4 / 0.2", close(pp.w[0], 0.4, 1e-6) && close(pp.w[1], 0.4, 1e-6) && close(pp.w[2], 0.2, 1e-6));
  void two;
}

// ---- virtualstain ------------------------------------------------------------------------------
{
  const tr = VS.trainingSet("nucleus"), c = VS.truthCell("nucleus", 3), y = VS.measureInput(c, 0.1), post = VS.posterior(tr, y, 0.1);
  let s = 0; for (const w of post.w) s += w;
  const rt = VS.readouts(c.target), rm = VS.readouts(VS.meanTarget(post));
  check("virtualstain: weights sum to 1; nucleus readout close to truth", close(s, 1, 1e-9) && Math.abs(rm.area - rt.area) / rt.area < 0.1 && close(rm.intensity, rt.intensity, 0.02), JSON.stringify({ rt, rm }));
  const ct = VS.truthCell("treated", 3), pt = VS.posterior(tr, VS.measureInput(ct, 0.1), 0.1);
  check("virtualstain: treatment effect vanishes (0.80 → 0.40)", close(VS.readouts(ct.target).intensity, 0.8, 1e-9) && close(VS.readouts(VS.meanTarget(pt)).intensity, 0.4, 0.02));
}

// ---- latentdiff: stored codes against the TS encoder; decoder against vaeimages ------------------
{
  let ok = true;
  CHECK_MASKS.forEach((b64, k) => {
    const bin = atob(b64), img = new Float32Array(784);
    for (let i = 0; i < 784; i++) img[i] = bin.charCodeAt(i) / 255;
    const e = shapeVae.encode(img);
    for (let i = 0; i < 2; i++) if (Math.abs(e.mu[i] - CHECK_MU[2 * k + i]) > 1e-4 || Math.abs(e.sd[i] - CHECK_SD[2 * k + i]) > 1e-4) ok = false;
  });
  check("latent codes: python encoder = TS encoder (10 masks)", ok);
  let e = 0; const a = LD.decode([0.3, -0.5]), b = shapeVae.decode([0.3, -0.5]); for (let i = 0; i < 784; i++) e = Math.max(e, Math.abs(a[i] - b[i]));
  check("latentdiff decoder = vaeimages decoder", e < 1e-6);
  const r = LD.sampleLatents({ source: "diffusion", kind: null, w: 0, steps: 40, eta: 0, sdScale: 1, n: 200, nPaths: 0, seed: 5 });
  const sh = LD.kindShares(LD.nearestCode(r.Z).kind);
  check("latent diffusion: kinds near the training shares", sh.every(v => v > 0.12 && v < 0.4), sh.join(" "));
}

// ---- sampleDiag sanity for the 2D data of genmetrics (shared with diffusion2d) --------------------
{
  const s = sampleDiag(makeDiag([1], [[0, 0]], 1), 5000, rng(1));
  check("sampleDiag unit variance", Math.abs(meanVar(Array.from({ length: 5000 }, (_, i) => s.X[2 * i])).v - 1) < 0.06);
}

console.log(`diffusion: ${passes} passed, ${fails} failed`);
if (fails) (globalThis as unknown as { process: { exit(c: number): void } }).process.exit(1);
