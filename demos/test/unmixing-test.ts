// Reference checks for lib/unmix.ts (spectral unmixing day): least squares, pseudo-inverse,
// residual, NNLS against brute force and KKT, fully constrained LS, conditioning, NMF
// monotonicity, vertex finding on pure-pixel data, colour deconvolution, Macenko, and the
// port of the earlier course's compensation demo (cytometry populations). Run:
//   node_modules/.bin/esbuild demos/test/unmixing-test.ts --bundle --platform=node --log-level=warning | node
import * as U from "../src/lib/unmix";
import { rng } from "../src/lib/random";
import { matMul } from "../src/lib/linalg";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const randMat = (r: ReturnType<typeof rng>, n: number, k: number) => Float64Array.from({ length: n * k }, () => r.unif());

// ---------------------------------------------------------------------------------------
current = "spectra";
{
  const d: U.Dye = { name: "a", peak: 520, width: 15, tail: 2 };
  check(close(U.emission(d, 520), 1), "peak = 1");
  check(close(U.emission(d, 505), U.emission(d, 550), 1e-12), "tail: 1 sd left = 1 sd right");
  // band covering everything collects (almost) all photons; bands partition additively
  check(close(U.bandFraction(d, { lo: 300, hi: 900 }), 1, 1e-6), "whole range = 1");
  const a = U.bandFraction(d, { lo: 480, hi: 530 }), b = U.bandFraction(d, { lo: 530, hi: 600 }), c = U.bandFraction(d, { lo: 480, hi: 600 });
  check(close(a + b, c, 1e-9), "bands add");
  const M = U.channelMatrix([d, { name: "b", peak: 580, width: 15, tail: 2 }], [{ lo: 500, hi: 540 }, { lo: 570, hi: 620 }]);
  const S = U.normaliseOwn(M, 2, 2);
  check(close(S[0], 1) && close(S[3], 1), "own channel = 1 after normalisation");
  check(S[2] > S[1], "green tail spills more into the red channel than red into green");
}

// ---------------------------------------------------------------------------------------
current = "least squares";
{
  const r = rng(11);
  for (let trial = 0; trial < 20; trial++) {
    const n = 3 + Math.floor(r.unif() * 10), k = 1 + Math.floor(r.unif() * Math.min(4, n));
    const M = randMat(r, n, k), Pi = U.pseudoInverse(M, n, k)!;
    const I = matMul(Pi, M, k, n, k);
    let err = 0;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) err = Math.max(err, Math.abs(I[i * k + j] - (i === j ? 1 : 0)));
    check(err < 1e-9, `M⁺M = I (n=${n}, k=${k}, err ${err})`);
    // noise-free recovery on a small stack
    const P = 7, Ab = Float32Array.from({ length: k * P }, () => r.unif());
    const Y = U.mix(M, n, k, Ab, P), Ah = U.applyToStack(Pi, k, n, Y, P);
    let e2 = 0;
    for (let i = 0; i < k * P; i++) e2 = Math.max(e2, Math.abs(Ah[i] - Ab[i]));
    check(e2 < 1e-5, `exact recovery without noise (err ${e2})`);
    // residual orthogonal to every column
    const y = Float64Array.from({ length: n }, () => r.unif());
    const a = U.lsPixel(M, n, k, y)!;
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let c = 0; c < n; c++) { let p = 0; for (let i = 0; i < k; i++) p += M[c * k + i] * a[i]; s += M[c * k + j] * (y[c] - p); }
      check(Math.abs(s) < 1e-10, `residual ⟂ column ${j} (${s})`);
    }
    // WLS with equal weights = LS
    const w = U.wlsPixel(M, n, k, y, new Float64Array(n).fill(2.5))!;
    check(a.every((v, i) => close(v, w[i], 1e-8)), "WLS with equal weights = LS");
  }
  // the square case is the inverse
  const S = [1, 0.1, 0.2, 1], Pi = U.pseudoInverse(S, 2, 2)!;
  check(close(Pi[0], 1 / 0.98) && close(Pi[1], -0.1 / 0.98) && close(Pi[2], -0.2 / 0.98), "2×2: M⁺ = M⁻¹");
}

// ---------------------------------------------------------------------------------------
current = "nnls";
{
  const r = rng(5);
  for (let trial = 0; trial < 300; trial++) {
    const n = 2 + Math.floor(r.unif() * 6), k = 1 + Math.floor(r.unif() * Math.min(4, n));
    const M = randMat(r, n, k), y = Float64Array.from({ length: n }, () => r.unif() * 2 - 0.6);
    const a = U.nnls(M, n, k, y), b = U.nnlsBrute(M, n, k, y);
    check(a.every((v, i) => Math.abs(v - b[i]) < 1e-8), `matches brute force (n=${n}, k=${k}): ${Array.from(a)} vs ${Array.from(b)}`);
    // KKT: a ≥ 0, gradient w = Mᵀ(y − Ma) ≤ 0, w_i = 0 where a_i > 0
    const w = U.mty(M, n, k, y), G = U.gram(M, n, k);
    for (let i = 0; i < k; i++) { let s = w[i]; for (let j = 0; j < k; j++) s -= G[i * k + j] * a[j]; w[i] = s; }
    check(a.every(v => v >= 0), "a ≥ 0");
    check(Array.from(w).every((v, i) => v <= 1e-9 && (a[i] === 0 || Math.abs(v) < 1e-9)), `KKT (w = ${Array.from(w)})`);
  }
  // fully constrained: sums to one, non-negative; inside the simplex it equals LS
  for (let trial = 0; trial < 100; trial++) {
    const n = 3 + Math.floor(r.unif() * 5), k = 2 + Math.floor(r.unif() * Math.min(3, n - 1));
    const M = randMat(r, n, k), y = Float64Array.from({ length: n }, () => r.unif());
    const a = U.fcls(M, n, k, y);
    check(Math.abs(a.reduce((s, v) => s + v, 0) - 1) < 1e-5 && a.every(v => v >= 0), `FCLS: Σa = 1, a ≥ 0 (${Array.from(a)})`);
    const t = Float64Array.from({ length: k }, () => r.unif()), ts = t.reduce((s, v) => s + v, 0);
    for (let j = 0; j < k; j++) t[j] /= ts;
    const yt = Float64Array.from({ length: n }, (_, c) => { let s = 0; for (let j = 0; j < k; j++) s += M[c * k + j] * t[j]; return s; });
    const af = U.fcls(M, n, k, yt), as = U.stoPixel(M, n, k, yt)!;
    check(af.every((v, j) => Math.abs(v - t[j]) < 1e-5) && as.every((v, j) => Math.abs(v - t[j]) < 1e-9), "noise-free mixture inside the simplex is recovered");
  }
}

// ---------------------------------------------------------------------------------------
current = "conditioning";
{
  // orthonormal columns: κ = 1, noise gain 1
  const Q = [1 / Math.SQRT2, 1 / Math.SQRT2, 1 / Math.SQRT2, -1 / Math.SQRT2, 0, 0];
  check(close(U.condition(Q, 3, 2), 1, 1e-9), "κ(orthonormal) = 1");
  check(Array.from(U.noiseGain(Q, 3, 2)).every(v => close(v, 1, 1e-9)), "noise gain 1");
  // the old exercise's similar dyes [[1, .9], [.9, 1]]: σ = 1.9, 0.1, κ = 19
  check(close(U.condition([1, 0.9, 0.9, 1], 2, 2), 19, 1e-9), "κ = 19 for similar dyes");
  // Monte Carlo: Cov(â) = σ²(MᵀM)⁻¹
  const r = rng(3), M = [1, 0.9, 0.8, 1, 0.3, 0.4], n = 3, k = 2, sd = 0.1, Pi = U.pseudoInverse(M, n, k)!;
  const g = U.noiseGain(M, n, k);
  let s0 = 0, s1 = 0;
  const T = 40000;
  for (let t = 0; t < T; t++) {
    const e = [sd * r.norm(), sd * r.norm(), sd * r.norm()];
    const a0 = Pi[0] * e[0] + Pi[1] * e[1] + Pi[2] * e[2], a1 = Pi[3] * e[0] + Pi[4] * e[1] + Pi[5] * e[2];
    s0 += a0 * a0; s1 += a1 * a1;
  }
  check(Math.abs(Math.sqrt(s0 / T) / (sd * g[0]) - 1) < 0.02 && Math.abs(Math.sqrt(s1 / T) / (sd * g[1]) - 1) < 0.02, "sd(â) = σ sqrt(diag (MᵀM)⁻¹)");
}

// ---------------------------------------------------------------------------------------
current = "endmembers";
{
  // pure-pixel data: three spectra over 8 channels, abundances on the simplex incl. pure pixels
  const r = rng(9), n = 8, k = 3, P = 400;
  const E = Float64Array.from({ length: n * k }, (_, i) => 0.05 + r.unif());
  for (let j = 0; j < k; j++) { let s = 0; for (let c = 0; c < n; c++) s += E[c * k + j]; for (let c = 0; c < n; c++) E[c * k + j] /= s; }
  const Ab = new Float32Array(k * P);
  for (let p = 0; p < P; p++) {
    const t = [r.unif(), r.unif(), r.unif()].map(v => -Math.log(v)), s = t[0] + t[1] + t[2];
    const sc = 0.5 + r.unif();
    for (let j = 0; j < k; j++) Ab[j * P + p] = sc * t[j] / s;
  }
  for (let j = 0; j < k; j++) { for (let i = 0; i < k; i++) Ab[i * P + j] = i === j ? 1.3 : 0; }   // pixels 0..2 are pure
  const Y = U.mix(E, n, k, Ab, P);
  const sv = U.stackSingularValues(Y, n, P);
  check(sv[2] > 1e-3 * sv[0] && sv[3] < 1e-6 * sv[0], `three singular values stand out (${Array.from(sv.slice(0, 5)).map(v => v.toExponential(2))})`);
  const v = U.vertexFind(Y, n, P, k);
  check(v.length === 3 && [0, 1, 2].every(j => v.includes(j)), `vertex finding returns the pure pixels (${v})`);
  // NMF error never increases
  const s = U.nmfInit(Y, n, P, k, 4);
  let prev = U.nmfError(Y, s), mono = true;
  for (let it = 0; it < 60; it++) { U.nmfStep(Y, s); const e = U.nmfError(Y, s); if (e > prev * (1 + 1e-12) + 1e-15) mono = false; prev = e; }
  check(mono, "NMF error non-increasing");
  const before = U.nmfError(Y, s); U.nmfNormalise(s);
  check(close(U.nmfError(Y, s), before, 1e-8), "normalising keeps WH");
  // fixed columns stay fixed
  const s2 = U.nmfInit(Y, n, P, k, 4, [U.normaliseSum(E, n, k).filter((_, i) => i % k === 0), null, null]);
  const w0 = Array.from({ length: n }, (_, c) => s2.W[c * k]);
  for (let it = 0; it < 10; it++) U.nmfStep(Y, s2);
  check(w0.every((x, c) => x === s2.W[c * k]), "known spectrum column stays fixed");
  // N-FINDR by hand
  const pts: [number, number][] = [[0, 0], [4, 0], [1, 1], [0, 3], [2, 1], [1, 2]];
  const lt = U.largestTriangle(pts);
  check(lt.area === 6 && lt.idx.join() === "0,1,3", "largest triangle");
  check(close(U.simplexVolume([[0, 0], [4, 0], [0, 3]]), 12), "simplex volume = |det| = 2 · area");
}

// ---------------------------------------------------------------------------------------
current = "stains";
{
  const sc = U.stainScene(64, 7, "he"), S = U.stainMatrix(U.STAINS.H, U.STAINS.E, U.STAINS.DAB), I0: [number, number, number] = [240, 235, 245];
  const img = U.renderBrightfield(sc, S, I0, { exact: true });
  const d = U.colourDeconvolve(img.R, img.G, img.B, I0, S, 1e-9)!;
  let err = 0;
  for (let k = 0; k < 3; k++) for (let p = 0; p < 64 * 64; p++) err = Math.max(err, Math.abs(d.c[k][p] - sc.conc[k][p]));
  check(err < 1e-4, `noise-free deconvolution with the true M recovers the maps (err ${err})`);
  // OD is additive: the OD of a two-stain pixel is the sum of the single-stain ODs
  const od = (c: number[]) => [0, 1, 2].map(i => -Math.log10(I0[i] * 10 ** -(S[i * 3] * c[0] + S[i * 3 + 1] * c[1]) / I0[i]));
  const a = od([0.5, 0]), b = od([0, 0.3]), ab = od([0.5, 0.3]);
  check(ab.every((v, i) => close(v, a[i] + b[i], 1e-12)), "OD adds");
  // residual vector ⟂ both stains
  const S2 = U.stainMatrix(U.STAINS.H, U.STAINS.E);
  check(Math.abs(S2[2] * S2[0] + S2[5] * S2[3] + S2[8] * S2[6]) < 1e-12 && Math.abs(S2[2] * S2[1] + S2[5] * S2[4] + S2[8] * S2[7]) < 1e-12, "cross-product column ⟂ stains");
  // Macenko on a noise-free H&E image finds vectors close to the true ones
  const big = U.stainScene(160, 7, "he"), im2 = U.renderBrightfield(big, S, I0, { exact: true });
  const d2 = U.colourDeconvolve(im2.R, im2.G, im2.B, I0, S, 1e-9)!;
  const m = U.macenko(d2.od, 0.15, 1)!;
  const ang = (u: number[], v: number[]) => Math.acos(Math.min(1, u[0] * v[0] + u[1] * v[1] + u[2] * v[2])) * 180 / Math.PI;
  const h = U.unit3(U.STAINS.H), e = U.unit3(U.STAINS.E);
  check(ang(m.v1, h) < 5 && ang(m.v2, e) < 5, `Macenko within 5° (${ang(m.v1, h).toFixed(2)}°, ${ang(m.v2, e).toFixed(2)}°)`);
}

// ---------------------------------------------------------------------------------------
current = "bleedthrough port";
{
  // numbers from the earlier course's compensation demo (simulate(spill(s21, s12), seed))
  const ref = [
    { s21: 0.25, s12: 0.02, seed: 1, mx: [139.96332169494588, 125.08052606270662, 148.30482847514494], my: [174.55603878994634, 118.9649198869994, 129.64395194696536], last: [3786.0804729379647, 3126.775342033872] },
    { s21: 0.32, s12: 0.08, seed: 2, mx: [44.74369921924382, 173.79832409678738, 153.1027262878275], my: [82.3682349403905, 189.62685325148783, 246.70555139271798], last: [2523.09795502103, 3610.301697945792] },
  ];
  for (const r of ref) {
    const c = U.cytometryCells(r.s21, r.s12, r.seed);
    check(r.mx.every((v, i) => close(c.mx[i], v, 1e-12)) && r.my.every((v, i) => close(c.my[i], v, 1e-12)) && close(c.mx[1599], r.last[0], 1e-12) && close(c.my[1599], r.last[1], 1e-12), `same cells as the old demo (seed ${r.seed})`);
  }
  // compensating with the true S: FITC+ population flat, median PE ≈ 105.29 (old demo)
  const c = U.cytometryCells(0.25, 0.02, 1), k = 1 - 0.25 * 0.02;
  const x: number[] = [], y: number[] = [];
  c.group.forEach((g, i) => { if (g === 1) { x.push((c.mx[i] - 0.02 * c.my[i]) / k); y.push((-0.25 * c.mx[i] + c.my[i]) / k); } });
  check(close(U.lsLine(x, y).b, -0.0036966070341697015, 1e-9), "FITC+ slope after compensation as in the old demo");
  check(close(U.median(y), 105.2885595098833, 1e-9), "FITC+ median PE as in the old demo");
  check(close(U.slopeThroughOrigin([1, 2, 3], [0.25, 0.5, 0.75]), 0.25), "slope through the origin");
  check(close(U.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1) && close(U.pearson([1, 2, 3, 4], [1, -1, -1, 1]), 0), "Pearson");
}

// ---------------------------------------------------------------------------------------
current = "scenes";
{
  const a = U.cellScene(128, 3), b = U.cellScene(128, 3);
  check(a.nucleus.every((v, i) => v === b.nucleus[i]) && a.mito.every((v, i) => v === b.mito[i]), "cell scene is seeded");
  const cnt = (m: Float32Array) => m.reduce((s, v) => s + (v > 0.5 ? 1 : 0), 0);
  check(cnt(a.nucleus) > 500 && cnt(a.mito) > 100 && cnt(a.membrane) > 100 && cnt(a.vesicles) > 5, `every component present (${cnt(a.nucleus)}, ${cnt(a.mito)}, ${cnt(a.membrane)}, ${cnt(a.vesicles)})`);
  const h = U.stainScene(160, 7, "hdab");
  check(h.positive.length > 3 && h.positive.length < h.nucleusCount, `H-DAB has positive and negative nuclei (${h.positive.length}/${h.nucleusCount})`);
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
