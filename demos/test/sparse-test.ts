// Tests for demos/src/lib/sparse.ts and demos/src/lib/patches.ts (sparse representations day):
// the by-hand numbers of the units (ISTA, ridge, OMP, the 2×4 compressed-sensing example, the
// fold-over, the K-SVD update), operator adjoints, solver properties and a recovery. Run alone:
//   node_modules/.bin/esbuild demos/test/sparse-test.ts --bundle --platform=node --log-level=warning | node
import {
  softVec, soft, lassoScalar, matrixOp, ista, fista, istaStep, omp, ridge, lambdaMax, lassoCD, lassoPath, lasso2dExact,
  penalised2d, minNormSolution, mutualCoherence, realFourierOp1d, fourierRows1d, gaussianMatrix, sparseSignal, l1Recover,
  partialFourierOp2d, kspaceMask, blurOp1d, blurOp2d, maskOp, basisOp1d, composeOp, unionOp, LinOp, powerIteration,
  lassoObjective, debias, denseMatrix, norm2, recoveryTrial, DONOHO_TANNER_L1, misfit2, pixelOp1d, Sym2,
} from "../src/lib/sparse";
import {
  extractPatches, aggregatePatches, batchOmp, dctDictionary, gram, randomPatchDictionary, ksvdUpdate, representationError,
  rank1, reconstruct, Codes, patchPCA, ksvdDenoise, patchInpaint, learnStep,
} from "../src/lib/patches";
import { soft as wsoft, WAVELETS } from "../src/lib/wavelet";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol;
const nearV = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9) => a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol);
const fmt = (a: ArrayLike<number>) => "(" + Array.from(a, v => v.toFixed(4)).join(", ") + ")";

// ---- penalties ----
const R = rng(11);
{
  const v = Float64Array.from({ length: 200 }, () => 6 * R.unif() - 3);
  const s = softVec(v, 0.7);
  check(Array.from(v).every((x, i) => s[i] === wsoft(x, 0.7)), "softVec equals wavelet.soft elementwise");
  const w = [4, -0.5, 1.2, -2.5, 0.3, 3];
  check(nearV(w.map(x => lassoScalar(x, 1, 2)), [2, -0.25, 0.6, -1.25, 0.15, 1.5]), "l2 shrinkage halves");
  check(nearV(w.map(x => lassoScalar(x, 1, 1)), [3, 0, 0.2, -1.5, 0, 2], 1e-12), "l1 soft threshold numbers");
  check(nearV(w.map(x => lassoScalar(x, 1, 0)), [4, 0, 1.2, -2.5, 0, 3]), "l0 hard threshold numbers");
  // l1/2 scalar: compare with brute force
  let ok = true;
  for (const x of [0.3, 0.8, 1.2, 2, 3.5, -2.2]) {
    let best = 0, bf = 0.5 * x * x;
    for (let c = -5; c <= 5; c += 1e-4) { const f = 0.5 * (c - x) ** 2 + Math.sqrt(Math.abs(c)); if (f < bf) { bf = f; best = c; } }
    if (Math.abs(lassoScalar(x, 1, 0.5) - best) > 2e-3) ok = false;
  }
  check(ok, "l1/2 scalar matches brute force");
}

// ---- the by-hand ISTA example (unit 2) ----
const A2 = Float64Array.from([1, 0.5, 0.5, 1]), y2 = Float64Array.from([2, 1]), op2 = matrixOp(A2, 2, 2);
{
  const prox = (v: Float64Array, t: number) => softVec(v, t * 0.5);
  const s1 = istaStep(op2, y2, new Float64Array(2), 0.4, prox);
  check(nearV(s1.z, [1.0, 0.8], 1e-12) && nearV(s1.x, [0.8, 0.6], 1e-12), "ISTA step 1: gradient (1, 0.8), threshold (0.8, 0.6) " + fmt(s1.x));
  const s2 = istaStep(op2, y2, s1.x, 0.4, prox);
  check(nearV(s2.x, [0.96, 0.58], 1e-12), "ISTA step 2: (0.96, 0.58) " + fmt(s2.x));
  const r = ista(op2, y2, { lambda: 0.5, step: 0.4, iters: 400 });
  check(nearV(r.x, [1.6, 0], 1e-8), "ISTA converges to (1.6, 0) " + fmt(r.x));
  let mono = true; for (let k = 1; k < r.history.length; k++) if (r.history[k] > r.history[k - 1] + 1e-12) mono = false;
  check(mono, "ISTA objective non-increasing");
  check(nearV(lasso2dExact([1.25, 1, 1.25], [2, 0], 0.5), [1.6, 0], 1e-12), "exact 2D lasso agrees (1.6, 0)");
  const rr = ridge(A2, 2, 2, y2, 0.5);
  check(nearV(rr, [1.1515151515, 0.4848484848], 1e-8) && rr[1] !== 0, "ridge (1.152, 0.485), no zero " + fmt(rr));
  const o = omp(A2, 2, 2, y2, { K: 2, tol: 1e-12 });
  check(nearV(o.x, [2, 0], 1e-12) && o.support.length === 1, "OMP returns (2, 0) after one atom " + fmt(o.x));
  check(near(powerIteration(op2), 2.25, 1e-9), "L = 2.25 for the by-hand matrix");
}

// ---- lambda max ----
{
  const m = 20, n = 30, A = gaussianMatrix(m, n, rng(3)), y = Float64Array.from({ length: m }, () => R.norm());
  const op = matrixOp(A, m, n), lm = lambdaMax(op, y);
  const x1 = lassoCD(A, m, n, y, lm);
  check(norm2(x1) === 0, "lasso at lambda_max is zero");
  const x2 = lassoCD(A, m, n, y, 0.99 * lm);
  check(norm2(x2) > 0, "lasso at 0.99 lambda_max is not zero");
  // FISTA vs ISTA after 100 iterations, and CD as reference
  const lam = 0.1 * lm;
  const fi = fista(op, y, { lambda: lam, iters: 100 }), is = ista(op, y, { lambda: lam, iters: 100 });
  check(fi.history[99] <= is.history[99] + 1e-12, `FISTA objective <= ISTA after 100 iterations (${fi.history[99].toFixed(6)} vs ${is.history[99].toFixed(6)})`);
  const cd = lassoCD(A, m, n, y, lam, undefined, 5000);
  const fcd = lassoObjective(op, y, cd, lam), f2 = lassoObjective(op, y, fista(op, y, { lambda: lam, iters: 3000 }).x, lam);
  check(near(fcd, f2, 1e-7), `CD and long FISTA reach the same objective (${fcd} vs ${f2})`);
  // path in 2D agrees with exact enumeration
  const B = Float64Array.from([1, 0.6, 0.2, 1, 0.5, -0.3]), yy = Float64Array.from([2, 1, -0.5]);
  const Q: Sym2 = [1 + 0.04 + 0.25, 0.6 + 0.2 - 0.15, 0.36 + 1 + 0.09];
  const lsq = ridge(B, 3, 2, yy, 0), lams = [3, 1.5, 0.8, 0.3, 0.1, 0.01];
  const path = lassoPath(B, 3, 2, yy, lams);
  let ok = true;
  for (let k = 0; k < lams.length; k++) { const e = lasso2dExact(Q, [lsq[0], lsq[1]], lams[k]); if (!nearV(e, path[k], 1e-7)) ok = false; }
  check(ok, "coordinate-descent lasso path matches exact 2D enumeration");
  // l2 and l0 in 2D
  const pr = penalised2d([1, 0, 1], [3, 1], 1, 2);
  check(nearV(pr, [1.5, 0.5]), "2D ridge with round contours halves");
  const p0 = penalised2d([1, 0, 1], [3, 1], 1.5, 0);
  check(nearV(p0, [3, 0]), "2D l0 with round contours = hard threshold " + fmt(p0));
  const ph = penalised2d([1, 0.5, 1], [3, 1], 1, 0.5);
  let bf = Infinity, bx = [0, 0];
  for (let a = -4; a <= 4; a += 0.01) for (let c = -4; c <= 4; c += 0.01) { const f = misfit2([1, 0.5, 1], [3, 1], [a, c]) + Math.sqrt(Math.abs(a)) + Math.sqrt(Math.abs(c)); if (f < bf) { bf = f; bx = [a, c]; } }
  const fph = misfit2([1, 0.5, 1], [3, 1], ph) + Math.sqrt(Math.abs(ph[0])) + Math.sqrt(Math.abs(ph[1]));
  check(fph <= bf + 1e-6, `2D l1/2 at least as good as a grid search (${fph} vs ${bf}, ${fmt(ph)} vs ${fmt(bx)})`);
}

// ---- budget picture: projection on the l1 ball of radius 2 from (3, 1) is (2, 0) ----
{
  // l1 ball projection = soft threshold with theta s.t. sum = 2
  const th = 1, p = [soft(3, th), soft(1, th)];
  check(nearV(p, [2, 0]), "l1-ball projection of (3, 1) with budget 2 is (2, 0)");
  const s = 2 / Math.hypot(3, 1);
  check(near(3 * s, 1.897366596, 1e-8) && near(1 * s, 0.632455532, 1e-8), "l2-ball projection (1.90, 0.63)");
}

// ---- compressed sensing: the 2×4 example ----
{
  const A = Float64Array.from([1, 1, 1, 1, 1, -1, 2, 0]), y = Float64Array.from([3, 6]);
  const mn = minNormSolution(A, 2, 4, y);
  check(nearV(mn, [1.2, -0.6, 2.1, 0.3], 1e-12), "min-norm solution (1.2, -0.6, 2.1, 0.3) " + fmt(mn));
  const l1 = l1Recover(A, 2, 4, y, { stages: 10, iters: 2000 });
  check(nearV(l1, [0, 0, 3, 0], 2e-3), "l1 recovers (0, 0, 3, 0) " + fmt(l1));
  check(near(mutualCoherence(A, 2, 4), 3 / Math.sqrt(10), 1e-12), "mutual coherence 3/sqrt(10) = 0.949");
}

// ---- fold-over: even DFT coefficients → ½(x_n + x_{n+N/2}) ----
{
  const N = 16, x = Float64Array.from({ length: N }, () => R.norm());
  const rows = fourierRows1d(N, "regular", 0, R, 2), op = realFourierOp1d(N, rows);
  const z = op.adjoint(op.forward(x));
  check(nearV(z, Array.from({ length: N }, (_, n) => 0.5 * (x[n] + x[(n + N / 2) % N])), 1e-12), "keep even frequencies: zero-filled = ½(x_n + x_{n+N/2})");
  check(rows.length === N / 2, "regular r = 2 keeps N/2 rows");
  const x8 = Float64Array.from([0, 0, 5, 0, 0, 0, 0, 0]), op8 = realFourierOp1d(8, fourierRows1d(8, "regular", 0, R, 2));
  check(nearV(op8.adjoint(op8.forward(x8)), [0, 0, 2.5, 0, 0, 0, 2.5, 0], 1e-12), "by-hand fold (0,0,2.5,0,0,0,2.5,0)");
  const op84 = realFourierOp1d(8, fourierRows1d(8, "regular", 0, R, 4));
  check(nearV(op84.adjoint(op84.forward(x8)), [1.25, 0, 1.25, 0, 1.25, 0, 1.25, 0], 1e-12), "every 4th: four copies of 1.25");
}

// ---- adjoints ----
function adjointOk(op: LinOp, label: string, tol = 1e-9): void {
  const x = Float64Array.from({ length: op.n }, () => R.norm()), y = Float64Array.from({ length: op.m }, () => R.norm());
  const a = op.forward(x), b = op.adjoint(y);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < op.m; i++) s1 += a[i] * y[i];
  for (let i = 0; i < op.n; i++) s2 += x[i] * b[i];
  check(Math.abs(s1 - s2) <= tol * (Math.abs(s1) + 1), `adjoint consistent: ${label} (${s1.toFixed(8)} vs ${s2.toFixed(8)})`);
}
{
  const W = { haar: WAVELETS.haar, db4: WAVELETS.db4 };
  adjointOk(realFourierOp1d(64, fourierRows1d(64, "random", 20, R)), "random real-Fourier rows");
  adjointOk(pixelOp1d(64, fourierRows1d(64, "random", 20, R)), "random pixels");
  adjointOk(blurOp1d(64, 2.5), "1D blur");
  adjointOk(blurOp2d(32, 32, 1.5), "2D blur");
  adjointOk(maskOp(Uint8Array.from({ length: 64 }, () => (R.unif() < 0.5 ? 1 : 0))), "mask");
  for (const b of ["spikes", "dct", "haar", "db4", "fourier"] as const) adjointOk(basisOp1d(64, b, W, 4), "basis " + b);
  adjointOk(composeOp(blurOp1d(64, 2), basisOp1d(64, "db4", W, 4)), "blur ∘ db4");
  adjointOk(unionOp(basisOp1d(64, "spikes"), basisOp1d(64, "dct")), "spikes ∪ DCT");
  for (const kind of ["centre", "regular", "random", "vardens", "radial"] as const) {
    const m = kspaceMask(32, kind, 4, rng(5));
    adjointOk(partialFourierOp2d(m, 32), "partial Fourier 2D " + kind);
    // symmetric mask → A^T A is a projection, norm 1
    const op = partialFourierOp2d(m, 32), x = Float64Array.from({ length: 1024 }, () => R.norm());
    const z1 = op.adjoint(op.forward(x)), z2 = op.adjoint(op.forward(z1));
    check(nearV(z1, z2, 1e-9), "partial Fourier 2D AᵀA is a projection: " + kind);
  }
  // orthonormality of the bases: adjoint ∘ forward = identity
  for (const b of ["dct", "haar", "db4", "fourier"] as const) {
    const op = basisOp1d(64, b, W, 4), c = Float64Array.from({ length: 64 }, () => R.norm());
    check(nearV(op.adjoint(op.forward(c)), c, 1e-10), "orthonormal basis " + b);
  }
  // regular 2D mask (every 2nd line) folds the image by half the field of view
  const n = 16, img = new Float64Array(n * n); img[3 * n + 5] = 1;
  const m2 = new Uint8Array(n * n); for (let ky = 0; ky < n; ky += 2) for (let kx = 0; kx < n; kx++) m2[ky * n + kx] = 1;
  const z = partialFourierOp2d(m2, n).adjoint(partialFourierOp2d(m2, n).forward(img));
  check(near(z[3 * n + 5], 0.5) && near(z[(3 + n / 2) * n + 5], 0.5), "every 2nd k-space line: ghost at half the field of view");
}

// ---- recovery ----
{
  const t = recoveryTrial(128, 50, 5, "gaussian", "l1", 42);
  check(t.success, `K = 5, N = 128, M = 50 Gaussian: l1 recovers (rel. error ${t.relErr.toExponential(2)})`);
  const t2 = recoveryTrial(128, 50, 5, "gaussian", "omp", 42);
  check(t2.success, `… and OMP too (rel. error ${t2.relErr.toExponential(2)})`);
  const t3 = recoveryTrial(128, 20, 15, "gaussian", "l1", 42);
  check(!t3.success, "K = 15 from M = 20 fails");
  check(DONOHO_TANNER_L1.delta.length === DONOHO_TANNER_L1.rho.length && near(DONOHO_TANNER_L1.rho[24], 0.3857), "DT table: rho(0.5) = 0.386");
  // debias on the true support gives exact recovery
  const N = 64, M = 24, A = gaussianMatrix(M, N, rng(9)), x = sparseSignal(N, 4, rng(10));
  const y = matrixOp(A, M, N).forward(x), sup = Array.from(x.keys()).filter(i => x[i] !== 0);
  check(nearV(debias(A, M, N, y, sup), x, 1e-9), "debias on the true support is exact");
  check(nearV(denseMatrix(matrixOp(A, M, N)), A, 0), "denseMatrix of a matrix op");
}

// ---- patches ----
{
  const w = 20, h = 16, img = Float64Array.from({ length: w * h }, () => R.unif() * 100);
  for (const stride of [1, 3]) {
    const P = extractPatches(img, w, h, 6, stride);
    check(nearV(aggregatePatches(P.Y, P.pos, P.n, 6, w, h), img, 1e-9), `extract + aggregate = identity (stride ${stride})`);
  }
  const P = extractPatches(img, w, h, 4, 1);
  check(P.n === 17 * 13, "count of overlapping 4×4 patches of 20×16");
  // Batch-OMP equals plain OMP
  const D = dctDictionary(4, 64), d = 16, k = 64;
  const codes = batchOmp(D, d, P.Y, 20, { K: 4 });
  let ok = true;
  const Dmat = new Float64Array(d * k); for (let j = 0; j < k; j++) for (let t = 0; t < d; t++) Dmat[t * k + j] = D[j * d + t];
  for (let i = 0; i < 20; i++) {
    const o = omp(Dmat, d, k, P.Y.subarray(i * d, (i + 1) * d), { K: 4 });
    const bx = new Float64Array(k); for (let s = 0; s < 4; s++) if (codes.idx[i * 4 + s] >= 0) bx[codes.idx[i * 4 + s]] = codes.val[i * 4 + s];
    if (!nearV(bx, o.x, 1e-8)) ok = false;
  }
  check(ok, "Batch-OMP equals plain OMP (20 patches, K = 4)");
  // orthonormal DCT dictionary: K = 16 is exact
  const D16 = dctDictionary(4, 16), G = gram(D16, 16);
  check(Array.from(G).every((v, i) => near(v, Math.floor(i / 16) === i % 16 ? 1 : 0, 1e-12)), "dctDictionary(p, p²) is orthonormal");
  const c16 = batchOmp(D16, 16, P.Y, P.n, { K: 16 });
  check(representationError(D16, 16, P.Y, c16) < 1e-18, "full DCT code is exact");
  // error target stops early
  const ce = batchOmp(D, d, P.Y, P.n, { K: 16, errTarget: 16 * 100 });
  check(Array.from(ce.err).every(e => e <= 1600 + 1e-6 || true) && Array.from({ length: P.n }, (_, i) => ce.err[i] <= 1600 + 1e-6 || ce.idx[i * ce.maxK + ce.maxK - 1] >= 0).every(Boolean), "Batch-OMP stops at the error target or at K");
  // K-SVD objective non-increasing over 5 iterations
  const img2 = Float64Array.from({ length: 32 * 32 }, (_, q) => 50 * Math.sin((q % 32) / 3) + 30 * ((q >> 5) > 16 ? 1 : 0) + 5 * R.norm());
  const P2 = extractPatches(img2, 32, 32, 5, 1), D2 = randomPatchDictionary(P2, 40, rng(4));
  let prev = Infinity, mono = true;
  for (let it = 0; it < 5; it++) {
    const c = batchOmp(D2, 25, P2.Y, P2.n, { K: 3 });
    const before = representationError(D2, 25, P2.Y, c);
    ksvdUpdate(D2, 25, P2.Y, c);
    const after = representationError(D2, 25, P2.Y, c);
    if (after > before + 1e-9) mono = false;
    console.log(`  K-SVD iteration ${it + 1}: after coding ${before.toFixed(3)}, after update ${after.toFixed(3)}`);
    prev = after;
  }
  check(mono, "K-SVD atom update never increases the error (5 iterations)");
  // learnStep runs for all methods
  for (const m of ["ksvd", "mod", "kmeans"] as const) {
    const Dm = randomPatchDictionary(P2, 16, rng(8));
    const e1 = learnStep(Dm, 25, P2, 2, m).error, e2 = learnStep(Dm, 25, P2, 2, m).error;
    check(Number.isFinite(e1) && Number.isFinite(e2), "learnStep " + m);
  }
  // the by-hand K-SVD update: residuals (2,1), (4,2), (−1,−1)
  const E = Float64Array.from([2, 1, 4, 2, -1, -1]), r1 = rank1(E, 2, 3, [1, 0], 200);
  const u = r1.u[0] < 0 ? Array.from(r1.u, v => -v) : Array.from(r1.u), sv = r1.u[0] < 0 ? Array.from(r1.sv, v => -v) : Array.from(r1.sv);
  check(near(u[0], 0.8841, 5e-4) && near(u[1], 0.4673, 5e-4), "K-SVD by hand: atom (0.884, 0.467) " + fmt(u));
  check(near(sv[0], 2.236, 2e-3) && near(sv[1], 4.471, 2e-3) && near(sv[2], -1.351, 2e-3), "K-SVD by hand: coefficients (2.24, 4.47, −1.35) " + fmt(sv));
  // PCA atoms orthonormal
  const B = patchPCA(P2, 5);
  let orth = true;
  for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) { let s = 0; for (let t = 0; t < 25; t++) s += B[a * 25 + t] * B[b * 25 + t]; if (!near(s, a === b ? 1 : 0, 1e-6)) orth = false; }
  check(orth, "patch PCA atoms orthonormal");
  // denoising improves the error; inpainting fills holes
  const clean = Float64Array.from({ length: 32 * 32 }, (_, q) => 100 + 60 * ((q % 32) > 12 ? 1 : 0) + 40 * ((q >> 5) > 20 ? 1 : 0));
  const noisy = Float64Array.from(clean, v => v + 15 * R.norm());
  const den = ksvdDenoise(noisy, 32, 32, dctDictionary(6, 144), 6, 15, { stride: 1 });
  const mse = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - clean[i]) ** 2; return s / a.length; };
  check(mse(den) < 0.4 * mse(noisy), `patch denoising lowers the MSE (${mse(noisy).toFixed(1)} → ${mse(den).toFixed(1)})`);
  const known = Uint8Array.from({ length: 32 * 32 }, () => (R.unif() < 0.6 ? 1 : 0));
  const holes = Float64Array.from(clean, (v, q) => (known[q] ? v : 0));
  const inp = patchInpaint(holes, known, 32, 32, dctDictionary(6, 144), 6, { stride: 1, K: 4 });
  check(mse(inp) < 0.05 * mse(holes), `patch inpainting fills 40 % missing pixels (${mse(holes).toFixed(0)} → ${mse(inp).toFixed(1)})`);
  void reconstruct; void (null as unknown as Codes);
}

console.log(`sparse-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
