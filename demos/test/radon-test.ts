// Reference checks for lib/radon.ts (projector / back-projector, filters, FBP, ART/SIRT)
// and the small-system model of the artsmall demo. Run:
//   node_modules/.bin/esbuild demos/test/radon-test.ts --bundle --platform=node --log-level=warning | node
import * as R from "../src/lib/radon";
import { rng } from "../src/lib/random";
import { fft1d, fft2dRect } from "../src/lib/fft";
import { sheppLogan } from "../src/lib/synth";
import * as S from "../src/artsmall/model";
import * as FS from "../src/fourierslice/model";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
let current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const randArr = (r: ReturnType<typeof rng>, n: number) => Float64Array.from({ length: n }, () => r.unif() - 0.3);

function testAdjoint(): void {
  current = "adjoint";
  const r = rng(3);
  for (const [n, m, nB, dR] of [[16, 7, 16, 1], [17, 12, 23, 0.8], [32, 45, 32, 1], [9, 5, 13, 1.3]] as const) {
    const g = R.geometry(n, R.angleSet(m, 180, 3.7), nB, dR);
    for (let t = 0; t < 3; t++) {
      const x = randArr(r, n * n), y = randArr(r, m * nB);
      const lhs = R.dotOf(R.project(x, g), y), rhs = R.dotOf(x, R.backproject(y, g));
      check(close(lhs, rhs, 1e-12), `<Ax,y>=<x,A^T y> n=${n}: ${lhs} vs ${rhs}`);
    }
    // the explicit matrix agrees with both
    const A = R.systemMatrix(g), x = randArr(r, n * n), y = randArr(r, m * nB);
    check(maxDiff(R.csrMul(A, x), R.project(x, g)) < 1e-12, "CSR A x = project");
    check(maxDiff(R.csrMulT(A, y), R.backproject(y, g)) < 1e-12, "CSR A^T y = backproject");
  }
  // ray norms
  const n = 12, th = 0.4, rho = 1.3;
  const { w } = R.rayWeights(n, th, rho);
  check(close(R.rayNorm2(n, th, rho), w.reduce((s, v) => s + v * v, 0)), "rayNorm2");
}

function testBasics(): void {
  current = "basics";
  const n = 8, r = rng(5);
  const x = randArr(r, n * n);
  const g = R.geometry(n, Float64Array.from([0, Math.PI / 2]));
  const p = R.project(x, g);
  // theta = 0: column sums (rho = x = col - cen); theta = 90: row sums, top row at +rho
  for (let col = 0; col < n; col++) {
    let s = 0; for (let row = 0; row < n; row++) s += x[row * n + col];
    check(close(p[col], s, 1e-12), `0 deg: column sum ${col}`);
  }
  for (let b = 0; b < n; b++) {
    const row = n - 1 - b;
    let s = 0; for (let col = 0; col < n; col++) s += x[row * n + col];
    check(close(p[n + b], s, 1e-12), `90 deg: row sum, bin ${b} = row ${row}`);
  }
  // mass: every projection of an object inside the field of view sums to the image total
  const N = 64, img = R.ellipseImage(N, R.sheppLoganEllipses(N)), G = R.geometry(N, R.angleSet(30));
  const sino = R.project(img, G), tot = img.reduce((s, v) => s + v, 0);
  for (let a = 0; a < 30; a++) {
    let s = 0; for (let b = 0; b < N; b++) s += sino[a * N + b];
    check(Math.abs(s - tot) < 0.003 * tot, `mass at angle ${a}: ${s} vs ${tot}`);
  }
  // a point mass traces rho = x cos + y sin
  const pt = new Float64Array(N * N);
  R.addPoint(pt, N, 10, -7, 1);
  const sp = R.project(pt, G);
  for (let a = 0; a < 30; a++) {
    let m0 = 0, m1 = 0;
    for (let b = 0; b < N; b++) { m0 += sp[a * N + b]; m1 += sp[a * N + b] * R.rhoOf(b, N); }
    check(Math.abs(m1 / m0 - R.pointRho(10, -7, G.angles[a])) < 0.1, `point centroid on its sinusoid, angle ${a}: ${m1 / m0} vs ${R.pointRho(10, -7, G.angles[a])}`);
  }
  // symmetry p(theta + 180, rho) = p(theta, -rho)
  const g2 = R.geometry(N, Float64Array.from([0.3, 0.3 + Math.PI]));
  const s2 = R.project(img, g2);
  let md = 0; for (let b = 0; b < N; b++) md = Math.max(md, Math.abs(s2[b] - s2[N + N - 1 - b]));
  check(md < 1e-9, `mirror symmetry, max diff ${md}`);
}

function testAnalytic(): void {
  current = "analytic";
  const n = 128, el = R.sheppLoganEllipses(n), g = R.geometry(n, R.angleSet(180));
  const img = R.ellipseImage(n, el, 4);
  // lib/synth's raster (2x supersampled) agrees with ours away from edges
  const syn = sheppLogan(n);
  let agree = 0; for (let i = 0; i < n * n; i++) if (Math.abs(syn.data[i] - img[i]) < 1e-6) agree++;
  check(agree / (n * n) > 0.9, `synth Shepp-Logan agrees on ${agree} pixels`);
  const exact = R.ellipseSinogram(el, g), num = R.project(img, g);
  let mx = 0; for (const v of exact) mx = Math.max(mx, v);
  const rel = R.rmseOf(num, exact) / mx;
  check(rel < 0.02, `projection vs analytic sinogram: rel RMSE ${rel.toFixed(4)}`);
  // disc of radius R: 2 v sqrt(R^2 - rho^2)
  const d = R.disc(0, 0, 20, 1);
  check(close(R.ellipseProjection(d, 0.7, 12), 2 * Math.sqrt(400 - 144)), "disc chord");
  // FBP of the analytic sinogram reconstructs the phantom
  const rec = R.fbp(exact, g, { kind: "ramp" });
  const mask = R.fovMask(n);
  const e = R.rmseOf(rec, img, mask);
  check(e < 0.06, `FBP(ramp) of exact sinogram: RMSE ${e.toFixed(4)}`);
  const recH = R.fbp(exact, g, { kind: "hann" });
  check(R.rmseOf(recH, img, mask) < 0.08, `FBP(hann): RMSE ${R.rmseOf(recH, img, mask).toFixed(4)}`);
  // plain back-projection is much worse (1/r blur)
  const bp = R.plainBackprojection(exact, g);
  check(R.rmseOf(bp, img, mask) > 5 * e, "plain back-projection is blurred");
  // no padding: worse than padded
  const recNP = R.fbp(exact, g, { kind: "ramp", pad: false });
  check(R.rmseOf(recNP, img, mask) > e, `no padding worse: ${R.rmseOf(recNP, img, mask).toFixed(4)} vs ${e.toFixed(4)}`);
}

function testFilters(): void {
  current = "filters";
  // the FFT filter with padding equals direct convolution with the full Ram-Lak kernel
  const r = rng(9), nB = 37, p = randArr(r, nB);
  const viaFFT = R.filterSinogram(p, 1, nB, { kind: "ramp", pad: true });
  const direct = R.filterDirect(p);
  check(maxDiff(viaFFT, direct) < 1e-12, `FFT ramp = direct convolution (${maxDiff(viaFFT, direct)})`);
  // the response is close to |f| (and positive at DC: 1/(4) - 2 sum 1/(pi k)^2 over odd k... )
  const L = 256, H = R.filterResponse("ramp", L);
  let md = 0; for (let k = 1; k < L / 2; k++) md = Math.max(md, Math.abs(H[k] - k / L));
  check(md < 0.01, `ramp response ~ |f| (max dev ${md.toFixed(4)})`);
  check(H[0] > 0 && H[0] < 0.01, `DC of the sampled ramp is small and positive: ${H[0]}`);
  // tapers equal 1 at DC and 0 beyond the cutoff
  for (const k of ["shepp-logan", "cosine", "hann"] as const) {
    check(close(R.taper(k, 0), 1), `${k} at 0`);
    check(R.taper(k, 0.3, 0.25) === 0, `${k} beyond cutoff`);
  }
  check(close(R.taper("hann", 0.25), 0.5), "hann at half the cutoff");
  // by hand (fbp unit): box (0,0,1,1,1,0,0) with the truncated kernel (-0.1, 0.25, -0.1)
  const box = [0, 0, 1, 1, 1, 0, 0], k3 = [-0.1, 0.25, -0.1];
  const out = box.map((_, i) => k3.reduce((s, kv, j) => s + kv * (box[i + j - 1] ?? 0), 0));
  check(maxDiff(out, [0, -0.1, 0.15, 0.05, 0.15, -0.1, 0]) < 1e-12, "by-hand filtered box");
}

function testFourierSlice(): void {
  current = "fourier slice";
  // 1D DFT of the 0-degree projection = the v = 0 row of the 2D DFT, exactly
  const n = 32, r = rng(4), img = randArr(r, n * n);
  const p = R.project(img, R.geometry(n, Float64Array.from([0])));
  const re = Float64Array.from(p), im = new Float64Array(n);
  fft1d(re, im);
  const Re = Float64Array.from(img), Im = new Float64Array(n * n);
  fft2dRect(Re, Im, n, n);
  check(maxDiff(re, Re.subarray(0, n)) < 1e-9 && maxDiff(im, Im.subarray(0, n)) < 1e-9, "0 deg slice exact");
  // 90 deg: the 1D DFT of the row sums (bin b = row n-1-b) equals the u = 0 column (reversed order)
  const q = R.project(img, R.geometry(n, Float64Array.from([Math.PI / 2])));
  const rr = new Float64Array(n), ii = new Float64Array(n);
  for (let row = 0; row < n; row++) rr[row] = q[n - 1 - row];
  fft1d(rr, ii);
  let md = 0; for (let v = 0; v < n; v++) md = Math.max(md, Math.abs(rr[v] - Re[v * n]), Math.abs(ii[v] - Im[v * n]));
  check(md < 1e-9, "90 deg slice exact");
}

function testSliceModel(): void {
  current = "fourierslice model";
  const n = 64, img = R.ellipseImage(n, R.sheppLoganEllipses(n), 3), G = FS.centredSpectrum(img, n);
  for (const [deg, tol] of [[0, 1e-9], [90, 1e-9], [30, 0.12], [45, 0.12]] as const) {
    const th = deg * Math.PI / 180, p = R.projectAngle(img, n, th);
    const P = FS.projectionSpectrum(p, n), sl = FS.sliceOf(G, th);
    const d = FS.relDiff(P.re, P.im, sl.re, sl.im);
    check(d < tol, `slice at ${deg} deg: rel diff ${d.toExponential(2)}`);
  }
  // direct reconstruction with many spokes and bilinear gridding is close to the image
  const g = R.geometry(n, R.angleSet(45)), sino = R.project(img, g);
  const rb = FS.directRecon(sino, n, g.angles, "bilinear"), rn = FS.directRecon(sino, n, g.angles, "nearest");
  const mask = R.fovMask(n), eb = R.rmseOf(rb.img, img, mask), en = R.rmseOf(rn.img, img, mask);
  check(eb < 0.15, `direct recon (bilinear) RMSE ${eb.toFixed(3)}`);
  check(en > eb, `nearest gridding worse: ${en.toFixed(3)} vs ${eb.toFixed(3)}`);
}

function testIterative(): void {
  current = "iterative";
  const n = 32, el = R.sheppLoganEllipses(n), img = R.ellipseImage(n, el, 4);
  const g = R.geometry(n, R.angleSet(48));
  const b = R.project(img, g), mask = R.fovMask(n);
  // SIRT from zero: the first step is a weighted back-projection; error decreases
  const sirt = new R.IterativeRecon(b, g, { method: "sirt" });
  const errs: number[] = [];
  for (let k = 0; k < 60; k++) { sirt.step(); errs.push(R.rmseOf(sirt.x, img, mask)); }
  check(errs[59] < errs[0] && errs[59] < errs[9], `SIRT error falls ${errs[0].toFixed(3)} -> ${errs[59].toFixed(3)}`);
  const s1 = new R.IterativeRecon(b, g, { method: "sirt" });
  s1.step();
  const { row, col } = R.rowColSums(g);
  const bp = R.backproject(b.map((v, i) => (row[i] > 1e-9 ? v / row[i] : 0)), g);
  check(maxDiff(s1.x, bp.map((v, p) => (col[p] > 1e-9 ? v / col[p] : 0))) < 1e-12, "SIRT step 1 = normalised back-projection");
  // ART reaches a smaller residual than SIRT in fewer sweeps
  const art = new R.IterativeRecon(b, g, { method: "art", omega: 1 });
  for (let k = 0; k < 10; k++) art.step();
  check(art.residual() < sirt.residual(), `ART 10 sweeps residual ${art.residual().toFixed(4)} < SIRT 60 ${sirt.residual().toFixed(4)}`);
  // non-negativity and TV keep the image >= 0
  const tv = new R.IterativeRecon(b, g, { method: "sirt", tv: 0.02 });
  for (let k = 0; k < 10; k++) tv.step();
  check(tv.x.every(v => v >= 0), "TV iterate non-negative");
  // noise: the same seed gives the same sinogram; I0 = inf gives no noise
  const n1 = R.noisySinogram(b, 1e3, 7), n2 = R.noisySinogram(b, 1e3, 7), n3 = R.noisySinogram(b, Infinity);
  check(maxDiff(n1.sino, n2.sino) === 0, "seeded noise");
  check(maxDiff(n3.sino, b) === 0, "no noise at I0 = inf");
  check(R.rmseOf(n1.sino, b) > 0, "noise present");
}

function testSmall(): void {
  current = "small systems";
  // the 2x2 example: rows and columns; (1,2;3,4) and (2,1;2,5) give the same data (3,7,4,6)
  const sys = S.smallSystem(2, ["rows", "cols"]);
  const a = [1, 2, 3, 4], b2 = [2, 1, 2, 5];
  check(maxDiff(S.apply(sys, a), [3, 7, 4, 6]) === 0, "data of (1,2;3,4)");
  check(maxDiff(S.apply(sys, b2), [3, 7, 4, 6]) === 0, "data of (2,1;2,5)");
  check(S.rankOf(sys) === 3, `rank 3 (got ${S.rankOf(sys)})`);
  const withDiag = S.smallSystem(2, ["rows", "cols", "diag"]);
  check(S.rankOf(withDiag) === 4, "rank 4 with the diagonals");
  // one Kaczmarz sweep from 0 in the order row 1, row 2, col 1, col 2
  const path = S.kaczmarzPath(sys, [3, 7, 4, 6], [0, 0, 0, 0], [0, 1, 2, 3], 1);
  const want = [[1.5, 1.5, 0, 0], [1.5, 1.5, 3.5, 3.5], [1, 1.5, 3, 3.5], [1, 2, 3, 4]];
  for (let k = 0; k < 4; k++) check(maxDiff(path[k + 1], want[k]) < 1e-12, `Kaczmarz step ${k + 1}: ${Array.from(path[k + 1])}`);
  // one SIRT step from zero (averaging): (1.75, 2.25, 2.75, 3.25)
  const s = S.sirtStep(sys, [3, 7, 4, 6], [0, 0, 0, 0], 1);
  check(maxDiff(s, [1.75, 2.25, 2.75, 3.25]) < 1e-12, `SIRT step: ${Array.from(s)}`);
  // A^T b = (7, 9, 11, 13)
  check(maxDiff(S.applyT(sys, [3, 7, 4, 6]), [7, 9, 11, 13]) === 0, "A^T b");
}

const t0 = Date.now();
testAdjoint();
testBasics();
testAnalytic();
testFilters();
testFourierSlice();
testSliceModel();
testIterative();
testSmall();
console.log(`radon-test: ${checks - failures}/${checks} checks passed (${Date.now() - t0} ms)`);
if (failures) process.exit(1);
