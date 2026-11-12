// Reference checks for the 2D image and numeric libraries in demos/src/lib/:
// random, fft, linalg, imageops, synth, histogram, metrics, label, distance, morph,
// pqueue, polar. Each routine is compared with a brute-force or textbook version on
// small random inputs. Run:
//   node_modules/.bin/esbuild demos/test/foundation-2d-test.ts --bundle --platform=node --log-level=warning | node
import { rng, gauss, poisson, uniform, shuffle, permutation, randInt, choice } from "../src/lib/random";
import * as F from "../src/lib/fft";
import * as LA from "../src/lib/linalg";
import * as IO from "../src/lib/imageops";
import * as SY from "../src/lib/synth";
import * as H from "../src/lib/histogram";
import * as ME from "../src/lib/metrics";
import * as LB from "../src/lib/label";
import * as DT from "../src/lib/distance";
import * as MO from "../src/lib/morph";
import { PriorityQueue, Heap, BucketQueue, Fifo } from "../src/lib/pqueue";
import * as PO from "../src/lib/polar";
import { fft2d as oldFft2d, rmse as oldRmse, makeMask } from "../src/fourier2d/fft";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
const section: Record<string, number> = {};
let current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  section[current] = (section[current] ?? 0) + 1;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const randImg = (r: ReturnType<typeof rng>, w: number, h: number, scale = 1) => IO.fromFunction(w, h, () => scale * r.unif());
const randMask = (r: ReturnType<typeof rng>, n: number, p: number) => Uint8Array.from({ length: n }, () => (r.unif() < p ? 1 : 0));

// ---------------------------------------------------------------------------------------
function testRandom(): void {
  current = "random";
  const a = rng(5), b = rng(5);
  for (let i = 0; i < 10; i++) check(a.unif() === b.unif(), "same seed, same sequence");
  const r = rng(11);
  for (const lam of [0.5, 4, 25, 60, 400]) {
    const N = 40000;
    let s = 0, s2 = 0, neg = false, nonint = false;
    for (let i = 0; i < N; i++) { const k = poisson(r, lam); s += k; s2 += k * k; if (k < 0) neg = true; if (k !== Math.floor(k)) nonint = true; }
    const m = s / N, v = s2 / N - m * m;
    check(!neg && !nonint, `poisson(${lam}) non-negative integers`);
    check(Math.abs(m - lam) < 5 * Math.sqrt(lam / N), `poisson(${lam}) mean ${m}`);
    check(Math.abs(v - lam) < 0.05 * lam + 0.05, `poisson(${lam}) variance ${v}`);
  }
  // P(K = k) for lambda = 50 against the pmf around the mean
  {
    const lam = 50, N = 200000, cnt = new Map<number, number>();
    for (let i = 0; i < N; i++) { const k = poisson(r, lam); cnt.set(k, (cnt.get(k) ?? 0) + 1); }
    for (const k of [40, 45, 50, 55, 60]) {
      const p = Math.exp(-lam + k * Math.log(lam) - lgamma(k + 1));
      const obs = (cnt.get(k) ?? 0) / N;
      check(Math.abs(obs - p) < 5 * Math.sqrt(p / N), `poisson pmf at ${k}: ${obs} vs ${p}`);
    }
  }
  let s = 0, s2 = 0;
  for (let i = 0; i < 40000; i++) { const g = gauss(r, 3, 2); s += g; s2 += g * g; }
  check(Math.abs(s / 40000 - 3) < 0.05 && Math.abs(Math.sqrt(s2 / 40000 - (s / 40000) ** 2) - 2) < 0.05, "gauss mean/sd");
  for (let i = 0; i < 1000; i++) { const u = uniform(r, -2, 3); check(u >= -2 && u < 3, "uniform range"); const k = randInt(r, 7); check(k >= 0 && k < 7 && k === Math.floor(k), "randInt"); }
  const p = permutation(r, 50);
  check([...p].sort((x, y) => x - y).every((v, i) => v === i), "permutation is a permutation");
  const arr = [1, 2, 3, 4, 5, 6];
  shuffle(r, arr);
  check([...arr].sort().join() === "1,2,3,4,5,6", "shuffle keeps elements");
  const c = [0, 0, 0];
  for (let i = 0; i < 30000; i++) c[choice(r, [1, 2, 7])]++;
  check(Math.abs(c[2] / 30000 - 0.7) < 0.02, "choice by weight");
}
function lgamma(x: number): number { let s = 0; for (let i = 2; i < x; i++) s += Math.log(i); return s; }

// ---------------------------------------------------------------------------------------
function testFFT(): void {
  current = "fft";
  const r = rng(21);
  for (const n of [1, 2, 4, 8, 16, 64]) {
    const re = Float64Array.from({ length: n }, () => r.norm()), im = Float64Array.from({ length: n }, () => r.norm());
    const ref = F.dft1d(re, im);
    const a = re.slice(), b = im.slice();
    F.fft1d(a, b);
    check(maxDiff(a, ref.re) < 1e-9 && maxDiff(b, ref.im) < 1e-9, `fft1d = dft (n=${n})`);
    F.fft1d(a, b, true);
    check(maxDiff(a.map(v => v / n), re) < 1e-12, `fft1d inverse round trip (n=${n})`);
  }
  // rectangular 2D vs separable DFT
  for (const [w, h] of [[8, 4], [4, 16], [16, 16], [1, 8]]) {
    const img = Float64Array.from({ length: w * h }, () => r.norm());
    const { re, im } = F.fftReal2d(img, w, h);
    // brute-force 2D DFT
    let err = 0;
    for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
      let sr = 0, si = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const ang = -2 * Math.PI * (u * x / w + v * y / h);
        sr += img[y * w + x] * Math.cos(ang); si += img[y * w + x] * Math.sin(ang);
      }
      err = Math.max(err, Math.abs(sr - re[v * w + u]), Math.abs(si - im[v * w + u]));
    }
    check(err < 1e-9, `fft2dRect = brute-force DFT (${w}x${h}), err ${err}`);
    check(maxDiff(F.ifftReal2d(re, im, w, h), img) < 1e-12, `fft2dRect round trip (${w}x${h})`);
  }
  // the fourier2d re-export behaves as before
  {
    const n = 16, img = Float64Array.from({ length: n * n }, () => r.unif());
    const a = img.slice(), b = new Float64Array(n * n), c = img.slice(), d = new Float64Array(n * n);
    oldFft2d(a, b, n); F.fft2dRect(c, d, n, n);
    check(maxDiff(a, c) === 0 && maxDiff(b, d) === 0, "fourier2d/fft.ts fft2d is lib/fft.ts");
    check(oldRmse([1, 2], [1, 4]) === Math.sqrt(2), "fourier2d/fft.ts rmse");
    check(makeMask(4, "all", 0, 0, true, 1, null).every(v => v === 1), "fourier2d makeMask still there");
  }
  // 3D vs brute force
  {
    const [nx, ny, nz] = [4, 2, 8], N = nx * ny * nz;
    const re = Float64Array.from({ length: N }, () => r.norm()), im = new Float64Array(N);
    const a = re.slice(), b = im.slice();
    F.fft3d(a, b, nx, ny, nz);
    let err = 0;
    for (let w = 0; w < nz; w++) for (let v = 0; v < ny; v++) for (let u = 0; u < nx; u++) {
      let sr = 0, si = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const ang = -2 * Math.PI * (u * x / nx + v * y / ny + w * z / nz), val = re[(z * ny + y) * nx + x];
        sr += val * Math.cos(ang); si += val * Math.sin(ang);
      }
      const k = (w * ny + v) * nx + u;
      err = Math.max(err, Math.abs(sr - a[k]), Math.abs(si - b[k]));
    }
    check(err < 1e-9, `fft3d = brute force, err ${err}`);
    F.fft3d(a, b, nx, ny, nz, true);
    check(maxDiff(a, re) < 1e-12 && maxDiff(b, im) < 1e-12, "fft3d round trip");
  }
  // circular convolution / correlation vs brute force
  {
    const w = 8, h = 4, A = Float64Array.from({ length: w * h }, () => r.norm()), B = Float64Array.from({ length: w * h }, () => r.norm());
    const conv = F.convolveFFT(A, B, w, h), corr = F.correlateFFT(A, B, w, h);
    let e1 = 0, e2 = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s1 = 0, s2 = 0;
      for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
        s1 += A[v * w + u] * B[((y - v + h) % h) * w + (x - u + w) % w];
        s2 += A[v * w + u] * B[((v + y) % h) * w + (u + x) % w];
      }
      e1 = Math.max(e1, Math.abs(s1 - conv[y * w + x])); e2 = Math.max(e2, Math.abs(s2 - corr[y * w + x]));
    }
    check(e1 < 1e-9, "convolveFFT = circular convolution");
    check(e2 < 1e-9, "correlateFFT = circular correlation");
    // correlation peak finds a circular shift
    const sh = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sh[((y + 1) % h) * w + (x + 3) % w] = A[y * w + x];
    const c = F.correlateFFT(A, sh, w, h);
    let best = 0; for (let k = 1; k < c.length; k++) if (c[k] > c[best]) best = k;
    check(best === 1 * w + 3, "correlation peak at the shift (3, 1)");
  }
  // shifts, padding
  {
    const w = 5, h = 4, a = Float64Array.from({ length: w * h }, (_, i) => i);
    check(maxDiff(F.ifftshift(F.fftshift(a, w, h), w, h), a) === 0, "ifftshift ∘ fftshift = id (odd width)");
    const s = F.fftshift(a, w, h);
    check(s[Math.floor(h / 2) * w + Math.floor(w / 2)] === 0, "fftshift moves index 0 to the centre");
    check(F.nextPow2(5) === 8 && F.nextPow2(8) === 8 && F.nextPow2(1) === 1, "nextPow2");
    const p = F.padTo(a, w, h, 8, 8);
    check(p.length === 64 && p[9] === a[6] && p[7] === 0, "padTo");
    check(maxDiff(F.crop(p, 8, 0, 0, w, h), a) === 0, "crop inverts padTo");
  }
}

// ---------------------------------------------------------------------------------------
function testLinalg(): void {
  current = "linalg";
  const r = rng(31);
  const randM = (m: number, n: number) => Float64Array.from({ length: m * n }, () => r.norm());
  const sym = (n: number) => { const A = randM(n, n); const S = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i * n + j] = A[i * n + j] + A[j * n + i]; return S; };
  for (const n of [1, 2, 3, 5, 10]) {
    const A = sym(n), { values, vectors } = LA.jacobiEigen(A, n);
    let err = 0;
    for (let k = 0; k < n; k++) {
      const v = LA.column(vectors, n, n, k), Av = LA.matVec(A, v, n, n);
      for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(Av[i] - values[k] * v[i]));
      if (k > 0) check(values[k] >= values[k - 1], "ascending eigenvalues");
    }
    check(err < 1e-9, `jacobiEigen A v = λ v (n=${n})`);
    const VtV = LA.matMul(LA.transpose(vectors, n, n), vectors, n, n, n);
    check(maxDiff(VtV, LA.identity(n)) < 1e-10, `eigenvectors orthonormal (n=${n})`);
    const d = LA.jacobiEigen(A, n, true).values;
    check(d[0] === values[n - 1], "descending option");
  }
  for (let t = 0; t < 200; t++) {
    const a = r.norm(), b = r.norm(), c = r.norm(), e = LA.eigSym2(a, b, c), J = LA.jacobiEigen([a, b, b, c], 2);
    check(close(e.l1, J.values[1], 1e-10) && close(e.l2, J.values[0], 1e-10), "eigSym2 values = Jacobi");
    const Av = [a * e.v1[0] + b * e.v1[1], b * e.v1[0] + c * e.v1[1]];
    check(Math.abs(Av[0] - e.l1 * e.v1[0]) < 1e-10 && Math.abs(Av[1] - e.l1 * e.v1[1]) < 1e-10, "eigSym2 vector");
    const d = r.norm(), g = LA.eig2(a, b, c, d);
    if (g.vectors) for (let k = 0; k < 2; k++) {
      const [x, y] = g.vectors[k], l = g.re[k];
      check(Math.abs(a * x + b * y - l * x) < 1e-8 && Math.abs(c * x + d * y - l * y) < 1e-8, "eig2 real eigenvector");
    } else check(close(g.re[0] * 2, a + d, 1e-10) && close(g.re[0] ** 2 + g.im[0] ** 2, a * d - b * c, 1e-10), "eig2 complex pair: trace and det");
    const S = LA.svd2(a, b, c, d);
    const US = [S.U[0] * S.s1, S.U[1] * S.s2, S.U[2] * S.s1, S.U[3] * S.s2];
    const rec = LA.matMul(US, LA.transpose(S.V, 2, 2), 2, 2, 2);
    check(maxDiff(rec, [a, b, c, d]) < 1e-10 && S.s1 >= S.s2 && S.s2 >= 0, `svd2 reconstructs (${maxDiff(rec, [a, b, c, d])})`);
    check(maxDiff(LA.matMul(LA.transpose(S.U, 2, 2), S.U, 2, 2, 2), LA.identity(2)) < 1e-12, "svd2 U orthogonal");
    const gs = LA.svd([a, b, c, d], 2, 2).S;
    check(close(gs[0], S.s1, 1e-10) && close(gs[1], S.s2, 1e-8), "svd2 = general svd");
  }
  for (const [m, n] of [[3, 3], [6, 3], [3, 6], [8, 5], [1, 4], [4, 1]]) {
    const A = randM(m, n), { U, S, V } = LA.svd(A, m, n);
    const US = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let k = 0; k < n; k++) US[i * n + k] = U[i * n + k] * S[k];
    check(maxDiff(LA.matMul(US, LA.transpose(V, n, n), m, n, n), A) < 1e-10, `svd reconstructs (${m}x${n})`);
    check(maxDiff(LA.matMul(LA.transpose(V, n, n), V, n, n, n), LA.identity(n)) < 1e-10, `svd V orthogonal (${m}x${n})`);
    for (let k = 1; k < n; k++) check(S[k] <= S[k - 1], "singular values descending");
    // singular values² = eigenvalues of AᵀA
    const ev = LA.jacobiEigen(LA.matMul(LA.transpose(A, m, n), A, n, m, n), n, true).values;
    check(maxDiff(Array.from(S, s => s * s), ev) < 1e-9, `svd² = eig(AᵀA) (${m}x${n})`);
    // pseudo-inverse: Penrose conditions
    const P = LA.pinv(A, m, n);
    const APA = LA.matMul(LA.matMul(A, P, m, n, m), A, m, m, n);
    check(maxDiff(APA, A) < 1e-9, `A A⁺ A = A (${m}x${n})`);
    const PAP = LA.matMul(LA.matMul(P, A, n, m, n), P, n, n, m);
    check(maxDiff(PAP, P) < 1e-9, `A⁺ A A⁺ = A⁺ (${m}x${n})`);
  }
  // least squares vs normal equations; null space; rank; cond
  {
    const m = 10, n = 3, A = randM(m, n), b = randM(m, 1);
    const x = LA.lstsq(A, b, m, n);
    const AtA = LA.matMul(LA.transpose(A, m, n), A, n, m, n), Atb = LA.matTVec(A, b, m, n);
    check(maxDiff(x, LA.solve(AtA, Atb, n)!) < 1e-9, "lstsq = normal equations");
    const B = new Float64Array([1, 2, 3, 2, 4, 6]);   // rank 1, 2 x 3
    check(LA.rank(B, 2, 3) === 1, "rank of a rank-1 matrix");
    const N = LA.nullSpace(B, 2, 3);
    check(N.length === 2 && N.every(v => maxDiff(LA.matVec(B, v, 2, 3), [0, 0]) < 1e-12), "null space vectors");
    check(LA.cond(B, 2, 3) === Infinity, "cond of singular is Infinity");
    check(close(LA.cond([2, 0, 0, 0.5], 2, 2), 4, 1e-12), "cond diag(2, .5) = 4");
  }
  for (const n of [1, 2, 3, 4, 7]) {
    const A = randM(n, n), b = randM(n, 1), x = LA.solve(A, b, n)!;
    check(maxDiff(LA.matVec(A, x, n, n), b) < 1e-9, `solve (n=${n})`);
    const Ai = LA.inverse(A, n)!;
    check(maxDiff(LA.matMul(A, Ai, n, n, n), LA.identity(n)) < 1e-9, `inverse (n=${n})`);
    const ev = LA.svd(A, n, n).S; let pd = 1; for (const s of ev) pd *= s;
    check(close(Math.abs(LA.det(A, n)), pd, 1e-9), `|det| = product of singular values (n=${n})`);
    const SPD = LA.matMul(LA.transpose(A, n, n), A, n, n, n); for (let i = 0; i < n; i++) SPD[i * n + i] += 0.1;
    const L = LA.cholesky(SPD, n)!;
    check(maxDiff(LA.matMul(L, LA.transpose(L, n, n), n, n, n), SPD) < 1e-9, `cholesky L Lᵀ = A (n=${n})`);
    check(maxDiff(LA.choleskySolve(L, b, n), LA.solve(SPD, b, n)!) < 1e-8, `choleskySolve (n=${n})`);
    check(close(LA.choleskyLogDet(L, n), Math.log(LA.det(SPD, n)), 1e-9), "cholesky log det");
  }
  check(LA.cholesky([1, 2, 2, 1], 2) === null, "cholesky rejects an indefinite matrix");
  check(LA.solve([1, 2, 2, 4], [1, 1], 2) === null, "solve reports singular");
  {
    const A = randM(3, 3), I3 = LA.inv3(A)!;
    check(maxDiff(I3, LA.inverse(A, 3)!) < 1e-9 && close(LA.det3(A), LA.det(A, 3), 1e-10), "inv3/det3 = general");
    const a = [1, 0, 0], b = [0, 1, 0];
    check(maxDiff(LA.cross3(a, b), [0, 0, 1]) === 0, "cross3");
    const X = new Float64Array([1, 2, 3, 4, 5, 9]), cv = LA.covariance(X, 3, 2);
    check(maxDiff(cv.mean, [3, 5]) < 1e-12 && close(cv.cov[0], 4) && close(cv.cov[1], 7) && close(cv.cov[3], 13), "covariance");
  }
}

// ---------------------------------------------------------------------------------------
function testImageops(): void {
  current = "imageops";
  const r = rng(41);
  // border modes
  const n = 4;
  check([-1, -2, 4, 5].map(i => IO.borderIndex(i, n, "replicate")).join() === "0,0,3,3", "replicate");
  check([-1, -2, 4, 5].map(i => IO.borderIndex(i, n, "mirror")).join() === "1,2,2,1", "mirror");
  check([-1, -2, 4, 5].map(i => IO.borderIndex(i, n, "wrap")).join() === "3,2,0,1", "wrap");
  check(IO.borderIndex(-1, n, "zero") === -1, "zero");
  // correlate2d vs brute force for all borders, random kernels; separable = 2D outer product
  for (const border of ["zero", "replicate", "mirror", "wrap"] as IO.Border[]) {
    const img = randImg(r, 7, 5), kw = 5, kh = 3, k = Float64Array.from({ length: kw * kh }, () => r.norm());
    const out = IO.correlate2d(img, k, kw, kh, border);
    let err = 0;
    for (let y = 0; y < 5; y++) for (let x = 0; x < 7; x++) {
      let s = 0;
      for (let j = 0; j < kh; j++) for (let i = 0; i < kw; i++) s += k[j * kw + i] * IO.pixel(img, x + i - 2, y + j - 1, border);
      err = Math.max(err, Math.abs(s - out.data[y * 7 + x]));
    }
    check(err < 1e-5, `correlate2d = brute force (${border})`);
    const kx = [r.norm(), r.norm(), r.norm(), r.norm(), r.norm()], ky = [r.norm(), r.norm(), r.norm()];
    const outer = new Float64Array(15); for (let j = 0; j < 3; j++) for (let i = 0; i < 5; i++) outer[j * 5 + i] = ky[j] * kx[i];
    check(maxDiff(IO.separable(img, kx, ky, border).data, IO.correlate2d(img, outer, 5, 3, border).data) < 1e-5, `separable = 2D (${border})`);
    // convolution = correlation with flipped kernel, check against definition at one pixel
    const cv = IO.convolve2d(img, k, kw, kh, border);
    let s = 0; for (let j = 0; j < kh; j++) for (let i = 0; i < kw; i++) s += k[j * kw + i] * IO.pixel(img, 3 - (i - 2), 2 - (j - 1), border);
    check(Math.abs(s - cv.data[2 * 7 + 3]) < 1e-5, `convolve2d definition (${border})`);
  }
  // Gaussian kernels: moments
  for (const s of [0.8, 1.5, 3]) {
    const g0 = IO.gaussianKernel(s, 0), g1 = IO.gaussianKernel(s, 1), g2 = IO.gaussianKernel(s, 2), R = g0.length >> 1;
    let m0 = 0, m2 = 0, d1 = 0, s2 = 0, s20 = 0;
    for (let i = -R; i <= R; i++) { m0 += g0[i + R]; m2 += g0[i + R] * i * i; d1 += g1[i + R] * i; s2 += g2[i + R] * i * i; s20 += g2[i + R]; }
    check(close(m0, 1, 1e-12) && Math.abs(Math.sqrt(m2) - s) < 0.02 * s + 0.05, `gaussian σ=${s}: sum 1, sd ≈ σ (${Math.sqrt(m2)})`);
    check(close(d1, 1, 1e-12), "first derivative kernel: ramp → 1");
    check(close(s2, 2, 1e-12) && Math.abs(s20) < 1e-12, "second derivative kernel: x² → 2, sum 0");
  }
  // derivatives of polynomials (interior, mirror border avoided by size)
  {
    const w = 40, h = 40, f = IO.fromFunction(w, h, (x, y) => 0.5 * x * x + 3 * x * y - 2 * y + 0.25 * y * y);
    const x0 = 20, y0 = 20, i = y0 * w + x0, s = 1.5;
    const lx = IO.gaussianDerivative(f, s, 1, 0).data[i], ly = IO.gaussianDerivative(f, s, 0, 1).data[i];
    const lxx = IO.gaussianDerivative(f, s, 2, 0).data[i], lxy = IO.gaussianDerivative(f, s, 1, 1).data[i], lyy = IO.gaussianDerivative(f, s, 0, 2).data[i];
    check(Math.abs(lx - (x0 + 3 * y0)) < 1e-2 && Math.abs(ly - (3 * x0 - 2 + 0.5 * y0)) < 1e-2, `Gaussian gradient of a quadratic (${lx}, ${ly})`);
    check(Math.abs(lxx - 1) < 1e-3 && Math.abs(lxy - 3) < 1e-3 && Math.abs(lyy - 0.5) < 1e-3, `Gaussian Hessian of a quadratic (${lxx}, ${lxy}, ${lyy})`);
    check(Math.abs(IO.logFilter(f, s).data[i] - 1.5) < 1e-3, "LoG of a quadratic = trace of Hessian");
    const sb = IO.sobel(f);
    check(Math.abs(sb.gx.data[i] / 8 - (x0 + 3 * y0)) < 1e-3, "Sobel x / 8 = slope");
    check(Math.abs(IO.laplacian(f).data[i] - 1.5) < 1e-3, "3×3 Laplacian of a quadratic");
  }
  // Gaussian blur: mass conserved with wrap; σ adds in quadrature (interior)
  {
    const img = randImg(r, 32, 32, 100), b = IO.gaussianBlur(img, 2, "wrap");
    check(close(IO.stats(b).mean, IO.stats(img).mean, 1e-5), "blur keeps the mean (wrap)");
    const d = IO.createGrey(64, 64); d.data[32 * 64 + 32] = 1;
    const two = IO.gaussianBlur(IO.gaussianBlur(d, 1.5), 2), one = IO.gaussianBlur(d, 2.5);
    check(maxDiff(two.data, one.data) < 2e-3 * IO.stats(one).max, "G(1.5) then G(2) ≈ G(2.5)");
  }
  // DoG ≈ (k−1)σ² LoG on a smooth image
  {
    const disc = SY.discs(64, 64, [{ x: 32, y: 32, r: 8 }]).image;
    const s = 3, k = 1.01, dog = IO.dogFilter(disc, s, k), lg = IO.logFilter(disc, s);
    const i = 32 * 64 + 32;
    check(Math.abs(dog.data[i] / ((k - 1) * s * s * lg.data[i]) - 1) < 0.04, `DoG ≈ (k−1)σ²·LoG (${dog.data[i] / ((k - 1) * s * s * lg.data[i])})`);
    // scale-normalised LoG peaks near r/√2
    let best = 0, bs = 0;
    for (let sg = 2; sg <= 10; sg += 0.25) { const v = -IO.logFilter(disc, sg, true, "mirror").data[i]; if (v > best) { best = v; bs = sg; } }
    check(Math.abs(bs - 8 / Math.SQRT2) <= 0.6, `normalised LoG of a disc r=8 peaks at σ=${bs}`);
  }
  // box filter vs brute force (all borders)
  for (const border of ["zero", "replicate", "mirror", "wrap"] as IO.Border[]) {
    const img = randImg(r, 9, 6), R = 2, b = IO.boxFilter(img, R, border);
    const k = new Float64Array(25).fill(1 / 25);
    check(maxDiff(b.data, IO.correlate2d(img, k, 5, 5, border).data) < 1e-5, `boxFilter = 5×5 mean (${border})`);
  }
  // sampling
  {
    const img = IO.fromFunction(6, 5, (x, y) => 2 * x - 3 * y + 1);
    check(close(IO.sampleBilinear(img, 2.3, 1.7), 2 * 2.3 - 3 * 1.7 + 1, 1e-6), "bilinear exact on a plane");
    check(IO.sampleNearest(img, 2.4, 1.6) === img.data[2 * 6 + 2], "nearest");
    const sq = randImg(r, 7, 7), q = IO.rotate(sq, Math.PI / 2, { interp: "bilinear" });
    let err = 0;
    for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) err = Math.max(err, Math.abs(q.data[y * 7 + x] - sq.data[x * 7 + (6 - y)]));
    check(err < 1e-5, "rotate +90° (counter-clockwise on screen) is an exact pixel permutation");
    const t = IO.translate(sq, 2, 1, "nearest");
    check(t.data[3 * 7 + 4] === sq.data[2 * 7 + 2] && t.data[0] === 0, "translate");
    const big = randImg(r, 16, 12), small = IO.resize(big, 8, 6, "area");
    check(close(small.data[0], (big.data[0] + big.data[1] + big.data[16] + big.data[17]) / 4, 1e-6), "area resize = 2×2 mean");
    check(maxDiff(IO.resize(big, 16, 12).data, big.data) < 1e-6, "resize to the same size is the identity");
    check(maxDiff(IO.boxDown2(big).data, small.data) < 1e-6, "boxDown2 = area resize by 2");
  }
  // pyramids
  {
    const img = randImg(r, 33, 20, 255), g = IO.gaussianPyramid(img, 4);
    check(g.length === 4 && g[1].width === 17 && g[1].height === 10 && g[3].width === 5, "pyramid sizes");
    const L = IO.laplacianPyramid(img, 4);
    check(maxDiff(IO.collapsePyramid(L).data, img.data) < 1e-3, "Laplacian pyramid collapses to the image");
    const up = IO.pyrUp(g[1], 33, 20);
    check(up.data[0] === g[1].data[0] && up.data[2] === g[1].data[1], "pyrUp places small pixel i at 2i");
  }
  // display bytes
  {
    const img = IO.fromArray(4, 1, [-10, 0, 127.5, 300]);
    check(Array.from(IO.toBytes(img)).join() === "0,0,128,255", "clip display");
    check(Array.from(IO.toBytes(img, { mode: "mid" })).join() === [Math.round(127.5 - 127.5 * 10 / 300), 128, Math.round(127.5 + 127.5 * 127.5 / 300), 255].join(), "mid display: 0 is mid-grey");
    check(Array.from(IO.toBytes(img, { mode: "abs" })).join() === [Math.round(255 * 10 / 300), 0, Math.round(255 * 127.5 / 300), 255].join(), "abs display");
    check(Array.from(IO.toBytes(img, { mode: "auto" }))[0] === 0 && IO.toBytes(img, { mode: "auto" })[3] === 255, "auto display");
    const rgba = IO.toRGBA(img, {}, Uint8Array.from({ length: 768 }, (_, i) => (i % 3 === 0 ? 255 - Math.floor(i / 3) : 7)));
    check(rgba[0] === 255 && rgba[1] === 7 && rgba[3] === 255, "lookup table display");
    const fromId = IO.fromImageData({ width: 1, height: 1, data: [100, 100, 100, 255] });
    check(close(fromId.data[0], 100, 1e-6), "fromImageData grey");
  }
}

// ---------------------------------------------------------------------------------------
function testSynth(): void {
  current = "synth";
  const a = SY.nuclei({ seed: 3 }), b = SY.nuclei({ seed: 3 });
  check(maxDiff(a.image.data, b.image.data) === 0, "nuclei reproducible");
  check(a.count >= 15, `nuclei: ${a.count} placed`);
  const sizes = LB.componentSizes(a.labels, a.count);
  check(sizes.slice(1).every(s => s > 100), "every nucleus has pixels");
  // every labelled object is roughly its ellipse area
  a.ellipses.forEach((e, k) => check(Math.abs(sizes[k + 1] / (Math.PI * e.a * e.b) - 1) < 0.35, `nucleus ${k + 1} area`));
  // touching pairs are adjacent in the label image
  for (const [p, q] of a.touchingPairs) {
    let adj = false;
    for (let y = 0; y < 256 && !adj; y++) for (let x = 0; x < 255; x++) {
      const u = a.labels[y * 256 + x], v = a.labels[y * 256 + x + 1], d = y < 255 ? a.labels[(y + 1) * 256 + x] : 0;
      if ((u === p && (v === q || d === q)) || (u === q && (v === p || d === p))) { adj = true; break; }
    }
    check(adj, `touching pair ${p}-${q} adjacent`);
  }
  check(a.touchingPairs.length >= 2, "some nuclei touch");
  const mIn = mean(a.image.data, a.labels, true), mOut = mean(a.image.data, a.labels, false);
  check(mIn > 150 && mOut < 70, `nuclei contrast ${mIn} vs ${mOut}`);
  const sp = SY.spots({ seed: 2 });
  check(sp.spots.length === 40, "40 spots");
  for (const s of sp.spots.slice(0, 5)) {
    const i = Math.round(s.y) * 256 + Math.round(s.x);
    check(sp.image.data[i] - sp.background.data[i] > s.amplitude * 0.5 - 20, "spot visible above the background");
  }
  const fl = SY.filaments({ seed: 4, curvature: 0 });
  check(fl.curves.length === 12 && fl.centreline.some(v => v === 1), "filaments generated");
  let on = 0, cnt = 0; for (let i = 0; i < fl.mask.length; i++) if (fl.centreline[i]) { on += fl.image.data[i]; cnt++; }
  check(on / cnt > 90, "filaments bright on their centreline");
  const sl = SY.sheppLogan(128);
  check(close(sl.data[64 * 128 + 64], 0.2, 1e-6) || close(sl.data[64 * 128 + 64], 0.3, 1e-6), `Shepp–Logan centre ${sl.data[64 * 128 + 64]}`);
  check(sl.data[0] === 0 && Math.abs(IO.stats(sl).max - 1) < 1e-6, "Shepp–Logan range 0..1");
  const cb = SY.checkerboard(8, 8, 2);
  check(cb.data[0] === 0 && cb.data[2] === 255 && cb.data[2 * 8 + 2] === 0, "checkerboard");
  const st = SY.siemensStar(64);
  check(IO.stats(st).min === 0 && IO.stats(st).max === 255, "Siemens star range");
  const noisy = SY.addPoissonNoise(IO.createGrey(100, 100, 20), 1, 5);
  const s2 = IO.stats(noisy);
  check(Math.abs(s2.mean - 20) < 0.5 && Math.abs(s2.sd * s2.sd - 20) < 2, "Poisson noise variance = mean");
}
function mean(d: ArrayLike<number>, lab: ArrayLike<number>, inside: boolean): number {
  let s = 0, n = 0;
  for (let i = 0; i < d.length; i++) if ((lab[i] > 0) === inside) { s += d[i]; n++; }
  return s / n;
}

// ---------------------------------------------------------------------------------------
function testHistogram(): void {
  current = "histogram";
  const r = rng(51);
  // float histogram against direct counting
  const a = Float64Array.from({ length: 5000 }, () => r.norm() * 3 - 1);
  const h = H.histogram(a, { bins: 20, min: -5, max: 5 });
  let inRange = 0;
  for (const v of a) if (v >= -5 && v <= 5) inRange++;
  check(h.total === inRange && h.below + h.above + h.total === a.length, "counts add up");
  for (let b = 0; b < 20; b++) {
    const lo = -5 + b * 0.5, hi = lo + 0.5;
    const c = a.filter(v => v >= lo && (b === 19 ? v <= hi : v < hi)).length;
    check(h.counts[b] === c, `bin ${b}`);
  }
  check(H.binOf(h, 5) === 19 && H.binOf(h, -5) === 0 && H.binOf(h, 0.1) === 10, "binOf (max goes to the last bin)");
  check(close(H.normalise(h).reduce((s, v) => s + v, 0), 1), "normalised sums to 1");
  const c = H.cumulative(h);
  check(close(c[19], 1) && c.every((v, i) => i === 0 || v >= c[i - 1]), "cumulative");
  check(close(H.density(h).reduce((s, v) => s + v * 0.5, 0), 1), "density integrates to 1");
  // integer histogram with negative and 16-bit values
  const ints = Int32Array.from({ length: 3000 }, () => Math.floor(r.unif() * 21) - 10);
  const hi = H.intHistogram(ints);
  check(hi.counts.length === 21 && H.binCentre(hi, 0) === -10 && H.binCentre(hi, 20) === 10, "int histogram range −10..10");
  for (let v = -10; v <= 10; v++) check(hi.counts[v + 10] === ints.filter(x => x === v).length, `int bin ${v}`);
  const u16 = Uint16Array.from({ length: 1000 }, () => Math.floor(r.unif() * 65536));
  const h16 = H.intHistogram(u16, 0, 65535);
  check(h16.counts.length === 65536 && h16.total === 1000 && h16.counts[u16[0]] >= 1, "16-bit histogram");
  // stats and percentiles
  const st = H.histStats(hi);
  const m = ints.reduce((s, v) => s + v, 0) / ints.length;
  check(close(st.mean, m, 1e-12), "histogram mean = data mean (integer bins)");
  const sorted = Float64Array.from(a).sort();
  check(close(H.percentile(a, 0.5), (sorted[2499] + sorted[2500]) / 2, 1e-12), "percentile median");
  check(Math.abs(H.histPercentile(H.histogram(a, { bins: 1000 }), 0.5) - H.percentile(a, 0.5)) < 0.05, "histogram percentile");
  // masked + weighted
  const mk = Uint8Array.from(a, (_, i) => i % 2);
  check(H.histogram(a, { bins: 10, min: -20, max: 20, mask: mk }).total === 2500, "mask");
  const wts = Float64Array.from(a, () => 2);
  check(H.histogram(a, { bins: 10, min: -20, max: 20, weights: wts }).total === 10000, "weights");
  // 2D
  const x = [0, 350, 359.9, 10, 180], y = [0, 0, 1, 1, 0.5];
  const H2 = H.histogram2d(x, y, { binsX: 36, binsY: 2, minX: 0, maxX: 360, minY: 0, maxY: 1, circularX: true });
  check(H2.total === 5 && H2.counts[0] === 1 && H2.counts[35] === 1 && H2.counts[36 + 35] === 1 && H2.counts[36 + 1] === 1 && H2.counts[36 + 18] === 1, "2D histogram with circular hue");
  const wrap = H.histogram2d([370], [0], { binsX: 36, binsY: 1, minX: 0, maxX: 360, minY: 0, maxY: 1, circularX: true });
  check(wrap.counts[1] === 1, "circular axis wraps 370° to 10°");
}

// ---------------------------------------------------------------------------------------
function testMetrics(): void {
  current = "metrics";
  const r = rng(61);
  const a = Float64Array.from({ length: 400 }, () => 255 * r.unif()), b = a.map(v => v + 5);
  check(close(ME.mse(a, b), 25) && close(ME.rmse(a, b), 5) && close(ME.mae(a, b), 5), "mse/rmse/mae");
  check(close(ME.psnr(a, b, 255), 10 * Math.log10(255 * 255 / 25)), "psnr with peak 255");
  check(ME.psnr(a, a, 1) === Infinity, "psnr identical = ∞");
  // SSIM: identical → 1; symmetric; decreases with noise; equals the window formula at one pixel
  const w = 32, h = 32, img = Float64Array.from({ length: w * h }, (_, i) => 128 + 60 * Math.sin(i % w / 3) + 20 * r.norm());
  const noisy = img.map(v => v + 30 * r.norm()), lessNoisy = img.map(v => v + 5 * r.norm());
  check(close(ME.ssim(img, img, w, h).mean, 1, 1e-6), "SSIM(a, a) = 1");
  check(close(ME.ssim(img, noisy, w, h).mean, ME.ssim(noisy, img, w, h).mean, 1e-9), "SSIM symmetric");
  check(ME.ssim(img, noisy, w, h).mean < ME.ssim(img, lessNoisy, w, h).mean, "SSIM decreases with noise");
  {
    const s = ME.ssim(img, noisy, w, h), x0 = 15, y0 = 12, g = IO.gaussianKernel(1.5, 0, 5);
    let ma = 0, mb = 0, saa = 0, sbb = 0, sab = 0;
    for (let j = -5; j <= 5; j++) for (let i = -5; i <= 5; i++) {
      const wt = g[i + 5] * g[j + 5], p = img[(y0 + j) * w + x0 + i], q = noisy[(y0 + j) * w + x0 + i];
      ma += wt * p; mb += wt * q; saa += wt * p * p; sbb += wt * q * q; sab += wt * p * q;
    }
    const C1 = (0.01 * 255) ** 2, C2 = (0.03 * 255) ** 2;
    const ref = ((2 * ma * mb + C1) * (2 * (sab - ma * mb) + C2)) / ((ma * ma + mb * mb + C1) * (saa - ma * ma + sbb - mb * mb + C2));
    check(Math.abs(s.map.data[y0 * w + x0] - ref) < 1e-4, `SSIM map = window formula (${s.map.data[y0 * w + x0]} vs ${ref})`);
  }
  // pixel scores vs set definitions
  for (let t = 0; t < 50; t++) {
    const p = randMask(r, 100, 0.4), q = randMask(r, 100, 0.4);
    let inter = 0, sp = 0, sq = 0, uni = 0;
    for (let i = 0; i < 100; i++) { inter += p[i] & q[i]; sp += p[i]; sq += q[i]; uni += p[i] | q[i]; }
    const c = ME.confusion(p, q);
    check(c.tp + c.fp + c.fn + c.tn === 100, "confusion counts");
    check(close(ME.dice(c), 2 * inter / (sp + sq)), "Dice = 2|A∩B|/(|A|+|B|)");
    check(close(ME.jaccard(c), inter / uni), "Jaccard = |A∩B|/|A∪B|");
    check(close(ME.jaccard(c), ME.dice(c) / (2 - ME.dice(c))), "J = D/(2−D)");
    check(close(ME.f1(c), 2 * ME.precision(c) * ME.recall(c) / (ME.precision(c) + ME.recall(c))), "F1 = harmonic mean");
  }
  check(ME.dice(ME.confusion([0, 0], [0, 0])) === 1, "Dice of two empty masks = 1");
  const M = ME.confusionMatrix([0, 1, 2, 2, 1], [0, 1, 1, 2, 2], 3);
  check(Array.from(M).join() === "1,0,0,0,1,1,0,1,1", "confusion matrix [true][pred]");
  const cc = ME.classCounts(M, 3, 1);
  check(cc.tp === 1 && cc.fp === 1 && cc.fn === 1 && cc.tn === 2, "one-vs-rest counts");
}

// ---------------------------------------------------------------------------------------
function floodLabels(mask: ArrayLike<number>, dims: number[], offs: number[][]): { count: number; same: (i: number, j: number) => boolean; lab: Int32Array } {
  const n = dims.reduce((a, b) => a * b, 1), lab = new Int32Array(n);
  let count = 0;
  const coord = (i: number) => { const c: number[] = []; for (const d of dims) { c.push(i % d); i = Math.floor(i / d); } return c; };
  const index = (c: number[]) => { let i = 0, s = 1; for (let k = 0; k < dims.length; k++) { i += c[k] * s; s *= dims[k]; } return i; };
  for (let s = 0; s < n; s++) {
    if (!mask[s] || lab[s]) continue;
    lab[s] = ++count;
    const st = [s];
    while (st.length) {
      const p = st.pop()!, c = coord(p);
      for (const o of offs) {
        const q = c.map((v, k) => v + o[k]);
        if (q.some((v, k) => v < 0 || v >= dims[k])) continue;
        const qi = index(q);
        if (mask[qi] && !lab[qi]) { lab[qi] = count; st.push(qi); }
      }
    }
  }
  return { count, lab, same: (i, j) => lab[i] === lab[j] };
}

function testLabel(): void {
  current = "label";
  const r = rng(71);
  for (let t = 0; t < 200; t++) {
    const w = 1 + randInt(r, 12), h = 1 + randInt(r, 10), m = randMask(r, w * h, 0.3 + 0.4 * r.unif());
    for (const conn of [4, 8] as LB.Conn2[]) {
      const L = LB.label2d(m, w, h, conn), ref = floodLabels(m, [w, h], LB.offsets2d(conn));
      check(L.count === ref.count, `2D count (${conn})`);
      // labels are a relabelling of the reference, in raster order
      check(Array.from(L.data).every((l, i) => l === ref.lab[i]), `2D labels = flood fill in raster order (${conn})`);
    }
  }
  for (let t = 0; t < 60; t++) {
    const nx = 1 + randInt(r, 6), ny = 1 + randInt(r, 6), nz = 1 + randInt(r, 5), m = randMask(r, nx * ny * nz, 0.35);
    for (const conn of [6, 18, 26] as LB.Conn3[]) {
      const L = LB.label3d(m, nx, ny, nz, conn), ref = floodLabels(m, [nx, ny, nz], LB.offsets3d(conn));
      check(L.count === ref.count && Array.from(L.data).every((l, i) => l === ref.lab[i]), `3D labels (${conn})`);
    }
  }
  check(LB.offsets3d(6).length === 6 && LB.offsets3d(18).length === 18 && LB.offsets3d(26).length === 26, "3D offset tables");
  const uf = new LB.UnionFind(10);
  uf.union(1, 2); uf.union(3, 4); uf.union(2, 4);
  check(uf.same(1, 3) && !uf.same(1, 5) && uf.size(4) === 4, "union–find");
  const L = LB.label2d([1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 4, 3, 4);
  const big = LB.removeSmall(L, 3);
  check(L.count === 2 && big.count === 1 && big.data[7] === 1 && big.data[0] === 0, "removeSmall");
  const bb = LB.boundingBoxes(L), ce = LB.centroids(L);
  check(bb[2].x0 === 0 && bb[2].x1 === 3 && bb[2].y0 === 1 && bb[2].y1 === 2 && close(ce[1].x, 0.5), "bounding boxes and centroids");
}

// ---------------------------------------------------------------------------------------
function testDistance(): void {
  current = "distance";
  const r = rng(81);
  for (let t = 0; t < 150; t++) {
    const w = 1 + randInt(r, 14), h = 1 + randInt(r, 12), m = randMask(r, w * h, 0.5 + 0.45 * r.unif());
    const sx = t % 3 === 0 ? 1 + r.unif() : 1, sy = t % 3 === 0 ? 1 + r.unif() : 1;
    const { dist, nearest } = DT.edtWithNearest(m, w, h, { spacing: [sx, sy] });
    const cb = DT.chamfer(m, w, h, "cityblock"), ch = DT.chamfer(m, w, h, "chessboard"), c34 = DT.chamfer(m, w, h, "3-4");
    let ok = true, okN = true, okC = true, okH = true, ok34 = true;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let best = Infinity, bc = Infinity, bh = Infinity;
      for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) if (!m[v * w + u]) {
        best = Math.min(best, Math.hypot(sx * (x - u), sy * (y - v)));
        bc = Math.min(bc, Math.abs(x - u) + Math.abs(y - v));
        bh = Math.min(bh, Math.max(Math.abs(x - u), Math.abs(y - v)));
      }
      const i = y * w + x;
      if (!(best === Infinity ? dist[i] === Infinity : Math.abs(dist[i] - best) < 1e-4)) ok = false;
      if (best < Infinity) {
        const q = nearest[i], u = q % w, v = (q - u) / w;
        if (m[q] || Math.abs(Math.hypot(sx * (x - u), sy * (y - v)) - best) > 1e-4) okN = false;
      }
      if (sx === 1 && sy === 1) {
        if (cb[i] !== bc) okC = false;
        if (ch[i] !== bh) okH = false;
        if (best < Infinity && !(c34[i] >= 0.942 * best - 1e-6 && c34[i] <= 1.055 * best + 1e-6)) ok34 = false;   // (dx + dy/3) vs Euclid: −5.7 % .. +5.4 %
      }
    }
    check(ok, `edt = brute force (${w}x${h}, spacing ${sx.toFixed(2)},${sy.toFixed(2)})`);
    check(okN, "nearest background pixel");
    check(okC && okH, "city-block and chessboard exact");
    check(ok34, "3-4 chamfer within bounds");
  }
  {
    const nx = 5, ny = 4, nz = 6, m = randMask(r, nx * ny * nz, 0.8), d = DT.edt3d(m, nx, ny, nz, { spacing: [1, 2, 0.5] });
    let ok = true;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      let best = Infinity;
      for (let c = 0; c < nz; c++) for (let b = 0; b < ny; b++) for (let a = 0; a < nx; a++)
        if (!m[(c * ny + b) * nx + a]) best = Math.min(best, Math.hypot(x - a, 2 * (y - b), 0.5 * (z - c)));
      if (Math.abs(d[(z * ny + y) * nx + x] - best) > 1e-4) ok = false;
    }
    check(ok, "edt3d = brute force (anisotropic)");
  }
  const sd = DT.signedDistance([0, 1, 1, 1, 0], 5, 1);
  check(Array.from(sd).join() === "1,-1,-2,-1,1", "signed distance");
}

// ---------------------------------------------------------------------------------------
function testMorph(): void {
  current = "morph";
  const r = rng(91);
  const ses = [MO.seSquare(1), MO.seDisk(2), MO.seDiamond(1), MO.seLine(5, 0.6), MO.seFromMask([1, 1, 0, 0, 1, 1], 3, 2, 0, 0)];
  for (let t = 0; t < 40; t++) {
    const w = 3 + randInt(r, 9), h = 3 + randInt(r, 8);
    const f = Float32Array.from({ length: w * h }, () => randInt(r, 10));
    for (const se of ses) {
      const e = MO.erode(f, w, h, se), d = MO.dilate(f, w, h, se);
      let ok = true;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let mn = Infinity, mx = -Infinity;
        for (let k = 0; k < se.dx.length; k++) {
          const xe = x + se.dx[k], ye = y + se.dy[k], xd = x - se.dx[k], yd = y - se.dy[k];
          if (xe >= 0 && xe < w && ye >= 0 && ye < h) mn = Math.min(mn, f[ye * w + xe]);
          if (xd >= 0 && xd < w && yd >= 0 && yd < h) mx = Math.max(mx, f[yd * w + xd]);
        }
        if (e[y * w + x] !== mn || d[y * w + x] !== mx) ok = false;
      }
      check(ok, "erosion/dilation = definition");
      // adjunction δ(f) ≤ g ⇔ f ≤ ε(g) gives: opening ≤ f ≤ closing, both idempotent
      const o = MO.open(f, w, h, se), c = MO.close(f, w, h, se);
      check(o.every((v, i) => v <= f[i]) && c.every((v, i) => v >= f[i]), "open ≤ f ≤ close");
      check(maxDiff(MO.open(o, w, h, se), o) === 0 && maxDiff(MO.close(c, w, h, se), c) === 0, "opening and closing idempotent");
      // binary = grey on 0/1
      const m = Uint8Array.from(f, v => (v > 5 ? 1 : 0));
      check(maxDiff(MO.dilateBinary(m, w, h, se), MO.dilate(m, w, h, se)) === 0, "binary dilation");
      // duality: erosion of the complement = complement of the dilation by the reflected SE
      const inv = m.map(v => 1 - v), ed = MO.erodeBinary(inv, w, h, se), dd = MO.dilateBinary(m, w, h, MO.seReflect(se));
      check(ed.every((v, i) => v === 1 - dd[i]), "duality ε_B(fᶜ) = (δ_B̌ f)ᶜ");
    }
    // median vs sort
    const md = MO.median(f, w, h, MO.seSquare(1));
    const x = 1, y = 1, nb: number[] = [];
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) nb.push(f[(y + j) * w + x + i]);
    nb.sort((p, q) => p - q);
    check(md[y * w + x] === nb[4], "median");
    // reconstruction = iterated geodesic dilation
    for (const conn of [4, 8] as LB.Conn2[]) {
      const marker = f.map(v => Math.max(0, v - randInt(r, 6)));
      const rec = MO.reconstructByDilation(marker, f, w, h, conn);
      let cur: Float32Array = Float32Array.from(marker, (v, i) => Math.min(v, f[i]));
      for (let it = 0; it < w * h; it++) { const nx = MO.geodesicDilate(cur, f, w, h, conn); if (maxDiff(nx, cur) === 0) break; cur = nx; }
      check(maxDiff(rec, cur) === 0, `reconstruction by dilation (${conn})`);
      const mk2 = f.map(v => v + randInt(r, 6));
      const rec2 = MO.reconstructByErosion(mk2, f, w, h, conn);
      let cur2: Float32Array = Float32Array.from(mk2, (v, i) => Math.max(v, f[i]));
      for (let it = 0; it < w * h; it++) { const nx = MO.geodesicErode(cur2, f, w, h, conn); if (maxDiff(nx, cur2) === 0) break; cur2 = nx; }
      check(maxDiff(rec2, cur2) === 0, `reconstruction by erosion (${conn})`);
      // regional maxima of integer images: f − rec(f − 1, f) > 0
      const rm = MO.regionalMaxima(f, w, h, conn), alt = MO.reconstructByDilation(f.map(v => v - 1), f, w, h, conn);
      check(rm.every((v, i) => v === (f[i] - alt[i] > 0 ? 1 : 0)), `regional maxima (${conn})`);
      const rmin = MO.regionalMinima(f, w, h, conn), altm = MO.reconstructByErosion(f.map(v => v + 1), f, w, h, conn);
      check(rmin.every((v, i) => v === (altm[i] - f[i] > 0 ? 1 : 0)), `regional minima (${conn})`);
      // h-minima: every regional minimum of HMIN has depth ≥ h in f; HMIN ≥ f and HMIN − f ≤ h
      const hm = MO.hMinTransform(f, w, h, 3, conn);
      check(hm.every((v, i) => v >= f[i] && v - f[i] <= 3 + 1e-6), `h-min transform bounds (${conn})`);
      const hx = MO.hMaxTransform(f, w, h, 3, conn);
      check(hx.every((v, i) => v <= f[i] && f[i] - v <= 3 + 1e-6), `h-max transform bounds (${conn})`);
      // h = 0 keeps every regional minimum
      check(MO.hMinima(f, w, h, 0.5, conn).every((v, i) => v === 0 || rmin[i] === 1), "h-minima are inside regional minima");
    }
    // minima imposition: the only regional minima are the markers
    const markers = new Uint8Array(w * h); markers[0] = 1; markers[w * h - 1] = 1;
    const imp = MO.imposeMinima(f, markers, w, h);
    const rmi = MO.regionalMinima(imp, w, h);
    check(rmi.every((v, i) => v === markers[i]), "imposed minima = markers");
  }
  // fill holes and clear border vs labelling
  for (let t = 0; t < 100; t++) {
    const w = 3 + randInt(r, 10), h = 3 + randInt(r, 10), m = randMask(r, w * h, 0.55);
    const fh = MO.fillHoles(m, w, h, 4), inv = m.map(v => 1 - v), L = floodLabels(inv, [w, h], LB.offsets2d(4));
    const border = new Set<number>();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if ((x === 0 || y === 0 || x === w - 1 || y === h - 1) && inv[y * w + x]) border.add(L.lab[y * w + x]);
    check(fh.every((v, i) => v === (m[i] || !border.has(L.lab[i]) ? 1 : 0)), "fill holes");
    const cb = MO.clearBorder(m, w, h, 8), L8 = floodLabels(m, [w, h], LB.offsets2d(8));
    const tb = new Set<number>();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if ((x === 0 || y === 0 || x === w - 1 || y === h - 1) && m[y * w + x]) tb.add(L8.lab[y * w + x]);
    check(cb.every((v, i) => v === (m[i] && !tb.has(L8.lab[i]) ? 1 : 0)), "clear border");
  }
  check(MO.seDisk(1).dx.length === 5 && MO.seDisk(2).dx.length === 13 && MO.seSquare(2).dx.length === 25, "SE sizes");
  // hit-or-miss finds isolated pixels
  const iso = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0];
  const hm = MO.hitOrMiss(iso, 4, 4, MO.seFromMask([1], 1, 1, 0, 0), MO.seFromMask([1, 1, 1, 1, 0, 1, 1, 1, 1], 3, 3));
  check(hm[5] === 1 && hm.reduce((s, v) => s + v, 0) === 1, "hit-or-miss isolated pixel");
}

// ---------------------------------------------------------------------------------------
function testPqueue(): void {
  current = "pqueue";
  const r = rng(101);
  const pq = new PriorityQueue(2);
  const items: [number, number, number][] = [];
  for (let i = 0; i < 2000; i++) { const p = randInt(r, 50); pq.push(i, p); items.push([p, i, i]); }
  items.sort((a, b) => a[0] - b[0] || a[2] - b[2]);
  let ok = true;
  for (const [p, it] of items) { if (pq.peekPriority() !== p || pq.pop() !== it) ok = false; }
  check(ok && pq.size === 0 && pq.pop() === -1, "PriorityQueue: sorted, FIFO among ties");
  // interleaved push/pop vs sorted array
  const ref: [number, number][] = [];
  let ok2 = true, ord = 0;
  for (let i = 0; i < 3000; i++) {
    if (r.unif() < 0.6 || ref.length === 0) { const p = r.unif(); pq.push(i, p); ref.push([p, ord++]); void i; }
    else { ref.sort((a, b) => a[0] - b[0] || a[1] - b[1]); const top = ref.shift()!; const pr = pq.peekPriority(); pq.pop(); if (pr !== top[0]) ok2 = false; }
  }
  check(ok2, "PriorityQueue interleaved");
  const hp = new Heap<number>((a, b) => a > b);
  const vals = Array.from({ length: 500 }, () => r.unif());
  vals.forEach(v => hp.push(v));
  const out: number[] = []; while (!hp.isEmpty()) out.push(hp.pop()!);
  check(out.every((v, i) => i === 0 || v <= out[i - 1]) && out.length === 500, "Heap max order");
  const bq = new BucketQueue(16);
  const bref: [number, number][] = [];
  let ok3 = true, seq = 0;
  for (let i = 0; i < 3000; i++) {
    if (r.unif() < 0.55 || bref.length === 0) { const p = randInt(r, 16); bq.push(i, p); bref.push([p, seq++ * 1e4 + i]); }
    else {
      bref.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const top = bref.shift()!;
      const lvl = bq.level, it = bq.pop();
      if (lvl !== top[0] || it !== top[1] % 1e4) ok3 = false;
    }
  }
  check(ok3, "BucketQueue: lowest level first, FIFO within a level");
  const q = new Fifo(1);
  for (let i = 0; i < 100; i++) q.push(i);
  let ok4 = true;
  for (let i = 0; i < 50; i++) if (q.shift() !== i) ok4 = false;
  for (let i = 100; i < 300; i++) q.push(i);
  for (let i = 50; i < 300; i++) if (q.shift() !== i) ok4 = false;
  check(ok4 && q.isEmpty() && q.shift() === -1, "Fifo order across growth and wrap-around");
}

// ---------------------------------------------------------------------------------------
function testPolar(): void {
  current = "polar";
  const n = 101, c = 50;
  // radially symmetric image: every column equal to f(r)
  const f = (rr: number) => 100 + 50 * Math.cos(rr / 6);
  const img = IO.fromFunction(n, n, (x, y) => f(Math.hypot(x - c, y - c)));
  const o: PO.PolarOptions = { cx: c, cy: c, rMax: 45, nR: 46, nTheta: 64 };
  const P = PO.toPolar(img, o);
  check(P.width === 64 && P.height === 46, "polar size: angles × radii");
  let err = 0;
  for (let i = 0; i < 46; i++) for (let j = 0; j < 64; j++) err = Math.max(err, Math.abs(P.data[i * 64 + j] - f(PO.radiusAt(o, i))));
  check(err < 1.5, `radially symmetric image → constant columns (err ${err.toFixed(3)})`);
  const prof = PO.radialProfile(P);
  check(Math.abs(prof[30] - f(30)) < 1, "radial profile");
  // angle convention: column of θ = 90° looks straight down (y down)
  const spot = IO.createGrey(n, n); spot.data[(c + 20) * n + c] = 1;
  const Ps = PO.toPolar(spot, o);
  check(Ps.data[20 * 64 + 16] === 1, "θ = 90° points down (+y)");
  // round trip on a smooth image
  const sm = IO.fromFunction(n, n, (x, y) => 100 + 30 * Math.sin(x / 9) * Math.cos(y / 7));
  const oo: PO.PolarOptions = { cx: c, cy: c, rMax: 40, nR: 161, nTheta: 720 };
  const back = PO.fromPolar(PO.toPolar(sm, oo), n, n, oo, NaN);
  let e2 = 0;
  for (let i = 0; i < n * n; i++) if (back.data[i] === back.data[i]) e2 = Math.max(e2, Math.abs(back.data[i] - sm.data[i]));
  check(e2 < 0.5, `polar round trip (err ${e2.toFixed(3)})`);
  // log-polar: scaling by s shifts rows by log(s)/Δ; rotation shifts columns
  const blob = (s: number, rot: number) => IO.fromFunction(n, n, (x, y) => {
    const dx = (x - c) / s, dy = (y - c) / s, u = Math.cos(rot) * dx + Math.sin(rot) * dy, v = -Math.sin(rot) * dx + Math.cos(rot) * dy;
    return 100 * Math.exp(-((u - 8) ** 2 + v * v) / 18) + 60 * Math.exp(-((u + 3) ** 2 + (v - 6) ** 2) / 10);
  });
  const lo: PO.PolarOptions = { cx: c, cy: c, rMin: 2, rMax: 48, nR: 128, nTheta: 128, log: true };
  const A = PO.toPolar(blob(1, 0), lo), B = PO.toPolar(blob(1.5, 0), lo), C = PO.toPolar(blob(1, 2 * Math.PI * 16 / 128), lo);
  const shiftErr = (X: IO.Grey, Y: IO.Grey, di: number, dj: number) => {
    let s = 0, m = 0;
    for (let i = 10; i < 100; i++) for (let j = 0; j < 128; j++) { s += Math.abs(Y.data[(i + di) * 128 + (j + dj) % 128] - X.data[i * 128 + j]); m++; }
    return s / m;
  };
  const step = Math.log(48 / 2) / 127, di = Math.round(Math.log(1.5) / step);
  check(shiftErr(A, B, di, 0) < 0.6 && shiftErr(A, B, di, 0) < shiftErr(A, B, 0, 0) / 5, `log-polar: scale 1.5 = row shift ${di}`);
  check(shiftErr(A, C, 0, 16) < 0.6 && shiftErr(A, C, 0, 16) < shiftErr(A, C, 0, 0) / 5, "log-polar: rotation = column shift");
}

// ---------------------------------------------------------------------------------------
const tests: [string, () => void][] = [
  ["random", testRandom], ["fft", testFFT], ["linalg", testLinalg], ["imageops", testImageops],
  ["synth", testSynth], ["histogram", testHistogram], ["metrics", testMetrics], ["label", testLabel],
  ["distance", testDistance], ["morph", testMorph], ["pqueue", testPqueue], ["polar", testPolar],
];
for (const [name, fn] of tests) {
  const t0 = Date.now();
  try { fn(); } catch (e) { current = name; check(false, `threw: ${(e as Error).stack}`); }
  console.log(`  ${name.padEnd(10)} ${String(section[name] ?? 0).padStart(6)} checks  ${Date.now() - t0} ms`);
}
// timings at demo sizes, for information
{
  const r = rng(3), img = randImg(r, 256, 256, 255);
  const time = (label: string, f: () => void) => { const t0 = Date.now(); f(); console.log(`  (${label}: ${Date.now() - t0} ms)`); };
  time("256² Gaussian blur σ=3", () => IO.gaussianBlur(img, 3));
  time("256² FFT forward+inverse", () => { const s = F.fftReal2d(img.data, 256, 256); F.ifftReal2d(s.re, s.im, 256, 256); });
  const m = Uint8Array.from(img.data, v => (v > 128 ? 1 : 0));
  time("256² exact EDT", () => DT.edt(m, 256, 256));
  time("256² label 8-conn", () => LB.label2d(m, 256, 256, 8));
  time("256² reconstruction (noise)", () => MO.reconstructByDilation(img.data.map(v => v - 40), img.data, 256, 256));
  const nuc = SY.nuclei({ seed: 9 }).image;
  time("256² h-maxima h=20 (nuclei)", () => MO.hMaxima(nuc.data, 256, 256, 20));
  time("256² grey opening, disk r=5", () => MO.open(img.data, 256, 256, MO.seDisk(5)));
  time("256² nuclei generator", () => SY.nuclei({ seed: 9 }));
}
console.log(`${checks - failures} of ${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
