// Tests for the deconvolution day's libraries (lib/psf.ts, lib/deconv.ts, lib/denoise.ts):
// reference values, brute-force versions and the identities the units teach. Run alone:
//   node_modules/.bin/esbuild demos/test/deconvolution-test.ts --bundle --platform=node --log-level=warning | node
import * as P from "../src/lib/psf";
import * as D from "../src/lib/deconv";
import * as N from "../src/lib/denoise";
import { solve } from "../src/lib/linalg";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const R = rng(2024);
const randArr = (n: number, s = 1) => Float64Array.from({ length: n }, () => s * R.unif());

// ---------------------------------------------------------------- psf.ts
check(close(P.besselJ1(1), 0.4400505857, 1e-7) && close(P.besselJ1(3.8317), 0, 1e-4) && close(P.besselJ1(10), 0.0434727462, 1e-6), "besselJ1 reference values");
check(close(P.airy(P.airyRadius(1.4, 500), 1.4, 500), 0, 1e-4), "first Airy zero at 0.61 λ/NA");
{
  const g = P.gaussianPSF(64, 64, 3);
  check(close(sum(g), 1), "Gaussian PSF sums to 1");
  const row = g.subarray(32 * 64, 33 * 64);
  check(Math.abs(P.fwhm(row) - 2.3548 * 3) < 0.05, `FWHM of a Gaussian = 2.355 σ (${P.fwhm(row)})`);
  for (const f of [P.motionPSF(64, 64, 9, 0.3), P.diskPSF(64, 64, 5), P.airyPSF(64, 64, { NA: 1, wavelength: 500, pixel: 50 })])
    check(close(sum(f), 1) && f.every(v => v >= 0), "PSF non-negative, sum 1");
  // pupil PSF in focus = Airy pattern sampled at pixel centres (n large to keep the tails)
  const o = { NA: 1.0, wavelength: 500, pixel: 40 };
  const pp = P.pupilPSF(128, o, 0);
  const ref = P.psfFromFunction(128, 128, (dx, dy) => P.airy(Math.hypot(dx, dy) * o.pixel, o.NA, o.wavelength), 1);
  const peak = Math.max(...ref);
  check(maxAbsDiff(pp, ref) < 0.02 * peak, `pupil PSF in focus ≈ Airy (max diff ${(maxAbsDiff(pp, ref) / peak).toFixed(4)} of the peak)`);
  // OTF of the pupil PSF = analytic Airy OTF (the pupil PSF is band-limited and periodic)
  const H = P.otf2d(pp, 128, 128), fc = P.cutoff(o.NA, o.wavelength, o.pixel);
  let worst = 0;
  for (let u = 0; u < 64; u++) worst = Math.max(worst, Math.abs(H.re[u] - P.airyOTF((u / 128) / fc)));
  check(worst < 0.02, `OTF of the in-focus PSF = analytic OTF (max diff ${worst.toFixed(4)})`);
  // defocus spreads the light: lower peak, same sum
  const pd = P.pupilPSF(128, o, 800);
  check(close(sum(pd), 1) && Math.max(...pd) < 0.5 * Math.max(...pp), "defocus lowers the peak, keeps the sum");
  // 3D: every widefield plane carries the same light; confocal planes far from focus are dark
  const o3 = { NA: 1.2, wavelength: 520, pixel: 100, zstep: 200, ri: 1.33 };
  const wf = P.widefieldPSF3d(32, 16, o3), cf = P.confocalPSF3d(32, 16, o3);
  const plane = (v: Float64Array, z: number) => sum(v.subarray(z * 1024, (z + 1) * 1024));
  check(close(plane(wf, 0), plane(wf, 8), 1e-6), "widefield: each plane has the same total light");
  check(plane(cf, 2) < 0.1 * plane(cf, 8), "confocal: out-of-focus planes are dim");
  check(close(sum(wf), 1) && close(sum(cf), 1), "3D PSFs sum to 1");
}

// ---------------------------------------------------------------- fast real convolver vs brute force
{
  for (const [w, h, d] of [[8, 4, 1], [4, 8, 1], [4, 4, 4], [8, 2, 4]]) {
    const n = w * h * d, k = randArr(n), x = randArr(n);
    const Op = d === 1 ? D.fftOperator2d(k, w, h) : D.fftOperator3d(k, w, h, d);
    const cx = w >> 1, cy = h >> 1, cz = d >> 1, idx = (X: number, Y: number, Z: number) => (((Z % d + d) % d) * h + ((Y % h + h) % h)) * w + ((X % w + w) % w);
    const conv = new Float64Array(n), corr = new Float64Array(n);
    for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) {
      let s = 0, t = 0;
      for (let c = 0; c < d; c++) for (let b = 0; b < h; b++) for (let a = 0; a < w; a++) {
        const kv = k[(c * h + b) * w + a], ox = a - cx, oy = b - cy, oz = c - cz;
        s += kv * x[idx(xx - ox, y - oy, z - oz)];
        t += kv * x[idx(xx + ox, y + oy, z + oz)];
      }
      conv[(z * h + y) * w + xx] = s; corr[(z * h + y) * w + xx] = t;
    }
    check(maxAbsDiff(Op.forward(x), conv) < 1e-12 * n && maxAbsDiff(Op.adjoint(x), corr) < 1e-12 * n, `fast convolver = brute force (${w}×${h}×${d})`);
  }
}

// ---------------------------------------------------------------- deconv.ts, linear methods
{
  const w = 32, h = 32, psf = P.gaussianPSF(w, h, 1.2), x = randArr(w * h);
  const Aop = D.fftOperator2d(psf, w, h), y = Aop.forward(x);
  const inv = D.linearRestore(y, psf, w, h, { kind: "inverse" });
  check(maxAbsDiff(inv, x) < 1e-6, `inverse filter without noise is exact (${maxAbsDiff(inv, x)})`);
  const wie = D.linearRestore(y, psf, w, h, { kind: "wiener", snr: 1e14, model: "flat" });
  check(maxAbsDiff(wie, inv) < 1e-4, "Wiener with noise → 0 equals the inverse filter");
  // Tikhonov energy = Wiener flat with λ = 1/SNR
  const t = D.linearRestore(y, psf, w, h, { kind: "tikhonov", lambda: 0.01, penalty: "energy" });
  const t2 = D.linearRestore(y, psf, w, h, { kind: "wiener", snr: 100, model: "flat" });
  check(maxAbsDiff(t, t2) < 1e-12, "Tikhonov (energy, λ) = Wiener (flat, SNR = 1/λ)");
  // roughness penalty keeps the mean; energy shrinks it by 1/(1 + λ)
  const tr = D.linearRestore(y, psf, w, h, { kind: "tikhonov", lambda: 0.5, penalty: "roughness" });
  const te = D.linearRestore(y, psf, w, h, { kind: "tikhonov", lambda: 0.5, penalty: "energy" });
  check(close(sum(tr), sum(y), 1e-9), "roughness-penalised Tikhonov keeps the mean (flux)");
  check(close(sum(te), sum(y) / 1.5, 1e-9), "energy-penalised Tikhonov scales the mean by 1/(1+λ)");
  // truncated inverse: frequencies with |H| ≤ ε are removed
  const c = D.gainContext(Aop.H, w, h);
  let okT = true;
  for (let i = 0; i < w * h; i++) { const g = D.gainFactor({ kind: "truncated", eps: 0.1 }, c, i); if (Math.sqrt(c.H2[i]) <= 0.1 ? g !== 0 : !close(g, 1 / c.H2[i])) okT = false; }
  check(okT, "truncated inverse gain");
  // Wiener gain numbers of the lecture: H = 0.5, SNR 100 → 1.92; H = 0.05 → 4
  const wg = (H: number, snr: number) => H / (H * H + 1 / snr);
  check(close(wg(0.5, 100), 1.923, 1e-3) && close(wg(0.05, 100), 4, 1e-9), "Wiener gains of the lecture");
  // padding with a taper keeps the image inside
  const pt = D.padTaper(x, w, h, 64, 64);
  let okP = true;
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) if (pt.data[(yy + pt.oy) * 64 + xx + pt.ox] !== x[yy * w + xx]) okP = false;
  check(okP, "padTaper keeps the image");
}

// ---------------------------------------------------------------- 1D regularisation
{
  // the 2-pixel blur A = [[0.75, 0.25], [0.25, 0.75]] with energy penalty, data (1, 0.5)
  const B2 = D.blur1dOf([0.25, 0.75]), y2 = [1, 0.5], E2 = D.penaltySymbol1d(2, "energy");
  const want: [number, number[]][] = [[0, [1.25, 0.25]], [0.1, [1.0390, 0.3247]], [1, [0.475, 0.275]]];
  for (const [lam, xs] of want) {
    const x = D.tikhonov1d(y2, B2, lam, E2);
    check(Math.abs(x[0] - xs[0]) < 1e-3 && Math.abs(x[1] - xs[1]) < 1e-3, `2-pixel Tikhonov λ = ${lam}: (${x[0].toFixed(4)}, ${x[1].toFixed(4)})`);
  }
  // 8-pixel case: closed form = normal equations solved directly, for each penalty
  const n = 8, ker = new Float64Array(n);
  ker[3] = 0.2; ker[4] = 0.5; ker[5] = 0.3;   // asymmetric kernel, centre at index 4
  const B = D.blur1dOf(ker), y = randArr(n);
  const A = new Float64Array(n * n);
  for (let j = 0; j < n; j++) { const e = new Float64Array(n); e[j] = 1; const col = D.blur1d(e, B); for (let i = 0; i < n; i++) A[i * n + j] = col[i]; }
  for (const pen of ["energy", "gradient", "roughness"] as D.Penalty1D[]) {
    const Dm = new Float64Array(n * n);
    for (let j = 0; j < n; j++) { const e = new Float64Array(n); e[j] = 1; const col = D.penaltyOp1d(e, pen); for (let i = 0; i < n; i++) Dm[i * n + j] = col[i]; }
    const lam = 0.3, M = new Float64Array(n * n), b = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0, t = 0; for (let k = 0; k < n; k++) { s += A[k * n + i] * A[k * n + j]; t += Dm[k * n + i] * Dm[k * n + j]; } M[i * n + j] = s + lam * t; }
    for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < n; k++) s += A[k * n + i] * y[k]; b[i] = s; }
    const xd = solve(M, b, n)!;
    const xf = D.tikhonov1d(y, B, lam, D.penaltySymbol1d(n, pen));
    check(maxAbsDiff(xd, xf) < 1e-9, `8-pixel Tikhonov (${pen}) = normal equations`);
  }
  // Landweber iterates = filter-factor formula
  const k = 37, xl = D.landweber1d(y, B, k), xf = D.applyFactors(y, B, D.landweberFactors(B.H2, k));
  check(maxAbsDiff(xl, xf) < 1e-9, "Landweber iterate = filter factors 1 − (1 − |H|²)^k");
  // filter factor numbers of the lecture
  const lf = (s: number, kk: number) => 1 - (1 - s * s) ** kk;
  check(close(lf(0.5, 1), 0.25) && Math.abs(lf(0.5, 4) - 0.684) < 1e-3 && Math.abs(lf(0.5, 16) - 0.990) < 1e-3 && Math.abs(lf(0.1, 16) - 0.149) < 1e-3 && Math.abs(lf(0.1, 100) - 0.634) < 1e-3, "Landweber factors of the lecture");
  // discrepancy principle on a 256-sample problem
  const m = 256, Bg = D.blur1dOf(D.gaussian1d(m, 3)), truth = Float64Array.from({ length: m }, (_, i) => (i > 80 && i < 160 ? 1 : 0));
  const sd = 0.02, data = D.blur1d(truth, Bg).map(v => v + sd * R.norm());
  const R2 = D.penaltySymbol1d(m, "roughness");
  const res = (lam: number) => D.rms(D.residual1d(D.tikhonov1d(data, Bg, lam, R2), data, Bg));
  const lamD = D.discrepancyLambda(res, sd, 1e-8, 1e3);
  check(Math.abs(res(lamD) - sd) < 1e-6, `discrepancy λ gives residual = σ_n (λ = ${lamD.toExponential(2)})`);
  // residual increases with λ
  check(res(1e-4) < res(1e-2) && res(1e-2) < res(1), "residual grows with λ");
  // non-negative Tikhonov: x ≥ 0 and the KKT conditions hold approximately
  const xn = D.tikhonovNonneg1d(data, Bg, 1e-3, R2, 2000);
  check(xn.every(v => v >= 0), "non-negative Tikhonov is ≥ 0");
  const obj = (x: Float64Array, lam: number) => D.norm2(D.residual1d(x, data, Bg)) ** 2 + lam * D.norm2(D.penaltyOp1d(x, "roughness")) ** 2;
  const xu = D.tikhonov1d(data, Bg, 1e-3, R2), xclip = xu.map(v => Math.max(0, v));
  check(obj(xn, 1e-3) <= obj(xclip, 1e-3) + 1e-12, "non-negative Tikhonov beats clipping the unconstrained solution");
  // TV: lower objective than the data itself and than the roughness solution
  const lamTV = 0.01, objTV = (x: Float64Array) => D.norm2(D.residual1d(x, data, Bg)) ** 2 + lamTV * Array.from(x, (v, i) => Math.sqrt((x[(i + 1) % m] - v) ** 2 + 1e-6)).reduce((a, b) => a + b, 0);
  const xtv = D.tv1d(data, Bg, lamTV, { beta: 1e-3, outer: 30 });
  check(objTV(xtv) < objTV(Float64Array.from(data)) && objTV(xtv) < objTV(xu), "TV (IRLS) lowers its objective");
  // L-curve corner on a synthetic L
  const rho = [1, 1.01, 1.02, 1.05, 2, 4, 8], eta = [100, 30, 10, 3, 2.9, 2.8, 2.7];
  check(D.lcurveCorner(rho, eta) === 3, `L-curve corner at the bend (${D.lcurveCorner(rho, eta)})`);
}

// ---------------------------------------------------------------- Richardson–Lucy
{
  const A = D.directOperator1d([0.25, 0.5, 0.25], 4), y = Float64Array.from([1, 2, 1, 0]);
  const x1 = D.rlStep(Float64Array.from([1, 1, 1, 1]), y, A), x2 = D.rlStep(x1, y, A);
  check(maxAbsDiff(x1, [1, 1.5, 1, 0.5]) < 1e-12, `RL step 1 by hand (${Array.from(x1)})`);
  check(maxAbsDiff(x2, [0.9, 1.95, 0.9, 0.25]) < 1e-12, `RL step 2 by hand (${Array.from(x2)})`);
  check(close(sum(x1), 4) && close(sum(x2), 4), "RL keeps the sum");
  check(maxAbsDiff(D.richardsonLucy(y, A, 2), x2) < 1e-12, "richardsonLucy from the flat start");
  // 2D: sum, non-negativity, monotone likelihood
  const w = 32, psf = P.gaussianPSF(w, w, 1.5), Op = D.fftOperator2d(psf, w, w);
  const truth = randArr(w * w, 50), data = Op.forward(truth).map(v => Math.round(v + 3 * R.norm() > 0 ? v + 3 * R.norm() : 0));
  let prev = -Infinity, okL = true, okS = true, okN = true;
  D.richardsonLucy(data, Op, 40, {
    onIter: (_k, x) => {
      const L = D.poissonLogLik(data, Op.forward(x));
      if (L < prev - 1e-6 * Math.abs(prev)) okL = false;
      prev = L;
      if (!close(sum(x), sum(data), 1e-8)) okS = false;
      if (!x.every(v => v >= 0)) okN = false;
    },
  });
  check(okL, "RL: the Poisson likelihood never decreases");
  check(okS, "RL: flux kept every iteration");
  check(okN, "RL: never negative");
  // with a background: flux of x = flux of the data − background
  const b = 5, datab = data.map(v => v + b);
  const xb = D.richardsonLucy(datab, Op, 30, { background: b });
  check(Math.abs(sum(xb) - (sum(datab) - b * w * w)) < 1e-3 * sum(datab), "RL with background keeps data − background flux");
  // 3D RL on a volume constant along z = 2D RL
  const nz = 4, psf3 = new Float64Array(w * w * nz);
  psf3.set(psf, (nz / 2) * w * w);
  const vol = new Float64Array(w * w * nz);
  for (let z = 0; z < nz; z++) vol.set(data, z * w * w);
  const Op3 = D.fftOperator3d(psf3, w, w, nz);
  const x3 = D.richardsonLucy(vol, Op3, 10), x2d = D.richardsonLucy(data, Op, 10);
  let ok3 = true;
  for (let z = 0; z < nz; z++) if (maxAbsDiff(x3.subarray(z * w * w, (z + 1) * w * w), x2d) > 1e-8 * Math.max(...x2d)) ok3 = false;
  check(ok3, "3D RL on a z-constant volume = 2D RL");
  // blind step: PSF stays non-negative, sums to 1 and respects the support
  let bx = D.flatStart(data), bp = P.gaussianPSF(w, w, 2.5);
  for (let i = 0; i < 5; i++) ({ x: bx, psf: bp } = D.blindStep(bx, bp, data, w, w, { supportRadius: 6 }));
  check(close(sum(bp), 1) && bp.every(v => v >= 0), "blind RL: PSF ≥ 0, sum 1");
  let okR = true;
  for (let yy = 0; yy < w; yy++) for (let xx = 0; xx < w; xx++) if ((xx - 16) ** 2 + (yy - 16) ** 2 > 36 && bp[yy * w + xx] !== 0) okR = false;
  check(okR, "blind RL: PSF zero outside the support");
}

// ---------------------------------------------------------------- denoise.ts
{
  const w = 8, h = 8, img = randArr(w * h, 100);
  check(maxAbsDiff(N.medianFilter(img, w, h, 1), N.medianBrute(img, w, h, 1)) < 1e-12, "median r = 1 = brute force");
  check(maxAbsDiff(N.medianFilter(img, w, h, 2), N.medianBrute(img, w, h, 2)) < 1e-12, "median r = 2 = brute force");
  check(maxAbsDiff(N.bilateral(img, w, h, 1.5, 20, 3), N.bilateralBrute(img, w, h, 1.5, 20, 3)) < 1e-9, "bilateral = brute force");
  check(maxAbsDiff(N.nonLocalMeans(img, w, h, 1, 2, 15), N.nonLocalMeansBrute(img, w, h, 1, 2, 15)) < 1e-9, "non-local means = brute force (patch 3×3, search 5×5)");
  check(maxAbsDiff(N.nonLocalMeans(img, w, h, 2, 3, 30), N.nonLocalMeansBrute(img, w, h, 2, 3, 30)) < 1e-9, "non-local means = brute force (patch 5×5, search 7×7)");
  // the by-hand row: median vs mean, window 3
  const row = [10, 10, 10, 200, 10, 10, 50, 50, 50];
  check(maxAbsDiff(N.median1d(row, 3), [10, 10, 10, 10, 10, 10, 50, 50, 50]) < 1e-12, "1D median of the by-hand row");
  // bilateral weight across a step: e^{-40²/(2·10²)} = e^{-8}
  check(close(Math.exp(-(40 ** 2) / (2 * 100)), Math.exp(-8)), "bilateral range weight e^{-8}");
  // Wiener denoiser keeps the mean
  const im2 = randArr(32 * 32, 10);
  check(Math.abs(sum(N.wienerDenoise(im2, 32, 32, 100)) - sum(im2)) < 1e-3 * sum(im2), "Wiener denoiser keeps the mean (gain ≈ 1 at f = 0)");
}

console.log(`deconvolution: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
