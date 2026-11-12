// Tests for the image formation day's demo code (psfotf/model.ts, resolve/model.ts) and the
// identities its units teach. Reference checks against closed forms and brute force. Run:
//   node_modules/.bin/esbuild demos/test/imageformation-test.ts --bundle --platform=node --log-level=warning | node
import * as P from "../src/lib/psf";
import * as D from "../src/lib/deconv";
import * as F from "../src/psfotf/model";
import * as R from "../src/resolve/model";
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
const Rn = rng(77);

// ------------------------------------------------------------- psfotf: PSF, OTF, blur
{
  const o: F.Optics = { kind: "pupil", NA: 1.4, wavelength: 520, defocus: 0, motion: 0, angle: 0 };
  const r = F.computePSF(o);
  check(close(sum(r.psf), 1, 1e-12) && r.psf.every(v => v >= 0), "pupil PSF sums to 1, non-negative");
  check(close(r.re[0], 1, 1e-12) && Math.abs(r.im[0]) < 1e-12, "H(0) = 1 for a PSF that sums to 1");
  // OTF along fx = analytic OTF of a circular pupil, 0 beyond the cut-off
  const fc = P.cutoff(o.NA, o.wavelength, F.PIXEL);
  let worst = 0, beyond = 0;
  for (let u = 0; u < F.N / 2; u++) {
    const s = (u / F.N) / fc;
    if (s < 1) worst = Math.max(worst, Math.abs(r.re[u] - P.airyOTF(s)));
    else beyond = Math.max(beyond, Math.abs(r.re[u]));
  }
  check(worst < 0.02, `in-focus OTF = (2/π)(acos s − s√(1−s²)) (max diff ${worst.toFixed(4)})`);
  check(beyond < 1e-9, `OTF = 0 beyond 2NA/λ (max ${beyond.toExponential(1)})`);
  // FWHM ≈ 0.51 λ/NA (Airy)
  const fw = F.psfFWHM(r.psf);
  check(Math.abs(fw - 0.5145 * 520 / 1.4) < 8, `Airy FWHM ≈ 0.51 λ/NA (${fw.toFixed(1)} nm)`);
  // Gaussian approximation: σ = 0.21 λ/NA, FWHM 2.355σ
  const g = F.computePSF({ ...o, kind: "gauss" });
  check(Math.abs(F.psfFWHM(g.psf) - 2.3548 * 0.21 * 520 / 1.4) < 3, "Gaussian approximation FWHM = 2.355 · 0.21 λ/NA");
  // defocus: OTF takes negative values (contrast reversal) for NA 0.7, 2 µm
  const dfc = F.computePSF({ ...o, NA: 0.7, defocus: 2000 });
  let mn = 0;
  for (let u = 0; u < F.N / 2; u++) mn = Math.min(mn, dfc.re[u]);
  check(mn < -0.04, `defocused OTF goes negative (min ${mn.toFixed(3)})`);
  check(close(sum(dfc.psf), 1, 1e-12), "defocused PSF sums to 1 (light is spread, not lost)");
  // motion blur along x with L = 1000 nm: OTF zeros near k/L along fx, none along fy
  const m = F.computePSF({ ...o, kind: "motion", motion: 1000 });
  const at = (f: number) => m.re[Math.round(f / 1000 * F.N * F.PIXEL)];
  check(Math.abs(at(1)) < 0.03 && Math.abs(at(2)) < 0.03, `motion OTF ≈ 0 at 1/L and 2/L (${at(1).toFixed(3)}, ${at(2).toFixed(3)})`);
  check(m.re[Math.round(1 / 1000 * F.N * F.PIXEL) * F.N] > 0.5, "motion OTF along fy (across the motion) is not zero at 1 cycle/µm");
  // the motion OTF equals Airy OTF × sinc(L f) (up to the antialiased line)
  let wm = 0;
  for (let u = 1; u < 40; u++) {
    const f = u / (F.N * F.PIXEL), s = Math.PI * f * 1000;
    wm = Math.max(wm, Math.abs(m.re[u] - r.re[u] * Math.sin(s) / s));
  }
  check(wm < 0.02, `motion OTF = lens OTF × sinc(Lf) (max diff ${wm.toFixed(4)})`);
}
{
  // blurWith (FFT) = direct circular convolution on a small image
  const n = 16, img = Float64Array.from({ length: n * n }, () => Rn.unif());
  const psf = P.gaussianPSF(n, n, 1.3), H = P.otf2d(psf, n, n);
  const fftRes = F.blurWith(img, H, n), direct = new Float64Array(n * n), c = n / 2;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let s = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s += img[((y - (j - c)) % n + n) % n * n + ((x - (i - c)) % n + n) % n] * psf[j * n + i];
    direct[y * n + x] = s;
  }
  check(maxAbsDiff(fftRes, direct) < 1e-12, "FFT blur = direct circular convolution");
  check(close(sum(fftRes), sum(img), 1e-12), "blur keeps the total intensity");
  // 1D example of the unit: three points (1, 0, 0, 2, 0, 1) blurred with (1, 2, 1)/4
  const x = [0, 1, 0, 0, 2, 0, 1, 0], k = [1, 2, 1].map(v => v / 4), y = x.map((_, i) => k[0] * (x[i - 1] ?? 0) + k[1] * x[i] + k[2] * (x[i + 1] ?? 0));
  check(maxAbsDiff(y, [0.25, 0.5, 0.25, 0.5, 1, 0.75, 0.5, 0.25]) < 1e-12, "three points ∗ (1,2,1)/4 by hand");
}

// ------------------------------------------------------------- resolve: two points, sampling
{
  const o = { NA: 1.4, wavelength: 520 };
  const ray = P.airyRadius(o.NA, o.wavelength);
  const dp = R.dip(ray, o);
  check(Math.abs(dp - 0.265) < 0.003, `Rayleigh dip ≈ 26.5 % (${(100 * dp).toFixed(2)} %)`);
  const sp = R.sparrowDistance(o) / (o.wavelength / o.NA);
  check(Math.abs(sp - 0.4735) < 0.002, `Sparrow ≈ 0.47 λ/NA (${sp.toFixed(4)})`);
  check(R.dip(0.99 * R.sparrowDistance(o), o) < 1e-6 && R.dip(1.1 * R.sparrowDistance(o), o) > 0, "no dip below Sparrow, a dip above");
  // Sparrow for a Gaussian PSF is 2σ (brute force on the curvature)
  const sig = 50, gp = (x: number, d: number) => Math.exp(-((x - d / 2) ** 2) / (2 * sig * sig)) + Math.exp(-((x + d / 2) ** 2) / (2 * sig * sig));
  const curv = (d: number) => gp(0.5, d) + gp(-0.5, d) - 2 * gp(0, d);
  check(curv(1.99 * sig) < 0 && curv(2.01 * sig) > 0, "Sparrow distance of a Gaussian = 2σ");
  // the Airy density integrates to the photon count
  let tot = 0;
  const h = 5;
  for (let y = -3000; y <= 3000; y += h) for (let x = -3000; x <= 3000; x += h) tot += R.airyDensity(Math.hypot(x, y), o, 1000) * h * h;
  check(Math.abs(tot - 1000) < 15, `Airy pattern carries all the photons (${tot.toFixed(1)} of 1000 within ±3 µm)`);
  // pixel sampling: expected counts, noise-free = expected
  const s = R.samplePixels(R.twoPointDensity(300, o, 1000), 8000, 8000, 65, false, 5, 3);
  check(s.counts.every((v, i) => v === s.expected[i]) && Math.abs(sum(s.expected) - 2000) < 20, `pixels collect the photons of both points (${sum(s.expected).toFixed(0)})`);
  // aliasing: period 300 nm sampled every 200 nm looks like 600 nm; sampled every 100 nm, unchanged
  check(close(R.apparentPeriod(300, 200), 600, 1e-9) && close(R.apparentPeriod(300, 100), 300, 1e-9), "apparent period after sampling");
  check(close(R.gratingContrast(186, o), 0, 1e-9) || R.gratingContrast(185, o) === 0, "a grating finer than λ/2NA has no contrast");
}
{
  // axial PSF: radial (Hankel) form = the FFT pupil PSF of lib/psf.ts, in and out of focus
  const opt = { NA: 1.2, wavelength: 500, ri: 1.518 }, px = 40, n = 128;
  for (const z of [0, 400]) {
    const fft = P.pupilPSF(n, { NA: opt.NA, wavelength: opt.wavelength, pixel: px, ri: opt.ri }, z);
    const ref0 = P.pupilPSF(n, { NA: opt.NA, wavelength: opt.wavelength, pixel: px, ri: opt.ri }, 0);
    const c = n / 2, rs = Float64Array.from({ length: 12 }, (_, i) => i * px);
    const hk = R.psfRZ(rs, [z], opt, 512), hk0 = R.psfRZ([0], [0], opt, 512)[0];
    let w = 0;
    for (let i = 0; i < 12; i++) w = Math.max(w, Math.abs(fft[c * n + c + i] / ref0[c * n + c] - hk[i] / hk0));
    check(w < 0.02, `Hankel PSF = FFT pupil PSF at z = ${z} nm (max diff ${w.toFixed(4)} of the peak)`);
  }
  const a = R.axialPSF({ NA: 1.4, wavelength: 520, ri: 1.518 }, false, 1500, 3000);
  check(Math.abs(a.fwhmLat - 0.5145 * 520 / 1.4) < 5, `lateral FWHM of the axial model = Airy FWHM (${a.fwhmLat.toFixed(1)})`);
  check(a.fwhmAx > 2.4 * a.fwhmLat, `PSF elongated along z (axial/lateral = ${(a.fwhmAx / a.fwhmLat).toFixed(2)})`);
  // first axial minimum of the widefield PSF near λ / (n (1 − cos α))
  for (const [NA, ri] of [[0.5, 1], [1.4, 1.518]]) {
    const zz = 520 / (ri * (1 - Math.sqrt(1 - (NA / ri) ** 2)));
    const zsN = Array.from({ length: 400 }, (_, i) => zz * (0.5 + i / 400)), v = R.psfRZ([0], zsN, { NA, wavelength: 520, ri });
    let k = 1;
    while (k < 399 && !(v[k] < v[k - 1] && v[k] <= v[k + 1])) k++;
    check(Math.abs(zsN[k] / zz - 1) < 0.06 && v[k] < 0.03, `NA ${NA}: first axial minimum at ${zsN[k].toFixed(0)} nm ≈ λ/n(1−cos α) = ${zz.toFixed(0)} nm (I = ${v[k].toFixed(4)})`);
  }
  const cf = R.axialPSF({ NA: 1.4, wavelength: 520, ri: 1.518 }, true, 1500, 3000);
  check(cf.fwhmLat < a.fwhmLat && cf.fwhmAx < a.fwhmAx, "confocal PSF narrower in x and z");
  // widefield: every plane carries the same light (integral over x, y of |h|² does not depend on z)
  const rs = Float64Array.from({ length: 400 }, (_, i) => (i + 0.5) * 10);
  const pl = (z: number) => { const v = R.psfRZ(rs, [z], { NA: 0.8, wavelength: 520, ri: 1 }); let s = 0; for (let i = 0; i < rs.length; i++) s += v[i] * 2 * Math.PI * rs[i] * 10; return s; };
  check(Math.abs(pl(1000) / pl(0) - 1) < 0.03, `widefield: out-of-focus planes carry the same light (${(pl(1000) / pl(0)).toFixed(3)})`);
}
{
  // beads: the averaged PSF of noise-free point beads at sub-pixel offsets = the true PSF
  const beads = R.beadPositions();
  check(beads.length === 30, "30 beads placed");
  const fine0 = R.beadImageFine(0), truth = R.truePSF();
  const field = R.beadField(fine0, beads, 5000, 0, false);
  const avg = R.averageBeads(field.counts, beads, 30, false);   // no background here; the border median would remove a bit of the PSF's tails
  check(Math.abs(avg.fwhm / truth.fwhm - 1) < 0.01, `noise-free average of 30 beads → true PSF (FWHM ${avg.fwhm.toFixed(1)} vs ${truth.fwhm.toFixed(1)})`);
  check(maxAbsDiff(avg.patch, truth.patch) < 0.01 * Math.max(...Array.from(truth.patch)), "averaged patch ≈ true patch");
  check(close(sum(avg.patch), 1, 1e-9), "averaged PSF normalised to sum 1");
  // a bead of diameter D widens the PSF like a blur of FWHM ≈ 0.53 D (sphere: variance R²/5)
  for (const Dm of [100, 200]) {
    const f = R.beadField(R.beadImageFine(Dm), beads, 5000, 0, false);
    const w = R.averageBeads(f.counts, beads, 30, true).fwhm, pred = Math.sqrt(truth.fwhm ** 2 + (0.527 * Dm) ** 2);
    check(Math.abs(w / pred - 1) < 0.03, `bead ${Dm} nm: FWHM ${w.toFixed(1)} ≈ √(PSF² + (0.53 D)²) = ${pred.toFixed(1)}`);
  }
  // forgetting a background widens the measured PSF
  const fb = R.beadField(fine0, beads, 500, 10, false);
  check(R.averageBeads(fb.counts, beads, 30, false).fwhm > 1.1 * truth.fwhm && Math.abs(R.averageBeads(fb.counts, beads, 30, true).fwhm / truth.fwhm - 1) < 0.01, "background left in → too wide; subtracted → right");
  // Fourier shift by an integer = roll; upsampling keeps the samples
  const p = Float64Array.from({ length: R.PATCH * R.PATCH }, () => Rn.unif());
  const sh = R.fourierShift(p, 2, -4);
  let wr = 0;
  for (let j = 0; j < R.PATCH; j++) for (let i = 0; i < R.PATCH; i++) wr = Math.max(wr, Math.abs(sh[((j - 4 + R.PATCH) % R.PATCH) * R.PATCH + (i + 2) % R.PATCH] - p[j * R.PATCH + i]));
  check(wr < 1e-12, "Fourier shift by an even number of pixels = circular roll");
  const up = R.upsample(truth.patch), m = R.PATCH * R.UP;
  let wu = 0;
  for (let j = 0; j < R.PATCH; j++) for (let i = 0; i < R.PATCH; i++) wu = Math.max(wu, Math.abs(up[j * R.UP * m + i * R.UP] - truth.patch[j * R.PATCH + i]));
  check(wu < 1e-3 * Math.max(...truth.patch), `Fourier upsampling passes through the original samples (${wu.toExponential(1)})`);
}

// ------------------------------------------------------------- ill-posed: inverse filter, circulant blur
{
  // without noise and with H ≠ 0 the inverse filter recovers the object exactly
  const w = 32, psf = P.gaussianPSF(w, w, 1.0), x = Float64Array.from({ length: w * w }, () => Rn.unif());
  const y = D.fftOperator2d(psf, w, w).forward(x);
  const inv = D.linearRestore(y, psf, w, w, { kind: "inverse" });
  check(maxAbsDiff(inv, x) < 1e-6, `inverse filter exact without noise (${maxAbsDiff(inv, x).toExponential(1)})`);
  // circulant blur (1,2,1)/4 on n = 4: eigenvalues 1, 0.5, 0, 0.5 for the waves k = 0..3
  const n = 4, A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => ((j - i + n) % n === 0 ? 0.5 : (j - i + n) % n === 1 || (j - i + n) % n === n - 1 ? 0.25 : 0)));
  const lam = [1, 0.5, 0, 0.5];
  for (let k = 0; k < n; k++) {
    const v = Array.from({ length: n }, (_, j) => Math.cos(2 * Math.PI * k * j / n));
    const Av = A.map(row => row.reduce((acc: number, a: number, j: number) => acc + a * v[j], 0));
    check(maxAbsDiff(Av, v.map(t => lam[k] * t)) < 1e-12, `circulant (1,2,1)/4: wave k = ${k} has eigenvalue ${lam[k]}`);
  }
  // the two 2 × 2 blurs: error growth along the difference direction 1/0.5 = 2 and 1/0.1 = 10
  for (const [a, b, gain] of [[0.75, 0.25, 2], [0.55, 0.45, 10]]) {
    const det = a * a - b * b, e = [0.01, -0.01], dx = [(a * e[0] - b * e[1]) / det, (-b * e[0] + a * e[1]) / det];
    check(close(Math.hypot(dx[0], dx[1]) / Math.hypot(e[0], e[1]), gain, 1e-9), `2 × 2 blur (${a}, ${b}): difference error × ${gain}`);
  }
}

console.log(`imageformation: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
