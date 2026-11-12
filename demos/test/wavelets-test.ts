// Tests for demos/src/lib/wavelet.ts (wavelets day): perfect reconstruction in 1D and 2D for
// every wavelet and boundary, energy preservation, filter identities and vanishing moments,
// the board example, integer lifting, the cascade algorithm, thresholding, and a brute-force
// check of the filter bank against the explicit transform matrix. Run alone:
//   node_modules/.bin/esbuild demos/test/wavelets-test.ts --bundle --platform=node --log-level=warning | node
import {
  WAVELETS, WaveletName, OrthoWavelet, Boundary, analysisStep, synthesisStep, wavedec, waverec, dwt, idwt, dwt2, idwt2,
  haarStates, haarInverse, haarMatrix, liftHaarInt, unliftHaarInt, cascade, hard, soft, madSigma, universalThreshold,
  denoise2, denoise1, keepLargest, tailErrors, testSignal, realFourier, realFourierInverse, subbandRect, bandAt, levelOf, branchSignals,
} from "../src/lib/wavelet";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return a.length === b.length ? m : Infinity;
};
const energy = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s; };
const R = rng(2024);
const randVec = (n: number, scale = 1) => Float64Array.from({ length: n }, () => scale * (2 * R.unif() - 1));

const ORTH = (Object.values(WAVELETS).filter(w => w.kind === "orth") as OrthoWavelet[]);
const ALL = Object.keys(WAVELETS) as WaveletName[];

// ---- filter identities ----
for (const w of ORTH) {
  const L = w.h.length;
  let sh = 0, sg = 0, ss = 0;
  for (let n = 0; n < L; n++) { sh += w.h[n]; sg += w.g[n]; ss += w.h[n] * w.h[n]; }
  check(Math.abs(sh - Math.SQRT2) < 1e-14, `${w.name}: sum h = sqrt2`);
  check(Math.abs(sg) < 1e-14, `${w.name}: sum g = 0`);
  check(Math.abs(ss - 1) < 1e-14, `${w.name}: sum h^2 = 1`);
  // orthogonality of even shifts: sum h[n] h[n+2m] = delta(m)
  for (let m = 1; 2 * m < L; m++) {
    let s = 0;
    for (let n = 0; n + 2 * m < L; n++) s += w.h[n] * w.h[n + 2 * m];
    check(Math.abs(s) < 1e-14, `${w.name}: even-shift orthogonality m=${m}`);
  }
  // vanishing moments: sum n^k g[n] = 0 for k < p, and not for k = p
  for (let k = 0; k <= w.p; k++) {
    let s = 0, scale = 0;
    for (let n = 0; n < L; n++) { s += n ** k * w.g[n]; scale += Math.abs(n ** k * w.g[n]); }
    if (k < w.p) check(Math.abs(s) < 1e-11 * Math.max(1, scale), `${w.name}: moment ${k} vanishes (${s})`);
    else check(Math.abs(s) > 1e-6 * scale, `${w.name}: moment ${k} does not vanish`);
  }
}
// db2 closed form
{
  const r3 = Math.sqrt(3), d = 4 * Math.SQRT2, h = [(1 + r3) / d, (3 + r3) / d, (3 - r3) / d, (1 - r3) / d];
  check(maxDiff(h, (WAVELETS.db2 as OrthoWavelet).h) < 1e-15, "db2 closed form");
}

// ---- one-level filter bank vs explicit matrix (brute force), and perfect reconstruction ----
for (const w of ORTH) for (const mode of ["periodic", "symmetric", "zero"] as Boundary[]) for (const n of [8, 16, 64, 10]) {
  if (mode === "periodic" && n < w.h.length) continue;
  const x = randVec(n);
  const st = analysisStep(x, w, mode);
  // brute force: build extended signal explicitly
  const ext = (i: number) => {
    if (mode === "periodic") return x[((i % n) + n) % n];
    if (mode === "zero") return i >= 0 && i < n ? x[i] : 0;
    let j = i;
    while (j < 0 || j >= n) j = j < 0 ? -1 - j : 2 * n - 1 - j;
    return x[j];
  };
  let bf = 0;
  for (let j = 0; j < st.a.length; j++) {
    const k = st.k0 + j;
    let a = 0, d = 0;
    for (let m = 0; m < w.h.length; m++) { a += w.h[m] * ext(2 * k + m); d += w.g[m] * ext(2 * k + m); }
    bf = Math.max(bf, Math.abs(a - st.a[j]), Math.abs(d - st.d[j]));
  }
  check(bf < 1e-12, `${w.name} ${mode} n=${n}: analysis = brute force`);
  const y = synthesisStep(st, w, mode);
  check(maxDiff(x, y) < 1e-10, `${w.name} ${mode} n=${n}: perfect reconstruction (${maxDiff(x, y)})`);
  if (mode === "periodic") check(Math.abs(energy(x) - energy(st.a) - energy(st.d)) < 1e-10, `${w.name} periodic: energy`);
  const br = branchSignals(x, w, mode);
  const sum = br.lowBack.map((v, i) => v + br.highBack[i]);
  check(maxDiff(sum, x) < 1e-10, `${w.name} ${mode}: branches add up to x`);
  // multi-level
  if (n % 8) continue;
  const dec = wavedec(x, w, 3, mode);
  check(maxDiff(waverec(dec.steps, w, mode), x) < 1e-9, `${w.name} ${mode} n=${n}: 3-level reconstruction`);
}

// dropping the high-pass branch aliases: low branch alone is not x, but is a projection (idempotent)
{
  const w = WAVELETS.db2 as OrthoWavelet, x = testSignal("chirp", 64);
  const st = analysisStep(x, w, "periodic");
  const low = synthesisStep(st, w, "periodic", { dropHigh: true });
  check(maxDiff(low, x) > 0.1, "dropHigh changes the signal");
  const low2 = synthesisStep(analysisStep(low, w, "periodic"), w, "periodic", { dropHigh: true });
  check(maxDiff(low, low2) < 1e-12, "low branch is a projection");
  const wrong = synthesisStep(st, w, "periodic", { wrongFilters: true });
  check(maxDiff(wrong, x) > 0.1, "wrong synthesis filters do not reconstruct");
}

// ---- Mallat-layout 1D transform: all wavelets, PR and energy ----
for (const name of ALL) {
  const w = WAVELETS[name];
  for (const n of [16, 64, 256]) {
    const x = randVec(n, 10), lev = Math.min(4, Math.log2(n) - 2);
    const c = dwt(x, w, lev), y = idwt(c, w, lev);
    check(maxDiff(x, y) < 1e-9, `${name} n=${n}: dwt/idwt round trip (${maxDiff(x, y)})`);
    if (w.kind === "orth") check(Math.abs(energy(c) - energy(x)) < 1e-8 * energy(x), `${name}: Parseval`);
    else check(Math.abs(energy(c) / energy(x) - 1) < (name === "cdf97" ? 0.25 : 0.6), `${name}: roughly energy preserving (${energy(c) / energy(x)})`);
  }
  // polynomial of degree < p gives zero details away from the borders (lifting: interior)
}
// periodic dwt of an orthogonal wavelet equals analysisStep(periodic)
for (const w of ORTH) {
  const x = randVec(32), c = dwt(x, w, 1), st = analysisStep(x, w, "periodic");
  check(maxDiff(c.subarray(0, 16), st.a) < 1e-13 && maxDiff(c.subarray(16), st.d) < 1e-13, `${w.name}: dwt level 1 = analysisStep`);
}
// vanishing moments on a sampled polynomial: details vanish away from the wrap-around
for (const w of ORTH) {
  const n = 64, L = w.h.length;
  for (let deg = 0; deg <= w.p; deg++) {
    const x = Float64Array.from({ length: n }, (_, i) => ((i - n / 2) / 8) ** deg);
    const st = analysisStep(x, w, "periodic");
    let m = 0;
    for (let k = 0; 2 * k + L - 1 < n; k++) m = Math.max(m, Math.abs(st.d[k]));
    const scale = Math.max(...Array.from(x, Math.abs));
    if (deg < w.p) check(m < 1e-11 * scale, `${w.name}: degree ${deg} gives zero interior details (${m})`);
    else check(m > 1e-8 * scale, `${w.name}: degree ${deg} gives nonzero details`);
  }
}

// ---- 2D ----
for (const name of ALL) {
  const w = WAVELETS[name];
  for (const [W, H, lev] of [[64, 64, 3], [32, 16, 2], [256, 256, 5]] as [number, number, number][]) {
    const img = randVec(W * H, 100), c = dwt2(img, W, H, w, lev), y = idwt2(c, W, H, w, lev);
    check(maxDiff(img, y) < 1e-8, `${name} ${W}x${H}: dwt2 round trip (${maxDiff(img, y)})`);
    if (w.kind === "orth") check(Math.abs(energy(c) - energy(img)) < 1e-8 * energy(img), `${name} 2D Parseval`);
  }
}
// 2x2 Haar block formulas: LL = (a+b+c+d)/2, V = (a-b+c-d)/2, H = (a+b-c-d)/2, D = (a-b-c+d)/2
{
  const [a, b, c, d] = [3, 7, 1, 4];
  const r = dwt2([a, b, c, d], 2, 2, WAVELETS.haar, 1);
  check(maxDiff(r, [(a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2]) < 1e-14, "2x2 Haar subbands");
  const rv = subbandRect(1, "V", 2, 2), rh = subbandRect(1, "H", 2, 2);
  check(rv.x === 1 && rv.y === 0 && rh.x === 0 && rh.y === 1, "subband rectangles");
  check(bandAt(1, 0, 2, 2, 1).band === "V" && bandAt(0, 1, 2, 2, 1).band === "H" && bandAt(1, 1, 2, 2, 1).band === "D" && bandAt(0, 0, 2, 2, 1).band === "LL", "bandAt");
  // a vertical edge lights only V
  const e = dwt2([0, 10, 0, 10], 2, 2, WAVELETS.haar, 1);
  check(e[1] !== 0 && e[2] === 0 && e[3] === 0, "vertical edge: only V");
}

// ---- Haar by hand: the board example ----
{
  const x = [4, 6, 10, 12, 8, 6, 5, 5];
  const st = haarStates(x, "half", 3);
  check(maxDiff(st[3], [7, 1, -3, 1, -1, -1, 1, 0]) < 1e-14, `board example: ${Array.from(st[3])}`);
  check(maxDiff(st[1], [5, 11, 7, 5, -1, -1, 1, 0]) < 1e-14, "board example level 1");
  check(maxDiff(haarInverse(st[3], "half", 3), x) < 1e-14, "board example inverse");
  const o = haarStates(x, "orth", 3);
  check(Math.abs(energy(o[3]) - energy(x)) < 1e-10, "orthonormal Haar keeps energy");
  check(Math.abs(energy(st[3]) - energy(x)) > 1, "half-normalised Haar does not");
  check(maxDiff(haarInverse(o[3], "orth", 3), x) < 1e-13, "orthonormal Haar inverse");
  // Haar matrix is orthogonal; equals dwt with Haar
  for (const N of [2, 4, 8, 16]) {
    const M = haarMatrix(N);
    let m = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += M[i * N + k] * M[j * N + k];
      m = Math.max(m, Math.abs(s - (i === j ? 1 : 0)));
    }
    check(m < 1e-13, `Haar matrix ${N}: H H^T = I`);
  }
  const lev = 3, c = dwt(x, WAVELETS.haar, lev);
  check(maxDiff(c, o[3]) < 1e-13, "haarStates(orth) = dwt(haar)");
  check(levelOf(0, 8, 3) === 0 && levelOf(1, 8, 3) === 3 && levelOf(2, 8, 3) === 2 && levelOf(4, 8, 3) === 1, "levelOf");
  // a spike touches log2 N + 1 coefficients
  for (let p = 0; p < 8; p++) {
    const e = new Float64Array(8); e[p] = 1;
    const nz = haarStates(e, "orth", 3)[3].filter(v => Math.abs(v) > 1e-12).length;
    check(nz === 4, `spike at ${p}: 4 nonzero Haar coefficients`);
  }
}

// ---- integer lifting ----
for (let t = 0; t < 200; t++) {
  const n = 2 * (1 + Math.floor(R.unif() * 20));
  const x = Int32Array.from({ length: n }, () => Math.floor(R.unif() * 512) - 256);
  check(maxDiff(unliftHaarInt(liftHaarInt(x)), x) === 0, "integer Haar lifting round trip");
  const lev = 1, c = dwt(x, WAVELETS.legall53, lev, true);
  check(c.every(Number.isInteger), "5/3 integer coefficients are integers");
  check(maxDiff(idwt(c, WAVELETS.legall53, lev, true), x) === 0, "5/3 integer round trip");
}
{
  const c = liftHaarInt([4, 6, 10, 12, 8, 6, 5, 5]);
  check(maxDiff(c, [5, 11, 7, 5, 2, 2, -2, 0]) === 0, `integer Haar board example: ${Array.from(c)}`);
}

// ---- cascade algorithm ----
{
  const w = WAVELETS.db2 as OrthoWavelet, it = 8, r = cascade(w, it), dt = 2 ** -it;
  let s = 0, s2 = 0, sp = 0, sp2 = 0;
  for (let k = 0; k < r.phi.length; k++) { s += r.phi[k] * dt; s2 += r.phi[k] ** 2 * dt; sp += r.psi[k] * dt; sp2 += r.psi[k] ** 2 * dt; }
  check(Math.abs(s - 1) < 1e-9, `cascade: integral of phi = 1 (${s})`);
  check(Math.abs(s2 - 1) < 0.02, `cascade: integral of phi^2 ≈ 1 (${s2})`);
  check(Math.abs(sp) < 1e-9, `cascade: integral of psi = 0 (${sp})`);
  check(Math.abs(sp2 - 1) < 0.02, `cascade: integral of psi^2 ≈ 1 (${sp2})`);
  // known value: db2 phi(1) = (1 + sqrt3)/2
  // known value: db2 phi(1) = (1 + sqrt3)/2; the cascade converges to it (slowly: phi is rough)
  const e8 = Math.abs(r.phi[2 ** it] - (1 + Math.sqrt(3)) / 2), r14 = cascade(w, 14), e14 = Math.abs(r14.phi[2 ** 14] - (1 + Math.sqrt(3)) / 2);
  check(e8 < 0.1 && e14 < e8 / 4, `db2 cascade converges to phi(1) = (1+√3)/2 (${e8}, ${e14})`);
  const hr = cascade(WAVELETS.haar as OrthoWavelet, 4);
  check(hr.phi.slice(0, 16).every(v => Math.abs(v - 1) < 1e-12), "Haar phi = 1 on [0,1)");
  check(hr.psi.slice(0, 8).every(v => Math.abs(v - 1) < 1e-12) && hr.psi.slice(8, 16).every(v => Math.abs(v + 1) < 1e-12), "Haar psi = +1, -1");
  // psi from phi by the two-scale relation, evaluated on the grid of the last iteration
  const r6 = cascade(w, 6), r5 = cascade(w, 5);
  let m = 0;
  for (let k = 0; k < r6.psi.length; k++) {
    let s = 0;
    for (let n = 0; n < 4; n++) { const j = k - n * 32; if (j >= 0 && j < r5.phi.length) s += Math.SQRT2 * w.g[n] * r5.phi[j]; }
    m = Math.max(m, Math.abs(s - r6.psi[k]));
  }
  check(m < 1e-12, "cascade psi = two-scale relation on phi");
  // psi is smooth-ish: no sample-to-sample sign flipping (neighbouring samples close)
  let jump = 0;
  for (let k = 1; k < r.psi.length; k++) jump = Math.max(jump, Math.abs(r.psi[k] - r.psi[k - 1]));
  check(jump < 0.5, `db2 psi has no fine-scale oscillation (${jump})`);
}

// ---- thresholding ----
{
  const w = [5, -1.5, 0.4, -3, 2.1, -0.2];
  check(maxDiff(w.map(v => hard(v, 2)), [5, 0, 0, -3, 2.1, 0]) < 1e-14, "hard threshold");
  check(maxDiff(w.map(v => soft(v, 2)), [3, 0, 0, -1, 0.1, 0]) < 1e-12, "soft threshold");
  check(Math.abs(universalThreshold(1, 65536) - 4.7096) < 1e-4, "universal threshold 256^2");
  const g = Float64Array.from({ length: 20001 }, () => R.norm() * 3);
  check(Math.abs(madSigma(g) - 3) < 0.1, `MAD sigma (${madSigma(g)})`);
  // soft thresholding minimises 1/2 (y - x)^2 + t |x| (brute force on a grid)
  for (const y of [-3, -0.5, 0.2, 1.7, 4]) {
    let best = Infinity, bx = 0;
    for (let x = -6; x <= 6; x += 1e-4) { const e = 0.5 * (y - x) ** 2 + 1.2 * Math.abs(x); if (e < best) { best = e; bx = x; } }
    check(Math.abs(bx - soft(y, 1.2)) < 2e-4, `soft = argmin (y=${y})`);
  }
}

// ---- denoising does something sensible ----
{
  const W = 64, img = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) img[y * W + x] = (x - 32) ** 2 + (y - 30) ** 2 < 300 ? 200 : 50;
  const noisy = img.map(v => v + 20 * R.norm());
  const mse = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - img[i]) ** 2; return s / a.length; };
  const d1 = denoise2(noisy, W, W, WAVELETS.haar, 4, 40, "soft", 1);
  const d16 = denoise2(noisy, W, W, WAVELETS.haar, 4, 40, "soft", 16);
  check(mse(d1.out) < mse(noisy) / 3, "denoise2 reduces the error");
  check(mse(d16.out) < mse(d1.out), "cycle spinning helps");
  const z = denoise2(img, W, W, WAVELETS.db4, 3, 0, "hard", 4);
  check(maxDiff(z.out, img) < 1e-9, "threshold 0 with cycle spinning returns the image");
  const s1 = testSignal("blocks", 256), n1 = s1.map(v => v + 0.5 * R.norm());
  const r1 = denoise1(n1, WAVELETS.haar, 5, 0, "soft", 4);
  check(maxDiff(r1, n1) < 1e-10, "denoise1 with t = 0 is the identity");
}

// ---- keep largest, tail errors, real Fourier ----
{
  const x = randVec(64);
  const k = keepLargest(x, 10);
  check(k.filter(v => v !== 0).length === 10, "keepLargest keeps 10");
  const te = tailErrors(x);
  const err = Math.sqrt(energy(x.map((v, i) => v - k[i])) / energy(x));
  check(Math.abs(te[10] - err) < 1e-12 && te[0] === 1 && te[64] === 0, "tailErrors");
  const s = testSignal("heavisine", 256), f = realFourier(s);
  check(Math.abs(energy(f) - energy(s)) < 1e-9 * energy(s), "real Fourier Parseval");
  check(maxDiff(realFourierInverse(f), s) < 1e-10, "real Fourier inverse");
  // a sine needs 1 real Fourier coefficient (sin at 4 cycles)
  const sn = testSignal("sine", 256), fs = realFourier(sn);
  check(fs.filter(v => Math.abs(v) > 1e-9).length === 1, "sine: one Fourier coefficient");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
