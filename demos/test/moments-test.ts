// Tests for lib/moments.ts (moments elective) and the moments-demo modes it feeds:
// closed forms for rectangles and discs, central-from-raw = direct, order 2 = lib/label.ts
// and the labelling day's regionProps, Hu invariants under rotation (exact for polygons,
// within a tolerance for digitised shapes), φ7 under mirroring, complex moments rotate by
// e^{i(p−q)α}, Legendre/discrete/Zernike bases orthogonal (documented tolerances), exact
// reconstructions of sums of basis functions, Legendre-from-geometric = direct, Zernike
// recurrence = factorial formula. Run alone:
//   node_modules/.bin/esbuild demos/test/moments-test.ts --bundle --platform=node --log-level=warning | node
import {
  rawMoments, centralFromRaw, centralMomentsDirect, normalisedMoments, huInvariants, huFromComplex, complexMoments,
  rotateMoments, flipV, polygonMoments, idx, unitFrame, legendre, legendreAveraged, unitCentres, discreteOrtho,
  legendreMoments, legendreReconstruct, monomialReconstruct, monomialGram1d, gaussSolve, zernikeRadial,
  zernikeRadialFactorial, discGrid, zernikeMoments, zernikeReconstruct, zernikeImage, relError, pose,
  polygonCoverageEvenOdd, flusserInvariants3, shiftMoments, binom,
} from "../src/lib/moments";
import { label2d, centroids } from "../src/lib/label";
import { regionProps } from "../src/moments/regionprops";
import { rasterise, rotatePoly, mirrorPoly, ellipsePoly, Poly } from "../src/lib/shapecanvas";
import { solve } from "../src/lib/linalg";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const relClose = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol * Math.max(Math.abs(a), Math.abs(b), 1e-300);

// ---- 1. polygon moments (Green's theorem) of a rectangle and a disc ----
{
  const a = 3, b = 5, N = 6;
  const m = polygonMoments([[0, 0], [a, 0], [a, b], [0, b]], N);
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++)
    check(close(m[idx(N, p, q)], a ** (p + 1) * b ** (q + 1) / ((p + 1) * (q + 1)), 1e-12), `rectangle m${p}${q}`);
  // clockwise orientation gives the same
  const m2 = polygonMoments([[0, 0], [0, b], [a, b], [a, 0]], N);
  check(close(m2[idx(N, 2, 1)], m[idx(N, 2, 1)]), "rectangle, other orientation");
  // a fine regular polygon approximates the disc: m00 = πr², μ20 = πr⁴/4, μ40 = πr⁶/8
  const r = 2, n = 4000, disc = Array.from({ length: n }, (_, i) => [r * Math.cos(2 * Math.PI * i / n), r * Math.sin(2 * Math.PI * i / n)] as [number, number]);
  const d = polygonMoments(disc, 4);
  check(relClose(d[0], Math.PI * r * r, 1e-5), "disc area");
  check(relClose(d[idx(4, 2, 0)], Math.PI * r ** 4 / 4, 1e-5), "disc mu20");
  check(relClose(d[idx(4, 4, 0)], Math.PI * r ** 6 / 8, 1e-5), "disc mu40");
  check(relClose(d[idx(4, 2, 2)], Math.PI * r ** 6 / 24, 1e-5), "disc mu22");
}

// ---- 2. pixel as a square: a pixel-aligned rectangle is integrated exactly ----
{
  const w = 20, h = 16, N = 5, f = new Float64Array(w * h);
  for (let y = 3; y <= 11; y++) for (let x = 4; x <= 15; x++) f[y * w + x] = 1;
  const sq = rawMoments(f, w, h, N, { pixel: "square" });
  const ex = polygonMoments([[3.5, 2.5], [15.5, 2.5], [15.5, 11.5], [3.5, 11.5]], N);
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) check(close(sq[idx(N, p, q)], ex[idx(N, p, q)], 1e-11), `square pixels exact m${p}${q}`);
  // μ20 (square) − μ20 (point) = m00/12
  const pt = centralFromRaw(rawMoments(f, w, h, N), N), sc = centralFromRaw(sq, N);
  check(close(sc[idx(N, 2, 0)] - pt[idx(N, 2, 0)], pt[0] / 12, 1e-10), "square adds m00/12 to mu20");
  check(close(sc[idx(N, 0, 2)] - pt[idx(N, 0, 2)], pt[0] / 12, 1e-10), "square adds m00/12 to mu02");
  // a single pixel: μ20 = 0 as a point, 1/12 as a square
  const one = new Float64Array(9); one[4] = 1;
  check(centralFromRaw(rawMoments(one, 3, 3, 2), 2)[idx(2, 2, 0)] === 0, "one pixel point mu20 = 0");
  check(close(centralFromRaw(rawMoments(one, 3, 3, 2, { pixel: "square" }), 2)[idx(2, 2, 0)], 1 / 12), "one pixel square mu20 = 1/12");
}

// ---- 3. central from raw = direct; grey images; frames ----
{
  const r = rng(5), w = 23, h = 17, N = 5, f = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) f[i] = r.unif() < 0.4 ? r.unif() * 3 : 0;
  for (const fr of [undefined, unitFrame(w, h), unitFrame(w, h, true)]) for (const pixel of ["point", "square"] as const) {
    const a = centralFromRaw(rawMoments(f, w, h, N, { frame: fr, pixel }), N), b = centralMomentsDirect(f, w, h, N, { frame: fr, pixel });
    for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) check(close(a[idx(N, p, q)], b[idx(N, p, q)], 1e-8), `central from raw p${p} q${q}`);
  }
  // shifting moments = moments of the shifted image
  const m = rawMoments(f, w, h, N), s = shiftMoments(m, N, 2, -1);
  const g = new Float64Array(w * h); // image moved by (+2, −1) within bounds: build by sum directly
  let ok = true;
  for (let p = 0; p <= 3; p++) for (let q = 0; p + q <= 3; q++) {
    let t = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) t += (x + 2) ** p * (y - 1) ** q * f[y * w + x];
    ok = ok && close(t, s[idx(N, p, q)], 1e-10);
  }
  check(ok, "shiftMoments"); void g;
  check(binom(10, 3) === 120 && binom(4, 0) === 1 && binom(3, 5) === 0, "binom");
}

// ---- 4. order 2 agrees with lib/label.ts and the labelling day's regionProps ----
{
  const shapes: Poly[] = [ellipsePoly(30, 28, 20, 9, 64, 0.7), [[5, 5], [20, 8], [12, 30]]];
  const mask = rasterise(shapes, 64, 64);
  const lab = label2d(mask, 64, 64, 8), cents = centroids(lab), props = regionProps(lab.data, 64, 64, lab.count, { perimeters: false });
  for (let l = 1; l <= lab.count; l++) {
    const one = new Float64Array(64 * 64);
    for (let i = 0; i < one.length; i++) one[i] = lab.data[i] === l ? 1 : 0;
    const m = rawMoments(one, 64, 64, 2), mu = centralFromRaw(m, 2), c = cents[l], rp = props.find(p => p.label === l)!;
    check(m[0] === c.n && close(m[idx(2, 1, 0)] / m[0], c.x) && close(m[idx(2, 0, 1)] / m[0], c.y), `label ${l}: area and centroid = label.ts`);
    check(close(mu[idx(2, 2, 0)], rp.mu20, 1e-9) && close(mu[idx(2, 1, 1)], rp.mu11, 1e-9) && close(mu[idx(2, 0, 2)], rp.mu02, 1e-9), `label ${l}: mu = regionProps`);
  }
}

// ---- 5. invariance: translation, scale, rotation (exact for polygons), mirror ----
const pear: Poly = (() => {
  // an asymmetric "pear": a disc of radius 10 joined to a smaller one, plus a bump
  const out: Poly = [];
  for (let i = 0; i < 200; i++) {
    const t = 2 * Math.PI * i / 200, r = 10 + 4 * Math.cos(t) + 1.5 * Math.sin(2 * t) + 0.8 * Math.cos(3 * t);
    out.push([32 + r * Math.cos(t), 32 + r * Math.sin(t)]);
  }
  return out;
})();
const N4 = 4;
function invariantsOfPoly(p: Poly) {
  const m = polygonMoments(p, N4), mu = centralFromRaw(m, N4), eta = normalisedMoments(mu, N4);
  return { m, mu, eta, hu: huInvariants(eta, N4), c: complexMoments(eta, N4) };
}
{
  const A = invariantsOfPoly(pear);
  const moved = invariantsOfPoly(pear.map(([x, y]) => [x + 7, y - 3] as [number, number]));
  check(close(moved.mu[idx(N4, 3, 0)], A.mu[idx(N4, 3, 0)], 1e-9) && !close(moved.m[idx(N4, 1, 0)], A.m[idx(N4, 1, 0)], 1e-3), "translation: mu same, m differs");
  const scaled = invariantsOfPoly(pear.map(([x, y]) => [2 * x, 2 * y] as [number, number]));
  check(close(scaled.mu[idx(N4, 2, 0)], 16 * A.mu[idx(N4, 2, 0)], 1e-9), "scale 2: mu20 × 2^4");
  check(close(scaled.mu[idx(N4, 3, 0)], 32 * A.mu[idx(N4, 3, 0)], 1e-9), "scale 2: mu30 × 2^5");
  for (let k = 0; k < 7; k++) check(relClose(scaled.hu[k], A.hu[k], 1e-8), `scale: hu ${k + 1}`);
  for (const ang of [0.3, 1.1, 2.5, -0.7]) {
    const R = invariantsOfPoly(rotatePoly(pear, ang));
    for (let k = 0; k < 7; k++) check(relClose(R.hu[k], A.hu[k], 1e-8), `rotation ${ang}: hu ${k + 1} (${R.hu[k]} vs ${A.hu[k]})`);
    // moments of the rotated polygon = rotateMoments of the original (y down: rotatePoly turns CCW on
    // screen, i.e. by −ang in the (x, y-down) frame)
    const rm = rotateMoments(A.mu, N4, -ang);
    for (const [p, q] of [[2, 0], [1, 1], [3, 0], [1, 2], [2, 2]]) check(close(rm[idx(N4, p, q)], R.mu[idx(N4, p, q)], 1e-8), `rotateMoments ${p}${q}`);
    // complex moments of η turn by e^{i(p−q)α}, α in the frame of the table
    const a = -ang;
    for (const [p, q] of [[2, 0], [3, 0], [2, 1], [1, 2], [1, 1], [3, 1]]) {
      const c0r = A.c.re[idx(N4, p, q)], c0i = A.c.im[idx(N4, p, q)], ph = (p - q) * a;
      const er = c0r * Math.cos(ph) - c0i * Math.sin(ph), ei = c0r * Math.sin(ph) + c0i * Math.cos(ph);
      check(close(R.c.re[idx(N4, p, q)], er, 1e-8) && close(R.c.im[idx(N4, p, q)], ei, 1e-8), `complex c${p}${q} rotates by e^{i(p−q)α}`);
    }
  }
  const M = invariantsOfPoly(mirrorPoly(pear, "x"));
  for (let k = 0; k < 6; k++) check(relClose(M.hu[k], A.hu[k], 1e-8), `mirror: hu ${k + 1} unchanged`);
  check(Math.abs(A.hu[6]) > 1e-12 && relClose(M.hu[6], -A.hu[6], 1e-8), `mirror: hu 7 flips (${A.hu[6]} → ${M.hu[6]})`);
  // complex form = real formulas (φ1..φ6 exactly; φ7 up to the frame's orientation)
  const hc = huFromComplex(A.c, N4);
  for (let k = 0; k < 6; k++) check(relClose(hc[k], A.hu[k], 1e-9), `hu ${k + 1} complex = real`);
  check(relClose(Math.abs(hc[6]), Math.abs(A.hu[6]), 1e-9), "hu 7 complex = real up to sign");
  // complex conjugation under flipV
  const cf = complexMoments(flipV(A.eta, N4), N4);
  check(close(cf.re[idx(N4, 3, 0)], A.c.re[idx(N4, 3, 0)]) && close(cf.im[idx(N4, 3, 0)], -A.c.im[idx(N4, 3, 0)]), "flipV conjugates c30");
  // Flusser's set: rotation invariant, and φ3 follows from it (Hu's set is dependent)
  const F0 = flusserInvariants3(A.c, N4), F1 = flusserInvariants3(invariantsOfPoly(rotatePoly(pear, 0.9)).c, N4);
  for (let k = 0; k < 6; k++) check(relClose(F0[k], F1[k], 1e-7) || Math.abs(F0[k] - F1[k]) < 1e-14, `flusser ${k} rotation invariant`);
  // pose normalisation: after rotating by −θ, μ11 = 0, μ20 ≥ μ02 and μ30 > 0
  const P = pose(A.m, N4), rmu = rotateMoments(A.mu, N4, -P.theta);
  check(Math.abs(rmu[idx(N4, 1, 1)]) < 1e-8 * rmu[idx(N4, 2, 0)] && rmu[idx(N4, 2, 0)] >= rmu[idx(N4, 0, 2)] && rmu[idx(N4, 3, 0)] > 0, "pose: principal axis on x, mu30 > 0");
  const P2 = pose(polygonMoments(rotatePoly(pear, 1.3), N4), N4), rmu2 = rotateMoments(centralFromRaw(polygonMoments(rotatePoly(pear, 1.3), N4), N4), N4, -P2.theta);
  check(close(rmu2[idx(N4, 3, 0)], rmu[idx(N4, 3, 0)], 1e-8) && close(rmu2[idx(N4, 1, 2)], rmu[idx(N4, 1, 2)], 1e-8), "pose: same normalised moments after rotation");
  const sq = pose(polygonMoments([[0, 0], [4, 0], [4, 4], [0, 4]], N4), N4);
  check(sq.ambiguous, "pose: a square is ambiguous");
  // disc and square: φ1 = 1/(2π) and 1/6
  const disc = Array.from({ length: 4000 }, (_, i) => [Math.cos(2 * Math.PI * i / 4000), Math.sin(2 * Math.PI * i / 4000)] as [number, number]);
  check(relClose(invariantsOfPoly(disc).hu[0], 1 / (2 * Math.PI), 1e-5), "disc phi1 = 1/2π");
  check(relClose(invariantsOfPoly([[0, 0], [3, 0], [3, 3], [0, 3]]).hu[0], 1 / 6, 1e-12), "square phi1 = 1/6");
}

// ---- 6. digitised shapes: invariance within a tolerance, improving with size ----
{
  const errAt = (scale: number) => {
    const base = pear.map(([x, y]) => [(x - 32) * scale + 64, (y - 32) * scale + 64] as [number, number]);
    const hu = (p: Poly) => { const f = polygonCoverageEvenOdd([p], 128, 128, 4); const m = rawMoments(f, 128, 128, N4); return huInvariants(normalisedMoments(centralFromRaw(m, N4), N4), N4); };
    const a = hu(base), b = hu(rotatePoly(base, 0.5));
    return Math.max(Math.abs(a[0] - b[0]) / a[0], Math.abs(a[1] - b[1]) / a[1]);
  };
  const small = errAt(0.6), large = errAt(3.5);
  check(large < 0.005, `digitised large pear: phi1, phi2 within 0.5 % under rotation (${large})`);
  check(small > large, `smaller shape has larger rotation error (${small} > ${large})`);
}

// ---- 7. Legendre, discrete polynomials: orthogonality and reconstruction ----
{
  // sampled Legendre on n pixel centres: Σ P_j P_k Δx vs 2/(2k+1) δ_jk; deviation grows with order
  const n = 64, xs = unitCentres(n), L = legendre(12, xs), dx = 2 / n;
  const dev = (N: number) => { let worst = 0; for (let j = 0; j <= N; j++) for (let k = 0; k <= N; k++) { let s = 0; for (let i = 0; i < n; i++) s += L[j][i] * L[k][i] * dx; worst = Math.max(worst, Math.abs(s - (j === k ? 2 / (2 * k + 1) : 0)) / Math.sqrt((2 / (2 * j + 1)) * (2 / (2 * k + 1)))); } return worst; };
  check(dev(4) < 0.01, `sampled Legendre on 64 pixels orthogonal within 1 % to order 4 (${dev(4)})`);
  check(dev(12) > dev(4), "sampled Legendre: deviation grows with order");
  // closed forms
  check(close(L[2][5], (3 * xs[5] ** 2 - 1) / 2, 1e-14) && close(L[3][7], (5 * xs[7] ** 3 - 3 * xs[7]) / 2, 1e-14), "P2, P3 closed form");
  // averaged Legendre: average of P_2 over [a, b] = (b³ − a³)/(2(b − a)) − 1/2
  const av = legendreAveraged(3, [0.3], 0.1);
  check(close(av[2][0], ((0.4 ** 3 - 0.2 ** 3) / 0.2) / 2 - 0.5, 1e-13), "averaged P2");
  // discrete orthonormal: exact
  const T = discreteOrtho(20, 40);
  let worst = 0;
  for (let j = 0; j <= 20; j++) for (let k = 0; k <= 20; k++) { let s = 0; for (let i = 0; i < 40; i++) s += T[j][i] * T[k][i]; worst = Math.max(worst, Math.abs(s - (j === k ? 1 : 0))); }
  check(worst < 1e-12, `discrete polynomials orthonormal (${worst})`);
  // reconstruction of a sum of basis functions: exact for the discrete basis
  const w = 32, f = new Float64Array(w * w), Td = discreteOrtho(6, w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) f[y * w + x] = 2 * Td[1][x] * Td[2][y] - Td[3][x] * Td[0][y] + 0.5 * Td[0][x] * Td[0][y];
  const a = legendreMoments(f, w, w, 6, "discrete");
  check(close(a[idx(6, 1, 2)], 2, 1e-12) && close(a[idx(6, 3, 0)], -1, 1e-12) && close(a[idx(6, 0, 0)], 0.5, 1e-12) && Math.abs(a[idx(6, 2, 2)]) < 1e-12, "discrete moments of a sum of basis functions");
  const rec = legendreReconstruct(a, 6, w, w, 3, "discrete");
  check(relError(rec, f) < 1e-12, "discrete reconstruction exact at N = 3");
  // sampled Legendre: P1(x)P2(y) is recovered to within the grid's non-orthogonality
  const g = new Float64Array(w * w), Ls = legendre(4, unitCentres(w));
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) g[y * w + x] = Ls[1][x] * Ls[2][y];
  const as = legendreMoments(g, w, w, 4, "sampled");
  check(Math.abs(as[idx(4, 1, 2)] - 1) < 0.01, `sampled Legendre moment of P1·P2 ≈ 1 (${as[idx(4, 1, 2)]})`);
  check(relError(legendreReconstruct(as, 4, w, w, 3, "sampled"), g) < 0.01, "sampled reconstruction of P1·P2 within 1 %");
  // Legendre moments from geometric moments = direct (sampled)
  const r = rng(11), h2 = new Float64Array(w * w);
  for (let i = 0; i < h2.length; i++) h2[i] = r.unif();
  const Nn = 5, m = rawMoments(h2, w, w, Nn, { frame: unitFrame(w, w), area: true }), aDirect = legendreMoments(h2, w, w, Nn, "sampled");
  // coefficients of P_n in powers of x from the recurrence
  const C: number[][] = [[1], [0, 1]];
  for (let k = 1; k < Nn; k++) { const nxt = new Array(k + 2).fill(0); for (let j = 0; j <= k; j++) nxt[j + 1] += (2 * k + 1) * C[k][j] / (k + 1); for (let j = 0; j < k; j++) nxt[j] -= k * C[k - 1][j] / (k + 1); C.push(nxt); }
  let okL = true;
  for (let p = 0; p <= Nn; p++) for (let q = 0; p + q <= Nn; q++) {
    let s = 0;
    for (let k = 0; k <= p; k++) for (let l = 0; l <= q; l++) s += C[p][k] * C[q][l] * m[idx(Nn, k, l)];
    s *= (2 * p + 1) * (2 * q + 1) / 4;
    okL = okL && close(s, aDirect[idx(Nn, p, q)], 1e-9);
  }
  check(okL, "Legendre moments from geometric moments = direct");
  // monomial reconstruction = discrete projection at low order (well conditioned), breaks down later
  const img = rasterise([ellipsePoly(16, 15, 10, 6, 40, 0.4)], w, w), fimg = Float64Array.from(img);
  const proj = (N: number) => legendreReconstruct(legendreMoments(fimg, w, w, N, "discrete"), N, w, w, N, "discrete");
  check(relError(monomialReconstruct(fimg, w, w, 4), proj(4)) < 1e-8, "monomial = discrete projection at N = 4");
  const e30 = relError(monomialReconstruct(fimg, w, w, 30), proj(30));
  check(!(e30 < 1e-3), `monomial reconstruction departs from the projection at N = 30 (${e30})`);
  // gaussSolve = linalg solve on a well-conditioned system
  const A = [4, 1, 0, 1, 3, 1, 0, 1, 2], bb = [1, 2, 3], x1 = gaussSolve(A, bb, 3)!, x2 = solve(A, bb, 3)!;
  check(x1.every((v, i) => close(v, x2[i], 1e-12)), "gaussSolve = solve");
  // Gram matrix of the monomials on [−1, 1]: 2/(j+k+1) for even j + k
  const G = monomialGram1d(2, 0, true);
  check(close(G[0], 2) && close(G[2], 2 / 3) && G[1] === 0 && close(G[8], 2 / 5), "continuous Gram matrix 3 × 3");
}

// ---- 8. Zernike ----
{
  let worst = 0;
  for (let n = 0; n <= 20; n++) for (let m = n % 2; m <= n; m += 2) for (const rho of [0, 0.13, 0.5, 0.77, 0.99, 1]) worst = Math.max(worst, Math.abs(zernikeRadial(n, m, rho) - zernikeRadialFactorial(n, m, rho)));
  check(worst < 1e-8, `Zernike recurrence = factorial formula for n ≤ 20 (${worst})`);
  check(close(zernikeRadial(4, 0, 0.5), 6 * 0.5 ** 4 - 6 * 0.25 + 1) && close(zernikeRadial(3, 1, 0.7), 3 * 0.343 - 1.4) && close(zernikeRadial(2, 2, 0.3), 0.09), "small R_nm closed forms");
  check(zernikeRadial(10, 4, 1) === 1 || close(zernikeRadial(10, 4, 1), 1), "R_nm(1) = 1");
  // continuous orthogonality of the radial part: ∫ R20 R00 ρ dρ = 0, ∫ R20² ρ dρ = 1/6
  const K = 20000; let s1 = 0, s2 = 0;
  for (let i = 0; i < K; i++) { const r = (i + 0.5) / K, R = zernikeRadial(2, 0, r); s1 += R * r / K; s2 += R * R * r / K; }
  check(Math.abs(s1) < 1e-8 && close(s2, 1 / 6, 1e-7), "radial orthogonality with weight ρ");
  // moments of an image that is a combination of V_nm: recovered within the disc discretisation
  const S = 96, g = discGrid(S, S), f = new Float64Array(S * S);
  const v42 = zernikeImage(4, 2, S), v31 = zernikeImage(3, 1, S), v00 = zernikeImage(0, 0, S);
  for (const i of g.pix) f[i] = 0.5 * v00.re[i] + 2 * v42.re[i] + 1 * v31.im[i];   // 2 Re V42 = V42 + V4,−2 → Z42 = 1
  const Z = zernikeMoments(f, g, 6), get = (n: number, m: number) => { const k = Z.nm.findIndex(([a, b]) => a === n && b === m); return [Z.re[k], Z.im[k]]; };
  const z42 = get(4, 2), z31 = get(3, 1), z00 = get(0, 0), z20 = get(2, 0);
  check(Math.abs(z42[0] - 1) < 0.02 && Math.abs(z42[1]) < 0.02, `Z42 ≈ 1 (${z42})`);
  check(Math.abs(z31[0]) < 0.02 && Math.abs(z31[1] + 0.5) < 0.02, `Z31 ≈ −0.5 i (${z31})`);
  check(Math.abs(z00[0] - 0.5) < 0.02 && Math.hypot(z20[0], z20[1]) < 0.02, `Z00 ≈ 0.5, Z20 ≈ 0 (${z00}, ${z20})`);
  const rec = zernikeReconstruct(Z, g, 6);
  check(relError(rec, f) < 0.02, `reconstruction of a sum of V_nm (${relError(rec, f)})`);
  // incremental reconstruction: the last partial equals the full reconstruction
  let last: Float64Array | null = null;
  zernikeReconstruct(Z, g, 6, (_n, img) => { last = img; });
  check(relError(last!, rec) < 1e-12, "incremental reconstruction");
  // rotation: |Z_nm| of a digitised shape and its rotated copy agree within a few per cent; phases turn by −mα... (y up: by −m·α for a CCW rotation of the image)
  const S2 = 128, base = pear.map(([x, y]) => [(x - 32) * 3 + 64, (y - 32) * 3 + 64] as [number, number]);
  const g2 = discGrid(S2, S2), ang = 0.6;
  const Za = zernikeMoments(polygonCoverageEvenOdd([base], S2, S2, 4), g2, 6), Zb = zernikeMoments(polygonCoverageEvenOdd([rotatePoly(base, ang, [64, 64])], S2, S2, 4), g2, 6);
  let worstMag = 0, worstPh = 0;
  Za.nm.forEach(([, m], k) => {
    const a = Math.hypot(Za.re[k], Za.im[k]), b = Math.hypot(Zb.re[k], Zb.im[k]);
    if (a < 0.02 * Math.hypot(Za.re[0], Za.im[0])) return;
    worstMag = Math.max(worstMag, Math.abs(a - b) / a);
    if (m > 0) { let d = Math.atan2(Zb.im[k], Zb.re[k]) - Math.atan2(Za.im[k], Za.re[k]) + m * ang; d = Math.atan2(Math.sin(d), Math.cos(d)); worstPh = Math.max(worstPh, Math.abs(d)); }
  });
  check(worstMag < 0.02, `|Z_nm| rotation invariant within 2 % (${worstMag})`);
  check(worstPh < 0.05, `phase of Z_nm turns by −mα (${worstPh})`);
}

console.log(`moments-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
