// Tests for the Ising / MRF day: lib/mrf.ts and lib/smooth.ts against brute force and the
// by-hand values of the units ising/, mcmc/, mrf-denoise/, gmrf/. Run alone:
//   node_modules/.bin/esbuild demos/test/ising-test.ts --bundle --platform=node --log-level=warning | node
import {
  lattice, enumerate, energy, stateSpins, randomSpins, metropolisSweep, gibbsSweep, icmSweep, meanFieldSweep,
  ringZ, transferMatrix, exactMAP, toSegProblem, flipCosts, fieldFromCosts, segmentationEnergy, conditionalPlus,
  acceptance, onsagerM, TC, magnetisation, stateIndex, disagreements, Lattice, Field,
} from "../src/lib/mrf";
import { segEnergy } from "../src/lib/graph";
import { thomas, gmrf1d, tv1d, huber1d, energy1d, tv2d, tv2dValue, weightedPathSolve, gmrfMatrix } from "../src/lib/smooth";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- enumeration: the 2x2 lattice by hand ----
{
  const L = lattice(2, 2, false);
  check(L.ep.length === 4, "2x2 free has 4 edges");
  for (const T of [1, 2, 4]) {
    const en = enumerate(L, 1, 0, T);
    const Z = 2 * Math.exp(4 / T) + 12 + 2 * Math.exp(-4 / T);
    check(close(en.Z, Z, 1e-12), `2x2 Z at T=${T}: ${en.Z} vs ${Z}`);
    check(en.levels.length === 3 && en.levels[0].count === 2 && en.levels[1].count === 12 && en.levels[2].count === 2, "2x2 degeneracies 2/12/2");
    let s = 0; for (const v of en.p) s += v;
    check(close(s, 1, 1e-12), "probabilities sum to 1");
  }
  const en1 = enumerate(L, 1, 0, 1);
  check(close(en1.levels[0].p, 2 * Math.exp(4) / (2 * Math.exp(4) + 12 + 2 * Math.exp(-4)), 1e-12), "p(aligned) at T = 1");
  // with a field, the all-+ state wins over all-minus by e^{2*4*h/T}
  const enh = enumerate(L, 1, 0.5, 1);
  check(close(enh.p[15] / enh.p[0], Math.exp(4), 1e-9), "field h = 0.5 favours all-+ by e^4");
}

// ---- 3x3 periodic: energy symmetries, brute-force Z ----
{
  const L = lattice(3, 3, true);
  check(L.ep.length === 18, "3x3 periodic has 18 edges");
  const en = enumerate(L, 1, 0.3, 2);
  let Z = 0;
  for (let k = 0; k < 512; k++) Z += Math.exp(-energy(L, stateSpins(k, 9), 1, 0.3).total / 2);
  check(close(en.Z, Z, 1e-12), "3x3 Z equals the direct sum");
  check(close(enumerate(L, 1, 0, 0.05).levels[0].p, 1, 1e-6), "T -> 0: all mass on the two ground states");
  const hot = enumerate(L, 1, 0, 1e6);
  check(Math.abs(hot.p[0] - 1 / 512) < 1e-6, "T -> infinity: uniform");
}

// ---- transfer matrix vs enumeration on rings ----
for (const n of [3, 4, 5, 8]) for (const [J, h, T] of [[1, 0, 1], [1, 0.4, 1.5], [-0.7, 0.2, 0.8]]) {
  const L = lattice(n, 1, true);
  const en = enumerate(L, J, h, T);
  check(close(ringZ(n, J, h, T), en.Z, 1e-10), `ring n=${n} J=${J} h=${h} T=${T}: ${ringZ(n, J, h, T)} vs ${en.Z}`);
  // trace of M^n directly
  const M = transferMatrix(J, h, T);
  let P = [[1, 0], [0, 1]];
  for (let i = 0; i < n; i++) P = [[P[0][0] * M[0][0] + P[0][1] * M[1][0], P[0][0] * M[0][1] + P[0][1] * M[1][1]], [P[1][0] * M[0][0] + P[1][1] * M[1][0], P[1][0] * M[0][1] + P[1][1] * M[1][1]]];
  check(close(P[0][0] + P[1][1], en.Z, 1e-10), "tr M^n = Z");
}

// ---- conditional probability and acceptance (mcmc by-hand values) ----
check(close(conditionalPlus(2, 2), 1 / (1 + Math.exp(-2)), 1e-12), "conditional (+,+,+,-) at T=2");
check(close(acceptance(4, 2), Math.exp(-2)) && close(acceptance(4, 0.5), Math.exp(-8)) && acceptance(-1, 1) === 1, "Metropolis acceptance");
{ // conditional from the enumeration equals the logistic formula
  const L = lattice(3, 3, false), T = 1.7, J = 0.8, h = 0.2;
  const en = enumerate(L, J, h, T);
  const s = stateSpins(0b101100110, 9);
  const kp = stateIndex(s) | (1 << 4), km = stateIndex(s) & ~(1 << 4);
  const u = J * (s[1] + s[3] + s[5] + s[7]) + h;
  check(close(en.p[kp] / (en.p[kp] + en.p[km]), conditionalPlus(u, T), 1e-10), "local conditional = logistic of the local field");
}

// ---- Gibbs and Metropolis sample the Gibbs distribution (3x3) ----
function chainHistogram(kind: "gibbs" | "metro", L: Lattice, J: number, h: Field, T: number, sweeps: number, seed: number): Float64Array {
  const r = rng(seed), s = randomSpins(L.n, r), H = new Float64Array(1 << L.n);
  for (let k = 0; k < 200; k++) kind === "gibbs" ? gibbsSweep(L, s, J, h, T, r, "random") : metropolisSweep(L, s, J, h, T, r, "random");
  for (let k = 0; k < sweeps; k++) {
    kind === "gibbs" ? gibbsSweep(L, s, J, h, T, r, "random") : metropolisSweep(L, s, J, h, T, r, "random");
    H[stateIndex(s)]++;
  }
  for (let k = 0; k < H.length; k++) H[k] /= sweeps;
  return H;
}
{
  const L = lattice(3, 3, true), J = 1, h = 0.2, T = 3;
  const en = enumerate(L, J, h, T);
  for (const kind of ["gibbs", "metro"] as const) {
    const H = chainHistogram(kind, L, J, h, T, 200000, 7);
    // compare per energy level (total variation distance)
    const lev = new Map<number, number>();
    for (let k = 0; k < H.length; k++) { const e = Math.round(en.E[k] * 1e6) / 1e6; lev.set(e, (lev.get(e) ?? 0) + H[k]); }
    let tvd = 0;
    for (const l of en.levels) tvd += Math.abs((lev.get(Math.round(l.E * 1e6) / 1e6) ?? 0) - l.p);
    check(tvd / 2 < 0.01, `${kind}: level histogram within 0.01 of exact (TVD ${(tvd / 2).toFixed(4)})`);
    let tvs = 0; for (let k = 0; k < H.length; k++) tvs += Math.abs(H[k] - en.p[k]);
    check(tvs / 2 < 0.03, `${kind}: state histogram TVD ${(tvs / 2).toFixed(4)}`);
  }
  // checkerboard order also samples correctly
  const r = rng(3), s = randomSpins(9, r), H = new Float64Array(512);
  const L2 = lattice(3, 3, false), en2 = enumerate(L2, 0.6, -0.1, 1.5);
  for (let k = 0; k < 200000; k++) { metropolisSweep(L2, s, 0.6, -0.1, 1.5, r, "checkerboard"); H[stateIndex(s)]++; }
  let tv = 0; for (let k = 0; k < 512; k++) tv += Math.abs(H[k] / 200000 - en2.p[k]);
  check(tv / 2 < 0.03, `checkerboard Metropolis TVD ${(tv / 2).toFixed(4)}`);
}

// ---- ICM never raises the energy; exact MAP by min-cut equals brute force ----
{
  const r = rng(11);
  for (let trial = 0; trial < 60; trial++) {
    const w = 3, h = 3, L = lattice(w, h, false);
    const J = 0.2 + 1.5 * r.unif();
    const f = Float64Array.from({ length: 9 }, () => 3 * (r.unif() - 0.5));
    let best = Infinity;
    for (let k = 0; k < 512; k++) best = Math.min(best, energy(L, stateSpins(k, 9), J, f).total);
    const map = exactMAP(w, h, J, f);
    check(close(map.energy, best, 1e-9), `min-cut MAP = brute force (${map.energy} vs ${best})`);
    // seg energy = Ising energy + J * #edges
    const P = toSegProblem(w, h, J, f);
    const Ls = Uint8Array.from(map.s, v => (v > 0 ? 1 : 0));
    check(close(segEnergy(P, Ls).total - J * L.ep.length, map.energy, 1e-9), "segmentation energy = Ising energy + const");
    // ICM monotone, ends above or at the MAP
    const s = randomSpins(9, r);
    let e = energy(L, s, J, f).total, ok = true;
    for (let k = 0; k < 20; k++) { icmSweep(L, s, J, f); const e2 = energy(L, s, J, f).total; if (e2 > e + 1e-12) ok = false; e = e2; }
    check(ok, "ICM energy non-increasing");
    check(e >= best - 1e-9, "ICM energy >= exact minimum");
  }
  // larger: min-cut <= annealing <= ... on a 24x24 noisy square
  const W = 24, N = W * W, truth = new Uint8Array(N);
  for (let y = 6; y < 18; y++) for (let x = 6; x < 18; x++) truth[y * W + x] = 1;
  const y = Uint8Array.from(truth, v => (r.unif() < 0.2 ? 1 - v : v));
  const { D0, D1 } = flipCosts(y, 0.2), f = fieldFromCosts(D0, D1), L = lattice(W, W, false), J = 0.8;
  const map = exactMAP(W, W, J, f);
  const s = Int8Array.from(y, v => (v ? 1 : -1));
  for (let k = 0; k < 30; k++) icmSweep(L, s, J, f);
  check(energy(L, s, J, f).total >= map.energy - 1e-9, "ICM >= min-cut on 24x24");
  const a = Int8Array.from(y, v => (v ? 1 : -1)), ra = rng(5);
  for (let k = 0; k < 400; k++) gibbsSweep(L, a, J, f, 3 * 0.985 ** k, ra);
  check(energy(L, a, J, f).total >= map.energy - 1e-9, "annealing >= min-cut on 24x24");
  check(energy(L, a, J, f).total - map.energy < 3, `slow annealing comes close to the min-cut energy (gap ${(energy(L, a, J, f).total - map.energy).toFixed(3)})`);
  // segmentation energy with flip costs agrees with the Ising energy up to a constant
  const e1 = segmentationEnergy(L, map.s, D0, D1, 2 * J).total, e2 = segmentationEnergy(L, s, D0, D1, 2 * J).total;
  check(close(e1 - e2, map.energy - energy(L, s, J, f).total, 1e-9), "segmentation form and Ising form differ by a constant");
  // mean field reaches a fixed point
  const m = Float64Array.from(f, v => Math.tanh(v));
  let d = 1;
  for (let k = 0; k < 500 && d > 1e-12; k++) d = meanFieldSweep(L, m, J, f, 1);
  check(d < 1e-10, "mean field converges");
  let res = 0;
  for (let p = 0; p < N; p++) res = Math.max(res, Math.abs(m[p] - Math.tanh(J * (L.nb.slice(4 * p, 4 * p + 4).reduce((t, q) => t + (q >= 0 ? m[q] : 0), 0)) + f[p])));
  check(res < 1e-9, "mean field fixed point m = tanh(J sum m + h)");
}

// ---- mrf-denoise by hand: 1x3, y = (+,-,+), eps = 0.2, J = 1 ----
{
  const L = lattice(3, 1, false), y = [1, 0, 1], { D0, D1 } = flipCosts(y, 0.2), f = fieldFromCosts(D0, D1);
  check(close(f[0], 0.5 * Math.log(4), 1e-12) && close(f[1], -0.5 * Math.log(4), 1e-12), "h = 1/2 log 4");
  let best = -1, bestE = Infinity;
  for (let k = 0; k < 8; k++) { const e = energy(L, stateSpins(k, 3), 1, f).total; if (e < bestE) { bestE = e; best = k; } }
  check(best === 7, "MAP of (+,-,+) with J = 1 is (+,+,+)");
  const s = Int8Array.from([1, -1, 1]);
  icmSweep(L, s, 1, f, "raster");
  check(s[0] === -1 && s[1] === -1 && s[2] === -1, "raster ICM from y ends in (-,-,-)");
}

// ---- Onsager ----
check(close(TC, 2.269185314213022, 1e-12), "T_c");
check(onsagerM(2.5) === 0 && onsagerM(1) > 0.99 && onsagerM(2.2) > 0.6 && onsagerM(2.2) < 0.8, "Onsager magnetisation");
{ // a 64x64 lattice at T = 1.5 started from all + stays near Onsager's m
  const L = lattice(64, 64, true), s = new Int8Array(L.n).fill(1), r = rng(2);
  let acc = 0;
  for (let k = 0; k < 300; k++) { metropolisSweep(L, s, 1, 0, 1.5, r); if (k >= 100) acc += magnetisation(s); }
  check(Math.abs(acc / 200 - onsagerM(1.5)) < 0.01, `64x64 at T=1.5: m=${(acc / 200).toFixed(4)} vs ${onsagerM(1.5).toFixed(4)}`);
  check(disagreements(L, s) >= 0, "disagreements");
}

// ---- smooth.ts ----
{
  // Thomas vs dense solve
  const r = rng(9);
  for (let t = 0; t < 20; t++) {
    const n = 2 + Math.floor(r.unif() * 12), lam = 0.1 + 5 * r.unif();
    const y = Float64Array.from({ length: n }, () => r.norm());
    const x = gmrf1d(y, lam), M = gmrfMatrix(n, lam);
    let err = 0;
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += M[i * n + j] * x[j]; err = Math.max(err, Math.abs(s - y[i])); }
    check(err < 1e-10, "Thomas solves (I + lambda L) x = y");
  }
  check(thomas([0, 1], [2, 2], [1, 0], [3, 3]).every((v, i) => close(v, [1, 1][i])), "thomas 2x2");
  const g = gmrf1d([0, 3, 0], 1);
  check(close(g[0], 0.75) && close(g[1], 1.5) && close(g[2], 0.75), "gmrf by hand (0.75, 1.5, 0.75)");
  const tv = tv1d([0, 0, 3, 3], 1);
  check([0.5, 0.5, 2.5, 2.5].every((v, i) => close(tv[i], v, 1e-12)), `tv by hand: ${Array.from(tv)}`);
  check(tv1d([1, 2, 3], 10).every(v => close(v, 2, 1e-12)), "large lambda: TV gives the mean");
  // TV against a brute-force minimiser (projected gradient on the dual)
  for (let t = 0; t < 30; t++) {
    const n = 2 + Math.floor(r.unif() * 30), lam = 0.05 + 2 * r.unif();
    const y = Float64Array.from({ length: n }, () => 3 * r.norm());
    const x = tv1d(y, lam);
    // dual: x = y - D^T z, |z_i| <= lam; projected gradient
    const z = new Float64Array(n - 1);
    const prim = (zz: Float64Array) => { const v = Float64Array.from(y); for (let i = 0; i < n - 1; i++) { v[i] += zz[i]; v[i + 1] -= zz[i]; } return v; };
    for (let it = 0; it < 20000; it++) {
      const v = prim(z);
      for (let i = 0; i < n - 1; i++) z[i] = Math.max(-lam, Math.min(lam, z[i] + 0.25 * (v[i + 1] - v[i])));
    }
    const ref = prim(z);
    const e1 = energy1d(x, y, "tv", lam).total, e2 = energy1d(ref, y, "tv", lam).total;
    check(e1 <= e2 + 1e-7, `Condat TV energy <= reference (${e1} vs ${e2}, n=${n})`);
    let dmax = 0; for (let i = 0; i < n; i++) dmax = Math.max(dmax, Math.abs(x[i] - ref[i]));
    check(dmax < 1e-4, `Condat TV = reference minimiser (max diff ${dmax.toExponential(2)})`);
  }
  // Huber: energy not above quadratic or TV solution evaluated with the Huber energy; small delta ~ TV
  for (let t = 0; t < 10; t++) {
    const n = 40, lam = 0.5 + 3 * r.unif(), delta = 0.2 + r.unif();
    const y = Float64Array.from({ length: n }, (_, i) => (i > 20 ? 2 : 0) + 0.3 * r.norm());
    const xh = huber1d(y, lam, delta, 200);
    const eh = energy1d(xh, y, "huber", lam, delta).total;
    check(eh <= energy1d(gmrf1d(y, lam), y, "huber", lam, delta).total + 1e-9, "Huber IRLS beats the quadratic solution");
    // stationarity: perturbations do not lower the energy
    let ok = true;
    for (let i = 0; i < n; i++) for (const e of [1e-4, -1e-4]) { const x2 = Float64Array.from(xh); x2[i] += e; if (energy1d(x2, y, "huber", lam, delta).total < eh - 1e-10) ok = false; }
    check(ok, "Huber IRLS is a local (= global, convex) minimum");
  }
  check(weightedPathSolve([1, 1, 1], 3).every(v => close(v, 1)), "constant signal unchanged");
  // 2D TV: image constant along y equals 1D TV per row
  const w = 20, h = 6, img = new Float64Array(w * h), row = Float64Array.from({ length: w }, (_, i) => (i >= 8 ? 3 : 0) + 0.5 * Math.sin(i));
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) img[j * w + i] = row[i];
  const x2 = tv2d(img, w, h, 0.7, 3000), x1 = tv1d(row, 0.7);
  let d2 = 0; for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) d2 = Math.max(d2, Math.abs(x2[j * w + i] - x1[i]));
  check(d2 < 2e-3, `tv2d matches tv1d on a row-constant image (max diff ${d2.toExponential(2)})`);
  check(tv2dValue(x2, w, h) < tv2dValue(img, w, h), "tv2d lowers the TV");
}

console.log(`ising-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
