// Image data for tools/figs/bilateral-nlm/figs.py (run by it; prints JSON). Uses lib/pde.ts, lib/denoise.ts.
import { bilateralWeights, nlmWeights, jointBilateral, testCard, addNoise, psnr255 } from "../../../demos/src/lib/pde";
import { bilateral, nonLocalMeans } from "../../../demos/src/lib/denoise";
const r = (a: ArrayLike<number>, d = 100) => Array.from(a, v => Math.round(v * d) / d);
const out: Record<string, unknown> = {};
const N = 128, card = testCard(), truth = card.truth, y = addNoise(truth, 10, 9);
out.noisy = r(y); out.truth = r(truth);
// weight maps
const bil = (x: number, yy: number) => { const b = bilateralWeights(y, N, N, x, yy, 2, 25); return { w: r(b.product, 1000), size: b.size, neff: b.neff }; };
const nlm = (x: number, yy: number) => { const m = nlmWeights(y, N, N, x, yy, 2, 10, 15); return { w: r(m.weights, 1000), size: m.size, neff: m.neff, self: m.selfWeight }; };
out.wb = { flat: bil(22, 22), edge: bil(6, 22), corner: bil(6, 6), tj: bil(82, 104) };
out.wn = { flat: nlm(22, 22), lattice: nlm(82, 20), unique: nlm(46, 76), stripes: nlm(94, 68) };
// N_eff maps
const nb = new Float64Array(N * N), nn = new Float64Array(N * N);
for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) { nb[yy * N + x] = bilateralWeights(y, N, N, x, yy, 2, 25).neff; nn[yy * N + x] = nlmWeights(y, N, N, x, yy, 2, 7, 15).neff; }
out.neffB = r(nb, 10); out.neffN = r(nn, 10);
// sigma_r = 0.5, 2, 8 sigma_n
out.sr = [5, 20, 80].map(s => { const u = bilateral(y, N, N, 2, s); return { s, u: r(u), psnr: psnr255(u, truth) }; });
// self-weight: h = 5 without and with the fix, h = 15
const nlmFix = (hh: number) => { const u = new Float64Array(N * N); for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) u[yy * N + x] = nlmWeights(y, N, N, x, yy, 2, 7, hh, true).estimate; return u; };
const h5 = nonLocalMeans(y, N, N, 2, 7, 5), h5f = nlmFix(5), h15 = nonLocalMeans(y, N, N, 2, 7, 15);
out.self = { h5: r(h5), h5f: r(h5f), h15: r(h15), psnr: [psnr255(h5, truth), psnr255(h5f, truth), psnr255(h15, truth), psnr255(y, truth)] };
// iterated bilateral
let u: Float64Array = y; const it: Record<string, number[]> = {};
for (let k = 1; k <= 10; k++) { u = bilateral(u, N, N, 2, 30); if (k === 1 || k === 3 || k === 10) it["k" + k] = r(u); }
out.iter = it;
// joint bilateral: a guide with an extra step across the ramp (x = 50)
const guide = Float64Array.from(truth);
for (let yy = 98; yy < 126; yy++) for (let x = 34; x < 66; x++) guide[yy * N + x] = x < 50 ? 150 : 90;
out.guide = r(guide); out.joint = r(jointBilateral(y, guide, N, N, 4, 20)); out.bil4 = r(bilateral(y, N, N, 4, 20));
console.log(JSON.stringify(out));
