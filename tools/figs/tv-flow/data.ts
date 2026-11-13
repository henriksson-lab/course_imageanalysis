// Image data for tools/figs/tv-flow/figs.py (run by it; prints JSON). Uses lib/smooth.ts and lib/pde.ts.
import { tv2d, tv1d } from "../../../demos/src/lib/smooth";
import { gaussianHeat, nonlinearStep, testCard, addNoise, psnr255, tvFlowStep } from "../../../demos/src/lib/pde";
const r = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 100) / 100);
const out: Record<string, unknown> = {};
const N = 128;
// TV flow as repeated ROF steps (implicit TV-flow steps of size h)
function flow(u0: Float64Array, T: number, h = 2): Float64Array { let u: Float64Array = Float64Array.from(u0); for (let t = 0; t < T - 1e-9; t += h) u = tv2d(u, N, N, Math.min(h, T - t), 80); return u; }
// discs of radius 4, 8, 16 (contrast 40 on 60) and a square
const D = new Float64Array(N * N).fill(60);
const discs: [number, number, number][] = [[20, 24, 4], [52, 24, 8], [96, 30, 16]];
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { for (const [cx, cy, rr] of discs) if ((x - cx) ** 2 + (y - cy) ** 2 <= rr * rr) D[y * N + x] = 100; if (x >= 24 && x < 72 && y >= 64 && y < 112) D[y * N + x] = 100; }
out.discs0 = r(D);
const times = [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 280, 320];
const contrast: number[][] = [];
let u: Float64Array = Float64Array.from(D), tPrev = 0;
const snaps: Record<string, number[]> = {};
for (const t of times) { if (t > tPrev) { u = flow(u, t - tPrev); tPrev = t; } const med = Float64Array.from(u).sort()[(N * N) >> 1]; contrast.push(discs.map(([cx, cy]) => u[cy * N + cx] - med)); if (t === 40 || t === 120 || t === 240) snaps["t" + t] = r(u); }
out.times = times; out.contrast = contrast; out.snaps = snaps;
// TV vs PM vs heat, each removing as much as the noise (SD(y − u) = σn = 15), as in the tvflow demo
const truth = testCard().truth, sn = 15, y = addNoise(truth, sn, 6);
const res = (u: ArrayLike<number>) => { let s = 0, s2 = 0; for (let i = 0; i < y.length; i++) { const d = y[i] - u[i]; s += d; s2 += d * d; } return Math.sqrt(s2 / y.length - (s / y.length) ** 2); };
let lo = 0, hi = 4 * sn + 10, tv: Float64Array = y;
for (let k = 0; k < 8; k++) { const m = (lo + hi) / 2; tv = tv2d(y, N, N, m, 80); if (res(tv) < sn) lo = m; else hi = m; }
const lam = (lo + hi) / 2; tv = tv2d(y, N, N, lam, 200);
let a = 0, b = 10, heat: Float64Array = y;
for (let k = 0; k < 22; k++) { const m = (a + b) / 2; heat = gaussianHeat(y, N, N, m); if (res(heat) < sn) a = m; else b = m; }
const sig = (a + b) / 2; heat = gaussianHeat(y, N, N, sig);
let pm: Float64Array = Float64Array.from(y), pmT = 0;
for (let k = 0; k < 4000 && res(pm) < sn; k++) { pm = nonlinearStep(pm, N, N, 0.05, "g2", sn, 1); pmT += 0.05; }
out.cmp = { y: r(y), truth: r(truth), tv: r(tv), heat: r(heat), pm: r(pm), lam, sig, pmT, sd: [res(tv), res(pm), res(heat)], psnr: [psnr255(tv, truth), psnr255(pm, truth), psnr255(heat, truth)] };
// 1D: ROF by hand and flow vs ROF on plateaus
out.rof2 = r(tv1d([0, 0, 0, 6, 6, 0, 0, 0], 2)); out.rof6 = r(tv1d([0, 0, 0, 6, 6, 0, 0, 0], 6));
// flow in 1D of the same row (eps small)
let f: Float64Array = Float64Array.from([0, 0, 0, 6, 6, 0, 0, 0]); for (let t = 0; t < 2 - 1e-9; t += 0.004) f = tvFlowStep(f, 8, 1, 0.004, 0.01);
out.flow2 = r(f);
console.log(JSON.stringify(out));
