// Data for the ista/ figures, computed with the demos' own code (demos/src/lib). Run by figs.py:
//   node data.js <outdir> <cell.f64>
// Writes deblur1d.json (1D profile restored by Tikhonov, TV and wavelet ℓ1) and inpaint.json
// (micrograph crop with missing pixels and scratches, restored with db4 ℓ1 + cycle spinning).
import { writeFileSync, readFileSync } from "fs";
import { blurOp1d, fista, waveletProx2d, maskOp, Vec, LinOp } from "../../../demos/src/lib/sparse";
import { WAVELETS, denoise1 } from "../../../demos/src/lib/wavelet";
import { tv1d } from "../../../demos/src/lib/smooth";
import { psnr } from "../../../demos/src/lib/metrics";
import { rng } from "../../../demos/src/lib/random";
import { randomMask, scratchMask } from "../../../demos/src/sparserestore/images";

declare const process: { argv: string[] };
const out = process.argv[2], cellPath = process.argv[3];

// ---- 1D deblurring: step, ramp, spike, small bump ----
const N = 256;
const x = new Float64Array(N);
for (let i = 0; i < N; i++) {
  if (i >= 30 && i < 80) x[i] = 1;
  if (i >= 100 && i < 150) x[i] = (i - 100) / 50;
  if (i >= 150 && i < 170) x[i] = 1;
  if (i === 200) x[i] = 1.5;
  if (i >= 220 && i < 232) x[i] = 0.25;   // small, low-contrast feature
}
const B = blurOp1d(N, 3), r = rng(4);
const y = Float64Array.from(B.forward(x), v => v + 0.02 * r.norm());
const rmse = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < N; i++) s += (a[i] - x[i]) ** 2; return Math.sqrt(s / N); };
// Tikhonov with a gradient penalty: FISTA with the exact prox (I + τλ DᵀD)⁻¹ is awkward in 1D without FFT,
// so use gradient descent on the smooth objective ½‖Bx − y‖² + (λ/2)‖Dx‖² (converged)
function tikhonov(lam: number): Vec {
  let z: Vec = Float64Array.from(y);
  const step = 1 / (1 + 4 * lam);
  for (let it = 0; it < 4000; it++) {
    const g = B.adjoint(Float64Array.from(B.forward(z), (v, i) => v - y[i]));
    for (let i = 0; i < N; i++) { const d = 2 * z[i] - z[(i + 1) % N] - z[(i + N - 1) % N]; g[i] += lam * d; }
    for (let i = 0; i < N; i++) z[i] -= step * g[i];
  }
  return z;
}
const sweep = (f: (l: number) => Vec, lams: number[]) => { let best = { l: 0, x: new Float64Array(N) as Vec, e: Infinity }; for (const l of lams) { const z = f(l); const e = rmse(z); if (e < best.e) best = { l, x: z, e }; } return best; };
const lamGrid = (a: number, b: number, n: number) => Array.from({ length: n }, (_, k) => a * Math.pow(b / a, k / (n - 1)));
const tik = sweep(tikhonov, lamGrid(1e-4, 1, 17));
const tvRun = (l: number) => fista(B, y, { lambda: l, iters: 600, step: 1, prox: (v, t) => tv1d(v, t * l), history: false }).x;
const tv = sweep(tvRun, lamGrid(1e-4, 0.03, 13));
const wRun = (l: number) => fista(B, y, { lambda: l, iters: 600, step: 1, prox: (v, t) => denoise1(v, WAVELETS.db4, 5, t * l, "soft", 8), history: false }).x;
const wv = sweep(wRun, lamGrid(1e-4, 0.03, 13));
writeFileSync(`${out}/deblur1d.json`, JSON.stringify({
  x: Array.from(x), y: Array.from(y), tik: Array.from(tik.x), tv: Array.from(tv.x), wav: Array.from(wv.x),
  lam: { tik: tik.l, tv: tv.l, wav: wv.l }, rmse: { y: rmse(y), tik: tik.e, tv: tv.e, wav: wv.e },
  bump: { truth: 0.25, tik: Math.max(...tik.x.slice(220, 232)), tv: Math.max(...tv.x.slice(220, 232)), wav: Math.max(...wv.x.slice(220, 232)) },
}));

// ---- 2D inpainting of a micrograph crop ----
const cell = new Float64Array(readFileSync(cellPath).buffer.slice(0));
const n = 128, crop = new Float64Array(n * n);
for (let yy = 0; yy < n; yy++) for (let xx = 0; xx < n; xx++) crop[yy * n + xx] = cell[(yy + 64) * 256 + xx + 64];
const res: Record<string, unknown> = { truth: Array.from(crop) };
for (const [name, mask] of [["random", randomMask(n, 0.5)], ["scratch", scratchMask(n)]] as [string, Uint8Array][]) {
  const A: LinOp = maskOp(mask);
  const yy = A.forward(crop);
  let s = 0, c = 0; for (let i = 0; i < n * n; i++) if (mask[i]) { s += yy[i]; c++; }
  const x0 = Float64Array.from(yy, (v, i) => (mask[i] ? v : s / c));
  const lam = 0.5;
  const rec = fista(A, yy, { lambda: lam, iters: 150, step: 1, x0, prox: (v, t) => waveletProx2d(v, n, n, WAVELETS.db4, 4, t * lam, 4), history: false }).x;
  res[name] = { mask: Array.from(mask), rec: Array.from(rec), psnrObs: psnr(x0, crop, 255), psnrRec: psnr(rec, crop, 255) };
}
writeFileSync(`${out}/inpaint.json`, JSON.stringify(res));
console.log("deblur λ", tik.l, tv.l, wv.l, "rmse", rmse(y), tik.e, tv.e, wv.e);
