// Data for the extra dwt2d slide figures, computed with the demo's own code (lib/wavelet.ts,
// dwt2d/images.ts, lib/fft.ts) so that the numbers match the dwt2d demo.
// Usage (called by figs.py): node data.js cell.f64 outdir
import * as fs from "fs";
import * as path from "path";
import { WAVELETS, WaveletName, dwt2, idwt2, keepLargest } from "../../../demos/src/lib/wavelet";
import { fft2d } from "../../../demos/src/lib/fft";
import { psnr } from "../../../demos/src/lib/metrics";
import { N, synthImage, blockDct } from "../../../demos/src/dwt2d/images";

const [cellPath, out] = process.argv.slice(2);
const cell = new Float64Array(fs.readFileSync(cellPath).buffer.slice(0));
const save = (name: string, a: Float64Array) => fs.writeFileSync(path.join(out, name + ".f64"), Buffer.from(a.buffer));
const info: Record<string, number> = {};

// one level of Haar on the cell, and the 4-level db2 pyramid (demo preset "pyramid")
save("cell_haar1", dwt2(cell, N, N, WAVELETS.haar, 1));
save("cell_db2_4", dwt2(cell, N, N, WAVELETS.db2, 4));

// keep the K largest at 1 % (cell, 4 levels): Haar, db4, Fourier, 8x8 DCT
const pct = 1, k = Math.round(pct / 100 * N * N);
for (const wn of ["haar", "db4"] as WaveletName[]) {
  const c = dwt2(cell, N, N, WAVELETS[wn], 4);
  const r = idwt2(keepLargest(c, k), N, N, WAVELETS[wn], 4);
  save("keep_" + wn, r); info["psnr_" + wn] = psnr(cell, r, 255);
}
{
  const re = Float64Array.from(cell), im = new Float64Array(N * N);
  fft2d(re, im, N);
  const mag = Float64Array.from(re, (v, i) => Math.hypot(v, im[i]));
  const order = Uint32Array.from({ length: N * N }, (_, i) => i).sort((a, b) => mag[b] - mag[a]);
  const r2 = new Float64Array(N * N), i2 = new Float64Array(N * N);
  for (let j = 0; j < k; j++) { r2[order[j]] = re[order[j]]; i2[order[j]] = im[order[j]]; }
  fft2d(r2, i2, N, true);
  save("keep_fourier", r2); info.psnr_fourier = psnr(cell, r2, 255);
  const d = blockDct(cell, N);
  const ad = Float64Array.from(d, Math.abs);
  const od = Uint32Array.from({ length: N * N }, (_, i) => i).sort((a, b) => ad[b] - ad[a]);
  const kd = new Float64Array(N * N);
  for (let j = 0; j < k; j++) kd[od[j]] = d[od[j]];
  const rd = blockDct(kd, N, true);
  save("keep_dct", rd); info.psnr_dct = psnr(cell, rd, 255);
}

// shift variance: square, Haar, 2 levels, shifted 1 px to the right (as the demo)
const sq = synthImage("square");
const sh = new Float64Array(N * N);
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) sh[y * N + x] = sq[y * N + ((x - 1 + N) % N)];
const c0 = dwt2(sq, N, N, WAVELETS.haar, 2), c1 = dwt2(sh, N, N, WAVELETS.haar, 2);
save("square", sq); save("square_c0", c0); save("square_c1", c1);
let dd = 0, tt = 0;
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (x >= N >> 2 || y >= N >> 2) { const i = y * N + x; dd += (c1[i] - c0[i]) ** 2; tt += c0[i] ** 2; }
info.shift_change = Math.sqrt(dd / tt);
fs.writeFileSync(path.join(out, "info.json"), JSON.stringify(info, null, 1));
