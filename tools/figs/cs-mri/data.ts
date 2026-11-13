// Data for the cs-mri/ figures, computed with the csmri demo's code (demos/src/lib/sparse.ts):
// masks, zero-filled and compressed-sensing reconstructions of the Shepp–Logan phantom and of
// the phantom with faint lesions. Run by figs.py:  node data.js <outdir>
import { writeFileSync } from "fs";
import { kspaceMask, partialFourierOp2d, waveletProx2d, LESION_PHANTOM, maskFraction, Vec, MaskKind } from "../../../demos/src/lib/sparse";
import { sheppLogan } from "../../../demos/src/lib/synth";
import { WAVELETS } from "../../../demos/src/lib/wavelet";
import { fft2d } from "../../../demos/src/lib/fft";
import { psnr, ssim } from "../../../demos/src/lib/metrics";
import { rng } from "../../../demos/src/lib/random";

declare const process: { argv: string[] };
const out = process.argv[2];
const n = 256;

function cs(truth: Vec, kind: MaskKind, accel: number, lam: number, iters = 150, noise = 0.002) {
  const mask = kspaceMask(n, kind, accel, rng(17), { q: 3 });
  const A = partialFourierOp2d(mask, n), rn = rng(5), y = A.forward(truth);
  for (let i = 0; i < y.length; i++) if (y[i] !== 0) y[i] += noise * rn.norm() / Math.SQRT2;
  const zf = A.adjoint(y);
  let x: Vec = Float64Array.from(zf), z: Vec = Float64Array.from(zf), t = 1;
  for (let k = 0; k < iters; k++) {
    const r = A.forward(z); for (let i = 0; i < r.length; i++) r[i] = y[i] - r[i];
    const g = A.adjoint(r), v = Float64Array.from(z, (zi, i) => zi + g[i]);
    const xn = lam > 0 ? waveletProx2d(v, n, n, WAVELETS.haar, 5, lam) : v;
    const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, b = (t - 1) / tn;
    for (let i = 0; i < v.length; i++) z[i] = xn[i] + b * (xn[i] - x[i]);
    x = xn; t = tn;
  }
  return {
    mask: Array.from(mask), zf: Array.from(zf), cs: Array.from(x), frac: maskFraction(mask),
    psnrZf: psnr(zf, truth, 1), psnrCs: psnr(x, truth, 1), ssimZf: ssim(zf, truth, n, n, { peak: 1 }).mean, ssimCs: ssim(x, truth, n, n, { peak: 1 }).mean,
  };
}

const shepp = Float64Array.from(sheppLogan(n).data), les = Float64Array.from(sheppLogan(n, LESION_PHANTOM).data);
const re = Float64Array.from(shepp), im = new Float64Array(n * n);
fft2d(re, im, n);
const klog = Array.from(re, (v, i) => Math.log10(1 + Math.hypot(v, im[i])));
const res: Record<string, unknown> = { n, shepp: Array.from(shepp), lesions: Array.from(les), klog };
res.centre = cs(shepp, "centre", 4, 0);
res.regular = cs(shepp, "regular", 2, 0);
res.vardens = cs(shepp, "vardens", 4, 0.01);
res.random = cs(shepp, "random", 4, 0.01);
res.les4 = cs(les, "vardens", 4, 0.015);
res.les8 = cs(les, "vardens", 8, 0.015);
writeFileSync(`${out}/mri.json`, JSON.stringify(res));
for (const k of ["centre", "regular", "vardens", "random", "les4", "les8"]) {
  const r = res[k] as { frac: number; psnrZf: number; psnrCs: number; ssimZf: number; ssimCs: number };
  console.log(k, "frac", r.frac.toFixed(3), "PSNR zf", r.psnrZf.toFixed(2), "cs", r.psnrCs.toFixed(2), "SSIM zf", r.ssimZf.toFixed(3), "cs", r.ssimCs.toFixed(3));
}
