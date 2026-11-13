// Data for the extra shrinkage slide figures, computed with the wavedenoise demo's own code and
// noise (rng(7), shapes image, σ_n = 20) so that the numbers match the demo.
// Usage (called by figs.py): node data.js outdir
import * as fs from "fs";
import * as path from "path";
import { WAVELETS, dwt2, subband, madSigma, denoise2 } from "../../../demos/src/lib/wavelet";
import { psnr } from "../../../demos/src/lib/metrics";
import { rng } from "../../../demos/src/lib/random";
import { N, synthImage } from "../../../demos/src/dwt2d/images";

const out = process.argv[2];
const save = (name: string, a: Float64Array) => fs.writeFileSync(path.join(out, name + ".f64"), Buffer.from(a.buffer));
const info: Record<string, number> = {};
const r = rng(7);
const noise = Float64Array.from({ length: N * N }, () => r.norm());
const truth = synthImage("shapes");
const sigma = 20;
const noisy = Float64Array.from(truth, (x, i) => x + sigma * noise[i]);
save("truth", truth); save("noisy", noisy);
info.psnr_noisy = psnr(truth, noisy, 255);

// finest D band (db4, 4 levels: the demo default) and the MAD estimate
const D = subband(dwt2(noisy, N, N, WAVELETS.db4, 4), N, N, 1, "D");
save("finestD", D);
info.sigma_hat_db4 = madSigma(D);
let m = 0; for (const v of D) m += v; m /= D.length;
let v2 = 0; for (const v of D) v2 += (v - m) ** 2; info.sd_D = Math.sqrt(v2 / D.length);

// Haar, soft, t = 3 σ̂ (σ̂ from the Haar D band, as the demo does for the Haar presets): 1, 4 and 16 shifts
const sH = madSigma(subband(dwt2(noisy, N, N, WAVELETS.haar, 4), N, N, 1, "D"));
info.sigma_hat_haar = sH;
for (const sp of [1, 4, 16]) {
  const o = denoise2(noisy, N, N, WAVELETS.haar, 4, 3 * sH, "soft", sp).out;
  save("haar_spin" + sp, o); info["psnr_haar_spin" + sp] = psnr(truth, o, 255);
}
fs.writeFileSync(path.join(out, "info.json"), JSON.stringify(info, null, 1));
