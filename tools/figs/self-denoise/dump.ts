// Data for the figures of the self-denoise unit, computed with the blindspot demo's own code.
// Run from the repository root (DIR = any scratch folder):
//   node_modules/.bin/esbuild tools/figs/self-denoise/dump.ts --bundle --platform=node --outfile=DIR/sd.js && node DIR/sd.js > DIR/sd.json
//   python3 -I tools/figs/self-denoise/figs.py DIR/sd.json
import { evaluate, noiseField, W } from "../../../demos/src/blindspot/model";
import { DEFAULT } from "../../../demos/src/blindspot/presets";
import { cellCrop } from "../../../demos/test/selfsup-cell";

const clean = cellCrop(64, 64, W, W);
const q = (a: ArrayLike<number>) => Array.from(a, v => Math.round(Math.max(0, Math.min(255, v))));
const r2 = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);
const out: any = { W, clean: q(clean) };
const white = evaluate(clean, { ...DEFAULT });
const mixed = evaluate(clean, { ...DEFAULT, mix: true });
const n2n = evaluate(clean, { ...DEFAULT, train: "n2n" });
const corr = evaluate(clean, { ...DEFAULT, noise: "corr" });
out.white = { noisy: q(white.noisy), n2v: q(white.out), gauss: q(white.cmpOut), weights: r2(white.filter.weights), psnr: white.psnr };
out.mixed = { img: q(mixed.out), psnr: mixed.psnr.out, c: mixed.c };
out.n2n = { img: q(n2n.out), psnr: n2n.psnr.out, weights: r2(n2n.filter.weights) };
out.corr = { noisy: q(corr.noisy), n2v: q(corr.out), weights: r2(corr.filter.weights), psnr: corr.psnr };
const stripes = evaluate(clean, { ...DEFAULT, noise: "stripes" });
out.stripes = { noisy: q(stripes.noisy), n2v: q(stripes.out), psnr: stripes.psnr };
// autocorrelation of the noise alone (offsets 0..4 along x), for the white and the correlated noise
for (const kind of ["white", "corr"] as const) {
  const n = noiseField(W, W, 25, kind, 101);
  const ac: number[] = [];
  for (let d = 0; d <= 4; d++) {
    let s = 0, s0 = 0;
    for (let y = 0; y < W; y++) for (let x = 0; x + d < W; x++) { s += n[y * W + x] * n[y * W + x + d]; s0 += n[y * W + x] ** 2; }
    ac.push(Math.round(1000 * s / s0) / 1000);
  }
  out["ac_" + kind] = ac;
  out["noise_" + kind] = q(Array.from(n, v => 128 + 2 * v));
}
console.log(JSON.stringify(out));
