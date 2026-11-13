// Dumps the data behind the compression day's slide figures, computed with the demos' own code
// (so the numbers match the demos). Bundle and run from the repo root:
//   python3 -I tools/figs/compression/figs.py celljson $S   (the micrograph as 256*256 integers)
//   node_modules/.bin/esbuild tools/figs/compression/dump.ts --bundle --platform=node --outfile=$S/dump.js
//   node $S/dump.js $S        → $S/dump.json, read by tools/figs/compression/figs.py
import * as fs from "fs";
import { smooth, shapes, withNoise, card, SIZE } from "../../../demos/src/jpeg/testimages";
import { residuals } from "../../../demos/src/lib/predict";
import { entropy } from "../../../demos/src/lib/entropy";
import { jpegGrey } from "../../../demos/src/lib/jpeg";
import { ssim, psnr } from "../../../demos/src/lib/metrics";
import { spots } from "../../../demos/src/lib/synth";
import * as M from "../../../demos/src/measuredamage/scene";
import { tune } from "../../../demos/src/samemse/distort";

const dir = process.argv[2];
const N = SIZE;
const cell = Float64Array.from(JSON.parse(fs.readFileSync(dir + "/cell.json", "utf8")) as number[]);
const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);
const out: Record<string, unknown> = {};

// ---- lossless: predict demo images
const sm = smooth(), sh = shapes(), smN10 = withNoise(sm, 10, 99, 255);
const fl = spots({ width: N, height: N, count: 60, sigma: 2, amplitude: [300, 3000], background: 300, slope: 150, bump: 250, photons: 1, seed: 5 });
const fluor12 = Float64Array.from(fl.image.data, v => Math.min(4095, Math.max(0, Math.round(v))));
out.smooth = arr(sm); out.shapes = arr(sh); out.smoothN10 = arr(smN10); out.fluor12 = arr(fluor12);
const preds = ["left", "average", "planar", "med"] as const;
const H: Record<string, Record<string, number>> = {};
for (const [k, img] of Object.entries({ smooth: sm, smoothN10: smN10, cell, fluor12, shapes: sh })) {
  H[k] = { pixels: entropy(img) };
  for (const p of preds) H[k][p] = entropy(residuals(img, N, N, p));
}
out.H = H;
out.res = {
  smooth_left: Array.from(residuals(sm, N, N, "left")), smooth_avg: Array.from(residuals(sm, N, N, "average")),
  smoothN10_avg: Array.from(residuals(smN10, N, N, "average")), cell_avg: Array.from(residuals(cell, N, N, "average")),
  shapes_left: Array.from(residuals(sh, N, N, "left")), shapes_med: Array.from(residuals(sh, N, N, "med")),
};

// ---- jpeg: rate-distortion of the micrograph (as the demo's rd tab: q = 5, 10, ..., 100)
const rd: { q: number; bpp: number; bytes: number; psnr: number; ssim: number }[] = [];
for (let q = 5; q <= 100; q += 5) {
  const r = jpegGrey(cell, N, N, { quality: q });
  rd.push({ q, bpp: r.bpp, bytes: r.bytes, psnr: psnr(cell, r.decoded, 255), ssim: ssim(cell, r.decoded, N, N, { peak: 255 }).mean });
}
out.rd = rd;
out.cell_q10 = arr(jpegGrey(cell, N, N, { quality: 10 }).decoded);
out.cell_q50 = arr(jpegGrey(cell, N, N, { quality: 50 }).decoded);
out.cell_q90 = arr(jpegGrey(cell, N, N, { quality: 90 }).decoded);
const cd = card();
out.card = arr(cd); out.card_q25 = arr(jpegGrey(cd, N, N, { quality: 25 }).decoded);

// ---- jpeg: same MSE (144) on the micrograph, as the samemse demo
const same: Record<string, { mse: number; ssim: number; img: number[] }> = {};
for (const id of ["shift", "blur", "jpeg"] as const) {
  const t = tune(id, cell, N, N, 144);
  same[id] = { mse: t.mse, ssim: ssim(cell, t.out, N, N, { peak: 255 }).mean, img: arr(t.out) };
}
out.same = same;

// ---- compress-science: the measuredamage scene
const sc = M.makeScene();
const T = M.detectionThreshold(sc.image), bg = M.backgroundMask(sc.spots, sc.clean), th = M.thirds(sc.spots);
const m0 = M.measure(sc.image, T, sc, bg);
const j30 = M.jpeg12(sc.image, 30), wv = M.waveletAtSize(sc.image, j30.bytes);
out.scene = arr(sc.image); out.scene_q30 = arr(j30.image); out.scene_wav = arr(wv.image);
out.scene_q90 = arr(M.jpeg12(sc.image, 90).image);
out.scene_bytes = { q30: j30.bytes, wav: wv.bytes };
out.scene_spots = sc.spots;
const curve: { q: number; bpp: number; bytes: number; found: number; falsePos: number; dim: number; bright: number; block: number }[] = [];
for (let q = 5; q <= 100; q += 5) {
  const d = M.jpeg12(sc.image, q), m = M.measure(d.image, T, sc, bg);
  curve.push({ q, bpp: d.bpp, bytes: d.bytes, found: m.det.found, falsePos: m.det.falsePos, dim: M.meanAbsChange(m0.integ, m.integ, th.dim), bright: M.meanAbsChange(m0.integ, m.integ, th.bright), block: M.blockingIndex(d.image) });
}
out.curve = curve;
const mw = M.measure(wv.image, T, sc, bg);
out.wavelet = { bytes: wv.bytes, bpp: wv.bpp, found: mw.det.found, falsePos: mw.det.falsePos, dim: M.meanAbsChange(m0.integ, mw.integ, th.dim), block: M.blockingIndex(wv.image) };
out.lossless = M.lossless(sc.image).bytes;
out.block0 = M.blockingIndex(sc.image);
out.spec_orig = arr(M.powerSpectrum(M.absDiff(sc.image)));
out.spec_q30 = arr(M.powerSpectrum(M.absDiff(j30.image)));
out.spec_wav = arr(M.powerSpectrum(M.absDiff(wv.image)));
fs.writeFileSync(dir + "/dump.json", JSON.stringify(out));
console.log(JSON.stringify({ same: Object.fromEntries(Object.entries(same).map(([k, v]) => [k, [v.mse, v.ssim]])), H, rd: rd.filter(r => [10, 50, 90].includes(r.q)), wavelet: out.wavelet, curve: curve.filter(c => [10, 30, 90, 95].includes(c.q)), block0: out.block0, lossless: out.lossless }, null, 1));
