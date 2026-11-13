// Data for the dictionaries/ figures, computed with the demos' code (demos/src/lib/patches.ts,
// sparse.ts, wavelet.ts, denoise.ts). Run by figs.py:  node data.js <outdir> <cell.f64>
// Micrograph crops: the 128 × 128 centre (as in the patchdict demo) and, as the training image of
// the restoration figures, the 128 × 128 crop at (128, 0) (as in the sparserestore demo).
import { writeFileSync, readFileSync } from "fs";
import {
  extractPatches, Patches, dctDictionary, randomPatchDictionary, learnStep, batchOmp, patchPCA, linearApproxError,
  representationError, gram, ksvdDenoise, patchInpaint, atomUsage,
} from "../../../demos/src/lib/patches";
import { denoise2, WAVELETS } from "../../../demos/src/lib/wavelet";
import { nonLocalMeans } from "../../../demos/src/lib/denoise";
import { psnr } from "../../../demos/src/lib/metrics";
import { rng, sampleIndices } from "../../../demos/src/lib/random";
import { textureImage, randomMask, scratchMask } from "../../../demos/src/sparserestore/images";
import { fista, maskOp, waveletProx2d } from "../../../demos/src/lib/sparse";

declare const process: { argv: string[] };
const out = process.argv[2], cell = new Float64Array(readFileSync(process.argv[3]).buffer.slice(0));
const n = 128, p = 8, d = 64;
const crop = (x0: number, y0: number) => { const o = new Float64Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) o[y * n + x] = cell[(y0 + y) * 256 + x0 + x]; return o; };
const img = crop(64, 64), train = crop(128, 0);
const A = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1e4) / 1e4);

function trainingSet(im: Float64Array, count = 4000, seed = 7): Patches {
  const all = extractPatches(im, n, n, p, 1), pick = sampleIndices(rng(seed), all.n, count);
  const Y = new Float64Array(pick.length * d), pos = new Int32Array(2 * pick.length);
  pick.forEach((q, i) => { Y.set(all.Y.subarray(q * d, (q + 1) * d), i * d); pos[2 * i] = all.pos[2 * q]; pos[2 * i + 1] = all.pos[2 * q + 1]; });
  return { Y, pos, n: pick.length, d, p };
}
function learn(P: Patches, atoms: number, K: number, iters: number, snapshotsAt: number[] = []) {
  const D = randomPatchDictionary(P, atoms, rng(3)), errs = [representationError(D, d, P.Y, batchOmp(D, d, P.Y, P.n, { K }))], snaps: Record<number, number[]> = { 0: A(D) };
  let codes = batchOmp(D, d, P.Y, P.n, { K });
  for (let it = 1; it <= iters; it++) { const r = learnStep(D, d, P, K, "ksvd"); codes = r.codes; errs.push(r.error); if (snapshotsAt.includes(it)) snaps[it] = A(D); }
  return { D, errs, snaps, usage: Array.from(atomUsage(codes, atoms)) };
}
const res: Record<string, unknown> = {};

// 1. learning on the micrograph centre (64 atoms, K = 4): snapshots at 0, 5, 20
const P = trainingSet(img);
const L64 = learn(P, 64, 4, 20, [5, 20]);
res.learning = { snaps: L64.snaps, errs: L64.errs, usage: L64.usage };
// 2. 128 atoms for PCA comparison, the patch code example and the restorations
const L128 = learn(P, 128, 4, 20);
const sub: Patches = { ...P, n: 1000, Y: P.Y.subarray(0, 1000 * d) };
const pca = patchPCA(sub, 16), dct = dctDictionary(p, 256), G1 = gram(L128.D, d), G2 = gram(dct, d);
const curves = { pca: [] as number[], dct: [] as number[], learned: [] as number[] };
for (let K = 1; K <= 8; K++) {
  curves.pca.push(linearApproxError(pca, d, sub, K));
  curves.dct.push(representationError(dct, d, sub.Y, batchOmp(dct, d, sub.Y, sub.n, { K, G: G2 })));
  curves.learned.push(representationError(L128.D, d, sub.Y, batchOmp(L128.D, d, sub.Y, sub.n, { K, G: G1 })));
}
res.pca = { atoms: A(pca), curves, learned: A(L128.D), usage: L128.usage };
res.odct = A(dct);
// patch code example
const px = 60, py = 40, y = new Float64Array(d);
for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) y[v * p + u] = img[(py + v) * n + px + u];
const c = batchOmp(L128.D, d, y, 1, { K: 3 });
res.code = { patch: A(y), idx: Array.from(c.idx), val: A(c.val), atoms: Array.from(c.idx, j => A(L128.D.subarray(j * d, (j + 1) * d))), err: c.err[0], pos: [px, py] };
// 3. texture atoms
const T = textureImage(n), PT = trainingSet(T);
const LT = learn(PT, 64, 4, 20);
res.texture = { image: A(T), atoms: A(LT.D), usage: LT.usage };
res.cellAtoms = { atoms: A(L64.D), usage: L64.usage };
// 4. denoising: σ = 20 on the centre crop; the learned dictionary comes from the training crop
const sigma = 20, rn = rng(11), noisy = Float64Array.from(img, v => v + sigma * rn.norm());
const LD = learn(trainingSet(train), 128, 4, 20).D;
let bestW = { t: 0, x: new Float64Array(0) as Float64Array, p: -Infinity };
for (const t of [0.75, 1, 1.25, 1.5, 2, 2.5, 3]) { const r = denoise2(noisy, n, n, WAVELETS.db4, 4, t * sigma, "soft", 4).out; const q = psnr(r, img, 255); if (q > bestW.p) bestW = { t, x: r, p: q }; }
let bestN = { h: 0, x: new Float64Array(0) as Float64Array, p: -Infinity };
for (const h of [0.6, 0.8, 1.0, 1.2]) { const r = nonLocalMeans(noisy, n, n, 2, 7, h * sigma); const q = psnr(r, img, 255); if (q > bestN.p) bestN = { h, x: r, p: q }; }
const den = { dct: ksvdDenoise(noisy, n, n, dct, p, sigma, { stride: 1 }), learned: ksvdDenoise(noisy, n, n, LD, p, sigma, { stride: 1 }) };
res.denoise = {
  truth: A(img), noisy: A(noisy), wavelet: A(bestW.x), nlm: A(bestN.x), dct: A(den.dct), learned: A(den.learned),
  psnr: { noisy: psnr(noisy, img, 255), wavelet: bestW.p, nlm: bestN.p, dct: psnr(den.dct, img, 255), learned: psnr(den.learned, img, 255) }, params: { t: bestW.t, h: bestN.h },
};
// 5. inpainting: 50 % random, scratches + hole; patch (learned) vs wavelet ℓ1 (as in ista/)
const inp: Record<string, unknown> = {};
for (const [name, mask] of [["random", randomMask(n, 0.5)], ["scratch", scratchMask(n)]] as [string, Uint8Array][]) {
  const obs = Float64Array.from(img, (v, i) => (mask[i] ? v : 0));
  const patch = patchInpaint(obs, mask, n, n, LD, p, { stride: 1, K: 6 });
  let s = 0, cnt = 0; for (let i = 0; i < n * n; i++) if (mask[i]) { s += obs[i]; cnt++; }
  const x0 = Float64Array.from(obs, (v, i) => (mask[i] ? v : s / cnt));
  const Aop = maskOp(mask);
  const wav = fista(Aop, obs, { lambda: 0.5, iters: 150, step: 1, x0, prox: (v, t) => waveletProx2d(v, n, n, WAVELETS.db4, 4, t * 0.5, 4), history: false }).x;
  inp[name] = { mask: Array.from(mask), patch: A(patch), wavelet: A(wav), psnrPatch: psnr(patch, img, 255), psnrWavelet: psnr(wav, img, 255) };
}
res.inpaint = inp;
writeFileSync(`${out}/dict.json`, JSON.stringify(res));
console.log("errors 64 atoms:", L64.errs.map(v => v.toFixed(1)).join(" "));
console.log("curves", JSON.stringify(curves));
console.log("denoise psnr", JSON.stringify((res.denoise as { psnr: unknown }).psnr), "params", JSON.stringify((res.denoise as { params: unknown }).params));
for (const k of ["random", "scratch"]) { const r = inp[k] as { psnrPatch: number; psnrWavelet: number }; console.log("inpaint", k, "patch", r.psnrPatch.toFixed(2), "wavelet", r.psnrWavelet.toFixed(2)); }
console.log("code", JSON.stringify({ idx: Array.from(c.idx), val: Array.from(c.val, v => v.toFixed(1)), err: c.err[0].toFixed(1) }));
