// Tests for the compression day's libraries: lib/entropy.ts, lib/huffman.ts, lib/dct.ts,
// lib/predict.ts, lib/jpeg.ts (and the SSIM of lib/metrics.ts against reference values).
// Run alone:
//   node_modules/.bin/esbuild demos/test/compression-test.ts --bundle --platform=node --log-level=warning | node
import { entropy, entropyFromCounts, information, rleEncode, rleDecode, binaryRuns, binaryRunsDecode, fixedLength } from "../src/lib/entropy";
import { huffman, kraftSum, averageLength, isPrefixFree, canonicalCodes, encode, decode, decodeTable, jpegCodes, DC_LUMA, AC_LUMA, DC_CHROMA, AC_CHROMA } from "../src/lib/huffman";
import { dctMatrix, dct1, idct1, dctFast, idctFast, dct2, idct2, dctBlock, idctBlock, blockDct, blockIdct, ZIGZAG8, realDftMatrix, separableBasis, energy } from "../src/lib/dct";
import { PREDICTORS, residuals, reconstruct, rowResiduals, predictFrom } from "../src/lib/predict";
import { LUMA_Q, tableForQuality, quantise, dequantise, blockSymbols, blockBits, blockBitString, sizeCategory, amplitudeBits, jpegGrey, jpegColour, rgbToYcc, yccToRgb, LUMA_CODES } from "../src/lib/jpeg";
import { ssim, psnr } from "../src/lib/metrics";
import { dft1d } from "../src/lib/fft";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return a.length === b.length ? m : Infinity;
};
const R = rng(2025);
const randVec = (n: number, s = 1) => Float64Array.from({ length: n }, () => s * (2 * R.unif() - 1));

// ---- entropy ----
check(Math.abs(entropyFromCounts([1, 1]) - 1) < 1e-12, "fair coin: 1 bit");
check(Math.abs(entropyFromCounts(new Array(256).fill(1)) - 8) < 1e-12, "256 equal levels: 8 bits");
check(entropyFromCounts([5]) === 0, "constant: 0 bits");
check(information(0.5) === 1 && information(1) === 0, "information");
check(Math.abs(entropyFromCounts([0.4, 0.2, 0.2, 0.1, 0.1]) - 2.1219280948873624) < 1e-12, "H(0.4,0.2,0.2,0.1,0.1) = 2.122");
check(Math.abs(entropy([3, -1, 3, -1, 70000, 3]) - entropyFromCounts([3, 2, 1])) < 1e-12, "entropy of signed / 16-bit values");
check(fixedLength(5) === 3 && fixedLength(256) === 8 && fixedLength(1) === 0, "fixed length");
for (let t = 0; t < 50; t++) {
  const v = Float64Array.from({ length: 1 + Math.floor(R.unif() * 40) }, () => Math.floor(R.unif() * 3));
  check(maxDiff(rleDecode(rleEncode(v)), v) === 0, "RLE round trip");
  const bits = Uint8Array.from(v, x => (x > 1 ? 1 : 0));
  check(maxDiff(binaryRunsDecode(binaryRuns(bits)), bits) === 0, "binary runs round trip");
}
check(JSON.stringify(binaryRuns([1, 1, 0, 0, 0, 1])) === "[0,2,3,1]", "binary runs start with a 0-run");

// ---- Huffman ----
{
  const p = [0.4, 0.2, 0.2, 0.1, 0.1];
  const a = huffman(p, "older"), b = huffman(p, "newer");
  check(JSON.stringify(Array.from(a.lengths)) === "[2,2,2,3,3]", "older tie rule: lengths 2,2,2,3,3 got " + Array.from(a.lengths));
  check(JSON.stringify(Array.from(b.lengths)) === "[1,2,3,4,4]", "newer tie rule: lengths 1,2,3,4,4 got " + Array.from(b.lengths));
  check(Math.abs(averageLength(p, a.lengths) - 2.2) < 1e-12 && Math.abs(averageLength(p, b.lengths) - 2.2) < 1e-12, "both trees: L = 2.2");
}
for (let t = 0; t < 300; t++) {
  const K = 2 + Math.floor(R.unif() * 15);
  const w = Array.from({ length: K }, () => (R.unif() < 0.15 ? 0 : Math.floor(1 + R.unif() * R.unif() * 100)));
  if (w.filter(x => x > 0).length < 2) w[0] = w[1] = 1;
  for (const tie of ["older", "newer"] as const) {
    const h = huffman(w, tie);
    const H = entropyFromCounts(w), L = averageLength(w, h.lengths);
    check(isPrefixFree(h.codes), "prefix-free");
    check(Math.abs(kraftSum(h.lengths) - 1) < 1e-12, "Kraft equality");
    check(H <= L + 1e-12 && L < H + 1, `H <= L < H + 1 (H ${H}, L ${L})`);
    w.forEach((x, i) => check((x > 0) === (h.lengths[i] > 0), "only used symbols get codes"));
    const msg = Array.from({ length: 30 }, () => { let s; do s = Math.floor(R.unif() * K); while (!w[s]); return s; });
    const bits = encode(msg, h.codes);
    check(JSON.stringify(decode(bits, h)) === JSON.stringify(msg), "Huffman encode/decode round trip");
    const can = canonicalCodes(h.lengths);
    check(isPrefixFree(can) && can.every((c, i) => c.length === h.lengths[i]), "canonical code has the same lengths");
    check(JSON.stringify(decodeTable(encode(msg, can), can)) === JSON.stringify(msg), "canonical round trip");
  }
  // optimality: no other prefix code beats it; compare with brute force over length vectors for K ≤ 5
  if (K <= 5 && w.every(x => x > 0)) {
    let best = Infinity;
    const rec = (i: number, L: number[]) => {
      if (i === K) { if (kraftSum(L) <= 1 + 1e-12) best = Math.min(best, averageLength(w, L)); return; }
      for (let l = 1; l <= K; l++) rec(i + 1, [...L, l]);
    };
    rec(0, []);
    check(Math.abs(best - averageLength(w, huffman(w).lengths)) < 1e-12, "Huffman is optimal (brute force over Kraft-feasible lengths)");
  }
}
{
  const one = huffman([0, 7, 0]);
  check(one.lengths[1] === 1 && one.codes[1] === "0", "single symbol gets one bit");
}

// ---- JPEG Huffman tables ----
for (const [name, t, nv] of [["DC luma", DC_LUMA, 12], ["AC luma", AC_LUMA, 162], ["DC chroma", DC_CHROMA, 12], ["AC chroma", AC_CHROMA, 162]] as const) {
  const total = t.bits.reduce((a, b) => a + b, 0);
  check(total === nv && t.vals.length === nv && new Set(t.vals).size === nv, `${name}: ${nv} distinct symbols`);
  const codes = jpegCodes(t);
  check(isPrefixFree(codes), `${name} prefix-free`);
  check(kraftSum(codes.map(c => c.length)) < 1, `${name}: Kraft < 1 (all-ones code reserved)`);
}
{
  const s = new Set(AC_LUMA.vals);
  let all = s.has(0) && s.has(0xf0);
  for (let r = 0; r < 16; r++) for (let z = 1; z <= 10; z++) all = all && s.has((r << 4) | z);
  check(all, "AC table covers EOB, ZRL and all (run, size)");
  const c = jpegCodes(AC_LUMA), d = jpegCodes(DC_LUMA);
  check(c[0x00] === "1010" && c[0x01] === "00" && c[0x02] === "01" && c[0x03] === "100" && c[0x11] === "1100" && c[0xf0] === "11111111001", "known AC luma codes");
  check(d[0] === "00" && d[1] === "010" && d[5] === "110" && d[6] === "1110" && d[11] === "111111110", "known DC luma codes");
}

// ---- DCT ----
for (const N of [2, 4, 8, 16]) {
  const D = dctMatrix(N);
  let err = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let s = 0; for (let k = 0; k < N; k++) s += D[i * N + k] * D[j * N + k];
    err = Math.max(err, Math.abs(s - (i === j ? 1 : 0)));
  }
  check(err < 1e-12, `DCT ${N}: orthonormal`);
}
{
  const D2 = dctMatrix(2);
  check(Math.abs(D2[0] - Math.SQRT1_2) < 1e-15 && Math.abs(D2[2] - Math.SQRT1_2) < 1e-15 && Math.abs(D2[3] + Math.SQRT1_2) < 1e-15, "2-point DCT = 45° rotation");
  const c = dct1([1, 2, 3, 4]);
  check(Math.abs(c[0] - 5) < 1e-12 && Math.abs(c[1] + 2.230442497387663) < 1e-12 && Math.abs(c[2]) < 1e-12 && Math.abs(c[3] + 0.15851266778110717) < 1e-12, "DCT of (1,2,3,4)");
}
for (const N of [8, 16, 64, 256]) {
  const x = randVec(N, 100);
  check(maxDiff(dctFast(x), dct1(x)) < 1e-9, `DCT via FFT = matrix (N=${N})`);
  check(maxDiff(idctFast(dctFast(x)), x) < 1e-9, `DCT round trip via FFT (N=${N})`);
  check(maxDiff(idct1(dct1(x)), x) < 1e-9, `DCT round trip by matrix (N=${N})`);
  check(Math.abs(energy(dct1(x)) - energy(x)) < 1e-8 * energy(x), `Parseval (N=${N})`);
}
{
  // DCT = DFT of the mirrored signal (reference dft1d)
  const N = 8, x = randVec(N), y = new Float64Array(2 * N);
  for (let n = 0; n < N; n++) { y[n] = x[n]; y[2 * N - 1 - n] = x[n]; }
  const Y = dft1d(y, new Float64Array(2 * N)), c = dct1(x);
  let err = 0;
  for (let k = 0; k < N; k++) {
    const a = -Math.PI * k / (2 * N), X = (Math.cos(a) * Y.re[k] - Math.sin(a) * Y.im[k]) / 2;
    err = Math.max(err, Math.abs(c[k] - (k ? Math.sqrt(2 / N) : Math.sqrt(1 / N)) * X));
    // the imaginary part after the phase shift is zero: the mirrored signal is symmetric
    err = Math.max(err, Math.abs((Math.sin(a) * Y.re[k] + Math.cos(a) * Y.im[k]) / 2));
  }
  check(err < 1e-12, "DCT is the DFT of the mirrored block");
}
{
  const w = 32, h = 16, img = randVec(w * h, 50);
  check(maxDiff(idct2(dct2(img, w, h), w, h), img) < 1e-9, "2D DCT round trip");
  check(Math.abs(energy(dct2(img, w, h)) - energy(img)) < 1e-8 * energy(img), "2D Parseval");
  check(maxDiff(blockIdct(blockDct(img, w, h), w, h), img) < 1e-9, "block DCT round trip");
  const b = randVec(64, 100), C = dctBlock(b);
  const M = separableBasis(dctMatrix(8), 8);
  let err = 0;
  for (let r = 0; r < 64; r++) { let s = 0; for (let i = 0; i < 64; i++) s += M[r * 64 + i] * b[i]; err = Math.max(err, Math.abs(s - C[r])); }
  check(err < 1e-9, "block DCT = separable basis matrix");
  check(maxDiff(idctBlock(C), b) < 1e-9, "block round trip");
  const flat = new Float64Array(64).fill(10), Cf = dctBlock(flat);
  check(Math.abs(Cf[0] - 80) < 1e-9 && Cf.slice(1).every(v => Math.abs(v) < 1e-9), "flat block: only DC (= 8 × mean)");
}
check(JSON.stringify(Array.from(ZIGZAG8.slice(0, 10))) === "[0,1,8,16,9,2,3,10,17,24]" && ZIGZAG8[63] === 63 && new Set(ZIGZAG8).size === 64, "zigzag order");
{
  const N = 8, A = realDftMatrix(N);
  let err = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let s = 0; for (let k = 0; k < N; k++) s += A[i * N + k] * A[j * N + k];
    err = Math.max(err, Math.abs(s - (i === j ? 1 : 0)));
  }
  check(err < 1e-12, "real DFT basis orthonormal");
}

// ---- prediction ----
{
  const w = 23, h = 17;
  for (const scale of [255, 4095]) {
    const img = Int32Array.from({ length: w * h }, (_, i) => Math.floor(R.unif() * (scale + 1)));
    for (const p of PREDICTORS) {
      const e = residuals(img, w, h, p);
      check(maxDiff(reconstruct(e, w, h, p), img) === 0, `predictor ${p}: exact round trip (${scale})`);
    }
  }
  check(JSON.stringify(rowResiduals([100, 102, 104, 103, 105, 107, 106, 108])) === "[100,2,2,-1,2,2,-1,2]", "row residuals");
  check(Math.abs(entropyFromCounts([5, 2]) - 0.863120568566631) < 1e-12, "H of residuals 0.86");
  check(predictFrom("paeth", 100, 120, 110) === 110 && predictFrom("paeth", 50, 60, 100) === 50 && predictFrom("paeth", 10, 30, 12) === 30, "Paeth");
  check(predictFrom("med", 100, 50, 120) === 50 && predictFrom("med", 100, 50, 20) === 100 && predictFrom("med", 100, 50, 70) === 80, "MED");
}

// ---- JPEG ----
check(tableForQuality(LUMA_Q, 50)[0] === 16 && tableForQuality(LUMA_Q, 25)[0] === 32 && tableForQuality(LUMA_Q, 75)[0] === 8 && tableForQuality(LUMA_Q, 90)[0] === 3, "IJG scaling");
{
  // against the libjpeg tables (read back from files written at these qualities)
  const ref: Record<number, number[]> = {
    10: [80, 55, 50, 80, 120, 200, 255, 255, 60, 60], 25: [32, 22, 20, 32, 48, 80, 102, 122, 24, 24],
    75: [8, 6, 5, 8, 12, 20, 26, 31, 6, 6], 90: [3, 2, 2, 3, 5, 8, 10, 12, 2, 2],
  };
  for (const q of [10, 25, 75, 90]) check(JSON.stringify(Array.from(tableForQuality(LUMA_Q, q).slice(0, 10))) === JSON.stringify(ref[q]), `table at quality ${q}`);
}
check(sizeCategory(0) === 0 && sizeCategory(1) === 1 && sizeCategory(-1) === 1 && sizeCategory(5) === 3 && sizeCategory(-255) === 8, "size categories");
check(amplitudeBits(5) === "101" && amplitudeBits(-5) === "010" && amplitudeBits(-1) === "0" && amplitudeBits(1) === "1", "amplitude bits");
for (let t = 0; t < 30; t++) {
  const c = randVec(64, 300), Q = tableForQuality(LUMA_Q, 5 + Math.floor(R.unif() * 95));
  const d = dequantise(quantise(c, Q), Q);
  let ok = true;
  for (let i = 0; i < 64; i++) ok = ok && Math.abs(d[i] - c[i]) <= Q[i] / 2 + 1e-9;
  check(ok, "quantisation error <= Q/2");
}
{
  // the hand example of the exercise: a mostly-zero quantised block
  const q = new Int32Array(64);
  q[0] = 12; q[1] = -3; q[8] = 2; q[16] = -1; q[2] = 1;   // zigzag: 12, -3, 2, -1, 0, 1, then zeros
  const s = blockSymbols(q, 9);
  const sym = s.ac.map(a => `${a.run}/${a.size}:${a.value}`).join(" ");
  check(s.dcDiff === 3 && s.dcSize === 2, "DC difference 3, size 2");
  check(sym === "0/2:-3 0/2:2 0/1:-1 1/1:1 0/0:0", "AC symbols " + sym);
  const bits = blockBits(s);
  // DC: size 2 code "011" + 2 bits; AC: 0/2 "01"+2, 0/2 "01"+2, 0/1 "00"+1, 1/1 "1100"+1, EOB "1010"
  check(bits === 5 + 4 + 4 + 3 + 5 + 4, "bit count of the hand example = 25, got " + bits);
  check(blockBitString(s) === "01111" + "0100" + "0110" + "000" + "11001" + "1010", "bit string of the hand example " + blockBitString(s));
}
{
  // a block with a long zero run needs ZRL
  const q = new Int32Array(64); q[ZIGZAG8[20]] = 1;
  const s = blockSymbols(q, 0);
  check(s.ac.length === 3 && s.ac[0].symbol === 0xf0 && s.ac[1].run === 3 && s.ac[2].symbol === 0, "ZRL for a run of 19 zeros");
}
{
  // a smooth test image: decoding error and sizes behave
  const w = 64, h = 48, img = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = 128 + 60 * Math.sin(x / 7) * Math.cos(y / 9) + (x > 40 ? 40 : 0);
  let prevBits = 0, prevErr = Infinity;
  for (const q of [10, 30, 50, 75, 90, 100]) {
    const r = jpegGrey(img, w, h, { quality: q });
    const e = psnr(img, r.decoded, 255);
    check(r.bits >= prevBits && e >= prevErr - 0.5 || prevErr === Infinity, `quality ${q}: more bits and better PSNR`);
    prevBits = r.bits; prevErr = e;
  }
  const q100 = jpegGrey(img, w, h, { quality: 100 });
  check(maxDiff(q100.decoded, img) <= 1.5, "quality 100: error about one grey level");
  const flat = jpegGrey(new Float64Array(w * h).fill(77), w, h, { quality: 50 });
  check(flat.nonzero <= (w * h) / 64 && maxDiff(flat.decoded, new Float64Array(w * h).fill(77)) <= 1, "flat image: only DCs");
  // block bits: the encoded total equals the sum over blocks with DC differences
  let total = 0, prev = 0;
  for (let by = 0; by < h / 8; by++) for (let bx = 0; bx < w / 8; bx++) {
    const q = new Int32Array(64);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) q[y * 8 + x] = q100.quant[(by * 8 + y) * w + bx * 8 + x];
    total += blockBits(blockSymbols(q, prev), LUMA_CODES); prev = q[0];
  }
  check(total === q100.bits, "bit total = sum of block bit counts");
}
{
  for (let t = 0; t < 20; t++) {
    const r = 255 * R.unif(), g = 255 * R.unif(), b = 255 * R.unif();
    const [y, cb, cr] = rgbToYcc(r, g, b), back = yccToRgb(y, cb, cr);
    check(maxDiff(back, [r, g, b]) < 1e-3, "Y'CbCr round trip");
  }
  const w = 32, h = 32, n = w * h, Rr = new Float64Array(n), G = new Float64Array(n), B = new Float64Array(n);
  for (let i = 0; i < n; i++) { const x = i % w; Rr[i] = x < 16 ? 220 : 30; G[i] = 40; B[i] = x < 16 ? 30 : 220; }
  const a = jpegColour(Rr, G, B, w, h, { quality: 90 }), s = jpegColour(Rr, G, B, w, h, { quality: 90, subsample: true });
  check(s.bits < a.bits, "4:2:0 needs fewer bits");
  check(psnr(Rr, a.rgb[0], 255) > 25, "colour JPEG decodes");
}

// ---- SSIM / PSNR against reference values (scikit-image 0.26: gaussian_weights, σ = 1.5,
//      population covariance, data_range 255) ----
{
  const h = 32, w = 40, a = new Float64Array(w * h), b = new Float64Array(w * h);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    a[i * w + j] = (i * 7 + j * 13) % 256;
    b[i * w + j] = Math.min(255, Math.max(0, a[i * w + j] + ((i * j) % 17) - 8));
  }
  const s = ssim(a, b, w, h, { peak: 255 }).mean;
  check(Math.abs(s - 0.9861565022034969) < 1e-6, "SSIM vs reference: " + s);
  check(Math.abs(psnr(a, b, 255) - 33.89207568481616) < 1e-9, "PSNR vs reference");
  check(Math.abs(ssim(a, a, w, h).mean - 1) < 1e-12, "SSIM(x, x) = 1");
}

console.log(`compression: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
