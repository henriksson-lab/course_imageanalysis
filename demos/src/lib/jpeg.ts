// The baseline JPEG pipeline as pure functions (no DOM, no file writing). Owner: the
// compression day (jpeg/ unit). Steps, for each 8 × 8 block of each channel:
//   level shift (−128) → DCT (lib/dct.ts) → quantise q = round(c / Q) → zigzag →
//   DC as the difference from the previous block, AC as (zero-run, size) symbols + amplitude
//   bits, Huffman-coded with the standard tables (lib/huffman.ts) → bit count.
// Decoding reverses each step; only the rounding in the quantiser (and chroma subsampling)
// loses information. The size is the exact number of entropy-coded bits plus a fixed header
// estimate (byte stuffing ignored).
// Colour: JFIF Y′CbCr (full range) and 4:2:0 subsampling by 2 × 2 averaging, upsampling by
// replication. (Kept here because lib/colour.ts has no Y′CbCr yet.)
import { dctBlock, idctBlock, ZIGZAG8 } from "./dct";
import { DC_LUMA, AC_LUMA, DC_CHROMA, AC_CHROMA, jpegCodes } from "./huffman";

/** Standard luminance quantisation table (Annex K.1), natural (row-major) order. */
export const LUMA_Q = Int32Array.from([
  16, 11, 10, 16, 24, 40, 51, 61,
  12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103, 99,
]);

/** Standard chrominance quantisation table (Annex K.2). */
export const CHROMA_Q = Int32Array.from([
  17, 18, 24, 47, 99, 99, 99, 99,
  18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99,
  47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
]);

/** IJG quality scaling in percent: 5000/q for q < 50, 200 − 2q otherwise (q clamped to 1..100). */
export function qualityScale(quality: number): number {
  const q = Math.min(100, Math.max(1, quality));
  return q < 50 ? 5000 / q : 200 - 2 * q;
}

/** Table scaled by s percent: Q' = ⌊(Q·s + 50)/100⌋, clamped to 1..255 (baseline). */
export function scaleTable(Q: ArrayLike<number>, s: number): Int32Array {
  return Int32Array.from(Q, v => Math.min(255, Math.max(1, Math.floor((v * s + 50) / 100))));
}

/** The table for a quality factor 1..100. */
export const tableForQuality = (Q: ArrayLike<number>, quality: number): Int32Array => scaleTable(Q, qualityScale(quality));

/** A flat table: every step equal. */
export const flatTable = (step: number): Int32Array => new Int32Array(64).fill(Math.min(255, Math.max(1, Math.round(step))));

/** Round half away from zero. */
const rnd = (v: number) => (v < 0 ? -Math.round(-v) : Math.round(v));

/** Quantise one block of DCT coefficients: q = round(c / Q). */
export function quantise(c: ArrayLike<number>, Q: ArrayLike<number>): Int32Array {
  return Int32Array.from({ length: 64 }, (_, i) => rnd(c[i] / Q[i]));
}

/** Dequantise: c' = q · Q. */
export function dequantise(q: ArrayLike<number>, Q: ArrayLike<number>): Float64Array {
  return Float64Array.from({ length: 64 }, (_, i) => q[i] * Q[i]);
}

/** Size category of a value: number of bits of |v| (0 for 0). */
export function sizeCategory(v: number): number {
  let a = Math.abs(v), s = 0;
  while (a > 0) { s++; a >>= 1; }
  return s;
}

/** The amplitude bits of v in its size category (JPEG: negative values as v − 1 in ones' complement). */
export function amplitudeBits(v: number): string {
  const s = sizeCategory(v);
  if (s === 0) return "";
  const u = v > 0 ? v : v + (1 << s) - 1;
  return u.toString(2).padStart(s, "0");
}

/** The coefficients of a quantised block in zigzag order. */
export const toZigzag = (q: ArrayLike<number>): Int32Array => Int32Array.from(ZIGZAG8, i => q[i]);

export interface AcSymbol { run: number; size: number; value: number; symbol: number }
export interface BlockSymbols {
  dcDiff: number;          // DC minus the previous block's DC
  dcSize: number;
  ac: AcSymbol[];          // including ZRL (run 15, size 0) and EOB (0, 0) entries
}

/** The entropy-coding symbols of one quantised block (natural order) after `prevDC`. */
export function blockSymbols(q: ArrayLike<number>, prevDC: number): BlockSymbols {
  const z = toZigzag(q);
  const dcDiff = z[0] - prevDC;
  const ac: AcSymbol[] = [];
  let last = 0;
  for (let i = 63; i >= 1; i--) if (z[i] !== 0) { last = i; break; }
  let run = 0;
  for (let i = 1; i <= last; i++) {
    if (z[i] === 0) { run++; continue; }
    while (run > 15) { ac.push({ run: 15, size: 0, value: 0, symbol: 0xf0 }); run -= 16; }
    const size = sizeCategory(z[i]);
    ac.push({ run, size, value: z[i], symbol: (run << 4) | size });
    run = 0;
  }
  if (last < 63) ac.push({ run: 0, size: 0, value: 0, symbol: 0x00 });
  return { dcDiff, dcSize: sizeCategory(dcDiff), ac };
}

export interface HuffTables { dc: string[]; ac: string[] }
export const LUMA_CODES: HuffTables = { dc: jpegCodes(DC_LUMA), ac: jpegCodes(AC_LUMA) };
export const CHROMA_CODES: HuffTables = { dc: jpegCodes(DC_CHROMA), ac: jpegCodes(AC_CHROMA) };

/** Number of bits of a block's symbols with the given Huffman tables. */
export function blockBits(s: BlockSymbols, t: HuffTables = LUMA_CODES): number {
  let b = t.dc[s.dcSize].length + s.dcSize;
  for (const a of s.ac) b += t.ac[a.symbol].length + a.size;
  return b;
}

/** The bit string of a block (for display). */
export function blockBitString(s: BlockSymbols, t: HuffTables = LUMA_CODES): string {
  let b = t.dc[s.dcSize] + amplitudeBits(s.dcDiff);
  for (const a of s.ac) b += t.ac[a.symbol] + amplitudeBits(a.value);
  return b;
}

/** Header bytes of a baseline JFIF file with the standard tables (SOI, APP0, DQT, SOF0, DHT, SOS, EOI). */
export const headerBytes = (colour: boolean): number => (colour ? 625 : 330);

export interface ChannelResult {
  decoded: Float64Array;   // 0..255, rounded and clamped
  coef: Float64Array;      // DCT coefficients before quantisation, block layout (pw × ph)
  quant: Int32Array;       // quantised coefficients, block layout
  bits: number;            // entropy-coded bits
  nonzero: number;         // non-zero quantised coefficients
  pw: number; ph: number;  // padded size (multiples of 8)
}

/** Edge-replicate a w × h image to multiples of 8. */
export function padTo8(img: ArrayLike<number>, w: number, h: number): { data: Float64Array; pw: number; ph: number } {
  const pw = Math.ceil(w / 8) * 8, ph = Math.ceil(h / 8) * 8, data = new Float64Array(pw * ph);
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) data[y * pw + x] = img[Math.min(y, h - 1) * w + Math.min(x, w - 1)];
  return { data, pw, ph };
}

/** Encode and decode one channel (values 0..255) with table Q; bits counted with tables t. */
export function codeChannel(img: ArrayLike<number>, w: number, h: number, Q: ArrayLike<number>, t: HuffTables = LUMA_CODES): ChannelResult {
  const { data, pw, ph } = padTo8(img, w, h);
  const coef = new Float64Array(pw * ph), quant = new Int32Array(pw * ph), rec = new Float64Array(pw * ph);
  const blk = new Float64Array(64);
  let bits = 0, nonzero = 0, prevDC = 0;
  for (let by = 0; by < ph / 8; by++) for (let bx = 0; bx < pw / 8; bx++) {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = data[(by * 8 + y) * pw + bx * 8 + x] - 128;
    const c = dctBlock(blk, 8), q = quantise(c, Q);
    const s = blockSymbols(q, prevDC);
    prevDC = q[0];
    bits += blockBits(s, t);
    const r = idctBlock(dequantise(q, Q), 8);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const i = (by * 8 + y) * pw + bx * 8 + x, k = y * 8 + x;
      coef[i] = c[k]; quant[i] = q[k]; rec[i] = r[k];
      if (q[k] !== 0) nonzero++;
    }
  }
  const decoded = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) decoded[y * w + x] = Math.min(255, Math.max(0, Math.round(rec[y * pw + x] + 128)));
  return { decoded, coef, quant, bits, nonzero, pw, ph };
}

export interface GreyJpeg extends ChannelResult { bytes: number; bpp: number; Q: Int32Array }

/** JPEG of a grey image (0..255) at a quality factor, or with an explicit table. */
export function jpegGrey(img: ArrayLike<number>, w: number, h: number, opts: { quality?: number; table?: ArrayLike<number> } = {}): GreyJpeg {
  const Q = opts.table ? Int32Array.from(opts.table) : tableForQuality(LUMA_Q, opts.quality ?? 75);
  const r = codeChannel(img, w, h, Q);
  const bytes = Math.ceil(r.bits / 8) + headerBytes(false);
  return { ...r, bytes, bpp: 8 * bytes / (w * h), Q };
}

// ---------------------------------------------------------------------------------------
// Colour

/** JFIF Y′CbCr from 8-bit R′G′B′ (full range, offsets 128). */
export function rgbToYcc(r: number, g: number, b: number): [number, number, number] {
  return [
    0.299 * r + 0.587 * g + 0.114 * b,
    128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
    128 + 0.5 * r - 0.418688 * g - 0.081312 * b,
  ];
}

/** Inverse of rgbToYcc. */
export function yccToRgb(y: number, cb: number, cr: number): [number, number, number] {
  return [y + 1.402 * (cr - 128), y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128), y + 1.772 * (cb - 128)];
}

/** 2 × 2 average (w, h even after edge replication): the 4:2:0 chroma plane. */
export function subsample2(c: ArrayLike<number>, w: number, h: number): { data: Float64Array; w: number; h: number } {
  const sw = Math.ceil(w / 2), sh = Math.ceil(h / 2), out = new Float64Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    let s = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) s += c[Math.min(h - 1, 2 * y + dy) * w + Math.min(w - 1, 2 * x + dx)];
    out[y * sw + x] = s / 4;
  }
  return { data: out, w: sw, h: sh };
}

/** Upsample by 2 (pixel replication) back to w × h. */
export function upsample2(c: ArrayLike<number>, sw: number, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = c[(y >> 1) * sw + (x >> 1)];
  return out;
}

export interface ColourJpeg {
  rgb: [Float64Array, Float64Array, Float64Array];   // decoded R, G, B (0..255, rounded)
  y: ChannelResult; cb: ChannelResult; cr: ChannelResult;
  bits: number; bytes: number; bpp: number; nonzero: number;
}

/** JPEG of an RGB image (three planes 0..255), 4:4:4 or 4:2:0. */
export function jpegColour(R: ArrayLike<number>, G: ArrayLike<number>, B: ArrayLike<number>, w: number, h: number,
  opts: { quality?: number; subsample?: boolean } = {}): ColourJpeg {
  const n = w * h, Y = new Float64Array(n), Cb = new Float64Array(n), Cr = new Float64Array(n);
  for (let i = 0; i < n; i++) { const t = rgbToYcc(R[i], G[i], B[i]); Y[i] = t[0]; Cb[i] = t[1]; Cr[i] = t[2]; }
  const ql = tableForQuality(LUMA_Q, opts.quality ?? 75), qc = tableForQuality(CHROMA_Q, opts.quality ?? 75);
  const y = codeChannel(Y, w, h, ql, LUMA_CODES);
  let cb: ChannelResult, cr: ChannelResult, cbUp: Float64Array, crUp: Float64Array;
  if (opts.subsample) {
    const sb = subsample2(Cb, w, h), sr = subsample2(Cr, w, h);
    cb = codeChannel(sb.data, sb.w, sb.h, qc, CHROMA_CODES); cr = codeChannel(sr.data, sr.w, sr.h, qc, CHROMA_CODES);
    cbUp = upsample2(cb.decoded, sb.w, w, h); crUp = upsample2(cr.decoded, sr.w, w, h);
  } else {
    cb = codeChannel(Cb, w, h, qc, CHROMA_CODES); cr = codeChannel(Cr, w, h, qc, CHROMA_CODES);
    cbUp = cb.decoded; crUp = cr.decoded;
  }
  const rgb: [Float64Array, Float64Array, Float64Array] = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let i = 0; i < n; i++) {
    const t = yccToRgb(y.decoded[i], cbUp[i], crUp[i]);
    for (let k = 0; k < 3; k++) rgb[k][i] = Math.min(255, Math.max(0, Math.round(t[k])));
  }
  const bits = y.bits + cb.bits + cr.bits, bytes = Math.ceil(bits / 8) + headerBytes(true);
  return { rgb, y, cb, cr, bits, bytes, bpp: 8 * bytes / n, nonzero: y.nonzero + cb.nonzero + cr.nonzero };
}
