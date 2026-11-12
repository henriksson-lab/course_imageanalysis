// Minimal greyscale PNG writer for node scripts (figures for the unit pages). No dependencies.
declare const require: (m: string) => any;
const zlib = require("zlib");

const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b: Uint8Array): number { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
/** grey: values 0..255, row-major, w × h. Returns PNG bytes. */
export function pngGrey(grey: Uint8Array, w: number, h: number): Uint8Array {
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 0;
  const raw = new Uint8Array((w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(grey.subarray(y * w, (y + 1) * w), y * (w + 1) + 1);
  const idat = new Uint8Array(zlib.deflateSync(raw));
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
/** Tile 28 × 28 images (ink 0..1, shown dark on white) into a grid with `gap` white pixels, upscaled by `up`. */
export function tile(imgs: ArrayLike<number>[], cols: number, gap = 2, up = 1): { grey: Uint8Array; w: number; h: number } {
  const rows = Math.ceil(imgs.length / cols), cw = 28 * up + gap;
  const w = cols * cw - gap, h = rows * cw - gap, grey = new Uint8Array(w * h).fill(255);
  imgs.forEach((im, k) => {
    const ox = (k % cols) * cw, oy = Math.floor(k / cols) * cw;
    for (let y = 0; y < 28 * up; y++) for (let x = 0; x < 28 * up; x++)
      grey[(oy + y) * w + ox + x] = Math.round(255 * (1 - Math.max(0, Math.min(1, im[Math.floor(y / up) * 28 + Math.floor(x / up)]))));
  });
  return { grey, w, h };
}
