// A small PNG decoder (8- and 16-bit grey, grey+alpha, RGB, RGBA; no palette, no interlace):
// enough for the sample images and the 16-bit label images of demos/dist/models/images/,
// which a canvas would reduce to 8 bits. The zlib inflate is passed in (browser:
// DecompressionStream("deflate"); node: zlib.inflateSync), so this file has no DOM.

export interface Png { w: number; h: number; channels: number; depth: number; data: Uint16Array }

export type Inflate = (z: Uint8Array) => Promise<Uint8Array>;

/** Browser inflate (zlib stream). */
export async function inflateBrowser(z: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate");
  const stream = new Blob([z as BlobPart]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function decodePng(bytes: Uint8Array, inflate: Inflate): Promise<Png> {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (bytes[i] !== sig[i]) throw new Error("not a PNG file");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = 8, w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
  const idat: Uint8Array[] = [];
  while (p < bytes.length) {
    const len = dv.getUint32(p), type = String.fromCharCode(...bytes.subarray(p + 4, p + 8));
    const body = bytes.subarray(p + 8, p + 8 + len);
    if (type === "IHDR") {
      w = dv.getUint32(p + 8); h = dv.getUint32(p + 12); depth = body[8]; ctype = body[9]; interlace = body[12];
    } else if (type === "IDAT") idat.push(body);
    else if (type === "IEND") break;
    p += 12 + len;
  }
  const channels = ({ 0: 1, 2: 3, 4: 2, 6: 4 } as Record<number, number>)[ctype];
  if (!channels || (depth !== 8 && depth !== 16) || interlace) throw new Error(`unsupported PNG (colour type ${ctype}, ${depth} bit)`);
  const zlen = idat.reduce((s, c) => s + c.length, 0);
  const z = new Uint8Array(zlen);
  let o = 0;
  for (const c of idat) { z.set(c, o); o += c.length; }
  const raw = await inflate(z);
  const bpp = channels * (depth / 8), stride = w * bpp;
  const cur = new Uint8Array(stride), prev = new Uint8Array(stride);
  const data = new Uint16Array(w * h * channels);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = v & 255;
    }
    const n = w * channels;
    for (let i = 0; i < n; i++) data[y * n + i] = depth === 16 ? (cur[2 * i] << 8) | cur[2 * i + 1] : cur[i];
    prev.set(cur);
  }
  return { w, h, channels, depth, data };
}
