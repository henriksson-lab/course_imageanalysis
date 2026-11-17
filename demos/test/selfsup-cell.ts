// Helper for the tests of the "Learning without labels" day (blindspot, maskfill): the course
// micrograph demos/src/fourier2d/cell256.png (8-bit grey, non-interlaced) decoded in node,
// grey levels 0..255, 256 × 256. Run the tests from the repository root.
declare const require: (m: string) => any;

export function loadCellNode(): { width: number; height: number; data: Float64Array } {
  const fs = require("fs"), zlib = require("zlib"), path = require("path");
  const g = globalThis as any;
  const buf: Uint8Array = fs.readFileSync(path.join(g.process.cwd(), "demos/src/fourier2d/cell256.png"));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 8, w = 0, h = 0;
  const idat: Uint8Array[] = [];
  while (p < buf.length) {
    const len = dv.getUint32(p), type = String.fromCharCode(...buf.subarray(p + 4, p + 8));
    if (type === "IHDR") { w = dv.getUint32(p + 8); h = dv.getUint32(p + 12); }
    if (type === "IDAT") idat.push(buf.subarray(p + 8, p + 8 + len));
    p += 12 + len;
  }
  const raw: Uint8Array = zlib.inflateSync(g.Buffer.concat(idat));
  const out = new Float64Array(w * h), prev = new Uint8Array(w), cur = new Uint8Array(w);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (w + 1)], row = raw.subarray(y * (w + 1) + 1, (y + 1) * (w + 1));
    for (let x = 0; x < w; x++) {
      const a = x > 0 ? cur[x - 1] : 0, b = prev[x], c = x > 0 ? prev[x - 1] : 0;
      let v = row[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    for (let x = 0; x < w; x++) out[y * w + x] = cur[x];
    prev.set(cur);
  }
  return { width: w, height: h, data: out };
}

/** A crop of the micrograph (grey levels 0..255). */
export function cellCrop(x0: number, y0: number, w: number, h: number): Float64Array {
  const c = loadCellNode(), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = c.data[(y + y0) * c.width + x + x0];
  return out;
}
