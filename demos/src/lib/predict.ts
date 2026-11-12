// Lossless predictive coding (DPCM) on integer images (no DOM). Owner: the compression day
// (lossless/ unit).
//
// Each pixel x is predicted from already-decoded neighbours
//       c  b          a = left (W), b = up (N), c = up-left (NW)
//       a  x
// and only the residual e = x − x̂ is stored. The decoder makes the same prediction in the
// same raster order and adds e back, so the round trip is exact.
// Borders: the first pixel is predicted as 0, the rest of the first row from the left
// neighbour, the rest of the first column from the pixel above (all predictors).
// Predictors: none (0), left (a), up (b), average ⌊(a+b)/2⌋, planar a + b − c,
// Paeth (PNG: whichever of a, b, c is closest to a + b − c) and MED (JPEG-LS / LOCO-I:
// the median of a, b and a + b − c, which picks min(a,b) or max(a,b) at an edge).

export type Predictor = "none" | "left" | "up" | "average" | "planar" | "paeth" | "med";

export const PREDICTORS: Predictor[] = ["none", "left", "up", "average", "planar", "paeth", "med"];

/** Prediction from the three neighbours (interior pixels). */
export function predictFrom(p: Predictor, a: number, b: number, c: number): number {
  switch (p) {
    case "none": return 0;
    case "left": return a;
    case "up": return b;
    case "average": return Math.floor((a + b) / 2);
    case "planar": return a + b - c;
    case "paeth": {
      const q = a + b - c, pa = Math.abs(q - a), pb = Math.abs(q - b), pc = Math.abs(q - c);
      return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
    case "med": {
      const mx = Math.max(a, b), mn = Math.min(a, b);
      return c >= mx ? mn : c <= mn ? mx : a + b - c;
    }
  }
}

/** Prediction of pixel (x, y) from the (decoded) image v, with the border rules above. */
export function predictAt(p: Predictor, v: ArrayLike<number>, w: number, x: number, y: number): number {
  if (p === "none") return 0;
  if (x === 0 && y === 0) return 0;
  if (y === 0) return v[x - 1];
  if (x === 0) return v[(y - 1) * w];
  const i = y * w + x;
  return predictFrom(p, v[i - 1], v[i - w], v[i - w - 1]);
}

/** Residuals e = x − x̂ of an integer image (signed integers). */
export function residuals(img: ArrayLike<number>, w: number, h: number, p: Predictor): Int32Array {
  const e = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    e[i] = Math.round(img[i]) - predictAt(p, img, w, x, y);
  }
  return e;
}

/** Predictions x̂ for every pixel (for display). */
export function predictions(img: ArrayLike<number>, w: number, h: number, p: Predictor): Int32Array {
  const out = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = predictAt(p, img, w, x, y);
  return out;
}

/** Decoder: rebuild the image from residuals in raster order. Exact inverse of `residuals`. */
export function reconstruct(e: ArrayLike<number>, w: number, h: number, p: Predictor): Int32Array {
  const v = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    v[i] = e[i] + predictAt(p, v, w, x, y);
  }
  return v;
}

/** Residuals of a 1D row with the left predictor (the first value is kept as it is). */
export function rowResiduals(row: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < row.length; i++) out.push(i === 0 ? row[0] : row[i] - row[i - 1]);
  return out;
}

/** Bytes for a real compressor: 8-bit data as (value mod 256), as PNG stores filtered rows;
 *  wider data as 16-bit little-endian two's complement. */
export function toBytes(values: ArrayLike<number>, bytesPerValue: 1 | 2): Uint8Array {
  const out = new Uint8Array(values.length * bytesPerValue);
  for (let i = 0; i < values.length; i++) {
    const v = Math.round(values[i]);
    if (bytesPerValue === 1) out[i] = ((v % 256) + 256) % 256;
    else { const u = ((v % 65536) + 65536) % 65536; out[2 * i] = u & 255; out[2 * i + 1] = u >> 8; }
  }
  return out;
}
