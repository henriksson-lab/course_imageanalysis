// 3 x 3 image filtering, kept free of DOM code so it can be tested in Node.
// Images are Float64Array of grey levels (row-major, row 0 at the top), usually 0..255.
// The kernel is 9 numbers, read row by row:  a b c / d e f / g h i.
// We compute the weighted sum WITHOUT flipping the kernel ("correlation"), which is
// what image software calls a filter; for symmetric kernels it is the same as convolution.

export type Border = "replicate" | "zero";

export interface Preset {
  id: string;
  label: string;
  k: number[];
  normalise: boolean;          // divide by the sum of the weights (when it is not 0)
  display: Display;            // the natural way to show the output
}

export type Display = "clip" | "abs" | "mid";

export const PRESETS: Preset[] = [
  { id: "identity", label: "identity", k: [0, 0, 0, 0, 1, 0, 0, 0, 0], normalise: false, display: "clip" },
  { id: "box", label: "box blur", k: [1, 1, 1, 1, 1, 1, 1, 1, 1], normalise: true, display: "clip" },
  { id: "gauss", label: "Gaussian", k: [1, 2, 1, 2, 4, 2, 1, 2, 1], normalise: true, display: "clip" },
  { id: "sharpen", label: "sharpen", k: [0, -1, 0, -1, 5, -1, 0, -1, 0], normalise: false, display: "clip" },
  { id: "laplace", label: "Laplacian", k: [0, 1, 0, 1, -4, 1, 0, 1, 0], normalise: false, display: "mid" },
  { id: "sobelx", label: "Sobel x", k: [-1, 0, 1, -2, 0, 2, -1, 0, 1], normalise: false, display: "mid" },
  { id: "sobely", label: "Sobel y", k: [-1, -2, -1, 0, 0, 0, 1, 2, 1], normalise: false, display: "mid" },
  { id: "emboss", label: "emboss", k: [-2, -1, 0, -1, 1, 1, 0, 1, 2], normalise: false, display: "clip" },
  { id: "shift", label: "shift", k: [0, 0, 0, 1, 0, 0, 0, 0, 0], normalise: false, display: "clip" },
];

export function kernelSum(k: number[]): number {
  return k.reduce((s, v) => s + v, 0);
}

/** The divisor actually used: the sum of the weights if normalising and the sum is not ~0, else 1. */
export function divisor(k: number[], normalise: boolean): number {
  const s = kernelSum(k);
  return normalise && Math.abs(s) > 1e-9 ? s : 1;
}

/** Pixel value at (row, col), with the border rule for positions outside the image. */
export function pixelAt(img: ArrayLike<number>, w: number, h: number, row: number, col: number, border: Border): number {
  if (row < 0 || row >= h || col < 0 || col >= w) {
    if (border === "zero") return 0;
    row = Math.min(h - 1, Math.max(0, row));
    col = Math.min(w - 1, Math.max(0, col));
  }
  return img[row * w + col];
}

/** The 3 x 3 neighbourhood of (row, col), row by row. */
export function neighbourhood(img: ArrayLike<number>, w: number, h: number, row: number, col: number, border: Border): number[] {
  const out: number[] = [];
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) out.push(pixelAt(img, w, h, row + dr, col + dc, border));
  return out;
}

/** Apply the kernel once: each output pixel = (sum of weight x pixel under it) / div. */
export function filter3(img: ArrayLike<number>, w: number, h: number, k: number[], div: number, border: Border): Float64Array {
  const out = new Float64Array(w * h);
  const [a, b, c, d, e, f, g, hh, i] = k;
  for (let r = 0; r < h; r++) {
    const inside = r > 0 && r < h - 1;
    for (let col = 0; col < w; col++) {
      let s: number;
      if (inside && col > 0 && col < w - 1) {       // fast path, no border checks
        const p = r * w + col;
        s = a * img[p - w - 1] + b * img[p - w] + c * img[p - w + 1]
          + d * img[p - 1] + e * img[p] + f * img[p + 1]
          + g * img[p + w - 1] + hh * img[p + w] + i * img[p + w + 1];
      } else {
        const n = neighbourhood(img, w, h, r, col, border);
        s = 0;
        for (let q = 0; q < 9; q++) s += k[q] * n[q];
      }
      out[r * w + col] = s / div;
    }
  }
  return out;
}

/** Apply the kernel `times` times in a row. */
export function filterRepeat(img: ArrayLike<number>, w: number, h: number, k: number[], div: number,
  border: Border, times: number): Float64Array {
  let cur: Float64Array = Float64Array.from(img);
  for (let t = 0; t < times; t++) cur = filter3(cur, w, h, k, div, border);
  return cur;
}

/** Map output values to grey levels 0..255 for display. Returns the grey levels and the value range. */
export function toGrey(v: Float64Array, mode: Display): { grey: Uint8ClampedArray; lo: number; hi: number } {
  let lo = Infinity, hi = -Infinity, amax = 0;
  for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; if (Math.abs(x) > amax) amax = Math.abs(x); }
  const grey = new Uint8ClampedArray(v.length);
  const sc = amax > 1e-9 ? amax : 1;
  for (let q = 0; q < v.length; q++) {
    const x = v[q];
    grey[q] = mode === "clip" ? Math.round(x)
      : mode === "abs" ? Math.round(255 * Math.abs(x) / sc)
      : Math.round(127.5 + 127.5 * x / sc);
  }
  return { grey, lo, hi };
}
