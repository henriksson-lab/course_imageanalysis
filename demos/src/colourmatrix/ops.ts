// Y'CbCr round trip with subsampling, and the mean colour difference of two images (no DOM).
import { srgbToLinear, linearRgbToXyz, xyzToLab, RGB } from "../lib/colour";
import { rgbToYcbcr, ycbcrToRgb } from "../colourcloud/colourx";

/** Average a channel over k x k blocks and spread the mean back (subsample + upsample). */
export function subsample(ch: Float32Array, w: number, h: number, k: number): Float32Array {
  if (k <= 1) return ch;
  const out = new Float32Array(w * h);
  for (let by = 0; by < h; by += k) for (let bx = 0; bx < w; bx += k) {
    let s = 0, n = 0;
    for (let y = by; y < Math.min(h, by + k); y++) for (let x = bx; x < Math.min(w, bx + k); x++) { s += ch[y * w + x]; n++; }
    s /= n;
    for (let y = by; y < Math.min(h, by + k); y++) for (let x = bx; x < Math.min(w, bx + k); x++) out[y * w + x] = s;
  }
  return out;
}

/** Y'CbCr round trip with subsampling of chroma (target "chroma") or luma ("luma"). Input
 *  and output: encoded R'G'B' in 0..255 (planar). */
export function ycbcrRoundTrip(R: Float32Array, G: Float32Array, B: Float32Array, w: number, h: number, k: number, target: "chroma" | "luma") {
  const n = w * h, Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let p = 0; p < n; p++) { const v = rgbToYcbcr(R[p], G[p], B[p]); Y[p] = v[0]; Cb[p] = v[1]; Cr[p] = v[2]; }
  const Ys = target === "luma" ? subsample(Y, w, h, k) : Y;
  const Cbs = target === "chroma" ? subsample(Cb, w, h, k) : Cb, Crs = target === "chroma" ? subsample(Cr, w, h, k) : Cr;
  const oR = new Float32Array(n), oG = new Float32Array(n), oB = new Float32Array(n);
  for (let p = 0; p < n; p++) { const c = ycbcrToRgb(Ys[p], Cbs[p], Crs[p]); oR[p] = c[0]; oG[p] = c[1]; oB[p] = c[2]; }
  return { Y: Ys, Cb: Cbs, Cr: Crs, R: oR, G: oG, B: oB };
}

/** Mean ΔE*ab between two encoded (0..255) images. */
export function meanDeltaE(a: Float32Array[], b: Float32Array[]): number {
  const n = a[0].length;
  const lab = (c: Float32Array[], p: number) => xyzToLab(linearRgbToXyz([0, 1, 2].map(k => srgbToLinear(Math.min(1, Math.max(0, c[k][p] / 255)))) as RGB));
  let s = 0;
  for (let p = 0; p < n; p++) { const x = lab(a, p), y = lab(b, p); s += Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); }
  return s / n;
}

