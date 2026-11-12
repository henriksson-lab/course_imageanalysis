// Gamma-encoded vs linear-light blur and downsampling (no DOM): test images as encoded sRGB
// in [0, 1], the operation done naively on encoded values or as decode -> op -> encode.
import { rng } from "../lib/random";
import { srgbToLinear, linearToSrgb } from "../lib/colour";
import { createGrey, gaussianBlur, Grey } from "../lib/imageops";

export const N = 128;
export type Img = [Float32Array, Float32Array, Float32Array]; // encoded sRGB in [0, 1], N x N
export type ImageId = "stripes" | "checker" | "lights" | "lines";
export type Op = "blur" | "down2" | "down4";

const newImg = (): Img => [new Float32Array(N * N), new Float32Array(N * N), new Float32Array(N * N)];

/** Test images, as encoded sRGB values in [0, 1]. */
export function makeImage(id: ImageId): Img {
  const I = newImg();
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x;
    let c = [0, 0, 0];
    if (id === "stripes") {
      // left: 1-pixel black/white stripes; right: uniform 128 (top) and 188 (bottom)
      if (x < 64) c = x % 2 ? [1, 1, 1] : [0, 0, 0];
      else { const v = (y < 64 ? 128 : 188) / 255; c = [v, v, v]; }
    } else if (id === "checker") {
      const red = (Math.floor(x / 32) + Math.floor(y / 32)) % 2 === 0;
      c = red ? [1, 0, 0] : [0, 1, 0];
    } else if (id === "lines") {
      const lines = [[10, 1], [22, 1], [34, 2], [48, 3]];
      const on = lines.some(([x0, w]) => x >= x0 && x < x0 + w) || (x >= 64 && (x + y) % 6 === 0) ;
      c = on ? [1, 1, 1] : [0, 0, 0];
      if (y > 100 && x < 64) c = x % 4 < 2 ? [1, 0.85, 0.3] : [0.05, 0.05, 0.25];
    }
    I[0][p] = c[0]; I[1][p] = c[1]; I[2][p] = c[2];
  }
  if (id === "lights") {
    // seeded small bright lights on a dark night sky, some saturated (clipped at 1)
    const r = rng(17), lin = [new Float32Array(N * N).fill(0.002), new Float32Array(N * N).fill(0.003), new Float32Array(N * N).fill(0.008)];
    const cols = [[1, 0.75, 0.35], [1, 0.3, 0.2], [0.4, 0.8, 1], [1, 1, 1], [0.3, 1, 0.4]];
    for (let k = 0; k < 40; k++) {
      const cx = 4 + r.unif() * (N - 8), cy = 4 + r.unif() * (N - 8), s = 0.6 + 0.9 * r.unif(), a = 1 + 5 * r.unif(), col = cols[k % cols.length];
      for (let y = Math.floor(cy - 5); y <= cy + 5; y++) for (let x = Math.floor(cx - 5); x <= cx + 5; x++) {
        if (x < 0 || y < 0 || x >= N || y >= N) continue;
        const g = a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
        for (let c = 0; c < 3; c++) lin[c][y * N + x] += g * col[c];
      }
    }
    for (let c = 0; c < 3; c++) for (let p = 0; p < N * N; p++) I[c][p] = linearToSrgb(Math.min(1, lin[c][p]));
  }
  return I;
}

/** The operation on one channel (values in whatever domain they come). */
function op(ch: Float32Array, o: Op, sigma: number): Float32Array {
  if (o === "blur") {
    const g: Grey = createGrey(N, N);
    g.data.set(ch);
    return Float32Array.from(gaussianBlur(g, sigma, "mirror").data);
  }
  // block average, shown at the original size (each block one colour)
  const k = o === "down2" ? 2 : 4, out = new Float32Array(N * N);
  for (let by = 0; by < N; by += k) for (let bx = 0; bx < N; bx += k) {
    let s = 0;
    for (let y = by; y < by + k; y++) for (let x = bx; x < bx + k; x++) s += ch[y * N + x];
    s /= k * k;
    for (let y = by; y < by + k; y++) for (let x = bx; x < bx + k; x++) out[y * N + x] = s;
  }
  return out;
}

/** Naive (on encoded values) and correct (in linear light) results, both encoded. */
export function process(I: Img, o: Op, sigma: number): { naive: Img; linear: Img } {
  const naive = I.map(ch => op(ch, o, sigma)) as Img;
  const linear = I.map(ch => {
    const l = op(Float32Array.from(ch, srgbToLinear), o, sigma);
    return Float32Array.from(l, v => linearToSrgb(Math.min(1, Math.max(0, v))));
  }) as Img;
  return { naive, linear };
}

/** Mean luminance Y (linear light) of an encoded image. */
export function meanLuminance(I: Img): number {
  let s = 0;
  for (let p = 0; p < N * N; p++) s += 0.2126 * srgbToLinear(I[0][p]) + 0.7152 * srgbToLinear(I[1][p]) + 0.0722 * srgbToLinear(I[2][p]);
  return s / (N * N);
}

