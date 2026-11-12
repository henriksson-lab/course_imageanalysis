// The two image VAEs of the vaeimages demo: int8 weights, forward passes, and preprocessing of
// a drawing. No DOM, so it can be run in node (demos/test/vae-check.ts computes the numbers
// quoted in the exercises with exactly this code).
// Adapted from digits.ts of the vaegen demo (3MB00A course); generalised to any weight set.
//
// Both networks: encoder 784 → 192 ReLU → (mean, log-variance) of a 2D normal;
// decoder 2 → 192 ReLU → 784 sigmoid (the "ink" of every pixel, 0..1).
import { MNIST_VAE, type Q8Layer } from "./mnist_weights";
import { SHAPE_VAE } from "./shape_weights";

export type V2 = [number, number];
interface Dense { W: Float32Array; b: Float32Array; nIn: number; nOut: number }
type Weights = { e1: Q8Layer; em: Q8Layer; ev: Q8Layer; d1: Q8Layer; d2: Q8Layer };

function deq(l: Q8Layer): Dense {
  const bin = atob(l.q), W = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const q = bin.charCodeAt(i), v = q > 127 ? q - 256 : q;
    W[i] = v * l.s[Math.floor(i / l.nIn)];
  }
  return { W, b: Float32Array.from(l.b), nIn: l.nIn, nOut: l.nOut };
}
function lin(d: Dense, x: ArrayLike<number>, relu: boolean): Float32Array {
  const y = new Float32Array(d.nOut);
  for (let o = 0; o < d.nOut; o++) {
    let s = d.b[o];
    const r = o * d.nIn;
    for (let i = 0; i < d.nIn; i++) s += d.W[r + i] * x[i];
    y[o] = relu ? Math.max(0, s) : s;
  }
  return y;
}

export interface ImageVae {
  encode(img: ArrayLike<number>): { mu: V2; sd: V2 };
  decode(z: ArrayLike<number>): Float32Array;
}
function makeVae(w: Weights): ImageVae {
  let net: { e1: Dense; em: Dense; ev: Dense; d1: Dense; d2: Dense } | null = null;
  const get = () => net ??= { e1: deq(w.e1), em: deq(w.em), ev: deq(w.ev), d1: deq(w.d1), d2: deq(w.d2) };
  return {
    decode(z) { const n = get(); return lin(n.d2, lin(n.d1, z, true), false).map(v => 1 / (1 + Math.exp(-v))); },
    encode(img) {
      const n = get(), h = lin(n.e1, img, true), m = lin(n.em, h, false), v = lin(n.ev, h, false);
      return { mu: [m[0], m[1]], sd: [Math.exp(0.5 * v[0]), Math.exp(0.5 * v[1])] };
    },
  };
}
export const digitVae = makeVae(MNIST_VAE);
export const shapeVae = makeVae(SHAPE_VAE);

/** Mean squared pixel error between two 784-pixel images. */
export function mse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < 784; i++) s += (a[i] - b[i]) ** 2;
  return s / 784;
}

/** A base64 string of 784 grey bytes → image with values in [0, 1]. */
export function fromB64(s: string): Float32Array {
  const bin = atob(s), out = new Float32Array(784);
  for (let i = 0; i < 784; i++) out[i] = bin.charCodeAt(i) / 255;
  return out;
}

/** Shift a 28 × 28 image so that its centre of mass is at pixel (13.5, 13.5). */
function centre(box: Float32Array): Float32Array {
  let s = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { s += box[i]; cx += box[i] * (i % 28); cy += box[i] * Math.floor(i / 28); }
  const dx = Math.round(13.5 - cx / s), dy = Math.round(13.5 - cy / s);
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const X = x + dx, Y = y + dy;
    if (X >= 0 && X < 28 && Y >= 0 && Y < 28) out[Y * 28 + X] = box[y * 28 + x];
  }
  return out;
}

/**
 * MNIST-style preprocessing of a drawing (ink in [0, 1], D×D): crop to the ink's bounding box,
 * scale the longer side to 20 px (box filter), then centre the centre of mass.
 */
export function preprocessDigit(ink: Float32Array, D: number): Float32Array | null {
  let x0 = D, x1 = -1, y0 = D, y1 = -1;
  for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) if (ink[y * D + x] > 0.05) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  if (x1 < 0) return null;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, sc = 20 / Math.max(bw, bh);
  const box = new Float32Array(784);
  const ox = (28 - bw * sc) / 2, oy = (28 - bh * sc) / 2;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const v = ink[y * D + x];
    if (v <= 0) continue;
    const tx = Math.floor(ox + (x - x0 + 0.5) * sc), ty = Math.floor(oy + (y - y0 + 0.5) * sc);
    if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) box[ty * 28 + tx] += v * sc * sc;
  }
  for (let i = 0; i < 784; i++) box[i] = Math.min(1, box[i]);
  return centre(box);
}

/** Preprocessing of a drawn cell mask: average D/28 × D/28 blocks (size is kept), centre of mass. */
export function preprocessShape(ink: Float32Array, D: number): Float32Array | null {
  const box = new Float32Array(784), f = D / 28;
  let any = false;
  for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) {
    const v = ink[y * D + x];
    if (v > 0.05) any = true;
    box[Math.min(27, Math.floor(y / f)) * 28 + Math.min(27, Math.floor(x / f))] += v / (f * f);
  }
  if (!any) return null;
  for (let i = 0; i < 784; i++) box[i] = Math.min(1, box[i]);
  return centre(box);
}
