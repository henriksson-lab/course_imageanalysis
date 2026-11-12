// The small CNN of the cnnviewer demo (inference only, int8 weights from train_cnn.py; no DOM,
// so the tests run it in node): conv 3×3, 1→8, ReLU, 2×2 max pool → conv 3×3, 8→16, ReLU,
// 2×2 max pool → flatten 16·7·7 = 784 → dense → softmax.
import { conv2d, relu, maxPool, dense, convFromQ8, tensor, type ConvLayer, type Tensor } from "../lib/cnn";
import { deq } from "../lib/q8";
import { softmax } from "../lib/nn";
import { CNN_SETS } from "./cnn_weights";

export interface Cnn { c1: ConvLayer; c2: ConvLayer; fc: { W: Float32Array; b: Float32Array; nOut: number } }
const cache: Record<string, Cnn> = {};
export function getCnn(set: string): Cnn {
  if (cache[set]) return cache[set];
  const w = CNN_SETS[set], fc = deq(w.fc);
  return (cache[set] = { c1: convFromQ8(w.c1, 3), c2: convFromQ8(w.c2, 3), fc: { W: fc.W, b: fc.b, nOut: fc.nOut } });
}

export interface CnnRun { a1: Tensor; p1: Tensor; a2: Tensor; p2: Tensor; logits: Float32Array; probs: Float64Array }

/** Forward pass on a 28 × 28 image with values 0..1 (ink). */
export function runCnn(img: ArrayLike<number>, set: string): CnnRun {
  const n = getCnn(set);
  const x = tensor(1, 28, 28, Float32Array.from(img));
  const a1 = relu(conv2d(x, n.c1)), p1 = maxPool(a1);
  const a2 = relu(conv2d(p1, n.c2)), p2 = maxPool(a2);
  const logits = dense(p2.data, n.fc.W, n.fc.b, n.fc.nOut);
  return { a1, p1, a2, p2, logits, probs: softmax(logits) };
}

/** Rotate (degrees, counter-clockwise on screen) about the centre and shift (pixels), bilinear, zero outside. */
export function transform(img: ArrayLike<number>, deg: number, dx: number, dy: number): Float32Array {
  const out = new Float32Array(784), a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), m = 13.5;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    // inverse map: output pixel → source position
    const u = x - dx - m, v = y - dy - m;
    const sx = c * u - s * v + m, sy = s * u + c * v + m;
    const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
    const at = (xx: number, yy: number) => (xx >= 0 && xx < 28 && yy >= 0 && yy < 28 ? img[yy * 28 + xx] : 0);
    out[y * 28 + x] = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
  }
  return out;
}

/** Occlusion map: probability of class k when a size × size square (value 0) covers each position (stride 2). */
export function occlusion(img: ArrayLike<number>, set: string, k: number, size = 6, stride = 2): { map: Float32Array; n: number } {
  const n = Math.floor((28 - size) / stride) + 1, map = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = Float32Array.from(img);
    for (let y = i * stride; y < i * stride + size; y++) for (let xx = j * stride; xx < j * stride + size; xx++) x[y * 28 + xx] = 0;
    map[i * n + j] = runCnn(x, set).probs[k];
  }
  return { map, n };
}
