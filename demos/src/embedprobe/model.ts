// Model of the embedprobe demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// Three kinds of per-pixel embeddings on a 64 × 64 grid (every second pixel of a 128 × 128
// image), each a vector per grid position:
//   raw      the 9 × 9 patch of grey levels around the pixel (81-D): a baseline;
//   stack    the random-forest day's multiscale filter stack (lib/featurestack.ts), standardised;
//   learned  the tiny self-supervised encoder of train_ssl.py (16-D, int8 weights, lib/cnn.ts).
// Tools: cosine similarity to one position; a logistic-regression linear probe trained on
// painted positions; a "prompt" = similarity to a click, thresholded, the connected
// component containing the click. Plus the "new microscope" shifts (gain, blur, noise).
import { conv2d, convFromQ8, relu, tensor, type ConvLayer } from "../lib/cnn";
import { featureStack } from "../lib/featurestack";
import { gaussianBlur } from "../lib/imageops";
import { label2d } from "../lib/label";
import { rng } from "../lib/ui";
import { SSL_LAYERS } from "./ssl_weights";
import { W } from "./scene";

export const G = W / 2;       // embedding grid
export type Kind = "raw" | "stack" | "learned";

export interface Embedding { d: number; z: Float32Array }   // z[p * d + c], p on the G × G grid, rows L2-normalised

let layers: Record<string, ConvLayer> | null = null;
function sslLayers(): Record<string, ConvLayer> {
  if (!layers) {
    layers = {};
    for (const n of ["c1", "c2", "c3", "c4"]) layers[n] = convFromQ8(SSL_LAYERS[n], 3);
    layers.c5 = convFromQ8(SSL_LAYERS.c5, 1);
  }
  return layers;
}

function normaliseRows(z: Float32Array, d: number): void {
  for (let p = 0; p < z.length / d; p++) {
    let s = 0; for (let c = 0; c < d; c++) s += z[p * d + c] ** 2;
    const k = s > 0 ? 1 / Math.sqrt(s) : 0;
    for (let c = 0; c < d; c++) z[p * d + c] *= k;
  }
}

/** The learned encoder on a 128 × 128 image of grey levels 0..255. */
export function learnedEmbedding(img: ArrayLike<number>): Embedding {
  const L = sslLayers();
  const x = tensor(1, W, W, Float32Array.from(img as ArrayLike<number>, v => v / 255));
  let h = relu(conv2d(x, L.c1, { pad: 1, stride: 2 }));
  h = relu(conv2d(h, L.c2, { pad: 1 }));
  h = relu(conv2d(h, L.c3, { pad: 2, dilation: 2 }));
  h = relu(conv2d(h, L.c4, { pad: 4, dilation: 4 }));
  const o = conv2d(h, L.c5, { pad: 0 });
  const d = o.c, z = new Float32Array(G * G * d);
  for (let c = 0; c < d; c++) for (let p = 0; p < G * G; p++) z[p * d + c] = o.data[c * G * G + p];
  normaliseRows(z, d);
  return { d, z };
}

/** Raw 9 × 9 patches (edge replicated) at every second pixel. */
export function rawEmbedding(img: ArrayLike<number>): Embedding {
  const d = 81, z = new Float32Array(G * G * d);
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
    const p = gy * G + gx;
    let c = 0;
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
      const x = Math.max(0, Math.min(W - 1, 2 * gx + dx)), y = Math.max(0, Math.min(W - 1, 2 * gy + dy));
      z[p * d + c++] = img[y * W + x] / 255;
    }
  }
  normaliseRows(z, d);
  return { d, z };
}

/** The filter stack (intensity, edges, blobs, ridges, texture at σ = 1, 2, 4), each channel standardised over the image. */
export function stackEmbedding(img: ArrayLike<number>): Embedding {
  const S = featureStack({ width: W, height: W, data: Float32Array.from(img as ArrayLike<number>) }, ["intensity", "edges", "blobs", "ridges", "texture"], [1, 2, 4]);
  const d = S.d, z = new Float32Array(G * G * d);
  for (let c = 0; c < d; c++) {
    let m = 0, v = 0;
    const ch = S.data.subarray(c * W * W, (c + 1) * W * W);
    for (const x of ch) m += x; m /= ch.length;
    for (const x of ch) v += (x - m) ** 2; const sd = Math.sqrt(v / ch.length) || 1;
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) z[(gy * G + gx) * d + c] = (ch[2 * gy * W + 2 * gx] - m) / sd;
  }
  normaliseRows(z, d);
  return { d, z };
}

/**
 * The embedding used by the demo: each dimension centred and scaled to unit variance over the
 * image, then every vector normalised to length 1 (so similarity = cosine of centred vectors).
 */
export function embedding(img: ArrayLike<number>, kind: Kind): Embedding {
  const E = kind === "raw" ? rawEmbedding(img) : kind === "stack" ? stackEmbedding(img) : learnedEmbedding(img);
  const { d, z } = E, n = z.length / d;
  for (let c = 0; c < d; c++) {
    let m = 0; for (let p = 0; p < n; p++) m += z[p * d + c]; m /= n;
    let v = 0; for (let p = 0; p < n; p++) v += (z[p * d + c] - m) ** 2;
    const k = 1 / (Math.sqrt(v / n) || 1);
    for (let p = 0; p < n; p++) z[p * d + c] = (z[p * d + c] - m) * k;
  }
  normaliseRows(z, d);
  return E;
}

/** Cosine similarity of every grid position to position q. */
export function similarity(E: Embedding, q: number): Float32Array {
  const { d, z } = E, out = new Float32Array(G * G);
  for (let p = 0; p < G * G; p++) { let s = 0; for (let c = 0; c < d; c++) s += z[p * d + c] * z[q * d + c]; out[p] = s; }
  return out;
}

export interface Probe { w: Float64Array; b: number; d: number }

/** Logistic regression on the embeddings of labelled positions (1 = object, 0 = background); plain gradient descent, small L2. */
export function trainProbe(E: Embedding, pos: number[], y: number[], iters = 300, lr = 2, l2 = 1e-3): Probe {
  const d = E.d, w = new Float64Array(d), n = pos.length;
  let b = 0;
  if (!n) return { w, b, d };
  const gw = new Float64Array(d);
  for (let it = 0; it < iters; it++) {
    gw.fill(0); let gb = 0;
    for (let k = 0; k < n; k++) {
      const o = pos[k] * d;
      let s = b; for (let c = 0; c < d; c++) s += w[c] * E.z[o + c];
      const e = 1 / (1 + Math.exp(-s)) - y[k];
      for (let c = 0; c < d; c++) gw[c] += e * E.z[o + c];
      gb += e;
    }
    for (let c = 0; c < d; c++) w[c] -= lr * (gw[c] / n + l2 * w[c]);
    b -= lr * gb / n;
  }
  return { w, b, d };
}

export function applyProbe(E: Embedding, P: Probe): Float32Array {
  const out = new Float32Array(G * G);
  for (let p = 0; p < G * G; p++) { let s = P.b; for (let c = 0; c < P.d; c++) s += P.w[c] * E.z[p * P.d + c]; out[p] = 1 / (1 + Math.exp(-s)); }
  return out;
}

/** Prompt: positions with similarity ≥ thr, the 4-connected component containing q. */
export function promptMask(sim: Float32Array, q: number, thr: number): Uint8Array {
  const m = Uint8Array.from(sim, v => (v >= thr ? 1 : 0));
  const L = label2d(m, G, G, 4), l = L.data[q];
  return Uint8Array.from(L.data, v => (l && v === l ? 1 : 0));
}

/** Truth mask on the grid (every second pixel). */
export function gridTruth(t: Uint8Array): Uint8Array {
  const out = new Uint8Array(G * G);
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) out[gy * G + gx] = t[2 * gy * W + 2 * gx];
  return out;
}

export function dice(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let i = 0, s = 0;
  for (let p = 0; p < a.length; p++) { if (a[p] && b[p]) i++; s += (a[p] ? 1 : 0) + (b[p] ? 1 : 0); }
  return s ? 2 * i / s : 1;
}

export interface Shift { gain: number; blur: number; noise: number }

/** "Another microscope": gain on the grey levels, optical blur, extra noise (seeded). Rounded to integers. */
export function shifted(img: Float32Array, s: Shift): Float32Array {
  let out: Float32Array = Float32Array.from(img, v => v * s.gain);
  if (s.blur > 0) out = gaussianBlur({ width: W, height: W, data: out }, s.blur).data as Float32Array;
  if (s.noise > 0) { const r = rng(17); out = out.map(v => v + s.noise * r.norm()); }
  return out.map(v => Math.round(Math.max(0, Math.min(255, v))));
}
