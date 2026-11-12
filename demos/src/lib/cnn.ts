// Inference building blocks of convolutional networks (owner: neural-networks day): multi-
// channel 2D correlation with padding / stride / dilation, ReLU, max pooling, nearest
// upsampling, transposed convolution, channel concatenation, dense layers. Float32Arrays in
// channel-major layout (data[(c * h + y) * w + x]). No DOM, no training (weights are trained
// offline and loaded with lib/q8.ts). Generalises kernel3x3/conv.ts (one channel, 3 × 3).
//
// Like every network library, "convolution" here is a correlation (the kernel is not
// flipped); with learned weights the flip does not matter.
import { deq, type Q8Layer } from "./q8";

export interface Tensor { c: number; h: number; w: number; data: Float32Array }

export const tensor = (c: number, h: number, w: number, data?: Float32Array): Tensor =>
  ({ c, h, w, data: data ?? new Float32Array(c * h * w) });

/** One channel of a tensor as its own (copied) array. */
export function channel(t: Tensor, c: number): Float32Array {
  return t.data.slice(c * t.h * t.w, (c + 1) * t.h * t.w);
}

/** A convolutional layer: cout kernels of size cin × k × k (row-major), one bias each. */
export interface ConvLayer { W: Float32Array; b: Float32Array; cin: number; cout: number; k: number }

export function convFromQ8(l: Q8Layer, k: number): ConvLayer {
  const d = deq(l);
  return { W: d.W, b: d.b, cout: d.nOut, cin: d.nIn / (k * k), k };
}

/** Output size of a convolution / pooling: ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1. */
export const outSize = (n: number, k: number, p: number, s = 1, d = 1): number => Math.floor((n + 2 * p - d * (k - 1) - 1) / s) + 1;

/**
 * y[o, i, j] = b[o] + Σ_c Σ_{u,v} W[o, c, u, v] · x[c, i·s + u·d − p, j·s + v·d − p]
 * with zeros outside the image. pad "same" = d(k − 1)/2 (keeps the size for stride 1).
 */
export function conv2d(x: Tensor, L: ConvLayer, o: { pad?: number | "same" | "valid"; stride?: number; dilation?: number } = {}): Tensor {
  const k = L.k, s = o.stride ?? 1, d = o.dilation ?? 1;
  const p = o.pad === undefined || o.pad === "same" ? (d * (k - 1)) >> 1 : o.pad === "valid" ? 0 : o.pad;
  if (L.cin !== x.c) throw new Error(`conv2d: ${x.c} input channels, kernel wants ${L.cin}`);
  const H = outSize(x.h, k, p, s, d), Wd = outSize(x.w, k, p, s, d);
  const y = tensor(L.cout, H, Wd), xs = x.data, hw = x.h * x.w;
  for (let oc = 0; oc < L.cout; oc++) {
    const out = y.data.subarray(oc * H * Wd, (oc + 1) * H * Wd);
    out.fill(L.b[oc]);
    for (let c = 0; c < x.c; c++) {
      const xo = c * hw;
      for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) {
        const wv = L.W[((oc * L.cin + c) * k + u) * k + v];
        if (wv === 0) continue;
        const dy = u * d - p, dx = v * d - p;
        for (let i = 0; i < H; i++) {
          const yy = i * s + dy;
          if (yy < 0 || yy >= x.h) continue;
          const row = xo + yy * x.w, orow = i * Wd;
          for (let j = 0; j < Wd; j++) {
            const xx = j * s + dx;
            if (xx >= 0 && xx < x.w) out[orow + j] += wv * xs[row + xx];
          }
        }
      }
    }
  }
  return y;
}

export function relu(x: Tensor): Tensor {
  const y = tensor(x.c, x.h, x.w);
  for (let i = 0; i < x.data.length; i++) y.data[i] = x.data[i] > 0 ? x.data[i] : 0;
  return y;
}

/** Max pooling with a size × size window and the given stride (default = size); no padding. */
export function maxPool(x: Tensor, size = 2, stride = size): Tensor {
  const H = outSize(x.h, size, 0, stride), W = outSize(x.w, size, 0, stride), y = tensor(x.c, H, W);
  for (let c = 0; c < x.c; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    let m = -Infinity;
    for (let u = 0; u < size; u++) for (let v = 0; v < size; v++) m = Math.max(m, x.data[(c * x.h + i * stride + u) * x.w + j * stride + v]);
    y.data[(c * H + i) * W + j] = m;
  }
  return y;
}

/** Nearest-neighbour upsampling by an integer factor. */
export function upsample(x: Tensor, f = 2): Tensor {
  const y = tensor(x.c, x.h * f, x.w * f);
  for (let c = 0; c < x.c; c++) for (let i = 0; i < y.h; i++) for (let j = 0; j < y.w; j++)
    y.data[(c * y.h + i) * y.w + j] = x.data[(c * x.h + Math.floor(i / f)) * x.w + Math.floor(j / f)];
  return y;
}

/**
 * Transposed convolution ("up-convolution"): every input value stamps its own weighted copy of
 * the kernel into the output, at stride s. W is laid out as [cin][cout][k][k]. Output size
 * (n − 1)s + k − 2p. It is the adjoint (transpose) of conv2d with the same k, s, p.
 */
export function convTranspose2d(x: Tensor, W: Float32Array, b: Float32Array, cout: number, k: number, s = 2, p = 0): Tensor {
  const H = (x.h - 1) * s + k - 2 * p, Wd = (x.w - 1) * s + k - 2 * p, y = tensor(cout, H, Wd);
  for (let oc = 0; oc < cout; oc++) y.data.subarray(oc * H * Wd, (oc + 1) * H * Wd).fill(b[oc] ?? 0);
  for (let c = 0; c < x.c; c++) for (let i = 0; i < x.h; i++) for (let j = 0; j < x.w; j++) {
    const v = x.data[(c * x.h + i) * x.w + j];
    if (v === 0) continue;
    for (let oc = 0; oc < cout; oc++) for (let u = 0; u < k; u++) for (let q = 0; q < k; q++) {
      const yy = i * s + u - p, xx = j * s + q - p;
      if (yy >= 0 && yy < H && xx >= 0 && xx < Wd) y.data[(oc * H + yy) * Wd + xx] += v * W[((c * cout + oc) * k + u) * k + q];
    }
  }
  return y;
}

/** Channels of a, then channels of b (same height and width): a U-Net skip connection. */
export function concat(a: Tensor, b: Tensor): Tensor {
  if (a.h !== b.h || a.w !== b.w) throw new Error("concat: sizes differ");
  const y = tensor(a.c + b.c, a.h, a.w);
  y.data.set(a.data, 0); y.data.set(b.data, a.data.length);
  return y;
}

/** Dense layer on the flattened tensor (channel-major order): out = W x + b. */
export function dense(x: ArrayLike<number>, W: Float32Array, b: Float32Array, nOut: number): Float32Array {
  const nIn = x.length, y = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    for (let i = 0; i < nIn; i++) s += W[o * nIn + i] * x[i];
    y[o] = s;
  }
  return y;
}

/** Shift a tensor by (dx, dy) pixels, filling with `fill`. */
export function shift(x: Tensor, dx: number, dy: number, fill = 0): Tensor {
  const y = tensor(x.c, x.h, x.w);
  y.data.fill(fill);
  for (let c = 0; c < x.c; c++) for (let i = 0; i < x.h; i++) for (let j = 0; j < x.w; j++) {
    const ii = i - dy, jj = j - dx;
    if (ii >= 0 && ii < x.h && jj >= 0 && jj < x.w) y.data[(c * x.h + i) * x.w + j] = x.data[(c * x.h + ii) * x.w + jj];
  }
  return y;
}

/**
 * Receptive field of a stack of layers (kernel k, stride s, dilation d each): size r and jump j
 * (input pixels between neighbouring units) after every layer. r_l = r_{l−1} + d_l(k_l − 1) j_{l−1},
 * j_l = j_{l−1} s_l, starting from r_0 = j_0 = 1.
 */
export function receptiveField(layers: { k: number; s?: number; d?: number }[]): { r: number[]; j: number[] } {
  const r = [1], j = [1];
  for (const L of layers) {
    r.push(r[r.length - 1] + (L.d ?? 1) * (L.k - 1) * j[j.length - 1]);
    j.push(j[j.length - 1] * (L.s ?? 1));
  }
  return { r, j };
}
