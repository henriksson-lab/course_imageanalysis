// A tiny multilayer perceptron for the teaching demos: dense layers, ReLU or identity,
// mean-squared-error loss, backpropagation and Adam. Flat Float64Arrays, no DOM, so it
// runs in node for tests. Small on purpose: networks of a few hundred weights, trained live.
//
// Layer l maps a[l] (size sizes[l]) to a[l+1] = act_l(W_l a[l] + b_l), W_l row-major
// (W[o * nIn + i]). Used by nnfit (1 → H → 1) and linae (2 → H → 1 → H → 2).
//
// Extended for the neural-networks day (nnplayground, backprop tests): activations tanh and
// sigmoid, softmax + cross-entropy and sigmoid + binary cross-entropy losses, seeded
// mini-batches, plain SGD with momentum next to Adam, weight decay and dropout (training only).
// The original functions (mlp, forward, trainStep with Adam) give bit-identical results for
// ReLU / identity networks (demos/test/neuralnets-test.ts checks nnfit and linae).
import { rng } from "./ui";

export type Act = "relu" | "id" | "tanh" | "sigmoid";

/** The activation φ(s). */
export function act(a: Act, s: number): number {
  return a === "relu" ? Math.max(0, s) : a === "id" ? s : a === "tanh" ? Math.tanh(s) : 1 / (1 + Math.exp(-s));
}
/** Its derivative φ'(s), given s (pre) and y = φ(s). ReLU'(0) is taken as 0. */
export function actDeriv(a: Act, s: number, y: number): number {
  return a === "relu" ? (s > 0 ? 1 : 0) : a === "id" ? 1 : a === "tanh" ? 1 - y * y : y * (1 - y);
}

export interface Mlp {
  sizes: number[];
  acts: Act[];
  W: Float64Array[];
  b: Float64Array[];
}

/** He-initialised network with a fixed seed. */
export function mlp(sizes: number[], acts: Act[], seed = 1): Mlp {
  const r = rng(seed);
  const W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 0; l + 1 < sizes.length; l++) {
    const nIn = sizes[l], nOut = sizes[l + 1], s = Math.sqrt((acts[l] === "relu" ? 2 : 1) / nIn);
    W.push(Float64Array.from({ length: nIn * nOut }, () => s * r.norm()));
    b.push(new Float64Array(nOut));
  }
  return { sizes, acts, W, b };
}

export function cloneMlp(n: Mlp): Mlp {
  return { sizes: [...n.sizes], acts: [...n.acts], W: n.W.map(w => w.slice()), b: n.b.map(v => v.slice()) };
}

/** All layer outputs for one input: a[0] = x, a[L] = network output. pre[l] = before the bend. */
export function forward(n: Mlp, x: ArrayLike<number>): { a: Float64Array[]; pre: Float64Array[] } {
  const a: Float64Array[] = [Float64Array.from(x)], pre: Float64Array[] = [];
  for (let l = 0; l < n.W.length; l++) {
    const nIn = n.sizes[l], nOut = n.sizes[l + 1], W = n.W[l], inp = a[l];
    const z = new Float64Array(nOut), y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = n.b[l][o];
      for (let i = 0; i < nIn; i++) s += W[o * nIn + i] * inp[i];
      z[o] = s;
      y[o] = act(n.acts[l], s);
    }
    pre.push(z); a.push(y);
  }
  return { a, pre };
}

export const predict = (n: Mlp, x: ArrayLike<number>): Float64Array => {
  const f = forward(n, x).a;
  return f[f.length - 1];
};

/** Mean squared error (averaged over samples and outputs). */
export function mse(n: Mlp, X: ArrayLike<number>[], Y: ArrayLike<number>[]): number {
  let s = 0, k = 0;
  X.forEach((x, i) => { const p = predict(n, x); for (let j = 0; j < p.length; j++) { s += (p[j] - Y[i][j]) ** 2; k++; } });
  return s / k;
}

/** Adam optimiser state for one network. */
export class Adam {
  t = 0;
  mW: Float64Array[]; vW: Float64Array[]; mb: Float64Array[]; vb: Float64Array[];
  constructor(n: Mlp, public lr = 0.01) {
    this.mW = n.W.map(w => new Float64Array(w.length)); this.vW = n.W.map(w => new Float64Array(w.length));
    this.mb = n.b.map(v => new Float64Array(v.length)); this.vb = n.b.map(v => new Float64Array(v.length));
  }
}

/** One full-batch gradient step on the mean squared error; returns the loss before the step. */
export function trainStep(n: Mlp, X: ArrayLike<number>[], Y: ArrayLike<number>[], opt: Adam): number {
  const L = n.W.length;
  const gW = n.W.map(w => new Float64Array(w.length)), gb = n.b.map(v => new Float64Array(v.length));
  const nOutL = n.sizes[L];
  const norm = 1 / (X.length * nOutL);
  let loss = 0;
  for (let s = 0; s < X.length; s++) {
    const { a, pre } = forward(n, X[s]);
    let d = new Float64Array(nOutL);
    for (let j = 0; j < nOutL; j++) { const e = a[L][j] - Y[s][j]; loss += e * e; d[j] = 2 * e * norm; }
    for (let l = L - 1; l >= 0; l--) {
      const nIn = n.sizes[l], nOut = n.sizes[l + 1];
      if (n.acts[l] === "relu") { for (let o = 0; o < nOut; o++) if (pre[l][o] <= 0) d[o] = 0; }
      else if (n.acts[l] !== "id") for (let o = 0; o < nOut; o++) d[o] *= actDeriv(n.acts[l], pre[l][o], a[l + 1][o]);
      const dPrev = new Float64Array(nIn);
      for (let o = 0; o < nOut; o++) {
        gb[l][o] += d[o];
        if (d[o] === 0) continue;
        for (let i = 0; i < nIn; i++) { gW[l][o * nIn + i] += d[o] * a[l][i]; dPrev[i] += d[o] * n.W[l][o * nIn + i]; }
      }
      d = dPrev;
    }
  }
  opt.t++;
  const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** opt.t, c2 = 1 - b2 ** opt.t;
  const upd = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
    for (let k = 0; k < p.length; k++) {
      m[k] = b1 * m[k] + (1 - b1) * g[k];
      v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
      p[k] -= opt.lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
    }
  };
  for (let l = 0; l < L; l++) { upd(n.W[l], gW[l], opt.mW[l], opt.vW[l]); upd(n.b[l], gb[l], opt.mb[l], opt.vb[l]); }
  return loss * norm;
}

// ---------------------------------------------------------------------------------------
// General training (neural-networks day): losses, mini-batches, SGD, weight decay, dropout.

/**
 * "mse": mean squared error (averaged over samples and outputs, as trainStep).
 * "bce": one output logit s, p = sigmoid(s), loss −[y log p + (1 − y) log(1 − p)], y ∈ {0, 1}.
 * "ce":  k output logits, p = softmax(s), loss −log p_true; targets are one-hot vectors.
 * bce and ce are averaged over samples; their gradient with respect to the logits is p − y.
 */
export type Loss = "mse" | "bce" | "ce";

/** Softmax of a score vector (stable: the largest score is subtracted first). */
export function softmax(z: ArrayLike<number>): Float64Array {
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) m = Math.max(m, z[i]);
  const p = new Float64Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
  for (let i = 0; i < z.length; i++) p[i] /= s;
  return p;
}

/** Network outputs turned into probabilities for a loss ("mse": unchanged). */
export function outputProbs(out: ArrayLike<number>, loss: Loss): Float64Array {
  if (loss === "ce") return softmax(out);
  if (loss === "bce") return Float64Array.of(1 / (1 + Math.exp(-out[0])));
  return Float64Array.from(out);
}

/** Loss of one sample, from the output (logits) and the target. */
export function sampleLoss(out: ArrayLike<number>, y: ArrayLike<number>, loss: Loss): number {
  if (loss === "mse") { let s = 0; for (let j = 0; j < out.length; j++) s += (out[j] - y[j]) ** 2; return s / out.length; }
  if (loss === "bce") {
    // −[y log σ(s) + (1−y) log(1−σ(s))] = softplus(s) − y s, computed stably
    const s = out[0];
    return Math.max(s, 0) + Math.log1p(Math.exp(-Math.abs(s))) - y[0] * s;
  }
  let m = -Infinity;
  for (let j = 0; j < out.length; j++) m = Math.max(m, out[j]);
  let se = 0;
  for (let j = 0; j < out.length; j++) se += Math.exp(out[j] - m);
  const lse = m + Math.log(se);
  let s = 0;
  for (let j = 0; j < out.length; j++) s += y[j] * (lse - out[j]);
  return s;
}

/** Mean loss over a data set (no dropout), plus the weight-decay term λ Σ w² if given. */
export function meanLoss(n: Mlp, X: ArrayLike<number>[], Y: ArrayLike<number>[], loss: Loss, decay = 0): number {
  if (!X.length) return NaN;
  let s = 0;
  for (let i = 0; i < X.length; i++) s += sampleLoss(predict(n, X[i]), Y[i], loss);
  return s / X.length + decay * weightNorm2(n);
}

/** Σ w² over all weights (not the biases). */
export function weightNorm2(n: Mlp): number {
  let s = 0;
  for (const W of n.W) for (let k = 0; k < W.length; k++) s += W[k] * W[k];
  return s;
}

export interface GradOptions {
  loss?: Loss;
  /** Weight decay λ: adds λ Σ w² to the loss, 2λw to the gradient (weights only). */
  decay?: number;
  /** Dropout probability for the hidden units (0 = off); needs `rand`. Inverted dropout:
   *  kept units are scaled by 1/(1 − p), so nothing changes at test time. */
  dropout?: number;
  rand?: () => number;
}

export interface Grads { gW: Float64Array[]; gb: Float64Array[]; loss: number }

/** Gradient of the mean loss over the samples `idx` (default all), by backpropagation. */
export function gradients(n: Mlp, X: ArrayLike<number>[], Y: ArrayLike<number>[], o: GradOptions = {}, idx?: ArrayLike<number>): Grads {
  const loss = o.loss ?? "mse", L = n.W.length, p = o.dropout ?? 0;
  const gW = n.W.map(w => new Float64Array(w.length)), gb = n.b.map(v => new Float64Array(v.length));
  const m = idx ? idx.length : X.length;
  let total = 0;
  for (let t = 0; t < m; t++) {
    const s = idx ? idx[t] : t;
    // forward with dropout masks on hidden layers
    const a: Float64Array[] = [Float64Array.from(X[s])], pre: Float64Array[] = [], masks: (Float64Array | null)[] = [];
    for (let l = 0; l < L; l++) {
      const nIn = n.sizes[l], nOut = n.sizes[l + 1], W = n.W[l], inp = a[l];
      const z = new Float64Array(nOut), y = new Float64Array(nOut);
      const mask = p > 0 && l < L - 1 && o.rand ? new Float64Array(nOut) : null;
      for (let j = 0; j < nOut; j++) {
        let v = n.b[l][j];
        for (let i = 0; i < nIn; i++) v += W[j * nIn + i] * inp[i];
        z[j] = v; y[j] = act(n.acts[l], v);
        if (mask) { mask[j] = o.rand!() < p ? 0 : 1 / (1 - p); y[j] *= mask[j]; }
      }
      pre.push(z); a.push(y); masks.push(mask);
    }
    const out = a[L], nOutL = n.sizes[L];
    total += sampleLoss(out, Y[s], loss);
    let d = new Float64Array(nOutL);
    if (loss === "mse") for (let j = 0; j < nOutL; j++) d[j] = 2 * (out[j] - Y[s][j]) / (nOutL * m);
    else { const pr = outputProbs(out, loss); for (let j = 0; j < nOutL; j++) d[j] = (pr[j] - Y[s][j]) / m; }
    for (let l = L - 1; l >= 0; l--) {
      const nIn = n.sizes[l], nOut = n.sizes[l + 1], mask = masks[l];
      // d holds ∂L/∂a[l+1] (after the mask): through the mask, then through φ
      for (let j = 0; j < nOut; j++) {
        if (mask) d[j] *= mask[j];
        if (n.acts[l] !== "id") d[j] *= actDeriv(n.acts[l], pre[l][j], act(n.acts[l], pre[l][j]));
      }
      const dPrev = new Float64Array(nIn);
      for (let j = 0; j < nOut; j++) {
        gb[l][j] += d[j];
        if (d[j] === 0) continue;
        for (let i = 0; i < nIn; i++) { gW[l][j * nIn + i] += d[j] * a[l][i]; dPrev[i] += d[j] * n.W[l][j * nIn + i]; }
      }
      d = dPrev;
    }
  }
  const lam = o.decay ?? 0;
  let lossOut = total / m;
  if (lam > 0) {
    lossOut += lam * weightNorm2(n);
    n.W.forEach((W, l) => { for (let k = 0; k < W.length; k++) gW[l][k] += 2 * lam * W[k]; });
  }
  return { gW, gb, loss: lossOut };
}

/** Plain (stochastic) gradient descent with optional momentum β: v ← βv + g, w ← w − ηv. */
export class Sgd {
  vW: Float64Array[]; vb: Float64Array[];
  constructor(n: Mlp, public lr = 0.1, public momentum = 0) {
    this.vW = n.W.map(w => new Float64Array(w.length)); this.vb = n.b.map(v => new Float64Array(v.length));
  }
}

/** Apply one update with a computed gradient (SGD with momentum, or Adam as in trainStep). */
export function applyGradients(n: Mlp, g: Grads, opt: Sgd | Adam): void {
  if (opt instanceof Sgd) {
    const upd = (p: Float64Array, gr: Float64Array, v: Float64Array) => {
      for (let k = 0; k < p.length; k++) { v[k] = opt.momentum * v[k] + gr[k]; p[k] -= opt.lr * v[k]; }
    };
    n.W.forEach((W, l) => { upd(W, g.gW[l], opt.vW[l]); upd(n.b[l], g.gb[l], opt.vb[l]); });
    return;
  }
  opt.t++;
  const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** opt.t, c2 = 1 - b2 ** opt.t;
  const upd = (p: Float64Array, gr: Float64Array, m: Float64Array, v: Float64Array) => {
    for (let k = 0; k < p.length; k++) {
      m[k] = b1 * m[k] + (1 - b1) * gr[k];
      v[k] = b2 * v[k] + (1 - b2) * gr[k] * gr[k];
      p[k] -= opt.lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
    }
  };
  n.W.forEach((W, l) => { upd(W, g.gW[l], opt.mW[l], opt.vW[l]); upd(n.b[l], g.gb[l], opt.mb[l], opt.vb[l]); });
}

/** A seeded shuffler that hands out mini-batches of indices; an epoch is one pass over all. */
export class Batcher {
  private order: number[] = [];
  private pos = 0;
  epoch = 0;
  private r: ReturnType<typeof rng>;
  constructor(public n: number, public size: number, seed = 1) { this.r = rng(seed); this.reshuffle(); }
  private reshuffle(): void {
    this.order = Array.from({ length: this.n }, (_, i) => i);
    for (let i = this.n - 1; i > 0; i--) { const j = Math.floor(this.r.unif() * (i + 1)); [this.order[i], this.order[j]] = [this.order[j], this.order[i]]; }
    this.pos = 0;
  }
  /** The next batch (size min(size, n)); starts a new shuffled epoch when the data run out. */
  next(): number[] {
    const b = Math.max(1, Math.min(this.size, this.n));
    if (this.pos + b > this.n) { this.reshuffle(); this.epoch++; }
    const out = this.order.slice(this.pos, this.pos + b);
    this.pos += b;
    return out;
  }
  /** Random numbers from the same stream (for dropout masks). */
  unif(): number { return this.r.unif(); }
}

/** One training step on the next mini-batch; returns the batch loss before the step. */
export function trainBatch(n: Mlp, X: ArrayLike<number>[], Y: ArrayLike<number>[], opt: Sgd | Adam, batcher: Batcher, o: GradOptions = {}): number {
  const idx = batcher.next();
  const g = gradients(n, X, Y, { ...o, rand: () => batcher.unif() }, idx);
  applyGradients(n, g, opt);
  return g.loss;
}
