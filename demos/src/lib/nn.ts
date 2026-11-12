// A tiny multilayer perceptron for the teaching demos: dense layers, ReLU or identity,
// mean-squared-error loss, backpropagation and Adam. Flat Float64Arrays, no DOM, so it
// runs in node for tests. Small on purpose: networks of a few hundred weights, trained live.
//
// Layer l maps a[l] (size sizes[l]) to a[l+1] = act_l(W_l a[l] + b_l), W_l row-major
// (W[o * nIn + i]). Used by nnfit (1 → H → 1) and linae (2 → H → 1 → H → 2).
import { rng } from "./ui";

export type Act = "relu" | "id";

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
      y[o] = n.acts[l] === "relu" ? Math.max(0, s) : s;
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
      if (n.acts[l] === "relu") for (let o = 0; o < nOut; o++) if (pre[l][o] <= 0) d[o] = 0;
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
