// Small dense networks for the demos of the day "GANs and normalising flows" (gan1d, ganring,
// translate1d, flow2d). No DOM, so the tests run them in node.
//
// Unlike lib/nn.ts (which trains on a fixed loss), a GAN needs the gradient of a loss with
// respect to the network's *input* (the discriminator's gradient flows back into the
// generator) and losses chosen outside the network. So: a forward pass over a batch that keeps
// its values, a backward pass from any output gradient that returns the input gradient and adds
// the parameter gradients into a buffer, and Adam with a choosable β1 (GANs use 0.5).
// Batches are flat row-major arrays (n × width); buffers are reused, so live training is fast.
import { rng } from "../lib/ui";

export type Act = "relu" | "tanh" | "lrelu" | "id";

export interface Net {
  sizes: number[];
  acts: Act[];
  /** W[l][o * nIn + i], row-major. */
  W: Float64Array[];
  b: Float64Array[];
}

const ACT_CODE: Record<Act, number> = { id: 0, relu: 1, lrelu: 2, tanh: 3 };
const actF = (a: Act, s: number) => a === "relu" ? (s > 0 ? s : 0) : a === "lrelu" ? (s > 0 ? s : 0.2 * s) : a === "tanh" ? Math.tanh(s) : s;

/** A network with layer sizes `sizes`, an activation per layer (the last usually "id"),
 *  weights ~ N(0, gain²·k/fan-in) (k = 2 for (leaky) ReLU) from a fixed seed, biases 0. */
export function net(sizes: number[], acts: Act[], seed: number, gain = 1): Net {
  const r = rng(seed), W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 0; l + 1 < sizes.length; l++) {
    const nIn = sizes[l], nOut = sizes[l + 1], s = gain * Math.sqrt((acts[l] === "relu" || acts[l] === "lrelu" ? 2 : 1) / nIn);
    W.push(Float64Array.from({ length: nIn * nOut }, () => s * r.norm()));
    b.push(new Float64Array(nOut));
  }
  return { sizes, acts, W, b };
}

export function cloneNet(n: Net): Net {
  return { sizes: [...n.sizes], acts: [...n.acts], W: n.W.map(w => w.slice()), b: n.b.map(v => v.slice()) };
}

/** The output for one input (no buffers kept). */
export function run(n: Net, x: ArrayLike<number>): Float64Array {
  let a = Float64Array.from(x);
  for (let l = 0; l < n.W.length; l++) {
    const nIn = n.sizes[l], nOut = n.sizes[l + 1], W = n.W[l], y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = n.b[l][o];
      for (let i = 0; i < nIn; i++) s += W[o * nIn + i] * a[i];
      y[o] = actF(n.acts[l], s);
    }
    a = y;
  }
  return a;
}

/** Parameter gradients (same shapes as the weights). */
export interface Grad { gW: Float64Array[]; gb: Float64Array[] }
export function zeroGrad(n: Net): Grad {
  return { gW: n.W.map(w => new Float64Array(w.length)), gb: n.b.map(v => new Float64Array(v.length)) };
}
export function clearGrad(g: Grad): void { g.gW.forEach(a => a.fill(0)); g.gb.forEach(a => a.fill(0)); }

/** Multiply every gradient by s (e.g. 1/batch). */
export function scaleGrad(g: Grad, s: number): void {
  for (const a of g.gW) for (let k = 0; k < a.length; k++) a[k] *= s;
  for (const a of g.gb) for (let k = 0; k < a.length; k++) a[k] *= s;
}

/** Forward and backward passes over batches, with reused buffers. */
export class Batch {
  /** a[l]: values of layer l (n × sizes[l]); a[0] is the input, a[L] the output. */
  a: Float64Array[] = [];
  private pre: Float64Array[] = [];
  private d: Float64Array[] = [];
  n = 0;
  constructor(public net: Net, private cap = 0) { this.alloc(Math.max(1, cap)); }
  private alloc(n: number): void {
    this.cap = n;
    const s = this.net.sizes;
    this.a = s.map(k => new Float64Array(n * k));
    this.pre = s.slice(1).map(k => new Float64Array(n * k));
    this.d = s.map(k => new Float64Array(n * k));
  }
  /** Forward pass of n inputs X (n × sizes[0]); returns the output buffer (n × sizes[L]). */
  forward(X: ArrayLike<number>, n: number): Float64Array {
    if (n > this.cap) this.alloc(n);
    this.n = n;
    const N = this.net, s = N.sizes, a0 = this.a[0];
    for (let k = 0; k < n * s[0]; k++) a0[k] = X[k];
    for (let l = 0; l < N.W.length; l++) {
      const nIn = s[l], nOut = s[l + 1], W = N.W[l], bb = N.b[l], code = ACT_CODE[N.acts[l]];
      const inp = this.a[l], z = this.pre[l], y = this.a[l + 1];
      for (let p = 0; p < n; p++) {
        const ip = p * nIn, op = p * nOut;
        for (let o = 0; o < nOut; o++) {
          let v = bb[o];
          const row = o * nIn;
          for (let i = 0; i < nIn; i++) v += W[row + i] * inp[ip + i];
          z[op + o] = v;
          y[op + o] = code === 0 ? v : code === 1 ? (v > 0 ? v : 0) : code === 2 ? (v > 0 ? v : 0.2 * v) : Math.tanh(v);
        }
      }
    }
    return this.a[s.length - 1];
  }
  /** Backward pass from dOut (n × sizes[L]) = ∂loss/∂output for the last forward batch: adds
   *  the parameter gradients into g (unless null); returns ∂loss/∂input (n × sizes[0]). */
  backward(dOut: ArrayLike<number>, g: Grad | null): Float64Array {
    const N = this.net, s = N.sizes, n = this.n, L = N.W.length;
    const dL = this.d[L];
    for (let k = 0; k < n * s[L]; k++) dL[k] = dOut[k];
    for (let l = L - 1; l >= 0; l--) {
      const nIn = s[l], nOut = s[l + 1], W = N.W[l], code = ACT_CODE[N.acts[l]];
      const d = this.d[l + 1], dPrev = this.d[l], inp = this.a[l], z = this.pre[l], y = this.a[l + 1];
      dPrev.fill(0, 0, n * nIn);
      const gw = g ? g.gW[l] : null, gb = g ? g.gb[l] : null;
      for (let p = 0; p < n; p++) {
        const ip = p * nIn, op = p * nOut;
        for (let o = 0; o < nOut; o++) {
          const v = z[op + o];
          const dd = d[op + o] * (code === 0 ? 1 : code === 1 ? (v > 0 ? 1 : 0) : code === 2 ? (v > 0 ? 1 : 0.2) : 1 - y[op + o] * y[op + o]);
          if (dd === 0) continue;
          const row = o * nIn;
          if (gw !== null) {
            gb![o] += dd;
            for (let i = 0; i < nIn; i++) { gw[row + i] += dd * inp[ip + i]; dPrev[ip + i] += dd * W[row + i]; }
          } else {
            for (let i = 0; i < nIn; i++) dPrev[ip + i] += dd * W[row + i];
          }
        }
      }
    }
    return this.d[0];
  }
}

/** Adam (Kingma and Ba) with a choosable β1; GANs commonly use β1 = 0.5. */
export class Adam {
  t = 0;
  private m: Grad; private v: Grad;
  constructor(n: Net, public lr: number, public beta1 = 0.5, public beta2 = 0.999) {
    this.m = zeroGrad(n); this.v = zeroGrad(n);
  }
  /** One step against the gradient `g` (already averaged over the batch). */
  step(n: Net, g: Grad): void {
    this.t++;
    const b1 = this.beta1, b2 = this.beta2, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t, lr = this.lr;
    const upd = (p: Float64Array, gr: Float64Array, m: Float64Array, v: Float64Array) => {
      for (let k = 0; k < p.length; k++) {
        m[k] = b1 * m[k] + (1 - b1) * gr[k];
        v[k] = b2 * v[k] + (1 - b2) * gr[k] * gr[k];
        p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8);
      }
    };
    for (let l = 0; l < n.W.length; l++) { upd(n.W[l], g.gW[l], this.m.gW[l], this.v.gW[l]); upd(n.b[l], g.gb[l], this.m.gb[l], this.v.gb[l]); }
  }
}

/** σ(s) = 1/(1 + e^(−s)) and the stable log σ(s), log(1 − σ(s)). */
export const sigmoid = (s: number) => 1 / (1 + Math.exp(-s));
export const logSigmoid = (s: number) => -(Math.max(-s, 0) + Math.log1p(Math.exp(-Math.abs(s))));
export const log1mSigmoid = (s: number) => logSigmoid(-s);
