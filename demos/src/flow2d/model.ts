// Model of the flow2d demo (no DOM; tested in demos/test/gans-test.ts). A normalising flow in
// 2D: K affine coupling layers map data x to z, which should look like N(0, I).
//
// Layer k changes one coordinate (d = k mod 2, alternating) and keeps the other (o):
//   forward (data → noise):  y_d = x_d · exp(s(x_o)) + t(x_o),   y_o = x_o,
//   inverse (noise → data):  x_d = (y_d − t(y_o)) · exp(−s(y_o)),  x_o = y_o,
//   log |det ∂y/∂x| = s(x_o)  (the Jacobian is triangular).
// s and t come from one small network per layer, 1 → H → H → 2 (tanh), with
// s = S_MAX · tanh(raw / S_MAX) to keep the scale factors in [e^(−S_MAX), e^(S_MAX)].
// Change of variables: log p(x) = log N(z; 0, I) + Σ_k s_k. Training minimises the mean
// negative log-likelihood −log p(x) over batches of data, with Adam.
import { rng } from "../lib/ui";
import { net, Batch, zeroGrad, clearGrad, Adam, run, type Net, type Grad } from "../gan1d/net";

export const S_MAX = 2;
export type DataId = "moons" | "ring" | "banana";

/** n points of a data set (flat x, y pairs), roughly within [−2.5, 2.5]². */
export function sampleData(id: DataId, n: number, r: ReturnType<typeof rng>): Float64Array {
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    let x: number, y: number;
    if (id === "moons") {
      const c = r.unif() < 0.5 ? 0 : 1, t = Math.PI * r.unif();
      // the classic two moons, centred and scaled to about [−2, 2] × [−1, 1]
      x = c === 0 ? Math.cos(t) : 1 - Math.cos(t); y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
      x = 1.6 * (x - 0.5) + 0.1 * r.norm(); y = 1.6 * (y - 0.25) + 0.1 * r.norm();
    } else if (id === "ring") {
      const t = 2 * Math.PI * r.unif(), rad = 1.6 + 0.12 * r.norm();
      x = rad * Math.cos(t); y = rad * Math.sin(t);
    } else {
      const u = r.norm();
      x = 1.2 * u; y = 0.6 * u * u - 1.2 + 0.3 * r.norm();
    }
    out[2 * i] = x; out[2 * i + 1] = y;
  }
  return out;
}

export class Flow {
  K: number;
  nets: Net[];
  private batches: Batch[];
  private grads: Grad[];
  private opts: Adam[];
  step = 0;
  history: { step: number; nll: number }[] = [];
  private r: ReturnType<typeof rng>;

  constructor(K: number, H: number, seed = 1, public lr = 3e-3) {
    this.K = K;
    // small final weights: every layer starts close to the identity
    this.nets = Array.from({ length: K }, (_, k) => {
      const nt = net([1, H, H, 2], ["tanh", "tanh", "id"], seed + 31 * k);
      nt.W[2].forEach((_, i) => { nt.W[2][i] *= 0.1; });
      return nt;
    });
    this.batches = this.nets.map(nt => new Batch(nt, 256));
    this.grads = this.nets.map(nt => zeroGrad(nt));
    this.opts = this.nets.map(nt => new Adam(nt, lr, 0.9));
    this.r = rng(seed + 1000);
  }

  /** s and t of layer k given the kept coordinate. */
  st(k: number, keep: number): [number, number] {
    const o = run(this.nets[k], [keep]);
    return [S_MAX * Math.tanh(o[0] / S_MAX), o[1]];
  }

  /** Data → noise through the first `upTo` layers (default all); returns z and Σ s. */
  forward(x: number, y: number, upTo = this.K): { z: [number, number]; logdet: number } {
    const v = [x, y];
    let ld = 0;
    for (let k = 0; k < upTo; k++) {
      const d = k % 2, o = 1 - d, [s, t] = this.st(k, v[o]);
      v[d] = v[d] * Math.exp(s) + t;
      ld += s;
    }
    return { z: [v[0], v[1]], logdet: ld };
  }

  /** Noise → data: the inverse of the layers from `from` down to 0 (default all). */
  inverse(zx: number, zy: number, from = this.K): [number, number] {
    const v = [zx, zy];
    for (let k = from - 1; k >= 0; k--) {
      const d = k % 2, o = 1 - d, [s, t] = this.st(k, v[o]);
      v[d] = (v[d] - t) * Math.exp(-s);
    }
    return [v[0], v[1]];
  }

  /** log p(x) by the change of variables. */
  logDensity(x: number, y: number): number {
    const { z, logdet } = this.forward(x, y);
    return -0.5 * (z[0] * z[0] + z[1] * z[1]) - Math.log(2 * Math.PI) + logdet;
  }

  /** One Adam step on the mean negative log-likelihood of a batch; returns it (before the step). */
  train(data: DataId, n = 128): number {
    const X = sampleData(data, n, this.r), K = this.K;
    // forward, keeping every layer's input and its (raw s, t)
    const vs: Float64Array[] = [X.slice()];
    const raws: Float64Array[] = [];
    for (let k = 0; k < K; k++) {
      const d = k % 2, o = 1 - d, cur = vs[k], keep = new Float64Array(n);
      for (let i = 0; i < n; i++) keep[i] = cur[2 * i + o];
      const out = Float64Array.from(this.batches[k].forward(keep, n));
      raws.push(out);
      const next = cur.slice();
      for (let i = 0; i < n; i++) {
        const s = S_MAX * Math.tanh(out[2 * i] / S_MAX), t = out[2 * i + 1];
        next[2 * i + d] = cur[2 * i + d] * Math.exp(s) + t;
      }
      vs.push(next);
    }
    // loss = mean(½|z|² + log 2π − Σ s); gradient starts at z / n
    let nll = 0;
    const zf = vs[K];
    const g = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      nll += 0.5 * (zf[2 * i] ** 2 + zf[2 * i + 1] ** 2) + Math.log(2 * Math.PI);
      g[2 * i] = zf[2 * i] / n; g[2 * i + 1] = zf[2 * i + 1] / n;
    }
    for (let k = K - 1; k >= 0; k--) {
      const d = k % 2, o = 1 - d, cur = vs[k], out = raws[k], dOut = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        const th = Math.tanh(out[2 * i] / S_MAX), s = S_MAX * th, es = Math.exp(s), gy = g[2 * i + d];
        nll -= s;
        // y_d = x_d e^s + t:  ∂/∂x_d = e^s,  ∂/∂s = x_d e^s (and −1 from −log det),  ∂/∂t = 1
        const dS = gy * cur[2 * i + d] * es - 1 / n;
        dOut[2 * i] = dS * (1 - th * th);     // through s = S_MAX tanh(raw / S_MAX)
        dOut[2 * i + 1] = gy;
        g[2 * i + d] = gy * es;
      }
      clearGrad(this.grads[k]);
      const dKeep = this.batches[k].backward(dOut, this.grads[k]);
      for (let i = 0; i < n; i++) g[2 * i + o] += dKeep[i];
      this.opts[k].lr = this.lr;
      this.opts[k].step(this.nets[k], this.grads[k]);
    }
    nll /= n;
    this.step++;
    this.history.push({ step: this.step, nll });
    return nll;
  }
}

/** Weights of all layers as one plain object (4 significant digits), and back. */
export function exportWeights(f: Flow): number[][][] {
  const r = (v: number) => +v.toPrecision(4);
  return f.nets.map(nt => [...nt.W.map(w => Array.from(w, r)), ...nt.b.map(b => Array.from(b, r))]);
}
export function importWeights(f: Flow, w: number[][][]): void {
  f.nets.forEach((nt, k) => {
    const L = nt.W.length;
    for (let l = 0; l < L; l++) { nt.W[l].set(w[k][l]); nt.b[l].set(w[k][L + l]); }
  });
}

/** The demo's flow: 6 layers, 24 hidden units, seed 1, learning rate 0.003, batches of 128 moons. */
export const K_LAYERS = 6, HIDDEN = 24;
