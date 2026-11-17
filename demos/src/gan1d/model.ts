// Model of the gan1d demo (no DOM; tested in demos/test/gans-test.ts). A GAN in one dimension.
//
// The smallest GAN, as in the by-hand exercises and the coding exercise (gans/src/gan/): the
// generator is a line x = a·z + b of the noise z ~ N(0, 1), so its samples are N(b, a²); the
// discriminator is a logistic curve of a quadratic, D(x) = σ(c0 + c1·x + c2·x²), which can
// represent the best discriminator between two Gaussians exactly (the log of a ratio of two
// Gaussians is quadratic). Both are trained by plain gradient descent with hand-written
// gradients. Data: one bump N(1, 0.5²), or two bumps (which a line generator cannot make: the
// unit gan-training uses it for mode dropping).
import { rng } from "../lib/ui";
import { sigmoid, logSigmoid, log1mSigmoid } from "./net";

export type Loss = "ns" | "mm";
export type DataId = "one" | "two";

/** The data distributions: a mixture of Gaussians (weights, means, sds). */
export const DATA: Record<DataId, { w: number[]; mu: number[]; sd: number[] }> = {
  one: { w: [1], mu: [1], sd: [0.5] },
  two: { w: [0.5, 0.5], mu: [0, 2], sd: [0.3, 0.3] },
};

export function dataPdf(id: DataId, x: number): number {
  const d = DATA[id];
  let p = 0;
  for (let k = 0; k < d.w.length; k++) p += d.w[k] * Math.exp(-0.5 * ((x - d.mu[k]) / d.sd[k]) ** 2) / (d.sd[k] * Math.sqrt(2 * Math.PI));
  return p;
}

export function sampleData(id: DataId, r: ReturnType<typeof rng>): number {
  const d = DATA[id];
  let u = r.unif(), k = 0;
  while (k < d.w.length - 1 && u > d.w[k]) { u -= d.w[k]; k++; }
  return d.mu[k] + d.sd[k] * r.norm();
}

export interface Options {
  lrG: number;
  lrD: number;
  /** Discriminator steps per generator step. */
  kD: number;
  loss: Loss;
  batch: number;
}

export interface History { step: number; lossD: number; lossG: number; a: number; b: number }

/** Mean over a batch of the discriminator's loss −[log D(real) + log(1 − D(fake))] given the logits. */
export function lossD(sReal: ArrayLike<number>, sFake: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < sReal.length; i++) s -= logSigmoid(sReal[i]);
  for (let i = 0; i < sFake.length; i++) s -= log1mSigmoid(sFake[i]);
  return s / sReal.length;
}

/** The generator's loss given the logits of its samples: "ns" −log D(G(z)), "mm" log(1 − D(G(z))). */
export function lossG(sFake: ArrayLike<number>, loss: Loss): number {
  let s = 0;
  for (let i = 0; i < sFake.length; i++) s += loss === "ns" ? -logSigmoid(sFake[i]) : log1mSigmoid(sFake[i]);
  return s / sFake.length;
}

/** ∂(generator loss)/∂(logit) for one sample: "ns" −(1 − D), "mm" −D. */
export const dLossGds = (s: number, loss: Loss) => loss === "ns" ? -(1 - sigmoid(s)) : -sigmoid(s);

/** The best discriminator D*(x) = p_data / (p_data + p_g). */
export const optimalD = (pData: number, pG: number) => pData + pG > 0 ? pData / (pData + pG) : 0.5;

export class Gan1d {
  data: DataId;
  o: Options;
  /** Replace the learned discriminator by the best one, D* = p/(p + q) (exact here). */
  optimal = false;
  step = 0;
  history: History[] = [];
  /** Generator x = a z + b; discriminator logit c0 + c1 x + c2 x². */
  a: number; b: number;
  c = [0, 0, 0];
  private r: ReturnType<typeof rng>;

  constructor(data: DataId, o: Options, seed = 1, start: [number, number] = [1, -1]) {
    this.data = data; this.o = { ...o };
    [this.a, this.b] = start;
    this.r = rng(seed);
  }

  /** The discriminator's logit at x. */
  logit(x: number): number {
    if (this.optimal) {
      // log p_data(x) − log p_g(x), with p_g = N(b, a²)
      const d = DATA[this.data];
      let p = 0;
      for (let k = 0; k < d.w.length; k++) p += d.w[k] * Math.exp(-0.5 * ((x - d.mu[k]) / d.sd[k]) ** 2) / d.sd[k];
      return Math.log(p) + 0.5 * ((x - this.b) / this.a) ** 2 + Math.log(Math.abs(this.a));
    }
    return this.c[0] + this.c[1] * x + this.c[2] * x * x;
  }
  /** ∂logit/∂x: the slope that pushes the generator's samples. */
  slope(x: number): number {
    if (!this.optimal) return this.c[1] + 2 * this.c[2] * x;
    const h = 1e-5;
    return (this.logit(x + h) - this.logit(x - h)) / (2 * h);
  }
  disc(x: number): number { return sigmoid(this.logit(x)); }
  gen(z: number): number { return this.a * z + this.b; }

  /** One training step: kD discriminator steps, then one generator step (plain gradient descent). */
  train(): void {
    const o = this.o, n = o.batch, r = this.r;
    let ld = NaN;
    for (let k = 0; k < (this.optimal ? 0 : o.kD); k++) {
      const xr = Float64Array.from({ length: n }, () => sampleData(this.data, r));
      const xf = Float64Array.from({ length: n }, () => this.a * r.norm() + this.b);
      // ∂L/∂s = σ(s) − 1 for real, σ(s) for fake; ∂s/∂c = (1, x, x²)
      const g = [0, 0, 0];
      const sr = xr.map(x => this.logit(x)), sf = xf.map(x => this.logit(x));
      ld = lossD(sr, sf);
      xr.forEach((x, i) => { const d = sigmoid(sr[i]) - 1; g[0] += d; g[1] += d * x; g[2] += d * x * x; });
      xf.forEach((x, i) => { const d = sigmoid(sf[i]); g[0] += d; g[1] += d * x; g[2] += d * x * x; });
      for (let j = 0; j < 3; j++) this.c[j] -= o.lrD * g[j] / n;
    }
    // generator step: chain rule through the (fixed) discriminator, x = a z + b
    let lg = 0, ga = 0, gb = 0;
    for (let i = 0; i < n; i++) {
      const z = r.norm(), x = this.a * z + this.b, s = this.logit(x);
      lg += o.loss === "ns" ? -logSigmoid(s) : log1mSigmoid(s);
      const dx = dLossGds(s, o.loss) * this.slope(x);
      ga += dx * z; gb += dx;
    }
    this.a -= o.lrG * ga / n; this.b -= o.lrG * gb / n;
    this.step++;
    this.history.push({ step: this.step, lossD: ld, lossG: lg / n, a: this.a, b: this.b });
  }

  /** Density of the generator's samples, N(b, a²). */
  genPdf(x: number): number { const a = Math.abs(this.a) + 1e-9; return Math.exp(-0.5 * ((x - this.b) / a) ** 2) / (a * Math.sqrt(2 * Math.PI)); }
}

/** 1D precision and recall of a Gaussian generator against the data, by probability mass:
 *  precision = share of generated mass where the data density is at least `t` times its peak;
 *  recall = share of data mass where the generator's density is at least `t` times its peak.
 *  (A smooth 1D stand-in for the k-nearest-neighbour version used for images.) */
export function precisionRecall1d(g: Gan1d, t = 0.05, lo = -5, hi = 6, n = 2000): { precision: number; recall: number } {
  const h = (hi - lo) / n, xs = Array.from({ length: n }, (_, i) => lo + (i + 0.5) * h);
  const pd = xs.map(x => dataPdf(g.data, x)), pg = xs.map(x => g.genPdf(x));
  const md = Math.max(...pd), mg = Math.max(...pg);
  let P = 0, R = 0, sg = 0, sd = 0;
  for (let i = 0; i < n; i++) { sg += pg[i]; sd += pd[i]; if (pd[i] >= t * md) P += pg[i]; if (pg[i] >= t * mg) R += pd[i]; }
  return { precision: P / sg, recall: R / sd };
}

/** Histogram density of samples on [lo, hi) with `bins` bins (integrates to the share inside). */
export function densityHist(xs: ArrayLike<number>, lo: number, hi: number, bins: number): Float64Array {
  const h = new Float64Array(bins), w = (hi - lo) / bins;
  for (let i = 0; i < xs.length; i++) { const k = Math.floor((xs[i] - lo) / w); if (k >= 0 && k < bins) h[k]++; }
  for (let k = 0; k < bins; k++) h[k] /= xs.length * w;
  return h;
}
