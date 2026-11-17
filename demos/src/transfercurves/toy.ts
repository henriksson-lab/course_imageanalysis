// The toy world of the day "Fine-tuning and transfer learning" (no DOM, so node tests and the
// figure scripts use exactly this code). Shared by the demos transfercurves, lowrank and
// domainshift.
//
// Images are 10 × 10 patches with one object each, grey values about 0..1.
//  - Source task ("pretraining", lots of labels): four generic shapes, bright on dark:
//    disc, ring, bar, two dots.
//  - Target task (your project, few labels): three cell phenotypes: interphase (a large dim
//    nucleus with dark nucleoli), mitotic (a bright compact plate), apoptotic (bright fragments).
//  - A Look describes the microscope: gain, background, blur, noise, inverted contrast.
// The network: 100 → 32 → 16 → K, ReLU, softmax + cross-entropy (lib/nn.ts), trained with
// Adam whose learning rate can differ per layer (0 = the layer is frozen).
import { mlp, cloneMlp, softmax, type Mlp } from "../lib/nn";
import { rng } from "../lib/ui";

export const P = 10;
export const NPIX = P * P;
export type Rand = ReturnType<typeof rng>;

export const SOURCE_CLASSES = ["disc", "ring", "bar", "two dots"];
export const TARGET_CLASSES = ["interphase", "mitotic", "apoptotic"];

/** How the microscope renders an object: pixel = 0.08 + offset + gain · object, blurred, plus noise. */
export interface Look {
  gain: number;       // brightness of the object (1 = as in training; negative = dark on bright)
  offset: number;     // added background
  blur: number;       // extra Gaussian blur σ in pixels (0 = none)
  noise: number;      // pixel noise standard deviation
}
/** The training microscope (fluorescence: bright objects on a dark background). */
export const LOOK_A: Look = { gain: 1, offset: 0, blur: 0, noise: 0.06 };
/** Another modality: dark, blurred objects on a bright background (bright field). */
export const LOOK_FAR: Look = { gain: -0.6, offset: 0.82, blur: 0.9, noise: 0.06 };

const sig = (t: number) => 1 / (1 + Math.exp(-t));

/** Draw one object of the given kind into a P × P float image (before the look's blur/noise). */
function drawObject(kind: string, r: Rand): Float64Array {
  const img = new Float64Array(NPIX);
  const cx = 4.5 + (r.unif() - 0.5) * 2.0, cy = 4.5 + (r.unif() - 0.5) * 2.0;
  const th = r.unif() * Math.PI, c = Math.cos(th), s = Math.sin(th);
  const A = 0.65 + 0.35 * r.unif();
  // a few random spots (dots, fragments, nucleoli)
  const spots = (n: number, rad: number, sd: number) => Array.from({ length: n }, () => {
    const a = r.unif() * 2 * Math.PI, d = rad * Math.sqrt(r.unif());
    return [cx + d * Math.cos(a), cy + d * Math.sin(a), sd] as [number, number, number];
  });
  let dots: [number, number, number][] = [];
  if (kind === "two dots") {
    const h = 1.4 + 0.6 * r.unif();
    dots = [[cx + h * c, cy + h * s, 0.8], [cx - h * c, cy - h * s, 0.8]];
  } else if (kind === "apoptotic") {
    // 4 to 6 fragments spread around the dying cell's centre
    const n = 4 + Math.floor(r.unif() * 3), a0 = r.unif() * 2 * Math.PI;
    dots = Array.from({ length: n }, (_, i) => {
      const a = a0 + (2 * Math.PI * (i + 0.3 * (r.unif() - 0.5))) / n, d = 1.6 + 1.4 * r.unif();
      return [cx + d * Math.cos(a), cy + d * Math.sin(a), 0.5] as [number, number, number];
    });
  }
  const nucleoli = kind === "interphase" ? spots(1 + Math.floor(r.unif() * 2), 1.4, 0.6) : [];
  const rad = kind === "disc" ? 2.0 + 1.0 * r.unif() : kind === "ring" ? 2.4 + 0.9 * r.unif() : 2.8 + 0.8 * r.unif();
  const len = kind === "bar" ? 3.0 + 0.8 * r.unif() : 2.2 + 0.7 * r.unif();
  const wid = kind === "bar" ? 0.9 + 0.3 * r.unif() : 0.65 + 0.2 * r.unif();
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
    const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy);
    const u = dx * c + dy * s, w = -dx * s + dy * c;
    let v = 0;
    switch (kind) {
      case "disc": v = A * sig((rad - d) / 0.45); break;
      case "ring": v = A * Math.exp(-((d - rad) ** 2) / (2 * 0.55 ** 2)); break;
      case "bar": case "mitotic": {
        const e = Math.sqrt((u / len) ** 2 + (w / wid) ** 2);
        v = (kind === "mitotic" ? 1.15 : 1) * A * sig((1 - e) / 0.18);
        break;
      }
      case "interphase": {
        v = 0.55 * A * sig((rad - d) / 0.5);
        for (const [px, py, sd] of nucleoli) v *= 1 - 0.75 * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sd * sd));
        break;
      }
    }
    for (const [px, py, sd] of dots) v += A * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sd * sd));
    img[y * P + x] = v;
  }
  return img;
}

/** Separable Gaussian blur of a P × P patch, edges clamped. */
export function blurPatch(img: Float64Array, sigma: number): Float64Array {
  if (sigma <= 0) return img.slice();
  const rad = Math.ceil(3 * sigma), k: number[] = [];
  let sum = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const cl = (i: number) => Math.max(0, Math.min(P - 1, i));
  const tmp = new Float64Array(NPIX), out = new Float64Array(NPIX);
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
    let s = 0; for (let i = -rad; i <= rad; i++) s += k[i + rad] * img[y * P + cl(x + i)]; tmp[y * P + x] = s;
  }
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
    let s = 0; for (let i = -rad; i <= rad; i++) s += k[i + rad] * tmp[cl(y + i) * P + x]; out[y * P + x] = s;
  }
  return out;
}

/** One patch: an object of `kind` seen through the microscope `look`. */
export function renderPatch(kind: string, r: Rand, look: Look = LOOK_A): Float64Array {
  let img = drawObject(kind, r);
  if (look.blur > 0) img = blurPatch(img, look.blur);
  for (let i = 0; i < NPIX; i++) img[i] = 0.08 + look.offset + look.gain * img[i] + look.noise * r.norm();
  return img;
}

export interface DataSet { X: Float64Array[]; y: number[]; k: number }

/** n patches of a task, classes in turn (balanced), with a fixed seed. The same seed with
 *  another look gives the same objects (and the same noise pattern) through that microscope. */
export function makeSet(task: "source" | "target", n: number, seed: number, look: Look = LOOK_A): DataSet {
  const classes = task === "source" ? SOURCE_CLASSES : TARGET_CLASSES;
  const r = rng(seed);
  const X: Float64Array[] = [], y: number[] = [];
  for (let i = 0; i < n; i++) { const c = i % classes.length; X.push(renderPatch(classes[c], r, look)); y.push(c); }
  return { X, y, k: classes.length };
}

// ---------------------------------------------------------------------------------------
// The network and its training. A small dense ReLU network (the Mlp type of lib/nn.ts) with
// its own allocation-free forward and backward passes, so that four networks train live.
// Frozen layers are not backpropagated through at all: their outputs are computed once.

export const SIZES = [NPIX, 32, 16];

/** A fresh network 100 → 32 → 16 → k (He initialisation, fixed seed). */
export function newNet(k: number, seed: number): Mlp {
  return mlp([...SIZES, k], ["relu", "relu", "id"], seed);
}

/** Output of layer l (after its ReLU; the last layer gives the scores). */
export function layerOut(net: Mlp, l: number, inp: ArrayLike<number>, out: Float64Array = new Float64Array(net.sizes[l + 1])): Float64Array {
  const nIn = net.sizes[l], nOut = net.sizes[l + 1], W = net.W[l], b = net.b[l], last = l === net.W.length - 1;
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * inp[i];
    out[o] = last || s > 0 ? s : 0;
  }
  return out;
}

/** The input of layer `from` (the features after layer from − 1) for every patch. */
export function featuresAt(net: Mlp, from: number, X: ArrayLike<number>[]): Float64Array[] {
  return X.map(x => {
    let a: Float64Array = Float64Array.from(x);
    for (let l = 0; l < from; l++) a = layerOut(net, l, a);
    return a;
  });
}

/** Class scores of a network for an input given at layer `from` (0 = the patch). */
export function scoresFrom(net: Mlp, from: number, a0: ArrayLike<number>): Float64Array {
  let a: Float64Array = Float64Array.from(a0);
  for (let l = from; l < net.W.length; l++) a = layerOut(net, l, a);
  return a;
}

/** Accuracy and mean cross-entropy, inputs given at layer `from`. */
export function evaluateFrom(net: Mlp, from: number, A: ArrayLike<number>[], y: number[]): { acc: number; loss: number } {
  let ok = 0, loss = 0;
  for (let i = 0; i < A.length; i++) {
    const p = softmax(scoresFrom(net, from, A[i]));
    let best = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[best]) best = j;
    if (best === y[i]) ok++;
    loss -= Math.log(Math.max(1e-12, p[y[i]]));
  }
  return { acc: ok / Math.max(1, A.length), loss: loss / Math.max(1, A.length) };
}

/** Accuracy and mean cross-entropy of a network on a data set. */
export function evaluate(net: Mlp, d: DataSet): { acc: number; loss: number } {
  return evaluateFrom(net, 0, d.X, d.y);
}

/**
 * Adam with one learning rate per layer. Layers below `from` (the first layer with a rate > 0)
 * are frozen and not backpropagated through; frozen layers above it (rate 0) still pass the
 * gradient on, but do not move.
 */
export class LayerAdam {
  t = 0;
  from: number;
  private m: Float64Array[]; private v: Float64Array[];
  private gW: Float64Array[]; private gb: Float64Array[];
  private act: Float64Array[]; private del: Float64Array[];
  constructor(public net: Mlp, public rates: number[]) {
    const L = net.W.length;
    this.from = rates.findIndex(r => r > 0);
    if (this.from < 0) this.from = L;
    this.m = net.W.map((w, l) => new Float64Array(w.length + net.b[l].length));
    this.v = net.W.map((w, l) => new Float64Array(w.length + net.b[l].length));
    this.gW = net.W.map(w => new Float64Array(w.length));
    this.gb = net.b.map(b => new Float64Array(b.length));
    this.act = net.sizes.map(n => new Float64Array(n));
    this.del = net.sizes.map(n => new Float64Array(n));
  }
  /**
   * One step on the mean cross-entropy (+ λ Σ w² of the trained layers if decay > 0) over the
   * samples `idx` (default all) of A, the inputs of layer `from`; returns the loss before the step.
   */
  step(A: ArrayLike<number>[], y: number[], idx?: number[], decay = 0): number {
    const net = this.net, L = net.W.length, f = this.from;
    if (f >= L) return NaN;
    for (let l = f; l < L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
    const m = idx ? idx.length : A.length;
    let loss = 0;
    for (let t = 0; t < m; t++) {
      const s = idx ? idx[t] : t;
      const a0 = A[s], af = this.act[f];
      for (let i = 0; i < af.length; i++) af[i] = a0[i];
      for (let l = f; l < L; l++) layerOut(net, l, this.act[l], this.act[l + 1]);
      // softmax + cross-entropy: δ = p − y
      const out = this.act[L], d = this.del[L], k = out.length;
      let mx = -Infinity; for (let j = 0; j < k; j++) mx = Math.max(mx, out[j]);
      let se = 0; for (let j = 0; j < k; j++) { d[j] = Math.exp(out[j] - mx); se += d[j]; }
      for (let j = 0; j < k; j++) d[j] /= se;
      loss -= Math.log(Math.max(1e-300, d[y[s]]));
      d[y[s]] -= 1;
      for (let j = 0; j < k; j++) d[j] /= m;
      for (let l = L - 1; l >= f; l--) {
        const nIn = net.sizes[l], nOut = net.sizes[l + 1], W = net.W[l], dl = this.del[l + 1], al = this.act[l];
        const gW = this.gW[l], gb = this.gb[l], dp = this.del[l];
        if (l > f) dp.fill(0);
        for (let o = 0; o < nOut; o++) {
          const g = dl[o];
          if (g === 0) continue;
          gb[o] += g;
          const row = o * nIn;
          for (let i = 0; i < nIn; i++) gW[row + i] += g * al[i];
          if (l > f) for (let i = 0; i < nIn; i++) dp[i] += g * W[row + i];
        }
        if (l > f) for (let i = 0; i < nIn; i++) if (al[i] <= 0) dp[i] = 0;   // ReLU of layer l − 1
      }
    }
    loss /= m;
    this.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    for (let l = f; l < L; l++) {
      const lr = this.rates[l];
      const W = net.W[l], b = net.b[l], mm = this.m[l], vv = this.v[l], nW = W.length;
      if (decay > 0) for (let q = 0; q < nW; q++) { loss += decay * W[q] * W[q]; this.gW[l][q] += 2 * decay * W[q]; }
      if (!(lr > 0)) continue;
      for (let q = 0; q < mm.length; q++) {
        const gr = q < nW ? this.gW[l][q] : this.gb[l][q - nW];
        mm[q] = b1 * mm[q] + (1 - b1) * gr;
        vv[q] = b2 * vv[q] + (1 - b2) * gr * gr;
        const dq = lr * (mm[q] / c1) / (Math.sqrt(vv[q] / c2) + 1e-8);
        if (q < nW) W[q] -= dq; else b[q - nW] -= dq;
      }
    }
    return loss;
  }
}

/** Pretraining on the source task: 2400 patches, mini-batches of 32, Adam 0.003. */
export const PRETRAIN = { n: 2400, seed: 101, netSeed: 7, steps: 1200, batch: 32, lr: 0.003 };

let cached: Mlp | null = null;
/** The pretrained source network (computed once per page, deterministic). */
export function pretrained(): Mlp {
  if (cached) return cached;
  const d = makeSet("source", PRETRAIN.n, PRETRAIN.seed);
  const net = newNet(d.k, PRETRAIN.netSeed);
  const opt = new LayerAdam(net, [PRETRAIN.lr, PRETRAIN.lr, PRETRAIN.lr]);
  const r = rng(PRETRAIN.seed + 1);
  const idx = new Array<number>(PRETRAIN.batch);
  for (let s = 0; s < PRETRAIN.steps; s++) {
    for (let i = 0; i < idx.length; i++) idx[i] = Math.floor(r.unif() * d.X.length);
    opt.step(d.X, d.y, idx);
  }
  cached = net;
  return net;
}

export type Strategy = "frozen" | "partial" | "full" | "scratch";
export const STRATEGIES: Strategy[] = ["frozen", "partial", "full", "scratch"];
export const STRATEGY_LABEL: Record<Strategy, string> = {
  frozen: "frozen + new head",
  partial: "last layers tuned",
  full: "all layers tuned",
  scratch: "from scratch",
};

/** The network a strategy starts from: the pretrained backbone with a new k-way head
 *  (or a fresh network for "scratch"), and the per-layer learning rates. */
export function startNet(strategy: Strategy, k: number, lr: number, headSeed = 11): { net: Mlp; rates: number[] } {
  if (strategy === "scratch") return { net: newNet(k, headSeed + 100), rates: [lr, lr, lr] };
  const base = cloneMlp(pretrained());
  const fresh = newNet(k, headSeed);
  base.sizes[3] = k; base.W[2] = fresh.W[2]; base.b[2] = fresh.b[2];
  const rates = strategy === "frozen" ? [0, 0, lr] : strategy === "partial" ? [0, lr, lr] : [lr, lr, lr];
  return { net: base, rates };
}

/** The backbone of `net` with the old source head put back (k = 4 source classes). */
export function withSourceHead(net: Mlp): Mlp {
  const old = cloneMlp(net), pre = pretrained();
  old.sizes[3] = pre.sizes[3]; old.W[2] = pre.W[2].slice(); old.b[2] = pre.b[2].slice();
  return old;
}

/** The source accuracy of a (fine-tuned) backbone with the old source head put back:
 *  how much of the old task it still knows (catastrophic forgetting). */
export function oldTaskAccuracy(net: Mlp, sourceTest: DataSet): number {
  return evaluate(withSourceHead(net), sourceTest).acc;
}

export interface CurvePoint { step: number; trainLoss: number; valLoss: number; valAcc: number }

/** Fine-tuning run of one strategy on a labelled target set, full batch. */
export class Run {
  net: Mlp; opt: LayerAdam; steps = 0; curve: CurvePoint[] = [];
  private A: Float64Array[]; private Av: Float64Array[];
  constructor(public strategy: Strategy, public train: DataSet, public val: DataSet, lr: number,
    public decay = 0, headSeed = 11, rates?: number[]) {
    const s = startNet(strategy, train.k, lr, headSeed);
    this.net = s.net;
    this.opt = new LayerAdam(this.net, rates ?? s.rates);
    this.A = featuresAt(this.net, this.opt.from, train.X);
    this.Av = featuresAt(this.net, this.opt.from, val.X);
    this.record(evaluateFrom(this.net, this.opt.from, this.A, train.y).loss);
  }
  private record(trainLoss: number): void {
    const v = evaluateFrom(this.net, this.opt.from, this.Av, this.val.y);
    this.curve.push({ step: this.steps, trainLoss, valLoss: v.loss, valAcc: v.acc });
  }
  /** n steps; records a curve point every `every` steps. */
  run(n: number, every = 5): void {
    for (let i = 0; i < n; i++) {
      const l = this.opt.step(this.A, this.train.y, undefined, this.decay);
      this.steps++;
      if (this.steps % every === 0) this.record(l);
    }
  }
  last(): CurvePoint { return this.curve[this.curve.length - 1]; }
  best(): CurvePoint { return this.curve.reduce((a, b) => (b.valAcc > a.valAcc ? b : a)); }
}

/** The first n patches of a balanced labelled set: n labels, classes in turn. */
export function subset(d: DataSet, n: number): DataSet {
  return { X: d.X.slice(0, n), y: d.y.slice(0, n), k: d.k };
}

/** The demo's choices of the number of labelled images, and the training steps per run. */
export const LABELS = [6, 9, 15, 30, 60, 90, 150, 300];
export const STEPS = 300;
