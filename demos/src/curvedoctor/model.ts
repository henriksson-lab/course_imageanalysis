// The training runs of the curve-doctor demo (unit "Reading training curves"): a small network
// on a three-armed spiral (three classes), healthy or with one thing wrong, trained live with
// Adam in mini-batches (lib/nn.ts). Each case also has a fix. The same cases as
// code/training/src/training_curves/run.rs (the numbers differ: other random numbers).
// No DOM, so the exercise numbers come from this code (demos/test/training-test.ts).
import { rng } from "../lib/ui";
import { mlp, gradients, applyGradients, Adam, Batcher, predict, outputProbs, type Mlp } from "../lib/nn";

/** Three-armed spiral: counts[k] points of class k in about [-1, 1]²; each arm turns by about
 *  200° from the centre outwards; `noise` is the sd of the scatter. Classes interleaved. */
export function spiral3(counts: [number, number, number], noise: number, seed: number): { X: number[][]; c: number[] } {
  const r = rng(seed), X: number[][] = [], c: number[] = [];
  const total = counts[0] + counts[1] + counts[2], left = [...counts];
  for (let i = 0; i < total; i++) {
    // the class with the largest share still to place
    let k = -1;
    for (let j = 0; j < 3; j++) if (left[j] > 0 && (k < 0 || left[j] / counts[j] > left[k] / counts[k])) k = j;
    left[k]--;
    const t = 0.15 + 0.85 * r.unif(), th = 2 * Math.PI * k / 3 + 3.5 * t;
    X.push([t * Math.cos(th) + noise * r.norm(), t * Math.sin(th) + noise * r.norm()]);
    c.push(k);
  }
  return { X, c };
}

export interface CaseConfig {
  hidden: number[];           // hidden layer sizes ([] = a linear classifier)
  lr: number; batch: number; epochs: number; decay: number;
  counts: [number, number, number];    // training points per class
  valCounts: [number, number, number];
  noise: number;
  zeroInputs: boolean;        // the bug: every input is 0
  shuffleLabels: boolean;     // the bug: labels shuffled against the inputs
  weighted: boolean;          // class weights total / (K n_k) in the loss
  seed: number;
}
export const BASE: CaseConfig = {
  hidden: [32, 32], lr: 0.01, batch: 32, epochs: 200, decay: 0.001, counts: [100, 100, 100], valCounts: [100, 100, 100],
  noise: 0.15, zeroInputs: false, shuffleLabels: false, weighted: false, seed: 1,
};

export type Diagnosis = "healthy" | "underfit" | "overfit" | "lrhigh" | "lrlow" | "deadinput" | "shuffled" | "imbalance";
export const DIAGNOSES: [Diagnosis, string][] = [
  ["healthy", "healthy"], ["underfit", "underfitting: model too simple"], ["overfit", "overfitting"],
  ["lrhigh", "learning rate too high"], ["lrlow", "learning rate too low"], ["deadinput", "bug: the inputs carry nothing"],
  ["shuffled", "bug: labels do not match the inputs"], ["imbalance", "class imbalance"],
];

export interface Case { id: string; diagnosis: Diagnosis; cfg: Partial<CaseConfig>; fix: Partial<CaseConfig>; fixLabel: string; explain: string }
/** The cases in the order of the demo's buttons ("run 1" … "run 8"), deliberately mixed. */
export const CASES: Case[] = [
  { id: "1", diagnosis: "overfit", cfg: { hidden: [64, 64], counts: [10, 10, 10], epochs: 400, batch: 30, decay: 0 }, fix: { counts: [100, 100, 100], hidden: [64, 64], epochs: 400, batch: 30, decay: 0.001 }, fixLabel: "10× more data + weight decay",
    explain: "30 training points and a large network: the training loss goes to zero while the validation loss turns up after a few dozen epochs. More data (or weight decay, or stopping at the lowest validation loss) cures it." },
  { id: "2", diagnosis: "deadinput", cfg: { zeroInputs: true }, fix: {}, fixLabel: "load the inputs",
    explain: "Both losses sit exactly at ln 3 = 1.099 from the first epoch: the network can only learn how common each class is, because every input is 0 (the images were never loaded). A loss stuck at ln K is the signature of inputs that carry nothing." },
  { id: "3", diagnosis: "lrhigh", cfg: { lr: 0.3 }, fix: {}, fixLabel: "learning rate 0.01",
    explain: "A fast first drop, then the loss jumps up and down and never settles: each step overshoots the valley. Lower the learning rate (or use a schedule that decays it)." },
  { id: "4", diagnosis: "healthy", cfg: {}, fix: {}, fixLabel: "nothing to fix",
    explain: "Both curves fall fast and then flatten together; the validation loss stays a little above the training loss. Nothing to fix (weight decay 0.001 keeps the curves together)." },
  { id: "5", diagnosis: "imbalance", cfg: { counts: [240, 50, 10], valCounts: [240, 50, 10] }, fix: { counts: [240, 50, 10], valCounts: [240, 50, 10], weighted: true }, fixLabel: "class weights in the loss",
    explain: "The curves look healthy and the accuracy is high, but 80 % of the points are class 1. Look at the recall of each class: the rare class is found least often. Weighting the loss by total / (K n_k) helps the rare class at a small cost in overall accuracy." },
  { id: "6", diagnosis: "underfit", cfg: { hidden: [] }, fix: {}, fixLabel: "two hidden layers",
    explain: "Both losses level off high (training about 0.5, validation about 0.6) and close together: a linear classifier cannot follow the spiral arms. Training longer does not help; a larger model does." },
  { id: "7", diagnosis: "shuffled", cfg: { shuffleLabels: true }, fix: {}, fixLabel: "fix the labels",
    explain: "The training loss falls slowly below ln 3 while the validation loss rises above it: the network memorises labels that have nothing to do with the inputs. Check that images and labels still belong together." },
  { id: "8", diagnosis: "lrlow", cfg: { lr: 0.0001 }, fix: {}, fixLabel: "learning rate 0.01",
    explain: "Both curves fall together, but slowly, and are still falling at the end: training stopped long before it was done. Raise the learning rate (or train much longer)." },
];

export interface EpochPoint { epoch: number; train: number; val: number; trainAcc: number; valAcc: number }

/** One training run of a case. */
export class CaseRun {
  net: Mlp; opt: Adam; batcher: Batcher; cfg: CaseConfig;
  X: number[][]; Y: number[][]; c: number[]; Xv: number[][]; Yv: number[][]; cv: number[];
  weights: number[];
  history: EpochPoint[] = [];
  epoch = 0;
  constructor(cfg: Partial<CaseConfig>) {
    const o = this.cfg = { ...BASE, ...cfg };
    const tr = spiral3(o.counts, o.noise, 1), va = spiral3(o.valCounts, o.noise, 2);
    const zero = (x: number[]) => (o.zeroInputs ? x.map(() => 0) : x);
    this.X = tr.X.map(zero); this.c = tr.c.slice();
    if (o.shuffleLabels) { const r = rng(9); for (let i = this.c.length - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); [this.c[i], this.c[j]] = [this.c[j], this.c[i]]; } }
    this.Xv = va.X.map(zero); this.cv = va.c;
    const oh = (k: number) => [0, 1, 2].map(j => (j === k ? 1 : 0));
    this.Y = this.c.map(oh); this.Yv = this.cv.map(oh);
    const n = o.counts[0] + o.counts[1] + o.counts[2];
    this.weights = o.weighted ? o.counts.map(m => n / (3 * m)) : [1, 1, 1];
    this.net = mlp([2, ...o.hidden, 3], [...o.hidden.map(() => "relu" as const), "id"], 3);
    this.opt = new Adam(this.net, o.lr);
    this.batcher = new Batcher(this.X.length, o.batch, o.seed);
    this.record();
  }
  /** Mean cross-entropy and accuracy on a set (no weights, no decay). */
  evaluate(X: number[][], c: number[]): [number, number] {
    let l = 0, ok = 0;
    X.forEach((x, i) => {
      const p = outputProbs(predict(this.net, x), "ce");
      l -= Math.log(Math.max(1e-12, p[c[i]]));
      if (p.indexOf(Math.max(...p)) === c[i]) ok++;
    });
    return [l / X.length, ok / X.length];
  }
  record(): void {
    const [train, trainAcc] = this.evaluate(this.X, this.c), [val, valAcc] = this.evaluate(this.Xv, this.cv);
    this.history.push({ epoch: this.epoch, train, val, trainAcc, valAcc });
  }
  /** One epoch of mini-batch steps. With class weights, each example's gradient is scaled by its
   *  class weight (one gradient call per class in the batch). */
  trainEpoch(): void {
    const steps = Math.ceil(this.X.length / this.cfg.batch);
    for (let s = 0; s < steps; s++) {
      const idx = this.batcher.next();
      const g = gradients(this.net, this.X, this.Y, { loss: "ce", decay: this.cfg.decay }, idx);
      if (this.cfg.weighted) {
        // gradient of (1/m) Σ w_c(n) ℓ_n: the sum over classes of w_k · (m_k/m) · (gradient of the class-k part)
        g.gW.forEach(a => a.fill(0)); g.gb.forEach(a => a.fill(0));
        for (let k = 0; k < 3; k++) {
          const sub = idx.filter(i => this.c[i] === k);
          if (!sub.length) continue;
          const gk = gradients(this.net, this.X, this.Y, { loss: "ce" }, sub), f = this.weights[k] * sub.length / idx.length;
          gk.gW.forEach((a, l) => a.forEach((v, q) => { g.gW[l][q] += f * v; }));
          gk.gb.forEach((a, l) => a.forEach((v, q) => { g.gb[l][q] += f * v; }));
        }
        const lam = this.cfg.decay;
        if (lam > 0) this.net.W.forEach((W, l) => W.forEach((w, q) => { g.gW[l][q] += 2 * lam * w; }));
      }
      applyGradients(this.net, g, this.opt);
    }
    this.epoch++;
    this.record();
  }
  done(): boolean { return this.epoch >= this.cfg.epochs; }
  /** Recall of each class on the validation set. */
  recalls(): number[] {
    const hit = [0, 0, 0], all = [0, 0, 0];
    this.Xv.forEach((x, i) => { const o = predict(this.net, x); all[this.cv[i]]++; if (o.indexOf(Math.max(...o)) === this.cv[i]) hit[this.cv[i]]++; });
    return hit.map((h, k) => (all[k] ? h / all[k] : NaN));
  }
  /** Class probabilities at an input point. */
  probs(x: number, y: number): Float64Array { return outputProbs(predict(this.net, [x, y]), "ce"); }
}

export const LN3 = Math.log(3);
