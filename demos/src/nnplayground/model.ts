// The network and training loop of the nnplayground demo (no DOM, so presets and the numbers
// quoted in the exercises can be computed in node with exactly this code).
//
// Input: a point (x, y) of the unit square, fed to the network as (2x − 1, 2y − 1).
// Two classes: one output logit, p = sigmoid, binary cross-entropy. Three or more classes:
// one logit per class, softmax, cross-entropy. Training: mini-batch SGD with momentum 0.9
// (lib/nn.ts), optional weight decay and dropout. Every point is in the training set or (with
// a validation fraction > 0) in the validation set, decided by a fixed hash of its index.
import { mlp, gradients, applyGradients, Sgd, Batcher, meanLoss, predict, forward, outputProbs, type Act, type Loss, type Mlp } from "../lib/nn";
import type { LPoint } from "../lib/datasets2d";

export interface Config {
  hidden: number[];          // units per hidden layer, e.g. [] or [2] or [8, 8]
  act: Act;                  // activation of the hidden layers
  lr: number;
  batch: number;             // mini-batch size; 0 = all training points (full batch)
  decay: number;             // weight decay λ
  dropout: number;           // dropout probability of hidden units during training
  momentum: number;
  seed: number;              // initial weights and batch order
  valFrac: number;           // fraction of points held out for validation
}

export const DEFAULT: Config = { hidden: [2], act: "relu", lr: 0.3, batch: 0, decay: 0, dropout: 0, momentum: 0.9, seed: 1, valFrac: 0 };

export const inputOf = (x: number, y: number): [number, number] => [2 * x - 1, 2 * y - 1];

/** Is point i in the validation set? A fixed, well-spread choice (golden-ratio sequence). */
export const isVal = (i: number, frac: number): boolean => frac > 0 && ((i * 0.6180339887 + 0.31) % 1) < frac;

export interface HistoryPoint { step: number; epoch: number; train: number; val: number; trainAcc: number; valAcc: number }

export class Trainer {
  net: Mlp;
  opt: Sgd;
  batcher: Batcher;
  loss: Loss;
  k: number;
  X: number[][] = []; Y: number[][] = []; Xv: number[][] = []; Yv: number[][] = [];
  cv: number[] = []; cvv: number[] = [];
  step = 0;
  history: HistoryPoint[] = [];

  constructor(points: LPoint[], k: number, public cfg: Config) {
    this.k = Math.max(2, k);
    this.loss = this.k === 2 ? "bce" : "ce";
    const nOut = this.k === 2 ? 1 : this.k;
    const sizes = [2, ...cfg.hidden, nOut];
    const acts: Act[] = [...cfg.hidden.map(() => cfg.act), "id"];
    this.net = mlp(sizes, acts, cfg.seed);
    this.opt = new Sgd(this.net, cfg.lr, cfg.momentum);
    this.setPoints(points);
    this.batcher = new Batcher(Math.max(1, this.X.length), cfg.batch > 0 ? cfg.batch : Math.max(1, this.X.length), cfg.seed + 1000);
    this.record();
  }

  target(c: number): number[] {
    if (this.k === 2) return [c === 1 ? 1 : 0];
    return Array.from({ length: this.k }, (_, j) => (j === c ? 1 : 0));
  }

  /** New points (keeps the weights); the batcher restarts. */
  setPoints(points: LPoint[]): void {
    this.X = []; this.Y = []; this.Xv = []; this.Yv = []; this.cv = []; this.cvv = [];
    points.forEach((p, i) => {
      const x = inputOf(p.x, p.y), y = this.target(p.c);
      if (isVal(i, this.cfg.valFrac)) { this.Xv.push(x); this.Yv.push(y); this.cvv.push(p.c); }
      else { this.X.push(x); this.Y.push(y); this.cv.push(p.c); }
    });
    if (this.batcher) this.batcher = new Batcher(Math.max(1, this.X.length), this.cfg.batch > 0 ? this.cfg.batch : Math.max(1, this.X.length), this.cfg.seed + 1000);
  }

  /** Class probabilities at a point of the unit square. */
  probs(x: number, y: number): Float64Array {
    const p = outputProbs(predict(this.net, inputOf(x, y)), this.loss);
    return this.k === 2 ? Float64Array.of(1 - p[0], p[0]) : p;
  }

  accuracy(X: number[][], c: number[]): number {
    if (!X.length) return NaN;
    let ok = 0;
    X.forEach((x, i) => {
      const p = outputProbs(predict(this.net, x), this.loss);
      const pred = this.k === 2 ? (p[0] > 0.5 ? 1 : 0) : p.indexOf(Math.max(...p));
      if (pred === c[i]) ok++;
    });
    return ok / X.length;
  }

  record(): void {
    this.history.push({
      step: this.step, epoch: this.step * Math.min(this.batcher.size, this.X.length) / Math.max(1, this.X.length),
      train: meanLoss(this.net, this.X, this.Y, this.loss), val: meanLoss(this.net, this.Xv, this.Yv, this.loss),
      trainAcc: this.accuracy(this.X, this.cv), valAcc: this.accuracy(this.Xv, this.cvv),
    });
  }

  /** n gradient steps on mini-batches; records the losses after each. */
  train(n: number): void {
    if (!this.X.length) return;
    for (let s = 0; s < n; s++) {
      const idx = this.batcher.next();
      const g = gradients(this.net, this.X, this.Y, { loss: this.loss, decay: this.cfg.decay, dropout: this.cfg.dropout, rand: () => this.batcher.unif() }, idx);
      applyGradients(this.net, g, this.opt);
      this.step++;
      this.record();
    }
  }

  /** Outputs of every hidden unit at a point: unit j of hidden layer l is a[l + 1][j]. */
  hiddenOutputs(x: number, y: number): Float64Array[] {
    const a = forward(this.net, inputOf(x, y)).a;
    return a.slice(1, a.length - 1);
  }
}

/** A single neuron for the by-hand mode: p(class 2) = sigmoid(w1 x' + w2 y' + b) in input units. */
export function setNeuron(t: Trainer, w1: number, w2: number, b: number): void {
  t.net.W[t.net.W.length - 1].set([w1, w2]);
  t.net.b[t.net.b.length - 1][0] = b;
}
