// The computations of the domainshift demo (no DOM). Every cell is measured by two features,
// its mean intensity and its size (both about 0..1). Two classes: interphase (dimmer, larger)
// and mitotic (brighter, smaller). Microscope A gives the training data; microscope B measures
// the same kind of cells with a different intensity scale: intensity_B = gain · intensity + offset,
// gain = 1 − 0.55 s, offset = 0.12 s for the shift s in [0, 1]. A linear classifier
// (logistic regression, Newton's method) is trained on A; the remedies:
//  - "as is": the A classifier on B;
//  - "normalise": both microscopes' features standardised with their own statistics
//    (mean and SD over all of that microscope's cells, no labels needed): re-estimated statistics;
//  - "augment": trained on A with random intensity gains and offsets;
//  - "fine-tune": refitted on 10 labelled B cells, penalised towards the A weights.
import { rng } from "../lib/ui";

export interface Cell { f: [number, number]; c: number }

/** n cells measured by microscope A; `fracMitotic` of them mitotic. */
export function cellsA(n: number, seed: number, fracMitotic = 0.5): Cell[] {
  const r = rng(seed), out: Cell[] = [];
  for (let i = 0; i < n; i++) {
    const c = r.unif() < fracMitotic ? 1 : 0;
    const I = c ? 0.62 + 0.08 * r.norm() : 0.42 + 0.08 * r.norm();
    const s = c ? 0.40 + 0.09 * r.norm() : 0.58 + 0.09 * r.norm();
    out.push({ f: [I, s], c });
  }
  return out;
}

export const gainOf = (s: number) => 1 - 0.55 * s;
export const offsetOf = (s: number) => 0.12 * s;

/** The same cells measured by microscope B at shift s. */
export function toB(cells: Cell[], s: number): Cell[] {
  const g = gainOf(s), o = offsetOf(s);
  return cells.map(c => ({ f: [g * c.f[0] + o, c.f[1]], c: c.c }));
}

export type W = [number, number, number];   // p(mitotic) = sigmoid(w0 + w1 f1 + w2 f2)
const sig = (t: number) => 1 / (1 + Math.exp(-t));
export const pMitotic = (w: W, f: [number, number]) => sig(w[0] + w[1] * f[0] + w[2] * f[1]);

/** Logistic regression by Newton's method, with the penalty λ‖w − prior‖² (bias included). */
export function fitLogistic(cells: Cell[], lambda = 1e-3, prior: W = [0, 0, 0], iters = 30): W {
  const w: W = [...prior] as W;
  for (let it = 0; it < iters; it++) {
    const g = [0, 0, 0], H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const c of cells) {
      const x = [1, c.f[0], c.f[1]], p = pMitotic(w, c.f), v = p * (1 - p);
      for (let a = 0; a < 3; a++) { g[a] += (p - c.c) * x[a]; for (let b = 0; b < 3; b++) H[a][b] += v * x[a] * x[b]; }
    }
    for (let a = 0; a < 3; a++) { g[a] += 2 * lambda * (w[a] - prior[a]); H[a][a] += 2 * lambda; }
    const d = solve3(H, g);
    let step = 0;
    for (let a = 0; a < 3; a++) { w[a] -= d[a]; step = Math.max(step, Math.abs(d[a])); }
    if (step < 1e-9) break;
  }
  return w;
}

function solve3(A: number[][], b: number[]): number[] {
  const det = (M: number[][]) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(A);
  return [0, 1, 2].map(k => det(A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / D);
}

export function accuracy(w: W, cells: Cell[]): number {
  let ok = 0;
  for (const c of cells) if ((pMitotic(w, c.f) > 0.5 ? 1 : 0) === c.c) ok++;
  return ok / cells.length;
}

/** Mean and SD of each feature over a set of cells (no labels used). */
export function stats(cells: Cell[]): { mean: [number, number]; sd: [number, number] } {
  const mean: [number, number] = [0, 0], sd: [number, number] = [0, 0];
  for (const c of cells) for (let k = 0; k < 2; k++) mean[k] += c.f[k] / cells.length;
  for (const c of cells) for (let k = 0; k < 2; k++) sd[k] += (c.f[k] - mean[k]) ** 2 / cells.length;
  return { mean, sd: [Math.sqrt(sd[0]), Math.sqrt(sd[1])] };
}

export function standardise(cells: Cell[], st = stats(cells)): Cell[] {
  return cells.map(c => ({ f: [(c.f[0] - st.mean[0]) / st.sd[0], (c.f[1] - st.mean[1]) / st.sd[1]], c: c.c }));
}

/** A shift score: the largest |mean_B − mean_A| / SD_A over the features (no labels needed). */
export function shiftScore(a: Cell[], b: Cell[]): number {
  const sa = stats(a), sb = stats(b);
  return Math.max(...[0, 1].map(k => Math.abs(sb.mean[k] - sa.mean[k]) / sa.sd[k]));
}

/** A with random intensity gains in [0.4, 1.2] and offsets in [−0.05, 0.2] (`copies` per cell). */
export function augment(cells: Cell[], seed: number, copies = 3): Cell[] {
  const r = rng(seed), out: Cell[] = [];
  for (const c of cells) for (let k = 0; k < copies; k++) {
    const g = 0.4 + 0.8 * r.unif(), o = -0.05 + 0.25 * r.unif();
    out.push({ f: [g * c.f[0] + o, c.f[1]], c: c.c });
  }
  return out;
}

export type Remedy = "none" | "normalise" | "augment" | "finetune";
export const REMEDIES: Remedy[] = ["none", "normalise", "augment", "finetune"];
export const REMEDY_LABEL: Record<Remedy, string> = {
  none: "as is", normalise: "normalise per microscope", augment: "trained with augmentation", finetune: "fine-tuned on 10 B cells",
};

export const SETUP = { nA: 400, seedA: 1, nB: 400, seedB: 2, augSeed: 3, nTune: 10 };

/** Everything trained on A, once. */
export function trainA(fracA = 0.5) {
  const A = cellsA(SETUP.nA, SETUP.seedA, fracA);
  const stA = stats(A);
  return {
    A, stA,
    wNone: fitLogistic(A),
    wNorm: fitLogistic(standardise(A, stA)),
    wAug: fitLogistic(augment(A, SETUP.augSeed)),
  };
}
export type Trained = ReturnType<typeof trainA>;

/** The B cells at shift s, and the accuracy of every remedy on them. */
export function evaluateB(T: Trained, s: number, fracB: number) {
  const B = toB(cellsA(SETUP.nB, SETUP.seedB, fracB), s);
  // 10 labelled B cells (5 of each class) for fine-tuning: from a separate draw
  const extra = toB(cellsA(200, SETUP.seedB + 10, 0.5), s);
  const tune = [...extra.filter(c => c.c === 0).slice(0, SETUP.nTune / 2), ...extra.filter(c => c.c === 1).slice(0, SETUP.nTune / 2)];
  const wTune = fitLogistic(tune, 0.05, T.wNone);
  const stB = stats(B);
  const acc: Record<Remedy, number> = {
    none: accuracy(T.wNone, B),
    normalise: accuracy(T.wNorm, standardise(B, stB)),
    augment: accuracy(T.wAug, B),
    finetune: accuracy(wTune, B),
  };
  return { B, tune, wTune, stB, acc, score: shiftScore(T.A, B) };
}
