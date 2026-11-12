// The pixel-classification pipeline without the DOM: painted label map + feature stack ->
// training set (capped, optionally class-balanced, with stroke ids as groups) -> any
// Classifier (lib/classifier.ts) -> per-pixel scores -> evaluation on a held-out region.
// Used by the demo, the tests and the solutions scripts.
import { FeatureStack, gatherPixels } from "../lib/featurestack";
import { TrainingSet, labelsAndUncertainty } from "../lib/classifier";
import { label2d } from "../lib/label";
import { rng } from "../lib/ui";
import { confusionMatrix } from "../lib/metrics";

export interface TrainingInfo {
  data: TrainingSet;
  /** Pixel index of every training sample. */
  pix: Int32Array;
  /** Painted pixels per class (before the cap), and how many were used. */
  painted: number[];
  used: number[];
  /** Number of strokes (connected painted pieces). */
  strokes: number;
}

/** Is pixel p inside the held-out region (x ≥ fraction · w)? */
export const inRegion = (p: number, w: number, fraction: number): boolean => (p % w) >= Math.round(fraction * w);

/**
 * Training set from the painted label map (values 1..k → classes 0..k-1), leaving out the
 * pixels for which `exclude(p)` is true, at most `cap` samples (random subset, the same
 * fraction from every class), with balancing weights n / (k · n_c) when `balance` is set.
 */
export function trainingSet(S: FeatureStack, labels: Uint8Array, o: { exclude?: (p: number) => boolean; cap?: number; balance?: boolean; seed?: number; k?: number } = {}): TrainingInfo | null {
  const N = S.w * S.h;
  let k = o.k ?? 0;
  const all: number[] = [];
  for (let p = 0; p < N; p++) if (labels[p] && !(o.exclude?.(p))) { all.push(p); if (!o.k) k = Math.max(k, labels[p]); }
  if (!all.length || k < 2) return null;
  const painted = new Array(k).fill(0);
  for (const p of all) painted[labels[p] - 1]++;
  // cap: keep a random subset of the same fraction of every class
  const cap = o.cap ?? 4000, r = rng(o.seed ?? 1);
  let pix = all;
  if (all.length > cap) {
    const keep = cap / all.length;
    pix = all.filter(() => r.unif() < keep);
  }
  const n = pix.length, y = new Int32Array(n), used = new Array(k).fill(0);
  pix.forEach((p, i) => { y[i] = labels[p] - 1; used[y[i]]++; });
  let w: Float64Array | undefined;
  if (o.balance) {
    const present = used.filter(c => c > 0).length;
    w = Float64Array.from(y, c => n / (present * used[c]));
  }
  // groups: connected pieces of each label (8-connected), numbered across labels
  const groupMap = new Int32Array(N);
  let offset = 0;
  for (let v = 1; v <= k; v++) {
    const mask = new Uint8Array(N);
    for (const p of all) if (labels[p] === v) mask[p] = 1;
    const L = label2d(mask, S.w, S.h, 8);
    for (const p of all) if (labels[p] === v) groupMap[p] = offset + L.data[p];
    offset += L.count;
  }
  const groups = Int32Array.from(pix, p => groupMap[p]);
  const X = gatherPixels(S, pix);
  return { data: { X, n, d: S.d, y, k, w, groups, names: S.names }, pix: Int32Array.from(pix), painted, used, strokes: offset };
}

export interface RegionScores {
  /** Painted pixels inside the region: how many, and the fraction classified as painted. */
  paintedN: number;
  paintedAcc: number;
  /** Against the ground truth (if given): accuracy, k × k confusion matrix (truth rows), Dice per class. */
  truthAcc: number;
  confusion: Float64Array | null;
  dice: number[];
}

/**
 * Scores in a region (`inside(p)`; all pixels if omitted). `truth` holds classes 1..K (0 =
 * unknown); predicted labels are 0-based classes, compared as class + 1.
 */
export function regionScores(pred: Uint8Array, labels: Uint8Array, truth: Uint8Array | null, kTruth: number, inside?: (p: number) => boolean): RegionScores {
  let pn = 0, pc = 0;
  const P: number[] = [], T: number[] = [];
  for (let p = 0; p < pred.length; p++) {
    if (inside && !inside(p)) continue;
    if (labels[p]) { pn++; if (pred[p] + 1 === labels[p]) pc++; }
    if (truth && truth[p]) { P.push(pred[p]); T.push(truth[p] - 1); }
  }
  let truthAcc = NaN, conf: Float64Array | null = null;
  const dice: number[] = [];
  if (truth && T.length) {
    conf = confusionMatrix(P, T, kTruth);
    let ok = 0, all = 0;
    for (let t = 0; t < kTruth; t++) for (let q = 0; q < kTruth; q++) { all += conf[t * kTruth + q]; if (t === q) ok += conf[t * kTruth + q]; }
    truthAcc = ok / all;
    for (let c = 0; c < kTruth; c++) {
      let tp = conf[c * kTruth + c], fp = 0, fn = 0;
      for (let q = 0; q < kTruth; q++) if (q !== c) { fp += conf[q * kTruth + c]; fn += conf[c * kTruth + q]; }
      dice.push(2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : NaN);
    }
  }
  return { paintedN: pn, paintedAcc: pn ? pc / pn : NaN, truthAcc, confusion: conf, dice };
}

export { labelsAndUncertainty };
