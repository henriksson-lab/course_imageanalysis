// Quality and agreement measures (no DOM). Two sections:
//  - image quality: MSE, RMSE, MAE, PSNR (with an explicit peak), SSIM and its map;
//  - segmentation / classification: confusion counts of two masks, Dice, Jaccard (IoU),
//    precision, recall, F1, accuracy, specificity, and a k-class confusion matrix.
// Images are flat arrays of equal length; masks count non-zero as foreground.
// Object-level evaluation (matching, AP) belongs to the evaluation day and calls the same
// precision/recall/f1 on object counts.

import { Grey, gaussianKernel, separable, createGrey } from "./imageops";

type A = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Image quality

/** Mean squared error. */
export function mse(a: A, b: A): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
  return s / a.length;
}

/** Root-mean-square difference. */
export function rmse(a: A, b: A): number {
  return Math.sqrt(mse(a, b));
}

/** Mean absolute difference. */
export function mae(a: A, b: A): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return s / a.length;
}

/**
 * Peak signal-to-noise ratio in dB, 10·log10(peak² / MSE). The peak is explicit on
 * purpose (255 for 8 bit, 4095 for 12 bit, 1 for [0, 1] data); Infinity for identical images.
 */
export function psnr(a: A, b: A, peak: number): number {
  const m = mse(a, b);
  return m === 0 ? Infinity : 10 * Math.log10(peak * peak / m);
}

/** PSNR from a known MSE. */
export const psnrFromMse = (m: number, peak: number): number => (m === 0 ? Infinity : 10 * Math.log10(peak * peak / m));

export interface SsimOptions {
  /** Dynamic range L of the data (255 for 8 bit). Default 255. */
  peak?: number;
  /** Gaussian window σ (default 1.5) and radius (default 5: the standard 11 × 11 window). */
  sigma?: number;
  radius?: number;
  K1?: number;   // default 0.01
  K2?: number;   // default 0.03
  /** Average only where the window lies fully inside the image (default true, as the reference code). */
  valid?: boolean;
}

/**
 * Structural similarity (Wang et al. 2004) of two w × h images: local means, variances and
 * covariance under a Gaussian window, SSIM = (2μaμb + C1)(2σab + C2) / ((μa² + μb² + C1)(σa² + σb² + C2)),
 * C1 = (K1·L)², C2 = (K2·L)². Returns the mean and the per-pixel map.
 */
export function ssim(a: A, b: A, w: number, h: number, opts: SsimOptions = {}): { mean: number; map: Grey } {
  const L = opts.peak ?? 255, sigma = opts.sigma ?? 1.5, r = opts.radius ?? 5;
  const C1 = ((opts.K1 ?? 0.01) * L) ** 2, C2 = ((opts.K2 ?? 0.03) * L) ** 2;
  const k = gaussianKernel(sigma, 0, r);
  const n = w * h;
  const mk = (f: (i: number) => number): Grey => { const g = createGrey(w, h); for (let i = 0; i < n; i++) g.data[i] = f(i); return g; };
  const blur = (g: Grey) => separable(g, k, k, "mirror");
  const ma = blur(mk(i => a[i])), mb = blur(mk(i => b[i]));
  const saa = blur(mk(i => a[i] * a[i])), sbb = blur(mk(i => b[i] * b[i])), sab = blur(mk(i => a[i] * b[i]));
  const map = createGrey(w, h);
  let sum = 0, cnt = 0;
  const valid = opts.valid ?? true;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, mua = ma.data[i], mub = mb.data[i];
    const va = saa.data[i] - mua * mua, vb = sbb.data[i] - mub * mub, cab = sab.data[i] - mua * mub;
    const s = ((2 * mua * mub + C1) * (2 * cab + C2)) / ((mua * mua + mub * mub + C1) * (va + vb + C2));
    map.data[i] = s;
    if (!valid || (x >= r && x < w - r && y >= r && y < h - r)) { sum += s; cnt++; }
  }
  return { mean: cnt ? sum / cnt : NaN, map };
}

// ---------------------------------------------------------------------------------------
// Segmentation and classification

/** Confusion counts. For object matching, tn is usually undefined (0). */
export interface Counts { tp: number; fp: number; fn: number; tn: number }

/** Compare a predicted and a true mask pixel by pixel (non-zero = foreground); optional `valid` mask. */
export function confusion(pred: A, truth: A, valid?: A): Counts {
  const c: Counts = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (let i = 0; i < pred.length; i++) {
    if (valid && !valid[i]) continue;
    const p = pred[i] ? 1 : 0, t = truth[i] ? 1 : 0;
    if (p && t) c.tp++; else if (p) c.fp++; else if (t) c.fn++; else c.tn++;
  }
  return c;
}

const ratio = (a: number, b: number, empty: number) => (b === 0 ? empty : a / b);

/** Dice = 2TP / (2TP + FP + FN) (= F1); 1 when both masks are empty. */
export const dice = (c: Counts): number => ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn, 1);
/** Jaccard (IoU) = TP / (TP + FP + FN); 1 when both masks are empty. Jaccard = Dice / (2 − Dice). */
export const jaccard = (c: Counts): number => ratio(c.tp, c.tp + c.fp + c.fn, 1);
/** Precision = TP / (TP + FP) (1 when nothing was predicted). */
export const precision = (c: Counts): number => ratio(c.tp, c.tp + c.fp, 1);
/** Recall (sensitivity, true-positive rate) = TP / (TP + FN) (1 when there is nothing to find). */
export const recall = (c: Counts): number => ratio(c.tp, c.tp + c.fn, 1);
/** Specificity (true-negative rate) = TN / (TN + FP). */
export const specificity = (c: Counts): number => ratio(c.tn, c.tn + c.fp, 1);
/** False-positive rate = FP / (FP + TN) = 1 − specificity (the x axis of a ROC curve). */
export const fpr = (c: Counts): number => ratio(c.fp, c.fp + c.tn, 0);
/** Accuracy = (TP + TN) / all. */
export const accuracy = (c: Counts): number => ratio(c.tp + c.tn, c.tp + c.tn + c.fp + c.fn, 1);
/** F1 = harmonic mean of precision and recall (= Dice for pixels). */
export const f1 = (c: Counts): number => dice(c);

/** Dice of two masks directly. */
export const diceMasks = (pred: A, truth: A): number => dice(confusion(pred, truth));
/** Jaccard (IoU) of two masks directly. */
export const jaccardMasks = (pred: A, truth: A): number => jaccard(confusion(pred, truth));

/**
 * k-class confusion matrix M (k × k, row-major): M[t*k + p] = number of items with true
 * class t predicted as p. Classes outside 0..k-1 are skipped.
 */
export function confusionMatrix(pred: A, truth: A, k: number): Float64Array {
  const M = new Float64Array(k * k);
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i], t = truth[i];
    if (p >= 0 && p < k && t >= 0 && t < k) M[t * k + p]++;
  }
  return M;
}

/** One-vs-rest counts of class c from a k × k confusion matrix. */
export function classCounts(M: A, k: number, c: number): Counts {
  let tp = 0, fp = 0, fn = 0, all = 0;
  for (let t = 0; t < k; t++) for (let p = 0; p < k; p++) {
    const v = M[t * k + p];
    all += v;
    if (t === c && p === c) tp += v; else if (p === c) fp += v; else if (t === c) fn += v;
  }
  return { tp, fp, fn, tn: all - tp - fp - fn };
}
