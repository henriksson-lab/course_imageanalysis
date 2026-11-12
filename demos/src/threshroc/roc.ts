// ROC and precision–recall curves of a threshold sweep, from the histograms of the
// object and background pixels (one pass over the grey levels). No DOM.
// (Belongs in the segmentation section of lib/metrics.ts; kept here because that file is
// owned elsewhere.)

export interface SweepPoint {
  t: number;              // threshold: object where I > t (t = −1: everything is object)
  tp: number; fp: number; fn: number; tn: number;
  tpr: number; fpr: number; precision: number; dice: number;
}

/**
 * Sweep t = −1, 0, …, L−1 over integer grey levels: object where I > t. `hObj[v]` and
 * `hBg[v]` count the object and background pixels with grey level v. TPR(t) and FPR(t)
 * are 1 − the cumulative histograms of the two classes.
 */
export function sweep(hObj: ArrayLike<number>, hBg: ArrayLike<number>): SweepPoint[] {
  const L = hObj.length;
  let P = 0, Nn = 0;
  for (let v = 0; v < L; v++) { P += hObj[v]; Nn += hBg[v]; }
  const out: SweepPoint[] = [];
  let tp = P, fp = Nn;   // t = −1: all pixels object
  for (let t = -1; t < L; t++) {
    if (t >= 0) { tp -= hObj[t]; fp -= hBg[t]; }
    const fn = P - tp, tn = Nn - fp;
    out.push({
      t, tp, fp, fn, tn,
      tpr: P > 0 ? tp / P : 1, fpr: Nn > 0 ? fp / Nn : 0,
      precision: tp + fp > 0 ? tp / (tp + fp) : 1,
      dice: 2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : 1,
    });
  }
  return out;
}

/** Area under the ROC curve (trapezoids over the sweep, FPR increasing). */
export function auc(pts: SweepPoint[]): number {
  const s = [...pts].sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr);
  let a = 0;
  for (let i = 1; i < s.length; i++) a += (s[i].fpr - s[i - 1].fpr) * (s[i].tpr + s[i - 1].tpr) / 2;
  return a;
}

/**
 * The probability that a random object pixel is brighter than a random background pixel
 * (ties count one half): equals the AUC of the sweep.
 */
export function probBrighter(hObj: ArrayLike<number>, hBg: ArrayLike<number>): number {
  let P = 0, Nn = 0, s = 0, below = 0;
  for (let v = 0; v < hObj.length; v++) { P += hObj[v]; Nn += hBg[v]; }
  for (let v = 0; v < hObj.length; v++) {
    s += hObj[v] * (below + 0.5 * hBg[v]);
    below += hBg[v];
  }
  return P > 0 && Nn > 0 ? s / (P * Nn) : NaN;
}
