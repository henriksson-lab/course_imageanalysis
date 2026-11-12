// Computation of the threshroc demo (no DOM), so scripts and tests can reproduce its numbers.
import { Grey, gaussianBlur } from "../lib/imageops";
import { intHistogram } from "../lib/histogram";
import { otsu, kittler, localThreshold } from "../lib/threshold";
import { blobMask, renderScene, quantise8 } from "../histthresh/scenes";
import { sweep, auc, SweepPoint } from "./roc";

export const N = 256;
export interface Settings { fraction: number; contrast: number; noise: number; light: number; smooth: boolean }

export interface RocModel {
  img: Grey; truth: Uint8Array; pts: SweepPoint[]; auc: number;
  marks: { otsu: number; kittler: number; dice: number };
  localMask: Uint8Array;
  /** Sweep point of threshold t (object where I > t). */
  at(t: number): SweepPoint;
}

export function rocModel(s: Settings): RocModel {
  const truth = blobMask(N, N, s.fraction, 6, 21);
  let img = renderScene(truth, { background: 70, contrast: s.contrast, noise: s.noise, gradient: s.light, edgeBlur: 1, seed: 4 });
  if (s.smooth) img = quantise8(gaussianBlur(img, 1.5));
  const hObj = intHistogram(img.data, 0, 255, truth).counts;
  const hBg = intHistogram(img.data, 0, 255, Uint8Array.from(truth, v => 1 - v)).counts;
  const pts = sweep(hObj, hBg);
  const hAll = intHistogram(img.data, 0, 255).counts;
  let b = 0;
  for (let i = 1; i < pts.length; i++) if (pts[i].dice > pts[b].dice) b = i;
  const marks = { otsu: otsu(hAll).t, kittler: kittler(hAll).t, dice: Math.max(0, pts[b].t) };
  // local threshold for bright objects: the local mean (51 × 51 window) plus 5 grey levels
  const T = localThreshold(img, "meanC", { r: 25, C: -5 });
  const localMask = Uint8Array.from(img.data, (v, i) => (v > T.data[i] ? 1 : 0));
  const at = (t: number) => pts[Math.max(0, Math.min(pts.length - 1, t + 1))];
  return { img, truth, pts, auc: auc(pts), marks, localMask, at };
}
