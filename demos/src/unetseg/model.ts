// Evaluation helpers of the unetseg demo (no DOM): masks from a probability map, the Otsu
// baseline (lib/threshold.ts), and Dice / IoU against the true mask (lib/metrics.ts).
import { otsu } from "../lib/threshold";
import { confusion, dice, jaccard, type Counts } from "../lib/metrics";

export function maskAt(prob: ArrayLike<number>, t: number): Uint8Array {
  return Uint8Array.from(prob as ArrayLike<number>, v => (v > t ? 1 : 0));
}

/** Otsu's threshold on the grey levels (rounded and clipped to 0..255); mask = I > t. */
export function otsuMask(img: ArrayLike<number>): { mask: Uint8Array; t: number } {
  const counts = new Float64Array(256);
  for (let i = 0; i < img.length; i++) counts[Math.max(0, Math.min(255, Math.round(img[i])))]++;
  const t = otsu(counts).t;
  return { mask: Uint8Array.from(img as ArrayLike<number>, v => (Math.round(v) > t ? 1 : 0)), t };
}

export function scores(pred: ArrayLike<number>, truth: ArrayLike<number>): { c: Counts; dice: number; iou: number } {
  const c = confusion(pred, truth);
  return { c, dice: dice(c), iou: jaccard(c) };
}

/** Per-pixel binary cross-entropy (mean) and soft Dice of a probability map against a mask. */
export function losses(prob: ArrayLike<number>, truth: ArrayLike<number>): { ce: number; softDice: number } {
  let ce = 0, pg = 0, ps = 0, gs = 0;
  for (let i = 0; i < prob.length; i++) {
    const p = Math.min(1 - 1e-7, Math.max(1e-7, prob[i])), g = truth[i];
    ce -= g * Math.log(p) + (1 - g) * Math.log(1 - p);
    pg += p * g; ps += p; gs += g;
  }
  return { ce: ce / prob.length, softDice: (2 * pg) / (ps + gs || 1) };
}
