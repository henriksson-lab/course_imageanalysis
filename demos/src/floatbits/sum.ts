// Summing many pixel values in low precision: the data and the table of prefix sums used
// by the floatbits demo (view "sum") and by the tests.
import { rng } from "../lib/ui";
import { F32, F16, rounder, sumPairwise } from "./fp";

export const METHOD_LABELS = [
  "float32 naive", "float32 pairwise", "float32 Kahan", "float16 naive (÷ 65535)", "float16 pairwise (÷ 65535)",
];

/** Simulated pixel values: round(mean + sd * z), clipped to 0..65535, fixed seed. */
export function pixels(n: number, mean: number, sd: number, seed = 21): Uint16Array {
  const r = rng(seed);
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.max(0, Math.min(65535, Math.round(mean + sd * r.norm())));
  return out;
}

/** Sums of the first 2^k values (k = kmin, kmin+1, ... while 2^k <= length) by every
 *  method in METHOD_LABELS, and the exact sum (double; exact, since every partial sum of
 *  integers stays below 2^53). Naive and Kahan come from one running pass. */
export function sumTable(px: Uint16Array, kmin = 4) {
  const r32 = rounder(F32), r16 = rounder(F16);
  const scaled = new Float64Array(px.length);
  for (let i = 0; i < px.length; i++) scaled[i] = r16(px[i] / 65535);
  const ks: number[] = [], exact: number[] = [], res: number[][] = METHOD_LABELS.map(() => []);
  let ex = 0, n32 = 0, n16 = 0, ks32 = 0, kc = 0, next = 2 ** kmin;
  for (let i = 0; i < px.length; i++) {
    const x = px[i];
    ex += x;
    n32 = r32(n32 + x);
    n16 = r16(n16 + scaled[i]);
    const y = r32(x - kc), t = r32(ks32 + y); kc = r32(r32(t - ks32) - y); ks32 = t;
    if (i + 1 === next) {
      const n = i + 1;
      ks.push(Math.log2(n)); exact.push(ex);
      res[0].push(n32);
      res[1].push(sumPairwise(px, 0, n, r32));
      res[2].push(ks32);
      res[3].push(n16 * 65535);
      res[4].push(sumPairwise(scaled, 0, n, r16) * 65535);
      next *= 2;
    }
  }
  return { ks, exact, res };
}
