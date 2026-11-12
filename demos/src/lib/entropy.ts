// Information and entropy of symbol histograms, run-length coding (no DOM).
// Owner: the compression day (entropy/ unit). The histogram itself comes from
// lib/histogram.ts (`intHistogram`: signed and 16-bit values fine).
//
//   information of an outcome with probability p:  I = −log2 p   (bits)
//   entropy of a histogram:  H = −Σ p_k log2 p_k = E[I]          (bits per symbol)
//
// H is the lower bound (Shannon) on the average length of a lossless code for independent
// symbols with these probabilities; `idealBits` = H · N is that bound for N symbols.
import { intHistogram } from "./histogram";

/** Information of an outcome with probability p, in bits (Infinity for p = 0). */
export const information = (p: number): number => (p > 0 ? -Math.log2(p) : Infinity);

/** Probabilities from counts (zeros stay zero). */
export function probabilities(counts: ArrayLike<number>): Float64Array {
  let s = 0;
  for (let i = 0; i < counts.length; i++) s += counts[i];
  const p = new Float64Array(counts.length);
  if (s > 0) for (let i = 0; i < counts.length; i++) p[i] = counts[i] / s;
  return p;
}

/** Entropy in bits of a histogram given as counts (or probabilities); empty bins ignored. */
export function entropyFromCounts(counts: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < counts.length; i++) s += counts[i];
  if (s <= 0) return 0;
  let H = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i];
    if (c > 0) { const p = c / s; H -= p * Math.log2(p); }
  }
  return H;
}

/** Entropy (bits per symbol) of the histogram of integer values (rounded); any range. */
export function entropy(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  return entropyFromCounts(intHistogram(values).counts);
}

/** The Shannon bound for coding these values one by one: H · N bits. */
export function idealBits(values: ArrayLike<number>): number {
  return entropy(values) * values.length;
}

/** Fixed-length code: ⌈log2 K⌉ bits per symbol for K distinct symbols (0 for K ≤ 1). */
export const fixedLength = (K: number): number => (K <= 1 ? 0 : Math.ceil(Math.log2(K)));

/** Number of distinct (rounded) values. */
export function distinct(values: ArrayLike<number>): number {
  const s = new Set<number>();
  for (let i = 0; i < values.length; i++) s.add(Math.round(values[i]));
  return s.size;
}

export interface Run { value: number; run: number }

/** Run-length encoding: maximal runs of equal values as (value, run) pairs. */
export function rleEncode(values: ArrayLike<number>): Run[] {
  const out: Run[] = [];
  for (let i = 0; i < values.length;) {
    let j = i + 1;
    while (j < values.length && values[j] === values[i]) j++;
    out.push({ value: values[i], run: j - i });
    i = j;
  }
  return out;
}

/** Inverse of rleEncode. */
export function rleDecode(runs: Run[]): Float64Array {
  let n = 0;
  for (const r of runs) n += r.run;
  const out = new Float64Array(n);
  let i = 0;
  for (const r of runs) { out.fill(r.value, i, i + r.run); i += r.run; }
  return out;
}

/**
 * Run lengths of a binary row read as alternating runs starting with 0 (fax style): the
 * first run may have length 0 when the row starts with 1. Decode with `binaryRunsDecode`.
 */
export function binaryRuns(bits: ArrayLike<number>): number[] {
  const runs: number[] = [];
  let cur = 0, len = 0;
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i] ? 1 : 0;
    if (b === cur) len++;
    else { runs.push(len); cur = b; len = 1; }
  }
  runs.push(len);
  return runs;
}

/** Inverse of binaryRuns. */
export function binaryRunsDecode(runs: number[]): Uint8Array {
  const out = new Uint8Array(runs.reduce((a, b) => a + b, 0));
  let i = 0, b = 0;
  for (const r of runs) { out.fill(b, i, i + r); i += r; b ^= 1; }
  return out;
}

/** Shuffle-invariance helper: entropy of the histogram and of the empirical pairs (x_i, x_{i+1}). */
export function pairEntropy(values: ArrayLike<number>): number {
  const m = new Map<string, number>();
  for (let i = 0; i + 1 < values.length; i++) {
    const k = Math.round(values[i]) + "," + Math.round(values[i + 1]);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return entropyFromCounts([...m.values()]);
}
