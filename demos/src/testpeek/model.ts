// Model of the testpeek demo (no DOM): k candidate methods (settings) with true accuracies a
// (all equal, or spread around a), each scored on a test set of n objects. Picking the best
// test score and reporting it is optimistic; re-testing the chosen method on fresh data shows
// by how much. With a validation set, the choice is made on validation scores and the test
// set is used once: no optimism.
import { rng, Rng, binomial } from "../lib/random";

export interface Settings {
  k: number;          // number of candidate methods / settings
  a: number;          // (mean) true accuracy
  spread: number;     // SD of the true accuracies around a
  n: number;          // objects in the test set (and in the validation set)
  validation: boolean;
}

export const DEFAULTS: Settings = { k: 20, a: 0.8, spread: 0, n: 100, validation: false };

/** Measured accuracy of a method with true accuracy p on n objects (binomial; normal approximation for large n). */
export function score(r: Rng, p: number, n: number): number {
  if (n <= 60) return binomial(r, n, p) / n;
  const z = r.norm(), c = Math.round(n * p + Math.sqrt(n * p * (1 - p)) * z);
  return Math.min(n, Math.max(0, c)) / n;
}

export interface Run {
  trueAcc: number[];
  test: number[];
  val: number[] | null;
  best: number;        // index of the chosen method
  reported: number;    // the number that goes in the paper
  fresh: number;       // the chosen method on a fresh test set
}

export function runOnce(s: Settings, r: Rng): Run {
  const trueAcc = Array.from({ length: s.k }, () => Math.min(0.999, Math.max(0.001, s.a + s.spread * r.norm())));
  const test = trueAcc.map(p => score(r, p, s.n));
  const val = s.validation ? trueAcc.map(p => score(r, p, s.n)) : null;
  const choose = val ?? test;
  let best = 0;
  choose.forEach((v, j) => { if (v > choose[best]) best = j; });
  return { trueAcc, test, val, best, reported: test[best], fresh: score(r, trueAcc[best], s.n) };
}

export interface Many { reported: Float64Array; fresh: Float64Array; truth: Float64Array; optimism: number; meanReported: number; meanFresh: number }

/** R repetitions: the reported score, the fresh re-test and the true accuracy of the chosen method. */
export function runMany(s: Settings, R: number, seed = 1): Many {
  const r = rng(seed), reported = new Float64Array(R), fresh = new Float64Array(R), truth = new Float64Array(R);
  for (let i = 0; i < R; i++) { const o = runOnce(s, r); reported[i] = o.reported; fresh[i] = o.fresh; truth[i] = o.trueAcc[o.best]; }
  const m = (x: Float64Array) => x.reduce((a, b) => a + b, 0) / x.length;
  return { reported, fresh, truth, optimism: m(reported) - m(fresh), meanReported: m(reported), meanFresh: m(fresh) };
}

export interface Preset { id: string; label: string; settings: Partial<Settings> }

export const PRESETS: Preset[] = [
  { id: "one", label: "one method: honest", settings: { k: 1 } },
  { id: "fifty", label: "50 settings, 100 objects", settings: { k: 50, n: 100 } },
  { id: "bigger", label: "bigger test set", settings: { k: 50, n: 1000 } },
  { id: "valid", label: "validation set fixes it", settings: { k: 50, n: 100, validation: true } },
  { id: "spread", label: "real differences", settings: { k: 20, n: 100, spread: 0.03 } },
];
