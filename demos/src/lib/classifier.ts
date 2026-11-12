// The common classifier interface (owner: random-forest day, day 14). No DOM.
//
// A Classifier trains on a feature table and returns a Model that scores new samples. The
// pixel-classification shell (demos/src/pixelclass/) and other demos that switch between
// classifiers (forestplay, momentclass, ...) only talk to this interface, so other days plug
// in their own classifier without touching the shell's logic:
//
//   import { registerClassifier, Classifier } from "../lib/classifier";
//   export const SVM: Classifier = {
//     id: "svm", name: "SVM (RBF)",
//     params: [{ key: "C", label: "C", min: -2, max: 3, step: 0.1, value: 0, log: true }, ...],
//     train(data, p, seed) { ...; return { k: data.k, d: data.d, predict(X, n, out) { ... } }; },
//   };
//   registerClassifier(SVM);
//
// Conventions:
//  - X is FEATURE-MAJOR: X[j * n + i] is feature j of sample i (n samples, d features); the
//    same layout as a pixel feature stack (lib/featurestack.ts), so a whole image can be
//    scored without copying.
//  - y holds classes 0..k-1. `w` (optional) are per-sample weights (class balancing); a
//    classifier that cannot use weights may resample instead. `groups` (optional) gives a
//    group id per sample (a painted stroke, a region, an image) for by-group validation.
//  - predict() returns n × k scores, row-major (out[i * k + c]), each row in [0, 1] summing
//    to 1: class probabilities (vote fractions for a forest; for a margin classifier e.g. a
//    logistic map of the decision value). The shell uses argmax as the label and
//    1 − max as the uncertainty.
//  - Parameters are numbers (booleans as 0/1); a param with `log: true` holds log10 of the
//    value on its slider, and `paramValue()` converts it.

export interface TrainingSet {
  /** Feature-major features: X[j * n + i]. */
  X: Float32Array | Float64Array;
  n: number;
  d: number;
  /** Class of each sample, 0..k-1. */
  y: Int32Array;
  k: number;
  /** Optional per-sample weights (default 1). */
  w?: Float64Array;
  /** Optional group id per sample (stroke, region, image). */
  groups?: Int32Array;
  /** Optional feature names. */
  names?: string[];
}

export interface ParamSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Default (slider units). */
  value: number;
  /** The slider holds log10 of the value. */
  log?: boolean;
  /** Upper bound depends on the number of features (e.g. m ≤ d): clamp to d. */
  maxIsD?: boolean;
}

export interface Model {
  k: number;
  d: number;
  /** Scores of n samples given feature-major X (n × k, rows sum to 1). `from`/`to` limit the samples scored. */
  predict(X: ArrayLike<number>, n: number, out?: Float32Array, from?: number, to?: number): Float32Array;
  /** Out-of-bag (or other built-in) error estimate, if the method has one. */
  oobError?(): number | null;
  /** Importance of each feature, if the method has one. */
  importance?(): Float64Array | null;
  /** One line for a readout (e.g. "50 trees, mean depth 11"). */
  summary?(): string;
}

export interface Classifier {
  id: string;
  name: string;
  /** One sentence for the demo's note line. */
  description?: string;
  params: ParamSpec[];
  /** Train on the data with parameter values (slider units, see ParamSpec.log) and a seed. */
  train(data: TrainingSet, params: Record<string, number>, seed: number): Model;
}

const REGISTRY = new Map<string, Classifier>();

/** Add (or replace) a classifier under its id. */
export function registerClassifier(c: Classifier): void { REGISTRY.set(c.id, c); }

/** The classifier with this id, or undefined. */
export function getClassifier(id: string): Classifier | undefined { return REGISTRY.get(id); }

/** All registered classifiers, in registration order. */
export function listClassifiers(): Classifier[] { return [...REGISTRY.values()]; }

/** Default parameter values of a classifier (slider units). */
export function defaultParams(c: Classifier): Record<string, number> {
  const p: Record<string, number> = {};
  for (const s of c.params) p[s.key] = s.value;
  return p;
}

/** The actual value of a parameter (undoes the log10 of log sliders). */
export function paramValue(c: Classifier, params: Record<string, number>, key: string): number {
  const s = c.params.find(q => q.key === key);
  const v = params[key] ?? s?.value ?? 0;
  return s?.log ? Math.pow(10, v) : v;
}

/** Labels (argmax) and uncertainty (1 − max score) from n × k scores. */
export function labelsAndUncertainty(P: ArrayLike<number>, n: number, k: number): { label: Uint8Array; uncertainty: Float32Array } {
  const label = new Uint8Array(n), unc = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0, bv = P[i * k];
    for (let c = 1; c < k; c++) if (P[i * k + c] > bv) { bv = P[i * k + c]; best = c; }
    label[i] = best; unc[i] = 1 - bv;
  }
  return { label, uncertainty: unc };
}
