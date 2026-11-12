// Model of the calibration demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// A simulated two-class problem: a feature x ~ N(+μ, 1) for class 1 and N(−μ, 1) for
// class 0 (equal priors), so the true probability of class 1 is σ(2μx). The "network"
// outputs the logit z = s·2μx: s = 1 is calibrated, s > 1 overconfident. Temperature
// scaling divides z by T (fitted on validation data by minimising the negative
// log-likelihood). Domain shift moves the test classes towards each other (μ → (1 − shift)μ)
// after T was fitted: the same logits, but now overconfident again.
import { rng } from "../lib/ui";

export interface Sample { z: number; y: number }      // logit of the network, true class

export const MU = 1.2;

/** n seeded samples (half per class) with network sharpness s and domain shift. */
export function simulate(n: number, s: number, shift: number, seed: number): Sample[] {
  const r = rng(seed), out: Sample[] = [], mu = MU * (1 - shift);
  for (let i = 0; i < n; i++) {
    const y = i % 2, x = (y ? mu : -mu) + r.norm();
    out.push({ z: s * 2 * MU * x, y });
  }
  return out;
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/** Predicted class, its confidence (the larger of p and 1 − p) and correctness, at temperature T. */
export function predictions(d: Sample[], T: number): { conf: number; correct: number; p: number }[] {
  return d.map(({ z, y }) => {
    const p = sigmoid(z / T), cls = p >= 0.5 ? 1 : 0;
    return { p, conf: Math.max(p, 1 - p), correct: cls === y ? 1 : 0 };
  });
}

export interface Reliability {
  edges: number[]; count: number[]; acc: number[]; conf: number[];
  ece: number; accuracy: number; nll: number;
}

/** Reliability diagram over confidence bins [0.5, 1] (nBins equal bins) and ECE = Σ n_b/n |acc_b − conf_b|. */
export function reliability(pr: { conf: number; correct: number }[], nBins = 5, lo = 0.5): Reliability {
  const edges = Array.from({ length: nBins + 1 }, (_, k) => lo + (1 - lo) * k / nBins);
  const count = new Array(nBins).fill(0), sa = new Array(nBins).fill(0), sc = new Array(nBins).fill(0);
  let acc = 0;
  for (const { conf, correct } of pr) {
    const b = Math.min(nBins - 1, Math.max(0, Math.floor((conf - lo) / (1 - lo) * nBins)));
    count[b]++; sa[b] += correct; sc[b] += conf; acc += correct;
  }
  let ece = 0;
  const A = count.map((c, b) => (c ? sa[b] / c : NaN)), C = count.map((c, b) => (c ? sc[b] / c : NaN));
  count.forEach((c, b) => { if (c) ece += c / pr.length * Math.abs(A[b] - C[b]); });
  return { edges, count, acc: A, conf: C, ece, accuracy: acc / pr.length, nll: NaN };
}

/** Mean negative log-likelihood of the true classes at temperature T. */
export function nll(d: Sample[], T: number): number {
  let s = 0;
  for (const { z, y } of d) { const p = sigmoid(z / T); s -= Math.log(Math.max(1e-12, y ? p : 1 - p)); }
  return s / d.length;
}

/** The temperature minimising the NLL (golden-section search on log T in [0.05, 20]). */
export function fitTemperature(d: Sample[]): number {
  let a = Math.log(0.05), b = Math.log(20);
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), e = a + g * (b - a);
  for (let it = 0; it < 80; it++) {
    if (nll(d, Math.exp(c)) < nll(d, Math.exp(e))) b = e; else a = c;
    c = b - g * (b - a); e = a + g * (b - a);
  }
  return Math.exp((a + b) / 2);
}
