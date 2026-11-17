// The computations of the activelearn demo (no DOM): a pool of unlabelled "cells" in a 2D
// feature space (two moons plus a small rare subtype of class 1 far from the boundary), a
// probabilistic classifier (logistic regression on Gaussian bumps, fitted by Newton's method),
// and the rules that choose which cell to label next.
import { rng } from "../lib/ui";
import { solve } from "../lib/linalg";

export interface Pt { x: number; y: number; c: number; rare: boolean; f?: Float64Array }

/** The features of a point, computed once and kept with it. */
export const phi = (p: Pt): Float64Array => p.f ?? (p.f = features(p.x, p.y));

/** n points: two noisy moons (classes 0 and 1) and a rare cluster of class 1 (6 % of the points). */
export function makePool(n: number, seed: number): Pt[] {
  const r = rng(seed), pts: Pt[] = [];
  const nRare = Math.round(0.06 * n);
  for (let i = 0; i < n - nRare; i++) {
    const c = i % 2, t = Math.PI * r.unif();
    const x = c === 0 ? Math.cos(t) : 1 - Math.cos(t), y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
    pts.push({ x: 0.1 + 0.8 * (x + 1) / 3 + 0.06 * r.norm(), y: 0.25 + 0.6 * (y + 0.5) / 1.5 + 0.06 * r.norm(), c, rare: false });
  }
  for (let i = 0; i < nRare; i++) pts.push({ x: 0.12 + 0.035 * r.norm(), y: 0.88 + 0.035 * r.norm(), c: 1, rare: true });
  // shuffle so that index order carries no information
  for (let i = pts.length - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); [pts[i], pts[j]] = [pts[j], pts[i]]; }
  return pts;
}

// ---- the classifier: p(class 1 | x) = sigmoid(w · φ(x)), φ = 1 and 7 × 7 Gaussian bumps ----
export const G = 7, SIGMA = 0.16, LAMBDA = 0.05;
export const NF = 1 + G * G;

export function features(x: number, y: number, out = new Float64Array(NF)): Float64Array {
  out[0] = 1;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const cx = (i + 0.5) / G, cy = (j + 0.5) / G;
    out[1 + i * G + j] = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * SIGMA * SIGMA));
  }
  return out;
}

const sigmoid = (t: number) => 1 / (1 + Math.exp(-t));

/** Weights of L2-regularised logistic regression (λ Σ w², not on the bias) by Newton's method. */
export function fit(pts: Pt[], idx: number[], w0?: Float64Array, iters = 4): Float64Array {
  const w = w0 ? w0.slice() : new Float64Array(NF);
  const Phi = idx.map(i => phi(pts[i]));
  for (let it = 0; it < iters; it++) {
    const g = new Float64Array(NF), H = new Float64Array(NF * NF);
    for (let a = 1; a < NF; a++) { g[a] = 2 * LAMBDA * w[a]; H[a * NF + a] = 2 * LAMBDA; }
    H[0] += 1e-6;
    Phi.forEach((f, k) => {
      let s = 0; for (let a = 0; a < NF; a++) s += w[a] * f[a];
      const p = sigmoid(s), e = p - pts[idx[k]].c, v = p * (1 - p);
      for (let a = 0; a < NF; a++) {
        g[a] += e * f[a];
        const va = v * f[a];
        for (let b = a; b < NF; b++) H[a * NF + b] += va * f[b];
      }
    });
    for (let a = 0; a < NF; a++) for (let b = 0; b < a; b++) H[a * NF + b] = H[b * NF + a];
    const d = solve(H, g, NF);
    if (!d) break;
    // damped Newton: halve the step until the penalised loss goes down
    const L0 = penalisedLoss(w, Phi, pts, idx);
    let t = 1, wn = w;
    for (let h = 0; h < 30; h++, t /= 2) {
      wn = w.map((v, a) => v - t * d[a]);
      if (penalisedLoss(wn, Phi, pts, idx) <= L0) break;
    }
    let step = 0;
    for (let a = 0; a < NF; a++) { step = Math.max(step, Math.abs(wn[a] - w[a])); w[a] = wn[a]; }
    if (step < 1e-6) break;
  }
  return w;
}

function penalisedLoss(w: Float64Array, Phi: Float64Array[], pts: Pt[], idx: number[]): number {
  let L = 0;
  Phi.forEach((f, k) => {
    let s = 0; for (let a = 0; a < NF; a++) s += w[a] * f[a];
    // −log p(c): softplus(s) − c s
    L += Math.max(s, 0) + Math.log1p(Math.exp(-Math.abs(s))) - pts[idx[k]].c * s;
  });
  for (let a = 1; a < NF; a++) L += LAMBDA * w[a] * w[a];
  return L;
}

export function prob(w: Float64Array, x: number, y: number, buf = new Float64Array(NF)): number {
  const f = features(x, y, buf);
  let s = 0; for (let a = 0; a < NF; a++) s += w[a] * f[a];
  return sigmoid(s);
}

/** p(class 1) of a point, with its cached features. */
export function probPt(w: Float64Array, p: Pt): number {
  const f = phi(p);
  let s = 0; for (let a = 0; a < NF; a++) s += w[a] * f[a];
  return sigmoid(s);
}

/** Accuracy on a test set, and the fraction of its rare cells classified correctly. */
export function score(w: Float64Array, test: Pt[]): { acc: number; rare: number } {
  let ok = 0, rareOk = 0, rareN = 0;
  for (const p of test) {
    const c = probPt(w, p) > 0.5 ? 1 : 0;
    if (c === p.c) ok++;
    if (p.rare) { rareN++; if (c === p.c) rareOk++; }
  }
  return { acc: ok / test.length, rare: rareN ? rareOk / rareN : NaN };
}

// ---- which cell to label next ----
export type Rule = "random" | "uncertain" | "diverse" | "both" | "mistakes";
export const RULES: Rule[] = ["random", "uncertain", "diverse", "both", "mistakes"];
export const RULE_LABEL: Record<Rule, string> = {
  random: "random",
  uncertain: "most uncertain",
  diverse: "most different",
  both: "uncertain + different",
  mistakes: "the model's mistakes",
};

/** The state of one labelling run: which pool cells are labelled, and the current model. */
export class Labeller {
  labelled: number[] = [];
  isLab: Uint8Array;
  dmin: Float64Array;          // distance of every pool cell to the nearest labelled cell
  p: Float64Array;             // current p(class 1) of every pool cell
  w: Float64Array = new Float64Array(NF);
  private r: ReturnType<typeof rng>;
  private turn = 0;
  private pending = -1;
  constructor(public pool: Pt[], start: number[], seed = 1) {
    this.isLab = new Uint8Array(pool.length);
    this.dmin = new Float64Array(pool.length).fill(Infinity);
    this.p = new Float64Array(pool.length).fill(0.5);
    this.r = rng(seed);
    for (const i of start) this.add(i, false);
    this.refit();
  }
  add(i: number, refit = true): void {
    if (this.isLab[i]) return;
    this.isLab[i] = 1; this.labelled.push(i);
    const q = this.pool[i];
    this.pool.forEach((p, j) => { const d = Math.hypot(p.x - q.x, p.y - q.y); if (d < this.dmin[j]) this.dmin[j] = d; });
    if (refit) this.refit();
  }
  refit(): void {
    this.w = fit(this.pool, this.labelled, this.w);
    this.pool.forEach((q, j) => { this.p[j] = probPt(this.w, q); });
  }
  /** The pool cell the rule proposes next (−1 if none is left). */
  propose(rule: Rule): number {
    let best = -1, bestV = -Infinity;
    const unl = this.pool.map((_, j) => j).filter(j => !this.isLab[j]);
    if (!unl.length) return -1;
    if (rule === "random") {
      // the same proposal until it is labelled (so the ringed cell is the one labelled next)
      if (this.pending < 0 || this.isLab[this.pending]) this.pending = unl[Math.floor(this.r.unif() * unl.length)];
      return this.pending;
    }
    const useDiverse = rule === "diverse" || (rule === "both" && this.turn % 2 === 1);
    if (rule === "mistakes") {
      // a person looks at the predictions and labels a cell the model gets wrong
      // (the most confidently wrong one first); if there is none, the most uncertain
      const wrong = unl.filter(j => (this.p[j] > 0.5 ? 1 : 0) !== this.pool[j].c);
      if (wrong.length) {
        for (const j of wrong) { const v = Math.abs(this.p[j] - 0.5); if (v > bestV) { bestV = v; best = j; } }
        return best;
      }
    }
    for (const j of unl) {
      const v = useDiverse ? this.dmin[j] : -Math.abs(this.p[j] - 0.5);
      if (v > bestV) { bestV = v; best = j; }
    }
    return best;
  }
  step(rule: Rule): number {
    const j = this.propose(rule);
    if (j >= 0) { this.add(j); this.turn++; }
    return j;
  }
}

/** Two cells of each class from the main moons (not the rare cluster), chosen by a seed. */
export function startSet(pool: Pt[], seed: number): number[] {
  const r = rng(seed), out: number[] = [];
  for (const c of [0, 1, 0, 1]) {
    let j: number;
    do { j = Math.floor(r.unif() * pool.length); } while (pool[j].c !== c || pool[j].rare || out.includes(j));
    out.push(j);
  }
  return out;
}

export interface Curve { n: number[]; acc: number[]; rare: number[] }

/** Mean learning curve of a rule: test accuracy against the number of labels, averaged over starts. */
export function learningCurve(pool: Pt[], test: Pt[], rule: Rule, maxN: number, starts: number[]): Curve {
  const n: number[] = [], acc: number[] = [], rare: number[] = [];
  for (let k = 4; k <= maxN; k++) { n.push(k); acc.push(0); rare.push(0); }
  for (const s of starts) {
    const L = new Labeller(pool, startSet(pool, s), s + 50);
    for (let t = 0; t < n.length; t++) {
      if (t > 0) L.step(rule);
      const sc = score(L.w, test);
      acc[t] += sc.acc / starts.length; rare[t] += sc.rare / starts.length;
    }
  }
  return { n, acc, rare };
}
