// Model of the mrfdenoise demo (no DOM, so the unit's numbers can be computed in node from
// this code). A 128 x 128 binary image x (1 = object), observed with flip noise (each pixel
// flipped with probability eps) or Gaussian noise of sd sigma_n on the grey levels 0 and 1.
// Posterior energy in segmentation form (as on the graph-cut day):
//     E(L) = sum_p D_p(L_p) + lambda * #(neighbour pairs with L_p != L_q)
// with D_p = -log p(y_p | L_p) up to a constant. In Ising form J = lambda / 2, h_p = (D_p(0) - D_p(1)) / 2.
// Methods: threshold (lambda = 0), ICM, Gibbs sampling at T = 1 (posterior marginals -> MPM),
// simulated annealing (Gibbs sweeps with geometric cooling, then ICM), mean field (T = 1), and
// the exact MAP by one minimum cut (lib/mrf.ts exactMAP -> lib/graph.ts solveSeg).
import { rng, Rng } from "../lib/random";
import { rasteriseEllipses } from "../lib/synth";
import { fromArray, gaussianBlur } from "../lib/imageops";
import {
  lattice, Lattice, flipCosts, gaussLevelCosts, fieldFromCosts, segmentationEnergy, icmSweep, gibbsSweep,
  meanFieldSweep, exactMAP, geometricT,
} from "../lib/mrf";

export const N = 128;
/** A Gibbs (MPM) run stops after this many sweeps. */
export const GIBBS_SWEEPS = 300;

export type ImageKind = "shapes" | "cell";
export type NoiseKind = "flip" | "gauss";
export type Method = "threshold" | "icm" | "gibbs" | "anneal" | "meanfield" | "exact";
export const METHODS: Method[] = ["threshold", "icm", "gibbs", "anneal", "meanfield", "exact"];
export const METHOD_LABEL: Record<Method, string> = {
  threshold: "threshold", icm: "ICM", gibbs: "Gibbs (MPM)", anneal: "annealing", meanfield: "mean field", exact: "exact MAP",
};

/** Synthetic test image: a disc, a square with sharp corners, a 1-pixel line and a 3-pixel line, a small dot. */
export function shapesTruth(): Uint8Array {
  const t = new Uint8Array(N * N);
  const disc = rasteriseEllipses(N, N, [{ x: 38, y: 40, a: 24, b: 24, angle: 0 }]);
  for (let p = 0; p < N * N; p++) if (disc[p]) t[p] = 1;
  for (let y = 18; y < 62; y++) for (let x = 74; x < 118; x++) t[y * N + x] = 1;   // square
  for (let x = 12; x < 116; x++) t[84 * N + x] = 1;                                 // 1-px line
  for (let y = 96; y < 99; y++) for (let x = 12; x < 116; x++) t[y * N + x] = 1;   // 3-px line
  for (let y = 108; y < 120; y++) for (let x = 56; x < 68; x++) t[y * N + x] = 1;  // small square
  return t;
}

/** Binary "truth" from the micrograph (grey 0..255, 128 x 128): dark structures = 1, after a
 *  light Gaussian blur (sigma = 1.5) so that the truth itself has no single-pixel specks. */
export function cellTruth(grey: ArrayLike<number>, threshold = 100): Uint8Array {
  const b = gaussianBlur(fromArray(N, N, grey), 1.5).data;
  return Uint8Array.from({ length: N * N }, (_, p) => (b[p] < threshold ? 1 : 0));
}

export interface Noise { kind: NoiseKind; eps: number; sd: number; seed: number }

/** Observation y: flipped labels (0/1) or grey values truth + Gaussian noise. */
export function observe(truth: Uint8Array, nz: Noise): Float64Array {
  const r = rng(nz.seed), y = new Float64Array(truth.length);
  for (let p = 0; p < y.length; p++) {
    y[p] = nz.kind === "flip" ? (r.unif() < nz.eps ? 1 - truth[p] : truth[p]) : truth[p] + nz.sd * r.norm();
  }
  return y;
}

/** Data costs D_p(0), D_p(1) (-log likelihood up to a constant). */
export function costs(y: Float64Array, nz: Noise): { D0: Float64Array; D1: Float64Array } {
  return nz.kind === "flip" ? flipCosts(y, nz.eps) : gaussLevelCosts(y, 0, 1, nz.sd);
}

export interface Problem {
  L: Lattice; truth: Uint8Array; y: Float64Array; D0: Float64Array; D1: Float64Array;
  lambda: number; J: number; h: Float64Array;
  /** exact MAP (spins) and its segmentation energy */
  map: Int8Array; Emin: number;
}

export function makeProblem(truth: Uint8Array, nz: Noise, lambda: number): Problem {
  const L = lattice(N, N, false), y = observe(truth, nz), { D0, D1 } = costs(y, nz);
  const h = fieldFromCosts(D0, D1), J = lambda / 2;
  const map = exactMAP(N, N, J, h).s;
  return { L, truth, y, D0, D1, lambda, J, h, map, Emin: segmentationEnergy(L, map, D0, D1, lambda).total };
}

/** Thresholded observation: each pixel takes its cheaper label (ties: background). */
export function thresholdSpins(P: Problem): Int8Array {
  return Int8Array.from({ length: P.L.n }, (_, p) => (P.D1[p] < P.D0[p] ? 1 : -1));
}

export const energyOf = (P: Problem, s: ArrayLike<number>) => segmentationEnergy(P.L, s, P.D0, P.D1, P.lambda);

/** Number of pixels where the estimate differs from the truth. */
export function wrong(P: Problem, s: ArrayLike<number>): number {
  let c = 0;
  for (let p = 0; p < P.L.n; p++) if ((s[p] > 0 ? 1 : 0) !== P.truth[p]) c++;
  return c;
}

export interface RunOptions { T0: number; coolSweeps: number; burnIn: number; seed: number }
export const DEFAULT_RUN: RunOptions = { T0: 4, coolSweeps: 300, burnIn: 20, seed: 1 };

/** One run of a method, advanced sweep by sweep. `estimate` is the current binary result. */
export class Run {
  readonly method: Method;
  readonly P: Problem;
  readonly o: RunOptions;
  s: Int8Array;              // current state (sample, or ICM / annealing state)
  estimate: Int8Array;       // the method's answer so far
  marginal: Float64Array | null = null;   // p(L_p = 1): Gibbs average, or (1 + m)/2 for mean field
  sweeps = 0;
  done = false;
  T = 1;
  hist: { k: number[]; E: number[] } = { k: [], E: [] };
  private r: Rng;
  private plus: Float64Array | null = null;
  private samples = 0;
  private m: Float64Array | null = null;

  constructor(method: Method, P: Problem, o: RunOptions = DEFAULT_RUN) {
    this.method = method; this.P = P; this.o = o;
    this.r = rng(o.seed);
    this.s = thresholdSpins(P);
    this.estimate = this.s;
    if (method === "threshold") this.done = true;
    if (method === "exact") { this.s = Int8Array.from(P.map); this.estimate = this.s; this.done = true; }
    if (method === "gibbs") { this.plus = new Float64Array(P.L.n); this.marginal = new Float64Array(P.L.n).fill(0.5); this.estimate = Int8Array.from(this.s); }
    if (method === "meanfield") {
      this.m = Float64Array.from(P.h, v => Math.tanh(v));
      this.marginal = Float64Array.from(this.m, v => (1 + v) / 2);
      this.estimate = Int8Array.from(this.m, v => (v >= 0 ? 1 : -1));
    }
    if (method === "anneal") this.T = o.T0;
    this.record();
  }

  get energy(): number { return energyOf(this.P, this.estimate).total; }

  private record(): void {
    this.hist.k.push(this.sweeps);
    this.hist.E.push(energyOf(this.P, this.method === "gibbs" ? this.s : this.estimate).total);
  }

  /** Run to the end (or maxSweeps). */
  finish(maxSweeps = 5000): void { while (!this.done && this.sweeps < maxSweeps) this.step(); }

  /** One sweep (no-op when done). */
  step(): void {
    if (this.done) return;
    const { P } = this, { L, J, h } = P;
    this.sweeps++;
    switch (this.method) {
      case "icm":
        if (icmSweep(L, this.s, J, h) === 0) this.done = true;
        this.estimate = this.s;
        break;
      case "anneal": {
        if (this.T > 0.05) {
          this.T = geometricT(this.o.T0, (0.05 / this.o.T0) ** (1 / this.o.coolSweeps), this.sweeps);
          gibbsSweep(L, this.s, J, h, this.T, this.r);
        } else {
          this.T = 0;
          if (icmSweep(L, this.s, J, h) === 0) this.done = true;
        }
        this.estimate = this.s;
        break;
      }
      case "gibbs": {
        gibbsSweep(L, this.s, J, h, 1, this.r);
        if (this.sweeps > this.o.burnIn) {
          this.samples++;
          const pl = this.plus!, mg = this.marginal!, est = this.estimate;
          for (let p = 0; p < L.n; p++) {
            if (this.s[p] > 0) pl[p]++;
            mg[p] = pl[p] / this.samples;
            est[p] = mg[p] > 0.5 ? 1 : -1;
          }
        } else this.estimate.set(this.s);
        if (this.sweeps >= GIBBS_SWEEPS) this.done = true;
        break;
      }
      case "meanfield": {
        const d = meanFieldSweep(L, this.m!, J, h, 1);
        const mg = this.marginal!, est = this.estimate;
        for (let p = 0; p < L.n; p++) { mg[p] = (1 + this.m![p]) / 2; est[p] = this.m![p] >= 0 ? 1 : -1; }
        if (d < 1e-6) this.done = true;
        break;
      }
    }
    this.record();
  }
}

export interface Preset {
  id: string; label: string;
  image: ImageKind; noise: Noise; lambda: number; method: Method; run: RunOptions;
  /** methods run to completion (Gibbs: GIBBS_SWEEPS) before showing `method`, so the energy plot compares them */
  also?: Method[];
}

const flip = (eps: number): Noise => ({ kind: "flip", eps, sd: 0.6, seed: 1 });
export const PRESETS: Preset[] = [
  { id: "icm", label: "ICM: fast, stuck", image: "shapes", noise: flip(0.2), lambda: 0.8, method: "icm", run: DEFAULT_RUN, also: ["exact"] },
  { id: "slow", label: "slow cooling", image: "shapes", noise: flip(0.2), lambda: 0.8, method: "anneal", run: { ...DEFAULT_RUN, coolSweeps: 1000 }, also: ["icm", "exact"] },
  { id: "quench", label: "quench", image: "shapes", noise: flip(0.2), lambda: 0.8, method: "anneal", run: { ...DEFAULT_RUN, coolSweeps: 10 }, also: ["icm", "exact"] },
  { id: "mpm", label: "MAP vs MPM", image: "shapes", noise: flip(0.2), lambda: 1.2, method: "gibbs", run: DEFAULT_RUN, also: ["exact"] },
  { id: "large", label: "λ too large", image: "shapes", noise: flip(0.2), lambda: 3, method: "exact", run: DEFAULT_RUN },
  { id: "zero", label: "λ = 0 is thresholding", image: "shapes", noise: flip(0.2), lambda: 0, method: "exact", run: DEFAULT_RUN, also: ["threshold"] },
  { id: "notbest", label: "exact MAP not the best image", image: "shapes", noise: flip(0.2), lambda: 1.6, method: "meanfield", run: DEFAULT_RUN, also: ["exact", "icm"] },
  { id: "gauss", label: "Gaussian noise, micrograph", image: "cell", noise: { kind: "gauss", eps: 0.2, sd: 0.8, seed: 1 }, lambda: 1, method: "exact", run: DEFAULT_RUN, also: ["threshold"] },
];
