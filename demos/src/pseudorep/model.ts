// Model of the pseudorep demo (no DOM): a two-level experiment (cells within experiments,
// two conditions) simulated by lib/stats.ts, analysed three ways (cells as n, experiment
// means as n, paired experiment means), and repeated many times from sufficient statistics
// (each experiment's mean and within-experiment sum of squares), which have exactly the
// distribution of the full simulation but cost O(E) per run instead of O(E·m).
import { Rng, rng, gauss } from "../lib/random";
import * as ST from "../lib/stats";

export interface Settings {
  E: number;            // experiments per condition (paired: experiments, each with both)
  m: number;            // cells per experiment and condition
  sdBetween: number;    // SD of the day-to-day effect u_e
  sdWithin: number;     // SD of cells within an experiment
  effect: number;       // true treatment effect
  design: ST.Design;
  batch: number;        // confounded design: shift of the batch that holds all treated experiments
}

export const DEFAULTS: Settings = { E: 3, m: 100, sdBetween: 1, sdWithin: 2, effect: 0, design: "independent", batch: 1 };

export const toOptions = (s: Settings): ST.TwoLevelOptions => ({ experiments: s.E, cells: s.m, sdBetween: s.sdBetween, sdWithin: s.sdWithin, effect: s.effect, design: s.design, batch: s.batch, mu: 10 });

/** One full simulation (every cell) and its three analyses. */
export function runOnce(s: Settings, seed: number): { data: ST.ExperimentData[]; res: ReturnType<typeof ST.analyseTwoLevel> } {
  const data = ST.simulateTwoLevel(rng(seed), toOptions(s));
  return { data, res: ST.analyseTwoLevel(data) };
}

/** Gamma(shape, 1) by Marsaglia–Tsang (shape ≥ 1; boosted below 1). */
export function gamma(r: Rng, shape: number): number {
  if (shape < 1) return gamma(r, shape + 1) * r.unif() ** (1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do { x = r.norm(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r.unif();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** χ² with k degrees of freedom. */
export const chi2 = (r: Rng, k: number): number => 2 * gamma(r, k / 2);

interface Group { means: number[]; ss: number[] }

function welch(m0: number, v0: number, n0: number, m1: number, v1: number, n1: number): number {
  const a = v0 / n0, b = v1 / n1, se = Math.sqrt(a + b), df = (a + b) ** 2 / (a * a / (n0 - 1) + b * b / (n1 - 1));
  return ST.tTwoSidedP((m1 - m0) / se, df);
}

/** p-values of the three analyses from experiment means and within sums of squares. */
function analyseSummaries(g0: Group, g1: Group, m: number, paired: boolean): { cells: number; means: number; paired: number } {
  const cellsStats = (g: Group) => {
    const E = g.means.length, N = E * m, mu = ST.mean(g.means);
    let ss = 0;
    for (let e = 0; e < E; e++) ss += g.ss[e] + m * (g.means[e] - mu) ** 2;
    return { mu, v: ss / (N - 1), N };
  };
  const c0 = cellsStats(g0), c1 = cellsStats(g1);
  const cells = welch(c0.mu, c0.v, c0.N, c1.mu, c1.v, c1.N);
  const means = welch(ST.mean(g0.means), ST.variance(g0.means), g0.means.length, ST.mean(g1.means), ST.variance(g1.means), g1.means.length);
  const pp = paired ? ST.tTestPaired(g0.means, g1.means).p : NaN;
  return { cells, means, paired: pp };
}

export interface Many { cells: number; means: number; paired: number; p: { cells: Float64Array; means: Float64Array; paired: Float64Array } }

/** R repetitions: fraction of p < 0.05 for each analysis (false-positive rate under no effect, power otherwise). */
export function runMany(s: Settings, R: number, seed = 1): Many {
  const r = rng(seed), E = s.E, m = s.m, sw = s.sdWithin;
  const pc = new Float64Array(R), pm = new Float64Array(R), pp = new Float64Array(R);
  const cellMean = (centre: number) => gauss(r, centre, sw / Math.sqrt(m));
  const ss = () => sw * sw * chi2(r, m - 1);
  for (let i = 0; i < R; i++) {
    const g0: Group = { means: [], ss: [] }, g1: Group = { means: [], ss: [] };
    if (s.design === "paired") {
      for (let e = 0; e < E; e++) {
        const u = gauss(r, 0, s.sdBetween);
        g0.means.push(cellMean(u)); g0.ss.push(ss());
        g1.means.push(cellMean(u + s.effect)); g1.ss.push(ss());
      }
    } else {
      const shift = s.design === "confounded" ? s.batch : 0;
      for (let e = 0; e < E; e++) { g0.means.push(cellMean(gauss(r, 0, s.sdBetween))); g0.ss.push(ss()); }
      for (let e = 0; e < E; e++) { g1.means.push(cellMean(gauss(r, 0, s.sdBetween) + s.effect + shift)); g1.ss.push(ss()); }
    }
    const a = analyseSummaries(g0, g1, m, s.design === "paired");
    pc[i] = a.cells; pm[i] = a.means; pp[i] = a.paired;
  }
  const frac = (p: Float64Array) => { let c = 0, n = 0; p.forEach(v => { if (Number.isFinite(v)) { n++; if (v < 0.05) c++; } }); return n ? c / n : NaN; };
  return { cells: frac(pc), means: frac(pm), paired: frac(pp), p: { cells: pc, means: pm, paired: pp } };
}

/** SE of the estimated effect with experiments as the unit, and what a cells-as-n analysis claims. */
export function seCurves(s: Settings, m: number): { true: number; claimed: number } {
  const vu = s.sdBetween ** 2, vw = s.sdWithin ** 2;
  const k = s.design === "paired" ? 0 : vu;   // pairing removes the day-to-day variance
  return { true: Math.sqrt(2 * (k + vw / m) / s.E), claimed: Math.sqrt(2 * (vu + vw) / (s.E * m)) };
}

export interface Preset { id: string; label: string; settings: Partial<Settings>; seed?: number }

export const PRESETS: Preset[] = [
  { id: "cells", label: "1000 cells, no effect", settings: { m: 1000, effect: 0 }, seed: 4 },
  { id: "more-cells", label: "more cells does not help", settings: { m: 1000, effect: 1 } },
  { id: "more-exp", label: "more experiments does", settings: { E: 10, m: 20, effect: 1 } },
  { id: "paired", label: "paired design wins", settings: { E: 3, m: 100, effect: 1, design: "paired" } },
  { id: "confounded", label: "confounded with the day", settings: { E: 3, m: 100, effect: 0, design: "confounded", batch: 1.5 } },
];
