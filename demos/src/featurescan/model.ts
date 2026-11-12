// Model of the featurescan demo (no DOM): a well plate (rows × columns), m features per well,
// a fraction of which carries a real treatment effect; plate effects (row and column
// gradients, edge wells) that every feature feels with its own sensitivity; three layouts
// (treated on the left half, randomised, randomised within every row). Every feature is
// tested treated vs control (Welch t-test over wells); hits uncorrected, Bonferroni and
// Benjamini–Hochberg, with the true and false hits known from the simulation.
import { rng } from "../lib/random";
import * as ST from "../lib/stats";

export interface Settings extends Omit<ST.PlateOptions, "rows" | "cols"> { big: boolean; alpha: number }

export const DEFAULTS: Settings = { big: true, features: 300, fracReal: 0, effect: 0.35, rowGradient: 0, colGradient: 0, edge: 0, layout: "random", alpha: 0.05 };

/** Seed of the plate shown by the demo (its "new plate" button counts up from here). */
export const SEED = 3;

export function makePlate(s: Settings, seed = SEED): ST.Plate {
  return ST.simulatePlate(rng(seed), { ...s, rows: s.big ? 16 : 8, cols: s.big ? 24 : 12 });
}

export interface Analysis {
  p: Float64Array;
  raw: number; bonf: number; bh: number;               // number of hits
  falseRaw: number; falseBonf: number; falseBH: number; // of which not real
  trueRaw: number; trueBonf: number; trueBH: number;
  nReal: number;
  bhThreshold: number;
  rejRaw: boolean[]; rejBonf: boolean[]; rejBH: boolean[];
}

export function analyse(pl: ST.Plate, alpha = 0.05): Analysis {
  const p = Float64Array.from(pl.values, v => {
    const a: number[] = [], b: number[] = [];
    v.forEach((x, i) => (pl.cond[i] ? b : a).push(x));
    return ST.tTestWelch(a, b).p;
  });
  const rejRaw = Array.from(p, v => v <= alpha), rejBonf = ST.bonferroni(p, alpha), bh = ST.benjaminiHochberg(p, alpha);
  const count = (r: boolean[], real?: boolean) => r.filter((x, i) => x && (real === undefined || pl.real[i] === real)).length;
  return {
    p, rejRaw, rejBonf, rejBH: bh.reject, bhThreshold: bh.threshold,
    raw: count(rejRaw), bonf: count(rejBonf), bh: count(bh.reject),
    falseRaw: count(rejRaw, false), falseBonf: count(rejBonf, false), falseBH: count(bh.reject, false),
    trueRaw: count(rejRaw, true), trueBonf: count(rejBonf, true), trueBH: count(bh.reject, true),
    nReal: pl.real.filter(Boolean).length,
  };
}

export interface Preset { id: string; label: string; settings: Partial<Settings> }

export const PRESETS: Preset[] = [
  { id: "null", label: "300 null features", settings: {} },
  { id: "real", label: "10 % real: BH finds most", settings: { fracReal: 0.1 } },
  { id: "confounded", label: "gradient + treated on the left", settings: { colGradient: 1.5, layout: "left" } },
  { id: "randomised", label: "randomised layout fixes it", settings: { colGradient: 1.5, layout: "random" } },
  { id: "edge", label: "edge wells, row-blocked layout", settings: { edge: 1, rowGradient: 1, layout: "blocked", fracReal: 0.1 } },
];
