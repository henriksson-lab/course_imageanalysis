// A seeded synthetic population of segmented cells with measured features (no DOM), for the
// feature-importance demo. Two phenotypes, "interphase" (class 0) and "mitotic" (class 1),
// that differ strongly in texture, moderately in size, weakly in intensity and solidity.
// Area, perimeter and equivalent diameter carry (almost) the same information:
// equivalent diameter is an exact function of area, perimeter is area plus a little noise.
// Two noise features (one continuous, one binary) carry no information at all.
import { rng } from "../lib/ui";

export interface FeatureInfo { id: string; label: string; short: string }

export const FEATURES: FeatureInfo[] = [
  { id: "area", label: "area (µm²)", short: "area" },
  { id: "perim", label: "perimeter (µm)", short: "perimeter" },
  { id: "eqd", label: "equivalent diameter (µm)", short: "eq. diam." },
  { id: "int", label: "mean intensity", short: "intensity" },
  { id: "tex", label: "texture (local s.d.)", short: "texture" },
  { id: "sol", label: "solidity", short: "solidity" },
  { id: "noise", label: "noise (continuous)", short: "noise" },
  { id: "coin", label: "noise (binary)", short: "coin" },
];

export const CLASS_NAMES = ["interphase", "mitotic"];

export interface Population { n: number; cols: Float64Array[]; y: Int32Array }

/** n objects (half of each class), features in the order of FEATURES. */
export function population(o: { n?: number; seed?: number } = {}): Population {
  const n = o.n ?? 400, r = rng(o.seed ?? 1);
  const cols = FEATURES.map(() => new Float64Array(n)), y = new Int32Array(n);
  const g = (m: number, s: number) => m + s * r.norm();
  for (let i = 0; i < n; i++) {
    const c = i < n / 2 ? 0 : 1;
    y[i] = c;
    const area = Math.max(80, c ? g(230, 50) : g(300, 60));
    cols[0][i] = Math.round(area * 10) / 10;
    cols[1][i] = Math.round(2 * Math.sqrt(Math.PI * area) * (1.08 + 0.02 * r.norm()) * 10) / 10;
    cols[2][i] = 2 * Math.sqrt(cols[0][i] / Math.PI);
    cols[3][i] = Math.round(c ? g(112, 15) : g(100, 15));
    cols[4][i] = Math.round(Math.max(1, c ? g(14, 2.5) : g(8, 2.5)) * 10) / 10;
    cols[5][i] = Math.round(Math.min(1, c ? g(0.95, 0.02) : g(0.93, 0.025)) * 1000) / 1000;
    cols[6][i] = r.unif();
    cols[7][i] = r.unif() < 0.5 ? 1 : 0;
  }
  return { n, cols, y };
}
