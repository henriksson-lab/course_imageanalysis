// Feature maps and presets of the svmlift demo (no DOM; also used by the solution scripts).
import { paraboloidMap, quadMap } from "../lib/svmkernels";
import { rings, xor, blobs, LPoint } from "../lib/datasets2d";
import { trainSVM, SVMModel } from "../lib/svm";

export type MapId = "paraboloid" | "saddle" | "quadratic";
export const MAPS: Record<MapId, { label: string; f: (x: number, y: number) => [number, number, number]; axes: string }> = {
  paraboloid: { label: "x₁² + x₂² (paraboloid)", f: paraboloidMap, axes: "(x₁, x₂, x₁² + x₂²)" },
  saddle: { label: "x₁x₂ (saddle)", f: (x, y) => [x, y, x * y], axes: "(x₁, x₂, x₁x₂)" },
  quadratic: { label: "(x₁², √2 x₁x₂, x₂²)", f: quadMap, axes: "(x₁², √2 x₁x₂, x₂²)" },
};

export interface Preset { id: string; label: string; map: MapId; points: () => LPoint[]; note: string }
const to11 = (pts: LPoint[]): LPoint[] => pts.map(p => ({ x: 2 * p.x - 1, y: 2 * p.y - 1, c: p.c }));
export const PRESETS: Preset[] = [
  { id: "ring", label: "ring → plane", map: "paraboloid", points: () => to11(rings({ n: 70, seed: 41, radii: [0.13, 0.36], noise: 0.03 }).points).map(p => ({ ...p, c: 1 - p.c })), note: "On the paraboloid the inner points sit low, the ring high: a flat plane separates them. Its cut through the paraboloid is a circle." },
  { id: "xor", label: "XOR needs x₁x₂", map: "saddle", points: () => to11(xor({ n: 70, seed: 42 }).points), note: "Lifted by x₁x₂, the two diagonals go up and down: the plane x₁x₂ = 0 separates them. Try the paraboloid: it does not help here." },
  { id: "blobs", label: "two blobs: lifting does not hurt", map: "paraboloid", points: () => to11(blobs([[0.32, 0.4], [0.68, 0.6]], { sd: 0.08, n: 50, seed: 43 }).points), note: "Separable in 2D already. In 3D the plane is nearly vertical and its cut is almost a straight line." },
  {
    id: "tworings", label: "two rings: one plane is not enough", map: "paraboloid",
    points: () => to11(rings({ n: 110, seed: 44, radii: [0.1, 0.26, 0.42], noise: 0.02 }).points).map(p => ({ ...p, c: p.c === 1 ? 1 : 0 })),
    note: "Blue inside, red ring, blue again outside: on the paraboloid red is a band in the middle height. No single plane cuts out a band: this map is not rich enough.",
  },
];


/** Lift the points by a map and train the linear SVM of the demo in 3D (null if only one class). */
export function liftAndTrain(P: LPoint[], map: MapId, C: number): SVMModel | null {
  const n = P.length, X = new Float64Array(3 * n), M = MAPS[map];
  P.forEach((p, i) => X.set(M.f(p.x, p.y), 3 * i));
  const nPos = P.filter(p => p.c === 1).length;
  if (nPos === 0 || nPos === n) return null;
  return trainSVM(X, n, 3, P.map(p => (p.c === 1 ? 1 : -1)), { C, tol: 1e-4 });
}
