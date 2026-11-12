// Point distribution model of a synthetic training set (no DOM): generate the set
// (lib/shapes.ts, fixed seed), align it by GPA (or not), PCA of the landmark vectors.
// Display units: aligned shapes are scaled so the RMS distance of a landmark from the
// centroid is 1 (centroid size √n), and drawn with y up.
import { Shape, Family, PDM, family, trainingSet, gpa, pdm, nPoints, centroidSize } from "../lib/shapes";

export interface ModelOptions {
  family: string;
  count: number;
  noise: number;
  align: boolean;
  corrupt: number;
  seed?: number;
}

export const POSE = { rot: 0.45, logScale: 0.2, shift: 0.6 };

export interface ShapeModel {
  fam: Family;
  /** Training shapes as generated (with pose), scaled to display units. */
  raw: Shape[];
  /** The shapes the PCA saw: GPA-aligned (or the raw ones), display units. */
  aligned: Shape[];
  model: PDM;
  corrupted: boolean[];
  /** Iterations GPA needed. */
  gpaIter: number;
}

export function buildModel(o: ModelOptions): ShapeModel {
  const fam = family(o.family);
  const ts = trainingSet(fam, { count: o.count, seed: o.seed ?? 3, noise: o.noise, ...POSE, corrupt: o.corrupt });
  const n = nPoints(ts.shapes[0]), D = Math.sqrt(n);
  // y up for display
  const flip = (s: Shape) => s.map((v, i) => (i % 2 ? -v : v));
  const meanSize = ts.shapes.reduce((t, s) => t + centroidSize(s), 0) / ts.shapes.length;
  const raw = ts.shapes.map(s => flip(s).map(v => v * D / meanSize));
  let aligned: Shape[], gpaIter = 0;
  if (o.align) {
    const g = gpa(raw, { maxIter: 30, tol: 1e-12 });
    aligned = g.aligned.map(s => s.map(v => v * D));
    gpaIter = g.iterations;
  } else aligned = raw.map(s => s.slice());
  return { fam, raw, aligned, model: pdm(aligned), corrupted: ts.corrupted, gpaIter };
}

/** Percent of the variance per mode and cumulative, for the first `k` modes. */
export function varianceShares(m: PDM, k: number): { share: number[]; cum: number[] } {
  const share: number[] = [], cum: number[] = [];
  let c = 0;
  for (let i = 0; i < k; i++) { const s = 100 * m.values[i] / m.total; c += s; share.push(s); cum.push(c); }
  return { share, cum };
}
