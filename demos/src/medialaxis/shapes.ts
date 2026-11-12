// Shapes and presets of the medialaxis demo (no DOM, so node scripts can use them).
import { rasterise, ellipsePoly, starPoly, Poly } from "../lib/shapecanvas";
import {
  Metric, distanceMap, medialAxis, reconstructFromDiscs, lantuejoul, lantuejoulReconstruct, skeletonize, skeletonGraph, boundarySpread,
} from "../lib/skeleton";

export const N = 256;

export type Method = "ridges" | "lantuejoul" | "thinning";
export type ShapeId = "rect" | "ellipse" | "L" | "star" | "dumbbell" | "ring";

export interface Shape { polys: Poly[]; rule: "evenodd" | "nonzero" }
export const SHAPES: Record<ShapeId, { label: string; make: () => Shape }> = {
  rect: { label: "rectangle", make: () => ({ polys: [[[40, 92], [216, 92], [216, 164], [40, 164]]], rule: "evenodd" }) },
  ellipse: { label: "ellipse", make: () => ({ polys: [ellipsePoly(128, 128, 100, 56, 16)], rule: "evenodd" }) },
  L: { label: "L", make: () => ({ polys: [[[50, 30], [110, 30], [110, 166], [214, 166], [214, 222], [50, 222]]], rule: "evenodd" }) },
  star: { label: "star", make: () => ({ polys: [starPoly(128, 136, 112, 46, 5)], rule: "evenodd" }) },
  dumbbell: { label: "dumbbell", make: () => ({ polys: [ellipsePoly(68, 128, 48, 48, 12), ellipsePoly(188, 128, 48, 48, 12), [[60, 114], [196, 114], [196, 142], [60, 142]]], rule: "nonzero" }) },
  ring: { label: "ring", make: () => ({ polys: [ellipsePoly(128, 128, 104, 84, 14), ellipsePoly(128, 128, 52, 34, 10)], rule: "evenodd" }) },
};

export interface State {
  shape: ShapeId; metric: Metric; method: Method; t: number; bump: number; bumpPos: number; pick: [number, number] | null;
  /** Keep only axis pixels whose nearest boundary points are at least λ apart (maximal-disc method). */
  lambda: number;
}
export interface Preset { id: string; label: string; s: Partial<State> }
export const BASE: State = { shape: "rect", metric: "euclidean", method: "ridges", t: 99, bump: 0, bumpPos: 0.12, pick: null, lambda: 0 };
export const PRESETS: Preset[] = [
  { id: "rect", label: "a rectangle forks at the ends", s: { shape: "rect", metric: "euclidean", method: "ridges", lambda: 8, pick: [45, 97] } },
  { id: "bump", label: "one bump, one branch", s: { shape: "ellipse", metric: "euclidean", method: "thinning", bump: 3, bumpPos: 0.2 } },
  { id: "ring", label: "the ring keeps its loop", s: { shape: "ring", metric: "euclidean", method: "thinning" } },
  { id: "star", label: "reconstruct the star from its axis", s: { shape: "star", metric: "chessboard", method: "lantuejoul" } },
  { id: "metric", label: "city-block vs Euclidean axis", s: { shape: "rect", metric: "cityblock", method: "ridges" } },
  { id: "fire", label: "grassfire", s: { shape: "dumbbell", metric: "euclidean", method: "thinning", t: 22 } },
]

/** Point at arc-length fraction s along a closed polygon, and the outward unit normal. */
export function boundaryPoint(p: Poly, s: number): { x: number; y: number; nx: number; ny: number } {
  const n = p.length, L: number[] = [];
  let tot = 0, area = 0;
  for (let i = 0; i < n; i++) {
    const a = p[i], b = p[(i + 1) % n];
    L.push(Math.hypot(b[0] - a[0], b[1] - a[1])); tot += L[i];
    area += a[0] * b[1] - b[0] * a[1];
  }
  let d = ((s % 1) + 1) % 1 * tot, i = 0;
  while (i < n - 1 && d > L[i]) { d -= L[i]; i++; }
  const a = p[i], b = p[(i + 1) % n], u = L[i] ? d / L[i] : 0;
  const tx = (b[0] - a[0]) / (L[i] || 1), ty = (b[1] - a[1]) / (L[i] || 1);
  // screen coordinates (y down): for area > 0 the polygon runs clockwise on screen, outward = (ty, −tx)
  const sg = area > 0 ? 1 : -1;
  return { x: a[0] + u * (b[0] - a[0]), y: a[1] + u * (b[1] - a[1]), nx: sg * ty, ny: -sg * tx };
}

/** The 256 × 256 mask of a shape with an optional bump (radius `bump` at arc fraction `pos` of the first polygon). */
export function shapeMask(shape: Shape, bump: number, pos: number): Uint8Array {
  const mask = rasterise(shape.polys, N, N, shape.rule);
  if (bump > 0) {
    const b = boundaryPoint(shape.polys[0], pos), r = bump, cx = b.x + b.nx * r * 0.45, cy = b.y + b.ny * r * 0.45;
    for (let y = Math.floor(cy - r - 1); y <= cy + r + 1; y++) for (let x = Math.floor(cx - r - 1); x <= cx + r + 1; x++)
      if (x >= 0 && y >= 0 && x < N && y < N && Math.hypot(x + 0.5 - cx, y + 0.5 - cy) < r) mask[y * N + x] = 1;
  }
  // keep a background frame, so the image border is never part of the shape
  for (let k = 0; k < N; k++) { mask[k] = 0; mask[(N - 1) * N + k] = 0; mask[k * N] = 0; mask[k * N + N - 1] = 0; }
  return mask;
}

export const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(2));

export interface ShapeAnalysis {
  mask: Uint8Array; dt: Float32Array; axis: Uint8Array; radius: Float32Array; recon: Uint8Array; maxDT: number;
  nShape: number; nAxis: number; missing: number; extra: number; ends: number; branches: number; pieces: number;
  info: string; note: string;
}

/** Everything the demo shows for a shape and a state (axis by the chosen method, its discs, the readout). */
export function analyseShape(shape: Shape, st: State): ShapeAnalysis {
  const mask = shapeMask(shape, st.bump, st.bumpPos);
  const lMetric: Metric = st.metric === "cityblock" ? "cityblock" : "chessboard";
  const useMetric: Metric = st.method === "lantuejoul" ? lMetric : st.metric;
  const dt = distanceMap(mask, N, N, useMetric);
  let maxDT = 1;
  for (let i = 0; i < N * N; i++) if (mask[i] && dt[i] > maxDT) maxDT = dt[i];
  let axis: Uint8Array, radius: Float32Array, recon: Uint8Array;
  if (st.method === "ridges") {
    axis = medialAxis(mask, N, N, useMetric, dt);
    if (st.lambda > 0) { const sp = boundarySpread(mask, N, N); for (let i = 0; i < N * N; i++) if (axis[i] && sp[i] < st.lambda) axis[i] = 0; }
    radius = dt;
    recon = reconstructFromDiscs(axis, radius, N, N, useMetric);
  } else if (st.method === "lantuejoul") {
    const se = lMetric === "cityblock" ? "cross" : "square";
    const L = lantuejoul(mask, N, N, se);
    axis = L.skeleton;
    radius = Float32Array.from(L.radius, r => r + 1);
    recon = lantuejoulReconstruct(L, N, N, se);
  } else {
    axis = skeletonize(mask, N, N);
    radius = dt;
    recon = reconstructFromDiscs(axis, radius, N, N, useMetric);
  }
  let nAxis = 0, nShape = 0, missing = 0, extra = 0;
  for (let i = 0; i < N * N; i++) {
    if (axis[i]) nAxis++;
    if (mask[i]) nShape++;
    if (mask[i] && !recon[i]) missing++;
    if (!mask[i] && recon[i]) extra++;
  }
  const g = skeletonGraph(skeletonize(axis, N, N), N, N);
  const ends = g.nodes.filter(n => n.kind === "end").length;
  const info = `shape: ${nShape} pixels, largest DT ${fmt(maxDT)}\n` +
    `axis: ${nAxis} pixels (${(100 * nAxis / Math.max(1, nShape)).toFixed(1)} %)\n` +
    `axis graph: ${ends} end points, ${g.edges.length} branches, ${g.components} piece${g.components === 1 ? "" : "s"}\n` +
    (missing || extra ? `reconstruction: ${missing} missing, ${extra} extra` : "reconstruction: exact");
  const note = st.method === "lantuejoul"
    ? `Lantuéjoul with B = ${lMetric === "cityblock" ? "the 3 × 3 cross (city-block discs)" : "the 3 × 3 square (chessboard discs)"}; colour = r of S_r.`
    : st.method === "thinning" ? "Thinning: connected and one pixel wide, but not exactly centred; radius = DT on it." : `Maximal discs: a pixel whose disc no neighbour's disc contains${st.lambda > 0 ? `; only where the two nearest boundary points are ≥ ${st.lambda} px apart` : ""}.`;
  return { mask, dt, axis, radius, recon, maxDT, nShape, nAxis, missing, extra, ends, branches: g.edges.length, pieces: g.components, info, note };
}
