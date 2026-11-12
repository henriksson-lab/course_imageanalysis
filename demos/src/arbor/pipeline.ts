// The measurement pipeline of the arbor demo, without DOM (so node scripts can compute the
// numbers quoted in the exercises): synthetic tree → image → multi-scale vesselness (edges
// day, lib/edges.ts) → hysteresis threshold → clean (area filter, fill small holes) →
// skeleton → prune → graph → measurements, compared with the ground-truth graph.
import { Grey } from "../lib/imageops";
import { frangi, geomScales, hysteresis } from "../lib/edges";
import { edt } from "../lib/distance";
import { label2d } from "../lib/label";
import { growTree, treeGraph, renderTree, Tree, TreeGraph } from "../lib/trees";
import {
  skeletonize, prune, skeletonGraph, SkelGraph, pathLength, chord, cyclomatic, shollProfile, edgePolylines, LengthEstimator,
} from "../lib/skeleton";

export interface GenParams { kind: "neuron" | "vessel"; seed: number; branching: number; tortuosity: number; noise: number; blur: number; spots: number }
export interface SegParams { sMin: number; sMax: number; c: number; tHigh: number; tLow: number; k: number }

export const N = 256;

export interface Generated { tree: Tree; truth: TreeGraph; image: Grey; centre: [number, number]; somaR: number }

export function generate(p: GenParams): Generated {
  const tree = growTree({ kind: p.kind, seed: p.seed, branching: p.branching, tortuosity: p.tortuosity });
  const truth = treeGraph(tree);
  const { image } = renderTree(tree, { noise: p.noise, blur: p.blur, spots: p.spots, seed: 100 + p.seed });
  const centre: [number, number] = tree.soma ? [tree.soma.x, tree.soma.y] : [N / 2, N / 2];
  return { tree, truth, image, centre, somaR: tree.soma ? tree.soma.r : 0 };
}

/** The σ values of the vesselness (3 scales from sMin to sMax, or one if equal). */
export const scalesOf = (s: SegParams): number[] => (s.sMax <= s.sMin + 1e-9 ? [s.sMin] : geomScales(s.sMin, s.sMax, 3));

export function vesselness(img: Grey, s: SegParams): { V: Float32Array; bestSigma: Float32Array } {
  const sig = scalesOf(s), F = frangi(img, sig, 0.5, s.c, true);
  const bestSigma = Float32Array.from(F.bestScale, k => (k >= 0 ? sig[k] : 0));
  return { V: F.response.data as Float32Array, bestSigma };
}

export interface Segmented { mask: Uint8Array; dt: Float32Array; skel0: Uint8Array; skel: Uint8Array; graph: SkelGraph; pruned: number }

export function segment(V: ArrayLike<number>, s: SegParams): Segmented {
  const mask = hysteresis({ width: N, height: N, data: V }, s.tLow, s.tHigh, 8).edges;
  // clean: drop specks below 30 pixels, fill holes up to 20 pixels
  const L = label2d(mask, N, N, 8), size = new Int32Array(L.count + 1);
  for (let i = 0; i < N * N; i++) size[L.data[i]]++;
  for (let i = 0; i < N * N; i++) if (L.data[i] && size[L.data[i]] < 30) mask[i] = 0;
  const inv = Uint8Array.from(mask, v => 1 - v), H = label2d(inv, N, N, 4), hs = new Int32Array(H.count + 1), touches = new Uint8Array(H.count + 1);
  for (let i = 0; i < N * N; i++) {
    hs[H.data[i]]++;
    const x = i % N, y = (i - x) / N;
    if (x === 0 || y === 0 || x === N - 1 || y === N - 1) touches[H.data[i]] = 1;
  }
  for (let i = 0; i < N * N; i++) if (H.data[i] && !touches[H.data[i]] && hs[H.data[i]] <= 20) mask[i] = 1;
  const dt = edt(mask, N, N);
  const skel0 = skeletonize(mask, N, N);
  const pr = s.k > 0 ? prune(skel0, N, N, s.k) : { skeleton: skel0, removed: 0 };
  return { mask, dt, skel0, skel: pr.skeleton, graph: skeletonGraph(pr.skeleton, N, N), pruned: pr.removed };
}

export interface Measures {
  length: Record<LengthEstimator, number>;
  ends: number;
  branchPoints: number;
  /** Branch points per 10 000 px². */
  density: number;
  /** Mean radius along the centreline (DT on the skeleton, outside the soma). */
  radius: number;
  /** Mean tortuosity of the edges with a chord of at least 10 px. */
  tortuosity: number;
  loops: number;
  /** Edges leaving the soma (neurons). */
  primaries: number;
  /** Positions of the branch points (outside the soma). */
  branchXY: [number, number][];
}

const inSoma = (g: Generated, x: number, y: number) => g.somaR > 0 && Math.hypot(x - g.centre[0], y - g.centre[1]) < g.somaR + 5;

export function measure(gen: Generated, seg: Segmented): Measures {
  const G = seg.graph, w = N;
  const length = { count: 0, weighted: 0, corrected: 0 } as Record<LengthEstimator, number>;
  for (const e of G.edges) for (const est of ["count", "weighted", "corrected"] as LengthEstimator[]) length[est] += pathLength(e.path, w, est);
  const junc = G.nodes.filter(n => n.kind === "junction");
  const branchXY = junc.filter(n => !inSoma(gen, n.x, n.y)).map(n => [n.x, n.y] as [number, number]);
  const somaNodes = new Set(junc.filter(n => inSoma(gen, n.x, n.y)).map(n => n.id));
  let primaries = 0;
  if (gen.somaR > 0) for (const e of G.edges) {
    const a = somaNodes.has(e.a), b = somaNodes.has(e.b);
    if (a !== b) primaries++;
  }
  let rs = 0, rn = 0;
  for (let i = 0; i < N * N; i++) if (seg.skel[i] && !inSoma(gen, i % N, Math.floor(i / N))) { rs += seg.dt[i]; rn++; }
  let ts = 0, tn = 0;
  for (const e of G.edges) { const c = chord(e.path, w); if (c >= 10) { ts += pathLength(e.path, w, "corrected") / c; tn++; } }
  return {
    length, ends: G.nodes.filter(n => n.kind === "end").length, branchPoints: branchXY.length,
    density: branchXY.length / (N * N / 1e4), radius: rn ? rs / rn : 0, tortuosity: tn ? ts / tn : 0,
    loops: cyclomatic(G), primaries, branchXY,
  };
}

export interface TruthMeasures { length: number; ends: number; branchPoints: number; density: number; radius: number; tortuosity: number; loops: number; primaries: number; branchXY: [number, number][] }

export function truthMeasures(gen: Generated): TruthMeasures {
  const T = gen.truth, t = gen.tree;
  const length = T.edges.reduce((s, e) => s + e.length, 0);
  const root = T.nodes.find(n => n.kind === "root" || n.kind === "soma");
  const ends = T.nodes.filter(n => n.kind === "end").length + (root && root.kind === "root" && root.degree === 1 ? 1 : 0);
  const branchXY = T.nodes.filter(n => n.kind === "branch" && !inSoma(gen, n.x, n.y)).map(n => [n.x, n.y] as [number, number]);
  let rs = 0, rn = 0;
  t.points.forEach((p, i) => { if (!(t.soma && i === 0)) { rs += p.r; rn++; } });
  const long = T.edges.filter(e => e.chord >= 10);
  return {
    length, ends, branchPoints: branchXY.length, density: branchXY.length / (N * N / 1e4), radius: rs / Math.max(1, rn),
    tortuosity: long.reduce((s, e) => s + e.length / e.chord, 0) / Math.max(1, long.length), loops: T.loops,
    primaries: t.soma ? (root?.degree ?? 0) : 0, branchXY,
  };
}

/** Greedy matching of branch points within `tol` px: matched, missed (truth only), spurious (measured only). */
export function matchPoints(truth: [number, number][], found: [number, number][], tol = 8): { matched: number; missed: [number, number][]; spurious: [number, number][] } {
  const pairs: [number, number, number][] = [];
  truth.forEach((a, i) => found.forEach((b, j) => { const d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d <= tol) pairs.push([d, i, j]); }));
  pairs.sort((p, q) => p[0] - q[0]);
  const ut = new Set<number>(), uf = new Set<number>();
  for (const [, i, j] of pairs) if (!ut.has(i) && !uf.has(j)) { ut.add(i); uf.add(j); }
  return { matched: ut.size, missed: truth.filter((_, i) => !ut.has(i)), spurious: found.filter((_, j) => !uf.has(j)) };
}

export const SHOLL_RADII = Array.from({ length: 60 }, (_, k) => 4 + 2 * k);

export function sholl(gen: Generated, seg: Segmented): { measured: number[]; truth: number[] } {
  const [cx, cy] = gen.centre;
  const meas = shollProfile(edgePolylines(seg.graph).map(L => L.map(([x, y]) => [x + 0.5, y + 0.5] as [number, number])), cx, cy, SHOLL_RADII);
  const tr = shollProfile(gen.truth.edges.map(e => e.line), cx, cy, SHOLL_RADII);
  return { measured: meas, truth: tr };
}
