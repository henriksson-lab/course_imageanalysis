// Model of the neighbours demo (no DOM): a seeded population of nuclei (Poisson, clustered
// or regular placement, never touching), cell bodies grown from the nuclei by a fixed
// distance, and small spots (children) mostly inside nuclei, a few in the cytoplasm and
// one on the border between two touching cells.
import { rasteriseEllipses, Ellipse } from "../lib/synth";
import { expandLabels, touching, withinDistance, nearestNeighbours, clarkEvans } from "../lib/neighbours";
import { rng, poisson } from "../lib/random";

export const W = 256, H = 256;
export const CELL_GROWTH = 9;

export type Pattern = "poisson" | "clustered" | "regular";

export interface Population {
  w: number; h: number;
  nuclei: Int32Array; count: number;
  xs: Float64Array; ys: Float64Array;
  cells: Int32Array;
  spots: Int32Array; nSpots: number;
  /** Index (label) of the spot placed on the border between two cells (0 if none). */
  borderSpot: number;
}

export interface PopOptions { pattern: Pattern; n: number; seed?: number }

/** Centres for n nuclei, by pattern; two nuclei never come closer than the sum of their radii + 2. */
function centres(o: PopOptions, radii: number[]): [number, number][] {
  const r = rng(o.seed ?? 1);
  const out: [number, number][] = [];
  const ok = (x: number, y: number) => {
    const k = out.length, rk = radii[k % radii.length];
    return x >= 4 && y >= 4 && x <= W - 5 && y <= H - 5 && out.every(([p, q], j) => Math.hypot(p - x, q - y) >= rk + radii[j % radii.length] + 2);
  };
  if (o.pattern === "regular") {
    const s = Math.sqrt(W * H / o.n), nx = Math.round(W / s), ny = Math.ceil(o.n / nx), sx = W / nx, sy = H / ny;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (out.length >= o.n) break;
      const x = (i + 0.5) * sx + (j % 2 ? 0.25 : -0.25) * sx + 0.12 * sx * (2 * r.unif() - 1);
      const y = (j + 0.5) * sy + 0.12 * sy * (2 * r.unif() - 1);
      if (ok(x, y)) out.push([x, y]);
    }
    return out;
  }
  if (o.pattern === "clustered") {
    const k = Math.max(2, Math.round(o.n / 9));
    const parents = Array.from({ length: k }, () => [30 + (W - 60) * r.unif(), 30 + (H - 60) * r.unif()]);
    let guard = 0;
    while (out.length < o.n && guard++ < 20000) {
      const [px, py] = parents[Math.floor(r.unif() * k)];
      const x = px + 17 * r.norm(), y = py + 17 * r.norm();
      if (ok(x, y)) out.push([x, y]);
    }
    return out;
  }
  let guard = 0;
  while (out.length < o.n && guard++ < 20000) {
    const x = W * r.unif(), y = H * r.unif();
    if (ok(x, y)) out.push([x, y]);
  }
  return out;
}

/** Disc of radius rad around (x, y) painted with label l where allowed(i). */
function paintDisc(lab: Int32Array, x: number, y: number, rad: number, l: number, allowed: (i: number) => boolean): number {
  let n = 0;
  for (let yy = Math.floor(y - rad); yy <= Math.ceil(y + rad); yy++) for (let xx = Math.floor(x - rad); xx <= Math.ceil(x + rad); xx++) {
    if (xx < 0 || yy < 0 || xx >= W || yy >= H || (xx - x) ** 2 + (yy - y) ** 2 > rad * rad) continue;
    const i = yy * W + xx;
    if (allowed(i)) { lab[i] = l; n++; }
  }
  return n;
}

export function makePopulation(o: PopOptions): Population {
  const r = rng((o.seed ?? 1) * 7919 + 13);
  const shapes = Array.from({ length: 100 }, () => { const a = 3.5 + 2 * r.unif(); return { a, b: a * (0.7 + 0.25 * r.unif()), angle: Math.PI * r.unif() }; });
  const cs = centres(o, shapes.map(s => s.a));
  const ell: Ellipse[] = cs.map(([x, y], k) => ({ x, y, ...shapes[k] }));
  const nuclei = rasteriseEllipses(W, H, ell);
  const count = ell.length;
  const xs = new Float64Array(count), ys = new Float64Array(count), n = new Float64Array(count);
  for (let i = 0; i < W * H; i++) { const l = nuclei[i]; if (l) { xs[l - 1] += i % W; ys[l - 1] += Math.floor(i / W); n[l - 1]++; } }
  for (let k = 0; k < count; k++) { xs[k] /= n[k]; ys[k] /= n[k]; }
  const cells = expandLabels(nuclei, W, H, CELL_GROWTH);
  // spots
  const spots = new Int32Array(W * H);
  let ns = 0;
  for (let k = 0; k < count; k++) {
    const m = poisson(r, 1.6);
    for (let j = 0; j < m; j++) {
      const t = 2 * Math.PI * r.unif(), rho = 1.5 * Math.sqrt(r.unif());
      const x = xs[k] + rho * Math.cos(t), y = ys[k] + rho * Math.sin(t);
      if (paintDisc(spots, x, y, 1.3, ns + 1, i => !spots[i])) ns++;
    }
  }
  // a few spots in the cytoplasm (inside a cell, outside every nucleus)
  let guard = 0, cyto = 0;
  while (cyto < 4 && guard++ < 5000) {
    const x = 6 + (W - 12) * r.unif(), y = 6 + (H - 12) * r.unif(), i = Math.round(y) * W + Math.round(x);
    if (!cells[i] || nuclei[i] || spots[i]) continue;
    let clear = true;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) { const j = i + dy * W + dx; if (nuclei[j] || spots[j]) clear = false; }
    if (!clear) continue;
    if (paintDisc(spots, x, y, 1.6, ns + 1, j => !spots[j])) { ns++; cyto++; }
  }
  // one spot on the border between two touching cells (closest pair to the image centre)
  let borderSpot = 0, best = Infinity, bi = -1;
  for (let y = 3; y < H - 3; y++) for (let x = 3; x < W - 4; x++) {
    const i = y * W + x, a = cells[i], b = cells[i + 1];
    if (a && b && a !== b && !nuclei[i] && !nuclei[i + 1]) {
      const d = Math.hypot(x - W / 2, y - H / 2);
      if (d < best) { best = d; bi = i; }
    }
  }
  if (bi >= 0 && paintDisc(spots, bi % W + 0.5, Math.floor(bi / W), 1.6, ns + 1, j => !spots[j])) { ns++; borderSpot = ns; }
  return { w: W, h: H, nuclei, count, xs, ys, cells, spots, nSpots: ns, borderSpot };
}

// ---------------------------------------------------------------------------------------
// Analysis (used by the demo and by the exercise numbers)

export type Def = "adjacent" | "within" | "expanded";
export type Border = "all" | "remove" | "guard";
export const GUARD = 24;

export interface Analysis {
  objects: Int32Array;
  nb: number[][];
  percent: Float64Array | null;
  /** Objects that are measured (border rule). */
  measured: boolean[];
  /** Objects whose region (Voronoi region for "within") touches the image border. */
  touchesBorder: boolean[];
}

/** Labels that touch the image border. */
export function borderLabels(lab: Int32Array, count: number): boolean[] {
  const b = new Array(count + 1).fill(false);
  for (let x = 0; x < W; x++) { b[lab[x]] = true; b[lab[(H - 1) * W + x]] = true; }
  for (let y = 0; y < H; y++) { b[lab[y * W]] = true; b[lab[y * W + W - 1]] = true; }
  b[0] = false;
  return b;
}

/** Neighbour lists for one definition and border rule. "remove" deletes objects whose cell body reaches the border. */
export function analyse(pop: Population, voronoi: Int32Array, def: Def, r: number, border: Border): Analysis {
  const c = pop.count;
  const bCell = borderLabels(pop.cells, c);
  const nucl = border === "remove" ? pop.nuclei.map(l => (bCell[l] ? 0 : l)) : pop.nuclei;
  const objects = def === "adjacent" ? expandLabels(nucl, W, H, CELL_GROWTH) : def === "expanded" ? expandLabels(nucl, W, H) : nucl;
  let nb: number[][], percent: Float64Array | null = null;
  if (def === "within") nb = withinDistance(objects, W, H, c, r);
  else { const T = touching(objects, W, H, c, 8); nb = T.neighbours; percent = T.percent; }
  const touchesBorder = borderLabels(def === "within" ? voronoi : objects, c);
  const measured = Array.from({ length: c + 1 }, (_, l) => {
    if (!l) return false;
    if (border === "remove") return !bCell[l];
    if (border === "guard") return pop.xs[l - 1] >= GUARD && pop.ys[l - 1] >= GUARD && pop.xs[l - 1] < W - GUARD && pop.ys[l - 1] < H - GUARD;
    return true;
  });
  return { objects, nb, percent, measured, touchesBorder };
}

export interface NN { first: number[]; firstDist: number[]; second: number[]; secondDist: number[]; angle: number[]; present: number }

/** Centroid nearest neighbours among the objects present (removed border objects are not neighbours). */
export function nnFor(pop: Population, a: Analysis, border: Border): NN {
  const keep = Array.from({ length: pop.count }, (_, k) => border !== "remove" || a.measured[k + 1]);
  const idx = keep.map((v, k) => (v ? k : -1)).filter(k => k >= 0);
  const r = nearestNeighbours(idx.map(k => pop.xs[k]), idx.map(k => pop.ys[k]));
  const out: NN = { first: [], firstDist: [], second: [], secondDist: [], angle: [], present: idx.length };
  for (let k = 0; k < pop.count; k++) { out.first.push(-1); out.firstDist.push(NaN); out.second.push(-1); out.secondDist.push(NaN); out.angle.push(NaN); }
  idx.forEach((k, j) => {
    out.first[k] = idx[r.first[j]] ?? -1; out.firstDist[k] = r.firstDist[j];
    out.second[k] = idx[r.second[j]] ?? -1; out.secondDist[k] = r.secondDist[j]; out.angle[k] = r.angle[j];
  });
  return out;
}

export interface Summary {
  counts: number[]; meanCount: number; interior: number; atBorder: number;
  nnd: number[]; rho: number; R: number; observed: number; expected: number; nn: NN;
}

const meanOf = (v: number[]) => (v.length ? v.reduce((x, y) => x + y, 0) / v.length : NaN);

/** Population summary of one analysis: neighbour counts, NN distances, Clark–Evans R. */
export function summarise(pop: Population, a: Analysis, border: Border): Summary {
  const ls = Array.from({ length: pop.count }, (_, k) => k + 1).filter(l => a.measured[l]);
  const counts = ls.map(l => a.nb[l].length);
  const nn = nnFor(pop, a, border);
  const nnd = ls.map(l => nn.firstDist[l - 1]);
  const rho = nn.present / (W * H);
  const ce = clarkEvans(nnd, nnd.length / rho);
  return {
    counts, meanCount: meanOf(counts),
    interior: meanOf(ls.filter(l => !a.touchesBorder[l]).map(l => a.nb[l].length)),
    atBorder: meanOf(ls.filter(l => a.touchesBorder[l]).map(l => a.nb[l].length)),
    nnd, rho, R: ce.R, observed: ce.observed, expected: ce.expected, nn,
  };
}
