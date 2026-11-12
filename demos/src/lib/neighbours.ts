// Relations between objects and between channels (owner: measurements day, units
// relations/ and coloc/). No DOM.
//
// Objects (label images, 0 = background, labels 1..count):
//  - touching(labels, w, h, conn): which labels touch (share a 4- or 8-neighbour pair),
//    with, per object, its boundary pixels (pixels with a neighbour of another value,
//    the image outside counting as background) and how many of them touch another object:
//    percent touching = 100 · touching / boundary.
//  - expandLabels: every background pixel takes the label of the nearest object pixel
//    (exact Euclidean, via the feature transform of lib/distance.ts), up to maxDist:
//    "expand until adjacent" = a Voronoi tessellation of the objects.
//  - withinDistance(labels, r): objects whose closest pixels are at most r apart
//    (r = 1: 4-adjacent, √2: 8-adjacent). Brute force over boundary pixels.
//  - nearestNeighbours(xs, ys): first and second closest point and the angle between the
//    two closest neighbours, seen from the point (degrees, 0..180).
//  - clarkEvans: R = mean nearest-neighbour distance / (0.5 / √density).
//  - assignParents: parent of each child by largest overlap or by the label at the rounded
//    centroid; flags children that overlap several parents.
// Channels (colocalisation over the pixels of a region mask):
//  - pearson, overlap coefficient Σ AB / √(Σ A² Σ B²), Manders M1/M2 with thresholds:
//    M1 = Σ A over pixels with A > tA and B > tB / Σ A over pixels with A > tA (M2 likewise),
//    so tA = tB = −∞ gives the unthresholded fractions Σ_{B>tB} A / Σ A; orthogonal
//    regression B = a A + b; Costes' threshold search; block-scramble randomisation.

import { edtWithNearest } from "./distance";
import { Rng, shuffle } from "./random";

type A = ArrayLike<number>;

const OFF4: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const OFF8: [number, number][] = [...OFF4, [1, 1], [1, -1], [-1, 1], [-1, -1]];

export interface Touching {
  /** Sorted neighbour labels per label (index 0 unused). */
  neighbours: number[][];
  /** Boundary pixel count per label. */
  boundary: Int32Array;
  /** Boundary pixels that touch another object, per label. */
  touching: Int32Array;
  /** Percent touching per label (NaN if no boundary). */
  percent: Float64Array;
}

/** Adjacency and boundary contact of the labels of a label image (see the header). */
export function touching(labels: A, w: number, h: number, count: number, conn: 4 | 8 = 8): Touching {
  const off = conn === 4 ? OFF4 : OFF8;
  const sets = Array.from({ length: count + 1 }, () => new Set<number>());
  const boundary = new Int32Array(count + 1), tch = new Int32Array(count + 1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const l = labels[y * w + x];
    if (!l) continue;
    let isB = false, isT = false;
    for (const [dx, dy] of off) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) { isB = true; continue; }
      const m = labels[yy * w + xx];
      if (m !== l) { isB = true; if (m) { isT = true; sets[l].add(m); } }
    }
    if (isB) boundary[l]++;
    if (isT) tch[l]++;
  }
  const percent = Float64Array.from({ length: count + 1 }, (_, l) => (boundary[l] ? 100 * tch[l] / boundary[l] : NaN));
  return { neighbours: sets.map(s => [...s].sort((a, b) => a - b)), boundary, touching: tch, percent };
}

/** Grow every object into the background up to maxDist (exact Euclidean; ties go to the first found). */
export function expandLabels(labels: A, w: number, h: number, maxDist = Infinity): Int32Array {
  const bg = new Uint8Array(w * h);
  let any = false;
  for (let i = 0; i < w * h; i++) { bg[i] = labels[i] ? 0 : 1; if (labels[i]) any = true; }
  const out = Int32Array.from(labels);
  if (!any) return out;
  const { dist, nearest } = edtWithNearest(bg, w, h);
  for (let i = 0; i < w * h; i++) if (bg[i] && nearest[i] >= 0 && dist[i] <= maxDist) out[i] = labels[nearest[i]];
  return out;
}

/** Boundary pixels (x, y) of each label (4-neighbour test). */
function boundaryPixels(labels: A, w: number, h: number, count: number): { xs: number[]; ys: number[] }[] {
  const out = Array.from({ length: count + 1 }, () => ({ xs: [] as number[], ys: [] as number[] }));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const l = labels[y * w + x];
    if (!l) continue;
    let b = false;
    for (const [dx, dy] of OFF4) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h || labels[yy * w + xx] !== l) { b = true; break; }
    }
    if (b) { out[l].xs.push(x); out[l].ys.push(y); }
  }
  return out;
}

/** Smallest pixel-to-pixel distance between every pair of labels that is ≤ rMax (else Infinity). */
export function gapMatrix(labels: A, w: number, h: number, count: number, rMax: number): Float64Array {
  const bp = boundaryPixels(labels, w, h, count);
  const D = new Float64Array((count + 1) * (count + 1)).fill(Infinity);
  // bounding circles for pruning
  const cx = new Float64Array(count + 1), cy = new Float64Array(count + 1), rad = new Float64Array(count + 1);
  for (let l = 1; l <= count; l++) {
    const { xs, ys } = bp[l];
    if (!xs.length) continue;
    cx[l] = xs.reduce((a, b) => a + b, 0) / xs.length; cy[l] = ys.reduce((a, b) => a + b, 0) / ys.length;
    for (let k = 0; k < xs.length; k++) rad[l] = Math.max(rad[l], Math.hypot(xs[k] - cx[l], ys[k] - cy[l]));
  }
  for (let a = 1; a <= count; a++) for (let b = a + 1; b <= count; b++) {
    if (!bp[a].xs.length || !bp[b].xs.length) continue;
    if (Math.hypot(cx[a] - cx[b], cy[a] - cy[b]) - rad[a] - rad[b] > rMax) continue;
    let m = Infinity;
    const A1 = bp[a], B1 = bp[b];
    for (let i = 0; i < A1.xs.length; i++) for (let j = 0; j < B1.xs.length; j++) {
      const d = (A1.xs[i] - B1.xs[j]) ** 2 + (A1.ys[i] - B1.ys[j]) ** 2;
      if (d < m) m = d;
    }
    m = Math.sqrt(m);
    if (m <= rMax) { D[a * (count + 1) + b] = m; D[b * (count + 1) + a] = m; }
  }
  return D;
}

/** Neighbours within distance r: labels whose closest pixels are at most r apart. */
export function withinDistance(labels: A, w: number, h: number, count: number, r: number): number[][] {
  const D = gapMatrix(labels, w, h, count, r);
  return Array.from({ length: count + 1 }, (_, a) => {
    const out: number[] = [];
    if (a) for (let b = 1; b <= count; b++) if (b !== a && D[a * (count + 1) + b] <= r) out.push(b);
    return out;
  });
}

export interface NearestResult {
  first: Int32Array; firstDist: Float64Array;
  second: Int32Array; secondDist: Float64Array;
  /** Angle (degrees) at the point between the directions to its two closest neighbours. */
  angle: Float64Array;
}

/** First and second nearest neighbour of each point (brute force; −1 if none). */
export function nearestNeighbours(xs: A, ys: A): NearestResult {
  const n = xs.length;
  const r: NearestResult = {
    first: new Int32Array(n).fill(-1), firstDist: new Float64Array(n).fill(Infinity),
    second: new Int32Array(n).fill(-1), secondDist: new Float64Array(n).fill(Infinity), angle: new Float64Array(n).fill(NaN),
  };
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = Math.hypot(xs[i] - xs[j], ys[i] - ys[j]);
      if (d < r.firstDist[i]) { r.second[i] = r.first[i]; r.secondDist[i] = r.firstDist[i]; r.first[i] = j; r.firstDist[i] = d; }
      else if (d < r.secondDist[i]) { r.second[i] = j; r.secondDist[i] = d; }
    }
    if (r.second[i] >= 0) {
      const a = r.first[i], b = r.second[i];
      const ux = xs[a] - xs[i], uy = ys[a] - ys[i], vx = xs[b] - xs[i], vy = ys[b] - ys[i];
      const c = (ux * vx + uy * vy) / (Math.hypot(ux, uy) * Math.hypot(vx, vy));
      r.angle[i] = Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
    }
  }
  return r;
}

/** Clark–Evans ratio R = mean NN distance / (0.5/√ρ), ρ = n / area. Expected NN distance under Poisson too. */
export function clarkEvans(nnDist: A, area: number): { R: number; observed: number; expected: number } {
  let s = 0, n = 0;
  for (let i = 0; i < nnDist.length; i++) if (Number.isFinite(nnDist[i])) { s += nnDist[i]; n++; }
  const observed = s / n, expected = 0.5 / Math.sqrt(nnDist.length / area);
  return { R: observed / expected, observed, expected };
}

/** Poisson nearest-neighbour distance density for density ρ: 2πρ r exp(−πρ r²). */
export const poissonNNDensity = (r: number, rho: number): number => 2 * Math.PI * rho * r * Math.exp(-Math.PI * rho * r * r);

export interface ParentResult {
  /** Parent label per child (0 = none). */
  parent: Int32Array;
  /** Fraction of the child's pixels inside its parent. */
  fraction: Float64Array;
  /** Number of different parents the child overlaps. */
  nParents: Int32Array;
  /** Child count per parent. */
  childCount: Int32Array;
}

/** Assign each child label a parent label, by largest overlap or by the label at its centroid. */
export function assignParents(child: A, parent: A, w: number, h: number, nChild: number, nParent: number,
  method: "overlap" | "centroid" = "overlap"): ParentResult {
  const ov = new Map<number, number>();
  const size = new Int32Array(nChild + 1), sx = new Float64Array(nChild + 1), sy = new Float64Array(nChild + 1);
  for (let i = 0; i < w * h; i++) {
    const c = child[i];
    if (!c) continue;
    size[c]++; sx[c] += i % w; sy[c] += Math.floor(i / w);
    const p = parent[i];
    if (p) ov.set(c * (nParent + 1) + p, (ov.get(c * (nParent + 1) + p) ?? 0) + 1);
  }
  const res: ParentResult = { parent: new Int32Array(nChild + 1), fraction: new Float64Array(nChild + 1), nParents: new Int32Array(nChild + 1), childCount: new Int32Array(nParent + 1) };
  for (let c = 1; c <= nChild; c++) {
    let best = 0, bestN = 0;
    for (let p = 1; p <= nParent; p++) {
      const n = ov.get(c * (nParent + 1) + p) ?? 0;
      if (n) res.nParents[c]++;
      if (n > bestN) { bestN = n; best = p; }
    }
    if (method === "centroid" && size[c]) {
      const x = Math.round(sx[c] / size[c]), y = Math.round(sy[c] / size[c]);
      best = parent[y * w + x];
      bestN = best ? (ov.get(c * (nParent + 1) + best) ?? 0) : 0;
    }
    res.parent[c] = best;
    res.fraction[c] = size[c] ? bestN / size[c] : 0;
    if (best) res.childCount[best]++;
  }
  return res;
}

// ---------------------------------------------------------------------------------------
// Colocalisation

/** Pearson correlation of a and b over the pixels where mask ≠ 0 (all pixels if no mask). */
export function pearsonMasked(a: A, b: A, mask?: A): number {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { n++; sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) {
    const u = a[i] - ma, v = b[i] - mb;
    sab += u * v; saa += u * u; sbb += v * v;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN;
}

/** Manders' overlap coefficient Σ AB / √(Σ A² Σ B²). */
export function overlapCoefficient(a: A, b: A, mask?: A): number {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return ab / Math.sqrt(aa * bb);
}

/** Manders M1, M2 with thresholds (see the header). */
export function manders(a: A, b: A, tA: number, tB: number, mask?: A): { m1: number; m2: number } {
  let a1 = 0, aAll = 0, b1 = 0, bAll = 0;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const ia = a[i] > tA, ib = b[i] > tB;
    if (ia) { aAll += a[i]; if (ib) a1 += a[i]; }
    if (ib) { bAll += b[i]; if (ia) b1 += b[i]; }
  }
  return { m1: aAll ? a1 / aAll : NaN, m2: bAll ? b1 / bAll : NaN };
}

/** Orthogonal (total least squares) regression line B = slope · A + intercept. */
export function orthoRegression(a: A, b: A, mask?: A): { slope: number; intercept: number } {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { n++; sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let saa = 0, sbb = 0, sab = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { const u = a[i] - ma, v = b[i] - mb; saa += u * u; sbb += v * v; sab += u * v; }
  const slope = sab === 0 ? 0 : (sbb - saa + Math.sqrt((sbb - saa) ** 2 + 4 * sab * sab)) / (2 * sab);
  return { slope, intercept: mb - slope * ma };
}

export interface CostesResult {
  /** False when the channels are not positively correlated (no regression line to walk down). */
  ok: boolean;
  tA: number; tB: number;
  /** (tA, r below thresholds) at every step tried. */
  trace: { tA: number; tB: number; r: number }[];
}

/**
 * Costes' automatic threshold: lower tA from max(A) in `steps` steps, with tB on the
 * orthogonal regression line, until the Pearson r of the pixels below the thresholds
 * (A ≤ tA or B ≤ tB) drops to 0 or below. Not defined (ok = false) unless the channels are
 * positively correlated with a positive regression slope.
 */
export function costes(a: A, b: A, mask?: A, steps = 100): CostesResult {
  const { slope, intercept } = orthoRegression(a, b, mask);
  if (!(slope > 0) || !(pearsonMasked(a, b, mask) > 0)) return { ok: false, tA: NaN, tB: NaN, trace: [] };
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { lo = Math.min(lo, a[i]); hi = Math.max(hi, a[i]); }
  const trace: CostesResult["trace"] = [];
  const below = new Uint8Array(a.length);
  let res = { tA: lo, tB: slope * lo + intercept };
  for (let s = 1; s <= steps; s++) {
    const tA = hi - (hi - lo) * s / steps, tB = slope * tA + intercept;
    for (let i = 0; i < a.length; i++) below[i] = (!mask || mask[i]) && (a[i] <= tA || b[i] <= tB) ? 1 : 0;
    const r = pearsonMasked(a, b, below);
    trace.push({ tA, tB, r });
    res = { tA, tB };
    if (!(r > 0)) break;
  }
  return { ok: true, ...res, trace };
}

/** Copy of b with its bs×bs blocks shuffled (whole blocks only; the remainder stays). */
export function blockScramble(b: A, w: number, h: number, bs: number, r: Rng): Float64Array {
  const nx = Math.floor(w / bs), ny = Math.floor(h / bs);
  const perm = shuffle(r, Int32Array.from({ length: nx * ny }, (_, i) => i));
  const out = Float64Array.from(b);
  for (let k = 0; k < nx * ny; k++) {
    const src = perm[k], sx = (src % nx) * bs, sy = Math.floor(src / nx) * bs, dx = (k % nx) * bs, dy = Math.floor(k / nx) * bs;
    for (let y = 0; y < bs; y++) for (let x = 0; x < bs; x++) out[(dy + y) * w + dx + x] = b[(sy + y) * w + sx + x];
  }
  return out;
}
