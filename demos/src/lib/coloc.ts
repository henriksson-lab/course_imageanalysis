// Colocalisation (owner: colocalisation day). No DOM.
//
// Pixel coefficients: Pearson, overlap, Manders (re-exported from lib/neighbours.ts, where the
// measurements day first wrote them), plus the Manders variant without the own-channel
// threshold, Manders' k1/k2, Spearman, the product of differences from the mean (PDM) and
// Li's ICQ. Significance: shift curves (r against a shift of B), block randomisation with a
// permutation p-value. Objects: cross-channel nearest neighbours, the fraction within d and
// its chance level, one-to-one pairing (Hungarian), the cross-K / L function and Monte Carlo
// envelopes, a small spot detector (LoG + local maxima + weighted centroid) and overlap /
// centroid relations between two label images. Closed forms: the normalised overlap of two
// Gaussian spots.
import { pearsonMasked, overlapCoefficient, manders, orthoRegression, costes, CostesResult, blockScramble } from "./neighbours";
import { normCdf } from "./stats";
import { rng, Rng, shuffle } from "./random";
import { hungarian } from "./eval";
import { parabolaOffset } from "./registration";
import { fromArray, logFilter } from "./imageops";

export { pearsonMasked, overlapCoefficient, manders, orthoRegression, costes, blockScramble };
export type { CostesResult };

type A = ArrayLike<number>;
const inMask = (mask: A | undefined, i: number) => !mask || mask[i] !== 0;

// ---------------------------------------------------------------------------------------
// Pixel coefficients

/** Manders without the own-channel threshold: M1 = Σ A[B > tB] / Σ A, M2 = Σ B[A > tA] / Σ B. */
export function mandersNoOwnThreshold(a: A, b: A, tA: number, tB: number, mask?: A): { m1: number; m2: number } {
  let a1 = 0, aAll = 0, b1 = 0, bAll = 0;
  for (let i = 0; i < a.length; i++) {
    if (!inMask(mask, i)) continue;
    aAll += a[i]; bAll += b[i];
    if (b[i] > tB) a1 += a[i];
    if (a[i] > tA) b1 += b[i];
  }
  return { m1: aAll ? a1 / aAll : NaN, m2: bAll ? b1 / bAll : NaN };
}

/** Manders' k1 = Σ AB / Σ A², k2 = Σ AB / Σ B² (overlap² = k1·k2). */
export function kCoefficients(a: A, b: A, mask?: A): { k1: number; k2: number } {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) if (inMask(mask, i)) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return { k1: ab / aa, k2: ab / bb };
}

/** Ranks 1..n with ties averaged. */
export function ranks(x: A): Float64Array {
  const n = x.length, idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => x[i] - x[j]);
  const r = new Float64Array(n);
  for (let s = 0; s < n;) {
    let e = s;
    while (e + 1 < n && x[idx[e + 1]] === x[idx[s]]) e++;
    const avg = (s + e) / 2 + 1;
    for (let k = s; k <= e; k++) r[idx[k]] = avg;
    s = e + 1;
  }
  return r;
}

/** Spearman's rank correlation over the mask: Pearson of the ranks (ties averaged). */
export function spearmanMasked(a: A, b: A, mask?: A): number {
  const ia: number[] = [];
  for (let i = 0; i < a.length; i++) if (inMask(mask, i)) ia.push(i);
  const ra = ranks(ia.map(i => a[i])), rb = ranks(ia.map(i => b[i]));
  return pearsonMasked(ra, rb);
}

/** Means of a and b over the mask. */
export function maskedMeans(a: A, b: A, mask?: A): { ma: number; mb: number; n: number } {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (inMask(mask, i)) { n++; sa += a[i]; sb += b[i]; }
  return { ma: sa / n, mb: sb / n, n };
}

/** Product of the differences from the mean, (A − Ā)(B − B̄), per pixel (0 outside the mask). */
export function pdm(a: A, b: A, mask?: A): Float64Array {
  const { ma, mb } = maskedMeans(a, b, mask), out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) if (inMask(mask, i)) out[i] = (a[i] - ma) * (b[i] - mb);
  return out;
}

export interface IcqResult {
  /** Share of positive PDM pixels − 0.5, in [−0.5, 0.5]. */
  icq: number;
  nPos: number; nNeg: number; n: number;
  /** Two-sided sign-test p-value (normal approximation, pixels assumed independent). */
  p: number;
}

/** Li's intensity correlation quotient. */
export function icq(a: A, b: A, mask?: A): IcqResult {
  const d = pdm(a, b, mask);
  let nPos = 0, nNeg = 0, n = 0;
  for (let i = 0; i < a.length; i++) if (inMask(mask, i)) { n++; if (d[i] > 0) nPos++; else if (d[i] < 0) nNeg++; }
  const z = (nPos - n / 2) / Math.sqrt(n / 4);
  return { icq: nPos / n - 0.5, nPos, nNeg, n, p: 2 * normCdf(-Math.abs(z)) };
}

// ---------------------------------------------------------------------------------------
// Shift curves and randomisation

/**
 * Pearson r between A(p) and B(p + t) for t = −maxShift..maxShift along x or y: pixels p in
 * the mask whose shifted partner lies inside the image. If B(x) = A(x − s), the peak is at t = s.
 */
export function shiftCurve(a: A, b: A, w: number, h: number, maxShift: number, mask?: A, axis: "x" | "y" = "x"): { shifts: number[]; r: number[] } {
  const shifts: number[] = [], r: number[] = [];
  for (let t = -maxShift; t <= maxShift; t++) {
    let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!inMask(mask, i)) continue;
      const xx = axis === "x" ? x + t : x, yy = axis === "y" ? y + t : y;
      if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
      const u = a[i], v = b[yy * w + xx];
      n++; sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v;
    }
    const cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    shifts.push(t); r.push(va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : NaN);
  }
  return { shifts, r };
}

/** Sub-pixel peak of a sampled curve: integer argmax + parabola through its neighbours. */
export function curvePeak(shifts: number[], r: number[]): { at: number; value: number; index: number } {
  let k = 0;
  for (let i = 1; i < r.length; i++) if (r[i] > r[k]) k = i;
  const off = k > 0 && k < r.length - 1 ? parabolaOffset(r[k - 1], r[k], r[k + 1]) : 0;
  return { at: shifts[k] + off, value: r[k], index: k };
}

/** Half width at half maximum of a peaked curve around its maximum (linear interpolation). */
export function halfWidth(shifts: number[], r: number[], base = 0): number {
  const k = curvePeak(shifts, r).index, half = base + (r[k] - base) / 2;
  const side = (dir: 1 | -1) => {
    for (let i = k; i + dir >= 0 && i + dir < r.length; i += dir) {
      if (r[i + dir] <= half) return Math.abs(shifts[i] + (shifts[i + dir] - shifts[i]) * (r[i] - half) / (r[i] - r[i + dir]) - shifts[k]);
    }
    return NaN;
  };
  return (side(1) + side(-1)) / 2;
}

/** Permutation p-value (1 + #{null ≥ observed}) / (N + 1). */
export function permutationP(nullValues: A, observed: number): number {
  let c = 0;
  for (let i = 0; i < nullValues.length; i++) if (nullValues[i] >= observed) c++;
  return (1 + c) / (nullValues.length + 1);
}

/**
 * Copy of b with its bs × bs blocks shuffled among the blocks that touch the mask (the region
 * of interest); blocks entirely outside it stay. Without a mask every whole block takes part
 * (as lib/neighbours blockScramble). Keeps (almost) the region's own intensity distribution,
 * where shuffling over the whole image would import background blocks into the region.
 */
export function blockScrambleMasked(b: A, w: number, h: number, bs: number, r: Rng, mask?: A): Float64Array {
  if (!mask) return blockScramble(b, w, h, bs, r);
  const nx = Math.floor(w / bs), ny = Math.floor(h / bs), inside: number[] = [];
  for (let k = 0; k < nx * ny; k++) {
    const x0 = (k % nx) * bs, y0 = Math.floor(k / nx) * bs;
    let any = false;
    for (let y = 0; y < bs && !any; y++) for (let x = 0; x < bs; x++) if (mask[(y0 + y) * w + x0 + x]) { any = true; break; }
    if (any) inside.push(k);
  }
  const perm = shuffle(r, Int32Array.from(inside));
  const out = Float64Array.from(b);
  inside.forEach((dst, i) => {
    const src = perm[i], sx = (src % nx) * bs, sy = Math.floor(src / nx) * bs, dx = (dst % nx) * bs, dy = Math.floor(dst / nx) * bs;
    for (let y = 0; y < bs; y++) for (let x = 0; x < bs; x++) out[(dy + y) * w + dx + x] = b[(sy + y) * w + sx + x];
  });
  return out;
}

/**
 * Block randomisation of B: bs × bs blocks shuffled within the region (blockScrambleMasked),
 * the statistic (default Pearson r) recomputed over the region, n times. p = permutation p.
 */
export function randomisationTest(a: A, b: A, w: number, h: number, o: { block: number; n: number; seed?: number; mask?: A; stat?: (a: A, b: A, mask?: A) => number }): { null: Float64Array; observed: number; p: number } {
  const stat = o.stat ?? pearsonMasked, r = rng(o.seed ?? 77);
  const observed = stat(a, b, o.mask), out = new Float64Array(o.n);
  for (let k = 0; k < o.n; k++) out[k] = stat(a, blockScrambleMasked(b, w, h, o.block, r, o.mask), o.mask);
  return { null: out, observed, p: permutationP(out, observed) };
}

// ---------------------------------------------------------------------------------------
// Objects and points

/** For each A point, the nearest B point (brute force): distance and index (−1 if B empty). */
export function crossNearest(ax: A, ay: A, bx: A, by: A): { dist: Float64Array; idx: Int32Array } {
  const dist = new Float64Array(ax.length).fill(Infinity), idx = new Int32Array(ax.length).fill(-1);
  for (let i = 0; i < ax.length; i++) for (let j = 0; j < bx.length; j++) {
    const d = Math.hypot(ax[i] - bx[j], ay[i] - by[j]);
    if (d < dist[i]) { dist[i] = d; idx[i] = j; }
  }
  return { dist, idx };
}

/** Share of distances ≤ d. */
export function fractionWithin(dist: A, d: number): number {
  let c = 0;
  for (let i = 0; i < dist.length; i++) if (dist[i] <= d) c++;
  return dist.length ? c / dist.length : NaN;
}

/** Chance that a Poisson process of density ρ has a point within d: 1 − exp(−ρπd²). */
export const chanceFraction = (rho: number, d: number): number => 1 - Math.exp(-rho * Math.PI * d * d);

/** Fraction corrected for chance: (f_obs − f_chance) / (1 − f_chance). */
export const correctedFraction = (fObs: number, fChance: number): number => (fObs - fChance) / (1 - fChance);

/**
 * One-to-one pairing of A and B points with distance ≤ dMax that maximises the number of
 * pairs, then minimises the total distance. The Hungarian method runs separately on each
 * connected component of the "within dMax" graph (exact, and fast for sparse spots), on a
 * padded square cost matrix. Returns B's index for each A (−1 if unpaired) and the distances.
 */
export function pairOneToOne(ax: A, ay: A, bx: A, by: A, dMax: number): { partner: Int32Array; dist: Float64Array } {
  const nA = ax.length, nB = bx.length;
  const partner = new Int32Array(nA).fill(-1), dist = new Float64Array(nA).fill(Infinity);
  // components over A ∪ B (B offset by nA)
  const parent = Int32Array.from({ length: nA + nB }, (_, i) => i);
  const find = (i: number): number => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const edges: [number, number, number][] = [];
  for (let i = 0; i < nA; i++) for (let j = 0; j < nB; j++) {
    const d = Math.hypot(ax[i] - bx[j], ay[i] - by[j]);
    if (d <= dMax) { edges.push([i, j, d]); parent[find(i)] = find(nA + j); }
  }
  const groups = new Map<number, { as: number[]; bs: number[] }>();
  for (const [i, j] of edges) {
    const g = find(i);
    if (!groups.has(g)) groups.set(g, { as: [], bs: [] });
    const G = groups.get(g)!;
    if (!G.as.includes(i)) G.as.push(i);
    if (!G.bs.includes(j)) G.bs.push(j);
  }
  const BIG = 1e6 * (dMax + 1);
  for (const { as, bs } of groups.values()) {
    const n = Math.max(as.length, bs.length), cost = new Float64Array(n * n).fill(BIG);
    as.forEach((i, r) => bs.forEach((j, c) => {
      const d = Math.hypot(ax[i] - bx[j], ay[i] - by[j]);
      if (d <= dMax) cost[r * n + c] = d;
    }));
    const col = hungarian(cost, n);
    as.forEach((i, r) => {
      const c = col[r];
      if (c >= 0 && c < bs.length && cost[r * n + c] < BIG) { partner[i] = bs[c]; dist[i] = cost[r * n + c]; }
    });
  }
  return { partner, dist };
}

/** Cross-K: K(r) = |W| / (nA nB) · #{(i, j): d_ij ≤ r}, no edge correction. */
export function crossK(ax: A, ay: A, bx: A, by: A, radii: A, area: number): Float64Array {
  const K = new Float64Array(radii.length), nA = ax.length, nB = bx.length;
  const rmax = Math.max(...Array.from(radii));
  for (let i = 0; i < nA; i++) for (let j = 0; j < nB; j++) {
    const d = Math.hypot(ax[i] - bx[j], ay[i] - by[j]);
    if (d > rmax) continue;
    for (let k = 0; k < radii.length; k++) if (d <= radii[k]) K[k]++;
  }
  for (let k = 0; k < radii.length; k++) K[k] *= area / (nA * nB);
  return K;
}

/** L(r) − r = √(K/π) − r (0 for independent points, without edge effects). */
export function crossLminusR(K: A, radii: A): Float64Array {
  return Float64Array.from(K as ArrayLike<number>, (k, i) => Math.sqrt(k / Math.PI) - radii[i]);
}

/** Pointwise min/max (and mean) of `n` simulated curves: a Monte Carlo envelope. */
export function envelope(sim: (r: Rng) => Float64Array, n: number, seed = 5): { lo: Float64Array; hi: Float64Array; mean: Float64Array } {
  const r = rng(seed);
  let lo: Float64Array | null = null, hi: Float64Array | null = null, mean: Float64Array | null = null;
  for (let k = 0; k < n; k++) {
    const c = sim(r);
    if (!lo) { lo = Float64Array.from(c); hi = Float64Array.from(c); mean = new Float64Array(c.length); }
    for (let i = 0; i < c.length; i++) { lo[i] = Math.min(lo[i], c[i]); hi![i] = Math.max(hi![i], c[i]); mean![i] += c[i] / n; }
  }
  return { lo: lo!, hi: hi!, mean: mean! };
}

/** Normalised overlap of two 2D Gaussian spots (σ1, σ2) whose centres are δ apart. */
export function gaussianOverlap(delta: number, s1: number, s2 = s1): number {
  const v = s1 * s1 + s2 * s2;
  return (2 * s1 * s2 / v) * Math.exp(-delta * delta / (2 * v));
}

/** Expected measured distance for true distance d and per-axis localisation errors σA, σB
 *  (the mean of a Rice distribution; Rayleigh for d = 0). */
export function meanMeasuredDistance(d: number, sA: number, sB: number): number {
  const s = Math.sqrt(sA * sA + sB * sB), x = d * d / (4 * s * s);
  // Rice mean: s √(π/2) L_{1/2}(−d²/2s²), L_{1/2}(−2x) = e^{−x}[(1 + 2x) I0(x) + 2x I1(x)]
  return s * Math.sqrt(Math.PI / 2) * Math.exp(-x) * ((1 + 2 * x) * besselI0(x) + 2 * x * besselI1(x));
}

function besselI0(x: number): number { let s = 1, t = 1; for (let k = 1; k < 200; k++) { t *= (x / 2) ** 2 / (k * k); s += t; if (t < 1e-16 * s) break; } return s; }
function besselI1(x: number): number { let t = x / 2, s = t; for (let k = 1; k < 200; k++) { t *= (x / 2) ** 2 / (k * (k + 1)); s += t; if (t < 1e-16 * s) break; } return s; }

export interface Spot { x: number; y: number; value: number }

/** Median by a numeric typed-array sort (same value as stats.median). */
function fastMedian(x: Float64Array): number {
  const s = Float64Array.from(x).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/**
 * Spot detector: LoG at σ (sign flipped, so spots are positive), local maxima in 3 × 3 above
 * median + k · robust sd of the response, located by the intensity-weighted centroid of the
 * background-subtracted image in a (2R+1)² window, R = ceil(1.5 σ).
 */
export function detectSpots(img: A, w: number, h: number, sigma: number, k = 5, mask?: A): Spot[] {
  const L = logFilter(fromArray(w, h, img), sigma).data;
  const resp = Float64Array.from(L, v => -v);
  const med = fastMedian(resp), mad = fastMedian(Float64Array.from(resp, v => Math.abs(v - med))) * 1.4826, thr = med + k * mad;
  const bg = fastMedian(Float64Array.from(img)), R = Math.ceil(1.5 * sigma), out: Spot[] = [];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, v = resp[i];
    if (v <= thr || !inMask(mask, i)) continue;
    let isMax = true;
    for (let dy = -1; dy <= 1 && isMax; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const u = resp[i + dy * w + dx];
      if (u > v || (u === v && dy * w + dx < 0)) { isMax = false; break; }
    }
    if (!isMax) continue;
    let sw = 0, sx = 0, sy = 0;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const wt = Math.max(0, img[yy * w + xx] - bg);
      sw += wt; sx += wt * xx; sy += wt * yy;
    }
    out.push(sw > 0 ? { x: sx / sw, y: sy / sw, value: v } : { x, y, value: v });
  }
  return out;
}

export interface ObjectRelation {
  /** Pixels of the A object, and how many of them lie in some B object. */
  area: number; overlapPx: number;
  /** Centroid of the A object and whether it lies in a B object. */
  cx: number; cy: number; centroidIn: boolean;
}

/** Per object of label image A: overlap with (any) B object and centroid-in-B. */
export function objectRelations(labA: A, nA: number, labB: A, w: number): ObjectRelation[] {
  const area = new Float64Array(nA + 1), ov = new Float64Array(nA + 1), sx = new Float64Array(nA + 1), sy = new Float64Array(nA + 1);
  for (let i = 0; i < labA.length; i++) {
    const l = labA[i];
    if (!l) continue;
    area[l]++; sx[l] += i % w; sy[l] += Math.floor(i / w);
    if (labB[i]) ov[l]++;
  }
  const out: ObjectRelation[] = [];
  for (let l = 1; l <= nA; l++) {
    const cx = sx[l] / area[l], cy = sy[l] / area[l];
    out.push({ area: area[l], overlapPx: ov[l], cx, cy, centroidIn: labB[Math.round(cy) * w + Math.round(cx)] !== 0 });
  }
  return out;
}
