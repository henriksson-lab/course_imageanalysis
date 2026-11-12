// Model of the stitch demo (no DOM): a 3 × 3 mosaic of 128² tiles cut from a 384² image at
// jittered (sub-pixel) stage positions, with noise, a brightness change and vignetting per
// tile; pairwise registration of neighbours by phase correlation + the four candidates
// (overlap NCC) + a parabola for the sub-pixel part; global positions by least squares.
// Tile k = (row r, column c) = 3r + c. Position = top-left corner in the big image.
import { rng, uniform } from "../lib/random";
import {
  correlate2d, findPeaks, fourCandidates, parabolaPeak2d, peakRatio, Surface, Candidate, TileEdge,
} from "../lib/registration";

export const T = 128, STEP = 104, GRID = 3, BIG = 384, NT = GRID * GRID;

export interface StitchSettings {
  seed: number;
  noise: number;       // sd per tile
  jitter: number;      // max stage error (px)
  vignette: number;    // brightness drop at the tile corners (0..1)
  empty: number;       // index of a tile over an empty region (−1: none)
  badPair: number;     // index into EDGES of a pair whose shared region is empty (−1: none)
}

export const DEFAULTS: StitchSettings = { seed: 3, noise: 6, jitter: 6, vignette: 0.25, empty: -1, badPair: -1 };

/** Neighbour pairs: 6 horizontal, then 6 vertical. */
export const EDGES: [number, number][] = (() => {
  const e: [number, number][] = [];
  for (let r = 0; r < GRID; r++) for (let c = 0; c + 1 < GRID; c++) e.push([GRID * r + c, GRID * r + c + 1]);
  for (let r = 0; r + 1 < GRID; r++) for (let c = 0; c < GRID; c++) e.push([GRID * r + c, GRID * (r + 1) + c]);
  return e;
})();

export const nominal = (k: number): [number, number] => [20 + (k % GRID) * STEP, 20 + Math.floor(k / GRID) * STEP];

/** Synthetic stand-in for the micrograph (node tests): fine texture (smoothed noise) plus blobs. */
export function syntheticBase(seed = 7): Float64Array {
  const r = rng(seed), n = new Float64Array(BIG * BIG).map(() => r.norm()), tmp = new Float64Array(BIG * BIG), out = new Float64Array(BIG * BIG);
  const k = [0.06, 0.24, 0.4, 0.24, 0.06];
  for (let y = 0; y < BIG; y++) for (let x = 0; x < BIG; x++) { let v = 0; for (let j = -2; j <= 2; j++) v += k[j + 2] * n[y * BIG + Math.min(BIG - 1, Math.max(0, x + j))]; tmp[y * BIG + x] = v; }
  for (let y = 0; y < BIG; y++) for (let x = 0; x < BIG; x++) { let v = 0; for (let j = -2; j <= 2; j++) v += k[j + 2] * tmp[Math.min(BIG - 1, Math.max(0, y + j)) * BIG + x]; out[y * BIG + x] = 150 + 60 * v; }
  for (let b = 0; b < 300; b++) {
    const cx = BIG * r.unif(), cy = BIG * r.unif(), s = 1.5 + 5 * r.unif() ** 2, h = (r.unif() < 0.5 ? -1 : 1) * (30 + 70 * r.unif());
    const R = Math.ceil(3 * s);
    for (let y = Math.max(0, Math.floor(cy - R)); y < Math.min(BIG, cy + R); y++)
      for (let x = Math.max(0, Math.floor(cx - R)); x < Math.min(BIG, cx + R); x++)
        out[y * BIG + x] += h * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
  }
  return out;
}

export interface Tiles {
  tiles: Float64Array[];
  pos: [number, number][];     // true positions
  base: Float64Array;          // the (possibly edited) big image
  s: StitchSettings;
}

function bilinear(img: Float64Array, x: number, y: number): number {
  x = Math.max(0, Math.min(BIG - 1.0001, x)); y = Math.max(0, Math.min(BIG - 1.0001, y));
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, i = y0 * BIG + x0;
  return (1 - fy) * ((1 - fx) * img[i] + fx * img[i + 1]) + fy * ((1 - fx) * img[i + BIG] + fx * img[i + BIG + 1]);
}

/** Cut the tiles. `base0`: the 384² image (null: synthetic). */
export function makeTiles(s: StitchSettings, base0: Float64Array | null): Tiles {
  const r = rng(s.seed), base = Float64Array.from(base0 ?? syntheticBase());
  const pos: [number, number][] = [];
  for (let k = 0; k < NT; k++) { const [x, y] = nominal(k); pos.push([x + uniform(r, -s.jitter, s.jitter), y + uniform(r, -s.jitter, s.jitter)]); }
  if (s.badPair >= 0) {
    // the region shared by the two tiles of this pair is empty (flat, the mean grey level)
    const [i, j] = EDGES[s.badPair];
    const x0 = Math.floor(Math.max(pos[i][0], pos[j][0])) - 2, x1 = Math.ceil(Math.min(pos[i][0], pos[j][0]) + T) + 2;
    const y0 = Math.floor(Math.max(pos[i][1], pos[j][1])) - 2, y1 = Math.ceil(Math.min(pos[i][1], pos[j][1]) + T) + 2;
    let m = 0;
    for (const v of base) m += v;
    m /= base.length;
    for (let y = Math.max(0, y0); y < Math.min(BIG, y1); y++) for (let x = Math.max(0, x0); x < Math.min(BIG, x1); x++) base[y * BIG + x] = m;
  }
  let mean = 0;
  for (const v of base) mean += v;
  mean /= base.length;
  const tiles: Float64Array[] = [];
  for (let k = 0; k < NT; k++) {
    const rn = rng(1000 * s.seed + k + 1), gain = uniform(rn, 0.85, 1.15), t = new Float64Array(T * T);
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      const v = k === s.empty ? mean : bilinear(base, pos[k][0] + x, pos[k][1] + y);
      const u = (x - (T - 1) / 2) / (T / 2), w = (y - (T - 1) / 2) / (T / 2);
      t[y * T + x] = gain * v * (1 - s.vignette * (u * u + w * w) / 2) + s.noise * rn.norm();
    }
    tiles.push(t);
  }
  return { tiles, pos, base, s };
}

export interface PairResult {
  edge: TileEdge;              // measured shift (position j − position i) and weight
  surface: Surface;
  candidates: Candidate[];
  ratio: number;               // peak ratio
  height: number;              // phase-correlation peak height
  ncc: number;                 // overlap NCC of the chosen candidate
  peakRank: number;            // which of the highest peaks gave it (0 = the highest)
}

/**
 * Register every neighbour pair: partial phase correlation (p = 0.7, more robust to noise than
 * p = 1 on these small overlaps), then the four candidates of each of the 5 highest peaks,
 * scored by NCC on their actual overlap (at least 1000 px); the best one wins. Its NCC is
 * the pair's confidence. Sub-pixel part: a parabola through the peak and its neighbours.
 */
export function registerPairs(t: Tiles, opts: { p?: number; peaks?: number } = {}): { edges: PairResult[]; ms: number } {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const out: PairResult[] = [];
  EDGES.forEach(([i, j]) => {
    const s = correlate2d(t.tiles[i], t.tiles[j], T, T, { p: opts.p ?? 0.7, eps: 1e-3 });
    const pks = findPeaks(s, opts.peaks ?? 5, 3);
    let best: { c: Candidate; pk: (typeof pks)[number] } | null = null, cands: Candidate[] = [];
    for (const pk of pks) {
      const cs = fourCandidates(t.tiles[i], t.tiles[j], T, T, pk.ix, pk.iy, MIN_OVERLAP);
      if (pk === pks[0]) cands = cs;
      for (const c of cs) if (Number.isFinite(c.ncc) && (!best || c.ncc > best.c.ncc)) best = { c, pk };
    }
    const pk = best ? best.pk : pks[0], c = best ? best.c : cands[0];
    const sub = parabolaPeak2d(s, pk.ix, pk.iy);
    const ncc = Number.isFinite(c.ncc) ? c.ncc : 0;
    out.push({
      edge: { i, j, dx: c.tx + sub.tx - pk.tx, dy: c.ty + sub.ty - pk.ty, w: ncc },
      surface: s, candidates: fourCandidates(t.tiles[i], t.tiles[j], T, T, pk.ix, pk.iy, MIN_OVERLAP),
      ratio: peakRatio(s, 3), height: pk.value, ncc, peakRank: pks.indexOf(pk),
    });
  });
  const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
  return { edges: out, ms };
}

export const MIN_OVERLAP = 1000;

/** Least-squares weights: NCC² for trusted pairs (NCC ≥ thr), almost nothing for the others. */
export function weighted(edges: PairResult[], thr: number, dropped: Set<number> = new Set()): TileEdge[] {
  return edges.map((e, k) => ({ ...e.edge, w: dropped.has(k) ? 1e-6 : e.ncc >= thr ? e.ncc * e.ncc : 1e-4 }));
}

/** The chain 0-1-2-5-4-3-6-7-8 through the grid (indices into EDGES). */
export const SNAKE: number[] = [[0, 1], [1, 2], [2, 5], [4, 5], [3, 4], [3, 6], [6, 7], [7, 8]]
  .map(([a, b]) => EDGES.findIndex(([i, j]) => i === a && j === b));

declare const performance: { now(): number } | undefined;

/** Position errors (px) of estimated positions vs truth, after matching tile 0. */
export function positionErrors(t: Tiles, x: ArrayLike<number>, y: ArrayLike<number>): number[] {
  const ox = t.pos[0][0] - x[0], oy = t.pos[0][1] - y[0];
  return t.pos.map((p, k) => Math.hypot(x[k] + ox - p[0], y[k] + oy - p[1]));
}
