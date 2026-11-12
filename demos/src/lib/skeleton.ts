// Skeletons (owner: skeleton day; no DOM). Binary masks are flat Uint8Array-like arrays
// (non-zero = object), row-major, w × h, y down. Pixels outside the image count as
// background. Tests: demos/test/skeleton-test.ts.
//
//  - 3 × 3 neighbourhoods: RING order N, NE, E, SE, S, SW, W, NW (clockwise on screen),
//    8-bit neighbour codes, three-state templates ("1" object, "0" background, "." don't care),
//    hit-or-miss with templates, rotations by 45°.
//  - Simple points (8-connected object, 4-connected background), counted by components in
//    the 8 neighbours; Yokoi's connectivity number as a cross-check.
//  - Thinning: sequential template thinning (the 8 rotated templates L1, L2), with a record
//    of every template application; parallel variant (cuts 2-pixel lines); simple-point
//    thinning with or without keeping end points; Zhang–Suen.
//  - Medial axis: distance maps (city-block, chessboard, Euclidean), centres of maximal
//    discs (= ridges of DT), reconstruction as a union of discs; Lantuéjoul's skeleton S_r
//    from erosions and openings, and its reconstruction ∪ S_r ⊕ rB.
//  - Skeleton graph: pixel classes (end, regular, junction), junction clusters, nodes and
//    edges with pixel paths, chain-code length estimators, pruning of short terminal edges,
//    cyclomatic number, tortuosity, radius along edges, Sholl crossings.
import { chamfer, edt, edtWithNearest } from "./distance";
import { label2d } from "./label";

type A = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Neighbourhoods and templates

/** Ring of the 8 neighbours as [dx, dy]: N, NE, E, SE, S, SW, W, NW (clockwise on screen). */
export const RING: [number, number][] = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
export const RING_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

export const isOn = (m: A, w: number, h: number, x: number, y: number): number =>
  x < 0 || y < 0 || x >= w || y >= h ? 0 : m[y * w + x] ? 1 : 0;

/** The 8 neighbours of (x, y) in RING order (0/1; outside = 0). */
export function neighbours(m: A, w: number, h: number, x: number, y: number): number[] {
  return RING.map(([dx, dy]) => isOn(m, w, h, x + dx, y + dy));
}

/** 8-bit code of the neighbours: bit k = RING[k]. */
export function neighbourCode(m: A, w: number, h: number, x: number, y: number): number {
  let c = 0;
  for (let k = 0; k < 8; k++) if (isOn(m, w, h, x + RING[k][0], y + RING[k][1])) c |= 1 << k;
  return c;
}

/** Number of object pixels among the 8 neighbours. */
export function degreeAt(m: A, w: number, h: number, x: number, y: number): number {
  let n = 0;
  for (const [dx, dy] of RING) n += isOn(m, w, h, x + dx, y + dy);
  return n;
}

/** A 3 × 3 template: three strings of "1" (object), "0" (background), "." (don't care). */
export type Template = string[];

/** Cell (row r, col c) of the 3 × 3 grid for ring position k. */
export const RING_CELL: [number, number][] = RING.map(([dx, dy]) => [dy + 1, dx + 1]);

/** Template states in RING order plus the centre: 1, 0 or −1 (don't care). */
export function templateRing(t: Template): { ring: number[]; centre: number } {
  const st = (ch: string) => (ch === "1" ? 1 : ch === "0" ? 0 : -1);
  return { ring: RING_CELL.map(([r, c]) => st(t[r][c])), centre: st(t[1][1]) };
}

export function templateFromRing(ring: number[], centre: number): Template {
  const g = [[".", ".", "."], [".", ".", "."], [".", ".", "."]];
  const ch = (v: number) => (v === 1 ? "1" : v === 0 ? "0" : ".");
  ring.forEach((v, k) => { const [r, c] = RING_CELL[k]; g[r][c] = ch(v); });
  g[1][1] = ch(centre);
  return g.map(r => r.join(""));
}

/** Rotate a template by `steps` × 45° clockwise (on screen); 2 steps = 90°. */
export function rotateTemplate(t: Template, steps: number): Template {
  const { ring, centre } = templateRing(t);
  const out = new Array(8);
  for (let k = 0; k < 8; k++) out[(((k + steps) % 8) + 8) % 8] = ring[k];
  return templateFromRing(out, centre);
}

/** Masks of a template for fast matching: (code & care) === want, and the centre state. */
export function templateBits(t: Template): { care: number; want: number; centre: number } {
  const { ring, centre } = templateRing(t);
  let care = 0, want = 0;
  ring.forEach((v, k) => { if (v >= 0) { care |= 1 << k; if (v) want |= 1 << k; } });
  return { care, want, centre };
}

/** Does the template fit at (x, y): every "1" on the object, every "0" on the background? */
export function matchAt(m: A, w: number, h: number, x: number, y: number, t: Template): boolean {
  const b = templateBits(t), c = isOn(m, w, h, x, y);
  if (b.centre >= 0 && c !== b.centre) return false;
  return (neighbourCode(m, w, h, x, y) & b.care) === b.want;
}

/** Hit-or-miss with a three-state template: (A ⊖ B1) ∩ (Aᶜ ⊖ B2), B1 = the "1" cells, B2 = the "0" cells. */
export function hitOrMissT(m: A, w: number, h: number, t: Template): Uint8Array {
  const out = new Uint8Array(w * h), b = templateBits(t);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = m[y * w + x] ? 1 : 0;
    if (b.centre >= 0 && c !== b.centre) continue;
    if ((neighbourCode(m, w, h, x, y) & b.care) === b.want) out[y * w + x] = 1;
  }
  return out;
}

/** Union of the hit-or-miss transforms of several templates. */
export function hitOrMissAny(m: A, w: number, h: number, ts: Template[]): Uint8Array {
  const out = new Uint8Array(w * h);
  for (const t of ts) { const r = hitOrMissT(m, w, h, t); for (let i = 0; i < out.length; i++) out[i] |= r[i]; }
  return out;
}

/** All 8 rotations (45° steps) of a template, without duplicates. */
export function rotations8(t: Template): Template[] {
  const out: Template[] = [], seen = new Set<string>();
  for (let s = 0; s < 8; s++) { const r = rotateTemplate(t, s), k = r.join("/"); if (!seen.has(k)) { seen.add(k); out.push(r); } }
  return out;
}

/** Named templates for the hit-or-miss demos. */
export const T_ISOLATED: Template = ["000", "010", "000"];
/** End point: exactly one neighbour (here north); its 8 rotations find every end point. */
export const T_END: Template = ["010", "010", "000"];
/** Outer corner of an object (top-left corner pixel). */
export const T_CORNER: Template = ["000", "011", "011"];
/** T-junction (W, E and S) and Y-junction (NW, NE and S) of a thin line. */
export const T_TJUNCTION: Template = ["000", "111", "010"];
export const T_YJUNCTION: Template = ["101", "010", "010"];
/** The two thinning templates (Golay's L): an edge pixel and a corner pixel. */
export const L1: Template = ["000", ".1.", "111"];
export const L2: Template = [".00", "110", ".1."];
/** The 8 thinning templates in the order they are applied: L1, L2, then each rotated by 90°. */
export const THIN_TEMPLATES: Template[] = [0, 2, 4, 6].flatMap(s => [rotateTemplate(L1, s), rotateTemplate(L2, s)]);
export const THIN_NAMES = ["L1", "L2", "L1 ↻90°", "L2 ↻90°", "L1 ↻180°", "L2 ↻180°", "L1 ↻270°", "L2 ↻270°"];

// ---------------------------------------------------------------------------------------
// Simple points

// adjacency of ring positions (8: Chebyshev distance 1; 4: city-block distance 1)
const ADJ8: boolean[][] = RING.map(a => RING.map(b => a !== b && Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])) === 1));
const ADJ4: boolean[][] = RING.map(a => RING.map(b => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1));
const IS4 = [true, false, true, false, true, false, true, false];

function ringComponents(on: boolean[], adj: boolean[][]): number[][] {
  const seen = new Array(8).fill(false), comps: number[][] = [];
  for (let s = 0; s < 8; s++) {
    if (!on[s] || seen[s]) continue;
    const comp: number[] = [], st = [s];
    seen[s] = true;
    while (st.length) {
      const k = st.pop()!;
      comp.push(k);
      for (let j = 0; j < 8; j++) if (on[j] && !seen[j] && adj[k][j]) { seen[j] = true; st.push(j); }
    }
    comps.push(comp.sort((a, b) => a - b));
  }
  return comps;
}

export interface SimpleTest {
  /** 8-connected components of object neighbours (each a list of ring positions). */
  fg: number[][];
  /** 4-connected components of background neighbours that touch a 4-neighbour of p. */
  bg: number[][];
  /** Number of object neighbours. */
  degree: number;
  simple: boolean;
  /** Short reason, e.g. "1 object part, 1 background part: simple". */
  reason: string;
}

/**
 * Is p (an object pixel with these 8 neighbours, RING order) simple: can it be removed
 * without changing the topology? Yes iff its object neighbours form exactly one 8-connected
 * part and its background neighbours exactly one 4-connected part touching p.
 */
export function simpleTest(nb: ArrayLike<number>): SimpleTest {
  const on = Array.from({ length: 8 }, (_, k) => !!nb[k]);
  const fg = ringComponents(on, ADJ8);
  const bg = ringComponents(on.map(v => !v), ADJ4).filter(c => c.some(k => IS4[k]));
  const degree = on.filter(Boolean).length;
  const simple = fg.length === 1 && bg.length === 1;
  let reason: string;
  if (degree === 0) reason = "isolated pixel: removing it deletes an object";
  else if (bg.length === 0) reason = "interior pixel: removing it makes a hole";
  else if (fg.length > 1) reason = `${fg.length} object parts: removing it splits the object`;
  else if (bg.length > 1) reason = `${bg.length} background parts: removing it merges a hole with the outside`;
  else reason = degree === 1 ? "end point: simple, but removing it shortens the line" : "1 object part, 1 background part: simple";
  return { fg, bg, degree, simple, reason };
}

export const isSimple = (nb: ArrayLike<number>): boolean => simpleTest(nb).simple;

/** Yokoi's 8-connectivity number (equals 1 exactly for simple border points). */
export function yokoi8(nb: ArrayLike<number>): number {
  const c = (k: number) => 1 - (nb[k % 8] ? 1 : 0);
  let n = 0;
  for (const k of [0, 2, 4, 6]) n += c(k) - c(k) * c(k + 1) * c(k + 2);
  return n;
}

// lookup table of simplicity by neighbour code
const SIMPLE_LUT = new Uint8Array(256);
for (let code = 0; code < 256; code++) SIMPLE_LUT[code] = isSimple(Array.from({ length: 8 }, (_, k) => (code >> k) & 1)) ? 1 : 0;
export const isSimpleCode = (code: number): boolean => SIMPLE_LUT[code] === 1;

// ---------------------------------------------------------------------------------------
// Thinning

export interface ThinStep {
  pass: number;
  /** Index of the template (into the template list); −1 for a parallel pass. */
  template: number;
  /** Pixels removed by this application. */
  removed: number[];
}

export interface ThinResult {
  skeleton: Uint8Array;
  /** Number of passes over all templates (the last one removes nothing). */
  passes: number;
  steps: ThinStep[];
}

/**
 * Template thinning: apply each template in turn, A ← A \ (A ⊛ T) (all matches of one
 * template removed together), repeat over all templates until nothing changes.
 * parallel: evaluate all templates on the same image and remove their union at once
 * (no subiterations; this can cut lines two pixels thick).
 */
export function thin(m: A, w: number, h: number, opts: { templates?: Template[]; parallel?: boolean; maxPasses?: number; record?: boolean } = {}): ThinResult {
  const ts = (opts.templates ?? THIN_TEMPLATES).map(templateBits);
  // padded copy (one background pixel around), so neighbour codes need no bounds checks
  const W = w + 2, pad = new Uint8Array(W * (h + 2));
  const off = RING.map(([dx, dy]) => dy * W + dx);
  let pix: number[] = [];   // padded indices of object pixels
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) { const p = (y + 1) * W + x + 1; pad[p] = 1; pix.push(p); }
  const code = (p: number) => {
    let c = 0;
    for (let k = 0; k < 8; k++) if (pad[p + off[k]]) c |= 1 << k;
    return c;
  };
  const unpad = (p: number) => { const x = (p % W) - 1, y = Math.floor(p / W) - 1; return y * w + x; };
  // only pixels with a background neighbour can match (every template has a "0")
  const inB = new Uint8Array(pad.length);
  let border: number[] = pix.filter(p => code(p) !== 255);
  for (const p of border) inB[p] = 1;
  const removeAll = (rem: number[]) => {
    for (const p of rem) pad[p] = 0;
    for (const p of rem) for (const o of off) { const q = p + o; if (pad[q] && !inB[q]) { inB[q] = 1; border.push(q); } }
  };
  const steps: ThinStep[] = [];
  let pass = 0;
  for (; pass < (opts.maxPasses ?? 10000); ) {
    pass++;
    let changed = 0;
    if (opts.parallel) {
      const rem: number[] = [];
      for (const p of border) { if (!pad[p]) continue; const c = code(p); if (ts.some(t => (c & t.care) === t.want)) rem.push(p); }
      removeAll(rem);
      changed += rem.length;
      if (opts.record) steps.push({ pass, template: -1, removed: rem.map(unpad) });
    } else {
      ts.forEach((t, k) => {
        const rem: number[] = [];
        for (const p of border) if (pad[p] && (code(p) & t.care) === t.want) rem.push(p);
        removeAll(rem);
        changed += rem.length;
        if (opts.record) steps.push({ pass, template: k, removed: rem.map(unpad) });
      });
    }
    border = border.filter(p => pad[p]);
    if (!changed) break;
  }
  pix = pix.filter(p => pad[p]);
  const skeleton = new Uint8Array(w * h);
  for (const p of pix) skeleton[unpad(p)] = 1;
  return { skeleton, passes: pass, steps };
}

/**
 * Remove the simple pixels that are not end points, one at a time in raster order, until
 * none is left (cleans the redundant pixels template thinning leaves at some junctions).
 */
export function removeSimple(sk: Uint8Array, w: number, h: number): number {
  let removed = 0;
  for (let changed = true; changed;) {
    changed = false;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!sk[i]) continue;
      const c = neighbourCode(sk, w, h, x, y);
      if (SIMPLE_LUT[c] && (c & (c - 1)) !== 0) { sk[i] = 0; removed++; changed = true; }
    }
  }
  return removed;
}

/** One-pixel skeleton: template thinning, then removal of the remaining simple non-end pixels. */
export function skeletonize(m: A, w: number, h: number): Uint8Array {
  const sk = thin(m, w, h).skeleton;
  removeSimple(sk, w, h);
  return sk;
}

export interface SimpleThinStep { pass: number; dir: number; index: number; removed: boolean; test: SimpleTest; end: boolean }

/**
 * Sequential thinning by simple points: in each pass, for each direction (N, E, S, W),
 * take the border pixels whose neighbour in that direction is background (at the start of
 * this subiteration), visit them in raster order, and remove each one that is simple (and
 * not an end point, if keepEnds) on the current image. Repeat until nothing changes.
 */
export function thinSimple(m: A, w: number, h: number, opts: { keepEnds?: boolean; maxPasses?: number; record?: boolean } = {}): { skeleton: Uint8Array; passes: number; steps: SimpleThinStep[] } {
  const keepEnds = opts.keepEnds ?? true;
  const sk = Uint8Array.from(m as ArrayLike<number>, v => (v ? 1 : 0));
  const steps: SimpleThinStep[] = [];
  let pass = 0;
  for (; pass < (opts.maxPasses ?? 10000);) {
    pass++;
    let changed = 0;
    for (const dir of [0, 2, 4, 6]) {
      const [dx, dy] = RING[dir];
      // the border pixels on this side, as the image is at the start of the subiteration
      const cand: number[] = [];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (sk[y * w + x] && !isOn(sk, w, h, x + dx, y + dy)) cand.push(y * w + x);
      for (const i of cand) {
        const x = i % w, y = (i - x) / w;
        const nb = neighbours(sk, w, h, x, y), test = simpleTest(nb), end = test.degree === 1;
        const rem = test.simple && !(keepEnds && end);
        if (rem) { sk[i] = 0; changed++; }
        if (opts.record) steps.push({ pass, dir, index: i, removed: rem, test, end });
      }
    }
    if (!changed) break;
  }
  return { skeleton: sk, passes: pass, steps };
}

/** Zhang–Suen thinning (two parallel subiterations per pass). */
export function zhangSuen(m: A, w: number, h: number): { skeleton: Uint8Array; passes: number } {
  const sk = Uint8Array.from(m as ArrayLike<number>, v => (v ? 1 : 0));
  let passes = 0;
  for (;;) {
    passes++;
    let changed = 0;
    for (const sub of [0, 1]) {
      const rem: number[] = [];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!sk[i]) continue;
        const p = neighbours(sk, w, h, x, y);   // P2..P9 = N, NE, E, SE, S, SW, W, NW
        const B = p.reduce((s, v) => s + v, 0);
        if (B < 2 || B > 6) continue;
        let Acount = 0;
        for (let k = 0; k < 8; k++) if (!p[k] && p[(k + 1) % 8]) Acount++;
        if (Acount !== 1) continue;
        const [P2, , P4, , P6, , P8] = p;
        if (sub === 0 ? (P2 * P4 * P6 === 0 && P4 * P6 * P8 === 0) : (P2 * P4 * P8 === 0 && P2 * P6 * P8 === 0)) rem.push(i);
      }
      for (const i of rem) sk[i] = 0;
      changed += rem.length;
    }
    if (!changed) break;
  }
  return { skeleton: sk, passes };
}

// ---------------------------------------------------------------------------------------
// Topology helpers

/** Number of 8-connected object components and 4-connected holes (background components not touching the border). */
export function topology(m: A, w: number, h: number): { components: number; holes: number; euler: number } {
  const comp = label2d(m, w, h, 8).count;
  // pad by one background pixel so the outside is one component
  const W = w + 2, H = h + 2, inv = new Uint8Array(W * H).fill(1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) inv[(y + 1) * W + x + 1] = 0;
  const holes = label2d(inv, W, H, 4).count - 1;
  return { components: comp, holes, euler: comp - holes };
}

// ---------------------------------------------------------------------------------------
// Distance maps and the medial axis

export type Metric = "cityblock" | "chessboard" | "euclidean";

/** Distance from each object pixel to the nearest background pixel (1 on the boundary). */
export function distanceMap(m: A, w: number, h: number, metric: Metric): Float32Array {
  return metric === "euclidean" ? edt(m, w, h) : chamfer(m, w, h, metric);
}

/** Distance between two pixels in a metric. */
export function metricDist(dx: number, dy: number, metric: Metric): number {
  const ax = Math.abs(dx), ay = Math.abs(dy);
  return metric === "cityblock" ? ax + ay : metric === "chessboard" ? Math.max(ax, ay) : Math.hypot(ax, ay);
}

/**
 * Centres of maximal discs (the discrete medial axis, the ridges of DT): object pixels p
 * whose disc {q : d(p, q) < DT(p)} is not contained in the disc of a neighbour (4
 * neighbours for city-block, 8 otherwise). For city-block and chessboard this is the test
 * DT(q) ≥ DT(p) + d(p, q). For Euclidean discs, containment is tested on the pixel grid
 * itself (a lattice disc can fit inside a neighbour's even when the round discs do not),
 * with the squared distances, which are integers.
 */
export function medialAxis(m: A, w: number, h: number, metric: Metric, dt?: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(w * h);
  if (metric === "euclidean") {
    const D2 = edt(m, w, h, { squared: true });
    const cache = new Map<number, Int32Array>();
    // M[k] = max over lattice points v with |v|² < D of |v + δ_k|², for the 8 neighbour offsets δ_k
    const maxShift = (D: number): Int32Array => {
      let M = cache.get(D);
      if (M) return M;
      M = new Int32Array(8).fill(-1);
      const R = Math.floor(Math.sqrt(Math.max(0, D - 1)));
      for (let y = -R; y <= R; y++) {
        const r2 = D - 1 - y * y;
        if (r2 < 0) continue;
        const xm = Math.floor(Math.sqrt(r2));
        for (const x of [-xm, xm]) for (let k = 0; k < 8; k++) {
          const v = (x + RING[k][0]) ** 2 + (y + RING[k][1]) ** 2;
          if (v > M[k]) M[k] = v;
        }
      }
      cache.set(D, M);
      return M;
    };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!m[i]) continue;
      const D = Math.round(D2[i]), M = maxShift(D);
      let maximal = true;
      for (let k = 0; k < 8 && maximal; k++) {
        // q = p − δ_k contains p's disc if every point of it is closer than √D_q to q
        const xx = x - RING[k][0], yy = y - RING[k][1];
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        if (Math.round(D2[yy * w + xx]) > M[k]) maximal = false;
      }
      if (maximal) out[i] = 1;
    }
    return out;
  }
  const D = dt ?? distanceMap(m, w, h, metric);
  const nb: [number, number][] = metric === "cityblock" ? [[0, -1], [1, 0], [0, 1], [-1, 0]] : RING;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!m[i]) continue;
    let maximal = true;
    for (const [dx, dy] of nb) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      if (D[yy * w + xx] >= D[i] + metricDist(dx, dy, metric) - 1e-6) { maximal = false; break; }
    }
    if (maximal) out[i] = 1;
  }
  return out;
}

/**
 * Spread of the nearest boundary points: for each object pixel, the largest distance between
 * its nearest background pixel and that of one of its 4 neighbours. Small on the smooth
 * part of the shape, large where two boundary pieces meet (on the medial axis). Keeping the
 * axis pixels with spread ≥ λ removes the branches caused by small boundary details
 * (the "λ-medial axis").
 */
export function boundarySpread(m: A, w: number, h: number): Float32Array {
  const { nearest } = edtWithNearest(m, w, h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!m[i] || nearest[i] < 0) continue;
    const a = nearest[i], ax = a % w, ay = (a - ax) / w;
    let best = 0;
    for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const b = nearest[yy * w + xx];
      if (b < 0) continue;
      const d = Math.hypot((b % w) - ax, Math.floor(b / w) - ay);
      if (d > best) best = d;
    }
    out[i] = best;
  }
  return out;
}

/** Union of the discs {q : d(p, q) < r(p)} over the pixels p with axis[p] = 1. */
export function reconstructFromDiscs(axis: A, radius: A, w: number, h: number, metric: Metric): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!axis[i]) continue;
    const r = radius[i], R = Math.ceil(r);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      if (metricDist(dx, dy, metric) < r - 1e-4 * Math.max(1, r)) out[yy * w + xx] = 1;
    }
  }
  return out;
}

// 3 × 3 binary erosion/dilation by the cross or the square (outside = background)
function erode3(a: Uint8Array, w: number, h: number, square: boolean): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!a[i]) continue;
    let ok = !!(isOn(a, w, h, x - 1, y) && isOn(a, w, h, x + 1, y) && isOn(a, w, h, x, y - 1) && isOn(a, w, h, x, y + 1));
    if (ok && square) ok = !!(isOn(a, w, h, x - 1, y - 1) && isOn(a, w, h, x + 1, y - 1) && isOn(a, w, h, x - 1, y + 1) && isOn(a, w, h, x + 1, y + 1));
    if (ok) out[i] = 1;
  }
  return out;
}
function dilate3(a: Uint8Array, w: number, h: number, square: boolean): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!a[y * w + x]) continue;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!square && dx && dy) continue;
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h) out[yy * w + xx] = 1;
    }
  }
  return out;
}

export interface LantuejoulResult {
  /** Union of the S_r. */
  skeleton: Uint8Array;
  /** r of the S_r a pixel belongs to (−1 if none). */
  radius: Int32Array;
  /** Number of pixels in S_0, S_1, ... */
  counts: number[];
}

/**
 * Lantuéjoul's skeleton with B = the 3 × 3 cross (city-block) or square (chessboard):
 * S_r = (A ⊖ rB) \ ((A ⊖ rB) ∘ B), for r = 0, 1, ... until A ⊖ rB is empty.
 */
export function lantuejoul(m: A, w: number, h: number, se: "cross" | "square"): LantuejoulResult {
  const sq = se === "square";
  let E: Uint8Array = Uint8Array.from(m as ArrayLike<number>, v => (v ? 1 : 0));
  const skeleton = new Uint8Array(w * h), radius = new Int32Array(w * h).fill(-1), counts: number[] = [];
  for (let r = 0; ; r++) {
    let any = false;
    for (let i = 0; i < E.length; i++) if (E[i]) { any = true; break; }
    if (!any) break;
    const En = erode3(E, w, h, sq), O = dilate3(En, w, h, sq);
    let n = 0;
    for (let i = 0; i < E.length; i++) if (E[i] && !O[i]) { skeleton[i] = 1; radius[i] = r; n++; }
    counts.push(n);
    E = En;
  }
  return { skeleton, radius, counts };
}

/** Reconstruction ∪_r S_r ⊕ rB: each S_r pixel gets a diamond (cross) or square of radius r. */
export function lantuejoulReconstruct(res: LantuejoulResult, w: number, h: number, se: "cross" | "square"): Uint8Array {
  const r1 = Float32Array.from(res.radius, r => r + 1);   // disc {d < r + 1} = {d ≤ r}
  return reconstructFromDiscs(res.skeleton, r1, w, h, se === "cross" ? "cityblock" : "chessboard");
}

// ---------------------------------------------------------------------------------------
// Skeleton graph

/** Pixel classes of a skeleton: 0 background, 1 end point, 2 regular, 3 junction, 4 isolated. */
export function classifySkeleton(sk: A, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!sk[i]) continue;
    const d = degreeAt(sk, w, h, x, y);
    out[i] = d === 0 ? 4 : d === 1 ? 1 : d === 2 ? 2 : 3;
  }
  return out;
}

export interface SkelNode {
  id: number;
  kind: "end" | "junction" | "loop" | "isolated";
  /** Pixel indices of the node (a junction cluster can have several). */
  pixels: number[];
  /** Mean position of its pixels (pixel coordinates). */
  x: number;
  y: number;
  /** Number of edge ends at the node (a self-loop counts twice). */
  degree: number;
}

export interface SkelEdge {
  id: number;
  a: number;
  b: number;
  /** Pixel path from a pixel of node a to a pixel of node b (both included). */
  path: number[];
}

export interface SkelGraph {
  w: number;
  h: number;
  nodes: SkelNode[];
  edges: SkelEdge[];
  /** 8-connected components of the skeleton. */
  components: number;
  /** Node id per pixel (−1 for path pixels and background). */
  nodeOf: Int32Array;
}

/**
 * Skeleton → graph. Nodes: end points (one neighbour) and clusters of 8-adjacent junction
 * pixels (three or more neighbours). Edges: the 8-connected pieces of the skeleton left
 * after removing the node pixels, each joined to the node(s) it touches; plus direct links
 * between adjacent pixels of different nodes. A closed loop without nodes gets one "loop"
 * node. An isolated pixel is a node without edges.
 */
export function skeletonGraph(sk: A, w: number, h: number): SkelGraph {
  const cls = classifySkeleton(sk, w, h), n = w * h;
  const nodeOf = new Int32Array(n).fill(-1), nodes: SkelNode[] = [];
  const addNode = (kind: SkelNode["kind"], pixels: number[]) => {
    const id = nodes.length;
    let sx = 0, sy = 0;
    for (const i of pixels) { sx += i % w; sy += Math.floor(i / w); nodeOf[i] = id; }
    nodes.push({ id, kind, pixels, x: sx / pixels.length, y: sy / pixels.length, degree: 0 });
    return id;
  };
  // junction clusters (8-connected), end points, isolated pixels, in raster order
  for (let i = 0; i < n; i++) {
    if (!cls[i] || nodeOf[i] >= 0) continue;
    if (cls[i] === 1) addNode("end", [i]);
    else if (cls[i] === 4) addNode("isolated", [i]);
    else if (cls[i] === 3) {
      const pix: number[] = [], st = [i];
      const seen = new Set([i]);
      while (st.length) {
        const p = st.pop()!;
        pix.push(p);
        const x = p % w, y = (p - x) / w;
        for (const [dx, dy] of RING) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (cls[q] === 3 && !seen.has(q)) { seen.add(q); st.push(q); }
        }
      }
      addNode("junction", pix.sort((a, b) => a - b));
    }
  }
  const edges: SkelEdge[] = [];
  const nbrs = (p: number): number[] => {
    const x = p % w, y = (p - x) / w, out: number[] = [];
    for (const [dx, dy] of RING) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && sk[yy * w + xx]) out.push(yy * w + xx);
    }
    return out;
  };
  const visited = new Uint8Array(n);
  const trace = (start: number): number[] => {
    const path = [start];
    visited[start] = 1;
    for (let cur = start; ;) {
      const next = nbrs(cur).find(q => nodeOf[q] < 0 && !visited[q]);
      if (next === undefined) break;
      visited[next] = 1; path.push(next); cur = next;
    }
    return path;
  };
  const addEdge = (path: number[]) => {
    const p0 = path[0], pk = path[path.length - 1];
    const n0 = nbrs(p0).filter(q => nodeOf[q] >= 0), nk = nbrs(pk).filter(q => nodeOf[q] >= 0);
    let qa = n0[0], qb = nk.find(q => nodeOf[q] !== nodeOf[qa]) ?? nk[0];
    if (path.length === 1 && nk.length > 1) qb = nk.find(q => nodeOf[q] !== nodeOf[qa]) ?? nk[nk.length - 1];
    if (qa === undefined || qb === undefined) return;
    const a = nodeOf[qa], b = nodeOf[qb];
    edges.push({ id: edges.length, a, b, path: [qa, ...path, qb] });
    nodes[a].degree++; nodes[b].degree++;
  };
  // pieces with an end (a path pixel touching a node)
  for (let i = 0; i < n; i++) {
    if (!sk[i] || nodeOf[i] >= 0 || visited[i]) continue;
    const ns = nbrs(i);
    const inPiece = ns.filter(q => nodeOf[q] < 0);
    if (inPiece.length <= 1 && ns.some(q => nodeOf[q] >= 0)) addEdge(trace(i));
  }
  // closed loops without nodes
  for (let i = 0; i < n; i++) {
    if (!sk[i] || nodeOf[i] >= 0 || visited[i]) continue;
    const id = addNode("loop", [i]);
    visited[i] = 1;
    const rest = nbrs(i).filter(q => nodeOf[q] < 0 && !visited[q]);
    if (!rest.length) continue;
    const path = trace(rest[0]);
    edges.push({ id: edges.length, a: id, b: id, path: [i, ...path, i] });
    nodes[id].degree += 2;
  }
  // direct links between adjacent pixels of different nodes
  const linked = new Set<string>();
  for (const nd of nodes) for (const p of nd.pixels) for (const q of nbrs(p)) {
    const o = nodeOf[q];
    if (o < 0 || o === nd.id) continue;
    const key = nd.id < o ? `${nd.id}-${o}` : `${o}-${nd.id}`;
    if (linked.has(key)) continue;
    linked.add(key);
    edges.push({ id: edges.length, a: nd.id, b: o, path: [p, q] });
    nd.degree++; nodes[o].degree++;
  }
  return { w, h, nodes, edges, components: label2d(sk, w, h, 8).count, nodeOf };
}

/** Cyclomatic number E − V + C (number of independent cycles = holes of the skeleton). */
export const cyclomatic = (g: SkelGraph): number => g.edges.length - g.nodes.length + g.components;

/** Freeman chain counts of a pixel path: axial steps, diagonal steps, corners (direction changes). */
export function chainCounts(path: number[], w: number): { ne: number; no: number; nc: number } {
  let ne = 0, no = 0, nc = 0, prev = -1;
  for (let k = 1; k < path.length; k++) {
    const a = path[k - 1], b = path[k], dx = (b % w) - (a % w), dy = Math.floor(b / w) - Math.floor(a / w);
    if (dx === 0 && dy === 0) continue;
    const dir = RING.findIndex(([u, v]) => u === dx && v === dy);
    if (dx && dy) no++; else ne++;
    if (prev >= 0 && dir !== prev) nc++;
    prev = dir;
  }
  return { ne, no, nc };
}

export type LengthEstimator = "count" | "weighted" | "corrected";
export const ESTIMATOR_NAMES: Record<LengthEstimator, string> = { count: "pixel steps", weighted: "1 / √2", corrected: "corrected" };

/** Length from chain counts: count = every step 1; weighted = 1 axial, √2 diagonal;
 *  corrected = Vossepoel & Smeulders (1982): 0.980 n_e + 1.406 n_o − 0.091 n_c. */
export function lengthFromCounts(c: { ne: number; no: number; nc: number }, est: LengthEstimator): number {
  if (est === "count") return c.ne + c.no;
  if (est === "weighted") return c.ne + Math.SQRT2 * c.no;
  return Math.max(0, 0.980 * c.ne + 1.406 * c.no - 0.091 * c.nc);
}

export const pathLength = (path: number[], w: number, est: LengthEstimator): number => lengthFromCounts(chainCounts(path, w), est);

/** Straight-line distance between the two ends of a path. */
export function chord(path: number[], w: number): number {
  const a = path[0], b = path[path.length - 1];
  return Math.hypot((b % w) - (a % w), Math.floor(b / w) - Math.floor(a / w));
}

/** Tortuosity = path length / chord (NaN for a closed loop). */
export function tortuosity(path: number[], w: number, est: LengthEstimator = "corrected"): number {
  const c = chord(path, w);
  return c > 0 ? pathLength(path, w, est) / c : NaN;
}

/** Mean of a map (e.g. the distance transform = local radius) along a path. */
export function meanAlong(path: number[], map: ArrayLike<number>): number {
  let s = 0;
  for (const i of path) s += map[i];
  return path.length ? s / path.length : NaN;
}

/** Total length of all edges. */
export function totalLength(g: SkelGraph, est: LengthEstimator): number {
  return g.edges.reduce((s, e) => s + pathLength(e.path, g.w, est), 0);
}

/**
 * Pruning: remove terminal edges (end point ↔ junction) shorter than k (or shorter than
 * k · DT at the junction, if `dt` is given), together with junction pixels left dangling,
 * re-thin, and repeat `passes` times (pruning can expose new short terminal edges).
 * A junction always keeps at least two edges (the longest ones).
 */
export function prune(sk: A, w: number, h: number, k: number, opts: { est?: LengthEstimator; dt?: ArrayLike<number>; passes?: number } = {}): { skeleton: Uint8Array; removed: number } {
  let cur: Uint8Array = Uint8Array.from(sk as ArrayLike<number>, v => (v ? 1 : 0));
  let removed = 0;
  const est = opts.est ?? "weighted";
  for (let pass = 0; pass < (opts.passes ?? 2); pass++) {
    const g = skeletonGraph(cur, w, h), clusterPix = new Set<number>();
    let any = false;
    // short terminal edges, grouped by their junction
    const cand = new Map<number, { e: SkelEdge; len: number }[]>();
    for (const e of g.edges) {
      const A = g.nodes[e.a], B = g.nodes[e.b];
      const endN = A.kind === "end" && B.kind === "junction" ? A : B.kind === "end" && A.kind === "junction" ? B : null;
      if (!endN) continue;
      const jn = endN === A ? B : A;
      const len = pathLength(e.path, w, est);
      const jpix = endN === A ? e.path[e.path.length - 1] : e.path[0];
      const lim = opts.dt ? k * opts.dt[jpix] : k;
      if (len >= lim) continue;
      if (!cand.has(jn.id)) cand.set(jn.id, []);
      cand.get(jn.id)!.push({ e, len });
    }
    for (const [jid, list] of cand) {
      const jn = g.nodes[jid];
      // a junction keeps at least two edges: spare the longest candidates if needed
      list.sort((p, q) => p.len - q.len);
      const nRemove = Math.max(0, Math.min(list.length, jn.degree - 2));
      for (const { e } of list.slice(0, nRemove)) {
        for (const p of e.path) if (g.nodeOf[p] !== jid) cur[p] = 0;
        for (const p of jn.pixels) clusterPix.add(p);
        removed++; any = true;
      }
    }
    if (!any) break;
    // dangling former junction pixels
    for (let changed = true; changed;) {
      changed = false;
      for (const p of clusterPix) {
        if (!cur[p]) continue;
        const x = p % w, y = (p - x) / w;
        if (degreeAt(cur, w, h, x, y) <= 1 && clusterPix.size > 1) {
          // keep it if it is the only pixel left of its cluster (it may be a real end)
          let others = 0;
          for (const q of clusterPix) if (q !== p && cur[q]) others++;
          if (!others) continue;
          cur[p] = 0; changed = true;
        }
      }
    }
    cur = skeletonize(cur, w, h);
  }
  return { skeleton: cur, removed };
}

// ---------------------------------------------------------------------------------------
// Sholl analysis

/** Number of times the polylines cross the circle of radius r around (cx, cy). */
export function shollCrossings(lines: [number, number][][], cx: number, cy: number, r: number): number {
  let n = 0;
  for (const L of lines) for (let k = 1; k < L.length; k++) {
    const d0 = Math.hypot(L[k - 1][0] - cx, L[k - 1][1] - cy) - r, d1 = Math.hypot(L[k][0] - cx, L[k][1] - cy) - r;
    if ((d0 < 0 && d1 >= 0) || (d0 >= 0 && d1 < 0)) n++;
  }
  return n;
}

/** Sholl profile N(r) for the given radii. */
export function shollProfile(lines: [number, number][][], cx: number, cy: number, radii: number[]): number[] {
  return radii.map(r => shollCrossings(lines, cx, cy, r));
}

/** Edge paths of a skeleton graph as polylines of pixel coordinates. */
export function edgePolylines(g: SkelGraph): [number, number][][] {
  return g.edges.map(e => e.path.map(i => [i % g.w, Math.floor(i / g.w)] as [number, number]));
}
