// A toy promptable segmenter (no DOM), built like the real one in three parts:
//  - "image encoder", run ONCE per image: smoothed copies of the image at three scales (the
//    intensity channels of the random-forest day's feature stack, lib/featurestack.ts) and a
//    background map (a large minimum filter of the coarsest copy, smoothed);
//  - "prompt encoder": what the clicks see: the value under each click and the local contrast
//    (brightest nearby pixel minus background);
//  - "mask decoder", run PER PROMPT, three candidates (as the real decoder's three outputs):
//      part   : pixels that look like the clicked pixel (fine scale, |L1 − L1(click)| small)
//      object : pixels brighter than halfway between background and the brightest nearby
//               pixel (fine scale): one nucleus
//      group  : the same on the coarsest scale at a low level: touching nuclei merge, a cell's
//               dim cytoplasm joins its nucleus
//    Each is a logit map ℓ (in units of 5 % of the local contrast), so ℓ = 0 is the border; the
//    candidate is the connected piece of ℓ > 0 at the click. Negative clicks claim the pixels
//    nearer to them than to the positive side (a toy rule); a box sets everything outside
//    it below zero and its centre acts as the click.
// The "predicted score" of a candidate is a PROXY (the real network has a trained head that
// regresses the true IoU): the stability of the mask at a wider threshold offset (±2 logits),
// i.e. how steep the logit map is at the mask's border. The demo shows it next to the true IoU,
// so one can see when it misleads.
// Automatic masks: a grid of single positive clicks, three candidates each, then the filters
// of the real pipeline (score, stability at ±1, box NMS), from samnet/post.ts.
import { featureStack } from "../lib/featurestack";
import { gaussianBlur } from "../lib/imageops";
import { label2d } from "../lib/label";
import { iouTable, match, objectScores, type ObjectScores } from "../lib/eval";
import * as P from "../samnet/post";

export const SIGMAS = [1, 2.5, 5];
export const LEVELS = ["part", "object", "group"] as const;
export type Level = typeof LEVELS[number];

/** Logits per unit of local contrast (20: one logit = 5 % of the contrast). */
export const GAIN = 20;
/** part: half-width of the band of "similar" values; object, group: level between background
 *  (0) and the brightest nearby pixel (1). */
export const LEVEL_PARAMS = { partBand: 0.12, objectLevel: 0.55, groupLevel: 0.22 };

export interface Encoded {
  w: number; h: number;
  /** Smoothed image at SIGMAS[0], [1], [2]. */
  L: Float32Array[];
  /** Background estimate. */
  B: Float32Array;
  ms: number;
}

export interface Point { x: number; y: number; label: 1 | 0 }
export type Box = [number, number, number, number];

export interface CandidateMask {
  level: Level;
  logits: Float32Array;
  mask: Uint8Array;
  area: number;
  /** Proxy score: stability at offset 2. */
  score: number;
  /** Stability at offset 1 (the filter of automatic masks). */
  stability: number;
}

export interface Decoded { cands: CandidateMask[]; ms: number; best: number }

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Separable minimum filter with a (2r+1)-wide square window, edges replicated. */
export function minFilter(a: Float32Array, w: number, h: number, r: number): Float32Array {
  const t = new Float32Array(w * h), o = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = Infinity;
    for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) m = Math.min(m, a[y * w + k]);
    t[y * w + x] = m;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = Infinity;
    for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) m = Math.min(m, t[k * w + x]);
    o[y * w + x] = m;
  }
  return o;
}

/** The "image encoder": smoothed copies at three scales and a background map. */
export function encode(img: Float32Array, w: number, h: number): Encoded {
  const t0 = now();
  const S = featureStack({ width: w, height: h, data: img }, ["intensity"], SIGMAS);
  const N = w * h;
  const L = SIGMAS.map((_, i) => S.data.slice(i * N, (i + 1) * N));
  const B = gaussianBlur({ width: w, height: h, data: minFilter(L[2], w, h, 20) }, 10).data;
  return { w, h, L, B, ms: now() - t0 };
}

/** Prompt encoder: value under the click (fine scale), background there, and the brightest
 *  medium-scale value within `r` px (medium: so a tiny bright nucleolus does not count). */
export function promptToken(E: Encoded, x: number, y: number, r = 6): { v: number; b: number; c: number } {
  const { w, h } = E, cx = Math.round(x), cy = Math.round(y);
  const p = cy * w + cx;
  let c = -Infinity;
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    if (dx * dx + dy * dy > r * r) continue;
    const xx = cx + dx, yy = cy + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
    c = Math.max(c, E.L[1][yy * w + xx]);
  }
  return { v: E.L[0][p], b: E.B[p], c };
}

/** Keep the connected piece (8-connected) of logit > 0 that contains a seed; a seed outside
 *  every piece snaps to the nearest piece within `snap` px. Logits of other pieces are pushed
 *  to −margin, so the mask is exactly the kept piece. */
export function keepPieces(logits: Float32Array, w: number, h: number, seeds: [number, number][], snap = 5, margin = 4): Uint8Array {
  const pos = new Uint8Array(w * h);
  for (let p = 0; p < pos.length; p++) pos[p] = logits[p] > 0 ? 1 : 0;
  const L = label2d(pos, w, h, 8);
  const keep = new Set<number>();
  for (const [sx, sy] of seeds) {
    const cx = Math.round(sx), cy = Math.round(sy);
    let id = 0, best = Infinity;
    for (let dy = -snap; dy <= snap; dy++) for (let dx = -snap; dx <= snap; dx++) {
      const xx = cx + dx, yy = cy + dy, d = dx * dx + dy * dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h || d > snap * snap) continue;
      const v = L.data[yy * w + xx];
      if (v && d < best) { best = d; id = v; }
    }
    if (id) keep.add(id);
  }
  const mask = new Uint8Array(w * h);
  for (let p = 0; p < mask.length; p++) {
    if (L.data[p] && keep.has(L.data[p])) mask[p] = 1;
    else if (logits[p] > -margin) logits[p] = Math.min(logits[p], -margin);
  }
  return mask;
}

/** Stability: area(ℓ > off) / area(ℓ > −off) (the IoU of the two nested masks). */
export function stability(logits: ArrayLike<number>, off: number): number {
  return P.stabilityScore(logits, 0, off);
}

/** The mask decoder: three candidates for one set of prompts. */
export function decode(E: Encoded, points: Point[], box: Box | null): Decoded {
  const t0 = now();
  const { w, h } = E, N = w * h;
  const pos = points.filter((p) => p.label === 1);
  const neg = points.filter((p) => p.label === 0);
  const seeds: [number, number][] = pos.length ? pos.map((p) => [p.x, p.y])
    : box ? [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]] : [];
  const cands: CandidateMask[] = [];
  if (!seeds.length) return { cands, ms: now() - t0, best: -1 };
  const toks = seeds.map(([x, y]) => promptToken(E, x, y, box ? Math.max(6, Math.min(box[2] - box[0], box[3] - box[1]) / 2) : 6));
  // Contrast: the brightest nearby pixel above the background (at least 15 grey levels).
  const tok = toks.reduce((a, t) => (t.c - t.b > a.c - a.b ? t : a));
  const K = Math.max(15, tok.c - tok.b);
  const maps: Float32Array[] = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
  const L1 = E.L[0], L3 = E.L[2], B = E.B;
  for (let p = 0; p < N; p++) {
    let part = -Infinity;
    for (const t of toks) part = Math.max(part, GAIN * (LEVEL_PARAMS.partBand - Math.abs(L1[p] - t.v) / K));
    maps[0][p] = part;
    maps[1][p] = GAIN * ((L1[p] - B[p]) / K - LEVEL_PARAMS.objectLevel);
    maps[2][p] = GAIN * ((L3[p] - B[p]) / K - LEVEL_PARAMS.groupLevel);
  }
  if (neg.length) {
    // Negative clicks claim the pixels nearer to them than to the positive side, which is
    // represented by the centre of the object-level mask of the positive clicks alone.
    const obj = keepPieces(maps[1].slice(), w, h, seeds);
    let n = 0, mx = 0, my = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (obj[y * w + x]) { n++; mx += x; my += y; }
    const anchors: [number, number][] = n ? [[mx / n, my / n], ...seeds] : seeds;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let dp = Infinity, dn = Infinity;
      for (const [sx, sy] of anchors) dp = Math.min(dp, Math.hypot(x - sx, y - sy));
      for (const n of neg) dn = Math.min(dn, Math.hypot(x - n.x, y - n.y));
      const v = 0.5 * (dn - dp); // logits: half a logit per pixel from the bisector
      const p = y * w + x;
      for (const m of maps) if (v < m[p]) m[p] = v;
    }
  }
  if (box) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dx = Math.max(box[0] - x, x - box[2], 0), dy = Math.max(box[1] - y, y - box[3], 0);
      if (dx || dy) { const p = y * w + x, v = -2 * (1 + Math.hypot(dx, dy)); for (const m of maps) if (v < m[p]) m[p] = v; }
    }
  }
  maps.forEach((logits, k) => {
    const mask = keepPieces(logits, w, h, seeds);
    let area = 0; for (let p = 0; p < N; p++) area += mask[p];
    cands.push({ level: LEVELS[k], logits, mask, area, score: area ? stability(logits, 2) : 0, stability: area ? stability(logits, 1) : 0 });
  });
  let best = 0;
  cands.forEach((c, k) => { if (c.score > cands[best].score) best = k; });
  return { cands, ms: now() - t0, best };
}

/** The true object at one level under (or within `reach` px of) a click; 0 if none. */
export function truthAt(truth: Int32Array, w: number, x: number, y: number, reach = 3): number {
  const h = truth.length / w;
  let id = 0, best = Infinity;
  const cx = Math.round(x), cy = Math.round(y);
  for (let dy = -reach; dy <= reach; dy++) for (let dx = -reach; dx <= reach; dx++) {
    const xx = cx + dx, yy = cy + dy;
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
    const v = truth[yy * w + xx], d = dx * dx + dy * dy;
    if (v && d < best) { best = d; id = v; }
  }
  return id;
}

/** IoU of a mask with the true object `id` of a label map. */
export function maskIoUWith(mask: Uint8Array, truth: Int32Array, id: number): number {
  let i = 0, u = 0;
  for (let p = 0; p < mask.length; p++) { const a = mask[p] ? 1 : 0, b = truth[p] === id ? 1 : 0; i += a & b; u += a | b; }
  return u ? i / u : 0;
}

/** IoU with the true object at a level that the click falls in; -1 when there is none. */
export function trueIoU(mask: Uint8Array, truth: Int32Array, w: number, x: number, y: number, reach = 3): number {
  const id = truthAt(truth, w, x, y, reach);
  return id ? maskIoUWith(mask, truth, id) : -1;
}

// ---- Automatic masks ----------------------------------------------------------------------

export interface AutoResult { cands: P.Candidate[]; ms: number; points: number }

/** Decode the points k0..k1-1 of an n x n grid (so the demo can run it in slices). */
export function autoSlice(E: Encoded, n: number, k0: number, k1: number, out: P.Candidate[]): void {
  const { w, h } = E;
  const grid = P.pointGrid(n);
  for (let i = k0; i < Math.min(k1, grid.length); i++) {
    const x = grid[i][0] * w - 0.5, y = grid[i][1] * h - 0.5;
    const d = decode(E, [{ x, y, label: 1 }], null);
    d.cands.forEach((c, k) => {
      const mb = P.maskBox(c.mask, w, h);
      out.push({ point: i, output: k, px: x, py: y, iou: c.score, stability: c.stability,
        mask: mb ? c.mask : null, box: mb ? mb.box : null, area: mb ? mb.area : 0 });
    });
  }
}

/** A grid of single positive clicks, three candidates each (score = the proxy). */
export function autoCandidates(E: Encoded, n: number): AutoResult {
  const t0 = now();
  const cands: P.Candidate[] = [];
  autoSlice(E, n, 0, n * n, cands);
  return { cands, ms: now() - t0, points: n * n };
}

export interface AutoFilters { score: number; stability: number; nms: number }
export const AUTO_DEFAULT: AutoFilters = { score: 0.8, stability: 0.9, nms: 0.7 };

/** The filters of automatic mask generation (samnet/post.ts), plus F1 against the nuclei. */
export function filterAuto(cands: P.Candidate[], points: number, f: AutoFilters, truth: Int32Array | null, w: number, h: number) {
  const r = P.filterCandidates(cands, { pointsPerSide: Math.round(Math.sqrt(points)), predIou: f.score, stability: f.stability, stabilityOffset: 1, nms: f.nms });
  let scores: ObjectScores | null = null;
  if (truth) scores = objectScores(match(iouTable(truth, P.labelImage(r.kept, w, h)), 0.5));
  return { ...r, scores };
}
