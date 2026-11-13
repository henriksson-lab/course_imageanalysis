// Scenes, the simulated single-shot head and its decoding for the yolodecode demo
// (Networks for cells and objects day, unit yolo/). No DOM, so demos/test/yolo-test.ts uses it.
//
// The input is 512 × 512 pixels, like the real network at its default size, so the numbers
// match the lecture: three grids with strides 8, 16 and 32 (64², 32², 16² = 5376 cells, in
// that order, each row-major). Every cell outputs 4 × 16 distance-bin logits (left, top, right,
// bottom: how many strides from the cell centre to the box side) and one logit per class
// ("cell", "debris"). Decoding: softmax per side, distance = expected bin, box = centre ±
// distances × stride; class probability = sigmoid(logit). A candidate is a (cell, class) pair
// above the score threshold; greedy NMS (lib/boxes.ts) per class or across classes.
//
// The "network" is simulated from the ground truth, the way it is trained: a cell is
// responsible for an object when its centre lies inside the object's box; its score grows with
// how central the cell is (FCOS centreness) and how well the stride suits the object's size; its
// distance distributions are peaks at the true distances (clamped to 0..15 bins) plus jitter.
// An "anchors" head (older detectors) predicts corrections (t_x, t_y, t_w, t_h) to one square
// template box per cell (side 4 strides); only anchors with IoU ≥ 0.5 with an object learn it.
import { type Box, boxIou, boxW, boxH, boxFromCentre, nmsGeneric, encodeBox, decodeBox, boxIouTable, type NmsResult } from "../lib/boxes";
import { match } from "../lib/eval";
import { createGrey, gaussianBlur, type Grey } from "../lib/imageops";
import { addGaussianNoise, type Ellipse } from "../lib/synth";
import { rng } from "../lib/ui";
import { softmax } from "../yolonet/post";

export const SIZE = 512;                 // input side in pixels
export const IMG = 256;                  // rendered image side (1 image pixel = 2 input pixels)
export const STRIDES = [8, 16, 32] as const;
export const BINS = 16;                  // distance bins 0..15 per side
export const CLASSES = ["cell", "debris"] as const;
export const NC = CLASSES.length;
export const NO = 4 * BINS + NC;         // raw numbers per cell (66)
export const ANCHOR_STRIDES = 4;         // anchor side in strides

export interface Grid { stride: number; g: number; start: number; count: number }

/** The three grids of a square input of side `size` (a multiple of 32). */
export function grids(size = SIZE): Grid[] {
  let start = 0;
  return STRIDES.map((stride) => { const g = size / stride, s = { stride, g, start, count: g * g }; start += g * g; return s; });
}
export const GRIDS = grids();
export const NCELLS = GRIDS.reduce((a, s) => a + s.count, 0);   // 5376

/** Number of output cells and of output numbers for a square input (classes C). */
export function outputSize(size: number, nc: number): { cells: number; numbers: number } {
  const cells = grids(size).reduce((a, s) => a + s.count, 0);
  return { cells, numbers: cells * (4 * BINS + nc) };
}

export interface CellInfo { scale: number; stride: number; i: number; j: number; cx: number; cy: number }

export function cellInfo(a: number): CellInfo {
  for (let k = 0; k < GRIDS.length; k++) {
    const s = GRIDS[k];
    if (a < s.start + s.count) {
      const r = a - s.start, i = Math.floor(r / s.g), j = r % s.g;
      return { scale: k, stride: s.stride, i, j, cx: (j + 0.5) * s.stride, cy: (i + 0.5) * s.stride };
    }
  }
  throw new Error("cell out of range");
}
export const cellIndex = (scale: number, i: number, j: number): number => GRIDS[scale].start + i * GRIDS[scale].g + j;
/** The cell of a given stride whose square contains the point (x, y). */
export function cellAt(scale: number, x: number, y: number): number {
  const s = GRIDS[scale];
  const j = Math.min(s.g - 1, Math.max(0, Math.floor(x / s.stride))), i = Math.min(s.g - 1, Math.max(0, Math.floor(y / s.stride)));
  return cellIndex(scale, i, j);
}

// ---- Scenes ---------------------------------------------------------------------------------

export interface Obj { e: Ellipse; cls: number; box: Box; alt?: number; name?: string }
export type SceneId = "mixed" | "large" | "rods";
export interface Scene { id: SceneId; objs: Obj[]; image: Grey }

/** Axis-aligned box of an ellipse. */
export function ellipseBox(e: Ellipse): Box {
  const c = Math.cos(e.angle), s = Math.sin(e.angle);
  const hx = Math.sqrt(e.a * e.a * c * c + e.b * e.b * s * s), hy = Math.sqrt(e.a * e.a * s * s + e.b * e.b * c * c);
  return { x0: e.x - hx, y0: e.y - hy, x1: e.x + hx, y1: e.y + hy };
}

const obj = (x: number, y: number, a: number, b: number, angle: number, cls: number, extra: Partial<Obj> = {}): Obj => {
  const e = { x, y, a, b, angle };
  return { e, cls, box: ellipseBox(e), ...extra };
};

function sceneObjects(id: SceneId): Obj[] {
  if (id === "mixed") return [
    obj(117, 96, 33, 26, 0, 0, { name: "B1 cell" }),            // box (84, 70)–(150, 122)
    obj(300, 92, 30, 24, 0.3, 0), obj(422, 160, 34, 27, -0.4, 0), obj(250, 250, 62, 48, 0.8, 0),
    obj(392, 336, 36, 26, 0.2, 0), obj(128, 412, 54, 44, -0.6, 0), obj(300, 450, 29, 24, 0, 0),
    obj(128, 250, 27, 22, 1.2, 0),
    obj(32, 160, 6, 6, 0, 0, { name: "small cell" }),           // box [26, 38] × [154, 166]: only stride-8 centres inside
    obj(470, 44, 8, 6, 0.4, 1), obj(210, 372, 7, 5, 0.5, 1), obj(468, 462, 9, 7, 1.0, 1),
    obj(56, 318, 14, 11, 0.3, 1, { alt: -0.5, name: "debris that looks like a cell" }),
  ];
  if (id === "large") return [
    obj(256, 260, 230, 150, 0, 0, { name: "large cell" }),     // box [26, 486] × [110, 410]
    obj(64, 62, 28, 22, 0.4, 0), obj(448, 64, 30, 24, -0.3, 0),
  ];
  // rods: a palisade of touching diagonal rods (their boxes overlap their neighbours' by ≈ 0.46)
  const o: Obj[] = [];
  for (let k = -3; k <= 3; k++) o.push(obj(256 - 21.2 * k, 256 + 21.2 * k, 70, 14, Math.PI / 4, 0));
  o.push(obj(80, 80, 28, 24, 0, 0), obj(432, 432, 30, 25, 0.5, 0));
  return o;
}

function render(objs: Obj[], seed: number): Grey {
  const img = createGrey(IMG, IMG, 30), r = rng(seed), f = IMG / SIZE;
  objs.forEach((o) => {
    const level = o.cls === 1 ? 235 : 140 + 60 * r.unif();
    const { x, y, a, b, angle } = o.e, c = Math.cos(angle), s = Math.sin(angle);
    const B = ellipseBox(o.e);
    for (let py = Math.max(0, Math.floor(B.y0 * f) - 1); py <= Math.min(IMG - 1, Math.ceil(B.y1 * f) + 1); py++)
      for (let px = Math.max(0, Math.floor(B.x0 * f) - 1); px <= Math.min(IMG - 1, Math.ceil(B.x1 * f) + 1); px++) {
        const X = (px + 0.5) / f - x, Y = (py + 0.5) / f - y;
        const u = (X * c + Y * s) / a, v = (-X * s + Y * c) / b;
        const q = u * u + v * v;
        if (q <= 1) {
          // cells: brighter rim-less body with a darker core (a nucleus); debris: flat and bright
          const val = o.cls === 1 ? level : level * (0.85 + 0.25 * Math.exp(-3 * q));
          img.data[py * IMG + px] = Math.max(img.data[py * IMG + px], val);
        }
      }
  });
  return addGaussianNoise(gaussianBlur(img, 0.8), 7, seed + 5);
}

const SEEDS: Record<SceneId, number> = { mixed: 3, large: 5, rods: 7 };
export function makeScene(id: SceneId): Scene {
  const objs = sceneObjects(id);
  return { id, objs, image: render(objs, SEEDS[id]) };
}

// ---- The simulated network ------------------------------------------------------------------

export interface SimParams { boxNoise: number; scoreNoise: number; seed: number }
export const SIM_DEFAULTS: SimParams = { boxNoise: 0.25, scoreNoise: 0.6, seed: 11 };

export interface Sim {
  scene: Scene;
  /** Raw output of the distance head: NCELLS × NO (4 × 16 bin logits, then NC class logits). */
  raw: Float32Array;
  /** Responsible object of every cell (−1: background) and its exact distances in strides (unclamped). */
  owner: Int32Array;
  target: Float32Array;
  /** The anchor head: t = (t_x, t_y, t_w, t_h) per cell, class logits, responsible object, its anchor IoU. */
  aT: Float32Array; aLogit: Float32Array; aOwner: Int32Array; aIou: Float32Array;
  /** Per object: best anchor IoU over all anchors (≥ 0.5: learnable). */
  bestAnchor: number[];
}

/** FCOS centreness of a point with distances l, t, r, b to the box sides (1 at the centre). */
export function centreness(l: number, t: number, r: number, b: number): number {
  if (l <= 0 || t <= 0 || r <= 0 || b <= 0) return 0;
  return Math.sqrt((Math.min(l, r) / Math.max(l, r)) * (Math.min(t, b) / Math.max(t, b)));
}

/** How well stride s suits an object of side S (largest box side): 1 at S = 10 s, falling off in log2. */
export const strideFit = (S: number, s: number): number => Math.exp(-((Math.log2(S / (10 * s))) ** 2) / (2 * 0.8 * 0.8));

export const anchorBox = (a: number): Box => {
  const c = cellInfo(a);
  return boxFromCentre(c.cx, c.cy, ANCHOR_STRIDES * c.stride, ANCHOR_STRIDES * c.stride);
};

/** Distance-bin logits for a peak at d (bins) of width w: a Gaussian on the bins, clamped to 0..15. */
export function peakLogits(d: number, w = 0.7): number[] {
  const dc = Math.min(BINS - 1, Math.max(0, d));
  return Array.from({ length: BINS }, (_, k) => -((k - dc) ** 2) / (2 * w * w));
}

export function simulate(scene: Scene, p: SimParams = SIM_DEFAULTS): Sim {
  const r = rng(p.seed), objs = scene.objs;
  const raw = new Float32Array(NCELLS * NO), owner = new Int32Array(NCELLS).fill(-1), target = new Float32Array(NCELLS * 4).fill(NaN);
  // which strides have at least one cell centre inside each object's box
  const fitRel = objs.map((o) => {
    const S = Math.max(boxW(o.box), boxH(o.box));
    const has = GRIDS.map((g) => {
      for (let i = 0; i < g.g; i++) for (let j = 0; j < g.g; j++) {
        const cx = (j + 0.5) * g.stride, cy = (i + 0.5) * g.stride;
        if (cx > o.box.x0 && cx < o.box.x1 && cy > o.box.y0 && cy < o.box.y1) return true;
      }
      return false;
    });
    const fits = GRIDS.map((g, k) => (has[k] ? strideFit(S, g.stride) : 0));
    const m = Math.max(...fits, 1e-12);
    return fits.map((f) => f / m);
  });
  for (let a = 0; a < NCELLS; a++) {
    const c = cellInfo(a), o = a * NO;
    let best = -1, bq = 0, bd = [0, 0, 0, 0];
    objs.forEach((ob, k) => {
      const d = [c.cx - ob.box.x0, c.cy - ob.box.y0, ob.box.x1 - c.cx, ob.box.y1 - c.cy];
      if (d.some((v) => v <= 0)) return;
      const q = Math.pow(centreness(d[0], d[1], d[2], d[3]), 0.7) * fitRel[k][c.scale];
      if (best < 0 || q > bq) { best = k; bq = q; bd = d; }
    });
    // distance bins
    if (best >= 0) {
      owner[a] = best;
      for (let s = 0; s < 4; s++) {
        target[a * 4 + s] = bd[s] / c.stride;
        const z = peakLogits(bd[s] / c.stride + p.boxNoise * r.norm());
        for (let k = 0; k < BINS; k++) raw[o + s * BINS + k] = z[k];
      }
    } else {
      for (let s = 0; s < 4; s++) for (let k = 0; k < BINS; k++) raw[o + s * BINS + k] = 0.3 * r.norm();
    }
    // class logits
    for (let cl = 0; cl < NC; cl++) raw[o + 4 * BINS + cl] = -6 + p.scoreNoise * r.norm();
    if (best >= 0) {
      const ob = objs[best], lg = -4 + 7 * bq + p.scoreNoise * r.norm();
      raw[o + 4 * BINS + ob.cls] = lg;
      if (ob.alt !== undefined) raw[o + 4 * BINS + (1 - ob.cls)] = lg + ob.alt + 0.3 * p.scoreNoise * r.norm();
    }
  }
  // anchor head
  const r2 = rng(p.seed + 101);
  const aT = new Float32Array(NCELLS * 4), aLogit = new Float32Array(NCELLS * NC), aOwner = new Int32Array(NCELLS).fill(-1), aIou = new Float32Array(NCELLS);
  const bestAnchor = objs.map(() => 0);
  for (let a = 0; a < NCELLS; a++) {
    const A = anchorBox(a);
    let best = -1, bi = 0;
    objs.forEach((ob, k) => {
      const v = boxIou(A, ob.box);
      if (v > bestAnchor[k]) bestAnchor[k] = v;
      if (v > bi) { bi = v; best = k; }
    });
    for (let cl = 0; cl < NC; cl++) aLogit[a * NC + cl] = -6 + p.scoreNoise * r2.norm();
    if (best >= 0 && bi >= 0.5) {
      aOwner[a] = best; aIou[a] = bi;
      const t = encodeBox(objs[best].box, A);
      for (let s = 0; s < 4; s++) aT[a * 4 + s] = t[s] + 0.2 * p.boxNoise * r2.norm();
      const lg = -1.5 + 7 * (bi - 0.5) / 0.5 + p.scoreNoise * r2.norm();
      aLogit[a * NC + objs[best].cls] = lg;
      if (objs[best].alt !== undefined) aLogit[a * NC + 1 - objs[best].cls] = lg + objs[best].alt!;
    }
  }
  return { scene, raw, owner, target, aT, aLogit, aOwner, aIou, bestAnchor };
}

// ---- Decoding -------------------------------------------------------------------------------

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

/** Expected bin of a distribution given as 16 logits at raw[off..]. */
export function expectedBin(raw: ArrayLike<number>, off: number): number {
  const p = softmax(raw, off, BINS);
  let e = 0;
  for (let k = 0; k < BINS; k++) e += k * p[k];
  return e;
}

/** The box of a cell from four distances (in strides): s (j + ½ ∓ d). */
export function boxFromDistances(a: number, d: ArrayLike<number>): Box {
  const c = cellInfo(a), s = c.stride;
  return { x0: c.cx - d[0] * s, y0: c.cy - d[1] * s, x1: c.cx + d[2] * s, y1: c.cy + d[3] * s };
}

export type Head = "dist" | "anchor";

export interface Decoded { head: Head; boxes: Box[]; dist: Float32Array; prob: Float32Array }

export function decodeAll(sim: Sim, head: Head): Decoded {
  const boxes: Box[] = new Array(NCELLS), dist = new Float32Array(NCELLS * 4), prob = new Float32Array(NCELLS * NC);
  for (let a = 0; a < NCELLS; a++) {
    if (head === "dist") {
      for (let s = 0; s < 4; s++) dist[a * 4 + s] = expectedBin(sim.raw, a * NO + s * BINS);
      boxes[a] = boxFromDistances(a, dist.subarray(a * 4, a * 4 + 4));
      for (let c = 0; c < NC; c++) prob[a * NC + c] = sigmoid(sim.raw[a * NO + 4 * BINS + c]);
    } else {
      boxes[a] = decodeBox(sim.aT.subarray(a * 4, a * 4 + 4), anchorBox(a));
      for (let c = 0; c < NC; c++) prob[a * NC + c] = sigmoid(sim.aLogit[a * NC + c]);
    }
  }
  return { head, boxes, dist, prob };
}

export interface Candidate { cell: number; cls: number; score: number }

/** All (cell, class) pairs with probability > conf, sorted by score (ties: cell, then class). */
export function candidates(dec: Decoded, conf: number, scales: boolean[] = [true, true, true]): Candidate[] {
  const out: Candidate[] = [];
  for (let a = 0; a < NCELLS; a++) {
    if (!scales[cellInfo(a).scale]) continue;
    for (let c = 0; c < NC; c++) { const p = dec.prob[a * NC + c]; if (p > conf) out.push({ cell: a, cls: c, score: p }); }
  }
  return out.sort((x, y) => y.score - x.score || x.cell - y.cell || x.cls - y.cls);
}

/** Greedy NMS of the candidates: per class (boxes of different classes never suppress each other) or across classes. */
export function runNms(dec: Decoded, cands: Candidate[], t: number, agnostic: boolean): NmsResult {
  return nmsGeneric(cands.length, cands.map((c) => c.score),
    (i, j) => (agnostic || cands[i].cls === cands[j].cls ? boxIou(dec.boxes[cands[i].cell], dec.boxes[cands[j].cell]) : 0), t);
}

export interface Eval { tp: number; fp: number; fn: number; f1: number; missed: number[] }

/** The evaluation day's matching at IoU ≥ tau, per class (a box must have the right class). */
export function evaluate(objs: Obj[], boxes: Box[], cls: number[], tau = 0.5): Eval {
  let tp = 0, fp = 0, fn = 0;
  const missed: number[] = [];
  for (let c = 0; c < NC; c++) {
    const ti = objs.map((o, k) => (o.cls === c ? k : -1)).filter((k) => k >= 0);
    const pi = cls.map((v, k) => (v === c ? k : -1)).filter((k) => k >= 0);
    const m = match(boxIouTable(ti.map((k) => objs[k].box), pi.map((k) => boxes[k])), tau);
    tp += m.tp; fp += m.fp; fn += m.fn;
    m.matchT.forEach((p, k) => { if (p < 0) missed.push(ti[k]); });
  }
  return { tp, fp, fn, f1: tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 1, missed: missed.sort((a, b) => a - b) };
}

/** Generalised IoU: IoU − |C \ (A ∪ B)| / |C|, C the smallest box enclosing both. */
export function giou(a: Box, b: Box): number {
  const ia = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const u = boxW(a) * boxH(a) + boxW(b) * boxH(b) - ia;
  const C = (Math.max(a.x1, b.x1) - Math.min(a.x0, b.x0)) * (Math.max(a.y1, b.y1) - Math.min(a.y0, b.y0));
  return ia / u - (C - u) / C;
}

/** Number of cell centres of each stride inside a box. */
export function centresInside(b: Box): number[] {
  return GRIDS.map((g) => {
    let n = 0;
    for (let i = 0; i < g.g; i++) for (let j = 0; j < g.g; j++) {
      const cx = (j + 0.5) * g.stride, cy = (i + 0.5) * g.stride;
      if (cx > b.x0 && cx < b.x1 && cy > b.y0 && cy < b.y1) n++;
    }
    return n;
  });
}

// ---- Editing one cell's output (the inspector) ------------------------------------------------

/** Probabilities of one side's 16 bins. */
export const sideProbs = (sim: Sim, a: number, side: number): number[] => softmax(sim.raw, a * NO + side * BINS, BINS);

/** Set one side's distribution from probabilities (stored as log-probabilities). */
export function setSideProbs(sim: Sim, a: number, side: number, p: ArrayLike<number>): void {
  let z = 0;
  for (let k = 0; k < BINS; k++) z += Math.max(0, p[k]);
  for (let k = 0; k < BINS; k++) sim.raw[a * NO + side * BINS + k] = Math.log(Math.max(1e-9, Math.max(0, p[k]) / (z || 1)));
}

/** A distribution from a list of (bin, mass) pairs. */
export function massAt(pairs: [number, number][]): number[] {
  const p = new Array(BINS).fill(0);
  for (const [k, m] of pairs) p[k] += m;
  return p;
}

/** Change bin k to probability v, scaling the other bins so that the sum stays 1. */
export function setBin(p: number[], k: number, v: number): number[] {
  v = Math.min(1, Math.max(0, v));
  const rest = p.reduce((s, x, i) => (i === k ? s : s + x), 0);
  return p.map((x, i) => (i === k ? v : rest > 0 ? (x * (1 - v)) / rest : (1 - v) / (BINS - 1)));
}

export const classLogit = (sim: Sim, a: number, c: number): number => sim.raw[a * NO + 4 * BINS + c];
export function setClassLogit(sim: Sim, a: number, c: number, v: number): void { sim.raw[a * NO + 4 * BINS + c] = v; }
