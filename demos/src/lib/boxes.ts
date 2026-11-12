// Axis-aligned boxes for detection (owner: modern deep learning day, unit detection/). No DOM.
//
// A box is given by its corners (x0, y0) top-left and (x1, y1) bottom-right in continuous
// pixel coordinates (a box covering pixels 0..3 is [0, 4]). Box IoU in closed form, greedy
// non-maximum suppression with a step record for the demos (also for any other overlap, e.g.
// the rasterised star-convex polygons of instancemaps), anchors on a grid, local maxima of a
// heat map, and an IoU table in the format of lib/eval.ts, so detections are evaluated with
// the evaluation day's matching, F1 and AP.
import { tableFromIou, type IouTable } from "./eval";

export interface Box { x0: number; y0: number; x1: number; y1: number }

export const boxW = (b: Box): number => Math.max(0, b.x1 - b.x0);
export const boxH = (b: Box): number => Math.max(0, b.y1 - b.y0);
export const boxArea = (b: Box): number => boxW(b) * boxH(b);
export const boxCentre = (b: Box): [number, number] => [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];

/** Box with centre (cx, cy), width w and height h. */
export function boxFromCentre(cx: number, cy: number, w: number, h: number): Box {
  return { x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2 };
}

/** Area of the intersection: the overlap of the x intervals times that of the y intervals. */
export function boxIntersection(a: Box, b: Box): number {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0), h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  return w > 0 && h > 0 ? w * h : 0;
}

/** Intersection over union of two boxes (0 if both are empty). */
export function boxIou(a: Box, b: Box): number {
  const i = boxIntersection(a, b), u = boxArea(a) + boxArea(b) - i;
  return u > 0 ? i / u : 0;
}

/** One round of greedy NMS: the box kept and the boxes it removed (with their IoU). */
export interface NmsStep { kept: number; removed: { j: number; iou: number }[] }

export interface NmsResult {
  keep: number[];          // kept indices, in the order they were kept (decreasing score)
  steps: NmsStep[];
  dropped: number[];       // below the score threshold, never considered
}

/**
 * Greedy non-maximum suppression for any overlap measure: sort by score, keep the best
 * remaining candidate, delete every remaining candidate whose overlap with it is > t, repeat.
 * Candidates with score < minScore are dropped first. Ties in score: lower index first.
 */
export function nmsGeneric(n: number, scores: ArrayLike<number>, overlap: (i: number, j: number) => number,
  t: number, minScore = -Infinity): NmsResult {
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
  const alive = new Uint8Array(n), dropped: number[] = [];
  for (const i of order) { if (scores[i] >= minScore) alive[i] = 1; else dropped.push(i); }
  const keep: number[] = [], steps: NmsStep[] = [];
  for (const i of order) {
    if (!alive[i]) continue;
    alive[i] = 0;
    keep.push(i);
    const removed: { j: number; iou: number }[] = [];
    for (const j of order) {
      if (!alive[j]) continue;
      const v = overlap(i, j);
      if (v > t) { alive[j] = 0; removed.push({ j, iou: v }); }
    }
    steps.push({ kept: i, removed });
  }
  return { keep, steps, dropped };
}

/** Greedy NMS of boxes with box IoU. */
export function nms(boxes: Box[], scores: ArrayLike<number>, t: number, minScore = -Infinity): NmsResult {
  return nmsGeneric(boxes.length, scores, (i, j) => boxIou(boxes[i], boxes[j]), t, minScore);
}

/** IoU table (rows = true boxes, columns = predicted boxes) in the format of lib/eval.ts. */
export function boxIouTable(truth: Box[], pred: Box[]): IouTable {
  return tableFromIou(truth.map(t => pred.map(p => boxIou(t, p))));
}

export interface Anchor extends Box { gx: number; gy: number; shape: number }

/**
 * Anchors: at the centre of every grid cell of size `stride`, one box per shape (w, h).
 */
export function anchorGrid(width: number, height: number, stride: number, shapes: [number, number][]): Anchor[] {
  const out: Anchor[] = [];
  for (let gy = 0; gy * stride < height; gy++) for (let gx = 0; gx * stride < width; gx++) {
    const cx = (gx + 0.5) * stride, cy = (gy + 0.5) * stride;
    shapes.forEach(([w, h], s) => out.push({ ...boxFromCentre(cx, cy, w, h), gx, gy, shape: s }));
  }
  return out;
}

/** For every true box, the anchor with the largest IoU and that IoU. */
export function bestAnchors(truth: Box[], anchors: Box[]): { index: number; iou: number }[] {
  return truth.map(t => {
    let index = -1, iou = 0;
    anchors.forEach((a, k) => { const v = boxIou(t, a); if (v > iou) { iou = v; index = k; } });
    return { index, iou };
  });
}

/**
 * Local maxima of a heat map: pixels that are the maximum of their (2r+1) × (2r+1) window
 * (ties: the first in raster order wins) and ≥ minValue. The 3 × 3 max-filter peak picking
 * of the Hough accumulator, with r = 1.
 */
export function localMaxima(f: ArrayLike<number>, w: number, h: number, r = 1, minValue = -Infinity): number[] {
  const out: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, v = f[p];
    if (!(v >= minValue)) continue;
    let ok = true;
    for (let dy = -r; dy <= r && ok; dy++) for (let dx = -r; dx <= r && ok; dx++) {
      if (!dx && !dy) continue;
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const q = yy * w + xx, u = f[q];
      if (u > v || (u === v && q < p)) ok = false;
    }
    if (ok) out.push(p);
  }
  return out;
}

/** Box regression target relative to an anchor: (t_x, t_y, t_w, t_h) with logs of the size ratios. */
export function encodeBox(b: Box, a: Box): [number, number, number, number] {
  const [cx, cy] = boxCentre(b), [ax, ay] = boxCentre(a);
  return [(cx - ax) / boxW(a), (cy - ay) / boxH(a), Math.log(boxW(b) / boxW(a)), Math.log(boxH(b) / boxH(a))];
}

/** Inverse of encodeBox. */
export function decodeBox(t: ArrayLike<number>, a: Box): Box {
  const [ax, ay] = boxCentre(a);
  return boxFromCentre(ax + t[0] * boxW(a), ay + t[1] * boxH(a), boxW(a) * Math.exp(t[2]), boxH(a) * Math.exp(t[3]));
}
