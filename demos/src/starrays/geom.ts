// Star-convex polygons (networks for cells and objects day, unit stardist/). No DOM.
//
// Geometry of the StarDist representation on a pixel grid: the length of each of n rays from a
// point to the first pixel outside its object (exact grid traversal), the polygon through the
// ray tips, its area by the triangle rule, the pixels it covers, polygon IoU, the objectness
// target (distance to the background, normalised per object), the best centre of a shape, and
// greedy polygon NMS with a bounding-box pre-check (via nmsGeneric of lib/boxes.ts).
//
// Coordinates as lib/shapecanvas.ts: x right, y DOWN; pixel (row y, column x) covers
// [x, x+1] × [y, y+1], its centre is (x + 0.5, y + 0.5). Ray k of n points along the angle
// 2πk/n measured from the x axis towards +y (so on screen the rays run clockwise; ray 0 points
// right, ray n/4 down).
//
// Candidate for lib/instances.ts (plan/cellnets-day.md, "Shared engineering"), together with
// the ray code of instancemaps/model.ts.
import { edt } from "../lib/distance";
import { nmsGeneric, type NmsResult } from "../lib/boxes";

export type Pt = [number, number];

/** Unit vector of ray k of n (angle 2πk/n, y down). */
export function rayDir(k: number, n: number): Pt {
  const a = (2 * Math.PI * k) / n;
  return [Math.cos(a), Math.sin(a)];
}

/**
 * Distance from (cx, cy) along (dx, dy) to the first point outside the object `id` of the label
 * image (or outside the image): exact traversal of the pixel squares the ray crosses. 0 if the
 * start pixel itself is outside.
 */
export function rayLength(lab: ArrayLike<number>, w: number, h: number, cx: number, cy: number,
  dx: number, dy: number, id = 1): number {
  let x = Math.floor(cx), y = Math.floor(cy);
  if (x < 0 || y < 0 || x >= w || y >= h || lab[y * w + x] !== id) return 0;
  const sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
  const ddx = Math.abs(dx) > 1e-12 ? Math.abs(1 / dx) : Infinity, ddy = Math.abs(dy) > 1e-12 ? Math.abs(1 / dy) : Infinity;
  let tx = ddx === Infinity ? Infinity : (dx > 0 ? x + 1 - cx : cx - x) * ddx;
  let ty = ddy === Infinity ? Infinity : (dy > 0 ? y + 1 - cy : cy - y) * ddy;
  for (let guard = 0; guard < 4 * (w + h); guard++) {
    let t: number;
    if (tx < ty) { t = tx; x += sx; tx += ddx; } else { t = ty; y += sy; ty += ddy; }
    if (x < 0 || y < 0 || x >= w || y >= h || lab[y * w + x] !== id) return t;
  }
  return 0;
}

/** The n ray lengths from (cx, cy) inside object `id`. */
export function raysAt(lab: ArrayLike<number>, w: number, h: number, cx: number, cy: number, n: number, id = 1): Float64Array {
  const r = new Float64Array(n);
  for (let k = 0; k < n; k++) { const [dx, dy] = rayDir(k, n); r[k] = rayLength(lab, w, h, cx, cy, dx, dy, id); }
  return r;
}

/** Area of the polygon through the ray tips: n triangles, ½ sin(2π/n) Σ r_k r_{k+1}. */
export function polyArea(r: ArrayLike<number>): number {
  const n = r.length;
  let s = 0;
  for (let k = 0; k < n; k++) s += r[k] * r[(k + 1) % n];
  return 0.5 * Math.sin((2 * Math.PI) / n) * s;
}

/** The ray tips (polygon vertices) in order. */
export function polyVertices(cx: number, cy: number, r: ArrayLike<number>): Pt[] {
  const n = r.length;
  return Array.from({ length: n }, (_, k) => { const [dx, dy] = rayDir(k, n); return [cx + r[k] * dx, cy + r[k] * dy] as Pt; });
}

/** Bounding box of a polygon given by rays: [x0, y0, x1, y1]. */
export function polyBox(cx: number, cy: number, r: ArrayLike<number>): [number, number, number, number] {
  let x0 = cx, y0 = cy, x1 = cx, y1 = cy;
  for (const [x, y] of polyVertices(cx, cy, r)) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return [x0, y0, x1, y1];
}

/** Indices of the pixels whose centres lie inside the polygon (fan of triangles from the centre). */
export function polygonPixels(cx: number, cy: number, r: ArrayLike<number>, w: number, h: number): Int32Array {
  const n = r.length, tips = polyVertices(cx, cy, r);
  const [bx0, by0, bx1, by1] = polyBox(cx, cy, r);
  const out: number[] = [], sector = (2 * Math.PI) / n;
  for (let y = Math.max(0, Math.floor(by0 - 0.5)); y <= Math.min(h - 1, Math.ceil(by1 - 0.5)); y++)
    for (let x = Math.max(0, Math.floor(bx0 - 0.5)); x <= Math.min(w - 1, Math.ceil(bx1 - 0.5)); x++) {
      // the pixel centre, nudged by a tiny fixed offset: a centre exactly on an edge then counts
      // for one of the two polygons sharing that edge (like a rasteriser's tie rule), so the
      // pixel count matches the area
      const px = x + 0.5 + 1.37e-6, py = y + 0.5 + 2.71e-6, dx = px - cx, dy = py - cy;
      let th = Math.atan2(dy, dx); if (th < 0) th += 2 * Math.PI;
      const k = Math.min(n - 1, Math.floor(th / sector)), k2 = (k + 1) % n;
      const [ax, ay] = tips[k], [qx, qy] = tips[k2];
      // inside the triangle (centre, tip k, tip k+1): on the same side of the far edge as the centre
      const cr = (qx - ax) * (py - ay) - (qy - ay) * (px - ax), crC = (qx - ax) * (cy - ay) - (qy - ay) * (cx - ax);
      if (cr * crC > 0) out.push(y * w + x);
    }
  return Int32Array.from(out);
}

/** IoU of two pixel sets (sorted or not) on an image of `size` pixels. */
export function pixelIou(a: ArrayLike<number>, b: ArrayLike<number>, size: number): number {
  const s = new Uint8Array(size);
  for (let i = 0; i < a.length; i++) s[a[i]] = 1;
  let inter = 0;
  for (let i = 0; i < b.length; i++) if (s[b[i]]) inter++;
  const u = a.length + b.length - inter;
  return u > 0 ? inter / u : 0;
}

/** Objectness target: Euclidean distance to the nearest pixel of another label or the background,
 *  divided by its maximum within each object (1 at the most central pixel, small at the border). */
export function objectness(lab: ArrayLike<number>, w: number, h: number): Float32Array {
  let nl = 0; for (let i = 0; i < lab.length; i++) nl = Math.max(nl, lab[i]);
  const out = new Float32Array(w * h), mx = new Float32Array(nl + 1);
  for (let l = 1; l <= nl; l++) {
    const m = new Uint8Array(w * h); let any = false;
    for (let i = 0; i < lab.length; i++) if (lab[i] === l) { m[i] = 1; any = true; }
    if (!any) continue;
    const d = edt(m, w, h);
    for (let i = 0; i < lab.length; i++) if (m[i]) { out[i] = d[i]; if (d[i] > mx[l]) mx[l] = d[i]; }
  }
  for (let i = 0; i < lab.length; i++) if (lab[i]) out[i] /= mx[lab[i]] || 1;
  return out;
}

export interface Fit { cx: number; cy: number; rays: Float64Array; pix: Int32Array; inter: number; iou: number; missed: number; extra: number }

/** The polygon of the rays from (cx, cy) compared with the object `id`. */
export function fitAt(lab: ArrayLike<number>, w: number, h: number, cx: number, cy: number, n: number, id = 1): Fit {
  const rays = raysAt(lab, w, h, cx, cy, n, id), pix = polygonPixels(cx, cy, rays, w, h);
  let area = 0; for (let i = 0; i < lab.length; i++) if (lab[i] === id) area++;
  let inter = 0; for (const p of pix) if (lab[p] === id) inter++;
  const u = area + pix.length - inter;
  return { cx, cy, rays, pix, inter, iou: u ? inter / u : 0, missed: area - inter, extra: pix.length - inter };
}

/** The pixel centre of object `id` whose polygon has the largest IoU with the object
 *  (ties: the more central pixel, by objectness). `stride` > 1 tests every stride-th pixel. */
export function bestCentre(lab: ArrayLike<number>, w: number, h: number, n: number, id = 1, stride = 1): Fit | null {
  const obj = objectness(lab, w, h);
  let best: Fit | null = null, bestKey = -Infinity;
  for (let y = 0; y < h; y += stride) for (let x = 0; x < w; x += stride) {
    const p = y * w + x;
    if (lab[p] !== id) continue;
    const f = fitAt(lab, w, h, x + 0.5, y + 0.5, n, id);
    const key = f.iou + 1e-6 * obj[p];
    if (key > bestKey) { bestKey = key; best = f; }
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// Analytic rays (for worked examples and tests)

/** Rays from the centre of an axis-aligned rectangle with half-sides hx, hy. */
export function rectRays(hx: number, hy: number, n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, k) => {
    const [dx, dy] = rayDir(k, n), c = Math.abs(dx), s = Math.abs(dy);
    return Math.min(c > 1e-12 ? hx / c : Infinity, s > 1e-12 ? hy / s : Infinity);
  });
}

/** Rays from the centre of an ellipse with semi-axes a (along x) and b. */
export function ellipseRays(a: number, b: number, n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, k) => {
    const [c, s] = rayDir(k, n);
    return (a * b) / Math.sqrt((b * c) ** 2 + (a * s) ** 2);
  });
}

// ---------------------------------------------------------------------------------------
// Polygon NMS

export interface Candidate { cx: number; cy: number; score: number; rays: Float64Array; pix: Int32Array; box: [number, number, number, number] }

export function makeCandidate(cx: number, cy: number, score: number, rays: Float64Array, w: number, h: number): Candidate {
  return { cx, cy, score, rays, pix: polygonPixels(cx, cy, rays, w, h), box: polyBox(cx, cy, rays) };
}

/**
 * Greedy polygon NMS: sort by score, keep the best, delete every candidate whose polygon IoU
 * with it exceeds t, repeat. Two polygons whose bounding boxes do not overlap cannot overlap,
 * so their IoU is never computed (the count is returned).
 */
export function polygonNms(c: Candidate[], t: number, size: number): NmsResult & { iouComputed: number } {
  const stamp = new Int32Array(size).fill(-1);
  let stamped = -1, iouComputed = 0;
  const overlap = (i: number, j: number): number => {
    const a = c[i].box, b = c[j].box;
    if (a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1]) return 0;
    iouComputed++;
    if (stamped !== i) { for (const q of c[i].pix) stamp[q] = i; stamped = i; }
    let inter = 0; for (const q of c[j].pix) if (stamp[q] === i) inter++;
    const u = c[i].pix.length + c[j].pix.length - inter;
    return u ? inter / u : 0;
  };
  const res = nmsGeneric(c.length, c.map(x => x.score), overlap, t);
  return { ...res, iouComputed };
}

/** Label image from kept candidates (in kept order = decreasing score): painted from the last to
 *  the first, so where polygons overlap the higher score wins. Labels 1..k in kept order. */
export function paintLabels(c: Candidate[], keep: number[], size: number): Int32Array {
  const lab = new Int32Array(size);
  for (let k = keep.length - 1; k >= 0; k--) for (const q of c[keep[k]].pix) lab[q] = k + 1;
  return lab;
}
