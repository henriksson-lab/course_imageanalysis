// Region properties of a label image in one scan (labelling day, unit regionprops/; used by
// the moments and regionfilter demos). No DOM.
//
// Coordinates: x = column, y = row (DOWN), pixel centres at integer (x, y). Raw moments
// m_pq = Σ x^p y^q over the pixels of a region; central moments μ_pq about the centroid;
// S = [μ20 μ11; μ11 μ02] / m00 (the covariance of the pixel coordinates, divided by n as in
// pca/). Its eigenvalues λ1 ≥ λ2 (descending), and θ = ½ atan2(2 μ11, μ20 − μ02), the
// angle of the major axis from the +x axis towards +y, i.e. CLOCKWISE as seen on screen;
// `orientation` is −θ in degrees (counter-clockwise on screen, the course convention).
//
// Perimeter estimators: edges (pixel edges between object and background; a digital disc
// gives ≈ 8r), chain (8-chain code through boundary pixel centres, steps 1 and √2, outer
// contour plus one contour per hole), crofton (crossings along lines at 0°, 45°, 90°, 135°).
// Euler number from Gray's bit-quads (local, no labelling), with the 8/4 pair.
//
// TODO(shared lib): these belong next to lib/label.ts (the plan's region-property section of
// label.ts); kept here because label.ts was created by another task.
import { eigSym2 } from "../lib/linalg";
import { label2d } from "../lib/label";

export type Pt = [number, number];

export interface Region {
  label: number;
  area: number;
  /** Centroid (pixel centres at integers). */
  cx: number; cy: number;
  /** Inclusive bounding box. */
  x0: number; y0: number; x1: number; y1: number;
  /** Central moments (sums, not divided by the area). */
  mu20: number; mu11: number; mu02: number;
  /** Covariance S = μ/m00: s11 = μ20/m00, s12 = μ11/m00, s22 = μ02/m00. */
  s11: number; s12: number; s22: number;
  l1: number; l2: number;
  /** Major-axis unit vector (y down) and θ (radians, from +x towards +y). */
  v1: Pt; theta: number;
  /** Orientation in degrees, counter-clockwise as seen on screen, in (−90, 90]. */
  orientation: number;
  eccentricity: number;
  elongation: number;
  /** Full axis lengths of the equivalent ellipse: 4√λ1, 4√λ2. */
  major: number; minor: number;
  extent: number;
  perimEdges: number;
  perimChain: number;
  perimCrofton: number;
  euler: number;
  holes: number;
  touchesBorder: boolean;
}

/** One scan for area, centroid, box and second moments; then per-region perimeters etc. on
 *  the region's cropped mask. `count` = number of labels (1..count). */
export function regionProps(lab: ArrayLike<number>, w: number, h: number, count: number, opts: { perimeters?: boolean } = {}): Region[] {
  const n = new Float64Array(count + 1), sx = new Float64Array(count + 1), sy = new Float64Array(count + 1);
  const sxx = new Float64Array(count + 1), sxy = new Float64Array(count + 1), syy = new Float64Array(count + 1);
  const x0 = new Int32Array(count + 1).fill(1 << 30), y0 = new Int32Array(count + 1).fill(1 << 30), x1 = new Int32Array(count + 1).fill(-1), y1 = new Int32Array(count + 1).fill(-1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const l = lab[y * w + x];
    if (!l || l > count) continue;
    n[l]++; sx[l] += x; sy[l] += y; sxx[l] += x * x; sxy[l] += x * y; syy[l] += y * y;
    if (x < x0[l]) x0[l] = x; if (x > x1[l]) x1[l] = x; if (y < y0[l]) y0[l] = y; if (y > y1[l]) y1[l] = y;
  }
  const out: Region[] = [];
  for (let l = 1; l <= count; l++) {
    if (!n[l]) continue;
    const A = n[l], cx = sx[l] / A, cy = sy[l] / A;
    const mu20 = sxx[l] - A * cx * cx, mu02 = syy[l] - A * cy * cy, mu11 = sxy[l] - A * cx * cy;
    const s11 = mu20 / A, s12 = mu11 / A, s22 = mu02 / A;
    const e = eigSym2(s11, s12, s22);
    const l1 = Math.max(0, e.l1), l2 = Math.max(0, e.l2);
    const bw = x1[l] - x0[l] + 1, bh = y1[l] - y0[l] + 1;
    let orient = -e.angle * 180 / Math.PI;
    if (orient <= -90) orient += 180;
    const r: Region = {
      label: l, area: A, cx, cy, x0: x0[l], y0: y0[l], x1: x1[l], y1: y1[l],
      mu20, mu11, mu02, s11, s12, s22, l1, l2, v1: e.v1, theta: e.angle, orientation: orient,
      eccentricity: l1 > 0 ? Math.sqrt(Math.max(0, 1 - l2 / l1)) : 0,
      elongation: l2 > 0 ? Math.sqrt(l1 / l2) : Infinity,
      major: 4 * Math.sqrt(l1), minor: 4 * Math.sqrt(l2), extent: A / (bw * bh),
      perimEdges: 0, perimChain: 0, perimCrofton: 0, euler: 0, holes: 0,
      touchesBorder: x0[l] === 0 || y0[l] === 0 || x1[l] === w - 1 || y1[l] === h - 1,
    };
    if (opts.perimeters !== false) {
      const c = cropLabel(lab, w, x0[l], y0[l], bw, bh, l);
      r.perimEdges = edgePerimeter(c.mask, c.w, c.h);
      r.perimChain = chainPerimeter(c.mask, c.w, c.h);
      r.perimCrofton = croftonPerimeter(c.mask, c.w, c.h);
      r.euler = eulerNumber(c.mask, c.w, c.h, 8);
      r.holes = countHoles(c.mask, c.w, c.h);
    }
    out.push(r);
  }
  return out;
}

/** Mask of label l inside its box, with a 1-pixel background frame. */
export function cropLabel(lab: ArrayLike<number>, w: number, bx: number, by: number, bw: number, bh: number, l: number): { mask: Uint8Array; w: number; h: number } {
  const cw = bw + 2, ch = bh + 2, mask = new Uint8Array(cw * ch);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) if (lab[(by + y) * w + bx + x] === l) mask[(y + 1) * cw + x + 1] = 1;
  return { mask, w: cw, h: ch };
}

/** Raw moment m_pq of a mask (pixel centres at integers). */
export function rawMoment(mask: ArrayLike<number>, w: number, h: number, p: number, q: number): number {
  let s = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) s += x ** p * y ** q;
  return s;
}

/** Number of pixel edges between object and background (outside the image = background). */
export function edgePerimeter(mask: ArrayLike<number>, w: number, h: number): number {
  let p = 0;
  const at = (x: number, y: number) => (x >= 0 && y >= 0 && x < w && y < h ? (mask[y * w + x] ? 1 : 0) : 0);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    p += (1 - at(x - 1, y)) + (1 - at(x + 1, y)) + (1 - at(x, y - 1)) + (1 - at(x, y + 1));
  }
  return p;
}

// 8 directions, clockwise as seen on screen (y down): E, SE, S, SW, W, NW, N, NE.
export const DIR8: Pt[] = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

/** Moore-neighbour tracing of the outer contour of the 8-connected component that contains
 *  `start` (which must be its first pixel in raster order). Returns the boundary pixel
 *  centres in order (clockwise on screen) and the chain codes (0..7, index into DIR8). */
export function traceContour(mask: ArrayLike<number>, w: number, h: number, start: number): { points: Pt[]; codes: number[] } {
  const at = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && !!mask[y * w + x];
  const sx = start % w, sy = (start - sx) / w;
  const points: Pt[] = [[sx, sy]], codes: number[] = [];
  // backtrack starts at the west neighbour (background: start is the first pixel in raster order)
  let cx = sx, cy = sy, b = 4;
  let first = -1;
  for (let guard = 0; guard < 4 * w * h + 8; guard++) {
    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (b + k) % 8;
      if (at(cx + DIR8[d][0], cy + DIR8[d][1])) { found = d; break; }
    }
    if (found < 0) break;                     // isolated pixel
    if (cx === sx && cy === sy) {
      if (first < 0) first = found;
      else if (found === first) break;          // back at the start, about to repeat the first move
    }
    cx += DIR8[found][0]; cy += DIR8[found][1];
    codes.push(found);
    points.push([cx, cy]);
    // new backtrack: the neighbour checked just before `found`, seen from the new pixel
    // (direction from the new pixel to that neighbour)
    const prev = (found + 7) % 8;
    const bx = cx - DIR8[found][0] + DIR8[prev][0], by = cy - DIR8[found][1] + DIR8[prev][1];
    b = DIR8.findIndex(([dx, dy]) => dx === bx - cx && dy === by - cy);
  }
  if (points.length > 1) points.pop();   // last point repeats the start
  return { points, codes };
}

const chainLength = (codes: number[]) => codes.reduce((s, c) => s + (c % 2 ? Math.SQRT2 : 1), 0);

/** 8-chain-code perimeter: outer contours of all 8-components plus, for every hole (a
 *  4-connected background component not touching the frame), the contour through the object
 *  pixels 4-adjacent to it (traced as the outer contour of the hole dilated by the cross). */
export function chainPerimeter(mask: ArrayLike<number>, w: number, h: number): number {
  let total = 0;
  const lab = label2d(mask, w, h, 8);
  const seen = new Uint8Array(lab.count + 1);
  for (let i = 0; i < w * h; i++) {
    const l = lab.data[i];
    if (!l || seen[l]) continue;
    seen[l] = 1;
    total += chainLength(traceContour(lab.data.map(v => (v === l ? 1 : 0)), w, h, i).codes);
  }
  const holes = holeMasks(mask, w, h);
  for (const hm of holes) {
    const d = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!hm[y * w + x]) continue;
      for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h) d[yy * w + xx] = 1;
      }
    }
    const s = d.indexOf(1);
    if (s >= 0) total += chainLength(traceContour(d, w, h, s).codes);
  }
  return total;
}

/** Cauchy–Crofton estimate: count object/background transitions along rows, columns and
 *  both diagonals (outside = background); P ≈ (π/8)(n0 + n90 + (n45 + n135)/√2). */
export function croftonPerimeter(mask: ArrayLike<number>, w: number, h: number): number {
  const at = (x: number, y: number) => (x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] ? 1 : 0);
  let n0 = 0, n90 = 0, n45 = 0, n135 = 0;
  for (let y = -1; y <= h; y++) for (let x = -1; x <= w; x++) {
    const v = at(x, y);
    if (v !== at(x + 1, y)) n0++;
    if (v !== at(x, y + 1)) n90++;
    if (v !== at(x + 1, y + 1)) n45++;
    if (v !== at(x - 1, y + 1)) n135++;
  }
  return Math.PI / 8 * (n0 + n90 + (n45 + n135) / Math.SQRT2);
}

/** Gray's bit-quad counts over all 2×2 windows (the image padded with background). */
export function bitQuads(mask: ArrayLike<number>, w: number, h: number): { q1: number; q3: number; qd: number; q2: number; q4: number } {
  const at = (x: number, y: number) => (x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] ? 1 : 0);
  let q1 = 0, q2 = 0, q3 = 0, q4 = 0, qd = 0;
  for (let y = -1; y < h; y++) for (let x = -1; x < w; x++) {
    const a = at(x, y), b = at(x + 1, y), c = at(x, y + 1), d = at(x + 1, y + 1), s = a + b + c + d;
    if (s === 1) q1++;
    else if (s === 3) q3++;
    else if (s === 4) q4++;
    else if (s === 2) { if (a === d) qd++; else q2++; }
  }
  return { q1, q2, q3, q4, qd };
}

/** Euler number from bit-quads: 8-connected objects E = (Q1 − Q3 − 2 QD)/4; 4-connected
 *  objects E = (Q1 − Q3 + 2 QD)/4. */
export function eulerNumber(mask: ArrayLike<number>, w: number, h: number, conn: 4 | 8 = 8): number {
  const q = bitQuads(mask, w, h);
  return (q.q1 - q.q3 + (conn === 8 ? -2 : 2) * q.qd) / 4;
}

/** Masks of the holes: background components (4-connected when the object is 8-connected,
 *  and vice versa) that do not touch the image frame. */
export function holeMasks(mask: ArrayLike<number>, w: number, h: number, objConn: 4 | 8 = 8): Uint8Array[] {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const lab = label2d(inv, w, h, objConn === 8 ? 4 : 8);
  const border = new Uint8Array(lab.count + 1);
  for (let x = 0; x < w; x++) { border[lab.data[x]] = 1; border[lab.data[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { border[lab.data[y * w]] = 1; border[lab.data[y * w + w - 1]] = 1; }
  const out: Uint8Array[] = [];
  for (let l = 1; l <= lab.count; l++) if (!border[l]) out.push(lab.data.map(v => (v === l ? 1 : 0)) as unknown as Uint8Array);
  return out.map(m => Uint8Array.from(m));
}

export function countHoles(mask: ArrayLike<number>, w: number, h: number, objConn: 4 | 8 = 8): number {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const lab = label2d(inv, w, h, objConn === 8 ? 4 : 8);
  const border = new Uint8Array(lab.count + 1);
  for (let x = 0; x < w; x++) { border[lab.data[x]] = 1; border[lab.data[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { border[lab.data[y * w]] = 1; border[lab.data[y * w + w - 1]] = 1; }
  let c = 0;
  for (let l = 1; l <= lab.count; l++) if (!border[l]) c++;
  return c;
}

/** Properties of a single binary mask (all foreground pixels as one region, even if it has
 *  several components; perimeters and Euler number then cover all of them). */
export function maskProps(mask: ArrayLike<number>, w: number, h: number): Region | null {
  const lab = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) lab[i] = mask[i] ? 1 : 0;
  const r = regionProps(lab, w, h, 1);
  return r[0] ?? null;
}
