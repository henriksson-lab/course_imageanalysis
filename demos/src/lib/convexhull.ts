// Convex hull and caliper geometry (owner: labelling day, unit shape-features/; used by
// measurements, skeleton and snakes). No DOM.
//
//  - convexHull(points): Andrew's monotone chain, O(n log n). Returns the hull vertices
//    without collinear points, counter-clockwise in a y-up frame (positive shoelace area),
//    which is CLOCKWISE as seen on screen with y down.
//  - maskHullPoints(mask, w, h, "corners" | "centres"): the candidate points of a binary
//    mask (pixel (x, y) covers [x, x+1] x [y, y+1]; its centre is (x+0.5, y+0.5)). Only the
//    leftmost and rightmost pixel of each row can be hull points, so 2 or 4 points per row.
//  - shoelaceArea (signed), polygonPerimeter.
//  - feretWidth(points, theta): w(theta) = extent of the projection onto (cos θ, sin θ).
//  - rotatingCalipers(hull): min width (with one caliper flush with a hull edge) and max
//    diameter (an antipodal pair), both in O(h).
//  - minEnclosingCircle(points, seed): Welzl's algorithm (iterative, seeded shuffle).
//  - bruteForce* versions for tests.

export type Pt = [number, number];

const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** Convex hull by Andrew's monotone chain. Collinear points are dropped. */
export function convexHull(points: Pt[]): Pt[] {
  const p = points.map(q => [q[0], q[1]] as Pt).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  // remove duplicates
  const u: Pt[] = [];
  for (const q of p) if (!u.length || u[u.length - 1][0] !== q[0] || u[u.length - 1][1] !== q[1]) u.push(q);
  if (u.length < 3) return u;
  const lower: Pt[] = [], upper: Pt[] = [];
  for (const q of u) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = u.length - 1; i >= 0; i--) {
    const q = u[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

/** Candidate hull points of a mask: pixel corners (default) or pixel centres, using only
 *  the extreme pixels of each row. Pixel (x, y) covers [x, x+1] x [y, y+1]. */
export function maskHullPoints(mask: ArrayLike<number>, w: number, h: number, mode: "corners" | "centres" = "corners"): Pt[] {
  const out: Pt[] = [];
  for (let y = 0; y < h; y++) {
    let a = -1, b = -1;
    for (let x = 0; x < w; x++) if (mask[y * w + x]) { if (a < 0) a = x; b = x; }
    if (a < 0) continue;
    if (mode === "centres") { out.push([a + 0.5, y + 0.5]); if (b !== a) out.push([b + 0.5, y + 0.5]); }
    else out.push([a, y], [a, y + 1], [b + 1, y], [b + 1, y + 1]);
  }
  return out;
}

/** Hull of a mask (corners or centres convention). */
export function maskHull(mask: ArrayLike<number>, w: number, h: number, mode: "corners" | "centres" = "corners"): Pt[] {
  return convexHull(maskHullPoints(mask, w, h, mode));
}

/** Signed shoelace area: positive for counter-clockwise vertices in a y-up frame. */
export function shoelaceArea(p: Pt[]): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const q = p[i], r = p[(i + 1) % p.length]; a += q[0] * r[1] - r[0] * q[1]; }
  return a / 2;
}

/** Length of the closed polygon. */
export function polygonPerimeter(p: Pt[]): number {
  let s = 0;
  if (p.length < 2) return 0;
  for (let i = 0; i < p.length; i++) { const q = p[i], r = p[(i + 1) % p.length]; s += Math.hypot(r[0] - q[0], r[1] - q[1]); }
  return s;
}

/** Caliper (Feret) diameter in direction theta: max − min of p·(cos θ, sin θ). Also returns
 *  the two extreme points. */
export function feretAt(points: Pt[], theta: number): { width: number; lo: Pt; hi: Pt } {
  const c = Math.cos(theta), s = Math.sin(theta);
  let mn = Infinity, mx = -Infinity, lo: Pt = [0, 0], hi: Pt = [0, 0];
  for (const q of points) {
    const t = q[0] * c + q[1] * s;
    if (t < mn) { mn = t; lo = q; }
    if (t > mx) { mx = t; hi = q; }
  }
  return { width: points.length ? mx - mn : 0, lo, hi };
}

export const feretWidth = (points: Pt[], theta: number): number => feretAt(points, theta).width;

/** w(θ) at n angles in [0, π). */
export function widthProfile(points: Pt[], n = 180): { theta: Float64Array; width: Float64Array } {
  const theta = new Float64Array(n), width = new Float64Array(n);
  for (let i = 0; i < n; i++) { theta[i] = Math.PI * i / n; width[i] = feretWidth(points, theta[i]); }
  return { theta, width };
}

export interface Calipers {
  /** Smallest width (min Feret) and the direction in which it is measured (radians, the
   *  normal of the flush hull edge). */
  minWidth: number;
  minAngle: number;
  /** Index i of the flush edge (hull[i] → hull[i+1]) and the opposite vertex index. */
  minEdge: [number, number];
  /** Largest distance between two hull points (max Feret), the pair and its direction. */
  maxDiameter: number;
  maxPair: [Pt, Pt];
  maxAngle: number;
}

/** Rotating calipers on a convex polygon in either orientation (as convexHull returns it). */
export function rotatingCalipers(hull: Pt[]): Calipers {
  const n = hull.length;
  if (n === 0) return { minWidth: 0, minAngle: 0, minEdge: [0, 0], maxDiameter: 0, maxPair: [[0, 0], [0, 0]], maxAngle: 0 };
  if (n === 1) return { minWidth: 0, minAngle: 0, minEdge: [0, 0], maxDiameter: 0, maxPair: [hull[0], hull[0]], maxAngle: 0 };
  if (n === 2) {
    const d = Math.hypot(hull[1][0] - hull[0][0], hull[1][1] - hull[0][1]);
    const a = Math.atan2(hull[1][1] - hull[0][1], hull[1][0] - hull[0][0]);
    return { minWidth: 0, minAngle: a + Math.PI / 2, minEdge: [0, 1], maxDiameter: d, maxPair: [hull[0], hull[1]], maxAngle: a };
  }
  const area2 = (i: number, j: number, k: number) => Math.abs(cross(hull[i % n], hull[j % n], hull[k % n]));
  const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  let best = Infinity, bestI = 0, bestJ = 0, dmax = 0, pa: Pt = hull[0], pb: Pt = hull[0];
  let j = 1;
  for (let i = 0; i < n; i++) {
    const i1 = (i + 1) % n;
    // advance j while the triangle (edge i, j+1) is larger: j+1 is farther from the edge line
    let guard = 0;
    while (area2(i, i1, j + 1) > area2(i, i1, j) + 1e-12 && guard++ < n) j = (j + 1) % n;
    const len = dist(hull[i], hull[i1]);
    const wdt = area2(i, i1, j) / len;
    if (wdt < best) { best = wdt; bestI = i; bestJ = j % n; }
    for (const [a, b] of [[hull[i], hull[j % n]], [hull[i1], hull[j % n]]] as [Pt, Pt][]) {
      const d = dist(a, b);
      if (d > dmax) { dmax = d; pa = a; pb = b; }
    }
  }
  const e0 = hull[bestI], e1 = hull[(bestI + 1) % n];
  const minAngle = Math.atan2(e1[1] - e0[1], e1[0] - e0[0]) + Math.PI / 2;
  return { minWidth: best, minAngle: normAngle(minAngle), minEdge: [bestI, bestJ], maxDiameter: dmax, maxPair: [pa, pb], maxAngle: normAngle(Math.atan2(pb[1] - pa[1], pb[0] - pa[0])) };
}

/** Angle mod π into [0, π). */
export function normAngle(a: number): number {
  let t = a % Math.PI;
  if (t < 0) t += Math.PI;
  if (t >= Math.PI - 1e-12) t = 0;
  return t;
}

/** Brute force: min over many angles of the width and max over all pairs of the distance. */
export function bruteForceFeret(points: Pt[]): { minWidth: number; maxDiameter: number } {
  let dmax = 0;
  for (let i = 0; i < points.length; i++) for (let k = i + 1; k < points.length; k++)
    dmax = Math.max(dmax, Math.hypot(points[i][0] - points[k][0], points[i][1] - points[k][1]));
  // min width: attained with a caliper flush with an edge between two points of the set
  let wmin = Infinity;
  for (let i = 0; i < points.length; i++) for (let k = i + 1; k < points.length; k++) {
    const a = Math.atan2(points[k][1] - points[i][1], points[k][0] - points[i][0]) + Math.PI / 2;
    const wd = feretWidth(points, a);
    // only edges with all points on one side define a support line
    const c = Math.cos(a), s = Math.sin(a), t = points[i][0] * c + points[i][1] * s;
    let lo = Infinity, hi = -Infinity;
    for (const q of points) { const u = q[0] * c + q[1] * s; lo = Math.min(lo, u); hi = Math.max(hi, u); }
    if (Math.abs(t - lo) < 1e-9 || Math.abs(t - hi) < 1e-9) wmin = Math.min(wmin, wd);
  }
  return { minWidth: points.length > 2 ? wmin : 0, maxDiameter: dmax };
}

/** Brute-force hull: points p such that p is a vertex (for tests): pairs (a, b) with all
 *  points on the left of a→b, collected as a set of vertices. */
export function bruteForceHullVertices(points: Pt[]): Pt[] {
  const out = new Map<string, Pt>();
  for (let i = 0; i < points.length; i++) for (let k = 0; k < points.length; k++) {
    if (i === k) continue;
    const a = points[i], b = points[k];
    if (a[0] === b[0] && a[1] === b[1]) continue;
    let ok = true;
    for (const q of points) {
      const c = cross(a, b, q);
      if (c < 0) { ok = false; break; }
      // collinear points beyond the segment: a, b would not be extreme
      if (c === 0) {
        const t = ((q[0] - a[0]) * (b[0] - a[0]) + (q[1] - a[1]) * (b[1] - a[1])) / ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);
        if (t < 0 || t > 1) { ok = false; break; }
      }
    }
    if (ok) { out.set(a.join(","), a); out.set(b.join(","), b); }
  }
  return [...out.values()];
}

// ---------------------------------------------------------------------------------------
// Minimum enclosing circle (Welzl)

export interface Circle { x: number; y: number; r: number }

function circle2(a: Pt, b: Pt): Circle {
  return { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, r: Math.hypot(a[0] - b[0], a[1] - b[1]) / 2 };
}

function circle3(a: Pt, b: Pt, c: Pt): Circle {
  const bx = b[0] - a[0], by = b[1] - a[1], cx = c[0] - a[0], cy = c[1] - a[1];
  const d = 2 * (bx * cy - by * cx);
  if (Math.abs(d) < 1e-12) {
    // collinear: the circle on the two farthest points
    const cands = [circle2(a, b), circle2(a, c), circle2(b, c)];
    return cands.reduce((m, q) => (q.r > m.r ? q : m));
  }
  const b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const ux = (cy * b2 - by * c2) / d, uy = (bx * c2 - cx * b2) / d;
  return { x: a[0] + ux, y: a[1] + uy, r: Math.hypot(ux, uy) };
}

const inCircle = (c: Circle, p: Pt) => Math.hypot(p[0] - c.x, p[1] - c.y) <= c.r * (1 + 1e-10) + 1e-10;

/** Smallest circle containing all points (Welzl, iterative form, seeded shuffle). */
export function minEnclosingCircle(points: Pt[], seed = 1): Circle {
  const p = points.slice();
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
  if (!p.length) return { x: 0, y: 0, r: 0 };
  let c: Circle = { x: p[0][0], y: p[0][1], r: 0 };
  for (let i = 1; i < p.length; i++) {
    if (inCircle(c, p[i])) continue;
    c = { x: p[i][0], y: p[i][1], r: 0 };
    for (let j = 0; j < i; j++) {
      if (inCircle(c, p[j])) continue;
      c = circle2(p[i], p[j]);
      for (let k = 0; k < j; k++) if (!inCircle(c, p[k])) c = circle3(p[i], p[j], p[k]);
    }
  }
  return c;
}
