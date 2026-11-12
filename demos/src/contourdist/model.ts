// Model of the contourdist demo (no DOM): a nucleus-like true outline, a predicted outline
// through draggable control points (closed Catmull–Rom spline) with an optional thin spike
// and a stray fragment, and the boundary metrics between them (Hausdorff, HD95, ASD) plus
// IoU and Dice of the enclosed regions. Coordinates in pixels, centre (0, 0).
import { SurfaceDistances, surfaceStats } from "../lib/eval";

export type Pt = [number, number];
export const NCTRL = 12;

/** Radius of the true outline at angle θ for an object of mean radius R. */
export const truthRadius = (R: number, th: number): number => R * (1 + 0.13 * Math.cos(2 * th - 0.4) + 0.06 * Math.cos(3 * th + 1) + 0.03 * Math.sin(5 * th));

/** The true outline as a closed polyline (last point not repeated). */
export function truthCurve(R: number, n = 240): Pt[] {
  return Array.from({ length: n }, (_, k) => { const th = 2 * Math.PI * k / n, r = truthRadius(R, th); return [r * Math.cos(th), r * Math.sin(th)] as Pt; });
}

/** Offset a closed polyline along its outward normals by d pixels (d > 0 = larger). */
export function offsetCurve(c: Pt[], d: number): Pt[] {
  const n = c.length;
  return c.map((p, k) => {
    const a = c[(k - 1 + n) % n], b = c[(k + 1) % n], tx = b[0] - a[0], ty = b[1] - a[1], L = Math.hypot(tx, ty) || 1;
    // counter-clockwise curve (y up as in the parametrisation): outward normal = (ty, −tx)
    return [p[0] + d * ty / L, p[1] - d * tx / L] as Pt;
  });
}

/** Closed Catmull–Rom spline through the control points, `per` samples per span. */
export function spline(ctrl: Pt[], per = 30): Pt[] {
  const n = ctrl.length, out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const p0 = ctrl[(i - 1 + n) % n], p1 = ctrl[i], p2 = ctrl[(i + 1) % n], p3 = ctrl[(i + 2) % n];
    for (let s = 0; s < per; s++) {
      const t = s / per, t2 = t * t, t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  return out;
}

/** Insert a thin spike (base width `base` px) at the curve point nearest to angle φ, reaching out to `tip`. */
export function withSpike(c: Pt[], phi: number, tip: Pt, base = 3): Pt[] {
  const n = c.length;
  let k0 = 0, bd = Infinity;
  c.forEach((p, k) => { const d = Math.abs(Math.atan2(Math.sin(Math.atan2(p[1], p[0]) - phi), Math.cos(Math.atan2(p[1], p[0]) - phi))); if (d < bd) { bd = d; k0 = k; } });
  let ka = k0, kb = k0, la = 0, lb = 0;
  while (la < base / 2 && ka > k0 - n / 4) { const p = c[(ka - 1 + n) % n], q = c[(ka + n) % n]; la += Math.hypot(p[0] - q[0], p[1] - q[1]); ka--; }
  while (lb < base / 2 && kb < k0 + n / 4) { const p = c[(kb + 1) % n], q = c[kb % n]; lb += Math.hypot(p[0] - q[0], p[1] - q[1]); kb++; }
  const out: Pt[] = [];
  for (let k = kb; k <= ka + n; k++) out.push(c[((k % n) + n) % n]);
  // out runs from kb around to ka; close with the spike between ka and kb
  out.push(tip);
  return out;
}

/** Points along a closed polyline every `step` pixels (for distances). */
export function resample(c: Pt[], step = 0.25): Pt[] {
  const out: Pt[] = [], n = c.length;
  for (let k = 0; k < n; k++) {
    const a = c[k], b = c[(k + 1) % n], L = Math.hypot(b[0] - a[0], b[1] - a[1]), m = Math.max(1, Math.round(L / step));
    for (let s = 0; s < m; s++) out.push([a[0] + (b[0] - a[0]) * s / m, a[1] + (b[1] - a[1]) * s / m]);
  }
  return out;
}

/** Distance from p to the nearest point of the closed polylines. Returns the distance and that point. */
export function nearestOn(p: Pt, curves: Pt[][]): { d: number; q: Pt } {
  let best = Infinity, q: Pt = p;
  for (const c of curves) {
    const n = c.length;
    for (let k = 0; k < n; k++) {
      const a = c[k], b = c[(k + 1) % n], vx = b[0] - a[0], vy = b[1] - a[1], L2 = vx * vx + vy * vy;
      const t = L2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2)) : 0;
      const x = a[0] + t * vx, y = a[1] + t * vy, d = Math.hypot(p[0] - x, p[1] - y);
      if (d < best) { best = d; q = [x, y]; }
    }
  }
  return { d: best, q };
}

/** Segments of closed polylines as flat arrays [ax, ay, vx, vy, 1/|v|²] for fast distance queries. */
function segments(curves: Pt[][]): Float64Array {
  const n = curves.reduce((s, c) => s + c.length, 0), out = new Float64Array(5 * n);
  let o = 0;
  for (const c of curves) for (let k = 0; k < c.length; k++) {
    const a = c[k], b = c[(k + 1) % c.length], vx = b[0] - a[0], vy = b[1] - a[1], L2 = vx * vx + vy * vy;
    out[o++] = a[0]; out[o++] = a[1]; out[o++] = vx; out[o++] = vy; out[o++] = L2 > 0 ? 1 / L2 : 0;
  }
  return out;
}

function minDist(px: number, py: number, S: Float64Array): number {
  let best = Infinity;
  for (let o = 0; o < S.length; o += 5) {
    const ax = S[o], ay = S[o + 1], vx = S[o + 2], vy = S[o + 3];
    let t = ((px - ax) * vx + (py - ay) * vy) * S[o + 4];
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const dx = px - ax - t * vx, dy = py - ay - t * vy, d = dx * dx + dy * dy;
    if (d < best) best = d;
  }
  return Math.sqrt(best);
}

/** Boundary distances between two outlines (each a list of closed curves), sampled every `step` px. */
export function curveDistances(A: Pt[] | Pt[][], B: Pt[] | Pt[][], step = 0.25): SurfaceDistances {
  const as = (Array.isArray(A[0][0]) ? A : [A]) as Pt[][], bs = (Array.isArray(B[0][0]) ? B : [B]) as Pt[][];
  const sa = as.flatMap(c => resample(c, step)), sb = bs.flatMap(c => resample(c, step));
  const SA = segments(as), SB = segments(bs);
  return surfaceStats(Float64Array.from(sa, p => minDist(p[0], p[1], SB)), Float64Array.from(sb, p => minDist(p[0], p[1], SA)));
}

/** Inside-intervals [x0, x1] of one closed polygon on the horizontal line y (even–odd rule). */
function rowIntervals(c: Pt[], y: number): [number, number][] {
  const xs: number[] = [];
  for (let i = 0, j = c.length - 1; i < c.length; j = i++) {
    const [xi, yi] = c[i], [xj, yj] = c[j];
    if ((yi > y) !== (yj > y)) xs.push(xi + (y - yi) * (xj - xi) / (yj - yi));
  }
  xs.sort((a, b) => a - b);
  const out: [number, number][] = [];
  for (let k = 0; k + 1 < xs.length; k += 2) out.push([xs[k], xs[k + 1]]);
  return out;
}

/** Union of intervals, sorted and merged. */
function unite(iv: [number, number][]): [number, number][] {
  iv.sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const v of iv) { const l = out[out.length - 1]; if (l && v[0] <= l[1]) l[1] = Math.max(l[1], v[1]); else out.push([v[0], v[1]]); }
  return out;
}

const measure = (iv: [number, number][]) => iv.reduce((s, v) => s + v[1] - v[0], 0);

function intersectLen(a: [number, number][], b: [number, number][]): number {
  let s = 0, i = 0, j = 0;
  while (i < a.length && j < b.length) {
    const lo = Math.max(a[i][0], b[j][0]), hi = Math.min(a[i][1], b[j][1]);
    if (hi > lo) s += hi - lo;
    if (a[i][1] < b[j][1]) i++; else j++;
  }
  return s;
}

/** IoU and Dice of the regions enclosed by A and by B (union of their curves), by scanlines `dy` px apart. */
export function overlap(A: Pt[][], B: Pt[][], dy = 0.1): { iou: number; dice: number; areaA: number; areaB: number } {
  const all = [...A, ...B].flat();
  const y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  let la = 0, lb = 0, li = 0;
  for (let y = y0 + dy / 2; y < y1; y += dy) {
    const ia = unite(A.flatMap(c => rowIntervals(c, y))), ib = unite(B.flatMap(c => rowIntervals(c, y)));
    la += measure(ia); lb += measure(ib); li += intersectLen(ia, ib);
  }
  const u = la + lb - li;
  return { iou: u ? li / u : 1, dice: la + lb ? 2 * li / (la + lb) : 1, areaA: la * dy, areaB: lb * dy };
}

// ---------------------------------------------------------------------------------------
// The prediction: the true outline moved by a smooth offset field. Each control point holds
// an offset [n, t] in pixels along the outward normal and the tangent of the true outline at
// its angle; between control points the offsets are interpolated (periodic Catmull–Rom). So
// zero offsets give exactly the truth, n = 1 everywhere a uniform one-pixel dilation, and the
// same pixel error can be put on a small and on a large object.

export interface Pred {
  offsets: Pt[];                                // NCTRL offsets [normal, tangent] in px
  spike: { on: boolean; phi: number; len: number };
  fragment: { on: boolean; u: number; v: number; r: number };   // position in units of R
}

export const ctrlAngle = (k: number): number => 2 * Math.PI * k / NCTRL;

/** Point, unit outward normal and unit tangent of the true outline at angle θ. */
export function frame(R: number, th: number): { p: Pt; n: Pt; t: Pt } {
  const P = (u: number): Pt => [truthRadius(R, u) * Math.cos(u), truthRadius(R, u) * Math.sin(u)];
  const e = 1e-4, a = P(th - e), b = P(th + e), tx = b[0] - a[0], ty = b[1] - a[1], L = Math.hypot(tx, ty);
  return { p: P(th), t: [tx / L, ty / L], n: [ty / L, -tx / L] };
}

/** Periodic Catmull–Rom interpolation of the control offsets at angle θ. */
function offsetAt(off: Pt[], th: number): Pt {
  const n = off.length, u = ((th / (2 * Math.PI)) * n % n + n) % n, i = Math.floor(u), t = u - i;
  const p0 = off[(i - 1 + n) % n], p1 = off[i], p2 = off[(i + 1) % n], p3 = off[(i + 2) % n];
  const t2 = t * t, t3 = t2 * t;
  const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
  return [f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])];
}

/** Control points of the prediction (screen positions of the handles). */
export function controlPoints(R: number, p: Pred): Pt[] {
  return p.offsets.map(([on, ot], k) => { const f = frame(R, ctrlAngle(k)); return [f.p[0] + on * f.n[0] + ot * f.t[0], f.p[1] + on * f.n[1] + ot * f.t[1]] as Pt; });
}

/** Offset [normal, tangent] of control point k that puts it at position q. */
export function offsetFor(R: number, k: number, q: Pt): Pt {
  const f = frame(R, ctrlAngle(k)), dx = q[0] - f.p[0], dy = q[1] - f.p[1];
  return [dx * f.n[0] + dy * f.n[1], dx * f.t[0] + dy * f.t[1]];
}

/** Uniform offset of d px along the normals. */
export const dilation = (_R: number, d: number): Pt[] => Array.from({ length: NCTRL }, () => [d, 0] as Pt);

/** The predicted outline as closed curves: the main outline (with spike) and the fragment. */
export function predCurves(R: number, p: Pred, n = 240): Pt[][] {
  let main: Pt[] = Array.from({ length: n }, (_, k) => {
    const th = 2 * Math.PI * k / n, f = frame(R, th), [on, ot] = offsetAt(p.offsets, th);
    return [f.p[0] + on * f.n[0] + ot * f.t[0], f.p[1] + on * f.n[1] + ot * f.t[1]] as Pt;
  });
  if (p.spike.on) {
    const f = frame(R, p.spike.phi), [on] = offsetAt(p.offsets, p.spike.phi), L = on + p.spike.len;
    main = withSpike(main, p.spike.phi, [f.p[0] + L * f.n[0], f.p[1] + L * f.n[1]]);
  }
  const out = [main];
  if (p.fragment.on) {
    const cx = p.fragment.u * R, cy = p.fragment.v * R, r = p.fragment.r;
    out.push(Array.from({ length: 24 }, (_, k) => [cx + r * Math.cos(2 * Math.PI * k / 24), cy + r * Math.sin(2 * Math.PI * k / 24)] as Pt));
  }
  return out;
}

export interface Settings { R: number; pred: Pred }

const zero = (): Pt[] => Array.from({ length: NCTRL }, () => [0, 0] as Pt);
const noSpike = { on: false, phi: 0.6, len: 14 }, noFrag = { on: false, u: 1.45, v: -1.25, r: 2 };

export interface Preset { id: string; label: string; make: () => Settings }

export const PRESETS: Preset[] = [
  { id: "free", label: "a typical outline", make: () => ({ R: 40, pred: { offsets: [[1.5, 0], [2, 1], [0.5, 0], [-1.5, -1], [-1, 0], [0.5, 1], [2.5, 0], [1, -1], [-0.5, 0], [-2, 0], [-1, 1], [1, 0]].map(v => v as Pt), spike: { ...noSpike }, fragment: { ...noFrag } } }) },
  { id: "dilate", label: "one-pixel dilation", make: () => ({ R: 40, pred: { offsets: dilation(40, 1), spike: { ...noSpike }, fragment: { ...noFrag } } }) },
  { id: "spike", label: "one thin spike", make: () => ({ R: 40, pred: { offsets: zero(), spike: { ...noSpike, on: true }, fragment: { ...noFrag } } }) },
  { id: "fragment", label: "a stray fragment", make: () => ({ R: 40, pred: { offsets: zero(), spike: { ...noSpike }, fragment: { ...noFrag, on: true } } }) },
  { id: "small", label: "same error, small object", make: () => ({ R: 8, pred: { offsets: dilation(8, 1), spike: { ...noSpike }, fragment: { ...noFrag } } }) },
];

export interface Result { dist: SurfaceDistances; iou: number; dice: number; truth: Pt[]; pred: Pt[][] }

export function evaluate(s: Settings): Result {
  const truth = truthCurve(s.R), pred = predCurves(s.R, s.pred);
  const dist = curveDistances([truth], pred, s.R < 15 ? 0.2 : 0.5);
  const ov = overlap([truth], pred, s.R < 15 ? 0.04 : 0.2);
  return { dist, iou: ov.iou, dice: ov.dice, truth, pred };
}
