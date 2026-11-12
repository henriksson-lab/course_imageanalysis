// Hough transforms (no DOM): lines over (theta, rho), circles over (a, b, r), the
// generalised Hough transform (R-table, rotation and scale), peak finding with
// non-maximum suppression on 2D and 3D accumulators, segments along a detected line, and
// recorders of the votes of one point (for the "click a point to see its votes" overlays).
// Owner: the Hough day (hough-lines/, hough-circles/, ght/, ransac/).
//
// Conventions
//  - Lines (shared with lib/radon.ts): world coordinates with the origin at the image
//    centre, x = col - (w - 1)/2 (right), y = (h - 1)/2 - row (UP), so angles turn
//    counter-clockwise on screen. A line is {x cos(theta) + y sin(theta) = rho}, theta in
//    [0, pi) the direction of its normal, rho signed. Accumulator: one row per angle,
//    acc[j * nRho + k], theta_j = j pi / nTheta, rho_k = (k - (nRho - 1)/2) dRho.
//  - Edge directions are lib/edges.ts's `theta`: the gradient direction on screen,
//    counter-clockwise, radians (atan2(-gy, gx)). In pixel coordinates (y down) the unit
//    gradient is (cos theta, -sin theta).
//  - Circles and the GHT work in pixel coordinates (col, row); accumulators are stacks of
//    w x h slices, slice-major: acc[s * w * h + row * w + col].
//  - Accumulators are flat Float32Arrays. A vote adds `weight` (default 1).

export interface Pt { x: number; y: number }
/** An oriented point: pixel coordinates and, optionally, its gradient direction (lib/edges.ts convention). */
export interface OrientedPt { x: number; y: number; theta?: number; strength?: number }

const TAU = 2 * Math.PI;
const wrapPi = (a: number) => { a %= Math.PI; return a < 0 ? a + Math.PI : a; };
const wrap2Pi = (a: number) => { a %= TAU; return a < 0 ? a + TAU : a; };

// ---------------------------------------------------------------------------------------
// Lines

export interface LineGeometry {
  /** Image size (for the world origin and the rho range). */
  w: number; h: number;
  nTheta: number; nRho: number; dRho: number;
  /** World origin in pixel coordinates. */
  cx: number; cy: number;
}

/** Geometry for a w x h image: nTheta angles over 180 degrees, bins of dRho covering +- half the diagonal. */
export function lineGeometry(w: number, h: number, nTheta = 180, dRho = 1): LineGeometry {
  const half = Math.hypot(w - 1, h - 1) / 2;
  const kHalf = Math.ceil(half / dRho);
  return { w, h, nTheta, nRho: 2 * kHalf + 1, dRho, cx: (w - 1) / 2, cy: (h - 1) / 2 };
}

export const thetaOf = (g: LineGeometry, j: number): number => j * Math.PI / g.nTheta;
export const rhoOf = (g: LineGeometry, k: number): number => (k - (g.nRho - 1) / 2) * g.dRho;
/** Continuous bin coordinate of rho (round it for the bin index). */
export const rhoBin = (g: LineGeometry, rho: number): number => rho / g.dRho + (g.nRho - 1) / 2;
/** World coordinates of a pixel. */
export const toWorld = (g: { cx: number; cy: number }, col: number, row: number): [number, number] => [col - g.cx, g.cy - row];
export const toPixel = (g: { cx: number; cy: number }, x: number, y: number): [number, number] => [x + g.cx, g.cy - y];
/** rho of the line with normal angle theta through world point (x, y): the point's sinusoid. */
export const sinusoid = (x: number, y: number, theta: number): number => x * Math.cos(theta) + y * Math.sin(theta);

export interface LineVoteOptions {
  /** Vote only within +- window (radians) of each point's gradient direction (needs theta); undefined = all angles. */
  gradWindow?: number;
  /** Use the point's strength as the vote weight (default false: one vote per bin). */
  weighted?: boolean;
}

export interface LineAccumulator { g: LineGeometry; acc: Float32Array; votes: number }

/**
 * The line Hough transform: every point (pixel coordinates) adds one vote to each bin
 * (theta_j, round(rho)) its sinusoid passes through; with `gradWindow` only to the angles
 * within the window around its gradient direction (taken modulo 180 degrees).
 */
export function houghLines(points: OrientedPt[], g: LineGeometry, o: LineVoteOptions = {}): LineAccumulator {
  const { nTheta, nRho } = g, acc = new Float32Array(nTheta * nRho);
  const cs = new Float64Array(nTheta), sn = new Float64Array(nTheta);
  for (let j = 0; j < nTheta; j++) { cs[j] = Math.cos(thetaOf(g, j)); sn[j] = Math.sin(thetaOf(g, j)); }
  const off = (nRho - 1) / 2, inv = 1 / g.dRho;
  let votes = 0;
  const useGrad = o.gradWindow !== undefined;
  const dth = Math.PI / nTheta;
  for (const p of points) {
    const x = p.x - g.cx, y = g.cy - p.y, wgt = o.weighted ? (p.strength ?? 1) : 1;
    let j0 = 0, j1 = nTheta - 1;
    if (useGrad && p.theta !== undefined) {
      const t = wrapPi(p.theta) / dth, half = o.gradWindow! / dth;
      j0 = Math.round(t - half); j1 = Math.round(t + half);
      if (j1 - j0 >= nTheta) { j0 = 0; j1 = nTheta - 1; }
    }
    for (let jj = j0; jj <= j1; jj++) {
      // an angle outside [0, pi) is the same line with theta - pi and -rho
      let j = jj, sgn = 1;
      if (j < 0) { j += nTheta; sgn = -1; } else if (j >= nTheta) { j -= nTheta; sgn = -1; }
      const k = Math.round(sgn * (x * cs[j] + y * sn[j]) * inv + off);
      if (k < 0 || k >= nRho) continue;
      acc[j * nRho + k] += wgt;
      votes++;
    }
  }
  return { g, acc, votes };
}

/** The bins one point votes for: [j, k] pairs (same rules as houghLines). */
export function lineVotesOf(p: OrientedPt, g: LineGeometry, o: LineVoteOptions = {}): [number, number][] {
  const a = houghLines([p], g, o).acc, out: [number, number][] = [];
  for (let i = 0; i < a.length; i++) if (a[i]) out.push([Math.floor(i / g.nRho), i % g.nRho]);
  return out;
}

export interface Peak2 { j: number; k: number; value: number }

/**
 * Peaks of a line accumulator: bins >= threshold that are the maximum of their
 * (2 win + 1)^2 window (win = 0: no NMS, every bin above the threshold). The window wraps
 * in theta: theta_j - pi is the same line as (theta_j, -rho). Ties: the first bin in
 * raster order wins. Sorted by value, strongest first; at most maxPeaks.
 */
export function linePeaks(L: LineAccumulator, threshold: number, win = 3, maxPeaks = 200): Peak2[] {
  const { acc, g } = L, { nTheta: nT, nRho: nR } = g, out: Peak2[] = [];
  for (let j = 0; j < nT; j++) for (let k = 0; k < nR; k++) {
    const v = acc[j * nR + k];
    if (v < threshold || v <= 0) continue;
    let ok = true;
    for (let dj = -win; dj <= win && ok; dj++) for (let dk = -win; dk <= win; dk++) {
      if (!dj && !dk) continue;
      let jj = j + dj, kk = k + dk;
      if (jj < 0) { jj += nT; kk = nR - 1 - kk; } else if (jj >= nT) { jj -= nT; kk = nR - 1 - kk; }
      if (kk < 0 || kk >= nR) continue;
      const u = acc[jj * nR + kk];
      // strictly larger neighbour, or an equal one earlier in raster order (after unwrapping)
      if (u > v || (u === v && (dj < 0 || (dj === 0 && dk < 0)))) { ok = false; break; }
    }
    if (ok) out.push({ j, k, value: v });
  }
  out.sort((a, b) => b.value - a.value);
  return out.slice(0, maxPeaks);
}

/** Dominant angle: the theta bin with the largest sum of squared votes (squares favour long lines over clutter). */
export function dominantAngle(L: LineAccumulator): { j: number; theta: number; profile: Float64Array } {
  const { acc, g } = L, prof = new Float64Array(g.nTheta);
  for (let j = 0; j < g.nTheta; j++) { let s = 0; for (let k = 0; k < g.nRho; k++) { const v = acc[j * g.nRho + k]; s += v * v; } prof[j] = s; }
  let jb = 0;
  for (let j = 1; j < g.nTheta; j++) if (prof[j] > prof[jb]) jb = j;
  return { j: jb, theta: thetaOf(g, jb), profile: prof };
}

/** End points (pixel coordinates) where the line (theta, rho) crosses the image rectangle, or null. */
export function clipLine(g: { w: number; h: number; cx: number; cy: number }, theta: number, rho: number): [number, number, number, number] | null {
  const c = Math.cos(theta), s = Math.sin(theta);
  // points x c + y s = rho in world coords; walk along direction (-s, c) from the foot point
  const fx = rho * c, fy = rho * s;
  const xs = [-g.cx - 0.5, g.w - 1 - g.cx + 0.5], ys = [g.cy - (g.h - 1) - 0.5, g.cy + 0.5];
  let t0 = -Infinity, t1 = Infinity;
  const dx = -s, dy = c;
  for (const [p, d, lo, hi] of [[fx, dx, xs[0], xs[1]], [fy, dy, ys[0], ys[1]]] as [number, number, number, number][]) {
    if (Math.abs(d) < 1e-12) { if (p < lo || p > hi) return null; continue; }
    let a = (lo - p) / d, b = (hi - p) / d;
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a); t1 = Math.min(t1, b);
  }
  if (t0 > t1) return null;
  const [c0, r0] = toPixel(g, fx + t0 * dx, fy + t0 * dy), [c1, r1] = toPixel(g, fx + t1 * dx, fy + t1 * dy);
  return [c0, r0, c1, r1];
}

export interface Segment { x0: number; y0: number; x1: number; y1: number; length: number; count: number }

/**
 * Segments along the line (theta, rho): the points within `tol` pixels of it, sorted along
 * the line, split where consecutive points are more than `gap` apart; pieces shorter than
 * `minLength` are dropped. End points in pixel coordinates (projected onto the line).
 */
export function lineSegments(points: Pt[], g: { cx: number; cy: number }, theta: number, rho: number, tol = 1, gap = 5, minLength = 10): Segment[] {
  const c = Math.cos(theta), s = Math.sin(theta), ts: number[] = [];
  for (const p of points) {
    const x = p.x - g.cx, y = g.cy - p.y;
    if (Math.abs(x * c + y * s - rho) <= tol) ts.push(-x * s + y * c);
  }
  ts.sort((a, b) => a - b);
  const out: Segment[] = [];
  const emit = (a: number, b: number, n: number) => {
    if (b - a < minLength) return;
    const [x0, y0] = toPixel(g, rho * c - a * s, rho * s + a * c), [x1, y1] = toPixel(g, rho * c - b * s, rho * s + b * c);
    out.push({ x0, y0, x1, y1, length: b - a, count: n });
  };
  let start = 0;
  for (let i = 1; i <= ts.length; i++) {
    if (i === ts.length || ts[i] - ts[i - 1] > gap) { if (ts.length) emit(ts[start], ts[i - 1], i - start); start = i; }
  }
  return out;
}

/** The (theta, rho) of the line through two world points (theta in [0, pi)). */
export function lineThrough(x1: number, y1: number, x2: number, y2: number): { theta: number; rho: number } {
  // the normal is perpendicular to the direction (x2 - x1, y2 - y1)
  let theta = Math.atan2(x2 - x1, -(y2 - y1));
  let rho = x1 * Math.cos(theta) + y1 * Math.sin(theta);
  if (theta < 0) { theta += Math.PI; rho = -rho; }
  if (theta >= Math.PI) { theta -= Math.PI; rho = -rho; }
  return { theta, rho };
}

// ---------------------------------------------------------------------------------------
// Circles

export type Polarity = "bright" | "dark" | "either";

/** Integer offsets (dx, dy) with |sqrt(dx^2 + dy^2) - r| < 0.5: a digital ring of radius r. */
export function ringOffsets(r: number): Int32Array {
  const R = Math.ceil(r + 0.5), out: number[] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const d = Math.hypot(dx, dy);
    if (Math.abs(d - r) < 0.5) out.push(dx, dy);
  }
  return Int32Array.from(out);
}

export interface CircleOptions {
  rMin: number; rMax: number;
  /** Vote along the gradient only (needs theta). */
  grad?: boolean;
  /** With gradient voting: bright disc on dark (centre along +gradient), dark on bright, or both. */
  polarity?: Polarity;
}

export interface CircleAccumulator { w: number; h: number; rMin: number; nR: number; acc: Float32Array; votes: number }

/**
 * The circle Hough transform for integer radii rMin..rMax. Full voting: each point adds a
 * vote at every pixel of the digital ring of radius r around it (a ring kernel). Gradient
 * voting: one (or two, polarity "either") vote per radius at p +- r u, u the unit gradient,
 * split bilinearly between the four nearest pixels.
 */
export function houghCircles(points: OrientedPt[], w: number, h: number, o: CircleOptions): CircleAccumulator {
  const nR = o.rMax - o.rMin + 1, S = w * h, acc = new Float32Array(nR * S);
  let votes = 0;
  if (!o.grad) {
    for (let ri = 0; ri < nR; ri++) {
      const r = o.rMin + ri, ring = ringOffsets(r), base = ri * S, R = Math.ceil(r + 0.5), m = ring.length / 2;
      const lin = new Int32Array(m);
      for (let q = 0; q < m; q++) lin[q] = ring[2 * q + 1] * w + ring[2 * q];
      for (const p of points) {
        const px = Math.round(p.x), py = Math.round(p.y);
        if (px >= R && py >= R && px < w - R && py < h - R) {
          // the whole ring is inside: no bounds checks
          const c = base + py * w + px;
          for (let q = 0; q < m; q++) acc[c + lin[q]] += 1;
          votes += m;
          continue;
        }
        for (let q = 0; q < ring.length; q += 2) {
          const a = px + ring[q], b = py + ring[q + 1];
          if (a < 0 || b < 0 || a >= w || b >= h) continue;
          acc[base + b * w + a] += 1; votes++;
        }
      }
    }
    return { w, h, rMin: o.rMin, nR, acc, votes };
  }
  const pol = o.polarity ?? "either";
  const signs = pol === "bright" ? [1] : pol === "dark" ? [-1] : [1, -1];
  for (const p of points) {
    if (p.theta === undefined) continue;
    const ux = Math.cos(p.theta), uy = -Math.sin(p.theta);
    for (let ri = 0; ri < nR; ri++) {
      const r = o.rMin + ri, base = ri * S;
      for (const sg of signs) {
        if (splat(acc, base, w, h, p.x + sg * r * ux, p.y + sg * r * uy, 1)) votes++;
      }
    }
  }
  return { w, h, rMin: o.rMin, nR, acc, votes };
}

/** Bilinear splat of `v` at (x, y) into slice `base`; false if outside. */
function splat(acc: Float32Array, base: number, w: number, h: number, x: number, y: number, v: number): boolean {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  if (x0 < -1 || y0 < -1 || x0 >= w || y0 >= h) return false;
  const put = (a: number, b: number, q: number) => { if (a >= 0 && b >= 0 && a < w && b < h && q > 0) acc[base + b * w + a] += q; };
  put(x0, y0, v * (1 - fx) * (1 - fy)); put(x0 + 1, y0, v * fx * (1 - fy));
  put(x0, y0 + 1, v * (1 - fx) * fy); put(x0 + 1, y0 + 1, v * fx * fy);
  return true;
}

/** Centre votes of one point at radius r (pixel coordinates, before splatting): the ring, or 1-2 points with gradient voting. */
export function circleVotesOf(p: OrientedPt, r: number, grad: boolean, polarity: Polarity = "either"): [number, number][] {
  if (!grad || p.theta === undefined) {
    const ring = ringOffsets(r), out: [number, number][] = [];
    for (let q = 0; q < ring.length; q += 2) out.push([Math.round(p.x) + ring[q], Math.round(p.y) + ring[q + 1]]);
    return out;
  }
  const ux = Math.cos(p.theta), uy = -Math.sin(p.theta);
  const signs = polarity === "bright" ? [1] : polarity === "dark" ? [-1] : [1, -1];
  return signs.map(sg => [p.x + sg * r * ux, p.y + sg * r * uy] as [number, number]);
}

/** One slice of a circle accumulator (radius index ri), as a view. */
export const circleSlice = (C: CircleAccumulator, ri: number): Float32Array => C.acc.subarray(ri * C.w * C.h, (ri + 1) * C.w * C.h);

/** Normalise each radius slice by 2 pi r: the value is then the fraction of the circumference that voted. */
export function normaliseByCircumference(C: CircleAccumulator): Float32Array {
  const S = C.w * C.h, out = new Float32Array(C.acc.length);
  for (let ri = 0; ri < C.nR; ri++) {
    const f = 1 / (TAU * (C.rMin + ri));
    for (let i = 0; i < S; i++) out[ri * S + i] = C.acc[ri * S + i] * f;
  }
  return out;
}

/** Maximum over the stack of slices, and the slice index of the maximum. */
export function maxOverSlices(acc: Float32Array, S: number, n: number): { max: Float32Array; arg: Int32Array } {
  const max = new Float32Array(S), arg = new Int32Array(S);
  for (let s = 0; s < n; s++) for (let i = 0; i < S; i++) { const v = acc[s * S + i]; if (v > max[i] || s === 0) { max[i] = v; arg[i] = s; } }
  return { max, arg };
}

export interface Peak3 { x: number; y: number; s: number; value: number }

/**
 * Peaks of a stack of n slices of size w x h: values >= threshold that are the maximum of a
 * (2 win + 1)^2 x (2 winS + 1) window (ties: first in raster order). Strongest first.
 */
export function peaks3(acc: Float32Array, w: number, h: number, n: number, threshold: number, win = 3, winS = 1, maxPeaks = 100): Peak3[] {
  const S = w * h, out: Peak3[] = [];
  for (let s = 0; s < n; s++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = acc[s * S + y * w + x];
    if (v < threshold || v <= 0) continue;
    let ok = true;
    for (let ds = -winS; ds <= winS && ok; ds++) {
      const ss = s + ds;
      if (ss < 0 || ss >= n) continue;
      for (let dy = -win; dy <= win && ok; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -win; dx <= win; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w || (!ds && !dy && !dx)) continue;
          const u = acc[ss * S + yy * w + xx];
          if (u > v || (u === v && (ds < 0 || (ds === 0 && (dy < 0 || (dy === 0 && dx < 0)))))) { ok = false; break; }
        }
      }
    }
    if (ok) out.push({ x, y, s, value: v });
  }
  out.sort((a, b) => b.value - a.value);
  return out.slice(0, maxPeaks);
}

/** Peaks of one 2D map (a single slice). */
export const peaks2 = (map: Float32Array, w: number, h: number, threshold: number, win = 3, maxPeaks = 100): Peak3[] =>
  peaks3(map, w, h, 1, threshold, win, 0, maxPeaks);

// ---------------------------------------------------------------------------------------
// Generalised Hough transform

export interface RTable {
  nBins: number;
  /** Per direction bin: the vectors (dx, dy) from the edge point to the reference point, pixel coordinates. */
  bins: [number, number][][];
  /** Number of entries. */
  size: number;
}

/** Direction bin of a gradient direction (radians, any range) among nBins over 360 degrees. */
export const dirBin = (theta: number, nBins: number): number => Math.floor(wrap2Pi(theta) / TAU * nBins + 0.5) % nBins;

/** Build the R-table of a template: for each oriented edge point, store ref - p under the bin of its direction. */
export function buildRTable(points: OrientedPt[], ref: Pt, nBins: number): RTable {
  const bins: [number, number][][] = Array.from({ length: nBins }, () => []);
  let size = 0;
  for (const p of points) {
    if (p.theta === undefined) continue;
    bins[dirBin(p.theta, nBins)].push([ref.x - p.x, ref.y - p.y]);
    size++;
  }
  return { nBins, bins, size };
}

/**
 * Rotate a pixel-coordinate vector by phi counter-clockwise ON SCREEN (y down) and scale by s.
 * A shape rotated this way has its gradient directions (lib/edges.ts convention) increased by phi.
 */
export function rotScale(v: [number, number], phi: number, s: number): [number, number] {
  const c = Math.cos(phi), sn = Math.sin(phi);
  return [s * (c * v[0] + sn * v[1]), s * (-sn * v[0] + c * v[1])];
}

export interface GHTOptions {
  /** Rotation angles (radians, CCW on screen) and scales to search; default [0] and [1]. */
  phis?: number[];
  scales?: number[];
  /** Accumulator cell size in pixels (default 1). */
  cell?: number;
  /**
   * Tolerance in cells (default 0). With tol > 0 a cell counts the number of distinct edge
   * points that vote within +-tol cells of it (each point at most once per cell), so an edge
   * pixel half a pixel off the true boundary still supports the right reference point.
   */
  tol?: number;
}

export interface GHTAccumulator { w: number; h: number; cell: number; phis: number[]; scales: number[]; acc: Float32Array; votes: number; lookups: number }

/**
 * GHT voting. For each scene point with direction theta and every (phi, s): look up the
 * table bin of theta - phi and vote for p + s R(phi) v for every vector v there. Slices are
 * ordered scale-major: slice = si * phis.length + pi. Accumulator cells of `cell` pixels.
 */
export function ghtVote(points: OrientedPt[], table: RTable, w: number, h: number, o: GHTOptions = {}): GHTAccumulator {
  const phis = o.phis ?? [0], scales = o.scales ?? [1], cell = o.cell ?? 1, tol = o.tol ?? 0;
  const W = Math.ceil(w / cell), H = Math.ceil(h / cell), S = W * H;
  const acc = new Float32Array(S * phis.length * scales.length);
  const stamp = tol > 0 ? new Int32Array(S).fill(-1) : null;
  let votes = 0, lookups = 0, stampId = 0;
  for (let si = 0; si < scales.length; si++) for (let pi = 0; pi < phis.length; pi++) {
    const phi = phis[pi], s = scales[si], c = Math.cos(phi), sn = Math.sin(phi), base = (si * phis.length + pi) * S;
    for (const p of points) {
      if (p.theta === undefined) continue;
      const bin = table.bins[dirBin(p.theta - phi, table.nBins)];
      lookups++;
      stampId++;
      for (const v of bin) {
        const x = p.x + s * (c * v[0] + sn * v[1]), y = p.y + s * (-sn * v[0] + c * v[1]);
        const a = Math.round(x / cell), b = Math.round(y / cell);
        if (a < -tol || b < -tol || a >= W + tol || b >= H + tol) continue;
        votes++;
        if (!stamp) { if (a >= 0 && b >= 0 && a < W && b < H) acc[base + b * W + a] += 1; continue; }
        for (let db = -tol; db <= tol; db++) {
          const bb = b + db;
          if (bb < 0 || bb >= H) continue;
          for (let da = -tol; da <= tol; da++) {
            const aa = a + da;
            if (aa < 0 || aa >= W) continue;
            const k = bb * W + aa;
            if (stamp[k] !== stampId) { stamp[k] = stampId; acc[base + k] += 1; }
          }
        }
      }
    }
  }
  return { w: W, h: H, cell, phis, scales, acc, votes, lookups };
}

/** The reference-point votes of one scene point for one (phi, s), pixel coordinates. */
export function ghtVotesOf(p: OrientedPt, table: RTable, phi = 0, s = 1): [number, number][] {
  if (p.theta === undefined) return [];
  return table.bins[dirBin(p.theta - phi, table.nBins)].map(v => {
    const [dx, dy] = rotScale(v, phi, s);
    return [p.x + dx, p.y + dy] as [number, number];
  });
}

// ---------------------------------------------------------------------------------------
// Polygons as oriented point sets (templates and synthetic scenes)

/**
 * Sample a closed polygon (pixel coordinates) every `step` pixels along its boundary, with
 * the gradient direction of a BRIGHT filled polygon on a dark background (the inward
 * normal), in the lib/edges.ts convention. Works for either vertex order.
 */
export function polygonEdgePoints(poly: [number, number][], step = 1): OrientedPt[] {
  const n = poly.length, out: OrientedPt[] = [];
  let area2 = 0;
  for (let i = 0; i < n; i++) { const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % n]; area2 += x0 * y1 - x1 * y0; }
  const orient = area2 > 0 ? 1 : -1;   // > 0: clockwise on screen (y down)
  for (let i = 0; i < n; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % n];
    const L = Math.hypot(x1 - x0, y1 - y0);
    if (L < 1e-9) continue;
    const tx = (x1 - x0) / L, ty = (y1 - y0) / L;
    // inward normal in pixel coordinates: for clockwise-on-screen order it is (-ty, tx)
    const nx = -ty * orient, ny = tx * orient;
    const theta = Math.atan2(-ny, nx);
    const m = Math.max(1, Math.round(L / step));
    for (let k = 0; k < m; k++) out.push({ x: x0 + (x1 - x0) * k / m, y: y0 + (y1 - y0) * k / m, theta });
  }
  return out;
}

/** Centroid of the vertices (a simple default reference point). */
export function vertexMean(poly: [number, number][]): Pt {
  let x = 0, y = 0;
  for (const [a, b] of poly) { x += a; y += b; }
  return { x: x / poly.length, y: y / poly.length };
}

/** Transform a polygon: rotate by phi (CCW on screen) and scale by s about `about`, then translate to `to`. */
export function placePolygon(poly: [number, number][], about: Pt, phi: number, s: number, to: Pt): [number, number][] {
  return poly.map(([x, y]) => {
    const [dx, dy] = rotScale([x - about.x, y - about.y], phi, s);
    return [to.x + dx, to.y + dy] as [number, number];
  });
}

/** Fill polygons into a w x h image (even-odd rule per polygon, with ss x ss supersampling): value where covered. */
export function fillPolygons(w: number, h: number, polys: { poly: [number, number][]; value: number }[], base = 0, ss = 3): Float32Array {
  const img = new Float32Array(w * h).fill(base);
  for (const { poly, value } of polys) {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of poly) { xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
    const y0 = Math.max(0, Math.floor(ymin) - 1), y1 = Math.min(h - 1, Math.ceil(ymax) + 1);
    const x0 = Math.max(0, Math.floor(xmin) - 1), x1 = Math.min(w - 1, Math.ceil(xmax) + 1);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      let cov = 0;
      for (let sy = 0; sy < ss; sy++) for (let sx = 0; sx < ss; sx++) {
        const px = x - 0.5 + (sx + 0.5) / ss, py = y - 0.5 + (sy + 0.5) / ss;
        if (insidePoly(poly, px, py)) cov++;
      }
      if (cov) { const f = cov / (ss * ss), i = y * w + x; img[i] = img[i] * (1 - f) + value * f; }
    }
  }
  return img;
}

export function insidePoly(poly: [number, number][], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Regular polygon approximating a circle (for "the circle template is the circle Hough transform"). */
export function circlePoly(cx: number, cy: number, r: number, n = 64): [number, number][] {
  return Array.from({ length: n }, (_, i) => [cx + r * Math.cos(TAU * i / n), cy + r * Math.sin(TAU * i / n)] as [number, number]);
}

// ---------------------------------------------------------------------------------------
// Counting (for readouts and the by-hand questions)

/** Number of line-accumulator bins and votes for nPoints edge points, with and without gradient voting. */
export function lineCost(nPoints: number, nTheta: number, nRho: number, gradBins = 1): { bins: number; votesFull: number; votesGrad: number } {
  return { bins: nTheta * nRho, votesFull: nPoints * nTheta, votesGrad: nPoints * gradBins };
}

