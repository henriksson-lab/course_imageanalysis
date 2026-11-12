// Level set toolkit (owner: level-set day; used by the demos levelsurface, curveflow, gac,
// chanvese). No DOM: safe to import from tests.
//
//   const phi = polygonSDF([poly], 128, 128);          // signed distance, negative inside
//   const T: SpeedTerms = { F: 1, eps: 0.5 };          // speed F − ε κ along the outward normal
//   const dt = 0.5 * cflLimit(T);
//   lsStep(phi, 128, 128, T, dt);                       // one forward-Euler step, upwind
//   const C = zeroContour(phi, 128, 128);               // marching squares (lib/marching.ts)
//   reinitExact(phi, 128, 128);                         // back to a signed distance function
//
// Conventions (printed on the slides)
//  - φ is a w × h grid, row-major, pixel spacing 1 (Δx = 1), y down. φ < 0 INSIDE the curve,
//    φ > 0 outside; the curve is the zero level. The outward normal is n = ∇φ/|∇φ|, the
//    curvature κ = ∇·n (a circle of radius r has κ = 1/r > 0).
//  - Level set equation: φ_t + F |∇φ| = 0, F = speed along the outward normal (F > 0 grows).
//  - Borders: replicate (one-sided differences become 0 across the image border).
//  - Upwind |∇φ| (Godunov, Osher–Sethian): for F > 0
//      |∇φ|² ≈ max(D⁻x, 0)² + min(D⁺x, 0)² + max(D⁻y, 0)² + min(D⁺y, 0)²
//    and for F < 0 the same with max and min swapped. D⁻x φ = φ(i) − φ(i−1), D⁺x φ = φ(i+1) − φ(i).
//  - The curvature term uses central differences (it is diffusion-like, not a wave).
//  - Chan–Vese with φ < 0 inside: c1 = mean inside, c2 = mean outside, and gradient descent
//      φ_t = δ_ε(φ) [ λ κ + ν + w1 (I − c1)² − w2 (I − c2)² ]
//    (positive bracket pushes the pixel outside).
import { marchingSquares, Contour, contourLength, contourArea } from "./marching";

export type Poly = [number, number][];

// =========================================================================================
// Building φ

/** φ(x, y) = f(x, y) at every pixel centre (x = column, y = row). */
export function fromFn(w: number, h: number, f: (x: number, y: number) => number): Float64Array {
  const phi = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) phi[y * w + x] = f(x, y);
  return phi;
}

/** Signed distance to a circle: √((x−cx)² + (y−cy)²) − r. */
export function sdfCircle(w: number, h: number, cx: number, cy: number, r: number): Float64Array {
  return fromFn(w, h, (x, y) => Math.hypot(x - cx, y - cy) - r);
}

/** Exact signed distance to an axis-aligned box (centre, half sides). */
export function sdfBox(px: number, py: number, cx: number, cy: number, hx: number, hy: number): number {
  const dx = Math.abs(px - cx) - hx, dy = Math.abs(py - cy) - hy;
  const ox = Math.max(dx, 0), oy = Math.max(dy, 0);
  return Math.sqrt(ox * ox + oy * oy) + Math.min(Math.max(dx, dy), 0);
}

/** Distance from (px, py) to the segment a–b. */
export function pointSegDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const ux = bx - ax, uy = by - ay, l2 = ux * ux + uy * uy;
  let t = l2 > 0 ? ((px - ax) * ux + (py - ay) * uy) / l2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const ex = px - ax - t * ux, ey = py - ay - t * uy;
  return Math.sqrt(ex * ex + ey * ey);
}

/** Is (px, py) inside the polygons (even–odd rule over all of them)? */
export function insideEvenOdd(px: number, py: number, polys: Poly[]): boolean {
  let inside = false;
  for (const P of polys) {
    for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
      const [xi, yi] = P[i], [xj, yj] = P[j];
      if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/**
 * Signed distance to closed polygons: exact distance to the nearest edge, negative inside
 * (even–odd rule). Coordinates in pixels (x = column, y = row).
 */
export function polygonSDF(polys: Poly[], w: number, h: number): Float64Array {
  const segs: number[] = [];
  for (const P of polys) for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length];
    segs.push(a[0], a[1], b[0], b[1]);
  }
  const d = distanceToSegments(segs, w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (insideEvenOdd(x, y, polys)) d[y * w + x] = -d[y * w + x];
  return d;
}

/**
 * Unsigned distance from every pixel centre to a set of segments (x0 y0 x1 y1 per segment).
 * Exact; segments are bucketed in 8 × 8 cells and searched ring by ring.
 */
export function distanceToSegments(segs: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h).fill(Infinity), ns = segs.length / 4;
  if (ns === 0) return out;
  const B = 8, bw = Math.ceil(w / B) + 1, bh = Math.ceil(h / B) + 1;
  const buckets: number[][] = Array.from({ length: bw * bh }, () => []);
  const clampB = (v: number, n: number) => Math.max(0, Math.min(n - 1, Math.floor(v / B)));
  for (let s = 0; s < ns; s++) {
    const x0 = segs[4 * s], y0 = segs[4 * s + 1], x1 = segs[4 * s + 2], y1 = segs[4 * s + 3];
    const i0 = clampB(Math.min(x0, x1), bw), i1 = clampB(Math.max(x0, x1), bw);
    const j0 = clampB(Math.min(y0, y1), bh), j1 = clampB(Math.max(y0, y1), bh);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) buckets[j * bw + i].push(s);
  }
  const maxR = Math.max(bw, bh);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const bi = clampB(x, bw), bj = clampB(y, bh);
    let best = Infinity;
    for (let r = 0; r <= maxR; r++) {
      // every cell in ring r is at least (r − 1) B away
      if (best <= (r - 1) * B) break;
      for (let j = bj - r; j <= bj + r; j++) {
        if (j < 0 || j >= bh) continue;
        const edge = j === bj - r || j === bj + r;
        for (let i = bi - r; i <= bi + r; i += edge ? 1 : 2 * r || 1) {
          if (i < 0 || i >= bw) continue;
          for (const s of buckets[j * bw + i]) {
            const d = pointSegDist(x, y, segs[4 * s], segs[4 * s + 1], segs[4 * s + 2], segs[4 * s + 3]);
            if (d < best) best = d;
          }
        }
      }
    }
    out[y * w + x] = best;
  }
  return out;
}

/** Union (min), intersection (max) and complement (−φ) of implicit shapes. */
export function union(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array { return Float64Array.from(a, (v, i) => Math.min(v, b[i])); }
export function intersection(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array { return Float64Array.from(a, (v, i) => Math.max(v, b[i])); }
export function complement(a: ArrayLike<number>): Float64Array { return Float64Array.from(a, v => -v); }

/** A regular-ish closed polygon from a radius function r(θ), n vertices (counter-clockwise on screen is not assumed). */
export function radialPoly(cx: number, cy: number, r: (t: number) => number, n = 120): Poly {
  return Array.from({ length: n }, (_, k) => { const t = 2 * Math.PI * k / n, rr = r(t); return [cx + rr * Math.cos(t), cy + rr * Math.sin(t)] as [number, number]; });
}

// =========================================================================================
// Derivatives

const at = (phi: ArrayLike<number>, w: number, h: number, x: number, y: number): number =>
  phi[(y < 0 ? 0 : y >= h ? h - 1 : y) * w + (x < 0 ? 0 : x >= w ? w - 1 : x)];

/** Central-difference gradient at pixel (x, y) (one-sided at the border). */
export function gradAt(phi: ArrayLike<number>, w: number, h: number, x: number, y: number): [number, number] {
  const gx = x === 0 ? at(phi, w, h, 1, y) - at(phi, w, h, 0, y) : x === w - 1 ? at(phi, w, h, x, y) - at(phi, w, h, x - 1, y) : (at(phi, w, h, x + 1, y) - at(phi, w, h, x - 1, y)) / 2;
  const gy = y === 0 ? at(phi, w, h, x, 1) - at(phi, w, h, x, 0) : y === h - 1 ? at(phi, w, h, x, y) - at(phi, w, h, x, y - 1) : (at(phi, w, h, x, y + 1) - at(phi, w, h, x, y - 1)) / 2;
  return [gx, gy];
}

/**
 * Curvature κ = ∇·(∇φ/|∇φ|) at pixel (x, y) by central differences (grid spacing `hs`):
 * κ = (φxx φy² − 2 φx φy φxy + φyy φx²) / (φx² + φy²)^{3/2}. Not clamped.
 */
export function curvatureAt(phi: ArrayLike<number>, w: number, h: number, x: number, y: number, hs = 1): number {
  const c = at(phi, w, h, x, y);
  const l = at(phi, w, h, x - 1, y), r = at(phi, w, h, x + 1, y), u = at(phi, w, h, x, y - 1), d = at(phi, w, h, x, y + 1);
  const fx = (r - l) / (2 * hs), fy = (d - u) / (2 * hs);
  const fxx = (r - 2 * c + l) / (hs * hs), fyy = (d - 2 * c + u) / (hs * hs);
  const fxy = (at(phi, w, h, x + 1, y + 1) - at(phi, w, h, x + 1, y - 1) - at(phi, w, h, x - 1, y + 1) + at(phi, w, h, x - 1, y - 1)) / (4 * hs * hs);
  const g2 = fx * fx + fy * fy;
  return (fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / Math.pow(g2 + 1e-12, 1.5);
}

/** κ |∇φ| at every pixel (central differences), with κ clamped to ±kmax (default 1/Δx = 1). */
export function curvatureTerm(phi: ArrayLike<number>, w: number, h: number, kmax = 1, out = new Float64Array(w * h)): Float64Array {
  for (let y = 0; y < h; y++) {
    const ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
      const c = phi[y * w + x], l = phi[y * w + xm], r = phi[y * w + xp], u = phi[ym * w + x], d = phi[yp * w + x];
      const fx = (r - l) / 2, fy = (d - u) / 2, fxx = r - 2 * c + l, fyy = d - 2 * c + u;
      const fxy = (phi[yp * w + xp] - phi[ym * w + xp] - phi[yp * w + xm] + phi[ym * w + xm]) / 4;
      const g2 = fx * fx + fy * fy, g = Math.sqrt(g2);
      let k = (fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / Math.pow(g2 + 1e-10, 1.5);
      k = k > kmax ? kmax : k < -kmax ? -kmax : k;
      out[y * w + x] = k * g;
    }
  }
  return out;
}

/** κ at every pixel, clamped to ±kmax. */
export function curvatureMap(phi: ArrayLike<number>, w: number, h: number, kmax = 1): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = curvatureAt(phi, w, h, x, y);
    out[y * w + x] = k > kmax ? kmax : k < -kmax ? -kmax : k;
  }
  return out;
}

/** Upwind (Godunov) |∇φ| at pixel p for a front moving outward (sign > 0) or inward (sign < 0). */
export function upwindGrad(phi: ArrayLike<number>, w: number, h: number, x: number, y: number, sign: number): number {
  const c = phi[y * w + x];
  const dmx = x > 0 ? c - phi[y * w + x - 1] : 0, dpx = x < w - 1 ? phi[y * w + x + 1] - c : 0;
  const dmy = y > 0 ? c - phi[(y - 1) * w + x] : 0, dpy = y < h - 1 ? phi[(y + 1) * w + x] - c : 0;
  if (sign > 0) {
    const a = Math.max(dmx, 0), b = Math.min(dpx, 0), e = Math.max(dmy, 0), f = Math.min(dpy, 0);
    return Math.sqrt(a * a + b * b + e * e + f * f);
  }
  const a = Math.min(dmx, 0), b = Math.max(dpx, 0), e = Math.min(dmy, 0), f = Math.max(dpy, 0);
  return Math.sqrt(a * a + b * b + e * e + f * f);
}

/** Central-difference |∇φ| at a pixel (zero across the border). */
export function centralGrad(phi: ArrayLike<number>, w: number, h: number, x: number, y: number): number {
  const xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1, ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
  const fx = (phi[y * w + xp] - phi[y * w + xm]) / 2, fy = (phi[yp * w + x] - phi[ym * w + x]) / 2;
  return Math.sqrt(fx * fx + fy * fy);
}

// =========================================================================================
// Evolution

export interface SpeedTerms {
  /** Normal speed (outward > 0): a constant or one value per pixel. */
  F?: number | ArrayLike<number>;
  /** Curvature weight ε: adds −ε κ to the speed (central differences). */
  eps?: number;
  /** Edge-stopping function g (one per pixel): multiplies F and the curvature term. */
  g?: ArrayLike<number>;
  /** Advection field v (one vx, vy per pixel): φ_t + v·∇φ = 0, upwind. For geodesic active
   *  contours v = −a ∇g. */
  vx?: ArrayLike<number>;
  vy?: ArrayLike<number>;
}

export type Scheme = "upwind" | "central";

/**
 * Largest stable time step (Δx = 1). Wave part: the front may move at most one pixel per step
 * in x plus y together; a diagonal front moves |F|/√2 in each, so Δt ≤ 1 / (√2 max|F g|)
 * (Δt ≤ 1/|F| in 1D). Advection: Δt ≤ 1 / max(|vx| + |vy|). Curvature: Δt ≤ 1 / (4 ε max g).
 * Returns the smallest (Infinity if all terms are zero).
 */
export function cflLimit(T: SpeedTerms, n?: number): number {
  const F = T.F ?? 0, g = T.g;
  const len = n ?? (typeof F === "number" ? (g ? g.length : 1) : F.length);
  let maxF = 0, maxG = 0, maxV = 0;
  for (let p = 0; p < len; p++) {
    const gp = g ? g[p] : 1, fp = typeof F === "number" ? F : F[p];
    maxF = Math.max(maxF, Math.abs(fp * gp));
    maxG = Math.max(maxG, gp);
    if (T.vx && T.vy) maxV = Math.max(maxV, Math.abs(T.vx[p]) + Math.abs(T.vy[p]));
  }
  let lim = Infinity;
  if (maxF + maxV > 0) lim = 1 / (Math.SQRT2 * maxF + maxV);
  if ((T.eps ?? 0) > 0 && maxG > 0) lim = Math.min(lim, 1 / (4 * (T.eps ?? 0) * maxG));
  return lim;
}

/**
 * One forward-Euler step of φ_t = −g F |∇φ| + ε g κ |∇φ| − v·∇φ, in place. `scheme` chooses
 * upwind (Godunov) or central differences for the wave and advection parts. Returns the
 * largest |Δφ| of the step.
 */
export function lsStep(phi: Float64Array, w: number, h: number, T: SpeedTerms, dt: number, scheme: Scheme = "upwind", work?: Float64Array): number {
  const n = w * h, upd = work ?? new Float64Array(n);
  const F = T.F ?? 0, eps = T.eps ?? 0, g = T.g;
  const kt = eps > 0 ? curvatureTerm(phi, w, h) : null;
  let maxD = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, gp = g ? g[p] : 1, fp = (typeof F === "number" ? F : F[p]) * gp;
    let r = 0;
    if (fp !== 0) r -= fp * (scheme === "upwind" ? upwindGrad(phi, w, h, x, y, fp) : centralGrad(phi, w, h, x, y));
    if (kt) r += eps * gp * kt[p];
    if (T.vx && T.vy) {
      const vx = T.vx[p], vy = T.vy[p], c = phi[p];
      let fx: number, fy: number;
      if (scheme === "upwind") {
        fx = vx > 0 ? (x > 0 ? c - phi[p - 1] : 0) : (x < w - 1 ? phi[p + 1] - c : 0);
        fy = vy > 0 ? (y > 0 ? c - phi[p - w] : 0) : (y < h - 1 ? phi[p + w] - c : 0);
      } else {
        fx = ((x < w - 1 ? phi[p + 1] : c) - (x > 0 ? phi[p - 1] : c)) / 2;
        fy = ((y < h - 1 ? phi[p + w] : c) - (y > 0 ? phi[p - w] : c)) / 2;
      }
      r -= vx * fx + vy * fy;
    }
    upd[p] = dt * r;
  }
  for (let p = 0; p < n; p++) { phi[p] += upd[p]; const a = Math.abs(upd[p]); if (a > maxD) maxD = a; }
  return maxD;
}

// =========================================================================================
// Contours and reinitialisation

/** The zero contour of φ (marching squares, linear interpolation). Lines run with the
 *  outside (φ > 0) on a fixed side; use `insideArea` for the enclosed area. */
export function zeroContour(phi: ArrayLike<number>, w: number, h: number, c = 0): Contour {
  return marchingSquares(phi, w, h, c, { saddle: "decider" });
}

/** Area of the region φ < 0 from the zero contour (sub-pixel, pixel units). */
export function insideArea(phi: ArrayLike<number>, w: number, h: number): number {
  return Math.abs(contourArea(marchingSquares(Float64Array.from(phi, v => -v), w, h, 0, { saddle: "decider" })));
}

/** Length of the zero contour. */
export function frontLength(phi: ArrayLike<number>, w: number, h: number): number {
  return contourLength(zeroContour(phi, w, h));
}

/** Number of pixels with φ < 0. */
export function insideCount(phi: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < phi.length; i++) if (phi[i] < 0) k++;
  return k;
}

/** Segments x0 y0 x1 y1 of a contour (for distance computations or drawing). */
export function contourSegments(C: Contour): Float64Array {
  const s = new Float64Array(2 * C.segments.length);
  for (let k = 0; k < C.segments.length; k += 2) {
    const a = C.segments[k], b = C.segments[k + 1];
    s[2 * k] = C.points[2 * a]; s[2 * k + 1] = C.points[2 * a + 1];
    s[2 * k + 2] = C.points[2 * b]; s[2 * k + 3] = C.points[2 * b + 1];
  }
  return s;
}

/**
 * Reinitialisation by recomputing distances: φ ← sign(φ) × (distance to the marching-squares
 * zero contour). Exact up to the linear interpolation of the crossings; in place.
 */
export function reinitExact(phi: Float64Array, w: number, h: number): void {
  const segs = contourSegments(zeroContour(phi, w, h));
  if (segs.length === 0) return;
  const d = distanceToSegments(segs, w, h);
  for (let p = 0; p < w * h; p++) phi[p] = phi[p] < 0 ? -d[p] : d[p];
}

/**
 * Reinitialisation by the PDE φ_τ = S(φ0)(1 − |∇φ|) (Sussman, Smereka & Osher 1994), with
 * the smoothed sign S = φ0/√(φ0² + 1) and Godunov upwinding; `iters` steps of size dτ. In place.
 */
export function reinitPDE(phi: Float64Array, w: number, h: number, iters = 20, dtau = 0.5): void {
  const phi0 = Float64Array.from(phi), upd = new Float64Array(w * h);
  for (let it = 0; it < iters; it++) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = y * w + x, s = phi0[p] / Math.sqrt(phi0[p] * phi0[p] + 1);
      // φ_τ + S (|∇φ| − 1) = 0: a front moving with speed S
      upd[p] = -dtau * s * (upwindGrad(phi, w, h, x, y, s) - 1);
    }
    for (let p = 0; p < w * h; p++) phi[p] += upd[p];
  }
}

/** Mean |∇φ| (central differences) over the pixels with |φ| < band. */
export function meanGradNearFront(phi: ArrayLike<number>, w: number, h: number, band = 3): number {
  let s = 0, k = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) if (Math.abs(phi[y * w + x]) < band) { s += centralGrad(phi, w, h, x, y); k++; }
  return k ? s / k : NaN;
}

// =========================================================================================
// Smoothed Heaviside and delta (Chan & Vese 2001)

/** H_ε(x) = ½(1 + (2/π) arctan(x/ε)): a soft step from 0 to 1. */
export const heaviside = (x: number, eps: number): number => 0.5 * (1 + (2 / Math.PI) * Math.atan(x / eps));
/** δ_ε(x) = H_ε'(x) = ε / (π (ε² + x²)): a soft spike, area 1. */
export const delta = (x: number, eps: number): number => eps / (Math.PI * (eps * eps + x * x));

// =========================================================================================
// Edge-stopping function (geodesic active contours)

/** g = 1/(1 + |∇I_σ|²/k²) from a gradient magnitude image; with the advection field −∇g. */
export function edgeStop(gradMag: ArrayLike<number>, w: number, h: number, k: number): { g: Float64Array; gx: Float64Array; gy: Float64Array } {
  const g = Float64Array.from(gradMag, m => 1 / (1 + (m * m) / (k * k)));
  const gx = new Float64Array(w * h), gy = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [a, b] = gradAt(g, w, h, x, y);
    gx[y * w + x] = a; gy[y * w + x] = b;
  }
  return { g, gx, gy };
}

// =========================================================================================
// Chan–Vese

export interface CVParams {
  /** Weight of the boundary length. */
  lambda: number;
  /** Weight of the inside area. */
  nu: number;
  /** Width ε of δ_ε (pixels). */
  eps: number;
  /** Data weights inside and outside (literature: λ1, λ2); default 1. */
  w1?: number;
  w2?: number;
}

/** Means inside (φ < 0) and outside (φ ≥ 0). An empty side gets the other side's mean. */
export function cvMeans(I: ArrayLike<number>, phi: ArrayLike<number>): { c1: number; c2: number; n1: number; n2: number } {
  let s1 = 0, s2 = 0, n1 = 0, n2 = 0;
  for (let p = 0; p < I.length; p++) if (phi[p] < 0) { s1 += I[p]; n1++; } else { s2 += I[p]; n2++; }
  const c1 = n1 ? s1 / n1 : n2 ? s2 / n2 : 0, c2 = n2 ? s2 / n2 : c1;
  return { c1, c2, n1, n2 };
}

export interface CVEnergy { dataIn: number; dataOut: number; length: number; area: number; total: number }

/** E = w1 Σ_in (I − c1)² + w2 Σ_out (I − c2)² + λ·Length + ν·Area (sharp regions, contour length). */
export function cvEnergy(I: ArrayLike<number>, phi: ArrayLike<number>, w: number, h: number, P: CVParams, c?: { c1: number; c2: number }): CVEnergy {
  const { c1, c2 } = c ?? cvMeans(I, phi);
  let dIn = 0, dOut = 0;
  for (let p = 0; p < I.length; p++) {
    if (phi[p] < 0) dIn += (I[p] - c1) ** 2; else dOut += (I[p] - c2) ** 2;
  }
  const length = frontLength(phi, w, h), area = insideCount(phi);
  const w1 = P.w1 ?? 1, w2 = P.w2 ?? 1;
  return { dataIn: dIn, dataOut: dOut, length, area, total: w1 * dIn + w2 * dOut + P.lambda * length + P.nu * area };
}

/**
 * One Chan–Vese step: means from the current regions, then
 * φ ← φ + Δt δ_ε(φ)[λκ + ν + w1(I − c1)² − w2(I − c2)²], with Δt chosen so that the largest
 * change is `cfl` pixels and the curvature part stays stable. In place; returns the means and Δt.
 */
export function cvStep(I: ArrayLike<number>, phi: Float64Array, w: number, h: number, P: CVParams, cfl = 0.5, work?: Float64Array): { c1: number; c2: number; dt: number } {
  const n = w * h, { c1, c2 } = cvMeans(I, phi), upd = work ?? new Float64Array(n);
  const w1 = P.w1 ?? 1, w2 = P.w2 ?? 1;
  const kap = P.lambda > 0 ? curvatureMap(phi, w, h) : null;
  let maxS = 0;
  for (let p = 0; p < n; p++) {
    const de = delta(phi[p], P.eps);
    const s = de * ((kap ? P.lambda * kap[p] : 0) + P.nu + w1 * (I[p] - c1) ** 2 - w2 * (I[p] - c2) ** 2);
    upd[p] = s;
    const a = Math.abs(s); if (a > maxS) maxS = a;
  }
  let dt = maxS > 0 ? cfl / maxS : 0;
  const dmax = 1 / (Math.PI * P.eps);
  if (P.lambda > 0) dt = Math.min(dt, 0.25 / (P.lambda * dmax));
  for (let p = 0; p < n; p++) phi[p] += dt * upd[p];
  return { c1, c2, dt };
}

// =========================================================================================
// Explicit polygon front (for the comparison with the implicit one)

/**
 * Move each vertex of a closed polygon along its (vertex) outward normal by dt (F − ε κ),
 * with κ from the turning angle. No resampling, no surgery: this is what goes wrong.
 * Works for either orientation of the polygon.
 */
export function explicitStep(P: Poly, F: number, eps: number, dt: number): Poly {
  const n = P.length;
  if (n < 3) return P;
  // signed area (y down): positive for clockwise-on-screen
  let A = 0;
  for (let i = 0; i < n; i++) { const [x0, y0] = P[i], [x1, y1] = P[(i + 1) % n]; A += x0 * y1 - x1 * y0; }
  const o = A > 0 ? 1 : -1;
  return P.map((p, i) => {
    const a = P[(i - 1 + n) % n], b = P[(i + 1) % n];
    const tx = b[0] - a[0], ty = b[1] - a[1], tl = Math.hypot(tx, ty) || 1;
    // outward normal: rotate the tangent; orientation decides which way is out
    const nx = o * ty / tl, ny = -o * tx / tl;
    // curvature from the circle through a, p, b: κ = 2 sin(turn) / |a − b|
    const ux = p[0] - a[0], uy = p[1] - a[1], vx = b[0] - p[0], vy = b[1] - p[1];
    const cr = ux * vy - uy * vx, la = Math.hypot(ux, uy), lb = Math.hypot(vx, vy);
    const k = la > 0 && lb > 0 ? o * 2 * cr / (la * lb * tl) : 0;
    const s = F - eps * k;
    return [p[0] + dt * s * nx, p[1] + dt * s * ny] as [number, number];
  });
}
