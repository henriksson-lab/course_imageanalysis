// Flatland: a 2D world seen by 1D pinhole cameras (no DOM). Owner: the geometry-from-images
// day (two-view/, sfm/; later radiance-fields/). Users: triangulate, bundleadjust, flatland
// demos; lib/bundle.ts. Everything a 3D reconstruction does happens here with one dimension
// less, so every camera, ray, point and residual fits in one picture.
//
// World: x right, y UP. A camera sits at (x, y) and looks along its heading θ (radians,
// counter-clockwise from +x). Camera coordinates: Z forward = (cos θ, sin θ), X to the right
// = (sin θ, −cos θ). Pixel u = f X / Z + c, pixel centres at integers 0 … n − 1.
//
// Contents: cameras, projection, rays · scenes (polygons with a texture along every edge) and
// 1D rendering by ray casting · triangulation (linear least squares on the rays; optimal by
// Gauss–Newton on the reprojection error) · plane (line) sweep photo-consistency.

import { rng } from "./random";

export type P = [number, number];

export interface Cam1D { x: number; y: number; theta: number; f: number; c: number; n: number }

export const fwd = (cam: Cam1D): P => [Math.cos(cam.theta), Math.sin(cam.theta)];
export const right = (cam: Cam1D): P => [Math.sin(cam.theta), -Math.cos(cam.theta)];

/** Camera coordinates (X right, Z forward) of a world point. */
export function toCam(cam: Cam1D, p: P): P {
  const dx = p[0] - cam.x, dy = p[1] - cam.y, s = Math.sin(cam.theta), c = Math.cos(cam.theta);
  return [dx * s - dy * c, dx * c + dy * s];
}

/** Pixel coordinate and depth Z of a world point (Z ≤ 0: behind the camera). */
export function project1d(cam: Cam1D, p: P): { u: number; Z: number } {
  const [X, Z] = toCam(cam, p);
  return { u: cam.f * X / Z + cam.c, Z };
}

/** Unit direction of the ray through pixel coordinate u. */
export function rayDir(cam: Cam1D, u: number): P {
  const x = (u - cam.c) / cam.f, F = fwd(cam), R = right(cam);
  const d: P = [F[0] + x * R[0], F[1] + x * R[1]], l = Math.hypot(d[0], d[1]);
  return [d[0] / l, d[1] / l];
}

/** World point on the ray of pixel u at depth Z (along the viewing axis). */
export function pointAtDepth(cam: Cam1D, u: number, Z: number): P {
  const x = (u - cam.c) / cam.f, F = fwd(cam), R = right(cam);
  return [cam.x + Z * (F[0] + x * R[0]), cam.y + Z * (F[1] + x * R[1])];
}

/** A camera at `pos` looking at `target`, with n pixels and focal length f (pixels); c = (n − 1)/2. */
export function lookAt1d(pos: P, target: P, f: number, n: number): Cam1D {
  return { x: pos[0], y: pos[1], theta: Math.atan2(target[1] - pos[1], target[0] - pos[0]), f, c: (n - 1) / 2, n };
}

/** Cameras evenly on a circle of radius R around `centre`, all looking at it. */
export function ring(k: number, R: number, f: number, n: number, centre: P = [0, 0], start = -Math.PI / 2, span = 2 * Math.PI): Cam1D[] {
  const full = Math.abs(span - 2 * Math.PI) < 1e-9;
  return Array.from({ length: k }, (_, i) => {
    const a = start + span * (full ? i / k : k > 1 ? i / (k - 1) : 0);
    return lookAt1d([centre[0] + R * Math.cos(a), centre[1] + R * Math.sin(a)], centre, f, n);
  });
}

// ---------------------------------------------------------------------------------------
// Scenes: closed polygons whose edges carry a texture (a function of arc length)

export interface Edge { a: P; b: P; tex: (s: number) => number; s0: number }
export interface Scene { edges: Edge[]; polys: P[][] }

/** 1D value noise in [0, 1] with feature size `cell` (world units). */
export function noise1d(seed: number, cell: number): (s: number) => number {
  const N = 257, r = rng(seed), g = Float32Array.from({ length: N }, () => r.unif());
  return s => {
    const f = s / cell, i = Math.floor(f), t = f - i, a = g[((i % N) + N) % N], b = g[(((i + 1) % N) + N) % N];
    const u = t * t * (3 - 2 * t);
    return a + (b - a) * u;
  };
}

/** Build a scene from polygons; `texFor(k, edgeIndex)` gives each edge's texture (default: textured). */
export function makeScene(polys: P[][], texFor?: (poly: number, edge: number) => ((s: number) => number) | null, seed = 3): Scene {
  const edges: Edge[] = [];
  polys.forEach((poly, k) => {
    let s0 = 0;
    for (let e = 0; e < poly.length; e++) {
      const a = poly[e], b = poly[(e + 1) % poly.length];
      const nz = noise1d(seed + 31 * k + e, 0.06);
      const tex = texFor?.(k, e) ?? ((s: number) => 0.15 + 0.8 * nz(s));
      edges.push({ a, b, tex, s0 });
      s0 += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
  });
  return { edges, polys };
}

/** First hit of a ray (origin o, unit direction d) with the scene: distance t, edge, arc length. */
export function castRay(scene: Scene, o: P, d: P, tMin = 1e-6): { t: number; edge: number; s: number } | null {
  let best: { t: number; edge: number; s: number } | null = null;
  scene.edges.forEach((e, k) => {
    const ex = e.b[0] - e.a[0], ey = e.b[1] - e.a[1];
    const den = d[0] * ey - d[1] * ex;
    if (Math.abs(den) < 1e-12) return;
    const wx = e.a[0] - o[0], wy = e.a[1] - o[1];
    const t = (wx * ey - wy * ex) / den, v = (wx * d[1] - wy * d[0]) / den;
    if (t > tMin && v >= 0 && v <= 1 && (!best || t < best.t)) best = { t, edge: k, s: v * Math.hypot(ex, ey) };
  });
  return best;
}

export interface Render1D { values: Float32Array; depth: Float32Array; edge: Int16Array }

/** The 1D photograph of a camera: per pixel the mean texture over `sub` rays (background 0.5 / depth ∞). */
export function render1d(scene: Scene, cam: Cam1D, sub = 3): Render1D {
  const values = new Float32Array(cam.n), depth = new Float32Array(cam.n), edge = new Int16Array(cam.n);
  for (let u = 0; u < cam.n; u++) {
    let s = 0;
    for (let k = 0; k < sub; k++) {
      const uu = u + (k + 0.5) / sub - 0.5, d = rayDir(cam, uu), h = castRay(scene, [cam.x, cam.y], d);
      s += h ? scene.edges[h.edge].tex(scene.edges[h.edge].s0 + h.s) : 0.5;
    }
    values[u] = s / sub;
    const h = castRay(scene, [cam.x, cam.y], rayDir(cam, u));
    depth[u] = h ? h.t * (rayDir(cam, u)[0] * fwd(cam)[0] + rayDir(cam, u)[1] * fwd(cam)[1]) : Infinity;
    edge[u] = h ? h.edge : -1;
  }
  return { values, depth, edge };
}

/** Is p visible from cam (in front, inside the image, nothing in between)? */
export function visible(scene: Scene, cam: Cam1D, p: P, tol = 1e-6): boolean {
  const { u, Z } = project1d(cam, p);
  if (!(Z > 0) || u < -0.5 || u > cam.n - 0.5) return false;
  const dx = p[0] - cam.x, dy = p[1] - cam.y, dist = Math.hypot(dx, dy);
  const h = castRay(scene, [cam.x, cam.y], [dx / dist, dy / dist]);
  return !h || h.t >= dist - 1e-4 - tol;
}

/** Linear interpolation of a 1D image at a real pixel coordinate (NaN outside). */
export function sample1d(v: ArrayLike<number>, u: number): number {
  if (u < 0 || u > v.length - 1) return NaN;
  const i = Math.min(v.length - 2, Math.floor(u)), t = u - i;
  return v[i] * (1 - t) + v[i + 1] * t;
}

// ---------------------------------------------------------------------------------------
// Triangulation

/** Least-squares intersection of the rays of pixels us[k] in cams[k] (sum of squared perpendicular distances). For two rays this is the midpoint. */
export function triangulateRays(cams: Cam1D[], us: number[]): P {
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  cams.forEach((cam, k) => {
    const d = rayDir(cam, us[k]), nx = -d[1], ny = d[0];       // normal of the ray
    const off = nx * cam.x + ny * cam.y;
    a11 += nx * nx; a12 += nx * ny; a22 += ny * ny; b1 += nx * off; b2 += ny * off;
  });
  const det = a11 * a22 - a12 * a12;
  if (Math.abs(det) < 1e-15) return [NaN, NaN];
  return [(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det];
}

/** Derivative of the pixel coordinate u with respect to the point p (2-vector). */
export function duDp(cam: Cam1D, p: P): P {
  const [X, Z] = toCam(cam, p), R = right(cam), F = fwd(cam);
  return [cam.f * (R[0] * Z - X * F[0]) / (Z * Z), cam.f * (R[1] * Z - X * F[1]) / (Z * Z)];
}

/** Optimal triangulation: minimise Σ (u_k − π_k(p))² by Gauss–Newton from the linear solution. */
export function triangulateOptimal(cams: Cam1D[], us: number[], iters = 10): P {
  let p = triangulateRays(cams, us);
  if (!Number.isFinite(p[0])) return p;
  for (let it = 0; it < iters; it++) {
    let a11 = 0, a12 = 0, a22 = 0, g1 = 0, g2 = 0;
    cams.forEach((cam, k) => {
      const r = us[k] - project1d(cam, p).u, J = duDp(cam, p);
      a11 += J[0] * J[0]; a12 += J[0] * J[1]; a22 += J[1] * J[1]; g1 += J[0] * r; g2 += J[1] * r;
    });
    const det = a11 * a22 - a12 * a12;
    if (Math.abs(det) < 1e-18) break;
    const dx = (a22 * g1 - a12 * g2) / det, dy = (a11 * g2 - a12 * g1) / det;
    p = [p[0] + dx, p[1] + dy];
    if (Math.hypot(dx, dy) < 1e-12) break;
  }
  return p;
}

/** The four corners of the uncertainty region of two cameras: rays at u1 ± δ meet rays at u2 ± δ. */
export function uncertaintyQuad(c1: Cam1D, u1: number, c2: Cam1D, u2: number, delta: number): P[] {
  const out: P[] = [];
  for (const [s1, s2] of [[-1, -1], [-1, 1], [1, 1], [1, -1]] as [number, number][]) out.push(triangulateRays([c1, c2], [u1 + s1 * delta, u2 + s2 * delta]));
  return out;
}

// ---------------------------------------------------------------------------------------
// Plane (line) sweep

export interface SweepResult {
  depths: number[];
  /** Photo-consistency cost per depth hypothesis (variance across views; NaN if < 2 views). */
  cost: number[];
  /** Number of views that see each hypothesis. */
  views: number[];
  best: number;
}

/**
 * Sweep depth hypotheses along the ray of pixel u of the reference camera: for each depth,
 * project the point (and its ±win neighbours on the fronto-parallel line) into every other
 * camera, sample their photographs, and score the variance across views (mean over the window).
 * Hypotheses seen by fewer than `minViews` cameras (default: all) get NaN.
 */
export function sweepPixel(cams: Cam1D[], images: ArrayLike<number>[], ref: number, u: number, depths: number[], win = 2, minViews = cams.length): SweepResult {
  const cost: number[] = [], views: number[] = [];
  for (const Z of depths) {
    let sum = 0, cnt = 0, vmin = Infinity;
    for (let k = -win; k <= win; k++) {
      const uu = u + k, v0 = sample1d(images[ref], uu);
      if (!Number.isFinite(v0)) continue;
      const p = pointAtDepth(cams[ref], uu, Z), vals = [v0];
      cams.forEach((cam, j) => {
        if (j === ref) return;
        const q = project1d(cam, p);
        if (!(q.Z > 0)) return;
        const v = sample1d(images[j], q.u);
        if (Number.isFinite(v)) vals.push(v);
      });
      if (vals.length < Math.max(2, minViews)) continue;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      sum += vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length;
      cnt++;
      vmin = Math.min(vmin, vals.length);
    }
    cost.push(cnt ? sum / cnt : NaN);
    views.push(cnt ? vmin : 0);
  }
  let best = -1;
  cost.forEach((c, k) => { if (Number.isFinite(c) && (best < 0 || c < cost[best])) best = k; });
  return { depths, cost, views, best };
}
