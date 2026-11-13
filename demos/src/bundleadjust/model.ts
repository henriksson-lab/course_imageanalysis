// The bundle adjustment problem of the bundleadjust demo (no DOM): an object outline in
// flatland, cameras around it, noisy observations, a perturbed starting guess, and the
// incremental reconstruction (initial pair → triangulate → resect next camera → triangulate
// → bundle adjust → …).
import { Cam1D, P, ring, project1d, visible, makeScene, Scene, triangulateRays, lookAt1d } from "../lib/flatland";
import { FlatBA, Obs, gaugeMask, levenbergMarquardt } from "../lib/bundle";
import { rng } from "../lib/random";
import { svd } from "../lib/linalg";

/**
 * Resection of a 1D camera by the DLT: u ≃ P (X, Y, 1) with P 2 × 3; each point gives one
 * linear equation u (p2 · X) − p1 · X = 0, so 5 points fix P up to scale. The pose (centre and
 * heading) follows from P with the known f, c. Needs ≥ 5 points.
 */
export function resect1d(X: P[], u: number[], base: Cam1D): Cam1D | null {
  const n = X.length;
  if (n < 5) return null;
  const A = new Float64Array(n * 6);
  X.forEach((p, i) => A.set([-p[0], -p[1], -1, u[i] * p[0], u[i] * p[1], u[i]], 6 * i));
  const V = svd(A, n, 6).V, q = Array.from({ length: 6 }, (_, k) => V[k * 6 + 5]);
  // second row = (Fx, Fy, −F·C) up to scale; fix the scale and the sign (points in front)
  let s = 1 / Math.hypot(q[3], q[4]);
  const z0 = s * (q[3] * X[0][0] + q[4] * X[0][1] + q[5]);
  if (z0 < 0) s = -s;
  const p = q.map(v => v * s);
  // C from p1 · (C, 1) = 0 and p2 · (C, 1) = 0
  const det = p[0] * p[4] - p[1] * p[3];
  if (Math.abs(det) < 1e-12) return null;
  const cx = (-p[2] * p[4] + p[1] * p[5]) / det, cy = (-p[0] * p[5] + p[3] * p[2]) / det;
  return { ...base, x: cx, y: cy, theta: Math.atan2(p[4], p[3]) };
}

export const F = 130, NPIX = 160;

export interface BAWorld {
  scene: Scene;
  cams: Cam1D[];
  pts: P[];
  obs: Obs[];
  /** Indices of observations replaced by outliers. */
  outliers: number[];
}

/** An outline r(φ) = 0.85 + 0.15 cos 3φ + 0.1 sin 2φ with nPts points on it; cameras on an arc. */
export function makeWorld(o: { nCams?: number; nPts?: number; arc?: number; radius?: number; sigma?: number; outliers?: number; seed?: number } = {}): BAWorld {
  const nC = o.nCams ?? 10, nP = o.nPts ?? 40, arc = o.arc ?? 2 * Math.PI, R = o.radius ?? 3, sigma = o.sigma ?? 0.2;
  const outline: P[] = [];
  for (let k = 0; k < 90; k++) {
    const a = 2 * Math.PI * k / 90, r = 0.85 + 0.15 * Math.cos(3 * a) + 0.1 * Math.sin(2 * a);
    outline.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  const scene = makeScene([outline]);
  const pts: P[] = Array.from({ length: nP }, (_, i) => outline[Math.round(i * 90 / nP) % 90]);
  const full = Math.abs(arc - 2 * Math.PI) < 1e-9;
  const cams = full ? ring(nC, R, F, NPIX, [0, 0], -Math.PI / 2 + 0.2) : ring(nC, R, F, NPIX, [0, 0], -Math.PI / 2 - arc / 2, arc);
  const r = rng(o.seed ?? 4);
  const obs: Obs[] = [];
  cams.forEach((c, j) => pts.forEach((p, i) => {
    if (!visible(scene, c, p, 1e-3)) return;
    obs.push({ cam: j, pt: i, u: project1d(c, p).u + sigma * r.norm() });
  }));
  // keep only points seen by at least 3 cameras (a 1D observation gives one equation, a point
  // has two unknowns: 3 views leave one to spare), and renumber them
  const count = pts.map((_, i) => obs.filter(o => o.pt === i).length);
  const keep = pts.map((_, i) => i).filter(i => count[i] >= 3), newIdx = new Map(keep.map((i, k) => [i, k]));
  const pts2 = keep.map(i => pts[i]);
  const obs2 = obs.filter(o => newIdx.has(o.pt)).map(o => ({ ...o, pt: newIdx.get(o.pt)! }));
  obs.length = 0; obs.push(...obs2);
  const outliers: number[] = [];
  const ro = rng(99);
  for (let k = 0; k < (o.outliers ?? 0); k++) {
    const i = Math.floor(ro.unif() * obs.length);
    obs[i] = { ...obs[i], u: obs[i].u + (ro.unif() < 0.5 ? -1 : 1) * (18 + 10 * ro.unif()) };
    outliers.push(i);
  }
  return { scene, cams, pts: pts2, obs, outliers };
}

/** The demo's gauge: camera 1 fixed, and the coordinate of the opposite camera along which the
 *  baseline is longest (x or y) fixed. Off: nothing fixed. */
export function gaugeFor(w: BAWorld, fix = true): boolean[] {
  const b = Math.floor(w.cams.length / 2), a = w.cams[0], c = w.cams[b];
  return gaugeMask(w.cams.length, w.pts.length, 0, b, fix, Math.abs(c.y - a.y) > Math.abs(c.x - a.x) ? 1 : 0);
}

/** Truth + a seeded perturbation of size s (positions in world units, angles s/3 rad); the gauge parameters stay at the truth. */
export function perturbed(w: BAWorld, s: number, fixed: boolean[], seed = 5): Float64Array {
  const th = FlatBA.pack(w.cams, w.pts), r = rng(seed), nc = 3 * w.cams.length;
  for (let k = 0; k < th.length; k++) {
    if (fixed[k]) continue;
    const isAngle = k < nc && k % 3 === 2;
    th[k] += (isAngle ? s / 3 : s) * r.norm();
  }
  return th;
}

/** 2D similarity (scale, rotation, translation) that best maps `a` onto `b` (least squares), applied to `a`. */
export function alignSimilarity(a: P[], b: P[]): { mapped: P[]; scale: number } {
  const n = a.length;
  let ax = 0, ay = 0, bx = 0, by = 0;
  for (let i = 0; i < n; i++) { ax += a[i][0] / n; ay += a[i][1] / n; bx += b[i][0] / n; by += b[i][1] / n; }
  let re = 0, im = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const zx = a[i][0] - ax, zy = a[i][1] - ay, wx = b[i][0] - bx, wy = b[i][1] - by;
    re += wx * zx + wy * zy; im += wy * zx - wx * zy; den += zx * zx + zy * zy;
  }
  const cr = re / den, ci = im / den;
  return { mapped: a.map(p => [bx + cr * (p[0] - ax) - ci * (p[1] - ay), by + ci * (p[0] - ax) + cr * (p[1] - ay)]), scale: Math.hypot(cr, ci) };
}

// ---------------------------------------------------------------------------------------
// Incremental reconstruction

export interface Incremental {
  stage: number;
  camOn: boolean[];
  ptOn: boolean[];
  theta: Float64Array;
  log: string;
}

/** Sub-problem over the active cameras and points; returns it and the index maps. */
export function subProblem(w: BAWorld, camOn: boolean[], ptOn: boolean[], freeCams: boolean) {
  const cIdx: number[] = [], pIdx: number[] = [];
  camOn.forEach((on, j) => { if (on) cIdx.push(j); });
  ptOn.forEach((on, i) => { if (on) pIdx.push(i); });
  const cMap = new Map(cIdx.map((j, k) => [j, k])), pMap = new Map(pIdx.map((i, k) => [i, k]));
  const obs = w.obs.filter(o => cMap.has(o.cam) && pMap.has(o.pt)).map(o => ({ cam: cMap.get(o.cam)!, pt: pMap.get(o.pt)!, u: o.u }));
  const prob = new FlatBA(cIdx.map(j => w.cams[j]), pIdx.length, obs);
  void freeCams;
  return { prob, cIdx, pIdx };
}

function extract(theta: Float64Array, nCams: number, cIdx: number[], pIdx: number[]): Float64Array {
  const t = new Float64Array(3 * cIdx.length + 2 * pIdx.length);
  cIdx.forEach((j, k) => t.set(theta.subarray(3 * j, 3 * j + 3), 3 * k));
  pIdx.forEach((i, k) => t.set(theta.subarray(3 * nCams + 2 * i, 3 * nCams + 2 * i + 2), 3 * cIdx.length + 2 * k));
  return t;
}
function writeBack(theta: Float64Array, sub: Float64Array, nCams: number, cIdx: number[], pIdx: number[]): void {
  cIdx.forEach((j, k) => theta.set(sub.subarray(3 * k, 3 * k + 3), 3 * j));
  pIdx.forEach((i, k) => theta.set(sub.subarray(3 * cIdx.length + 2 * k, 3 * cIdx.length + 2 * k + 2), 3 * nCams + 2 * i));
}

/** Triangulate every inactive point seen by ≥ 2 active cameras (least squares on the rays). Returns how many. */
function triangulateNew(w: BAWorld, inc: Incremental): number {
  const nC = w.cams.length;
  const camsNow = w.cams.map((c, j) => ({ ...c, x: inc.theta[3 * j], y: inc.theta[3 * j + 1], theta: inc.theta[3 * j + 2] }));
  let added = 0;
  w.pts.forEach((_, i) => {
    if (inc.ptOn[i]) return;
    const os = w.obs.filter(o => o.pt === i && inc.camOn[o.cam]);
    if (os.length < 2) return;
    const p = triangulateRays(os.map(o => camsNow[o.cam]), os.map(o => o.u));
    if (!Number.isFinite(p[0])) return;
    inc.theta[3 * nC + 2 * i] = p[0]; inc.theta[3 * nC + 2 * i + 1] = p[1];
    inc.ptOn[i] = true; added++;
  });
  return added;
}

export function startIncremental(w: BAWorld): Incremental {
  const nC = w.cams.length;
  const theta = FlatBA.pack(w.cams.map(c => lookAt1d([0, 0], [1, 0], F, NPIX)).map(c => ({ ...c, x: 0, y: 0, theta: 0 })), w.pts.map(() => [0, 0] as P));
  // the initial pair: cameras 1 and 2 from two-view geometry (here: their true poses; the gauge)
  for (const j of [0, 1]) theta.set([w.cams[j].x, w.cams[j].y, w.cams[j].theta], 3 * j);
  const inc: Incremental = { stage: 0, camOn: w.cams.map((_, j) => j < 2), ptOn: w.pts.map(() => false), theta, log: "" };
  const n = triangulateNew(w, inc);
  inc.log = `initial pair: cameras 1 and 2, ${n} points triangulated`;
  void nC;
  return inc;
}

/** Next step: resect the next camera from the reconstructed points, triangulate new points, bundle adjust. */
export function stepIncremental(w: BAWorld, inc: Incremental): boolean {
  const nC = w.cams.length, j = inc.camOn.indexOf(false);
  if (j < 0) return false;
  // resection: camera j's pose from the points it sees that are already reconstructed
  const seen = w.obs.filter(o => o.cam === j && inc.ptOn[o.pt]);
  if (seen.length < 3) return false;
  // linear start (DLT for a 1D camera), refined by LM; if that does not fit (noise makes the
  // DLT unstable for points on one arc), also try starts around the points and keep the best
  const Xs = seen.map(o => [inc.theta[3 * nC + 2 * o.pt], inc.theta[3 * nC + 2 * o.pt + 1]] as P), us = seen.map(o => o.u);
  const resect = new FlatBA([w.cams[j]], seen.length, seen.map((o, k) => ({ cam: 0, pt: k, u: o.u })));
  const fixedPts = Array.from({ length: 3 + 2 * seen.length }, (_, k) => k >= 3);
  const mx = Xs.reduce((a, p) => a + p[0], 0) / Xs.length, my = Xs.reduce((a, p) => a + p[1], 0) / Xs.length;
  const starts: Cam1D[] = [];
  const dlt = resect1d(Xs, us, w.cams[j]);
  if (dlt) starts.push(dlt);
  for (let k = 0; k < 16; k++) { const a = 2 * Math.PI * k / 16; starts.push(lookAt1d([mx + 3 * Math.cos(a), my + 3 * Math.sin(a)], [mx, my], F, NPIX)); }
  let res: { theta: Float64Array; costs: number[]; mu: number } = { theta: new Float64Array(0), costs: [Infinity], mu: 0 };
  for (const g of starts) {
    const r = levenbergMarquardt(resect, FlatBA.pack([g], Xs), { fixed: fixedPts, maxIter: 60 });
    const ok = resect.cams(r.theta)[0], front = Xs.every(p => project1d(ok, p).Z > 0);
    if (front && r.costs[r.costs.length - 1] < res.costs[res.costs.length - 1]) res = r;
    if (res.costs[res.costs.length - 1] < 4 * seen.length) break;
  }
  if (!res.theta.length) return false;
  inc.theta.set(res.theta.subarray(0, 3), 3 * j);
  inc.camOn[j] = true;
  const added = triangulateNew(w, inc);
  // bundle adjust the active set (gauge: camera 1 fixed, x of camera 2 fixed)
  const { prob, cIdx, pIdx } = subProblem(w, inc.camOn, inc.ptOn, true);
  const sub = extract(inc.theta, nC, cIdx, pIdx);
  const fixed = gaugeMask(cIdx.length, pIdx.length);
  const ba = levenbergMarquardt(prob, sub, { fixed, maxIter: 20 });
  writeBack(inc.theta, ba.theta, nC, cIdx, pIdx);
  inc.stage++;
  const rms = Math.sqrt(ba.costs[ba.costs.length - 1] / Math.max(1, prob.obs.length));
  inc.log = `camera ${j + 1}: resected from ${seen.length} points, ${added} new points, bundle adjusted (RMS ${rms.toFixed(2)} px)`;
  return true;
}
