// Calibration problems of the dltcalib demo (no DOM): a hidden camera photographs a known
// target (two orthogonal checkerboard walls, or one flat checkerboard in several poses) with
// seeded pixel noise and optional radial distortion. Estimation by the DLT (normalised or
// not), Zhang's method from homographies, and refinement of the reprojection error by
// Levenberg–Marquardt (lib/bundle.ts) with a numerical Jacobian.
import { Camera, makeK, lookAtCamera, project, twoPlaneTarget, planarTarget, dltCamera, decomposeP, homographyDLT, zhangK, poseFromHomography, axisAngleToR, RToAxisAngle, V3, V2 } from "../lib/camera";
import { LSProblem, levenbergMarquardt, lmStep } from "../lib/bundle";
import { rng } from "../lib/random";

export const IMG_W = 640, IMG_H = 480;
export const K_TRUE = makeK(820, 335, 245);

export interface Target3D { X: V3[]; cam: Camera; obs: V2[] }

export function makeTarget3D(sigma: number, k1: number, seed = 3): Target3D {
  const X = twoPlaneTarget(6, 0.04);
  const cam = lookAtCamera(K_TRUE, [0.4, 0.34, 0.47], [0.085, 0.14, 0.085], 0, k1, 0);
  const r = rng(seed);
  const obs = X.map(p => { const q = project(cam, p); return [q.u + sigma * r.norm(), q.v + sigma * r.norm()] as V2; });
  return { X, cam, obs };
}

export interface PlanarViews { X: V3[]; cams: Camera[]; obs: V2[][] }

/** Views of a flat 8 × 6 board. `tilt` scales how much the views are tilted (0 = all fronto-parallel). */
export function makePlanar(nViews: number, sigma: number, k1: number, tilt = 1, seed = 5): PlanarViews {
  const X = planarTarget(8, 6, 0.03);
  const poses: [V3, V3][] = [
    [[0.35, -0.25, 0.1], [0.01, -0.01, 0.55]],
    [[-0.3, 0.3, -0.12], [-0.02, 0.01, 0.6]],
    [[0.15, 0.4, 0.2], [0.02, 0.02, 0.5]],
    [[-0.4, -0.2, -0.05], [0, 0.015, 0.62]],
    [[0.25, 0.1, -0.3], [-0.015, -0.02, 0.58]],
  ];
  const r = rng(seed);
  const cams: Camera[] = [], obs: V2[][] = [];
  for (let k = 0; k < nViews; k++) {
    const [w, t] = poses[k % poses.length];
    // the board faces the camera: rotate by π about x so that its normal points to the camera
    const R = axisAngleToR(w.map(v => 1.5 * v * tilt));
    const flip = Float64Array.from([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    const Rm = mul(R, flip);
    const cam: Camera = { K: K_TRUE, R: Rm, t: [...t], k1, k2: 0 };
    cams.push(cam);
    obs.push(X.map(p => { const q = project(cam, p); return [q.u + sigma * r.norm(), q.v + sigma * r.norm()] as V2; }));
  }
  return { X, cams, obs };
}
function mul(A: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j]; C[i * 3 + j] = s; }
  return C;
}

// ---------------------------------------------------------------------------------------
// Parameter vectors: [fx, fy, cx, cy, s, k1, k2, then per view: w (3), t (3)]

export const N_INTR = 7;
export function pack(K: ArrayLike<number>, k1: number, k2: number, poses: { R: ArrayLike<number>; t: V3 }[]): Float64Array {
  const th = new Float64Array(N_INTR + 6 * poses.length);
  th.set([K[0], K[4], K[2], K[5], 0, k1, k2]);   // skew: 0 (held fixed in the refinement)
  poses.forEach((p, v) => { th.set(RToAxisAngle(p.R), N_INTR + 6 * v); th.set(p.t, N_INTR + 6 * v + 3); });
  return th;
}
export function camOf(th: ArrayLike<number>, v: number): Camera {
  const o = N_INTR + 6 * v;
  return {
    K: Float64Array.from([th[0], th[4], th[2], 0, th[1], th[3], 0, 0, 1]),
    R: axisAngleToR([th[o], th[o + 1], th[o + 2]]), t: [th[o + 3], th[o + 4], th[o + 5]], k1: th[5], k2: th[6],
  };
}

/** Reprojection residuals over all views, numerical Jacobian (central differences). */
export class CalibProblem implements LSProblem {
  readonly nParams: number;
  constructor(readonly X: V3[], readonly obs: V2[][], readonly use: boolean[][]) { this.nParams = N_INTR + 6 * obs.length; }
  residuals(th: ArrayLike<number>): Float64Array {
    const out: number[] = [];
    this.obs.forEach((o, v) => {
      const cam = camOf(th, v);
      o.forEach((x, i) => { if (!this.use[v][i]) return; const q = project(cam, this.X[i]); out.push(x[0] - q.u, x[1] - q.v); });
    });
    return Float64Array.from(out);
  }
  evaluate(th: Float64Array, wantJ: boolean): { r: Float64Array; J?: Float64Array } {
    const r = this.residuals(th);
    if (!wantJ) return { r };
    const m = r.length, n = this.nParams, J = new Float64Array(m * n);
    for (let k = 0; k < n; k++) {
      const h = 1e-6 * Math.max(1, Math.abs(th[k]));
      const tp = Float64Array.from(th), tm = Float64Array.from(th);
      tp[k] += h; tm[k] -= h;
      const rp = this.residuals(tp), rm = this.residuals(tm);
      for (let i = 0; i < m; i++) J[i * n + k] = (rp[i] - rm[i]) / (2 * h);
    }
    return { r, J };
  }
}

/** Parameters held fixed: skew always; k1, k2 unless the distortion is modelled. */
export function fixedMask(nViews: number, modelDistortion: boolean): boolean[] {
  const m = new Array(N_INTR + 6 * nViews).fill(false);
  m[4] = true;
  if (!modelDistortion) { m[5] = true; m[6] = true; }
  return m;
}

export function rmsOf(r: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < r.length; i++) s += r[i] * r[i];
  return Math.sqrt(s / Math.max(1, r.length / 2));
}

/** DLT on the selected points, decomposed into K, R, t. */
export function dltEstimate(t: Target3D, use: boolean[], normalise: boolean) {
  const idx = use.map((u, i) => (u ? i : -1)).filter(i => i >= 0);
  if (idx.length < 6) return null;
  const d = dltCamera(idx.map(i => t.X[i]), idx.map(i => t.obs[i]), normalise);
  const dec = decomposeP(d.P);
  return { ...d, ...dec };
}

/** Zhang: homographies of all views, then K; poses from the homographies. */
export function zhangEstimate(pv: PlanarViews) {
  const Hs = pv.obs.map(o => homographyDLT(pv.X.map(p => [p[0], p[1]]), o, true).H);
  const z = zhangK(Hs);
  const poses = z.K ? Hs.map(H => poseFromHomography(z.K!, H)) : null;
  return { Hs, K: z.K, S: z.S, rows: z.rows, poses };
}

export { levenbergMarquardt, lmStep };
