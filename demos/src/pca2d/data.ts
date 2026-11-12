// Data and computations for the pca2d demo (no DOM, so exercise numbers come from this code).
// Every point is a two-pixel image (p1, p2), grey levels in [0, 1]. The covariance is divided
// by n, as in the autoencoder unit, and "neighbouring pixels" is the same data set as the
// linae demo there, so the two units quote the same numbers.
import { rng } from "../lib/ui";
import { makeData as linaeData } from "../linae/data";

export type V2 = [number, number];
export type Kind = "pixels" | "four" | "round" | "unrelated" | "outlier";
export const KINDS: Kind[] = ["pixels", "four", "round", "unrelated", "outlier"];

export function makeData(kind: Kind): V2[] {
  if (kind === "pixels") return linaeData("pixels");
  // The four images of the exercise (by hand): mean (0.5, 0.5), S = [0.05 0.03; 0.03 0.05].
  if (kind === "four") return [[0.2, 0.4], [0.4, 0.2], [0.6, 0.8], [0.8, 0.6]];
  const out: V2[] = [];
  if (kind === "round") {             // two independent pixels with the same spread
    const r = rng(31);
    for (let i = 0; i < 80; i++) out.push([0.5 + 0.12 * r.norm(), 0.5 + 0.12 * r.norm()]);
  } else if (kind === "unrelated") {  // independent pixels, pixel 1 varies much more
    const r = rng(32);
    for (let i = 0; i < 80; i++) out.push([0.5 + 0.18 * r.norm(), 0.5 + 0.05 * r.norm()]);
  } else {                            // neighbouring pixels plus one hot pixel
    const r = rng(33);
    for (let i = 0; i < 40; i++) { const b = 0.3 + 0.4 * r.unif(), d = 0.03 * r.norm(); out.push([b + d, b - d]); }
    out.push([0.95, 0.05]);
  }
  return out;
}

/** Mean, covariance S = [[a, b], [b, c]] (divided by n), eigenvalues l1 ≥ l2 and the PC1 angle θ1. */
export interface Pca { m: V2; a: number; b: number; c: number; l1: number; l2: number; theta: number }
export function pca(P: V2[]): Pca {
  const n = P.length;
  const m: V2 = [P.reduce((s, p) => s + p[0], 0) / n, P.reduce((s, p) => s + p[1], 0) / n];
  let a = 0, b = 0, c = 0;
  for (const p of P) { const x = p[0] - m[0], y = p[1] - m[1]; a += x * x; b += x * y; c += y * y; }
  a /= n; b /= n; c /= n;
  const tr = a + c, dt = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  let theta = 0.5 * Math.atan2(2 * b, a - c);       // in (−90°, 90°]
  if (dt < 1e-15) theta = 0;                       // round cloud: every direction is an eigenvector
  return { m, a, b, c, l1: tr / 2 + dt, l2: tr / 2 - dt, theta };
}

export const unit = (th: number): V2 => [Math.cos(th), Math.sin(th)];

/** Origin of the line: the mean (centred) or (0, 0) (not centred). */
export const origin = (pc: Pca, centred: boolean): V2 => centred ? pc.m : [0, 0];

/** Score of p along the unit direction at angle th, through `o`. */
export const score = (p: V2, o: V2, th: number): number => (p[0] - o[0]) * Math.cos(th) + (p[1] - o[1]) * Math.sin(th);
/** Reconstruction from one score: back along the line. */
export const decode = (s: number, o: V2, th: number): V2 => [o[0] + s * Math.cos(th), o[1] + s * Math.sin(th)];

/** Mean squared score (the variance along u when centred) and mean squared reconstruction error. */
export function along(P: V2[], o: V2, th: number): { meanSq: number; err: number; total: number } {
  let ms = 0, tot = 0;
  for (const p of P) { const s = score(p, o, th); ms += s * s; tot += (p[0] - o[0]) ** 2 + (p[1] - o[1]) ** 2; }
  ms /= P.length; tot /= P.length;
  return { meanSq: ms, err: tot - ms, total: tot };
}

/** uᵀSu: the variance of the scores along the angle th (closed form, same as along(…).meanSq centred). */
export const varAlong = (pc: Pca, th: number): number => {
  const u = unit(th);
  return pc.a * u[0] * u[0] + 2 * pc.b * u[0] * u[1] + pc.c * u[1] * u[1];
};

/** Wrap an angle to (−90°, 90°]: a line has no arrow. */
export function wrapAngle(a: number): number {
  while (a > Math.PI / 2) a -= Math.PI;
  while (a <= -Math.PI / 2) a += Math.PI;
  return a;
}
