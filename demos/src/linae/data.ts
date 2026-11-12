// Data and computations for the linae demo (no DOM, so exercise numbers come from this code).
// Every point is a two-pixel image (p1, p2), grey levels in [0, 1].
//   "pixels":  pairs of neighbouring pixels from a smooth image: both pixels about equally
//              bright, so the cloud lies along the diagonal.
//   "spot":    a bright Gaussian spot (sd 0.5 pixel) at position t ∈ [-0.5, 1.5] seen by two
//              pixels at positions 0 and 1, plus a little noise: one true number per image,
//              but the images lie on a curve.
import { rng } from "../lib/ui";
import { Adam, mlp, trainStep, forward, type Mlp } from "../lib/nn";

export type V2 = [number, number];
export const N_PTS = 150;

export function makeData(kind: "pixels" | "spot"): V2[] {
  const r = rng(kind === "pixels" ? 21 : 22), out: V2[] = [];
  for (let i = 0; i < N_PTS; i++) {
    if (kind === "pixels") {
      const b = 0.15 + 0.7 * r.unif(), d = 0.06 * r.norm();
      out.push([b + d, b - d]);
    } else {
      const t = -0.5 + 2 * (i + r.unif()) / N_PTS;
      const g = (c: number) => Math.exp(-((t - c) ** 2) / (2 * 0.25));
      out.push([g(0) + 0.015 * r.norm(), g(1) + 0.015 * r.norm()]);
    }
  }
  return out;
}

export function mean(P: V2[]): V2 {
  return [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length];
}

/** Covariance (divided by n) and the PCA angle (direction of the largest eigenvector) in radians. */
export function pca(P: V2[]): { m: V2; a: number; b: number; c: number; theta: number; l1: number; l2: number } {
  const m = mean(P);
  let a = 0, b = 0, c = 0;
  for (const p of P) { const x = p[0] - m[0], y = p[1] - m[1]; a += x * x; b += x * y; c += y * y; }
  a /= P.length; b /= P.length; c /= P.length;
  const theta = 0.5 * Math.atan2(2 * b, a - c);
  const tr = a + c, dt = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  return { m, a, b, c, theta, l1: tr / 2 + dt, l2: tr / 2 - dt };
}

/** Linear 1-number autoencoder along unit direction u = (cos θ, sin θ) through the mean. */
export function linEncode(p: V2, m: V2, th: number): number { return (p[0] - m[0]) * Math.cos(th) + (p[1] - m[1]) * Math.sin(th); }
export function linDecode(s: number, m: V2, th: number): V2 { return [m[0] + s * Math.cos(th), m[1] + s * Math.sin(th)]; }
/** Mean over images of the squared reconstruction error ||x − x̂||² (summed over the 2 pixels). */
export function linError(P: V2[], th: number): number {
  const m = mean(P);
  return P.reduce((s, p) => { const q = linDecode(linEncode(p, m, th), m, th); return s + (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; }, 0) / P.length;
}

/** The nonlinear autoencoder 2 → 16 → 1 → 16 → 2 (ReLU), fixed seed. */
export const AE_SIZES = [2, 16, 1, 16, 2];
export function makeAE(): { net: Mlp; opt: Adam } {
  const net = mlp(AE_SIZES, ["relu", "id", "relu", "id"], 3);
  return { net, opt: new Adam(net, 0.01) };
}
export const aeCode = (net: Mlp, p: V2): number => forward(net, p).a[2][0];
/** Decode a code z with the decoder half of the network (layers 2 and 3). */
export function aeDecode(net: Mlp, z: number): V2 {
  const h = new Float64Array(16);
  for (let o = 0; o < 16; o++) h[o] = Math.max(0, net.b[2][o] + net.W[2][o] * z);
  const y: V2 = [net.b[3][0], net.b[3][1]];
  for (let j = 0; j < 2; j++) for (let o = 0; o < 16; o++) y[j] += net.W[3][j * 16 + o] * h[o];
  return y;
}
export function aeError(net: Mlp, P: V2[]): number {
  return P.reduce((s, p) => { const q = forward(net, p).a[4]; return s + (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; }, 0) / P.length;
}
export function aeTrain(net: Mlp, opt: Adam, P: V2[], steps: number): void {
  for (let k = 0; k < steps; k++) trainStep(net, P, P, opt);
}
