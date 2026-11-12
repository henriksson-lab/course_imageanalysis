// Data and start values for the nnfit demo (no DOM, so the exercise numbers can be
// computed in node with exactly this code).
import { rng } from "../lib/ui";
import type { Mlp } from "../lib/nn";

export const N_PIX = 100;
const sig = (t: number) => 1 / (1 + Math.exp(-t));

/** Noise-free profile: background 0.15, a bright nucleus on [0.12, 0.42], a dimmer one on [0.6, 0.88]. */
export function trueProfile(x: number): number {
  const box = (a: number, b: number) => sig((x - a) / 0.012) * sig((b - x) / 0.012);
  return 0.15 + 0.7 * box(0.12, 0.42) + 0.35 * box(0.6, 0.88);
}

/** The line scan: N_PIX pixels at x = (i + 0.5) / N_PIX, Gaussian noise sd 0.03, seed 5. */
export function profileData(): { X: number[][]; Y: number[][] } {
  const r = rng(5), X: number[][] = [], Y: number[][] = [];
  for (let i = 0; i < N_PIX; i++) {
    const x = (i + 0.5) / N_PIX;
    X.push([x]); Y.push([trueProfile(x) + 0.03 * r.norm()]);
  }
  return { X, Y };
}

/**
 * Start values: hidden unit k gets a hinge at a random position in [0, 1] (slope ±3..8, random sign),
 * small random output weights, output shift = mean intensity. Fixed seed.
 */
export function initHinges(net: Mlp, Y: number[][], seed: number): void {
  const r = rng(seed + 100), H = net.sizes[1];
  for (let k = 0; k < H; k++) {
    const w = (r.unif() < 0.5 ? -1 : 1) * (3 + 5 * r.unif());
    const xk = r.unif();
    net.W[0][k] = w; net.b[0][k] = -w * xk;
    net.W[1][k] = 0.1 * r.norm();
  }
  net.b[1][0] = Y.reduce((s, y) => s + y[0], 0) / Y.length;
}
