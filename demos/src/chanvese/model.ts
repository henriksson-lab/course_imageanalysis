// Model of the chanvese demo (no DOM; used by the tests): the synthetic test images
// (grey levels 0..1) with their ground truth, and the start shapes.
import * as LS from "../lib/levelset";
import { rng, gauss } from "../lib/random";

export const N = 128;
export type SceneId = "shapes" | "gradient";

const cov = (x: number, y: number, cx: number, cy: number, r: number) => Math.max(0, Math.min(1, r - Math.hypot(x - cx, y - cy) + 0.5));

/**
 * Noise-free test images, grey levels in [0, 1], and the true object mask.
 * "shapes": background 0.3, objects 0.7: a ring (disk with a hole), a disk with a blurred edge,
 * a square, a small disk and a thin bar (3 pixels wide). "gradient": the same objects at 0.75
 * on a background that rises from 0.05 (left) to 0.6 (right).
 */
export function scene(id: SceneId): { img: Float64Array; mask: Uint8Array } {
  const img = new Float64Array(N * N), mask = new Uint8Array(N * N);
  const lo = (x: number) => id === "shapes" ? 0.3 : 0.05 + 0.55 * x / (N - 1), hi = id === "shapes" ? 0.7 : 0.75;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    // ring at (36, 38): outer radius 22, hole radius 9
    const ring = Math.min(cov(x, y, 36, 38, 22), 1 - cov(x, y, 36, 38, 9));
    // blurred disk at (92, 92), radius 18, edge width ~ 6 pixels (logistic profile)
    const blur = 1 / (1 + Math.exp((Math.hypot(x - 92, y - 92) - 18) / 2.2));
    // square at (94, 34), half side 14
    const sq = Math.max(0, Math.min(1, 14.5 - Math.abs(x - 94))) * Math.max(0, Math.min(1, 14.5 - Math.abs(y - 34)));
    // small disk at (30, 96), radius 10; thin bar from (50, 112) to (70, 112)... vertical bar x 60..62, y 70..118
    const small = cov(x, y, 30, 96, 10);
    const bar = (x >= 59 && x <= 61 && y >= 72 && y <= 118) ? 1 : 0;
    const a = Math.max(ring, blur, sq, small, bar);
    img[y * N + x] = lo(x) + a * (hi - lo(x));
    mask[y * N + x] = a >= 0.5 ? 1 : 0;
  }
  return { img, mask };
}

/** Gaussian noise of standard deviation sd (fixed seed). */
export function addNoise(img: ArrayLike<number>, sd: number, seed = 11): Float64Array {
  const r = rng(seed);
  return Float64Array.from(img, v => v + (sd > 0 ? gauss(r, 0, sd) : 0));
}

export type InitId = "circle" | "grid" | "hole" | "ring";

/** Start shapes are signed distances divided by SCALE (so |∇φ| = 1/SCALE): the width of
 *  δ_ε is then about ε × SCALE pixels, and φ has no flat parts. */
export const SCALE = 10;
const scaled = (phi: Float64Array) => phi.map(v => v / SCALE);

/** Start shapes: one large circle in the middle, a grid of small blobs, a circle around the
 *  ring (its hole is far inside), or a small circle in the ring's hole. */
export function initPhi(id: InitId): Float64Array {
  if (id === "circle") return scaled(LS.sdfCircle(N, N, 64, 64, 40));
  if (id === "ring") return scaled(LS.sdfCircle(N, N, 36, 38, 27));
  if (id === "hole") return scaled(LS.sdfCircle(N, N, 36, 38, 5));
  // the classic robust start: a grid of small blobs, φ = −sin(πx/5) sin(πy/5)
  return LS.fromFn(N, N, (x, y) => -Math.sin(Math.PI * x / 5) * Math.sin(Math.PI * y / 5));
}

/** Pixels whose label (φ < 0 = object) differs from the truth. */
export function wrongPixels(phi: ArrayLike<number>, mask: ArrayLike<number>): number {
  let k = 0;
  for (let p = 0; p < mask.length; p++) if ((phi[p] < 0 ? 1 : 0) !== mask[p]) k++;
  return k;
}
