// Synthetic test image of the localstats demo (no DOM, so the tests and the exercise
// numbers can use it in node).
import { rng } from "../lib/random";

export const N = 256;

/** Synthetic test image: six "cells" of different contrast on a brightness gradient, with noise. Integers 0..255. */
export function synthImage(seed = 7): Float64Array {
  const r = rng(seed), img = new Float64Array(N * N);
  const cells = [
    { x: 48, y: 64, c: 6 }, { x: 128, y: 64, c: 14 }, { x: 208, y: 64, c: 30 },
    { x: 48, y: 184, c: 60 }, { x: 128, y: 184, c: 10 }, { x: 208, y: 184, c: 40 },
  ];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 40 + 140 * x / (N - 1) + 30 * y / (N - 1);
    for (const c of cells) {
      const d = Math.hypot(x - c.x, y - c.y);
      if (d < 26) v += c.c * (d < 10 ? 1.8 : 1) * (0.8 + 0.2 * Math.cos(x * 0.9) * Math.cos(y * 0.9));
    }
    v += 3 * r.norm();
    img[y * N + x] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return img;
}

