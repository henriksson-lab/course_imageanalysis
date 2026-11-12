// The synthetic test image of the isocontour demo (no DOM).
import { Grey, fromFunction } from "../lib/imageops";

export const N = 128;

/** The synthetic image: two Gaussian blobs (saddle value between them ≈ 128) and two discs
 *  joined by a 3-pixel neck, background 30. */
export function synthImage(): Grey {
  const disc = (x: number, y: number, cx: number, cy: number, r: number) => 0.5 * (1 + Math.tanh((r - Math.hypot(x - cx, y - cy)) / 0.8));
  const neck = (x: number, y: number) => (x > 40 && x < 92 ? 0.5 * (1 + Math.tanh((1.5 - Math.abs(y - 98)) / 0.6)) : 0);
  return fromFunction(N, N, (x, y) => {
    const g = 175 * (Math.exp(-((x - 42) ** 2 + (y - 42) ** 2) / (2 * 11 ** 2)) + Math.exp(-((x - 82) ** 2 + (y - 46) ** 2) / (2 * 11 ** 2)));
    const d = Math.max(disc(x, y, 34, 98, 12), disc(x, y, 96, 98, 13), neck(x, y));
    return 30 + g + 170 * d;
  });
}

