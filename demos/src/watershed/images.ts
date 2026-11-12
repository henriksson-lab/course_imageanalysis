// Seeded synthetic test image for the watershed demo (no DOM): 256 × 256, background 80,
// bright blobs (175; two pairs touch), and a "cell" outlined by a dark membrane (35) whose
// outline has a gap on its left side. Blurred (σ = 1.2), Gaussian noise of the given sd.
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";

export const N = 256;

export interface Blob { x: number; y: number; r: number }
export const BLOBS: Blob[] = [
  { x: 62, y: 66, r: 25 }, { x: 103, y: 84, r: 21 },     // a touching pair
  { x: 58, y: 168, r: 22 }, { x: 96, y: 200, r: 17 },    // a second pair
  { x: 150, y: 44, r: 16 }, { x: 40, y: 228, r: 12 },
];
/** The cell with a membrane: centre, radius, membrane half-width, gap (centre angle, half-width in radians). */
export const RING = { x: 178, y: 160, r: 52, half: 2.5, gapAt: Math.PI, gapHalf: 0.22 };

export function synthClean(): Grey {
  const img = createGrey(N, N, 80);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    for (const b of BLOBS) if ((x - b.x) ** 2 + (y - b.y) ** 2 <= b.r * b.r) img.data[i] = 175;
    const d = Math.hypot(x - RING.x, y - RING.y), a = Math.atan2(y - RING.y, x - RING.x);
    let da = Math.abs(a - RING.gapAt);
    da = Math.min(da, 2 * Math.PI - da);
    if (Math.abs(d - RING.r) <= RING.half && da > RING.gapHalf) img.data[i] = 35;
    // a nucleus inside the ring cell
    if ((x - RING.x - 8) ** 2 + (y - RING.y + 4) ** 2 <= 15 * 15) img.data[i] = 150;
  }
  return gaussianBlur(img, 1.2);
}

export function synth(noise: number, seed = 11): Grey {
  const c = synthClean();
  return noise > 0 ? addGaussianNoise(c, noise, seed) : c;
}

/** Marker positions for the presets: [x, y, label] with label 1 = background. */
export const CLICKS_OBJECTS: [number, number, number][] = [
  [62, 66, 2], [103, 84, 3], [58, 168, 4], [96, 200, 5], [150, 44, 6], [40, 228, 7], [178, 190, 8], [20, 20, 1],
];
export const CLICKS_LEAK: [number, number, number][] = [[178, 190, 2], [240, 16, 1]];
