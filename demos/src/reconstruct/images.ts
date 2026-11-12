// Seeded binary test mask for the reconstruction demo (no DOM): 256 × 256 blobs (unions of
// discs), some touching the image border, some with holes, some joined by thin bridges,
// plus specks.
import { rng } from "../lib/random";

export const N = 256;

/** The synthetic nuclei of this demo (smoother than the lib default, so h-maxima work without a
 *  pre-filter), and the h of the "one dot per nucleus?" preset. */
export const NUCLEI_OPTS = { seed: 3, count: 22, texture: 6, noise: 4, blur: 1.5 };
export const NUCLEI_H = 15;

function disc(m: Uint8Array, cx: number, cy: number, r: number, v: number): void {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(N - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(N - 1, Math.ceil(cx + r)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) m[y * N + x] = v;
}

export function blobs(seed = 9): Uint8Array {
  const r = rng(seed), m = new Uint8Array(N * N), centres: { x: number; y: number; R: number }[] = [];
  // blobs on the border
  const border: [number, number][] = [[0, 70], [255, 160], [120, 0], [190, 255], [0, 230]];
  for (const [x, y] of border) { const R = 16 + 8 * r.unif(); disc(m, x, y, R, 1); disc(m, x + (x ? -1 : 1) * 10, y + (y ? -1 : 1) * 8, R * 0.7, 1); }
  for (let k = 0, tries = 0; k < 13 && tries < 4000; tries++) {
    const R = 11 + 9 * r.unif(), x = 30 + r.unif() * (N - 60), y = 30 + r.unif() * (N - 60);
    if (centres.some(c => Math.hypot(c.x - x, c.y - y) < c.R + R + 12)) continue;
    if (border.some(([bx, by]) => Math.hypot(bx - x, by - y) < R + 34)) continue;
    centres.push({ x, y, R });
    disc(m, x, y, R, 1);
    const a = 2 * Math.PI * r.unif();
    disc(m, x + 0.6 * R * Math.cos(a), y + 0.6 * R * Math.sin(a), 0.7 * R, 1);
    k++;
  }
  // holes in every second blob
  centres.forEach((c, k) => { if (k % 2 === 0) disc(m, c.x + 2, c.y - 1, Math.max(2.5, c.R * 0.3), 0); });
  // thin bridges between a few neighbouring blobs
  for (let k = 0; k + 1 < centres.length && k < 6; k += 3) {
    const a = centres[k], b = centres.slice(k + 1).reduce((best, c) => (Math.hypot(c.x - a.x, c.y - a.y) < Math.hypot(best.x - a.x, best.y - a.y) ? c : best));
    const L = Math.hypot(b.x - a.x, b.y - a.y);
    for (let t = 0; t <= L; t += 0.5) {
      const x = Math.round(a.x + (b.x - a.x) * t / L), y = Math.round(a.y + (b.y - a.y) * t / L);
      m[y * N + x] = 1; m[y * N + Math.min(N - 1, x + 1)] = 1;
    }
  }
  // specks
  for (let k = 0; k < 45; k++) {
    const x = 3 + Math.floor(r.unif() * (N - 6)), y = 3 + Math.floor(r.unif() * (N - 6));
    m[y * N + x] = 1; if (r.unif() < 0.5) m[y * N + x + 1] = 1; if (r.unif() < 0.3) m[(y + 1) * N + x] = 1;
  }
  return m;
}
