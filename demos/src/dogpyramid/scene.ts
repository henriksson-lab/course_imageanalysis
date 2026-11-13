// The dogpyramid test image (no DOM, so the tests and figure scripts can use it).
import { Grey, createGrey } from "../lib/imageops";
import { rng } from "../lib/random";

/** The test image: ramp, discs and squares of three sizes (bright and dark), a long bar, noise. */
export function makeScene(n = 192, noise = 2, seed = 3): Grey {
  const g = createGrey(n, n), r = rng(seed);
  const discs: [number, number, number, number][] = [[30, 32, 3, 200], [70, 30, 6, 200], [130, 36, 12, 210], [36, 150, 12, 30], [100, 160, 6, 30], [160, 150, 3, 35], [96, 92, 3, 210]];
  const squares: [number, number, number, number][] = [[40, 92, 6, 205], [150, 92, 12, 200], [100, 125, 3, 35]];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let v = 80 + 40 * x / n;
    for (const [cx, cy, rad, val] of discs) { const c = Math.max(0, Math.min(1, rad + 0.5 - Math.hypot(x - cx, y - cy))); v = (1 - c) * v + c * val; }
    for (const [cx, cy, h, val] of squares) { const c = Math.max(0, Math.min(1, h + 0.5 - Math.abs(x - cx))) * Math.max(0, Math.min(1, h + 0.5 - Math.abs(y - cy))); v = (1 - c) * v + c * val; }
    // the long bar (2 px half width) from (20, 62) to (172, 62), its brightness varying slowly along it
    const cb = Math.max(0, Math.min(1, 2.5 - Math.abs(y - 62))) * Math.max(0, Math.min(1, 1 + Math.min(x - 20, 172 - x)));
    v = (1 - cb) * v + cb * (178 + 22 * Math.sin(2 * Math.PI * x / 21));
    g.data[y * n + x] = v + noise * r.norm();
  }
  return g;
}

