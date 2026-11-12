// Seeded binary test images for the granulometry demo (no DOM). 256 × 256, 1 = object.
import { rng } from "../lib/random";
import { nuclei } from "../lib/synth";

export const SIZE = 256;

function disc(m: Uint8Array, cx: number, cy: number, r: number): void {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(SIZE - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(SIZE - 1, Math.ceil(cx + r)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) m[y * SIZE + x] = 1;
}

/** Non-overlapping discs of the given radii (gap ≥ `gap` pixels), plus `specks` tiny blobs. */
export function twoSizes(seed = 3, radii: [number, number][] = [[6, 22], [13, 9]], specks = 40): Uint8Array {
  const r = rng(seed), m = new Uint8Array(SIZE * SIZE), placed: { x: number; y: number; r: number }[] = [];
  for (const [rad, count] of [...radii].sort((a, b) => b[0] - a[0])) {
    for (let k = 0, tries = 0; k < count && tries < 5000; tries++) {
      const x = rad + 2 + r.unif() * (SIZE - 2 * rad - 4), y = rad + 2 + r.unif() * (SIZE - 2 * rad - 4);
      if (placed.some(p => Math.hypot(p.x - x, p.y - y) < p.r + rad + 4)) continue;
      placed.push({ x, y, r: rad }); disc(m, x, y, rad); k++;
    }
  }
  for (let k = 0, tries = 0; k < specks && tries < 5000; tries++) {
    const x = 2 + Math.floor(r.unif() * (SIZE - 4)), y = 2 + Math.floor(r.unif() * (SIZE - 4));
    if (placed.some(p => Math.hypot(p.x - x, p.y - y) < p.r + 3)) continue;
    m[y * SIZE + x] = 1;
    if (r.unif() < 0.5) m[y * SIZE + x + 1] = 1;
    k++;
  }
  return m;
}

/** Discs of radius 8 in touching pairs and chains of three, plus single discs. */
export function touching(seed = 5): Uint8Array {
  const r = rng(seed), m = new Uint8Array(SIZE * SIZE), rad = 8, groups: { x: number; y: number }[] = [];
  for (let k = 0, tries = 0; k < 14 && tries < 5000; tries++) {
    const x = 34 + r.unif() * (SIZE - 68), y = 34 + r.unif() * (SIZE - 68);
    if (groups.some(p => Math.hypot(p.x - x, p.y - y) < 58)) continue;
    groups.push({ x, y });
    const n = k % 3 === 0 ? 1 : k % 3 === 1 ? 2 : 3, a = r.unif() * Math.PI;
    for (let j = 0; j < n; j++) {
      const d = (j - (n - 1) / 2) * 2 * rad * 0.85;
      disc(m, x + d * Math.cos(a), y + d * Math.sin(a), rad);
    }
    k++;
  }
  return m;
}

/** Horizontal and vertical bars, 3 pixels wide and 24..48 long, plus specks. */
export function lines(seed = 7): Uint8Array {
  const r = rng(seed), m = new Uint8Array(SIZE * SIZE), boxes: [number, number, number, number][] = [];
  for (let k = 0, tries = 0; k < 26 && tries < 5000; tries++) {
    const len = 24 + Math.floor(r.unif() * 25), horiz = k % 2 === 0;
    const bw = horiz ? len : 3, bh = horiz ? 3 : len;
    const x = 4 + Math.floor(r.unif() * (SIZE - bw - 8)), y = 4 + Math.floor(r.unif() * (SIZE - bh - 8));
    if (boxes.some(([a, b, c, d]) => x < a + c + 5 && a < x + bw + 5 && y < b + d + 5 && b < y + bh + 5)) continue;
    boxes.push([x, y, bw, bh]);
    for (let yy = y; yy < y + bh; yy++) for (let xx = x; xx < x + bw; xx++) m[yy * SIZE + xx] = 1;
    k++;
  }
  return m;
}

/** The synthetic nuclei image (lib/synth.ts, seed 3), thresholded halfway between background and nuclei. */
export function nucleiMask(seed = 3): Uint8Array {
  const n = nuclei({ seed, count: 22 });
  return Uint8Array.from(n.image.data, v => (v > 115 ? 1 : 0));
}
