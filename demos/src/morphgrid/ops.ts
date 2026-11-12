// Binary morphology on small grids, written out pixel by pixel so the stepper can show each
// decision (no DOM; tested against lib/morph.ts in demos/test/morphology-test.ts).
// Offsets are [dx, dy] (x right, y down) relative to the SE origin.
//   erosion   (A ⊖ B)(x) = 1  iff  x + b ∈ A for every b ∈ B      ("the SE fits")
//   dilation  (A ⊕ B)(x) = 1  iff  x − b ∈ A for some b ∈ B       ("the reflected SE hits")
// Pixels outside the grid have the value `outside` (0 or 1).

export type Off = [number, number];

/** Offsets of the set cells of a k × k grid, relative to the origin cell `origin` (index). */
export function seFromGrid(values: ArrayLike<number>, k: number, origin: number): Off[] {
  const ox = origin % k, oy = Math.floor(origin / k), out: Off[] = [];
  for (let i = 0; i < k * k; i++) if (values[i]) out.push([(i % k) - ox, Math.floor(i / k) - oy]);
  return out;
}

export const reflect = (se: Off[]): Off[] => se.map(([dx, dy]) => [-dx, -dy] as Off);

export function at(a: ArrayLike<number>, w: number, h: number, x: number, y: number, outside: number): number {
  return x < 0 || y < 0 || x >= w || y >= h ? outside : a[y * w + x] ? 1 : 0;
}

/** Values under the SE at pixel (x, y): x + b for erosion, x − b for dilation. */
export function probe(a: ArrayLike<number>, w: number, h: number, x: number, y: number, se: Off[], kind: "erode" | "dilate", outside: number): number[] {
  const s = kind === "erode" ? 1 : -1;
  return se.map(([dx, dy]) => at(a, w, h, x + s * dx, y + s * dy, outside));
}

export function erodeGrid(a: ArrayLike<number>, w: number, h: number, se: Off[], outside = 0): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = probe(a, w, h, x, y, se, "erode", outside).every(v => v === 1) ? 1 : 0;
  return out;
}

export function dilateGrid(a: ArrayLike<number>, w: number, h: number, se: Off[], outside = 0): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = probe(a, w, h, x, y, se, "dilate", outside).some(v => v === 1) ? 1 : 0;
  return out;
}

export const complement = (a: ArrayLike<number>): Uint8Array => Uint8Array.from(a, v => (v ? 0 : 1));

export const openGrid = (a: ArrayLike<number>, w: number, h: number, se: Off[], outside = 0) =>
  dilateGrid(erodeGrid(a, w, h, se, outside), w, h, se, outside);
export const closeGrid = (a: ArrayLike<number>, w: number, h: number, se: Off[], outside = 0) =>
  erodeGrid(dilateGrid(a, w, h, se, outside), w, h, se, outside);

/** Two-pass chamfer distance on a small grid (city-block: 4 steps, chessboard: 8 steps),
 *  distance from each object pixel to the nearest background pixel inside the grid;
 *  `INF` where there is no background. */
export const INF = 999;
export const FORWARD: Record<"cityblock" | "chessboard", Off[]> = {
  cityblock: [[0, -1], [-1, 0]],
  chessboard: [[-1, -1], [0, -1], [1, -1], [-1, 0]],
};

export function chamferGrid(a: ArrayLike<number>, w: number, h: number, metric: "cityblock" | "chessboard"): Int32Array {
  const d = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = a[i] ? INF : 0;
  for (const s of [1, -1]) {
    for (let k = 0; k < w * h; k++) {
      const i = s > 0 ? k : w * h - 1 - k, x = i % w, y = (i - x) / w;
      if (!d[i]) continue;
      for (const [dx, dy] of FORWARD[metric]) {
        const xx = x + s * dx, yy = y + s * dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        d[i] = Math.min(d[i], d[yy * w + xx] + 1);
      }
    }
  }
  return d;
}
