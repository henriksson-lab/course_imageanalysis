// Binary reconstruction as a breadth-first flood (no DOM; tested against lib/morph.ts in
// demos/test/morphology-test.ts). Iterated geodesic dilation of a marker inside a mask with
// the 3 × 3 square (8-connected) or the cross (4-connected) reaches, after k iterations,
// exactly the mask pixels within k steps of the marker: so one BFS gives the reconstruction
// AND the iteration at which each pixel joins (for the replay slider).

/** Iteration at which each pixel joins R_mask(marker): 0 for marker pixels inside the mask,
 *  k for pixels k steps away inside the mask, −1 if never reached. */
export function geodesicSteps(marker: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number, conn: 4 | 8 = 8): Int32Array {
  const n = w * h, d = new Int32Array(n).fill(-1), q = new Int32Array(n);
  let head = 0, tail = 0;
  for (let i = 0; i < n; i++) if (marker[i] && mask[i]) { d[i] = 0; q[tail++] = i; }
  const nb = conn === 8 ? [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] : [[0, -1], [-1, 0], [1, 0], [0, 1]];
  while (head < tail) {
    const p = q[head++], x = p % w, y = (p - x) / w;
    for (const [dx, dy] of nb) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const r = yy * w + xx;
      if (mask[r] && d[r] < 0) { d[r] = d[p] + 1; q[tail++] = r; }
    }
  }
  return d;
}

export const frameOf = (w: number, h: number): Uint8Array =>
  Uint8Array.from({ length: w * h }, (_, i) => { const x = i % w, y = (i - x) / w; return x === 0 || y === 0 || x === w - 1 || y === h - 1 ? 1 : 0; });

/** Reconstruction R_mask(marker) (8-connected unless conn = 4). */
export function reconstructBinary(marker: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number, conn: 4 | 8 = 8): Uint8Array {
  return Uint8Array.from(geodesicSteps(marker, mask, w, h, conn), v => (v >= 0 ? 1 : 0));
}

/** Fill holes: complement of the background reachable from the frame (background 4-connected). */
export function fillHolesRec(a: ArrayLike<number>, w: number, h: number): Uint8Array {
  const bg = Uint8Array.from(a, v => (v ? 0 : 1));
  const reach = reconstructBinary(frameOf(w, h), bg, w, h, 4);
  return Uint8Array.from(reach, v => (v ? 0 : 1));
}

/** Clear border: A minus the objects reachable from the frame (objects 8-connected). */
export function clearBorderRec(a: ArrayLike<number>, w: number, h: number): Uint8Array {
  const touch = reconstructBinary(frameOf(w, h), a, w, h, 8);
  return Uint8Array.from(a, (v, i) => (v && !touch[i] ? 1 : 0));
}
