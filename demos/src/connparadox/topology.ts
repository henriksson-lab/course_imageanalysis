// Digital topology checks for the connparadox demo (labelling day, unit label-connectivity/).
// No DOM. Foreground components with one connectivity, background components with another;
// "enclosed" background components are those that do not touch the image frame (holes in
// 2D, cavities in 3D).
import { label2d, label3d, Conn3 } from "../lib/label";

export interface Analysis {
  fg: number;
  bg: number;
  enclosed: number;
  fgLabels: Int32Array;
  bgLabels: Int32Array;
  /** Which background labels are enclosed (index = label). */
  isEnclosed: Uint8Array;
}

function enclosedOf(bg: Int32Array, count: number, touches: (i: number) => boolean): Uint8Array {
  const out = new Uint8Array(count + 1).fill(1);
  out[0] = 0;
  for (let i = 0; i < bg.length; i++) if (bg[i] && touches(i)) out[bg[i]] = 0;
  return out;
}

export function analyse2d(mask: ArrayLike<number>, w: number, h: number, fgConn: 4 | 8, bgConn: 4 | 8): Analysis {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const f = label2d(mask, w, h, fgConn), b = label2d(inv, w, h, bgConn);
  const isEnclosed = enclosedOf(b.data, b.count, i => { const x = i % w, y = (i - x) / w; return x === 0 || y === 0 || x === w - 1 || y === h - 1; });
  let enclosed = 0;
  for (let l = 1; l <= b.count; l++) enclosed += isEnclosed[l];
  return { fg: f.count, bg: b.count, enclosed, fgLabels: f.data, bgLabels: b.data, isEnclosed };
}

export function analyse3d(mask: ArrayLike<number>, n: number, fgConn: Conn3, bgConn: Conn3): Analysis {
  const N = n * n * n, inv = new Uint8Array(N);
  for (let i = 0; i < N; i++) inv[i] = mask[i] ? 0 : 1;
  const f = label3d(mask, n, n, n, fgConn), b = label3d(inv, n, n, n, bgConn);
  const isEnclosed = enclosedOf(b.data, b.count, i => {
    const x = i % n, y = Math.floor(i / n) % n, z = Math.floor(i / (n * n));
    return x === 0 || y === 0 || z === 0 || x === n - 1 || y === n - 1 || z === n - 1;
  });
  let enclosed = 0;
  for (let l = 1; l <= b.count; l++) enclosed += isEnclosed[l];
  return { fg: f.count, bg: b.count, enclosed, fgLabels: f.data, bgLabels: b.data, isEnclosed };
}

/** The connectivity that complements c (8 ↔ 4, 26 ↔ 6, 18 ↔ 6). */
export const complement2 = (c: 4 | 8): 4 | 8 => (c === 8 ? 4 : 8);
export const complement3 = (c: Conn3): Conn3 => (c === 6 ? 26 : 6);

/** One-line verdict: compare the chosen background connectivity with the consistent one. */
export function verdict2d(mask: ArrayLike<number>, w: number, h: number, fgConn: 4 | 8, bgConn: 4 | 8): { text: string; ok: boolean } {
  const a = analyse2d(mask, w, h, fgConn, bgConn);
  if (bgConn !== fgConn) return { ok: true, text: `Consistent pair (${fgConn}/${bgConn}): every closed curve separates an inside from the outside, and only a closed curve does.` };
  const ref = analyse2d(mask, w, h, fgConn, complement2(fgConn));
  if (fgConn === 8 && a.enclosed < ref.enclosed) return { ok: false, text: `8/8: the curve is closed, yet the inside leaks out through the diagonal gaps (${ref.enclosed - a.enclosed} enclosed region${ref.enclosed - a.enclosed > 1 ? "s" : ""} lost).` };
  if (fgConn === 4 && a.enclosed > ref.enclosed) return { ok: false, text: `4/4: the curve is broken into ${a.fg} pieces, yet the inside is cut off from the outside.` };
  return { ok: false, text: `${fgConn}/${bgConn} is not a consistent pair; this picture happens to show no conflict.` };
}

export function verdict3d(mask: ArrayLike<number>, n: number, fgConn: Conn3, bgConn: Conn3): { text: string; ok: boolean } {
  const okPair = (fgConn === 26 && bgConn === 6) || (fgConn === 6 && bgConn === 26) || (fgConn === 18 && bgConn === 6) || (fgConn === 6 && bgConn === 18);
  if (okPair) return { ok: true, text: `Consistent pair (${fgConn}/${bgConn}).` };
  const a = analyse3d(mask, n, fgConn, bgConn), ref = analyse3d(mask, n, fgConn, complement3(fgConn));
  if (a.enclosed < ref.enclosed) return { ok: false, text: `${fgConn}/${bgConn}: the shell is closed, yet the cavity leaks out through a corner or an edge.` };
  if (a.enclosed > ref.enclosed) return { ok: false, text: `${fgConn}/${bgConn}: the shell has a gap at a corner or an edge, yet the cavity is sealed.` };
  return { ok: false, text: `${fgConn}/${bgConn} is not a consistent pair; this volume happens to show no conflict.` };
}

/** Brute-force 3D components of an n³ volume (for tests). */
export function bruteLabels3d(m: ArrayLike<number>, n: number, conn: Conn3): Int32Array {
  const N = n * n * n, l = new Int32Array(N);
  for (let i = 0; i < N; i++) l[i] = m[i] ? i + 1 : 0;
  let changed = true;
  while (changed) {
    changed = false;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = (z * n + y) * n + x;
      if (!l[i]) continue;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const k = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
        if (!k || (k === 2 && conn < 18) || (k === 3 && conn < 26)) continue;
        const xx = x + dx, yy = y + dy, zz = z + dz;
        if (xx < 0 || yy < 0 || zz < 0 || xx >= n || yy >= n || zz >= n) continue;
        const j = (zz * n + yy) * n + xx;
        if (l[j] && l[j] < l[i]) { l[i] = l[j]; changed = true; }
      }
    }
  }
  return l;
}

const diamond = (n: number, cx: number, cy: number, r: number) => Array.from({ length: n }, (_, y) =>
  Array.from({ length: n }, (_, x) => (Math.abs(x - cx) + Math.abs(y - cy) === r ? "1" : "0")).join(""));

export const PARADOX_GRIDS: Record<string, string[]> = {
  diamond: diamond(12, 5, 6, 4),
  staircase: Array.from({ length: 12 }, (_, y) => Array.from({ length: 12 }, (_, x) => (x === 11 - y ? "1" : "0")).join("")),
  square: Array.from({ length: 12 }, (_, y) => Array.from({ length: 12 }, (_, x) => ((y === 2 || y === 9) && x >= 2 && x <= 9) || ((x === 2 || x === 9) && y >= 2 && y <= 9) ? "1" : "0").join("")),
  twoblobs: ["000000000000", "011100000000", "011100000000", "011100000000", "000011100000", "000011100000", "000011100000", "000000000000", "000000011100", "000000011100", "000000011100", "000000000000"],
};

/** 3D presets on a 5 × 5 × 5 grid: lists of [x, y, z] voxels. */
export function preset3d(id: string): Uint8Array {
  const n = 5, m = new Uint8Array(n * n * n), set = (x: number, y: number, z: number) => { m[(z * n + y) * n + x] = 1; };
  if (id === "corner") { set(1, 1, 1); set(2, 2, 2); set(3, 3, 3); }
  else if (id === "edge") { set(1, 1, 2); set(2, 2, 2); set(3, 3, 2); }
  else if (id === "shell" || id === "leaky") {
    for (let z = 1; z <= 3; z++) for (let y = 1; y <= 3; y++) for (let x = 1; x <= 3; x++) if (!(x === 2 && y === 2 && z === 2)) set(x, y, z);
    if (id === "leaky") m[(1 * n + 1) * n + 1] = 0;
  }
  return m;
}
