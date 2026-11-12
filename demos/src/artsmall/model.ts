// Tiny tomography systems for the artsmall demo and the by-hand exercises (no DOM): an
// n x n image (n = 2 or 3) measured by ray SUMS (weight 1 per pixel on the ray) along rows,
// columns and the two diagonal directions; Kaczmarz (ART) and SIRT steps on dense rows.
// Pixels are numbered row-major: for 2 x 2, (a, b; c, d) = (0, 1; 2, 3).
// Angle names follow the course convention (theta = direction of the line's normal):
// 0 deg = vertical rays = column sums; 90 deg = row sums; 45 deg = the diagonals
// running from top-left to bottom-right (a + d); 135 deg = the other diagonals (b + c).
import { rank, nullSpace } from "../lib/linalg";

export type RaySet = "rows" | "cols" | "diag" | "anti";
export const SET_ANGLE: Record<RaySet, string> = { cols: "0°", diag: "45°", rows: "90°", anti: "135°" };

export interface Ray { pix: number[]; set: RaySet; label: string }
export interface SmallSystem { n: number; rays: Ray[] }

/** All rays of the chosen sets (each set in a fixed order: top to bottom, left to right). */
export function smallSystem(n: number, sets: RaySet[]): SmallSystem {
  const rays: Ray[] = [];
  for (const set of sets) {
    if (set === "rows") for (let r = 0; r < n; r++) rays.push({ set, label: `row ${r + 1}`, pix: Array.from({ length: n }, (_, c) => r * n + c) });
    if (set === "cols") for (let c = 0; c < n; c++) rays.push({ set, label: `col ${c + 1}`, pix: Array.from({ length: n }, (_, r) => r * n + c) });
    if (set === "diag" || set === "anti") {
      // diag: col - row = k (k from n-1 down to -(n-1): top-right corner first); anti: col + row = k
      for (let t = 0; t < 2 * n - 1; t++) {
        const pix: number[] = [];
        for (let r = 0; r < n; r++) {
          const c = set === "diag" ? (n - 1 - t) + r : t - r;
          if (c >= 0 && c < n) pix.push(r * n + c);
        }
        rays.push({ set, label: `${set === "diag" ? "↘" : "↙"} ${t + 1}`, pix });
      }
    }
  }
  return { n, rays };
}

/** Dense matrix A (rays x pixels), row-major. */
export function matrix(s: SmallSystem): Float64Array {
  const N = s.n * s.n, A = new Float64Array(s.rays.length * N);
  s.rays.forEach((r, i) => r.pix.forEach(p => { A[i * N + p] = 1; }));
  return A;
}

export function apply(s: SmallSystem, x: ArrayLike<number>): Float64Array {
  return Float64Array.from(s.rays.map(r => r.pix.reduce((acc, p) => acc + x[p], 0)));
}

export function applyT(s: SmallSystem, y: ArrayLike<number>): Float64Array {
  const x = new Float64Array(s.n * s.n);
  s.rays.forEach((r, i) => r.pix.forEach(p => { x[p] += y[i]; }));
  return x;
}

export const rankOf = (s: SmallSystem): number => rank(matrix(s), s.rays.length, s.n * s.n);

/** Basis of the null space (images with zero data), each scaled so its largest entry is 1. */
export function nullImages(s: SmallSystem): Float64Array[] {
  return nullSpace(matrix(s), s.rays.length, s.n * s.n).map(v => {
    let m = 0;
    for (const x of v) if (Math.abs(x) > Math.abs(m)) m = x;
    return v.map(x => Math.round(x / m * 1e9) / 1e9);
  });
}

/** One Kaczmarz step on ray i: spread the error of that ray evenly over its pixels (times omega). */
export function kaczmarzStep(s: SmallSystem, b: ArrayLike<number>, x: ArrayLike<number>, i: number, omega = 1): Float64Array {
  const r = s.rays[i], out = Float64Array.from(x);
  const err = b[i] - r.pix.reduce((acc, p) => acc + x[p], 0);
  for (const p of r.pix) out[p] += omega * err / r.pix.length;
  return out;
}

/** Iterates of Kaczmarz over the given ray order (first entry = x0). */
export function kaczmarzPath(s: SmallSystem, b: ArrayLike<number>, x0: ArrayLike<number>, order: number[], omega = 1): Float64Array[] {
  const path: Float64Array[] = [Float64Array.from(x0)];
  for (const i of order) path.push(kaczmarzStep(s, b, path[path.length - 1], i, omega));
  return path;
}

/** One SIRT step: x + omega W_c A^T W_r (b - A x), with W_r, W_c the inverse row and column sums. */
export function sirtStep(s: SmallSystem, b: ArrayLike<number>, x: ArrayLike<number>, omega = 1): Float64Array {
  const ax = apply(s, x);
  const rr = Float64Array.from(s.rays.map((r, i) => (b[i] - ax[i]) / r.pix.length));
  const bt = applyT(s, rr), cs = applyT(s, s.rays.map(() => 1));
  return Float64Array.from(x, (v, p) => v + omega * (cs[p] > 0 ? bt[p] / cs[p] : 0));
}

// ---------------------------------------------------------------------------------------
// Two pixels: each measurement is a line a1 x1 + a2 x2 = b in the (x1, x2) plane.

export interface Line2 { a: [number, number]; b: number }

/** Orthogonal projection of x onto the line (relaxed by omega). */
export function project2(l: Line2, x: [number, number], omega = 1): [number, number] {
  const nn = l.a[0] ** 2 + l.a[1] ** 2, r = (l.b - (l.a[0] * x[0] + l.a[1] * x[1])) / nn;
  return [x[0] + omega * r * l.a[0], x[1] + omega * r * l.a[1]];
}

/** ART path: `sweeps` passes over the lines in order, every intermediate point. */
export function artPath2(lines: Line2[], x0: [number, number], sweeps: number, omega = 1): [number, number][] {
  const pts: [number, number][] = [x0];
  let x = x0;
  for (let s = 0; s < sweeps; s++) for (const l of lines) { x = project2(l, x, omega); pts.push(x); }
  return pts;
}

/** SIRT path in the plane: x + omega * mean over lines of the projections' corrections. */
export function sirtPath2(lines: Line2[], x0: [number, number], iters: number, omega = 1): [number, number][] {
  const pts: [number, number][] = [x0];
  let x = x0;
  for (let k = 0; k < iters; k++) {
    let d0 = 0, d1 = 0;
    for (const l of lines) { const p = project2(l, x); d0 += p[0] - x[0]; d1 += p[1] - x[1]; }
    x = [x[0] + omega * d0 / lines.length, x[1] + omega * d1 / lines.length];
    pts.push(x);
  }
  return pts;
}

/** Least-squares point of the lines (normal equations, 2 x 2). */
export function lsq2(lines: Line2[]): [number, number] | null {
  let a = 0, b = 0, d = 0, r0 = 0, r1 = 0;
  for (const l of lines) {
    a += l.a[0] ** 2; b += l.a[0] * l.a[1]; d += l.a[1] ** 2;
    r0 += l.a[0] * l.b; r1 += l.a[1] * l.b;
  }
  const det = a * d - b * b;
  if (Math.abs(det) < 1e-12) return null;
  return [(d * r0 - b * r1) / det, (a * r1 - b * r0) / det];
}
