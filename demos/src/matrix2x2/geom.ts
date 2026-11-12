// Pure 2x2 matrix helpers for the matrix2x2 demo (no DOM, so they can be unit-tested).
// A matrix is [a, b, c, d], row-major: ( a b ; c d ). Columns: (a, c) and (b, d).

export type M2 = [number, number, number, number];

export const ID: M2 = [1, 0, 0, 1];

/** A·(x, y). */
export function apply(m: M2, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y, m[2] * x + m[3] * y];
}

/** The product A·B ("first B, then A"). */
export function mul(A: M2, B: M2): M2 {
  return [
    A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
    A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
  ];
}

/** Signed area factor ad - bc. */
export function areaFactor(m: M2): number {
  return m[0] * m[3] - m[1] * m[2];
}

/** Inverse, or null when the matrix collapses the plane (columns on one line). */
export function inverse(m: M2): M2 | null {
  const k = areaFactor(m);
  if (Math.abs(k) < 1e-9) return null;
  return [m[3] / k, -m[1] / k, -m[2] / k, m[0] / k];
}

/** Remove floating-point dust, e.g. cos(90°) = 6e-17 -> 0. */
export function clean(v: number): number {
  const r = Math.round(v * 1e9) / 1e9;
  return r === 0 ? 0 : r;
}

/** Rotation by `deg` degrees counter-clockwise. */
export function rotation(deg: number): M2 {
  const t = deg * Math.PI / 180;
  return [clean(Math.cos(t)), clean(-Math.sin(t)), clean(Math.sin(t)), clean(Math.cos(t))];
}

export function same(A: M2, B: M2, tol = 1e-9): boolean {
  return A.every((v, i) => Math.abs(v - B[i]) < tol);
}

/** Round to a step (for snapping dragged handles), without float dust. */
export function snap(v: number, step: number): number {
  return clean(Math.round(v / step) * step);
}

/** Short number: at most 2 decimals, no trailing zeros, real minus sign. */
export function fmt(v: number, dec = 2): string {
  const r = Number(v.toFixed(dec));
  const s = (r === 0 ? 0 : r).toString();
  return s.replace("-", "−");
}

export interface EigDir { theta: number; lambda: number; }  // theta in [0, pi), radians

/**
 * Directions v = (cos t, sin t) with A v parallel to v (found without determinants:
 * A v is parallel to v exactly when the cross product v x Av is zero).
 * Returns "all" when every direction qualifies (a multiple of the identity).
 */
export function eigenDirections(m: M2): EigDir[] | "all" {
  const [a, b, c, d] = m;
  const eps = 1e-9;
  // cross(v, Av) = c x^2 + (d - a) x y - b y^2 for v = (x, y)
  const dirs: number[] = [];
  if (Math.abs(b) < eps) {
    dirs.push(Math.PI / 2);                                // the y-axis
    if (Math.abs(d - a) > eps) dirs.push(Math.atan(-c / (d - a)));
    else if (Math.abs(c) < eps) return "all";
  } else {
    const D = (d - a) * (d - a) + 4 * b * c;
    if (D > eps) {
      const s = Math.sqrt(D);
      dirs.push(Math.atan(((d - a) + s) / (2 * b)), Math.atan(((d - a) - s) / (2 * b)));
    } else if (D > -eps) {
      dirs.push(Math.atan((d - a) / (2 * b)));
    }
  }
  const out: EigDir[] = [];
  for (let t of dirs) {
    t = ((t % Math.PI) + Math.PI) % Math.PI;
    if (Math.abs(t - Math.PI) < 1e-12) t = 0;
    if (out.some(e => Math.abs(e.theta - t) < 1e-9)) continue;
    const x = Math.cos(t), y = Math.sin(t);
    const [ax, ay] = apply(m, x, y);
    out.push({ theta: t, lambda: clean(ax * x + ay * y) });
  }
  return out.sort((p, q) => q.lambda - p.lambda);
}

/** Smallest angle (radians) between the line at angle t and the line at angle e (lines, mod pi). */
export function lineGap(t: number, e: number): number {
  const g = (((t - e) % Math.PI) + Math.PI) % Math.PI;
  return Math.min(g, Math.PI - g);
}
