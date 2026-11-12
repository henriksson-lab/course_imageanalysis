// The functions of the levelsurface demo (no DOM; used by the tests and for the exercise
// numbers): shapes as signed distances in world coordinates x, y ∈ [−3, 3] (y up), and three
// functions φ with the same zero contour.
import { sdfBox } from "../lib/levelset";

export const EXT = 3;

export type ShapeId = "circle" | "ellipse" | "two" | "square" | "egg";
export type Mode = "sdf" | "x3" | "sq";
export const SHAPES: { id: ShapeId; label: string; R: number }[] = [
  { id: "circle", label: "circle", R: 1.5 },
  { id: "ellipse", label: "ellipse", R: 2.2 },
  { id: "two", label: "two circles", R: 1 },
  { id: "square", label: "square", R: 1.5 },
  { id: "egg", label: "egg crate", R: 1 },
];
export const MODES: { id: Mode; label: string }[] = [
  { id: "sdf", label: "distance d" },
  { id: "x3", label: "3 × d" },
  { id: "sq", label: "d + d²/2R" },
];

// ---- the functions ----
const ELL_A = 2.2, ELL_B = 1.2;
/** Distance from (x, y) to the ellipse (a cos t, b sin t): coarse search over t, then Newton
 *  on the derivative of the squared distance. */
function ellipseDistance(x: number, y: number): number {
  const a = ELL_A, b = ELL_B, d2 = (t: number) => (a * Math.cos(t) - x) ** 2 + (b * Math.sin(t) - y) ** 2;
  let t = 0, best = Infinity;
  for (let k = 0; k < 64; k++) { const s = 2 * Math.PI * k / 64, v = d2(s); if (v < best) { best = v; t = s; } }
  for (let it = 0; it < 20; it++) {
    const c = Math.cos(t), s = Math.sin(t);
    const f1 = -2 * a * s * (a * c - x) + 2 * b * c * (b * s - y);                       // d/dt of d2
    const f2 = 2 * a * a * s * s - 2 * a * c * (a * c - x) + 2 * b * b * c * c - 2 * b * s * (b * s - y);
    if (f2 <= 0) break;
    const tn = t - f1 / f2;
    if (Math.abs(tn - t) < 1e-14) { t = tn; break; }
    t = tn;
  }
  return Math.sqrt(Math.min(best, d2(t)));
}

/** Signed distance (or, for the egg crate, the given function) at world point (x, y); y up. */
export function baseFn(shape: ShapeId, x: number, y: number): number {
  switch (shape) {
    case "circle": return Math.hypot(x, y) - 1.5;
    case "two": return Math.min(Math.hypot(x - 1.2, y) - 1, Math.hypot(x + 1.2, y) - 1);
    case "square": return sdfBox(x, y, 0, 0, 1.5, 1.5);
    case "egg": return Math.sin(Math.PI * x / 2) * Math.sin(Math.PI * y / 2);
    case "ellipse": {
      const d = ellipseDistance(x, y);
      return (x / ELL_A) ** 2 + (y / ELL_B) ** 2 < 1 ? -d : d;
    }
  }
}

export function phiFn(shape: ShapeId, mode: Mode, x: number, y: number): number {
  const d = baseFn(shape, x, y);
  if (shape === "egg") return d;
  const R = SHAPES.find(s => s.id === shape)!.R;
  return mode === "x3" ? 3 * d : mode === "sq" ? d + d * d / (2 * R) : d;
}

/** Sample on an n × n grid; row 0 = top (y = +3), column 0 = left (x = −3). */
export function sample(shape: ShapeId, mode: Mode, n: number): Float64Array {
  const v = new Float64Array(n * n), hs = 2 * EXT / (n - 1);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) v[j * n + i] = phiFn(shape, mode, -EXT + i * hs, EXT - j * hs);
  return v;
}

/** φ, gradient, |∇φ| and κ at a world point by central differences of the function (h = 0.01). */
export function geometryAt(shape: ShapeId, mode: Mode, x: number, y: number) {
  const h = 0.01, f = (a: number, b: number) => phiFn(shape, mode, a, b);
  const c = f(x, y), l = f(x - h, y), r = f(x + h, y), d = f(x, y - h), u = f(x, y + h);
  const fx = (r - l) / (2 * h), fy = (u - d) / (2 * h);
  const fxx = (r - 2 * c + l) / (h * h), fyy = (u - 2 * c + d) / (h * h);
  const fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h);
  const g = Math.hypot(fx, fy);
  const k = (fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / (g * g * g);
  return { phi: c, fx, fy, g, k };
}

