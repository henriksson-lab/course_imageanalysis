// Shapes for the moments demo (analytic outlines as fine polygons) and their exact
// ("continuous") measures, against which the digitised estimates are compared. No DOM.
import { Poly, Pt, ellipsePoly, starPoly } from "../lib/shapecanvas";

export const disc = (cx: number, cy: number, r: number, n = 120): Poly => ellipsePoly(cx, cy, r, r, n);

/** A crescent: the disc (cx, cy, r) minus the disc of the same radius shifted by d to the right. */
export function crescent(cx: number, cy: number, r: number, d: number, n = 60): Poly {
  const y = Math.sqrt(Math.max(0, r * r - d * d / 4)), t0 = Math.atan2(y, d / 2);
  const out: Pt[] = [];
  for (let i = 0; i <= n; i++) { const t = t0 + (2 * Math.PI - 2 * t0) * i / n; out.push([cx + r * Math.cos(t), cy - r * Math.sin(t)]); }
  for (let i = 1; i < n; i++) { const t = Math.PI + t0 - 2 * t0 * i / n; out.push([cx + d + r * Math.cos(t), cy - r * Math.sin(t)]); }
  return out;
}

/** Union of two overlapping discs of radius r with centres d apart (d < 2r), along x. */
export function twoDiscs(cx: number, cy: number, r: number, d: number, n = 60): Poly {
  const t1 = Math.acos(Math.min(1, d / (2 * r))), out: Pt[] = [];
  const a = cx - d / 2, b = cx + d / 2;
  for (let i = 0; i <= n; i++) { const t = t1 + (2 * Math.PI - 2 * t1) * i / n; out.push([a + r * Math.cos(t), cy - r * Math.sin(t)]); }
  for (let i = 1; i < n; i++) { const t = -(Math.PI - t1) + (2 * Math.PI - 2 * t1) * i / n; out.push([b + r * Math.cos(t), cy - r * Math.sin(t)]); }
  return out;
}

/** A rod: rectangle of length L between the cap centres, half-width r, rotated by angle
 *  (counter-clockwise on screen). */
export function rod(cx: number, cy: number, L: number, r: number, angle = 0, n = 24): Poly {
  const pts: Pt[] = [];
  for (let i = 0; i <= n; i++) { const t = -Math.PI / 2 + Math.PI * i / n; pts.push([L / 2 + r * Math.cos(t), r * Math.sin(t)]); }
  for (let i = 0; i <= n; i++) { const t = Math.PI / 2 + Math.PI * i / n; pts.push([-L / 2 + r * Math.cos(t), r * Math.sin(t)]); }
  const c = Math.cos(angle), s = Math.sin(angle);
  return pts.map(([x, y]) => [cx + c * x + s * y, cy - s * x + c * y] as Pt);
}

export const rect = (x0: number, y0: number, x1: number, y1: number): Poly => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
export const star = starPoly;

function inside(p: Poly, [x, y]: Pt): boolean {
  let c = false;
  for (let a = 0, b = p.length - 1; a < p.length; b = a++) {
    const [xa, ya] = p[a], [xb, yb] = p[b];
    if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) c = !c;
  }
  return c;
}

/** +1 for an outer outline, −1 for a hole (even-odd: inside an odd number of others). */
function signs(polys: Poly[]): number[] {
  return polys.map((p, i) => (polys.filter((q, j) => j !== i && p.length && inside(q, p[0])).length % 2 ? -1 : 1));
}

export interface Exact { area: number; perimeter: number; cx: number; cy: number; s11: number; s12: number; s22: number }

/** Exact area, perimeter, centroid and covariance of the region bounded by the polygons
 *  (even-odd rule; overlapping separate outlines are counted twice). Green's theorem. */
export function exactMeasures(polys: Poly[]): Exact | null {
  const sg = signs(polys);
  let A = 0, P = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0, Syy = 0;
  polys.forEach((p, k) => {
    let a = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < p.length; i++) {
      const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length], c = x0 * y1 - x1 * y0;
      a += c; sx += (x0 + x1) * c; sy += (y0 + y1) * c;
      sxx += (x0 * x0 + x0 * x1 + x1 * x1) * c; syy += (y0 * y0 + y0 * y1 + y1 * y1) * c;
      sxy += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c;
      P += Math.hypot(x1 - x0, y1 - y0);
    }
    const o = Math.sign(a) * sg[k];   // orient each outline, then add or subtract it
    A += o * a / 2; Sx += o * sx / 6; Sy += o * sy / 6; Sxx += o * sxx / 12; Syy += o * syy / 12; Sxy += o * sxy / 24;
  });
  if (A <= 1e-9) return null;
  const cx = Sx / A, cy = Sy / A;
  return { area: A, perimeter: P, cx, cy, s11: Sxx / A - cx * cx, s12: Sxy / A - cx * cy, s22: Syy / A - cy * cy };
}
