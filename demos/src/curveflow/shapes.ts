// Start shapes of the curveflow demo as signed distance functions on a 128 × 128 grid, and
// the explicit polygons taken from their zero contours. No DOM (used by the tests).
import * as LS from "../lib/levelset";
import { Contour } from "../lib/marching";

export const N = 128;
export type ShapeId = "circle" | "two" | "dumbbell" | "star" | "square";

/** The initial φ of a shape: a signed distance function. */
export function shapePhi(id: ShapeId): Float64Array {
  let phi: Float64Array;
  switch (id) {
    case "circle": return LS.sdfCircle(N, N, 64, 64, 30);
    case "two": return LS.union(LS.sdfCircle(N, N, 40, 64, 18), LS.sdfCircle(N, N, 88, 64, 18));
    case "square": return LS.fromFn(N, N, (x, y) => LS.sdfBox(x, y, 64, 64, 32, 32));
    case "star": return LS.polygonSDF([LS.radialPoly(64, 64, t => 32 + 13 * Math.cos(5 * t), 300)], N, N);
    case "dumbbell":
      phi = LS.fromFn(N, N, (x, y) => Math.min(Math.hypot(x - 34, y - 64) - 22, Math.hypot(x - 94, y - 64) - 22, LS.sdfBox(x, y, 64, 64, 30, 6)));
      LS.reinitExact(phi, N, N);
      return phi;
  }
}

/** Closed polylines of a contour, resampled to about one vertex per pixel. */
export function contourPolys(C: Contour): LS.Poly[] {
  const out: LS.Poly[] = [];
  for (const L of C.lines) {
    if (!L.closed || L.ids.length < 4) continue;
    const pts = L.ids.map(i => [C.points[2 * i], C.points[2 * i + 1]] as [number, number]);
    const P: LS.Poly = [pts[0]];
    let acc = 0;
    for (let k = 1; k < pts.length; k++) {
      const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
      acc += d;
      if (acc >= 1) { P.push(pts[k]); acc = 0; }
    }
    if (P.length >= 4) out.push(P);
  }
  return out;
}

