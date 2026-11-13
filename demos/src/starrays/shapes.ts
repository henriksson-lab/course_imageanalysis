// Shape presets of the starrays demo on a 64 × 64 grid (pixel units, y down). No DOM.
import { ellipsePoly, type Poly, type Pt } from "../lib/shapecanvas";

/** A band between radii r0 and r1 over the angles a0..a1 (a C, a bent rod). */
export function arcBand(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number, m = 40): Poly {
  const out: Pt[] = [];
  for (let i = 0; i <= m; i++) { const a = a0 + (a1 - a0) * i / m; out.push([cx + r1 * Math.cos(a), cy + r1 * Math.sin(a)]); }
  for (let i = m; i >= 0; i--) { const a = a0 + (a1 - a0) * i / m; out.push([cx + r0 * Math.cos(a), cy + r0 * Math.sin(a)]); }
  return out;
}
export const deg = Math.PI / 180;
export const SHAPES: { id: string; label: string; polys: () => Poly[] }[] = [
  { id: "disc", label: "disc", polys: () => [ellipsePoly(32, 32, 18, 18, 64)] },
  { id: "ellipse", label: "ellipse 2:1", polys: () => [ellipsePoly(32, 32, 24, 12, 64)] },
  { id: "rect", label: "rectangle 8:4", polys: () => [[[12, 22], [52, 22], [52, 42], [12, 42]]] },
  { id: "L", label: "L", polys: () => [[[12, 8], [26, 8], [26, 42], [54, 42], [54, 56], [12, 56]]] },
  { id: "C", label: "C", polys: () => [arcBand(34, 32, 12, 23, 40 * deg, 320 * deg)] },
  { id: "ring", label: "ring", polys: () => [ellipsePoly(32, 32, 23, 23, 64), ellipsePoly(32, 32, 11, 11, 48)] },
  { id: "rod", label: "bent rod", polys: () => [arcBand(32, 50, 21, 28, 180 * deg, 360 * deg)] },
  // five-pointed star with its tips along the rays of n = 5 (0°, 72°, …): 5 rays hit the tips
  { id: "star", label: "star", polys: () => [Array.from({ length: 10 }, (_, i) => { const a = i * 36 * deg, r = i % 2 ? 11 : 26; return [32 + r * Math.cos(a), 32 + r * Math.sin(a)] as Pt; })] },
];

