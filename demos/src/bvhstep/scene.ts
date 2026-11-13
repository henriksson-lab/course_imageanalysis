// Scenes of the bvhstep demo (no DOM; also used by the number and figure scripts).
import { rng } from "../lib/random";
import * as B from "../lib/bvh";

export type Layout = "uniform" | "clustered" | "stadium" | "wall";
export type Struct = "none" | "median" | "sah" | "grid";
export interface Prim { kind: 0 | 1; a: number[] }
export interface State { n: number; layout: Layout; struct: Struct; cells: number; level: number; o: [number, number]; aim: [number, number]; step: number }

/** Seeded scene of n primitives in [0, 100]². Object size shrinks like 1/√n (constant density). */
export function makeScene(n: number, layout: Layout, seed = 7): Prim[] {
  const r = rng(seed + n * 31 + layout.length), U = (a: number, b: number) => a + (b - a) * r.unif();
  const s = Math.min(4, 28 / Math.sqrt(n));
  const prims: Prim[] = [];
  const centres: [number, number][] = [[28, 30], [70, 72], [72, 25], [30, 75]];
  for (let i = 0; i < n; i++) {
    let x: number, y: number;
    if (layout === "uniform" || layout === "wall") { x = U(4, 96); y = U(4, 96); }
    else if (layout === "clustered") { const c = centres[i % 4]; x = c[0] + 7 * r.norm(); y = c[1] + 7 * r.norm(); }
    else {
      if (i % 20 === 0) { const a = U(0, 2 * Math.PI); x = 50 + 46 * Math.cos(a); y = 50 + 46 * Math.sin(a); }
      else { const a = U(0, 2 * Math.PI), rr = 3 * Math.sqrt(r.unif()); x = 54 + rr * Math.cos(a); y = 46 + rr * Math.sin(a); }
    }
    x = Math.max(1, Math.min(99, x)); y = Math.max(1, Math.min(99, y));
    const sz = layout === "stadium" && i % 20 !== 0 ? s * 0.25 : s;
    if (layout === "wall" && i === 0) { prims.push({ kind: 1, a: [5, 52, 95, 48] }); continue; }
    if (i % 3) prims.push({ kind: 0, a: [x, y, sz * U(0.3, 0.6)] });
    else { const a = U(0, Math.PI), L = sz * U(0.8, 1.6); prims.push({ kind: 1, a: [x - L / 2 * Math.cos(a), y - L / 2 * Math.sin(a), x + L / 2 * Math.cos(a), y + L / 2 * Math.sin(a)] }); }
  }
  return prims;
}
export const primBox = (p: Prim): B.AABB => p.kind === 0 ? { min: [p.a[0] - p.a[2], p.a[1] - p.a[2]], max: [p.a[0] + p.a[2], p.a[1] + p.a[2]] } : { min: [Math.min(p.a[0], p.a[2]), Math.min(p.a[1], p.a[3])], max: [Math.max(p.a[0], p.a[2]), Math.max(p.a[1], p.a[3])] };
export const primHit = (p: Prim, o: number[], d: number[], tmax: number): number => p.kind === 0 ? B.rayDisc2(o, d, p.a[0], p.a[1], p.a[2], tmax) : B.raySegment2(o, d, p.a[0], p.a[1], p.a[2], p.a[3], tmax);

