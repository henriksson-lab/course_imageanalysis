// Presets of the raster demo (no DOM; used by main.ts and by the exercise scripts).
import { WinVertex, ndcDepth } from "../lib/raster";

export const FAR = 10;
export type View = "colour" | "depth" | "bary" | "edge";
export interface Tri { p: [number, number][]; w: number[]; col: [number, number, number][]; uv?: [number, number][] }
export interface State { res: number; tris: Tri[]; cull: boolean; topLeft: boolean; persp: boolean; near: number; bits: number; view: View }
export interface Preset { id: string; label: string; make: () => State }

export const RES: [number, number][] = [[16, 12], [32, 24], [64, 48]];
const RGBV: [number, number, number][] = [[1, 0.15, 0.15], [0.15, 0.85, 0.2], [0.2, 0.35, 1]];
const ORANGE: [number, number, number][] = [[1, 0.65, 0.1], [1, 0.65, 0.1], [1, 0.65, 0.1]];
const base = (o: Partial<State>): State => ({ res: 1, tris: [], cull: false, topLeft: true, persp: true, near: 1, bits: 24, view: "colour", ...o });

/** A floor quad seen in perspective: two triangles with texture coordinates and true w. */
function floorQuad(): Tri[] {
  const [W, H] = RES[2], f = 1.2, asp = W / H;
  const P: [number, number, number][] = [[-1.6, -1, -1.3], [1.6, -1, -1.3], [1.6, -1, -9], [-1.6, -1, -9]];
  const s = P.map(([x, y, z]) => { const w = -z; return [(f * x / asp / w + 1) / 2 * W, (f * y / w + 1) / 2 * H, w] as [number, number, number]; });
  const uv: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const mk = (ix: number[]): Tri => ({ p: ix.map(k => [s[k][0], s[k][1]] as [number, number]), w: ix.map(k => s[k][2]), col: ix.map(() => [1, 1, 1] as [number, number, number]), uv: ix.map(k => uv[k]) });
  return [mk([0, 1, 2]), mk([0, 2, 3])];
}

export const PRESETS: Preset[] = [
  { id: "one", label: "one triangle", make: () => base({ tris: [{ p: [[4, 3], [26, 8], [12, 21]], w: [2, 4, 3], col: RGBV }] }) },
  { id: "intersect", label: "two triangles intersect", make: () => base({ tris: [
    { p: [[3, 3], [27, 5], [10, 21]], w: [2, 8, 2], col: RGBV },
    { p: [[8, 18], [14, 2], [29, 15]], w: [4, 4, 4], col: ORANGE }] }) },
  { id: "shared", label: "a shared edge", make: () => base({ res: 0, tris: [
    { p: [[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]], w: [3, 3, 3], col: RGBV },
    { p: [[13.5, 1.5], [13.5, 10.5], [1.5, 10.5]], w: [3, 3, 3], col: ORANGE }] }) },
  { id: "cull", label: "flipped winding is culled", make: () => base({ cull: true, tris: [
    { p: [[3, 3], [15, 4], [8, 20]], w: [3, 3, 3], col: RGBV },
    { p: [[17, 4], [24, 20], [29, 3]], w: [3, 3, 3], col: ORANGE }] }) },
  { id: "perspective", label: "affine vs perspective-correct", make: () => base({ res: 2, persp: false, tris: floorQuad() }) },
  { id: "precision", label: "near plane 0.01 vs 1", make: () => base({ near: 0.01, bits: 14, view: "colour", tris: [
    { p: [[3, 4], [29, 6], [12, 21]], w: [8, 9, 8], col: RGBV },
    { p: [[6, 18], [16, 3], [29, 16]], w: [8.5, 8.5, 8.5], col: ORANGE }] }) },
];


/** The triangles of a state in window coordinates (depth z from w, the near plane and FAR). */
export function winTris(st: State): WinVertex[][] {
  return st.tris.map(t => t.p.map((q, v) => ({
    x: q[0], y: q[1], w: t.w[v], z: ndcDepth(t.w[v], st.near, FAR),
    attr: [...t.col[v], ...(t.uv ? t.uv[v] : [0, 0]), t.uv ? 1 : 0],
  })));
}

/** The demo's fragment shader: vertex colours, or a checkerboard from the texture coordinates. */
export function shade(a: number[]): [number, number, number] {
  if (a[5] > 0.5) {
    const c = (Math.floor(a[3] * 8) + Math.floor(a[4] * 8)) % 2 ? 0.92 : 0.22;
    return [c, c, c * 0.9 + 0.08];
  }
  return [a[0], a[1], a[2]];
}
