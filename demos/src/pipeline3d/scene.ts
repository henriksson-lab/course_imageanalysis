// The scene of the pipeline3d demo (no DOM; used by main.ts and by the exercise scripts):
// a small house (10 vertices), the state of the demo, its presets, and the matrices M, V, P.
import * as M from "../lib/mat4";
import type { Mat4, Vec3 } from "../lib/mat4";
import { OrbitCamera, MeshData } from "../lib/gl3d";

export type Space = "model" | "world" | "view" | "ndc" | "screen";
export const SCREEN: [number, number] = [40, 30];

// ---- the object: a house (box + roof), 10 vertices ----
export const HV: Vec3[] = [
  [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
  [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
  [0, 1, -0.5], [0, 1, 0.5],
];
const WALL: Vec3 = [0.85, 0.72, 0.5], ROOF: Vec3 = [0.8, 0.25, 0.2], FRONT: Vec3 = [0.45, 0.65, 0.9], BACK: Vec3 = [0.6, 0.6, 0.62], FLOOR: Vec3 = [0.4, 0.4, 0.4];
/** Faces as counter-clockwise polygons seen from outside, with a colour. */
export const HF: [number[], Vec3][] = [
  [[4, 5, 7, 9, 6], FRONT], [[1, 0, 2, 8, 3], BACK],
  [[0, 4, 6, 2], WALL], [[5, 1, 3, 7], WALL], [[0, 1, 5, 4], FLOOR],
  [[6, 9, 8, 2], ROOF], [[7, 3, 8, 9], ROOF],
];
export const HT: [number, number, number, Vec3][] = HF.flatMap(([q, c]) => q.slice(1, -1).map((_, k) => [q[0], q[k + 1], q[k + 2], c] as [number, number, number, Vec3]));

export function houseMesh(P: Vec3[]): MeshData {
  const pos: number[] = [], col: number[] = [];
  for (const [a, b, c, k] of HT) for (const v of [a, b, c]) { pos.push(...P[v]); col.push(...k); }
  return { positions: pos, colors: col };
}

export interface State { space: Space; yaw: number; sx: number; tz: number; az: number; el: number; dist: number; fov: number; near: number; far: number; ortho: boolean; pick: number }
export const DEF: State = { space: "view", yaw: 30, sx: 1, tz: 0, az: 35, el: 20, dist: 4, fov: 40, near: 1, far: 8, ortho: false, pick: 9 };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "ortho", label: "orthographic vs perspective", s: { ortho: true, space: "ndc" } },
  { id: "dolly", label: "dolly zoom", s: { dist: 9, fov: 18.4, near: 6, far: 12, space: "screen" } },
  { id: "near", label: "near plane too close", s: { near: 0.05, far: 30, space: "ndc" } },
  { id: "behind", label: "a vertex behind the camera", s: { tz: 1, dist: 1.5, az: 40, el: 40, near: 0.1, space: "screen", pick: 9 } },
  { id: "scale", label: "non-uniform scale", s: { sx: 2.5, space: "world" } },
];


/** M (model → world), V (world → view), P (view → clip, for the 40 × 30 screen) of a state. */
export function matrices(st: State): { Mm: Mat4; V: Mat4; P: Mat4; cam: OrbitCamera } {
  const Mm = M.mul(M.translation(0, 0, st.tz), M.rotationY(st.yaw * Math.PI / 180), M.scaling(st.sx, 1, 1));
  const cam = new OrbitCamera({ distance: st.dist, azimuth: st.az * Math.PI / 180, elevation: st.el * Math.PI / 180, fovy: st.fov * Math.PI / 180, near: st.near, far: st.far, orthographic: st.ortho });
  return { Mm, V: cam.view(), P: cam.projection(SCREEN[0] / SCREEN[1]), cam };
}
