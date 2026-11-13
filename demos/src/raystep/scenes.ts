// Scenes and presets of the raystep demo (ray-tracing day, units raycast/ and whitted/).
import type { Obj, Scene, Material } from "../lib/raytrace";
import type { Vec3 } from "../lib/mat4";

export type SceneId = "basic" | "one" | "mirror" | "mirrors" | "glass" | "glassfloor" | "tile";

const red: Material = { kind: "diffuse", colour: [0.8, 0.15, 0.12] };
const blue: Material = { kind: "diffuse", colour: [0.2, 0.35, 0.85] };
const yellow: Material = { kind: "diffuse", colour: [0.9, 0.75, 0.15] };
const ground: Material = { kind: "diffuse", colour: [0.85, 0.85, 0.85], checker: true };

/** Sphere A (the one the radius slider changes) is always object 0. */
export function makeScene(id: SceneId, radius: number): Scene {
  const A = (mat: Material, c: Vec3 = [-1, radius, -1.5]): Obj => ({ name: "sphere A", type: "sphere", c, r: radius, mat });
  const B: Obj = { name: "sphere B", type: "sphere", c: [1.4, 0.6, -3], r: 0.6, mat: blue };
  const T: Obj = { name: "triangle", type: "triangle", a: [-0.6, 0, -4.5], b: [1.6, 0, -5], c: [0.3, 2.2, -4.7], mat: yellow };
  const G: Obj = { name: "ground", type: "plane", p: [0, 0, 0], n: [0, 1, 0], mat: ground };
  const lights = [{ p: [-3, 5, 2] as Vec3, colour: [0.95, 0.95, 0.9] as [number, number, number] }];
  const sky: [number, number, number] = [0.55, 0.7, 0.9];
  switch (id) {
    case "one": return { objects: [A(red, [0, 0, -3])], lights, background: [0.15, 0.15, 0.2] };
    case "mirror": return { objects: [A({ kind: "mirror", colour: [0.95, 0.95, 0.95] }), B, T, G], lights, background: sky };
    case "glass": return { objects: [A({ kind: "glass", colour: [1, 1, 1], ior: 1.5 }), B, T, G], lights, background: sky };
    case "mirrors": return {
      objects: [A(red, [0, radius, -2.5]),
        { name: "left mirror", type: "plane", p: [-2.2, 0, 0], n: [1, 0, 0], mat: { kind: "mirror", colour: [0.9, 0.95, 0.9] } },
        { name: "right mirror", type: "plane", p: [2.2, 0, 0], n: [-1, 0, 0], mat: { kind: "mirror", colour: [0.9, 0.95, 0.9] } }, G],
      lights: [{ p: [0, 5, 2], colour: [0.95, 0.95, 0.9] }], background: sky,
    };
    case "glassfloor": return {
      objects: [A(red), B, T, { name: "glass floor", type: "plane", p: [0, 0, 0], n: [0, 1, 0], mat: { kind: "glass", colour: [1, 1, 1], ior: 1.5 } },
        { name: "lower floor", type: "plane", p: [0, -0.6, 0], n: [0, 1, 0], mat: ground }],
      lights, background: sky,
    };
    case "tile": return {
      objects: [A(red), B, T, G,
        { name: "low tile", type: "triangle", a: [0.2, 0.12, -0.6], b: [2.2, 0.12, -0.6], c: [1.2, 0.12, -2.0], mat: yellow }],
      lights, background: sky,
    };
    default: return { objects: [A(red), B, T, G], lights, background: sky };
  }
}

export interface State {
  scene: SceneId;
  W: number; H: number;
  fov: number;
  eye: [number, number, number];
  ortho: boolean;
  radius: number;
  view: "shaded" | "depth" | "normal" | "id" | "bary";
  depth: number;
  shadows: boolean; reflection: boolean; refraction: boolean; fresnel: boolean;
  /** log10 of ε; −8 means ε = 0 */
  epsLog: number;
  inspect: [number, number];
  /** Pixels traced so far, in raster order (−1 = all). */
  traced: number;
}

export const DEF0: State = {
  scene: "basic", W: 32, H: 24, fov: 60, eye: [0, 1, 3], ortho: false, radius: 0.8, view: "shaded",
  depth: 0, shadows: false, reflection: false, refraction: false, fresnel: false, epsLog: -4, inspect: [11, 13], traced: -1,
};
export const DEF1: State = { ...DEF0, depth: 3, shadows: true, reflection: true, refraction: true, fresnel: true };

export interface Preset { id: string; label: string; s: Partial<State> }

export const PRESETS0: Preset[] = [
  { id: "one", label: "one sphere", s: { scene: "one", W: 16, H: 12, inspect: [8, 6], traced: 0 } },
  { id: "miss", label: "a ray that misses", s: { scene: "one", W: 16, H: 12, inspect: [2, 2] } },
  { id: "behind", label: "sphere behind the camera", s: { scene: "basic", eye: [-1, 0.8, -3], inspect: [16, 12] } },
  { id: "inside", label: "eye inside the sphere", s: { scene: "basic", eye: [-1, 0.8, -1.5], inspect: [16, 12] } },
  { id: "bary", label: "the triangle: barycentrics", s: { scene: "basic", view: "bary", W: 64, H: 48, inspect: [34, 25] } },
  { id: "ortho", label: "orthographic vs perspective", s: { scene: "basic", ortho: true } },
  { id: "count", label: "count the tests", s: { scene: "basic", W: 64, H: 48, inspect: [32, 24] } },
];
export const PRESETS1: Preset[] = [
  { id: "acne", label: "shadow acne (ε = 0)", s: { scene: "basic", epsLog: -8, depth: 1, W: 64, H: 48, inspect: [20, 40] } },
  { id: "leak", label: "light leaks (huge ε)", s: { scene: "tile", epsLog: -0.7, depth: 1, W: 64, H: 48, inspect: [40, 37] } },
  { id: "mirror", label: "mirror ball", s: { scene: "mirror", W: 64, H: 48, inspect: [22, 26] } },
  { id: "mirrors", label: "two facing mirrors", s: { scene: "mirrors", depth: 6, W: 64, H: 48, inspect: [10, 26] } },
  { id: "glass", label: "glass ball", s: { scene: "glass", depth: 5, W: 64, H: 48, inspect: [22, 26] } },
  { id: "depth0", label: "depth 0 vs 5", s: { scene: "glass", depth: 0, W: 64, H: 48, inspect: [22, 26] } },
  { id: "floor", label: "Fresnel on a glass floor", s: { scene: "glassfloor", depth: 4, W: 64, H: 48, inspect: [32, 30] } },
];
