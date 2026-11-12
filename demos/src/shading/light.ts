// Local reflection models of the shading demo (no DOM; tested in demos/test/graphics3d-test.ts).
// All vectors are unit vectors pointing AWAY from the surface point: n (normal), l (to the
// light), v (to the eye).
export type V3 = [number, number, number];
export type Model = "lambert" | "phong" | "blinn";

export const dot3 = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export function unit(a: V3): V3 { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

/** Mirror direction of l about n: r = 2 (n·l) n − l. */
export const reflect = (n: V3, l: V3): V3 => { const d = 2 * dot3(n, l); return [d * n[0] - l[0], d * n[1] - l[1], d * n[2] - l[2]]; };

/** Half vector h = (l + v) / |l + v|. */
export const halfVector = (l: V3, v: V3): V3 => unit([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);

export interface Material { ka: number; kd: number; ks: number; shininess: number }

/** Ambient + diffuse (Lambert) and the specular term of the chosen model (0 for Lambert).
 *  No specular where the light is behind the surface. */
export function reflectance(n: V3, l: V3, v: V3, model: Model, m: Material): { diffuse: number; specular: number } {
  const nl = dot3(n, l);
  const diffuse = m.ka + m.kd * Math.max(0, nl);
  if (model === "lambert" || nl <= 0) return { diffuse, specular: 0 };
  const s = model === "phong" ? Math.max(0, dot3(reflect(n, l), v)) : Math.max(0, dot3(n, halfVector(l, v)));
  return { diffuse, specular: m.ks * s ** m.shininess };
}
