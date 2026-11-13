// Ray tracing for the course (owner: ray-tracing day; no DOM). Units raycast/, whitted/,
// pathtrace/ and the demos raystep, pathtracer (CPU reference and WebGL fallback).
//
// Conventions: a ray is r(t) = o + t d with t > 0 in front of the origin; with ‖d‖ = 1, t is a
// distance. Camera looks down −z (OpenGL/WebGL, as in lib/mat4.ts), vertical field of view.
// Normals are unit vectors; a surface's normal is its geometric "outside" (sphere: away from the
// centre; triangle: (b − a) × (c − a)); `facing` flips it towards the incoming ray.
//
//  - intersections: `solveSphere` (stable quadratic, both roots), `solveSphereNaive` (textbook
//    formula, for the cancellation test), `hitPlane`, `solveTriangle` (Möller–Trumbore: t, u, v,
//    det), `hitQuad`, `intersect`, `nearestHit` (with a test counter), `normalAt`;
//  - optics: `reflect`, `refract` (null on total internal reflection), `fresnel` (exact,
//    unpolarised), `fresnelSP`, `schlick`;
//  - `cameraRay(i, j, W, H, cam)`: the ray through the centre of pixel (i, j) (j = row from the top);
//  - `traceWhitted`: an instrumented Whitted tracer that returns the colour and the ray tree;
//  - `tracePath`: a small path tracer (uniform / cosine / cosine + light sampling, Russian
//    roulette, a "Whitted" comparison mode), the reference for the WebGL path tracer, and
//    `Progressive` (running mean and variance per pixel, auxiliary normal/albedo/depth buffers).
import { Vec3, add, sub, scale3, dot, cross, normalize } from "./mat4";
import type { RGB } from "./colour";
import { cosineHemisphere, uniformHemisphere, toWorld } from "./montecarlo";

export interface Ray { o: Vec3; d: Vec3 }
export const pointAt = (o: Vec3, d: Vec3, t: number): Vec3 => [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];

export type MatKind = "diffuse" | "mirror" | "glass" | "light";
export interface Material {
  kind: MatKind;
  /** Albedo (diffuse), tint (mirror, glass). Linear RGB in [0, 1]. */
  colour: RGB;
  /** Refractive index of a glass object (the outside is air, 1). */
  ior?: number;
  /** Emitted radiance of a light (path tracer). */
  emit?: RGB;
  /** Fraction reflected like a mirror by a diffuse surface (Whitted only; e.g. a polished floor). */
  refl?: number;
  /** Checkerboard texture on a plane (Whitted demo). */
  checker?: boolean;
}

export type Shape =
  | { type: "sphere"; c: Vec3; r: number }
  | { type: "plane"; p: Vec3; n: Vec3 }
  | { type: "triangle"; a: Vec3; b: Vec3; c: Vec3; na?: Vec3; nb?: Vec3; nc?: Vec3 }
  /** Parallelogram p + s u + t v, s, t ∈ [0, 1]; normal u × v. */
  | { type: "quad"; p: Vec3; u: Vec3; v: Vec3 };
export type Obj = Shape & { name: string; mat: Material };

export interface Hit { t: number; obj: number; p: Vec3; n: Vec3; u: number; v: number }
export interface Counter { tests: number; rays: number }
export const newCounter = (): Counter => ({ tests: 0, rays: 0 });

// ------------------------------------------------------------------------------------------
// Intersections

/** Both roots of ‖o + t d − c‖² = r² (t1 ≤ t2), NaN when the discriminant is negative.
 *  With b' = d·(o − c), k = ‖o − c‖² − r², a = d·d: t = (−b' ± √(b'² − a k)) / a, computed the
 *  stable way: q = −(b' + sign(b')√disc), t = q/a and t = k/q (Vieta), so no cancellation. */
export function solveSphere(o: Vec3, d: Vec3, c: Vec3, r: number): { b: number; k: number; a: number; disc: number; t1: number; t2: number } {
  const oc = sub(o, c), a = dot(d, d), b = dot(d, oc), k = dot(oc, oc) - r * r;
  const disc = b * b - a * k;
  if (disc < 0) return { a, b, k, disc, t1: NaN, t2: NaN };
  const s = Math.sqrt(disc), q = -(b + (b >= 0 ? s : -s));
  let t1 = q / a, t2 = q !== 0 ? k / q : -b / a;
  if (t1 > t2) [t1, t2] = [t2, t1];
  return { a, b, k, disc, t1, t2 };
}

/** The textbook formula (−b' ± √disc)/a, which loses the small root to cancellation. */
export function solveSphereNaive(o: Vec3, d: Vec3, c: Vec3, r: number, f: (x: number) => number = x => x): [number, number] {
  const oc = sub(o, c), a = f(dot(d, d)), b = f(dot(d, oc)), k = f(f(dot(oc, oc)) - f(r * r));
  const disc = f(f(b * b) - f(a * k));
  if (disc < 0) return [NaN, NaN];
  const s = f(Math.sqrt(disc));
  return [f(f(-b - s) / a), f(f(-b + s) / a)];
}

/** Nearest root in (tmin, tmax) of the sphere, or Infinity. */
export function hitSphere(o: Vec3, d: Vec3, c: Vec3, r: number, tmin = 0, tmax = Infinity): number {
  const s = solveSphere(o, d, c, r);
  if (!(s.disc >= 0)) return Infinity;
  if (s.t1 > tmin && s.t1 < tmax) return s.t1;
  if (s.t2 > tmin && s.t2 < tmax) return s.t2;
  return Infinity;
}

/** t where the ray meets the plane through p with normal n: (p − o)·n / (d·n); Infinity if parallel. */
export function hitPlane(o: Vec3, d: Vec3, p: Vec3, n: Vec3): number {
  const den = dot(d, n);
  if (Math.abs(den) < 1e-15) return Infinity;
  return dot(sub(p, o), n) / den;
}

/** Möller–Trumbore: solve [−d  e1  e2] (t, u, v)ᵀ = o − a by Cramer's rule with triple
 *  products. Returns the raw solution (det ≈ 0: parallel); inside iff u, v ≥ 0 and u + v ≤ 1.
 *  Barycentric coordinates (b_a, b_b, b_c) = (1 − u − v, u, v). */
export function solveTriangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3): { t: number; u: number; v: number; det: number } {
  const e1 = sub(b, a), e2 = sub(c, a), pv = cross(d, e2), det = dot(e1, pv);
  if (Math.abs(det) < 1e-15) return { t: NaN, u: NaN, v: NaN, det };
  const inv = 1 / det, tv = sub(o, a), u = dot(tv, pv) * inv, qv = cross(tv, e1);
  const v = dot(d, qv) * inv, t = dot(e2, qv) * inv;
  return { t, u, v, det };
}

export const insideTriangle = (u: number, v: number): boolean => u >= 0 && v >= 0 && u + v <= 1;

/** Ray against a parallelogram p + s u + t v: (t, s, t') or null. */
export function hitQuad(o: Vec3, d: Vec3, p: Vec3, u: Vec3, v: Vec3): { t: number; s: number; w: number } | null {
  const pv = cross(d, v), det = dot(u, pv);
  if (Math.abs(det) < 1e-15) return null;
  const inv = 1 / det, tv = sub(o, p), s = dot(tv, pv) * inv;
  if (s < 0 || s > 1) return null;
  const qv = cross(tv, u), w = dot(d, qv) * inv;
  if (w < 0 || w > 1) return null;
  return { t: dot(v, qv) * inv, s, w };
}

/** Nearest t in (tmin, tmax) of one object (with its surface parameters), or null. */
export function intersect(ob: Shape, o: Vec3, d: Vec3, tmin = 0, tmax = Infinity): { t: number; u: number; v: number } | null {
  switch (ob.type) {
    case "sphere": { const t = hitSphere(o, d, ob.c, ob.r, tmin, tmax); return t < Infinity ? { t, u: 0, v: 0 } : null; }
    case "plane": { const t = hitPlane(o, d, ob.p, ob.n); return t > tmin && t < tmax ? { t, u: 0, v: 0 } : null; }
    case "triangle": {
      const s = solveTriangle(o, d, ob.a, ob.b, ob.c);
      return s.t > tmin && s.t < tmax && insideTriangle(s.u, s.v) ? { t: s.t, u: s.u, v: s.v } : null;
    }
    case "quad": { const q = hitQuad(o, d, ob.p, ob.u, ob.v); return q && q.t > tmin && q.t < tmax ? { t: q.t, u: q.s, v: q.w } : null; }
  }
}

/** Geometric (outside) unit normal at a hit point; triangles interpolate vertex normals if given. */
export function normalAt(ob: Shape, p: Vec3, u: number, v: number): Vec3 {
  switch (ob.type) {
    case "sphere": return scale3(sub(p, ob.c), 1 / ob.r);
    case "plane": return normalize(ob.n);
    case "triangle":
      if (ob.na && ob.nb && ob.nc) return normalize(add(add(scale3(ob.na, 1 - u - v), scale3(ob.nb, u)), scale3(ob.nc, v)));
      return normalize(cross(sub(ob.b, ob.a), sub(ob.c, ob.a)));
    case "quad": return normalize(cross(ob.u, ob.v));
  }
}

/** The nearest hit over all objects (brute force: every object is tested). */
export function nearestHit(objs: readonly Shape[], o: Vec3, d: Vec3, tmin = 0, tmax = Infinity, cnt?: Counter): Hit | null {
  let best: Hit | null = null, tb = tmax;
  for (let i = 0; i < objs.length; i++) {
    if (cnt) cnt.tests++;
    const h = intersect(objs[i], o, d, tmin, tb);
    if (h) { tb = h.t; best = { t: h.t, obj: i, p: [0, 0, 0], n: [0, 0, 0], u: h.u, v: h.v }; }
  }
  if (best) { best.p = pointAt(o, d, best.t); best.n = normalAt(objs[best.obj], best.p, best.u, best.v); }
  return best;
}

// ------------------------------------------------------------------------------------------
// Optics

/** Mirror reflection r = d − 2(d·n)n (n unit; works for either side). */
export const reflect = (d: Vec3, n: Vec3): Vec3 => { const k = 2 * dot(d, n); return [d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2]]; };

/** Snell's law in vector form. n faces the incoming ray (d·n < 0), eta = n1/n2, d unit.
 *  t = eta d + (eta c1 − √(1 − eta²(1 − c1²))) n with c1 = −d·n; null on total internal reflection. */
export function refract(d: Vec3, n: Vec3, eta: number): Vec3 | null {
  const c1 = -dot(d, n), k = 1 - eta * eta * (1 - c1 * c1);
  if (k < 0) return null;
  const f = eta * c1 - Math.sqrt(k);
  return [eta * d[0] + f * n[0], eta * d[1] + f * n[1], eta * d[2] + f * n[2]];
}

/** Reflectances for s and p polarisation at incidence cosine cosi from medium n1 into n2 (1, 1 on TIR). */
export function fresnelSP(cosi: number, n1: number, n2: number): { rs: number; rp: number; tir: boolean } {
  cosi = Math.min(1, Math.abs(cosi));
  const sint = n1 / n2 * Math.sqrt(Math.max(0, 1 - cosi * cosi));
  if (sint >= 1) return { rs: 1, rp: 1, tir: true };
  const cost = Math.sqrt(1 - sint * sint);
  const rs = ((n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost)) ** 2;
  const rp = ((n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi)) ** 2;
  return { rs, rp, tir: false };
}

/** Exact unpolarised Fresnel reflectance R = (R_s + R_p)/2. */
export const fresnel = (cosi: number, n1: number, n2: number): number => { const f = fresnelSP(cosi, n1, n2); return (f.rs + f.rp) / 2; };

/** Head-on reflectance R0 = ((n1 − n2)/(n1 + n2))². */
export const fresnelR0 = (n1: number, n2: number): number => ((n1 - n2) / (n1 + n2)) ** 2;

/** Schlick's approximation R0 + (1 − R0)(1 − cos θ)⁵, using the angle in the less dense medium. */
export function schlick(cosi: number, n1: number, n2: number): number {
  cosi = Math.min(1, Math.abs(cosi));
  let c = cosi;
  if (n1 > n2) {
    const sint = n1 / n2 * Math.sqrt(Math.max(0, 1 - cosi * cosi));
    if (sint >= 1) return 1;
    c = Math.sqrt(1 - sint * sint);
  }
  const r0 = fresnelR0(n1, n2);
  return r0 + (1 - r0) * (1 - c) ** 5;
}

// ------------------------------------------------------------------------------------------
// Camera

export interface Camera {
  eye: Vec3;
  /** Point looked at (default: straight down −z). */
  target?: Vec3;
  up?: Vec3;
  /** Vertical field of view in degrees. */
  fov: number;
  /** Parallel rays (orthographic); then `height` is the full height of the view in world units. */
  ortho?: boolean;
  height?: number;
}

/** Camera basis: right r, up u, forward f (looking along f = −z of view space). */
export function cameraBasis(cam: Camera): { r: Vec3; u: Vec3; f: Vec3 } {
  const tgt = cam.target ?? [cam.eye[0], cam.eye[1], cam.eye[2] - 1];
  const f = normalize(sub(tgt, cam.eye)), r = normalize(cross(f, cam.up ?? [0, 1, 0])), u = cross(r, f);
  return { r, u, f };
}

/** The ray through the centre (i + ½, j + ½) of pixel (i, j) (j counts rows from the top) of a
 *  W × H image: the point (x_ndc · tan(fov/2) · W/H, y_ndc · tan(fov/2), −1) in view space. */
export function cameraRay(i: number, j: number, W: number, H: number, cam: Camera, jx = 0.5, jy = 0.5): Ray {
  const { r, u, f } = cameraBasis(cam);
  const xn = 2 * (i + jx) / W - 1, yn = 1 - 2 * (j + jy) / H, asp = W / H;
  if (cam.ortho) {
    const h = (cam.height ?? 2) / 2;
    return { o: add(cam.eye, add(scale3(r, xn * h * asp), scale3(u, yn * h))), d: f };
  }
  const th = Math.tan(cam.fov * Math.PI / 360);
  return { o: [...cam.eye] as Vec3, d: normalize(add(f, add(scale3(r, xn * th * asp), scale3(u, yn * th)))) };
}

// ------------------------------------------------------------------------------------------
// Whitted ray tracing, instrumented

export interface PointLight { p: Vec3; colour: RGB }
export interface Scene { objects: Obj[]; lights: PointLight[]; background: RGB; ambient?: number }

export type RayKind = "primary" | "reflect" | "refract" | "shadow";
export interface RayNode {
  kind: RayKind;
  o: Vec3;
  d: Vec3;
  /** Distance to the hit (or to the light for a shadow ray; Infinity for a miss). */
  t: number;
  /** Fraction of the pixel colour this ray carries. */
  weight: number;
  /** Object hit (−1: none; for a shadow ray: the blocker, −1 = light reached). */
  obj: number;
  /** Light index (shadow rays). */
  light?: number;
  depth: number;
  colour: RGB;
  children: RayNode[];
  note?: string;
}

export interface WhittedOptions {
  /** Maximum recursion depth (0 = ray casting: primary rays and shading only). */
  depth: number;
  shadows: boolean;
  reflection: boolean;
  refraction: boolean;
  /** Split glass by the Fresnel reflectance; off: all light is refracted (reflected on TIR). */
  fresnel: boolean;
  /** Offset of secondary-ray origins along the normal. */
  eps: number;
  /** Round hit points to float32, as a GPU stores them (makes shadow acne visible at ε = 0). */
  float32?: boolean;
  /** Rays with a smaller weight are not traced. */
  minWeight?: number;
  /** Keep the ray tree (costs memory; only for the inspected pixel). */
  tree?: boolean;
}

export const WHITTED_DEFAULTS: WhittedOptions = { depth: 3, shadows: true, reflection: true, refraction: true, fresnel: true, eps: 1e-4, float32: true, minWeight: 1e-3 };

const BLACK: RGB = [0, 0, 0];
const mulc = (a: RGB, b: RGB): RGB => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const addc = (a: RGB, b: RGB): RGB => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sclc = (a: RGB, s: number): RGB => [a[0] * s, a[1] * s, a[2] * s];
/** o + t d computed in float32 arithmetic, as a GPU would (each operation rounded). */
const fr = Math.fround;
export const pointAtF32 = (o: Vec3, d: Vec3, t: number): Vec3 => { const tt = fr(t); return [fr(fr(o[0]) + fr(tt * fr(d[0]))), fr(fr(o[1]) + fr(tt * fr(d[1]))), fr(fr(o[2]) + fr(tt * fr(d[2])))]; };

/** Albedo at a point (checkerboard planes alternate between the colour and a darker copy). */
export function albedoAt(ob: Obj, p: Vec3): RGB {
  const m = ob.mat;
  if (m.checker && ob.type === "plane") {
    const k = (Math.floor(p[0]) + Math.floor(p[2])) & 1;
    return k ? m.colour : sclc(m.colour, 0.45);
  }
  return m.colour;
}

/** Lambert shading with shadow rays (ambient + Σ lights ρ I max(0, n·l)). */
function localShade(sc: Scene, ob: Obj, p: Vec3, nf: Vec3, o: WhittedOptions, cnt: Counter, node: RayNode | null, depth: number): RGB {
  const alb = albedoAt(ob, p);
  let c = sclc(alb, sc.ambient ?? 0.08);
  sc.lights.forEach((L, li) => {
    const tl = sub(L.p, p), dist = Math.hypot(tl[0], tl[1], tl[2]), l = scale3(tl, 1 / dist);
    const cos = dot(nf, l);
    if (cos <= 0) return;
    let blocked = -1;
    if (o.shadows) {
      const so = add(p, scale3(nf, o.eps));
      cnt.rays++;
      const h = nearestHit(sc.objects, so, l, 0, dist - o.eps, cnt);
      if (h) blocked = h.obj;
      if (node && o.tree) node.children.push({ kind: "shadow", o: so, d: l, t: h ? h.t : dist, weight: node.weight, obj: blocked, light: li, depth: depth + 1, colour: BLACK, children: [] });
    }
    if (blocked < 0) c = addc(c, sclc(mulc(alb, L.colour), cos));
  });
  return c;
}

/** Trace one ray with Whitted's recursion: local shading + R · reflected + (1 − R) · refracted. */
export function traceWhitted(sc: Scene, o: Vec3, d: Vec3, opt: WhittedOptions, cnt: Counter = newCounter(), kind: RayKind = "primary", depth = 0, weight = 1): { colour: RGB; node: RayNode } {
  cnt.rays++;
  const node: RayNode = { kind, o, d, t: Infinity, weight, obj: -1, depth, colour: BLACK, children: [] };
  const h = nearestHit(sc.objects, o, d, 0, Infinity, cnt);
  if (!h) { node.colour = sc.background; return { colour: sc.background, node }; }
  node.t = h.t; node.obj = h.obj;
  const ob = sc.objects[h.obj], m = ob.mat;
  const p = opt.float32 ? pointAtF32(o, d, h.t) : h.p;
  const front = dot(d, h.n) < 0, nf: Vec3 = front ? h.n : scale3(h.n, -1);
  const canGo = depth < opt.depth, minW = opt.minWeight ?? 1e-3;
  const sub_ = (k: RayKind, dir: Vec3, side: number, w: number): RGB => {
    if (!canGo || weight * w < minW) return BLACK;
    const r = traceWhitted(sc, add(p, scale3(nf, side * opt.eps)), dir, opt, cnt, k, depth + 1, weight * w);
    if (opt.tree) node.children.push(r.node);
    return r.colour;
  };
  let col: RGB;
  const kindEff = (m.kind === "mirror" && !opt.reflection) || (m.kind === "glass" && !opt.refraction && !opt.reflection) ? "diffuse" : m.kind;
  if (kindEff === "light") col = m.emit ?? m.colour;
  else if (kindEff === "diffuse") {
    col = localShade(sc, ob, p, nf, opt, cnt, node, depth);
    const k = opt.reflection ? (m.refl ?? 0) : 0;
    if (k > 0) col = addc(sclc(col, 1 - k), sclc(sub_("reflect", reflect(d, nf), 1, k), k));
  } else if (kindEff === "mirror") {
    col = mulc(m.colour, sub_("reflect", reflect(d, nf), 1, 1));
    if (!canGo) node.note = "depth limit";
  } else {
    // glass
    const n1 = front ? 1 : (m.ior ?? 1.5), n2 = front ? (m.ior ?? 1.5) : 1;
    const cosi = -dot(d, nf);
    const tdir = refract(d, nf, n1 / n2);
    let R = tdir ? (opt.fresnel ? fresnel(cosi, n1, n2) : 0) : 1;
    if (!opt.refraction) R = opt.fresnel ? fresnel(cosi, n1, n2) : 1;
    if (!opt.reflection) R = 0;
    let c: RGB = BLACK;
    if (R > 0) c = addc(c, sclc(sub_("reflect", reflect(d, nf), 1, R), R));
    if (R < 1 && tdir && opt.refraction) c = addc(c, sclc(sub_("refract", tdir, -1, 1 - R), 1 - R));
    if (!tdir) node.note = "total internal reflection";
    col = mulc(m.colour, c);
    if (!canGo) node.note = "depth limit";
  }
  node.colour = col;
  return { colour: col, node };
}

/** Number of rays in a ray tree (shadow rays included). */
export function treeSize(n: RayNode): number { return 1 + n.children.reduce((s, c) => s + treeSize(c), 0); }

/** Render a W × H image with traceWhitted (linear RGB, row-major from the top). */
export function renderWhitted(sc: Scene, cam: Camera, W: number, H: number, opt: WhittedOptions, cnt: Counter = newCounter()): Float32Array {
  const out = new Float32Array(3 * W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const r = cameraRay(i, j, W, H, cam);
    const c = traceWhitted(sc, r.o, r.d, { ...opt, tree: false }, cnt).colour;
    out.set(c, 3 * (j * W + i));
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// Path tracing (CPU reference)

export type Sampling = "uniform" | "cosine" | "light";
export interface PathOptions {
  /** Maximum number of bounces (1 = direct light only). */
  maxBounces: number;
  russianRoulette: boolean;
  sampling: Sampling;
  /** Whitted-style comparison: direct light from the light's centre with a hard shadow, mirror and glass only. */
  whitted?: boolean;
}
export interface PathScene { objects: Obj[]; /** indices of emissive quads, for light sampling */ lights: number[] }

/** Unit vector, random in [0, 1). */
export type U01 = () => number;

/** Direct light from the emissive quads at x (normal n, albedo ρ): one sample per light,
 *  ρ/π · L_e · cosθ_x cosθ_ℓ / r² · A (the solid-angle → area change of variables). */
function directLight(sc: PathScene, x: Vec3, n: Vec3, alb: RGB, rnd: U01, centre: boolean, eps: number): RGB {
  let L: RGB = [0, 0, 0];
  for (const li of sc.lights) {
    const q = sc.objects[li];
    if (q.type !== "quad") continue;
    const s = centre ? 0.5 : rnd(), w = centre ? 0.5 : rnd();
    const y = add(q.p, add(scale3(q.u, s), scale3(q.v, w)));
    const cr = cross(q.u, q.v), A = Math.hypot(cr[0], cr[1], cr[2]), nl = scale3(cr, 1 / A);
    const toL = sub(y, x), d2 = dot(toL, toL), dist = Math.sqrt(d2), wi = scale3(toL, 1 / dist);
    const cx = dot(n, wi), cl = -dot(nl, wi);
    if (cx <= 0 || cl <= 0) continue;
    const h = nearestHit(sc.objects, add(x, scale3(n, eps)), wi, 0, dist - 2 * eps);
    if (h && h.obj !== li) continue;
    const e = q.mat.emit ?? [1, 1, 1], g = cx * cl / d2 * A / Math.PI;
    L = addc(L, sclc(mulc(alb, e), g));
  }
  return L;
}

/** One path sample of the radiance arriving along the ray (o, d). */
export function tracePath(sc: PathScene, o: Vec3, d: Vec3, rnd: U01, opt: PathOptions, eps = 1e-4): RGB {
  let L: RGB = [0, 0, 0], T: RGB = [1, 1, 1], specular = true;
  const nee = opt.sampling === "light" || !!opt.whitted;
  for (let k = 0; ; k++) {
    const h = nearestHit(sc.objects, o, d);
    if (!h) break;
    const ob = sc.objects[h.obj], m = ob.mat;
    const front = dot(d, h.n) < 0, nf: Vec3 = front ? h.n : scale3(h.n, -1);
    if (m.kind === "light") {
      if (front && (specular || !nee)) L = addc(L, mulc(T, m.emit ?? [1, 1, 1]));
      break;
    }
    if (k >= opt.maxBounces) break;
    const p = h.p;
    if (m.kind === "diffuse") {
      if (nee) L = addc(L, mulc(T, directLight(sc, p, nf, m.colour, rnd, !!opt.whitted, eps)));
      if (opt.whitted) break;
      let wi: Vec3;
      if (opt.sampling === "uniform") { wi = toWorld(uniformHemisphere(rnd(), rnd()), nf); const c = dot(wi, nf); T = sclc(mulc(T, m.colour), 2 * c); }
      else { wi = toWorld(cosineHemisphere(rnd(), rnd()), nf); T = mulc(T, m.colour); }
      o = add(p, scale3(nf, eps)); d = wi; specular = false;
    } else if (m.kind === "mirror") {
      d = reflect(d, nf); o = add(p, scale3(nf, eps)); T = mulc(T, m.colour); specular = true;
    } else {
      const n1 = front ? 1 : (m.ior ?? 1.5), n2 = front ? (m.ior ?? 1.5) : 1;
      const tdir = refract(d, nf, n1 / n2), R = tdir ? fresnel(-dot(d, nf), n1, n2) : 1;
      if (rnd() < R) { d = reflect(d, nf); o = add(p, scale3(nf, eps)); }
      else { d = normalize(tdir!); o = add(p, scale3(nf, -eps)); }
      T = mulc(T, m.colour); specular = true;
    }
    if (opt.russianRoulette && k >= 2) {
      const q = Math.min(0.95, Math.max(T[0], T[1], T[2]));
      if (rnd() >= q) break;
      T = sclc(T, 1 / q);
    }
  }
  return L;
}

/** Hash-based random numbers per (pixel, sample, seed): the same stream as the WebGL shader. */
export function pcg(v: number): number {
  let s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}
export function pixelRng(px: number, sample: number, seed: number): U01 {
  let s = pcg((px >>> 0) ^ pcg((sample >>> 0) ^ pcg(seed >>> 0)));
  return () => { s = pcg(s); return s / 4294967296; };
}

/** Progressive rendering on the CPU: running sum and sum of squares (of luminance) per pixel. */
export class Progressive {
  readonly sum: Float64Array;
  readonly sumSq: Float64Array;
  spp = 0;
  constructor(readonly sc: PathScene, readonly cam: Camera, readonly W: number, readonly H: number, public opt: PathOptions, readonly seed = 1) {
    this.sum = new Float64Array(3 * W * H); this.sumSq = new Float64Array(W * H);
  }
  reset(): void { this.sum.fill(0); this.sumSq.fill(0); this.spp = 0; }
  /** Add one sample to every pixel (rows j0..j1 only, for spreading work over frames). */
  addSample(j0 = 0, j1 = this.H): void {
    const { W } = this;
    for (let j = j0; j < j1; j++) for (let i = 0; i < W; i++) {
      const px = j * W + i, rnd = pixelRng(px, this.spp, this.seed);
      const r = cameraRay(i, j, W, this.H, this.cam, rnd(), rnd());
      const c = tracePath(this.sc, r.o, r.d, rnd, this.opt);
      this.sum[3 * px] += c[0]; this.sum[3 * px + 1] += c[1]; this.sum[3 * px + 2] += c[2];
      const y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      this.sumSq[px] += y * y;
    }
    if (j1 >= this.H) this.spp++;
  }
  /** Mean radiance (linear RGB). */
  mean(out = new Float32Array(3 * this.W * this.H)): Float32Array {
    const k = this.spp > 0 ? 1 / this.spp : 1;   // during the first sample: the rows done so far
    for (let i = 0; i < out.length; i++) out[i] = this.sum[i] * k;
    return out;
  }
}

/** First-hit auxiliary buffers (normal as RGB in [0, 1], albedo, depth) through pixel centres. */
export function auxBuffers(sc: PathScene, cam: Camera, W: number, H: number): { normal: Float32Array; albedo: Float32Array; depth: Float32Array } {
  const normal = new Float32Array(3 * W * H), albedo = new Float32Array(3 * W * H), depth = new Float32Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const r = cameraRay(i, j, W, H, cam), h = nearestHit(sc.objects, r.o, r.d), px = j * W + i;
    if (!h) { depth[px] = Infinity; continue; }
    const nf = dot(r.d, h.n) < 0 ? h.n : scale3(h.n, -1);
    normal.set([0.5 + 0.5 * nf[0], 0.5 + 0.5 * nf[1], 0.5 + 0.5 * nf[2]], 3 * px);
    const m = sc.objects[h.obj].mat;
    albedo.set(m.kind === "light" ? [1, 1, 1] : m.colour, 3 * px);
    depth[px] = h.t;
  }
  return { normal, albedo, depth };
}

// ------------------------------------------------------------------------------------------
// The closed box used by the path tracer (shared with lib/glpathtrace.ts)

export interface BoxSceneOptions {
  diffuse?: boolean;
  mirror?: boolean;
  glass?: boolean;
  /** Side of the square ceiling light (the emitted power is kept constant). */
  lightSize?: number;
}

/** A closed box x ∈ [−1, 1], y ∈ [0, 2], z ∈ [−1, 1] with an open front (white floor, ceiling and
 *  back, red left wall, green right wall), a square ceiling light and up to three spheres. */
export function boxScene(o: BoxSceneOptions = {}): PathScene & { camera: Camera } {
  const white: Material = { kind: "diffuse", colour: [0.73, 0.73, 0.73] };
  const red: Material = { kind: "diffuse", colour: [0.65, 0.06, 0.05] };
  const green: Material = { kind: "diffuse", colour: [0.12, 0.45, 0.15] };
  const s = o.lightSize ?? 0.5, P = 12 * 0.25;   // power ∝ L_e · area, fixed: the 0.5 × 0.5 light has L_e = 12
  const Le = P / (s * s);
  const objs: Obj[] = [
    { name: "floor", type: "quad", p: [-1, 0, 1], u: [2, 0, 0], v: [0, 0, -2], mat: white },
    { name: "ceiling", type: "quad", p: [-1, 2, -1], u: [2, 0, 0], v: [0, 0, 2], mat: white },
    { name: "back wall", type: "quad", p: [-1, 0, -1], u: [2, 0, 0], v: [0, 2, 0], mat: white },
    { name: "left wall", type: "quad", p: [-1, 0, 1], u: [0, 0, -2], v: [0, 2, 0], mat: red },
    { name: "right wall", type: "quad", p: [1, 0, -1], u: [0, 0, 2], v: [0, 2, 0], mat: green },
    { name: "light", type: "quad", p: [-s / 2, 1.999, -s / 2], u: [s, 0, 0], v: [0, 0, s], mat: { kind: "light", colour: [1, 1, 1], emit: [Le, Le, Le] } },
  ];
  if (o.diffuse ?? true) objs.push({ name: "diffuse ball", type: "sphere", c: [-0.5, 0.35, -0.35], r: 0.35, mat: { kind: "diffuse", colour: [0.75, 0.75, 0.75] } });
  if (o.mirror ?? true) objs.push({ name: "mirror ball", type: "sphere", c: [0.5, 0.4, -0.45], r: 0.4, mat: { kind: "mirror", colour: [0.95, 0.95, 0.95] } });
  if (o.glass ?? true) objs.push({ name: "glass ball", type: "sphere", c: [0.15, 0.3, 0.45], r: 0.3, mat: { kind: "glass", colour: [1, 1, 1], ior: 1.5 } });
  return { objects: objs, lights: [5], camera: { eye: [0, 1, 2.9], target: [0, 1, 0], fov: 40 } };
}
