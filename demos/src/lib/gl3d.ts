// Minimal WebGL2 renderer for every 3D demo of the course (owner: 3D-graphics day; first
// used by the marching-cubes day). No 3D libraries, no network.
//
//   const view = new GLView(host, { aspect: 4 / 3 });
//   const mesh = view.addMesh(sphereMesh(0.5), { shading: "smooth", lighting: "blinn-phong", wireframe: "overlay" });
//   view.addLines(axesLines(1), { width: 3 });
//   viewButtons(panel, view);                // front / side / top / iso buttons (no keyboard)
//   mesh.set({ shading: "flat" });           // re-renders on demand
//
// What it does
//  - Objects: meshes (flat or smooth normals; unlit, Lambert or Blinn–Phong; wireframe overlay
//    or wireframe only, via barycentric coordinates), points (round sprites, per-point
//    colour), thick lines (screen-space quads; WebGL's own lines are 1 px), a slice plane with
//    a 2D texture, and volumes as 3D textures with maximum-intensity projection, ray-marched
//    compositing (transfer function as a 256-entry RGBA LUT texture) or a first-hit isosurface.
//  - An orbit camera per viewport (M per object, V and P from the camera, all exposed), rotated
//    by pointer drag and by buttons for standard views; several viewports in one canvas (one
//    WebGL context; browsers allow only ~16 live contexts per page).
//  - The context is created only while the view is on screen (IntersectionObserver) and released
//    when it leaves; all data stays on the CPU side and is re-uploaded on return.
//  - devicePixelRatio-aware canvas; pointer positions via getBoundingClientRect, so it works in
//    reveal.js's CSS-scaled slides. No keyboard handling (arrow keys belong to reveal.js).
//  - Without WebGL2 a clear message is shown instead of the canvas.
//
// Conventions: world space right-handed, y up; view space looks down -z (WebGL). Matrices are
// column-major Float32Array (lib/mat4.ts). Volumes are stored (z, y, x): value at voxel
// (x, y, z) is data[(z * ny + y) * nx + x].
import * as M from "./mat4";
import type { Mat4, Vec3 } from "./mat4";
import { ColorMap, grey, colormapLUT, scalarToRGBA, RGB, hexToRgb } from "./colour";
import { htmlEl, button } from "./ui";

// =======================================================================================
// Camera

export type StandardView = "front" | "back" | "left" | "right" | "top" | "bottom" | "iso";

/** Orbit camera around a target: azimuth about +y (0 = looking from +z), elevation above the
 *  x-z plane, distance. Perspective (vertical field of view `fovy`) or orthographic (the
 *  visible height then equals what the perspective view shows at the target). */
export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 3;
  azimuth = 0.6;
  elevation = 0.45;
  fovy = 40 * Math.PI / 180;
  near = 0.01;
  far = 100;
  orthographic = false;

  constructor(init: Partial<Pick<OrbitCamera, "target" | "distance" | "azimuth" | "elevation" | "fovy" | "near" | "far" | "orthographic">> = {}) {
    Object.assign(this, init);
  }

  /** Camera position in world space. */
  eye(): Vec3 {
    const ce = Math.cos(this.elevation);
    return [
      this.target[0] + this.distance * ce * Math.sin(this.azimuth),
      this.target[1] + this.distance * Math.sin(this.elevation),
      this.target[2] + this.distance * ce * Math.cos(this.azimuth),
    ];
  }

  /** View matrix V (world -> view space). */
  view(): Mat4 { return M.lookAt(this.eye(), this.target, [0, 1, 0]); }

  /** Projection matrix P for a viewport aspect ratio (width / height). */
  projection(aspect: number): Mat4 {
    if (this.orthographic) {
      const h = this.distance * Math.tan(this.fovy / 2);
      return M.ortho(-h * aspect, h * aspect, -h, h, this.near, this.far);
    }
    return M.perspective(this.fovy, aspect, this.near, this.far);
  }

  /** Rotate about the target (radians). Elevation is kept just short of ±90°. */
  orbit(dAzimuth: number, dElevation: number): void {
    this.azimuth += dAzimuth;
    const lim = Math.PI / 2 - 1e-3;
    this.elevation = Math.max(-lim, Math.min(lim, this.elevation + dElevation));
  }

  zoom(factor: number): void { this.distance = Math.max(this.near * 2, this.distance * factor); }

  setView(v: StandardView): void {
    const lim = Math.PI / 2 - 1e-3;
    const t: Record<StandardView, [number, number]> = {
      front: [0, 0], back: [Math.PI, 0], right: [Math.PI / 2, 0], left: [-Math.PI / 2, 0],
      top: [0, lim], bottom: [0, -lim], iso: [Math.PI / 4, Math.atan(1 / Math.SQRT2)],
    };
    [this.azimuth, this.elevation] = t[v];
  }

  copyFrom(c: OrbitCamera): void {
    this.target = [...c.target]; this.distance = c.distance; this.azimuth = c.azimuth; this.elevation = c.elevation;
    this.fovy = c.fovy; this.near = c.near; this.far = c.far; this.orthographic = c.orthographic;
  }
}

// =======================================================================================
// Geometry data (CPU side)

type Num = ArrayLike<number>;

export interface MeshData {
  /** x, y, z per vertex. */
  positions: Num;
  /** Three vertex indices per triangle (omit for a plain triangle list). */
  indices?: Num;
  /** Per-vertex normals (default: area-weighted average of the face normals). */
  normals?: Num;
  /** Per-vertex RGB in [0, 1] (used when the style has vertexColors). */
  colors?: Num;
}

export interface PointData {
  positions: Num;
  /** Per-point RGB in [0, 1] (default: the style colour). */
  colors?: Num;
}

export interface LineData {
  /** Segments: x0 y0 z0 x1 y1 z1 per segment. */
  positions: Num;
  /** Per-segment RGB in [0, 1] (default: the style colour). */
  colors?: Num;
}

const triIndices = (d: MeshData): Num => d.indices ?? Uint32Array.from({ length: d.positions.length / 3 }, (_, i) => i);

/** Smooth vertex normals: sum of the (area-weighted) face normals around each vertex. */
export function computeVertexNormals(positions: Num, indices: Num): Float32Array {
  const n = new Float32Array(positions.length);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    const ux = positions[3 * b] - positions[3 * a], uy = positions[3 * b + 1] - positions[3 * a + 1], uz = positions[3 * b + 2] - positions[3 * a + 2];
    const vx = positions[3 * c] - positions[3 * a], vy = positions[3 * c + 1] - positions[3 * a + 1], vz = positions[3 * c + 2] - positions[3 * a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const v of [a, b, c]) { n[3 * v] += nx; n[3 * v + 1] += ny; n[3 * v + 2] += nz; }
  }
  for (let v = 0; v < n.length; v += 3) {
    const l = Math.hypot(n[v], n[v + 1], n[v + 2]) || 1;
    n[v] /= l; n[v + 1] /= l; n[v + 2] /= l;
  }
  return n;
}

/** Axis-aligned box from min to max, 24 vertices (one normal per face), outward CCW winding. */
export function cubeMesh(min: Vec3 = [-0.5, -0.5, -0.5], max: Vec3 = [0.5, 0.5, 0.5]): MeshData {
  const c = (i: number): Vec3 => [i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]];
  const P: number[] = [], N: number[] = [], I: number[] = [];
  for (const [q, n] of CUBE_FACES) {
    const base = P.length / 3;
    for (const v of q) { P.push(...c(v)); N.push(...n); }
    I.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions: Float32Array.from(P), normals: Float32Array.from(N), indices: Uint32Array.from(I) };
}

/** Faces of the unit cube: corner index = x + 2y + 4z; quads CCW seen from outside. */
const CUBE_FACES: [number[], Vec3][] = [
  [[0, 2, 3, 1], [0, 0, -1]], [[4, 5, 7, 6], [0, 0, 1]],
  [[0, 4, 6, 2], [-1, 0, 0]], [[5, 1, 3, 7], [1, 0, 0]],
  [[0, 1, 5, 4], [0, -1, 0]], [[6, 7, 3, 2], [0, 1, 0]],
];

/** UV sphere of radius r with `seg` segments around and seg/2 from pole to pole. */
export function sphereMesh(r = 0.5, seg = 32, centre: Vec3 = [0, 0, 0]): MeshData {
  const P: number[] = [], N: number[] = [], I: number[] = [], rings = Math.max(2, Math.floor(seg / 2));
  for (let j = 0; j <= rings; j++) {
    const th = Math.PI * j / rings;
    for (let i = 0; i <= seg; i++) {
      const ph = 2 * Math.PI * i / seg, n: Vec3 = [Math.sin(th) * Math.sin(ph), Math.cos(th), Math.sin(th) * Math.cos(ph)];
      N.push(...n); P.push(centre[0] + r * n[0], centre[1] + r * n[1], centre[2] + r * n[2]);
    }
  }
  for (let j = 0; j < rings; j++) for (let i = 0; i < seg; i++) {
    const a = j * (seg + 1) + i, b = a + seg + 1;
    if (j > 0) I.push(a, b, a + 1);
    if (j < rings - 1) I.push(a + 1, b, b + 1);
  }
  return { positions: Float32Array.from(P), normals: Float32Array.from(N), indices: Uint32Array.from(I) };
}

/**
 * Height field of a w x h grid (values[row * w + col]) as a mesh in the x-z plane, centred
 * at the origin with the longer side 1; height y = scale * (v - lo) / (hi - lo). Coloured by
 * `map` (vertex colours). Row 0 is at the back (-z), as an image seen from above-front.
 */
export function heightFieldMesh(values: Num, w: number, h: number, o: { scale?: number; lo?: number; hi?: number; map?: ColorMap } = {}): MeshData {
  let lo = o.lo, hi = o.hi;
  if (lo === undefined || hi === undefined) {
    let a = Infinity, b = -Infinity;
    for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) { a = Math.min(a, v); b = Math.max(b, v); } }
    lo ??= a; hi ??= b;
  }
  const s = o.scale ?? 0.3, L = Math.max(w - 1, h - 1) || 1, map = o.map;
  const P = new Float32Array(3 * w * h), C = new Float32Array(3 * w * h), I: number[] = [];
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const p = j * w + i, t = hi! > lo! ? (values[p] - lo!) / (hi! - lo!) : 0;
    P[3 * p] = (i - (w - 1) / 2) / L; P[3 * p + 1] = s * (Number.isFinite(t) ? t : 0); P[3 * p + 2] = (j - (h - 1) / 2) / L;
    const c = map ? map(t) : [0.75, 0.75, 0.78];
    C[3 * p] = c[0]; C[3 * p + 1] = c[1]; C[3 * p + 2] = c[2];
  }
  for (let j = 0; j + 1 < h; j++) for (let i = 0; i + 1 < w; i++) {
    const a = j * w + i;
    I.push(a, a + w, a + 1, a + 1, a + w, a + w + 1);
  }
  const indices = Uint32Array.from(I);
  return { positions: P, colors: C, indices, normals: computeVertexNormals(P, indices) };
}

/** x (red), y (green), z (blue) axes from the origin, as line segments. */
export function axesLines(len = 1, origin: Vec3 = [0, 0, 0]): LineData {
  const [x, y, z] = origin;
  return {
    positions: [x, y, z, x + len, y, z, x, y, z, x, y + len, z, x, y, z, x, y, z + len],
    colors: [0.84, 0.15, 0.16, 0.17, 0.63, 0.17, 0.12, 0.47, 0.71],
  };
}

/** The 12 edges of an axis-aligned box. */
export function boxLines(min: Vec3 = [-0.5, -0.5, -0.5], max: Vec3 = [0.5, 0.5, 0.5]): LineData {
  const c = (i: number): Vec3 => [i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]];
  const P: number[] = [];
  for (let i = 0; i < 8; i++) for (const b of [1, 2, 4]) if (!(i & b)) P.push(...c(i), ...c(i | b));
  return { positions: P };
}

// =======================================================================================
// Scene objects

/** Fields shared by every object. */
abstract class GLObject {
  /** Model matrix M (object -> world). */
  model: Mat4 = M.identity();
  visible = true;
  /** Indices of the viewports that show this object (default: all). */
  viewports: number[] | null = null;
  /** @internal bumped when the data changes (re-upload). */
  version = 0;
  /** @internal GPU resources, valid for context generation `gen`. */
  gpu: { gen: number; version: number; res: any } | null = null;
  constructor(protected view: GLView) {}
  /** Set the model matrix. */
  setModel(m: Mat4): void { this.model = m; this.view.render(); }
  setVisible(v: boolean): void { this.visible = v; this.view.render(); }
  remove(): void { this.view.remove(this); }
}

export type Shading = "flat" | "smooth";
export type Lighting = "unlit" | "lambert" | "blinn-phong";

export interface MeshStyle {
  color: RGB;
  /** Use the per-vertex colours of the data. */
  vertexColors: boolean;
  shading: Shading;
  lighting: Lighting;
  /** "overlay": edges drawn over the shaded surface; "only": edges alone. */
  wireframe: "off" | "overlay" | "only";
  wireColor: RGB;
  /** Edge width in (roughly) pixels. */
  wireWidth: number;
  opacity: number;
  shininess: number;
  ambient: number;
}

export class MeshObject extends GLObject {
  data: MeshData;
  style: MeshStyle;
  constructor(view: GLView, data: MeshData, style: Partial<MeshStyle>) {
    super(view);
    this.data = data;
    this.style = { color: [0.55, 0.7, 0.88], vertexColors: !!data.colors && !style.color, shading: "smooth", lighting: "blinn-phong", wireframe: "off", wireColor: [0.1, 0.1, 0.15], wireWidth: 1.2, opacity: 1, shininess: 40, ambient: 0.25, ...style };
  }
  setData(d: MeshData): void { this.data = d; this.version++; this.view.render(); }
  set(s: Partial<MeshStyle>): void { Object.assign(this.style, s); this.view.render(); }
}

export interface PointStyle {
  color: RGB;
  /** Diameter in CSS pixels. */
  size: number;
  /** Light sphere-like shading of each dot. */
  shaded: boolean;
}

export class PointsObject extends GLObject {
  data: PointData;
  style: PointStyle;
  constructor(view: GLView, data: PointData, style: Partial<PointStyle>) {
    super(view);
    this.data = data;
    this.style = { color: [0.12, 0.47, 0.71], size: 6, shaded: true, ...style };
  }
  setData(d: PointData): void { this.data = d; this.version++; this.view.render(); }
  set(s: Partial<PointStyle>): void { Object.assign(this.style, s); if (s.color && !this.data.colors) this.version++; this.view.render(); }
}

export interface LineStyle {
  color: RGB;
  /** Width in CSS pixels. */
  width: number;
}

export class LinesObject extends GLObject {
  data: LineData;
  style: LineStyle;
  constructor(view: GLView, data: LineData, style: Partial<LineStyle>) {
    super(view);
    this.data = data;
    this.style = { color: [0.2, 0.2, 0.2], width: 2, ...style };
  }
  setData(d: LineData): void { this.data = d; this.version++; this.view.render(); }
  set(s: Partial<LineStyle>): void { Object.assign(this.style, s); if (s.color && !this.data.colors) this.version++; this.view.render(); }
}

/** A 2D image for a slice: scalar values through a colour map, or RGBA bytes. */
export type SliceImage =
  | { values: Num; w: number; h: number; lo?: number; hi?: number; map?: ColorMap }
  | { rgba: Uint8ClampedArray | Uint8Array; w: number; h: number };

export interface SliceStyle {
  /** Corners in world space: bottom-left, bottom-right, top-right, top-left of the image
   *  (image row 0 is the top edge). */
  corners: [Vec3, Vec3, Vec3, Vec3];
  filter: "nearest" | "linear";
  opacity: number;
}

export class SliceObject extends GLObject {
  image: SliceImage;
  style: SliceStyle;
  constructor(view: GLView, image: SliceImage, style: Partial<SliceStyle>) {
    super(view);
    this.image = image;
    this.style = { corners: [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], filter: "nearest", opacity: 1, ...style };
  }
  setImage(img: SliceImage): void { this.image = img; this.version++; this.view.render(); }
  /** Move the plane (and change filter/opacity) without re-uploading the texture. */
  set(s: Partial<SliceStyle>): void { Object.assign(this.style, s); this.view.render(); }
}

export type VolumeMode = "mip" | "dvr" | "iso";

export interface VolumeData {
  /** Values stored (z, y, x): data[(z * ny + y) * nx + x]. */
  data: Num;
  nx: number;
  ny: number;
  nz: number;
  /** Value range mapped to [0, 1] (default: the data range). */
  lo?: number;
  hi?: number;
  /** Voxel spacing (Δx, Δy, Δz); sets the box proportions (default 1, 1, 1). */
  spacing?: Vec3;
}

export interface VolumeStyle {
  mode: VolumeMode;
  /** Transfer function: a colour map (alpha = the map's lightness ramp t) or a 256-entry RGBA LUT. */
  transfer: ColorMap | Uint8ClampedArray;
  /** Samples along the box diagonal (default 256). */
  steps: number;
  /** Isosurface level in normalised units [0, 1] (mode "iso"). */
  iso: number;
  /** Multiplies the transfer-function opacity (mode "dvr"). */
  opacity: number;
  /** Colour of the isosurface (mode "iso"; default from the transfer function at `iso`). */
  isoColor: RGB | null;
  /** Show image row 0 at the top (+y up flips the data's y). Default true. */
  flipY: boolean;
}

export class VolumeObject extends GLObject {
  vol: VolumeData;
  style: VolumeStyle;
  /** @internal */ tfVersion = 0;
  constructor(view: GLView, vol: VolumeData, style: Partial<VolumeStyle>) {
    super(view);
    this.vol = vol;
    this.style = { mode: "mip", transfer: grey, steps: 256, iso: 0.5, opacity: 1, isoColor: null, flipY: true, ...style };
    this.model = VolumeObject.boxModel(vol);
  }
  /** Model matrix that maps the unit cube to a box centred at the origin with the volume's
   *  physical proportions and longest side 1. */
  static boxModel(v: VolumeData): Mat4 {
    const sp = v.spacing ?? [1, 1, 1], s: Vec3 = [v.nx * sp[0], v.ny * sp[1], v.nz * sp[2]], m = Math.max(...s);
    return M.multiply(M.scaling(s[0] / m, s[1] / m, s[2] / m), M.translation(-0.5, -0.5, -0.5));
  }
  setVolume(v: VolumeData, keepModel = false): void { this.vol = v; if (!keepModel) this.model = VolumeObject.boxModel(v); this.version++; this.view.render(); }
  set(s: Partial<VolumeStyle>): void { if (s.transfer) this.tfVersion++; Object.assign(this.style, s); this.view.render(); }
}

/** A transfer function LUT: colours from `map`, opacity rising from 0 at `start` to `maxAlpha` at 1. */
export function transferLUT(map: ColorMap, start = 0.2, maxAlpha = 0.8, gamma = 1.5): Uint8ClampedArray {
  const lut = colormapLUT(map, 256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255, a = t <= start ? 0 : maxAlpha * ((t - start) / (1 - start)) ** gamma;
    lut[4 * i + 3] = Math.round(255 * a);
  }
  return lut;
}

// =======================================================================================
// Shaders

const MESH_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in vec3 aBary;
uniform mat4 uModel, uView, uProj;
uniform mat3 uNormalMat;
out vec3 vPos; out vec3 vNormal; out vec3 vColor; out vec3 vBary;
void main() {
  vec4 p = uView * uModel * vec4(aPos, 1.0);
  vPos = p.xyz; vNormal = uNormalMat * aNormal; vColor = aColor; vBary = aBary;
  gl_Position = uProj * p;
}`;

const MESH_FS = `#version 300 es
precision highp float;
in vec3 vPos; in vec3 vNormal; in vec3 vColor; in vec3 vBary;
uniform int uFlat, uLight, uWire, uVColor;
uniform vec3 uColor, uWireColor, uLightDir;
uniform float uOpacity, uShininess, uAmbient, uWireWidth;
out vec4 o;
void main() {
  vec3 base = uVColor == 1 ? vColor : uColor;
  vec3 V = normalize(-vPos);
  vec3 n = uFlat == 1 ? normalize(cross(dFdx(vPos), dFdy(vPos))) : normalize(vNormal);
  if (dot(n, V) < 0.0) n = -n;               // two-sided lighting
  vec3 col = base;
  if (uLight > 0) {
    vec3 L = normalize(uLightDir);
    float d = max(dot(n, L), 0.0);
    col = base * (uAmbient + (1.0 - uAmbient) * d);
    if (uLight == 2 && d > 0.0) col += vec3(0.35) * pow(max(dot(n, normalize(L + V)), 0.0), uShininess);
  }
  float a = uOpacity;
  if (uWire > 0) {
    vec3 e = smoothstep(vec3(0.0), fwidth(vBary) * uWireWidth, vBary);
    float edge = 1.0 - min(min(e.x, e.y), e.z);
    if (uWire == 2) { if (edge < 0.02) discard; col = uWireColor; a = edge; }
    else col = mix(col, uWireColor, edge);
  }
  o = vec4(col * a, a);
}`;

const POINTS_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uModel, uView, uProj;
uniform float uSize;
out vec3 vColor;
void main() { gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); gl_PointSize = uSize; vColor = aColor; }`;

const POINTS_FS = `#version 300 es
precision highp float;
in vec3 vColor;
uniform int uShade;
out vec4 o;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  vec3 col = uShade == 1 ? vColor * (0.55 + 0.45 * sqrt(1.0 - r2)) : vColor;
  o = vec4(col, 1.0);
}`;

const LINES_VS = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 aA;
layout(location=2) in vec3 aB;
layout(location=3) in vec3 aColor;
uniform mat4 uModel, uView, uProj;
uniform vec2 uViewport;
uniform float uWidth;
out vec3 vColor;
void main() {
  mat4 mvp = uProj * uView * uModel;
  vec4 ca = mvp * vec4(aA, 1.0), cb = mvp * vec4(aB, 1.0);
  vec2 sa = ca.xy / ca.w * uViewport * 0.5, sb = cb.xy / cb.w * uViewport * 0.5;
  vec2 d = sb - sa; float L = length(d);
  d = L > 1e-6 ? d / L : vec2(1.0, 0.0);
  vec2 nrm = vec2(-d.y, d.x);
  vec4 c = aCorner.x < 0.5 ? ca : cb;
  vec2 off = (nrm * aCorner.y + d * (aCorner.x * 2.0 - 1.0)) * uWidth * 0.5;
  c.xy += off / (uViewport * 0.5) * c.w;
  gl_Position = c; vColor = aColor;
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 o;
void main() { o = vec4(vColor, 1.0); }`;

const SLICE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uModel, uView, uProj;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }`;

const SLICE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform float uOpacity;
out vec4 o;
void main() { vec4 c = texture(uTex, vUV); float a = c.a * uOpacity; o = vec4(c.rgb * a, a); }`;

const VOL_VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uModel, uView, uProj;
out vec3 vLocal;
void main() { vLocal = aPos; gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }`;

const VOL_FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vLocal;
uniform sampler3D uVol;
uniform sampler2D uTF;
uniform vec3 uEye, uDir, uTexel, uIsoColor;
uniform int uOrtho, uMode, uSteps, uFlipY, uUseIsoColor;
uniform float uIso, uOpacity;
out vec4 o;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float sampleV(vec3 p) { return texture(uVol, uFlipY == 1 ? vec3(p.x, 1.0 - p.y, p.z) : p).r; }
void main() {
  vec3 dir = normalize(uOrtho == 1 ? uDir : vLocal - uEye);
  vec3 org = uOrtho == 1 ? vLocal - dir * 4.0 : uEye;
  vec3 dd = mix(dir, vec3(1e-6), lessThan(abs(dir), vec3(1e-6)));
  vec3 t0 = -org / dd, t1 = (1.0 - org) / dd;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float tn = max(max(max(tmn.x, tmn.y), tmn.z), 0.0), tf = min(min(tmx.x, tmx.y), tmx.z);
  if (tf <= tn) discard;
  float dt = 1.7320508 / float(uSteps);
  float t = tn + dt * hash(gl_FragCoord.xy);
  if (uMode == 0) {                       // maximum intensity projection
    float m = 0.0;
    for (int i = 0; i < 4096; i++) { if (t > tf) break; m = max(m, sampleV(org + dir * t)); t += dt; }
    o = vec4(texture(uTF, vec2(m, 0.5)).rgb, 1.0);
  } else if (uMode == 1) {                // front-to-back compositing
    vec3 C = vec3(0.0); float A = 0.0;
    float k = 256.0 / float(uSteps);      // opacity correction to the reference step
    for (int i = 0; i < 4096; i++) {
      if (t > tf || A > 0.995) break;
      vec4 s = texture(uTF, vec2(sampleV(org + dir * t), 0.5));
      float a = 1.0 - pow(1.0 - clamp(s.a * uOpacity, 0.0, 0.999), k);
      C += (1.0 - A) * a * s.rgb; A += (1.0 - A) * a;
      t += dt;
    }
    o = vec4(C, A);
  } else {                                // first hit of the isosurface
    float prev = sampleV(org + dir * t);
    for (int i = 0; i < 4096; i++) {
      if (t > tf) discard;
      float v = sampleV(org + dir * t);
      if (v >= uIso) {
        float a = t - dt, b = t;          // refine by bisection
        for (int j = 0; j < 6; j++) { float m = 0.5 * (a + b); if (sampleV(org + dir * m) >= uIso) b = m; else a = m; }
        vec3 p = org + dir * b;
        vec3 g = vec3(sampleV(p + vec3(uTexel.x, 0, 0)) - sampleV(p - vec3(uTexel.x, 0, 0)),
                      sampleV(p + vec3(0, uTexel.y, 0)) - sampleV(p - vec3(0, uTexel.y, 0)),
                      sampleV(p + vec3(0, 0, uTexel.z)) - sampleV(p - vec3(0, 0, uTexel.z)));
        vec3 n = length(g) > 0.0 ? -normalize(g) : -dir;
        float d = abs(dot(n, -dir));
        vec3 base = uUseIsoColor == 1 ? uIsoColor : texture(uTF, vec2(uIso, 0.5)).rgb;
        vec3 h = normalize(-dir + normalize(-dir + vec3(0.3, 0.5, 0.0)));
        o = vec4(base * (0.25 + 0.75 * d) + vec3(0.25) * pow(max(dot(n, h), 0.0), 30.0), 1.0);
        return;
      }
      prev = v;
      t += dt;
    }
    discard;
  }
}`;

// =======================================================================================
// The view

export interface Viewport {
  /** Rectangle as fractions of the canvas, origin top-left. */
  x: number; y: number; w: number; h: number;
  camera: OrbitCamera;
  /** Clear colour (default: the view's background). */
  background?: RGB;
  /** Caption drawn in the top-left corner (HTML, ≥ 18 px on slides). */
  label?: string;
  /** Rotate the camera by dragging in this viewport (default true). */
  interactive?: boolean;
}

export interface GLViewOptions {
  /** Width / height of the canvas (default 4/3). */
  aspect?: number;
  /** Viewports (default: one full-size viewport with a new OrbitCamera). */
  viewports?: Viewport[];
  /** Default clear colour (default: the demo panel colour). */
  background?: RGB;
  /** Create the context only while on screen (default true). */
  lazy?: boolean;
  /** Extra sentence for the no-WebGL2 message (e.g. "The static figure above shows ..."). */
  fallback?: string;
  /** Mouse-wheel zoom (default false: the wheel scrolls the page). */
  wheelZoom?: boolean;
  /** Called after every camera change by the user. */
  onCamera?: (vp: number) => void;
  /** Called after every rendered frame. */
  onRender?: () => void;
  ariaLabel?: string;
  /** Light direction in view space (default from the upper left, towards the scene). */
  lightDir?: Vec3;
}

interface Programs {
  mesh: WebGLProgram; points: WebGLProgram; lines: WebGLProgram; slice: WebGLProgram; vol: WebGLProgram;
  quadCorner: WebGLBuffer; cube: { vao: WebGLVertexArrayObject; n: number };
  loc: Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>;
}

/** Is WebGL2 available in this browser? (Creates and immediately releases a context.) */
export function webgl2Available(): boolean {
  try {
    const c = document.createElement("canvas"), gl = c.getContext("webgl2");
    if (!gl) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch { return false; }
}

let contextGeneration = 0;

/** A canvas with one or more viewports and a list of objects, rendered on demand. */
export class GLView {
  readonly root: HTMLDivElement;
  canvas: HTMLCanvasElement | null = null;
  gl: WebGL2RenderingContext | null = null;
  viewports: Viewport[];
  readonly objects: GLObject[] = [];
  readonly opts: GLViewOptions;
  /** Set when WebGL2 is missing or failed (the message is shown). */
  error: string | null = null;
  private progs: Programs | null = null;
  private gen = 0;
  private frame = 0;
  private labels: HTMLDivElement[] = [];
  private msg: HTMLDivElement | null = null;
  private io: IntersectionObserver | null = null;
  private ro: ResizeObserver | null = null;
  private offTimer = 0;
  private destroyed = false;

  constructor(parent: HTMLElement, opts: GLViewOptions = {}) {
    this.opts = opts;
    this.root = htmlEl("div", "qb-gl", "", parent);
    this.root.style.aspectRatio = String(opts.aspect ?? 4 / 3);
    this.root.setAttribute("role", "img");
    this.root.setAttribute("aria-label", opts.ariaLabel ?? "3D view; drag to rotate");
    this.viewports = opts.viewports ?? [{ x: 0, y: 0, w: 1, h: 1, camera: new OrbitCamera() }];
    this.installPointer();
    this.syncLabels();
    if (opts.lazy === false || typeof IntersectionObserver === "undefined") this.activate();
    else {
      this.io = new IntersectionObserver(es => {
        const on = es.some(e => e.isIntersecting);
        clearTimeout(this.offTimer);
        if (on) this.activate();
        else this.offTimer = window.setTimeout(() => this.deactivate(), 400);
      }, { rootMargin: "100px" });
      this.io.observe(this.root);
    }
  }

  /** The first viewport's camera. */
  get camera(): OrbitCamera { return this.viewports[0].camera; }

  /** Is a WebGL context live right now? */
  get active(): boolean { return !!this.gl; }

  // ---- objects ----
  addMesh(d: MeshData, s: Partial<MeshStyle> = {}): MeshObject { return this.add(new MeshObject(this, d, s)); }
  addPoints(d: PointData, s: Partial<PointStyle> = {}): PointsObject { return this.add(new PointsObject(this, d, s)); }
  addLines(d: LineData, s: Partial<LineStyle> = {}): LinesObject { return this.add(new LinesObject(this, d, s)); }
  addSlice(img: SliceImage, s: Partial<SliceStyle> = {}): SliceObject { return this.add(new SliceObject(this, img, s)); }
  addVolume(v: VolumeData, s: Partial<VolumeStyle> = {}): VolumeObject { return this.add(new VolumeObject(this, v, s)); }

  private add<T extends GLObject>(o: T): T { this.objects.push(o); this.render(); return o; }

  remove(o: GLObject): void {
    const i = this.objects.indexOf(o);
    if (i < 0) return;
    this.objects.splice(i, 1);
    if (this.gl && o.gpu?.gen === this.gen) this.freeRes(o.gpu.res);
    o.gpu = null;
    this.render();
  }

  /** Replace the viewports (e.g. switch between one and two views). */
  setViewports(v: Viewport[]): void { this.viewports = v; this.syncLabels(); this.render(); }

  /** Request a redraw on the next animation frame (cheap to call often). */
  render(): void {
    if (!this.gl || this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }

  /** World-space ray under a pointer event: viewport index, origin and unit direction (or null). */
  pointerRay(ev: { clientX: number; clientY: number }): { vp: number; origin: Vec3; dir: Vec3; ndc: [number, number] } | null {
    const hit = this.viewportAt(ev);
    if (!hit) return null;
    const vp = this.viewports[hit.i], r = this.root.getBoundingClientRect();
    const inv = M.invert(M.multiply(vp.camera.projection(vp.w * r.width / (vp.h * r.height)), vp.camera.view()));
    if (!inv) return null;
    const a = M.transformPoint(inv, [hit.nx, hit.ny, -1]), b = M.transformPoint(inv, [hit.nx, hit.ny, 1]);
    return { vp: hit.i, origin: a, dir: M.normalize(M.sub(b, a)), ndc: [hit.nx, hit.ny] };
  }

  /** Stop observing and release the context. */
  destroy(): void { this.destroyed = true; this.io?.disconnect(); this.deactivate(); }

  // ---- lifecycle ----
  private activate(): void {
    if (this.gl || this.destroyed || this.error) return;
    const cv = document.createElement("canvas");
    cv.className = "qb-gl-canvas";
    const gl = cv.getContext("webgl2", { antialias: true, premultipliedAlpha: true, preserveDrawingBuffer: true });
    if (!gl) { this.fail("This 3D view needs WebGL2, which this browser (or its current settings, e.g. hardware acceleration switched off) does not provide."); return; }
    try { this.progs = this.compileAll(gl); }
    catch (e) { gl.getExtension("WEBGL_lose_context")?.loseContext(); this.fail("WebGL2 is available, but the shaders did not compile: " + String(e)); return; }
    this.root.prepend(cv);
    this.canvas = cv; this.gl = gl; this.gen = ++contextGeneration;
    cv.addEventListener("webglcontextlost", ev => { ev.preventDefault(); this.deactivate(); });
    if (typeof ResizeObserver !== "undefined") { this.ro = new ResizeObserver(() => this.render()); this.ro.observe(this.root); }
    this.render();
  }

  private deactivate(): void {
    if (this.frame) { cancelAnimationFrame(this.frame); this.frame = 0; }
    this.ro?.disconnect(); this.ro = null;
    const gl = this.gl;
    this.gl = null; this.progs = null;
    for (const o of this.objects) o.gpu = null;
    if (gl) gl.getExtension("WEBGL_lose_context")?.loseContext();
    this.canvas?.remove(); this.canvas = null;
  }

  private fail(text: string): void {
    this.error = text;
    this.msg?.remove();
    this.msg = htmlEl("div", "qb-gl-msg", text + (this.opts.fallback ? " " + this.opts.fallback : ""), this.root);
    this.root.style.aspectRatio = "auto";
  }

  private syncLabels(): void {
    this.labels.forEach(l => l.remove());
    this.labels = this.viewports.filter(v => v.label).map(v => {
      const d = htmlEl("div", "qb-gl-label", v.label!, this.root);
      d.style.left = `calc(${(v.x * 100).toFixed(3)}% + 6px)`;
      d.style.top = `calc(${(v.y * 100).toFixed(3)}% + 4px)`;
      return d;
    });
  }

  // ---- pointer ----
  private viewportAt(ev: { clientX: number; clientY: number }): { i: number; nx: number; ny: number } | null {
    const r = this.root.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
    for (let i = this.viewports.length - 1; i >= 0; i--) {
      const v = this.viewports[i];
      if (fx >= v.x && fx <= v.x + v.w && fy >= v.y && fy <= v.y + v.h) return { i, nx: 2 * (fx - v.x) / v.w - 1, ny: 1 - 2 * (fy - v.y) / v.h };
    }
    return null;
  }

  private installPointer(): void {
    const el = this.root;
    el.style.touchAction = "none";
    let drag: { i: number; x: number; y: number } | null = null;
    el.addEventListener("pointerdown", ev => {
      const hit = this.viewportAt(ev);
      if (!hit || this.viewports[hit.i].interactive === false) return;
      ev.preventDefault();
      el.setPointerCapture(ev.pointerId);
      drag = { i: hit.i, x: ev.clientX, y: ev.clientY };
    });
    el.addEventListener("pointermove", ev => {
      if (!drag) return;
      const r = el.getBoundingClientRect(), v = this.viewports[drag.i];
      const dx = (ev.clientX - drag.x) / (r.width * v.w), dy = (ev.clientY - drag.y) / (r.height * v.h);
      drag.x = ev.clientX; drag.y = ev.clientY;
      v.camera.orbit(-dx * Math.PI * 1.2, dy * Math.PI);
      this.opts.onCamera?.(drag.i);
      this.render();
    });
    const up = () => { drag = null; };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    if (this.opts.wheelZoom) el.addEventListener("wheel", ev => {
      const hit = this.viewportAt(ev);
      if (!hit) return;
      ev.preventDefault();
      this.viewports[hit.i].camera.zoom(Math.exp(ev.deltaY * 0.001));
      this.opts.onCamera?.(hit.i);
      this.render();
    }, { passive: false });
  }

  // ---- GL setup ----
  private compileAll(gl: WebGL2RenderingContext): Programs {
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
      return s;
    };
    const prog = (vs: string, fs: string) => {
      const p = gl.createProgram()!;
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
      return p;
    };
    const quadCorner = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quadCorner);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 0, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    // unit cube (back faces are drawn for ray marching)
    const cvao = gl.createVertexArray()!;
    gl.bindVertexArray(cvao);
    const P: number[] = [];
    for (const [q] of CUBE_FACES) for (const k of [0, 1, 2, 0, 2, 3]) { const i = q[k]; P.push(i & 1, (i >> 1) & 1, (i >> 2) & 1); }
    const cb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return {
      mesh: prog(MESH_VS, MESH_FS), points: prog(POINTS_VS, POINTS_FS), lines: prog(LINES_VS, FLAT_FS),
      slice: prog(SLICE_VS, SLICE_FS), vol: prog(VOL_VS, VOL_FS),
      quadCorner, cube: { vao: cvao, n: P.length / 3 }, loc: new Map(),
    };
  }

  private u(p: WebGLProgram, name: string): WebGLUniformLocation | null {
    const P = this.progs!;
    let m = P.loc.get(p);
    if (!m) { m = new Map(); P.loc.set(p, m); }
    if (!m.has(name)) m.set(name, this.gl!.getUniformLocation(p, name));
    return m.get(name)!;
  }

  private freeRes(res: any): void {
    const gl = this.gl!;
    for (const v of Object.values(res ?? {})) {
      if (v instanceof WebGLBuffer) gl.deleteBuffer(v);
      else if (v instanceof WebGLTexture) gl.deleteTexture(v);
      else if (v instanceof WebGLVertexArrayObject) gl.deleteVertexArray(v);
    }
  }

  private buffer(data: Float32Array): WebGLBuffer {
    const gl = this.gl!, b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  private attrib(loc: number, buf: WebGLBuffer, size: number, stride = 0, offset = 0, divisor = 0): void {
    const gl = this.gl!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    gl.vertexAttribDivisor(loc, divisor);
  }

  /** Upload (or re-upload) an object's data if needed; returns its GPU resources. */
  private resources(o: GLObject): any {
    if (o.gpu && o.gpu.gen === this.gen && o.gpu.version === o.version) return o.gpu.res;
    if (o.gpu && o.gpu.gen === this.gen) this.freeRes(o.gpu.res);
    const gl = this.gl!;
    let res: any;
    if (o instanceof MeshObject) {
      // de-index: three vertices per triangle, each with barycentric coordinates (wireframe)
      const d = o.data, idx = triIndices(d), nt = Math.floor(idx.length / 3);
      const nrm = d.normals ?? computeVertexNormals(d.positions, idx);
      const P = new Float32Array(9 * nt), N = new Float32Array(9 * nt), C = new Float32Array(9 * nt), B = new Float32Array(9 * nt);
      for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
        const v = idx[3 * t + k], q = 9 * t + 3 * k;
        for (let c = 0; c < 3; c++) {
          P[q + c] = d.positions[3 * v + c]; N[q + c] = nrm[3 * v + c];
          C[q + c] = d.colors ? d.colors[3 * v + c] : 1;
        }
        B[q + k] = 1;
      }
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const bp = this.buffer(P), bn = this.buffer(N), bc = this.buffer(C), bb = this.buffer(B);
      this.attrib(0, bp, 3); this.attrib(1, bn, 3); this.attrib(2, bc, 3); this.attrib(3, bb, 3);
      gl.bindVertexArray(null);
      res = { vao, bp, bn, bc, bb, count: 3 * nt };
    } else if (o instanceof PointsObject) {
      const n = Math.floor(o.data.positions.length / 3);
      const C = o.data.colors ? Float32Array.from(o.data.colors) : Float32Array.from({ length: 3 * n }, (_, i) => o.style.color[i % 3]);
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const bp = this.buffer(Float32Array.from(o.data.positions)), bc = this.buffer(C);
      this.attrib(0, bp, 3); this.attrib(1, bc, 3);
      gl.bindVertexArray(null);
      res = { vao, bp, bc, count: n };
    } else if (o instanceof LinesObject) {
      const n = Math.floor(o.data.positions.length / 6), S = new Float32Array(9 * n);
      for (let s = 0; s < n; s++) {
        for (let c = 0; c < 6; c++) S[9 * s + c] = o.data.positions[6 * s + c];
        for (let c = 0; c < 3; c++) S[9 * s + 6 + c] = o.data.colors ? o.data.colors[3 * s + c] : o.style.color[c];
      }
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      this.attrib(0, this.progs!.quadCorner, 2);
      const bs = this.buffer(S);
      this.attrib(1, bs, 3, 36, 0, 1); this.attrib(2, bs, 3, 36, 12, 1); this.attrib(3, bs, 3, 36, 24, 1);
      gl.bindVertexArray(null);
      res = { vao, bs, count: n };
    } else if (o instanceof SliceObject) {
      const img = o.image, tex = gl.createTexture()!;
      const rgba = "rgba" in img ? Uint8Array.from(img.rgba) : new Uint8Array(scalarToRGBA(img.values, { map: img.map ?? grey, lo: img.lo, hi: img.hi }).buffer);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, img.w, img.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const vao = gl.createVertexArray()!, bp = gl.createBuffer()!, bu = this.buffer(new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]));
      gl.bindVertexArray(vao);
      this.attrib(0, bp, 3); this.attrib(1, bu, 2);
      gl.bindVertexArray(null);
      res = { vao, bp, bu, tex };
    } else if (o instanceof VolumeObject) {
      const v = o.vol, n = v.nx * v.ny * v.nz;
      let lo = v.lo, hi = v.hi;
      if (lo === undefined || hi === undefined) {
        let a = Infinity, b = -Infinity;
        for (let i = 0; i < n; i++) { const x = v.data[i]; if (x < a) a = x; if (x > b) b = x; }
        lo ??= a; hi ??= b;
      }
      const k = hi > lo ? 255 / (hi - lo) : 0, bytes = new Uint8Array(n);
      for (let i = 0; i < n; i++) { const q = (v.data[i] - lo) * k; bytes[i] = q < 0 ? 0 : q > 255 ? 255 : q; }
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, v.nx, v.ny, v.nz, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
      res = { tex, tf: gl.createTexture()!, tfVersion: -1 };
    }
    o.gpu = { gen: this.gen, version: o.version, res };
    return res;
  }

  // ---- drawing ----
  private draw(): void {
    const gl = this.gl, cv = this.canvas;
    if (!gl || !cv || !this.progs) return;
    const dpr = window.devicePixelRatio || 1, r = this.root.getBoundingClientRect();
    const cssW = Math.max(this.root.clientWidth, r.width), cssH = Math.max(this.root.clientHeight, r.height);
    const W = Math.max(1, Math.round(cssW * dpr)), H = Math.max(1, Math.round(cssH * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const pxPerCss = W / Math.max(1, this.root.clientWidth || cssW);
    gl.enable(gl.SCISSOR_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const bgDefault = this.opts.background ?? panelColour(this.root);
    this.viewports.forEach((vp, vi) => {
      const x = Math.round(vp.x * W), y = Math.round((1 - vp.y - vp.h) * H), w = Math.round(vp.w * W), h = Math.round(vp.h * H);
      gl.viewport(x, y, w, h); gl.scissor(x, y, w, h);
      const bg = vp.background ?? bgDefault;
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
      const V = vp.camera.view(), P = vp.camera.projection(w / h);
      const objs = this.objects.filter(o => o.visible && (!o.viewports || o.viewports.includes(vi)));
      const transparent = (o: GLObject) => (o instanceof MeshObject && (o.style.opacity < 1 || o.style.wireframe === "only")) || (o instanceof SliceObject && o.style.opacity < 1);
      // opaque first, then volumes and transparent objects without depth writes
      for (const o of objs) if (!(o instanceof VolumeObject) && !transparent(o)) this.drawObject(o, V, P, vp, w, h, pxPerCss);
      gl.depthMask(false);
      for (const o of objs) if (o instanceof VolumeObject) this.drawObject(o, V, P, vp, w, h, pxPerCss);
      for (const o of objs) if (transparent(o)) this.drawObject(o, V, P, vp, w, h, pxPerCss);
      gl.depthMask(true);
    });
    gl.disable(gl.SCISSOR_TEST);
    this.opts.onRender?.();
  }

  private common(p: WebGLProgram, o: GLObject, V: Mat4, P: Mat4): void {
    const gl = this.gl!;
    gl.useProgram(p);
    gl.uniformMatrix4fv(this.u(p, "uModel"), false, o.model);
    gl.uniformMatrix4fv(this.u(p, "uView"), false, V);
    gl.uniformMatrix4fv(this.u(p, "uProj"), false, P);
  }

  private drawObject(o: GLObject, V: Mat4, P: Mat4, vp: Viewport, w: number, h: number, pxPerCss: number): void {
    const gl = this.gl!, pr = this.progs!, res = this.resources(o);
    if (!res) return;
    if (o instanceof MeshObject) {
      const p = pr.mesh, s = o.style;
      this.common(p, o, V, P);
      gl.uniformMatrix3fv(this.u(p, "uNormalMat"), false, M.normalMatrix(M.multiply(V, o.model)));
      gl.uniform1i(this.u(p, "uFlat"), s.shading === "flat" ? 1 : 0);
      gl.uniform1i(this.u(p, "uLight"), s.lighting === "unlit" ? 0 : s.lighting === "lambert" ? 1 : 2);
      gl.uniform1i(this.u(p, "uWire"), s.wireframe === "off" ? 0 : s.wireframe === "overlay" ? 1 : 2);
      gl.uniform1i(this.u(p, "uVColor"), s.vertexColors && o.data.colors ? 1 : 0);
      gl.uniform3fv(this.u(p, "uColor"), s.color);
      gl.uniform3fv(this.u(p, "uWireColor"), s.wireColor);
      gl.uniform3fv(this.u(p, "uLightDir"), this.opts.lightDir ?? [-0.35, 0.55, 0.75]);
      gl.uniform1f(this.u(p, "uOpacity"), s.opacity);
      gl.uniform1f(this.u(p, "uShininess"), s.shininess);
      gl.uniform1f(this.u(p, "uAmbient"), s.ambient);
      gl.uniform1f(this.u(p, "uWireWidth"), s.wireWidth * pxPerCss);
      gl.bindVertexArray(res.vao);
      gl.drawArrays(gl.TRIANGLES, 0, res.count);
    } else if (o instanceof PointsObject) {
      const p = pr.points;
      this.common(p, o, V, P);
      const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array;
      gl.uniform1f(this.u(p, "uSize"), Math.min(range[1], o.style.size * pxPerCss));
      gl.uniform1i(this.u(p, "uShade"), o.style.shaded ? 1 : 0);
      gl.bindVertexArray(res.vao);
      gl.drawArrays(gl.POINTS, 0, res.count);
    } else if (o instanceof LinesObject) {
      const p = pr.lines;
      this.common(p, o, V, P);
      gl.uniform2f(this.u(p, "uViewport"), w, h);
      gl.uniform1f(this.u(p, "uWidth"), o.style.width * pxPerCss);
      gl.bindVertexArray(res.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, res.count);
    } else if (o instanceof SliceObject) {
      const p = pr.slice, c = o.style.corners;
      this.common(p, o, V, P);
      gl.bindBuffer(gl.ARRAY_BUFFER, res.bp);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...c[0], ...c[1], ...c[2], ...c[3]]), gl.DYNAMIC_DRAW);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, res.tex);
      const f = o.style.filter === "linear" ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
      gl.uniform1i(this.u(p, "uTex"), 0);
      gl.uniform1f(this.u(p, "uOpacity"), o.style.opacity);
      gl.bindVertexArray(res.vao);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    } else if (o instanceof VolumeObject) {
      const p = pr.vol, s = o.style, v = o.vol;
      if (res.tfVersion !== o.tfVersion) {
        const lut = s.transfer instanceof Uint8ClampedArray ? s.transfer : transferLUT(s.transfer, s.mode === "dvr" ? 0.2 : 0, 1, s.mode === "dvr" ? 1.5 : 1);
        gl.bindTexture(gl.TEXTURE_2D, res.tf);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(lut.buffer, lut.byteOffset, 1024));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        res.tfVersion = o.tfVersion;
      }
      this.common(p, o, V, P);
      const inv = M.invert(o.model) ?? M.identity(), cam = vp.camera;
      gl.uniform3fv(this.u(p, "uEye"), M.transformPoint(inv, cam.eye()));
      gl.uniform3fv(this.u(p, "uDir"), M.transformDir(inv, M.sub(cam.target, cam.eye())));
      gl.uniform1i(this.u(p, "uOrtho"), cam.orthographic ? 1 : 0);
      gl.uniform1i(this.u(p, "uMode"), s.mode === "mip" ? 0 : s.mode === "dvr" ? 1 : 2);
      gl.uniform1i(this.u(p, "uSteps"), Math.max(8, Math.min(2048, Math.round(s.steps))));
      gl.uniform1i(this.u(p, "uFlipY"), s.flipY ? 1 : 0);
      gl.uniform1f(this.u(p, "uIso"), s.iso);
      gl.uniform1f(this.u(p, "uOpacity"), s.opacity);
      gl.uniform1i(this.u(p, "uUseIsoColor"), s.isoColor ? 1 : 0);
      gl.uniform3fv(this.u(p, "uIsoColor"), s.isoColor ?? [1, 1, 1]);
      gl.uniform3f(this.u(p, "uTexel"), 1 / v.nx, 1 / v.ny, 1 / v.nz);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, res.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, res.tf);
      gl.uniform1i(this.u(p, "uVol"), 0); gl.uniform1i(this.u(p, "uTF"), 1);
      gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT);
      if (s.mode === "iso") gl.depthMask(true);
      gl.bindVertexArray(pr.cube.vao);
      gl.drawArrays(gl.TRIANGLES, 0, pr.cube.n);
      gl.disable(gl.CULL_FACE); gl.depthMask(false);
      gl.activeTexture(gl.TEXTURE0);
    }
    gl.bindVertexArray(null);
  }
}

/** The demo panel colour (--qb-panel) of an element, as RGB, for the default background. */
function panelColour(el: Element): RGB {
  const s = getComputedStyle(el).getPropertyValue("--qb-panel").trim();
  if (/^#[0-9a-f]{3,6}$/i.test(s)) return hexToRgb(s);
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (m) { const [r, g, b] = m[1].split(",").map(Number); return [r / 255, g / 255, b / 255]; }
  return [0.965, 0.969, 0.976];
}

/** A row of buttons for standard views of viewport `vp` (pointer only; no keyboard). */
export function viewButtons(parent: Element, view: GLView, vp = 0, views: StandardView[] = ["front", "right", "top", "iso"], zoom = true): HTMLElement {
  const row = htmlEl("div", "qb-presets qb-gl-views", "", parent);
  const names: Record<StandardView, string> = { front: "front", back: "back", left: "left", right: "side", top: "top", bottom: "bottom", iso: "3/4 view" };
  for (const v of views) button(row, names[v], () => { view.viewports[vp].camera.setView(v); view.opts.onCamera?.(vp); view.render(); });
  if (zoom) {
    button(row, "+", () => { view.viewports[vp].camera.zoom(1 / 1.2); view.opts.onCamera?.(vp); view.render(); });
    button(row, "−", () => { view.viewports[vp].camera.zoom(1.2); view.opts.onCamera?.(vp); view.render(); });
  }
  return row;
}
