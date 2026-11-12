// Volume rendering for the course (owner: 3D-graphics day, unit `volrender/`).
//
// Pure part (no DOM; tested in demos/test/graphics3d-test.ts):
//  - headPhantom(): the synthetic 64 × 64 × 40 "head" with anisotropic voxels (1 × 1 × 1.6 mm):
//    soft tissue, a skull shell, brain, two ventricles, a thin bright vessel tree and a dark
//    air cavity, seeded noise; values 0..255;
//  - transfer functions: piecewise-linear opacity over the value (opacity per step of ONE
//    voxel, the reference step) and a colour map; tfAlpha, tfColour, tfLUT;
//  - compositing along one ray, instrumented: front-to-back (C += T α c, T *= 1 − α) and
//    back-to-front (C = α c + (1 − α) C), opacity correction α' = 1 − (1 − α)^(Δs/Δs₀), early
//    termination at T < ε, MIP / MinIP / average, the first crossing of an isovalue (refined by
//    linear interpolation, as in marching squares);
//  - a CPU ray caster over the whole image that returns samples per pixel (the numbers quoted
//    in the exercises come from it, not from the GPU).
// DOM / WebGL2 part:
//  - VolumeRenderer: one canvas, one WebGL2 context, a full-screen ray marcher with the same
//    modes, step size, opacity correction on/off, early termination, nearest/trilinear,
//    spacing correct/ignored, orthographic/perspective and a samples-per-pixel heat map;
//  - transferEditor(): histogram of the volume, draggable opacity points, colour-map choice.
//
// Ray conventions: positions in the volume's world box (centred at the origin, longest side 1,
// y up, so the image row 0 of a slice is at the top); step sizes in voxels of the smallest
// spacing; t = distance from where the ray enters the box.
import type { Volume, Vec3 } from "./volume";
import { volumeFromFunction, softStep, ellipsoidDistance, trilinear } from "./volume";
import { rng } from "./random";
import { ColorMap, grey, inferno, viridis, mapFromAnchors, RGB } from "./colour";
import * as M from "./mat4";
import type { Mat4 } from "./mat4";
import { OrbitCamera } from "./gl3d";
import { svgEl, htmlEl, svgPoint, button } from "./ui";

// =========================================================================================
// The phantom

export const PHANTOM_VALUES = { air: 0, cavity: 8, csf: 40, soft: 70, brain: 100, bone: 170, vessel: 230 };
export const PHANTOM_SPACING: Vec3 = [1, 1, 1.6];

/** Distance (≥ 0) from p to the segment a–b. */
function segDist(p: Vec3, a: Vec3, b: Vec3): number {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const t = Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / (ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)));
  return Math.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]);
}

/** The vessel tree (mm, volume coordinates: x right, y DOWN, z along the stack). */
export const VESSELS: [Vec3, Vec3][] = [
  [[32, 60, 31], [32, 34, 31]],
  [[32, 34, 31], [22, 24, 27]], [[22, 24, 27], [18, 16, 31]], [[22, 24, 27], [26, 13, 25]],
  [[32, 34, 31], [42, 24, 35]], [[42, 24, 35], [46, 16, 31]], [[42, 24, 35], [38, 13, 37]],
];

/** Synthetic head, 64 × 64 × 40 voxels of 1 × 1 × 1.6 mm, values 0..255 (see PHANTOM_VALUES). */
export function headPhantom(o: { noise?: number; seed?: number } = {}): Volume {
  const V = PHANTOM_VALUES, w = 0.6;
  const head = { centre: [32, 33, 31] as Vec3, radii: [26, 29, 27] as Vec3 };
  const skullO = { centre: [32, 29, 31] as Vec3, radii: [22.5, 22.5, 23.5] as Vec3 };
  const skullI = { centre: [32, 29, 31] as Vec3, radii: [20, 20, 21] as Vec3 };
  const vents = [{ centre: [28, 27, 31] as Vec3, radii: [2.5, 6, 4] as Vec3 }, { centre: [36, 27, 31] as Vec3, radii: [2.5, 6, 4] as Vec3 }];
  const cavity = { centre: [32, 55, 24] as Vec3, radii: [6, 3, 5] as Vec3 };
  const mix = (a: number, b: number, s: number) => a + s * (b - a);
  const vol = volumeFromFunction(64, 64, 40, PHANTOM_SPACING, (x, y, z) => {
    const p: Vec3 = [x, y, z];
    let v = V.air;
    v = mix(v, V.soft, softStep(ellipsoidDistance(head, p), w));
    v = mix(v, V.bone, softStep(ellipsoidDistance(skullO, p), w));
    v = mix(v, V.brain, softStep(ellipsoidDistance(skullI, p), w));
    for (const e of vents) v = mix(v, V.csf, softStep(ellipsoidDistance(e, p), w));
    v = mix(v, V.cavity, softStep(ellipsoidDistance(cavity, p), w));
    let dv = Infinity;
    for (const [a, b] of VESSELS) dv = Math.min(dv, segDist(p, a, b));
    v = mix(v, V.vessel, softStep(1.3 - dv, 0.45));
    return v;
  });
  vol.unit = "mm";
  const sd = o.noise ?? 4;
  if (sd > 0) {
    const r = rng(o.seed ?? 7);
    for (let i = 0; i < vol.data.length; i++) vol.data[i] = Math.max(0, Math.min(255, vol.data[i] + sd * r.norm()));
  }
  return vol;
}

// =========================================================================================
// Transfer functions

export type MapName = "grey" | "bone" | "inferno" | "viridis";
export const BONE: ColorMap = mapFromAnchors(["#000000", "#3b2a2a", "#a0603c", "#e8c08a", "#fff8ee"]);
export const MAPS: Record<MapName, ColorMap> = { grey, bone: BONE, inferno, viridis };

/** Opacity control points (value 0..255, opacity per one-voxel step 0..1), sorted by value. */
export interface TF { points: [number, number][]; map: MapName }

/** Piecewise-linear opacity; constant beyond the first and last points. */
export function tfAlpha(tf: TF, v: number): number {
  const P = tf.points;
  if (!P.length) return 0;
  if (v <= P[0][0]) return P[0][1];
  for (let k = 1; k < P.length; k++) if (v <= P[k][0]) {
    const [a, fa] = P[k - 1], [b, fb] = P[k];
    return b > a ? fa + (v - a) / (b - a) * (fb - fa) : fb;
  }
  return P[P.length - 1][1];
}

export const tfColour = (tf: TF, v: number): RGB => MAPS[tf.map](Math.max(0, Math.min(1, v / 255)));

/** 256-entry RGBA LUT (alpha = opacity per one-voxel step). */
export function tfLUT(tf: TF): Uint8ClampedArray {
  const out = new Uint8ClampedArray(1024);
  for (let i = 0; i < 256; i++) {
    const c = tfColour(tf, i);
    out[4 * i] = 255 * c[0]; out[4 * i + 1] = 255 * c[1]; out[4 * i + 2] = 255 * c[2]; out[4 * i + 3] = Math.round(255 * tfAlpha(tf, i));
  }
  return out;
}

/** Opacity of a step Δs, given the opacity α₀ of the reference step Δs₀:
 *  the segment's transmittance is (1 − α₀)^(Δs/Δs₀). */
export const correctAlpha = (a0: number, ds: number, ds0 = 1): number => 1 - Math.pow(1 - Math.min(a0, 0.999999), ds / ds0);

/** Opacity of a segment of length Δs with extinction τ: α = 1 − e^(−τ Δs). */
export const segmentAlpha = (tau: number, ds: number): number => 1 - Math.exp(-tau * ds);

// =========================================================================================
// Compositing along one ray

export interface CompositeStep { C: RGB; T: number }

/** Front to back: C ← C + T α c, T ← T (1 − α). Stops when T < eps (0 = never).
 *  Returns the final colour, transmittance, the number of samples used and every partial sum. */
export function frontToBack(alphas: number[], cols: RGB[], eps = 0): { C: RGB; T: number; used: number; steps: CompositeStep[] } {
  let C: RGB = [0, 0, 0], T = 1, used = 0;
  const steps: CompositeStep[] = [];
  for (let i = 0; i < alphas.length; i++) {
    if (eps > 0 && T < eps) break;
    const a = alphas[i];
    C = [C[0] + T * a * cols[i][0], C[1] + T * a * cols[i][1], C[2] + T * a * cols[i][2]];
    T *= 1 - a; used++;
    steps.push({ C, T });
  }
  return { C, T, used, steps };
}

/** Back to front (the painter's order): C ← α c + (1 − α) C, starting from the far end.
 *  Partial sums are listed in processing order (far sample first). */
export function backToFront(alphas: number[], cols: RGB[]): { C: RGB; T: number; steps: CompositeStep[] } {
  let C: RGB = [0, 0, 0], T = 1;
  const steps: CompositeStep[] = [];
  for (let i = alphas.length - 1; i >= 0; i--) {
    const a = alphas[i];
    C = [a * cols[i][0] + (1 - a) * C[0], a * cols[i][1] + (1 - a) * C[1], a * cols[i][2] + (1 - a) * C[2]];
    T *= 1 - a;
    steps.push({ C, T });
  }
  return { C, T, steps };
}

export type RayMode = "mip" | "minip" | "avg" | "dvr" | "iso";

export interface RayOptions {
  mode: RayMode;
  /** Step between samples in voxels (of the smallest spacing). */
  step: number;
  /** Opacity correction for the step (default true). */
  correction?: boolean;
  /** Early termination threshold on T (0 = off). */
  eps?: number;
  /** Isovalue (mode "iso"). */
  iso?: number;
  /** MinIP ignores samples outside the body: only samples between the first and last sample
   *  above this value count (default 30). */
  body?: number;
}

export interface RayResult {
  /** Displayed colour (DVR, iso) or grey value 0..255 as [v, v, v] (projections). */
  C: RGB;
  T: number;
  value: number;
  /** Samples actually taken (early termination stops counting). */
  used: number;
  /** Parameter of the isosurface hit (in samples, fractional), or −1. */
  hit: number;
  alphas: number[];
  cols: RGB[];
}

/** Render one ray from its samples (values 0..255, step `step` voxels apart). */
export function renderRay(values: number[], tf: TF, o: RayOptions): RayResult {
  const n = values.length, corr = o.correction ?? true, eps = o.eps ?? 0;
  const alphas = values.map(v => { const a = tfAlpha(tf, v); return corr ? correctAlpha(a, o.step) : Math.min(1, a); });
  const cols = values.map(v => tfColour(tf, v));
  const res: RayResult = { C: [0, 0, 0], T: 1, value: 0, used: n, hit: -1, alphas, cols };
  if (o.mode === "mip") res.value = n ? Math.max(...values) : 0;
  else if (o.mode === "avg") res.value = n ? values.reduce((a, b) => a + b, 0) / n : 0;
  else if (o.mode === "minip") {
    const body = o.body ?? 30, first = values.findIndex(v => v > body);
    let last = -1;
    for (let i = n - 1; i >= 0; i--) if (values[i] > body) { last = i; break; }
    res.value = first < 0 ? 0 : Math.min(...values.slice(first, last + 1));
  } else if (o.mode === "dvr") {
    const r = frontToBack(alphas, cols, eps);
    res.C = r.C; res.T = r.T; res.used = r.used;
  } else {
    const c = o.iso ?? 128;
    for (let i = 1; i < n; i++) if (values[i - 1] < c && values[i] >= c) { res.hit = i - 1 + (c - values[i - 1]) / (values[i] - values[i - 1]); res.used = i + 1; break; }
    if (res.hit < 0 && n && values[0] >= c) { res.hit = 0; res.used = 1; }
  }
  if (o.mode !== "dvr" && o.mode !== "iso") res.C = [res.value, res.value, res.value];
  return res;
}

// =========================================================================================
// Geometry of the volume box and a CPU ray caster

/** Model matrix: texture coordinates (s, t, r) ∈ [0, 1]³ → world box (centred, longest side 1,
 *  world y = −image y). `ignoreSpacing` uses spacing 1, 1, 1 (the volume stretches). */
export function boxModel(v: Volume, ignoreSpacing = false): Mat4 {
  const sp = ignoreSpacing ? [1, 1, 1] : v.spacing;
  const e = [v.nx * sp[0], v.ny * sp[1], v.nz * sp[2]], m = Math.max(...e);
  return M.mul(M.scaling(e[0] / m, -e[1] / m, e[2] / m), M.translation(-0.5, -0.5, -0.5));
}

/** Length of one voxel of the smallest spacing, in world units. */
export function voxelWorld(v: Volume, ignoreSpacing = false): number {
  const sp = ignoreSpacing ? [1, 1, 1] : v.spacing;
  return Math.min(...sp) / Math.max(v.nx * sp[0], v.ny * sp[1], v.nz * sp[2]);
}

/** Entry and exit of the ray o + t d through the unit cube (slab method), or null. */
export function rayBox(o: Vec3, d: Vec3): [number, number] | null {
  let t0 = -Infinity, t1 = Infinity;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-12) { if (o[k] < 0 || o[k] > 1) return null; continue; }
    let a = (0 - o[k]) / d[k], b = (1 - o[k]) / d[k];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a); t1 = Math.min(t1, b);
  }
  t0 = Math.max(t0, 0);
  return t1 > t0 ? [t0, t1] : null;
}

/** Value at texture coordinates (s, t, r) by trilinear (or nearest) interpolation; 0 outside. */
export function sampleTex(v: Volume, s: number, t: number, r: number, nearest = false): number {
  const x = s * v.nx - 0.5, y = t * v.ny - 0.5, z = r * v.nz - 0.5;
  if (nearest) {
    const i = Math.min(v.nx - 1, Math.max(0, Math.round(x))), j = Math.min(v.ny - 1, Math.max(0, Math.round(y))), k = Math.min(v.nz - 1, Math.max(0, Math.round(z)));
    return v.data[(k * v.ny + j) * v.nx + i];
  }
  return trilinear(v, Math.max(0, Math.min(v.nx - 1, x)), Math.max(0, Math.min(v.ny - 1, y)), Math.max(0, Math.min(v.nz - 1, z)));
}

export interface ViewOptions extends RayOptions { ignoreSpacing?: boolean; nearest?: boolean }

/** World-space ray through NDC (x, y) of a camera with matrix PV, as texture-space origin and
 *  direction per WORLD unit, plus the world length of the direction. */
export function pixelRay(invPV: Mat4, invModel: Mat4, x: number, y: number): { o: Vec3; d: Vec3 } {
  const a = M.transformPoint(invPV, [x, y, -1]), b = M.transformPoint(invPV, [x, y, 1]);
  const dir = M.normalize(M.sub(b, a));
  return { o: M.transformPoint(invModel, a) as Vec3, d: M.transformDir(invModel, dir) as Vec3 };
}

/** Cast every ray of a w × h image on the CPU; returns the samples taken per pixel (and the
 *  samples a ray would take without early termination) and the rendered values. */
export function castImage(v: Volume, tf: TF, PV: Mat4, w: number, h: number, o: ViewOptions): { used: Float32Array; full: Float32Array; results: (RayResult | null)[] } {
  const model = boxModel(v, o.ignoreSpacing), invModel = M.invert(model)!, invPV = M.invert(PV)!;
  const dt = o.step * voxelWorld(v, o.ignoreSpacing);
  const used = new Float32Array(w * h), full = new Float32Array(w * h), results: (RayResult | null)[] = [];
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const { o: ro, d } = pixelRay(invPV, invModel, (i + 0.5) / w * 2 - 1, (j + 0.5) / h * 2 - 1);
    const hit = rayBox(ro, d);
    if (!hit) { results.push(null); continue; }
    const vals: number[] = [];
    for (let t = hit[0] + 0.5 * dt; t < hit[1]; t += dt) vals.push(sampleTex(v, ro[0] + t * d[0], ro[1] + t * d[1], ro[2] + t * d[2], o.nearest));
    const r = renderRay(vals, tf, o);
    used[j * w + i] = r.used; full[j * w + i] = vals.length;
    results.push(r);
  }
  return { used, full, results };
}

// =========================================================================================
// WebGL2 ray marcher

const VS = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
uniform sampler3D uVol;
uniform sampler2D uTF;
uniform mat4 uInvPV, uInvModel;
uniform int uMode, uCorrect, uShowCount, uMaxSteps;
uniform float uDt, uStepVox, uEps, uIso, uBody, uGain;
uniform vec3 uBg;
out vec4 o;
float S(vec3 p) { return texture(uVol, p).r * 255.0; }
vec3 heat(float x) { x = clamp(x, 0.0, 1.0); return vec3(smoothstep(0.0, 0.55, x), smoothstep(0.25, 0.9, x), smoothstep(0.7, 1.0, x) * 0.9 + 0.1 * (1.0 - x)); }
void main() {
  vec4 a4 = uInvPV * vec4(vNdc, -1.0, 1.0), b4 = uInvPV * vec4(vNdc, 1.0, 1.0);
  vec3 a = a4.xyz / a4.w, b = b4.xyz / b4.w, dir = normalize(b - a);
  vec3 ro = (uInvModel * vec4(a, 1.0)).xyz, rd = (uInvModel * vec4(dir, 0.0)).xyz;
  vec3 dd = mix(rd, vec3(1e-9), lessThan(abs(rd), vec3(1e-9)));
  vec3 t0 = -ro / dd, t1 = (1.0 - ro) / dd, tmn = min(t0, t1), tmx = max(t0, t1);
  float tn = max(max(max(tmn.x, tmn.y), tmn.z), 0.0), tf = min(min(tmx.x, tmx.y), tmx.z);
  if (tf <= tn) { o = vec4(uBg, 1.0); return; }
  float t = tn + 0.5 * uDt;
  int used = 0;
  vec3 col = uBg;
  if (uMode == 0 || uMode == 1 || uMode == 2) {           // MIP, MinIP, average
    float m = uMode == 1 ? 1e9 : 0.0, s = 0.0, n = 0.0, mIn = 0.0; bool entered = false;
    for (int i = 0; i < 4096; i++) {
      if (t > tf || i >= uMaxSteps) break;
      float v = S(ro + rd * t); used++;
      if (uMode == 0) m = max(m, v);
      else if (uMode == 2) { s += v; n += 1.0; }
      else { if (v > uBody) entered = true; if (entered) { m = min(m, v); if (v > uBody) mIn = m; } }
      t += uDt;
    }
    float val = uMode == 0 ? m : uMode == 2 ? s / max(n, 1.0) * uGain : (entered ? mIn : 0.0);
    col = vec3(clamp(val / 255.0, 0.0, 1.0));
  } else if (uMode == 3) {                                 // DVR, front to back
    vec3 C = vec3(0.0); float T = 1.0;
    for (int i = 0; i < 4096; i++) {
      if (t > tf || i >= uMaxSteps) break;
      if (uEps > 0.0 && T < uEps) break;
      vec4 c = texture(uTF, vec2((S(ro + rd * t) + 0.5) / 256.0, 0.5)); used++;
      float al = uCorrect == 1 ? 1.0 - pow(1.0 - min(c.a, 0.999), uStepVox) : c.a;
      C += T * al * c.rgb; T *= 1.0 - al;
      t += uDt;
    }
    col = C + T * uBg;
  } else {                                                 // first hit of the isosurface
    float prev = S(ro + rd * t); used++;
    for (int i = 0; i < 4096; i++) {
      t += uDt;
      if (t > tf || i >= uMaxSteps) break;
      float v = S(ro + rd * t); used++;
      if (prev < uIso && v >= uIso) {
        float th = t - uDt + uDt * (uIso - prev) / (v - prev);
        vec3 p = ro + rd * th;
        vec3 h = 1.0 / vec3(textureSize(uVol, 0));
        vec3 g = vec3(S(p + vec3(h.x, 0, 0)) - S(p - vec3(h.x, 0, 0)), S(p + vec3(0, h.y, 0)) - S(p - vec3(0, h.y, 0)), S(p + vec3(0, 0, h.z)) - S(p - vec3(0, 0, h.z))) / (2.0 * h);
        vec3 nw = normalize((transpose(uInvModel) * vec4(g, 0.0)).xyz);
        vec3 n = -nw, L = -dir;
        float d = max(dot(n, L), 0.0);
        vec3 base = texture(uTF, vec2((uIso + 0.5) / 256.0, 0.5)).rgb * 0.8 + 0.2;
        col = base * (0.2 + 0.8 * d) + vec3(0.3) * pow(max(dot(n, L), 0.0), 40.0);
        break;
      }
      prev = v;
    }
  }
  if (uShowCount == 1) col = heat(float(used) / float(uMaxSteps));
  o = vec4(col, 1.0);
}`;

export interface RenderSettings {
  mode: RayMode;
  step: number;
  correction: boolean;
  eps: number;
  iso: number;
  nearest: boolean;
  ignoreSpacing: boolean;
  showCount: boolean;
}

/** A WebGL2 volume ray marcher on its own canvas; drag to orbit. Renders on demand. */
export class VolumeRenderer {
  readonly root: HTMLDivElement;
  readonly canvas: HTMLCanvasElement;
  camera = new OrbitCamera({ distance: 2.2, azimuth: 0.5, elevation: 0.25, fovy: 35 * Math.PI / 180 });
  settings: RenderSettings = { mode: "dvr", step: 0.5, correction: true, eps: 0, iso: 135, nearest: false, ignoreSpacing: false, showCount: false };
  onCamera: (() => void) | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private prog: WebGLProgram | null = null;
  private volTex: WebGLTexture | null = null;
  private tfTex: WebGLTexture | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private vol: Volume | null = null;
  private frame = 0;
  private loc = new Map<string, WebGLUniformLocation | null>();

  constructor(parent: HTMLElement, fallback = "") {
    this.root = htmlEl("div", "qb-gl", "", parent);
    this.root.style.aspectRatio = "1";
    this.canvas = htmlEl("canvas", "qb-gl-canvas", "", this.root);
    const gl = this.canvas.getContext("webgl2", { preserveDrawingBuffer: true });
    if (!gl) { htmlEl("div", "qb-gl-msg", "This 3D view needs WebGL2. " + fallback, this.root); this.root.style.aspectRatio = "auto"; this.canvas.remove(); return; }
    this.gl = gl;
    try { this.setup(gl); } catch (e) { this.gl = null; htmlEl("div", "qb-gl-msg", "WebGL2 shaders failed: " + String(e), this.root); }
    this.installPointer();
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => this.render()).observe(this.root);
  }

  private setup(gl: WebGL2RenderingContext): void {
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
    this.prog = p;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.volTex = gl.createTexture(); this.tfTex = gl.createTexture();
  }

  private u(name: string): WebGLUniformLocation | null {
    if (!this.loc.has(name)) this.loc.set(name, this.gl!.getUniformLocation(this.prog!, name));
    return this.loc.get(name)!;
  }

  setVolume(v: Volume): void {
    this.vol = v;
    const gl = this.gl;
    if (!gl) return;
    const bytes = new Uint8Array(v.data.length);
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.max(0, Math.min(255, Math.round(v.data[i])));
    gl.bindTexture(gl.TEXTURE_3D, this.volTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, v.nx, v.ny, v.nz, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.render();
  }

  setTF(tf: TF): void {
    const gl = this.gl;
    if (!gl) return;
    gl.bindTexture(gl.TEXTURE_2D, this.tfTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(tfLUT(tf).buffer));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    this.render();
  }

  set(s: Partial<RenderSettings>): void { Object.assign(this.settings, s); this.render(); }

  /** The camera's projection × view for the canvas (square). */
  pv(): Mat4 { return M.multiply(this.camera.projection(1), this.camera.view()); }

  render(): void {
    if (!this.gl || this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }

  private draw(): void {
    const gl = this.gl, v = this.vol;
    if (!gl || !v || !this.prog) return;
    const dpr = window.devicePixelRatio || 1, css = this.root.clientWidth || 400;
    const W = Math.round(css * dpr);
    if (this.canvas.width !== W) { this.canvas.width = W; this.canvas.height = W; }
    gl.viewport(0, 0, W, W);
    const s = this.settings;
    gl.useProgram(this.prog);
    const model = boxModel(v, s.ignoreSpacing);
    gl.uniformMatrix4fv(this.u("uInvPV"), false, M.invert(this.pv())!);
    gl.uniformMatrix4fv(this.u("uInvModel"), false, M.invert(model)!);
    const modes: Record<RayMode, number> = { mip: 0, minip: 1, avg: 2, dvr: 3, iso: 4 };
    gl.uniform1i(this.u("uMode"), modes[s.mode]);
    gl.uniform1i(this.u("uCorrect"), s.correction ? 1 : 0);
    gl.uniform1i(this.u("uShowCount"), s.showCount ? 1 : 0);
    const dt = s.step * voxelWorld(v, s.ignoreSpacing);
    gl.uniform1i(this.u("uMaxSteps"), Math.min(4096, Math.ceil(Math.sqrt(3) / voxelWorld(v, s.ignoreSpacing) / s.step) + 2));
    gl.uniform1f(this.u("uDt"), dt);
    gl.uniform1f(this.u("uStepVox"), s.step);
    gl.uniform1f(this.u("uEps"), s.eps);
    gl.uniform1f(this.u("uIso"), s.iso);
    gl.uniform1f(this.u("uBody"), 30);
    gl.uniform1f(this.u("uGain"), 2);
    gl.uniform3f(this.u("uBg"), 0.08, 0.09, 0.12);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.volTex);
    const f = s.nearest ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, f);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.tfTex);
    gl.uniform1i(this.u("uVol"), 0); gl.uniform1i(this.u("uTF"), 1);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
  }

  private installPointer(): void {
    const el = this.root;
    el.style.touchAction = "none";
    let drag: [number, number] | null = null;
    el.addEventListener("pointerdown", ev => { drag = [ev.clientX, ev.clientY]; el.setPointerCapture(ev.pointerId); ev.preventDefault(); });
    el.addEventListener("pointermove", ev => {
      if (!drag) return;
      const r = el.getBoundingClientRect();
      this.camera.orbit(-(ev.clientX - drag[0]) / r.width * Math.PI * 1.2, (ev.clientY - drag[1]) / r.height * Math.PI);
      drag = [ev.clientX, ev.clientY];
      this.render(); this.onCamera?.();
    });
    const up = () => { drag = null; };
    el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
  }
}

// =========================================================================================
// Transfer-function editor

export interface TFEditor { svg: SVGSVGElement; set(tf: TF): void; get(): TF }

/** Histogram of `data` (log counts, grey bars), the opacity curve with draggable points and a
 *  strip of the colour map; buttons choose the colour map. Calls onChange on every edit. */
export function transferEditor(parent: Element, init: TF, data: ArrayLike<number> | null, onChange: (tf: TF) => void, o: { width?: number; height?: number; aMax?: number } = {}): TFEditor {
  const W = o.width ?? 380, H = o.height ?? 150, aMax = o.aMax ?? 1;
  const ml = 40, mr = 10, mt = 8, mb = 44;
  let tf: TF = { points: init.points.map(p => [...p] as [number, number]), map: init.map };
  const wrap = htmlEl("div", "qb-tfedit", "", parent);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "qb-plot-svg" }, wrap);
  svg.style.width = "100%"; svg.style.touchAction = "none";
  const X = (v: number) => ml + v / 255 * (W - ml - mr), Y = (a: number) => H - mb - a / aMax * (H - mb - mt);
  const iX = (x: number) => Math.max(0, Math.min(255, (x - ml) / (W - ml - mr) * 255)), iY = (y: number) => Math.max(0, Math.min(aMax, (H - mb - y) / (H - mb - mt) * aMax));
  // histogram
  if (data) {
    const bins = new Float64Array(64);
    for (let i = 0; i < data.length; i++) bins[Math.max(0, Math.min(63, Math.floor(data[i] / 4)))]++;
    const lmax = Math.log10(1 + Math.max(...bins));
    bins.forEach((c, b) => {
      const h = Math.log10(1 + c) / lmax * (H - mb - mt);
      svgEl("rect", { x: X(4 * b), y: H - mb - h, width: X(4) - X(0) + 0.3, height: h, fill: "var(--qb-grid, #ddd)" }, svg);
    });
  }
  svgEl("line", { x1: ml, y1: H - mb, x2: W - mr, y2: H - mb, stroke: "var(--qb-axis, #444)" }, svg);
  svgEl("line", { x1: ml, y1: mt, x2: ml, y2: H - mb, stroke: "var(--qb-axis, #444)" }, svg);
  const txt = (x: number, y: number, s: string, anchor = "middle") => { const t = svgEl("text", { x, y, "text-anchor": anchor, "font-size": 16 }, svg); t.textContent = s; return t; };
  txt(ml - 4, mt + 12, aMax.toFixed(aMax < 1 ? 1 : 0), "end"); txt(ml - 4, H - mb, "0", "end");
  txt(ml - 26, (H - mb + mt) / 2, "α").setAttribute("font-style", "italic");
  [0, 64, 128, 192, 255].forEach(v => txt(X(v), H - mb + 34, String(v)));
  const strip = svgEl("g", {}, svg);
  const poly = svgEl("polyline", { fill: "none", stroke: "var(--qb-c2, #d62728)", "stroke-width": 3 }, svg);
  const handles = svgEl("g", {}, svg);
  const mapRow = htmlEl("div", "qb-presets", "", wrap);
  const mapBtns = (Object.keys(MAPS) as MapName[]).map(m => button(mapRow, m, () => { tf.map = m; redraw(); onChange(get()); }));

  function redraw(): void {
    strip.innerHTML = "";
    for (let i = 0; i < 64; i++) {
      const c = tfColour(tf, 4 * i + 2);
      svgEl("rect", { x: X(4 * i), y: H - mb + 3, width: X(4) - X(0) + 0.4, height: 12, fill: `rgb(${c.map(x => Math.round(255 * x)).join(",")})` }, strip);
    }
    poly.setAttribute("points", [[0, tfAlpha(tf, 0)] as [number, number], ...tf.points, [255, tfAlpha(tf, 255)] as [number, number]].map(([v, a]) => `${X(v)},${Y(a)}`).join(" "));
    handles.innerHTML = "";
    tf.points.forEach((p, k) => {
      const c = svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: 8, fill: "#fff", stroke: "var(--qb-c2, #d62728)", "stroke-width": 3 }, handles);
      c.style.cursor = "grab";
      c.addEventListener("pointerdown", ev => {
        ev.preventDefault(); ev.stopPropagation();
        c.setPointerCapture(ev.pointerId);
        const move = (e: PointerEvent) => {
          const [x, y] = svgPoint(svg, e);
          const lo = k > 0 ? tf.points[k - 1][0] : 0, hi = k < tf.points.length - 1 ? tf.points[k + 1][0] : 255;
          tf.points[k] = [Math.round(Math.max(lo, Math.min(hi, iX(x)))), Math.round(iY(y) * 1000) / 1000];
          redraw(); onChange(get());
        };
        const up = () => { c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); };
        c.addEventListener("pointermove", move); c.addEventListener("pointerup", up);
      });
    });
    mapBtns.forEach((b, i) => b.classList.toggle("active", (Object.keys(MAPS) as MapName[])[i] === tf.map));
  }
  function get(): TF { return { points: tf.points.map(p => [...p] as [number, number]), map: tf.map }; }
  redraw();
  return { svg, get, set: (t: TF) => { tf = { points: t.points.map(p => [...p] as [number, number]), map: t.map }; redraw(); } };
}
