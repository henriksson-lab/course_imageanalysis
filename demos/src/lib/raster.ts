// A small CPU software rasteriser (owner: 3D-graphics day, unit `rasterise/`). No DOM: safe
// to import from tests. It does what a GPU does after the vertex stage, slowly and visibly:
//
//  - window ("screen") coordinates: x right, y UP, pixel (i, j) covers [i, i+1] × [j, j+1] and
//    its centre is (i + 0.5, j + 0.5) (the OpenGL convention); counter-clockwise = front;
//  - edge functions E_ab(p) = (b − a) × (p − a) (z component, twice the signed area of a, b, p);
//    inside = all three ≥ 0 with the top-left rule for pixels exactly on an edge;
//  - barycentric coordinates (b_a, b_b, b_c) = (E_bc, E_ca, E_ab)(p) / E_ab(c);
//  - z-buffer on NDC depth (smaller = nearer), back-face culling by the sign of the area;
//  - attributes interpolated affinely (wrong under perspective) or perspective-correctly
//    (interpolate f/w and 1/w, then divide);
//  - clipping of a clip-space triangle against the near plane (z = −w) before the divide.
//
//   const t = createTarget(32, 24);
//   rasteriseTriangle(t, [a, b, c], { cull: true, topLeft: true, perspective: true }, (attr) => attr);
//   const steps = rasteriseSteps([tri1, tri2], 32, 24, opts);   // one record per pixel visit

export type V2 = [number, number];
export type V4 = [number, number, number, number];

/** A vertex in window coordinates: x, y (pixels, y up), z = NDC depth in [-1, 1], w = the clip
 *  w (the view-space distance −z_view for a perspective camera, 1 for orthographic), and
 *  any number of attributes (colour, normal, texture coordinate, ...). */
export interface WinVertex { x: number; y: number; z: number; w: number; attr: number[] }

export interface RasterOptions {
  /** Skip triangles whose screen area is negative (clockwise = back facing). */
  cull?: boolean;
  /** Pixels exactly on a shared edge belong to one triangle only (default true). */
  topLeft?: boolean;
  /** Perspective-correct attribute interpolation (default true). */
  perspective?: boolean;
  /** Depth test (default true). */
  depthTest?: boolean;
  /** Store depth as an integer of this many bits (default: a float, no rounding). */
  depthBits?: number;
}

/** NDC depth rounded to a b-bit integer depth buffer (and back to NDC). */
export function quantiseDepth(z: number, bits: number): number {
  const m = 2 ** bits - 1;
  return Math.round((Math.min(1, Math.max(-1, z)) + 1) / 2 * m) / m * 2 - 1;
}

/** Colour (RGB in [0, 1]), depth (NDC z, initialised to +∞) and the id of the triangle that
 *  wrote each pixel (−1 = none). Row 0 is the BOTTOM row (y up). */
export interface Target { w: number; h: number; color: Float32Array; depth: Float32Array; id: Int32Array }

export function createTarget(w: number, h: number, bg: [number, number, number] = [0, 0, 0]): Target {
  const color = new Float32Array(3 * w * h);
  for (let i = 0; i < w * h; i++) { color[3 * i] = bg[0]; color[3 * i + 1] = bg[1]; color[3 * i + 2] = bg[2]; }
  return { w, h, color, depth: new Float32Array(w * h).fill(Infinity), id: new Int32Array(w * h).fill(-1) };
}

/** Edge function E_ab(p) = (b − a) × (p − a): positive when p lies to the LEFT of a → b (y up). */
export function edge(a: { x: number; y: number }, b: { x: number; y: number }, px: number, py: number): number {
  return (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
}

/** Twice the signed area of triangle a, b, c (positive = counter-clockwise, y up). */
export const area2 = (a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }): number => edge(a, b, c.x, c.y);

/** Top-left rule for a counter-clockwise triangle in y-up coordinates: a "left" edge runs
 *  downwards (dy < 0); a "top" edge is horizontal and runs right to left. */
export function isTopLeft(a: { x: number; y: number }, b: { x: number; y: number }): boolean {
  const dx = b.x - a.x, dy = b.y - a.y;
  return dy < 0 || (dy === 0 && dx < 0);
}

/** Barycentric coordinates of p in triangle a, b, c (any orientation; null if degenerate). */
export function barycentric(a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }, px: number, py: number): [number, number, number] | null {
  const A = area2(a, b, c);
  if (A === 0) return null;
  return [edge(b, c, px, py) / A, edge(c, a, px, py) / A, edge(a, b, px, py) / A];
}

/** Interpolate attributes with screen-space barycentrics `bs`: affine, or perspective-correct
 *  using each vertex's w (f = Σ b_i f_i / w_i  /  Σ b_i / w_i). */
export function interpolate(vs: WinVertex[], bs: number[], perspective: boolean): number[] {
  const n = vs[0].attr.length, out = new Array<number>(n).fill(0);
  if (!perspective) {
    for (let k = 0; k < n; k++) out[k] = bs[0] * vs[0].attr[k] + bs[1] * vs[1].attr[k] + bs[2] * vs[2].attr[k];
    return out;
  }
  const q = [bs[0] / vs[0].w, bs[1] / vs[1].w, bs[2] / vs[2].w], s = q[0] + q[1] + q[2];
  for (let k = 0; k < n; k++) out[k] = (q[0] * vs[0].attr[k] + q[1] * vs[1].attr[k] + q[2] * vs[2].attr[k]) / s;
  return out;
}

/** Perspective-correct barycentrics (the weights that interpolate in 3D, not on the screen). */
export function perspectiveBary(vs: WinVertex[], bs: number[]): [number, number, number] {
  const q = [bs[0] / vs[0].w, bs[1] / vs[1].w, bs[2] / vs[2].w], s = q[0] + q[1] + q[2];
  return [q[0] / s, q[1] / s, q[2] / s];
}

/** What happened at one pixel visit (for the step-through demo). */
export interface Fragment {
  tri: number;
  /** Pixel indices (j counted from the bottom). */
  i: number;
  j: number;
  /** Edge functions E_bc, E_ca, E_ab at the pixel centre (after reordering to CCW). */
  E: [number, number, number];
  inside: boolean;
  /** On an edge (some E = 0) and rejected or accepted by the top-left rule. */
  onEdge: boolean;
  /** Screen-space and perspective-correct barycentrics (w.r.t. the triangle's own a, b, c). */
  bary: [number, number, number];
  baryP: [number, number, number];
  /** Interpolated depth (NDC z), the depth already stored, and the test result. */
  z: number;
  zBefore: number;
  passed: boolean;
  attr: number[];
  /** Colour written (if passed). */
  color: [number, number, number] | null;
}

/** Shader: attributes in, RGB out. */
export type FragmentShader = (attr: number[], i: number, j: number, bary: [number, number, number]) => [number, number, number];

/**
 * Rasterise one triangle into `t`. Calls `shade` for every covered pixel that passes the depth
 * test and writes its colour and depth. Returns the fragments visited if `record` is given
 * (every pixel of the bounding box, in rows from the bottom). Returns false if culled.
 */
export function rasteriseTriangle(t: Target, tri: WinVertex[], opts: RasterOptions, shade: FragmentShader, id = 0, record?: Fragment[]): boolean {
  const topLeft = opts.topLeft ?? true, persp = opts.perspective ?? true, dt = opts.depthTest ?? true;
  let [a, b, c] = tri;
  let order = [0, 1, 2];
  const A = area2(a, b, c);
  if (A === 0) return false;
  if (A < 0) {
    if (opts.cull) return false;
    [b, c] = [c, b]; order = [0, 2, 1];       // rasterise as counter-clockwise
  }
  const AA = Math.abs(A);
  const x0 = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x))), x1 = Math.min(t.w - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
  const y0 = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y))), y1 = Math.min(t.h - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
  const tl = [isTopLeft(b, c), isTopLeft(c, a), isTopLeft(a, b)];
  for (let j = y0; j <= y1; j++) for (let i = x0; i <= x1; i++) {
    const px = i + 0.5, py = j + 0.5;
    const E: [number, number, number] = [edge(b, c, px, py), edge(c, a, px, py), edge(a, b, px, py)];
    let inside = true, onEdge = false;
    for (let k = 0; k < 3; k++) {
      if (E[k] < 0) inside = false;
      else if (E[k] === 0) { onEdge = true; if (topLeft && !tl[k]) inside = false; }
    }
    // barycentrics w.r.t. the triangle's own vertex order
    const bo = [E[0] / AA, E[1] / AA, E[2] / AA];
    const bary: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 3; k++) bary[order[k]] = bo[k];
    const baryP = perspectiveBary(tri, bary);
    const zi = bary[0] * tri[0].z + bary[1] * tri[1].z + bary[2] * tri[2].z;   // depth: affine in screen space
    const z = opts.depthBits ? quantiseDepth(zi, opts.depthBits) : zi;
    const p = j * t.w + i, zBefore = t.depth[p];
    const passed = inside && (!dt || z < zBefore) && zi >= -1 && zi <= 1;
    let attr: number[] = [];
    if (inside) attr = interpolate(tri, bary, persp);
    let col: [number, number, number] | null = null;
    if (passed) {
      col = shade(attr, i, j, persp ? baryP : bary);
      t.color[3 * p] = col[0]; t.color[3 * p + 1] = col[1]; t.color[3 * p + 2] = col[2];
      t.depth[p] = z; t.id[p] = id;
    }
    if (record) record.push({ tri: id, i, j, E, inside, onEdge, bary, baryP, z, zBefore, passed, attr, color: col });
  }
  return true;
}

/** Rasterise a list of triangles (window coordinates) and record every pixel visit in order. */
export function rasteriseSteps(tris: WinVertex[][], w: number, h: number, opts: RasterOptions, shade: (tri: number) => FragmentShader, bg: [number, number, number] = [0, 0, 0]): { target: Target; steps: Fragment[]; culled: boolean[] } {
  const target = createTarget(w, h, bg), steps: Fragment[] = [], culled: boolean[] = [];
  tris.forEach((tri, k) => { culled.push(!rasteriseTriangle(target, tri, opts, shade(k), k, steps)); });
  return { target, steps, culled };
}

// -------------------------------------------------------------------------------------------
// Depth

/** NDC depth of a point at view-space distance d (= −z_view > 0) for near n and far f
 *  (OpenGL perspective matrix): z_ndc = A + B / z_view with A = (f + n)/(f − n), B = 2fn/(f − n). */
export function ndcDepth(d: number, n: number, f: number): number {
  const A = (f + n) / (f - n), B = 2 * f * n / (f - n);
  return A - B / d;
}

// -------------------------------------------------------------------------------------------
// Clip space → window, with near-plane clipping

/** A vertex in clip space (x, y, z, w) with attributes. */
export interface ClipVertex { p: V4; attr: number[] }

const lerpV = (a: ClipVertex, b: ClipVertex, s: number): ClipVertex => ({
  p: [0, 1, 2, 3].map(k => a.p[k] + s * (b.p[k] - a.p[k])) as V4,
  attr: a.attr.map((v, k) => v + s * (b.attr[k] - v)),
});

/** Sutherland–Hodgman clipping of a polygon against the near plane z ≥ −w (i.e. z + w ≥ 0).
 *  Done in clip space, before the divide, where everything is still linear. */
export function clipNear(poly: ClipVertex[]): ClipVertex[] {
  const out: ClipVertex[] = [];
  const d = (v: ClipVertex) => v.p[2] + v.p[3];
  for (let k = 0; k < poly.length; k++) {
    const a = poly[k], b = poly[(k + 1) % poly.length], da = d(a), db = d(b);
    if (da >= 0) out.push(a);
    if ((da >= 0) !== (db >= 0)) out.push(lerpV(a, b, da / (da - db)));
  }
  return out;
}

/** Perspective divide and viewport: clip → NDC → window (W × H pixels, y up). */
export function toWindow(v: ClipVertex, W: number, H: number): WinVertex {
  const [x, y, z, w] = v.p;
  return { x: (x / w + 1) * 0.5 * W, y: (y / w + 1) * 0.5 * H, z: z / w, w, attr: v.attr };
}

/** Clip a clip-space triangle against the near plane, divide, map to the window and rasterise
 *  the resulting fan. Returns the number of triangles drawn (0 if culled or clipped away). */
export function drawClipTriangle(t: Target, tri: ClipVertex[], opts: RasterOptions, shade: FragmentShader, id = 0): number {
  const poly = clipNear(tri);
  if (poly.length < 3) return 0;
  const win = poly.map(v => toWindow(v, t.w, t.h));
  // cull on the whole (planar) polygon, not on the fan pieces
  if (opts.cull) {
    let s = 0;
    for (let k = 0; k < win.length; k++) { const a = win[k], b = win[(k + 1) % win.length]; s += a.x * b.y - b.x * a.y; }
    if (s < 0) return 0;
  }
  let n = 0;
  for (let k = 1; k + 1 < win.length; k++) if (rasteriseTriangle(t, [win[0], win[k], win[k + 1]], { ...opts, cull: false }, shade, id)) n++;
  return n;
}

/** Target colour as RGBA bytes, flipped so that row 0 of the image is the TOP (for canvas). */
export function targetToRGBA(t: Target, out = new Uint8ClampedArray(4 * t.w * t.h)): Uint8ClampedArray {
  for (let j = 0; j < t.h; j++) for (let i = 0; i < t.w; i++) {
    const p = j * t.w + i, q = ((t.h - 1 - j) * t.w + i) * 4;
    out[q] = 255 * t.color[3 * p]; out[q + 1] = 255 * t.color[3 * p + 1]; out[q + 2] = 255 * t.color[3 * p + 2]; out[q + 3] = 255;
  }
  return out;
}
