// Small 4x4 matrix and 3-vector maths for the WebGL renderer (lib/gl3d.ts) and for demos
// that show the matrices themselves. Conventions are WebGL/OpenGL ones:
//  - matrices are Float32Array(16) in COLUMN-MAJOR order: element (row r, column c) is m[4*c + r];
//  - vectors are columns, so a point is transformed as M * p and `multiply(a, b)` = a * b
//    (b is applied first);
//  - view space is right-handed, the camera looks down -z; NDC z is in [-1, 1].
// No DOM: safe to import from tests.

export type Mat4 = Float32Array;
export type Mat3 = Float32Array;
export type Vec3 = [number, number, number];

/** The identity matrix (a new array). */
export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** Element (row r, column c). */
export const at = (m: Mat4, r: number, c: number): number => m[4 * c + r];

/** Build a matrix from 16 numbers given ROW by row (as written on paper). */
export function fromRows(rows: number[]): Mat4 {
  const m = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[4 * c + r] = rows[4 * r + c];
  return m;
}

/** a * b (apply b first, then a). */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[4 * k + r] * b[4 * c + k];
    o[4 * c + r] = s;
  }
  return o;
}

/** Product of several matrices, left to right: mul(A, B, C) = A * B * C. */
export function mul(...ms: Mat4[]): Mat4 {
  return ms.reduce((acc, m) => multiply(acc, m), identity());
}

export function transpose(m: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[4 * c + r] = m[4 * r + c];
  return o;
}

export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

export function scaling(x: number, y = x, z = x): Mat4 {
  const m = identity();
  m[0] = x; m[5] = y; m[10] = z;
  return m;
}

/** Rotation by `angle` radians about the x, y or z axis (right-handed: counter-clockwise
 *  when the axis points at the viewer). */
export function rotationX(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle), m = identity();
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}
export function rotationY(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle), m = identity();
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}
export function rotationZ(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle), m = identity();
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
  return m;
}

/** Rotation by `angle` radians about an arbitrary axis (Rodrigues). */
export function rotation(axis: Vec3, angle: number): Mat4 {
  const [x, y, z] = normalize(axis), c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return fromRows([
    t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
    0, 0, 0, 1,
  ]);
}

/** Perspective projection (as gluPerspective): vertical field of view `fovy` in radians.
 *  Maps the view-space frustum to the NDC cube [-1, 1]^3 (near plane to z = -1). */
export function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2), m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = (far + near) / (near - far); m[11] = -1;
  m[14] = 2 * far * near / (near - far);
  return m;
}

/** Orthographic projection of the box [l,r] x [b,t] x [-near,-far] to the NDC cube. */
export function ortho(l: number, r: number, b: number, t: number, near: number, far: number): Mat4 {
  const m = identity();
  m[0] = 2 / (r - l); m[5] = 2 / (t - b); m[10] = -2 / (far - near);
  m[12] = -(r + l) / (r - l); m[13] = -(t + b) / (t - b); m[14] = -(far + near) / (far - near);
  return m;
}

/** View matrix of a camera at `eye` looking at `target` (as gluLookAt): V = R * T(-eye). */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]): Mat4 {
  const zA = normalize(sub(eye, target));          // camera z axis (points backwards)
  let xA = cross(up, zA);
  if (length(xA) < 1e-9) xA = cross([0, 0, 1], zA); // up parallel to the view direction
  xA = normalize(xA);
  const yA = cross(zA, xA);
  return fromRows([
    xA[0], xA[1], xA[2], -dot(xA, eye),
    yA[0], yA[1], yA[2], -dot(yA, eye),
    zA[0], zA[1], zA[2], -dot(zA, eye),
    0, 0, 0, 1,
  ]);
}

/** Inverse of a general 4x4 matrix (cofactors). Returns null if singular. */
export function invert(m: Mat4): Mat4 | null {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3], a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11], a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  const d = 1 / det, o = new Float32Array(16);
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
  return o;
}

/** The normal matrix M^{-T} restricted to 3x3 (column-major Float32Array(9)): transforms
 *  normals so they stay perpendicular to transformed surfaces under non-uniform scaling. */
export function normalMatrix(m: Mat4): Mat3 {
  const a = m[0], b = m[4], c = m[8], d = m[1], e = m[5], f = m[9], g = m[2], h = m[6], i = m[10];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const k = det ? 1 / det : 0;
  // inverse (row-major) = adj / det; its transpose in column-major has the same layout as
  // the inverse in row-major order.
  const inv = [
    A, -(b * i - c * h), b * f - c * e,
    B, a * i - c * g, -(a * f - c * d),
    C, -(a * h - b * g), a * e - b * d,
  ];
  return Float32Array.from(inv, v => v * k);
}

/** Upper-left 3x3 of a 4x4 matrix (column-major), e.g. to transform directions. */
export function upper3(m: Mat4): Mat3 {
  return Float32Array.from([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
}

/** M * (x, y, z, w). */
export function transformVec4(m: Mat4, v: [number, number, number, number]): [number, number, number, number] {
  const o: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
  return o;
}

/** Transform a point (w = 1) and divide by the resulting w. */
export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const [x, y, z, w] = transformVec4(m, [p[0], p[1], p[2], 1]);
  return [x / w, y / w, z / w];
}

/** Transform a direction (w = 0): no translation. */
export function transformDir(m: Mat4, v: Vec3): Vec3 {
  const [x, y, z] = transformVec4(m, [v[0], v[1], v[2], 0]);
  return [x, y, z];
}

/** 3x3 (column-major) times a vector. */
export function mat3Vec(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

/** Matrix as text, one row per line, fixed decimals and aligned columns (for readouts). */
export function formatMat(m: Mat4 | Mat3, decimals = 2): string {
  const n = m.length === 9 ? 3 : 4;
  const cells: string[][] = [];
  for (let r = 0; r < n; r++) {
    const row: string[] = [];
    for (let c = 0; c < n; c++) {
      const v = m[n * c + r];
      row.push((Math.abs(v) < 0.5 * 10 ** -decimals ? 0 : v).toFixed(decimals));
    }
    cells.push(row);
  }
  const w = Math.max(...cells.flat().map(s => s.length));
  return cells.map(r => r.map(s => s.padStart(w)).join("  ")).join("\n");
}

// ---- 3-vectors ----
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export function normalize(a: Vec3): Vec3 {
  const l = length(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}
