// 3D images (volumes) for the course demos (owner: marching-cubes day, unit `volumes3d/`).
// No DOM: safe to import from tests.
//
// A volume is a 3D array of values plus the geometry that places it in the world:
//  - values stored (z, y, x): the value of voxel (x, y, z) is data[(z * ny + y) * nx + x];
//    image axes as in `images/`: x right, y down (row), z along the stack;
//  - voxel spacing (Δx, Δy, Δz) in physical units (µm, mm), often anisotropic (Δz > Δx);
//  - the world position of voxel (0, 0, 0) (`origin`) and, optionally, the directions of the
//    three voxel axes (`axes`, a row-major 3×3 rotation; default the identity).
// The voxel → world map is the 4×4 affine A = [R·diag(Δ) | origin]; world → voxel is A⁻¹.
//
//   const v = nucleiStack();                       // anisotropic synthetic stack
//   const A = voxelToWorld(v);                     // row-major 4×4
//   const p = applyAffine(A, [2, 3, 1]);           // world position of voxel (2, 3, 1)
//   const b = gaussianBlur3d(v, 0.4);              // σ = 0.4 µm in every direction
//   const m = mip(v, "z");                         // maximum along z, a 2D image
//   const s = reslice(v, plane);                   // oblique slice by trilinear interpolation
import { rng, gauss, Rng } from "./random";
import { gaussianKernel } from "./imageops";
import { inverse } from "./linalg";

export type Vec3 = [number, number, number];
export type Axis = "x" | "y" | "z";

export interface Volume {
  nx: number;
  ny: number;
  nz: number;
  /** Values stored (z, y, x). */
  data: Float32Array;
  /** Voxel spacing (Δx, Δy, Δz) in physical units. */
  spacing: Vec3;
  /** World position of voxel (0, 0, 0). */
  origin: Vec3;
  /** Optional directions of the voxel axes: row-major 3×3 rotation (columns = x, y, z axes). */
  axes?: number[];
  /** Physical unit of the spacing, for labels (default "µm"). */
  unit?: string;
}

/** A 2D image (row-major, y down), as returned by slices and projections. */
export interface Slice2D { width: number; height: number; data: Float32Array }

// ------------------------------------------------------------------------------------------
// Construction and access

export function createVolume(nx: number, ny: number, nz: number, spacing: Vec3 = [1, 1, 1], fill = 0, origin: Vec3 = [0, 0, 0]): Volume {
  const data = new Float32Array(nx * ny * nz);
  if (fill) data.fill(fill);
  return { nx, ny, nz, data, spacing: [...spacing] as Vec3, origin: [...origin] as Vec3 };
}

export function cloneVolume(v: Volume, data?: Float32Array): Volume {
  return { ...v, data: data ?? new Float32Array(v.data), spacing: [...v.spacing] as Vec3, origin: [...v.origin] as Vec3, axes: v.axes ? [...v.axes] : undefined };
}

/** Index of voxel (x, y, z) in `data`. */
export const vidx = (v: Volume, x: number, y: number, z: number): number => (z * v.ny + y) * v.nx + x;

/** Value at integer voxel (x, y, z), clamped to the volume (replicate border). */
export function voxel(v: Volume, x: number, y: number, z: number): number {
  x = x < 0 ? 0 : x >= v.nx ? v.nx - 1 : x;
  y = y < 0 ? 0 : y >= v.ny ? v.ny - 1 : y;
  z = z < 0 ? 0 : z >= v.nz ? v.nz - 1 : z;
  return v.data[(z * v.ny + y) * v.nx + x];
}

/** Volume from a function of the WORLD position of each voxel centre (spacing and origin applied). */
export function volumeFromFunction(nx: number, ny: number, nz: number, spacing: Vec3, f: (x: number, y: number, z: number) => number, origin: Vec3 = [0, 0, 0]): Volume {
  const v = createVolume(nx, ny, nz, spacing, 0, origin);
  let i = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
    v.data[i++] = f(origin[0] + x * spacing[0], origin[1] + y * spacing[1], origin[2] + z * spacing[2]);
  return v;
}

/** Minimum and maximum value. */
export function volumeRange(v: Volume): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v.data.length; i++) { const a = v.data[i]; if (a < lo) lo = a; if (a > hi) hi = a; }
  return [lo, hi];
}

/** Physical size of the volume along x, y, z (number of voxels × spacing). */
export const extentOf = (v: Volume): Vec3 => [v.nx * v.spacing[0], v.ny * v.spacing[1], v.nz * v.spacing[2]];

// ------------------------------------------------------------------------------------------
// Voxel <-> world

/** The voxel → world affine as a row-major 4×4 (16 numbers): x_world = A [i, j, k, 1]ᵀ. */
export function voxelToWorld(v: Volume): number[] {
  const R = v.axes ?? [1, 0, 0, 0, 1, 0, 0, 0, 1], s = v.spacing, o = v.origin;
  return [
    R[0] * s[0], R[1] * s[1], R[2] * s[2], o[0],
    R[3] * s[0], R[4] * s[1], R[5] * s[2], o[1],
    R[6] * s[0], R[7] * s[1], R[8] * s[2], o[2],
    0, 0, 0, 1,
  ];
}

/** The world → voxel affine A⁻¹ (row-major 4×4). */
export function worldToVoxel(v: Volume): number[] {
  const inv = inverse(voxelToWorld(v), 4);
  if (!inv) throw new Error("singular voxel-to-world matrix");
  return Array.from(inv);
}

/** Apply a row-major 4×4 affine to a point. */
export function applyAffine(A: ArrayLike<number>, p: ArrayLike<number>): Vec3 {
  return [
    A[0] * p[0] + A[1] * p[1] + A[2] * p[2] + A[3],
    A[4] * p[0] + A[5] * p[1] + A[6] * p[2] + A[7],
    A[8] * p[0] + A[9] * p[1] + A[10] * p[2] + A[11],
  ];
}

/** Format a row-major 4×4 with fixed decimals, one row per line. */
export function formatAffine(A: ArrayLike<number>, digits = 2): string {
  const s = Array.from(A, a => (Math.abs(a) < 1e-12 ? 0 : a).toFixed(digits));
  const w = Math.max(...s.map(t => t.length));
  return [0, 1, 2, 3].map(r => "[ " + s.slice(4 * r, 4 * r + 4).map(t => t.padStart(w)).join("  ") + " ]").join("\n");
}

// ------------------------------------------------------------------------------------------
// Interpolation

/** Trilinear interpolation at a (fractional) voxel position; outside: replicate border, or
 *  `outside` if given. The weighted sum of the 8 surrounding voxels. */
export function trilinear(v: Volume, x: number, y: number, z: number, outside?: number): number {
  const e = 1e-6;
  if (outside !== undefined && (x < -e || y < -e || z < -e || x > v.nx - 1 + e || y > v.ny - 1 + e || z > v.nz - 1 + e)) return outside;
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const fx = x - x0, fy = y - y0, fz = z - z0;
  let s = 0;
  for (let c = 0; c < 8; c++) {
    const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
    const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
    if (w !== 0) s += w * voxel(v, x0 + dx, y0 + dy, z0 + dz);
  }
  return s;
}

/** Trilinear interpolation of 8 corner values (corner k = x + 2y + 4z) at (fx, fy, fz) in [0, 1]³. */
export function trilinear8(c: ArrayLike<number>, fx: number, fy: number, fz: number): number {
  let s = 0;
  for (let k = 0; k < 8; k++) s += c[k] * (k & 1 ? fx : 1 - fx) * (k & 2 ? fy : 1 - fy) * (k & 4 ? fz : 1 - fz);
  return s;
}

/** Value at a WORLD position (world → voxel, then trilinear). */
export function sampleWorld(v: Volume, p: Vec3, outside?: number, inv = worldToVoxel(v)): number {
  const q = applyAffine(inv, p);
  return trilinear(v, q[0], q[1], q[2], outside);
}

// ------------------------------------------------------------------------------------------
// Slices, reslicing, projections

/** Orthogonal slice: "z" gives an xy image (width nx, height ny), "y" an xz image (width nx,
 *  height nz, z down), "x" a yz image (width ny, height nz). Plain array indexing. */
export function sliceAxis(v: Volume, axis: Axis, index: number): Slice2D {
  const { nx, ny, nz } = v;
  if (axis === "z") {
    const k = Math.max(0, Math.min(nz - 1, Math.round(index)));
    return { width: nx, height: ny, data: v.data.slice(k * nx * ny, (k + 1) * nx * ny) };
  }
  if (axis === "y") {
    const j = Math.max(0, Math.min(ny - 1, Math.round(index))), out = new Float32Array(nx * nz);
    for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) out[z * nx + x] = v.data[(z * ny + j) * nx + x];
    return { width: nx, height: nz, data: out };
  }
  const i = Math.max(0, Math.min(nx - 1, Math.round(index))), out = new Float32Array(ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) out[z * ny + y] = v.data[(z * ny + y) * nx + i];
  return { width: ny, height: nz, data: out };
}

/** A plane in world space: centre, two orthonormal in-plane directions u (image x) and
 *  w (image y), the output size in pixels and the pixel size in world units. */
export interface Plane { centre: Vec3; u: Vec3; w: Vec3; width: number; height: number; step: number }

/** World position of output pixel (i, j) of a plane (pixel centres, image y down along w). */
export function planePoint(p: Plane, i: number, j: number): Vec3 {
  const a = (i - (p.width - 1) / 2) * p.step, b = (j - (p.height - 1) / 2) * p.step;
  return [p.centre[0] + a * p.u[0] + b * p.w[0], p.centre[1] + a * p.u[1] + b * p.w[1], p.centre[2] + a * p.u[2] + b * p.w[2]];
}

/** Oblique reslicing: for every output pixel, world point → voxel (A⁻¹) → trilinear.
 *  Inverse mapping, as for 2D transforms in `images/`, one dimension up. */
export function reslice(v: Volume, p: Plane, outside = NaN): Slice2D {
  const inv = worldToVoxel(v), out = new Float32Array(p.width * p.height);
  for (let j = 0; j < p.height; j++) for (let i = 0; i < p.width; i++)
    out[j * p.width + i] = sampleWorld(v, planePoint(p, i, j), outside, inv);
  return { width: p.width, height: p.height, data: out };
}

/** Plane through `centre` with normal given by two tilt angles (radians): first about the
 *  y axis (tilting z towards x), then about the x axis. Zero tilt = an xy plane. */
export function tiltedPlane(centre: Vec3, tiltY: number, tiltX: number, width: number, height: number, step: number): Plane {
  const cy = Math.cos(tiltY), sy = Math.sin(tiltY), cx = Math.cos(tiltX), sx = Math.sin(tiltX);
  // rotation R = Rx(tiltX) * Ry(tiltY); columns are the rotated x and y axes.
  const u: Vec3 = [cy, sx * sy, -cx * sy];
  const w: Vec3 = [0, cx, sx];
  return { centre, u, w, width, height, step };
}

/** Resample to (nearly) isotropic voxels of size `step` (default: the smallest spacing),
 *  by trilinear interpolation; the physical extent stays the same. */
export function resampleIsotropic(v: Volume, step = Math.min(...v.spacing)): Volume {
  const n = (k: number) => Math.max(1, Math.round((k === 0 ? v.nx : k === 1 ? v.ny : v.nz) * v.spacing[k] / step));
  const nx = n(0), ny = n(1), nz = n(2);
  const out = createVolume(nx, ny, nz, [step, step, step], 0, v.origin);
  out.axes = v.axes ? [...v.axes] : undefined; out.unit = v.unit;
  // output voxel centre i*step maps to input index i*step/Δ (same origin)
  let q = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
    out.data[q++] = trilinear(v, x * step / v.spacing[0], y * step / v.spacing[1], z * step / v.spacing[2]);
  return out;
}

/** Maximum-intensity projection along an axis (same image layout as `sliceAxis`). */
export function mip(v: Volume, axis: Axis = "z"): Slice2D {
  const { nx, ny, nz } = v;
  const W = axis === "x" ? ny : nx, H = axis === "z" ? ny : nz;
  const out = new Float32Array(W * H).fill(-Infinity);
  let i = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const a = v.data[i++];
    const o = axis === "z" ? y * nx + x : axis === "y" ? z * nx + x : z * ny + y;
    if (a > out[o]) out[o] = a;
  }
  return { width: W, height: H, data: out };
}

// ------------------------------------------------------------------------------------------
// Filters

function blur1d(src: Float32Array, dst: Float32Array, v: Volume, axis: 0 | 1 | 2, k: Float64Array): void {
  const { nx, ny, nz } = v, r = (k.length - 1) / 2;
  const n = axis === 0 ? nx : axis === 1 ? ny : nz, stride = axis === 0 ? 1 : axis === 1 ? nx : nx * ny;
  const line = new Float64Array(n);
  const [A, B] = axis === 0 ? [ny, nz] : axis === 1 ? [nx, nz] : [nx, ny];
  for (let b = 0; b < B; b++) for (let a = 0; a < A; a++) {
    const base = axis === 0 ? (b * ny + a) * nx : axis === 1 ? b * nx * ny + a : b * nx + a;
    for (let i = 0; i < n; i++) line[i] = src[base + i * stride];
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = -r; t <= r; t++) { const j = i + t; s += k[t + r] * line[j < 0 ? 0 : j >= n ? n - 1 : j]; }
      dst[base + i * stride] = s;
    }
  }
}

/** Separable Gaussian blur with σ in PHYSICAL units (one value, or one per axis); the σ in
 *  voxels along axis i is σᵢ / Δᵢ, so an isotropic physical blur is anisotropic in voxels.
 *  Replicate border. σ ≤ 0 along an axis leaves that axis alone. */
export function gaussianBlur3d(v: Volume, sigma: number | Vec3): Volume {
  const s: Vec3 = typeof sigma === "number" ? [sigma, sigma, sigma] : sigma;
  let a = new Float32Array(v.data), b = new Float32Array(v.data.length);
  for (const ax of [0, 1, 2] as const) {
    const sv = s[ax] / v.spacing[ax];
    if (!(sv > 0.05)) continue;
    blur1d(a, b, v, ax, gaussianKernel(sv, 0, Math.max(1, Math.ceil(3 * sv))));
    [a, b] = [b, a];
  }
  return cloneVolume(v, a);
}

/** Gradient by central differences at every voxel, in value per PHYSICAL unit
 *  (one-sided at the border). `ignoreSpacing` divides by voxels instead (the classic mistake). */
export function gradient3d(v: Volume, ignoreSpacing = false): { gx: Float32Array; gy: Float32Array; gz: Float32Array } {
  const { nx, ny, nz, data } = v, n = data.length;
  const gx = new Float32Array(n), gy = new Float32Array(n), gz = new Float32Array(n);
  const [dx, dy, dz] = ignoreSpacing ? [1, 1, 1] : v.spacing;
  let i = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) {
    const xm = x > 0 ? 1 : 0, xp = x < nx - 1 ? 1 : 0, ym = y > 0 ? nx : 0, yp = y < ny - 1 ? nx : 0;
    const zm = z > 0 ? nx * ny : 0, zp = z < nz - 1 ? nx * ny : 0;
    gx[i] = (data[i + xp] - data[i - xm]) / ((xp + xm) * dx || 1);
    gy[i] = (data[i + yp] - data[i - ym]) / (((yp + ym) / nx) * dy || 1);
    gz[i] = (data[i + zp] - data[i - zm]) / (((zp + zm) / (nx * ny)) * dz || 1);
  }
  return { gx, gy, gz };
}

/** Unit normals −∇I/‖∇I‖ at fractional VOXEL positions (x, y, z triples), from the grid
 *  gradient interpolated trilinearly: a bright object gets outward normals. */
export function gradientNormals(v: Volume, voxelPositions: ArrayLike<number>, grad = gradient3d(v)): Float32Array {
  const n = new Float32Array(voxelPositions.length);
  const gv = (arr: Float32Array) => ({ ...v, data: arr });
  const GX = gv(grad.gx), GY = gv(grad.gy), GZ = gv(grad.gz);
  for (let i = 0; i < voxelPositions.length; i += 3) {
    const x = voxelPositions[i], y = voxelPositions[i + 1], z = voxelPositions[i + 2];
    const a = trilinear(GX, x, y, z), b = trilinear(GY, x, y, z), c = trilinear(GZ, x, y, z);
    const l = Math.hypot(a, b, c) || 1;
    n[i] = -a / l; n[i + 1] = -b / l; n[i + 2] = -c / l;
  }
  return n;
}

/** Pad with `width` voxels of `value` on every side (so surfaces touching the border close). */
export function padVolume(v: Volume, width = 1, value = 0): Volume {
  const nx = v.nx + 2 * width, ny = v.ny + 2 * width, nz = v.nz + 2 * width;
  const o: Vec3 = [v.origin[0] - width * v.spacing[0], v.origin[1] - width * v.spacing[1], v.origin[2] - width * v.spacing[2]];
  const out = createVolume(nx, ny, nz, v.spacing, value, o);
  out.unit = v.unit;
  for (let z = 0; z < v.nz; z++) for (let y = 0; y < v.ny; y++) for (let x = 0; x < v.nx; x++)
    out.data[((z + width) * ny + y + width) * nx + x + width] = v.data[(z * v.ny + y) * v.nx + x];
  return out;
}

/** Add Gaussian noise of standard deviation `sd` (seeded). */
export function addNoise3d(v: Volume, sd: number, seed = 1): Volume {
  const out = cloneVolume(v);
  if (sd <= 0) return out;
  const r: Rng = rng(seed);
  for (let i = 0; i < out.data.length; i++) out.data[i] += gauss(r, 0, sd);
  return out;
}

/** Binary mask (1 where value > c) as a volume. */
export function thresholdVolume(v: Volume, c: number): Volume {
  const out = cloneVolume(v, new Float32Array(v.data.length));
  for (let i = 0; i < v.data.length; i++) out.data[i] = v.data[i] > c ? 1 : 0;
  return out;
}

// ------------------------------------------------------------------------------------------
// Synthetic volumes (generated in code; no files)

/** Smooth 0..1 profile across a boundary at signed distance d (positive inside), edge width w. */
export const softStep = (d: number, w: number): number => w <= 0 ? (d > 0 ? 1 : 0) : 0.5 * (1 + Math.tanh(d / w));

/** An ellipsoid in world space: centre, semi-axes, rotation (row-major 3×3; columns = its axes). */
export interface Ellipsoid { centre: Vec3; radii: Vec3; rot?: number[]; value?: number }

/** Approximate signed distance (positive inside) of a point to an ellipsoid surface. */
export function ellipsoidDistance(e: Ellipsoid, p: Vec3): number {
  const d: Vec3 = [p[0] - e.centre[0], p[1] - e.centre[1], p[2] - e.centre[2]];
  const R = e.rot;
  const q = R ? [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]] : d;
  const k = Math.hypot(q[0] / e.radii[0], q[1] / e.radii[1], q[2] / e.radii[2]);
  return (1 - k) * Math.min(...e.radii);
}

/** Rotation about the y axis (row-major 3×3). */
export function rotY(a: number): number[] { const c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 1, 0, -s, 0, c]; }
/** Rotation about the x axis (row-major 3×3). */
export function rotX(a: number): number[] { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, -s, 0, s, c]; }

/** Sum of soft-edged ellipsoids on a background, sampled at voxel centres. */
export function ellipsoidVolume(nx: number, ny: number, nz: number, spacing: Vec3, objs: Ellipsoid[], o: { background?: number; edge?: number } = {}): Volume {
  const bg = o.background ?? 0, w = o.edge ?? 0.5 * Math.min(...spacing);
  return volumeFromFunction(nx, ny, nz, spacing, (x, y, z) => {
    let s = bg;
    for (const e of objs) s += (e.value ?? 1) * softStep(ellipsoidDistance(e, [x, y, z]), w);
    return s;
  });
}

/** The synthetic nuclei stack of `volumes3d/` (µm): 64 × 64 × 20 voxels of 0.2 × 0.2 × 0.5 µm,
 *  grey values about 10 (background) to 200 (nuclei), seeded noise. Nuclei A and B overlap in
 *  x-y but lie at different depths (they merge in the z-MIP, not in 3D); C is an elongated cell
 *  tilted 35° in the x-z plane; D is a small isolated nucleus. */
export const NUCLEI_OBJECTS: (Ellipsoid & { name: string })[] = [
  { name: "A", centre: [4.6, 6.0, 2.8], radii: [2.0, 2.0, 1.5], value: 190 },
  { name: "B", centre: [7.6, 6.6, 7.2], radii: [2.0, 2.0, 1.5], value: 190 },
  { name: "C", centre: [6.6, 10.6, 5.0], radii: [3.6, 1.2, 1.2], rot: rotY(-35 * Math.PI / 180), value: 170 },
  { name: "D", centre: [10.6, 2.4, 5.0], radii: [1.2, 1.2, 1.2], value: 160 },
];
export const NUCLEI_SPACING: Vec3 = [0.2, 0.2, 0.5];

export function nucleiStack(o: { noise?: number; seed?: number; nx?: number; ny?: number; nz?: number; spacing?: Vec3 } = {}): Volume {
  const sp = o.spacing ?? NUCLEI_SPACING;
  const v = ellipsoidVolume(o.nx ?? 64, o.ny ?? 64, o.nz ?? 20, sp, NUCLEI_OBJECTS, { background: 10, edge: 0.15 });
  v.unit = "µm";
  const out = addNoise3d(v, o.noise ?? 6, o.seed ?? 7);
  out.unit = "µm";
  return out;
}

/** Metaballs: Σ exp(−‖p − cᵢ‖² / 2s²) on an n³ grid with unit spacing (two blobs that merge
 *  as the iso-value drops). Centres in voxel units. */
export function metaballs(n: number, centres: Vec3[], s: number): Volume {
  return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => {
    let a = 0;
    for (const c of centres) a += Math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * s * s));
    return a;
  });
}

/** Signed-distance-like sphere: value R − ‖p − centre‖ (positive inside), unit spacing; the
 *  zero level set is the sphere. `centre` defaults to the middle of the grid. */
export function sphereSDF(n: number, R: number, centre?: Vec3): Volume {
  const c = centre ?? [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
  return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => R - Math.hypot(x - c[0], y - c[1], z - c[2]));
}

/** Torus (ring) around the z axis: value r − dist to the core circle of radius Rmaj (unit spacing). */
export function torusSDF(n: number, Rmaj: number, r: number, centre?: Vec3): Volume {
  const c = centre ?? [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
  return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => {
    const q = Math.hypot(x - c[0], y - c[1]) - Rmaj;
    return r - Math.hypot(q, z - c[2]);
  });
}

/** A ball of radius R (world units) as a grey image: 1 inside, 0 outside, with a smooth
 *  edge of width `edge` (0 = binary), sampled on an n³ grid with the given spacing, centred. */
export function ballVolume(n: number, R: number, spacing: Vec3 = [1, 1, 1], edge = 0, nz = n): Volume {
  const c: Vec3 = [(n - 1) / 2 * spacing[0], (n - 1) / 2 * spacing[1], (nz - 1) / 2 * spacing[2]];
  return volumeFromFunction(n, n, nz, spacing, (x, y, z) => softStep(R - Math.hypot(x - c[0], y - c[1], z - c[2]), edge));
}
