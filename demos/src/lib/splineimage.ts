// Splines on images (owner: the splines day). No DOM.
//
// - The cubic B-spline prefilter (Unser, Aldroubi & Eden 1991): coefficients c with
//   (c_{k−1} + 4 c_k + c_{k+1}) / 6 = f_k, as a causal + anti-causal first-order recursion
//   with pole z₁ = √3 − 2, mirror (whole-sample symmetric) boundary; 2D = rows then columns.
// - Samplers at real (x, y) (pixel centres at integers): nearest, bilinear, cubic convolution
//   (Keys, a = −½) and cubic B-spline (from coefficients; prefilter off = plain smoothing), plus
//   the exact gradient of the cubic B-spline interpolant.
// - Frequency responses of the interpolators (for the "quality in frequency" plot).
// - Free-form deformation (FFD) on a uniform control grid: displacement field and Jacobian
//   determinant, degree 3 (cubic B-spline) or degree 1 (bilinear) for comparison.
import { Grey, createGrey, borderIndex, Border } from "./imageops";
import { bspline3, bspline3d, bspline1 } from "./spline";

export const Z1 = Math.sqrt(3) - 2;

/** In place: interpolation coefficients of one line (mirror boundary). */
export function prefilterLine(c: Float64Array): void {
  const n = c.length, z = Z1;
  if (n === 1) return;
  for (let k = 0; k < n; k++) c[k] *= 6; // gain (1 − z)(1 − 1/z)
  // causal initialisation, exact for the mirror extension (period 2n − 2)
  const zn = Math.pow(z, n - 1);
  let s = c[0] + zn * c[n - 1], zk = z;
  const z2n = zn * zn;
  for (let k = 1; k < n - 1; k++) { s += (zk + z2n / zk) * c[k]; zk *= z; }
  c[0] = s / (1 - z2n);
  for (let k = 1; k < n; k++) c[k] += z * c[k - 1];
  // anti-causal initialisation
  c[n - 1] = (z / (z * z - 1)) * (c[n - 1] + z * c[n - 2]);
  for (let k = n - 2; k >= 0; k--) c[k] = z * (c[k + 1] - c[k]);
}

/** Cubic B-spline coefficients of an image (prefilter along rows, then columns). */
export function bsplineCoeffs(img: Grey): Grey {
  const { width: w, height: h } = img, out = createGrey(w, h);
  const row = new Float64Array(w), col = new Float64Array(h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = img.data[y * w + x];
    prefilterLine(row);
    for (let x = 0; x < w; x++) out.data[y * w + x] = row[x];
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = out.data[y * w + x];
    prefilterLine(col);
    for (let y = 0; y < h; y++) out.data[y * w + x] = col[y];
  }
  return out;
}

/** 1D coefficients of a signal (copy). */
export function bsplineCoeffs1d(f: ArrayLike<number>): Float64Array {
  const c = Float64Array.from(f);
  prefilterLine(c);
  return c;
}

/** Σ c_k β³(x − k) for a 1D coefficient array (mirror boundary). */
export function bsplineValue1d(c: ArrayLike<number>, x: number): number {
  const k0 = Math.floor(x) - 1;
  let s = 0;
  for (let k = k0; k <= k0 + 3; k++) s += c[borderIndex(k, c.length, "mirror")] * bspline3(x - k);
  return s;
}

/** Cubic B-spline value at real (x, y) from coefficients (16 terms, mirror boundary). */
export function sampleBSpline(coef: Grey, x: number, y: number, border: Border = "mirror"): number {
  const { width: w, height: h, data } = coef;
  const x0 = Math.floor(x) - 1, y0 = Math.floor(y) - 1;
  const wx = [0, 0, 0, 0], wy = [0, 0, 0, 0], ix = [0, 0, 0, 0], iy = [0, 0, 0, 0];
  for (let k = 0; k < 4; k++) {
    wx[k] = bspline3(x - (x0 + k)); wy[k] = bspline3(y - (y0 + k));
    ix[k] = borderIndex(x0 + k, w, border); iy[k] = borderIndex(y0 + k, h, border);
  }
  let s = 0;
  for (let j = 0; j < 4; j++) {
    if (iy[j] < 0) continue;
    let r = 0;
    for (let i = 0; i < 4; i++) if (ix[i] >= 0) r += wx[i] * data[iy[j] * w + ix[i]];
    s += wy[j] * r;
  }
  return s;
}

/** Exact gradient (∂/∂x, ∂/∂y) of the cubic B-spline surface at (x, y). */
export function gradBSpline(coef: Grey, x: number, y: number): [number, number] {
  const { width: w, height: h, data } = coef;
  const x0 = Math.floor(x) - 1, y0 = Math.floor(y) - 1;
  let gx = 0, gy = 0;
  for (let j = 0; j < 4; j++) {
    const yy = borderIndex(y0 + j, h, "mirror"), by = bspline3(y - (y0 + j)), dby = bspline3d(y - (y0 + j));
    for (let i = 0; i < 4; i++) {
      const xx = borderIndex(x0 + i, w, "mirror"), v = data[yy * w + xx];
      gx += bspline3d(x - (x0 + i)) * by * v;
      gy += bspline3(x - (x0 + i)) * dby * v;
    }
  }
  return [gx, gy];
}

/** Keys' cubic convolution kernel (a = −½): interpolating, C¹, reproduces quadratics. */
export function keys(x: number, a = -0.5): number {
  const t = Math.abs(x);
  if (t < 1) return (a + 2) * t * t * t - (a + 3) * t * t + 1;
  if (t < 2) return a * t * t * t - 5 * a * t * t + 8 * a * t - 4 * a;
  return 0;
}

/** Cubic-convolution value at real (x, y) (4 × 4 neighbourhood). */
export function sampleKeys(img: Grey, x: number, y: number, border: Border = "mirror"): number {
  const { width: w, height: h, data } = img;
  const x0 = Math.floor(x) - 1, y0 = Math.floor(y) - 1;
  let s = 0;
  for (let j = 0; j < 4; j++) {
    const iy = borderIndex(y0 + j, h, border);
    if (iy < 0) continue;
    const ky = keys(y - (y0 + j));
    let r = 0;
    for (let i = 0; i < 4; i++) { const ix = borderIndex(x0 + i, w, border); if (ix >= 0) r += keys(x - (x0 + i)) * data[iy * w + ix]; }
    s += ky * r;
  }
  return s;
}

export type SplineInterp = "nearest" | "bilinear" | "keys" | "bspline" | "bspline-nopre";

export const INTERP_LABEL: Record<SplineInterp, string> = {
  nearest: "nearest", bilinear: "bilinear", keys: "cubic convolution", bspline: "cubic B-spline", "bspline-nopre": "B-spline, no prefilter",
};

/** A sampler (x, y) → value for an image, preparing coefficients once when needed. */
export function sampler(img: Grey, interp: SplineInterp, border: Border = "mirror"): (x: number, y: number) => number {
  const { width: w, height: h, data } = img;
  const px = (x: number, y: number) => {
    const ix = borderIndex(x, w, border), iy = borderIndex(y, h, border);
    return ix < 0 || iy < 0 ? 0 : data[iy * w + ix];
  };
  switch (interp) {
    case "nearest": return (x, y) => px(Math.round(x), Math.round(y));
    case "bilinear": return (x, y) => {
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      return (1 - fy) * ((1 - fx) * px(x0, y0) + fx * px(x0 + 1, y0)) + fy * ((1 - fx) * px(x0, y0 + 1) + fx * px(x0 + 1, y0 + 1));
    };
    case "keys": return (x, y) => sampleKeys(img, x, y, border);
    case "bspline": { const c = bsplineCoeffs(img); return (x, y) => sampleBSpline(c, x, y, border); }
    case "bspline-nopre": return (x, y) => sampleBSpline(img, x, y, border);
  }
}

/** Inverse-mapping warp with any sampler; points that map outside [−0.5, size − 0.5] get `outside`. */
export function warpWith(img: Grey, map: (x: number, y: number) => [number, number], interp: SplineInterp, outside: number | null = null): Grey {
  const { width: w, height: h } = img, out = createGrey(w, h), s = sampler(img, interp);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [sx, sy] = map(x, y);
    out.data[y * w + x] = outside !== null && (sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5) ? outside : s(sx, sy);
  }
  return out;
}

/** Rotate about the centre by `angle` (radians, counter-clockwise on screen) with a sampler. */
export function rotateWith(img: Grey, angle: number, interp: SplineInterp): Grey {
  const cx = (img.width - 1) / 2, cy = (img.height - 1) / 2, c = Math.cos(angle), s = Math.sin(angle);
  return warpWith(img, (x, y) => { const dx = x - cx, dy = y - cy; return [cx + c * dx - s * dy, cy + s * dx + c * dy]; }, interp);
}

// ---------------------------------------------------------------------------------------
// Frequency responses (ω in radians per sample, 0 … π is the band)

const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

/** Frequency response of the continuous interpolation kernel at ω (rad/sample). */
export function interpResponse(interp: SplineInterp, omega: number): number {
  const f = omega / (2 * Math.PI);
  switch (interp) {
    case "nearest": return sinc(f);
    case "bilinear": return sinc(f) ** 2;
    case "bspline-nopre": return sinc(f) ** 4;
    case "bspline": return sinc(f) ** 4 / ((2 + Math.cos(omega)) / 3);
    case "keys": {
      // numeric Fourier transform of the even, compactly supported kernel
      let s = 0;
      const N = 400;
      for (let k = 0; k < N; k++) { const x = (k + 0.5) * 2 / N; s += keys(x) * Math.cos(omega * x) * 2 / N; }
      return 2 * s;
    }
  }
}

// ---------------------------------------------------------------------------------------
// Free-form deformation on a uniform control grid

export interface FFDGrid {
  /** spacing h in pixels; control point (k, l) sits at (k h, l h), k = −1 … nx, l = −1 … ny */
  h: number;
  nx: number; ny: number;
  /** displacements, index ((l + 1) * (nx + 2) + (k + 1)) */
  dx: Float64Array; dy: Float64Array;
  degree: 1 | 3;
}

/** A grid covering a w × h image (one extra ring of control points outside). */
export function ffdGrid(w: number, hgt: number, spacing: number, degree: 1 | 3 = 3): FFDGrid {
  const nx = Math.round((w - 1) / spacing) + 1, ny = Math.round((hgt - 1) / spacing) + 1;
  const n = (nx + 2) * (ny + 2);
  return { h: spacing, nx, ny, dx: new Float64Array(n), dy: new Float64Array(n), degree };
}

export const ffdIndex = (g: FFDGrid, k: number, l: number): number => (l + 1) * (g.nx + 2) + (k + 1);

/** Displacement (u, v) at pixel position (x, y), and optionally the Jacobian matrix of the
 *  displacement [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]. */
export function ffdAt(g: FFDGrid, x: number, y: number, jac?: number[]): [number, number] {
  const h = g.h, sx = x / h, sy = y / h;
  const B = g.degree === 3 ? bspline3 : bspline1;
  const dB = g.degree === 3 ? bspline3d : (t: number) => (Math.abs(t) < 1 ? -Math.sign(t) : 0);
  const r = g.degree === 3 ? 2 : 1;
  const k0 = Math.ceil(sx - r), l0 = Math.ceil(sy - r);
  let u = 0, v = 0, ux = 0, uy = 0, vx = 0, vy = 0;
  for (let l = l0; l <= l0 + 2 * r; l++) {
    if (l < -1 || l > g.ny) continue;
    const by = B(sy - l), dby = dB(sy - l) / h;
    for (let k = k0; k <= k0 + 2 * r; k++) {
      if (k < -1 || k > g.nx) continue;
      const bx = B(sx - k), dbx = dB(sx - k) / h, i = ffdIndex(g, k, l);
      const wgt = bx * by;
      u += wgt * g.dx[i]; v += wgt * g.dy[i];
      if (jac) { ux += dbx * by * g.dx[i]; uy += bx * dby * g.dx[i]; vx += dbx * by * g.dy[i]; vy += bx * dby * g.dy[i]; }
    }
  }
  if (jac) { jac[0] = ux; jac[1] = uy; jac[2] = vx; jac[3] = vy; }
  return [u, v];
}

/** Jacobian determinant det(I + ∇u) of the map x → x + u(x) at (x, y). */
export function ffdJacobianDet(g: FFDGrid, x: number, y: number): number {
  const J = [0, 0, 0, 0];
  ffdAt(g, x, y, J);
  return (1 + J[0]) * (1 + J[3]) - J[1] * J[2];
}

/** Displacement field on the pixel grid (separable sums), u and v as w × h arrays. */
export function ffdField(g: FFDGrid, w: number, hgt: number): { u: Float32Array; v: Float32Array } {
  const B = g.degree === 3 ? bspline3 : bspline1, nk = g.nx + 2, nl = g.ny + 2;
  const bx = new Float64Array(w * nk), by = new Float64Array(hgt * nl);
  for (let x = 0; x < w; x++) for (let k = -1; k <= g.nx; k++) bx[x * nk + k + 1] = B(x / g.h - k);
  for (let y = 0; y < hgt; y++) for (let l = -1; l <= g.ny; l++) by[y * nl + l + 1] = B(y / g.h - l);
  const u = new Float32Array(w * hgt), v = new Float32Array(w * hgt);
  const rowU = new Float64Array(nk), rowV = new Float64Array(nk);
  for (let y = 0; y < hgt; y++) {
    rowU.fill(0); rowV.fill(0);
    for (let l = 0; l < nl; l++) {
      const b = by[y * nl + l];
      if (!b) continue;
      for (let k = 0; k < nk; k++) { rowU[k] += b * g.dx[l * nk + k]; rowV[k] += b * g.dy[l * nk + k]; }
    }
    for (let x = 0; x < w; x++) {
      let su = 0, sv = 0;
      for (let k = 0; k < nk; k++) { const b = bx[x * nk + k]; if (b) { su += b * rowU[k]; sv += b * rowV[k]; } }
      u[y * w + x] = su; v[y * w + x] = sv;
    }
  }
  return { u, v };
}

/** Deform an image forward: the content at x moves to T(x) = x + u(x). Rendered by inverse
 *  mapping: for each output pixel y, solve s + u(s) = y by the fixed-point iteration
 *  s ← y − u(s) (converges while the map does not fold), then sample the input at s.
 *  Pixels whose source falls outside the image get 0. */
export function ffdWarp(img: Grey, g: FFDGrid, interp: SplineInterp = "bilinear", iterations = 6): Grey {
  const { width: w, height: hgt } = img, F = ffdField(g, w, hgt);
  const fu = { width: w, height: hgt, data: F.u } as Grey, fv = { width: w, height: hgt, data: F.v } as Grey;
  const lin = (G: Grey, x: number, y: number) => {
    const xc = Math.min(Math.max(x, 0), w - 1), yc = Math.min(Math.max(y, 0), hgt - 1);
    const x0 = Math.min(Math.floor(xc), w - 2), y0 = Math.min(Math.floor(yc), hgt - 2), ax = xc - x0, ay = yc - y0, d = G.data;
    return (1 - ay) * ((1 - ax) * d[y0 * w + x0] + ax * d[y0 * w + x0 + 1]) + ay * ((1 - ax) * d[(y0 + 1) * w + x0] + ax * d[(y0 + 1) * w + x0 + 1]);
  };
  return warpWith(img, (x, y) => {
    let sx = x - F.u[y * w + x], sy = y - F.v[y * w + x];
    for (let it = 1; it < iterations; it++) { const nx = x - lin(fu, sx, sy), ny = y - lin(fv, sx, sy); sx = nx; sy = ny; }
    return [sx, sy];
  }, interp, 0);
}
