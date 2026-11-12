// Polar and log-polar resampling (no DOM), with bilinear interpolation.
//
// The unwrapped image has one COLUMN per angle and one ROW per radius:
//   column j ↔ angle θ_j = θ0 + 2π j / nTheta   (x right, y down: θ increases clockwise on screen)
//   row i    ↔ radius r_i = rMin + (rMax − rMin) i / (nR − 1)                (linear)
//             r_i = rMin · (rMax / rMin)^(i / (nR − 1))                      (log, rMin > 0)
// and samples the input at (cx + r cos θ, cy + r sin θ). With log spacing, scaling the
// image about the centre shifts rows and rotating it shifts columns (circularly), which
// is the trick behind Fourier–Mellin registration.

import { Grey, createGrey, sampleBilinear, Border } from "./imageops";

export interface PolarOptions {
  cx: number;           // centre (pixel coordinates)
  cy: number;
  rMax: number;
  rMin?: number;        // default 0 (linear) or 1 (log)
  nR: number;           // rows (radii)
  nTheta: number;       // columns (angles), covering a full turn
  log?: boolean;        // log-polar spacing of the radii
  theta0?: number;      // angle of column 0 (default 0: pointing right)
  border?: Border;      // input outside the image (default "zero")
}

const rMinOf = (o: PolarOptions) => o.rMin ?? (o.log ? 1 : 0);

/** Radius of row i. */
export function radiusAt(o: PolarOptions, i: number): number {
  const r0 = rMinOf(o), t = o.nR > 1 ? i / (o.nR - 1) : 0;
  return o.log ? r0 * Math.pow(o.rMax / r0, t) : r0 + (o.rMax - r0) * t;
}

/** Angle of column j. */
export const angleAt = (o: PolarOptions, j: number): number => (o.theta0 ?? 0) + 2 * Math.PI * j / o.nTheta;

/** Real row index of radius r (inverse of radiusAt). */
export function rowOf(o: PolarOptions, r: number): number {
  const r0 = rMinOf(o);
  if (o.nR <= 1) return 0;
  return o.log ? (o.nR - 1) * Math.log(r / r0) / Math.log(o.rMax / r0) : (o.nR - 1) * (r - r0) / (o.rMax - r0);
}

/** Real column index of angle θ, in [0, nTheta). */
export function colOf(o: PolarOptions, theta: number): number {
  const t = (theta - (o.theta0 ?? 0)) / (2 * Math.PI) * o.nTheta;
  return ((t % o.nTheta) + o.nTheta) % o.nTheta;
}

/** Unwrap: nTheta × nR image (width = angles, height = radii). */
export function toPolar(img: Grey, o: PolarOptions): Grey {
  if (o.log && !(rMinOf(o) > 0)) throw new Error("toPolar: log spacing needs rMin > 0");
  const out = createGrey(o.nTheta, o.nR), border = o.border ?? "zero";
  const cs = new Float64Array(o.nTheta), sn = new Float64Array(o.nTheta);
  for (let j = 0; j < o.nTheta; j++) { const a = angleAt(o, j); cs[j] = Math.cos(a); sn[j] = Math.sin(a); }
  for (let i = 0; i < o.nR; i++) {
    const r = radiusAt(o, i);
    for (let j = 0; j < o.nTheta; j++)
      out.data[i * o.nTheta + j] = sampleBilinear(img, o.cx + r * cs[j], o.cy + r * sn[j], border);
  }
  return out;
}

/**
 * Back to Cartesian: a w × h image whose pixel (x, y) takes the polar image at its
 * (r, θ), bilinear in r and θ (θ wraps around). Outside [rMin, rMax] → `fill`.
 */
export function fromPolar(polar: Grey, w: number, h: number, o: PolarOptions, fill = 0): Grey {
  const out = createGrey(w, h), nT = o.nTheta, r0 = rMinOf(o);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x - o.cx, dy = y - o.cy, r = Math.hypot(dx, dy);
    if (r < r0 - 1e-9 || r > o.rMax + 1e-9 || (o.log && r <= 0)) { out.data[y * w + x] = fill; continue; }
    const fi = Math.min(o.nR - 1, Math.max(0, rowOf(o, Math.max(r, r0)))), fj = colOf(o, Math.atan2(dy, dx));
    const i0 = Math.floor(fi), j0 = Math.floor(fj), a = fi - i0, b = fj - j0;
    const i1 = Math.min(o.nR - 1, i0 + 1), j1 = (j0 + 1) % nT, jj0 = j0 % nT;
    const p = polar.data;
    out.data[y * w + x] = (1 - a) * ((1 - b) * p[i0 * nT + jj0] + b * p[i0 * nT + j1]) + a * ((1 - b) * p[i1 * nT + jj0] + b * p[i1 * nT + j1]);
  }
  return out;
}

/** Mean over angles for every radius: the radial profile (length nR) of a polar image. */
export function radialProfile(polar: Grey): Float64Array {
  const p = new Float64Array(polar.height);
  for (let i = 0; i < polar.height; i++) {
    let s = 0;
    for (let j = 0; j < polar.width; j++) s += polar.data[i * polar.width + j];
    p[i] = s / polar.width;
  }
  return p;
}
