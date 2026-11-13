// Phasors (no DOM): the normalised first (or h-th) Fourier coefficient of a sampled periodic
// curve. Written for fluorescence decays (time bins over one laser period), and usable for
// spectra (wavelength bins: the spectral phasor of the unmixing day).
//
//   P = Σ_j y_j e^{+iω_h t_j} / Σ_j y_j,   t_j = (j + ½)Δ,  ω_h = 2πh / (nΔ)
//   g = Re P, s = Im P.
// Sign: the Fourier unit's c_k uses e^{−iωt}, so P = conj(c_1) / c_0 (s plotted upwards).
// A single exponential lies on the universal semicircle |P − ½| = ½; mixtures are convex
// combinations with photon fractions as weights; convolution with an IRF multiplies phasors.

import { histogram2d, Histogram2D } from "./histogram";

type A = ArrayLike<number>;
export interface Ph { g: number; s: number }

/** Phasor of a sampled periodic curve (bin centres), harmonic h. */
export function phasor(y: A, harmonic = 1): Ph {
  const n = y.length;
  let G = 0, S = 0, I = 0;
  for (let j = 0; j < n; j++) {
    const a = 2 * Math.PI * harmonic * (j + 0.5) / n;
    G += y[j] * Math.cos(a); S += y[j] * Math.sin(a); I += y[j];
  }
  return I > 0 ? { g: G / I, s: S / I } : { g: NaN, s: NaN };
}

/** Phasor maps of a time-major stack (stack[j * P + p]); also the total per pixel. */
export function phasorStack(stack: A, nBins: number, P: number, harmonic = 1): { g: Float32Array; s: Float32Array; I: Float32Array } {
  const g = new Float32Array(P), s = new Float32Array(P), I = new Float32Array(P);
  for (let j = 0; j < nBins; j++) {
    const a = 2 * Math.PI * harmonic * (j + 0.5) / nBins, c = Math.cos(a), sn = Math.sin(a), off = j * P;
    for (let p = 0; p < P; p++) { const v = stack[off + p]; g[p] += v * c; s[p] += v * sn; I[p] += v; }
  }
  for (let p = 0; p < P; p++) { if (I[p] > 0) { g[p] /= I[p]; s[p] /= I[p]; } else { g[p] = NaN; s[p] = NaN; } }
  return { g, s, I };
}

/** Phasor of one exponential, ωτ given: (1, ωτ) / (1 + ω²τ²). */
export function semicircle(omegaTau: number): Ph {
  const q = 1 + omegaTau * omegaTau;
  return { g: 1 / q, s: omegaTau / q };
}

/** Phase and modulation lifetimes of a phasor: τ_φ = s / (g ω), τ_m = √(1/m² − 1) / ω. */
export function lifetimeFromPhasor(p: Ph, omega: number): { tauPhi: number; tauM: number } {
  const m2 = p.g * p.g + p.s * p.s;
  return { tauPhi: p.s / (p.g * omega), tauM: m2 > 0 && m2 <= 1 ? Math.sqrt(1 / m2 - 1) / omega : NaN };
}

/** Phase lifetime of a phasor: τ_φ = s / (g ω). */
export const tauPhi = (p: Ph, omega: number): number => p.s / (p.g * omega);

// complex helpers
export const cmul = (a: Ph, b: Ph): Ph => ({ g: a.g * b.g - a.s * b.s, s: a.g * b.s + a.s * b.g });
export function cdiv(a: Ph, b: Ph): Ph {
  const d = b.g * b.g + b.s * b.s;
  return { g: (a.g * b.g + a.s * b.s) / d, s: (a.s * b.g - a.g * b.s) / d };
}

/** Calibration factor from a reference of known lifetime: K = P_ref,true / P_ref,measured.
 *  Multiply every measured phasor by K. */
export function calibrationFactor(refMeasured: Ph, tauRef: number, omega: number): Ph {
  return cdiv(semicircle(omega * tauRef), refMeasured);
}

/** Calibrate a phasor (or apply any complex factor). */
export const calibrate = (p: Ph, K: Ph): Ph => cmul(p, K);

/** Calibrate phasor maps in place. */
export function calibrateMaps(g: Float32Array, s: Float32Array, K: Ph): void {
  for (let p = 0; p < g.length; p++) { const a = g[p], b = s[p]; g[p] = a * K.g - b * K.s; s[p] = a * K.s + b * K.g; }
}

/** Effect of sampling with nBins bin integrals at bin centres: the phasor is multiplied by
 *  sinc(πh/n) = sin(πh/n) / (πh/n) (first order; aliased harmonics neglected). */
export function binningCorrection(nBins: number, harmonic = 1): number {
  const x = Math.PI * harmonic / nBins;
  return Math.sin(x) / x;
}

/** Position of p projected onto the segment a → b: 0 at a, 1 at b (lever rule). */
export function fractionOnSegment(p: Ph, a: Ph, b: Ph): number {
  const dx = b.g - a.g, dy = b.s - a.s, L2 = dx * dx + dy * dy;
  return L2 > 0 ? ((p.g - a.g) * dx + (p.s - a.s) * dy) / L2 : 0;
}

/** Barycentric coordinates of p in the triangle (a, b, c): fractions that sum to 1. */
export function barycentric3(p: Ph, a: Ph, b: Ph, c: Ph): [number, number, number] {
  const det = (b.g - a.g) * (c.s - a.s) - (c.g - a.g) * (b.s - a.s);
  if (Math.abs(det) < 1e-15) return [NaN, NaN, NaN];
  const l2 = ((p.g - a.g) * (c.s - a.s) - (c.g - a.g) * (p.s - a.s)) / det;
  const l3 = ((b.g - a.g) * (p.s - a.s) - (p.g - a.g) * (b.s - a.s)) / det;
  return [1 - l2 - l3, l2, l3];
}

/** Intersections of the line p0 + t·d with the semicircle (g − ½)² + s² = ¼ (s ≥ 0), as
 *  lifetimes (ascending), or [] if it misses. */
export function lineSemicircle(p0: Ph, d: Ph, omega: number): number[] {
  const ox = p0.g - 0.5, oy = p0.s;
  const A = d.g * d.g + d.s * d.s, B = 2 * (ox * d.g + oy * d.s), C = ox * ox + oy * oy - 0.25;
  const disc = B * B - 4 * A * C;
  if (A === 0 || disc < 0) return [];
  const out: number[] = [];
  for (const sg of [-1, 1]) {
    const t = (-B + sg * Math.sqrt(disc)) / (2 * A), g = p0.g + t * d.g, s = p0.s + t * d.s;
    if (s >= -1e-9 && g > 1e-9) out.push(Math.max(0, s) / (g * omega));
  }
  return out.sort((a, b) => a - b);
}

/** Principal axis of a (weighted) cloud of phasors: centroid and unit direction, and the two
 *  lifetimes where that line meets the semicircle. */
export function lineThroughCloud(g: A, s: A, omega: number, weights?: A): { centre: Ph; dir: Ph; taus: number[] } {
  let W = 0, mg = 0, ms = 0;
  for (let i = 0; i < g.length; i++) { const w = weights ? weights[i] : 1; if (!(w > 0) || g[i] !== g[i] || s[i] !== s[i]) continue; W += w; mg += w * g[i]; ms += w * s[i]; }
  mg /= W; ms /= W;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < g.length; i++) {
    const w = weights ? weights[i] : 1; if (!(w > 0) || g[i] !== g[i] || s[i] !== s[i]) continue;
    const dx = g[i] - mg, dy = s[i] - ms; sxx += w * dx * dx; sxy += w * dx * dy; syy += w * dy * dy;
  }
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const dir = { g: Math.cos(th), s: Math.sin(th) }, centre = { g: mg, s: ms };
  return { centre, dir, taus: lineSemicircle(centre, dir, omega) };
}

/** 2D histogram of phasor maps over g ∈ [gLo, gHi], s ∈ [0, sHi] (rows: s from 0 upwards). */
export function phasorHistogram(g: A, s: A, opts: { bins?: number; mask?: A; gLo?: number; gHi?: number; sHi?: number; weights?: A } = {}): Histogram2D {
  const bins = opts.bins ?? 100;
  const gLo = opts.gLo ?? 0, gHi = opts.gHi ?? 1, sHi = opts.sHi ?? 0.6;
  return histogram2d(g, s, { binsX: bins, binsY: Math.round(bins * sHi / (gHi - gLo)), minX: gLo, maxX: gHi, minY: 0, maxY: sHi, mask: opts.mask, weights: opts.weights });
}

/** Pixels whose phasor lies within `radius` of `centre`. */
export function cursorMask(g: A, s: A, centre: Ph, radius: number, mask?: A): Uint8Array {
  const out = new Uint8Array(g.length), r2 = radius * radius;
  for (let p = 0; p < g.length; p++) {
    if (mask && !mask[p]) continue;
    const dx = g[p] - centre.g, dy = s[p] - centre.s;
    if (dx * dx + dy * dy <= r2) out[p] = 1;
  }
  return out;
}
