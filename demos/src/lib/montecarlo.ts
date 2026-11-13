// Monte Carlo integration (owner: ray-tracing day, unit pathtrace/; no DOM).
//
//  - `Welford`: running mean and variance (numerically stable, one pass);
//  - hemisphere sampling around +z: `uniformHemisphere` (pdf 1/2π), `cosineHemisphere`
//    (Malley's method: a uniform point on the unit disc lifted to the hemisphere, pdf cos θ/π),
//    `sampleDisc`, `toWorld` (local z → a given normal);
//  - 1D: `TabulatedPdf` (piecewise-constant pdf, sampled by the inverse CDF), `stratified`,
//    `russianRoulette`;
//  - the 2D "half-circle" versions used by the mcpixel demo: directions are angles θ ∈ (−π/2, π/2)
//    from the normal; `uniformAngle` (pdf 1/π), `cosineAngle` (θ = asin(2u − 1), pdf cos θ / 2);
//  - `estimate(f, sample, n)`: the estimator (1/N) Σ f(X)/p(X).
import type { Vec3 } from "./mat4";

/** Running mean and variance (Welford). */
export class Welford {
  n = 0; mean = 0; m2 = 0;
  add(x: number): void { this.n++; const d = x - this.mean; this.mean += d / this.n; this.m2 += d * (x - this.mean); }
  /** Sample variance (n − 1 in the denominator). */
  variance(): number { return this.n > 1 ? this.m2 / (this.n - 1) : 0; }
  sd(): number { return Math.sqrt(this.variance()); }
  /** Standard error of the mean, sd/√n. */
  sem(): number { return this.n > 0 ? this.sd() / Math.sqrt(this.n) : 0; }
}

/** Uniform direction on the hemisphere z ≥ 0; pdf 1/(2π) per steradian. */
export function uniformHemisphere(u1: number, u2: number): Vec3 {
  const z = u1, r = Math.sqrt(Math.max(0, 1 - z * z)), phi = 2 * Math.PI * u2;
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

/** Uniform point on the unit disc (polar: r = √u1). */
export function sampleDisc(u1: number, u2: number): [number, number] {
  const r = Math.sqrt(u1), phi = 2 * Math.PI * u2;
  return [r * Math.cos(phi), r * Math.sin(phi)];
}

/** Cosine-weighted direction (Malley): a uniform disc point lifted to the hemisphere; pdf cos θ / π. */
export function cosineHemisphere(u1: number, u2: number): Vec3 {
  const [x, y] = sampleDisc(u1, u2);
  return [x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y))];
}

/** Rotate a local direction (z = up) into the frame of the unit normal n. */
export function toWorld(v: Vec3, n: Vec3): Vec3 {
  // Duff et al. (2017) orthonormal basis
  const s = n[2] >= 0 ? 1 : -1, a = -1 / (s + n[2]), b = n[0] * n[1] * a;
  const t: Vec3 = [1 + s * n[0] * n[0] * a, s * b, -s * n[0]];
  const bt: Vec3 = [b, s + n[1] * n[1] * a, -n[1]];
  return [v[0] * t[0] + v[1] * bt[0] + v[2] * n[0], v[0] * t[1] + v[1] * bt[1] + v[2] * n[1], v[0] * t[2] + v[1] * bt[2] + v[2] * n[2]];
}

/** A piecewise-constant pdf on [a, b] from (non-negative) weights per bin; sampled by the inverse CDF. */
export class TabulatedPdf {
  readonly cdf: Float64Array;
  readonly dens: Float64Array;
  readonly h: number;
  constructor(readonly a: number, readonly b: number, weights: ArrayLike<number>) {
    const n = weights.length, h = (b - a) / n;
    this.h = h;
    const tot = Array.from(weights).reduce((s, w) => s + Math.max(0, w), 0) || 1;
    this.cdf = new Float64Array(n + 1); this.dens = new Float64Array(n);
    for (let i = 0; i < n; i++) { const p = Math.max(0, weights[i]) / tot; this.dens[i] = p / h; this.cdf[i + 1] = this.cdf[i] + p; }
    this.cdf[n] = 1;
  }
  /** x = F⁻¹(u). */
  sample(u: number): number {
    const c = this.cdf;
    let lo = 0, hi = c.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c[m] <= u) lo = m; else hi = m; }
    const w = c[lo + 1] - c[lo];
    return this.a + this.h * (lo + (w > 0 ? (u - c[lo]) / w : 0.5));
  }
  pdf(x: number): number {
    if (x < this.a || x > this.b) return 0;
    const i = Math.min(this.dens.length - 1, Math.floor((x - this.a) / this.h));
    return this.dens[i];
  }
}

/** n stratified numbers in [0, 1): one random point in each of n equal strata. */
export function stratified(n: number, u: () => number): number[] {
  return Array.from({ length: n }, (_, i) => (i + u()) / n);
}

/** Russian roulette: continue with probability q; the caller divides the weight by q if true. */
export const russianRoulette = (q: number, u: number): boolean => u < q;

// ---- 2D (half-circle) sampling: angles from the normal in (−π/2, π/2) ----

/** Uniform angle; pdf 1/π. */
export const uniformAngle = (u: number): number => Math.PI * (u - 0.5);
export const uniformAnglePdf = (th: number): number => (Math.abs(th) <= Math.PI / 2 ? 1 / Math.PI : 0);
/** Cosine-weighted angle θ = asin(2u − 1); pdf cos θ / 2. */
export const cosineAngle = (u: number): number => Math.asin(2 * u - 1);
export const cosineAnglePdf = (th: number): number => (Math.abs(th) <= Math.PI / 2 ? Math.cos(th) / 2 : 0);

/** The Monte Carlo estimate (1/N) Σ f(X_k)/p(X_k) with X_k ~ p. */
export function estimate(f: (x: number) => number, sample: () => number, pdf: (x: number) => number, n: number): number {
  let s = 0;
  for (let k = 0; k < n; k++) { const x = sample(), p = pdf(x); if (p > 0) s += f(x) / p; }
  return s / n;
}
