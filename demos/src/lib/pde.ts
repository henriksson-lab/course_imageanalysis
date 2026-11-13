// Diffusion PDEs for smoothing (owner: the edge-preserving smoothing day, units heat-equation/,
// perona-malik/, tv-flow/, bilateral-nlm/, smoothers/). No DOM: also used by the tests.
//
// Images are row-major Float64Arrays of size w × h (h = 1 for a 1D profile). Grid spacing 1.
//   heatStep(u, w, h, dt)               // u + dt ∇²u, 5-point stencil; stable for dt ≤ 1/4 (2D), 1/2 (1D)
//   implicitStep1d(u, dt)               // (I + dt L) u_new = u  (Thomas, = gmrf1d of lib/smooth.ts)
//   implicitStep2d(u, w, h, dt)         // the same in 2D, Neumann border, diagonalised by the DCT
//   nonlinearStep(u, w, h, dt, "g2", K) // Perona–Malik: flux g(|d|) d between each pair of neighbours
//   tvFlowStep(u, w, h, dt, eps)        // u_t = div(∇u / |∇u|_eps); stable for dt ≤ eps/4
// Border "mirror" = no flux leaves the image (Neumann; the mean is kept); "zero" = the image is
// surrounded by value 0 (Dirichlet; heat leaks out).
//
// Also: the edge-stopping functions and their flux, the von Neumann amplification factor, the
// three energies of the energy picture (Dirichlet, TV, Perona–Malik), profile scores, a
// synthetic test card with ground truth, and the per-pixel weights of the bilateral filter
// and non-local means (weight maps, effective number of averaged pixels, joint bilateral).
// The weight helpers are candidates for lib/denoise.ts (owner: deconvolution day), the test
// card for lib/synth.ts.
import { gmrf1d, thomas, tv2dValue } from "./smooth";
import { dct2, idct2 } from "./dct";
import { fromArray, gaussianBlur, gaussianKernel } from "./imageops";
import { rng } from "./random";

export type HeatBorder = "mirror" | "zero";
export type Diffusivity = "linear" | "g1" | "g2";

// ---------------------------------------------------------------------------------------
// Linear diffusion

/** One explicit step of the heat equation u_t = ∇²u (5-point stencil; 1D if h = 1). */
export function heatStep(u: ArrayLike<number>, w: number, h: number, dt: number, border: HeatBorder = "mirror"): Float64Array {
  const out = new Float64Array(w * h);
  const z = border === "zero";
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, c = u[p];
    let s = 0;
    s += (x > 0 ? u[p - 1] : z ? 0 : c) - c;
    s += (x < w - 1 ? u[p + 1] : z ? 0 : c) - c;
    if (h > 1) {
      s += (y > 0 ? u[p - w] : z ? 0 : c) - c;
      s += (y < h - 1 ? u[p + w] : z ? 0 : c) - c;
    }
    out[p] = c + dt * s;
  }
  return out;
}

/** n explicit steps (convenience). */
export function heatSteps(u: ArrayLike<number>, w: number, h: number, dt: number, n: number, border: HeatBorder = "mirror"): Float64Array {
  let v: Float64Array = Float64Array.from(u);
  for (let k = 0; k < n; k++) v = heatStep(v, w, h, dt, border);
  return v;
}

/** One implicit (backward Euler) step in 1D: solve (I + dt L) v = u. Mirror border = free ends
 *  (exactly the quadratic smoother gmrf1d with λ = dt); zero border = value 0 outside. */
export function implicitStep1d(u: ArrayLike<number>, dt: number, border: HeatBorder = "mirror"): Float64Array {
  if (border === "mirror") return gmrf1d(u, dt);
  const n = u.length, a = new Float64Array(n).fill(-dt), b = new Float64Array(n).fill(1 + 2 * dt), c = new Float64Array(n).fill(-dt);
  a[0] = 0; c[n - 1] = 0;
  return thomas(a, b, c, u);
}

/** One implicit step in 2D with mirror (Neumann) border: the orthonormal DCT-II diagonalises the
 *  Neumann Laplacian, eigenvalues (2 − 2cos(πk/w)) + (2 − 2cos(πl/h)). */
export function implicitStep2d(u: ArrayLike<number>, w: number, h: number, dt: number): Float64Array {
  const C = dct2(u, w, h);
  for (let l = 0; l < h; l++) {
    const ly = 2 - 2 * Math.cos(Math.PI * l / h);
    for (let k = 0; k < w; k++) C[l * w + k] /= 1 + dt * (2 - 2 * Math.cos(Math.PI * k / w) + ly);
  }
  return idct2(C, w, h);
}

/** Largest stable explicit time step for diffusion in `dim` dimensions with conductivity ≤ gmax. */
export const stabilityLimit = (dim: number, gmax = 1): number => 1 / (2 * dim * gmax);

/** Von Neumann amplification factor of one step for the Fourier mode e^{iωj}, r = dt/dx². */
export function amplification(r: number, omega: number, scheme: "explicit" | "implicit" = "explicit"): number {
  const s = 4 * r * Math.sin(omega / 2) ** 2;
  return scheme === "explicit" ? 1 - s : 1 / (1 + s);
}

/** Gaussian blur with the border that matches the heat step: "mirror" = half-sample symmetric
 *  extension (… b a | a b …, no flux across the border), "zero" = 0 outside. 1D if h = 1. */
export function gaussianHeat(u: ArrayLike<number>, w: number, h: number, sigma: number, border: HeatBorder = "mirror"): Float64Array {
  if (!(sigma > 0)) return Float64Array.from(u);
  const k = gaussianKernel(sigma, 0), r = (k.length - 1) / 2;
  const idx = (i: number, n: number): number => {
    if (border === "zero") return i < 0 || i >= n ? -1 : i;
    const p = 2 * n;
    let m = ((i % p) + p) % p;
    return m < n ? m : p - 1 - m;
  };
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) { const q = idx(x + j, w); if (q >= 0) s += k[j + r] * u[y * w + q]; }
    tmp[y * w + x] = s;
  }
  if (h === 1) return tmp;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) { const q = idx(y + j, h); if (q >= 0) s += k[j + r] * tmp[q * w + x]; }
    out[y * w + x] = s;
  }
  return out;
}

/** σ of the Gaussian that equals running u_t = ∇²u for time t. */
export const sigmaOfTime = (t: number): number => Math.sqrt(2 * t);
export const timeOfSigma = (sigma: number): number => sigma * sigma / 2;

// ---------------------------------------------------------------------------------------
// Edge-stopping diffusion (Perona–Malik)

/** Edge-stopping function g(s): linear 1, g1 = e^{−(s/K)²}, g2 = 1/(1 + (s/K)²). */
export function diffusivity(kind: Diffusivity, s: number, K: number): number {
  if (kind === "linear") return 1;
  const q = (s / K) ** 2;
  return kind === "g1" ? Math.exp(-q) : 1 / (1 + q);
}

/** Flux φ(s) = s g(s). It falls for s > K (g2) or s > K/√2 (g1): backward diffusion. */
export const flux = (kind: Diffusivity, s: number, K: number): number => s * diffusivity(kind, s, K);

/** Where the flux peaks (forward ↔ backward diffusion). */
export const fluxPeak = (kind: Diffusivity, K: number): number => (kind === "g1" ? K / Math.SQRT2 : kind === "g2" ? K : Infinity);

/** One explicit Perona–Malik step: between each pair of 4-neighbours the flux g(|d̃|) d, where d
 *  is the difference of u and d̃ the difference of the image used for the conductivity (u
 *  itself, or u blurred with σ = preSigma: Catté et al.). Mirror border (no flux out). */
export function nonlinearStep(u: ArrayLike<number>, w: number, h: number, dt: number, kind: Diffusivity, K: number, preSigma = 0): Float64Array {
  const v = preSigma > 0 ? gaussianBlur(fromArray(w, h, u), preSigma, "mirror").data : u;
  const out = Float64Array.from(u);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (x < w - 1) {
      const d = u[p + 1] - u[p], f = dt * diffusivity(kind, Math.abs(v[p + 1] - v[p]), K) * d;
      out[p] += f; out[p + 1] -= f;
    }
    if (y < h - 1) {
      const d = u[p + w] - u[p], f = dt * diffusivity(kind, Math.abs(v[p + w] - v[p]), K) * d;
      out[p] += f; out[p + w] -= f;
    }
  }
  return out;
}

/** Conductivity map g(|∇v|) with v = u or u blurred by preSigma (central differences). */
export function conductivityMap(u: ArrayLike<number>, w: number, h: number, kind: Diffusivity, K: number, preSigma = 0): Float64Array {
  const v = preSigma > 0 ? gaussianBlur(fromArray(w, h, u), preSigma, "mirror").data : u;
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    const gx = 0.5 * (v[y * w + Math.min(w - 1, x + 1)] - v[y * w + Math.max(0, x - 1)]);
    const gy = h > 1 ? 0.5 * (v[Math.min(h - 1, y + 1) * w + x] - v[Math.max(0, y - 1) * w + x]) : 0;
    out[p] = diffusivity(kind, Math.hypot(gx, gy), K);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Total variation flow

/** One explicit step of u_t = div(∇u/|∇u|_ε), |∇u|_ε = √(|∇u|² + ε²) from forward differences
 *  at each pixel; the conductivity between two neighbours is the mean of their two values.
 *  Conductivity ≤ 1/ε, so the step must satisfy dt ≤ ε/4 (2D) or ε/2 (1D). */
export function tvFlowStep(u: ArrayLike<number>, w: number, h: number, dt: number, eps: number): Float64Array {
  const n = w * h, g = new Float64Array(n), out = Float64Array.from(u);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    const gx = x < w - 1 ? u[p + 1] - u[p] : 0, gy = y < h - 1 ? u[p + w] - u[p] : 0;
    g[p] = 1 / Math.sqrt(gx * gx + gy * gy + eps * eps);
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (x < w - 1) { const f = dt * (h > 1 ? 0.5 * (g[p] + g[p + 1]) : g[p]) * (u[p + 1] - u[p]); out[p] += f; out[p + 1] -= f; }
    if (y < h - 1) { const f = dt * 0.5 * (g[p] + g[p + w]) * (u[p + w] - u[p]); out[p] += f; out[p + w] -= f; }
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Energies (forward differences, Neumann border)

function forEachGradient(u: ArrayLike<number>, w: number, h: number, f: (s: number) => number): number {
  let e = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    const gx = x < w - 1 ? u[p + 1] - u[p] : 0, gy = y < h - 1 ? u[p + w] - u[p] : 0;
    e += f(Math.hypot(gx, gy));
  }
  return e;
}
/** ½ Σ |∇u|² (heat equation = gradient descent on this). */
export const dirichletEnergy = (u: ArrayLike<number>, w: number, h: number): number => forEachGradient(u, w, h, s => 0.5 * s * s);
/** Σ |∇u| (TV flow). */
export const tvValue = (u: ArrayLike<number>, w: number, h: number): number => tv2dValue(u, w, h);
/** Σ (K²/2) ln(1 + |∇u|²/K²) (Perona–Malik g2). */
export const pmEnergy = (u: ArrayLike<number>, w: number, h: number, K: number): number => forEachGradient(u, w, h, s => 0.5 * K * K * Math.log(1 + (s / K) ** 2));
/** The three penalties Ψ(s) of one gradient size s. */
export function penaltyOf(kind: "heat" | "tv" | "pm", s: number, K = 1): number {
  return kind === "heat" ? 0.5 * s * s : kind === "tv" ? Math.abs(s) : 0.5 * K * K * Math.log(1 + (s / K) ** 2);
}

// ---------------------------------------------------------------------------------------
// Profile scores

/** 10–90 % rise width of an edge in profile p between indices i0 < i1 (lo/hi from the ends). */
export function edgeWidth1090(p: ArrayLike<number>, i0: number, i1: number): number {
  const lo = p[i0], hi = p[i1], up = hi > lo;
  const at = (f: number): number => {
    const t = lo + f * (hi - lo);
    for (let i = i0; i < i1; i++) {
      const a = p[i], b = p[i + 1];
      if ((up && a <= t && b >= t) || (!up && a >= t && b <= t)) return b === a ? i : i + (t - a) / (b - a);
    }
    return NaN;
  };
  return Math.abs(at(0.9) - at(0.1));
}

/** Staircase index of a ramp segment: n Σd² / (Σ|d|)² over its n differences. 1 for a perfect
 *  ramp (all steps equal); n when the whole rise sits in one jump. */
export function staircaseIndex(p: ArrayLike<number>, i0: number, i1: number): number {
  let s1 = 0, s2 = 0, n = 0;
  for (let i = i0; i < i1; i++) { const d = p[i + 1] - p[i]; s1 += Math.abs(d); s2 += d * d; n++; }
  return s1 > 0 ? n * s2 / (s1 * s1) : 1;
}

// ---------------------------------------------------------------------------------------
// The test card (synthetic, with ground truth)

export interface CardFeature { name: string; x: number; y: number; r?: number }
export interface TestCard { w: number; h: number; truth: Float64Array; features: CardFeature[]; profileRow: number }

const BG = 60, FG = 180;

/** 128 × 128 synthetic test card (values 0..255): a square (steps, corners), discs of radius
 *  2, 3, 5 and 8, a dot lattice, one unique spot (a ring), fine stripes, and a bottom band with a
 *  step, a ramp, a 1-px line and a soft step along row `profileRow` = 112. */
export function testCard(): TestCard {
  const w = 128, h = 128, t = new Float64Array(w * h).fill(BG);
  const set = (x: number, y: number, v: number) => { if (x >= 0 && y >= 0 && x < w && y < h) t[y * w + x] = v; };
  // square with corners
  for (let y = 6; y < 40; y++) for (let x = 6; x < 40; x++) set(x, y, FG);
  // discs (anti-aliased by 4×4 supersampling)
  const disc = (cx: number, cy: number, r: number, v: number) => {
    for (let y = Math.floor(cy - r - 1); y <= Math.ceil(cy + r + 1); y++) for (let x = Math.floor(cx - r - 1); x <= Math.ceil(cx + r + 1); x++) {
      let c = 0;
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if ((x + (a + 0.5) / 4 - 0.5 - cx) ** 2 + (y + (b + 0.5) / 4 - 0.5 - cy) ** 2 <= r * r) c++;
      if (c) set(x, y, t[y * w + x] * (1 - c / 16) + v * c / 16);
    }
  };
  disc(52, 9, 2, 140); disc(52, 22, 3, 140); disc(52, 39, 5, 140); disc(22, 62, 8, 140);
  // dot lattice (period 6) in x 68..122, y 6..40
  for (let j = 0; j < 6; j++) for (let i = 0; i < 10; i++) disc(70 + 6 * i, 8 + 6 * j, 1.5, FG);
  // unique spot: a small ring
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = Math.hypot(x - 46, y - 76);
    if (d >= 2.2 && d <= 4.2) set(x, y, FG);
  }
  // fine stripes (period 4) in x 68..122, y 50..88
  for (let y = 50; y < 88; y++) for (let x = 68; x < 122; x++) set(x, y, BG + 40 + 40 * Math.sin(2 * Math.PI * x / 4 + 0.6));
  // bottom band, y 98..126: step at x = 10, ramp from x = 34 to 66 (down), 1-px line at x = 82, soft step at x = 100
  for (let y = 98; y < 126; y++) for (let x = 0; x < w; x++) {
    let v = BG;
    if (x >= 10 && x < 34) v = FG;
    else if (x >= 34 && x < 66) v = FG + (BG - FG) * (x - 34) / 32;
    if (x === 82) v = FG;
    if (x >= 92) v = BG + (FG - BG) / (1 + Math.exp(-(x - 100) / 1.5));
    t[y * w + x] = v;
  }
  return {
    w, h, truth: t, profileRow: 112,
    features: [
      { name: "square", x: 22, y: 22 }, { name: "corner", x: 6, y: 6 }, { name: "step", x: 10, y: 112 },
      { name: "ramp", x: 50, y: 112 }, { name: "line", x: 82, y: 112 }, { name: "soft step", x: 100, y: 112 },
      { name: "disc r = 2", x: 52, y: 9, r: 2 }, { name: "disc r = 3", x: 52, y: 22, r: 3 }, { name: "disc r = 5", x: 52, y: 39, r: 5 }, { name: "disc r = 8", x: 22, y: 62, r: 8 },
      { name: "lattice", x: 82, y: 20 }, { name: "unique spot", x: 46, y: 76 }, { name: "stripes", x: 94, y: 68 },
    ],
  };
}

/** Add Gaussian noise of standard deviation sd (seeded). */
export function addNoise(u: ArrayLike<number>, sd: number, seed = 1): Float64Array {
  const r = rng(seed), out = new Float64Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = u[i] + sd * r.norm();
  return out;
}

/** Salt-and-pepper noise: a fraction `frac` of pixels set to lo or hi (seeded). */
export function addSaltPepper(u: ArrayLike<number>, frac: number, lo = 0, hi = 255, seed = 1): Float64Array {
  const r = rng(seed), out = Float64Array.from(u);
  for (let i = 0; i < u.length; i++) if (r.unif() < frac) out[i] = r.unif() < 0.5 ? lo : hi;
  return out;
}

/** Poisson noise with `photons` expected counts at value 255 (seeded; normal approximation
 *  above 30 counts, exact below), rescaled back to 0..255. */
export function addPoisson(u: ArrayLike<number>, photons: number, seed = 1): Float64Array {
  const r = rng(seed), out = new Float64Array(u.length), k = photons / 255;
  for (let i = 0; i < u.length; i++) {
    const lam = Math.max(0, u[i] * k);
    let n: number;
    if (lam > 30) n = Math.max(0, Math.round(lam + Math.sqrt(lam) * r.norm()));
    else { const L = Math.exp(-lam); let p = 1; n = -1; do { n++; p *= r.unif(); } while (p > L); }
    out[i] = n / k;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Neighbourhood filters: weights at one pixel (candidates for lib/denoise.ts)

const mirrorIdx = (i: number, n: number): number => {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
};

/** Effective number of averaged pixels (Σw)²/Σw²: the noise variance falls by this factor. */
export function effectiveN(wts: ArrayLike<number>): number {
  let s = 0, s2 = 0;
  for (let i = 0; i < wts.length; i++) { s += wts[i]; s2 += wts[i] * wts[i]; }
  return s2 > 0 ? s * s / s2 : 0;
}

export interface WeightMap { r: number; size: number; spatial: Float64Array; range: Float64Array; product: Float64Array; estimate: number; neff: number }

/** Bilateral weights of all pixels in the (2r+1)² window around (x, y): spatial factor,
 *  range factor (from `guide`, default the image itself: joint bilateral otherwise), product. */
export function bilateralWeights(img: ArrayLike<number>, w: number, h: number, x: number, y: number, sigmaS: number, sigmaR: number, radius = Math.ceil(2 * sigmaS), guide?: ArrayLike<number>): WeightMap {
  const r = Math.max(1, radius), size = 2 * r + 1, g = guide ?? img;
  const spatial = new Float64Array(size * size), range = new Float64Array(size * size), product = new Float64Array(size * size);
  const c = g[y * w + x];
  let s = 0, sw = 0;
  for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
    const q = mirrorIdx(y + j, h) * w + mirrorIdx(x + i, w), k = (j + r) * size + i + r;
    spatial[k] = Math.exp(-(i * i + j * j) / (2 * sigmaS * sigmaS));
    range[k] = Math.exp(-((g[q] - c) ** 2) / (2 * sigmaR * sigmaR));
    product[k] = spatial[k] * range[k];
    s += product[k] * img[q]; sw += product[k];
  }
  return { r, size, spatial, range, product, estimate: s / sw, neff: effectiveN(product) };
}

/** Joint (cross) bilateral filter: range weights from `guide`, values from `img`. */
export function jointBilateral(img: ArrayLike<number>, guide: ArrayLike<number>, w: number, h: number, sigmaS: number, sigmaR: number, radius = Math.ceil(2 * sigmaS)): Float64Array {
  const r = Math.max(1, radius), out = new Float64Array(w * h), kr = 1 / (2 * sigmaR * sigmaR);
  const ws: number[] = [];
  for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) ws.push(Math.exp(-(i * i + j * j) / (2 * sigmaS * sigmaS)));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = guide[y * w + x];
    let s = 0, sw = 0, t = 0;
    for (let j = -r; j <= r; j++) {
      const yy = mirrorIdx(y + j, h) * w;
      for (let i = -r; i <= r; i++, t++) {
        const q = yy + mirrorIdx(x + i, w), d = guide[q] - c, wt = ws[t] * Math.exp(-d * d * kr);
        s += wt * img[q]; sw += wt;
      }
    }
    out[y * w + x] = s / sw;
  }
  return out;
}

export interface NlmMap { S: number; size: number; weights: Float64Array; d2: Float64Array; estimate: number; neff: number; selfWeight: number; top: { x: number; y: number; w: number }[] }

/** Non-local means weights at (x, y): patch (2P+1)², search window (2S+1)², w = e^{−d²/h²} with
 *  d² the MEAN squared patch difference (as lib/denoise.ts nonLocalMeans). selfFix: the pixel's
 *  own weight is replaced by the largest weight of any other pixel. Mirror border. */
export function nlmWeights(img: ArrayLike<number>, w: number, h: number, x: number, y: number, P: number, S: number, hh: number, selfFix = false): NlmMap {
  const size = 2 * S + 1, weights = new Float64Array(size * size), d2 = new Float64Array(size * size), area = (2 * P + 1) ** 2;
  const at = (xx: number, yy: number) => img[mirrorIdx(yy, h) * w + mirrorIdx(xx, w)];
  let best = 0;
  for (let dy = -S; dy <= S; dy++) for (let dx = -S; dx <= S; dx++) {
    let s = 0;
    for (let j = -P; j <= P; j++) for (let i = -P; i <= P; i++) { const d = at(x + i, y + j) - at(x + dx + i, y + dy + j); s += d * d; }
    const k = (dy + S) * size + dx + S;
    d2[k] = s / area;
    weights[k] = Math.exp(-d2[k] / (hh * hh));
    if (dx || dy) best = Math.max(best, weights[k]);
  }
  const ck = S * size + S;
  if (selfFix) weights[ck] = best;
  let s = 0, sw = 0;
  for (let dy = -S; dy <= S; dy++) for (let dx = -S; dx <= S; dx++) { const k = (dy + S) * size + dx + S; s += weights[k] * at(x + dx, y + dy); sw += weights[k]; }
  const order = Array.from(weights.keys()).filter(k => k !== ck).sort((a, b) => weights[b] - weights[a]).slice(0, 5);
  return {
    S, size, weights, d2, estimate: s / sw, neff: effectiveN(weights), selfWeight: weights[ck] / sw,
    top: order.map(k => ({ x: x + (k % size) - S, y: y + Math.floor(k / size) - S, w: weights[k] })),
  };
}

/** Non-local means of a 1D signal (patch 2P+1, all positions whose patch lies inside). Returns
 *  the estimate at index i, the candidate positions, their mean squared distances and weights. */
export function nlm1dAt(x: ArrayLike<number>, i: number, P: number, hh: number): { pos: number[]; d2: number[]; w: number[]; estimate: number } {
  const pos: number[] = [], d2: number[] = [], wt: number[] = [];
  for (let j = P; j < x.length - P; j++) {
    let s = 0;
    for (let k = -P; k <= P; k++) s += (x[i + k] - x[j + k]) ** 2;
    pos.push(j); d2.push(s / (2 * P + 1)); wt.push(Math.exp(-s / (2 * P + 1) / (hh * hh)));
  }
  let num = 0, den = 0;
  pos.forEach((j, k) => { num += wt[k] * x[j]; den += wt[k]; });
  return { pos, d2, w: wt, estimate: num / den };
}

/** PSNR on 0..255 (convenience; lib/metrics.ts psnr). */
export function psnr255(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return 10 * Math.log10(255 * 255 / (s / a.length));
}
