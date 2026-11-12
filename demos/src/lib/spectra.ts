// Spectra, cone and colour-matching functions, and chromaticity (owner: colour day,
// plan/colour-day.md). No DOM. Conversions between colour spaces and the colour maps live
// in lib/colour.ts; this file holds what starts from a spectrum.
//
// Sampling: 36 bins of 10 nm, centred at 380, 390, ..., 730 nm (WL). A spectrum is a
// Float64Array of 36 non-negative values. Integrals over wavelength become sums over the
// bins, so "the eye is a linear map" is literally a 3 x 36 matrix (A_XYZ, A_LMS, ...).
//
// Data, and why no tables are copied in:
//  - CIE 1931 2° colour-matching functions: the analytic multi-lobe fit of Wyman, Sloan &
//    Shirley (2013), "Simple analytic approximations to the CIE XYZ color matching
//    functions", JCGT 2(2). It is within about 1 % of the tabulated functions, which is far
//    below what a projector shows.
//  - Cone fundamentals: L, M, S = the Hunt–Pointer–Estévez matrix times x̄, ȳ, z̄. This makes
//    the cones exactly a change of basis of the CMFs (the same 3D row space), which is the
//    point the unit makes; no measured cone tables (and no licence question) are needed.
//  - CIE D65: the standard relative spectral power distribution at 10 nm (CIE 15).
//  - Blackbody: Planck's law, computed. LED, fluorescent tube and the camera are synthetic
//    (seeded shapes), labelled as such in the demos.

export type V3 = [number, number, number];

/** Wavelengths of the 36 bins, 380..730 nm. */
export const WL: Float64Array = Float64Array.from({ length: 36 }, (_, i) => 380 + 10 * i);
export const NB = 36;
export const DL = 10;

// ---------------------------------------------------------------------------------------
// Colour-matching functions and cones

const lobe = (l: number, mu: number, s1: number, s2: number) => {
  const t = (l - mu) / (l < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
};

/** CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at wavelength l (nm), analytic fit.
 *  The fit's tails are poor for chromaticity: beyond 650 nm its ȳ decays too slowly (the
 *  locus would fold back), so there ȳ = x̄ · ρ(λ) with ρ going from its 650 nm value to the
 *  tabulated 0.361 at 700 nm; below 420 nm likewise (see below). */
export function cmf(l: number): V3 {
  const fit = (q: number): V3 => [
    1.056 * lobe(q, 599.8, 37.9, 31.0) + 0.362 * lobe(q, 442.0, 16.0, 26.7) - 0.065 * lobe(q, 501.1, 20.4, 26.2),
    0.821 * lobe(q, 568.8, 46.9, 40.5) + 0.286 * lobe(q, 530.9, 16.3, 31.1),
    1.217 * lobe(q, 437.0, 11.8, 36.0) + 0.681 * lobe(q, 459.0, 26.0, 13.8),
  ];
  const v = fit(l);
  if (l > 650) {
    const f650 = fit(650), r650 = f650[1] / f650[0], t = Math.min(1, (l - 650) / 50);
    v[1] = v[0] * (r650 + t * (0.361 - r650));
  }
  if (l < 420) {
    // the fit's ȳ tail is too high and its z̄ too low below 420 nm: keep x̄, and take ȳ, z̄
    // from x̄ with ratios going to the tabulated values (0.028, 4.72) at 400 nm and below
    const f420 = fit(420), ry = f420[1] / f420[0], rz = f420[2] / f420[0], t = Math.min(1, (420 - l) / 20);
    v[1] = v[0] * (ry + t * (0.028 - ry));
    v[2] = v[0] * (rz + t * (4.72 - rz));
  }
  return v;
}

/** Hunt–Pointer–Estévez XYZ -> LMS (rows). */
export const M_XYZ_TO_LMS: number[][] = [
  [0.4002, 0.7076, -0.0808],
  [-0.2263, 1.1653, 0.0457],
  [0, 0, 0.9182],
];

/** Cone fundamentals l̄, m̄, s̄ at wavelength l (a linear combination of the CMFs). */
export function cones(l: number): V3 {
  return mat3v(M_XYZ_TO_LMS, cmf(l));
}

/** A synthetic three-filter camera: Gaussian-like R, G, B sensitivities (not a linear
 *  combination of the cones: the camera's null space differs from the eye's). */
export function camera(l: number): V3 {
  return [
    0.95 * lobe(l, 600, 28, 40),
    0.9 * lobe(l, 535, 32, 34),
    1.0 * lobe(l, 460, 26, 30),
  ];
}

export type Observer = "xyz" | "lms" | "camera" | "deutan";

/** 3 x 36 sensitivity matrix (row-major) of an observer, rows = channels. The deuteranope
 *  has no M cones: its second row is zero (rank 2). */
export function sensitivity(obs: Observer): Float64Array {
  const A = new Float64Array(3 * NB);
  for (let i = 0; i < NB; i++) {
    const f = obs === "xyz" ? cmf(WL[i]) : obs === "camera" ? camera(WL[i]) : cones(WL[i]);
    A[i] = f[0];
    A[NB + i] = obs === "deutan" ? 0 : f[1];
    A[2 * NB + i] = f[2];
  }
  return A;
}

// ---------------------------------------------------------------------------------------
// Light sources (relative spectral power, 36 bins)

/** CIE D65, relative spectral power at 380..730 nm (10 nm), 100 at 560 nm. */
export const D65_SPD: Float64Array = Float64Array.from([
  49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81,
  109.35, 107.8, 104.79, 107.69, 104.41, 104.05, 100.0, 96.33, 95.79, 88.69, 90.01, 89.6,
  87.7, 83.29, 83.7, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35, 61.6, 69.89,
]);

/** Blackbody of temperature T (K), relative power, normalised to 100 at 560 nm. */
export function blackbody(T: number): Float64Array {
  const c2 = 1.4388e-2; // m K
  const p = (l: number) => { const m = l * 1e-9; return 1 / (m ** 5 * (Math.exp(c2 / (m * T)) - 1)); };
  const k = 100 / p(560);
  return Float64Array.from(WL, l => k * p(l));
}

/** Synthetic white LED: a blue emitter at 450 nm plus a broad yellow phosphor. */
export function whiteLED(): Float64Array {
  return Float64Array.from(WL, l => 150 * lobe(l, 450, 10, 12) + 95 * lobe(l, 565, 50, 70));
}

/** Synthetic fluorescent tube: mercury lines (405, 436, 546, 578 nm) and narrow-band
 *  phosphors (450, 545, 610 nm) on a weak continuum. Spiky. */
export function fluorescent(): Float64Array {
  const lines: [number, number][] = [[405, 30], [436, 90], [546, 160], [578, 40], [610, 150], [450, 40], [490, 20]];
  return Float64Array.from(WL, l => {
    let v = 12 + 25 * lobe(l, 600, 60, 60);
    for (const [c, a] of lines) v += a * lobe(l, c, 6, 6);
    return v;
  });
}

/** Narrow-band RGB LED lamp: three peaks (465, 530, 625 nm), about 20 nm wide. */
export function rgbLED(): Float64Array {
  return Float64Array.from(WL, l => 140 * lobe(l, 465, 10, 10) + 120 * lobe(l, 530, 12, 12) + 130 * lobe(l, 625, 10, 10));
}

/** A white LED tuned to be an exact metamer of D65 for the eye (cones): a blue emitter and
 *  two phosphor humps, with amplitudes solved from 3 equations (L, M, S equal). */
export function metamericLED(): Float64Array {
  const comps = [Float64Array.from(WL, l => lobe(l, 450, 9, 11)), Float64Array.from(WL, l => lobe(l, 540, 35, 45)), Float64Array.from(WL, l => lobe(l, 615, 30, 40))];
  const w = fitMixture(D65_SPD, comps, sensitivity("lms"));
  return Float64Array.from(WL, (_, i) => w[0] * comps[0][i] + w[1] * comps[1][i] + w[2] * comps[2][i]);
}

/** Monochromatic light: all power in the bin nearest to l (power p). */
export function monochromatic(l: number, p = 100): Float64Array {
  const s = new Float64Array(NB);
  s[Math.max(0, Math.min(NB - 1, Math.round((l - 380) / 10)))] = p;
  return s;
}

/** Flat (equal-energy) spectrum. */
export const flat = (v = 100): Float64Array => new Float64Array(NB).fill(v);

// ---------------------------------------------------------------------------------------
// Spectrum -> XYZ, and the display swatch

/** Normalisation k so that D65 has Y = 1: XYZ = k Σ s(λ) x̄(λ). */
const K_D65 = (() => { let y = 0; for (let i = 0; i < NB; i++) y += D65_SPD[i] * cmf(WL[i])[1]; return 1 / y; })();

/** Apply a 3 x 36 matrix (row-major) to a spectrum. */
export function apply(A: ArrayLike<number>, s: ArrayLike<number>): V3 {
  const out: V3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) { let t = 0; for (let i = 0; i < NB; i++) t += A[c * NB + i] * s[i]; out[c] = t; }
  return out;
}

const A_XYZ = sensitivity("xyz");

/** XYZ of a light spectrum, scaled so that D65 at the same power (100 at 560 nm) has Y = 1. */
export function spectrumToXyz(s: ArrayLike<number>): V3 {
  const v = apply(A_XYZ, s);
  return [K_D65 * v[0], K_D65 * v[1], K_D65 * v[2]];
}

/** Response of an observer, normalised so that D65 gives (1, 1, 1) in every channel that
 *  exists (a white-balanced eye or camera: the numbers are comparable between observers). */
export function response(obs: Observer, s: ArrayLike<number>): V3 {
  const A = sensitivity(obs), w = apply(A, D65_SPD), v = apply(A, s);
  return [w[0] ? v[0] / w[0] : 0, w[1] ? v[1] / w[1] : 0, w[2] ? v[2] / w[2] : 0];
}

/** XYZ of D65 with the analytic CMFs (Y = 1); used as the reference white of swatches. */
export const WHITE_XYZ: V3 = spectrumToXyz(D65_SPD);

/** sRGB (linear) <- XYZ, with a diagonal correction so that WHITE_XYZ maps exactly to
 *  (1, 1, 1): the small white-point error of the analytic CMFs then does not tint D65. */
const XYZ_TO_LIN: number[][] = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.969266, 1.8760108, 0.041556],
  [0.0556434, -0.2040259, 1.0572252],
];
const WHITE_LIN = mat3v(XYZ_TO_LIN, WHITE_XYZ);

/** Linear sRGB of an XYZ triple (white-calibrated; may lie outside [0, 1]). */
export function xyzToLinearSrgbCal(xyz: V3): V3 {
  const v = mat3v(XYZ_TO_LIN, xyz);
  return [v[0] / WHITE_LIN[0], v[1] / WHITE_LIN[1], v[2] / WHITE_LIN[2]];
}

/** Display colour of an XYZ triple: linear sRGB, clipped into the gamut by desaturating
 *  towards the grey of the same luminance (and then scaling down if still above 1).
 *  `outOfGamut` says whether anything had to be changed. */
export function displayColour(xyz: V3): { lin: V3; outOfGamut: boolean } {
  let v = xyzToLinearSrgbCal(xyz);
  let out = false;
  const Y = Math.max(0, xyz[1]);
  const mn = Math.min(v[0], v[1], v[2]);
  if (mn < -1e-6) {
    out = true;
    const t = Y / (Y - mn); // mix towards grey Y until the smallest channel is 0
    v = [Y + t * (v[0] - Y), Y + t * (v[1] - Y), Y + t * (v[2] - Y)];
  }
  const mx = Math.max(v[0], v[1], v[2]);
  if (mx > 1 + 1e-6) { out = true; v = [v[0] / mx, v[1] / mx, v[2] / mx]; }
  return { lin: [Math.max(0, v[0]), Math.max(0, v[1]), Math.max(0, v[2])], outOfGamut: out };
}

/** XYZ -> chromaticity (x, y). Black gives the white point's chromaticity. */
export function xyzToXy(xyz: V3): [number, number] {
  const s = xyz[0] + xyz[1] + xyz[2];
  if (s <= 1e-12) { const w = WHITE_XYZ, ws = w[0] + w[1] + w[2]; return [w[0] / ws, w[1] / ws]; }
  return [xyz[0] / s, xyz[1] / s];
}

/** (x, y, Y) -> XYZ. */
export function xyYToXyz(x: number, y: number, Y: number): V3 {
  if (y <= 1e-12) return [0, 0, 0];
  return [x * Y / y, Y, (1 - x - y) * Y / y];
}

/** Points of the spectral locus (x, y) for monochromatic light from l0 to l1 nm. */
export function spectralLocus(l0 = 380, l1 = 700, step = 1): { l: number; x: number; y: number }[] {
  const out: { l: number; x: number; y: number }[] = [];
  for (let l = l0; l <= l1 + 1e-9; l += step) {
    const [x, y] = xyzToXy(cmf(l));
    out.push({ l, x, y });
  }
  return out;
}

/** An approximate display colour for monochromatic light of wavelength l (for drawing
 *  curves and axes): the chromaticity of the spectral locus at constant lightness, pushed
 *  into the sRGB gamut. Returns display sRGB in [0, 1]. */
export function wavelengthToSrgb(l: number): V3 {
  const [x, y] = xyzToXy(cmf(l));
  const lin = xyzToLinearSrgbCal(xyYToXyz(x, y, 0.35)).map(v => Math.max(0, v));
  const mx = Math.max(lin[0], lin[1], lin[2], 1e-6);
  // fade at the ends of the visible range, where the eye is insensitive
  const f = l < 420 ? 0.35 + 0.65 * (l - 380) / 40 : l > 680 ? 0.35 + 0.65 * (730 - l) / 50 : 1;
  return lin.map(v => encode(Math.max(0, Math.min(1, f * v / mx)))) as V3;
}

const encode = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

// ---------------------------------------------------------------------------------------
// Displays: primaries, white point and the RGB -> XYZ matrix

export interface Display { name: string; r: [number, number]; g: [number, number]; b: [number, number]; w: [number, number] }

export const D65_XY: [number, number] = [0.3127, 0.329];
export const SRGB_DISPLAY: Display = { name: "sRGB", r: [0.64, 0.33], g: [0.3, 0.6], b: [0.15, 0.06], w: D65_XY };
export const P3_DISPLAY: Display = { name: "Display P3", r: [0.68, 0.32], g: [0.265, 0.69], b: [0.15, 0.06], w: D65_XY };
export const REC2020_DISPLAY: Display = { name: "Rec. 2020", r: [0.708, 0.292], g: [0.17, 0.797], b: [0.131, 0.046], w: D65_XY };

/** The 3 x 3 matrix (rows) taking linear RGB of a display to XYZ (white Y = 1): the columns
 *  are the primaries' XYZ, each scaled so that R = G = B = 1 gives the white point. */
export function rgbToXyzMatrix(d: Display): { M: number[][]; scales: V3 } {
  const col = (p: [number, number]): V3 => [p[0] / p[1], 1, (1 - p[0] - p[1]) / p[1]];
  const P = [col(d.r), col(d.g), col(d.b)]; // columns
  const Pm = [[P[0][0], P[1][0], P[2][0]], [P[0][1], P[1][1], P[2][1]], [P[0][2], P[1][2], P[2][2]]];
  const W = col(d.w);
  const s = mat3v(inv3(Pm), W);
  return { M: Pm.map(row => [row[0] * s[0], row[1] * s[1], row[2] * s[2]]), scales: s };
}

// ---------------------------------------------------------------------------------------
// Metamers: the null space of a 3 x n map

/** Projector onto the null space of A (3 x n, row-major, full row rank):
 *  P = I - Aᵀ (A Aᵀ)⁻¹ A, as a function v -> P v. Rank-deficient A (a dichromat) uses only
 *  the non-zero rows. */
export function nullProjector(A: ArrayLike<number>, n = NB): (v: ArrayLike<number>) => Float64Array {
  const rows: number[] = [];
  for (let c = 0; c < 3; c++) { let t = 0; for (let i = 0; i < n; i++) t += A[c * n + i] ** 2; if (t > 1e-20) rows.push(c); }
  const k = rows.length;
  const G = new Float64Array(k * k);
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { let t = 0; for (let i = 0; i < n; i++) t += A[rows[a] * n + i] * A[rows[b] * n + i]; G[a * k + b] = t; }
  const Gi = invSmall(G, k);
  return (v: ArrayLike<number>) => {
    const Av = rows.map(r => { let t = 0; for (let i = 0; i < n; i++) t += A[r * n + i] * v[i]; return t; });
    const c = rows.map((_, a) => { let t = 0; for (let b = 0; b < k; b++) t += Gi[a * k + b] * Av[b]; return t; });
    const out = Float64Array.from(v);
    for (let a = 0; a < k; a++) for (let i = 0; i < n; i++) out[i] -= A[rows[a] * n + i] * c[a];
    return out;
  };
}

/** A smooth seeded random wiggle (sum of a few sinusoids over the 36 bins). */
export function wiggle(unif: () => number, n = NB): Float64Array {
  const v = new Float64Array(n);
  for (let k = 1; k <= 6; k++) {
    const a = (unif() * 2 - 1) / k ** 0.5, ph = unif() * 2 * Math.PI;
    for (let i = 0; i < n; i++) v[i] += a * Math.sin(Math.PI * k * i / (n - 1) * 1.7 + ph);
  }
  return v;
}

/** A metamer of s for the observer A: s + t·d with d = P(wiggle) in the null space of A,
 *  (or d = P(dir) for a given direction) and t as large as possible (times `fill`) while
 *  keeping every bin in [0, hi]. Then
 *  A·(result) = A·s exactly (up to rounding), although the spectra differ. */
export function makeMetamer(s: ArrayLike<number>, A: ArrayLike<number>, unif: () => number, fill = 0.92, hi = Infinity, dir?: ArrayLike<number>): Float64Array {
  const P = nullProjector(A);
  const d = P(dir ?? wiggle(unif));
  const maxStep = (sign: number) => {
    let t = Infinity;
    for (let i = 0; i < NB; i++) {
      const di = sign * d[i];
      if (di < 0) t = Math.min(t, -s[i] / di);
      if (di > 0 && hi < Infinity) t = Math.min(t, (hi - s[i]) / di);
    }
    return t;
  };
  const tp = maxStep(1), tn = maxStep(-1);
  const sign = tp >= tn ? 1 : -1, t = fill * Math.min(sign > 0 ? tp : tn, 1e6);
  return Float64Array.from(s, (v, i) => Math.min(hi, Math.max(0, v + sign * t * d[i])));
}

/** Least-squares weights w (k components) so that A·(Σ w_j c_j) ≈ A·target (3 equations);
 *  with k = 3 the match is exact if the 3 x 3 system is regular. */
export function fitMixture(target: ArrayLike<number>, comps: ArrayLike<number>[], A: ArrayLike<number>): number[] {
  const t = apply(A, target), C = comps.map(c => apply(A, c)), k = comps.length;
  // normal equations (CᵀC) w = Cᵀ t, C is 3 x k with columns C[j]
  const G = new Float64Array(k * k), r = new Float64Array(k);
  for (let a = 0; a < k; a++) {
    r[a] = C[a][0] * t[0] + C[a][1] * t[1] + C[a][2] * t[2];
    for (let b = 0; b < k; b++) G[a * k + b] = C[a][0] * C[b][0] + C[a][1] * C[b][1] + C[a][2] * C[b][2];
  }
  const Gi = invSmall(G, k);
  return Array.from({ length: k }, (_, a) => { let v = 0; for (let b = 0; b < k; b++) v += Gi[a * k + b] * r[b]; return v; });
}

// ---------------------------------------------------------------------------------------
// Small matrix helpers (3 x 3 and k x k for k <= 3)

export function mat3v(M: number[][], v: ArrayLike<number>): V3 {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}

export function inv3(M: number[][]): number[][] {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

export function mat3mul(A: number[][], B: number[][]): number[][] {
  return [0, 1, 2].map(r => [0, 1, 2].map(c => A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c]));
}

function invSmall(G: Float64Array, k: number): Float64Array {
  if (k === 1) return Float64Array.of(1 / G[0]);
  if (k === 2) { const d = G[0] * G[3] - G[1] * G[2]; return Float64Array.of(G[3] / d, -G[1] / d, -G[2] / d, G[0] / d); }
  const m = inv3([[G[0], G[1], G[2]], [G[3], G[4], G[5]], [G[6], G[7], G[8]]]);
  return Float64Array.from(m.flat());
}
