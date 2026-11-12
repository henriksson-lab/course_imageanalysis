// Models of the resolve demo (no DOM), imported by demos/test/imageformation-test.ts.
//  - two points / grating on a pixel grid with photon noise (mode "twopoints"),
//  - the axial (xz) PSF of a widefield and a confocal microscope (mode "axial"), computed
//    with the radial (Hankel) form of the pupil integral: the in-focus and defocused PSF of
//    a circular pupil is |∫ P(ρ) e^{iφ(ρ,z)} J0(2πρr) 2πρ dρ|², one 1D integral per (r, z),
//  - a simulated bead slide and the bead-averaged PSF (mode "beads").
import * as P from "../lib/psf";
import { fft2dRect, freq } from "../lib/fft";
import { poisson } from "../lib/random";
import { rng } from "../lib/ui";

// ---------------------------------------------------------------------------------------
// Special functions

/** Bessel function of the first kind, order 0 (rational approximations, |error| < 1e-8). */
export function besselJ0(x: number): number {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    const a = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
    const b = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
    return a / b;
  }
  const z = 8 / ax, y = z * z, xx = ax - 0.785398164;
  const p = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  const q = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p - z * Math.sin(xx) * q);
}

// ---------------------------------------------------------------------------------------
// Two points and a grating

export interface Optics2 { NA: number; wavelength: number }

/** Photons per nm² at distance r from a point that emits `photons` (Airy pattern, integral = photons). */
export function airyDensity(r: number, o: Optics2, photons: number): number {
  return photons * P.airy(r, o.NA, o.wavelength) * Math.PI * o.NA * o.NA / (o.wavelength * o.wavelength);
}

/** Continuous intensity (relative, peak of one point = 1) along the line through two points at ±d/2. */
export function twoPointProfile(x: number, d: number, o: Optics2): number {
  return P.airy(Math.abs(x - d / 2), o.NA, o.wavelength) + P.airy(Math.abs(x + d / 2), o.NA, o.wavelength);
}

/** Dip between two equal points: 1 − I(midpoint) / I(peak), from the continuous profile (0 if no dip). */
export function dip(d: number, o: Optics2): number {
  let mx = 0;
  const step = Math.max(0.5, d / 400);
  for (let x = 0; x <= d; x += step) mx = Math.max(mx, twoPointProfile(x, d, o));
  return Math.max(0, 1 - twoPointProfile(0, d, o) / mx);
}

/** Sparrow distance (nm): the separation at which the midpoint curvature of the profile changes sign. */
export function sparrowDistance(o: Optics2): number {
  const curv = (d: number) => { const h = 0.5; return twoPointProfile(h, d, o) + twoPointProfile(-h, d, o) - 2 * twoPointProfile(0, d, o); };
  let lo = 0.3 * o.wavelength / o.NA, hi = 0.6 * o.wavelength / o.NA;   // curv < 0 at lo (one peak), > 0 at hi (dip)
  for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (curv(m) > 0) hi = m; else lo = m; }
  return (lo + hi) / 2;
}

export interface Sampled { nx: number; ny: number; xs: Float64Array; expected: Float64Array; counts: Float64Array }

/**
 * Pixel image of a field of width × height nm with pixel `pixel` nm, pixel centres at k·pixel
 * (a pixel always sits on the midpoint x = 0). `density(x, y)` is in photons per nm²; each
 * pixel integrates it with ss × ss sub-samples. counts = Poisson(expected) with a fixed seed
 * (or expected itself when noise is off).
 */
export function samplePixels(density: (x: number, y: number) => number, width: number, height: number, pixel: number, noise: boolean, seed = 5, ss = 5): Sampled {
  const kx = Math.max(1, Math.floor(width / 2 / pixel)), ky = Math.max(0, Math.floor(height / 2 / pixel));
  const nx = 2 * kx + 1, ny = 2 * ky + 1, area = pixel * pixel / (ss * ss);
  const xs = Float64Array.from({ length: nx }, (_, i) => (i - kx) * pixel);
  const expected = new Float64Array(nx * ny), counts = new Float64Array(nx * ny), r = rng(seed);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const cx = (i - kx) * pixel, cy = (j - ky) * pixel;
    let s = 0;
    for (let b = 0; b < ss; b++) for (let a = 0; a < ss; a++) s += density(cx + ((a + 0.5) / ss - 0.5) * pixel, cy + ((b + 0.5) / ss - 0.5) * pixel);
    expected[j * nx + i] = s * area;
  }
  for (let k = 0; k < counts.length; k++) counts[k] = noise ? poisson(r, expected[k]) : expected[k];
  return { nx, ny, xs, expected, counts };
}

/** Photon density of two points (each `photons`) at (±d/2, 0). */
export const twoPointDensity = (d: number, o: Optics2, photons: number) => (x: number, y: number) =>
  airyDensity(Math.hypot(x - d / 2, y), o, photons) + airyDensity(Math.hypot(x + d / 2, y), o, photons);

/** Contrast that survives for a sinusoidal grating of period p (nm): the in-focus OTF at 1/p. */
export const gratingContrast = (p: number, o: Optics2): number => P.airyOTF((1 / p) / (2 * o.NA / o.wavelength));

/** Photon density (per nm²) of the image of a grating 1 + cos(2πx/p), mean `perUm2` photons per µm². */
export const gratingDensity = (p: number, o: Optics2, perUm2: number) => {
  const m = gratingContrast(p, o);
  return (x: number) => perUm2 * 1e-6 * (1 + m * Math.cos(2 * Math.PI * x / p));
};

/** Period (nm) that a grating of period p appears to have when sampled with pixel Δ (aliasing), or p if Δ ≤ p/2. */
export function apparentPeriod(p: number, pixel: number): number {
  const f = 1 / p, fs = 1 / pixel;
  const fa = Math.abs(f - Math.round(f / fs) * fs);
  return fa < 1e-12 ? Infinity : 1 / fa;
}

// ---------------------------------------------------------------------------------------
// Axial PSF (scalar, circular pupil; refractive index n of the immersion medium)

export interface AxialOptions { NA: number; wavelength: number; ri: number }

/**
 * Intensity PSF on a grid of radii rs (nm) × axial positions zs (nm), peak (r = 0, z = 0)
 * normalised to 1. Returns row-major [z][r]. The defocus phase is the same as lib/psf.ts:
 * φ = 2π z (√((n/λ)² − ρ²) − n/λ), ρ the pupil frequency up to NA/λ.
 */
export function psfRZ(rs: ArrayLike<number>, zs: ArrayLike<number>, o: AxialOptions, M = 256): Float64Array {
  const kmax = o.NA / o.wavelength, kn = o.ri / o.wavelength, dk = kmax / M;
  const nr = rs.length, nz = zs.length;
  const J = new Float64Array(nr * M), w = new Float64Array(M), kz = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    const rho = (m + 0.5) * dk;
    w[m] = rho * dk;
    kz[m] = Math.sqrt(Math.max(0, kn * kn - rho * rho)) - kn;
    for (let i = 0; i < nr; i++) J[i * M + m] = besselJ0(2 * Math.PI * rho * rs[i]);
  }
  const out = new Float64Array(nr * nz), c = new Float64Array(M), s = new Float64Array(M);
  for (let j = 0; j < nz; j++) {
    for (let m = 0; m < M; m++) { const ph = 2 * Math.PI * zs[j] * kz[m]; c[m] = Math.cos(ph) * w[m]; s[m] = Math.sin(ph) * w[m]; }
    for (let i = 0; i < nr; i++) {
      let re = 0, im = 0;
      const base = i * M;
      for (let m = 0; m < M; m++) { const jj = J[base + m]; re += jj * c[m]; im += jj * s[m]; }
      out[j * nr + i] = re * re + im * im;
    }
  }
  // the peak of a perfect pupil is at r = 0, z = 0: (Σ w)²
  let s0 = 0;
  for (let m = 0; m < M; m++) s0 += w[m];
  for (let k = 0; k < out.length; k++) out[k] /= s0 * s0;
  return out;
}

/** Axial Nyquist step λ / (2n(1 − cos α)), sin α = NA/n. */
export const axialNyquist = (o: AxialOptions): number => o.wavelength / (2 * o.ri * (1 - Math.sqrt(1 - (o.NA / o.ri) ** 2)));
/** Rule-of-thumb axial resolution 2nλ/NA². */
export const axialResolution = (o: AxialOptions): number => 2 * o.ri * o.wavelength / (o.NA * o.NA);

export interface AxialPSF { nx: number; nz: number; X: number; Z: number; img: Float64Array; lateral: Float64Array; axial: Float64Array; xs: Float64Array; zs: Float64Array; fwhmLat: number; fwhmAx: number }

/**
 * xz image (nz rows, nx columns) over ±X (lateral) × ±Z (axial), nm. Widefield: |h|²;
 * confocal (point pinhole, same wavelength): |h|⁴. Also the lateral (z = 0) and axial
 * (x = 0) profiles and their FWHMs in nm.
 */
export function axialPSF(o: AxialOptions, confocal: boolean, X: number, Z: number, nx = 97, nz = 193): AxialPSF {
  const kx = (nx - 1) / 2, rs = Float64Array.from({ length: kx + 1 }, (_, i) => i * X / kx);
  const zs = Float64Array.from({ length: nz }, (_, j) => (j - (nz - 1) / 2) * 2 * Z / (nz - 1));
  const rz = psfRZ(rs, zs, o);
  const img = new Float64Array(nx * nz);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const v = rz[j * (kx + 1) + Math.abs(i - kx)];
    img[(nz - 1 - j) * nx + i] = confocal ? v * v : v;        // row 0 = +Z (top)
  }
  const xs = Float64Array.from({ length: nx }, (_, i) => (i - kx) * X / kx);
  const lateral = Float64Array.from({ length: nx }, (_, i) => img[((nz - 1) / 2) * nx + i]);
  const axial = Float64Array.from({ length: nz }, (_, j) => img[(nz - 1 - j) * nx + kx]);
  return { nx, nz, X, Z, img, lateral, axial, xs, zs, fwhmLat: P.fwhm(lateral) * X / kx, fwhmAx: P.fwhm(axial) * 2 * Z / (nz - 1) };
}

// ---------------------------------------------------------------------------------------
// Beads

export const BEAD_PIXEL = 65;           // nm: 6.5 µm camera pixels behind a 100× objective
export const BEAD_OPTICS: Optics2 = { NA: 1.4, wavelength: 520 };
const SUB = 8;                          // fine samples per pixel (along each axis)
const FN = 256;                         // fine grid: 32 pixels = 2.08 µm
export const FIELD = 160;               // field of 160 × 160 pixels (10.4 µm)
export const PATCH = 16;                // measurement patch, centre at index 8
export const UP = 8;                    // Fourier upsampling of the averaged patch

/**
 * The image of one bead on the fine grid (FN × FN samples of BEAD_PIXEL/SUB nm at
 * half-sample offsets, sum 1): the Airy PSF convolved with the projection of a uniform sphere
 * of diameter `diameter` nm (thickness ∝ √(R² − r²)).
 */
export function beadImageFine(diameter: number, o: Optics2 = BEAD_OPTICS): Float64Array {
  const d = BEAD_PIXEL / SUB, c = FN / 2;
  const airy = new Float64Array(FN * FN);
  for (let y = 0; y < FN; y++) for (let x = 0; x < FN; x++) airy[y * FN + x] = P.airy(Math.hypot(x - c + 0.5, y - c + 0.5) * d, o.NA, o.wavelength);
  const R = diameter / 2 / d;
  if (R < 0.5) return P.normalise(airy);
  const sph = new Float64Array(FN * FN);
  for (let y = 0; y < FN; y++) for (let x = 0; x < FN; x++) {
    // 4 × 4 sub-samples per fine sample, so small spheres are smooth
    let s = 0;
    for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
      const r2 = (x - c + (a + 0.5) / 4 - 0.5) ** 2 + (y - c + (b + 0.5) / 4 - 0.5) ** 2;
      if (r2 < R * R) s += Math.sqrt(R * R - r2);
    }
    sph[y * FN + x] = s;
  }
  const A = { re: P.toOrigin(airy, FN, FN), im: new Float64Array(FN * FN) }, K = { re: P.toOrigin(sph, FN, FN), im: new Float64Array(FN * FN) };
  fft2dRect(A.re, A.im, FN, FN); fft2dRect(K.re, K.im, FN, FN);
  for (let i = 0; i < FN * FN; i++) { const a = A.re[i], b = A.im[i]; A.re[i] = a * K.re[i] - b * K.im[i]; A.im[i] = a * K.im[i] + b * K.re[i]; }
  fft2dRect(A.re, A.im, FN, FN, true);
  const out = new Float64Array(FN * FN);
  for (let y = 0; y < FN; y++) for (let x = 0; x < FN; x++) out[((y + c) % FN) * FN + (x + c) % FN] = Math.max(0, A.re[y * FN + x]);
  return P.normalise(out);
}

/** Fraction of the bead's light in pixel (i, j) for a bead at pixel (0, 0) + (ox, oy)/SUB, i, j = −16..15. */
function pixelFraction(fine: Float64Array, i: number, j: number, ox: number, oy: number): number {
  let s = 0;
  for (let b = 0; b < SUB; b++) {
    const ky = SUB * j - SUB / 2 + b - oy + FN / 2;
    if (ky < 0 || ky >= FN) continue;
    for (let a = 0; a < SUB; a++) {
      const kx = SUB * i - SUB / 2 + a - ox + FN / 2;
      if (kx >= 0 && kx < FN) s += fine[ky * FN + kx];
    }
  }
  return s;
}

export interface Bead { px: number; py: number; ox: number; oy: number }

/** 30 bead positions (seeded): integer pixel + offset in 1/8 pixel, at least 20 pixels apart, 12 from the border. */
export function beadPositions(seed = 3, count = 30): Bead[] {
  const r = rng(seed), out: Bead[] = [];
  let tries = 0;
  while (out.length < count && tries++ < 100000) {
    const px = 12 + Math.floor(r.unif() * (FIELD - 24)), py = 12 + Math.floor(r.unif() * (FIELD - 24));
    const ox = Math.floor(r.unif() * SUB) - SUB / 2, oy = Math.floor(r.unif() * SUB) - SUB / 2;
    if (out.every(b => Math.hypot(b.px - px, b.py - py) >= 20)) out.push({ px, py, ox, oy });
  }
  return out;
}

/** Expected photons per pixel of the bead slide, and the noisy counts (seeded Poisson). */
export function beadField(fine: Float64Array, beads: Bead[], photons: number, background: number, noise = true, seed = 9): { expected: Float64Array; counts: Float64Array } {
  const expected = new Float64Array(FIELD * FIELD).fill(background);
  for (const b of beads) for (let j = -16; j < 16; j++) for (let i = -16; i < 16; i++) {
    const x = b.px + i, y = b.py + j;
    if (x >= 0 && x < FIELD && y >= 0 && y < FIELD) expected[y * FIELD + x] += photons * pixelFraction(fine, i, j, b.ox, b.oy);
  }
  const r = rng(seed), counts = new Float64Array(FIELD * FIELD);
  for (let k = 0; k < counts.length; k++) counts[k] = noise ? poisson(r, expected[k]) : expected[k];
  return { expected, counts };
}

/** The PATCH × PATCH patch around pixel (px, py) (index 8 = the bead's pixel). */
export function patchAt(img: Float64Array, px: number, py: number): Float64Array {
  const out = new Float64Array(PATCH * PATCH), h = PATCH / 2;
  for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) out[j * PATCH + i] = img[(py - h + j) * FIELD + px - h + i];
  return out;
}

/** Median of the patch's border pixels: the local background. */
export function borderMedian(p: Float64Array): number {
  const v: number[] = [];
  for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) if (i === 0 || j === 0 || i === PATCH - 1 || j === PATCH - 1) v.push(p[j * PATCH + i]);
  v.sort((a, b) => a - b);
  return (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2;
}

/** Sub-pixel centre: centroid of the positive part in a 7 × 7 window, re-centred twice. */
export function centroid(p: Float64Array): [number, number] {
  let cx = PATCH / 2, cy = PATCH / 2;
  for (let it = 0; it < 3; it++) {
    const ix = Math.round(cx), iy = Math.round(cy);
    let s = 0, sx = 0, sy = 0;
    for (let j = iy - 3; j <= iy + 3; j++) for (let i = ix - 3; i <= ix + 3; i++) {
      if (i < 0 || j < 0 || i >= PATCH || j >= PATCH) continue;
      const v = Math.max(0, p[j * PATCH + i]);
      s += v; sx += v * i; sy += v * j;
    }
    if (s <= 0) break;
    cx = sx / s; cy = sy / s;
  }
  return [cx, cy];
}

/** Centroid of the positive part in the 7 × 7 window around the patch centre. */
function windowCentroid(p: Float64Array): [number, number] {
  const c = PATCH / 2;
  let s = 0, sx = 0, sy = 0;
  for (let j = c - 3; j <= c + 3; j++) for (let i = c - 3; i <= c + 3; i++) { const v = Math.max(0, p[j * PATCH + i]); s += v; sx += v * i; sy += v * j; }
  return s > 0 ? [sx / s, sy / s] : [c, c];
}

/** Shift a PATCH × PATCH image by (sx, sy) pixels with the Fourier shift theorem (circular). */
export function fourierShift(p: Float64Array, sx: number, sy: number): Float64Array {
  const n = PATCH, re = Float64Array.from(p), im = new Float64Array(n * n);
  fft2dRect(re, im, n, n);
  for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
    const fu = freq(u, n), fv = freq(v, n);
    // the Nyquist bin of a real signal cannot carry a sub-pixel shift: keep only its real part
    const a = -2 * Math.PI * ((fu === -n / 2 ? 0 : fu) * sx + (fv === -n / 2 ? 0 : fv) * sy) / n;
    const c = Math.cos(a), s = Math.sin(a), k = v * n + u, x = re[k], y = im[k];
    re[k] = x * c - y * s; im[k] = x * s + y * c;
  }
  fft2dRect(re, im, n, n, true);
  return re;
}

/** Fourier (band-limited) upsampling of a PATCH × PATCH image by UP, keeping values per original pixel. */
export function upsample(p: Float64Array): Float64Array {
  const n = PATCH, m = n * UP, re = Float64Array.from(p), im = new Float64Array(n * n);
  fft2dRect(re, im, n, n);
  const R = new Float64Array(m * m), I = new Float64Array(m * m);
  for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
    const fu = freq(u, n), fv = freq(v, n);
    // split the Nyquist bins evenly between +n/2 and −n/2 so the result stays real
    const wu = fu === -n / 2 ? 0.5 : 1, wv = fv === -n / 2 ? 0.5 : 1;
    const targets: [number, number][] = [[fu, fv]];
    if (fu === -n / 2) targets.push([n / 2, fv]);
    if (fv === -n / 2) for (const t of [...targets]) targets.push([t[0], n / 2]);
    for (const [a, b] of targets) {
      const k = ((b + m) % m) * m + (a + m) % m;
      R[k] += re[v * n + u] * wu * wv; I[k] += im[v * n + u] * wu * wv;
    }
  }
  fft2dRect(R, I, m, m, true);
  const out = new Float64Array(m * m);
  for (let i = 0; i < m * m; i++) out[i] = R[i] * UP * UP;   // the inverse divides by m² = UP²·n²: ×UP² keeps the original pixel values
  return out;
}

export interface AveragedPSF { patch: Float64Array; up: Float64Array; fwhm: number }

/** FWHM (nm) of an upsampled centred PSF: mean of the x and y profiles through the centre. */
export function upFWHM(up: Float64Array): number {
  const m = PATCH * UP, c = (PATCH / 2) * UP;
  const rx = Array.from({ length: m }, (_, i) => up[c * m + i]), ry = Array.from({ length: m }, (_, j) => up[j * m + c]);
  return (P.fwhm(rx) + P.fwhm(ry)) / 2 * BEAD_PIXEL / UP;
}

/**
 * Average the first `count` beads: background (border median, if `subtract`) removed,
 * sub-pixel centroid, Fourier shift to the patch centre, mean, normalised to sum 1.
 */
export function averageBeads(img: Float64Array, beads: Bead[], count: number, subtract: boolean): AveragedPSF {
  const acc = new Float64Array(PATCH * PATCH);
  for (const b of beads.slice(0, count)) {
    const p = patchAt(img, b.px, b.py);
    const bg = subtract ? borderMedian(p) : 0;
    for (let k = 0; k < p.length; k++) p[k] -= bg;
    // centroid, then refine: shift the bead to the centre and re-measure in a window centred
    // there, until the window is symmetric about the bead (removes the truncation bias)
    let [cx, cy] = centroid(p), s = fourierShift(p, PATCH / 2 - cx, PATCH / 2 - cy);
    for (let it = 0; it < 4; it++) {
      const [ex, ey] = windowCentroid(s);
      cx += ex - PATCH / 2; cy += ey - PATCH / 2;
      s = fourierShift(p, PATCH / 2 - cx, PATCH / 2 - cy);
    }
    for (let k = 0; k < acc.length; k++) acc[k] += s[k];
  }
  let sum = 0;
  for (const v of acc) sum += v;
  if (sum !== 0) for (let k = 0; k < acc.length; k++) acc[k] /= sum;
  const up = upsample(acc);
  return { patch: acc, up, fwhm: upFWHM(up) };
}

/** The true PSF as these pixels record a point at a pixel centre (noise-free, no background). */
export function truePSF(o: Optics2 = BEAD_OPTICS): AveragedPSF {
  const fine = beadImageFine(0, o), acc = new Float64Array(PATCH * PATCH);
  for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) acc[j * PATCH + i] = pixelFraction(fine, i - PATCH / 2, j - PATCH / 2, 0, 0);
  P.normalise(acc);
  const up = upsample(acc);
  return { patch: acc, up, fwhm: upFWHM(up) };
}
