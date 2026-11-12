// The forward model of the psfotf demo (no DOM): a PSF from the pupil (with defocus), a
// motion-blurred or a Gaussian PSF, its OTF, and the blur of a test object by FFT.
// Field: N × N pixels of PIXEL nm (256 × 40 nm = 10.24 µm). Everything here is imported by
// demos/test/imageformation-test.ts.
import * as P from "../lib/psf";
import { fft2dRect, freq } from "../lib/fft";
import { siemensStar } from "../lib/synth";
import { rng } from "../lib/ui";

export const N = 256;
export const PIXEL = 40;          // nm per pixel in the sample plane

export type PsfKind = "pupil" | "motion" | "gauss";
export interface Optics {
  kind: PsfKind;
  NA: number;
  wavelength: number;             // nm
  defocus: number;                // nm (pupil kind only)
  motion: number;                 // nm, length of the motion path (motion kind)
  angle: number;                  // degrees, counter-clockwise on screen (motion kind)
}

/** Refractive index of the immersion medium: oil for NA ≥ 1, air below. */
export const immersion = (NA: number): number => (NA >= 1 ? 1.518 : 1.0);

export interface PsfResult { psf: Float64Array; re: Float64Array; im: Float64Array }

/** Centred N × N PSF (sum 1) and its OTF (lib/fft layout). */
export function computePSF(o: Optics, n = N, pixel = PIXEL): PsfResult {
  const pupil = { NA: o.NA, wavelength: o.wavelength, pixel, ri: immersion(o.NA) };
  if (o.kind === "gauss") {
    const psf = P.gaussianPSF(n, n, P.airySigma(o.NA, o.wavelength, pixel));
    return { psf, ...P.otf2d(psf, n, n) };
  }
  if (o.kind === "pupil") {
    const psf = P.pupilPSF(n, pupil, o.defocus);
    return { psf, ...P.otf2d(psf, n, n) };
  }
  // motion: the in-focus diffraction PSF dragged along a line (OTFs multiply)
  const a = P.otf2d(P.pupilPSF(n, pupil, 0), n, n);
  const m = P.otf2d(P.motionPSF(n, n, o.motion / pixel, o.angle * Math.PI / 180), n, n);
  const re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    re[i] = a.re[i] * m.re[i] - a.im[i] * m.im[i];
    im[i] = a.re[i] * m.im[i] + a.im[i] * m.re[i];
  }
  const r2 = re.slice(), i2 = im.slice();
  fft2dRect(r2, i2, n, n, true);
  const psf = new Float64Array(n * n), h = n >> 1;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) psf[((y + h) % n) * n + (x + h) % n] = Math.max(0, r2[y * n + x]);
  return { psf: P.normalise(psf), re, im };
}

/** Circular blur of an n × n object by multiplying its spectrum with the OTF. */
export function blurWith(obj: ArrayLike<number>, H: { re: Float64Array; im: Float64Array }, n = N): Float64Array {
  const re = Float64Array.from(obj), im = new Float64Array(n * n);
  fft2dRect(re, im, n, n, false);
  for (let i = 0; i < n * n; i++) {
    const a = re[i], b = im[i];
    re[i] = a * H.re[i] - b * H.im[i];
    im[i] = a * H.im[i] + b * H.re[i];
  }
  fft2dRect(re, im, n, n, true);
  return re;
}

/** The OTF along the horizontal frequency axis: frequency (cycles/µm) and real part, u = 0..umax. */
export function otfCut(H: { re: Float64Array }, umax: number, n = N, pixel = PIXEL): { f: Float64Array; v: Float64Array } {
  const f = new Float64Array(umax + 1), v = new Float64Array(umax + 1);
  for (let u = 0; u <= umax; u++) { f[u] = u / (n * pixel) * 1000; v[u] = H.re[u]; }
  return { f, v };
}

/** Crop of the centred spectrum (|u|, |v| ≤ k) as a (2k+1)² image, row 0 = highest positive v... flipped so +v is up. */
export function spectrumCrop(values: ArrayLike<number>, k: number, n = N): Float64Array {
  const m = 2 * k + 1, out = new Float64Array(m * m);
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
    const u = i - k, v = k - j;
    out[j * m + i] = values[((v + n) % n) * n + (u + n) % n];
  }
  return out;
}

/** Centre crop (2k+1)² of a centred n × n array. */
export function centreCrop(a: ArrayLike<number>, k: number, n = N): Float64Array {
  const m = 2 * k + 1, c = n >> 1, out = new Float64Array(m * m);
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) out[j * m + i] = a[(c - k + j) * n + c - k + i];
  return out;
}

/** FWHM (nm) of a centred PSF, along x through the centre. */
export function psfFWHM(psf: ArrayLike<number>, n = N, pixel = PIXEL): number {
  const c = n >> 1;
  return P.fwhm(Array.from({ length: n }, (_, x) => psf[c * n + x])) * pixel;
}

// ---------------------------------------------------------------------------------------
// Test objects (values 0..1, N × N)

export const STAR_SPOKES = 16;
export type ObjectId = "cell" | "star" | "beads" | "bars";

/** Siemens star with 16 spoke pairs: the spoke period shrinks towards the centre (period 2πr/16). */
export const starObject = (n = N): Float64Array => Float64Array.from(siemensStar(n, STAR_SPOKES, 0, 1).data);

/** 40 point-like beads (2 × 2 pixels = 80 nm) at seeded random positions on a dark field. */
export function beadObject(n = N, seed = 4): Float64Array {
  const r = rng(seed), out = new Float64Array(n * n);
  const pts: [number, number][] = [];
  let tries = 0;
  while (pts.length < 40 && tries++ < 10000) {
    const x = 8 + Math.floor(r.unif() * (n - 18)), y = 8 + Math.floor(r.unif() * (n - 18));
    if (pts.every(([a, b]) => Math.hypot(a - x, b - y) > 14)) pts.push([x, y]);
  }
  // two close pairs, 240 nm and 160 nm apart (6 and 4 pixels)
  pts.push([40, 230], [46, 230], [90, 230], [94, 230]);
  for (const [x, y] of pts) for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) out[(y + j) * n + x + i] = 1;
  return out;
}

/** Bar groups (4 bright bars each) with periods 160 … 800 nm, finest top left. */
export const BAR_PERIODS = [160, 200, 240, 320, 400, 480, 640, 800];
export function barObject(n = N, pixel = PIXEL): Float64Array {
  const out = new Float64Array(n * n);
  BAR_PERIODS.forEach((p, g) => {
    const pp = p / pixel, half = pp / 2, col = g % 4, row = Math.floor(g / 4);
    const x0 = 10 + col * 62, y0 = 22 + row * 120;
    for (let y = y0; y < y0 + 90; y++) for (let x = x0; x < x0 + 4 * pp; x++) {
      if (x >= n) continue;
      out[y * n + x] = (x - x0) % pp < half ? 1 : 0;
    }
  });
  return out;
}

/** Wavelength (nm, 380–750) to an approximate sRGB colour, for the slider. */
export function wavelengthColour(l: number): string {
  let r = 0, g = 0, b = 0;
  if (l < 440) { r = (440 - l) / 60; b = 1; }
  else if (l < 490) { g = (l - 440) / 50; b = 1; }
  else if (l < 510) { g = 1; b = (510 - l) / 20; }
  else if (l < 580) { r = (l - 510) / 70; g = 1; }
  else if (l < 645) { r = 1; g = (645 - l) / 65; }
  else r = 1;
  const k = l < 420 ? 0.3 + 0.7 * (l - 380) / 40 : l > 700 ? 0.3 + 0.7 * (750 - l) / 50 : 1;
  const c = (v: number) => Math.round(255 * Math.pow(v * k, 0.8));
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

/** Frequency index along one axis → cycles/µm. */
export const cyclesPerMicron = (u: number, n = N, pixel = PIXEL): number => freq(u, n) / (n * pixel) * 1000;
