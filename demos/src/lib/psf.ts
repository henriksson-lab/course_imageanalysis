// Point spread functions and optical transfer functions (no DOM). Owner: the deconvolution
// day (written there first); the image formation day uses and extends it.
//
// Conventions
//  - A PSF is a flat Float64Array, row-major, CENTRED: the optical axis sits at index
//    (h/2)·w + w/2 (and plane d/2 in 3D), as fftshift puts frequency 0. `toOrigin` moves the
//    centre to index 0, which is what FFT convolution needs. Every PSF sums to 1 (flux kept).
//  - Distances are in pixels unless a function takes `pixel` (nm per pixel) and `wavelength`
//    (nm); NA is the numerical aperture, `ri` the refractive index of the immersion medium.
//  - Frequencies are in cycles per pixel (−0.5..0.5); the incoherent cut-off of a circular
//    pupil is fc = 2·NA/λ cycles per nm, i.e. 2·NA·pixel/λ cycles per pixel.
//  - An OTF is the 2D/3D FFT (lib/fft.ts layout, frequency 0 at index 0) of the PSF moved
//    to the origin. For a symmetric PSF it is real.
import { fft1d, fft2dRect, fft3d, freq, ifftshift } from "./fft";

// ---------------------------------------------------------------------------------------
// Special functions

/** Bessel function of the first kind, order 1 (rational approximations, |error| < 1e-8). */
export function besselJ1(x: number): number {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    const a = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))));
    const b = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return a / b;
  }
  const z = 8 / ax, y = z * z, xx = ax - 2.356194491;
  const p = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
  const q = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p - z * Math.sin(xx) * q);
  return x < 0 ? -ans : ans;
}

/** Airy intensity (2 J1(v)/v)², peak 1, at radius r (nm): v = 2π·NA·r/λ. */
export function airy(r: number, NA: number, wavelength: number): number {
  const v = 2 * Math.PI * NA * r / wavelength;
  if (Math.abs(v) < 1e-8) return 1;
  const a = 2 * besselJ1(v) / v;
  return a * a;
}

/** Radius of the first dark ring of the Airy pattern (nm): 0.61 λ / NA. Also the Rayleigh distance. */
export const airyRadius = (NA: number, wavelength: number): number => 0.6098 * wavelength / NA;

/** Incoherent cut-off frequency of a circular pupil, in cycles per pixel: 2 NA · pixel / λ. */
export const cutoff = (NA: number, wavelength: number, pixel: number): number => 2 * NA * pixel / wavelength;

/** Analytic OTF of an aberration-free circular pupil at normalised frequency s = f / fc (0 beyond 1). */
export function airyOTF(s: number): number {
  s = Math.abs(s);
  if (s >= 1) return 0;
  return (2 / Math.PI) * (Math.acos(s) - s * Math.sqrt(1 - s * s));
}

// ---------------------------------------------------------------------------------------
// Basic helpers

/** Divide by the sum, so the PSF keeps the total flux. Returns the same array. */
export function normalise(psf: Float64Array): Float64Array {
  let s = 0;
  for (let i = 0; i < psf.length; i++) s += psf[i];
  if (s !== 0) for (let i = 0; i < psf.length; i++) psf[i] /= s;
  return psf;
}

/** Move a centred w × h PSF so its centre is at index 0 (for FFT convolution). */
export function toOrigin(psf: ArrayLike<number>, w: number, h: number): Float64Array {
  return ifftshift(psf, w, h);
}

/** Centred 2D PSF from a function of the offset (dx, dy) in pixels, integrated over each pixel with ss × ss samples. */
export function psfFromFunction(w: number, h: number, f: (dx: number, dy: number) => number, ss = 1): Float64Array {
  const out = new Float64Array(w * h), cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++)
      s += f(x - cx + (i + 0.5) / ss - 0.5, y - cy + (j + 0.5) / ss - 0.5);
    out[y * w + x] = s;
  }
  return normalise(out);
}

// ---------------------------------------------------------------------------------------
// 2D PSF models

/** Gaussian PSF with standard deviation `sigma` pixels (sampled at pixel centres; sigma 0 = a point). */
export function gaussianPSF(w: number, h: number, sigma: number): Float64Array {
  if (sigma <= 0) return pointPSF(w, h);
  return psfFromFunction(w, h, (dx, dy) => Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)));
}

/** A single bright pixel: no blur. */
export function pointPSF(w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  out[Math.floor(h / 2) * w + Math.floor(w / 2)] = 1;
  return out;
}

/** Airy PSF (in-focus, aberration-free circular pupil), each pixel integrated with ss × ss samples. */
export function airyPSF(w: number, h: number, o: { NA: number; wavelength: number; pixel: number }, ss = 3): Float64Array {
  return psfFromFunction(w, h, (dx, dy) => airy(Math.hypot(dx, dy) * o.pixel, o.NA, o.wavelength), ss);
}

/**
 * Airy PSF defined through its OTF: the analytic incoherent OTF of a circular pupil on the
 * n × n frequency grid, transformed back. Exactly band-limited (H = 0 beyond the cut-off,
 * up to rounding), so "invisible" frequencies really are invisible. Centred, sum 1.
 */
export function airyPSFBandlimited(w: number, h: number, o: { NA: number; wavelength: number; pixel: number }): Float64Array {
  const fc = cutoff(o.NA, o.wavelength, o.pixel), re = new Float64Array(w * h), im = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) re[v * w + u] = airyOTF(Math.hypot(freq(u, w) / w, freq(v, h) / h) / fc);
  fft2dRect(re, im, w, h, true);
  for (let i = 0; i < w * h; i++) re[i] = Math.max(0, re[i]);
  return normalise(fftshiftOrigin(re, w, h));
}

/** Gaussian σ (pixels) that best matches the Airy core: σ ≈ 0.21 λ / NA. */
export const airySigma = (NA: number, wavelength: number, pixel: number): number => 0.21 * wavelength / NA / pixel;

/**
 * Linear motion blur: a uniform line of `length` pixels at `angle` radians (counter-clockwise
 * on screen, 0 = to the right), antialiased by splatting 8 points per pixel bilinearly.
 */
export function motionPSF(w: number, h: number, length: number, angle = 0): Float64Array {
  const out = new Float64Array(w * h), cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  if (length <= 0) return pointPSF(w, h);
  const n = Math.max(2, Math.ceil(length * 8)), c = Math.cos(angle), s = -Math.sin(angle);
  for (let k = 0; k < n; k++) {
    const t = ((k + 0.5) / n - 0.5) * length, x = cx + t * c, y = cy + t * s;
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const put = (xx: number, yy: number, v: number) => {
      const i = ((yy % h) + h) % h * w + ((xx % w) + w) % w;
      out[i] += v;
    };
    put(x0, y0, (1 - fx) * (1 - fy)); put(x0 + 1, y0, fx * (1 - fy));
    put(x0, y0 + 1, (1 - fx) * fy); put(x0 + 1, y0 + 1, fx * fy);
  }
  return normalise(out);
}

/** Geometric defocus: a uniform disk of `radius` pixels (antialiased with 4 × 4 samples per pixel). */
export function diskPSF(w: number, h: number, radius: number): Float64Array {
  if (radius <= 0.5) return pointPSF(w, h);
  return psfFromFunction(w, h, (dx, dy) => (dx * dx + dy * dy <= radius * radius ? 1 : 0), 4);
}

export interface PupilOptions {
  NA: number;
  wavelength: number;     // nm (in vacuum)
  pixel: number;          // nm per pixel
  ri?: number;            // refractive index of the medium (default 1.518, oil)
}

/** Amplitude spectrum of the pupil (lib/fft layout, n × n) with the defocus phase for an offset z (nm). */
function pupilField(n: number, o: PupilOptions, z: number): { re: Float64Array; im: Float64Array } {
  const ri = o.ri ?? 1.518, kmax = o.NA / o.wavelength, kn = ri / o.wavelength, df = 1 / (n * o.pixel);
  const re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
    const kx = freq(u, n) * df, ky = freq(v, n) * df, k2 = kx * kx + ky * ky;
    if (k2 > kmax * kmax) continue;
    const phi = 2 * Math.PI * z * (Math.sqrt(Math.max(0, kn * kn - k2)) - kn);
    re[v * n + u] = Math.cos(phi); im[v * n + u] = Math.sin(phi);
  }
  return { re, im };
}

/**
 * Scalar pupil-based PSF |F⁻¹(P·e^{iφ})|² of an n × n field, with defocus z (nm) giving the
 * phase φ = 2π z (√((ri/λ)² − k²) − ri/λ). In focus (z = 0) this is the Airy pattern sampled
 * at pixel centres. Centred and normalised to sum 1.
 */
export function pupilPSF(n: number, o: PupilOptions, z = 0): Float64Array {
  const f = pupilField(n, o, z);
  fft2dRect(f.re, f.im, n, n, true);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) out[i] = f.re[i] * f.re[i] + f.im[i] * f.im[i];
  return normalise(fftshiftOrigin(out, n, n));
}

/** Move an origin-based array to the centre (fftshift for even sizes). */
function fftshiftOrigin(a: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h), sx = w >> 1, sy = h >> 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[((y + sy) % h) * w + (x + sx) % w] = a[y * w + x];
  return out;
}

// ---------------------------------------------------------------------------------------
// 3D PSFs (nx × nx lateral, nz planes, centred at plane nz/2)

export interface PSF3DOptions extends PupilOptions {
  zstep: number;          // nm between planes
}

/**
 * Widefield 3D PSF: one pupil PSF per defocus z = (k − nz/2)·zstep. Each plane carries the
 * same total light (out-of-focus light is spread, not lost); the volume sums to 1.
 * Index z·nx·nx + y·nx + x, centred at (nx/2, nx/2, nz/2).
 */
export function widefieldPSF3d(nx: number, nz: number, o: PSF3DOptions, oversample = 2): Float64Array {
  // The amplitude is computed on an oversampled field (os·nx) and the INTENSITY folded back
  // to nx: folding the amplitude instead would add interference between the wrapped copies.
  const out = new Float64Array(nx * nx * nz), m = nx * oversample, c = m >> 1, cn = nx >> 1;
  for (let k = 0; k < nz; k++) {
    const p = pupilPSF(m, o, (k - Math.floor(nz / 2)) * o.zstep);
    const base = k * nx * nx;
    for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
      const yy = (((y - c + cn) % nx) + nx) % nx, xx = (((x - c + cn) % nx) + nx) % nx;
      out[base + yy * nx + xx] += p[y * m + x];
    }
  }
  return normalise(out);
}

/**
 * Confocal 3D PSF with a point-like pinhole: excitation × emission, here both taken as the
 * widefield PSF at the same wavelength (PSF²). Out-of-focus planes become much dimmer.
 */
export function confocalPSF3d(nx: number, nz: number, o: PSF3DOptions, oversample = 2): Float64Array {
  // square on the oversampled field, then fold (squaring after folding would mix copies)
  const m = nx * oversample, big = widefieldPSF3d(m, nz, o, 1), out = new Float64Array(nx * nx * nz), c = m >> 1, cn = nx >> 1;
  for (let k = 0; k < nz; k++) for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
    const v = big[(k * m + y) * m + x], yy = (((y - c + cn) % nx) + nx) % nx, xx = (((x - c + cn) % nx) + nx) % nx;
    out[(k * nx + yy) * nx + xx] += v * v;
  }
  return normalise(out);
}

/** Gaussian 3D PSF with lateral σxy and axial σz (voxels), centred, sum 1. */
export function gaussianPSF3d(nx: number, nz: number, sxy: number, sz: number): Float64Array {
  const out = new Float64Array(nx * nx * nz), c = Math.floor(nx / 2), cz = Math.floor(nz / 2);
  for (let z = 0; z < nz; z++) for (let y = 0; y < nx; y++) for (let x = 0; x < nx; x++) {
    const dx = x - c, dy = y - c, dz = z - cz;
    out[(z * nx + y) * nx + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sxy * sxy) - dz * dz / (2 * sz * sz));
  }
  return normalise(out);
}

/** Move a centred nx × ny × nz volume so its centre is at index 0. */
export function toOrigin3d(v: ArrayLike<number>, nx: number, ny: number, nz: number): Float64Array {
  const out = new Float64Array(nx * ny * nz), cx = Math.floor(nx / 2), cy = Math.floor(ny / 2), cz = Math.floor(nz / 2);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
    out[(((z - cz + nz) % nz) * ny + ((y - cy + ny) % ny)) * nx + ((x - cx + nx) % nx)] = v[(z * ny + y) * nx + x];
  return out;
}

/** The xz slice (through y = ny/2) of a centred volume: nz rows of nx values. */
export function xzSlice(v: ArrayLike<number>, nx: number, ny: number, nz: number, y = Math.floor(ny / 2)): Float64Array {
  const out = new Float64Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) out[z * nx + x] = v[(z * ny + y) * nx + x];
  return out;
}

// ---------------------------------------------------------------------------------------
// OTFs and measurements

/** OTF (complex, lib/fft layout) of a centred w × h PSF. Real and ≤ 1 for a symmetric PSF summing to 1. */
export function otf2d(psf: ArrayLike<number>, w: number, h: number): { re: Float64Array; im: Float64Array } {
  const re = toOrigin(psf, w, h), im = new Float64Array(w * h);
  fft2dRect(re, im, w, h, false);
  return { re, im };
}

/** OTF of a centred 3D PSF (nx × ny × nz). */
export function otf3d(psf: ArrayLike<number>, nx: number, ny: number, nz: number): { re: Float64Array; im: Float64Array } {
  const re = toOrigin3d(psf, nx, ny, nz), im = new Float64Array(nx * ny * nz);
  fft3d(re, im, nx, ny, nz, false);
  return { re, im };
}

/** OTF of a centred 1D kernel of length n (power of two), lib/fft layout. */
export function otf1d(kernel: ArrayLike<number>, n: number): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(n), im = new Float64Array(n), c = Math.floor(n / 2);
  for (let i = 0; i < n; i++) re[(i - c + n) % n] = kernel[i];
  fft1d(re, im, false);
  return { re, im };
}

/** Radial frequency (cycles per pixel) of each index of a w × h spectrum in lib/fft layout. */
export function radialFrequency(w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) out[v * w + u] = Math.hypot(freq(u, w) / w, freq(v, h) / h);
  return out;
}

/**
 * Radial average of a spectrum-shaped array (lib/fft layout) in `bins` rings over
 * frequencies 0..fmax (cycles per pixel). Returns ring centres and means (NaN for empty rings).
 */
export function radialAverage(values: ArrayLike<number>, w: number, h: number, bins = 64, fmax = 0.5): { f: Float64Array; v: Float64Array } {
  const sum = new Float64Array(bins), cnt = new Float64Array(bins), fr = radialFrequency(w, h);
  for (let i = 0; i < w * h; i++) {
    const b = Math.floor(fr[i] / fmax * bins + 0.5);   // ring 0 = frequency 0 only (and nearest)
    if (b < bins) { sum[b] += values[i]; cnt[b]++; }
  }
  const f = new Float64Array(bins), v = new Float64Array(bins);
  for (let b = 0; b < bins; b++) { f[b] = b * fmax / bins; v[b] = cnt[b] ? sum[b] / cnt[b] : NaN; }
  return { f, v };
}

/** Full width at half maximum of a 1D profile around its maximum (linear interpolation), in samples. */
export function fwhm(profile: ArrayLike<number>, base = 0): number {
  let im = 0;
  for (let i = 1; i < profile.length; i++) if (profile[i] > profile[im]) im = i;
  const half = base + (profile[im] - base) / 2;
  let l = im, r = im;
  while (l > 0 && profile[l - 1] > half) l--;
  while (r < profile.length - 1 && profile[r + 1] > half) r++;
  const left = l > 0 ? l - (profile[l] - half) / (profile[l] - profile[l - 1]) : l;
  const right = r < profile.length - 1 ? r + (profile[r] - half) / (profile[r] - profile[r + 1]) : r;
  return right - left;
}
