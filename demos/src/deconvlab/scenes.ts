// Test objects and the forward model for the deconvolution demos (no DOM).
// A scene is a function of field coordinates (x, y) in pixels of an N × N field, values
// about 0..1. `render` samples it with 2 × 2 sub-samples per pixel (area sampling), on the
// field itself or on a finer / wider grid for the "realistic" simulation.
import { rng } from "../lib/ui";
import { poisson } from "../lib/random";
import * as P from "../lib/psf";
import { RealConvolver, spectrum2d } from "../lib/deconv";

export type SceneFn = (x: number, y: number) => number;
export type SceneId = "phantom" | "star" | "beads" | "cell";

const smoothstep = (e: number) => (e <= -1 ? 1 : e >= 1 ? 0 : 0.5 - 0.5 * Math.sin(e * Math.PI / 2));
const ell = (x: number, y: number, cx: number, cy: number, a: number, b: number, th: number) => {
  const c = Math.cos(th), s = Math.sin(th), dx = x - cx, dy = y - cy;
  const u = (c * dx + s * dy) / a, v = (-s * dx + c * dy) / b;
  return (Math.sqrt(u * u + v * v) - 1) * Math.min(a, b);   // ≈ signed distance in pixels
};

/** Synthetic cell phantom (seeded): cell body, nucleus, nucleolus, granules, filaments; a second cell cut by the border. */
export function phantom(N: number, seed = 7): SceneFn {
  const r = rng(seed), k = N / 128;
  const cx = 60 * k, cy = 66 * k;
  const gran: [number, number, number][] = [];
  while (gran.length < 14) {
    const x = cx + (r.unif() - 0.5) * 80 * k, y = cy + (r.unif() - 0.5) * 60 * k;
    if (ell(x, y, cx, cy, 44 * k, 32 * k, 0.3) < -3 && ell(x, y, cx + 4 * k, cy - 2 * k, 17 * k, 13 * k, -0.2) > 2) gran.push([x, y, 0.25 + 0.35 * r.unif()]);
  }
  const fil = [0, 1, 2].map(i => ({ a: 0.4 + i * 0.9 + 0.2 * r.unif(), off: (i - 1) * 9 * k }));
  return (x, y) => {
    let v = 0.05;
    v += 0.3 * smoothstep(ell(x, y, cx, cy, 44 * k, 32 * k, 0.3) / 1.2);
    v += 0.25 * smoothstep(ell(x, y, cx + 4 * k, cy - 2 * k, 17 * k, 13 * k, -0.2) / 1.0);
    v += 0.3 * smoothstep(ell(x, y, cx + 8 * k, cy - 4 * k, 4 * k, 3.5 * k, 0) / 0.8);
    for (const [gx, gy, a] of gran) v += a * Math.exp(-((x - gx) ** 2 + (y - gy) ** 2) / (2 * 0.8 * 0.8 * k * k));
    // filaments: straight segments through the cytoplasm, 1 px wide
    if (ell(x, y, cx, cy, 44 * k, 32 * k, 0.3) < -2 && ell(x, y, cx + 4 * k, cy - 2 * k, 17 * k, 13 * k, -0.2) > 1)
      for (const f of fil) { const d = Math.abs(-Math.sin(f.a) * (x - cx) + Math.cos(f.a) * (y - cy) - f.off - 20 * k); v += 0.18 * Math.exp(-d * d / (2 * 0.5 * 0.5 * k * k)); }
    // a second cell, half outside the right border
    v += 0.35 * smoothstep(ell(x, y, 128 * k, 20 * k, 22 * k, 16 * k, 0.5) / 1.2);
    return v;
  };
}

/** Siemens star with `spokes` black/white sector pairs, radius 0.45 N. */
export function star(N: number, spokes = 18): SceneFn {
  const c = N / 2, R = 0.45 * N;
  return (x, y) => {
    const dx = x - c, dy = y - c, r = Math.hypot(dx, dy);
    if (r > R) return 0.1;
    return Math.sin(spokes * Math.atan2(dy, dx)) > 0 ? 0.8 : 0.1;
  };
}

/** Bead positions of the "beads" scene in a 128 field: [x, y, brightness]. */
export const BEADS: [number, number, number][] = [
  [24, 24, 1], [24 + 6, 24, 0.25],          // a 1 : 4 pair, 6 px apart
  [24, 50, 0.5], [24 + 4, 50, 0.5],         // a close equal pair
  [24, 76, 0.5], [24 + 9, 76, 0.5],         // a wider pair
  [24, 104, 0.8],                            // a single bead
  [60, 30, 0.6], [58, 92, 0.3],
];

/** Beads (point-like, σ = 0.6 px) of several brightnesses and pairs, a step edge and a bar. */
export function beads(N: number): SceneFn {
  const k = N / 128;
  return (x, y) => {
    let v = 0.05;
    for (const [bx, by, a] of BEADS) v += 2.5 * a * Math.exp(-((x - bx * k) ** 2 + (y - by * k) ** 2) / (2 * 0.6 * 0.6 * k * k));
    if (x > 88 * k) v += 0.45;                              // step edge, runs off top and bottom
    if (Math.abs(x - 72 * k) < 1.5 * k && y > 50 * k && y < 80 * k) v += 0.5;   // a thin bar
    return v;
  };
}

/** The micrograph (a 2N × 2N array, values 0..1) as a scene on an N field, mirrored outside. */
export function micrograph(img: ArrayLike<number>, M: number, N: number): SceneFn {
  const s = M / N;
  const at = (i: number, j: number) => {
    const m = (t: number) => { const p = 2 * (M - 1); t = ((t % p) + p) % p; return t < M ? t : p - t; };
    return img[m(j) * M + m(i)];
  };
  return (x, y) => {
    const u = (x + 0.5) * s - 0.5, v = (y + 0.5) * s - 0.5, i = Math.floor(u), j = Math.floor(v), fx = u - i, fy = v - j;
    return (1 - fy) * ((1 - fx) * at(i, j) + fx * at(i + 1, j)) + fy * ((1 - fx) * at(i, j + 1) + fx * at(i + 1, j + 1));
  };
}

/** Sample a scene on an n × n grid: pixel (i, j) covers field [x0 + i/s, x0 + (i+1)/s) (s = pixels per field pixel). */
export function render(f: SceneFn, n: number, s = 1, x0 = 0, y0 = 0): Float64Array {
  const out = new Float64Array(n * n), d = 1 / s;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = x0 + (i + 0.5) * d - 0.5, y = y0 + (j + 0.5) * d - 0.5;   // pixel centre in field coordinates
    out[j * n + i] = 0.25 * (f(x - d / 4, y - d / 4) + f(x + d / 4, y - d / 4) + f(x - d / 4, y + d / 4) + f(x + d / 4, y + d / 4));
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// PSFs and the forward model

export type PsfKind = "gauss" | "airy" | "motion";
export interface PsfSpec { kind: PsfKind; width: number; angle?: number }
export const WAVELENGTH = 520, PIXEL = 65;   // nm; Airy: NA 1.0 → first zero 4.9 px, cut-off 0.25 cycles/px

/** Centred n × n PSF; `scale` = grid pixels per field pixel; `factor` widens it (PSF used for restoration). */
export function makePSF(n: number, spec: PsfSpec, scale = 1, factor = 1): Float64Array {
  if (spec.kind === "gauss") return P.gaussianPSF(n, n, spec.width * factor * scale);
  if (spec.kind === "airy") return P.airyPSFBandlimited(n, n, { NA: spec.width / factor, wavelength: WAVELENGTH, pixel: PIXEL / scale });
  return P.motionPSF(n, n, spec.width * factor * scale, spec.angle ?? 0);
}

export interface NoiseSpec { kind: "gauss" | "photons"; level: number }   // σ_n as a fraction of 1, or photons at value 1

/** Noise standard deviation (in image units) for a given clean image. */
export function noiseSD(noise: NoiseSpec, clean: ArrayLike<number>): number {
  if (noise.kind === "gauss") return noise.level;
  let m = 0;
  for (let i = 0; i < clean.length; i++) m += Math.max(0, clean[i]);
  return Math.sqrt(m / clean.length / noise.level);
}

export function addNoise(clean: ArrayLike<number>, noise: NoiseSpec, seed = 11): Float64Array {
  const r = rng(seed), out = new Float64Array(clean.length);
  if (noise.kind === "gauss") for (let i = 0; i < out.length; i++) out[i] = clean[i] + noise.level * r.norm();
  else for (let i = 0; i < out.length; i++) out[i] = poisson(r, Math.max(0, clean[i]) * noise.level) / noise.level;
  return out;
}

/** Circular convolution of an n × n image with a centred PSF. */
export function blur(img: ArrayLike<number>, psf: ArrayLike<number>, n: number): Float64Array {
  const H = spectrum2d(P.toOrigin(psf, n, n), n, n);
  return new RealConvolver(n, n, 1, H).apply(img);
}

/**
 * The forward model. ideal: truth on the N field, circular blur with the same PSF as the
 * restoration will use (the "inverse crime"). realistic: the scene continues N/2 beyond
 * every border and is blurred on a 2× finer grid, then binned 2 × 2 and cropped: blur
 * reaches in from outside the field, and the PSF is not exactly the discrete one.
 */
export function simulate(scene: SceneFn, N: number, psf: PsfSpec, realistic: boolean): { truth: Float64Array; clean: Float64Array } {
  const truth = render(scene, N);
  if (!realistic) return { truth, clean: blur(truth, makePSF(N, psf), N) };
  const M = 4 * N;                                // fine grid: 2N field pixels at 2× = 4N samples
  const fine = render(scene, M, 2, -N / 2, -N / 2);
  const bl = blur(fine, makePSF(M, psf, 2), M);
  const clean = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const X = 2 * (x + N / 2), Y = 2 * (y + N / 2);
    clean[y * N + x] = 0.25 * (bl[Y * M + X] + bl[Y * M + X + 1] + bl[(Y + 1) * M + X] + bl[(Y + 1) * M + X + 1]);
  }
  return { truth, clean };
}
