// The eight smoothers of the smoothercompare demo, each with one main parameter, and the
// scores read off the test card (no DOM: also used by the figure script of smoothers/).
import { gaussianHeat, nonlinearStep, testCard, addNoise, addSaltPepper, addPoisson, psnr255, edgeWidth1090, staircaseIndex } from "../lib/pde";
import { medianFilter, bilateral, nonLocalMeans, wienerDenoise } from "../lib/denoise";
import { tv2d } from "../lib/smooth";
import { denoise2, wavelet } from "../lib/wavelet";
import { ssim } from "../lib/metrics";

export type MethodId = "gauss" | "median" | "pm" | "tv" | "bilateral" | "nlm" | "wavelet" | "wiener";
export interface Method {
  id: MethodId; label: string; param: string; lo: number; hi: number; step: number; log?: boolean;
  /** What the method believes about the image. */
  prior: string;
  run(y: Float64Array, p: number, sigmaN: number): Float64Array;
}
const N = 128;

export const METHODS: Method[] = [
  { id: "gauss", label: "Gaussian (= heat)", param: "σ", lo: 0.3, hi: 4, step: 0.05, prior: "the image is smooth", run: (y, p) => gaussianHeat(y, N, N, p) },
  { id: "median", label: "median", param: "radius", lo: 1, hi: 3, step: 1, prior: "outliers are rare; structures wider than the window", run: (y, p) => medianFilter(y, N, N, Math.round(p)) },
  { id: "pm", label: "Perona–Malik", param: "time", lo: 0.1, hi: 20, step: 0.1, log: true, prior: "piecewise smooth, edges higher than K",
    run: (y, p, s) => { let u: Float64Array = Float64Array.from(y); const K = Math.max(5, 1.2 * s); for (let t = 0; t < p - 1e-9; t += 0.2) u = nonlinearStep(u, N, N, Math.min(0.2, p - t), "g2", K, 1); return u; } },
  { id: "tv", label: "TV", param: "λ", lo: 1, hi: 60, step: 0.5, prior: "piecewise constant, short total edge length", run: (y, p) => tv2d(y, N, N, p, 200) },
  { id: "bilateral", label: "bilateral", param: "σr", lo: 5, hi: 150, step: 1, prior: "nearby similar values belong together", run: (y, p) => bilateral(y, N, N, 2, p) },
  { id: "nlm", label: "non-local means", param: "h", lo: 3, hi: 60, step: 0.5, prior: "the image repeats itself", run: (y, p) => nonLocalMeans(y, N, N, 2, 6, p) },
  { id: "wavelet", label: "wavelet shrinkage", param: "threshold", lo: 2, hi: 150, step: 1, prior: "few large wavelet coefficients", run: (y, p) => denoise2(y, N, N, wavelet("db2"), 3, p, "soft", 4).out },
  { id: "wiener", label: "Wiener", param: "SNR", lo: 0.001, hi: 10, step: 0.001, log: true, prior: "a known power spectrum (1/f²)", run: (y, p) => wienerDenoise(y, N, N, p) },
];

export type NoiseKind = "gauss" | "poisson" | "saltpepper";
export interface Scenario { image: "card" | "cell"; noise: NoiseKind; level: number }

/** Noisy image for a scenario (seeded). For Poisson, level = photons at 255. */
export function noisyOf(truth: Float64Array, sc: Scenario): Float64Array {
  if (sc.noise === "gauss") return addNoise(truth, sc.level, 13);
  if (sc.noise === "poisson") return addPoisson(truth, sc.level, 13);
  return addSaltPepper(truth, sc.level / 100, 0, 255, 13);
}
/** The noise SD seen by the methods that need it (Perona–Malik's K). */
export function noiseSd(truth: Float64Array, y: Float64Array): number {
  let s = 0, s2 = 0;
  for (let i = 0; i < y.length; i++) { const d = y[i] - truth[i]; s += d; s2 += d * d; }
  return Math.sqrt(s2 / y.length - (s / y.length) ** 2);
}

export interface Scores { psnr: number; ssim: number; edge: number; line: number; disc: number; stairs: number }

/** Scores on the test card: PSNR, SSIM, 10–90 % width of the step at x = 10, contrast kept of the
 *  1-px line (x = 82) and of the r = 3 disc (3 × 3 means), staircase index of the ramp (x = 34 … 66); the
 *  profile is the mean of rows 100 … 123. */
export function scores(u: Float64Array, truth: Float64Array): Scores {
  const prof = new Float64Array(N), tp = new Float64Array(N);
  for (let x = 0; x < N; x++) { let s = 0, t = 0; for (let y = 100; y < 124; y++) { s += u[y * N + x]; t += truth[y * N + x]; } prof[x] = s / 24; tp[x] = t / 24; }
  const line = (prof[82] - 0.5 * (prof[79] + prof[85])) / (tp[82] - 0.5 * (tp[79] + tp[85]));
  const m3 = (a: Float64Array, x: number, y: number) => { let s = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) s += a[(y + j) * N + x + i]; return s / 9; };
  const disc = (m3(u, 52, 22) - m3(u, 60, 22)) / (m3(truth, 52, 22) - m3(truth, 60, 22));
  return { psnr: psnr255(u, truth), ssim: ssim(u, truth, N, N).mean, edge: edgeWidth1090(prof, 4, 16), line, disc, stairs: staircaseIndex(prof, 36, 64) };
}

export const cardTruth = (): Float64Array => testCard().truth;
