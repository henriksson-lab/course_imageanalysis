// Model of the noisespectrum demo (no DOM; used by demos/test/diffusion-test.ts and the
// figure script tools/figs/diffusion/). The "Gaussian image model": the image's own radially
// averaged power spectrum P(f); white noise of level σ has power N²σ² per coefficient.
import { fft2d } from "../lib/fft";
import { rng } from "../lib/ui";

export const N = 128;

/** Normalise to mean 0, variance 1. */
export function standardise(v: ArrayLike<number>): Float64Array {
  let m = 0; for (let i = 0; i < v.length; i++) m += v[i]; m /= v.length;
  let s = 0; for (let i = 0; i < v.length; i++) s += (v[i] - m) ** 2; s = Math.sqrt(s / v.length);
  return Float64Array.from(v, x => (x - m) / s);
}

export const freqIdx = (k: number) => k <= N / 2 ? k : k - N;

/** Radially averaged power |X|² in integer frequency bins 0..N/2 (cycles per image), and the per-coefficient model P(k). */
export function powerModel(x: Float64Array): { radial: Float64Array; P: Float64Array } {
  const re = Float64Array.from(x), im = new Float64Array(N * N);
  fft2d(re, im, N);
  const sum = new Float64Array(N / 2 + 2), cnt = new Float64Array(N / 2 + 2);
  for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
    const f = Math.hypot(freqIdx(u), freqIdx(v)), b = Math.round(f);
    if (b <= N / 2) { sum[b] += re[v * N + u] ** 2 + im[v * N + u] ** 2; cnt[b]++; }
  }
  const radial = new Float64Array(N / 2 + 1);
  for (let b = 0; b <= N / 2; b++) radial[b] = cnt[b] ? sum[b] / cnt[b] : 0;
  const P = new Float64Array(N * N);
  for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
    const f = Math.min(N / 2, Math.hypot(freqIdx(u), freqIdx(v))), b = Math.floor(f), t = f - b;
    P[v * N + u] = b >= N / 2 ? radial[N / 2] : (1 - t) * radial[b] + t * radial[b + 1];
  }
  P[0] = 0; // the mean is 0
  return { radial, P };
}


/** One noise level: the noisy image, the Wiener estimate, the crossover frequency and both errors. */
export function analyse(x: Float64Array, model: { radial: Float64Array; P: Float64Array }, sigma: number, noise: Float64Array): { noisy: Float64Array; wiener: Float64Array; crossover: number; mseNoisy: number; mseWiener: number } {
  const noisy = Float64Array.from(x, (v, i) => v + sigma * noise[i]);
  const nf = N * N * sigma * sigma, re = Float64Array.from(noisy), im = new Float64Array(N * N);
  fft2d(re, im, N);
  for (let i = 0; i < N * N; i++) { const g = model.P[i] / (model.P[i] + nf); re[i] *= g; im[i] *= g; }
  fft2d(re, im, N, true);
  let fc = 0;
  for (let b = 1; b <= N / 2; b++) if (model.radial[b] > nf) fc = b;
  let eN = 0, eW = 0;
  for (let i = 0; i < N * N; i++) { eN += (noisy[i] - x[i]) ** 2; eW += (re[i] - x[i]) ** 2; }
  return { noisy, wiener: re, crossover: fc, mseNoisy: eN / (N * N), mseWiener: eW / (N * N) };
}

/** The fixed noise field of the demo (seed 3). */
export function noiseField(): Float64Array { const r = rng(3), v = new Float64Array(N * N); for (let i = 0; i < v.length; i++) v[i] = r.norm(); return v; }

/** A sample of the Gaussian image model: white noise shaped by √P / N. */
export function gaussianSample(P: Float64Array, seed: number): Float64Array {
  const r = rng(seed), sr = new Float64Array(N * N), si = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) sr[i] = r.norm();
  fft2d(sr, si, N);
  for (let i = 0; i < N * N; i++) { const g = Math.sqrt(P[i]) / N; sr[i] *= g; si[i] *= g; }
  fft2d(sr, si, N, true);
  return sr;
}
