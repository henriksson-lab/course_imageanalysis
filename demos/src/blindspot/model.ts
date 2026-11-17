// Model of the blindspot demo (no DOM; tested in demos/test/blindspot-test.ts).
//
// Self-supervised denoising with a linear filter over a (2r+1)² window, trained by least
// squares on ONE noisy image (unit self-denoise/):
//   n2v       Noise2Void: input = target = the noisy image, the centre weight forced to 0
//             (the blind spot);
//   ident     the same without the blind spot: the best prediction of a pixel is itself;
//   n2n       Noise2Noise: target = a second, independent noisy image of the same thing;
//   sup       supervised: target = the clean image (only possible in a simulation);
//   supblind  supervised with the blind spot (the best blind-spot filter: Noise2Void should
//             find the same one when the noise is independent from pixel to pixel).
// Optionally the pixel is "mixed back": out = pred + c (noisy − pred), c = 1 − σ²/mean(noisy − pred)²
// with the noise level σ known. Noise: independent Gaussian, correlated (white noise blurred
// with a Gaussian of 1 px, rescaled to σ), or stripes (one random offset per row plus
// independent noise, half the variance each).
import { gaussianDenoise, nonLocalMeans } from "../lib/denoise";
import { solve } from "../lib/linalg";
import { psnr } from "../lib/metrics";
import { rng } from "../lib/ui";

export type NoiseKind = "white" | "corr" | "stripes";
export type Train = "n2v" | "ident" | "n2n" | "sup" | "supblind";
export const TRAIN_LABEL: Record<Train, string> = {
  n2v: "Noise2Void (blind spot)", ident: "no blind spot", n2n: "Noise2Noise (2 noisy images)",
  sup: "supervised (clean target)", supblind: "supervised, blind spot",
};

/** Noise of standard deviation `sigma` (see the header), seeded. */
export function noiseField(w: number, h: number, sigma: number, kind: NoiseKind, seed: number): Float64Array {
  const r = rng(seed), n = new Float64Array(w * h);
  for (let i = 0; i < n.length; i++) n[i] = r.norm();
  if (kind === "white") return n.map(v => sigma * v);
  if (kind === "stripes") {
    const rowOff = Float64Array.from({ length: h }, () => r.norm());
    const s = sigma / Math.SQRT2;
    return n.map((v, i) => s * v + s * rowOff[Math.floor(i / w)]);
  }
  // correlated: blur with a Gaussian of 1 px (radius 4, wrap-around), rescale to sigma
  const k = Float64Array.from({ length: 9 }, (_, i) => Math.exp(-((i - 4) ** 2) / 2));
  const ks = k.reduce((a, b) => a + b, 0);
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0; for (let d = -4; d <= 4; d++) s += k[d + 4] * n[y * w + ((x + d + w) % w)];
    tmp[y * w + x] = s / ks;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0; for (let d = -4; d <= 4; d++) s += k[d + 4] * tmp[((y + d + h) % h) * w + x];
    out[y * w + x] = s / ks;
  }
  let ss = 0; for (const v of out) ss += v * v;
  const sd = Math.sqrt(ss / out.length);
  return out.map(v => sigma * v / sd);
}

/** Mirror index (the edge pixel repeated, as in the course library). */
function mir(i: number, n: number): number {
  const j = ((i % (2 * n)) + 2 * n) % (2 * n);
  return j < n ? j : 2 * n - 1 - j;
}

/** The (2r+1)² window values around (x, y), row by row (mirror border), without the centre if blind. */
export function window(img: ArrayLike<number>, w: number, h: number, x: number, y: number, r: number, blind: boolean): number[] {
  const v: number[] = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    if (blind && dx === 0 && dy === 0) continue;
    v.push(img[mir(y + dy, h) * w + mir(x + dx, w)]);
  }
  return v;
}

export interface Filter { r: number; weights: Float64Array; bias: number }   // weights: (2r+1)², row by row

/** Least-squares fit of the filter (normal equations; one row per pixel: window values and a 1). */
export function fitFilter(input: ArrayLike<number>, target: ArrayLike<number>, w: number, h: number, r: number, blind: boolean): Filter {
  const n = (2 * r + 1) ** 2, d = (blind ? n - 1 : n) + 1;
  const A = new Float64Array(d * d), b = new Float64Array(d);
  const row = new Float64Array(d);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = window(input, w, h, x, y, r, blind);
    for (let i = 0; i < d - 1; i++) row[i] = v[i];
    row[d - 1] = 1;
    const t = target[y * w + x];
    for (let i = 0; i < d; i++) {
      b[i] += row[i] * t;
      const ri = row[i];
      for (let j = i; j < d; j++) A[i * d + j] += ri * row[j];
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) A[i * d + j] = A[j * d + i];
  const sol = solve(A, b, d) ?? new Float64Array(d);
  const weights = new Float64Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) weights[i] = blind && i === (n - 1) / 2 ? 0 : sol[k++];
  return { r, weights, bias: sol[d - 1] };
}

/** Apply a filter (mirror border). */
export function applyFilter(img: ArrayLike<number>, w: number, h: number, f: Filter): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = window(img, w, h, x, y, f.r, false);
    let s = f.bias;
    for (let i = 0; i < v.length; i++) s += f.weights[i] * v[i];
    out[y * w + x] = s;
  }
  return out;
}

/** Mix the pixel back into a blind-spot prediction, with the noise level known. */
export function mixBack(noisy: ArrayLike<number>, pred: ArrayLike<number>, sigma: number): { out: Float64Array; c: number } {
  let s = 0;
  for (let i = 0; i < noisy.length; i++) s += (noisy[i] - pred[i]) ** 2;
  const c = Math.max(0, 1 - sigma * sigma / (s / noisy.length));
  return { out: Float64Array.from(pred, (p, i) => p + c * (noisy[i] - p)), c };
}

/** Train the filter of one kind. `noisy2` is the second noisy image (for Noise2Noise). */
export function train(kind: Train, clean: ArrayLike<number>, noisy: ArrayLike<number>, noisy2: ArrayLike<number>, w: number, h: number, r: number): Filter {
  switch (kind) {
    case "n2v": return fitFilter(noisy, noisy, w, h, r, true);
    case "ident": return fitFilter(noisy, noisy, w, h, r, false);
    case "n2n": return fitFilter(noisy, noisy2, w, h, r, false);
    case "sup": return fitFilter(noisy, clean, w, h, r, false);
    case "supblind": return fitFilter(noisy, clean, w, h, r, true);
  }
}

// ---------------------------------------------------------------------------------------
// Everything the demo shows for one setting.

export type Cmp = "gauss" | "nlm";
export interface Settings { noise: NoiseKind; sigma: number; r: number; train: Train; mix: boolean; cmp: Cmp; gsig: number; nlmH: number }
export interface Result {
  noisy: Float64Array; out: Float64Array; cmpOut: Float64Array; filter: Filter; c: number;
  psnr: { noisy: number; out: number; cmp: number; all: Record<Train, number>; mixed: number };
}

export const W = 128;

/** Train every kind (for the table), return the selected one and the comparison filter. */
export function evaluate(clean: Float64Array, s: Settings): Result {
  const noisy = Float64Array.from(clean), noisy2 = Float64Array.from(clean);
  const n1 = noiseField(W, W, s.sigma, s.noise, 101), n2 = noiseField(W, W, s.sigma, s.noise, 202);
  for (let i = 0; i < noisy.length; i++) { noisy[i] += n1[i]; noisy2[i] += n2[i]; }
  const all = {} as Record<Train, number>;
  let filter: Filter | null = null, out: Float64Array = noisy;
  let n2vOut: Float64Array = noisy;
  for (const k of Object.keys(TRAIN_LABEL) as Train[]) {
    const f = train(k, clean, noisy, noisy2, W, W, s.r);
    const o = applyFilter(noisy, W, W, f);
    all[k] = psnr(o, clean, 255);
    if (k === "n2v") n2vOut = o;
    if (k === s.train) { filter = f; out = o; }
  }
  const mixed = mixBack(noisy, n2vOut, s.sigma);
  let c = NaN;
  if (s.mix && (s.train === "n2v" || s.train === "supblind")) {
    const m = s.train === "n2v" ? mixed : mixBack(noisy, out, s.sigma);
    out = m.out; c = m.c;
  }
  const cmpOut = s.cmp === "gauss" ? gaussianDenoise(noisy, W, W, s.gsig) : nonLocalMeans(noisy, W, W, 1, 5, s.nlmH);
  return {
    noisy, out, cmpOut, filter: filter!, c,
    psnr: { noisy: psnr(noisy, clean, 255), out: psnr(out, clean, 255), cmp: psnr(cmpOut, clean, 255), all, mixed: psnr(mixed.out, clean, 255) },
  };
}
