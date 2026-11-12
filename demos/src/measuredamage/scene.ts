// The measuredamage demo's data, codecs and measurements (no DOM, so node scripts can compute
// the numbers quoted in the exercise).
// Scene: a synthetic 12-bit fluorescence image with ground truth: 60 Gaussian spots of varying
// brightness (peak 60..1500, log-uniform) and size (σ 1.0..2.2) inside a cell-shaped body with a
// darker nucleus, Poisson noise (one photon per grey level), fixed seed.
// Codecs: lossless (prediction + entropy code: identical pixels, size = MED-residual entropy);
// JPEG at quality q after a 12 → 8-bit rescale (0..max → 0..255) (lib/jpeg.ts); a simple wavelet codec (CDF 9/7,
// 5 levels, uniform dead-zone quantiser, size = entropy of the quantised coefficients per band)
// with its step chosen to match a given size.
// Measurements (the same pipeline on every image): spot detection (difference of Gaussians,
// a fixed threshold set on the original, 8-connected components, matching to the true spots
// within 3 px), per-spot integrated intensity in an aperture minus the local background,
// background mean and standard deviation, and the share of high-frequency power.
import { fromArray, gaussianBlur } from "../lib/imageops";
import { rng } from "../lib/random";
import { poisson } from "../lib/random";
import { label2d, centroids } from "../lib/label";
import { jpegGrey } from "../lib/jpeg";
import { residuals } from "../lib/predict";
import { entropy } from "../lib/entropy";
import { dwt2, idwt2, WAVELETS, subbandRect, Band } from "../lib/wavelet";
import { fft2d } from "../lib/fft";

export const N = 256;
export const MAX12 = 4095;

export interface Spot { x: number; y: number; sigma: number; amp: number; total: number }
export interface Scene { image: Float64Array; clean: Float64Array; spots: Spot[] }

/** The scene (deterministic). */
export function makeScene(seed = 12): Scene {
  const r = rng(seed);
  const ang = 20 * Math.PI / 180, ca = Math.cos(ang), sa = Math.sin(ang);
  const inCell = (x: number, y: number) => {
    const u = ((x - 128) * ca + (y - 128) * sa) / 105, v = (-(x - 128) * sa + (y - 128) * ca) / 78;
    return Math.sqrt(u * u + v * v);
  };
  const inNuc = (x: number, y: number) => Math.hypot((x - 120) / 36, (y - 122) / 28);
  const clean = new Float64Array(N * N);
  const sig = (d: number, w: number) => 1 / (1 + Math.exp((d - 1) * w));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    clean[y * N + x] = 150 + 400 * sig(inCell(x, y), 8) - 180 * sig(inNuc(x, y), 6);
  const spots: Spot[] = [];
  for (let tries = 0; spots.length < 60 && tries < 20000; tries++) {
    const x = 10 + (N - 20) * r.unif(), y = 10 + (N - 20) * r.unif();
    if (inCell(x, y) > 0.85 || inNuc(x, y) < 1.3) continue;
    if (spots.some(s => Math.hypot(s.x - x, s.y - y) < 10)) continue;
    const sigma = 1 + 1.2 * r.unif(), amp = 60 * Math.pow(25, r.unif());
    spots.push({ x, y, sigma, amp, total: 2 * Math.PI * sigma * sigma * amp });
  }
  for (const s of spots) {
    const R = Math.ceil(4 * s.sigma);
    for (let y = Math.max(0, Math.floor(s.y) - R); y <= Math.min(N - 1, Math.floor(s.y) + R); y++)
      for (let x = Math.max(0, Math.floor(s.x) - R); x <= Math.min(N - 1, Math.floor(s.x) + R); x++)
        clean[y * N + x] += s.amp * Math.exp(-((x - s.x) ** 2 + (y - s.y) ** 2) / (2 * s.sigma * s.sigma));
  }
  const r2 = rng(seed + 1000), image = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) image[i] = Math.min(MAX12, poisson(r2, clean[i]));
  return { image, clean, spots };
}

// ---------------------------------------------------------------------------------------
// Codecs

export interface Decoded { image: Float64Array; bytes: number; bpp: number; label: string }

/** Lossless: the image itself; size = entropy of the MED residuals (ideal coder) + 100 bytes. */
export function lossless(img: Float64Array): Decoded {
  const bytes = Math.ceil(entropy(residuals(img, N, N, "med")) * N * N / 8) + 100;
  return { image: img, bytes, bpp: 8 * bytes / (N * N), label: "lossless" };
}

/** Raw 16-bit storage size. */
export const RAW_BYTES = N * N * 2;

/** JPEG at quality q of the image rescaled to 8 bit (0..max → 0..255), decoded back to 12 bit. */
export function jpeg12(img: Float64Array, q: number): Decoded {
  let mx = 1;
  for (const v of img) if (v > mx) mx = v;
  const k = 255 / mx;
  const r = jpegGrey(img.map(v => Math.round(v * k)), N, N, { quality: q });
  return { image: r.decoded.map(v => v / k), bytes: r.bytes, bpp: r.bpp, label: `JPEG q = ${q}` };
}

const W97 = WAVELETS.cdf97, LEVELS = 5;

/** Wavelet codec with quantiser step `step` (12-bit units at the finest level's scale). */
export function waveletCode(img: Float64Array, step: number): Decoded {
  const c = dwt2(img, N, N, W97, LEVELS);
  const q = new Int32Array(N * N);
  for (let i = 0; i < N * N; i++) { const v = c[i] / step; q[i] = v < 0 ? -Math.floor(-v) : Math.floor(v); }   // dead zone
  // size: entropy of each subband's integers
  let bits = 0;
  const bands: { x: number; y: number; w: number; h: number }[] = [subbandRect(LEVELS, "LL", N, N)];
  for (let l = 1; l <= LEVELS; l++) for (const b of ["H", "V", "D"] as Band[]) bands.push(subbandRect(l, b, N, N));
  for (const b of bands) {
    const vals = new Float64Array(b.w * b.h);
    for (let y = 0; y < b.h; y++) for (let x = 0; x < b.w; x++) vals[y * b.w + x] = q[(b.y + y) * N + b.x + x];
    bits += entropy(vals) * vals.length + 64;   // + a small table cost per band
  }
  const deq = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) deq[i] = q[i] === 0 ? 0 : (q[i] + (q[i] > 0 ? 0.5 : -0.5)) * step;
  const rec = idwt2(deq, N, N, W97, LEVELS);
  const image = rec.map(v => Math.min(MAX12, Math.max(0, Math.round(v))));
  const bytes = Math.ceil(bits / 8) + 100;
  return { image, bytes, bpp: 8 * bytes / (N * N), label: `wavelet, step ${step.toFixed(1)}` };
}

/** The wavelet codec at about `bytes` bytes (bisection on the step, log scale). */
export function waveletAtSize(img: Float64Array, bytes: number): Decoded {
  let lo = 0.5, hi = 4000, best = waveletCode(img, hi);
  for (let k = 0; k < 24; k++) {
    const mid = Math.sqrt(lo * hi), r = waveletCode(img, mid);
    if (Math.abs(r.bytes - bytes) < Math.abs(best.bytes - bytes)) best = r;
    if (r.bytes > bytes) lo = mid; else hi = mid;
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// Measurements

/** Difference of Gaussians (σ 1.2 and 6): spots positive, smooth background removed. */
export function dog(img: Float64Array): Float64Array {
  const g = fromArray(N, N, img);
  const a = gaussianBlur(g, 1.2).data, b = gaussianBlur(g, 6).data;
  return Float64Array.from(a, (v, i) => v - b[i]);
}

/** Detection threshold, set once on the original: 5 robust standard deviations of the DoG. */
export function detectionThreshold(img: Float64Array): number {
  const d = Array.from(dog(img)).sort((a, b) => a - b);
  const med = d[d.length >> 1];
  const mad = Array.from(d, v => Math.abs(v - med)).sort((a, b) => a - b)[d.length >> 1];
  return med + 5 * 1.4826 * mad;
}

export interface Detection { found: number; missed: number; falsePos: number; components: number; merged: number; centres: { x: number; y: number }[] }

/** Detect spots (DoG > T, components ≥ 3 px) and match them to the true spots within 3 px. */
export function detect(img: Float64Array, T: number, spots: Spot[]): Detection {
  const d = dog(img), mask = Uint8Array.from(d, v => (v > T ? 1 : 0));
  const lab = label2d(mask, N, N, 8), cs = centroids(lab).slice(1).filter(c => c.n >= 3);
  const used = new Uint8Array(cs.length);
  let found = 0;
  for (const s of spots) {
    let best = -1, bd = 3;
    cs.forEach((c, j) => { const dd = Math.hypot(c.x - s.x, c.y - s.y); if (!used[j] && dd <= bd) { bd = dd; best = j; } });
    if (best >= 0) { used[best] = 1; found++; }
  }
  // merged: components that contain the centre pixels of two or more true spots
  const hits = new Map<number, number>();
  for (const s of spots) { const l = lab.data[Math.round(s.y) * N + Math.round(s.x)]; if (l) hits.set(l, (hits.get(l) ?? 0) + 1); }
  let merged = 0;
  for (const v of hits.values()) if (v > 1) merged++;
  return { found, missed: spots.length - found, falsePos: cs.length - used.reduce((a, b) => a + b, 0), components: cs.length, merged, centres: cs.map(c => ({ x: c.x, y: c.y })) };
}

/** Integrated intensity of each spot: sum over r ≤ 3.5 minus the mean of the annulus 5 ≤ r ≤ 7. */
export function integrated(img: Float64Array, spots: Spot[]): Float64Array {
  return Float64Array.from(spots, s => {
    let sum = 0, n = 0, bs = 0, bn = 0;
    for (let y = Math.floor(s.y) - 8; y <= Math.floor(s.y) + 8; y++) for (let x = Math.floor(s.x) - 8; x <= Math.floor(s.x) + 8; x++) {
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      const r = Math.hypot(x - s.x, y - s.y), v = img[y * N + x];
      if (r <= 3.5) { sum += v; n++; } else if (r >= 5 && r <= 7) { bs += v; bn++; }
    }
    return sum - n * (bn ? bs / bn : 0);
  });
}

/** Background mask: inside the cell body, farther than 9 px from every spot. */
export function backgroundMask(spots: Spot[], clean: Float64Array): Uint8Array {
  const m = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (clean[y * N + x] < 450) continue;
    if (spots.every(s => Math.hypot(s.x - x, s.y - y) > 9)) m[y * N + x] = 1;
  }
  return m;
}

/** Mean and standard deviation over a mask. */
export function maskedStats(img: Float64Array, m: Uint8Array): { mean: number; sd: number } {
  let s = 0, s2 = 0, n = 0;
  for (let i = 0; i < img.length; i++) if (m[i]) { s += img[i]; s2 += img[i] * img[i]; n++; }
  const mean = s / n;
  return { mean, sd: Math.sqrt(Math.max(0, s2 / n - mean * mean)) };
}

/** Power spectrum (unshifted) of an image with its mean removed. */
export function powerSpectrum(img: Float64Array): Float64Array {
  let mean = 0;
  for (const v of img) mean += v;
  mean /= img.length;
  const re = Float64Array.from(img, v => v - mean), im = new Float64Array(N * N);
  fft2d(re, im, N);
  return Float64Array.from(re, (v, i) => v * v + im[i] * im[i]);
}

/** Share of the power at radial frequency above N/4 (fine texture and noise). */
export function highFreqShare(img: Float64Array): number {
  const P = powerSpectrum(img);
  let hi = 0, tot = 0;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const fx = x < N / 2 ? x : x - N, fy = y < N / 2 ? y : y - N, p = P[y * N + x];
    tot += p;
    if (Math.hypot(fx, fy) > N / 4) hi += p;
  }
  return hi / tot;
}

export interface Measures { det: Detection; integ: Float64Array; bg: { mean: number; sd: number }; hf: number }

export function measure(img: Float64Array, T: number, sc: Scene, bgMask: Uint8Array): Measures {
  return { det: detect(img, T, sc.spots), integ: integrated(img, sc.spots), bg: maskedStats(img, bgMask), hf: highFreqShare(img) };
}

/** Mean relative change (%) of the integrated intensity, over the spots with index in `idx`. */
export function meanChange(a: Float64Array, b: Float64Array, idx: number[]): number {
  let s = 0;
  for (const i of idx) s += (b[i] - a[i]) / Math.max(1, Math.abs(a[i]));
  return 100 * s / Math.max(1, idx.length);
}

/** Spot indices of the dimmest and brightest thirds by true integrated intensity. */
export function thirds(spots: Spot[]): { dim: number[]; bright: number[] } {
  const ord = spots.map((_, i) => i).sort((a, b) => spots[a].total - spots[b].total), k = Math.floor(spots.length / 3);
  return { dim: ord.slice(0, k), bright: ord.slice(spots.length - k) };
}

/** Mean absolute relative change (%) of the integrated intensity over the spots in `idx`. */
export function meanAbsChange(a: Float64Array, b: Float64Array, idx: number[]): number {
  let s = 0;
  for (const i of idx) s += Math.abs(b[i] - a[i]) / Math.max(1, Math.abs(a[i]));
  return 100 * s / Math.max(1, idx.length);
}
