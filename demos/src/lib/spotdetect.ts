// Spot (point) detection, owned by the point-detection day (no DOM): spot filters (Gaussian,
// zero-mean Gaussian = matched filter under an unknown background, LoG, DoG, top-hat +
// Gaussian, à trous multiscale product), the noise gain of a filter, MAD noise estimates,
// local noise maps and the Anscombe transform, the Euler-characteristic estimate of the
// number of noise maxima above a threshold, 1D helpers (filters, local maxima, h-maxima /
// prominence) for the peaks1d demo, a seeded spot-image generator that allows crowding
// (pixel-integrated spots, no minimum distance), and distance-gated one-to-one point
// matching (greedy or optimal, built on eval.hungarian).
// Local maxima come from edges.localMaxima2D (no new 2D peak finder).
import { Grey, createGrey, separable, gaussianKernel, gaussianBlur, logFilter, dogFilter, boxFilter, Border, fromArray } from "./imageops";
import { localMaxima2D, correlate1d } from "./edges";
import { whiteTopHat, seDisk } from "./morph";
import { hungarian } from "./eval";
import { rng } from "./ui";
import { poisson } from "./random";
import { median } from "./stats";

// ---------------------------------------------------------------------------------------
// Filters

export type SpotFilter = "none" | "gauss" | "zmgauss" | "log" | "dog" | "tophat" | "atrous";
export const SPOT_FILTER_LABELS: Record<SpotFilter, string> = {
  none: "none", gauss: "Gaussian", zmgauss: "zero-mean Gaussian", log: "LoG", dog: "DoG", tophat: "top-hat + Gaussian", atrous: "à trous product",
};
export const DOG_RATIO = 1.6;

/** Radius of the zero-mean matched filter window (and of the top-hat disk). */
export const templateRadius = (sigma: number): number => Math.max(2, Math.ceil(3 * sigma));

/**
 * Linear spot filters (bright spots give positive responses). `gauss` and `none` subtract the
 * global median (a single background level), `zmgauss` is Σ (p_k − p̄) I_k over a
 * (2R+1)² window with p the normalised Gaussian (the matched filter with an unknown constant
 * background), `log` is −σ²∇²G, `dog` is G_σ − G_{1.6σ}, `tophat` is a white top-hat with a
 * disk of radius R, then the Gaussian, minus its median.
 */
export function spotFilter(img: Grey, kind: Exclude<SpotFilter, "atrous">, sigma: number, border: Border = "mirror"): Grey {
  const sub = (g: Grey): Grey => { const m = median(g.data); for (let i = 0; i < g.data.length; i++) g.data[i] -= m; return g; };
  switch (kind) {
    case "none": return sub(createGreyFrom(img));
    case "gauss": return sub(gaussianBlur(img, sigma, border));
    case "zmgauss": {
      const R = templateRadius(sigma), k = gaussianKernel(sigma, 0, R);
      const g = separable(img, k, k, border), b = boxFilter(img, R, border);
      for (let i = 0; i < g.data.length; i++) g.data[i] -= b.data[i];
      return g;
    }
    case "log": { const g = logFilter(img, sigma, true, border); for (let i = 0; i < g.data.length; i++) g.data[i] = -g.data[i]; return g; }
    case "dog": { const g = dogFilter(img, sigma, DOG_RATIO, border); for (let i = 0; i < g.data.length; i++) g.data[i] = -g.data[i]; return g; }
    case "tophat": {
      const t = whiteTopHat(img.data, img.width, img.height, seDisk(templateRadius(sigma) + 1));
      return sub(gaussianBlur({ width: img.width, height: img.height, data: t }, sigma, border));
    }
  }
}

function createGreyFrom(img: Grey): Grey { return { width: img.width, height: img.height, data: Float32Array.from(img.data) }; }

/** The 2D kernel of a linear spot filter (filter applied to a unit impulse), size n × n, centre n>>1. */
export function filterKernel(kind: "none" | "gauss" | "zmgauss" | "log" | "dog", sigma: number, n = 81): Grey {
  const d = createGrey(n, n), c = n >> 1;
  d.data[c * n + c] = 1;
  if (kind === "none") return d;
  if (kind === "gauss") return gaussianBlur(d, sigma, "zero");
  if (kind === "zmgauss") {
    const R = templateRadius(sigma), k = gaussianKernel(sigma, 0, R), g = separable(d, k, k, "zero");
    const inv = 1 / ((2 * R + 1) ** 2);
    for (let y = c - R; y <= c + R; y++) for (let x = c - R; x <= c + R; x++) g.data[y * n + x] -= inv;
    return g;
  }
  const g = kind === "log" ? logFilter(d, sigma, true, "zero") : dogFilter(d, sigma, DOG_RATIO, "zero");
  for (let i = 0; i < g.data.length; i++) g.data[i] = -g.data[i];
  return g;
}

/** √Σ w² of a kernel: the noise sd after filtering white noise of sd 1. */
export function filterResponseNoise(w: ArrayLike<number>): number {
  let s = 0; for (let i = 0; i < w.length; i++) s += w[i] * w[i];
  return Math.sqrt(s);
}

/** Σ_k w_k p_k: the filter's peak response to a unit-amplitude Gaussian spot of width s centred on a pixel. */
export function spotGain(w: Grey, s: number): number {
  const n = w.width, c = n >> 1;
  let g = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) g += w.data[y * n + x] * Math.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * s * s));
  return g;
}

/** Noise gain √Σw² of a linear spot filter (top-hat: the Gaussian's gain, an approximation). */
const gainCache = new Map<string, number>();
export function noiseGain(kind: SpotFilter, sigma: number): number {
  const k = kind === "tophat" ? "gauss" : kind === "atrous" ? "none" : kind;
  const key = k + ":" + sigma.toFixed(4);
  let v = gainCache.get(key);
  if (v === undefined) { v = filterResponseNoise(filterKernel(k, sigma).data); gainCache.set(key, v); }
  return v;
}

// ---------------------------------------------------------------------------------------
// À trous (undecimated, B3 spline)

export const B3 = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];

/** The B3 kernel with 2^(j−1) − 1 zeros ("holes") between its taps, for level j ≥ 1. */
export function b3Dilated(j: number): Float64Array {
  const step = 1 << (j - 1), k = new Float64Array(4 * step + 1);
  for (let i = 0; i < 5; i++) k[i * step] = B3[i];
  return k;
}

/** À trous decomposition: smooth images a_1..a_J and detail images w_j = a_{j−1} − a_j (a_0 = input). */
export function atrous(img: Grey, J: number, border: Border = "mirror"): { w: Grey[]; a: Grey[] } {
  const w: Grey[] = [], a: Grey[] = [];
  let prev = img;
  for (let j = 1; j <= J; j++) {
    const k = b3Dilated(j), cur = separable(prev, k, k, border), d = createGrey(img.width, img.height);
    for (let i = 0; i < d.data.length; i++) d.data[i] = prev.data[i] - cur.data[i];
    w.push(d); a.push(cur); prev = cur;
  }
  return { w, a };
}

/** Reconstruction a_J + Σ w_j (telescoping sum). */
export function atrousRecon(w: Grey[], last: Grey): Grey {
  const out = createGrey(last.width, last.height);
  for (let i = 0; i < out.data.length; i++) { let s = last.data[i]; for (const d of w) s += d.data[i]; out.data[i] = s; }
  return out;
}

/** σ of the noise in w_j for white noise of sd 1 (computed from the impulse response, cached). */
const atrousGainCache: number[] = [];
export function atrousNoiseGain(j: number): number {
  if (atrousGainCache[j] === undefined) {
    const n = 4 * (1 << j) + 9, d = createGrey(n, n), c = n >> 1;
    d.data[c * n + c] = 1;
    atrousGainCache[j] = filterResponseNoise(atrous(d, j, "zero").w[j - 1].data);
  }
  return atrousGainCache[j];
}

/**
 * Olivo-Marin multiscale product: P = Π_{j∈scales} w̃_j with w̃_j = w_j where w_j > k σ_j,
 * else 0 (σ_j = MAD estimate of w_j). Spots persist over scales, noise does not.
 */
export function atrousProduct(img: Grey, scales: number[], k: number, dec?: { w: Grey[]; a: Grey[] }): { product: Grey; sigmas: number[]; dec: { w: Grey[]; a: Grey[] } } {
  const J = Math.max(...scales), D = dec && dec.w.length >= J ? dec : atrous(img, J);
  const sigmas = D.w.map(g => madNoise(g.data));
  const P = createGrey(img.width, img.height);
  P.data.fill(1);
  for (const j of scales) {
    const g = D.w[j - 1], t = k * sigmas[j - 1];
    for (let i = 0; i < P.data.length; i++) P.data[i] *= g.data[i] > t ? g.data[i] : 0;
  }
  return { product: P, sigmas, dec: D };
}

// ---------------------------------------------------------------------------------------
// Noise estimates

/** MAD noise estimate: median |x − median(x)| / 0.6745. */
export function madNoise(x: ArrayLike<number>): number {
  const m = median(x), d = Float64Array.from(x, v => Math.abs(v - m));
  return median(d) / 0.6744897501960817;
}

/** Noise sd of an image estimated from its finest à trous level: MAD(w_1) / gain(w_1). */
export function estimateNoise(img: Grey): number {
  const w1 = atrous(img, 1).w[0];
  return madNoise(w1.data) / atrousNoiseGain(1);
}

/** Anscombe variance-stabilising transform 2√(x + 3/8): Poisson counts → noise sd ≈ 1. */
export const anscombe = (x: number): number => 2 * Math.sqrt(Math.max(0, x) + 0.375);

/** Local noise sd map √(gain · background + read²), background from a wide Gaussian blur. */
export function localNoiseMap(img: Grey, gain = 1, readNoise = 0, bgSigma = 8): Grey {
  const bg = gaussianBlur(img, bgSigma, "mirror");
  for (let i = 0; i < bg.data.length; i++) bg.data[i] = Math.sqrt(Math.max(1e-6, gain * Math.max(0, bg.data[i]) + readNoise * readNoise));
  return bg;
}

/**
 * Euler-characteristic heuristic: expected number of maxima above u (in sd units) of white
 * noise smoothed with a Gaussian of width σ_f over A pixels:
 * A · 1/(2σ_f²) · u e^{−u²/2} / (2π)^{3/2}.
 */
export function eulerExpectedMaxima(area: number, sigmaF: number, u: number): number {
  return area / (2 * sigmaF * sigmaF) * u * Math.exp(-u * u / 2) / Math.pow(2 * Math.PI, 1.5);
}

/** Naive estimate: (number of maxima) · P(Z > u). */
export const upperTail = (u: number): number => 0.5 * erfcApprox(u / Math.SQRT2);
function erfcApprox(x: number): number {
  // Numerical Recipes erfc (rel. error < 1.2e-7)
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** White Gaussian noise (sd 1), size w × h, seeded. */
export function whiteNoise(w: number, h: number, seed: number): Grey {
  const r = rng(seed), g = createGrey(w, h);
  for (let i = 0; i < g.data.length; i++) g.data[i] = r.norm();
  return g;
}

/**
 * Values (in units of the filtered noise sd) of all local maxima (3 × 3 window, r = 1)
 * of seeded white noise after a Gaussian of width σ_f. Border of 4σ_f excluded.
 */
export function noiseMaximaValues(size: number, sigmaF: number, seed: number, r = 1, kind: "gauss" | "log" = "gauss"): { values: Float64Array; area: number } {
  const n = whiteNoise(size, size, seed);
  const f = kind === "gauss" ? gaussianBlur(n, sigmaF, "wrap") : spotFilter(n, "log", sigmaF, "wrap");
  const g = noiseGain(kind, sigmaF);
  for (let i = 0; i < f.data.length; i++) f.data[i] /= g;
  const m = localMaxima2D(f, -Infinity, r, r);
  return { values: Float64Array.from(m, p => p.value), area: (size - 2 * r) ** 2 };
}

/** Mean number of noise maxima above each k (sd units) per image, over `seeds` seeded images. */
export function noiseMaximaCount(size: number, sigmaF: number, ks: number[], seeds: number[], kind: "gauss" | "log" = "gauss"): { counts: number[]; maxima: number } {
  const counts = ks.map(() => 0);
  let maxima = 0;
  for (const s of seeds) {
    const { values } = noiseMaximaValues(size, sigmaF, s, 1, kind);
    maxima += values.length;
    for (const v of values) ks.forEach((k, i) => { if (v > k) counts[i]++; });
  }
  return { counts: counts.map(c => c / seeds.length), maxima: maxima / seeds.length };
}

// ---------------------------------------------------------------------------------------
// Detection

export interface Detection { x: number; y: number; value: number; snr: number }

export interface DetectOptions {
  filter: SpotFilter;
  sigma: number;           // filter σ (px)
  radius: number;          // maxima window radius r: (2r+1)² window
  k: number;               // threshold in noise sd of the filtered image (à trous: per scale)
  noise?: "mad" | number;  // noise sd of the input image (default MAD estimate)
  scales?: number[];       // à trous scales (default [2, 3])
}

/** Filter, find local maxima, threshold at k noise sd of the response. */
export function detectSpots(img: Grey, o: DetectOptions): { response: Grey; dets: Detection[]; noise: number; respNoise: number; candidates: number } {
  const noise = typeof o.noise === "number" ? o.noise : estimateNoise(img);
  if (o.filter === "atrous") {
    const { product } = atrousProduct(img, o.scales ?? [2, 3], o.k);
    const m = localMaxima2D(product, 0, o.radius, 2);
    return { response: product, dets: m.map(p => ({ ...p, snr: NaN })), noise, respNoise: NaN, candidates: m.length };
  }
  const response = spotFilter(img, o.filter, o.sigma);
  const respNoise = noise * noiseGain(o.filter, o.sigma);
  const all = localMaxima2D(response, -Infinity, o.radius, 2);
  const t = o.k * respNoise;
  const dets = all.filter(p => p.value > t).map(p => ({ ...p, snr: p.value / respNoise }));
  return { response, dets, noise, respNoise, candidates: all.length };
}

// ---------------------------------------------------------------------------------------
// Synthetic spot images (crowding allowed)

export interface SpotTruth { x: number; y: number; amplitude: number }

export interface SpotImageOptions {
  size?: number;            // square image, default 256
  count?: number;
  sigma?: number;           // PSF σ (px), default 1.5
  amplitude?: [number, number];
  background?: number;      // mean level
  slope?: number;           // left → right increase
  bump?: number;            // broad bright patch
  patch?: number;           // a bright square background patch (falsespots: Poisson demo)
  noise?: number;           // Gaussian noise sd
  poisson?: boolean;        // Poisson noise (image in photons) instead
  minDistance?: number;     // default 0: spots may overlap
  margin?: number;          // keep spots this far from the border (default 4)
  seed?: number;
  list?: SpotTruth[];       // fixed positions instead of random
}

const erf = (x: number): number => 1 - erfcApprox(x);
/** Fraction of a unit Gaussian (centre c, width s) falling on pixel i (covering [i−½, i+½]). */
export function pixelFraction(i: number, c: number, s: number): number {
  const k = 1 / (Math.SQRT2 * s);
  return 0.5 * (erf((i + 0.5 - c) * k) - erf((i - 0.5 - c) * k));
}

/** Pixel-integrated Gaussian spots on a smooth background with Gaussian or Poisson noise, truth known. */
export function spotImage(o: SpotImageOptions = {}): { image: Grey; clean: Grey; background: Grey; truth: SpotTruth[] } {
  const n = o.size ?? 256, s = o.sigma ?? 1.5, r = rng(o.seed ?? 1), [a0, a1] = o.amplitude ?? [20, 120];
  const margin = o.margin ?? 4, minD = o.minDistance ?? 0;
  let truth: SpotTruth[] = o.list ? o.list.map(t => ({ ...t })) : [];
  if (!o.list) for (let tries = 0; truth.length < (o.count ?? 60) && tries < 20000; tries++) {
    const x = margin + r.unif() * (n - 1 - 2 * margin), y = margin + r.unif() * (n - 1 - 2 * margin);
    const amp = a0 + (a1 - a0) * r.unif();
    if (minD > 0 && truth.some(t => Math.hypot(t.x - x, t.y - y) < minD)) continue;
    truth.push({ x, y, amplitude: amp });
  }
  const bg = createGrey(n, n), bl = o.background ?? 40, slope = o.slope ?? 0, bump = o.bump ?? 0;
  const bx = 0.62 * n, by = 0.4 * n, bs = 0.22 * n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let v = bl + slope * x / (n - 1) + bump * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * bs * bs));
    if (o.patch && x >= n / 2 && y < n / 2) v += o.patch;
    bg.data[y * n + x] = v;
  }
  const clean = createGrey(n, n);
  clean.data.set(bg.data);
  // peak amplitude A of the point-sampled Gaussian = total signal A·2πs²; pixel-integrated
  const R = Math.ceil(4 * s), tot = 2 * Math.PI * s * s;
  const fx = new Float64Array(2 * R + 2), fy = new Float64Array(2 * R + 2);
  for (const t of truth) {
    const x0 = Math.round(t.x) - R, y0 = Math.round(t.y) - R;
    for (let i = 0; i <= 2 * R; i++) { fx[i] = pixelFraction(x0 + i, t.x, s); fy[i] = pixelFraction(y0 + i, t.y, s); }
    for (let j = 0; j <= 2 * R; j++) {
      const yy = y0 + j; if (yy < 0 || yy >= n) continue;
      for (let i = 0; i <= 2 * R; i++) {
        const xx = x0 + i; if (xx < 0 || xx >= n) continue;
        clean.data[yy * n + xx] += t.amplitude * tot * fx[i] * fy[j];
      }
    }
  }
  const image = createGrey(n, n), rn = rng((o.seed ?? 1) * 7919 + 17);
  for (let i = 0; i < n * n; i++)
    image.data[i] = o.poisson ? poisson(rn, Math.max(0, clean.data[i])) : clean.data[i] + (o.noise ?? 10) * rn.norm();
  return { image, clean, background: bg, truth };
}

// ---------------------------------------------------------------------------------------
// 1D helpers (peaks1d)

/** Gaussian (σ) of a 1D signal, mirror border. */
export const gauss1d = (f: ArrayLike<number>, sigma: number): Float64Array => correlate1d(f, gaussianKernel(sigma, 0), "mirror");

/** 1D spot filters, as spotFilter: gauss/none minus the median; zmgauss; log (−σ² g''); dog; atrous (product of levels 2..3, thresholded at k). */
export function filter1d(f: ArrayLike<number>, kind: SpotFilter, sigma: number): Float64Array {
  const sub = (g: Float64Array) => { const m = median(g); for (let i = 0; i < g.length; i++) g[i] -= m; return g; };
  switch (kind) {
    case "none": case "tophat": return sub(Float64Array.from(f));
    case "gauss": return sub(gauss1d(f, sigma));
    case "zmgauss": {
      const R = templateRadius(sigma), k = gaussianKernel(sigma, 0, R), inv = 1 / (2 * R + 1);
      return correlate1d(f, k.map(v => v - inv), "mirror");
    }
    case "log": { const k = gaussianKernel(sigma, 2); return correlate1d(f, k.map(v => -v * sigma * sigma), "mirror"); }
    case "dog": { const a = gauss1d(f, sigma), b = gauss1d(f, DOG_RATIO * sigma); return a.map((v, i) => v - b[i]); }
    case "atrous": return Float64Array.from(f);
  }
}

/** 1D kernel of a linear 1D spot filter (for drawing to scale), offsets −R..R. */
export function kernel1d(kind: SpotFilter, sigma: number): Float64Array {
  const n = 2 * Math.ceil(4 * DOG_RATIO * Math.max(sigma, 0.5)) + 1, d = new Float64Array(n);
  d[n >> 1] = 1;
  if (kind === "none" || kind === "tophat" || kind === "atrous") return d;
  if (kind === "gauss") return gauss1d(d, sigma);
  return filter1d(d, kind, sigma);
}

/** 1D à trous: detail levels w_1..w_J and the last smooth a_J. */
export function atrous1d(f: ArrayLike<number>, J: number): { w: Float64Array[]; a: Float64Array[] } {
  const w: Float64Array[] = [], a: Float64Array[] = [];
  let prev: Float64Array = Float64Array.from(f);
  for (let j = 1; j <= J; j++) {
    const cur = correlate1d(prev, b3Dilated(j), "mirror");
    w.push(prev.map((v, i) => v - cur[i])); a.push(cur); prev = cur;
  }
  return { w, a };
}

/** 1D multiscale product of levels `scales`, each kept where w_j > k·σ_j (MAD). */
export function atrousProduct1d(f: ArrayLike<number>, scales: number[], k: number): Float64Array {
  const { w } = atrous1d(f, Math.max(...scales)), P = new Float64Array(f.length).fill(1);
  for (const j of scales) {
    const g = w[j - 1], t = k * madNoise(g);
    for (let i = 0; i < P.length; i++) P[i] *= g[i] > t ? g[i] : 0;
  }
  return P;
}

/**
 * 1D local maxima: i is a candidate if f[i] ≥ every value within ±r, and among equal values
 * (a plateau) only the leftmost survives, if the plateau ends lower on its right (a step of a
 * staircase is not a maximum). Values must exceed `threshold`.
 */
export function localMaxima1d(f: ArrayLike<number>, r: number, threshold = -Infinity): number[] {
  const out: number[] = [];
  for (let i = 0; i < f.length; i++) {
    const v = f[i];
    if (!(v > threshold)) continue;
    let ok = true;
    for (let d = -r; d <= r && ok; d++) {
      if (!d) continue;
      const j = i + d; if (j < 0 || j >= f.length) continue;
      if (f[j] > v || (d < 0 && f[j] === v)) ok = false;
    }
    if (ok && i + 1 < f.length && f[i + 1] === v) {   // a plateau: a maximum only if it ends lower
      let j = i + 1; while (j < f.length && f[j] === v) j++;
      if (j < f.length && f[j] > v) ok = false;
    }
    if (ok) out.push(i);
  }
  return out;
}

/**
 * Prominence of each strict-or-plateau local maximum (r = 1): its height above the higher of
 * the two lowest points ("saddles") separating it from higher ground on either side (the
 * signal's ends count as −∞ … i.e. the global maximum's prominence is its height above the
 * lowest saddle on its sides, or above the minimum of the signal). h-maxima keep peaks with prominence ≥ h.
 */
export function prominences(f: ArrayLike<number>): { i: number; prominence: number; saddle: number }[] {
  const peaks = localMaxima1d(f, 1), n = f.length, out: { i: number; prominence: number; saddle: number }[] = [];
  let gmin = Infinity; for (let i = 0; i < n; i++) gmin = Math.min(gmin, f[i]);
  for (const p of peaks) {
    const v = f[p];
    let lmin = v, rmin = v, lHigher = false, rHigher = false;
    for (let j = p - 1; j >= 0; j--) { if (f[j] > v) { lHigher = true; break; } lmin = Math.min(lmin, f[j]); }
    for (let j = p + 1; j < n; j++) { if (f[j] > v) { rHigher = true; break; } rmin = Math.min(rmin, f[j]); }
    let saddle: number;
    if (lHigher && rHigher) saddle = Math.max(lmin, rmin);
    else if (lHigher) saddle = lmin;
    else if (rHigher) saddle = rmin;
    else saddle = gmin;
    out.push({ i: p, prominence: v - saddle, saddle });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Matching points within a gate (one-to-one)

export type Pt = [number, number];
export interface PointMatch { pairs: { t: number; d: number; dist: number }[]; tp: number; fp: number; fn: number; rmse: number; bias: Pt; precision: number; recall: number; f1: number; jaccard: number }

function finishMatch(truth: Pt[], det: Pt[], pairs: { t: number; d: number; dist: number }[]): PointMatch {
  const tp = pairs.length, fp = det.length - tp, fn = truth.length - tp;
  let se = 0, bx = 0, by = 0;
  for (const p of pairs) { se += p.dist * p.dist; bx += det[p.d][0] - truth[p.t][0]; by += det[p.d][1] - truth[p.t][1]; }
  const precision = det.length ? tp / det.length : 1, recall = truth.length ? tp / truth.length : 1;
  return {
    pairs, tp, fp, fn, rmse: tp ? Math.sqrt(se / tp) : NaN, bias: tp ? [bx / tp, by / tp] : [NaN, NaN],
    precision, recall, f1: tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 1, jaccard: tp + fp + fn ? tp / (tp + fp + fn) : 1,
  };
}

/** Candidate pairs within the gate (grid-bucketed), sorted by distance. */
function gatePairs(truth: Pt[], det: Pt[], gate: number): { t: number; d: number; dist: number }[] {
  const cell = Math.max(gate, 1e-9), grid = new Map<string, number[]>();
  det.forEach((p, j) => { const k = Math.floor(p[0] / cell) + "," + Math.floor(p[1] / cell); (grid.get(k) ?? grid.set(k, []).get(k)!).push(j); });
  const out: { t: number; d: number; dist: number }[] = [];
  truth.forEach((p, i) => {
    const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) for (const j of grid.get((cx + dx) + "," + (cy + dy)) ?? []) {
      const dist = Math.hypot(det[j][0] - p[0], det[j][1] - p[1]);
      if (dist <= gate) out.push({ t: i, d: j, dist });
    }
  });
  return out.sort((a, b) => a.dist - b.dist || a.t - b.t || a.d - b.d);
}

/**
 * One-to-one matching of detections to truth within a gate radius. Greedy: closest pair first.
 * Optimal: the most matches, then the smallest total distance (Hungarian, per connected
 * component of the gate graph).
 */
export function matchPoints(truth: Pt[], det: Pt[], gate: number, method: "greedy" | "optimal" = "optimal"): PointMatch {
  const cand = gatePairs(truth, det, gate);
  if (method === "greedy") {
    const ut = new Uint8Array(truth.length), ud = new Uint8Array(det.length), pairs: typeof cand = [];
    for (const c of cand) if (!ut[c.t] && !ud[c.d]) { ut[c.t] = 1; ud[c.d] = 1; pairs.push(c); }
    return finishMatch(truth, det, pairs);
  }
  // components (union–find over truth indices 0..T−1 and detections T..T+D−1)
  const T = truth.length, par = Int32Array.from({ length: T + det.length }, (_, i) => i);
  const find = (i: number): number => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
  for (const c of cand) { const a = find(c.t), b = find(T + c.d); if (a !== b) par[a] = b; }
  const comps = new Map<number, typeof cand>();
  for (const c of cand) { const r = find(c.t); (comps.get(r) ?? comps.set(r, []).get(r)!).push(c); }
  const pairs: typeof cand = [];
  for (const cs of comps.values()) {
    const ts = [...new Set(cs.map(c => c.t))], ds = [...new Set(cs.map(c => c.d))];
    if (cs.length === 1) { pairs.push(cs[0]); continue; }
    const n = Math.max(ts.length, ds.length), eps = 1 / (n + 1), wgt = new Float64Array(n * n);
    const ti = new Map(ts.map((t, i) => [t, i])), di = new Map(ds.map((d, i) => [d, i]));
    for (const c of cs) wgt[ti.get(c.t)! * n + di.get(c.d)!] = 1 + eps * (gate - c.dist) / (gate || 1);
    let wmax = 0; for (const v of wgt) wmax = Math.max(wmax, v);
    const col = hungarian(Float64Array.from(wgt, v => wmax - v), n);
    for (let i = 0; i < ts.length; i++) {
      const j = col[i];
      if (j >= 0 && j < ds.length && wgt[i * n + j] > 0) {
        const t = ts[i], d = ds[j];
        pairs.push({ t, d, dist: Math.hypot(det[d][0] - truth[t][0], det[d][1] - truth[t][1]) });
      }
    }
  }
  return finishMatch(truth, det, pairs.sort((a, b) => a.t - b.t));
}

/** Matching scores for a range of gate radii. */
export function f1VsGate(truth: Pt[], det: Pt[], gates: number[], method: "greedy" | "optimal" = "optimal"): PointMatch[] {
  return gates.map(g => matchPoints(truth, det, g, method));
}

export { fromArray };
