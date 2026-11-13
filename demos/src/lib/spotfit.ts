// Sub-pixel spot localisation, owned by the point-detection day (no DOM): pixel-integrated
// Gaussian (and Airy) spot models, the centroid, three-point parabola and log-parabola
// (Gaussian) estimators, least-squares and Poisson maximum-likelihood fits of one or several
// spots with a known PSF width (Levenberg–Marquardt / Fisher scoring, with an iteration
// record), the Cramér–Rao / Thompson / Mortensen precision formulas, and the model-selection
// rule for "one spot or two". Coordinates: pixel (i, j) covers [i − ½, i + ½] × [j − ½, j + ½].
import { solve } from "./linalg";
import { besselJ1 } from "./psf";
import { parabolaOffset } from "./registration";
import { pixelFraction } from "./spotdetect";
import { rng } from "./ui";
import { poisson } from "./random";

const SQ2PI = Math.sqrt(2 * Math.PI);
const phi = (z: number): number => Math.exp(-0.5 * z * z) / SQ2PI;

// ---------------------------------------------------------------------------------------
// Models

/** Expected counts of one pixel-integrated Gaussian spot (N photons, width s) on background B, w × h pixels. */
export function pixelGaussian(x0: number, y0: number, s: number, N: number, B: number, w: number, h = w): Float64Array {
  const fx = Float64Array.from({ length: w }, (_, i) => pixelFraction(i, x0, s));
  const fy = Float64Array.from({ length: h }, (_, j) => pixelFraction(j, y0, s));
  const out = new Float64Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) out[j * w + i] = B + N * fx[i] * fy[j];
  return out;
}

/**
 * Pixel-integrated Airy spot (supersampled 7 × 7 per pixel) whose Gaussian approximation has
 * width s: (2J1(v)/v)² with v = 1.3195 r/s (σ = 0.21 λ/NA); normalised over the plane.
 */
export function pixelAiry(x0: number, y0: number, s: number, N: number, B: number, w: number, h = w): Float64Array {
  const k = 2 * Math.PI * 0.21 / s, norm = k * k / (4 * Math.PI), ss = 7, out = new Float64Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    let acc = 0;
    for (let b = 0; b < ss; b++) for (let a = 0; a < ss; a++) {
      const x = i - 0.5 + (a + 0.5) / ss - x0, y = j - 0.5 + (b + 0.5) / ss - y0, v = k * Math.hypot(x, y);
      const f = v < 1e-8 ? 1 : (2 * besselJ1(v) / v) ** 2;
      acc += f;
    }
    out[j * w + i] = B + N * norm * acc / (ss * ss);
  }
  return out;
}

/** Poisson sample of expected counts, seeded. */
export function poissonImage(mu: ArrayLike<number>, seed: number): Float64Array {
  const r = rng(seed);
  return Float64Array.from(mu, m => poisson(r, Math.max(0, m)));
}

// ---------------------------------------------------------------------------------------
// Simple estimators (positions in pixel coordinates)

/** The brightest pixel. */
export function peakPixel(img: ArrayLike<number>, w: number, h = w): [number, number] {
  let b = 0; for (let i = 1; i < w * h; i++) if (img[i] > img[b]) b = i;
  return [b % w, Math.floor(b / w)];
}

/**
 * Centroid in a (2m+1)² window centred on pixel (cx, cy), after subtracting `bg` from every
 * pixel (no clipping).
 */
export function centroid(img: ArrayLike<number>, w: number, h: number, cx: number, cy: number, m: number, bg = 0): [number, number] {
  let s = 0, sx = 0, sy = 0;
  for (let y = Math.max(0, cy - m); y <= Math.min(h - 1, cy + m); y++) for (let x = Math.max(0, cx - m); x <= Math.min(w - 1, cx + m); x++) {
    const v = img[y * w + x] - bg;
    s += v; sx += v * x; sy += v * y;
  }
  return s !== 0 ? [sx / s, sy / s] : [cx, cy];
}

/** Three-point log-parabola (exact for a point-sampled Gaussian): offset from the middle sample. */
export function gauss3Point(cm: number, c0: number, cp: number): number {
  const e = 1e-9, a = Math.log(Math.max(e, cm)), b = Math.log(Math.max(e, c0)), c = Math.log(Math.max(e, cp));
  const d = a - 2 * b + c;
  return d < 0 ? Math.max(-1, Math.min(1, (a - c) / (2 * d))) : 0;
}

/** Three-point parabola (on raw values), as registration.parabolaOffset. */
export const parabola3Point = parabolaOffset;

/** Apply a three-point rule along x and y through pixel (cx, cy), after subtracting bg. */
export function threePoint(img: ArrayLike<number>, w: number, h: number, cx: number, cy: number, rule: (a: number, b: number, c: number) => number, bg = 0): [number, number] {
  const at = (x: number, y: number) => img[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))] - bg;
  return [cx + rule(at(cx - 1, cy), at(cx, cy), at(cx + 1, cy)), cy + rule(at(cx, cy - 1), at(cx, cy), at(cx, cy + 1))];
}

// ---------------------------------------------------------------------------------------
// Least squares / maximum likelihood fits of n spots with known width

export interface SpotParams { x: number; y: number; N: number }
export interface FitResult { spots: SpotParams[]; B: number; rss: number; nll: number; iterations: { spots: SpotParams[]; B: number; rss: number }[]; model: Float64Array; converged: boolean }

/** Model and Jacobian (columns: per spot x, y, N; then B) of n pixel-integrated Gaussians. */
function modelJac(p: Float64Array, n: number, s: number, w: number, h: number, wantJ: boolean): { mu: Float64Array; J: Float64Array | null } {
  const P = 3 * n + 1, mu = new Float64Array(w * h).fill(p[3 * n]), J = wantJ ? new Float64Array(w * h * P) : null;
  for (let k = 0; k < n; k++) {
    const x0 = p[3 * k], y0 = p[3 * k + 1], N = p[3 * k + 2];
    const fx = new Float64Array(w), fy = new Float64Array(h), dx = new Float64Array(w), dy = new Float64Array(h);
    for (let i = 0; i < w; i++) { fx[i] = pixelFraction(i, x0, s); dx[i] = (phi((i - 0.5 - x0) / s) - phi((i + 0.5 - x0) / s)) / s; }
    for (let j = 0; j < h; j++) { fy[j] = pixelFraction(j, y0, s); dy[j] = (phi((j - 0.5 - y0) / s) - phi((j + 0.5 - y0) / s)) / s; }
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const q = j * w + i;
      mu[q] += N * fx[i] * fy[j];
      if (J) { J[q * P + 3 * k] = N * dx[i] * fy[j]; J[q * P + 3 * k + 1] = N * fx[i] * dy[j]; J[q * P + 3 * k + 2] = fx[i] * fy[j]; }
    }
  }
  if (J) for (let q = 0; q < w * h; q++) J[q * P + 3 * n] = 1;
  return { mu, J };
}

function objective(mu: Float64Array, img: ArrayLike<number>, mle: boolean): number {
  let s = 0;
  for (let q = 0; q < mu.length; q++) {
    if (mle) { const m = Math.max(1e-9, mu[q]); s += m - img[q] * Math.log(m); }
    else s += (img[q] - mu[q]) ** 2;
  }
  return s;
}

/**
 * Fit n spots (known width s) and a constant background to an image by least squares
 * (Levenberg–Marquardt on Σ(I − μ)²) or by Poisson maximum likelihood (Fisher scoring with
 * damping on Σ(μ − I ln μ)). `iterations` records every accepted step (for the step-through).
 */
export function fitSpots(img: ArrayLike<number>, w: number, h: number, s: number, init: SpotParams[], B0: number,
  o: { mle?: boolean; maxIter?: number; tol?: number } = {}): FitResult {
  const n = init.length, P = 3 * n + 1, mle = !!o.mle, maxIter = o.maxIter ?? 30;
  let p = new Float64Array(P);
  init.forEach((sp, k) => { p[3 * k] = sp.x; p[3 * k + 1] = sp.y; p[3 * k + 2] = Math.max(1, sp.N); });
  p[3 * n] = B0;
  const unpack = (q: Float64Array) => ({ spots: init.map((_, k) => ({ x: q[3 * k], y: q[3 * k + 1], N: q[3 * k + 2] })), B: q[3 * n] });
  let { mu, J } = modelJac(p, n, s, w, h, true), f = objective(mu, img, mle), lam = 1e-3, converged = false;
  const iterations = [{ ...unpack(p), rss: rssOf(mu, img) }];
  for (let it = 0; it < maxIter; it++) {
    const A = new Float64Array(P * P), g = new Float64Array(P);
    for (let q = 0; q < w * h; q++) {
      const wt = mle ? 1 / Math.max(1e-6, mu[q]) : 1, r = img[q] - mu[q];
      for (let a = 0; a < P; a++) {
        const ja = J![q * P + a];
        if (ja === 0) continue;
        g[a] += wt * ja * r;
        for (let b = a; b < P; b++) A[a * P + b] += wt * ja * J![q * P + b];
      }
    }
    for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) A[a * P + b] = A[b * P + a];
    let accepted = false;
    for (let tries = 0; tries < 12; tries++) {
      const D = Float64Array.from(A);
      for (let a = 0; a < P; a++) D[a * P + a] *= 1 + lam;
      const step = solve(D, g, P);
      if (!step) { lam *= 10; continue; }
      const pn = p.map((v, i) => v + (i % 3 !== 2 && i !== 3 * n ? Math.max(-1, Math.min(1, step[i])) : step[i]));
      for (let k = 0; k < n; k++) {
        pn[3 * k] = Math.max(-0.5, Math.min(w - 0.5, pn[3 * k])); pn[3 * k + 1] = Math.max(-0.5, Math.min(h - 0.5, pn[3 * k + 1]));
        pn[3 * k + 2] = Math.max(1e-3, pn[3 * k + 2]);
      }
      if (mle) pn[3 * n] = Math.max(1e-3, pn[3 * n]);
      const mn = modelJac(pn, n, s, w, h, true), fn = objective(mn.mu, img, mle);
      if (fn <= f) {
        const small = Math.max(...Array.from(step).map((v, i) => (i % 3 === 2 || i === 3 * n) ? 0 : Math.abs(v))) < (o.tol ?? 1e-5);
        p = pn; mu = mn.mu; J = mn.J; f = fn; lam = Math.max(1e-7, lam / 10); accepted = true;
        iterations.push({ ...unpack(p), rss: rssOf(mu, img) });
        if (small) converged = true;
        break;
      }
      lam *= 10;
    }
    if (!accepted || converged) { converged = true; break; }
  }
  const u = unpack(p);
  return { ...u, rss: rssOf(mu, img), nll: objective(mu, img, true), iterations, model: mu, converged };
}

function rssOf(mu: Float64Array, img: ArrayLike<number>): number { let s = 0; for (let q = 0; q < mu.length; q++) s += (img[q] - mu[q]) ** 2; return s; }

/** One Gauss–Newton step for x0 alone in 1D (point-sampled Gaussian A·g(x − x0) + B with A, B, s known). */
export function gaussNewtonStep1d(xs: number[], ys: number[], x0: number, A: number, s: number, B: number): { dx: number; J: number[]; r: number[] } {
  const J: number[] = [], r: number[] = [];
  let JtJ = 0, Jtr = 0;
  xs.forEach((x, i) => {
    const g = Math.exp(-((x - x0) ** 2) / (2 * s * s)), m = B + A * g, d = A * g * (x - x0) / (s * s);
    J.push(d); r.push(ys[i] - m); JtJ += d * d; Jtr += d * (ys[i] - m);
  });
  return { dx: Jtr / JtJ, J, r };
}

/** Linear least squares for amplitude A and background B with known shape values g_k: minimise Σ(I_k − B − A g_k)². */
export function amplitudeBackgroundLSQ(I: ArrayLike<number>, g: ArrayLike<number>): { A: number; B: number } {
  let n = 0, sg = 0, sgg = 0, sI = 0, sgI = 0;
  for (let k = 0; k < I.length; k++) { n++; sg += g[k]; sgg += g[k] * g[k]; sI += I[k]; sgI += g[k] * I[k]; }
  const det = n * sgg - sg * sg, A = (n * sgI - sg * sI) / det, B = (sI - A * sg) / n;
  return { A, B };
}

// ---------------------------------------------------------------------------------------
// Precision

/** Cramér–Rao bound (variance per axis) for N photon positions from a Gaussian of width s: s²/N. */
export const crlbGaussian = (s: number, N: number): number => s * s / N;

/** Thompson, Larson & Webb (2002): variance ≈ (s² + a²/12)/N + 8π s⁴ b/(a² N²). */
export const thompsonVar = (s: number, a: number, b: number, N: number): number =>
  (s * s + a * a / 12) / N + 8 * Math.PI * s ** 4 * b / (a * a * N * N);

/** Mortensen et al. (2010), least squares: (s_a²/N)(16/9 + 8π s_a² b/(N a²)), s_a² = s² + a²/12. */
export function mortensenLSQ(s: number, a: number, b: number, N: number): number {
  const sa2 = s * s + a * a / 12;
  return sa2 / N * (16 / 9 + 8 * Math.PI * sa2 * b / (N * a * a));
}

/** Mortensen et al. (2010), maximum likelihood: (s_a²/N)/(1 + ∫₀¹ ln t/(1 + t/τ) dt), τ = 2π s_a² b/(N a²). */
export function mortensenMLE(s: number, a: number, b: number, N: number): number {
  const sa2 = s * s + a * a / 12, tau = 2 * Math.PI * sa2 * b / (N * a * a);
  if (tau <= 0) return sa2 / N;
  // midpoint rule in u with t = u² (the integrand has a log singularity at 0)
  let I = 0; const M = 4000;
  for (let i = 0; i < M; i++) { const u = (i + 0.5) / M, t = u * u; I += Math.log(t) / (1 + t / tau) * 2 * u / M; }
  return sa2 / N / (1 + I);
}

// ---------------------------------------------------------------------------------------
// One spot or two?

/** Penalised comparison: keep n + 1 spots if RSS_n − RSS_{n+1} > 3 c σ_n² (c ≈ 2: AIC-like; ln(#pixels): BIC-like). */
export function selectSpotCount(rss1: number, rss2: number, sigmaN: number, c = 2): { keepTwo: boolean; gain: number; penalty: number } {
  const gain = rss1 - rss2, penalty = 3 * c * sigmaN * sigmaN;
  return { keepTwo: gain > penalty, gain, penalty };
}

/** Number of maxima of the sum of two equal Gaussians of width s at ±d/2 (1D, dense sampling). */
export function twoGaussianMaxima(d: number, s: number): number {
  const xs = 2000, lo = -d / 2 - 4 * s, hi = d / 2 + 4 * s;
  let count = 0, prev = NaN, prevUp = false;
  for (let i = 0; i <= xs; i++) {
    const x = lo + (hi - lo) * i / xs, v = Math.exp(-((x - d / 2) ** 2) / (2 * s * s)) + Math.exp(-((x + d / 2) ** 2) / (2 * s * s));
    if (!Number.isNaN(prev)) { const up = v > prev; if (prevUp && !up) count++; prevUp = up; }
    prev = v;
  }
  return count;
}
