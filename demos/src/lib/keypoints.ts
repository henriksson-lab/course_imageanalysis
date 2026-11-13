// Keypoints across scale and rotation (no DOM). Owner: the keypoints day (keypoints/).
//
// - dogPyramid: octaves of Gaussian blurs (S steps per octave, k = 2^(1/S), S + 3 images per
//   octave, incremental blur) and their differences (S + 2 DoGs per octave);
// - dogCandidates (26-neighbour extrema), refineExtremum (3D quadratic fit, moving to the
//   neighbour when an offset exceeds 0.5), contrastTest, edgeRatioTest (tr²/det of the 2 × 2
//   spatial Hessian), detectDoG with a stage record for the step-through demo;
// - detectHarris (one σ), detectHarrisMultiScale (normalised Harris + Laplacian scale
//   selection), detectFAST (a ring of 16, n contiguous);
// - orientationHistogram / assignOrientations (36 bins, a second keypoint above 80 %);
// - repeatability against a known transform, keepStrongest.
//
// Coordinates: x right, y down, pixel centres at integers, in the ORIGINAL image's pixels.
// σ in original pixels. θ is the dominant gradient direction counter-clockwise on the screen
// (as in lib/edges.ts screenAngle): the direction vector in image coordinates is
// (cos θ, −sin θ).

import { Grey, createGrey, warp, combine, gaussianKernel } from "./imageops";
import { solve, eigSym2 } from "./linalg";
import { parabolaOffset } from "./registration";

export interface Keypoint {
  x: number; y: number;
  /** Scale in original pixels. */
  sigma: number;
  /** Orientation, radians, counter-clockwise on the screen. */
  theta: number;
  /** Detector strength (|D| for DoG, the corner response otherwise). */
  response: number;
  /** Octave and (continuous) level in the DoG pyramid; −1 when not from the pyramid. */
  octave: number;
  level: number;
}

// ---------------------------------------------------------------------------------------
// A fast separable Gaussian (replicate border); same kernels as imageops.gaussianBlur

function pass(src: Float32Array, dst: Float32Array, w: number, h: number, k: Float64Array, horizontal: boolean): void {
  const r = k.length >> 1, n = k.length;
  if (horizontal) {
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let x = 0; x < w; x++) {
        let s = 0;
        if (x >= r && x + r < w) { const b = o + x - r; for (let i = 0; i < n; i++) s += k[i] * src[b + i]; }
        else for (let i = 0; i < n; i++) { let xx = x - r + i; xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx; s += k[i] * src[o + xx]; }
        dst[o + x] = s;
      }
    }
  } else {
    dst.fill(0);
    for (let j = 0; j < n; j++) {
      const kj = k[j];
      for (let y = 0; y < h; y++) {
        let yy = y - r + j; yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy;
        const so = yy * w, d0 = y * w;
        for (let x = 0; x < w; x++) dst[d0 + x] += kj * src[so + x];
      }
    }
  }
}

/** Gaussian blur (σ ≤ 0: copy), separable, replicate border; derivative orders as imageops.gaussianKernel. */
export function fastGauss(img: Grey, sigma: number, ox: 0 | 1 | 2 = 0, oy: 0 | 1 | 2 = 0): Grey {
  const w = img.width, h = img.height;
  if (!(sigma > 0)) return { width: w, height: h, data: Float32Array.from(img.data) };
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  pass(img.data, tmp, w, h, gaussianKernel(sigma, ox), true);
  pass(tmp, out, w, h, gaussianKernel(sigma, oy), false);
  return { width: w, height: h, data: out };
}

// ---------------------------------------------------------------------------------------
// The octave pyramid

export interface Octave {
  /** S + 3 blurred images. */
  gauss: Grey[];
  /** S + 2 differences gauss[i + 1] − gauss[i]. */
  dog: Grey[];
  /** Original pixels per pixel of this octave (2^o, halved when the first octave is upsampled). */
  step: number;
  /** σ of gauss[i] in this octave's pixels: σ0 k^i. */
  sigmas: number[];
}

export interface DoGPyramid {
  octaves: Octave[];
  S: number; k: number; sigma0: number; upsample: boolean;
  width: number; height: number;
}

export interface PyramidOptions {
  sigma0?: number;
  /** Steps per octave. */
  S?: number;
  /** Maximum number of octaves (stops when an octave would be smaller than 12 px). */
  octaves?: number;
  /** Double the image before the first octave (catches the smallest blobs). */
  upsample?: boolean;
  /** Blur already present in the input (pixels). */
  inBlur?: number;
}

/** σ in octave pixels of level i: σ0 k^i. Absolute σ (original px) = σ0 k^i · step. */
export function levelSigma(sigma0: number, S: number, i: number): number { return sigma0 * Math.pow(2, i / S); }

export function dogPyramid(img: Grey, o: PyramidOptions = {}): DoGPyramid {
  const sigma0 = o.sigma0 ?? 1.6, S = o.S ?? 3, up = !!o.upsample, inBlur = o.inBlur ?? 0.5;
  const k = Math.pow(2, 1 / S);
  let base: Grey = img, have = inBlur;
  if (up) { base = warp(img, 2 * img.width, 2 * img.height, (x, y) => [x / 2, y / 2], "bilinear", "replicate"); have = 2 * inBlur; }
  base = fastGauss(base, Math.sqrt(Math.max(0.01, sigma0 * sigma0 - have * have)));
  const octaves: Octave[] = [];
  const maxO = o.octaves ?? 8;
  let step = up ? 0.5 : 1;
  for (let oc = 0; oc < maxO; oc++) {
    if (oc > 0 && Math.min(base.width, base.height) < 12) break;
    const gauss: Grey[] = [base], sigmas = [sigma0];
    for (let i = 1; i < S + 3; i++) {
      const s0 = levelSigma(sigma0, S, i - 1), s1 = levelSigma(sigma0, S, i);
      gauss.push(fastGauss(gauss[i - 1], Math.sqrt(s1 * s1 - s0 * s0)));
      sigmas.push(s1);
    }
    const dog = gauss.slice(0, -1).map((g, i) => combine(gauss[i + 1], g, (a, b) => a - b));
    octaves.push({ gauss, dog, step, sigmas });
    // next octave: gauss[S] has σ = 2σ0; keep every second pixel (small pixel i = big pixel 2i)
    const g = gauss[S], w = Math.ceil(g.width / 2), h = Math.ceil(g.height / 2), nb = createGrey(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) nb.data[y * w + x] = g.data[2 * y * g.width + 2 * x];
    base = nb; step *= 2;
  }
  return { octaves, S, k, sigma0, upsample: up, width: img.width, height: img.height };
}

/** Absolute σ (original px) of octave o, continuous level s. */
export function pyramidSigma(p: DoGPyramid, o: number, s: number): number {
  return levelSigma(p.sigma0, p.S, s) * p.octaves[o].step;
}

/** Octave and continuous level for an absolute σ (clamped to the pyramid). */
export function placeSigma(p: DoGPyramid, sigma: number): { octave: number; level: number } {
  const step0 = p.octaves[0].step;
  const t = Math.log2(sigma / (p.sigma0 * step0)) * p.S;          // levels above octave 0, level 0
  let o = Math.floor(t / p.S);
  o = Math.max(0, Math.min(p.octaves.length - 1, o));
  const level = Math.max(0, Math.min(p.S + 2, t - o * p.S));
  return { octave: o, level };
}

// ---------------------------------------------------------------------------------------
// Candidates, refinement and the two tests

export interface Candidate { octave: number; level: number; ix: number; iy: number; value: number }

/** Extrema over the 26 neighbours on the interior DoG levels 1..S, with |D| ≥ preThreshold. */
export function dogCandidates(p: DoGPyramid, preThreshold: number, border = 1): Candidate[] {
  const out: Candidate[] = [];
  p.octaves.forEach((oc, o) => {
    const w = oc.dog[0].width, h = oc.dog[0].height;
    for (let s = 1; s <= p.S; s++) {
      const D0 = oc.dog[s - 1].data, D1 = oc.dog[s].data, D2 = oc.dog[s + 1].data;
      for (let y = border; y < h - border; y++) for (let x = border; x < w - border; x++) {
        const i = y * w + x, v = D1[i];
        if (Math.abs(v) < preThreshold) continue;
        const isMax = v > 0;
        let ok = true;
        for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1 && ok; dx++) {
          const j = i + dy * w + dx;
          const a = D0[j], b = D2[j], c = D1[j];
          if (isMax) { if (a >= v || b >= v || (j !== i && c >= v)) ok = false; }
          else if (a <= v || b <= v || (j !== i && c <= v)) ok = false;
        }
        if (ok) out.push({ octave: o, level: s, ix: x, iy: y, value: v });
      }
    }
  });
  return out;
}

export interface Refined {
  ok: boolean;
  /** Final integer sample and the sub-sample offset (x, y, level). */
  ix: number; iy: number; is: number;
  offset: [number, number, number];
  /** Interpolated peak value D + ½ ∇Dᵀ h. */
  value: number;
  /** Spatial Hessian of D at the final sample: dxx, dxy, dyy. */
  dxx: number; dxy: number; dyy: number;
  /** Gradient and 3 × 3 Hessian used for the last step. */
  grad: [number, number, number];
  hess: number[];
  /** Number of moves to a neighbour. */
  moves: number;
  /** Position in original pixels and absolute σ. */
  x: number; y: number; sigma: number;
}

/** Gradient and Hessian of D at an integer sample by central differences. */
export function dogDerivatives(oc: Octave, x: number, y: number, s: number): { grad: [number, number, number]; hess: number[]; v: number } {
  const w = oc.dog[0].width;
  const D = (ss: number, dx: number, dy: number) => oc.dog[ss].data[(y + dy) * w + x + dx];
  const v = D(s, 0, 0);
  const gx = (D(s, 1, 0) - D(s, -1, 0)) / 2, gy = (D(s, 0, 1) - D(s, 0, -1)) / 2, gs = (D(s + 1, 0, 0) - D(s - 1, 0, 0)) / 2;
  const dxx = D(s, 1, 0) - 2 * v + D(s, -1, 0), dyy = D(s, 0, 1) - 2 * v + D(s, 0, -1), dss = D(s + 1, 0, 0) - 2 * v + D(s - 1, 0, 0);
  const dxy = (D(s, 1, 1) - D(s, -1, 1) - D(s, 1, -1) + D(s, -1, -1)) / 4;
  const dxs = (D(s + 1, 1, 0) - D(s + 1, -1, 0) - D(s - 1, 1, 0) + D(s - 1, -1, 0)) / 4;
  const dys = (D(s + 1, 0, 1) - D(s + 1, 0, -1) - D(s - 1, 0, 1) + D(s - 1, 0, -1)) / 4;
  return { grad: [gx, gy, gs], hess: [dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], v };
}

/** Fit a 3D quadratic around the candidate; move to the neighbour while an offset exceeds 0.5. */
export function refineExtremum(p: DoGPyramid, c: Candidate, maxMoves = 5): Refined {
  const oc = p.octaves[c.octave], w = oc.dog[0].width, h = oc.dog[0].height;
  let x = c.ix, y = c.iy, s = c.level, moves = 0;
  let res: Refined | null = null;
  for (let it = 0; it <= maxMoves; it++) {
    const { grad, hess, v } = dogDerivatives(oc, x, y, s);
    const h3 = solve(hess, grad.map(g => -g), 3);
    const off: [number, number, number] = h3 ? [h3[0], h3[1], h3[2]] : [0, 0, 0];
    const conv = h3 !== null && Math.abs(off[0]) < 0.5 && Math.abs(off[1]) < 0.5 && Math.abs(off[2]) < 0.5;
    const value = v + 0.5 * (grad[0] * off[0] + grad[1] * off[1] + grad[2] * off[2]);
    res = {
      ok: conv, ix: x, iy: y, is: s, offset: off, value, dxx: hess[0], dxy: hess[1], dyy: hess[4], grad, hess, moves,
      x: (x + off[0]) * oc.step, y: (y + off[1]) * oc.step, sigma: levelSigma(p.sigma0, p.S, s + off[2]) * oc.step,
    };
    if (conv || !h3) break;
    if (it === maxMoves) break;
    x += Math.round(off[0]); y += Math.round(off[1]); s += Math.round(off[2]); moves++;
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1 || s < 1 || s > p.S) { res.ok = false; res.moves = moves; break; }
  }
  return res!;
}

/** Edge test: keep if det > 0 and tr²/det < (r + 1)²/r. */
export function edgeRatioTest(dxx: number, dxy: number, dyy: number, r = 10): { pass: boolean; ratio: number; limit: number } {
  const tr = dxx + dyy, det = dxx * dyy - dxy * dxy, limit = (r + 1) * (r + 1) / r;
  const ratio = det > 0 ? tr * tr / det : Infinity;
  return { pass: det > 0 && ratio < limit, ratio, limit };
}

export const contrastTest = (value: number, threshold: number): boolean => Math.abs(value) >= threshold;

export type Stage = "unstable" | "contrast" | "edge" | "kept";
export interface CandidateRecord { cand: Candidate; refined: Refined; stage: Stage; edge: { ratio: number; limit: number } }

export interface DoGOptions extends PyramidOptions {
  /** Contrast threshold on |D| after refinement (grey levels of a 0..255 image). */
  contrast?: number;
  /** Edge ratio r. */
  edgeR?: number;
  /** Keep only the strongest n (after the tests). */
  maxPoints?: number;
  /** Skip refinement (positions at the sample; for comparison). */
  noRefine?: boolean;
}

export interface DoGResult { pyr: DoGPyramid; kps: Keypoint[]; records: CandidateRecord[] }

/** DoG keypoints: candidates → refinement → contrast test → edge test (θ = 0; see assignOrientations). */
export function detectDoG(img: Grey, o: DoGOptions = {}, pyr?: DoGPyramid): DoGResult {
  const p = pyr ?? dogPyramid(img, o);
  const thr = o.contrast ?? 3, r = o.edgeR ?? 10;
  const cands = dogCandidates(p, 0.5 * thr);
  const records: CandidateRecord[] = [];
  const kps: Keypoint[] = [];
  for (const c of cands) {
    let ref: Refined;
    if (o.noRefine) {
      const oc = p.octaves[c.octave], d = dogDerivatives(oc, c.ix, c.iy, c.level);
      ref = { ok: true, ix: c.ix, iy: c.iy, is: c.level, offset: [0, 0, 0], value: c.value, dxx: d.hess[0], dxy: d.hess[1], dyy: d.hess[4], grad: d.grad, hess: d.hess, moves: 0, x: c.ix * oc.step, y: c.iy * oc.step, sigma: levelSigma(p.sigma0, p.S, c.level) * oc.step };
    } else ref = refineExtremum(p, c);
    const e = edgeRatioTest(ref.dxx, ref.dxy, ref.dyy, r);
    let stage: Stage;
    if (!ref.ok) stage = "unstable";
    else if (!contrastTest(ref.value, thr)) stage = "contrast";
    else if (!e.pass) stage = "edge";
    else stage = "kept";
    records.push({ cand: c, refined: ref, stage, edge: { ratio: e.ratio, limit: e.limit } });
    if (stage === "kept") kps.push({ x: ref.x, y: ref.y, sigma: ref.sigma, theta: 0, response: Math.abs(ref.value), octave: c.octave, level: ref.is + ref.offset[2] });
  }
  return { pyr: p, kps: o.maxPoints ? keepStrongest(kps, o.maxPoints) : kps, records };
}

/** The n strongest keypoints (by response). */
export function keepStrongest(kps: Keypoint[], n: number): Keypoint[] {
  return kps.slice().sort((a, b) => b.response - a.response).slice(0, n);
}

// ---------------------------------------------------------------------------------------
// Harris (one scale, many scales) and FAST

function maxima2D(R: Grey, thr: number, border: number): { x: number; y: number; v: number }[] {
  const { width: w, height: h, data } = R, out: { x: number; y: number; v: number }[] = [];
  for (let y = border; y < h - border; y++) for (let x = border; x < w - border; x++) {
    const v = data[y * w + x];
    if (!(v > thr)) continue;
    let ok = true;
    for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const u = data[(y + dy) * w + x + dx];
      if (u > v || (u === v && (dy < 0 || (dy === 0 && dx < 0)))) { ok = false; break; }
    }
    if (ok) out.push({ x, y, v });
  }
  return out;
}

/** Sub-pixel position of a 2D maximum by two 1D parabolas. */
function subpix(R: Grey, x: number, y: number): [number, number] {
  const w = R.width, d = R.data, v = d[y * w + x];
  return [x + parabolaOffset(d[y * w + x - 1], v, d[y * w + x + 1]), y + parabolaOffset(d[(y - 1) * w + x], v, d[(y + 1) * w + x])];
}

/** Harris response det J − k (tr J)² with J = G_ρ * (∇L ∇Lᵀ), ∇L at σ_D (as lib/edges.ts harris + structureTensor, with the fast blur). */
export function harrisResponse(img: Grey, sigmaD: number, rho: number, k = 0.05): Grey {
  const gx = fastGauss(img, sigmaD, 1, 0), gy = fastGauss(img, sigmaD, 0, 1), n = img.data.length;
  const pxx = createGrey(img.width, img.height), pxy = createGrey(img.width, img.height), pyy = createGrey(img.width, img.height);
  for (let i = 0; i < n; i++) { const a = gx.data[i], b = gy.data[i]; pxx.data[i] = a * a; pxy.data[i] = a * b; pyy.data[i] = b * b; }
  const jxx = fastGauss(pxx, rho), jxy = fastGauss(pxy, rho), jyy = fastGauss(pyy, rho), R = createGrey(img.width, img.height);
  for (let i = 0; i < n; i++) { const a = jxx.data[i], b = jxy.data[i], c = jyy.data[i]; R.data[i] = a * c - b * b - k * (a + c) ** 2; }
  return R;
}

/** Smaller eigenvalue of the structure tensor (Förstner / Shi–Tomasi), for comparison. */
export function minEigResponse(img: Grey, sigmaD: number, rho: number): Grey {
  const gx = fastGauss(img, sigmaD, 1, 0), gy = fastGauss(img, sigmaD, 0, 1), n = img.data.length;
  const p = [createGrey(img.width, img.height), createGrey(img.width, img.height), createGrey(img.width, img.height)];
  for (let i = 0; i < n; i++) { const a = gx.data[i], b = gy.data[i]; p[0].data[i] = a * a; p[1].data[i] = a * b; p[2].data[i] = b * b; }
  const [jxx, jxy, jyy] = p.map(q => fastGauss(q, rho)), R = createGrey(img.width, img.height);
  for (let i = 0; i < n; i++) R.data[i] = Math.max(0, eigSym2(jxx.data[i], jxy.data[i], jyy.data[i]).l2);
  return R;
}

export interface HarrisOptions { k?: number; /** ρ = rhoFactor · σ_D */ rhoFactor?: number; /** threshold relative to the strongest response */ rel?: number; maxPoints?: number; border?: number }

/** Harris corners at one derivative scale σ_D (window ρ = 1.5 σ_D), normalised by σ_D⁴. σ of a keypoint = ρ. */
export function detectHarris(img: Grey, sigmaD = 1.5, o: HarrisOptions = {}): Keypoint[] {
  const k = o.k ?? 0.05, rho = (o.rhoFactor ?? 1.5) * sigmaD, n4 = sigmaD ** 4;
  const R = harrisResponse(img, sigmaD, rho, k);
  let mx = 0; for (const v of R.data) mx = Math.max(mx, v);
  const pts = maxima2D(R, (o.rel ?? 0.01) * mx, o.border ?? Math.max(2, Math.ceil(2 * rho)));
  const kps = pts.map(p => { const [x, y] = subpix(R, p.x, p.y); return { x, y, sigma: rho, theta: 0, response: p.v * n4, octave: -1, level: 0 }; });
  return o.maxPoints ? keepStrongest(kps, o.maxPoints) : kps;
}

/** Scales of the multi-scale Harris detector: σ_n = σ_first · ratio^n. */
export function harrisScales(first = 1.2, ratio = 1.4, n = 8): number[] {
  return Array.from({ length: n }, (_, i) => first * Math.pow(ratio, i));
}

/**
 * Multi-scale Harris (Harris–Laplace style): at each integration scale σ_n (σ_D = 0.7 σ_n),
 * the normalised Harris response σ_D⁴(det J − k tr²J) has local maxima; keep one only if the
 * normalised Laplacian σ_n²|∇²L| at that point is larger than at the neighbouring scales.
 * σ of a keypoint = σ_n.
 */
export function detectHarrisMultiScale(img: Grey, scales = harrisScales(), o: HarrisOptions = {}): Keypoint[] {
  const k = o.k ?? 0.05;
  // large scales are computed on a 2^m-times smaller image (σ / 2^m ≥ 1.2), which is equivalent and much faster
  const small: Grey[] = [img];
  const factor = scales.map(s => { let m = 0; while (s / 2 ** (m + 1) >= 1.2 && img.width / 2 ** (m + 1) >= 24) m++; return m; });
  const imgAt = (m: number): Grey => {
    while (small.length <= m) { const g = fastGauss(small[small.length - 1], 0.8), w = Math.ceil(g.width / 2), h = Math.ceil(g.height / 2), d = createGrey(w, h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d.data[y * w + x] = g.data[2 * y * g.width + 2 * x]; small.push(d); }
    return small[m];
  };
  const lapAt = scales.map((s, n) => {
    const f = 2 ** factor[n], im = imgAt(factor[n]), t = s / f;
    const a = fastGauss(im, t, 2, 0), b = fastGauss(im, t, 0, 2);
    return combine(a, b, (u, v) => t * t * Math.abs(u + v));
  });
  const resp = scales.map((s, n) => {
    const f = 2 ** factor[n], t = s / f, sd = 0.7 * t, R = harrisResponse(imgAt(factor[n]), sd, t, k), n4 = sd ** 4;
    for (let i = 0; i < R.data.length; i++) R.data[i] *= n4;
    return R;
  });
  const lapSample = (n: number, x: number, y: number): number => {
    const f = 2 ** factor[n], L = lapAt[n], u = Math.max(0, Math.min(L.width - 1, Math.round(x / f))), v = Math.max(0, Math.min(L.height - 1, Math.round(y / f)));
    return L.data[v * L.width + u];
  };
  let mx = 0; for (const R of resp) for (const v of R.data) mx = Math.max(mx, v);
  const kps: Keypoint[] = [];
  scales.forEach((s, n) => {
    const f = 2 ** factor[n];
    const pts = maxima2D(resp[n], (o.rel ?? 0.01) * mx, o.border ?? Math.max(2, Math.ceil(2 * s / f)));
    for (const p of pts) {
      const X = p.x * f, Y = p.y * f, L = lapSample(n, X, Y);
      const lo = n > 0 ? lapSample(n - 1, X, Y) : -Infinity, hi = n < scales.length - 1 ? lapSample(n + 1, X, Y) : -Infinity;
      if (!(L > lo && L > hi)) continue;
      const [x, y] = subpix(resp[n], p.x, p.y);
      kps.push({ x: x * f, y: y * f, sigma: s, theta: 0, response: p.v, octave: -1, level: n });
    }
  });
  return o.maxPoints ? keepStrongest(kps, o.maxPoints) : kps;
}

/** The FAST ring: 16 offsets on a Bresenham circle of radius 3, clockwise from the top. */
export const FAST_RING: [number, number][] = [[0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3], [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3]];

/** FAST test at one pixel: +1 (n contiguous brighter than c + t), −1 (darker than c − t) or 0; and the score. */
export function fastTest(img: Grey, x: number, y: number, t: number, n = 9): { kind: number; score: number } {
  const w = img.width, d = img.data, c = d[y * w + x];
  const v = FAST_RING.map(([dx, dy]) => d[(y + dy) * w + x + dx]);
  for (const sign of [1, -1]) {
    let run = 0, best = 0;
    for (let i = 0; i < 32; i++) {
      const u = v[i % 16];
      if (sign * (u - c) > t) { run++; best = Math.max(best, run); } else run = 0;
    }
    if (Math.min(best, 16) >= n) {
      let score = 0;
      for (const u of v) score += Math.max(0, sign * (u - c) - t);
      return { kind: sign, score };
    }
  }
  return { kind: 0, score: 0 };
}

/** FAST corners with 3 × 3 non-maximum suppression on the score; σ of a keypoint = `sigma` (no scale). */
export function detectFAST(img: Grey, t = 20, n = 9, o: { maxPoints?: number; sigma?: number } = {}): Keypoint[] {
  const w = img.width, h = img.height, S = createGrey(w, h);
  for (let y = 3; y < h - 3; y++) for (let x = 3; x < w - 3; x++) S.data[y * w + x] = fastTest(img, x, y, t, n).score;
  const kps = maxima2D(S, 0, 3).map(p => ({ x: p.x, y: p.y, sigma: o.sigma ?? 2, theta: 0, response: p.v, octave: -1, level: 0 }));
  return o.maxPoints ? keepStrongest(kps, o.maxPoints) : kps;
}

// ---------------------------------------------------------------------------------------
// Orientation

export interface OrientationOptions { bins?: number; peakRatio?: number; /** window σ = winFactor · σ */ winFactor?: number; smooth?: number }

/** Blurred image of the pyramid nearest to a keypoint's σ, with the keypoint in its pixels. */
export function levelImage(p: DoGPyramid, kp: { x: number; y: number; sigma: number }): { img: Grey; cx: number; cy: number; s: number; step: number; octave: number; level: number } {
  const { octave, level } = placeSigma(p, kp.sigma), oc = p.octaves[octave];
  const li = Math.max(0, Math.min(oc.gauss.length - 1, Math.round(level)));
  return { img: oc.gauss[li], cx: kp.x / oc.step, cy: kp.y / oc.step, s: kp.sigma / oc.step, step: oc.step, octave, level: li };
}

/**
 * Histogram of gradient directions (counter-clockwise on the screen, bin b covers
 * [b, b + 1)·360°/bins) around a keypoint, weighted by magnitude × Gaussian (σ_w = 1.5 σ),
 * then smoothed circularly (`smooth` passes of [1 1 1]/3). Peaks ≥ peakRatio · max that are
 * local maxima; their angle refined by a parabola through three bins.
 */
export function orientationHistogram(p: DoGPyramid, kp: { x: number; y: number; sigma: number }, o: OrientationOptions = {}): { hist: Float64Array; peaks: number[]; raw: Float64Array } {
  const nb = o.bins ?? 36, ratio = o.peakRatio ?? 0.8, L = levelImage(p, kp);
  const sw = (o.winFactor ?? 1.5) * L.s, R = Math.max(1, Math.round(3 * sw));
  const img = L.img, w = img.width, h = img.height, d = img.data;
  const raw = new Float64Array(nb);
  const x0 = Math.round(L.cx), y0 = Math.round(L.cy);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const x = x0 + dx, y = y0 + dy;
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) continue;
    const rx = x - L.cx, ry = y - L.cy, r2 = rx * rx + ry * ry;
    if (r2 > R * R) continue;
    const gx = (d[y * w + x + 1] - d[y * w + x - 1]) / 2, gy = (d[(y + 1) * w + x] - d[(y - 1) * w + x]) / 2;
    const m = Math.hypot(gx, gy);
    if (m === 0) continue;
    let a = Math.atan2(-gy, gx); if (a < 0) a += 2 * Math.PI;
    const b = Math.min(nb - 1, Math.floor(a / (2 * Math.PI) * nb));
    raw[b] += m * Math.exp(-r2 / (2 * sw * sw));
  }
  let hist = Float64Array.from(raw);
  for (let pass = 0; pass < (o.smooth ?? 2); pass++) {
    const t = new Float64Array(nb);
    for (let b = 0; b < nb; b++) t[b] = (hist[(b + nb - 1) % nb] + hist[b] + hist[(b + 1) % nb]) / 3;
    hist = t;
  }
  let mx = 0; for (const v of hist) mx = Math.max(mx, v);
  const peaks: number[] = [];
  if (mx > 0) for (let b = 0; b < nb; b++) {
    const l = hist[(b + nb - 1) % nb], c = hist[b], r = hist[(b + 1) % nb];
    if (c >= ratio * mx && c > l && c >= r) {
      const off = parabolaOffset(l, c, r);
      let a = (b + 0.5 + off) / nb * 2 * Math.PI;
      a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      peaks.push(a);
    }
  }
  // strongest peak first
  peaks.sort((a, b) => hist[Math.floor(b / (2 * Math.PI) * nb) % nb] - hist[Math.floor(a / (2 * Math.PI) * nb) % nb]);
  return { hist, peaks, raw };
}

/** One keypoint per orientation peak (a second peak above 80 % gives a second keypoint at the same place). */
export function assignOrientations(p: DoGPyramid, kps: Keypoint[], o: OrientationOptions = {}): Keypoint[] {
  const out: Keypoint[] = [];
  for (const kp of kps) {
    const { peaks } = orientationHistogram(p, kp, o);
    if (!peaks.length) { out.push({ ...kp }); continue; }
    for (const a of peaks) out.push({ ...kp, theta: a });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Repeatability

/** A 2D map from image 1 to image 2 (e.g. a Transform2D applied by lib/transforms2d.ts). */
export type PointMap = (x: number, y: number) => [number, number];

export interface RepeatOptions {
  eps?: number;
  /** |log(s σ1 / σ2)| < scaleTol; Infinity ignores the scale (single-scale detectors). */
  scaleTol?: number;
  /** Scale factor of the map (for the scale test). */
  scale?: number;
  /** Image sizes; points within `margin` of the border of the other image are not counted. */
  w1: number; h1: number; w2: number; h2: number; margin?: number;
}

export interface RepeatResult {
  rep: number; repeated: number; n1: number; n2: number;
  /** Per keypoint of image 1: −1 outside the common region, else the index of its repeated partner or −2 if none. */
  partner: Int32Array;
}

/** Fraction of points (in the common region) found again within eps px at a matching scale (each point of image 2 used once). */
export function repeatability(k1: Keypoint[], k2: Keypoint[], T: PointMap, Tinv: PointMap, o: RepeatOptions): RepeatResult {
  const eps = o.eps ?? 2.5, tol = o.scaleTol ?? Math.log(1.5), s = o.scale ?? 1, m = o.margin ?? 4;
  const in2 = (x: number, y: number) => x >= m && y >= m && x <= o.w2 - 1 - m && y <= o.h2 - 1 - m;
  const in1 = (x: number, y: number) => x >= m && y >= m && x <= o.w1 - 1 - m && y <= o.h1 - 1 - m;
  const partner = new Int32Array(k1.length).fill(-1);
  let n1 = 0, rep = 0;
  // grid of image-2 keypoints for speed
  const cell = Math.max(4, eps), gw = Math.ceil(o.w2 / cell) + 1, gh = Math.ceil(o.h2 / cell) + 1;
  const grid: number[][] = Array.from({ length: gw * gh }, () => []);
  k2.forEach((p, j) => { const gx = Math.floor(p.x / cell), gy = Math.floor(p.y / cell); if (gx >= 0 && gy >= 0 && gx < gw && gy < gh) grid[gy * gw + gx].push(j); });
  // candidate pairs within eps (and at a matching scale), then one-to-one, nearest first
  const pairs: [number, number, number][] = [];
  k1.forEach((p, i) => {
    const [x, y] = T(p.x, p.y);
    if (!in2(x, y)) return;
    n1++;
    partner[i] = -2;
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
    for (let v = gy - 1; v <= gy + 1; v++) for (let u = gx - 1; u <= gx + 1; u++) {
      if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
      for (const j of grid[v * gw + u]) {
        const q = k2[j], d = Math.hypot(q.x - x, q.y - y);
        if (d < eps && (tol === Infinity || Math.abs(Math.log(s * p.sigma / q.sigma)) < tol)) pairs.push([i, j, d]);
      }
    }
  });
  pairs.sort((a, b) => a[2] - b[2]);
  const used = new Uint8Array(k2.length);
  for (const [i, j] of pairs) if (partner[i] === -2 && !used[j]) { partner[i] = j; used[j] = 1; rep++; }
  let n2 = 0;
  for (const q of k2) { const [x, y] = Tinv(q.x, q.y); if (in1(x, y)) n2++; }
  const den = Math.min(n1, n2);
  return { rep: den > 0 ? Math.min(1, rep / den) : 0, repeated: rep, n1, n2, partner };
}
