// Closed outlines as periodic complex signals: Fourier descriptors and matching (owner:
// shape-matching day, unit fourier-descriptors/; also used by the moments demo). No DOM.
//
// An outline is a list of points [x, y] in image orientation (x right, y down), read as the
// complex signal z_n = x_n + i y_n, n = 0 … N−1. Its descriptors are
//   c_k = (1/N) Σ_n z_n e^{−2πi k n/N},   k signed (index j ↔ k = freq(j, N), lib/fft.ts),
// so z_n = Σ_k c_k e^{2πi k n/N}. Multiplying by e^{iα} turns a point CLOCKWISE on screen
// by α (the course convention); outlines traced clockwise on screen have positive shoelace
// area and a dominant c_1.
//
//  - traceBoundary(mask, w, h): outer boundary of the object that holds the first pixel in
//    raster order (Moore tracing, 8-connected), pixel centres at integer coordinates, clockwise.
//  - resampleArcLength(poly, n, offset): n points equally spaced in arc length along the
//    closed polygon (offset = fraction of the length where sample 0 sits); resampleByIndex:
//    equally spaced in vertex index (what you get without arc-length resampling).
//  - descriptors(z) → Coeffs; coef(c, k); evalSeries / reconstruct (|k| ≤ K, the Nyquist term
//    split between ±N/2 so the curve between samples is the natural interpolation); epicycles.
//  - the five rules as operations on outlines: translateOutline, scaleOutline, rotateOutline,
//    restart (start m samples later), reverse (walk the other way), mirror (y → −y),
//    mirrorTraced (mirror, then walked clockwise again: c_k → conj(c_k)).
//  - matchOutlines(a, b): best rotation + scale (one complex λ) and starting point m for
//    B ≈ λ·A(start m) + t, for all m at once by one FFT of conj(c^A_k)·c^B_k; optional mirror.
//  - magnitudeSignature / magnitudeDistance: |c_k|/|c_1| (rotation, scale, start invariant).
//  - firstEllipse, ellipticDescriptors (Kuhl & Giardina, normalised by the first ellipse).
//  - fourierOutline(pts, k): smoothed outline + magnitudes (moved here from moments/features.ts).
import type { Pt } from "./convexhull";
import { dft1d, fft1d, isPow2, freq } from "./fft";

export type { Pt };

/** Descriptors c_j for j = 0 … n−1 (j ↔ signed k = freq(j, n)), already divided by n. */
export interface Coeffs { re: Float64Array; im: Float64Array; n: number }

// ---------------------------------------------------------------------------------------
// From a mask to an outline

/** Outer boundary (pixel centres, clockwise on screen) of the 8-connected object that contains
 *  the first foreground pixel in raster order (or the pixel index `start`, which must have
 *  background to its left). A single pixel gives one point. */
export function traceBoundary(mask: ArrayLike<number>, w: number, h: number, start?: number): Pt[] {
  let s = start ?? -1;
  if (s < 0) for (let i = 0; i < w * h; i++) if (mask[i]) { s = i; break; }
  if (s < 0) return [];
  const fg = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && !!mask[y * w + x];
  // neighbours clockwise on screen, starting east
  const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const sx = s % w, sy = (s - sx) / w;
  // one step: from pixel (x, y) with backtrack direction b (towards a background neighbour),
  // scan clockwise from b and move to the first foreground neighbour
  const step = (x: number, y: number, b: number): [number, number, number] | null => {
    for (let j = 1; j <= 8; j++) {
      const d = (b + j) % 8;
      if (fg(x + DX[d], y + DY[d])) {
        // new backtrack: the neighbour examined just before, seen from the new pixel
        const pd = (b + j - 1) % 8, px = x + DX[pd], py = y + DY[pd], nx = x + DX[d], ny = y + DY[d];
        let nb = 0;
        for (let q = 0; q < 8; q++) if (nx + DX[q] === px && ny + DY[q] === py) { nb = q; break; }
        return [nx, ny, nb];
      }
    }
    return null;
  };
  const pts: Pt[] = [[sx, sy]];
  const first = step(sx, sy, 4);              // west of the first pixel is background
  if (!first) return pts;
  let [x, y, b] = first;
  for (let it = 0; it < 4 * w * h + 8; it++) {
    if (x === sx && y === sy) {
      const nxt = step(x, y, b)!;
      if (nxt[0] === first[0] && nxt[1] === first[1]) break;   // back where we started
    }
    pts.push([x, y]);
    const nxt = step(x, y, b)!;
    [x, y, b] = nxt;
  }
  return pts;
}

/** Signed shoelace area: positive for an outline walked clockwise on screen (y down). */
export function signedArea(poly: Pt[]): number {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2;
}

/** The polygon walked clockwise on screen (reversed if it was anticlockwise). */
export function clockwise(poly: Pt[]): Pt[] {
  return signedArea(poly) < 0 ? reverse(poly) : poly.slice();
}

/** Length of the closed polygon. */
export function polygonLength(poly: Pt[]): number {
  let L = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    L += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return L;
}

/** n points equally spaced in arc length along the closed polygon; sample 0 sits at the
 *  fraction `offset` of the length after vertex 0. */
export function resampleArcLength(poly: Pt[], n: number, offset = 0): Pt[] {
  const m = poly.length;
  if (m === 0) return [];
  if (m === 1) return Array.from({ length: n }, () => [poly[0][0], poly[0][1]] as Pt);
  const cum = new Float64Array(m + 1);
  for (let i = 0; i < m; i++) {
    const a = poly[i], b = poly[(i + 1) % m];
    cum[i + 1] = cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  const L = cum[m], out: Pt[] = [];
  const off = ((offset % 1) + 1) % 1;
  for (let i = 0; i < n; i++) {
    let t = L * (off + i / n);
    if (t >= L) t -= L;
    // binary search for the segment containing t
    let lo = 0, hi = m - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (cum[mid] <= t) lo = mid; else hi = mid - 1; }
    const a = poly[lo], b = poly[(lo + 1) % m], f = (t - cum[lo]) / Math.max(1e-300, cum[lo + 1] - cum[lo]);
    out.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
  }
  return out;
}

/** n points equally spaced in vertex index (each polygon edge counts as one step, whatever its
 *  length): the outline without arc-length resampling. */
export function resampleByIndex(poly: Pt[], n: number): Pt[] {
  const m = poly.length, out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const t = m * i / n, j = Math.floor(t), f = t - j, a = poly[j % m], b = poly[(j + 1) % m];
    out.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Descriptors and reconstruction

/** c_k = (1/N) Σ z_n e^{−2πikn/N} (FFT when N is a power of two). */
export function descriptors(z: Pt[]): Coeffs {
  const n = z.length;
  let re: Float64Array = Float64Array.from(z, p => p[0]), im: Float64Array = Float64Array.from(z, p => p[1]);
  if (isPow2(n)) fft1d(re, im, false);
  else ({ re, im } = dft1d(re, im, false));
  for (let j = 0; j < n; j++) { re[j] /= n; im[j] /= n; }
  return { re, im, n };
}

/** The outline back from its descriptors (all of them): z_n = Σ_k c_k e^{2πikn/N}. */
export function samplesFromCoeffs(c: Coeffs): Pt[] {
  const re = c.re.slice(), im = c.im.slice(), n = c.n;
  let R: Float64Array = re, I: Float64Array = im;
  if (isPow2(n)) fft1d(R, I, true);
  else ({ re: R, im: I } = dft1d(re, im, true));
  return Array.from({ length: n }, (_, j) => [R[j], I[j]] as Pt);
}

const idx = (k: number, n: number) => ((k % n) + n) % n;

/** c_k for a signed k (wraps around modulo N). */
export function coef(c: Coeffs, k: number): [number, number] {
  const j = idx(k, c.n);
  return [c.re[j], c.im[j]];
}

/** The terms kept with |k| ≤ K, as (k, weight) pairs; for even N and K ≥ N/2 the Nyquist
 *  coefficient is split between k = N/2 and k = −N/2 (weight ½ each). */
function terms(n: number, K: number): { j: number; k: number; wt: number }[] {
  const out: { j: number; k: number; wt: number }[] = [];
  for (let j = 0; j < n; j++) {
    const k = freq(j, n);
    if (n % 2 === 0 && j === n / 2) {
      if (K >= n / 2) { out.push({ j, k: n / 2, wt: 0.5 }, { j, k: -n / 2, wt: 0.5 }); }
      continue;
    }
    if (Math.abs(k) <= K) out.push({ j, k, wt: 1 });
  }
  return out;
}

/** The Fourier series with |k| ≤ K at parameter t ∈ [0, 1) (t = n/N is sample n). */
export function evalSeries(c: Coeffs, K: number, t: number): Pt {
  let x = 0, y = 0;
  for (const { j, k, wt } of terms(c.n, K)) {
    const a = 2 * Math.PI * k * t, cs = Math.cos(a), sn = Math.sin(a);
    x += wt * (c.re[j] * cs - c.im[j] * sn); y += wt * (c.re[j] * sn + c.im[j] * cs);
  }
  return [x, y];
}

/** m points of the outline rebuilt from |k| ≤ K (a smoothed outline for small K). */
export function reconstruct(c: Coeffs, K: number, m: number): Pt[] {
  return Array.from({ length: m }, (_, i) => evalSeries(c, K, i / m));
}

/** Epicycles at parameter t: the chain of circles k = 0, 1, −1, 2, −2, … (|k| ≤ K). Each
 *  entry is the centre of circle k, its radius |c_k|, and the point reached after adding it. */
export function epicycles(c: Coeffs, K: number, t: number): { k: number; centre: Pt; r: number; tip: Pt }[] {
  const ts = terms(c.n, K).sort((a, b) => Math.abs(a.k) - Math.abs(b.k) || b.k - a.k);
  const out: { k: number; centre: Pt; r: number; tip: Pt }[] = [];
  let x = 0, y = 0;
  for (const { j, k, wt } of ts) {
    const a = 2 * Math.PI * k * t, cs = Math.cos(a), sn = Math.sin(a);
    const vx = wt * (c.re[j] * cs - c.im[j] * sn), vy = wt * (c.re[j] * sn + c.im[j] * cs);
    const centre: Pt = [x, y];
    x += vx; y += vy;
    out.push({ k, centre, r: Math.hypot(vx, vy), tip: [x, y] });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// What a change does: the five rules as operations on outlines

export const translateOutline = (z: Pt[], dx: number, dy: number): Pt[] => z.map(p => [p[0] + dx, p[1] + dy] as Pt);
export const scaleOutline = (z: Pt[], s: number): Pt[] => z.map(p => [s * p[0], s * p[1]] as Pt);
/** Multiply by e^{iα}: turns the outline about the origin by α, clockwise on screen. */
export function rotateOutline(z: Pt[], alpha: number): Pt[] {
  const c = Math.cos(alpha), s = Math.sin(alpha);
  return z.map(p => [c * p[0] - s * p[1], s * p[0] + c * p[1]] as Pt);
}
/** Start m samples later: z'_n = z_{n+m}. */
export const restart = (z: Pt[], m: number): Pt[] => z.map((_, n) => z[idx(n + m, z.length)]);
/** Walk the other way round from the same start: z'_n = z_{−n}. */
export const reverse = (z: Pt[]): Pt[] => z.map((_, n) => z[idx(-n, z.length)]);
/** Mirror image y → −y (conjugate every sample); walked anticlockwise afterwards. */
export const mirror = (z: Pt[]): Pt[] => z.map(p => [p[0], -p[1]] as Pt);
/** Mirror image walked clockwise again from the same start, z'_n = conj(z_{−n}): c_k → conj(c_k). */
export const mirrorTraced = (z: Pt[]): Pt[] => reverse(mirror(z));

/** Similarity applied about the origin: z ↦ s e^{iα} z + (tx, ty). */
export const similarity = (z: Pt[], s: number, alpha: number, tx = 0, ty = 0): Pt[] =>
  translateOutline(scaleOutline(rotateOutline(z, alpha), s), tx, ty);

/** Centroid of the samples (= c_0). */
export function sampleMean(z: Pt[]): Pt {
  let x = 0, y = 0;
  for (const p of z) { x += p[0]; y += p[1]; }
  return [x / z.length, y / z.length];
}

// ---------------------------------------------------------------------------------------
// Matching over rotation, scale and starting point

export interface MatchScores {
  /** q(m) = Σ_{k≠0, |k|≤K} conj(c^A_k) c^B_k e^{−2πikm/N}, for m = 0 … N−1. */
  qRe: Float64Array; qIm: Float64Array;
  /** |q(m)| / √(Σ|c^A_k|² Σ|c^B_k|²): 1 for a perfect match (a normalised correlation). */
  score: Float64Array;
  energyA: number; energyB: number;
}

/** The match score for every starting point m at once: one FFT of conj(c^A_k)·c^B_k. */
export function matchScores(cA: Coeffs, cB: Coeffs, K = Infinity): MatchScores {
  const n = cA.n;
  if (cB.n !== n) throw new Error("matchScores: both outlines need the same number of samples");
  let re: Float64Array = new Float64Array(n), im: Float64Array = new Float64Array(n);
  let eA = 0, eB = 0;
  for (let j = 1; j < n; j++) {
    if (Math.abs(freq(j, n)) > K) continue;
    const ar = cA.re[j], ai = cA.im[j], br = cB.re[j], bi = cB.im[j];
    re[j] = ar * br + ai * bi; im[j] = ar * bi - ai * br;      // conj(a)·b
    eA += ar * ar + ai * ai; eB += br * br + bi * bi;
  }
  if (isPow2(n)) fft1d(re, im, false);                       // forward: Σ_k P_k e^{−2πikm/N}
  else ({ re, im } = dft1d(re, im, false));
  const nrm = Math.sqrt(eA * eB) || 1;
  const score = Float64Array.from(re, (r, m) => Math.hypot(r, im[m]) / nrm);
  return { qRe: re, qIm: im, score, energyA: eA, energyB: eB };
}

export interface Match {
  /** Best starting point: B_n ≈ λ·A_{n+m} + t. */
  m: number;
  /** λ = s e^{iα}: scale s, angle α (radians, clockwise on screen). */
  lambda: [number, number]; scale: number; angle: number;
  t: [number, number];
  /** Normalised score at the peak (1 = perfect) and the residual fraction 1 − score². */
  score: number; residual: number;
  /** True if the mirror image of A matched better. */
  mirrored: boolean;
  /** Scores over m for A, and for its mirror image (if tried). */
  scores: Float64Array; scoresMirror: Float64Array | null;
  /** A (or its mirror image) moved onto B: λ·(A_{n+m} − c^A_0) + c^B_0. */
  aligned: Pt[];
}

function bestOf(s: MatchScores): number {
  let m = 0;
  for (let i = 1; i < s.score.length; i++) if (s.score[i] > s.score[m]) m = i;
  return m;
}

/** Match outline B to outline A (same number of samples) over rotation, scale, translation and
 *  starting point; with `mirror`, also to A's mirror image (walked clockwise again). */
export function matchOutlines(a: Pt[], b: Pt[], o: { mirror?: boolean; K?: number } = {}): Match {
  const cA = descriptors(a), cB = descriptors(b), K = o.K ?? Infinity;
  const sN = matchScores(cA, cB, K);
  let use = sN, src = a, cS = cA, mirrored = false, sM: MatchScores | null = null;
  if (o.mirror) {
    const am = mirrorTraced(a), cM = descriptors(am);
    sM = matchScores(cM, cB, K);
    if (sM.score[bestOf(sM)] > sN.score[bestOf(sN)] + 1e-12) { use = sM; src = am; cS = cM; mirrored = true; }
  }
  const m = bestOf(use);
  const lr = use.qRe[m] / (use.energyA || 1), li = use.qIm[m] / (use.energyA || 1);
  const tx = cB.re[0] - (lr * cS.re[0] - li * cS.im[0]), ty = cB.im[0] - (lr * cS.im[0] + li * cS.re[0]);
  const aligned = restart(src, m).map(p => [lr * p[0] - li * p[1] + tx, lr * p[1] + li * p[0] + ty] as Pt);
  const score = use.score[m];
  return {
    m, lambda: [lr, li], scale: Math.hypot(lr, li), angle: Math.atan2(li, lr), t: [tx, ty],
    score, residual: Math.max(0, 1 - score * score), mirrored,
    scores: sN.score, scoresMirror: sM ? sM.score : null, aligned,
  };
}

/** Mean squared distance between two outlines sample by sample. */
export function meanSqDist(a: Pt[], b: Pt[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2;
  return s / a.length;
}

// ---------------------------------------------------------------------------------------
// Magnitudes, invariants, the first ellipse

/** |c_k| / |c_1| for k = −K … −1, 1 … K (in that order): unchanged by translation, scale,
 *  rotation and starting point (outlines walked in the same direction). */
export function magnitudeSignature(c: Coeffs, K: number): Float64Array {
  const [r1, i1] = coef(c, 1), m1 = Math.hypot(r1, i1) || 1, out: number[] = [];
  for (let k = -K; k <= K; k++) {
    if (k === 0) continue;
    const [r, i] = coef(c, k);
    out.push(Math.hypot(r, i) / m1);
  }
  return Float64Array.from(out);
}

/** Euclidean distance between the magnitude signatures (0: indistinguishable by magnitudes). */
export function magnitudeDistance(cA: Coeffs, cB: Coeffs, K: number): number {
  const a = magnitudeSignature(cA, K), b = magnitudeSignature(cB, K);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

/** The ellipse of c_{±1}: semi-axes |c_1| ± |c_{−1}|, orientation ψ = (arg c_1 + arg c_{−1})/2 of
 *  the major axis (clockwise from +x on screen), and the parameter angle θ0 = (arg c_{−1} − arg c_1)/2
 *  at which the ellipse is at the end of its major axis. */
export function firstEllipse(c: Coeffs): { a: number; b: number; psi: number; theta0: number } {
  const [r1, i1] = coef(c, 1), [rm, im] = coef(c, -1);
  const p = Math.hypot(r1, i1), q = Math.hypot(rm, im), a1 = Math.atan2(i1, r1), am = Math.atan2(im, rm);
  return { a: p + q, b: Math.abs(p - q), psi: (a1 + am) / 2, theta0: (am - a1) / 2 };
}

/** Elliptic Fourier descriptors (Kuhl & Giardina) for harmonics 1 … K, normalised by the first
 *  ellipse (start at the end of its major axis, major axis along +x, length 1):
 *    x(θ) = Σ a_n cos nθ + b_n sin nθ,   y(θ) = Σ c_n cos nθ + d_n sin nθ.
 *  Then a_1 = 1, b_1 = c_1 = 0 and d_1 = ±(minor / major). Without normalisation with `raw`. */
export function ellipticDescriptors(c: Coeffs, K: number, raw = false): { a: number[]; b: number[]; c: number[]; d: number[] } {
  const e = firstEllipse(c), out = { a: [] as number[], b: [] as number[], c: [] as number[], d: [] as number[] };
  const norm = (k: number): [number, number] => {
    const [r, i] = coef(c, k);
    if (raw) return [r, i];
    const ang = k * e.theta0 - e.psi, cs = Math.cos(ang), sn = Math.sin(ang);
    return [(r * cs - i * sn) / e.a, (r * sn + i * cs) / e.a];
  };
  for (let n = 1; n <= K; n++) {
    const [pr, pi] = norm(n), [qr, qi] = norm(-n);
    out.a.push(pr + qr); out.b.push(-(pi - qi)); out.c.push(pi + qi); out.d.push(pr - qr);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Shape features (labelling day): smoothing by truncation (moved from moments/features.ts)

/** Fourier descriptors of a closed outline z(t) = x + i y: keep the coefficients with
 *  frequency |f| ≤ k (and the mean), reconstruct. Returns the smoothed outline and the
 *  magnitudes |c_f| / |c_1| for f = 1..kmax (translation- and scale-invariant). */
export function fourierOutline(pts: Pt[], k: number): { smooth: Pt[]; mags: number[] } {
  const n = pts.length;
  if (n < 3) return { smooth: pts, mags: [] };
  const re = Float64Array.from(pts, p => p[0]), im = Float64Array.from(pts, p => p[1]);
  const F = dft1d(re, im, false);
  const fr = F.re, fi = F.im;
  const kr = new Float64Array(n), ki = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const f = j <= n / 2 ? j : j - n;
    if (Math.abs(f) <= k) { kr[j] = fr[j]; ki[j] = fi[j]; }
  }
  const B = dft1d(kr, ki, true);
  const smooth: Pt[] = Array.from({ length: n }, (_, j) => [B.re[j] / n, B.im[j] / n] as Pt);
  // the boundary is traced clockwise on screen: the dominant coefficient is f = 1 or f = −1
  const m1 = Math.max(Math.hypot(fr[1], fi[1]), Math.hypot(fr[n - 1], fi[n - 1])) || 1;
  const mags: number[] = [];
  for (let f = 1; f <= Math.min(12, Math.floor(n / 2) - 1); f++) mags.push((Math.hypot(fr[f], fi[f]) + Math.hypot(fr[n - f], fi[n - f])) / m1);
  return { smooth, mags };
}
