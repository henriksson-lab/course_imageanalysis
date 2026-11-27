// Registration by correlation (owner: the FFT registration day). No DOM; runs under node.
//
// Convention used everywhere in the course (and here): the SHIFT t of image B relative to
// image A is where B must be placed on A so that the two agree,
//     B(x) ≈ A(x + t),
// i.e. B's frame is t to the right of / below A's frame. The score of a candidate shift
// compares the pixels lying on top of each other, A(x + t) and B(x):
//     c(t) = Σ_x A(x + t) B(x)            (cross-correlation)
// and its spectrum is ĉ_k = A_k · conj(B_k). A correlation SURFACE is stored like an image
// of size W × H, entry [ty mod H][tx mod W] = c(tx, ty) (shift 0 in the corner, as an FFT
// returns it); `signed` maps an index back to a shift in (−W/2, W/2].
//
// Contents: tapers · 1D scores and brute force (with an operation counter) · FFT
// correlation in 1D and 2D (CC, partial normalisation |·|^p, phase correlation, low-pass
// weight, taper, zero-padding) · masked NCC by FFT (overlap NCC for every shift; template
// matching) · peaks, peak ratio, the four candidates of a circular peak · sub-pixel
// (parabola, centroid, upsampled DFT) · Fourier (phase-ramp) shifts · global stitching by
// weighted least squares on a tile graph, spanning trees.

import { fft1d, fft2dRect, dft1d, freq, isPow2 } from "./fft";
import { solve } from "./linalg";
import { toPolar } from "./polar";
import { Grey, Border, sampleBilinear } from "./imageops";

export interface Complex { re: Float64Array; im: Float64Array }
export interface Surface { data: Float64Array; w: number; h: number }

/** Signed shift of circular index i in a period of n: 0..n/2 stay, the rest become negative. */
export const signed = (i: number, n: number): number => (i <= n / 2 ? i : i - n);
/** Circular index of a signed shift. */
export const wrapIndex = (t: number, n: number): number => ((t % n) + n) % n;

// ---------------------------------------------------------------------------------------
// Tapers

/** Hann taper of length n: ½(1 − cos(2πi/(n−1))), 0 at both ends, 1 in the middle. */
export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  if (n === 1) { w[0] = 1; return w; }
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
  return w;
}

/** Tukey taper: flat 1 in the middle, Hann-shaped flanks covering a fraction `alpha` of n. */
export function tukey(n: number, alpha = 0.5): Float64Array {
  const w = new Float64Array(n).fill(1);
  if (alpha <= 0 || n < 2) return w;
  const edge = alpha * (n - 1) / 2;
  for (let i = 0; i < n; i++) {
    const d = Math.min(i, n - 1 - i);
    if (d < edge) w[i] = 0.5 * (1 - Math.cos(Math.PI * d / edge));
  }
  return w;
}

/** Separable 2D taper (w × h) from a 1D taper function. */
export function taper2d(w: number, h: number, kind: "hann" | "tukey" = "hann", alpha = 0.5): Float64Array {
  const tx = kind === "hann" ? hann(w) : tukey(w, alpha), ty = kind === "hann" ? hann(h) : tukey(h, alpha);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = ty[y] * tx[x];
  return out;
}

/** Separable 3D Hann taper. */
export function hann3d(nx: number, ny: number, nz: number): Float64Array {
  const a = hann(nx), b = hann(ny), c = hann(nz), out = new Float64Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) out[(z * ny + y) * nx + x] = c[z] * b[y] * a[x];
  return out;
}

/** Mean of an array. */
export function mean(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return a.length ? s / a.length : 0;
}

/** Prepare an image for the FFT: optionally subtract its mean, then multiply by a taper. */
export function prepare(a: ArrayLike<number>, opts: { meanFree?: boolean; taper?: Float64Array | null } = {}): Float64Array {
  const m = opts.meanFree ? mean(a) : 0, out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - m) * (opts.taper ? opts.taper[i] : 1);
  return out;
}

// ---------------------------------------------------------------------------------------
// 1D scores (brute force), as on the sliding picture

export type Score = "ssd" | "cc" | "ncc";

export interface Overlap1D { n: number; ssd: number; cc: number; ncc: number; ops: number }

/**
 * Scores of placing B (length nb) at offset t on A (length na): pairs A(x + t), B(x) for the
 * x where both exist. ssd and cc are SUMS over the overlap; ncc is the correlation
 * coefficient on the overlap (NaN if n < 2 or a side is flat). `circular`: indices of A
 * wrap (needs na = nb) and the overlap is always complete. ops = multiply-adds used.
 */
export function overlapScores1d(a: ArrayLike<number>, b: ArrayLike<number>, t: number, circular = false): Overlap1D {
  const na = a.length, nb = b.length;
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, ssd = 0;
  for (let x = 0; x < nb; x++) {
    let i = x + t;
    if (circular) i = wrapIndex(i, na);
    else if (i < 0 || i >= na) continue;
    const u = a[i], v = b[x];
    n++; sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v; ssd += (u - v) * (u - v);
  }
  let ncc = NaN;
  if (n >= 2) {
    const va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    if (va > 1e-12 * (saa + 1e-300) && vb > 1e-12 * (sbb + 1e-300)) ncc = (sab - sa * sb / n) / Math.sqrt(va * vb);
  }
  return { n, ssd, cc: sab, ncc, ops: n };
}

/** All shifts t in [tMin, tMax] (linear) or 0..n−1 (circular), brute force, with the total multiply-add count. */
export function bruteForce1d(a: ArrayLike<number>, b: ArrayLike<number>, opts: { circular?: boolean; tMin?: number; tMax?: number } = {}):
  { t: number[]; ssd: Float64Array; cc: Float64Array; ncc: Float64Array; n: Int32Array; ops: number } {
  const circ = !!opts.circular;
  const t0 = circ ? 0 : opts.tMin ?? -(b.length - 1), t1 = circ ? a.length - 1 : opts.tMax ?? a.length - 1;
  const m = t1 - t0 + 1, ts: number[] = [];
  const ssd = new Float64Array(m), cc = new Float64Array(m), ncc = new Float64Array(m), n = new Int32Array(m);
  let ops = 0;
  for (let k = 0; k < m; k++) {
    const r = overlapScores1d(a, b, t0 + k, circ);
    ts.push(t0 + k); ssd[k] = r.ssd; cc[k] = r.cc; ncc[k] = r.ncc; n[k] = r.n; ops += r.ops;
  }
  return { t: ts, ssd, cc, ncc, n, ops };
}

// ---------------------------------------------------------------------------------------
// FFT correlation, 1D

/** DFT of a real 1D signal (FFT for powers of two, else the O(n²) reference). */
export function dftReal(a: ArrayLike<number>): Complex {
  const n = a.length;
  if (isPow2(n)) { const re = Float64Array.from(a), im = new Float64Array(n); fft1d(re, im); return { re, im }; }
  return dft1d(a, new Float64Array(n));
}

/** Inverse DFT (divided by n), real part. */
export function idftReal(X: Complex): Float64Array {
  const n = X.re.length;
  let re: Float64Array, im: Float64Array;
  if (isPow2(n)) { re = Float64Array.from(X.re); im = Float64Array.from(X.im); fft1d(re, im, true); }
  else ({ re, im } = dft1d(X.re, X.im, true));
  for (let i = 0; i < n; i++) re[i] /= n;
  return re;
}

export interface SpectrumOptions {
  /** Exponent of the normalisation: 0 = plain cross-correlation, 1 = phase correlation. */
  p?: number;
  /** Regularisation, relative to the mean magnitude of the cross-power (default 1e-3). */
  eps?: number;
  /** Gaussian low-pass weight, sd in cycles per pixel (0 or undefined: none). */
  lowpass?: number;
}

/**
 * Normalised cross-power spectrum R_k = P_k / (|P_k| + ε·mean|P|)^p, P = A·conj(B), times an
 * optional Gaussian low-pass weight. `fx(k)`, `fy(k)` give the frequency in cycles/pixel.
 */
export function crossPower(A: Complex, B: Complex, weightAt: ((k: number) => number) | null, o: SpectrumOptions = {}): Complex {
  const n = A.re.length, re = new Float64Array(n), im = new Float64Array(n), mag = new Float64Array(n);
  const p = o.p ?? 0;
  let mm = 0;
  for (let k = 0; k < n; k++) {
    re[k] = A.re[k] * B.re[k] + A.im[k] * B.im[k];
    im[k] = A.im[k] * B.re[k] - A.re[k] * B.im[k];
    mag[k] = Math.hypot(re[k], im[k]); mm += mag[k];
  }
  mm /= n;
  const eps = (o.eps ?? 1e-3) * mm;
  for (let k = 0; k < n; k++) {
    let s = p === 0 ? 1 : 1 / Math.pow(mag[k] + eps, p);
    if (!Number.isFinite(s)) s = 0;
    if (weightAt) s *= weightAt(k);
    re[k] *= s; im[k] *= s;
  }
  return { re, im };
}

/** 1D correlation c(t) = Σ a(x+t) b(x) by the DFT (circular over the common length). */
export function correlate1d(a: ArrayLike<number>, b: ArrayLike<number>, o: SpectrumOptions & { pad?: boolean } = {}):
  { c: Float64Array; A: Complex; B: Complex; R: Complex; n: number } {
  const L = Math.max(a.length, b.length);
  const n = o.pad ? 2 * L : L;
  const pa = new Float64Array(n), pb = new Float64Array(n);
  pa.set(Array.from(a)); pb.set(Array.from(b));
  const A = dftReal(pa), B = dftReal(pb);
  const lp = o.lowpass;
  const R = crossPower(A, B, lp ? (k => Math.exp(-((freq(k, n) / n) ** 2) / (2 * lp * lp))) : null, o);
  return { c: idftReal(R), A, B, R, n };
}

// ---------------------------------------------------------------------------------------
// FFT correlation, 2D

export interface Corr2DOptions extends SpectrumOptions {
  /** Subtract each image's mean first (default true). */
  meanFree?: boolean;
  /** Multiply both by a Hann taper before the FFT. */
  taper?: boolean;
  /** Zero-pad to 2w × 2h: linear instead of circular correlation. */
  pad?: boolean;
}

export interface Corr2D extends Surface { A: Complex; B: Complex; R: Complex; a: Float64Array; b: Float64Array }

/** Forward 2D FFT of a real w × h image (powers of two). */
export function fftReal(a: ArrayLike<number>, w: number, h: number): Complex {
  const re = Float64Array.from(a), im = new Float64Array(w * h);
  fft2dRect(re, im, w, h);
  return { re, im };
}

/** Inverse 2D FFT (divided by w·h), real part. */
export function ifftReal(X: Complex, w: number, h: number): Float64Array {
  const re = Float64Array.from(X.re), im = Float64Array.from(X.im);
  fft2dRect(re, im, w, h, true);
  return re;
}

/** Gaussian low-pass weight on a w × h FFT grid, sd in cycles/pixel. */
export function lowpassWeight(w: number, h: number, sd: number): (k: number) => number {
  return k => {
    const fx = freq(k % w, w) / w, fy = freq(Math.floor(k / w), h) / h;
    return Math.exp(-(fx * fx + fy * fy) / (2 * sd * sd));
  };
}

/**
 * Correlation surface of two w × h images (powers of two): c(t) = Σ a(x+t) b(x), by the FFT.
 * p = 0: cross-correlation; p = 1: phase correlation (a pure circular shift gives a peak of
 * height 1 at the shift). Returns the surface (2w × 2h with `pad`) and the spectra used.
 */
export function correlate2d(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, o: Corr2DOptions = {}): Corr2D {
  const tap = o.taper ? taper2d(w, h) : null;
  const pa = prepare(a, { meanFree: o.meanFree ?? true, taper: tap }), pb = prepare(b, { meanFree: o.meanFree ?? true, taper: tap });
  const W = o.pad ? 2 * w : w, H = o.pad ? 2 * h : h;
  const A = fftReal(o.pad ? padCorner(pa, w, h, W, H) : pa, W, H), B = fftReal(o.pad ? padCorner(pb, w, h, W, H) : pb, W, H);
  const R = crossPower(A, B, o.lowpass ? lowpassWeight(W, H, o.lowpass) : null, o);
  return { data: ifftReal(R, W, H), w: W, h: H, A, B, R, a: pa, b: pb };
}

function padCorner(a: ArrayLike<number>, w: number, h: number, W: number, H: number): Float64Array {
  const out = new Float64Array(W * H);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * W + x] = a[y * w + x];
  return out;
}

/** Value of a surface at a signed shift (tx, ty). */
export const at = (s: Surface, tx: number, ty: number): number => s.data[wrapIndex(ty, s.h) * s.w + wrapIndex(tx, s.w)];

/** Brute-force circular correlation of two w × h images (reference for the tests; O(N²)). */
export function bruteCircular2d(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let ty = 0; ty < h; ty++) for (let tx = 0; tx < w; tx++) {
    let s = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) s += a[((y + ty) % h) * w + (x + tx) % w] * b[y * w + x];
    out[ty * w + tx] = s;
  }
  return out;
}

/**
 * Scores at one shift for two w × h images (linear: only the overlap counts). Returns the
 * overlap size n, SSD and CC as sums, NCC on the overlap. ops = multiply-adds.
 */
export function overlapScores2d(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, tx: number, ty: number): Overlap1D {
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, ssd = 0;
  const x0 = Math.max(0, -tx), x1 = Math.min(w, w - tx), y0 = Math.max(0, -ty), y1 = Math.min(h, h - ty);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const u = a[(y + ty) * w + x + tx], v = b[y * w + x];
    n++; sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v; ssd += (u - v) * (u - v);
  }
  let ncc = NaN;
  if (n >= 2) {
    const va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    if (va > 1e-12 * (saa + 1e-300) && vb > 1e-12 * (sbb + 1e-300)) ncc = (sab - sa * sb / n) / Math.sqrt(va * vb);
  }
  return { n, ssd, cc: sab, ncc, ops: n };
}

/** Brute-force NCC over a square search window |tx|, |ty| ≤ r; returns the best shift and the op count. */
export function bruteForceNcc2d(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, r: number, minOverlap = 1):
  { tx: number; ty: number; ncc: number; ops: number } {
  let best = { tx: 0, ty: 0, ncc: -Infinity }, ops = 0;
  for (let ty = -r; ty <= r; ty++) for (let tx = -r; tx <= r; tx++) {
    const s = overlapScores2d(a, b, w, h, tx, ty);
    ops += s.n;
    if (s.n >= minOverlap && s.ncc > best.ncc) best = { tx, ty, ncc: s.ncc };
  }
  return { ...best, ops };
}

/** Rough flop counts used on the slides: one complex FFT of M points ≈ 5 M log2 M. */
export const fftFlops = (M: number): number => 5 * M * Math.log2(M);

// ---------------------------------------------------------------------------------------
// Masked NCC by FFT (overlap NCC for every shift; template matching)

/**
 * NCC of A(x + t) and B(x) over their overlap, for every shift t, by six FFT correlations
 * (local sums of A, A², B, B², AB and the overlap size). a is wa × ha, b is wb × hb; the
 * result is a P × Q surface (powers of two, P ≥ wa + wb, Q ≥ ha + hb, so no wrap-around):
 * entry at signed shift t = NCC, NaN where the overlap has fewer than `minOverlap` pixels or
 * a side is flat. Also returns the overlap size per shift.
 */
export function maskedNcc(a: ArrayLike<number>, wa: number, ha: number, b: ArrayLike<number>, wb: number, hb: number, minOverlap = 2):
  Surface & { n: Float64Array } {
  let P = 1, Q = 1;
  while (P < wa + wb) P <<= 1;
  while (Q < ha + hb) Q <<= 1;
  const ma = new Float64Array(wa * ha).fill(1), mb = new Float64Array(wb * hb).fill(1);
  const a2 = Float64Array.from(a, v => v * v), b2 = Float64Array.from(b, v => v * v);
  const F = (x: ArrayLike<number>, w: number, h: number) => fftReal(padCorner(x, w, h, P, Q), P, Q);
  const FA = F(a, wa, ha), FA2 = F(a2, wa, ha), FMa = F(ma, wa, ha);
  const FB = F(b, wb, hb), FB2 = F(b2, wb, hb), FMb = F(mb, wb, hb);
  const corr = (X: Complex, Y: Complex) => ifftReal(crossPower(X, Y, null, { p: 0 }), P, Q);
  const sab = corr(FA, FB), sa = corr(FA, FMb), saa = corr(FA2, FMb), sb = corr(FMa, FB), sbb = corr(FMa, FB2), n = corr(FMa, FMb);
  const out = new Float64Array(P * Q);
  for (let i = 0; i < P * Q; i++) {
    const m = Math.round(n[i]);
    n[i] = m;
    if (m < Math.max(2, minOverlap)) { out[i] = NaN; continue; }
    const va = saa[i] - sa[i] * sa[i] / m, vb = sbb[i] - sb[i] * sb[i] / m;
    const scale = Math.max(Math.abs(saa[i]), Math.abs(sbb[i]), 1e-300);
    out[i] = va > 1e-9 * scale && vb > 1e-9 * scale ? (sab[i] - sa[i] * sb[i] / m) / Math.sqrt(va * vb) : NaN;
  }
  return { data: out, w: P, h: Q, n };
}

/**
 * Template matching: NCC of a template T (tw × th) placed with its top-left corner at every
 * position (u, v) of an image I (W × H, powers of two) where it fits completely. The image
 * spectra are computed once; `match(T)` costs two FFTs. Output: W × H map, NaN where the
 * template does not fit.
 */
export class TemplateMatcher {
  private FI: Complex; private SI: Float64Array; private SII: Float64Array;
  constructor(readonly img: ArrayLike<number>, readonly W: number, readonly H: number, readonly tw: number, readonly th: number) {
    this.FI = fftReal(img, W, H);
    const FI2 = fftReal(Float64Array.from(img, v => v * v), W, H);
    const m = new Float64Array(W * H);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) m[y * W + x] = 1;
    const FM = fftReal(m, W, H);
    this.SI = ifftReal(crossPower(this.FI, FM, null), W, H);
    this.SII = ifftReal(crossPower(FI2, FM, null), W, H);
  }
  /** NCC map for template T (tw × th values). */
  match(T: ArrayLike<number>): Float64Array {
    const { W, H, tw, th } = this, n = tw * th, mt = mean(T);
    let stt = 0;
    const t0 = new Float64Array(W * H);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const v = T[y * tw + x] - mt; t0[y * W + x] = v; stt += v * v; }
    const num = ifftReal(crossPower(this.FI, fftReal(t0, W, H), null), W, H);
    const out = new Float64Array(W * H).fill(NaN);
    for (let v = 0; v + th <= H; v++) for (let u = 0; u + tw <= W; u++) {
      const i = v * W + u, vi = this.SII[i] - this.SI[i] * this.SI[i] / n;
      out[i] = vi > 1e-9 * Math.abs(this.SII[i]) && stt > 0 ? num[i] / Math.sqrt(vi * stt) : NaN;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------------------
// Peaks

export interface Peak { tx: number; ty: number; ix: number; iy: number; value: number }

/**
 * The k highest local peaks of a surface (circular), each at least `minDist` (Chebyshev, in
 * pixels, circular) from the higher ones. NaN entries are skipped. Shifts are signed.
 */
export function findPeaks(s: Surface, k = 2, minDist = 3): Peak[] {
  const { w, h, data } = s, peaks: Peak[] = [];
  const idx = Array.from({ length: w * h }, (_, i) => i).filter(i => Number.isFinite(data[i]));
  idx.sort((i, j) => data[j] - data[i]);
  const cd = (a: number, b: number, n: number) => { const d = Math.abs(a - b) % n; return Math.min(d, n - d); };
  for (const i of idx) {
    const ix = i % w, iy = (i - ix) / w;
    if (peaks.some(p => Math.max(cd(p.ix, ix, w), cd(p.iy, iy, h)) < minDist)) continue;
    peaks.push({ ix, iy, tx: signed(ix, w), ty: signed(iy, h), value: data[i] });
    if (peaks.length >= k) break;
  }
  return peaks;
}

/** Highest peak / second-highest peak (at least minDist away). Large = trustworthy. */
export function peakRatio(s: Surface, minDist = 3): number {
  const p = findPeaks(s, 2, minDist);
  if (p.length < 2) return Infinity;
  return p[1].value > 0 ? p[0].value / p[1].value : Infinity;
}

export interface Candidate { tx: number; ty: number; overlap: number; ncc: number }

/**
 * The four shifts a circular peak at index (ix, iy) of a w × h surface can mean:
 * tx ∈ {ix, ix − w}, ty ∈ {iy, iy − h}, each scored by NCC on the actual overlap of a and b
 * (both w × h). Candidates with fewer than minOverlap pixels get ncc = NaN. Sorted best first.
 */
export function fourCandidates(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, ix: number, iy: number, minOverlap = 1): Candidate[] {
  const out: Candidate[] = [];
  for (const ty of [iy, iy - h]) for (const tx of [ix, ix - w]) {
    if (Math.abs(tx) >= w || Math.abs(ty) >= h) continue;
    const s = overlapScores2d(a, b, w, h, tx, ty);
    out.push({ tx, ty, overlap: s.n, ncc: s.n >= minOverlap ? s.ncc : NaN });
  }
  const key = (c: Candidate) => (Number.isFinite(c.ncc) ? c.ncc : -Infinity);
  return out.sort((p, q) => key(q) - key(p));
}

// ---------------------------------------------------------------------------------------
// Sub-pixel

/** Vertex of the parabola through (−1, cm), (0, c0), (1, cp): offset from the middle sample. */
export function parabolaOffset(cm: number, c0: number, cp: number): number {
  const d = cm - 2 * c0 + cp;
  return d < 0 ? Math.max(-0.5, Math.min(0.5, (cm - cp) / (2 * d))) : 0;
}

/** Centroid of three samples (negative values clipped to 0): offset from the middle. */
export function centroidOffset(cm: number, c0: number, cp: number): number {
  const a = Math.max(0, cm), b = Math.max(0, c0), c = Math.max(0, cp), s = a + b + c;
  return s > 0 ? (c - a) / s : 0;
}

/**
 * Continuous inverse DFT of a spectrum R (length n) at real position x:
 * c(x) = (1/n) Σ_k Re(R_k e^{2πi k x / n}), with signed k (the Nyquist term as a cosine).
 * At integer x this equals the inverse DFT. This is the "upsampled DFT".
 */
export function idftAt(R: Complex, x: number): number {
  const n = R.re.length;
  let s = 0;
  for (let k = 0; k < n; k++) {
    const f = freq(k, n);
    if (2 * k === n) { s += R.re[k] * Math.cos(Math.PI * x); continue; }
    const a = 2 * Math.PI * f * x / n;
    s += R.re[k] * Math.cos(a) - R.im[k] * Math.sin(a);
  }
  return s / n;
}

/**
 * Sub-pixel peak by the upsampled DFT (Guizar-Sicairos et al. 2008), 1D: evaluate c(x) on a
 * grid of spacing 1/U within ±0.75 px of the integer peak t0 and return the best position.
 */
export function upsampledPeak1d(R: Complex, t0: number, U = 20): { x: number; xs: number[]; cs: number[] } {
  const xs: number[] = [], cs: number[] = [];
  const m = Math.ceil(0.75 * U);
  let best = t0, bv = -Infinity;
  for (let j = -m; j <= m; j++) {
    const x = t0 + j / U, c = idftAt(R, x);
    xs.push(x); cs.push(c);
    if (c > bv) { bv = c; best = x; }
  }
  return { x: best, xs, cs };
}

/**
 * 2D upsampled DFT around the integer peak (tx0, ty0) of the inverse transform of R (w × h):
 * c(x, y) on a (2m+1)² grid of spacing 1/U (m = ceil(0.75 U)), computed as E_y R E_xᵀ with
 * small DFT matrices (cost ≈ w·h·(2m+1)). Returns the best sub-pixel shift.
 */
export function upsampledPeak2d(R: Complex, w: number, h: number, tx0: number, ty0: number, U = 20): { tx: number; ty: number; value: number } {
  const m = Math.ceil(0.75 * U), g = 2 * m + 1;
  const xs = Array.from({ length: g }, (_, j) => tx0 + (j - m) / U), ys = Array.from({ length: g }, (_, j) => ty0 + (j - m) / U);
  // T[ky][jx] = Σ_kx R[ky,kx] e^{2πi fx x_j / w}
  const Tr = new Float64Array(h * g), Ti = new Float64Array(h * g);
  const cx = new Float64Array(w * g), sx = new Float64Array(w * g);
  for (let kx = 0; kx < w; kx++) for (let j = 0; j < g; j++) {
    const a = 2 * Math.PI * freq(kx, w) * xs[j] / w;
    cx[kx * g + j] = Math.cos(a); sx[kx * g + j] = Math.sin(a);
  }
  for (let ky = 0; ky < h; ky++) for (let kx = 0; kx < w; kx++) {
    const rr = R.re[ky * w + kx], ri = R.im[ky * w + kx];
    if (rr === 0 && ri === 0) continue;
    for (let j = 0; j < g; j++) {
      const c = cx[kx * g + j], s = sx[kx * g + j];
      Tr[ky * g + j] += rr * c - ri * s; Ti[ky * g + j] += rr * s + ri * c;
    }
  }
  let best = { tx: tx0, ty: ty0, value: -Infinity };
  for (let jy = 0; jy < g; jy++) {
    const cy = new Float64Array(h), sy = new Float64Array(h);
    for (let ky = 0; ky < h; ky++) { const a = 2 * Math.PI * freq(ky, h) * ys[jy] / h; cy[ky] = Math.cos(a); sy[ky] = Math.sin(a); }
    for (let jx = 0; jx < g; jx++) {
      let s = 0;
      for (let ky = 0; ky < h; ky++) s += Tr[ky * g + jx] * cy[ky] - Ti[ky * g + jx] * sy[ky];
      s /= w * h;
      if (s > best.value) best = { tx: xs[jx], ty: ys[jy], value: s };
    }
  }
  return best;
}

/** Sub-pixel refinement of an integer peak of a surface by a parabola in x and in y separately. */
export function parabolaPeak2d(s: Surface, ix: number, iy: number): { tx: number; ty: number } {
  const v = (dx: number, dy: number) => s.data[wrapIndex(iy + dy, s.h) * s.w + wrapIndex(ix + dx, s.w)];
  return { tx: signed(ix, s.w) + parabolaOffset(v(-1, 0), v(0, 0), v(1, 0)), ty: signed(iy, s.h) + parabolaOffset(v(0, -1), v(0, 0), v(0, 1)) };
}

// ---------------------------------------------------------------------------------------
// Fourier shifts (the shift theorem used backwards)

/**
 * Shift a real 1D signal by s samples (any real s) with the phase ramp: out(x) = in(x − s),
 * circularly, band-limited interpolation. (The Nyquist term is multiplied by cos(πs).)
 */
export function fourierShift1d(a: ArrayLike<number>, s: number): Float64Array {
  const n = a.length, X = dftReal(a);
  for (let k = 0; k < n; k++) {
    if (2 * k === n) { X.re[k] *= Math.cos(Math.PI * s); X.im[k] *= Math.cos(Math.PI * s); continue; }
    const ang = -2 * Math.PI * freq(k, n) * s / n, c = Math.cos(ang), sn = Math.sin(ang);
    const r = X.re[k] * c - X.im[k] * sn, i = X.re[k] * sn + X.im[k] * c;
    X.re[k] = r; X.im[k] = i;
  }
  return idftReal(X);
}

/** 2D phase-ramp shift of a w × h image (powers of two): out(x, y) = in(x − sx, y − sy), circularly. */
export function fourierShift2d(a: ArrayLike<number>, w: number, h: number, sx: number, sy: number): Float64Array {
  const X = fftReal(a, w, h);
  for (let ky = 0; ky < h; ky++) for (let kx = 0; kx < w; kx++) {
    const k = ky * w + kx;
    const nyx = 2 * kx === w, nyy = 2 * ky === h;
    const ax = nyx ? 0 : -2 * Math.PI * freq(kx, w) * sx / w, ay = nyy ? 0 : -2 * Math.PI * freq(ky, h) * sy / h;
    const amp = (nyx ? Math.cos(Math.PI * sx) : 1) * (nyy ? Math.cos(Math.PI * sy) : 1);
    const c = amp * Math.cos(ax + ay), s = amp * Math.sin(ax + ay);
    const r = X.re[k] * c - X.im[k] * s, i = X.re[k] * s + X.im[k] * c;
    X.re[k] = r; X.im[k] = i;
  }
  return ifftReal(X, w, h);
}

// ---------------------------------------------------------------------------------------
// Global alignment of a mosaic

/** A measured pairwise shift: position of tile j minus position of tile i, with a weight. */
export interface TileEdge { i: number; j: number; dx: number; dy: number; w: number }

/**
 * Tile positions that best agree with the pairwise shifts, by weighted least squares:
 * minimise Σ w (x_j − x_i − dx)² + (same in y), with tile `fixed` at (0, 0). The normal
 * equations are L x = b with L the weighted graph Laplacian (pinned row/column removed).
 * Returns null when the graph is not connected. Residuals per edge: (x_j − x_i − dx, …).
 */
export function solveGlobal(n: number, edges: TileEdge[], fixed = 0):
  { x: Float64Array; y: Float64Array; rx: number[]; ry: number[]; L: Float64Array } | null {
  const L = new Float64Array(n * n), bx = new Float64Array(n), by = new Float64Array(n);
  for (const e of edges) {
    L[e.i * n + e.i] += e.w; L[e.j * n + e.j] += e.w; L[e.i * n + e.j] -= e.w; L[e.j * n + e.i] -= e.w;
    bx[e.j] += e.w * e.dx; bx[e.i] -= e.w * e.dx; by[e.j] += e.w * e.dy; by[e.i] -= e.w * e.dy;
  }
  const keep = Array.from({ length: n }, (_, k) => k).filter(k => k !== fixed), m = n - 1;
  const Lr = new Float64Array(m * m), rx = new Float64Array(m), ry = new Float64Array(m);
  keep.forEach((p, a) => { rx[a] = bx[p]; ry[a] = by[p]; keep.forEach((q, b) => { Lr[a * m + b] = L[p * n + q]; }); });
  if (!connected(n, edges)) return null;
  const sx = m ? solve(Lr, rx, m) : new Float64Array(0), sy = m ? solve(Lr, ry, m) : new Float64Array(0);
  if (!sx || !sy) return null;
  const x = new Float64Array(n), y = new Float64Array(n);
  keep.forEach((p, a) => { x[p] = sx[a]; y[p] = sy[a]; });
  return { x, y, rx: edges.map(e => x[e.j] - x[e.i] - e.dx), ry: edges.map(e => y[e.j] - y[e.i] - e.dy), L };
}

function connected(n: number, edges: TileEdge[]): boolean {
  const seen = new Uint8Array(n), stack = [0];
  seen[0] = 1;
  while (stack.length) {
    const u = stack.pop()!;
    for (const e of edges) {
      const v = e.i === u ? e.j : e.j === u ? e.i : -1;
      if (v >= 0 && !seen[v]) { seen[v] = 1; stack.push(v); }
    }
  }
  return seen.every(s => s === 1);
}

/** Maximum spanning tree by weight (Prim), returned as a list of edge indices. */
export function maxSpanningTree(n: number, edges: TileEdge[]): number[] {
  const inTree = new Uint8Array(n), out: number[] = [];
  inTree[0] = 1;
  for (let step = 1; step < n; step++) {
    let best = -1;
    edges.forEach((e, k) => {
      if (inTree[e.i] === inTree[e.j]) return;
      if (best < 0 || e.w > edges[best].w) best = k;
    });
    if (best < 0) break;
    out.push(best);
    inTree[edges[best].i] = 1; inTree[edges[best].j] = 1;
  }
  return out;
}

/** Positions by chaining the shifts along a tree (a list of edge indices) from tile `root` at (0, 0). */
export function chainPositions(n: number, edges: TileEdge[], tree: number[], root = 0): { x: Float64Array; y: Float64Array } {
  const x = new Float64Array(n).fill(NaN), y = new Float64Array(n).fill(NaN);
  x[root] = 0; y[root] = 0;
  let changed = true;
  while (changed) {
    changed = false;
    for (const k of tree) {
      const e = edges[k];
      if (Number.isFinite(x[e.i]) && !Number.isFinite(x[e.j])) { x[e.j] = x[e.i] + e.dx; y[e.j] = y[e.i] + e.dy; changed = true; }
      else if (Number.isFinite(x[e.j]) && !Number.isFinite(x[e.i])) { x[e.i] = x[e.j] - e.dx; y[e.i] = y[e.j] - e.dy; changed = true; }
    }
  }
  return { x, y };
}

// ---------------------------------------------------------------------------------------
// Rotation and scale as shifts: polar and log-polar resampling, rotational correlation,
// Fourier–Mellin registration (added by the shape-matching day, unit fourier-mellin/; built
// on lib/polar.ts). Angles in radians, positive = clockwise on screen (x right, y down), as
// everywhere in the course.

/** Rotation (about the image centre), scale and shift of a similarity transform. */
export interface Similarity { angle: number; scale: number; tx: number; ty: number }

/**
 * A w × h view of `src` rotated by `angle` about the centre, scaled by `scale`, then shifted
 * by (tx, ty), by inverse mapping with bilinear lookup:
 *   out(x) = src(cs + R(−angle)(x − c − t) / scale),   c = (w/2, h/2),
 * cs = (cx, cy) the matching point of the source (default c). Outside the source → `border`.
 */
export function warpSimilarity(src: Grey, w: number, h: number, T: Similarity,
  o: { cx?: number; cy?: number; border?: Border } = {}): Float64Array {
  const out = new Float64Array(w * h), c = Math.cos(T.angle), s = Math.sin(T.angle);
  const ox = w / 2, oy = h / 2, cx = o.cx ?? ox, cy = o.cy ?? oy, border = o.border ?? "zero";
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = (x - ox - T.tx) / T.scale, dy = (y - oy - T.ty) / T.scale;
    // R(−α) = [[cos α, sin α], [−sin α, cos α]] (R(α) turns clockwise on screen)
    out[y * w + x] = sampleBilinear(src, cx + c * dx + s * dy, cy - s * dx + c * dy, border);
  }
  return out;
}

/** A Grey view of a flat array (no copy when it is already a Float32Array). */
function asGrey(a: ArrayLike<number>, w: number, h: number): Grey {
  return { width: w, height: h, data: a instanceof Float32Array ? a : Float32Array.from(a) };
}

/**
 * The high-pass weight of Reddy & Chatterji (1996) on a centred w × h frequency grid:
 * (1 − X)(2 − X) with X = cos(π fx) cos(π fy), f in cycles per pixel. 0 at frequency 0,
 * rising like f² near it; damps the strong low frequencies, which carry little angle.
 */
export function highpassWeight(w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = Math.cos(Math.PI * (x - w / 2) / w) * Math.cos(Math.PI * (y - h / 2) / h);
    out[y * w + x] = (1 - X) * (2 - X);
  }
  return out;
}

/**
 * Magnitude spectrum |F| of a real w × h image (powers of two), centred: frequency 0 at
 * (w/2, h/2). The mean is removed first; optional Hann taper and high-pass weight.
 */
export function centredMagnitude(a: ArrayLike<number>, w: number, h: number,
  o: { taper?: boolean; highpass?: boolean } = {}): Float64Array {
  const p = prepare(a, { meanFree: true, taper: o.taper ? taper2d(w, h) : null });
  const F = fftReal(p, w, h), out = new Float64Array(w * h), hp = o.highpass ? highpassWeight(w, h) : null;
  for (let ky = 0; ky < h; ky++) for (let kx = 0; kx < w; kx++) {
    const i = ((ky + h / 2) % h) * w + (kx + w / 2) % w, k = ky * w + kx;
    out[i] = Math.hypot(F.re[k], F.im[k]) * (hp ? hp[i] : 1);
  }
  return out;
}

/** Grid of a log-polar spectrum: nTheta columns over HALF a turn, nR log-spaced radii rMin…rMax. */
export interface LogPolarGrid { nTheta: number; nR: number; rMin: number; rMax: number }

/** Size of one log-radius row: ln(rMax / rMin) / (nR − 1). A scale s moves rows by ln(s) / step. */
export const logStep = (g: LogPolarGrid): number => Math.log(g.rMax / g.rMin) / (g.nR - 1);

/**
 * Log-polar resampling of a centred spectrum (w × h, frequency 0 at (w/2, h/2)). |F| of a real
 * image is point-symmetric, so half a turn is enough: column j ↔ θ = π j / nTheta (clockwise
 * from the +x axis), row i ↔ r = rMin (rMax / rMin)^(i / (nR − 1)). Bilinear (toPolar of
 * lib/polar.ts on a full turn of 2·nTheta columns, of which the first half is kept).
 */
export function logPolarSpectrum(mag: ArrayLike<number>, w: number, h: number, g: LogPolarGrid): Float64Array {
  const full = toPolar(asGrey(mag, w, h), { cx: w / 2, cy: h / 2, rMin: g.rMin, rMax: g.rMax, nR: g.nR, nTheta: 2 * g.nTheta, log: true });
  const out = new Float64Array(g.nTheta * g.nR);
  for (let i = 0; i < g.nR; i++) for (let j = 0; j < g.nTheta; j++) out[i * g.nTheta + j] = full.data[i * 2 * g.nTheta + j];
  return out;
}

/**
 * Rotational correlation of two polar images (nTheta columns per row, nTheta a power of two;
 * one row per ring): score(t) = Σ_rings weight_i Σ_j A(i, j) B(i, j + t), circular in j: every
 * angle at once, one FFT along θ per ring (the inverse FFT of Σ_i w_i conj(A_i,m) B_i,m). The
 * peak is at t = the rotation of B relative to A, in columns (clockwise positive). Options:
 * remove each ring's mean first (default true), weights per ring (e.g. the radius, for an
 * area-weighted sum), and `phase` (normalise the summed cross-power: a sharper peak).
 */
export function rotationalCorrelation(pa: ArrayLike<number>, pb: ArrayLike<number>, nTheta: number, nRings: number,
  o: { meanFree?: boolean; weights?: ArrayLike<number>; phase?: boolean } = {}): Float64Array {
  const re = new Float64Array(nTheta), im = new Float64Array(nTheta);
  for (let i = 0; i < nRings; i++) {
    const ar = new Float64Array(nTheta), br = new Float64Array(nTheta);
    let ma = 0, mb = 0;
    for (let j = 0; j < nTheta; j++) { ar[j] = pa[i * nTheta + j]; br[j] = pb[i * nTheta + j]; ma += ar[j]; mb += br[j]; }
    if (o.meanFree ?? true) for (let j = 0; j < nTheta; j++) { ar[j] -= ma / nTheta; br[j] -= mb / nTheta; }
    const A = dftReal(ar), B = dftReal(br), wi = o.weights ? o.weights[i] : 1;
    for (let m = 0; m < nTheta; m++) {      // conj(A) · B
      re[m] += wi * (A.re[m] * B.re[m] + A.im[m] * B.im[m]);
      im[m] += wi * (A.re[m] * B.im[m] - A.im[m] * B.re[m]);
    }
  }
  if (o.phase) {
    let mm = 0;
    for (let m = 0; m < nTheta; m++) mm += Math.hypot(re[m], im[m]) / nTheta;
    for (let m = 0; m < nTheta; m++) { const s = 1 / (Math.hypot(re[m], im[m]) + 1e-6 * mm); re[m] *= s; im[m] *= s; }
  }
  return idftReal({ re, im });
}

export interface FMOptions {
  taper?: boolean;      // Hann taper on both images before |F| (default true)
  highpass?: boolean;   // Reddy–Chatterji high-pass on |F| (default true)
  both?: boolean;       // test θ and θ + 180° by the translation peak (default true)
  grid?: Partial<LogPolarGrid>;   // default nTheta 256, nR 128, rMin 2, rMax w/2 − 2
}

/** One candidate angle with its translation step. */
export interface FMCandidate {
  angle: number;            // radians, clockwise positive
  tx: number; ty: number;   // the shift of the similarity B = T(A) (see warpSimilarity)
  height: number; ratio: number;  // of the translation phase-correlation peak
  unwarped: Float64Array;   // B with rotation and scale undone (≈ A shifted)
  surface: Surface;         // the translation phase-correlation surface
}

export interface FMResult {
  grid: LogPolarGrid;
  magA: Float64Array; magB: Float64Array;   // centred |F| (after taper / high-pass)
  lpA: Float64Array; lpB: Float64Array;     // log-polar, nTheta × nR
  surface: Surface;          // phase correlation in (θ, log r): index (θ shift mod nTheta, row shift mod nR)
  thetaBins: number; rhoBins: number;      // sub-pixel peak (signed bins; rhoBins = −ln(s) / step)
  peakHeight: number; peakRatio: number;
  angle: number;             // from the peak, in (−90°, 90°]
  scale: number;
  candidates: FMCandidate[]; // angle and angle + 180° (only the first with both = false)
  best: FMCandidate;         // the candidate with the higher translation peak
}

/**
 * Fourier–Mellin registration (Reddy & Chatterji 1996) of two w × h images (powers of two),
 * B ≈ A rotated by α, scaled by s and shifted by t (as warpSimilarity). Steps: |F| of both
 * (translation drops out), log-polar resampling (rotation → shift along θ, scale → shift along
 * log r), phase correlation → α (mod 180°) and s; undo them on B; phase-correlate with A for
 * the shift, for α and α + 180°, and keep the higher peak.
 */
export function fourierMellin(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, o: FMOptions = {}): FMResult {
  const taper = o.taper ?? true, highpass = o.highpass ?? true;
  const grid: LogPolarGrid = { nTheta: 256, nR: 128, rMin: 2, rMax: Math.min(w, h) / 2 - 2, ...o.grid };
  const magA = centredMagnitude(a, w, h, { taper, highpass }), magB = centredMagnitude(b, w, h, { taper, highpass });
  const lpA = logPolarSpectrum(magA, w, h, grid), lpB = logPolarSpectrum(magB, w, h, grid);
  // score(t) = Σ lpB(x + t) lpA(x): peak at t = (α in columns, −ln s / step)
  const surface = correlate2d(lpB, lpA, grid.nTheta, grid.nR, { p: 1, eps: 1e-6 });
  const pk = findPeaks(surface, 1)[0];
  const sub = parabolaPeak2d(surface, pk.ix, pk.iy);
  const angle = sub.tx * Math.PI / grid.nTheta, scale = Math.exp(-sub.ty * logStep(grid));
  const angles = (o.both ?? true) ? [angle, angle + Math.PI] : [angle];
  const candidates = angles.map(al => translationStep(a, b, w, h, al, scale, taper));
  const best = candidates.reduce((p, q) => (q.height > p.height ? q : p));
  return {
    grid, magA, magB, lpA, lpB, surface, thetaBins: sub.tx, rhoBins: sub.ty, peakHeight: pk.value,
    peakRatio: peakRatio(surface), angle, scale, candidates, best,
  };
}

/** Undo rotation `angle` and `scale` on B, phase-correlate with A, convert the shift back. */
function translationStep(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, angle: number, scale: number, taper: boolean): FMCandidate {
  // B'(x) = B(c + s R(α)(x − c)) ≈ A(x − R(−α) t / s)
  const unwarped = warpSimilarity(asGrey(b, w, h), w, h, { angle: -angle, scale: 1 / scale, tx: 0, ty: 0 });
  const surface = correlate2d(a, unwarped, w, h, { p: 1, taper, eps: 1e-6 });
  const pk = findPeaks(surface, 1)[0], sub = parabolaPeak2d(surface, pk.ix, pk.iy);
  // B'(x) = A(x + t') with t' = −R(−α) t / s  ⇒  t = −s R(α) t'
  const c = Math.cos(angle), s = Math.sin(angle);
  const tx = -scale * (c * sub.tx - s * sub.ty), ty = -scale * (s * sub.tx + c * sub.ty);
  return { angle, tx, ty, height: pk.value, ratio: peakRatio(surface), unwarped, surface };
}
