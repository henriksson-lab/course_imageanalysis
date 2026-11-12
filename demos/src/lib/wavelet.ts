// Discrete wavelet transforms for the course demos (owner: wavelets day). No DOM, so it
// runs under node (tests: demos/test/wavelets-test.ts).
//
// Conventions
//  - Orthogonal filters (Haar, Daubechies dbN, symlet sym4) are given by the low-pass h in
//    Daubechies' order (db2 = (1+√3, 3+√3, 3−√3, 1−√3)/(4√2)); the high-pass is the
//    quadrature mirror g[n] = (−1)^n h[L−1−n]. Analysis is a correlation followed by keeping
//    every second sample: a[k] = Σ_n h[n] x[2k+n], d[k] = Σ_n g[n] x[2k+n]
//    (= convolution with the time-reversed filter). Haar: a = (x0 + x1)/√2, d = (x0 − x1)/√2.
//  - Synthesis is the transpose: x[2k+n] += h[n] a[k] + g[n] d[k].
//  - Boundaries: "periodic" (wrap; N/2 coefficients per branch, exactly orthogonal),
//    "symmetric" (half-sample mirror, x[−1] = x[0]) and "zero" (zero padding). The last two
//    keep every coefficient whose filter window touches the signal ((N + L − 2)/2 per
//    branch), which makes the reconstruction exact for any extension.
//  - Biorthogonal CDF 9/7 and LeGall 5/3 are computed by lifting with whole-sample
//    symmetric extension (as in JPEG 2000): N/2 coefficients per branch, exact inverse,
//    scaled by √2 (low) and 1/√2 (high) so they are close to orthonormal. 5/3 also has an
//    integer mode (no scaling, floors inside the lifting steps).
//  - Mallat layout: a multi-level 1D transform of length N is stored as
//    [a_J | d_J | … | d_2 | d_1]; level 1 is the finest. In 2D, one level turns the top-left
//    w × h block into LL (top-left), V (top-right: high-pass along x, i.e. vertical edges),
//    H (bottom-left: high-pass along y, horizontal edges) and D (bottom-right).

import { fft1d } from "./fft";

export type WaveletName = "haar" | "db2" | "db3" | "db4" | "db8" | "sym4" | "cdf97" | "legall53";
export type Boundary = "periodic" | "symmetric" | "zero";

export interface OrthoWavelet {
  kind: "orth";
  name: WaveletName;
  label: string;
  /** Low-pass analysis/synthesis filter (Σh = √2, Σh² = 1). */
  h: Float64Array;
  /** High-pass g[n] = (−1)^n h[L−1−n]. */
  g: Float64Array;
  /** Number of vanishing moments. */
  p: number;
}

export interface LiftWavelet {
  kind: "lift";
  name: WaveletName;
  label: string;
  /** Lifting coefficients: predict/update pairs, applied in order (odd += c·(left + right) for predict). */
  steps: { type: "predict" | "update"; c: number }[];
  /** Final scaling K: low ← low·√2/K, high ← high·K/√2. */
  K: number;
  p: number;
}

export type Wavelet = OrthoWavelet | LiftWavelet;

/** High-pass quadrature mirror filter g[n] = (−1)^n h[L−1−n]. */
export function qmf(h: ArrayLike<number>): Float64Array {
  const L = h.length, g = new Float64Array(L);
  for (let n = 0; n < L; n++) g[n] = (n % 2 ? -1 : 1) * h[L - 1 - n];
  return g;
}

function ortho(name: WaveletName, label: string, h: number[], p: number): OrthoWavelet {
  const hh = Float64Array.from(h);
  return { kind: "orth", name, label, h: hh, g: qmf(hh), p };
}

const S2 = Math.SQRT1_2;
// Computed by spectral factorisation at 50 digits (minimum phase for dbN; for sym4 the root
// choice with the most nearly linear phase). Σh = √2 and Σh² = 1 to machine precision.
export const WAVELETS: Record<WaveletName, Wavelet> = {
  haar: ortho("haar", "Haar", [S2, S2], 1),
  db2: ortho("db2", "db2", [0.48296291314453414, 0.83651630373780791, 0.22414386804201338, -0.12940952255126038], 2),
  db3: ortho("db3", "db3", [0.33267055295008262, 0.80689150931109258, 0.45987750211849157, -0.13501102001025459,
    -0.085441273882026662, 0.035226291885709537], 3),
  db4: ortho("db4", "db4", [0.2303778133088965, 0.71484657055291565, 0.63088076792985891, -0.027983769416859854,
    -0.18703481171909308, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032], 4),
  db8: ortho("db8", "db8", [0.05441584224310401, 0.31287159091429997, 0.67563073629728981, 0.58535468365420671,
    -0.015829105256349306, -0.28401554296154693, 0.00047248457391328277, 0.12874742662047846,
    -0.017369301001807546, -0.044088253930794752, 0.013981027917398282, 0.0087460940474057767,
    -0.0048703529934515743, -0.00039174037337694705, 0.00067544940645056937, -0.00011747678412476953], 8),
  sym4: ortho("sym4", "sym4", [0.032223100604051468, -0.012603967262031304, -0.099219543576633533, 0.29785779560530605,
    0.80373875180513208, 0.49761866763277499, -0.029635527646002492, -0.075765714789502213], 4),
  cdf97: {
    kind: "lift", name: "cdf97", label: "CDF 9/7", p: 4, K: 1.230174104914001,
    steps: [
      { type: "predict", c: -1.586134342059924 }, { type: "update", c: -0.052980118572961 },
      { type: "predict", c: 0.882911075530934 }, { type: "update", c: 0.443506852043971 },
    ],
  },
  legall53: {
    kind: "lift", name: "legall53", label: "LeGall 5/3", p: 2, K: 1,
    steps: [{ type: "predict", c: -0.5 }, { type: "update", c: 0.25 }],
  },
};

/** Look up a wavelet by name (unknown names give Haar). */
export function wavelet(name: string): Wavelet {
  return (WAVELETS as Record<string, Wavelet>)[name] ?? WAVELETS.haar;
}

/** Filter length (lifting wavelets: the analysis low-pass length, 9 or 5). */
export function filterLength(w: Wavelet): number {
  return w.kind === "orth" ? w.h.length : w.name === "cdf97" ? 9 : 5;
}

// ---------------------------------------------------------------------------------------
// Boundary extension

/** Index into [0, n) for an extended position i (symmetric = half-sample mirror, period 2n). */
export function extendIndex(i: number, n: number, mode: Boundary): number {
  if (mode === "periodic") return ((i % n) + n) % n;
  if (mode === "zero") return i >= 0 && i < n ? i : -1;
  const p = 2 * n;
  let j = ((i % p) + p) % p;
  if (j >= n) j = p - 1 - j;
  return j;
}

function at(x: ArrayLike<number>, i: number, n: number, mode: Boundary): number {
  const j = extendIndex(i, n, mode);
  return j < 0 ? 0 : x[j];
}

// ---------------------------------------------------------------------------------------
// One level, orthogonal filters, any boundary (the filter bank)

export interface Step {
  /** Low-pass (approximation) coefficients. */
  a: Float64Array;
  /** High-pass (detail) coefficients. */
  d: Float64Array;
  /** Index k of a[0] and d[0]: coefficient k uses samples 2k … 2k+L−1. */
  k0: number;
  /** Length of the input. */
  n: number;
}

/** Range of coefficient indices for a signal of length n and filter length L. */
export function coeffRange(n: number, L: number, mode: Boundary): { k0: number; count: number } {
  if (mode === "periodic") return { k0: 0, count: Math.ceil(n / 2) };
  const k0 = Math.ceil(-(L - 1) / 2), k1 = Math.floor((n - 1) / 2);
  return { k0, count: k1 - k0 + 1 };
}

/** Analysis: filter with h and g, keep every second sample. */
export function analysisStep(x: ArrayLike<number>, w: OrthoWavelet, mode: Boundary = "periodic"): Step {
  const n = x.length, L = w.h.length, { k0, count } = coeffRange(n, L, mode);
  const a = new Float64Array(count), d = new Float64Array(count);
  for (let j = 0; j < count; j++) {
    const s = 2 * (k0 + j);
    let sa = 0, sd = 0;
    for (let m = 0; m < L; m++) { const v = at(x, s + m, n, mode); sa += w.h[m] * v; sd += w.g[m] * v; }
    a[j] = sa; d[j] = sd;
  }
  return { a, d, k0, n };
}

/** Options to break the synthesis on purpose (filterbank demo). */
export interface SynthesisOptions {
  /** Leave out the high-pass branch (d = 0). */
  dropHigh?: boolean;
  /** Leave out the low-pass branch (a = 0). */
  dropLow?: boolean;
  /** Use the time-reversed filters for synthesis (the wrong ones). */
  wrongFilters?: boolean;
}

/** Synthesis: upsample by 2, filter, add the two branches (the transpose of analysisStep). */
export function synthesisStep(st: Step, w: OrthoWavelet, mode: Boundary = "periodic", opts: SynthesisOptions = {}): Float64Array {
  const { a, d, k0, n } = st, L = w.h.length;
  const h = opts.wrongFilters ? Float64Array.from(w.h).reverse() : w.h;
  const g = opts.wrongFilters ? Float64Array.from(w.g).reverse() : w.g;
  const y = new Float64Array(n);
  for (let j = 0; j < a.length; j++) {
    const s = 2 * (k0 + j), aj = opts.dropLow ? 0 : a[j], dj = opts.dropHigh ? 0 : d[j];
    for (let m = 0; m < L; m++) {
      const i = s + m;
      let t: number;
      if (mode === "periodic") t = ((i % n) + n) % n;
      else if (i >= 0 && i < n) t = i;
      else continue;
      y[t] += h[m] * aj + g[m] * dj;
    }
  }
  return y;
}

/** Branch signals of one level for drawing: filtered (before ↓2), and each branch after synthesis. */
export function branchSignals(x: ArrayLike<number>, w: OrthoWavelet, mode: Boundary = "periodic"): {
  lowFull: Float64Array; highFull: Float64Array; step: Step; lowBack: Float64Array; highBack: Float64Array;
} {
  const n = x.length, L = w.h.length;
  const lowFull = new Float64Array(n), highFull = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sa = 0, sd = 0;
    for (let m = 0; m < L; m++) { const v = at(x, i + m, n, mode); sa += w.h[m] * v; sd += w.g[m] * v; }
    lowFull[i] = sa; highFull[i] = sd;
  }
  const step = analysisStep(x, w, mode);
  return {
    lowFull, highFull, step,
    lowBack: synthesisStep(step, w, mode, { dropHigh: true }),
    highBack: synthesisStep(step, w, mode, { dropLow: true }),
  };
}

/** Multi-level decomposition with any boundary: details[0] is level 1 (finest). */
export function wavedec(x: ArrayLike<number>, w: OrthoWavelet, levels: number, mode: Boundary = "periodic"): { approx: Step; steps: Step[] } {
  const steps: Step[] = [];
  let cur: ArrayLike<number> = x;
  for (let j = 0; j < levels; j++) {
    const st = analysisStep(cur, w, mode);
    steps.push(st);
    cur = st.a;
  }
  return { approx: steps[steps.length - 1], steps };
}

/** Inverse of wavedec. `steps[j].a` of the coarsest step is the approximation; finer `a`s are recomputed. */
export function waverec(steps: Step[], w: OrthoWavelet, mode: Boundary = "periodic"): Float64Array {
  let a = steps[steps.length - 1].a;
  for (let j = steps.length - 1; j >= 0; j--) {
    a = synthesisStep({ ...steps[j], a }, w, mode);
  }
  return a;
}

// ---------------------------------------------------------------------------------------
// Non-expansive in-place 1D transforms (periodic orthogonal, or symmetric lifting)

/** One forward level on buf[0..n) (n even): result [a | d] in place. */
export function forward1(buf: Float64Array, n: number, w: Wavelet, tmp: Float64Array = new Float64Array(n), integer = false): void {
  const h2 = n >> 1;
  if (w.kind === "orth") {
    const L = w.h.length, h = w.h, g = w.g;
    for (let k = 0; k < h2; k++) {
      let sa = 0, sd = 0;
      for (let m = 0; m < L; m++) { let i = 2 * k + m; if (i >= n) i %= n; const v = buf[i]; sa += h[m] * v; sd += g[m] * v; }
      tmp[k] = sa; tmp[h2 + k] = sd;
    }
  } else {
    const s = tmp.subarray(0, h2), d = tmp.subarray(h2, n);
    for (let k = 0; k < h2; k++) { s[k] = buf[2 * k]; d[k] = buf[2 * k + 1]; }
    liftSteps(s, d, w, integer, false);
  }
  for (let i = 0; i < n; i++) buf[i] = tmp[i];
}

/** One inverse level on buf[0..n): [a | d] → signal, in place. */
export function inverse1(buf: Float64Array, n: number, w: Wavelet, tmp: Float64Array = new Float64Array(n), integer = false): void {
  const h2 = n >> 1;
  if (w.kind === "orth") {
    const L = w.h.length, h = w.h, g = w.g;
    tmp.fill(0, 0, n);
    for (let k = 0; k < h2; k++) {
      const a = buf[k], d = buf[h2 + k];
      for (let m = 0; m < L; m++) { let i = 2 * k + m; if (i >= n) i %= n; tmp[i] += h[m] * a + g[m] * d; }
    }
    for (let i = 0; i < n; i++) buf[i] = tmp[i];
  } else {
    const s = Float64Array.from(buf.subarray(0, h2)), d = Float64Array.from(buf.subarray(h2, n));
    liftSteps(s, d, w, integer, true);
    for (let k = 0; k < h2; k++) { buf[2 * k] = s[k]; buf[2 * k + 1] = d[k]; }
  }
}

/**
 * Lifting on even samples s and odd samples d (whole-sample symmetric extension:
 * s[n2] = s[n2−1] mirrored through the last odd sample, d[−1] = d[0]).
 * Forward: predict  d[k] += c (s[k] + s[k+1]);  update  s[k] += c (d[k−1] + d[k]); then scale.
 * Integer mode (5/3 only): d[k] −= floor((s[k] + s[k+1]) / 2), s[k] += floor((d[k−1] + d[k] + 2) / 4).
 */
function liftSteps(s: Float64Array, d: Float64Array, w: LiftWavelet, integer: boolean, inverse: boolean): void {
  const n2 = s.length;
  const sAt = (k: number) => s[k >= n2 ? n2 - 1 : k];
  const dAt = (k: number) => d[k < 0 ? 0 : k];
  const steps = inverse ? [...w.steps].reverse() : w.steps;
  const sign = inverse ? -1 : 1;
  if (!integer && inverse) {
    const ks = w.K / Math.SQRT2, kd = Math.SQRT2 / w.K;
    for (let k = 0; k < n2; k++) { s[k] *= ks; d[k] *= kd; }
  }
  for (const st of steps) {
    if (st.type === "predict") {
      for (let k = 0; k < n2; k++) {
        const v = sAt(k) + sAt(k + 1);
        d[k] += sign * (integer ? -Math.floor(v / 2) : st.c * v);
      }
    } else {
      for (let k = 0; k < n2; k++) {
        const v = dAt(k - 1) + dAt(k);
        s[k] += sign * (integer ? Math.floor((v + 2) / 4) : st.c * v);
      }
    }
  }
  if (!integer && !inverse) {
    const ks = Math.SQRT2 / w.K, kd = w.K / Math.SQRT2;
    for (let k = 0; k < n2; k++) { s[k] *= ks; d[k] *= kd; }
  }
}

/** Largest number of levels for length n (each level halves; stop at length 2). */
export function maxLevels(n: number): number {
  let j = 0;
  while (n % 2 === 0 && n >= 4) { n /= 2; j++; }
  return j;
}

/** Multi-level 1D DWT in Mallat layout [a_J | d_J | … | d_1] (n divisible by 2^levels). */
export function dwt(x: ArrayLike<number>, w: Wavelet, levels: number, integer = false): Float64Array {
  const c = Float64Array.from(x), tmp = new Float64Array(c.length);
  let n = c.length;
  for (let j = 0; j < levels; j++) { forward1(c, n, w, tmp, integer); n >>= 1; }
  return c;
}

/** Inverse of dwt. */
export function idwt(c: ArrayLike<number>, w: Wavelet, levels: number, integer = false): Float64Array {
  const x = Float64Array.from(c), tmp = new Float64Array(x.length);
  let n = x.length >> levels;
  for (let j = 0; j < levels; j++) { n <<= 1; inverse1(x, n, w, tmp, integer); }
  return x;
}

/** Where level j (1 = finest) of a 1D Mallat vector of length n sits: details [start, start+len). */
export function detailRange1d(n: number, level: number): { start: number; len: number } {
  const len = n >> level;
  return { start: len, len };
}

// ---------------------------------------------------------------------------------------
// 2D separable transform (Mallat layout)

/** Forward 2D DWT of a W × H image (row-major), `levels` levels, in a new array. */
export function dwt2(img: ArrayLike<number>, W: number, H: number, w: Wavelet, levels: number): Float64Array {
  const c = Float64Array.from(img);
  const line = new Float64Array(Math.max(W, H)), tmp = new Float64Array(Math.max(W, H));
  let cw = W, ch = H;
  for (let j = 0; j < levels; j++) {
    for (let y = 0; y < ch; y++) {
      const o = y * W;
      for (let x = 0; x < cw; x++) line[x] = c[o + x];
      forward1(line, cw, w, tmp);
      for (let x = 0; x < cw; x++) c[o + x] = line[x];
    }
    for (let x = 0; x < cw; x++) {
      for (let y = 0; y < ch; y++) line[y] = c[y * W + x];
      forward1(line, ch, w, tmp);
      for (let y = 0; y < ch; y++) c[y * W + x] = line[y];
    }
    cw >>= 1; ch >>= 1;
  }
  return c;
}

/** Inverse of dwt2. */
export function idwt2(coef: ArrayLike<number>, W: number, H: number, w: Wavelet, levels: number): Float64Array {
  const c = Float64Array.from(coef);
  const line = new Float64Array(Math.max(W, H)), tmp = new Float64Array(Math.max(W, H));
  for (let j = levels - 1; j >= 0; j--) {
    const cw = W >> j, ch = H >> j;
    for (let x = 0; x < cw; x++) {
      for (let y = 0; y < ch; y++) line[y] = c[y * W + x];
      inverse1(line, ch, w, tmp);
      for (let y = 0; y < ch; y++) c[y * W + x] = line[y];
    }
    for (let y = 0; y < ch; y++) {
      const o = y * W;
      for (let x = 0; x < cw; x++) line[x] = c[o + x];
      inverse1(line, cw, w, tmp);
      for (let x = 0; x < cw; x++) c[o + x] = line[x];
    }
  }
  return c;
}

export type Band = "LL" | "H" | "V" | "D";

/** Rectangle of a subband in the Mallat layout. LL only exists at the coarsest level. */
export function subbandRect(level: number, band: Band, W: number, H: number): { x: number; y: number; w: number; h: number } {
  const w = W >> level, h = H >> level;
  switch (band) {
    case "LL": return { x: 0, y: 0, w, h };
    case "V": return { x: w, y: 0, w, h };
    case "H": return { x: 0, y: h, w, h };
    default: return { x: w, y: h, w, h };
  }
}

/** Which subband a coefficient position (x, y) belongs to. */
export function bandAt(x: number, y: number, W: number, H: number, levels: number): { level: number; band: Band } {
  for (let j = 1; j <= levels; j++) {
    const w = W >> j, h = H >> j;
    if (x >= w || y >= h) return { level: j, band: x >= w && y >= h ? "D" : x >= w ? "V" : "H" };
  }
  return { level: levels, band: "LL" };
}

/** Mask (1 = detail coefficient, 0 = coarsest LL) for a Mallat layout. */
export function detailMask(W: number, H: number, levels: number): Uint8Array {
  const m = new Uint8Array(W * H).fill(1), w = W >> levels, h = H >> levels;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) m[y * W + x] = 0;
  return m;
}

/** Values of one subband, copied out. */
export function subband(c: ArrayLike<number>, W: number, H: number, level: number, band: Band): Float64Array {
  const r = subbandRect(level, band, W, H), out = new Float64Array(r.w * r.h);
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) out[y * r.w + x] = c[(r.y + y) * W + r.x + x];
  return out;
}

// ---------------------------------------------------------------------------------------
// The Haar transform step by step (haarstep demo)

/** Normalisation of the Haar step: "half" s = (a+b)/2, d = (a−b)/2; "orth" with 1/√2. */
export type HaarNorm = "half" | "orth";

/** Mallat vectors after 0, 1, …, levels Haar levels (states[0] = x). */
export function haarStates(x: ArrayLike<number>, norm: HaarNorm, levels = maxLevels(x.length) + 1): Float64Array[] {
  const k = norm === "half" ? 0.5 : Math.SQRT1_2;
  const states = [Float64Array.from(x)];
  let n = x.length;
  for (let j = 0; j < levels && n >= 2; j++) {
    const prev = states[states.length - 1], c = Float64Array.from(prev), h2 = n >> 1;
    for (let i = 0; i < h2; i++) { c[i] = k * (prev[2 * i] + prev[2 * i + 1]); c[h2 + i] = k * (prev[2 * i] - prev[2 * i + 1]); }
    states.push(c);
    n = h2;
  }
  return states;
}

/** Inverse of a full Haar transform with `levels` levels (a = s + d, b = s − d for "half"). */
export function haarInverse(c: ArrayLike<number>, norm: HaarNorm, levels: number): Float64Array {
  const k = norm === "half" ? 1 : Math.SQRT1_2;
  const x = Float64Array.from(c);
  let n = x.length >> levels;
  for (let j = 0; j < levels; j++) {
    const prev = Float64Array.from(x.subarray(0, 2 * n));
    for (let i = 0; i < n; i++) { x[2 * i] = k * (prev[i] + prev[n + i]); x[2 * i + 1] = k * (prev[i] - prev[n + i]); }
    n *= 2;
  }
  return x;
}

/** The N × N orthonormal Haar matrix (rows = basis vectors in Mallat order), full depth. */
export function haarMatrix(N: number): Float64Array {
  const levels = maxLevels(N) + 1, M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    const e = new Float64Array(N); e[i] = 1;
    const c = haarStates(e, "orth", levels)[levels];
    for (let r = 0; r < N; r++) M[r * N + i] = c[r];
  }
  return M;
}

/** Level (1 = finest) of entry i of a 1D Mallat vector of length n with `levels` levels; 0 = approximation. */
export function levelOf(i: number, n: number, levels: number): number {
  if (i < n >> levels) return 0;
  for (let j = levels; j >= 1; j--) if (i < n >> (j - 1)) return j;
  return 1;
}

// ---------------------------------------------------------------------------------------
// Lifting with integers (Haar and LeGall 5/3)

/** Integer Haar by lifting, one level: d = b − a, s = a + floor(d/2). Returns [s | d]. */
export function liftHaarInt(x: ArrayLike<number>): Int32Array {
  const n2 = x.length >> 1, c = new Int32Array(x.length);
  for (let k = 0; k < n2; k++) {
    const d = x[2 * k + 1] - x[2 * k];
    c[k] = x[2 * k] + Math.floor(d / 2); c[n2 + k] = d;
  }
  return c;
}

/** Inverse of liftHaarInt: a = s − floor(d/2), b = a + d. */
export function unliftHaarInt(c: ArrayLike<number>): Int32Array {
  const n2 = c.length >> 1, x = new Int32Array(c.length);
  for (let k = 0; k < n2; k++) {
    const d = c[n2 + k], a = c[k] - Math.floor(d / 2);
    x[2 * k] = a; x[2 * k + 1] = a + d;
  }
  return x;
}

// ---------------------------------------------------------------------------------------
// Scaling function and wavelet by the cascade algorithm

/** φ and ψ at t = k / 2^iterations, k = 0 … (L−1)·2^iterations, by iterating the two-scale relation. */
export function cascade(w: OrthoWavelet, iterations: number): { t: Float64Array; phi: Float64Array; psi: Float64Array } {
  const L = w.h.length;
  let v = Float64Array.of(1);
  const up = (u: Float64Array, f: Float64Array) => {
    const out = new Float64Array(2 * (u.length - 1) + f.length);
    for (let i = 0; i < u.length; i++) for (let m = 0; m < f.length; m++) out[2 * i + m] += Math.SQRT2 * f[m] * u[i];
    return out;
  };
  let prev = v;
  for (let i = 0; i < iterations; i++) { prev = v; v = up(v, w.h); }
  // ψ(k/2^i) = √2 Σ_n g_n φ_{i−1}[k − n·2^(i−1)]: the filter g dilated to the previous grid
  let psi0: Float64Array;
  if (iterations > 0) {
    const s = 2 ** (iterations - 1);
    psi0 = new Float64Array(prev.length + (L - 1) * s);
    for (let n = 0; n < L; n++) for (let k = 0; k < prev.length; k++) psi0[k + n * s] += Math.SQRT2 * w.g[n] * prev[k];
  } else psi0 = Float64Array.from(v);
  const n = (L - 1) * 2 ** iterations + 1, T = new Float64Array(n), phi = new Float64Array(n), psi = new Float64Array(n);
  for (let k = 0; k < n; k++) { T[k] = k / 2 ** iterations; phi[k] = v[k] ?? 0; psi[k] = psi0[k] ?? 0; }
  return { t: T, phi, psi };
}

// ---------------------------------------------------------------------------------------
// Thresholding and denoising

/** Hard threshold: keep w if |w| > t, else 0. */
export const hard = (w: number, t: number): number => (Math.abs(w) > t ? w : 0);
/** Soft threshold: sign(w)·max(|w| − t, 0). */
export const soft = (w: number, t: number): number => (w > t ? w - t : w < -t ? w + t : 0);

/** Median of |values| / 0.6745: robust noise estimate from detail coefficients. */
export function madSigma(values: ArrayLike<number>): number {
  const a = Float64Array.from(values, Math.abs).sort(), n = a.length;
  if (!n) return 0;
  const med = n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
  return med / 0.6744897501960817;
}

/** Universal threshold σ √(2 ln n). */
export const universalThreshold = (sigma: number, n: number): number => sigma * Math.sqrt(2 * Math.log(n));

/** Apply a threshold to the detail coefficients of a 2D Mallat layout (LL untouched), in place. Returns how many details survive. */
export function thresholdDetails2(c: Float64Array, W: number, H: number, levels: number, t: number, kind: "hard" | "soft"): number {
  const lw = W >> levels, lh = H >> levels;
  let kept = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (x < lw && y < lh) continue;
    const i = y * W + x, v = c[i];
    const r = kind === "hard" ? hard(v, t) : soft(v, t);
    c[i] = r;
    if (r !== 0) kept++;
  }
  return kept;
}

/** Circular shift of a W × H image by (dx, dy). */
export function shift2(img: ArrayLike<number>, W: number, H: number, dx: number, dy: number): Float64Array {
  const out = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    const yy = (((y + dy) % H) + H) % H;
    for (let x = 0; x < W; x++) out[yy * W + ((((x + dx) % W) + W) % W)] = img[y * W + x];
  }
  return out;
}

/**
 * Wavelet shrinkage of a W × H image: DWT, threshold the details, inverse. With `shifts` > 1
 * (cycle spinning) the image is shifted by (dx, dy) for dx, dy = 0 … s−1 with s = √shifts,
 * denoised, shifted back and the results averaged. Returns the estimate and the fraction of
 * detail coefficients kept (unshifted transform).
 */
export function denoise2(img: ArrayLike<number>, W: number, H: number, w: Wavelet, levels: number, t: number,
  kind: "hard" | "soft", shifts = 1): { out: Float64Array; kept: number } {
  const s = Math.max(1, Math.round(Math.sqrt(shifts)));
  const out = new Float64Array(W * H);
  let kept = 0;
  for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) {
    const sh = dx || dy ? shift2(img, W, H, dx, dy) : Float64Array.from(img);
    const c = dwt2(sh, W, H, w, levels);
    const k = thresholdDetails2(c, W, H, levels, t, kind);
    if (!dx && !dy) kept = k / (W * H - (W >> levels) * (H >> levels));
    const r = idwt2(c, W, H, w, levels);
    const back = dx || dy ? shift2(r, W, H, -dx, -dy) : r;
    for (let i = 0; i < out.length; i++) out[i] += back[i];
  }
  const k = 1 / (s * s);
  for (let i = 0; i < out.length; i++) out[i] *= k;
  return { out, kept };
}

/** 1D shrinkage (periodic / lifting transform, Mallat layout), with optional cycle spinning over `shifts` shifts. */
export function denoise1(x: ArrayLike<number>, w: Wavelet, levels: number, t: number, kind: "hard" | "soft", shifts = 1): Float64Array {
  const n = x.length, out = new Float64Array(n), a0 = n >> levels;
  for (let s = 0; s < shifts; s++) {
    const sh = new Float64Array(n);
    for (let i = 0; i < n; i++) sh[(i + s) % n] = x[i];
    const c = dwt(sh, w, levels);
    for (let i = a0; i < n; i++) c[i] = kind === "hard" ? hard(c[i], t) : soft(c[i], t);
    const r = idwt(c, w, levels);
    for (let i = 0; i < n; i++) out[i] += r[(i + s) % n] / shifts;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Keeping the largest coefficients

/** Threshold value such that the `k` largest |values| (among `mask` = 1 entries, if given) are > it. */
export function kthLargestAbs(values: ArrayLike<number>, k: number): number {
  if (k <= 0) return Infinity;
  const a = Float64Array.from(values, Math.abs).sort();
  if (k >= a.length) return -1;
  return a[a.length - k - 1];
}

/** Copy with all but the k largest |values| set to 0 (ties broken by position). */
export function keepLargest(values: ArrayLike<number>, k: number): Float64Array {
  const n = values.length, out = new Float64Array(n);
  if (k <= 0) return out;
  if (k >= n) { out.set(Array.from(values)); return out; }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => Math.abs(values[j]) - Math.abs(values[i]) || i - j);
  for (let r = 0; r < k; r++) out[idx[r]] = values[idx[r]];
  return out;
}

/** Relative error ‖dropped‖ / ‖all‖ after keeping the k largest, for every k = 0 … n (orthonormal bases). */
export function tailErrors(values: ArrayLike<number>): Float64Array {
  const sq = Float64Array.from(values, v => v * v).sort().reverse(), n = sq.length;
  let total = 0;
  for (let i = 0; i < n; i++) total += sq[i];
  const out = new Float64Array(n + 1);
  let tail = total;
  out[0] = total > 0 ? 1 : 0;
  for (let k = 1; k <= n; k++) { tail -= sq[k - 1]; out[k] = total > 0 ? Math.sqrt(Math.max(0, tail) / total) : 0; }
  return out;
}

// ---------------------------------------------------------------------------------------
// Test signals (Donoho & Johnstone 1994 and simple ones), t = i / n

export type SignalName = "spike" | "step" | "twosteps" | "ramp" | "sine" | "chirp" | "blocks" | "bumps" | "heavisine" | "doppler" | "quadratic" | "cubic";

export const SIGNAL_LABELS: Record<SignalName, string> = {
  spike: "spike", step: "step", twosteps: "two steps", ramp: "ramp", sine: "sine", chirp: "chirp",
  blocks: "Blocks", bumps: "Bumps", heavisine: "HeaviSine", doppler: "Doppler", quadratic: "quadratic", cubic: "cubic",
};

const DJ_T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81];
const BLOCKS_H = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2];
const BUMPS_H = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2];
const BUMPS_W = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005];

/** A test signal of length n. */
export function testSignal(name: SignalName, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / n;
    let v = 0;
    switch (name) {
      case "spike": v = i === Math.round(0.37 * n) ? 1 : 0; break;
      case "step": v = t >= 0.37 ? 1 : 0; break;
      case "twosteps": v = (t >= 0.3 ? 1 : 0) - (t >= 0.62 ? 1.5 : 0); break;
      case "ramp": v = t; break;
      case "sine": v = Math.sin(2 * Math.PI * 4 * t); break;
      case "chirp": v = Math.sin(2 * Math.PI * (2 * t + 30 * t * t)); break;
      case "blocks": for (let j = 0; j < 11; j++) v += BLOCKS_H[j] * (t - DJ_T[j] >= 0 ? 1 : 0); break;
      case "bumps": for (let j = 0; j < 11; j++) v += BUMPS_H[j] * (1 + Math.abs((t - DJ_T[j]) / BUMPS_W[j])) ** -4; break;
      case "heavisine": v = 4 * Math.sin(4 * Math.PI * t) - Math.sign(t - 0.3) - Math.sign(0.72 - t); break;
      case "doppler": { const e = 0.05; v = Math.sqrt(t * (1 - t)) * Math.sin(2 * Math.PI * (1 + e) / (t + e)); break; }
      case "quadratic": v = 4 * (t - 0.5) ** 2; break;
      case "cubic": v = 8 * (t - 0.2) * (t - 0.5) * (t - 0.8) + 0.5; break;
    }
    x[i] = v;
  }
  return x;
}

// ---------------------------------------------------------------------------------------
// Real orthonormal Fourier basis (for "keep the K largest" comparisons in 1D)

/** Coefficients of x in the real orthonormal DFT basis (DC, cos/sin pairs, Nyquist); length n, Parseval exact. */
export function realFourier(x: ArrayLike<number>): Float64Array {
  const n = x.length, re = Float64Array.from(x), im = new Float64Array(n);
  fft1d(re, im, false);
  const s = 1 / Math.sqrt(n), c = new Float64Array(n);
  c[0] = re[0] * s;
  c[n - 1] = re[n / 2] * s;
  for (let k = 1; k < n / 2; k++) { c[2 * k - 1] = Math.SQRT2 * re[k] * s; c[2 * k] = Math.SQRT2 * im[k] * s; }
  return c;
}

/** Inverse of realFourier. */
export function realFourierInverse(c: ArrayLike<number>): Float64Array {
  const n = c.length, re = new Float64Array(n), im = new Float64Array(n), s = Math.sqrt(n);
  re[0] = c[0] * s; re[n / 2] = c[n - 1] * s;
  for (let k = 1; k < n / 2; k++) {
    re[k] = c[2 * k - 1] * s / Math.SQRT2; im[k] = c[2 * k] * s / Math.SQRT2;
    re[n - k] = re[k]; im[n - k] = -im[k];
  }
  fft1d(re, im, true);
  for (let i = 0; i < n; i++) re[i] /= n;
  return re;
}
