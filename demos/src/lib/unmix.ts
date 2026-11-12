// Linear unmixing (owner: spectral unmixing day, plan/unmixing-day.md). No DOM.
//
// The model of the whole day: a pixel with N channel values is y = M a + noise, where the
// K columns of M are the components' spectra as seen through the channels, and a holds the
// K abundances. Everything below is about building M, solving for a, and finding M when it
// is unknown.
//
// Layouts: matrices are row-major Float64Arrays with explicit sizes (as lib/linalg.ts);
// an N-channel image stack of P pixels is one Float32Array, channel-major: value of channel
// c at pixel p is Y[c * P + p]; abundance maps likewise (K * P).
//
// Sections: parametric spectra and filter bands · forward model (Poisson) · least squares,
// pseudo-inverse, WLS · NNLS (Lawson–Hanson) and fully constrained LS · conditioning ·
// endmember discovery (singular values, vertex finding, N-FINDR, NMF) · stains (optical
// density, stain matrices, Macenko) · seeded synthetic scenes (fluorescent cell, H&E/H-DAB
// brightfield, two-colour cytometry populations) · small statistics.
import { rng } from "./random";
import { cholesky, choleskySolve, svd, jacobiEigen, cross3, inv3, det } from "./linalg";
import { poisson, gauss } from "./random";

type A = ArrayLike<number>;

// =======================================================================================
// Parametric spectra and filter bands

/** An emission spectrum: a split Gaussian with peak `peak` (nm), left standard deviation
 *  `width` and right standard deviation `width * tail` (a long red tail for tail > 1). */
export interface Dye { name: string; peak: number; width: number; tail: number }

/** Relative emission of a dye at wavelength l (1 at the peak). */
export function emission(d: Dye, l: number): number {
  const s = l < d.peak ? d.width : d.width * d.tail, t = (l - d.peak) / s;
  return Math.exp(-0.5 * t * t);
}

/** A band-pass filter: transmission 1 on [lo, hi], 0 elsewhere. */
export interface Band { lo: number; hi: number }

/** Wavelength range of the day's plots and integrals. */
export const L_MIN = 400, L_MAX = 750;

/** Total emission of a dye over all wavelengths (numerically, 0.5 nm steps over 300..900 nm). */
export function totalEmission(d: Dye): number {
  let s = 0;
  for (let l = 300; l <= 900; l += 0.5) s += emission(d, l) * 0.5;
  return s;
}

/** Fraction of a dye's emitted photons that pass a band: ∫ s(λ) f(λ) dλ / ∫ s(λ) dλ
 *  (midpoint rule, 0.5 nm steps). */
export function bandFraction(d: Dye, b: Band, tot = totalEmission(d)): number {
  let s = 0;
  const step = 0.5, n = Math.max(0, Math.round((b.hi - b.lo) / step));
  for (let i = 0; i < n; i++) s += emission(d, b.lo + (i + 0.5) * step) * step;
  return s / tot;
}

/** The N × K mixing matrix: entry (c, k) = fraction of dye k's photons collected in band c. */
export function channelMatrix(dyes: Dye[], bands: Band[]): Float64Array {
  const n = bands.length, k = dyes.length, M = new Float64Array(n * k);
  dyes.forEach((d, j) => { const t = totalEmission(d); bands.forEach((b, c) => { M[c * k + j] = bandFraction(d, b, t); }); });
  return M;
}

/** n contiguous bands of equal width covering [lo, hi] (a spectral detector / lambda stack). */
export function contiguousBands(lo: number, hi: number, n: number): Band[] {
  const w = (hi - lo) / n;
  return Array.from({ length: n }, (_, i) => ({ lo: lo + i * w, hi: lo + (i + 1) * w }));
}

/** Columns divided by their entry in the dye's "own" channel (row j for column j): the
 *  spillover matrix with ones on the diagonal (square case; rows ≥ K are scaled too). */
export function normaliseOwn(M: A, n: number, k: number): Float64Array {
  const S = Float64Array.from(M);
  for (let j = 0; j < k; j++) {
    const own = j < n ? M[j * k + j] : 1;
    for (let c = 0; c < n; c++) S[c * k + j] = own > 0 ? M[c * k + j] / own : 0;
  }
  return S;
}

// =======================================================================================
// Forward model

/** Y = M A (noise-free), N × P from N × K and K × P. */
export function mix(M: A, n: number, k: number, Ab: A, P: number): Float32Array {
  const Y = new Float32Array(n * P);
  for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) {
    const m = M[c * k + j];
    if (m === 0) continue;
    const yo = c * P, ao = j * P;
    for (let p = 0; p < P; p++) Y[yo + p] += m * Ab[ao + p];
  }
  return Y;
}

/** Photon counts: Poisson(photons · (M A) + offset) per channel and pixel, seeded.
 *  With noise = false the expected counts are returned. */
export function forward(M: A, n: number, k: number, Ab: A, P: number,
  o: { photons: number; offset?: number; seed?: number; noise?: boolean }): Float32Array {
  const Y = mix(M, n, k, Ab, P), off = o.offset ?? 0, r = rng(o.seed ?? 1), noisy = o.noise ?? true;
  for (let i = 0; i < Y.length; i++) {
    const lam = Math.max(0, o.photons * Y[i] + off);
    Y[i] = noisy ? poisson(r, lam) : lam;
  }
  return Y;
}

// =======================================================================================
// Least squares

/** G = Mᵀ M (K × K). */
export function gram(M: A, n: number, k: number): Float64Array {
  const G = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    let s = 0;
    for (let c = 0; c < n; c++) s += M[c * k + i] * M[c * k + j];
    G[i * k + j] = s;
  }
  return G;
}

/** Mᵀ y (length K). */
export function mty(M: A, n: number, k: number, y: A): Float64Array {
  const b = new Float64Array(k);
  for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) b[j] += M[c * k + j] * y[c];
  return b;
}

/** Pseudo-inverse M⁺ = (MᵀM)⁻¹Mᵀ (K × N) via the Cholesky factor of MᵀM; null if the
 *  columns are linearly dependent. For N = K this is M⁻¹. */
export function pseudoInverse(M: A, n: number, k: number): Float64Array | null {
  const L = cholesky(gram(M, n, k), k);
  if (!L) return null;
  const P = new Float64Array(k * n), e = new Float64Array(k);
  for (let c = 0; c < n; c++) {
    for (let j = 0; j < k; j++) e[j] = M[c * k + j];
    const x = choleskySolve(L, e, k);
    for (let j = 0; j < k; j++) P[j * n + c] = x[j];
  }
  return P;
}

/** B Y for a K × N matrix B and an N-channel stack: the same small matrix applied to every
 *  pixel (unmixing an image is one matrix product). Optional offset subtracted first and a
 *  scale (1/photons) applied after. */
export function applyToStack(B: A, k: number, n: number, Y: A, P: number, offset = 0, scale = 1): Float32Array {
  const out = new Float32Array(k * P);
  for (let j = 0; j < k; j++) for (let c = 0; c < n; c++) {
    const b = B[j * n + c] * scale;
    if (b === 0) continue;
    const yo = c * P, oo = j * P;
    for (let p = 0; p < P; p++) out[oo + p] += b * (Y[yo + p] - offset);
  }
  return out;
}

/** Least-squares abundances of one pixel (null if MᵀM is singular). */
export function lsPixel(M: A, n: number, k: number, y: A): Float64Array | null {
  const L = cholesky(gram(M, n, k), k);
  return L ? choleskySolve(L, mty(M, n, k, y), k) : null;
}

/** Weighted least squares: minimise Σ_c w_c (y_c − (M a)_c)²; a = (MᵀWM)⁻¹MᵀWy. */
export function wlsPixel(M: A, n: number, k: number, y: A, w: A): Float64Array | null {
  const G = new Float64Array(k * k), b = new Float64Array(k);
  for (let c = 0; c < n; c++) {
    const wc = w[c];
    for (let i = 0; i < k; i++) {
      const mi = M[c * k + i] * wc;
      b[i] += mi * y[c];
      for (let j = 0; j < k; j++) G[i * k + j] += mi * M[c * k + j];
    }
  }
  const L = cholesky(G, k);
  return L ? choleskySolve(L, b, k) : null;
}

/** Residual stack R = Y − scale · M A (same layout as Y). */
export function residualStack(M: A, n: number, k: number, Ab: A, Y: A, P: number, scale = 1, offset = 0): Float32Array {
  const R = new Float32Array(n * P);
  for (let i = 0; i < n * P; i++) R[i] = Y[i] - offset;
  for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) {
    const m = scale * M[c * k + j], ro = c * P, ao = j * P;
    for (let p = 0; p < P; p++) R[ro + p] -= m * Ab[ao + p];
  }
  return R;
}

/** Root-sum-square over channels of a stack: one number per pixel. */
export function stackNorm(R: A, n: number, P: number): Float32Array {
  const out = new Float32Array(P);
  for (let c = 0; c < n; c++) for (let p = 0; p < P; p++) out[p] += R[c * P + p] ** 2;
  for (let p = 0; p < P; p++) out[p] = Math.sqrt(out[p]);
  return out;
}

// =======================================================================================
// Constraints: NNLS and fully constrained least squares

/**
 * Non-negative least squares, min ‖y − M a‖² subject to a ≥ 0, by the Lawson–Hanson active
 * set method written on the normal equations: it only needs G = MᵀM and b = Mᵀy, so the
 * K × K Gram matrix can be precomputed once per image. Returns a (length K).
 */
export function nnlsGram(G: A, b: A, k: number, maxIter = 3 * k + 10): Float64Array {
  const x = new Float64Array(k), passive = new Uint8Array(k);
  const grad = () => { const w = new Float64Array(k); for (let i = 0; i < k; i++) { let s = b[i]; for (let j = 0; j < k; j++) s -= G[i * k + j] * x[j]; w[i] = s; } return w; };
  const solveP = (): Float64Array => {
    const idx: number[] = [];
    for (let i = 0; i < k; i++) if (passive[i]) idx.push(i);
    const m = idx.length, Gp = new Float64Array(m * m), bp = new Float64Array(m), z = new Float64Array(k);
    idx.forEach((i, r) => { bp[r] = b[i]; idx.forEach((j, c) => { Gp[r * m + c] = G[i * k + j]; }); });
    const L = cholesky(Gp, m);
    if (L) { const s = choleskySolve(L, bp, m); idx.forEach((i, r) => { z[i] = s[r]; }); }
    return z;
  };
  const tol = 1e-12 * Math.max(1, ...Array.from(b, Math.abs));
  for (let outer = 0; outer < maxIter; outer++) {
    const w = grad();
    let t = -1, best = tol;
    for (let i = 0; i < k; i++) if (!passive[i] && w[i] > best) { best = w[i]; t = i; }
    if (t < 0) break;                       // KKT holds: done
    passive[t] = 1;
    for (let inner = 0; inner < 2 * k + 5; inner++) {
      const z = solveP();
      let ok = true;
      for (let i = 0; i < k; i++) if (passive[i] && z[i] <= 0) { ok = false; break; }
      if (ok) { x.set(z); break; }
      // step from x towards z until the first passive variable hits zero
      let alpha = Infinity;
      for (let i = 0; i < k; i++) if (passive[i] && z[i] <= 0) alpha = Math.min(alpha, x[i] / (x[i] - z[i]));
      for (let i = 0; i < k; i++) {
        x[i] += alpha * (z[i] - x[i]);
        if (passive[i] && Math.abs(x[i]) < 1e-14) { x[i] = 0; passive[i] = 0; }
      }
    }
  }
  return x;
}

/** NNLS of one pixel. */
export function nnls(M: A, n: number, k: number, y: A): Float64Array {
  return nnlsGram(gram(M, n, k), mty(M, n, k, y), k);
}

/** Brute-force NNLS for small K (tests): over all 2^K supports, the unconstrained LS on
 *  that support, kept if non-negative; the feasible one with the smallest residual. */
export function nnlsBrute(M: A, n: number, k: number, y: A): Float64Array {
  let best = new Float64Array(k), bestR = Infinity;
  for (let mask = 0; mask < 1 << k; mask++) {
    const idx = [...Array(k).keys()].filter(i => mask & (1 << i)), m = idx.length;
    const Ms = new Float64Array(n * m);
    for (let c = 0; c < n; c++) idx.forEach((j, r) => { Ms[c * m + r] = M[c * k + j]; });
    const x = new Float64Array(k);
    if (m > 0) {
      const s = lsPixel(Ms, n, m, y);
      if (!s || s.some(v => v < 0)) continue;
      idx.forEach((j, r) => { x[j] = s[r]; });
    }
    let r2 = 0;
    for (let c = 0; c < n; c++) { let p = 0; for (let j = 0; j < k; j++) p += M[c * k + j] * x[j]; r2 += (y[c] - p) ** 2; }
    if (r2 < bestR - 1e-12) { bestR = r2; best = x; }
  }
  return best;
}

/** Fully constrained least squares (a ≥ 0, Σ a = 1), Heinz & Chang: NNLS on M and y
 *  augmented with a row δ·(1, …, 1) and the value δ. Large δ enforces the sum. */
export function fcls(M: A, n: number, k: number, y: A, delta?: number): Float64Array {
  let scale = 0;
  for (let i = 0; i < n * k; i++) scale = Math.max(scale, Math.abs(M[i]));
  const d = delta ?? 1e3 * (scale || 1);
  const Ma = new Float64Array((n + 1) * k), ya = new Float64Array(n + 1);
  Ma.set(Array.from(M).slice(0, n * k));
  for (let j = 0; j < k; j++) Ma[n * k + j] = d;
  for (let c = 0; c < n; c++) ya[c] = y[c];
  ya[n] = d;
  return nnls(Ma, n + 1, k, ya);
}

/** Sum-to-one LS without non-negativity (equality constraint, closed form by a Lagrange
 *  multiplier): a = a_LS + G⁻¹1 (1 − 1ᵀa_LS)/(1ᵀG⁻¹1). */
export function stoPixel(M: A, n: number, k: number, y: A): Float64Array | null {
  const G = gram(M, n, k), L = cholesky(G, k);
  if (!L) return null;
  const a = choleskySolve(L, mty(M, n, k, y), k), g = choleskySolve(L, new Float64Array(k).fill(1), k);
  const s1 = a.reduce((s, v) => s + v, 0), s2 = g.reduce((s, v) => s + v, 0);
  for (let j = 0; j < k; j++) a[j] += g[j] * (1 - s1) / s2;
  return a;
}

// =======================================================================================
// Conditioning

/** Singular values of M (descending, length K). */
export function singularValuesOf(M: A, n: number, k: number): Float64Array {
  return svd(M, n, k).S.slice(0, Math.min(n, k));
}

/** Condition number κ = σ_max / σ_min of the N × K matrix M. */
export function condition(M: A, n: number, k: number): number {
  const s = singularValuesOf(M, n, k), smin = s[s.length - 1];
  return smin > 0 ? s[0] / smin : Infinity;
}

/** Noise amplification: Cov(â) = σ² (MᵀM)⁻¹ ⇒ standard deviation of each abundance per unit
 *  channel noise, sqrt of the diagonal of (MᵀM)⁻¹. */
export function noiseGain(M: A, n: number, k: number): Float64Array {
  const L = cholesky(gram(M, n, k), k), out = new Float64Array(k).fill(Infinity);
  if (!L) return out;
  for (let j = 0; j < k; j++) { const e = new Float64Array(k); e[j] = 1; out[j] = Math.sqrt(choleskySolve(L, e, k)[j]); }
  return out;
}

// =======================================================================================
// Endmember discovery

/** Singular values of the N × P pixel matrix (not centred), from the N × N matrix Y Yᵀ:
 *  descending, length N. */
export function stackSingularValues(Y: A, n: number, P: number): Float64Array {
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0;
    const io = i * P, jo = j * P;
    for (let p = 0; p < P; p++) s += Y[io + p] * Y[jo + p];
    G[i * n + j] = G[j * n + i] = s;
  }
  const { values } = jacobiEigen(G, n, true);
  return values.map(v => Math.sqrt(Math.max(0, v)));
}

/** Principal axes of the pixel cloud (centred covariance, N × N): eigenvalues descending
 *  and eigenvectors as columns, plus the mean spectrum. */
export function stackPCA(Y: A, n: number, P: number, weights?: A): { mean: Float64Array; values: Float64Array; vectors: Float64Array } {
  const mean = new Float64Array(n);
  let wsum = 0;
  for (let p = 0; p < P; p++) { const w = weights ? weights[p] : 1; wsum += w; for (let c = 0; c < n; c++) mean[c] += w * Y[c * P + p]; }
  for (let c = 0; c < n; c++) mean[c] /= wsum || 1;
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0;
    for (let p = 0; p < P; p++) { const w = weights ? weights[p] : 1; if (w) s += w * (Y[i * P + p] - mean[i]) * (Y[j * P + p] - mean[j]); }
    C[i * n + j] = C[j * n + i] = s / (wsum || 1);
  }
  const { values, vectors } = jacobiEigen(C, n, true);
  return { mean, values, vectors };
}

/** Pixels divided by their total over channels (the sum-to-one normalisation): the cone
 *  becomes a simplex. Pixels with a total below `minTotal` become NaN. */
export function normaliseSum(Y: A, n: number, P: number, minTotal = 1e-12): Float32Array {
  const out = new Float32Array(n * P);
  for (let p = 0; p < P; p++) {
    let s = 0;
    for (let c = 0; c < n; c++) s += Y[c * P + p];
    for (let c = 0; c < n; c++) out[c * P + p] = s > minTotal ? Y[c * P + p] / s : NaN;
  }
  return out;
}

/**
 * Vertex finding by successive projections ("most extreme pixel, project out, repeat"; the
 * successive projection algorithm, close to VCA): on the sum-normalised pixels whose total
 * is at least `minTotal`, pick the pixel of largest norm, project every pixel onto the
 * orthogonal complement of it, repeat. Returns up to K pixel indices (stops early if
 * `steps` < K). Each step is one vertex of the simplex.
 */
export function vertexFind(Y: A, n: number, P: number, K: number, minTotal = 0, steps = K): number[] {
  const cand: number[] = [];
  const R: Float64Array[] = [];
  for (let p = 0; p < P; p++) {
    let s = 0;
    for (let c = 0; c < n; c++) s += Y[c * P + p];
    if (s > Math.max(minTotal, 1e-12)) {
      cand.push(p);
      const v = new Float64Array(n);
      for (let c = 0; c < n; c++) v[c] = Y[c * P + p] / s;
      R.push(v);
    }
  }
  const picked: number[] = [];
  for (let step = 0; step < Math.min(K, steps); step++) {
    let best = -1, bestN = -1;
    R.forEach((v, i) => { let s = 0; for (let c = 0; c < n; c++) s += v[c] * v[c]; if (s > bestN) { bestN = s; best = i; } });
    if (best < 0 || bestN < 1e-24) break;
    picked.push(cand[best]);
    const u = Float64Array.from(R[best]), un = Math.sqrt(bestN);
    for (let c = 0; c < n; c++) u[c] /= un;
    for (const v of R) { let d = 0; for (let c = 0; c < n; c++) d += v[c] * u[c]; for (let c = 0; c < n; c++) v[c] -= d * u[c]; }
  }
  return picked;
}

/** Area of the triangle with vertices a, b, c in 2D. */
export const triangleArea = (a: [number, number], b: [number, number], c: [number, number]): number =>
  Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;

/** N-FINDR by brute force in 2D: the three points spanning the largest triangle. */
export function largestTriangle(pts: [number, number][]): { idx: [number, number, number]; area: number } {
  let best: [number, number, number] = [0, 1, 2], area = -1;
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) for (let k = j + 1; k < pts.length; k++) {
    const a = triangleArea(pts[i], pts[j], pts[k]);
    if (a > area) { area = a; best = [i, j, k]; }
  }
  return { idx: best, area };
}

/** Volume (up to the factor 1/(K−1)!) of the simplex with vertices E (K points in K−1
 *  dimensions, row per point): |det [1 … 1; E]|. */
export function simplexVolume(E: number[][]): number {
  const K = E.length, D = new Float64Array(K * K);
  for (let i = 0; i < K; i++) { D[i] = 1; for (let d = 0; d < K - 1; d++) D[(d + 1) * K + i] = E[i][d]; }
  return Math.abs(det(D, K));
}

/** NMF state: Y (N × P) ≈ W (N × K) H (K × P), all non-negative. */
export interface NMF { n: number; k: number; P: number; W: Float64Array; H: Float64Array; fixed: boolean[] }

/** Seeded positive start. Columns of W given in `known` (length K, null = free) are fixed.
 *  W columns are scaled to sum 1; H starts at the mean intensity. */
export function nmfInit(Y: A, n: number, P: number, k: number, seed = 1, known: (A | null)[] = []): NMF {
  const r = rng(seed), W = new Float64Array(n * k), H = new Float64Array(k * P);
  let mean = 0;
  for (let i = 0; i < n * P; i++) mean += Y[i];
  mean /= n * P;
  for (let j = 0; j < k; j++) {
    const kn = known[j];
    let s = 0;
    for (let c = 0; c < n; c++) { const v = kn ? kn[c] : 0.1 + r.unif(); W[c * k + j] = v; s += v; }
    for (let c = 0; c < n; c++) W[c * k + j] /= s || 1;
  }
  for (let i = 0; i < k * P; i++) H[i] = (0.5 + r.unif()) * mean * n / k;
  return { n, k, P, W, H, fixed: Array.from({ length: k }, (_, j) => !!known[j]) };
}

/** ‖Y − WH‖²_F. */
export function nmfError(Y: A, s: NMF): number {
  const { n, k, P, W, H } = s;
  let e = 0;
  for (let c = 0; c < n; c++) for (let p = 0; p < P; p++) {
    let v = 0;
    for (let j = 0; j < k; j++) v += W[c * k + j] * H[j * P + p];
    e += (Y[c * P + p] - v) ** 2;
  }
  return e;
}

/** One round of Lee–Seung multiplicative updates for the Frobenius error:
 *  H ← H ⊙ (WᵀY) ⊘ (WᵀWH), then W ← W ⊙ (YHᵀ) ⊘ (WHHᵀ) for the free columns.
 *  The error never increases (Lee & Seung 2001). No rescaling inside, so that property
 *  holds step by step; use nmfNormalise for display. */
export function nmfStep(Y: A, s: NMF): void {
  const { n, k, P, W, H } = s, eps = 1e-12;
  // H update
  const WtW = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { let v = 0; for (let c = 0; c < n; c++) v += W[c * k + i] * W[c * k + j]; WtW[i * k + j] = v; }
  const WtY = new Float64Array(k * P);
  for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) { const w = W[c * k + j]; if (w === 0) continue; const yo = c * P, o = j * P; for (let p = 0; p < P; p++) WtY[o + p] += w * Y[yo + p]; }
  const col = new Float64Array(k);
  for (let p = 0; p < P; p++) {
    for (let j = 0; j < k; j++) col[j] = H[j * P + p];
    for (let i = 0; i < k; i++) {
      let d = 0;
      for (let j = 0; j < k; j++) d += WtW[i * k + j] * col[j];
      H[i * P + p] = col[i] * WtY[i * P + p] / (d + eps);
    }
  }
  if (s.fixed.every(f => f)) return;
  // W update
  const HHt = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = i; j < k; j++) { let v = 0; const io = i * P, jo = j * P; for (let p = 0; p < P; p++) v += H[io + p] * H[jo + p]; HHt[i * k + j] = HHt[j * k + i] = v; }
  const YHt = new Float64Array(n * k);
  for (let c = 0; c < n; c++) for (let j = 0; j < k; j++) { let v = 0; const yo = c * P, ho = j * P; for (let p = 0; p < P; p++) v += Y[yo + p] * H[ho + p]; YHt[c * k + j] = v; }
  for (let c = 0; c < n; c++) {
    const row = Array.from({ length: k }, (_, j) => W[c * k + j]);
    for (let j = 0; j < k; j++) {
      if (s.fixed[j]) continue;
      let d = 0;
      for (let i = 0; i < k; i++) d += row[i] * HHt[i * k + j];
      W[c * k + j] = row[j] * YHt[c * k + j] / (d + eps);
    }
  }
}

/** Rescale every free column of W to sum 1 and the matching row of H inversely (WH unchanged). */
export function nmfNormalise(s: NMF): void {
  const { n, k, P, W, H } = s;
  for (let j = 0; j < k; j++) {
    if (s.fixed[j]) continue;
    let t = 0;
    for (let c = 0; c < n; c++) t += W[c * k + j];
    if (!(t > 0)) continue;
    for (let c = 0; c < n; c++) W[c * k + j] /= t;
    for (let p = 0; p < P; p++) H[j * P + p] *= t;
  }
}

/** Match found spectra to true ones (columns of N × K matrices, both scaled to sum 1):
 *  greedy by largest cosine similarity. Returns perm[trueIndex] = foundIndex and the angles. */
export function matchColumns(T: A, F: A, n: number, k: number): { perm: number[]; angleDeg: number[] } {
  const cos = (i: number, j: number) => {
    let a = 0, b = 0, c = 0;
    for (let r = 0; r < n; r++) { a += T[r * k + i] * F[r * k + j]; b += T[r * k + i] ** 2; c += F[r * k + j] ** 2; }
    return a / Math.sqrt(b * c || 1);
  };
  const perm = new Array(k).fill(-1), used = new Set<number>(), angleDeg = new Array(k).fill(NaN);
  const pairs: [number, number, number][] = [];
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) pairs.push([cos(i, j), i, j]);
  pairs.sort((a, b) => b[0] - a[0]);
  for (const [c, i, j] of pairs) if (perm[i] < 0 && !used.has(j)) { perm[i] = j; used.add(j); angleDeg[i] = Math.acos(Math.min(1, c)) * 180 / Math.PI; }
  return { perm, angleDeg };
}

// =======================================================================================
// Stains: optical density and colour deconvolution

/** Ruifrok & Johnston (2001) optical-density vectors (R, G, B) of the common stains. */
export const STAINS: Record<"H" | "E" | "DAB", [number, number, number]> = {
  H: [0.65, 0.70, 0.29], E: [0.07, 0.99, 0.11], DAB: [0.27, 0.57, 0.78],
};

export const unit3 = (v: A): [number, number, number] => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

/** Optical density of a transmitted intensity: −log10(I / I0), with I clipped at `floor`. */
export const opticalDensity = (I: number, I0: number, floor = 1): number => -Math.log10(Math.max(I, floor) / I0);

/** The 3 × 3 stain matrix with the (unit) stain vectors as COLUMNS: od = S c. With only two
 *  stains, the third column is their normalised cross product (the "residual" channel). */
export function stainMatrix(v1: A, v2: A, v3?: A | null): Float64Array {
  const a = unit3(v1), b = unit3(v2);
  let c: [number, number, number];
  if (v3) c = unit3(v3);
  else { const x = cross3(a, b); c = unit3([Math.abs(x[0]), Math.abs(x[1]), Math.abs(x[2])]); }
  return Float64Array.from([a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]]);
}

/** Concentration maps from an RGB image: c = S⁻¹ od per pixel. Inputs 0..255 per channel. */
export function colourDeconvolve(R: A, G: A, B: A, I0: [number, number, number], S: A, floor = 1): { c: Float32Array[]; od: Float32Array[] } | null {
  const Si = inv3(S);
  if (!Si) return null;
  const P = R.length, c = [new Float32Array(P), new Float32Array(P), new Float32Array(P)], od = [new Float32Array(P), new Float32Array(P), new Float32Array(P)];
  for (let p = 0; p < P; p++) {
    const o = [opticalDensity(R[p], I0[0], floor), opticalDensity(G[p], I0[1], floor), opticalDensity(B[p], I0[2], floor)];
    for (let i = 0; i < 3; i++) { od[i][p] = o[i]; c[i][p] = Si[i * 3] * o[0] + Si[i * 3 + 1] * o[1] + Si[i * 3 + 2] * o[2]; }
  }
  return { c, od };
}

/**
 * Macenko et al. (2009) stain estimate: keep pixels with OD > beta in every channel; the
 * plane of the OD cloud from its two leading principal directions (uncentred, as in the
 * paper); the angle of each pixel in that plane; the two stain vectors at the alpha and
 * 100 − alpha percentiles of the angle (robust extremes = two-component vertex finding).
 * The vector with the larger red OD is returned first (hematoxylin).
 */
export function macenko(od: Float32Array[], beta = 0.15, alpha = 1): { v1: [number, number, number]; v2: [number, number, number]; used: number } | null {
  const P = od[0].length, idx: number[] = [];
  for (let p = 0; p < P; p++) if (od[0][p] > beta && od[1][p] > beta && od[2][p] > beta) idx.push(p);
  if (idx.length < 10) return null;
  const C = new Float64Array(9);
  for (const p of idx) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i * 3 + j] += od[i][p] * od[j][p];
  const { vectors } = jacobiEigen(C, 3, true);
  let e1 = [vectors[0], vectors[3], vectors[6]], e2 = [vectors[1], vectors[4], vectors[7]];
  if (e1[0] + e1[1] + e1[2] < 0) e1 = e1.map(v => -v);
  // second axis: sign chosen so that the first stain found has positive projection
  const ang = idx.map(p => { const x = od[0][p] * e1[0] + od[1][p] * e1[1] + od[2][p] * e1[2], y = od[0][p] * e2[0] + od[1][p] * e2[1] + od[2][p] * e2[2]; return Math.atan2(y, x); });
  ang.sort((a, b) => a - b);
  const q = (f: number) => ang[Math.min(ang.length - 1, Math.max(0, Math.round(f * (ang.length - 1))))];
  const lo = q(alpha / 100), hi = q(1 - alpha / 100);
  const vec = (t: number) => unit3([Math.cos(t) * e1[0] + Math.sin(t) * e2[0], Math.cos(t) * e1[1] + Math.sin(t) * e2[1], Math.cos(t) * e1[2] + Math.sin(t) * e2[2]]);
  let v1 = vec(lo), v2 = vec(hi);
  if (v2[0] > v1[0]) [v1, v2] = [v2, v1];
  return { v1, v2, used: idx.length };
}

// =======================================================================================
// Seeded synthetic scenes

const smoothstep = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/** Separable Gaussian blur of a w × h map (replicate border), in place-safe (returns new). */
export function blurMap(a: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma <= 0) return Float32Array.from(a);
  const R = Math.ceil(3 * sigma), ker = new Float64Array(2 * R + 1);
  let s = 0;
  for (let i = -R; i <= R; i++) { ker[i + R] = Math.exp(-0.5 * (i / sigma) ** 2); s += ker[i + R]; }
  for (let i = 0; i < ker.length; i++) ker[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let i = -R; i <= R; i++) v += ker[i + R] * a[y * w + Math.min(w - 1, Math.max(0, x + i))];
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let i = -R; i <= R; i++) v += ker[i + R] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x];
    out[y * w + x] = v;
  }
  return out;
}

/** The day's fluorescent cell (seeded): abundance maps in 0..~1 for nucleus, mitochondria,
 *  membrane, autofluorescence and double-labelled vesicles, plus a cell mask.
 *  The same cell is used by bleedthrough, lsunmix, unmixcloud and endmembers. */
export interface CellScene {
  n: number;
  nucleus: Float32Array; mito: Float32Array; membrane: Float32Array; auto: Float32Array; vesicles: Float32Array;
  cell: Uint8Array;
}

export function cellScene(n = 128, seed = 3): CellScene {
  const r = rng(seed), P = n * n, s = n / 128;
  const cx = 64 * s, cy = 64 * s, ra = 54 * s, rb = 44 * s, th = 0.35;
  const nx = 56 * s, ny = 60 * s, na = 20 * s, nb = 15 * s, nth = 0.6;
  const ell = (x: number, y: number, x0: number, y0: number, a: number, b: number, t: number) => {
    const dx = x - x0, dy = y - y0, u = dx * Math.cos(t) + dy * Math.sin(t), v = -dx * Math.sin(t) + dy * Math.cos(t);
    return Math.sqrt((u / a) ** 2 + (v / b) ** 2);
  };
  // wobbly outline: radius modulation
  const wob = Array.from({ length: 4 }, () => [0.04 * r.unif(), 2 + Math.floor(4 * r.unif()), 6.3 * r.unif()]);
  const cellD = (x: number, y: number) => {
    const a = Math.atan2(y - cy, x - cx);
    let f = 1;
    for (const [amp, k, ph] of wob) f += amp * Math.sin(k * a + ph);
    return ell(x, y, cx, cy, ra * f, rb * f, th);
  };
  const cell = new Uint8Array(P), nucleus = new Float32Array(P), membrane = new Float32Array(P);
  const auto = new Float32Array(P), mito = new Float32Array(P), vesicles = new Float32Array(P);
  const waves = Array.from({ length: 4 }, () => [r.unif() * 0.12 / s, r.unif() * 0.12 / s, 6.3 * r.unif()]);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const p = y * n + x, d = cellD(x, y), dn = ell(x, y, nx, ny, na, nb, nth);
    cell[p] = d < 1 ? 1 : 0;
    const edge = (d - 1) * Math.min(ra, rb);       // ~ distance to the outline in pixels
    membrane[p] = Math.exp(-0.5 * (edge / (1.1 * s)) ** 2);
    nucleus[p] = (1 - smoothstep((dn - 0.92) / 0.12)) * (0.8 + 0.2 * Math.sin(0.5 * x / s + 0.3) * Math.cos(0.43 * y / s));
    if (d < 1) {
      let w = 0.6;
      for (const [a, b, ph] of waves) w += 0.15 * Math.sin(a * x + b * y + ph);
      auto[p] = Math.max(0, w) * (1 - smoothstep((d - 0.85) / 0.15)) * (1 - 0.7 * (1 - smoothstep((dn - 0.9) / 0.2)));
    }
  }
  // mitochondria: seeded random walks in the cytoplasm
  const line = new Float32Array(P);
  for (let m = 0, tries = 0; m < 26 && tries < 400; tries++) {
    let x = cx + (r.unif() - 0.5) * 2 * ra, y = cy + (r.unif() - 0.5) * 2 * rb;
    if (cellD(x, y) > 0.85 || ell(x, y, nx, ny, na, nb, nth) < 1.25) continue;
    m++;
    let a = 6.3 * r.unif();
    const len = (8 + 14 * r.unif()) * s;
    for (let t = 0; t < len; t += 0.5) {
      a += 0.25 * gauss(r);
      x += 0.5 * Math.cos(a); y += 0.5 * Math.sin(a);
      if (cellD(x, y) > 0.9 || ell(x, y, nx, ny, na, nb, nth) < 1.1) break;
      const ix = Math.round(x), iy = Math.round(y);
      if (ix >= 0 && iy >= 0 && ix < n && iy < n) line[iy * n + ix] = 1;
    }
  }
  const mb = blurMap(line, n, n, 0.9 * s);
  let mmax = 0;
  for (let p = 0; p < P; p++) mmax = Math.max(mmax, mb[p]);
  for (let p = 0; p < P; p++) mito[p] = Math.min(1, 1.4 * mb[p] / (mmax || 1));
  // vesicles (double-labelled dots) and a few autofluorescent granules
  const dots = (out: Float32Array, count: number, rad: number, amp: number) => {
    for (let m = 0, tries = 0; m < count && tries < 500; tries++) {
      const x = cx + (r.unif() - 0.5) * 2 * ra, y = cy + (r.unif() - 0.5) * 2 * rb;
      if (cellD(x, y) > 0.82 || ell(x, y, nx, ny, na, nb, nth) < 1.2) continue;
      m++;
      for (let yy = Math.max(0, Math.floor(y - 3 * rad)); yy <= Math.min(n - 1, Math.ceil(y + 3 * rad)); yy++)
        for (let xx = Math.max(0, Math.floor(x - 3 * rad)); xx <= Math.min(n - 1, Math.ceil(x + 3 * rad)); xx++)
          out[yy * n + xx] = Math.max(out[yy * n + xx], amp * Math.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / rad ** 2));
    }
  };
  dots(vesicles, 9, 1.5 * s, 1);
  dots(auto, 7, 1.3 * s, 0.9);
  return { n, nucleus, mito, membrane, auto, vesicles, cell };
}

/** Synthetic brightfield slide by Beer–Lambert from known stain maps (seeded).
 *  mode "he": hematoxylin nuclei, eosin cytoplasm/stroma, lumens; "hdab": hematoxylin
 *  counterstain with some nuclei DAB-positive (brown), little eosin. Concentrations in OD
 *  units along the unit stain vectors; `positive` lists the DAB-positive nucleus labels. */
export interface StainScene {
  n: number; mode: "he" | "hdab";
  conc: [Float32Array, Float32Array, Float32Array];   // H, E, DAB
  nucleusLabel: Int32Array; positive: number[]; nucleusCount: number;
}

export function stainScene(n = 160, seed = 7, mode: "he" | "hdab" = "he"): StainScene {
  const r = rng(seed), P = n * n, s = n / 160;
  const cH = new Float32Array(P), cE = new Float32Array(P), cD = new Float32Array(P), lab = new Int32Array(P);
  const waves = Array.from({ length: 5 }, () => [r.unif() * 0.08 / s, r.unif() * 0.08 / s, r.unif() * 6.3]);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let e = mode === "he" ? 0.55 : 0.10;
    for (const [a, b, ph] of waves) e += (mode === "he" ? 0.09 : 0.03) * Math.sin(a * x + b * y + ph);
    const l1 = Math.hypot(x - 0.28 * n, y - 0.72 * n) / (0.15 * n), l2 = Math.hypot((x - 0.78 * n) / (0.14 * n), (y - 0.25 * n) / (0.09 * n));
    const lumen = Math.max(1 - smoothstep((l1 - 0.85) / 0.3), 1 - smoothstep((l2 - 0.85) / 0.3));
    cE[y * n + x] = Math.max(0, e * (1 - lumen));
  }
  const positive: number[] = [];
  let count = 0;
  for (let k = 0, tries = 0; k < 30 && tries < 600; tries++) {
    const x0 = r.unif() * n, y0 = r.unif() * n, a = (5 + 3 * r.unif()) * s, b = a * (0.6 + 0.35 * r.unif()), th = r.unif() * Math.PI;
    const dens = 0.55 + 0.35 * r.unif(), pos = mode === "hdab" && r.unif() < 0.4;
    if (cE[Math.floor(y0) * n + Math.floor(x0)] < (mode === "he" ? 0.2 : 0.05)) continue;
    // no overlap with an earlier nucleus
    let clash = false;
    for (let y = Math.max(0, Math.floor(y0 - a - 2)); y < Math.min(n, y0 + a + 2) && !clash; y++) for (let x = Math.max(0, Math.floor(x0 - a - 2)); x < Math.min(n, x0 + a + 2); x++) if (lab[y * n + x]) { clash = true; break; }
    if (clash) continue;
    k++; count++;
    if (pos) positive.push(count);
    for (let y = Math.max(0, Math.floor(y0 - a - 2)); y < Math.min(n, y0 + a + 2); y++) for (let x = Math.max(0, Math.floor(x0 - a - 2)); x < Math.min(n, x0 + a + 2); x++) {
      const dx = x - x0, dy = y - y0, u = dx * Math.cos(th) + dy * Math.sin(th), v = -dx * Math.sin(th) + dy * Math.cos(th);
      const d = Math.sqrt((u / a) ** 2 + (v / b) ** 2);
      if (d < 1.2) {
        const w = 1 - smoothstep((d - 0.8) / 0.35), p = y * n + x, tex = 0.85 + 0.15 * Math.sin(1.7 * x / s + 2.3 * y / s);
        const h = w * dens * tex * (pos ? 0.45 : 1);
        cH[p] = Math.max(cH[p], h);
        if (pos) cD[p] = Math.max(cD[p], w * (0.5 + 0.25 * dens) * tex);
        if (d < 1) lab[p] = count;
        cE[p] *= 1 - 0.6 * w;
      }
    }
  }
  return { n, mode, conc: [cH, cE, cD], nucleusLabel: lab, positive, nucleusCount: count };
}

/** Render a stain scene to 8-bit RGB by Beer–Lambert: I_c = I0_c 10^(−Σ_k c_k s_k,c), with
 *  optional Gaussian detector noise (sd in grey levels) and rounding; `exact` skips noise and
 *  rounding (for tests). Stain vectors as unit columns of S (3 × 3: H, E, DAB). */
export function renderBrightfield(sc: StainScene, S: A, I0: [number, number, number], o: { noise?: number; seed?: number; exact?: boolean } = {}): { R: Float32Array; G: Float32Array; B: Float32Array } {
  const P = sc.n * sc.n, out = [new Float32Array(P), new Float32Array(P), new Float32Array(P)], r = rng(o.seed ?? 5);
  for (let p = 0; p < P; p++) for (let c = 0; c < 3; c++) {
    let od = 0;
    for (let k = 0; k < 3; k++) od += S[c * 3 + k] * sc.conc[k][p];
    let v = I0[c] * 10 ** -od;
    if (!o.exact) { v += (o.noise ?? 0) * gauss(r); v = Math.min(255, Math.max(0, Math.round(v))); }
    out[c][p] = v;
  }
  return { R: out[0], G: out[1], B: out[2] };
}

/** The earlier course's four flow-cytometry populations (unstained, FITC+, PE+, FITC+ PE+:
 *  600, 400, 400, 200 cells; log-normal true amounts with σ = 0.3; detector noise sd 30),
 *  measured = S × true + noise, S = [[1, s12], [s21, 1]]. Same random stream as the old
 *  demo, so its numbers are reproduced exactly. */
export function cytometryCells(s21: number, s12: number, seed = 1): { group: number[]; tx: Float64Array; ty: Float64Array; mx: Float64Array; my: Float64Array } {
  const r = rng(seed);
  const spec: [number, number, number, number][] = [[0, 600, 100, 100], [1, 400, 3000, 100], [2, 400, 100, 2000], [3, 200, 3000, 2000]];
  const N = 1600, group: number[] = [], tx = new Float64Array(N), ty = new Float64Array(N), mx = new Float64Array(N), my = new Float64Array(N);
  let i = 0;
  for (const [g, m, f, q] of spec) for (let k = 0; k < m; k++, i++) {
    const tf = Math.exp(Math.log(f) + 0.3 * r.norm()), tp = Math.exp(Math.log(q) + 0.3 * r.norm());
    group.push(g); tx[i] = tf; ty[i] = tp;
    mx[i] = tf + s12 * tp + 30 * r.norm();
    my[i] = s21 * tf + tp + 30 * r.norm();
  }
  return { group, tx, ty, mx, my };
}

// =======================================================================================
// Small statistics

/** Pearson correlation of two equally long arrays (optionally only where mask is set). */
export function pearson(a: A, b: A, mask?: A): number {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { n++; sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb || 1);
}

/** Least-squares line y = a + b x. */
export function lsLine(x: A, y: A): { a: number; b: number } {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const b = sxx > 0 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

/** Slope of the line through the origin, Σxy / Σx² (spillover from a single-stained control). */
export function slopeThroughOrigin(x: A, y: A): number {
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sxy += x[i] * y[i]; sxx += x[i] * x[i]; }
  return sxx > 0 ? sxy / sxx : 0;
}

export function median(v: A): number {
  const s = Array.from(v).sort((a, b) => a - b), h = s.length >> 1;
  return s.length === 0 ? NaN : s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}
