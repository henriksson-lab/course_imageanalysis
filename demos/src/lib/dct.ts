// The discrete cosine transform (DCT-II and its inverse, the DCT-III), orthonormal, and the
// helpers of transform coding (no DOM). Owner: the compression day (dct/ unit).
//
//   c_k = α_k Σ_{n=0}^{N-1} x_n cos(π (2n+1) k / 2N),  α_0 = √(1/N), α_k = √(2/N) (k ≥ 1).
//
// Orthonormal, so the inverse is the transpose and Parseval holds (Σ c² = Σ x²).
// - `dctMatrix(N)`: the N × N matrix D (row k = basis vector k), exact, any N.
// - `dct1`/`idct1`: by the matrix (any N); `dctFast`/`idctFast`: via lib/fft.ts for N a power
//   of two, using the mirrored signal (x, reversed x) of length 2N: its DFT is
//   2·e^{iπk/2N}·X_k, so the DCT is the DFT of the block mirrored (no jump at the ends).
// - `dct2`/`idct2`: separable 2D on a whole w × h image (C = D_h X D_wᵀ).
// - `blockDct`/`blockIdct`: the same on every B × B block (JPEG: B = 8); w, h multiples of B.
// - `ZIGZAG8` / `zigzag(N)`: the JPEG scan order (low to high frequency).
// - `realDftMatrix(N)`: the DFT written as an orthonormal real basis (cos and sin rows), for
//   comparing the DFT with the DCT on a block.
// - `basisImage(A, N, u, v)`: the 2D basis image of a separable basis A (rows = 1D vectors).
import { fft1d } from "./fft";

/** Orthonormal DCT-II matrix, row-major N × N: D[k*N + n] = α_k cos(π(2n+1)k/2N). */
export function dctMatrix(N: number): Float64Array {
  const D = new Float64Array(N * N);
  for (let k = 0; k < N; k++) {
    const a = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
    for (let n = 0; n < N; n++) D[k * N + n] = a * Math.cos(Math.PI * (2 * n + 1) * k / (2 * N));
  }
  return D;
}

const matCache = new Map<number, Float64Array>();
function cachedDct(N: number): Float64Array {
  let D = matCache.get(N);
  if (!D) { D = dctMatrix(N); matCache.set(N, D); }
  return D;
}

/** c = A x for a row-major N × N matrix. */
export function applyMatrix(A: ArrayLike<number>, x: ArrayLike<number>, N: number): Float64Array {
  const c = new Float64Array(N);
  for (let k = 0; k < N; k++) { let s = 0; for (let n = 0; n < N; n++) s += A[k * N + n] * x[n]; c[k] = s; }
  return c;
}

/** x = Aᵀ c (the inverse of an orthonormal A). */
export function applyTranspose(A: ArrayLike<number>, c: ArrayLike<number>, N: number): Float64Array {
  const x = new Float64Array(N);
  for (let k = 0; k < N; k++) { const ck = c[k]; if (ck === 0) continue; for (let n = 0; n < N; n++) x[n] += A[k * N + n] * ck; }
  return x;
}

/** Orthonormal DCT-II of any length, by the matrix (O(N²)). */
export const dct1 = (x: ArrayLike<number>): Float64Array => applyMatrix(cachedDct(x.length), x, x.length);
/** Inverse (DCT-III) of any length, by the matrix. */
export const idct1 = (c: ArrayLike<number>): Float64Array => applyTranspose(cachedDct(c.length), c, c.length);

/** Orthonormal DCT-II via a 2N-point FFT of the mirrored signal (N a power of two). */
export function dctFast(x: ArrayLike<number>): Float64Array {
  const N = x.length, re = new Float64Array(2 * N), im = new Float64Array(2 * N);
  for (let n = 0; n < N; n++) { re[n] = x[n]; re[2 * N - 1 - n] = x[n]; }
  fft1d(re, im);
  const c = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const a = -Math.PI * k / (2 * N);   // X_k = Re(e^{-iπk/2N} Y_k) / 2
    const X = (Math.cos(a) * re[k] - Math.sin(a) * im[k]) / 2;
    c[k] = (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N)) * X;
  }
  return c;
}

/** Inverse of dctFast (DCT-III) via a 2N-point inverse FFT (N a power of two). */
export function idctFast(c: ArrayLike<number>): Float64Array {
  const N = c.length, re = new Float64Array(2 * N), im = new Float64Array(2 * N);
  for (let k = 0; k < N; k++) {
    const s = (k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N)) * c[k], a = Math.PI * k / (2 * N);
    re[k] = s * Math.cos(a); im[k] = s * Math.sin(a);
  }
  fft1d(re, im, true);   // unnormalised: Σ_k Z_k e^{+2πi nk/2N}
  return re.slice(0, N);
}

const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0;
const fwd = (x: ArrayLike<number>) => (isPow2(x.length) && x.length > 8 ? dctFast(x) : dct1(x));
const inv = (c: ArrayLike<number>) => (isPow2(c.length) && c.length > 8 ? idctFast(c) : idct1(c));

function separable2(a: ArrayLike<number>, w: number, h: number, f: (v: ArrayLike<number>) => Float64Array): Float64Array {
  const out = Float64Array.from(a), row = new Float64Array(w), col = new Float64Array(h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = out[y * w + x];
    const r = f(row);
    for (let x = 0; x < w; x++) out[y * w + x] = r[x];
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = out[y * w + x];
    const r = f(col);
    for (let y = 0; y < h; y++) out[y * w + x] = r[y];
  }
  return out;
}

/** 2D orthonormal DCT of a whole w × h image (row-major); coefficient (u, v) at v*w + u. */
export const dct2 = (img: ArrayLike<number>, w: number, h: number): Float64Array => separable2(img, w, h, fwd);
/** Inverse of dct2. */
export const idct2 = (c: ArrayLike<number>, w: number, h: number): Float64Array => separable2(c, w, h, inv);

/** 2D DCT of one B × B block (row-major, length B²): C = D X Dᵀ. */
export function dctBlock(block: ArrayLike<number>, B = 8): Float64Array {
  const D = cachedDct(B), T = new Float64Array(B * B), C = new Float64Array(B * B);
  for (let y = 0; y < B; y++) for (let u = 0; u < B; u++) {      // rows: T = X Dᵀ
    let s = 0; for (let x = 0; x < B; x++) s += block[y * B + x] * D[u * B + x]; T[y * B + u] = s;
  }
  for (let v = 0; v < B; v++) for (let u = 0; u < B; u++) {      // columns: C = D T
    let s = 0; for (let y = 0; y < B; y++) s += D[v * B + y] * T[y * B + u]; C[v * B + u] = s;
  }
  return C;
}

/** Inverse of dctBlock: X = Dᵀ C D. */
export function idctBlock(C: ArrayLike<number>, B = 8): Float64Array {
  const D = cachedDct(B), T = new Float64Array(B * B), X = new Float64Array(B * B);
  for (let y = 0; y < B; y++) for (let u = 0; u < B; u++) {
    let s = 0; for (let v = 0; v < B; v++) s += D[v * B + y] * C[v * B + u]; T[y * B + u] = s;
  }
  for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) {
    let s = 0; for (let u = 0; u < B; u++) s += T[y * B + u] * D[u * B + x]; X[y * B + x] = s;
  }
  return X;
}

/** Copy block (bx, by) of size B out of a w-wide image. */
export function getBlock(img: ArrayLike<number>, w: number, bx: number, by: number, B = 8): Float64Array {
  const b = new Float64Array(B * B);
  for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) b[y * B + x] = img[(by * B + y) * w + bx * B + x];
  return b;
}

/** Write a B × B block back into a w-wide image. */
export function putBlock(img: Float64Array, w: number, bx: number, by: number, b: ArrayLike<number>, B = 8): void {
  for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) img[(by * B + y) * w + bx * B + x] = b[y * B + x];
}

/** DCT of every B × B block; the coefficients of a block stay in the block's place. */
export function blockDct(img: ArrayLike<number>, w: number, h: number, B = 8): Float64Array {
  const out = new Float64Array(w * h);
  for (let by = 0; by < h / B; by++) for (let bx = 0; bx < w / B; bx++) putBlock(out, w, bx, by, dctBlock(getBlock(img, w, bx, by, B), B), B);
  return out;
}

/** Inverse of blockDct. */
export function blockIdct(c: ArrayLike<number>, w: number, h: number, B = 8): Float64Array {
  const out = new Float64Array(w * h);
  for (let by = 0; by < h / B; by++) for (let bx = 0; bx < w / B; bx++) putBlock(out, w, bx, by, idctBlock(getBlock(c, w, bx, by, B), B), B);
  return out;
}

/**
 * Zigzag scan order of an N × N block: zigzag(N)[i] = row-major index (v*N + u) of the i-th
 * coefficient. Starts at DC, then (0,1), (1,0), (2,0), (1,1), (0,2), ... as in JPEG.
 */
export function zigzag(N: number): Int32Array {
  const out = new Int32Array(N * N);
  let i = 0;
  for (let s = 0; s < 2 * N - 1; s++) {
    // even diagonals run up-right (row decreasing), odd ones down-left
    if (s % 2 === 0) for (let v = Math.min(s, N - 1); v >= Math.max(0, s - N + 1); v--) out[i++] = v * N + (s - v);
    else for (let v = Math.max(0, s - N + 1); v <= Math.min(s, N - 1); v++) out[i++] = v * N + (s - v);
  }
  return out;
}

/** The JPEG zigzag order for 8 × 8 blocks. */
export const ZIGZAG8: Int32Array = zigzag(8);

/**
 * The DFT of length N as an orthonormal real basis (row-major N × N): row 0 the constant,
 * then for k = 1 .. N/2 − 1 the pair √(2/N)cos(2πkn/N), √(2/N)sin(2πkn/N), and last the
 * alternating cos(πn)/√N. Rows are ordered by frequency, so it compares with the DCT.
 * Spans the same frequency pairs (±k) as the complex DFT, and is periodic like it.
 */
export function realDftMatrix(N: number): Float64Array {
  const A = new Float64Array(N * N);
  let r = 0;
  for (let n = 0; n < N; n++) A[n] = 1 / Math.sqrt(N);
  r = 1;
  for (let k = 1; k < N / 2; k++) {
    for (let n = 0; n < N; n++) {
      A[r * N + n] = Math.sqrt(2 / N) * Math.cos(2 * Math.PI * k * n / N);
      A[(r + 1) * N + n] = Math.sqrt(2 / N) * Math.sin(2 * Math.PI * k * n / N);
    }
    r += 2;
  }
  if (N % 2 === 0) for (let n = 0; n < N; n++) A[r * N + n] = (n % 2 ? -1 : 1) / Math.sqrt(N);
  return A;
}

/** Frequency (cycles per N samples) of row r of realDftMatrix(N): 0, 1, 1, 2, 2, ..., N/2. */
export const realDftFreq = (r: number): number => Math.ceil(r / 2);

/**
 * The 2D separable basis built from a 1D basis A (row-major N × N): the N² × N² matrix whose
 * row v*N + u is the image A_v(y)·A_u(x), flattened row-major. Orthonormal if A is.
 */
export function separableBasis(A: ArrayLike<number>, N: number): Float64Array {
  const M = new Float64Array(N ** 4);
  for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
    const r = (v * N + u) * N * N;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) M[r + y * N + x] = A[v * N + y] * A[u * N + x];
  }
  return M;
}

/** The basis image (u, v) of the separable basis A, as an N × N block. */
export function basisImage(A: ArrayLike<number>, N: number, u: number, v: number): Float64Array {
  const b = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) b[y * N + x] = A[v * N + y] * A[u * N + x];
  return b;
}

/** Sum of squares. */
export function energy(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}
