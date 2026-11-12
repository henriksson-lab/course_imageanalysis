// Kernel functions of the SVM day (owner: SVM day). No DOM.
//
// Course convention: "kernel" alone is a convolution kernel; these are KERNEL FUNCTIONS
// k(x, z) = φ(x)·φ(z), similarities between two feature vectors. The file is deliberately not
// called kernels.ts.
//
//   linear      k(x, z) = x·z
//   polynomial  k(x, z) = (x·z + c)^p
//   RBF         k(x, z) = exp(−γ ‖x − z‖²)   (= the Gaussian affinity exp(−d²/2σ²) of the
//                                             graph-cut n-links and of ncut, with γ = 1/(2σ²))
//
// Also: explicit feature maps for the 2D lift demo (svmlift) and the by-hand exercises, the
// kernel (Gram) matrix, and a Mercer check (smallest eigenvalue of a kernel matrix).
import { jacobiEigen } from "./linalg";

export type KernelType = "linear" | "poly" | "rbf";

export interface KernelSpec {
  type: KernelType;
  /** RBF width: γ = 1 / (length scale)². */
  gamma?: number;
  /** Polynomial degree p. */
  degree?: number;
  /** Polynomial offset c. */
  coef0?: number;
}

/** A kernel function on two vectors stored at offsets `ia`, `ib` (length d) of flat arrays. */
export type KernelFn = (a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number, d: number) => number;

export function dotAt(a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number, d: number): number {
  let s = 0;
  for (let j = 0; j < d; j++) s += a[ia + j] * b[ib + j];
  return s;
}

export function sqDistAt(a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number, d: number): number {
  let s = 0;
  for (let j = 0; j < d; j++) { const t = a[ia + j] - b[ib + j]; s += t * t; }
  return s;
}

/** RBF kernel function from a squared distance. */
export const rbfFromSqDist = (d2: number, gamma: number): number => Math.exp(-gamma * d2);

/** The Gaussian affinity exp(−Δ²/(2σ²)) of graph-cut n-links and ncut: the RBF with γ = 1/(2σ²). */
export const gaussianAffinity = (delta: number, sigma: number): number => rbfFromSqDist(delta * delta, 1 / (2 * sigma * sigma));

/** γ ↔ σ: γ = 1/(2σ²). */
export const gammaFromSigma = (sigma: number): number => 1 / (2 * sigma * sigma);

/** The kernel function of a spec, on flat arrays. */
export function kernelFn(spec: KernelSpec): KernelFn {
  if (spec.type === "linear") return dotAt;
  if (spec.type === "poly") {
    const p = spec.degree ?? 2, c = spec.coef0 ?? 1;
    return (a, ia, b, ib, d) => Math.pow(dotAt(a, ia, b, ib, d) + c, p);
  }
  const g = spec.gamma ?? 1;
  return (a, ia, b, ib, d) => Math.exp(-g * sqDistAt(a, ia, b, ib, d));
}

/** k(x, z) for two plain vectors. */
export function kernel(spec: KernelSpec, x: ArrayLike<number>, z: ArrayLike<number>): number {
  return kernelFn(spec)(x, 0, z, 0, x.length);
}

/** Kernel matrix K[i n + j] = k(x_i, x_j) of n row-major vectors of length d. */
export function kernelMatrix(spec: KernelSpec, X: ArrayLike<number>, n: number, d: number): Float64Array {
  const K = new Float64Array(n * n), f = kernelFn(spec);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    const v = f(X, i * d, X, j * d, d);
    K[i * n + j] = v; K[j * n + i] = v;
  }
  return K;
}

/** Eigenvalues of a symmetric matrix (descending): a kernel matrix must have none below 0. */
export function kernelEigenvalues(K: ArrayLike<number>, n: number): Float64Array {
  return jacobiEigen(Float64Array.from(K), n, true).values;
}

/** Is K positive semidefinite (all eigenvalues ≥ −tol · largest |eigenvalue|)? */
export function isPSD(K: ArrayLike<number>, n: number, tol = 1e-9): boolean {
  const ev = kernelEigenvalues(K, n);
  const big = Math.max(1e-300, ...Array.from(ev, Math.abs));
  return ev.every(v => v >= -tol * big);
}

// ---------------------------------------------------------------------------------------
// Explicit feature maps (2D input)

/** φ(x) = (x1², √2 x1 x2, x2²): φ(x)·φ(z) = (x·z)². */
export function quadMap(x1: number, x2: number): [number, number, number] {
  return [x1 * x1, Math.SQRT2 * x1 * x2, x2 * x2];
}

/** The 6-coordinate map of (x·z + 1)²: (1, √2 x1, √2 x2, x1², √2 x1 x2, x2²). */
export function quadMap6(x1: number, x2: number): number[] {
  return [1, Math.SQRT2 * x1, Math.SQRT2 * x2, x1 * x1, Math.SQRT2 * x1 * x2, x2 * x2];
}

/** The paraboloid lift (x1, x2, x1² + x2²). */
export function paraboloidMap(x1: number, x2: number): [number, number, number] {
  return [x1, x2, x1 * x1 + x2 * x2];
}

/** Number of coordinates of the explicit map of (x·z + 1)^p for d inputs: C(d + p, p). */
export function polyFeatureCount(d: number, p: number): number {
  let r = 1;
  for (let i = 1; i <= p; i++) r = r * (d + i) / i;
  return Math.round(r);
}
