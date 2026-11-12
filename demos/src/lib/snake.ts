// Active contours ("snakes", Kass, Witkin & Terzopoulos 1988) on closed polygons.
// Owner: snakes day (demos snake, snakespectrum, asm). No DOM.
//
// A contour is n points x_i = (x[i], y[i]), indices mod n (closed), in image coordinates
// (x right, y down, pixel centres at integers). The internal energy is
//   E_int = Σ_i ½α|x_{i+1} − x_i|² + ½β|x_{i−1} − 2x_i + x_{i+1}|²,
// whose gradient is A x (the same matrix for the x and the y coordinates), with A the cyclic
// pentadiagonal matrix with rows [β, −α−4β, 2α+6β, −α−4β, β]. The internal force is −A x.
// Explicit step:       x ← x + τ(−A x + f(x))
// Semi-implicit step:  (A + γI) x_new = γ x + f(x),  γ = 1/τ   (Kass et al.)
//
// External forces live on a grid (ForceField) and are sampled bilinearly at the points:
// the edge force ∇f of the edge map f = |∇(G_σ * I)|² (normalised to max 1), gradient vector
// flow (Xu & Prince 1998), and the balloon force κ n_i along the outward normal (Cohen 1991).
import { Grey, gradient } from "./imageops";
import { inverse } from "./linalg";

export interface Contour { x: Float64Array; y: Float64Array }

export function contour(xs: ArrayLike<number>, ys: ArrayLike<number>): Contour {
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

export function cloneContour(c: Contour): Contour { return { x: c.x.slice(), y: c.y.slice() }; }

/** n points on a circle, starting at angle 0 and going counter-clockwise on screen (y down). */
export function circleContour(cx: number, cy: number, r: number, n: number, phase = 0): Contour {
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = phase + 2 * Math.PI * i / n;
    x[i] = cx + r * Math.cos(t); y[i] = cy - r * Math.sin(t);
  }
  return { x, y };
}

/** Signed area ½ Σ (x_i y_{i+1} − x_{i+1} y_i); its sign gives the orientation. */
export function signedArea(c: Contour): number {
  const n = c.x.length;
  let s = 0;
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; s += c.x[i] * c.y[j] - c.x[j] * c.y[i]; }
  return s / 2;
}

export function perimeter(c: Contour): number {
  const n = c.x.length;
  let s = 0;
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; s += Math.hypot(c.x[j] - c.x[i], c.y[j] - c.y[i]); }
  return s;
}

/** Resample to `m` points equally spaced along the arc length (starting at point 0). */
export function resample(c: Contour, m = c.x.length): Contour {
  const n = c.x.length, cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; cum[i + 1] = cum[i] + Math.hypot(c.x[j] - c.x[i], c.y[j] - c.y[i]); }
  const L = cum[n], x = new Float64Array(m), y = new Float64Array(m);
  if (!(L > 0)) { x.fill(c.x[0]); y.fill(c.y[0]); return { x, y }; }
  let seg = 0;
  for (let k = 0; k < m; k++) {
    const s = L * k / m;
    while (seg < n - 1 && cum[seg + 1] <= s) seg++;
    const j = (seg + 1) % n, len = cum[seg + 1] - cum[seg], t = len > 0 ? (s - cum[seg]) / len : 0;
    x[k] = c.x[seg] + t * (c.x[j] - c.x[seg]); y[k] = c.y[seg] + t * (c.y[j] - c.y[seg]);
  }
  return { x, y };
}

// ---------------------------------------------------------------------------------------
// The internal energy and its matrix

/** The cyclic pentadiagonal n × n matrix A (row-major); entries add up when the stencil wraps (n < 5). */
export function snakeMatrix(n: number, alpha: number, beta: number): Float64Array {
  const A = new Float64Array(n * n);
  const w = [beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta];
  for (let i = 0; i < n; i++) for (let d = -2; d <= 2; d++) {
    const j = ((i + d) % n + n) % n;
    A[i * n + j] += w[d + 2];
  }
  return A;
}

/** Eigenvalue of A for the Fourier mode k: 4α sin²(πk/n) + 16β sin⁴(πk/n). */
export function snakeEigenvalue(k: number, n: number, alpha: number, beta: number): number {
  const s2 = Math.sin(Math.PI * k / n) ** 2;
  return 4 * alpha * s2 + 16 * beta * s2 * s2;
}

/** All n eigenvalues, in mode order k = 0..n−1 (not sorted). */
export function snakeEigenvalues(n: number, alpha: number, beta: number): Float64Array {
  return Float64Array.from({ length: n }, (_, k) => snakeEigenvalue(k, n, alpha, beta));
}

/** Largest eigenvalue of A (exactly 4α + 16β for even n). */
export function lambdaMax(n: number, alpha: number, beta: number): number {
  let m = 0;
  for (let k = 0; k < n; k++) m = Math.max(m, snakeEigenvalue(k, n, alpha, beta));
  return m;
}

/** Per-step multiplier of mode k: 1 − τλ_k (explicit) or γ/(γ + λ_k) = 1/(1 + τλ_k) (semi-implicit). */
export function amplification(lambda: number, tau: number, rule: "explicit" | "implicit"): number {
  return rule === "explicit" ? 1 - tau * lambda : 1 / (1 + tau * lambda);
}

/** A v for the cyclic stencil (no matrix needed). */
export function applyA(v: ArrayLike<number>, alpha: number, beta: number): Float64Array {
  const n = v.length, out = new Float64Array(n);
  const w = [beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let d = -2; d <= 2; d++) s += w[d + 2] * v[((i + d) % n + n) % n];
    out[i] = s;
  }
  return out;
}

/** Internal force −A x at every point: elastic part α(x_{i−1} − 2x_i + x_{i+1}) plus the rigid part. */
export function internalForce(c: Contour, alpha: number, beta: number): Contour {
  const fx = applyA(c.x, alpha, beta), fy = applyA(c.y, alpha, beta);
  for (let i = 0; i < fx.length; i++) { fx[i] = -fx[i]; fy[i] = -fy[i]; }
  return { x: fx, y: fy };
}

/** E_int split into its elastic (α) and rigid (β) parts. */
export function internalEnergy(c: Contour, alpha: number, beta: number): { elastic: number; rigid: number; total: number } {
  const n = c.x.length;
  let el = 0, ri = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, h = (i - 1 + n) % n;
    el += (c.x[j] - c.x[i]) ** 2 + (c.y[j] - c.y[i]) ** 2;
    ri += (c.x[h] - 2 * c.x[i] + c.x[j]) ** 2 + (c.y[h] - 2 * c.y[i] + c.y[j]) ** 2;
  }
  return { elastic: 0.5 * alpha * el, rigid: 0.5 * beta * ri, total: 0.5 * alpha * el + 0.5 * beta * ri };
}

/** Pre-factored semi-implicit solver: M = (A + γI)⁻¹ (dense; fine for n ≤ ~200). */
export interface ImplicitSolver { n: number; alpha: number; beta: number; gamma: number; inv: Float64Array }

export function implicitSolver(n: number, alpha: number, beta: number, gamma: number): ImplicitSolver {
  const A = snakeMatrix(n, alpha, beta);
  for (let i = 0; i < n; i++) A[i * n + i] += gamma;
  const inv = inverse(A, n);
  if (!inv) throw new Error("A + γI is singular");
  return { n, alpha, beta, gamma, inv };
}

/** Explicit (gradient-descent) step x ← x + τ(−A x + f). `f` may be null (internal forces only). */
export function explicitStep(c: Contour, alpha: number, beta: number, tau: number, f: Contour | null): Contour {
  const Ax = applyA(c.x, alpha, beta), Ay = applyA(c.y, alpha, beta), n = c.x.length;
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = c.x[i] + tau * (-Ax[i] + (f ? f.x[i] : 0));
    y[i] = c.y[i] + tau * (-Ay[i] + (f ? f.y[i] : 0));
  }
  return { x, y };
}

/** Semi-implicit step (A + γI) x_new = γ x + f. */
export function implicitStep(c: Contour, S: ImplicitSolver, f: Contour | null): Contour {
  const n = c.x.length;
  if (n !== S.n) throw new Error("solver size");
  const bx = new Float64Array(n), by = new Float64Array(n);
  for (let i = 0; i < n; i++) { bx[i] = S.gamma * c.x[i] + (f ? f.x[i] : 0); by[i] = S.gamma * c.y[i] + (f ? f.y[i] : 0); }
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0;
    const row = i * n;
    for (let j = 0; j < n; j++) { const m = S.inv[row + j]; sx += m * bx[j]; sy += m * by[j]; }
    x[i] = sx; y[i] = sy;
  }
  return { x, y };
}

/** Unit outward normals (from the central-difference tangent and the orientation). */
export function normals(c: Contour): Contour {
  const n = c.x.length, s = signedArea(c) >= 0 ? 1 : -1;
  const nx = new Float64Array(n), ny = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, h = (i - 1 + n) % n;
    const tx = c.x[j] - c.x[h], ty = c.y[j] - c.y[h], L = Math.hypot(tx, ty) || 1;
    nx[i] = s * ty / L; ny[i] = -s * tx / L;
  }
  return { x: nx, y: ny };
}

/** Balloon force κ n_i (κ > 0 inflates, κ < 0 deflates). */
export function balloonForce(c: Contour, kappa: number): Contour {
  const N = normals(c);
  for (let i = 0; i < N.x.length; i++) { N.x[i] *= kappa; N.y[i] *= kappa; }
  return N;
}

// ---------------------------------------------------------------------------------------
// External force fields

export interface ForceField { width: number; height: number; fx: Float32Array; fy: Float32Array }

/** Edge map f = |∇(G_σ * I)|², scaled to max 1 (E_ext = −w f is low on edges). */
export function edgeMap(img: Grey, sigma: number): Grey {
  const g = gradient(img, sigma, "replicate");
  const out: Grey = { width: img.width, height: img.height, data: new Float32Array(img.data.length) };
  let m = 0;
  for (let i = 0; i < out.data.length; i++) { const v = g.mag.data[i] ** 2; out.data[i] = v; if (v > m) m = v; }
  if (m > 0) for (let i = 0; i < out.data.length; i++) out.data[i] /= m;
  return out;
}

/** Central-difference gradient (∂f/∂x, ∂f/∂y) of a grid; one-sided at the border. */
export function gridGradient(f: Grey): ForceField {
  const { width: w, height: h } = f, fx = new Float32Array(w * h), fy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const xl = Math.max(0, x - 1), xr = Math.min(w - 1, x + 1), yu = Math.max(0, y - 1), yd = Math.min(h - 1, y + 1);
    fx[i] = (f.data[y * w + xr] - f.data[y * w + xl]) / Math.max(1, xr - xl);
    fy[i] = (f.data[yd * w + x] - f.data[yu * w + x]) / Math.max(1, yd - yu);
  }
  return { width: w, height: h, fx, fy };
}

/** Largest |v| over the field. */
export function fieldMax(F: ForceField): number {
  let m = 0;
  for (let i = 0; i < F.fx.length; i++) m = Math.max(m, Math.hypot(F.fx[i], F.fy[i]));
  return m;
}

/** Multiply a field by a constant (new field). */
export function scaleField(F: ForceField, s: number): ForceField {
  return { width: F.width, height: F.height, fx: F.fx.map(v => v * s), fy: F.fy.map(v => v * s) };
}

/**
 * Gradient vector flow (Xu & Prince 1998): starting from v = ∇f, iterate
 *   v ← v + Δt [ μ ∇²v − |∇f|² (v − ∇f) ]
 * (5-point Laplacian, replicated border). Stable for Δt ≤ 1/(4μ). Near edges v stays ≈ ∇f,
 * in flat regions it is a smooth extension that points towards the edges.
 */
export function gvf(grad: ForceField, mu: number, iterations: number, dt = Math.min(1, 0.25 / Math.max(mu, 1e-9)), init?: ForceField): ForceField {
  const { width: w, height: h } = grad, N = w * h;
  const u = Float32Array.from(init ? init.fx : grad.fx), v = Float32Array.from(init ? init.fy : grad.fy);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) b[i] = grad.fx[i] ** 2 + grad.fy[i] ** 2;
  const lu = new Float32Array(N), lv = new Float32Array(N);
  for (let it = 0; it < iterations; it++) {
    for (let y = 0; y < h; y++) {
      const yu = y > 0 ? y - 1 : 0, yd = y < h - 1 ? y + 1 : h - 1;
      for (let x = 0; x < w; x++) {
        const xl = x > 0 ? x - 1 : 0, xr = x < w - 1 ? x + 1 : w - 1, i = y * w + x;
        lu[i] = u[y * w + xl] + u[y * w + xr] + u[yu * w + x] + u[yd * w + x] - 4 * u[i];
        lv[i] = v[y * w + xl] + v[y * w + xr] + v[yu * w + x] + v[yd * w + x] - 4 * v[i];
      }
    }
    for (let i = 0; i < N; i++) {
      u[i] += dt * (mu * lu[i] - b[i] * (u[i] - grad.fx[i]));
      v[i] += dt * (mu * lv[i] - b[i] * (v[i] - grad.fy[i]));
    }
  }
  return { width: w, height: h, fx: u, fy: v };
}

/**
 * GVF computed fast: first on a half-resolution grid (2 × 2 averages of ∇f, which reaches
 * twice as far per iteration), then upsampled and refined with `fineIter` full-resolution
 * iterations. The result points the same way as the full-resolution GVF far from edges
 * and equals it near them.
 */
export function gvfFast(grad: ForceField, mu: number, coarseIter: number, fineIter: number): ForceField {
  const { width: w, height: h } = grad, cw = Math.ceil(w / 2), ch = Math.ceil(h / 2);
  const cfx = new Float32Array(cw * ch), cfy = new Float32Array(cw * ch);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y >> 1) * cw + (x >> 1);
    // per coarse pixel the gradient is twice as large; average of the 4 fine pixels
    cfx[i] += grad.fx[y * w + x] * 0.5; cfy[i] += grad.fy[y * w + x] * 0.5;
  }
  const coarse = gvf({ width: cw, height: ch, fx: cfx, fy: cfy }, mu, coarseIter);
  const ux = new Float32Array(w * h), uy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = (x - 0.5) / 2, Y = (y - 0.5) / 2;
    ux[y * w + x] = sampleGrid(coarse.fx, cw, ch, X, Y) / 2;
    uy[y * w + x] = sampleGrid(coarse.fy, cw, ch, X, Y) / 2;
  }
  return gvf(grad, mu, fineIter, undefined, { width: w, height: h, fx: ux, fy: uy });
}

/** v / √(|v|² + ε²): unit length where |v| ≫ ε, so the far field is as strong as the near field. */
export function normaliseField(F: ForceField, eps: number): ForceField {
  const fx = new Float32Array(F.fx.length), fy = new Float32Array(F.fy.length);
  for (let i = 0; i < fx.length; i++) { const k = 1 / Math.sqrt(F.fx[i] ** 2 + F.fy[i] ** 2 + eps * eps); fx[i] = F.fx[i] * k; fy[i] = F.fy[i] * k; }
  return { width: F.width, height: F.height, fx, fy };
}

/** One explicit GVF iteration on a 1D signal (for the by-hand check): v ← v + Δt[μ v'' − g²(v − g)]. */
export function gvf1dStep(v: ArrayLike<number>, g: ArrayLike<number>, mu: number, dt: number): Float64Array {
  const n = v.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const l = v[Math.max(0, i - 1)], r = v[Math.min(n - 1, i + 1)];
    out[i] = v[i] + dt * (mu * (l - 2 * v[i] + r) - g[i] * g[i] * (v[i] - g[i]));
  }
  return out;
}

/** Bilinear sample of a scalar grid at (x, y) (replicated border). */
export function sampleGrid(data: ArrayLike<number>, w: number, h: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(w - 1, x)), cy = Math.max(0, Math.min(h - 1, y));
  const x0 = Math.min(w - 2, Math.floor(cx)), y0 = Math.min(h - 2, Math.floor(cy));
  const fx = cx - x0, fy = cy - y0, i = y0 * w + x0;
  return (1 - fy) * ((1 - fx) * data[i] + fx * data[i + 1]) + fy * ((1 - fx) * data[i + w] + fx * data[i + w + 1]);
}

/** The field sampled bilinearly at every contour point, times `weight`. */
export function sampleField(F: ForceField, c: Contour, weight = 1): Contour {
  const n = c.x.length, x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = weight * sampleGrid(F.fx, F.width, F.height, c.x[i], c.y[i]);
    y[i] = weight * sampleGrid(F.fy, F.width, F.height, c.x[i], c.y[i]);
  }
  return { x, y };
}

/** External energy Σ_i −w f(x_i) for an edge map f. */
export function externalEnergy(f: Grey, c: Contour, weight = 1): number {
  let s = 0;
  for (let i = 0; i < c.x.length; i++) s -= weight * sampleGrid(f.data, f.width, f.height, c.x[i], c.y[i]);
  return s;
}

/** Add contours point-wise (forces). */
export function addForces(...fs: (Contour | null)[]): Contour | null {
  const g = fs.filter((f): f is Contour => f !== null);
  if (!g.length) return null;
  const out = cloneContour(g[0]);
  for (const f of g.slice(1)) for (let i = 0; i < out.x.length; i++) { out.x[i] += f.x[i]; out.y[i] += f.y[i]; }
  return out;
}

// ---------------------------------------------------------------------------------------
// Fourier modes of a contour (the eigenvectors of A)

/**
 * Amplitude of each frequency k = 0..⌊n/2⌋ in the contour: the DFT of x and of y, with the
 * coefficients of k and n − k combined, so a pure mode a cos(2πki/n) has amplitude |a|
 * (frequency 0 is the centroid). Under the internal force alone, the amplitude of mode k is
 * multiplied by the amplification factor of λ_k at every step.
 */
export function modeAmplitudes(c: Contour): Float64Array {
  const n = c.x.length, K = Math.floor(n / 2), out = new Float64Array(K + 1);
  const pw = (v: Float64Array, k: number) => {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) { const a = 2 * Math.PI * k * i / n; re += v[i] * Math.cos(a); im -= v[i] * Math.sin(a); }
    return re * re + im * im;
  };
  for (let k = 0; k <= K; k++) {
    const both = pw(c.x, k) + pw(c.y, k);
    // |X_k|² + |X_{n−k}|² = 2|X_k|² for 0 < k < n/2; a cos mode of amplitude a gives |X_k| = na/2
    out[k] = (k === 0 || 2 * k === n) ? Math.sqrt(both) / n : 2 * Math.sqrt(both) / n;
  }
  return out;
}
