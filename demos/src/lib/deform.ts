// Non-rigid registration with a cubic B-spline free-form deformation (no DOM; runs under node).
// Owner: the optical flow day, unit nonrigid-reg/ (demos ffdreg, jointhist). The splines day
// (planned) owns spline theory: when its lib/spline.ts exists, bspline3/bspline3d should move
// there and be re-exported from here.
//
// Conventions: images are Grey with values 0..1 for registration; the deformation d is a
// backward map: the registered image is W(x) = M(x + d(x)) (moving M sampled where the fixed
// pixel x "comes from"). Control point (k, l) sits at (x0 + kΔ, y0 + lΔ) with x0 = y0 = −Δ, and
//   d(x, y) = Σ_kl c_kl β³((x − x0)/Δ − k) β³((y − y0)/Δ − l).
import { Grey, createGrey, fromFunction, sampleBilinear, gaussianBlur, gaussianDerivative } from "./imageops";
import { rng } from "./random";

/** Cubic B-spline: 2/3 − t² + |t|³/2 for |t| < 1, (2 − |t|)³/6 for 1 ≤ |t| < 2, else 0. */
export function bspline3(t: number): number {
  const a = Math.abs(t);
  if (a < 1) return 2 / 3 - a * a + 0.5 * a * a * a;
  if (a < 2) { const b = 2 - a; return b * b * b / 6; }
  return 0;
}

/** Derivative of the cubic B-spline. */
export function bspline3d(t: number): number {
  const a = Math.abs(t), s = t < 0 ? -1 : 1;
  if (a < 1) return s * (-2 * a + 1.5 * a * a);
  if (a < 2) { const b = 2 - a; return -s * 0.5 * b * b; }
  return 0;
}

/** A control grid with x and y coefficients (row-major ny × nx). */
export interface FFDGrid { nx: number; ny: number; spacing: number; x0: number; y0: number; cx: Float64Array; cy: Float64Array }

/** A zero grid covering a w × h image with control points every `spacing` px (one extra ring outside). */
export function makeGrid(w: number, h: number, spacing: number): FFDGrid {
  const nx = Math.floor((w - 1) / spacing) + 4, ny = Math.floor((h - 1) / spacing) + 4;
  return { nx, ny, spacing, x0: -spacing, y0: -spacing, cx: new Float64Array(nx * ny), cy: new Float64Array(nx * ny) };
}

export function cloneGrid(g: FFDGrid): FFDGrid {
  return { ...g, cx: Float64Array.from(g.cx), cy: Float64Array.from(g.cy) };
}

/** Position of control point (k, l) in pixels. */
export const knotPos = (g: FFDGrid, k: number, l: number): [number, number] => [g.x0 + k * g.spacing, g.y0 + l * g.spacing];

/** Per-pixel B-spline weights along one axis: first control index k0 and four weights. */
interface AxisWeights { k0: Int32Array; wt: Float64Array }
function axisWeights(n: number, origin: number, spacing: number, nk: number): AxisWeights {
  const k0 = new Int32Array(n), wt = new Float64Array(4 * n);
  for (let x = 0; x < n; x++) {
    const t = (x - origin) / spacing, f = Math.floor(t) - 1;
    k0[x] = f;
    for (let j = 0; j < 4; j++) {
      const k = f + j;
      wt[4 * x + j] = k >= 0 && k < nk ? bspline3(t - k) : 0;
    }
  }
  return { k0, wt };
}

/** The displacement field (u, v) of a grid on a w × h image. */
export function ffdDisplacement(g: FFDGrid, w: number, h: number): { u: Float32Array; v: Float32Array } {
  const ax = axisWeights(w, g.x0, g.spacing, g.nx), ay = axisWeights(h, g.y0, g.spacing, g.ny);
  const u = new Float32Array(w * h), v = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const l0 = ay.k0[y];
    for (let x = 0; x < w; x++) {
      const k0 = ax.k0[x];
      let su = 0, sv = 0;
      for (let j = 0; j < 4; j++) {
        const wy = ay.wt[4 * y + j]; if (!wy) continue;
        const row = (l0 + j) * g.nx;
        for (let i = 0; i < 4; i++) {
          const wx = ax.wt[4 * x + i]; if (!wx) continue;
          const c = row + k0 + i, ww = wx * wy;
          su += ww * g.cx[c]; sv += ww * g.cy[c];
        }
      }
      u[y * w + x] = su; v[y * w + x] = sv;
    }
  }
  return { u, v };
}

/** Transpose of ffdDisplacement: gradient wrt the coefficients of Σ_x gu(x) u(x) + gv(x) v(x). */
export function ffdAdjoint(g: FFDGrid, w: number, h: number, gu: ArrayLike<number>, gv: ArrayLike<number>): { cx: Float64Array; cy: Float64Array } {
  const ax = axisWeights(w, g.x0, g.spacing, g.nx), ay = axisWeights(h, g.y0, g.spacing, g.ny);
  const cx = new Float64Array(g.nx * g.ny), cy = new Float64Array(g.nx * g.ny);
  for (let y = 0; y < h; y++) {
    const l0 = ay.k0[y];
    for (let x = 0; x < w; x++) {
      const k0 = ax.k0[x], p = y * w + x, a = gu[p], b = gv[p];
      if (!a && !b) continue;
      for (let j = 0; j < 4; j++) {
        const wy = ay.wt[4 * y + j]; if (!wy) continue;
        const row = (l0 + j) * g.nx;
        for (let i = 0; i < 4; i++) {
          const wx = ax.wt[4 * x + i]; if (!wx) continue;
          const c = row + k0 + i, ww = wx * wy;
          cx[c] += ww * a; cy[c] += ww * b;
        }
      }
    }
  }
  return { cx, cy };
}

/** Halve the grid spacing without changing the deformation (cubic B-spline subdivision, per axis). */
export function refineGrid(g: FFDGrid, w: number, h: number): FFDGrid {
  const f = makeGrid(w, h, g.spacing / 2);
  // 1D rule (old knot i at (i−1)Δ, new knot j at (j−1)Δ/2): j = 2i − 1 → (c_{i−1} + 6c_i + c_{i+1})/8; j = 2i → (c_i + c_{i+1})/2
  const sub = (src: Float64Array, n: number, stride: number, off: number, m: number): Float64Array => {
    const at = (i: number) => src[off + Math.max(0, Math.min(n - 1, i)) * stride];
    const out = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      if (j % 2 === 1) { const i = (j + 1) / 2; out[j] = (at(i - 1) + 6 * at(i) + at(i + 1)) / 8; }
      else { const i = j / 2; out[j] = (at(i) + at(i + 1)) / 2; }
    }
    return out;
  };
  for (const key of ["cx", "cy"] as const) {
    // rows first (x direction), then columns
    const tmp = new Float64Array(g.ny * f.nx);
    for (let l = 0; l < g.ny; l++) tmp.set(sub(g[key], g.nx, 1, l * g.nx, f.nx), l * f.nx);
    for (let k = 0; k < f.nx; k++) {
      const col = sub(tmp, g.ny, f.nx, k, f.ny);
      for (let l = 0; l < f.ny; l++) f[key][l * f.nx + k] = col[l];
    }
  }
  return f;
}

/** Determinant of the Jacobian of x ↦ x + d(x) (central differences): > 1 expansion, < 1 compression, ≤ 0 folding. */
export function jacobianDeterminant(u: ArrayLike<number>, v: ArrayLike<number>, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const d = (a: ArrayLike<number>, x: number, y: number, ax: 0 | 1): number => {
    const i = y * w + x;
    if (ax === 0) return w < 2 ? 0 : x === 0 ? a[i + 1] - a[i] : x === w - 1 ? a[i] - a[i - 1] : (a[i + 1] - a[i - 1]) / 2;
    return h < 2 ? 0 : y === 0 ? a[i + w] - a[i] : y === h - 1 ? a[i] - a[i - w] : (a[i + w] - a[i - w]) / 2;
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out[y * w + x] = (1 + d(u, x, y, 0)) * (1 + d(v, x, y, 1)) - d(u, x, y, 1) * d(v, x, y, 0);
  return out;
}

/**
 * Bending penalty of the grid: Σ over coefficients of the squared second differences
 * (Δ²_k c)² + (Δ²_l c)² + 2(Δ_k Δ_l c)², both components, divided by Δ² · (image area) so
 * that it approximates the mean of |∂²d|² per pixel. Returns the value and its gradient.
 */
export function bendingEnergy(g: FFDGrid, w: number, h: number): { value: number; gx: Float64Array; gy: Float64Array } {
  const { nx, ny } = g, s = 1 / (g.spacing * g.spacing * w * h);
  const gx = new Float64Array(nx * ny), gy = new Float64Array(nx * ny);
  let value = 0;
  for (const [c, gr] of [[g.cx, gx], [g.cy, gy]] as const) {
    for (let l = 0; l < ny; l++) for (let k = 0; k < nx; k++) {
      const i = l * nx + k;
      if (k > 0 && k < nx - 1) { const e = c[i - 1] - 2 * c[i] + c[i + 1]; value += s * e * e; gr[i - 1] += 2 * s * e; gr[i] -= 4 * s * e; gr[i + 1] += 2 * s * e; }
      if (l > 0 && l < ny - 1) { const e = c[i - nx] - 2 * c[i] + c[i + nx]; value += s * e * e; gr[i - nx] += 2 * s * e; gr[i] -= 4 * s * e; gr[i + nx] += 2 * s * e; }
      if (k < nx - 1 && l < ny - 1) {
        const e = c[i + nx + 1] - c[i + 1] - c[i + nx] + c[i];
        value += 2 * s * e * e;
        gr[i + nx + 1] += 4 * s * e; gr[i + 1] -= 4 * s * e; gr[i + nx] -= 4 * s * e; gr[i] += 4 * s * e;
      }
    }
  }
  return { value, gx, gy };
}

// ---------------------------------------------------------------------------------------
// Similarity measures (as costs to minimise) with their gradients wrt the warped image

export type Measure = "ssd" | "ncc" | "mi";

export interface CostResult { cost: number; grad: Float64Array; score: number }

/**
 * Cost of the warped image W against the fixed F, with dCost/dW per pixel.
 *  ssd: mean (W − F)² / var(F)   (score = mean squared difference)
 *  ncc: 1 − NCC                  (score = NCC)
 *  mi:  −MI / H(F)              (score = MI in bits). F binned hard, W binned linearly
 *       (Parzen-like), so MI is differentiable in W.
 * Values are expected in 0..1.
 */
export function similarityCost(F: ArrayLike<number>, W: ArrayLike<number>, measure: Measure, bins = 24): CostResult {
  const n = F.length, grad = new Float64Array(n);
  if (measure === "ssd") {
    let m = 0; for (let i = 0; i < n; i++) m += F[i]; m /= n;
    let vf = 0; for (let i = 0; i < n; i++) vf += (F[i] - m) ** 2; vf = vf / n || 1;
    let s = 0;
    for (let i = 0; i < n; i++) { const d = W[i] - F[i]; s += d * d; grad[i] = 2 * d / (n * vf); }
    return { cost: s / n / vf, grad, score: s / n };
  }
  if (measure === "ncc") {
    let mf = 0, mw = 0; for (let i = 0; i < n; i++) { mf += F[i]; mw += W[i]; } mf /= n; mw /= n;
    let sff = 0, sww = 0, sfw = 0;
    for (let i = 0; i < n; i++) { const a = F[i] - mf, b = W[i] - mw; sff += a * a; sww += b * b; sfw += a * b; }
    const sf = Math.sqrt(sff / n) || 1e-12, sw = Math.sqrt(sww / n) || 1e-12, ncc = sfw / (n * sf * sw);
    for (let i = 0; i < n; i++) grad[i] = -((F[i] - mf) / sf - ncc * (W[i] - mw) / sw) / (n * sw);
    return { cost: 1 - ncc, grad, score: ncc };
  }
  // mutual information with linear binning of W
  const B = bins, P = new Float64Array(B * B), a = new Int32Array(n), b0 = new Int32Array(n), fr = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.max(0, Math.min(B - 1, Math.floor(F[i] * B)));
    let t = Math.max(0, Math.min(B - 1, W[i] * (B - 1))), k = Math.floor(t);
    if (k >= B - 1) k = B - 2;
    b0[i] = k; fr[i] = t - k;
    P[a[i] * B + k] += (1 - fr[i]) / n; P[a[i] * B + k + 1] += fr[i] / n;
  }
  const pa = new Float64Array(B), pb = new Float64Array(B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) { pa[i] += P[i * B + j]; pb[j] += P[i * B + j]; }
  let ha = 0, mi = 0;
  for (let i = 0; i < B; i++) if (pa[i] > 0) ha -= pa[i] * Math.log2(pa[i]);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) { const p = P[i * B + j]; if (p > 0) mi += p * Math.log2(p / (pa[i] * pb[j])); }
  const eps = 1e-12, L = (i: number, j: number) => Math.log2((P[i * B + j] + eps) / (pb[j] + eps));
  const scale = (B - 1) / (n * (ha || 1));
  for (let i = 0; i < n; i++) grad[i] = -scale * (L(a[i], b0[i] + 1) - L(a[i], b0[i]));
  return { cost: -mi / (ha || 1), grad, score: mi };
}

/** Mutual information (bits) from hard-binned values in [0, 1]: also H(A), H(B), H(A, B) and the joint counts (rows = A). */
export function mutualInformation(A: ArrayLike<number>, Bv: ArrayLike<number>, bins: number, mask?: ArrayLike<number>):
  { mi: number; ha: number; hb: number; hab: number; counts: Float64Array; n: number } {
  const counts = new Float64Array(bins * bins);
  let n = 0;
  for (let i = 0; i < A.length; i++) {
    if (mask && !mask[i]) continue;
    const a = Math.max(0, Math.min(bins - 1, Math.floor(A[i] * bins))), b = Math.max(0, Math.min(bins - 1, Math.floor(Bv[i] * bins)));
    counts[a * bins + b]++; n++;
  }
  const H = (c: ArrayLike<number>) => { let h = 0; for (let i = 0; i < c.length; i++) if (c[i] > 0) { const p = c[i] / n; h -= p * Math.log2(p); } return h; };
  const ra = new Float64Array(bins), rb = new Float64Array(bins);
  for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) { ra[i] += counts[i * bins + j]; rb[j] += counts[i * bins + j]; }
  const ha = H(ra), hb = H(rb), hab = H(counts);
  return { mi: n ? ha + hb - hab : 0, ha, hb, hab, counts, n };
}

/** Mean squared difference and NCC over a mask. */
export function ssdNcc(A: ArrayLike<number>, B: ArrayLike<number>, mask?: ArrayLike<number>): { ssd: number; ncc: number; n: number } {
  let n = 0, ma = 0, mb = 0;
  for (let i = 0; i < A.length; i++) if (!mask || mask[i]) { n++; ma += A[i]; mb += B[i]; }
  if (!n) return { ssd: NaN, ncc: NaN, n };
  ma /= n; mb /= n;
  let saa = 0, sbb = 0, sab = 0, ss = 0;
  for (let i = 0; i < A.length; i++) if (!mask || mask[i]) {
    const a = A[i] - ma, b = B[i] - mb;
    saa += a * a; sbb += b * b; sab += a * b; ss += (A[i] - B[i]) ** 2;
  }
  return { ssd: ss / n, ncc: saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0, n };
}

// ---------------------------------------------------------------------------------------
// Intensity remaps ("a second stain") and test deformations

export type Remap = "identity" | "gain" | "inverted" | "nonmonotone";
export const REMAP_NAMES: Record<Remap, string> = { identity: "same contrast", gain: "gain + offset", inverted: "inverted", nonmonotone: "non-monotone" };

/** Map a value in 0..1: identity, gain + offset (0.25 + 0.5 v), inverted (1 − v), non-monotone (a hump: mid-grey bright, both ends dark). */
export function remap(v: number, kind: Remap): number {
  switch (kind) {
    case "identity": return v;
    case "gain": return 0.25 + 0.5 * v;
    case "inverted": return 1 - v;
    case "nonmonotone": { const t = Math.max(0, Math.min(1, v)); return 1 - Math.abs(2 * t - 1) ** 1.2; }
  }
}

/** Seeded smooth random deformation: a grid with spacing `spacing` and Gaussian coefficients of sd `amp` px (outer ring zero). */
export function randomGrid(w: number, h: number, spacing: number, amp: number, seed = 1): FFDGrid {
  const g = makeGrid(w, h, spacing), R = rng(seed);
  for (let l = 0; l < g.ny; l++) for (let k = 0; k < g.nx; k++) {
    const i = l * g.nx + k, a = R.norm(), b = R.norm();
    if (k === 0 || l === 0 || k === g.nx - 1 || l === g.ny - 1) continue;
    g.cx[i] = amp * a; g.cy[i] = amp * b;
  }
  return g;
}

/**
 * The moving image for a known deformation: M(x + d(x)) = F(x), i.e. M(y) = F(y − g(y)) with
 * g(y) = d(y − g(y)) solved by fixed-point steps. Then registering with d recovers F exactly.
 */
export function deformForTruth(F: Grey, u: ArrayLike<number>, v: ArrayLike<number>, iters = 15): Grey {
  const w = F.width, h = F.height, U = { width: w, height: h, data: u } as Grey, V = { width: w, height: h, data: v } as Grey;
  return fromFunction(w, h, (x, y) => {
    let gx = 0, gy = 0;
    for (let k = 0; k < iters; k++) { gx = sampleBilinear(U, x - gx, y - gy); gy = sampleBilinear(V, x - gx, y - gy); }
    return sampleBilinear(F, x - gx, y - gy);
  });
}

/** Warp the moving image by a displacement field: W(x) = M(x + d(x)). */
export function warpMoving(M: Grey, u: ArrayLike<number>, v: ArrayLike<number>): Grey {
  const w = M.width, h = M.height, out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x; out.data[i] = sampleBilinear(M, x + u[i], y + v[i]); }
  return out;
}

// ---------------------------------------------------------------------------------------
// Registration by gradient descent

export interface RegOptions {
  measure: Measure;
  /** Weight of the bending penalty. */
  lambda: number;
  /** Final control-point spacing in px. */
  spacing: number;
  /** Start with a grid this many times coarser (1, 2 or 4) and refine by halving; images are blurred more on coarse grids. */
  coarse: number;
  bins?: number;
  /** Image blur (σ, px) at the finest grid. */
  blur?: number;
  /** Refine when the step size falls below this many px (or after `maxPerLevel` steps). */
  minStep?: number;
  maxPerLevel?: number;
}

/**
 * Gradient descent on cost = similarity + λ·bending over the control coefficients. Each step
 * moves the coefficients along the negative gradient so that the largest change is `step` px;
 * the step grows by 1.2 after a success and halves after a failure (then the old grid is kept).
 */
export class FFDRegistration {
  grid: FFDGrid;
  step = 1;
  iterations = 0;
  history: number[] = [];
  level: number;
  private F!: Grey; private M!: Grey; private Mx!: Grey; private My!: Grey;
  private sinceRefine = 0;
  current: { cost: number; sim: number; bend: number; score: number; u: Float32Array; v: Float32Array; W: Grey } | null = null;

  constructor(readonly fixed: Grey, readonly moving: Grey, public o: RegOptions, start?: FFDGrid) {
    this.level = start ? 1 : Math.max(1, o.coarse);
    this.grid = start ? cloneGrid(start) : makeGrid(fixed.width, fixed.height, o.spacing * this.level);
    this.prepare();
  }

  /** Blur for the current grid level. */
  private prepare(): void {
    const s = (this.o.blur ?? 1) * this.level;
    this.F = gaussianBlur(this.fixed, s); this.M = gaussianBlur(this.moving, s);
    this.Mx = gaussianDerivative(this.M, 0.7, 1, 0); this.My = gaussianDerivative(this.M, 0.7, 0, 1);
    this.current = this.evaluate(this.grid);
  }

  /** Cost of a grid on the current (blurred) images. */
  evaluate(g: FFDGrid) {
    const w = this.fixed.width, h = this.fixed.height;
    const { u, v } = ffdDisplacement(g, w, h), W = warpMoving(this.M, u, v);
    const sim = similarityCost(this.F.data, W.data, this.o.measure, this.o.bins ?? 24);
    const bend = bendingEnergy(g, w, h);
    return { cost: sim.cost + this.o.lambda * bend.value, sim: sim.cost, bend: bend.value, score: sim.score, u, v, W, simGrad: sim.grad, bendGrad: bend };
  }

  /** One descent step. Returns true while it can still make progress. */
  iterate(): boolean {
    const w = this.fixed.width, h = this.fixed.height;
    const cur = this.evaluate(this.grid);
    // dCost/dc = Σ_x dCost/dW(x) ∇M(x + d(x)) β_kl(x) + λ dBend/dc
    const gu = new Float64Array(w * h), gv = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, px = x + cur.u[i], py = y + cur.v[i];
      gu[i] = cur.simGrad[i] * sampleBilinear(this.Mx, px, py);
      gv[i] = cur.simGrad[i] * sampleBilinear(this.My, px, py);
    }
    const G = ffdAdjoint(this.grid, w, h, gu, gv);
    let gmax = 0;
    for (let i = 0; i < G.cx.length; i++) {
      G.cx[i] += this.o.lambda * cur.bendGrad.gx[i]; G.cy[i] += this.o.lambda * cur.bendGrad.gy[i];
      gmax = Math.max(gmax, Math.abs(G.cx[i]), Math.abs(G.cy[i]));
    }
    this.iterations++; this.sinceRefine++;
    if (gmax > 0) {
      const trial = cloneGrid(this.grid), k = this.step / gmax;
      for (let i = 0; i < G.cx.length; i++) { trial.cx[i] -= k * G.cx[i]; trial.cy[i] -= k * G.cy[i]; }
      const t = this.evaluate(trial);
      if (t.cost < cur.cost) { this.grid = trial; this.step = Math.min(4, this.step * 1.2); this.current = t; }
      else { this.step *= 0.5; this.current = cur; }
    }
    this.history.push(this.current!.cost);
    const stuck = this.step < (this.o.minStep ?? 0.02) || this.sinceRefine >= (this.o.maxPerLevel ?? 60);
    if (stuck && this.level > 1) {
      this.grid = refineGrid(this.grid, w, h); this.level /= 2; this.step = 1; this.sinceRefine = 0; this.prepare();
      return true;
    }
    return !stuck;
  }
}

/** Mean |d − d_true| over pixels at least `margin` from the border. */
export function displacementError(u: ArrayLike<number>, v: ArrayLike<number>, ut: ArrayLike<number>, vt: ArrayLike<number>, w: number, h: number, margin = 8): number {
  let s = 0, n = 0;
  for (let y = margin; y < h - margin; y++) for (let x = margin; x < w - margin; x++) { const i = y * w + x; s += Math.hypot(u[i] - ut[i], v[i] - vt[i]); n++; }
  return n ? s / n : NaN;
}
