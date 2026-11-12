// Image moments to any order (owner: the moments elective; units moments-basis/,
// moment-invariants/, orthogonal-moments/, zernike/). No DOM.
//
// One object (or one grey image) at a time, to any order, in float64. lib/label.ts keeps
// its one-scan accumulation to order 2 for all labels at once; the order-2 results here
// agree with it (demos/test/moments-test.ts).
//
// Conventions
// - Images are flat row-major arrays f[y * w + x] (masks or grey values), pixel (x, y) has
//   its centre at the integer coordinates (x, y), x right, y DOWN.
// - A moment table of order N is a Float64Array of size (N+1)², entry [p * (N+1) + q] holds
//   the moment of x^p y^q. All p, q ≤ N are filled (the separable sums give them for free);
//   only p + q ≤ N are "moments up to order N".
// - A `Frame` maps pixel indices to the coordinates the moments use:
//   u = (x − x0)·sx, v = (y − y0)·sy. `pixelFrame` is the identity (pixel units, origin at
//   the top-left pixel centre), `unitFrame(w, h)` maps the image onto [−1, 1]² (pixel
//   centres at ±(1 − 1/w), …), `yUp` flips v so that angles turn counter-clockwise on screen.
// - Pixel as a point ("point": f(x, y)·u^p v^q) or as a unit square ("square": the average of
//   u^p v^q over the pixel). In pixel units, "square" adds m00/12 to μ20 and μ02.
// - With `area: true`, sums are multiplied by the pixel area sx·sy (Riemann sums of the
//   integrals); without, they are plain sums over pixels (the labelling day's m_pq).

export type Table = Float64Array;

export interface Frame { x0: number; y0: number; sx: number; sy: number }

export const pixelFrame: Frame = { x0: 0, y0: 0, sx: 1, sy: 1 };

/** Map pixel centres onto [−1, 1]² (the square for Legendre moments, the disc for Zernike). */
export function unitFrame(w: number, h: number, yUp = false): Frame {
  return { x0: (w - 1) / 2, y0: (h - 1) / 2, sx: 2 / w, sy: (yUp ? -2 : 2) / h };
}

/** Map the image onto [0, 1]² (pixel centres at (i + ½)/n): the badly conditioned choice. */
export function zeroOneFrame(w: number, h: number): Frame {
  return { x0: -0.5, y0: -0.5, sx: 1 / w, sy: 1 / h };
}

export type PixelModel = "point" | "square";

export interface MomentOptions {
  frame?: Frame;
  pixel?: PixelModel;
  area?: boolean;
}

export const idx = (N: number, p: number, q: number): number => p * (N + 1) + q;
export const at = (t: Table, N: number, p: number, q: number): number => t[p * (N + 1) + q];

/** Number of moments of order ≤ N: (N+1)(N+2)/2. */
export const countUpTo = (N: number): number => (N + 1) * (N + 2) / 2;

/** Binomial coefficient (exact for the small n used here). */
export function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 1; i <= k; i++) r = r * (n - k + i) / i;
  return Math.round(r);
}

// ---------------------------------------------------------------------------------------
// Separable sums

/** Rows of a 1D basis sampled on n pixels: B[k] is a Float64Array of length n. */
export type Basis1D = Float64Array[];

/** Powers u^k (point) or the pixel averages of u^k (square), k = 0..N, for pixel indices
 *  0..n−1 with u = (i − c)·s. */
export function powerBasis(n: number, N: number, c: number, s: number, pixel: PixelModel = "point"): Basis1D {
  const B: Basis1D = [];
  for (let k = 0; k <= N; k++) B.push(new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const u = (i - c) * s;
    if (pixel === "point") {
      let v = 1;
      for (let k = 0; k <= N; k++) { B[k][i] = v; v *= u; }
    } else {
      // average of t^k over [u − |s|/2, u + |s|/2] = (b^{k+1} − a^{k+1}) / ((k+1)(b − a))
      const a = u - Math.abs(s) / 2, b = u + Math.abs(s) / 2;
      let pa = a, pb = b;
      for (let k = 0; k <= N; k++) { B[k][i] = (pb - pa) / ((k + 1) * (b - a)); pa *= a; pb *= b; }
    }
  }
  return B;
}

/** T[p, q] = Σ_y Σ_x Bx[p][x] · By[q][y] · f(x, y), for p ≤ Nx, q ≤ Ny; returned as a
 *  (Nx+1) × (Ny+1) row-major table. Cost O(Nx·w·h + Nx·Ny·h). */
export function separableSums(f: ArrayLike<number>, w: number, h: number, Bx: Basis1D, By: Basis1D): Float64Array {
  const Nx = Bx.length - 1, Ny = By.length - 1;
  // row sums: R[p][y] = Σ_x Bx[p][x] f(x, y)
  const R = new Float64Array((Nx + 1) * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let p = 0; p <= Nx; p++) {
      const b = Bx[p];
      let s = 0;
      for (let x = 0; x < w; x++) { const v = f[o + x]; if (v) s += b[x] * v; }
      R[p * h + y] = s;
    }
  }
  const T = new Float64Array((Nx + 1) * (Ny + 1));
  for (let p = 0; p <= Nx; p++) for (let q = 0; q <= Ny; q++) {
    const b = By[q];
    let s = 0;
    for (let y = 0; y < h; y++) s += R[p * h + y] * b[y];
    T[p * (Ny + 1) + q] = s;
  }
  return T;
}

/** Raw (geometric) moments m_pq, p, q ≤ N, of an image or mask. */
export function rawMoments(f: ArrayLike<number>, w: number, h: number, N: number, o: MomentOptions = {}): Table {
  const fr = o.frame ?? pixelFrame, px = o.pixel ?? "point";
  const T = separableSums(f, w, h, powerBasis(w, N, fr.x0, fr.sx, px), powerBasis(h, N, fr.y0, fr.sy, px));
  if (o.area) { const a = Math.abs(fr.sx * fr.sy); for (let i = 0; i < T.length; i++) T[i] *= a; }
  return T;
}

/** Central moments computed directly (coordinates relative to the centroid). */
export function centralMomentsDirect(f: ArrayLike<number>, w: number, h: number, N: number, o: MomentOptions = {}): Table {
  const fr = o.frame ?? pixelFrame;
  const m = rawMoments(f, w, h, 1, { ...o, pixel: "point" });
  const cx = m[idx(1, 1, 0)] / m[0], cy = m[idx(1, 0, 1)] / m[0];
  // centroid in pixel indices: u = (x − x0)·sx = cx  →  x = x0 + cx/sx
  return rawMoments(f, w, h, N, { ...o, frame: { x0: fr.x0 + cx / fr.sx, y0: fr.y0 + cy / fr.sy, sx: fr.sx, sy: fr.sy } });
}

/** Centroid (x̄, ȳ) = (m10/m00, m01/m00) of a raw table. */
export function centroidOf(m: Table, N: number): [number, number] {
  return [m[idx(N, 1, 0)] / m[0], m[idx(N, 0, 1)] / m[0]];
}

/** Central moments from raw moments by the binomial theorem (no second pass):
 *  μ_pq = Σ_k Σ_l C(p,k) C(q,l) (−x̄)^{p−k} (−ȳ)^{q−l} m_kl. */
export function centralFromRaw(m: Table, N: number): Table {
  const [cx, cy] = centroidOf(m, N);
  return shiftMoments(m, N, -cx, -cy);
}

/** Moments of the image translated by (dx, dy): Σ (u+dx)^p (v+dy)^q f. */
export function shiftMoments(m: Table, N: number, dx: number, dy: number): Table {
  const out = new Float64Array((N + 1) * (N + 1));
  for (let p = 0; p <= N; p++) for (let q = 0; q <= N; q++) {
    let s = 0;
    for (let k = 0; k <= p; k++) for (let l = 0; l <= q; l++)
      s += binom(p, k) * binom(q, l) * dx ** (p - k) * dy ** (q - l) * m[idx(N, k, l)];
    out[idx(N, p, q)] = s;
  }
  return out;
}

/** Normalised central moments η_pq = μ_pq / μ00^{1 + (p+q)/2} (scale invariant; for grey
 *  images also invariant to a contrast factor a·f). Entries with p + q < 2 are left as μ/μ00^…
 *  (η00 = 1, η10 = η01 = 0). */
export function normalisedMoments(mu: Table, N: number): Table {
  const out = new Float64Array(mu.length), m00 = mu[0];
  for (let p = 0; p <= N; p++) for (let q = 0; q <= N; q++) out[idx(N, p, q)] = mu[idx(N, p, q)] / m00 ** (1 + (p + q) / 2);
  return out;
}

/** Moments after the linear map u' = a u + b v, v' = c u + d v (up to the Jacobian: the
 *  result is Σ u'^p v'^q f, i.e. the moments of the point masses moved by the map). */
export function linearMapMoments(t: Table, N: number, a: number, b: number, c: number, d: number): Table {
  const out = new Float64Array((N + 1) * (N + 1));
  for (let p = 0; p <= N; p++) for (let q = 0; q <= N; q++) {
    if (p + q > N) continue;
    // (a u + b v)^p (c u + d v)^q = Σ_i Σ_j C(p,i) a^i b^{p−i} u^i v^{p−i} · C(q,j) c^j d^{q−j} u^j v^{q−j}
    let s = 0;
    for (let i = 0; i <= p; i++) for (let j = 0; j <= q; j++)
      s += binom(p, i) * binom(q, j) * a ** i * b ** (p - i) * c ** j * d ** (q - j) * t[idx(N, i + j, p + q - i - j)];
    out[idx(N, p, q)] = s;
  }
  return out;
}

/** Rotate the coordinates by α (u' = u cos α − v sin α, v' = u sin α + v cos α): the
 *  moments of the image rotated by α in the (u, v) frame (counter-clockwise if v points up). */
export function rotateMoments(t: Table, N: number, alpha: number): Table {
  const c = Math.cos(alpha), s = Math.sin(alpha);
  return linearMapMoments(t, N, c, -s, s, c);
}

/** Flip v → −v (switch between y down and y up): m_pq → (−1)^q m_pq. */
export function flipV(t: Table, N: number): Table {
  const out = t.slice();
  for (let p = 0; p <= N; p++) for (let q = 1; q <= N; q += 2) out[idx(N, p, q)] = -out[idx(N, p, q)];
  return out;
}

// ---------------------------------------------------------------------------------------
// Polygons (Green's theorem)

/** Exact moments ∬ x^p y^q dx dy of the region inside a simple polygon (absolute value of the
 *  signed result, so either orientation works), p + q ≤ N. Green's theorem turns the area
 *  integral into a sum over the edges (Steger 1996):
 *  m_pq = 1/((p+q+2)(p+q+1)C(p+q,p)) Σ_i (x_i y_{i+1} − x_{i+1} y_i)
 *         Σ_k Σ_l C(k+l, l) C(p+q−k−l, q−l) x_i^k x_{i+1}^{p−k} y_i^l y_{i+1}^{q−l}. */
export function polygonMoments(poly: ArrayLike<[number, number]> | [number, number][], N: number): Table {
  const P = poly as [number, number][], n = P.length, out = new Float64Array((N + 1) * (N + 1));
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const [x0, y0] = P[i], [x1, y1] = P[(i + 1) % n], cr = x0 * y1 - x1 * y0;
      let inner = 0;
      for (let k = 0; k <= p; k++) for (let l = 0; l <= q; l++)
        inner += binom(k + l, l) * binom(p + q - k - l, q - l) * x0 ** k * x1 ** (p - k) * y0 ** l * y1 ** (q - l);
      s += cr * inner;
    }
    out[idx(N, p, q)] = s / ((p + q + 2) * (p + q + 1) * binom(p + q, p));
  }
  if (out[0] < 0) for (let i = 0; i < out.length; i++) out[i] = -out[i];
  return out;
}

// ---------------------------------------------------------------------------------------
// Complex moments and rotation invariants

export interface Complex { re: number; im: number }

/** Complex moments c_pq = Σ (u + iv)^p (u − iv)^q f, p, q ≤ N with p + q ≤ N, from a real
 *  table (raw, central or normalised). Rotating the image by α (counter-clockwise in the
 *  (u, v) frame) multiplies c_pq by e^{i(p−q)α}; a reflection v → −v conjugates it. */
export function complexMoments(t: Table, N: number): { re: Table; im: Table } {
  const re = new Float64Array((N + 1) * (N + 1)), im = new Float64Array((N + 1) * (N + 1));
  // i^k (−i)^l = i^{k − l}·(−1)^l ... computed as a unit complex number
  const ipow = (e: number): [number, number] => { const r = ((e % 4) + 4) % 4; return r === 0 ? [1, 0] : r === 1 ? [0, 1] : r === 2 ? [-1, 0] : [0, -1]; };
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) {
    let sr = 0, si = 0;
    for (let k = 0; k <= p; k++) for (let l = 0; l <= q; l++) {
      // (u + iv)^p = Σ C(p,k) u^{p−k} (iv)^k ; (u − iv)^q = Σ C(q,l) u^{q−l} (−iv)^l
      const coef = binom(p, k) * binom(q, l) * (l % 2 ? -1 : 1);
      const [a, b] = ipow(k + l);
      const m = t[idx(N, p + q - k - l, k + l)];
      sr += coef * a * m; si += coef * b * m;
    }
    re[idx(N, p, q)] = sr; im[idx(N, p, q)] = si;
  }
  return { re, im };
}

/** Hu's seven invariants from the normalised central moments η (order ≥ 3 table). */
export function huInvariants(eta: Table, N: number): number[] {
  const e = (p: number, q: number) => eta[idx(N, p, q)];
  const n20 = e(2, 0), n02 = e(0, 2), n11 = e(1, 1), n30 = e(3, 0), n03 = e(0, 3), n21 = e(2, 1), n12 = e(1, 2);
  const a = n30 + n12, b = n21 + n03;
  return [
    n20 + n02,
    (n20 - n02) ** 2 + 4 * n11 ** 2,
    (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
    a ** 2 + b ** 2,
    (n30 - 3 * n12) * a * (a ** 2 - 3 * b ** 2) + (3 * n21 - n03) * b * (3 * a ** 2 - b ** 2),
    (n20 - n02) * (a ** 2 - b ** 2) + 4 * n11 * a * b,
    (3 * n21 - n03) * a * (a ** 2 - 3 * b ** 2) - (n30 - 3 * n12) * b * (3 * a ** 2 - b ** 2),
  ];
}

/** Hu's invariants in complex form (the same numbers): φ1 = c11, φ2 = |c20|², φ3 = |c30|²,
 *  φ4 = |c21|², φ5 = Re(c30 c12³), φ6 = Re(c20 c12²), φ7 = Im(c30 c12³), with c from η.
 *  The sign of φ7 depends on the orientation of the frame (y up or down). */
export function huFromComplex(c: { re: Table; im: Table }, N: number): number[] {
  const g = (p: number, q: number): Complex => ({ re: c.re[idx(N, p, q)], im: c.im[idx(N, p, q)] });
  const mul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  const abs2 = (a: Complex) => a.re * a.re + a.im * a.im;
  const c12 = g(1, 2), c12sq = mul(c12, c12), c12cu = mul(c12sq, c12);
  const p30 = mul(g(3, 0), c12cu), p20 = mul(g(2, 0), c12sq);
  return [g(1, 1).re, abs2(g(2, 0)), abs2(g(3, 0)), abs2(g(2, 1)), p30.re, p20.re, p30.im];
}

/** Flusser's independent and complete set of rotation invariants up to order 3 (from c of η):
 *  c11, c21·c12, Re(c20 c12²), Im(c20 c12²), Re(c30 c12³), Im(c30 c12³). */
export function flusserInvariants3(c: { re: Table; im: Table }, N: number): number[] {
  const g = (p: number, q: number): Complex => ({ re: c.re[idx(N, p, q)], im: c.im[idx(N, p, q)] });
  const mul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  const c12 = g(1, 2), c12sq = mul(c12, c12), c12cu = mul(c12sq, c12);
  const p20 = mul(g(2, 0), c12sq), p30 = mul(g(3, 0), c12cu), c21 = g(2, 1);
  return [g(1, 1).re, c21.re * c12.re - c21.im * c12.im, p20.re, p20.im, p30.re, p30.im];
}

/** The usual log transform for comparing invariants: −sign(φ)·log10|φ| (0 for φ = 0). */
export function logFeature(v: number): number {
  return v === 0 ? 0 : -Math.sign(v) * Math.log10(Math.abs(v));
}

/** All the invariant-related numbers of an image in one call (pixel frame unless given). */
export function invariantsOf(f: ArrayLike<number>, w: number, h: number, o: MomentOptions = {}): { m: Table; mu: Table; eta: Table; hu: number[]; N: number } {
  const N = 4;
  const m = rawMoments(f, w, h, N, o), mu = centralFromRaw(m, N), eta = normalisedMoments(mu, N);
  return { m, mu, eta, hu: huInvariants(eta, N), N };
}

// ---------------------------------------------------------------------------------------
// Pose normalisation

export interface Pose {
  /** Centroid in the table's frame. */
  cx: number; cy: number;
  /** Area (μ00) and the scale factor 1/√μ00 that maps the area to 1. */
  area: number; scale: number;
  /** Angle of the major axis, ½ atan2(2μ11, μ20 − μ02), plus π if needed so that μ30 > 0
   *  after rotating the major axis onto u. */
  theta: number;
  /** Eigenvalues of S = μ/μ00 (descending); equal values mean the angle is undefined. */
  l1: number; l2: number;
  /** True when λ1 ≈ λ2 (relative gap below 1e-3) or |μ30| ≈ 0 after rotation. */
  ambiguous: boolean;
}

export function pose(m: Table, N: number): Pose {
  const [cx, cy] = centroidOf(m, N);
  const mu = centralFromRaw(m, N);
  const a = mu[idx(N, 2, 0)], b = mu[idx(N, 1, 1)], c = mu[idx(N, 0, 2)], A = mu[0];
  let theta = 0.5 * Math.atan2(2 * b, a - c);
  const tr = (a + c) / A, dd = Math.sqrt(((a - c) / A) ** 2 / 4 + (b / A) ** 2);
  const l1 = tr / 2 + dd, l2 = tr / 2 - dd;
  let ambiguous = l1 - l2 < 1e-3 * l1;
  if (N >= 3) {
    const r = rotateMoments(mu, N, -theta);
    const m30 = r[idx(N, 3, 0)], scale3 = Math.abs(A) * (l1 + l2) ** 1.5;
    if (m30 < 0) theta += Math.PI;
    if (Math.abs(m30) < 1e-3 * scale3) ambiguous = true;
  }
  return { cx, cy, area: A, scale: 1 / Math.sqrt(Math.abs(A)), theta, l1, l2, ambiguous };
}

// ---------------------------------------------------------------------------------------
// Legendre polynomials, discrete orthogonal polynomials

/** P_0..P_N at the points xs (three-term recurrence (n+1)P_{n+1} = (2n+1)x P_n − n P_{n−1}). */
export function legendre(N: number, xs: ArrayLike<number>): Basis1D {
  const n = xs.length, B: Basis1D = [];
  for (let k = 0; k <= N; k++) B.push(new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    let p0 = 1, p1 = x;
    B[0][i] = 1;
    if (N >= 1) B[1][i] = x;
    for (let k = 1; k < N; k++) { const p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1); B[k + 1][i] = p2; p0 = p1; p1 = p2; }
  }
  return B;
}

/** Averages of P_0..P_N over the intervals [x − hw, x + hw] (exact: ∫P_n = (P_{n+1} − P_{n−1})/(2n+1)). */
export function legendreAveraged(N: number, xs: ArrayLike<number>, hw: number): Basis1D {
  const a = Array.from(xs, x => x - hw), b = Array.from(xs, x => x + hw);
  const La = legendre(N + 1, a), Lb = legendre(N + 1, b), n = xs.length, B: Basis1D = [];
  for (let k = 0; k <= N; k++) {
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const F = (L: Basis1D, x: number) => (k === 0 ? x : (L[k + 1][i] - L[k - 1][i]) / (2 * k + 1));
      r[i] = (F(Lb, b[i]) - F(La, a[i])) / (2 * hw);
    }
    B.push(r);
  }
  return B;
}

/** Pixel centres of n pixels mapped onto [−1, 1]: x_i = −1 + (2i + 1)/n. */
export function unitCentres(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => -1 + (2 * i + 1) / n);
}

/** Discrete polynomials orthonormal on the n pixel centres (Σ_i T_j(x_i) T_k(x_i) = δ_jk):
 *  the discrete Tchebichef polynomials up to normalisation. Modified Gram–Schmidt (twice)
 *  on the sampled Legendre polynomials, which are already nearly orthogonal. N < n. */
export function discreteOrtho(N: number, n: number): Basis1D {
  const L = legendre(N, unitCentres(n)), B: Basis1D = [];
  for (let k = 0; k <= N; k++) {
    const v = L[k].slice();
    for (let pass = 0; pass < 2; pass++) for (const u of B) {
      let d = 0;
      for (let i = 0; i < n; i++) d += u[i] * v[i];
      for (let i = 0; i < n; i++) v[i] -= d * u[i];
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += v[i] * v[i];
    s = Math.sqrt(s) || 1;
    for (let i = 0; i < n; i++) v[i] /= s;
    if (v[n - 1] < 0) for (let i = 0; i < n; i++) v[i] = -v[i];   // same sign convention as P_k(1) = 1
    B.push(v);
  }
  return B;
}

export type OrthoVariant = "sampled" | "integrated" | "discrete";

/** A separable orthogonal basis on n pixels for coefficients up to N:
 *  coefficient a_pq = Σ f(x, y) A_p(x) A_q(y), reconstruction f̂ = Σ a_pq S_p(x) S_q(y).
 *  "sampled": Legendre at pixel centres, A = (2k+1)/2·P_k·Δx (the usual discretised integral);
 *  "integrated": the pixel averages of P_k (exact integrals for a piecewise-constant image);
 *  "discrete": orthonormal discrete polynomials (exactly orthogonal on the grid). */
export function orthoBasis(n: number, N: number, variant: OrthoVariant): { A: Basis1D; S: Basis1D } {
  if (variant === "discrete") { const T = discreteOrtho(N, n); return { A: T, S: T }; }
  const xs = unitCentres(n), dx = 2 / n;
  const S = variant === "sampled" ? legendre(N, xs) : legendreAveraged(N, xs, dx / 2);
  const A = S.map((r, k) => r.map(v => v * (2 * k + 1) / 2 * dx));
  return { A, S };
}

/** Legendre (or discrete) moments a_pq, p, q ≤ N, of a w × h image mapped onto [−1, 1]². */
export function legendreMoments(f: ArrayLike<number>, w: number, h: number, N: number, variant: OrthoVariant = "sampled"): Table {
  const bx = orthoBasis(w, N, variant), by = h === w ? bx : orthoBasis(h, N, variant);
  return separableSums(f, w, h, bx.A, by.A);
}

/** f̂_N = Σ_{p+q ≤ N} a_pq S_p(x) S_q(y), from a coefficient table of order Nt ≥ N. */
export function legendreReconstruct(a: Table, Nt: number, w: number, h: number, N: number, variant: OrthoVariant = "sampled"): Float64Array {
  const bx = orthoBasis(w, N, variant), by = h === w ? bx : orthoBasis(h, N, variant);
  return reconstructSeparable((p, q) => a[idx(Nt, p, q)], bx.S, by.S, w, h, N);
}

/** Σ_{p+q ≤ N} coef(p, q) Sx_p(x) Sy_q(y), separably. */
export function reconstructSeparable(coef: (p: number, q: number) => number, Sx: Basis1D, Sy: Basis1D, w: number, h: number, N: number): Float64Array {
  const out = new Float64Array(w * h), g = new Float64Array(h);
  for (let p = 0; p <= N; p++) {
    g.fill(0);
    for (let q = 0; q <= N - p; q++) { const c = coef(p, q); if (c === 0) continue; const b = Sy[q]; for (let y = 0; y < h; y++) g[y] += c * b[y]; }
    const bx = Sx[p];
    for (let y = 0; y < h; y++) { const gy = g[y]; if (gy === 0) continue; const o = y * w; for (let x = 0; x < w; x++) out[o + x] += gy * bx[x]; }
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Monomials: Gram matrix and reconstruction from geometric moments (ill-conditioned on purpose)

/** Gram matrix G_jk = Σ_i x_i^{j+k} (or ∫_{−1}^{1} x^{j+k} dx with `continuous`) of the
 *  monomials 1, x, …, x^N on n pixel centres in [−1, 1]. Row-major (N+1)². */
export function monomialGram1d(N: number, n: number, continuous = false): Float64Array {
  const G = new Float64Array((N + 1) * (N + 1));
  const S = new Float64Array(2 * N + 1);
  if (continuous) for (let k = 0; k <= 2 * N; k++) S[k] = k % 2 ? 0 : 2 / (k + 1);
  else { const xs = unitCentres(n), dx = 2 / n; for (const x of xs) { let v = dx; for (let k = 0; k <= 2 * N; k++) { S[k] += v; v *= x; } } }
  for (let j = 0; j <= N; j++) for (let k = 0; k <= N; k++) G[j * (N + 1) + k] = S[j + k];
  return G;
}

/** Gaussian elimination with partial pivoting, in place on copies; null if singular. Used for
 *  the monomial system so that its rounding errors are the demo's subject (no refinement). */
export function gaussSolve(Ain: ArrayLike<number>, bin: ArrayLike<number>, n: number): Float64Array | null {
  const A = Float64Array.from(Ain), b = Float64Array.from(bin);
  for (let c = 0; c < n; c++) {
    let piv = c, best = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) { const v = Math.abs(A[r * n + c]); if (v > best) { best = v; piv = r; } }
    if (best === 0 || !Number.isFinite(best)) return null;
    if (piv !== c) { for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; } const t = b[c]; b[c] = b[piv]; b[piv] = t; }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const fct = A[r * n + c] / d;
      if (fct === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= fct * A[c * n + k];
      b[r] -= fct * b[c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k]; x[r] = s / A[r * n + r]; }
  return x;
}

/** Least-squares polynomial of total degree ≤ N from the geometric moments m_pq (pixel
 *  centres mapped to [−1, 1]² by default, or any frame, e.g. `zeroOneFrame` for [0, 1]²;
 *  area-weighted sums): solve the Gram system G c = m with
 *  G_(pq),(rs) = Σx x^{p+r} · Σy y^{q+s}, then f̂ = Σ c_pq x^p y^q. In exact arithmetic this
 *  equals the discrete orthogonal projection; in float64 it breaks down as N grows. */
export function monomialReconstruct(f: ArrayLike<number>, w: number, h: number, N: number, pixel: PixelModel = "point", frame: Frame = unitFrame(w, h)): Float64Array {
  const fr = frame;
  const m = rawMoments(f, w, h, N, { frame: fr, pixel, area: true });
  const Bx = powerBasis(w, 2 * N, fr.x0, fr.sx, pixel), By = powerBasis(h, 2 * N, fr.y0, fr.sy, pixel);
  const dx = Math.abs(fr.sx), dy = Math.abs(fr.sy);
  const Sx = Bx.map(r => r.reduce((s, v) => s + v * dx, 0)), Sy = By.map(r => r.reduce((s, v) => s + v * dy, 0));
  if (pixel === "square") {
    // with pixel averages as basis functions the Gram entries are Σ avg(x^p)·avg(x^r)·Δx
    return monomialReconstructGeneral(f, w, h, N, pixel, fr);
  }
  const pairs: [number, number][] = [];
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) pairs.push([p, q]);
  const K = pairs.length, G = new Float64Array(K * K), rhs = new Float64Array(K);
  pairs.forEach(([p, q], i) => {
    rhs[i] = m[idx(N, p, q)];
    pairs.forEach(([r, s], j) => { G[i * K + j] = Sx[p + r] * Sy[q + s]; });
  });
  const c = gaussSolve(G, rhs, K);
  const out = new Float64Array(w * h);
  if (!c) return out.fill(NaN);
  const Px = powerBasis(w, N, fr.x0, fr.sx), Py = powerBasis(h, N, fr.y0, fr.sy);
  const coef = new Map<number, number>();
  pairs.forEach(([p, q], i) => coef.set(p * 1000 + q, c[i]));
  return reconstructSeparable((p, q) => coef.get(p * 1000 + q) ?? 0, Px, Py, w, h, N);
}

function monomialReconstructGeneral(f: ArrayLike<number>, w: number, h: number, N: number, pixel: PixelModel, fr: Frame): Float64Array {
  const dx = Math.abs(fr.sx), dy = Math.abs(fr.sy);
  const Bx = powerBasis(w, N, fr.x0, fr.sx, pixel), By = powerBasis(h, N, fr.y0, fr.sy, pixel);
  const gx = (p: number, r: number) => { let s = 0; for (let i = 0; i < w; i++) s += Bx[p][i] * Bx[r][i]; return s * dx; };
  const gy = (q: number, s2: number) => { let s = 0; for (let i = 0; i < h; i++) s += By[q][i] * By[s2][i]; return s * dy; };
  const m = separableSums(f, w, h, Bx, By);
  const pairs: [number, number][] = [];
  for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) pairs.push([p, q]);
  const K = pairs.length, G = new Float64Array(K * K), rhs = new Float64Array(K);
  const GX = new Float64Array((N + 1) * (N + 1)), GY = new Float64Array((N + 1) * (N + 1));
  for (let p = 0; p <= N; p++) for (let r = 0; r <= N; r++) { GX[p * (N + 1) + r] = gx(p, r); GY[p * (N + 1) + r] = gy(p, r); }
  pairs.forEach(([p, q], i) => {
    rhs[i] = m[idx(N, p, q)] * dx * dy;
    pairs.forEach(([r, s], j) => { G[i * K + j] = GX[p * (N + 1) + r] * GY[q * (N + 1) + s]; });
  });
  const c = gaussSolve(G, rhs, K);
  const out = new Float64Array(w * h);
  if (!c) return out.fill(NaN);
  const coef = new Map<number, number>();
  pairs.forEach(([p, q], i) => coef.set(p * 1000 + q, c[i]));
  return reconstructSeparable((p, q) => coef.get(p * 1000 + q) ?? 0, Bx, By, w, h, N);
}

// ---------------------------------------------------------------------------------------
// Zernike polynomials and moments on the unit disc

/** R_n^m(ρ) by the factorial formula Σ_k (−1)^k (n−k)! / (k! ((n+m)/2−k)! ((n−m)/2−k)!) ρ^{n−2k}
 *  (reference; cancellation makes it inaccurate for large n). m ≥ 0, n − m even. */
export function zernikeRadialFactorial(n: number, m: number, rho: number): number {
  m = Math.abs(m);
  if ((n - m) % 2) return 0;
  const fact = (k: number) => { let r = 1; for (let i = 2; i <= k; i++) r *= i; return r; };
  let s = 0;
  for (let k = 0; k <= (n - m) / 2; k++)
    s += (k % 2 ? -1 : 1) * fact(n - k) / (fact(k) * fact((n + m) / 2 - k) * fact((n - m) / 2 - k)) * rho ** (n - 2 * k);
  return s;
}

/** R_m^m, R_{m+2}^m, …, R_N^m at the points rhos (Kintner's recurrence in n). Returns an array
 *  indexed by n (entries for n − m odd or n < m are empty). */
export function zernikeRadialTable(m: number, N: number, rhos: ArrayLike<number>): Float64Array[] {
  m = Math.abs(m);
  const L = rhos.length, R: Float64Array[] = [];
  for (let n = 0; n <= N; n++) R.push(new Float64Array(n >= m && (n - m) % 2 === 0 ? L : 0));
  if (m > N) return R;
  for (let i = 0; i < L; i++) R[m][i] = rhos[i] ** m;
  if (m + 2 <= N) for (let i = 0; i < L; i++) { const r = rhos[i]; R[m + 2][i] = (m + 2) * r ** (m + 2) - (m + 1) * r ** m; }
  for (let n = m + 4; n <= N; n += 2) {
    const K1 = (n + m) * (n - m) * (n - 2) / 2, K2 = 2 * n * (n - 1) * (n - 2);
    const K3 = -m * m * (n - 1) - n * (n - 1) * (n - 2), K4 = -n * (n + m - 2) * (n - m - 2) / 2;
    const a = R[n - 2], b = R[n - 4], out = R[n];
    for (let i = 0; i < L; i++) { const r2 = rhos[i] * rhos[i]; out[i] = ((K2 * r2 + K3) * a[i] + K4 * b[i]) / K1; }
  }
  return R;
}

/** One radial polynomial R_n^m(ρ) via the recurrence. */
export function zernikeRadial(n: number, m: number, rho: number): number {
  m = Math.abs(m);
  if (n < m || (n - m) % 2) return 0;
  return zernikeRadialTable(m, n, [rho])[n][0];
}

/** All (n, m) with 0 ≤ m ≤ n ≤ N and n − m even (m ≥ 0: Z_{n,−m} = conj Z_nm for real f). */
export function zernikeIndices(N: number): [number, number][] {
  const out: [number, number][] = [];
  for (let n = 0; n <= N; n++) for (let m = n % 2; m <= n; m += 2) out.push([n, m]);
  return out;
}

export interface DiscGrid {
  w: number; h: number;
  /** Pixel indices inside the disc, and their ρ and θ (θ counter-clockwise on screen). */
  pix: Int32Array; rho: Float64Array; theta: Float64Array;
  /** Pixel area in disc units (the disc has radius 1). */
  dA: number;
  cx: number; cy: number; r: number;
}

/** The pixels of a w × h image whose centres lie in the disc of radius r (pixels) around
 *  (cx, cy); default: the inscribed disc, centre (w−1)/2, (h−1)/2, radius min(w, h)/2. */
export function discGrid(w: number, h: number, o: { cx?: number; cy?: number; r?: number } = {}): DiscGrid {
  const cx = o.cx ?? (w - 1) / 2, cy = o.cy ?? (h - 1) / 2, r = o.r ?? Math.min(w, h) / 2;
  const pix: number[] = [], rho: number[] = [], th: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = (x - cx) / r, v = -(y - cy) / r, p = Math.hypot(u, v);
    if (p <= 1) { pix.push(y * w + x); rho.push(p); th.push(Math.atan2(v, u)); }
  }
  return { w, h, pix: Int32Array.from(pix), rho: Float64Array.from(rho), theta: Float64Array.from(th), dA: 1 / (r * r), cx, cy, r };
}

export interface ZernikeSet {
  N: number;
  /** Z_nm for m ≥ 0, in the order of zernikeIndices(N). */
  nm: [number, number][];
  re: Float64Array; im: Float64Array;
}

/** Zernike moments Z_nm = (n+1)/π Σ f R_nm(ρ) e^{−imθ} ΔA over the disc pixels, m ≥ 0. */
export function zernikeMoments(f: ArrayLike<number>, g: DiscGrid, N: number): ZernikeSet {
  const nm = zernikeIndices(N), re = new Float64Array(nm.length), im = new Float64Array(nm.length);
  const L = g.pix.length, fv = new Float64Array(L);
  for (let i = 0; i < L; i++) fv[i] = f[g.pix[i]];
  const pos = new Map<number, number>();
  nm.forEach(([n, m], k) => pos.set(n * 1000 + m, k));
  for (let m = 0; m <= N; m++) {
    const R = zernikeRadialTable(m, N, g.rho);
    const c = new Float64Array(L), s = new Float64Array(L);
    for (let i = 0; i < L; i++) { c[i] = fv[i] * Math.cos(m * g.theta[i]); s[i] = fv[i] * Math.sin(m * g.theta[i]); }
    for (let n = m; n <= N; n += 2) {
      const r = R[n];
      let a = 0, b = 0;
      for (let i = 0; i < L; i++) { a += r[i] * c[i]; b += r[i] * s[i]; }
      const k = pos.get(n * 1000 + m)!, fac = (n + 1) / Math.PI * g.dA;
      re[k] = fac * a; im[k] = -fac * b;
    }
  }
  return { N, nm, re, im };
}

/** f̂_N = Σ_{n ≤ N} Σ_m Z_nm V_nm on the disc pixels (0 outside the disc). With `curve`, also
 *  returns the image after each order 0..N via `onOrder(n, partial)` (incremental). */
export function zernikeReconstruct(Z: ZernikeSet, g: DiscGrid, N: number, onOrder?: (n: number, img: Float64Array) => void): Float64Array {
  const out = new Float64Array(g.w * g.h), L = g.pix.length;
  const acc = new Float64Array(L);
  // radial tables per m up to N
  const Rm: Float64Array[][] = [];
  for (let m = 0; m <= N; m++) Rm.push(zernikeRadialTable(m, N, g.rho));
  const cosm: Float64Array[] = [], sinm: Float64Array[] = [];
  for (let m = 0; m <= N; m++) { cosm.push(g.theta.map(t => Math.cos(m * t))); sinm.push(g.theta.map(t => Math.sin(m * t))); }
  const pos = new Map<number, number>();
  Z.nm.forEach(([n, m], k) => pos.set(n * 1000 + m, k));
  for (let n = 0; n <= N; n++) {
    for (let m = n % 2; m <= n; m += 2) {
      const k = pos.get(n * 1000 + m);
      if (k === undefined) continue;
      const a = Z.re[k], b = Z.im[k], R = Rm[m][n], c = cosm[m], s = sinm[m];
      if (m === 0) for (let i = 0; i < L; i++) acc[i] += a * R[i];
      else for (let i = 0; i < L; i++) acc[i] += 2 * R[i] * (a * c[i] - b * s[i]);
    }
    if (onOrder) { const img = new Float64Array(g.w * g.h); for (let i = 0; i < L; i++) img[g.pix[i]] = acc[i]; onOrder(n, img); }
  }
  for (let i = 0; i < L; i++) out[g.pix[i]] = acc[i];
  return out;
}

/** V_nm sampled on a size × size image (inscribed disc; NaN outside), real and imaginary part,
 *  θ counter-clockwise on screen. m may be negative. */
export function zernikeImage(n: number, m: number, size: number): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(size * size).fill(NaN), im = new Float64Array(size * size).fill(NaN);
  const g = discGrid(size, size), R = zernikeRadialTable(Math.abs(m), n, g.rho)[n];
  for (let i = 0; i < g.pix.length; i++) {
    const t = m * g.theta[i];
    re[g.pix[i]] = R[i] * Math.cos(t); im[g.pix[i]] = R[i] * Math.sin(t);
  }
  return { re, im };
}

// ---------------------------------------------------------------------------------------
// Small helpers

/** Relative RMS error ‖a − b‖ / ‖b‖ over all pixels (or over `mask`). */
export function relError(a: ArrayLike<number>, b: ArrayLike<number>, mask?: ArrayLike<number>): number {
  let s = 0, t = 0;
  for (let i = 0; i < b.length; i++) {
    if (mask && !mask[i]) continue;
    const d = a[i] - b[i];
    s += Number.isFinite(d) ? d * d : Infinity; t += b[i] * b[i];
  }
  return Math.sqrt(s / (t || 1));
}

/** Pixel averages of a polygon region (anti-aliased mask) by supersampling the even-odd rule
 *  at k × k points per pixel; pixel (x, y) covers [x, x+1] × [y, y+1] in polygon units.
 *  (Same convention as lib/shapecanvas.ts `rasterise`, which samples the centre only.) */
export function polygonCoverageEvenOdd(polys: [number, number][][], w: number, h: number, k = 4): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let sy = 0; sy < k; sy++) {
    const py = y + (sy + 0.5) / k;
    const xs: number[] = [];
    for (const p of polys) for (let i = 0; i < p.length; i++) {
      const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length];
      if ((y0 <= py) !== (y1 <= py)) xs.push(x0 + (py - y0) * (x1 - x0) / (y1 - y0));
    }
    xs.sort((a, b) => a - b);
    for (let j = 0; j + 1 < xs.length; j += 2) {
      for (let sx = 0; sx < k; sx++) {
        const off = (sx + 0.5) / k;
        const xa = Math.max(0, Math.ceil(xs[j] - off)), xb = Math.min(w, Math.ceil(xs[j + 1] - off));
        for (let x = xa; x < xb; x++) out[y * w + x] += 1 / (k * k);
      }
    }
  }
  return out;
}
