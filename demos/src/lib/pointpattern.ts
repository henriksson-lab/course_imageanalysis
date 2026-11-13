// Spatial point patterns (owner: the spatial statistics day; no DOM). Positions in µm in a
// window W = [0, w] × [0, h], either a full rectangle or a mask raster with 1 µm cells
// (mask[iy·w + ix] covers [ix, ix+1) × [iy, iy+1)). Intensity is ρ (points per µm²; the
// literature writes λ, which the course keeps for the weight of a prior).
//
//   const win = rectWindow(200, 200);
//   const p = thomas(rng(1), 0.0006, 8, 4, win);          // κ, μ, σ
//   const radii = linspace(0, 50, 51);
//   const K = Kest(p, win, radii, "translation"), L = Lfun(K), g = pcf(p, win, radii, 2, "translation");
//   const env = pointwiseEnvelope(sims); const mad = madTest(obsL, simsL, radii, [0, 50]);
//
// Simulators (seeded with lib/random.ts): binomial and Poisson CSR, inhomogeneous Poisson by
// thinning, Thomas and Matérn cluster, sequential inhibition (hard core), jittered lattice,
// colonies (clusters with a hard core), tight pairs, detection artefacts (merge, split), two-type
// generators, random relabelling, toroidal shift. Summaries: quadrat counts with the χ² test,
// kernel intensity with edge correction, G, F, J, K, L, pair correlation g, cross-K and cross-g,
// the mark connection function, the inhomogeneous K; edge corrections none / border / translation /
// isotropic. Tests: Monte Carlo rank p-value, pointwise and global (MAD) envelopes, DCLF, pooled K.
import { Rng, poisson, gauss, permutation } from "./random";
import { edt } from "./distance";
import { logGamma, normCdf } from "./stats";
import { correlateFFT } from "./fft";

type A = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Points and windows

export interface Points { x: Float64Array; y: Float64Array; marks?: Int32Array }

export function makePoints(xs: A, ys: A, marks?: A): Points {
  return { x: Float64Array.from(xs), y: Float64Array.from(ys), marks: marks ? Int32Array.from(marks) : undefined };
}
export const count = (p: Points): number => p.x.length;

/** Keep points i with keep(i). */
export function filterPoints(p: Points, keep: (i: number) => boolean): Points {
  const xs: number[] = [], ys: number[] = [], ms: number[] = [];
  for (let i = 0; i < p.x.length; i++) if (keep(i)) { xs.push(p.x[i]); ys.push(p.y[i]); if (p.marks) ms.push(p.marks[i]); }
  return makePoints(xs, ys, p.marks ? ms : undefined);
}
/** The points of one type. */
export const ofType = (p: Points, t: number): Points => filterPoints(p, i => !!p.marks && p.marks[i] === t);
/** Concatenate patterns; marks become the pattern index unless `keepMarks`. */
export function concatPoints(list: Points[], keepMarks = false): Points {
  const xs: number[] = [], ys: number[] = [], ms: number[] = [];
  list.forEach((p, k) => { for (let i = 0; i < p.x.length; i++) { xs.push(p.x[i]); ys.push(p.y[i]); ms.push(keepMarks && p.marks ? p.marks[i] : k); } });
  return makePoints(xs, ys, ms);
}

export interface Window {
  w: number; h: number;
  /** undefined = the full rectangle */
  mask?: Uint8Array;
  area: number;
  /** distance (µm) from each mask cell centre to the outside (masks only) */
  bdist?: Float32Array;
  /** cached set covariance |W ∩ (W + v)| for integer offsets (masks only) */
  cov?: { data: Float64Array; n: number };
}

export function rectWindow(w: number, h: number): Window { return { w, h, area: w * h }; }

export function maskWindow(mask: A, w: number, h: number): Window {
  const m = Uint8Array.from(mask, v => (v ? 1 : 0));
  let area = 0;
  for (let i = 0; i < m.length; i++) area += m[i];
  // pad by one so the raster border counts as outside
  const W = w + 2, H = h + 2, pad = new Uint8Array(W * H);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pad[(y + 1) * W + x + 1] = m[y * w + x];
  const d = edt(pad, W, H), bd = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) bd[y * w + x] = m[y * w + x] ? Math.max(0, d[(y + 1) * W + x + 1] - 0.5) : 0;
  return { w, h, mask: m, area, bdist: bd };
}

/** A smooth tissue-like blob (radius modulated by a few seeded harmonics), optionally elongated
 *  (`aspect` < 1 squeezes y) and with a round hole (a vessel) at (hx, hy) of radius hr. */
export function blobMask(r: Rng, w: number, h: number, o: { radius?: number; wobble?: number; aspect?: number; hole?: [number, number, number]; cx?: number; cy?: number } = {}): Uint8Array {
  const R = o.radius ?? 0.42 * Math.min(w, h), wob = o.wobble ?? 0.12, asp = o.aspect ?? 1;
  const cx = o.cx ?? w / 2, cy = o.cy ?? h / 2;
  const amp: number[] = [], ph: number[] = [];
  for (let k = 2; k <= 5; k++) { amp.push(wob * (0.5 + r.unif()) / (k - 1)); ph.push(2 * Math.PI * r.unif()); }
  const m = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x + 0.5 - cx, dy = (y + 0.5 - cy) / asp, th = Math.atan2(dy, dx);
    let rad = R;
    for (let k = 0; k < amp.length; k++) rad *= 1 + amp[k] * Math.cos((k + 2) * th + ph[k]);
    let inside = Math.hypot(dx, dy) < rad;
    if (inside && o.hole) inside = Math.hypot(x + 0.5 - o.hole[0], y + 0.5 - o.hole[1]) >= o.hole[2];
    m[y * w + x] = inside ? 1 : 0;
  }
  return m;
}

/** A curved band of tissue of the given width (an elongated mask). */
export function bandMask(w: number, h: number, width: number, bend = 0.25): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const t = (x + 0.5) / w, yc = h / 2 + bend * h * Math.sin(2 * Math.PI * (t - 0.25)) * 0.8;
    m[y * w + x] = Math.abs(y + 0.5 - yc) < width / 2 && x > 4 && x < w - 4 ? 1 : 0;
  }
  return m;
}

export function inside(win: Window, x: number, y: number): boolean {
  if (!(x >= 0 && y >= 0 && x < win.w && y < win.h)) return false;
  return !win.mask || win.mask[Math.floor(y) * win.w + Math.floor(x)] === 1;
}

/** Distance from (x, y) to the window border (outside → 0). */
export function borderDistance(win: Window, x: number, y: number): number {
  if (!inside(win, x, y)) return 0;
  if (!win.mask) return Math.min(x, y, win.w - x, win.h - y);
  const ix = Math.floor(x), iy = Math.floor(y);
  return win.bdist![iy * win.w + ix];
}

/** Points that lie in the window (crops a pattern to a smaller window). */
export const clip = (p: Points, win: Window): Points => filterPoints(p, i => inside(win, p.x[i], p.y[i]));

// ---------------------------------------------------------------------------------------
// Simulation

/** n independent uniform points in W (the binomial process; rejection in masks). */
export function binomialProcess(r: Rng, n: number, win: Window): Points {
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    for (let t = 0; t < 100000; t++) { x = r.unif() * win.w; y = r.unif() * win.h; if (inside(win, x, y)) break; }
    xs[i] = x; ys[i] = y;
  }
  return { x: xs, y: ys };
}

/** Homogeneous Poisson process (CSR) with intensity ρ: N ~ Poisson(ρ|W|), then N uniform points. */
export const poissonProcess = (r: Rng, rho: number, win: Window): Points => binomialProcess(r, poisson(r, rho * win.area), win);

/** Inhomogeneous Poisson process by thinning a CSR pattern of intensity ρmax. */
export function inhomPoisson(r: Rng, rhoFn: (x: number, y: number) => number, rhoMax: number, win: Window): Points {
  const p = poissonProcess(r, rhoMax, win);
  return filterPoints(p, i => r.unif() * rhoMax < rhoFn(p.x[i], p.y[i]));
}

/** n points of an inhomogeneous density (conditional on n): rejection against ρmax. */
export function inhomBinomial(r: Rng, n: number, rhoFn: (x: number, y: number) => number, rhoMax: number, win: Window): Points {
  const xs: number[] = [], ys: number[] = [];
  for (let t = 0; xs.length < n && t < 1e6; t++) {
    const x = r.unif() * win.w, y = r.unif() * win.h;
    if (inside(win, x, y) && r.unif() * rhoMax < rhoFn(x, y)) { xs.push(x); ys.push(y); }
  }
  return makePoints(xs, ys);
}

/** Thomas cluster process: parents Poisson(κ) in an expanded box, Poisson(μ) offspring each, N(0, σ²I)
 *  displacement; offspring outside W are dropped. Intensity κμ. marks = parent index. */
export function thomas(r: Rng, kappa: number, mu: number, sigma: number, win: Window): Points {
  return clusterProcess(r, kappa, mu, win, 4 * sigma, () => [gauss(r, 0, sigma), gauss(r, 0, sigma)]);
}

/** Matérn cluster process: offspring uniform in a disc of radius R around each parent. */
export function maternCluster(r: Rng, kappa: number, mu: number, R: number, win: Window): Points {
  return clusterProcess(r, kappa, mu, win, R, () => {
    const a = 2 * Math.PI * r.unif(), d = R * Math.sqrt(r.unif());
    return [d * Math.cos(a), d * Math.sin(a)];
  });
}

function clusterProcess(r: Rng, kappa: number, mu: number, win: Window, reach: number, off: () => [number, number]): Points {
  const ext = rectWindow(win.w + 2 * reach, win.h + 2 * reach);
  const np = poisson(r, kappa * ext.area);
  const xs: number[] = [], ys: number[] = [], ms: number[] = [];
  for (let k = 0; k < np; k++) {
    const px = r.unif() * ext.w - reach, py = r.unif() * ext.h - reach, m = poisson(r, mu);
    for (let j = 0; j < m; j++) {
      const [dx, dy] = off(), x = px + dx, y = py + dy;
      if (inside(win, x, y)) { xs.push(x); ys.push(y); ms.push(k); }
    }
  }
  return makePoints(xs, ys, ms);
}

/** Simple sequential inhibition: add uniform points, reject those closer than h to an earlier one,
 *  until n points or `maxTries` rejections in a row. */
export function ssi(r: Rng, n: number, h: number, win: Window, maxTries = 2000): Points {
  const xs: number[] = [], ys: number[] = [], h2 = h * h;
  const grid = new SpatialGrid(win.w, win.h, Math.max(h, 1));
  let fails = 0;
  while (xs.length < n && fails < maxTries) {
    const x = r.unif() * win.w, y = r.unif() * win.h;
    if (!inside(win, x, y)) continue;
    if (grid.anyWithin(x, y, h2, xs, ys)) { fails++; continue; }
    grid.add(xs.length, x, y); xs.push(x); ys.push(y); fails = 0;
  }
  return makePoints(xs, ys);
}

/** A square lattice of the given spacing, randomly offset, each point jittered by N(0, jitter²). */
export function jitteredLattice(r: Rng, spacing: number, jitter: number, win: Window): Points {
  const ox = r.unif() * spacing, oy = r.unif() * spacing, xs: number[] = [], ys: number[] = [];
  for (let gy = oy - spacing; gy < win.h + spacing; gy += spacing)
    for (let gx = ox - spacing; gx < win.w + spacing; gx += spacing) {
      const x = gx + gauss(r, 0, jitter), y = gy + gauss(r, 0, jitter);
      if (inside(win, x, y)) { xs.push(x); ys.push(y); }
    }
  return makePoints(xs, ys);
}

/** Colonies: parents Poisson(κ); in each colony Poisson(μ) cells placed uniformly in a disc of
 *  radius R by sequential inhibition with minimum distance h (also between colonies). */
export function colonies(r: Rng, kappa: number, mu: number, R: number, h: number, win: Window): Points {
  const ext = rectWindow(win.w + 2 * R, win.h + 2 * R), np = poisson(r, kappa * ext.area);
  const xs: number[] = [], ys: number[] = [], ms: number[] = [], h2 = h * h;
  const grid = new SpatialGrid(win.w, win.h, Math.max(h, 1));
  for (let k = 0; k < np; k++) {
    const px = r.unif() * ext.w - R, py = r.unif() * ext.h - R, m = poisson(r, mu);
    let placed = 0;
    for (let t = 0; placed < m && t < 60 * m + 60; t++) {
      const a = 2 * Math.PI * r.unif(), d = R * Math.sqrt(r.unif()), x = px + d * Math.cos(a), y = py + d * Math.sin(a);
      if (!inside(win, x, y)) { placed++; continue; }   // the colony continues outside W
      if (grid.anyWithin(x, y, h2, xs, ys)) continue;
      grid.add(xs.length, x, y); xs.push(x); ys.push(y); ms.push(k); placed++;
    }
  }
  return makePoints(xs, ys, ms);
}

/** nPairs pair centres at random; the two points of a pair are d apart (random direction). */
export function tightPairs(r: Rng, nPairs: number, d: number, win: Window): Points {
  const c = binomialProcess(r, nPairs, win), xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < nPairs; i++) {
    const a = Math.PI * r.unif();
    for (const s of [-0.5, 0.5]) {
      const x = c.x[i] + s * d * Math.cos(a), y = c.y[i] + s * d * Math.sin(a);
      if (inside(win, x, y)) { xs.push(x); ys.push(y); }
    }
  }
  return makePoints(xs, ys);
}

/** Detection artefact: points closer than d are merged into one at their mean (greedy, closest
 *  pairs first, repeated until no pair is closer than d). */
export function mergeClose(p: Points, d: number): Points {
  let xs = Array.from(p.x), ys = Array.from(p.y), ws = xs.map(() => 1);
  for (;;) {
    let best = d * d, bi = -1, bj = -1;
    for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) {
      const q = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2;
      if (q < best) { best = q; bi = i; bj = j; }
    }
    if (bi < 0) break;
    const W = ws[bi] + ws[bj];
    xs[bi] = (xs[bi] * ws[bi] + xs[bj] * ws[bj]) / W; ys[bi] = (ys[bi] * ws[bi] + ys[bj] * ws[bj]) / W; ws[bi] = W;
    xs.splice(bj, 1); ys.splice(bj, 1); ws.splice(bj, 1);
  }
  return makePoints(xs, ys);
}

/** Detection artefact: a fraction of the points is split into two, d apart (over-segmentation). */
export function splitSome(r: Rng, p: Points, frac: number, d: number, win: Window): Points {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < p.x.length; i++) {
    if (r.unif() < frac) {
      const a = Math.PI * r.unif();
      for (const s of [-0.5, 0.5]) {
        const x = p.x[i] + s * d * Math.cos(a), y = p.y[i] + s * d * Math.sin(a);
        if (inside(win, x, y)) { xs.push(x); ys.push(y); }
      }
    } else { xs.push(p.x[i]); ys.push(p.y[i]); }
  }
  return makePoints(xs, ys);
}

/** Random labelling: the same marks, permuted over the positions. */
export function relabel(r: Rng, p: Points): Points {
  const perm = permutation(r, p.x.length), m = new Int32Array(p.x.length);
  for (let i = 0; i < m.length; i++) m[i] = p.marks ? p.marks[perm[i]] : 0;
  return { x: p.x, y: p.y, marks: m };
}

/** Random labelling with probabilities: each point independently type 1 with probability p1, else 0. */
export function labelRandom(r: Rng, p: Points, p1: number): Points {
  const m = new Int32Array(p.x.length);
  for (let i = 0; i < m.length; i++) m[i] = r.unif() < p1 ? 1 : 0;
  return { x: p.x, y: p.y, marks: m };
}

/** Shift a pattern by (dx, dy) with wrap-around in the rectangle [0, w] × [0, h]. */
export function toroidalShift(p: Points, dx: number, dy: number, win: Window): Points {
  const m = (v: number, L: number) => ((v % L) + L) % L;
  return { x: p.x.map(v => m(v + dx, win.w)), y: p.y.map(v => m(v + dy, win.h)), marks: p.marks };
}

/** Uniform grid of buckets for neighbour queries. */
export class SpatialGrid {
  readonly nx: number; readonly ny: number; readonly cell: number;
  readonly cells: number[][];
  constructor(w: number, h: number, cell: number) {
    this.cell = cell; this.nx = Math.max(1, Math.ceil(w / cell)); this.ny = Math.max(1, Math.ceil(h / cell));
    this.cells = Array.from({ length: this.nx * this.ny }, () => []);
  }
  private idx(x: number, y: number): number {
    const cx = Math.min(this.nx - 1, Math.max(0, Math.floor(x / this.cell))), cy = Math.min(this.ny - 1, Math.max(0, Math.floor(y / this.cell)));
    return cy * this.nx + cx;
  }
  add(i: number, x: number, y: number): void { this.cells[this.idx(x, y)].push(i); }
  anyWithin(x: number, y: number, r2: number, xs: A, ys: A): boolean {
    const cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell);
    for (let gy = cy - 1; gy <= cy + 1; gy++) for (let gx = cx - 1; gx <= cx + 1; gx++) {
      if (gx < 0 || gy < 0 || gx >= this.nx || gy >= this.ny) continue;
      for (const j of this.cells[gy * this.nx + gx]) if ((xs[j] - x) ** 2 + (ys[j] - y) ** 2 < r2) return true;
    }
    return false;
  }
}

// ---------------------------------------------------------------------------------------
// First order: quadrats and intensity

export interface QuadratResult {
  nx: number; ny: number;
  /** counts per quadrat (row-major), NaN for quadrats not fully in the window */
  counts: Float64Array;
  m: number; mean: number; variance: number; vmr: number;
  X2: number; df: number;
  /** P(χ² ≥ X²): small = clustered */
  pClustered: number;
  /** P(χ² ≤ X²): small = regular */
  pRegular: number;
}

/** Quadrat counts on an nx × ny grid over [0, w] × [0, h]; in masks only quadrats fully inside count. */
export function quadratCounts(p: Points, win: Window, nx: number, ny: number): QuadratResult {
  const qw = win.w / nx, qh = win.h / ny, counts = new Float64Array(nx * ny);
  if (win.mask) {
    for (let qy = 0; qy < ny; qy++) for (let qx = 0; qx < nx; qx++) {
      let full = true;
      for (let y = Math.floor(qy * qh); y < Math.ceil((qy + 1) * qh) && full; y++)
        for (let x = Math.floor(qx * qw); x < Math.ceil((qx + 1) * qw) && full; x++) if (!win.mask[y * win.w + x]) full = false;
      if (!full) counts[qy * nx + qx] = NaN;
    }
  }
  for (let i = 0; i < p.x.length; i++) {
    const qx = Math.min(nx - 1, Math.floor(p.x[i] / qw)), qy = Math.min(ny - 1, Math.floor(p.y[i] / qh));
    counts[qy * nx + qx] += 1;
  }
  let m = 0, s = 0;
  for (const c of counts) if (Number.isFinite(c)) { m++; s += c; }
  const mean = s / Math.max(1, m);
  let ss = 0;
  for (const c of counts) if (Number.isFinite(c)) ss += (c - mean) ** 2;
  const variance = m > 1 ? ss / (m - 1) : NaN, vmr = variance / mean;
  const X2 = mean > 0 ? ss / mean : NaN, df = m - 1;
  const cdf = chi2Cdf(X2, df);
  return { nx, ny, counts, m, mean, variance, vmr, X2, df, pClustered: 1 - cdf, pRegular: cdf };
}

/** Regularised lower incomplete gamma P(a, x) (series / continued fraction, Numerical Recipes). */
export function gammaP(a: number, x: number): number {
  if (!(x > 0)) return 0;
  const gln = logGamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a); b += 2;
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
}

/** χ² CDF with k degrees of freedom. */
export const chi2Cdf = (x: number, k: number): number => (k > 0 ? gammaP(k / 2, x / 2) : NaN);

/** Upper χ² quantile by bisection: the x with P(χ²_k ≥ x) = alpha. */
export function chi2Upper(alpha: number, k: number): number {
  let lo = 0, hi = k + 50 * Math.sqrt(2 * k) + 50;
  for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (1 - chi2Cdf(mid, k) > alpha) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

/** Gaussian edge factor e(x) = ∫_W k_σ(x − y) dy (closed form for rectangles, a sum over cells for masks). */
export function edgeFactor(win: Window, sigma: number, x: number, y: number): number {
  if (!win.mask) {
    const fx = normCdf((win.w - x) / sigma) - normCdf(-x / sigma), fy = normCdf((win.h - y) / sigma) - normCdf(-y / sigma);
    return fx * fy;
  }
  const R = Math.ceil(3 * sigma), x0 = Math.max(0, Math.floor(x - R)), x1 = Math.min(win.w - 1, Math.ceil(x + R));
  const y0 = Math.max(0, Math.floor(y - R)), y1 = Math.min(win.h - 1, Math.ceil(y + R));
  let s = 0;
  const c = 1 / (2 * Math.PI * sigma * sigma);
  for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++)
    if (win.mask[yy * win.w + xx]) s += c * Math.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2 * sigma * sigma));
  return s;
}

/** Kernel estimate of the intensity on an nx × ny grid of cell centres (NaN outside the window). */
export function kernelIntensity(p: Points, win: Window, sigma: number, nx: number, ny: number, edge = true): Float64Array {
  const out = new Float64Array(nx * ny), c = 1 / (2 * Math.PI * sigma * sigma), cut = (4 * sigma) ** 2;
  for (let gy = 0; gy < ny; gy++) for (let gx = 0; gx < nx; gx++) {
    const x = (gx + 0.5) * win.w / nx, y = (gy + 0.5) * win.h / ny;
    if (!inside(win, x, y)) { out[gy * nx + gx] = NaN; continue; }
    let s = 0;
    for (let i = 0; i < p.x.length; i++) {
      const d2 = (p.x[i] - x) ** 2 + (p.y[i] - y) ** 2;
      if (d2 < cut) s += c * Math.exp(-d2 / (2 * sigma * sigma));
    }
    out[gy * nx + gx] = edge ? s / Math.max(edgeFactor(win, sigma, x, y), 1e-6) : s;
  }
  return out;
}

/** Kernel intensity at the data points; leave-one-out (the point itself excluded) by default. */
export function intensityAtPoints(p: Points, win: Window, sigma: number, leaveOneOut = true, edge = true): Float64Array {
  const n = p.x.length, out = new Float64Array(n), c = 1 / (2 * Math.PI * sigma * sigma);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (j === i && leaveOneOut) continue;
      s += c * Math.exp(-((p.x[i] - p.x[j]) ** 2 + (p.y[i] - p.y[j]) ** 2) / (2 * sigma * sigma));
    }
    out[i] = edge ? s / Math.max(edgeFactor(win, sigma, p.x[i], p.y[i]), 1e-6) : s;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Nearest neighbours: G, F, J

export function linspace(a: number, b: number, n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => a + (b - a) * i / Math.max(1, n - 1));
}

/** Nearest-neighbour distance and index of every point (brute force, n ≤ a few thousand). */
export function nnDistances(p: Points): { d: Float64Array; idx: Int32Array } {
  const n = p.x.length, d = new Float64Array(n).fill(Infinity), idx = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const q = (p.x[i] - p.x[j]) ** 2 + (p.y[i] - p.y[j]) ** 2;
    if (q < d[i]) { d[i] = q; idx[i] = j; }
    if (q < d[j]) { d[j] = q; idx[j] = i; }
  }
  for (let i = 0; i < n; i++) d[i] = Math.sqrt(d[i]);
  return { d, idx };
}

export type NNCorr = "none" | "border";

/** Ĝ(r): fraction of points whose nearest neighbour is within r. border = reduced sample
 *  (only points at least r from the border count). */
export function Gest(p: Points, win: Window, radii: A, corr: NNCorr = "border"): Float64Array {
  const { d } = nnDistances(p), n = d.length, out = new Float64Array(radii.length);
  const b = Float64Array.from({ length: n }, (_, i) => borderDistance(win, p.x[i], p.y[i]));
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      if (corr === "border" && b[i] < r) continue;
      den++; if (d[i] <= r) num++;
    }
    out[k] = den > 0 ? num / den : NaN;
  }
  return out;
}

/** Test locations on a regular grid inside the window (spacing `step`). */
export function gridLocations(win: Window, step: number): Points {
  const xs: number[] = [], ys: number[] = [];
  for (let y = step / 2; y < win.h; y += step) for (let x = step / 2; x < win.w; x += step) if (inside(win, x, y)) { xs.push(x); ys.push(y); }
  return makePoints(xs, ys);
}

/** Distance from each location to the nearest point of p (the distance transform of the point image). */
export function emptySpaceDistances(loc: Points, p: Points): Float64Array {
  const out = new Float64Array(loc.x.length);
  for (let k = 0; k < loc.x.length; k++) {
    let best = Infinity;
    for (let i = 0; i < p.x.length; i++) { const q = (p.x[i] - loc.x[k]) ** 2 + (p.y[i] - loc.y[k]) ** 2; if (q < best) best = q; }
    out[k] = Math.sqrt(best);
  }
  return out;
}

/** F̂(r): fraction of test locations whose nearest point is within r (grid spacing `step`). */
export function Fest(p: Points, win: Window, radii: A, corr: NNCorr = "border", step = 0): Float64Array {
  const loc = gridLocations(win, step || Math.max(win.w, win.h) / 64), e = emptySpaceDistances(loc, p);
  const b = Float64Array.from({ length: e.length }, (_, k) => borderDistance(win, loc.x[k], loc.y[k]));
  const out = new Float64Array(radii.length);
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    let num = 0, den = 0;
    for (let i = 0; i < e.length; i++) {
      if (corr === "border" && b[i] < r) continue;
      den++; if (e[i] <= r) num++;
    }
    out[k] = den > 0 ? num / den : NaN;
  }
  return out;
}

/** Ĵ(r) = (1 − Ĝ) / (1 − F̂). */
export function Jfun(G: A, F: A): Float64Array {
  return Float64Array.from({ length: G.length }, (_, k) => (1 - F[k] > 1e-9 ? (1 - G[k]) / (1 - F[k]) : NaN));
}

/** CSR: G(r) = F(r) = 1 − exp(−ρπr²). */
export const csrG = (r: number, rho: number): number => 1 - Math.exp(-rho * Math.PI * r * r);

// ---------------------------------------------------------------------------------------
// Second order: K, L, g with edge corrections

export type KCorr = "none" | "border" | "translation" | "isotropic";

/** Fraction of the circle of radius d centred at (x, y) that lies inside the window. */
export function circleFraction(win: Window, x: number, y: number, d: number, nAng = 64): number {
  if (d <= 0) return 1;
  if (!win.mask) {
    // exact for rectangles: angles where the circle crosses the four edge lines, test arc midpoints
    const ang: number[] = [0, 2 * Math.PI];
    const addX = (c: number) => { const t = (c - x) / d; if (Math.abs(t) <= 1) { const a = Math.acos(t); ang.push(a, 2 * Math.PI - a); } };
    const addY = (c: number) => { const t = (c - y) / d; if (Math.abs(t) <= 1) { const a = Math.asin(t); ang.push((a + 2 * Math.PI) % (2 * Math.PI), Math.PI - a); } };
    addX(0); addX(win.w); addY(0); addY(win.h);
    ang.sort((a, b) => a - b);
    let inArc = 0;
    for (let k = 0; k + 1 < ang.length; k++) {
      const a0 = ang[k], a1 = ang[k + 1];
      if (a1 - a0 < 1e-12) continue;
      const am = (a0 + a1) / 2, px = x + d * Math.cos(am), py = y + d * Math.sin(am);
      if (px >= 0 && px <= win.w && py >= 0 && py <= win.h) inArc += a1 - a0;
    }
    return inArc / (2 * Math.PI);
  }
  let c = 0;
  for (let k = 0; k < nAng; k++) { const a = 2 * Math.PI * (k + 0.5) / nAng; if (inside(win, x + d * Math.cos(a), y + d * Math.sin(a))) c++; }
  return c / nAng;
}

/** Set covariance |W ∩ (W + (dx, dy))| (exact for rectangles; FFT autocorrelation of the mask, cached). */
export function setCovariance(win: Window, dx: number, dy: number): number {
  if (!win.mask) return Math.max(0, win.w - Math.abs(dx)) * Math.max(0, win.h - Math.abs(dy));
  if (!win.cov) {
    let n = 1;
    while (n < 2 * Math.max(win.w, win.h)) n *= 2;
    const a = new Float64Array(n * n);
    for (let y = 0; y < win.h; y++) for (let x = 0; x < win.w; x++) a[y * n + x] = win.mask[y * win.w + x];
    win.cov = { data: correlateFFT(a, a, n, n), n };
  }
  const { data, n } = win.cov, ix = Math.round(dx), iy = Math.round(dy);
  if (Math.abs(ix) >= n / 2 || Math.abs(iy) >= n / 2) return 0;
  return Math.max(0, data[((iy + n) % n) * n + ((ix + n) % n)]);
}

/** Ordered pairs (i ≠ j) closer than rmax with their distance, edge weight and the border distance of i. */
export interface PairData {
  i: Int32Array; j: Int32Array; d: Float64Array; w: Float64Array;
  /** border distance of every point */
  b: Float64Array;
  n: number; area: number; corr: KCorr;
}

export function pairData(p: Points, win: Window, rmax: number, corr: KCorr = "translation"): PairData {
  const n = p.x.length, I: number[] = [], J: number[] = [], D: number[] = [], Wt: number[] = [];
  const grid = new SpatialGrid(win.w, win.h, Math.max(rmax, 1));
  for (let i = 0; i < n; i++) grid.add(i, p.x[i], p.y[i]);
  const r2 = rmax * rmax;
  for (let i = 0; i < n; i++) {
    const cx = Math.floor(p.x[i] / grid.cell), cy = Math.floor(p.y[i] / grid.cell);
    for (let gy = cy - 1; gy <= cy + 1; gy++) for (let gx = cx - 1; gx <= cx + 1; gx++) {
      if (gx < 0 || gy < 0 || gx >= grid.nx || gy >= grid.ny) continue;
      for (const j of grid.cells[gy * grid.nx + gx]) {
        if (j === i) continue;
        const dx = p.x[j] - p.x[i], dy = p.y[j] - p.y[i], q = dx * dx + dy * dy;
        if (q > r2) continue;
        const d = Math.sqrt(q);
        let w = 1;
        if (corr === "translation") w = win.area / Math.max(setCovariance(win, dx, dy), 1e-9);
        else if (corr === "isotropic") w = 1 / Math.max(circleFraction(win, p.x[i], p.y[i], d), 0.25);
        I.push(i); J.push(j); D.push(d); Wt.push(w);
      }
    }
  }
  const b = Float64Array.from({ length: n }, (_, i) => borderDistance(win, p.x[i], p.y[i]));
  return { i: Int32Array.from(I), j: Int32Array.from(J), d: Float64Array.from(D), w: Float64Array.from(Wt), b, n, area: win.area, corr };
}

/** Selection of pairs by type: from-type a to-type b (−1 = any). */
interface Sel { marks?: Int32Array; a?: number; b?: number }
function selCounts(pd: PairData, s: Sel): { na: number; nb: number } {
  if (!s.marks || s.a === undefined || s.a < 0) return { na: pd.n, nb: pd.n };
  let na = 0, nb = 0;
  for (let i = 0; i < pd.n; i++) { if (s.marks[i] === s.a) na++; if (s.marks[i] === s.b) nb++; }
  return { na, nb };
}
function selOk(pd: PairData, k: number, s: Sel): boolean {
  if (!s.marks || s.a === undefined || s.a < 0) return true;
  return s.marks[pd.i[k]] === s.a && s.marks[pd.j[k]] === s.b;
}

/** K̂(r) from pair data; optional types (cross-K from type a to type b). */
export function KfromPairs(pd: PairData, radii: A, sel: Sel = {}): Float64Array {
  const out = new Float64Array(radii.length), { na, nb } = selCounts(pd, sel);
  const same = !sel.marks || sel.a === undefined || sel.a < 0 || sel.a === sel.b;
  const nbOther = same ? nb - 1 : nb;
  if (pd.corr === "border") {
    for (let k = 0; k < radii.length; k++) {
      const r = radii[k];
      let cnt = 0, centres = 0;
      for (let i = 0; i < pd.n; i++) if (pd.b[i] >= r && (!sel.marks || sel.a === undefined || sel.a < 0 || sel.marks[i] === sel.a)) centres++;
      for (let q = 0; q < pd.d.length; q++) if (pd.d[q] <= r && pd.b[pd.i[q]] >= r && selOk(pd, q, sel)) cnt++;
      out[k] = centres > 0 && nbOther > 0 ? cnt / centres / (nbOther / pd.area) : NaN;
    }
    return out;
  }
  const norm = pd.area / Math.max(1, na * nbOther);
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    let s = 0;
    for (let q = 0; q < pd.d.length; q++) if (pd.d[q] <= r && selOk(pd, q, sel)) s += pd.w[q];
    out[k] = s * norm;
  }
  return out;
}

/** Epanechnikov kernel with half-width h. */
export const epanechnikov = (u: number, h: number): number => (Math.abs(u) < h ? 0.75 / h * (1 - (u / h) ** 2) : 0);

/** ĝ(r) from pair data (Epanechnikov kernel of half-width h); border correction uses translation-free weights 1. */
export function pcfFromPairs(pd: PairData, radii: A, h: number, sel: Sel = {}): Float64Array {
  const out = new Float64Array(radii.length), { na, nb } = selCounts(pd, sel);
  const same = !sel.marks || sel.a === undefined || sel.a < 0 || sel.a === sel.b;
  const norm = pd.area / Math.max(1, na * (same ? nb - 1 : nb));
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    if (r <= 0) { out[k] = NaN; continue; }
    let s = 0;
    for (let q = 0; q < pd.d.length; q++) {
      const u = r - pd.d[q];
      if (u <= -h || u >= h || !selOk(pd, q, sel)) continue;
      s += pd.w[q] * epanechnikov(u, h) / (2 * Math.PI * Math.max(pd.d[q], 1e-9));
    }
    out[k] = s * norm;
  }
  return out;
}

const rmaxOf = (radii: A, extra = 0) => Math.max(...Array.from(radii)) + extra;

/** K̂(r) with an edge correction. */
export const Kest = (p: Points, win: Window, radii: A, corr: KCorr = "translation"): Float64Array =>
  KfromPairs(pairData(p, win, rmaxOf(radii), corr), radii);

/** L(r) = √(K/π). */
export const Lfun = (K: A): Float64Array => Float64Array.from(K, v => Math.sqrt(Math.max(0, v) / Math.PI));
/** L(r) − r. */
export const LminusR = (K: A, radii: A): Float64Array => Float64Array.from(K, (v, k) => Math.sqrt(Math.max(0, v) / Math.PI) - radii[k]);

/** ĝ(r) (bandwidth h; border correction falls back to translation weights). */
export const pcf = (p: Points, win: Window, radii: A, h: number, corr: KCorr = "translation"): Float64Array =>
  pcfFromPairs(pairData(p, win, rmaxOf(radii, h), corr === "border" ? "translation" : corr), radii, h);

/** Cross-K from type a to type b of a marked pattern. */
export const Kcross = (p: Points, win: Window, radii: A, a: number, b: number, corr: KCorr = "translation"): Float64Array =>
  KfromPairs(pairData(p, win, rmaxOf(radii), corr), radii, { marks: p.marks, a, b });

/** Cross pair correlation from type a to type b. */
export const pcfCross = (p: Points, win: Window, radii: A, h: number, a: number, b: number, corr: KCorr = "translation"): Float64Array =>
  pcfFromPairs(pairData(p, win, rmaxOf(radii, h), corr === "border" ? "translation" : corr), radii, h, { marks: p.marks, a, b });

/** Mark connection function p_ab(r): of the (ordered) pairs at distance r, the fraction with
 *  types {a, b} in either order (a ≠ b), or a–a (a = b). Under random labelling: 2·p_a·p_b, or p_a². */
export function markConnection(pd: PairData, marks: Int32Array, radii: A, h: number, a: number, b: number): Float64Array {
  const out = new Float64Array(radii.length);
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    let num = 0, den = 0;
    for (let q = 0; q < pd.d.length; q++) {
      const u = r - pd.d[q];
      if (u <= -h || u >= h) continue;
      const wt = pd.w[q] * epanechnikov(u, h);
      den += wt;
      const mi = marks[pd.i[q]], mj = marks[pd.j[q]];
      if ((mi === a && mj === b) || (a !== b && mi === b && mj === a)) num += wt;
    }
    out[k] = den > 0 ? num / den : NaN;
  }
  return out;
}

/** Inhomogeneous K: (1/|W|) Σ_{i≠j} w_ij [d_ij ≤ r] / (ρ(x_i) ρ(x_j)), ρ at the points given. */
export function KinhomFromPairs(pd: PairData, rhoAt: A, radii: A): Float64Array {
  const out = new Float64Array(radii.length);
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k];
    let s = 0;
    for (let q = 0; q < pd.d.length; q++) if (pd.d[q] <= r) s += pd.w[q] / (rhoAt[pd.i[q]] * rhoAt[pd.j[q]]);
    out[k] = s / pd.area;
  }
  return out;
}
export const Kinhom = (p: Points, win: Window, radii: A, rhoAt: A, corr: KCorr = "translation"): Float64Array =>
  KinhomFromPairs(pairData(p, win, rmaxOf(radii), corr === "border" ? "translation" : corr), rhoAt, radii);

/** Thomas process: K(r) = πr² + (1 − exp(−r²/4σ²))/κ. */
export const thomasK = (r: number, kappa: number, sigma: number): number => Math.PI * r * r + (1 - Math.exp(-r * r / (4 * sigma * sigma))) / kappa;
/** Thomas process: g(r) = 1 + exp(−r²/4σ²)/(4πκσ²). */
export const thomasPcf = (r: number, kappa: number, sigma: number): number => 1 + Math.exp(-r * r / (4 * sigma * sigma)) / (4 * Math.PI * kappa * sigma * sigma);

// ---------------------------------------------------------------------------------------
// Monte Carlo tests

/** Monte Carlo p-values of an observed statistic against simulated ones (ties count against H0). */
export function mcRank(obs: number, sims: A): { rank: number; pUpper: number; pLower: number; pTwo: number } {
  let ge = 0, le = 0;
  for (let k = 0; k < sims.length; k++) { if (sims[k] >= obs) ge++; if (sims[k] <= obs) le++; }
  const s = sims.length, pUpper = (ge + 1) / (s + 1), pLower = (le + 1) / (s + 1);
  return { rank: s + 1 - ge, pUpper, pLower, pTwo: Math.min(1, 2 * Math.min(pUpper, pLower)) };
}

/** Pointwise min/max envelope (and mean) of simulated curves. */
export function pointwiseEnvelope(sims: A[]): { lo: Float64Array; hi: Float64Array; mean: Float64Array } {
  const m = sims[0].length, lo = new Float64Array(m).fill(Infinity), hi = new Float64Array(m).fill(-Infinity), mean = new Float64Array(m);
  for (const c of sims) for (let k = 0; k < m; k++) { lo[k] = Math.min(lo[k], c[k]); hi[k] = Math.max(hi[k], c[k]); mean[k] += c[k] / sims.length; }
  return { lo, hi, mean };
}

/** Indices of the radii inside [r1, r2]. */
export function rangeIdx(radii: A, range: [number, number]): number[] {
  const out: number[] = [];
  for (let k = 0; k < radii.length; k++) if (radii[k] >= range[0] - 1e-9 && radii[k] <= range[1] + 1e-9) out.push(k);
  return out;
}

/** Does the curve leave the pointwise envelope somewhere in the index set? */
export function escapes(curve: A, env: { lo: A; hi: A }, idx: number[]): number[] {
  return idx.filter(k => curve[k] < env.lo[k] - 1e-12 || curve[k] > env.hi[k] + 1e-12);
}

/** Maximum absolute deviation from a reference curve over the index set. */
export function madStat(curve: A, ref: A, idx: number[]): number {
  let m = 0;
  for (const k of idx) if (Number.isFinite(curve[k])) m = Math.max(m, Math.abs(curve[k] - ref[k]));
  return m;
}

/** DCLF statistic: Σ (curve − ref)² over the index set (times the r step). */
export function dclfStat(curve: A, ref: A, idx: number[], dr = 1): number {
  let s = 0;
  for (const k of idx) if (Number.isFinite(curve[k])) s += (curve[k] - ref[k]) ** 2 * dr;
  return s;
}

/** MAD test: T = max |curve − ref| over the range, ranked among the simulations; and the global
 *  envelope ref ± T_crit, T_crit = the k-th largest simulated T with k = ⌊α(s + 1)⌋. */
export function madTest(obs: A, sims: A[], ref: A, idx: number[], alpha = 0.05): { T0: number; Ts: Float64Array; p: number; rank: number; Tcrit: number; lo: Float64Array; hi: Float64Array } {
  const T0 = madStat(obs, ref, idx), Ts = Float64Array.from(sims, c => madStat(c, ref, idx));
  const { pUpper, rank } = mcRank(T0, Ts);
  const sorted = Array.from(Ts).sort((a, b) => b - a), k = Math.max(1, Math.floor(alpha * (sims.length + 1)));
  const Tcrit = sorted[Math.min(sorted.length - 1, k - 1)];
  const lo = Float64Array.from(ref, v => v - Tcrit), hi = Float64Array.from(ref, v => v + Tcrit);
  return { T0, Ts, p: pUpper, rank, Tcrit, lo, hi };
}

/** Pooled K of several images: Σ n_k² K_k / Σ n_k². */
export function pooledK(curves: A[], counts: A): Float64Array {
  const m = curves[0].length, out = new Float64Array(m);
  let W = 0;
  curves.forEach((c, k) => { const w = counts[k] * counts[k]; W += w; for (let i = 0; i < m; i++) out[i] += w * c[i]; });
  for (let i = 0; i < m; i++) out[i] /= W;
  return out;
}
