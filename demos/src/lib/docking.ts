// FFT docking in 2D (owner: shape-matching day, unit fft-docking; no DOM, runs under node).
// A receptor on a grid as a thin surface layer (value 1) around a core (value ρ < 0); a ligand
// as 1 on its cells. The score of a ligand pose is the sum of the receptor grid under the
// ligand's cells: contacts count +1, clashes ρ. For one ligand angle the score of every shift
// is one circular correlation, computed with the FFT (Katchalski-Katzir et al. 1992); the
// receptor's transform is computed once and cached, the ligand's once per angle.
//
// Conventions: grids are row-major (index y·w + x), y down. A ligand is a list of cell
// offsets from its reference cell (the cell that holds its centroid); the score map S(d) is
// indexed by the position d of that reference cell: S(d) = Σ_x a(x) b(x − d) (circular).
// Angles are in degrees; positive angles turn clockwise on screen (the course convention).
import { dilateBinary, seDisk } from "./morph";
import { fft2dRect, fft1d } from "./fft";

export type Pt = [number, number];
export type Poly = Pt[];

// ---------------------------------------------------------------------------------------
// Geometry

/** Turn a polygon by `deg` degrees clockwise on screen (y down) about `about`. */
export function rotatePolyCW(p: Poly, deg: number, about: Pt = [0, 0]): Poly {
  const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return p.map(([x, y]) => {
    const dx = x - about[0], dy = y - about[1];
    return [about[0] + c * dx - s * dy, about[1] + s * dx + c * dy] as Pt;
  });
}

/** Centroid of a polygon's area. */
export function centroid(p: Poly): Pt {
  let A = 0, cx = 0, cy = 0;
  for (let i = 0; i < p.length; i++) {
    const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length], c = x0 * y1 - x1 * y0;
    A += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c;
  }
  return [cx / (3 * A), cy / (3 * A)];
}

/** The polygon moved so that its centroid is at (0, 0). */
export function centred(p: Poly): Poly {
  const [cx, cy] = centroid(p);
  return p.map(([x, y]) => [x - cx, y - cy] as Pt);
}

/** Point in polygon (even-odd rule). */
export function inside(p: Poly, x: number, y: number): boolean {
  let c = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i], [xj, yj] = p[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** Distance from a point to the outline of a polygon. */
export function distToOutline(p: Poly, x: number, y: number): number {
  let best = Infinity;
  for (let i = 0; i < p.length; i++) {
    const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length];
    const dx = x1 - x0, dy = y1 - y0, L = dx * dx + dy * dy;
    const t = L > 0 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / L)) : 0;
    best = Math.min(best, Math.hypot(x - x0 - t * dx, y - y0 - t * dy));
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// Grids

/** A receptor's two layers on a w × h grid: the core (the receptor's own cells) and the
 *  surface layer (cells within `thickness` cells of the core, outside it: dilation minus
 *  the object, with a disk). */
export interface Layers { w: number; h: number; core: Uint8Array; surface: Uint8Array }

export function layers(mask: ArrayLike<number>, w: number, h: number, thickness: number): Layers {
  const core = Uint8Array.from(mask, v => (v ? 1 : 0));
  const surface = new Uint8Array(w * h);
  if (thickness > 0) {
    const d = dilateBinary(core, w, h, seDisk(thickness));
    for (let i = 0; i < w * h; i++) surface[i] = d[i] && !core[i] ? 1 : 0;
  }
  return { w, h, core, surface };
}

/** The receptor grid a: 1 on the surface layer, ρ in the core, 0 elsewhere. */
export function receptorGrid(L: Layers, rho: number): Float64Array {
  const a = new Float64Array(L.w * L.h);
  for (let i = 0; i < a.length; i++) a[i] = L.core[i] ? rho : L.surface[i] ? 1 : 0;
  return a;
}

/** A ligand: cell offsets from its reference cell. */
export interface Ligand { dx: Int32Array; dy: Int32Array }

/** The cells of a ligand polygon (given with its centroid at (0, 0), in world units), turned by
 *  `deg` about the centroid, on a grid of spacing `h`: cell (i, j) belongs to the ligand when
 *  its centre (i·h, j·h) lies inside the turned polygon. */
export function ligandCells(poly: Poly, deg: number, h: number): Ligand {
  const p = rotatePolyCW(poly, deg);
  let r = 0;
  for (const [x, y] of p) r = Math.max(r, Math.hypot(x, y));
  const R = Math.ceil(r / h) + 1, xs: number[] = [], ys: number[] = [];
  for (let j = -R; j <= R; j++) for (let i = -R; i <= R; i++) if (inside(p, i * h, j * h)) { xs.push(i); ys.push(j); }
  return { dx: Int32Array.from(xs), dy: Int32Array.from(ys) };
}

/** The ligand as an M × M grid b with its reference cell at (0, 0), negative offsets wrapped
 *  around (so that b(x − d) puts the reference cell at d). */
export function ligandGrid(L: Ligand, M: number): Float64Array {
  const b = new Float64Array(M * M);
  for (let k = 0; k < L.dx.length; k++) b[(((L.dy[k] % M) + M) % M) * M + (((L.dx[k] % M) + M) % M)] = 1;
  return b;
}

// ---------------------------------------------------------------------------------------
// Scores

/** The score of one pose, directly: the sum of the receptor grid under the ligand's cells with
 *  its reference cell at (x, y) (wrapping around), and how many of its cells lie on the surface
 *  layer (contacts) and in the core (clashes). */
export function scoreAt(a: ArrayLike<number>, R: Layers, L: Ligand, x: number, y: number): { score: number; contact: number; clash: number } {
  const M = R.w;
  let score = 0, contact = 0, clash = 0;
  for (let k = 0; k < L.dx.length; k++) {
    const i = (((y + L.dy[k]) % M) + M) % M * M + (((x + L.dx[k]) % M) + M) % M;
    score += a[i];
    contact += R.surface[i];
    clash += R.core[i];
  }
  return { score, contact, clash };
}

/** Brute force, dense: S(d) = Σ_x a(x) b(x − d) for every shift d, over every cell of both
 *  w × h grids (circular). w·h multiply-adds per shift: the cost the FFT avoids. */
export function scoreDirect(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number): Float64Array {
  const S = new Float64Array(w * h);
  for (let dy = 0; dy < h; dy++) for (let dx = 0; dx < w; dx++) {
    let s = 0;
    for (let y = 0; y < h; y++) {
      const yb = ((y - dy) % h + h) % h;
      for (let x = 0; x < w; x++) s += a[y * w + x] * b[yb * w + ((x - dx) % w + w) % w];
    }
    S[dy * w + dx] = s;
  }
  return S;
}

/** Brute force over the ligand's cells only: for every shift, one addition per ligand cell
 *  (M² · cells additions). Same result as `scoreDirect`. */
export function scoreSparse(a: ArrayLike<number>, M: number, L: Ligand): Float64Array {
  const S = new Float64Array(M * M), n = L.dx.length;
  for (let k = 0; k < n; k++) {
    const ox = ((L.dx[k] % M) + M) % M, oy = ((L.dy[k] % M) + M) % M;
    for (let y = 0; y < M; y++) {
      const ya = (y + oy) % M;
      for (let x = 0; x < M; x++) S[y * M + x] += a[ya * M + (x + ox) % M];
    }
  }
  return S;
}

/** 1D score with the FFT (length a power of two): S(d) = Σ_x a(x) b(x − d), circular. */
export function score1dFFT(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const n = a.length;
  const ar = Float64Array.from(a), ai = new Float64Array(n), br = Float64Array.from(b), bi = new Float64Array(n);
  fft1d(ar, ai); fft1d(br, bi);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) {   // A_k · conj(B_k)
    re[k] = ar[k] * br[k] + ai[k] * bi[k];
    im[k] = ai[k] * br[k] - ar[k] * bi[k];
  }
  fft1d(re, im, true);
  return re.map(v => v / n);
}

/** FFT scorer for one receptor grid (M × M, M a power of two). The receptor's transform is
 *  computed once; every `map` call costs two FFTs (the ligand forward, the product back). */
export class Docker {
  readonly M: number;
  private Ar: Float64Array;
  private Ai: Float64Array;
  /** FFTs of size M × M done so far (the receptor's included). */
  ffts = 1;
  constructor(a: ArrayLike<number>, M: number) {
    this.M = M;
    this.Ar = Float64Array.from(a); this.Ai = new Float64Array(M * M);
    fft2dRect(this.Ar, this.Ai, M, M);
  }
  /** The score map S(d) of one ligand orientation for every shift d. */
  map(L: Ligand): Float64Array {
    const M = this.M, br = ligandGrid(L, M), bi = new Float64Array(M * M);
    fft2dRect(br, bi, M, M);
    for (let k = 0; k < M * M; k++) {   // A_k · conj(B_k), in place in (br, bi)
      const xr = this.Ar[k] * br[k] + this.Ai[k] * bi[k];
      const xi = this.Ai[k] * br[k] - this.Ar[k] * bi[k];
      br[k] = xr; bi[k] = xi;
    }
    fft2dRect(br, bi, M, M, true);
    this.ffts += 2;
    return br;
  }
}

/** Two scores in one complex correlation: receptor a = s_R + i·√w·c_R, ligand
 *  b = s_L + i·√w·c_L. Returns Re Σ_x a(x) b(x − d) = Σ s_R s_L − w Σ c_R c_L for every d
 *  (contact of the two surfaces minus w times the overlap of the two cores). The FFT route
 *  needs conj(FFT(conj b)), because the formula A·conj(B) conjugates b. */
export function complexScore(sR: ArrayLike<number>, cR: ArrayLike<number>, sL: ArrayLike<number>, cL: ArrayLike<number>, M: number, w: number): Float64Array {
  const q = Math.sqrt(w), n = M * M;
  const ar = Float64Array.from(sR), ai = Float64Array.from(cR, v => q * v);
  const br = Float64Array.from(sL), bi = Float64Array.from(cL, v => -q * v);   // conj(b)
  fft2dRect(ar, ai, M, M); fft2dRect(br, bi, M, M);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) {   // A · conj(FFT(conj b))
    re[k] = ar[k] * br[k] + ai[k] * bi[k];
    im[k] = ai[k] * br[k] - ar[k] * bi[k];
  }
  fft2dRect(re, im, M, M, true);
  return re;
}

// ---------------------------------------------------------------------------------------
// Peaks and poses

export interface Pose { angle: number; x: number; y: number; score: number }

/** The `k` highest local maxima of a score map inside the box [x0, x1) × [y0, y1), at least
 *  `minDist` cells apart (chessboard distance). */
export function localMaxima(S: ArrayLike<number>, M: number, box: [number, number, number, number], k: number, minDist = 2): { x: number; y: number; score: number }[] {
  const [x0, y0, x1, y1] = box, cand: { x: number; y: number; score: number }[] = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const v = S[y * M + x];
    let max = true;
    for (let dy = -1; dy <= 1 && max; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const u = S[(((y + dy) % M) + M) % M * M + (((x + dx) % M) + M) % M];
      if (u > v || (u === v && (dy < 0 || (dy === 0 && dx < 0)))) { max = false; break; }
    }
    if (max) cand.push({ x, y, score: v });
  }
  cand.sort((p, q) => q.score - p.score);
  const out: { x: number; y: number; score: number }[] = [];
  for (const c of cand) {
    if (out.some(o => Math.max(Math.abs(o.x - c.x), Math.abs(o.y - c.y)) < minDist)) continue;
    out.push(c);
    if (out.length >= k) break;
  }
  return out;
}

/** Signed difference of two angles in degrees, in (−180, 180]. */
export const angleDiff = (a: number, b: number): number => { const d = ((a - b) % 360 + 540) % 360 - 180; return d === -180 ? 180 : d; };

/** Non-maximum suppression across angles: the best `k` poses, dropping a pose when a better
 *  one lies within `minDist` cells AND within `minAngle` degrees of it (the same fit found at
 *  a neighbouring angle). */
export function topPoses(poses: Pose[], k: number, minDist: number, minAngle: number): Pose[] {
  const sorted = [...poses].sort((p, q) => q.score - p.score), out: Pose[] = [];
  for (const p of sorted) {
    if (out.some(o => Math.hypot(o.x - p.x, o.y - p.y) <= minDist && Math.abs(angleDiff(o.angle, p.angle)) <= minAngle)) continue;
    out.push(p);
    if (out.length >= k) break;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Counting

/** Real operations of one complex FFT of M samples (the course's estimate 5 M log₂ M). */
export const fftFlops = (M: number): number => 5 * M * Math.log2(M);

/** Number of evenly spread 3D orientations for an angular step (degrees): 8π² / Δθ³. */
export const orientations3d = (stepDeg: number): number => 8 * Math.PI * Math.PI / Math.pow(stepDeg * Math.PI / 180, 3);
