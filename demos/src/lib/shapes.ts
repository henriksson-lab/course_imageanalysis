// Synthetic landmark shapes, Procrustes alignment and point distribution models.
// Owner: snakes day (demos procrustes, shapepca, asm). No DOM.
//
// A shape is n landmarks stored interleaved, Float64Array [x0, y0, x1, y1, ...], in image
// orientation (x right, y down). Every family generates its landmarks from a few hidden
// parameters, so landmark i means the same anatomical point on every shape (built-in
// correspondence). Training sets add landmark noise and a random similarity pose, with a
// fixed seed, so the lecture and the exercise show the same set.
//
// Procrustes (2D, closed form): align p onto q (both centred) with the matrix
//   sR = [[a, −b], [b, a]],  a = Σ p·q / Σ|p|²,  b = Σ (p_x q_y − p_y q_x) / Σ|p|².
// (The plan puts the 2 × 2 Procrustes in lib/linalg.ts; it lives here because linalg.ts is
// owned by another day. It only needs closed forms.)
import { rng } from "./random";
import { jacobiEigen } from "./linalg";
import { Grey, createGrey, gaussianBlur } from "./imageops";

export type Shape = Float64Array;
type R = ReturnType<typeof rng>;

export const nPoints = (s: Shape): number => s.length >> 1;

// ---------------------------------------------------------------------------------------
// Shape families

export interface ParamSpec { name: string; mean: number; sd: number; min?: number; max?: number }

export interface Family {
  id: string;
  label: string;
  /** Number of landmarks. */
  n: number;
  /** Hidden parameters (independent Gaussians, clamped to [min, max]). */
  params: ParamSpec[];
  /** Landmarks of the shape with these parameter values, centred near 0, size about 1. */
  make(p: number[]): Shape;
  /** Indices of the "anatomical" landmarks (tips, corners), drawn larger. */
  anchors: number[];
}

/** Kidney-shaped nucleus outline: 32 landmarks, 0 at the dent. Parameters: elongation, dent depth, bend. */
const nucleus: Family = {
  id: "nucleus", label: "nucleus", n: 32,
  params: [
    { name: "elongation", mean: 0.15, sd: 0.1, min: -0.15, max: 0.45 },
    { name: "dent", mean: 0.22, sd: 0.1, min: 0, max: 0.5 },
    { name: "bend", mean: 0, sd: 0.15 },
  ],
  anchors: [0, 8, 16, 24],
  make([e, d, c]) {
    const n = 32, s = new Float64Array(2 * n), a = 0.62 * (1 + e), b = 0.62 * (1 - e) * 0.85;
    for (let i = 0; i < n; i++) {
      const t = Math.PI + 2 * Math.PI * i / n;                     // i = 0 at the left (the dent)
      const dt = Math.atan2(Math.sin(t - Math.PI), Math.cos(t - Math.PI));
      const r = 1 - d * Math.exp(-((dt / 0.55) ** 2));
      let x = a * Math.cos(t) * r, y = b * Math.sin(t) * r;
      y += c * (x * x - 0.15);                                     // banana bend
      s[2 * i] = x; s[2 * i + 1] = y;
    }
    return s;
  },
};

/** Leaf outline: 24 landmarks (0 = base, 12 = tip). Parameters: width, where widest, midrib bend. */
const leaf: Family = {
  id: "leaf", label: "leaf", n: 24,
  params: [
    { name: "width", mean: 0.36, sd: 0.07, min: 0.15, max: 0.6 },
    { name: "widest", mean: 0, sd: 0.35, min: -0.8, max: 0.8 },
    { name: "bend", mean: 0, sd: 0.15 },
  ],
  anchors: [0, 12],
  make([W, q, c]) {
    const n = 24, s = new Float64Array(2 * n), half = 12;
    const at = (u: number, side: number, k: number) => {       // u in [0, 1] from base to tip
      const w = W * Math.sin(Math.PI * u) ** 0.9 * (1 + q * (0.5 - u)) * side;
      const y = 1 - 2 * u, x = w + c * Math.sin(Math.PI * u) * 1.2;
      s[2 * k] = x; s[2 * k + 1] = y;
    };
    for (let k = 0; k <= half; k++) at(k / half, 1, k);           // right side, base → tip
    for (let k = 1; k < half; k++) at(1 - k / half, -1, half + k); // left side, tip → base
    return s;
  },
};

/** Hand outline (palm view, fingers up): 29 landmarks. Parameters: finger spread, thumb angle, finger length. */
const hand: Family = {
  id: "hand", label: "hand", n: 29,
  params: [
    { name: "spread", mean: 0, sd: 1, min: -2.2, max: 2.2 },
    { name: "thumb", mean: 0, sd: 0.25, min: -0.6, max: 0.6 },
    { name: "length", mean: 1, sd: 0.08, min: 0.8, max: 1.2 },
  ],
  anchors: [0, 4, 9, 14, 19, 24, 28],
  make([spread, thumb, len]) {
    const pts: number[] = [];
    const push = (x: number, y: number) => { pts.push(x, y); };
    const finger = (bx: number, by: number, ang: number, L: number, hw: number) => {
      // ang: angle from straight up, positive = towards +x; points: base-l, mid-l, tip, mid-r, base-r
      const dx = Math.sin(ang), dy = -Math.cos(ang), px = Math.cos(ang), py = Math.sin(ang);
      push(bx - hw * px, by - hw * py);
      push(bx + 0.8 * L * dx - 0.8 * hw * px, by + 0.8 * L * dy - 0.8 * hw * py);
      push(bx + L * dx, by + L * dy);
      push(bx + 0.8 * L * dx + 0.8 * hw * px, by + 0.8 * L * dy + 0.8 * hw * py);
      push(bx + hw * px, by + hw * py);
    };
    push(-0.36, 0.62);                                              // 0: wrist, left
    push(-0.42, 0.36);                                              // 1: palm, left side
    // thumb from the left side of the palm, pointing up and out (base-l is the lower point)
    const ta = -1.0 + thumb;                                        // angle from straight up
    {
      const bx = -0.44, by = 0.16, L = 0.48 * len, hw = 0.1;
      const dx = Math.sin(ta), dy = -Math.cos(ta), px = Math.cos(ta), py = Math.sin(ta);
      // going clockwise on screen around the hand: lower edge of the thumb first (towards −p)
      push(bx - hw * px * 0 - 0.02, by + hw);                       // 2: thumb base, lower
      push(bx + 0.55 * L * dx - 0.85 * hw * px, by + 0.55 * L * dy - 0.85 * hw * py); // 3
      push(bx + L * dx, by + L * dy);                               // 4: thumb tip
      push(bx + 0.55 * L * dx + 0.85 * hw * px, by + 0.55 * L * dy + 0.85 * hw * py); // 5
      push(bx + 0.06, by - hw * 1.2);                               // 6: thumb base, upper
    }
    const bases = [-0.29, -0.1, 0.09, 0.27], lens = [0.66, 0.74, 0.69, 0.53], offs = [-1.5, -0.5, 0.5, 1.5];
    for (let j = 0; j < 4; j++) {
      const ang = offs[j] * 0.08 + spread * offs[j] * 0.13;
      finger(bases[j], -0.12, ang, lens[j] * len, 0.08);           // 7–26
    }
    push(0.42, 0.1);                                                // 27: palm, right side
    push(0.38, 0.62);                                               // 28: wrist, right
    return Float64Array.from(pts);
  },
};

export const FAMILIES: Family[] = [hand, nucleus, leaf];

export function family(id: string | undefined): Family {
  return FAMILIES.find(f => f.id === id) ?? FAMILIES[0];
}

/** Draw hidden parameters (clamped). */
export function sampleParams(f: Family, r: R, spreadScale = 1): number[] {
  return f.params.map(p => {
    const v = p.mean + p.sd * spreadScale * r.norm();
    return Math.max(p.min ?? -Infinity, Math.min(p.max ?? Infinity, v));
  });
}

export const meanParams = (f: Family): number[] => f.params.map(p => p.mean);

// ---------------------------------------------------------------------------------------
// Similarity transforms

export interface Pose { s: number; theta: number; tx: number; ty: number; reflect?: boolean }

/** Apply x ↦ s R(θ) x + t (θ counter-clockwise in the maths sense of the stored numbers). */
export function transform(sh: Shape, P: Pose): Shape {
  const c = P.s * Math.cos(P.theta), d = P.s * Math.sin(P.theta), out = new Float64Array(sh.length);
  for (let i = 0; i < sh.length; i += 2) {
    const x = P.reflect ? -sh[i] : sh[i], y = sh[i + 1];
    out[i] = c * x - d * y + P.tx; out[i + 1] = d * x + c * y + P.ty;
  }
  return out;
}

/** Apply the 2 × 2 matrix [[a, −b], [b, a]] and a translation. */
export function applySim(sh: Shape, a: number, b: number, tx = 0, ty = 0): Shape {
  const out = new Float64Array(sh.length);
  for (let i = 0; i < sh.length; i += 2) { out[i] = a * sh[i] - b * sh[i + 1] + tx; out[i + 1] = b * sh[i] + a * sh[i + 1] + ty; }
  return out;
}

export function centroid(sh: Shape): [number, number] {
  let x = 0, y = 0;
  const n = nPoints(sh);
  for (let i = 0; i < sh.length; i += 2) { x += sh[i]; y += sh[i + 1]; }
  return [x / n, y / n];
}

export function translateShape(sh: Shape, dx: number, dy: number): Shape {
  const out = sh.slice();
  for (let i = 0; i < sh.length; i += 2) { out[i] += dx; out[i + 1] += dy; }
  return out;
}

export function centre(sh: Shape): Shape {
  const [cx, cy] = centroid(sh);
  return translateShape(sh, -cx, -cy);
}

/** Centroid size √(Σ |x_i − x̄|²). */
export function centroidSize(sh: Shape): number {
  const [cx, cy] = centroid(sh);
  let s = 0;
  for (let i = 0; i < sh.length; i += 2) s += (sh[i] - cx) ** 2 + (sh[i + 1] - cy) ** 2;
  return Math.sqrt(s);
}

export function scaleShape(sh: Shape, k: number): Shape { return sh.map(v => v * k); }

export function sumSq(a: Shape, b: Shape): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

// ---------------------------------------------------------------------------------------
// Procrustes

export interface ProcrustesFit {
  /** sR = [[a, −b], [b, a]] (times diag(−1, 1) on the right when reflected). */
  a: number; b: number;
  s: number; theta: number;
  reflect: boolean;
  /** Translation applied after sR: aligned = sR p + t. */
  tx: number; ty: number;
  /** Residual sum of squares Σ |aligned_i − q_i|². */
  rss: number;
  aligned: Shape;
}

/**
 * Least-squares similarity aligning p onto q: minimise Σ |sR p_i + t − q_i|².
 * Options: scale (default true; false = rigid), reflection (default false: R is a rotation).
 */
export function procrustes(p: Shape, q: Shape, opts: { scale?: boolean; reflection?: boolean } = {}): ProcrustesFit {
  const useScale = opts.scale !== false;
  const [px, py] = centroid(p), [qx, qy] = centroid(q);
  const fit = (refl: boolean): ProcrustesFit => {
    let spp = 0, sab = 0, scr = 0;
    for (let i = 0; i < p.length; i += 2) {
      const x = (refl ? -1 : 1) * (p[i] - px), y = p[i + 1] - py, u = q[i] - qx, v = q[i + 1] - qy;
      spp += x * x + y * y; sab += x * u + y * v; scr += x * v - y * u;
    }
    let a = spp > 0 ? sab / spp : 1, b = spp > 0 ? scr / spp : 0;
    if (!useScale) { const h = Math.hypot(a, b) || 1; a /= h; b /= h; }
    const s = Math.hypot(a, b), theta = Math.atan2(b, a);
    // aligned = M (p − p̄) + q̄, M = [[a, −b], [b, a]] · diag(±1, 1)
    const aligned = new Float64Array(p.length);
    let rss = 0;
    for (let i = 0; i < p.length; i += 2) {
      const x = (refl ? -1 : 1) * (p[i] - px), y = p[i + 1] - py;
      aligned[i] = a * x - b * y + qx; aligned[i + 1] = b * x + a * y + qy;
      rss += (aligned[i] - q[i]) ** 2 + (aligned[i + 1] - q[i + 1]) ** 2;
    }
    // t such that aligned = M p + t
    const mx = (refl ? -a : a) * px - b * py, my = (refl ? -b : b) * px + a * py;
    return { a, b, s, theta, reflect: refl, tx: qx - mx, ty: qy - my, rss, aligned };
  };
  const f0 = fit(false);
  if (!opts.reflection) return f0;
  const f1 = fit(true);
  return f1.rss < f0.rss - 1e-12 ? f1 : f0;
}

/** Inverse of the fitted similarity applied to y: M⁻¹ (y − t). */
export function invertFit(f: ProcrustesFit, y: Shape): Shape {
  const det = f.a * f.a + f.b * f.b, out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i += 2) {
    const u = y[i] - f.tx, v = y[i + 1] - f.ty;
    // inverse of [[a, −b], [b, a]] is [[a, b], [−b, a]] / det
    let x = (f.a * u + f.b * v) / det;
    const yy = (-f.b * u + f.a * v) / det;
    if (f.reflect) x = -x;
    out[i] = x; out[i + 1] = yy;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Generalised Procrustes analysis

export interface GPAResult {
  aligned: Shape[];
  mean: Shape;
  /** Change of the mean (sum of squares) after each iteration. */
  changes: number[];
  iterations: number;
}

/** Centre a shape and scale it to centroid size 1. */
export function normalise(sh: Shape): Shape {
  const c = centre(sh), k = centroidSize(c);
  return k > 0 ? scaleShape(c, 1 / k) : c;
}

/**
 * GPA: centre and normalise every shape, take shape 0 as the first mean; then repeat
 * { align every shape to the mean (rotation + scale); mean = average; normalise the mean and
 * align it to the reference shape 0 (fixes its pose) } until the mean changes by less than `tol`.
 */
export function gpa(shapes: Shape[], opts: { maxIter?: number; tol?: number; reflection?: boolean } = {}): GPAResult {
  const maxIter = opts.maxIter ?? 20, tol = opts.tol ?? 1e-10;
  const ref = normalise(shapes[0]);
  let mean: Shape = ref.slice();
  let aligned = shapes.map(normalise);
  const changes: number[] = [];
  let it = 0;
  for (; it < maxIter; it++) {
    aligned = aligned.map(s => procrustes(s, mean, { reflection: opts.reflection }).aligned);
    const m = new Float64Array(mean.length);
    for (const s of aligned) for (let i = 0; i < m.length; i++) m[i] += s[i] / aligned.length;
    const mn = procrustes(normalise(m), ref, { scale: false }).aligned;
    const ch = sumSq(mn, mean);
    changes.push(ch);
    mean = normalise(mn);
    if (ch < tol) { it++; break; }
  }
  aligned = aligned.map(s => procrustes(s, mean, { reflection: opts.reflection }).aligned);
  return { aligned, mean, changes, iterations: it };
}

/** One GPA iteration from a given mean (for step-by-step display). */
export function gpaStep(aligned: Shape[], mean: Shape, ref: Shape): { aligned: Shape[]; mean: Shape; change: number } {
  const al = aligned.map(s => procrustes(s, mean).aligned);
  const m = new Float64Array(mean.length);
  for (const s of al) for (let i = 0; i < m.length; i++) m[i] += s[i] / al.length;
  const mn = normalise(procrustes(normalise(m), ref, { scale: false }).aligned);
  return { aligned: al, mean: mn, change: sumSq(mn, mean) };
}

// ---------------------------------------------------------------------------------------
// Point distribution model

export interface PDM {
  mean: Shape;
  /** Eigenvalues, descending (variances of the b_k), length 2n. */
  values: Float64Array;
  /** Mode k as a shape-sized vector (unit length): modes[k]. */
  modes: Float64Array[];
  /** Total variance (trace of S). */
  total: number;
}

/** PCA of aligned shapes; covariance divided by the number of shapes s (as in the PCA unit). */
export function pdm(aligned: Shape[], mean?: Shape): PDM {
  const s = aligned.length, d = aligned[0].length;
  const mu = mean ?? (() => { const m = new Float64Array(d); for (const a of aligned) for (let i = 0; i < d; i++) m[i] += a[i] / s; return m; })();
  const S = new Float64Array(d * d);
  for (const a of aligned) {
    for (let i = 0; i < d; i++) {
      const yi = a[i] - mu[i];
      if (yi === 0) continue;
      for (let j = i; j < d; j++) S[i * d + j] += yi * (a[j] - mu[j]) / s;
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) S[i * d + j] = S[j * d + i];
  const { values, vectors } = jacobiEigen(S, d, true);
  const modes: Float64Array[] = [];
  for (let k = 0; k < d; k++) {
    const v = new Float64Array(d);
    for (let r = 0; r < d; r++) v[r] = vectors[r * d + k];
    // sign convention: largest component positive
    let big = 0;
    for (let r = 0; r < d; r++) if (Math.abs(v[r]) > Math.abs(big)) big = v[r];
    if (big < 0) for (let r = 0; r < d; r++) v[r] = -v[r];
    modes.push(v);
  }
  let total = 0;
  for (let i = 0; i < d; i++) total += S[i * d + i];
  for (let k = 0; k < d; k++) values[k] = Math.max(0, values[k]);
  return { mean: mu, values, modes, total };
}

/** Number of modes that explain at least `frac` of the variance. */
export function modesFor(m: PDM, frac: number): number {
  let c = 0;
  for (let k = 0; k < m.values.length; k++) { c += m.values[k]; if (c >= frac * m.total - 1e-15) return k + 1; }
  return m.values.length;
}

/** x̄ + Σ_k b_k φ_k. */
export function synthesise(m: PDM, b: ArrayLike<number>): Shape {
  const x = m.mean.slice();
  for (let k = 0; k < b.length; k++) { const v = m.modes[k]; for (let i = 0; i < x.length; i++) x[i] += b[k] * v[i]; }
  return x;
}

/** b = Φᵀ (x − x̄) for the first k modes. */
export function projectB(m: PDM, x: Shape, k: number): Float64Array {
  const b = new Float64Array(k);
  for (let j = 0; j < k; j++) { let s = 0; const v = m.modes[j]; for (let i = 0; i < x.length; i++) s += v[i] * (x[i] - m.mean[i]); b[j] = s; }
  return b;
}

/** Clamp each |b_k| ≤ limit·√λ_k. Returns a new array. */
export function clampB(m: PDM, b: ArrayLike<number>, limit = 3): Float64Array {
  return Float64Array.from(b, (v, k) => { const L = limit * Math.sqrt(m.values[k]); return Math.max(-L, Math.min(L, v)); });
}

// ---------------------------------------------------------------------------------------
// Training sets

export interface SetOptions {
  count: number;
  seed?: number;
  /** Landmark noise SD, in shape units (shapes are about 2 units tall). */
  noise?: number;
  /** Random pose: rotation SD (radians), scale range (log SD), translation SD. */
  rot?: number; logScale?: number; shift?: number;
  /** Fraction of shapes whose landmark numbering is shifted by one (bad correspondence). */
  corrupt?: number;
}

export interface TrainingSet { shapes: Shape[]; params: number[][]; poses: Pose[]; corrupted: boolean[] }

export function trainingSet(f: Family, o: SetOptions): TrainingSet {
  const r = rng(o.seed ?? 1);
  const shapes: Shape[] = [], params: number[][] = [], poses: Pose[] = [], corrupted: boolean[] = [];
  for (let j = 0; j < o.count; j++) {
    const p = sampleParams(f, r);
    let s: Shape = f.make(p);
    const noise = o.noise ?? 0;
    const nz = Float64Array.from(s, () => r.norm());
    if (noise > 0) s = s.map((v, i) => v + noise * nz[i]);
    const P: Pose = { s: Math.exp((o.logScale ?? 0) * r.norm()), theta: (o.rot ?? 0) * r.norm(), tx: (o.shift ?? 0) * r.norm(), ty: (o.shift ?? 0) * r.norm() };
    s = transform(s, P);
    const bad = r.unif() < (o.corrupt ?? 0);
    if (bad) s = shiftLandmarks(s, 1);
    shapes.push(s); params.push(p); poses.push(P); corrupted.push(bad);
  }
  return { shapes, params, poses, corrupted };
}

/** Renumber landmarks cyclically by `k` (landmark i gets the position of landmark i + k). */
export function shiftLandmarks(s: Shape, k: number): Shape {
  const n = nPoints(s), out = new Float64Array(s.length);
  for (let i = 0; i < n; i++) { const j = ((i + k) % n + n) % n; out[2 * i] = s[2 * j]; out[2 * i + 1] = s[2 * j + 1]; }
  return out;
}

// ---------------------------------------------------------------------------------------
// Rasterising a shape into a grey image

/** Fraction of each pixel inside the polygon (scanline, `sub` sub-rows, exact horizontal coverage). */
export function polygonCoverage(sh: Shape, w: number, h: number, sub = 4): Float32Array {
  const out = new Float32Array(w * h), n = nPoints(sh), xs: number[] = [];
  for (let y = 0; y < h; y++) {
    for (let k = 0; k < sub; k++) {
      const yy = y - 0.5 + (k + 0.5) / sub;
      xs.length = 0;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, y0 = sh[2 * i + 1], y1 = sh[2 * j + 1];
        if ((y0 <= yy && y1 > yy) || (y1 <= yy && y0 > yy)) {
          const t = (yy - y0) / (y1 - y0);
          xs.push(sh[2 * i] + t * (sh[2 * j] - sh[2 * i]));
        }
      }
      xs.sort((a, b) => a - b);
      for (let m = 0; m + 1 < xs.length; m += 2) {
        // span [a, b) in continuous x; pixel x covers [x − 0.5, x + 0.5)
        const a = xs[m] + 0.5, b = xs[m + 1] + 0.5;
        const ia = Math.max(0, Math.floor(a)), ib = Math.min(w - 1, Math.floor(b));
        for (let x = ia; x <= ib; x++) {
          const cov = Math.min(b, x + 1) - Math.max(a, x);
          if (cov > 0) out[y * w + x] += cov / sub;
        }
      }
    }
  }
  return out;
}

export interface RenderOptions {
  width?: number; height?: number;
  /** Grey levels of background and object (0..1). */
  bg?: number; fg?: number;
  blur?: number;
  noise?: number;
  seed?: number;
  /** Number of random clutter blobs (discs of the object's grey level). */
  clutter?: number;
  /** Occluding bar across the image: [x0, y0, x1, y1, thickness] with the background grey. */
  occluder?: [number, number, number, number, number] | null;
}

/** Render the filled shape (image coordinates) with blur, clutter, an occluder and Gaussian noise. */
export function renderShape(sh: Shape, o: RenderOptions = {}): Grey {
  const w = o.width ?? 128, h = o.height ?? 128, bg = o.bg ?? 0.25, fg = o.fg ?? 0.75;
  const cov = polygonCoverage(sh, w, h);
  let img = createGrey(w, h);
  for (let i = 0; i < w * h; i++) img.data[i] = bg + (fg - bg) * Math.min(1, cov[i]);
  const r = rng(o.seed ?? 7);
  for (let c = 0; c < (o.clutter ?? 0); c++) {
    const cx = r.unif() * w, cy = r.unif() * h, rad = 3 + 4 * r.unif(), g = bg + (fg - bg) * (0.6 + 0.4 * r.unif());
    for (let y = Math.max(0, Math.floor(cy - rad - 1)); y < Math.min(h, cy + rad + 2); y++)
      for (let x = Math.max(0, Math.floor(cx - rad - 1)); x < Math.min(w, cx + rad + 2); x++) {
        const a = Math.max(0, Math.min(1, rad + 0.5 - Math.hypot(x - cx, y - cy)));
        if (a > 0) img.data[y * w + x] = img.data[y * w + x] * (1 - a) + g * a;
      }
  }
  if (o.occluder) {
    const [x0, y0, x1, y1, th] = o.occluder, L = Math.hypot(x1 - x0, y1 - y0) || 1, ux = (x1 - x0) / L, uy = (y1 - y0) / L;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const t = (x - x0) * ux + (y - y0) * uy, dist = Math.abs(-(x - x0) * uy + (y - y0) * ux);
      if (t >= 0 && t <= L) { const a = Math.max(0, Math.min(1, th / 2 + 0.5 - dist)); if (a > 0) img.data[y * w + x] = img.data[y * w + x] * (1 - a) + bg * a; }
    }
  }
  if ((o.blur ?? 0) > 0) img = gaussianBlur(img, o.blur!);
  if ((o.noise ?? 0) > 0) for (let i = 0; i < w * h; i++) img.data[i] += o.noise! * r.norm();
  return img;
}

/** Outward unit normals at the landmarks of a closed shape (central differences, orientation-aware). */
export function shapeNormals(sh: Shape): Float64Array {
  const n = nPoints(sh), out = new Float64Array(2 * n);
  let area = 0;
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; area += sh[2 * i] * sh[2 * j + 1] - sh[2 * j] * sh[2 * i + 1]; }
  const sg = area >= 0 ? 1 : -1;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, k = (i - 1 + n) % n;
    const tx = sh[2 * j] - sh[2 * k], ty = sh[2 * j + 1] - sh[2 * k + 1], L = Math.hypot(tx, ty) || 1;
    out[2 * i] = sg * ty / L; out[2 * i + 1] = -sg * tx / L;
  }
  return out;
}
