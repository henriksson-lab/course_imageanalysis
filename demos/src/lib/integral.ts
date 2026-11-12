// Integral images (summed-area tables). Owner: the integral-images day (units
// integral-image/, box-filters/, rect-features/, viola-jones/). No DOM.
//
// Convention used everywhere (demos, exercises): the table is padded with a zero row and
// column, so it is (w + 1) × (h + 1) with stride w + 1, and
//   S[y][x] = sum of I(i, j) over 0 <= i < x, 0 <= j < y      (x across, y down).
// The sum over the half-open rectangle [x0, x1) × [y0, y1) is then
//   S[y1][x1] − S[y0][x1] − S[y1][x0] + S[y0][x0]               (D − B − C + A)
// with no special cases at the image border.
//
// Tables can be built in three number types, to show the numerical traps:
//   "f64": float64 (exact for integer images up to 2^53),
//   "f32": float32, every partial sum rounded (Math.fround), as a float32 accumulator,
//   "u32": unsigned 32-bit integers with wrap-around (mod 2^32); rectangle sums are still
//          right whenever the true rectangle sum is < 2^32 (the wraps cancel).

export type TableKind = "f64" | "f32" | "u32";

export interface Table {
  /** Image width and height (the table is (w + 1) × (h + 1)). */
  w: number;
  h: number;
  kind: TableKind;
  data: Float64Array | Float32Array | Uint32Array;
}

// ---------------------------------------------------------------------------------------
// 1D

/** Prefix sums with a leading zero: P[0] = 0, P[i] = v[0] + ... + v[i-1]. */
export function prefixSum(v: ArrayLike<number>): Float64Array {
  const p = new Float64Array(v.length + 1);
  for (let i = 0; i < v.length; i++) p[i + 1] = p[i] + v[i];
  return p;
}

/** Sum of v[a..b] (inclusive, 0-based) from the prefix sums: P[b + 1] − P[a]. */
export function rangeSum(p: ArrayLike<number>, a: number, b: number): number {
  return p[b + 1] - p[a];
}

// ---------------------------------------------------------------------------------------
// 2D tables

/**
 * Summed-area table of `img` (w × h, row-major), optionally of f(value) (e.g. v => v * v).
 * Built in one raster pass with S(x, y) = I + S(left) + S(up) − S(up-left).
 */
export function integralImage(img: ArrayLike<number>, w: number, h: number, kind: TableKind = "f64",
  f?: (v: number) => number): Table {
  const W = w + 1;
  const n = W * (h + 1);
  if (kind === "u32") {
    const S = new Uint32Array(n);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const v = f ? f(img[y * w + x]) : img[y * w + x];
        const i = (y + 1) * W + x + 1;
        S[i] = (((v >>> 0) + S[i - 1] + S[i - W]) - S[i - W - 1]) >>> 0;
      }
    return { w, h, kind, data: S };
  }
  if (kind === "f32") {
    const S = new Float32Array(n), fr = Math.fround;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const v = fr(f ? f(img[y * w + x]) : img[y * w + x]);
        const i = (y + 1) * W + x + 1;
        S[i] = fr(fr(fr(v + S[i - 1]) + S[i - W]) - S[i - W - 1]);
      }
    return { w, h, kind, data: S };
  }
  const S = new Float64Array(n);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = f ? f(img[y * w + x]) : img[y * w + x];
      const i = (y + 1) * W + x + 1;
      S[i] = v + S[i - 1] + S[i - W] - S[i - W - 1];
    }
  return { w, h, kind, data: S };
}

/** Table entry S[y][x] (0 <= x <= w, 0 <= y <= h). */
export function at(T: Table, x: number, y: number): number {
  return T.data[y * (T.w + 1) + x];
}

/**
 * Sum over [x0, x1) × [y0, y1) (no clipping; corners must lie in the table). Four lookups,
 * three additions, whatever the size. For "u32" the result is taken mod 2^32; for "f32"
 * every operation is rounded to float32.
 */
export function rectSum(T: Table, x0: number, y0: number, x1: number, y1: number): number {
  const W = T.w + 1, S = T.data;
  const A = S[y0 * W + x0], B = S[y0 * W + x1], C = S[y1 * W + x0], D = S[y1 * W + x1];
  if (T.kind === "u32") return ((((D - B) >>> 0) - C + A) >>> 0);
  if (T.kind === "f32") { const fr = Math.fround; return fr(fr(fr(D - B) - C) + A); }
  return D - B - C + A;
}

/** Clip [x0, x1) × [y0, y1) to the image; returns the sum and the number of pixels inside. */
export function rectSumClipped(T: Table, x0: number, y0: number, x1: number, y1: number): { sum: number; n: number } {
  const a = Math.max(0, Math.min(T.w, x0)), b = Math.max(0, Math.min(T.w, x1));
  const c = Math.max(0, Math.min(T.h, y0)), d = Math.max(0, Math.min(T.h, y1));
  if (b <= a || d <= c) return { sum: 0, n: 0 };
  return { sum: rectSum(T, a, c, b, d), n: (b - a) * (d - c) };
}

/** Brute-force sum over [x0, x1) × [y0, y1) (reference, and the "direct" count). */
export function directSum(img: ArrayLike<number>, w: number, x0: number, y0: number, x1: number, y1: number): number {
  let s = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) s += img[y * w + x];
  return s;
}

/** Additions for one rectangle sum: direct = pixels − 1, with the table always 3. */
export function additionCounts(rw: number, rh: number): { direct: number; table: number } {
  return { direct: Math.max(0, rw * rh - 1), table: 3 };
}

// ---------------------------------------------------------------------------------------
// Box filters and local statistics

/**
 * Mean over the (2r + 1) × (2r + 1) window around each pixel, with the window clipped to
 * the image and divided by the number of pixels actually inside (no dark frame).
 * `r` may be a number or a per-pixel radius map.
 */
export function boxMean(img: ArrayLike<number>, w: number, h: number, r: number | ArrayLike<number>, T?: Table): Float64Array {
  const S = T ?? integralImage(img, w, h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const rr = typeof r === "number" ? r : r[y * w + x];
      const { sum, n } = rectSumClipped(S, x - rr, y - rr, x + rr + 1, y + rr + 1);
      out[y * w + x] = sum / n;
    }
  return out;
}

export interface LocalStatsOptions {
  /** Number type of the two tables (default f64). */
  kind?: TableKind;
  /** Subtract this value from every pixel before building the tables (variance is unchanged). */
  offset?: number;
}

export interface LocalStats {
  mean: Float64Array;
  /** Local variance mean(I²) − mean(I)², as computed (may be negative through rounding). */
  variance: Float64Array;
  /** sqrt(max(0, variance)). */
  sd: Float64Array;
  /** How many pixels got a negative variance. */
  negatives: number;
}

/**
 * Local mean, variance and standard deviation in the clipped (2r + 1)² window, from the
 * tables of I and I². With kind "f32" the final formula is evaluated in float32 too; with
 * "u32" the window sums are exact integers (if they fit 32 bits) and the formula is float64.
 */
export function localStats(img: ArrayLike<number>, w: number, h: number, r: number, opts: LocalStatsOptions = {}): LocalStats {
  const kind = opts.kind ?? "f64", c = opts.offset ?? 0;
  const shifted = c === 0 ? img : Float64Array.from({ length: w * h }, (_, i) => img[i] - c);
  // "u32" with an offset: negative values are stored modulo 2^32 (two's complement), so
  // the window sums of I − c are right modulo 2^32 too and are read back as signed.
  const T1 = integralImage(shifted, w, h, kind);
  const T2 = integralImage(shifted, w, h, kind, v => v * v);
  const mean = new Float64Array(w * h), variance = new Float64Array(w * h), sd = new Float64Array(w * h);
  const fr = Math.fround;
  let negatives = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const a = rectSumClipped(T1, x - r, y - r, x + r + 1, y + r + 1);
      const b = rectSumClipped(T2, x - r, y - r, x + r + 1, y + r + 1);
      let s1 = a.sum, s2 = b.sum;
      if (kind === "u32") s1 = s1 | 0; // signed reading of a window sum of I − c (|sum| < 2^31)
      let m: number, v: number;
      if (kind === "f32") { m = fr(s1 / a.n); v = fr(fr(s2 / a.n) - fr(m * m)); }
      else { m = s1 / a.n; v = s2 / a.n - m * m; }
      const i = y * w + x;
      mean[i] = m + c; variance[i] = v; sd[i] = Math.sqrt(Math.max(0, v));
      if (v < 0) negatives++;
    }
  return { mean, variance, sd, negatives };
}

/** Exact local variance by two passes over each window (reference; O(N r²)). */
export function localVarianceDirect(img: ArrayLike<number>, w: number, h: number, r: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      let s = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { s += img[yy * w + xx]; n++; }
      const m = s / n;
      let q = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { const d = img[yy * w + xx] - m; q += d * d; }
      out[y * w + x] = q / n;
    }
  return out;
}

/** Local contrast normalisation (I − m) / (s + eps). */
export function contrastNormalise(img: ArrayLike<number>, st: LocalStats, eps: number): Float64Array {
  return Float64Array.from({ length: st.mean.length }, (_, i) => (img[i] - st.mean[i]) / (st.sd[i] + eps));
}

/** Bits needed for the table of a b-bit image with n pixels: b + ceil(log2 n). */
export function bitsNeeded(bitDepth: number, pixels: number): number {
  return bitDepth + Math.ceil(Math.log2(pixels));
}

// ---------------------------------------------------------------------------------------
// Box filters by four methods (same clipped-window rule, so the outputs agree); used by
// boxspeed to time them honestly. All take a k × k window, k odd.

export type BoxMethod = "direct" | "separable" | "running" | "integral";

/** Additions per pixel of each method for a k × k box (border effects ignored). */
export function boxOpsPerPixel(method: BoxMethod, k: number): number {
  switch (method) {
    case "direct": return k * k - 1;
    case "separable": return 2 * (k - 1);
    case "running": return 4;    // per pass: add the entering, subtract the leaving pixel
    case "integral": return 6;   // build: 3 per pixel; lookup: 3 per pixel
  }
}

/** Box mean by the chosen method (clipped window, divided by the pixel count). */
export function boxFilterBy(method: BoxMethod, img: ArrayLike<number>, w: number, h: number, k: number): Float64Array {
  const r = (k - 1) >> 1;
  const out = new Float64Array(w * h);
  if (method === "integral") return boxMean(img, w, h, r);
  if (method === "direct") {
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
        let s = 0;
        for (let yy = y0; yy < y1; yy++) { const row = yy * w; for (let xx = x0; xx < x1; xx++) s += img[row + xx]; }
        out[y * w + x] = s / ((x1 - x0) * (y1 - y0));
      }
    }
    return out;
  }
  // separable and running: row sums, then column sums, then divide by the clipped count
  const tmp = new Float64Array(w * h);
  if (method === "separable") {
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
        let s = 0;
        for (let xx = x0; xx < x1; xx++) s += img[y * w + xx];
        tmp[y * w + x] = s;
      }
    for (let x = 0; x < w; x++)
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
        let s = 0;
        for (let yy = y0; yy < y1; yy++) s += tmp[yy * w + x];
        out[y * w + x] = s;
      }
  } else {
    for (let y = 0; y < h; y++) {
      let s = 0;
      for (let xx = 0; xx < Math.min(w, r + 1); xx++) s += img[y * w + xx];
      for (let x = 0; x < w; x++) {
        tmp[y * w + x] = s;
        if (x + r + 1 < w) s += img[y * w + x + r + 1];
        if (x - r >= 0) s -= img[y * w + x - r];
      }
    }
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let yy = 0; yy < Math.min(h, r + 1); yy++) s += tmp[yy * w + x];
      for (let y = 0; y < h; y++) {
        out[y * w + x] = s;
        if (y + r + 1 < h) s += tmp[(y + r + 1) * w + x];
        if (y - r >= 0) s -= tmp[(y - r) * w + x];
      }
    }
  }
  for (let y = 0; y < h; y++) {
    const ny = Math.min(h, y + r + 1) - Math.max(0, y - r);
    for (let x = 0; x < w; x++) out[y * w + x] /= ny * (Math.min(w, x + r + 1) - Math.max(0, x - r));
  }
  return out;
}

/** 1D impulse response of n passes of a box of width k (odd): a vector of length n(k−1)+1, sum 1. */
export function repeatedBox1d(k: number, n: number): Float64Array {
  let v = Float64Array.of(1);
  for (let p = 0; p < n; p++) {
    const out = new Float64Array(v.length + k - 1);
    for (let i = 0; i < v.length; i++) for (let j = 0; j < k; j++) out[i + j] += v[i] / k;
    v = out;
  }
  return v;
}

/** Variance of n passes of a box of width k: n (k² − 1) / 12. */
export function repeatedBoxVariance(k: number, n: number): number {
  return n * (k * k - 1) / 12;
}

// ---------------------------------------------------------------------------------------
// Rectangle (Haar-like) features

/** A rectangle of a template in cell units, with a weight (e.g. +1 / −1). */
export interface WRect { x: number; y: number; w: number; h: number; weight: number }

/** A feature template: weighted rectangles on a grid of `cols` × `rows` cells. */
export interface Template { id: string; label: string; cols: number; rows: number; rects: WRect[] }

/** The standard templates (cells: the feature of size (cols·sx) × (rows·sy) pixels). */
export const TEMPLATES: Template[] = [
  { id: "edge-x", label: "2-rect, left − right", cols: 2, rows: 1, rects: [{ x: 0, y: 0, w: 1, h: 1, weight: 1 }, { x: 1, y: 0, w: 1, h: 1, weight: -1 }] },
  { id: "edge-y", label: "2-rect, top − bottom", cols: 1, rows: 2, rects: [{ x: 0, y: 0, w: 1, h: 1, weight: 1 }, { x: 0, y: 1, w: 1, h: 1, weight: -1 }] },
  { id: "line-x", label: "3-rect, 2·centre − sides", cols: 3, rows: 1, rects: [{ x: 0, y: 0, w: 1, h: 1, weight: -1 }, { x: 1, y: 0, w: 1, h: 1, weight: 2 }, { x: 2, y: 0, w: 1, h: 1, weight: -1 }] },
  { id: "line-y", label: "3-rect, 2·middle − top − bottom", cols: 1, rows: 3, rects: [{ x: 0, y: 0, w: 1, h: 1, weight: -1 }, { x: 0, y: 1, w: 1, h: 1, weight: 2 }, { x: 0, y: 2, w: 1, h: 1, weight: -1 }] },
  { id: "checker", label: "4-rect, diagonal", cols: 2, rows: 2, rects: [{ x: 0, y: 0, w: 1, h: 1, weight: 1 }, { x: 1, y: 0, w: 1, h: 1, weight: -1 }, { x: 0, y: 1, w: 1, h: 1, weight: -1 }, { x: 1, y: 1, w: 1, h: 1, weight: 1 }] },
  { id: "centre", label: "centre − surround", cols: 3, rows: 3, rects: [{ x: 1, y: 1, w: 1, h: 1, weight: 1 }, { x: 0, y: 0, w: 3, h: 3, weight: -1 / 9 }] },
];

export function templateById(id: string): Template {
  return TEMPLATES.find(t => t.id === id) ?? TEMPLATES[0];
}

/** A table lookup with its coefficient: response = Σ c · S[y][x]. */
export interface Corner { x: number; y: number; c: number }

/**
 * The distinct table corners of a template placed at (x, y) with cell size sx × sy, with
 * their merged coefficients (shared corners read once; corners whose coefficients cancel
 * are dropped). The number of lookups is derived from this list, not hard-coded.
 */
export function templateCorners(t: Template, x: number, y: number, sx: number, sy: number): Corner[] {
  const m = new Map<string, Corner>();
  const add = (cx: number, cy: number, c: number) => {
    const key = cx + "," + cy;
    const e = m.get(key);
    if (e) e.c += c; else m.set(key, { x: cx, y: cy, c });
  };
  for (const r of t.rects) {
    const x0 = x + r.x * sx, y0 = y + r.y * sy, x1 = x0 + r.w * sx, y1 = y0 + r.h * sy;
    add(x1, y1, r.weight); add(x1, y0, -r.weight); add(x0, y1, -r.weight); add(x0, y0, r.weight);
  }
  return [...m.values()].filter(c => Math.abs(c.c) > 1e-12);
}

/** Evaluate a list of corners on a table (f64 or f32 table). */
export function evalCorners(T: Table, corners: Corner[], dx = 0, dy = 0): number {
  const W = T.w + 1;
  let s = 0;
  for (const c of corners) s += c.c * T.data[(c.y + dy) * W + c.x + dx];
  return s;
}

/** The same feature by summing the pixels directly (reference). */
export function templateDirect(img: ArrayLike<number>, w: number, t: Template, x: number, y: number, sx: number, sy: number): number {
  let s = 0;
  for (const r of t.rects)
    s += r.weight * directSum(img, w, x + r.x * sx, y + r.y * sy, x + (r.x + r.w) * sx, y + (r.y + r.h) * sy);
  return s;
}

/** Number of pixels a direct evaluation reads (sum of the rectangle areas). */
export function templatePixels(t: Template, sx: number, sy: number): number {
  return t.rects.reduce((a, r) => a + r.w * sx * r.h * sy, 0);
}

/**
 * Response map of a template with cell size sx × sy at every position: out[y·w + x] is the
 * feature with its centre at (x, y) (top-left = centre − half the size), NaN where it does
 * not fit. Optionally divided by `norm` (e.g. the area of one rectangle, giving a mean
 * difference).
 */
export function responseMap(T: Table, t: Template, sx: number, sy: number, norm = 1): Float64Array {
  const { w, h } = T, fw = t.cols * sx, fh = t.rows * sy;
  const out = new Float64Array(w * h).fill(NaN);
  const corners = templateCorners(t, 0, 0, sx, sy);
  const hx = Math.floor(fw / 2), hy = Math.floor(fh / 2);
  for (let y = 0; y + fh <= h; y++)
    for (let x = 0; x + fw <= w; x++)
      out[(y + hy) * w + x + hx] = evalCorners(T, corners, x, y) / norm;
  return out;
}

/** Number of placements of a template with a × b cells (cell sizes ≥ 1) in a W × H window. */
export function poolSize(a: number, b: number, W: number, H: number): number {
  let n = 0;
  for (let cw = 1; a * cw <= W; cw++)
    for (let ch = 1; b * ch <= H; ch++) n += (W - a * cw + 1) * (H - b * ch + 1);
  return n;
}

// ---------------------------------------------------------------------------------------
// The box group of a pixel feature stack (recalled from the random forests day; this
// day adds the box features). Every feature costs a few lookups whatever its scale.

export type BoxFeature = "mean" | "sd" | "dob" | "haar-x" | "haar-y";
export const BOX_FEATURES: { id: BoxFeature; label: string }[] = [
  { id: "mean", label: "box mean" },
  { id: "sd", label: "box SD" },
  { id: "dob", label: "centre − surround" },
  { id: "haar-x", label: "edge x" },
  { id: "haar-y", label: "edge y" },
];

/**
 * One box feature at scale s (pixels) for every pixel, with clipped windows:
 *  mean: mean over the (2s+1)² box; sd: its standard deviation;
 *  dob: mean over the (2s+1)² box minus the mean over the (4s+1)² box (≈ DoG, blob detector);
 *  haar-x / haar-y: mean of the right (lower) half minus the left (upper) half of the
 *  (2s) × (2s+1) box: a box-smoothed derivative.
 */
export function boxFeature(img: ArrayLike<number>, w: number, h: number, f: BoxFeature, s: number,
  T1?: Table, T2?: Table): Float64Array {
  const S1 = T1 ?? integralImage(img, w, h);
  const out = new Float64Array(w * h);
  const mean = (x0: number, y0: number, x1: number, y1: number) => { const q = rectSumClipped(S1, x0, y0, x1, y1); return q.n ? q.sum / q.n : 0; };
  if (f === "sd") {
    const S2 = T2 ?? integralImage(img, w, h, "f64", v => v * v);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const a = rectSumClipped(S1, x - s, y - s, x + s + 1, y + s + 1), b = rectSumClipped(S2, x - s, y - s, x + s + 1, y + s + 1);
        const m = a.sum / a.n;
        out[y * w + x] = Math.sqrt(Math.max(0, b.sum / b.n - m * m));
      }
    return out;
  }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v: number;
      if (f === "mean") v = mean(x - s, y - s, x + s + 1, y + s + 1);
      else if (f === "dob") v = mean(x - s, y - s, x + s + 1, y + s + 1) - mean(x - 2 * s, y - 2 * s, x + 2 * s + 1, y + 2 * s + 1);
      else if (f === "haar-x") v = mean(x + 1, y - s, x + s + 1, y + s + 1) - mean(x - s + 1, y - s, x + 1, y + s + 1);
      else v = mean(x - s, y + 1, x + s + 1, y + s + 1) - mean(x - s, y - s + 1, x + s + 1, y + 1);
      out[y * w + x] = v;
    }
  return out;
}
