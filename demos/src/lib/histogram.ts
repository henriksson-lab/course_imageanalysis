// Histograms (no DOM; drawing goes through lib/plot.ts). Works for 8-bit, 16-bit, signed
// integer and float data alike.
//
// A Histogram has `bins` equal bins of width `binWidth` starting at `lo`:
// bin i covers [lo + i·binWidth, lo + (i+1)·binWidth), the last bin also includes its upper
// edge. `intHistogram` makes one bin per integer value v, centred on v (lo = vmin − 0.5),
// so bin i holds the value vmin + i — negative values and 16-bit data included.

export interface Histogram {
  counts: Float64Array;     // (weighted) count per bin
  lo: number;               // lower edge of bin 0
  binWidth: number;
  total: number;            // sum of counts (values outside the range are not counted)
  below: number;            // number of values < lo (not in counts)
  above: number;            // number of values > upper edge (not in counts)
}

export interface HistOptions {
  bins?: number;            // default 256
  min?: number;             // range; default the data minimum
  max?: number;             // default the data maximum
  weights?: ArrayLike<number>;
  mask?: ArrayLike<number>; // only pixels with mask[i] ≠ 0
}

/** Minimum and maximum of the (masked) values, ignoring NaN. */
export function range(a: ArrayLike<number>, mask?: ArrayLike<number>): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const v = a[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

/** Histogram of real values with equal bins over [min, max]. */
export function histogram(a: ArrayLike<number>, opts: HistOptions = {}): Histogram {
  const bins = opts.bins ?? 256;
  let { min, max } = opts.min !== undefined && opts.max !== undefined ? { min: opts.min, max: opts.max } : range(a, opts.mask);
  if (opts.min !== undefined) min = opts.min;
  if (opts.max !== undefined) max = opts.max;
  if (!Number.isFinite(min)) { min = 0; max = 1; }
  if (max <= min) max = min + 1;
  const h: Histogram = { counts: new Float64Array(bins), lo: min, binWidth: (max - min) / bins, total: 0, below: 0, above: 0 };
  for (let i = 0; i < a.length; i++) {
    if (opts.mask && !opts.mask[i]) continue;
    const v = a[i], wt = opts.weights ? opts.weights[i] : 1;
    if (v < min) { h.below += wt; continue; }
    if (v > max) { h.above += wt; continue; }
    if (v !== v) continue;   // NaN
    const b = Math.min(bins - 1, Math.floor((v - min) / h.binWidth));
    h.counts[b] += wt;
    h.total += wt;
  }
  return h;
}

/**
 * One bin per integer value from vmin to vmax (default: the data range, values rounded).
 * Fine for 16-bit data (65 536 bins) and negative values.
 */
export function intHistogram(a: ArrayLike<number>, vmin?: number, vmax?: number, mask?: ArrayLike<number>): Histogram {
  if (vmin === undefined || vmax === undefined) {
    const r = range(a, mask);
    vmin = vmin ?? Math.round(r.min); vmax = vmax ?? Math.round(r.max);
    if (!Number.isFinite(vmin)) { vmin = 0; vmax = 0; }
  }
  const bins = vmax - vmin + 1;
  const h: Histogram = { counts: new Float64Array(bins), lo: vmin - 0.5, binWidth: 1, total: 0, below: 0, above: 0 };
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const v = Math.round(a[i]);
    if (v < vmin) { h.below++; continue; }
    if (v > vmax) { h.above++; continue; }
    h.counts[v - vmin]++;
    h.total++;
  }
  return h;
}

/** Bin index of value v (may be < 0 or ≥ bins when outside the range). */
export function binOf(h: Histogram, v: number): number {
  const b = Math.floor((v - h.lo) / h.binWidth);
  return b === h.counts.length && v === h.lo + h.counts.length * h.binWidth ? b - 1 : b;
}

/** Centre value of bin i. */
export const binCentre = (h: Histogram, i: number): number => h.lo + (i + 0.5) * h.binWidth;

/** Bin centres as an array. */
export function binCentres(h: Histogram): Float64Array {
  return Float64Array.from(h.counts, (_, i) => binCentre(h, i));
}

/** p_i = count_i / total (sums to 1): the normalised histogram, an estimate of the probability per bin. */
export function normalise(h: Histogram): Float64Array {
  const p = new Float64Array(h.counts.length), t = h.total || 1;
  for (let i = 0; i < p.length; i++) p[i] = h.counts[i] / t;
  return p;
}

/** Probability density per bin: p_i / binWidth (integrates to 1, comparable to a pdf curve). */
export function density(h: Histogram): Float64Array {
  const p = normalise(h);
  for (let i = 0; i < p.length; i++) p[i] /= h.binWidth;
  return p;
}

/** Cumulative histogram: c_i = Σ_{j ≤ i} count_j (or the fraction, with `normalised`). */
export function cumulative(h: Histogram, normalised = true): Float64Array {
  const c = new Float64Array(h.counts.length), t = normalised ? h.total || 1 : 1;
  let s = 0;
  for (let i = 0; i < c.length; i++) { s += h.counts[i]; c[i] = s / t; }
  return c;
}

/** Mean and variance of the binned values (bin centres weighted by counts). */
export function histStats(h: Histogram): { mean: number; variance: number; sd: number } {
  let s = 0, s2 = 0;
  for (let i = 0; i < h.counts.length; i++) { const c = binCentre(h, i); s += h.counts[i] * c; s2 += h.counts[i] * c * c; }
  const t = h.total || 1, mean = s / t, variance = Math.max(0, s2 / t - mean * mean);
  return { mean, variance, sd: Math.sqrt(variance) };
}

/** Value below which a fraction q (0..1) of the counts lies, interpolated linearly inside the bin. */
export function histPercentile(h: Histogram, q: number): number {
  const target = q * h.total;
  let s = 0;
  for (let i = 0; i < h.counts.length; i++) {
    const c = h.counts[i];
    if (s + c >= target && c > 0) return h.lo + (i + (target - s) / c) * h.binWidth;
    s += c;
  }
  return h.lo + h.counts.length * h.binWidth;
}

/** Exact percentile of raw data (sorts a copy; linear interpolation between order statistics). */
export function percentile(a: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(a).sort();
  if (s.length === 0) return NaN;
  const t = q * (s.length - 1), i = Math.floor(t), f = t - i;
  return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i];
}

/** Index of the largest bin (first one on ties). */
export function mode(h: Histogram): number {
  let b = 0;
  for (let i = 1; i < h.counts.length; i++) if (h.counts[i] > h.counts[b]) b = i;
  return b;
}

export interface Histogram2D {
  counts: Float64Array;   // binsY rows × binsX columns, row-major (counts[iy*binsX + ix])
  binsX: number; binsY: number;
  loX: number; widthX: number;
  loY: number; widthY: number;
  total: number;
}

/**
 * Joint histogram of two channels (e.g. hue–saturation, or two fluorescence channels).
 * `circularX` wraps x values into [minX, maxX) (for hue in degrees or angles).
 */
export function histogram2d(x: ArrayLike<number>, y: ArrayLike<number>, opts: {
  binsX?: number; binsY?: number; minX?: number; maxX?: number; minY?: number; maxY?: number;
  circularX?: boolean; mask?: ArrayLike<number>; weights?: ArrayLike<number>;
} = {}): Histogram2D {
  const bx = opts.binsX ?? 64, by = opts.binsY ?? 64;
  const rx = range(x, opts.mask), ry = range(y, opts.mask);
  const minX = opts.minX ?? rx.min, maxX = opts.maxX ?? (rx.max > rx.min ? rx.max : rx.min + 1);
  const minY = opts.minY ?? ry.min, maxY = opts.maxY ?? (ry.max > ry.min ? ry.max : ry.min + 1);
  const H: Histogram2D = { counts: new Float64Array(bx * by), binsX: bx, binsY: by, loX: minX, widthX: (maxX - minX) / bx,
    loY: minY, widthY: (maxY - minY) / by, total: 0 };
  const spanX = maxX - minX;
  for (let i = 0; i < x.length; i++) {
    if (opts.mask && !opts.mask[i]) continue;
    let u = x[i];
    const v = y[i];
    if (opts.circularX) u = minX + (((u - minX) % spanX) + spanX) % spanX;
    if (u < minX || u > maxX || v < minY || v > maxY || u !== u || v !== v) continue;
    const ix = Math.min(bx - 1, Math.floor((u - minX) / H.widthX)), iy = Math.min(by - 1, Math.floor((v - minY) / H.widthY));
    const wt = opts.weights ? opts.weights[i] : 1;
    H.counts[iy * bx + ix] += wt;
    H.total += wt;
  }
  return H;
}
