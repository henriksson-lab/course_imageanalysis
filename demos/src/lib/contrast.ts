// Point operations and histogram-based contrast transforms (owner: the contrast unit,
// contrast/). No DOM. Used by the demos tonecurve, equalise, clahe and displayvsdata and by
// tools/figs/contrast/data.ts. Tests: demos/test/contrast-test.ts.
//
// Conventions
//  - An integer image has L possible values 0..L-1 (256 for 8 bits, 4096 for 12 bits).
//  - A LUT (lookup table) is a Float64Array of length L: lut[v] = output for input v.
//    Display LUTs have outputs in 0..255 (not rounded; round when showing).
//  - A curve is a function of one raw value v (a number) returning the display value.
//  - Equalisation and matching work on histogram counts h[0..L-1]; the mapping is the
//    cumulative histogram: s_k = round((Lout - 1) * c_k / N), c_k = sum_{j <= k} h_j.
//  - CLAHE: tiles with their own clipped histograms; each pixel blends the curves of the
//    four nearest tile centres (bilinear weights). The clip limit is beta * (mean count per
//    bin), so beta is the largest contrast gain relative to a straight line.
//
// Also: seeded synthetic scenes for these demos (the 12-bit fluorescence image of the
// window16 demo, a low-contrast "foggy" scene, a faint cell on a flat noisy background, and
// a control/treated pair with known masks). They live here, not in lib/synth.ts, until the
// owner of synth.ts moves them.
import { rng } from "./ui";
import { poisson } from "./random";
import { Grey, createGrey } from "./imageops";

export type Lut = Float64Array;
export type Curve = (v: number) => number;

// ---------------------------------------------------------------------------------------
// LUTs

/** Tabulate a curve for the L input values 0..L-1. */
export function lutFromFunction(f: Curve, L: number): Lut {
  const lut = new Float64Array(L);
  for (let v = 0; v < L; v++) lut[v] = f(v);
  return lut;
}

/** Apply a LUT: out[i] = lut[round(data[i]) - vmin] (indices clamped to the table). */
export function applyLut(data: ArrayLike<number>, lut: ArrayLike<number>, vmin = 0, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(data.length), n = lut.length - 1;
  for (let i = 0; i < data.length; i++) {
    let k = Math.round(data[i]) - vmin;
    if (k < 0) k = 0; else if (k > n) k = n;
    o[i] = lut[k];
  }
  return o;
}

/** The same for a Grey image. */
export function applyLutGrey(img: Grey, lut: ArrayLike<number>, vmin = 0): Grey {
  return { width: img.width, height: img.height, data: applyLut(img.data, lut, vmin) };
}

/** Two LUTs in a row are one LUT: (b ∘ a)[v] = b[round(a[v])] (a's outputs index b). */
export function composeLut(a: ArrayLike<number>, b: ArrayLike<number>): Lut {
  const out = new Float64Array(a.length), n = b.length - 1;
  for (let v = 0; v < a.length; v++) out[v] = b[Math.max(0, Math.min(n, Math.round(a[v])))];
  return out;
}

/** True if the LUT never falls (strict: always rises). */
export function isMonotone(lut: ArrayLike<number>, strict = false): boolean {
  for (let v = 1; v < lut.length; v++) if (strict ? lut[v] <= lut[v - 1] : lut[v] < lut[v - 1]) return false;
  return true;
}

/** Slope of the LUT at v (central difference; one-sided at the ends). */
export function slopeOf(lut: ArrayLike<number>, v: number): number {
  const n = lut.length - 1, k = Math.max(0, Math.min(n, Math.round(v)));
  if (n === 0) return 0;
  if (k === 0) return lut[1] - lut[0];
  if (k === n) return lut[n] - lut[n - 1];
  return (lut[k + 1] - lut[k - 1]) / 2;
}

/** Number of distinct values (after rounding) in the data. */
export function distinctCount(data: ArrayLike<number>): number {
  const s = new Set<number>();
  for (let i = 0; i < data.length; i++) s.add(Math.round(data[i]));
  return s.size;
}

// ---------------------------------------------------------------------------------------
// Curves (raw value v -> display value in 0..out, default out = 255)

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);

/** The linear window of highbit/: lo -> 0, hi -> out, clipped (not rounded). */
export function windowCurve(lo: number, hi: number, out = 255): Curve {
  return v => out * clamp01((v - lo) / (hi - lo || 1));
}

/** Gamma on the window [lo, hi]: out * t^γ with t = (v - lo)/(hi - lo) clipped. */
export function gammaCurve(gamma: number, lo: number, hi: number, out = 255): Curve {
  return v => out * Math.pow(clamp01((v - lo) / (hi - lo || 1)), gamma);
}

/** Log: c * ln(1 + (v - lo)); c = "fit" scales hi to out. Values below lo give 0. */
export function logCurve(c: number | "fit", lo: number, hi: number, out = 255): Curve {
  const k = c === "fit" ? out / Math.log(1 + Math.max(1e-9, hi - lo)) : c;
  return v => Math.min(out, k * Math.log(1 + Math.max(0, v - lo)));
}

/** S-curve: a logistic centred at `centre` with width `width` (raw units), rescaled so
 *  that the ends of [lo, hi] map exactly to 0 and out. */
export function sigmoidCurve(centre: number, width: number, lo: number, hi: number, out = 255): Curve {
  const s = (v: number) => 1 / (1 + Math.exp(-(v - centre) / Math.max(1e-9, width)));
  const a = s(lo), b = s(hi);
  return v => out * clamp01((s(v) - a) / (b - a || 1));
}

/** Piecewise-linear curve through points (x sorted); flat outside. `monotone` makes the
 *  y values non-decreasing (each y at least the previous one). */
export function piecewiseLinear(points: [number, number][], monotone = false): Curve {
  const pts = points.map(p => [p[0], p[1]] as [number, number]).sort((a, b) => a[0] - b[0]);
  if (monotone) for (let i = 1; i < pts.length; i++) pts[i][1] = Math.max(pts[i][1], pts[i - 1][1]);
  return v => {
    if (v <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      if (v <= pts[i][0]) {
        const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
        return x1 > x0 ? y0 + (y1 - y0) * (v - x0) / (x1 - x0) : y1;
      }
    }
    return pts[pts.length - 1][1];
  };
}

/** Negative: v -> L - 1 - v, rescaled to the display range. */
export function negativeCurve(L: number, out = 255): Curve {
  return v => out * (L - 1 - v) / (L - 1);
}

/** A threshold: 0 below t, out from t on (the steepest point operation). */
export function thresholdCurve(t: number, out = 255): Curve {
  return v => (v >= t ? out : 0);
}

// ---------------------------------------------------------------------------------------
// Equalisation and matching

/** Integer histogram with L bins of the values 0..L-1 (rounded, clamped). */
export function levelHistogram(data: ArrayLike<number>, L: number, mask?: ArrayLike<number>): Float64Array {
  const h = new Float64Array(L);
  for (let i = 0; i < data.length; i++) {
    if (mask && !mask[i]) continue;
    const k = Math.round(data[i]);
    h[k < 0 ? 0 : k >= L ? L - 1 : k]++;
  }
  return h;
}

/** Running sums c_k = Σ_{j≤k} h_j. */
export function cumulativeCounts(h: ArrayLike<number>): Float64Array {
  const c = new Float64Array(h.length);
  let s = 0;
  for (let k = 0; k < h.length; k++) { s += h[k]; c[k] = s; }
  return c;
}

/**
 * Equalisation LUT from histogram counts h (input levels 0..h.length-1) to Lout output
 * levels: s_k = round((Lout - 1) · c_k / N) ("plain"), or with the darkest occurring value
 * mapped to 0: round((Lout - 1)(c_k - c_min)/(N - c_min)) ("cmin"). `round = false` gives
 * the unrounded curve (for drawing).
 */
export function equaliseLut(h: ArrayLike<number>, Lout: number, variant: "plain" | "cmin" = "plain", round = true): Lut {
  const c = cumulativeCounts(h), N = c[c.length - 1] || 1;
  let cmin = 0;
  if (variant === "cmin") for (let k = 0; k < h.length; k++) if (h[k] > 0) { cmin = c[k]; break; }
  const lut = new Float64Array(h.length), den = Math.max(1e-12, N - cmin);
  for (let k = 0; k < h.length; k++) {
    const s = (Lout - 1) * Math.max(0, c[k] - cmin) / den;
    lut[k] = round ? Math.round(s) : s;
  }
  return lut;
}

/**
 * Equalisation of arbitrary (e.g. float) values: the mapping uses only the order of the
 * distinct values, so any strictly rising transform applied first gives the same output.
 * out[i] = round((Lout - 1) · (number of pixels ≤ data[i]) / N).
 */
export function equaliseValues(data: ArrayLike<number>, Lout: number): Float32Array {
  const n = data.length, idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => data[a] - data[b]);
  const out = new Float32Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && data[idx[j + 1]] === data[idx[i]]) j++;
    const s = Math.round((Lout - 1) * (j + 1) / n);
    for (let k = i; k <= j; k++) out[idx[k]] = s;
    i = j + 1;
  }
  return out;
}

/**
 * Exact equalisation: order all pixels by value, break ties (at random with a seed, or by
 * the 3 × 3 local mean, then at random), and give level floor(rank · Lout / N). Every level
 * gets N/Lout pixels (exactly when Lout divides N), at the price of an invented order.
 */
export function equaliseExact(data: ArrayLike<number>, Lout: number, tieBreak: "random" | "localMean" = "random",
  seed = 1, w = 0, h = 0): Float32Array {
  const n = data.length, r = rng(seed);
  const key2 = new Float64Array(n), key3 = new Float64Array(n);
  for (let i = 0; i < n; i++) key3[i] = r.unif();
  if (tieBreak === "localMean" && w > 0 && h > 0) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0, m = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) { s += data[yy * w + xx]; m++; }
      }
      key2[y * w + x] = s / m;
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => (data[a] - data[b]) || (key2[a] - key2[b]) || (key3[a] - key3[b]));
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) out[idx[k]] = Math.floor(k * Lout / n);
  return out;
}

/**
 * Histogram matching: T = G⁻¹ ∘ F with the generalised inverse G⁻¹(u) = min{v' : G(v') ≥ u}.
 * src and target are counts (any totals); returns T[k] = target level for source level k.
 */
export function matchLut(src: ArrayLike<number>, target: ArrayLike<number>): Lut {
  const F = cumulativeCounts(src), G = cumulativeCounts(target);
  const NF = F[F.length - 1] || 1, NG = G[G.length - 1] || 1;
  const lut = new Float64Array(src.length);
  let j = 0;
  for (let k = 0; k < src.length; k++) {
    const u = F[k] / NF;
    while (j < G.length - 1 && G[j] / NG < u - 1e-12) j++;
    lut[k] = j;
  }
  return lut;
}

/** A flat target histogram with L levels and N pixels in all. */
export function flatTarget(L: number, N = 1): Float64Array {
  return new Float64Array(L).fill(N / L);
}

/** A Gaussian-shaped target (mean, sd in levels), scaled to N pixels in all. */
export function gaussianTarget(L: number, mean: number, sd: number, N = 1): Float64Array {
  const t = Float64Array.from({ length: L }, (_, k) => Math.exp(-0.5 * ((k - mean) / sd) ** 2));
  const s = t.reduce((a, b) => a + b, 0);
  return t.map(v => v * N / s);
}

// ---------------------------------------------------------------------------------------
// CLAHE

/**
 * Clip every bin at `limit` and spread the clipped excess evenly over all bins. With
 * `iterate`, repeat until no bin exceeds the limit (so the slope bound is exact).
 * Returns the clipped histogram and the excess removed in the first pass.
 */
export function clipHistogram(h: ArrayLike<number>, limit: number, iterate = true): { clipped: Float64Array; excess: number } {
  const B = h.length, c = Float64Array.from(h);
  let first = -1;
  for (let it = 0; it < 200; it++) {
    let ex = 0;
    for (let k = 0; k < B; k++) if (c[k] > limit) { ex += c[k] - limit; c[k] = limit; }
    if (first < 0) first = ex;
    if (ex <= 1e-9 * (limit * B)) break;
    const add = ex / B;
    for (let k = 0; k < B; k++) c[k] += add;
    if (!iterate) break;
  }
  return { clipped: c, excess: first };
}

export interface ClaheOptions {
  tilesX: number;
  tilesY: number;
  /** β: clip at β × (mean count per bin); Infinity = no clipping (plain adaptive equalisation). */
  clip: number;
  /** Number of histogram bins over `range` (default 256). */
  bins?: number;
  /** Value range [lo, hi] binned (default: the image min and max). */
  range?: [number, number];
  /** Blend the four nearest tile curves (default true); false: each tile its own curve (seams). */
  interpolate?: boolean;
  /** Repeat the redistribution until no bin exceeds the limit (default true). */
  iterate?: boolean;
  /** Output range 0..out (default 255). */
  out?: number;
}

export interface ClaheResult {
  out: Grey;
  /** One curve per tile (row-major over tiles), bin index -> output value. */
  tileLuts: Float64Array[];
  /** Clipped histograms per tile. */
  tileHists: Float64Array[];
  /** Raw histograms per tile. */
  rawHists: Float64Array[];
  /** Tile centres in pixel coordinates. */
  centres: [number, number][];
  /** Bin index of each pixel. */
  binOfPixel: Uint16Array;
  bins: number;
  range: [number, number];
  /** Pixels per tile. */
  tileN: number[];
}

/** Bin index (0..bins-1) of value v over [lo, hi]. */
export function binIndex(v: number, lo: number, hi: number, bins: number): number {
  const b = Math.floor((v - lo) / (hi - lo || 1) * bins);
  return b < 0 ? 0 : b >= bins ? bins - 1 : b;
}

/** The (up to) four tile centres around pixel (x, y) and their bilinear weights. */
export function claheWeights(x: number, y: number, W: number, H: number, tilesX: number, tilesY: number): { tile: number; w: number }[] {
  const tw = W / tilesX, th = H / tilesY;
  // position in "tile-centre" coordinates: centre of tile i is at (i + 0.5) * tw
  let gx = (x + 0.5) / tw - 0.5, gy = (y + 0.5) / th - 0.5;
  gx = Math.max(0, Math.min(tilesX - 1, gx)); gy = Math.max(0, Math.min(tilesY - 1, gy));
  const x0 = Math.min(tilesX - 1, Math.floor(gx)), y0 = Math.min(tilesY - 1, Math.floor(gy));
  const x1 = Math.min(tilesX - 1, x0 + 1), y1 = Math.min(tilesY - 1, y0 + 1);
  const fx = gx - x0, fy = gy - y0;
  const out: { tile: number; w: number }[] = [];
  const add = (tx: number, ty: number, w: number) => {
    const t = ty * tilesX + tx, e = out.find(o => o.tile === t);
    if (e) e.w += w; else out.push({ tile: t, w });
  };
  add(x0, y0, (1 - fx) * (1 - fy)); add(x1, y0, fx * (1 - fy));
  add(x0, y1, (1 - fx) * fy); add(x1, y1, fx * fy);
  return out.filter(o => o.w > 0 || out.length === 1);
}

/** Tile index of pixel (x, y). */
export function tileOf(x: number, y: number, W: number, H: number, tilesX: number, tilesY: number): number {
  const tx = Math.min(tilesX - 1, Math.floor(x * tilesX / W)), ty = Math.min(tilesY - 1, Math.floor(y * tilesY / H));
  return ty * tilesX + tx;
}

/** Contrast-limited adaptive histogram equalisation (with tilesX = tilesY = 1 and clip = ∞:
 *  global equalisation with `bins` levels). */
export function clahe(img: Grey, o: ClaheOptions): ClaheResult {
  const W = img.width, H = img.height, bins = o.bins ?? 256, outMax = o.out ?? 255;
  let lo: number, hi: number;
  if (o.range) [lo, hi] = o.range;
  else { lo = Infinity; hi = -Infinity; for (const v of img.data) { if (v < lo) lo = v; if (v > hi) hi = v; } if (hi <= lo) hi = lo + 1; }
  const nT = o.tilesX * o.tilesY;
  const binOfPixel = new Uint16Array(W * H);
  const rawHists = Array.from({ length: nT }, () => new Float64Array(bins));
  const tileN = new Array(nT).fill(0);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, b = binIndex(img.data[i], lo, hi, bins), t = tileOf(x, y, W, H, o.tilesX, o.tilesY);
    binOfPixel[i] = b; rawHists[t][b]++; tileN[t]++;
  }
  const tileHists: Float64Array[] = [], tileLuts: Float64Array[] = [];
  for (let t = 0; t < nT; t++) {
    const N = tileN[t] || 1;
    const hc = Number.isFinite(o.clip) ? clipHistogram(rawHists[t], Math.max(1, o.clip) * N / bins, o.iterate ?? true).clipped : Float64Array.from(rawHists[t]);
    tileHists.push(hc);
    const lut = new Float64Array(bins);
    let s = 0;
    for (let b = 0; b < bins; b++) { s += hc[b]; lut[b] = outMax * s / N; }
    tileLuts.push(lut);
  }
  const centres: [number, number][] = [];
  for (let ty = 0; ty < o.tilesY; ty++) for (let tx = 0; tx < o.tilesX; tx++)
    centres.push([(tx + 0.5) * W / o.tilesX - 0.5, (ty + 0.5) * H / o.tilesY - 0.5]);
  const out = createGrey(W, H);
  const interp = o.interpolate ?? true;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, b = binOfPixel[i];
    if (!interp) { out.data[i] = tileLuts[tileOf(x, y, W, H, o.tilesX, o.tilesY)][b]; continue; }
    let v = 0;
    for (const { tile, w } of claheWeights(x, y, W, H, o.tilesX, o.tilesY)) v += w * tileLuts[tile][b];
    out.data[i] = v;
  }
  return { out, tileLuts, tileHists, rawHists, centres, binOfPixel, bins, range: [lo, hi], tileN };
}

/** Largest contrast gain of a tile curve relative to a straight line: max_b h̃(b)·B/N. */
export function maxGain(clipped: ArrayLike<number>, N: number): number {
  let m = 0;
  for (let b = 0; b < clipped.length; b++) m = Math.max(m, clipped[b]);
  return m * clipped.length / (N || 1);
}

// ---------------------------------------------------------------------------------------
// Measurements

/** Mean of the values inside a mask (all pixels without a mask). */
export function maskedMean(data: ArrayLike<number>, mask?: ArrayLike<number>): number {
  let s = 0, n = 0;
  for (let i = 0; i < data.length; i++) if (!mask || mask[i]) { s += data[i]; n++; }
  return n ? s / n : NaN;
}

/** Median of the values inside a mask. */
export function maskedMedian(data: ArrayLike<number>, mask?: ArrayLike<number>): number {
  const a: number[] = [];
  for (let i = 0; i < data.length; i++) if (!mask || mask[i]) a.push(data[i]);
  if (!a.length) return NaN;
  a.sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Standard deviation of the values inside a mask. */
export function maskedSd(data: ArrayLike<number>, mask?: ArrayLike<number>): number {
  const m = maskedMean(data, mask);
  let s = 0, n = 0;
  for (let i = 0; i < data.length; i++) if (!mask || mask[i]) { s += (data[i] - m) ** 2; n++; }
  return n > 1 ? Math.sqrt(s / (n - 1)) : 0;
}

// ---------------------------------------------------------------------------------------
// Scenes (seeded)

export const SCENE_N = 256;

/** The 12-bit fluorescence image of the window16 demo (same generator and seed): offset
 *  100 ADU, background, four cells with bright nuclei, two saturated spots, shot and read
 *  noise. Values 0..4095. */
export function fluorescence12(seed = 3): Uint16Array {
  const N = SCENE_N, r = rng(seed), out = new Uint16Array(N * N);
  const cells = [
    [70, 80, 46, 34, 0.4, 700, 2600], [175, 70, 40, 30, -0.3, 500, 1900],
    [120, 175, 55, 38, 0.1, 900, 3300], [210, 190, 30, 26, 0.8, 350, 1200],
  ];
  const spots = [[132, 160], [62, 92]];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let s = 60 + 25 * Math.sin(j / 37) * Math.cos(i / 53);
    for (const [cx, cy, rx, ry, a, cyto, nuc] of cells) {
      const dx = j - cx, dy = i - cy, c = Math.cos(a), sn = Math.sin(a);
      const u = (c * dx + sn * dy) / rx, v = (-sn * dx + c * dy) / ry;
      const d2 = u * u + v * v;
      s += cyto / (1 + Math.exp((Math.sqrt(d2) - 1) * 12));
      s += (nuc - cyto) * Math.exp(-d2 / (2 * 0.16));
    }
    for (const [x, y] of spots) s += 12000 * Math.exp(-((j - x) ** 2 + (i - y) ** 2) / 4);
    const v = 100 + s + Math.sqrt(s) * r.norm() + 3 * r.norm();
    out[i * N + j] = Math.max(0, Math.min(4095, Math.round(v)));
  }
  return out;
}

/** A low-contrast 8-bit scene ("fog"): shapes with small contrast on a smooth gradient
 *  that spans much more than their contrast, plus noise. Values 0..255 (integers). */
export function foggyScene(seed = 5): Uint8Array {
  const N = SCENE_N, r = rng(seed), out = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    // haze: brighter towards the top right; the scene itself is dim and flat
    let v = 120 + 50 * (x / N) - 45 * (y / N);
    const disc = (cx: number, cy: number, rad: number) => Math.hypot(x - cx, y - cy) < rad;
    if (disc(60, 70, 26)) v += 14;
    if (disc(180, 60, 18)) v -= 12;
    if (disc(90, 185, 34)) v -= 10;
    if (disc(200, 180, 22)) v += 9;
    if (x > 120 && x < 150 && y > 100 && y < 230) v += 7;          // a faint bar
    if (Math.abs(y - 128 - 0.35 * (x - 128)) < 2.5 && x < 110) v -= 12; // a thin line
    v += 2.5 * r.norm();
    out[y * N + x] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return out;
}

/** A flat noisy background with one bright and one faint cell (8-bit). The case where
 *  equalisation blows up the background noise. */
export function faintCellScene(seed = 6): Uint8Array {
  const N = SCENE_N, r = rng(seed), out = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 30;
    const d1 = Math.hypot((x - 80) / 34, (y - 90) / 28), d2 = Math.hypot((x - 180) / 30, (y - 170) / 36);
    v += 150 / (1 + Math.exp((d1 - 1) * 10)) + 60 * Math.exp(-d1 * d1 / 0.18);
    v += 14 / (1 + Math.exp((d2 - 1) * 10)) + 10 * Math.exp(-d2 * d2 / 0.18);
    v += 3 * r.norm();
    out[y * N + x] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return out;
}

export interface ControlTreated {
  w: number;
  h: number;
  control: Uint16Array;
  treated: Uint16Array;
  /** Nucleus mask, shared by both images (1 = nucleus). */
  nuclei: Uint8Array;
  /** Cell index per pixel (0 = none, 1..n). */
  cellOf: Uint8Array;
  /** Background region far from all cells (1 = background reference). */
  background: Uint8Array;
  /** Signal above background per cell in the control (ADU). */
  signal: number[];
  offset: number;
  factor: number;
}

/** A synthetic control/treated pair (12-bit): the same nuclei, signal above the offset of
 *  `offset` ADU; in the treated image every nucleus is `factor` times brighter above the
 *  offset. Poisson (shot) noise, seeded. Sharp-edged nuclei, so masks are well defined. */
export function controlTreated(seed = 7, factor = 1.5, w = 192, h = 192): ControlTreated {
  const offset = 100;
  const cells: [number, number, number, number, number, number][] = [
    // cx, cy, rx, ry, angle, control signal above offset
    [44, 46, 20, 15, 0.3, 900], [128, 38, 17, 14, -0.4, 650], [150, 112, 22, 17, 0.6, 1300],
    [60, 132, 18, 21, 0.1, 600], [112, 164, 15, 12, -0.2, 800], [170, 172, 12, 11, 0.0, 1000],
  ];
  const nuclei = new Uint8Array(w * h), cellOf = new Uint8Array(w * h), background = new Uint8Array(w * h);
  const sig = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let near = false;
    cells.forEach(([cx, cy, rx, ry, a, s], k) => {
      const dx = x - cx, dy = y - cy, c = Math.cos(a), sn = Math.sin(a);
      const u = (c * dx + sn * dy) / rx, v = (-sn * dx + c * dy) / ry, d = Math.hypot(u, v);
      if (d <= 1) { nuclei[i] = 1; cellOf[i] = k + 1; sig[i] = s; }
      if (d < 1.6) near = true;
    });
    if (!near) background[i] = 1;
  }
  const r1 = rng(seed), r2 = rng(seed + 1000);
  const control = new Uint16Array(w * h), treated = new Uint16Array(w * h);
  for (let i = 0; i < w * h; i++) {
    control[i] = Math.min(4095, poisson(r1, offset + sig[i]));
    treated[i] = Math.min(4095, poisson(r2, offset + factor * sig[i]));
  }
  return { w, h, control, treated, nuclei, cellOf, background, signal: cells.map(c => c[5]), offset, factor };
}
