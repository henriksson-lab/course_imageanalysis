// Per-object measurements: the catalogue of the measurements day (owner: measurements day,
// unit intensity-features/; used by texture-features/, relations/, coloc/ and the random
// forest day). No DOM.
//
// Conventions (said on the catalogue slides):
//  - an object is a set of pixel indices Ω (y*w + x); pixel centres at integer (x, y), y down;
//  - standard deviation divides by n (the pixels are the whole population of the object);
//  - quartiles and median: linear interpolation between order statistics (type 7, as
//    lib/stats.ts); MAD = median |I − median| (unscaled; 1.4826·MAD estimates σ for normal data);
//  - edge band: object pixels with a 4-neighbour outside the object (object minus its erosion
//    by the 3×3 cross, outside the image counts as background);
//  - radial distribution: ρ = d_c / (d_c + d_e) with d_c the straight-line distance to the
//    centre point and d_e the exact Euclidean distance transform (to the nearest background
//    pixel, lib/distance.ts); ring k of n holds ρ in [k/n, (k+1)/n) (ρ = 1 goes to ring n−1).
//    FracAtD_k = share of the intensity in ring k; MeanFrac_k = FracAtD_k / (share of the
//    pixels in ring k); RadialCV_k = CV (sd/mean, sd over n) of the mean ring intensity over
//    `sectors` angular sectors around the centre (empty sectors skipped).
//  - local background: median of a ring around the object, gap < distance ≤ gap + width,
//    excluding pixels of any object.
//  - robust z: (x − median(controls)) / (1.4826 · MAD(controls)).
//
// Builds on lib/distance.ts (edt), lib/morph.ts (erosion), lib/stats.ts (quantile, mad).

import { edt } from "./distance";
import { erodeBinary, seDiamond } from "./morph";
import { quantile, median, mad } from "./stats";

type A = ArrayLike<number>;

/** Pixel indices of each label 1..count (index 0 unused). */
export function objectPixels(labels: A, count: number): Int32Array[] {
  const n = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) { const l = labels[i]; if (l > 0 && l <= count) n[l]++; }
  const out = Array.from({ length: count + 1 }, (_, l) => new Int32Array(n[l]));
  const k = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) { const l = labels[i]; if (l > 0 && l <= count) out[l][k[l]++] = i; }
  return out;
}

/** Binary mask of a set of pixel indices. */
export function maskFromPixels(px: A, w: number, h: number): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let i = 0; i < px.length; i++) m[px[i]] = 1;
  return m;
}

/** Indices of the non-zero pixels of a mask. */
export function pixelsOf(mask: A): Int32Array {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  const out = new Int32Array(n);
  for (let i = 0, k = 0; i < mask.length; i++) if (mask[i]) out[k++] = i;
  return out;
}

// ---------------------------------------------------------------------------------------
// Statistics of the intensity values

export interface IntensityStats {
  n: number;
  integrated: number;
  mean: number;
  sd: number;
  min: number;
  max: number;
  q1: number;
  median: number;
  q3: number;
  mad: number;
}

/** Statistics of a list of intensity values (sd over n). */
export function intensityStats(values: A): IntensityStats {
  const n = values.length;
  if (!n) return { n: 0, integrated: 0, mean: NaN, sd: NaN, min: NaN, max: NaN, q1: NaN, median: NaN, q3: NaN, mad: NaN };
  let s = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) { const v = values[i]; s += v; if (v < mn) mn = v; if (v > mx) mx = v; }
  const m = s / n;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (values[i] - m) ** 2;
  return {
    n, integrated: s, mean: m, sd: Math.sqrt(ss / n), min: mn, max: mx,
    q1: quantile(values, 0.25), median: median(values), q3: quantile(values, 0.75), mad: mad(values),
  };
}

/** The intensity values of an image at a set of pixel indices. */
export function valuesAt(img: A, px: A): Float64Array {
  return Float64Array.from({ length: px.length }, (_, k) => img[px[k]]);
}

/** Edge band: mask pixels with a 4-neighbour outside the mask (= mask minus its erosion by the cross). */
export function edgeBand(mask: A, w: number, h: number): Uint8Array {
  const er = erodeBinary(mask, w, h, seDiamond(1), true);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] && !er[i] ? 1 : 0;
  return out;
}

// ---------------------------------------------------------------------------------------
// Centroids and mass displacement

/** Geometric centroid (mean x, mean y) of a set of pixels. */
export function centroidOf(px: A, w: number): [number, number] {
  let sx = 0, sy = 0;
  for (let k = 0; k < px.length; k++) { const i = px[k]; sx += i % w; sy += Math.floor(i / w); }
  return [sx / px.length, sy / px.length];
}

/** Intensity-weighted centroid Σ I x / Σ I. */
export function weightedCentroid(img: A, px: A, w: number): [number, number] {
  let s = 0, sx = 0, sy = 0;
  for (let k = 0; k < px.length; k++) {
    const i = px[k], v = img[i];
    s += v; sx += v * (i % w); sy += v * Math.floor(i / w);
  }
  return [sx / s, sy / s];
}

/** Distance between the geometric and the intensity-weighted centroid (pixels × spacing). */
export function massDisplacement(img: A, px: A, w: number, spacing = 1): number {
  const [gx, gy] = centroidOf(px, w), [cx, cy] = weightedCentroid(img, px, w);
  return Math.hypot(cx - gx, cy - gy) * spacing;
}

// ---------------------------------------------------------------------------------------
// Radial distribution

export interface RadialResult {
  /** Ring index per pixel of the mask (−1 outside). */
  ring: Int8Array;
  /** Normalised radial position ρ per pixel (NaN outside). */
  rho: Float32Array;
  fracAtD: Float64Array;
  /** Share of the object's pixels in each ring. */
  areaFrac: Float64Array;
  meanFrac: Float64Array;
  radialCV: Float64Array;
}

/**
 * Radial distribution of the intensity in `nBins` rings by normalised distance from the
 * centre (cx, cy): ρ = d_c / (d_c + d_e). See the header for the definitions.
 */
export function radialDistribution(img: A, mask: A, w: number, h: number, nBins: number,
  centre: [number, number], sectors = 8): RadialResult {
  const de = edt(mask, w, h);
  const ring = new Int8Array(w * h).fill(-1), rho = new Float32Array(w * h).fill(NaN);
  const I = new Float64Array(nBins), N = new Float64Array(nBins);
  const secSum = new Float64Array(nBins * sectors), secN = new Float64Array(nBins * sectors);
  let tot = 0, totN = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) continue;
    const dc = Math.hypot(x - centre[0], y - centre[1]);
    const r = dc / (dc + de[i]);
    const k = Math.min(nBins - 1, Math.floor(r * nBins));
    ring[i] = k; rho[i] = r;
    I[k] += img[i]; N[k]++; tot += img[i]; totN++;
    let a = Math.atan2(y - centre[1], x - centre[0]);
    if (a < 0) a += 2 * Math.PI;
    const s = Math.min(sectors - 1, Math.floor(a / (2 * Math.PI) * sectors));
    secSum[k * sectors + s] += img[i]; secN[k * sectors + s]++;
  }
  const fracAtD = new Float64Array(nBins), areaFrac = new Float64Array(nBins), meanFrac = new Float64Array(nBins), radialCV = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    fracAtD[k] = tot ? I[k] / tot : NaN;
    areaFrac[k] = totN ? N[k] / totN : NaN;
    meanFrac[k] = N[k] ? fracAtD[k] / areaFrac[k] : NaN;
    const ms: number[] = [];
    for (let s = 0; s < sectors; s++) if (secN[k * sectors + s]) ms.push(secSum[k * sectors + s] / secN[k * sectors + s]);
    if (ms.length < 2) { radialCV[k] = NaN; continue; }
    const m = ms.reduce((a, b) => a + b, 0) / ms.length;
    const sd = Math.sqrt(ms.reduce((a, b) => a + (b - m) ** 2, 0) / ms.length);
    radialCV[k] = m ? sd / m : NaN;
  }
  return { ring, rho, fracAtD, areaFrac, meanFrac, radialCV };
}

// ---------------------------------------------------------------------------------------
// Local background

/**
 * Ring around an object: pixels whose distance to the object is in (gap, gap + width],
 * and that belong to no object (`occupied`, optional, non-zero = some object).
 */
export function backgroundRing(mask: A, w: number, h: number, gap: number, width: number, occupied?: A): Uint8Array {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const d = edt(inv, w, h);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (inv[i] && d[i] > gap && d[i] <= gap + width && !(occupied && occupied[i])) out[i] = 1;
  return out;
}

// ---------------------------------------------------------------------------------------
// One row of the intensity table

export interface IntensityRow extends IntensityStats {
  edgeMean: number;
  edgeIntegrated: number;
  /** edge mean / mean */
  edgeRatio: number;
  cx: number; cy: number;
  wx: number; wy: number;
  massDisplacement: number;
}

/** All intensity columns of one object (mask) in one image. */
export function intensityRow(img: A, mask: A, w: number, h: number, spacing = 1): IntensityRow {
  const px = pixelsOf(mask);
  const st = intensityStats(valuesAt(img, px));
  const ev = valuesAt(img, pixelsOf(edgeBand(mask, w, h)));
  let es = 0;
  for (const v of ev) es += v;
  const em = ev.length ? es / ev.length : NaN;
  const [cx, cy] = centroidOf(px, w), [wx, wy] = weightedCentroid(img, px, w);
  return { ...st, edgeMean: em, edgeIntegrated: es, edgeRatio: em / st.mean, cx, cy, wx, wy, massDisplacement: Math.hypot(wx - cx, wy - cy) * spacing };
}

/** The intensity row of every label 1..count. */
export function intensityTable(img: A, labels: A, w: number, h: number, count: number, spacing = 1): IntensityRow[] {
  const pix = objectPixels(labels, count);
  const rows: IntensityRow[] = [];
  for (let l = 1; l <= count; l++) rows.push(intensityRow(img, maskFromPixels(pix[l], w, h), w, h, spacing));
  return rows;
}

// ---------------------------------------------------------------------------------------
// Units, aggregation and normalisation

export type Dim = "px" | "px2" | "intensity" | "none";

/** Convert a value from pixel units to physical units (pixel size `s`, e.g. µm per pixel). */
export function toPhysical(v: number, dim: Dim, s: number): number {
  return dim === "px" ? v * s : dim === "px2" ? v * s * s : v;
}

/** Aggregate values per group (e.g. per well): "mean", "median" or "count". */
export function aggregate(values: A, groups: A, nGroups: number, how: "mean" | "median" | "count" = "median"): Float64Array {
  const lists: number[][] = Array.from({ length: nGroups }, () => []);
  for (let i = 0; i < values.length; i++) lists[groups[i]]?.push(values[i]);
  return Float64Array.from(lists, l => (how === "count" ? l.length : !l.length ? NaN : how === "mean" ? l.reduce((a, b) => a + b, 0) / l.length : median(l)));
}

/** z-score against controls: (x − mean(c)) / sd(c), sd over n − 1. */
export function zScore(values: A, controls: A): Float64Array {
  const n = controls.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += controls[i];
  m /= n;
  let s = 0;
  for (let i = 0; i < n; i++) s += (controls[i] - m) ** 2;
  const sd = Math.sqrt(s / (n - 1));
  return Float64Array.from(values, v => (v - m) / sd);
}

/** Robust z-score against controls: (x − median(c)) / (1.4826 · MAD(c)). */
export function robustZ(values: A, controls: A): Float64Array {
  const m = median(controls), s = 1.4826 * mad(controls);
  return Float64Array.from(values, v => (v - m) / s);
}
