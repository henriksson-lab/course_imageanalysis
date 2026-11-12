// White balance on a spectrally rendered chart (no DOM). 24 patches with seeded smooth
// reflectance spectra are lit by an illuminant and seen by an observer (the CIE eye -> linear
// sRGB, or a synthetic camera), so the ground truth (the same chart under D65) is known.
// Estimates of the illuminant's white: grey world, white patch (max-RGB), a clicked grey
// card; corrections: a diagonal matrix on encoded values, on linear RGB, in cone space (von
// Kries) or the Bradford basis, or a full 3 x 3 least-squares fit to the chart.
import * as SP from "../lib/spectra";
import { srgbToLinear, linearToSrgb, linearRgbToXyz, xyzToLab, RGB } from "../lib/colour";
import { rng } from "../lib/random";
import { adaptationMatrix } from "../colourcloud/colourx";

export type ChartKind = "chart" | "forest";
export type Illum = "d65" | "blackbody" | "fluorescent" | "rgbled" | "whiteled";
export type WbObserver = "eye" | "camera";
export type Estimate = "none" | "greyworld" | "whitepatch" | "card" | "lstsq";
export type Space = "encoded" | "rgb" | "lms" | "bradford";

export const COLS = 6, ROWS = 4, NP = COLS * ROWS;
const lobe = (l: number, c: number, w: number) => Math.exp(-0.5 * ((l - c) / w) ** 2);

/** 24 reflectance spectra. "chart": varied colours + a bottom row of neutral greys.
 *  "forest": greens, browns and a sky (no neutral row, except one white flower). */
export function reflectances(kind: ChartKind, seed = 21): Float64Array[] {
  const r = rng(seed), out: Float64Array[] = [];
  for (let k = 0; k < NP; k++) {
    const s = new Float64Array(SP.NB);
    if (kind === "chart") {
      if (k >= 18) { s.fill([0.9, 0.59, 0.36, 0.19, 0.09, 0.031][k - 18]); out.push(s); continue; }
      const base = 0.04 + 0.1 * r.unif(), c = 410 + 300 * (k / 18) + 25 * (r.unif() - 0.5), w = 25 + 45 * r.unif(), a = 0.35 + 0.35 * r.unif();
      const c2 = 400 + 320 * r.unif(), a2 = 0.25 * r.unif();
      for (let i = 0; i < SP.NB; i++) s[i] = base + a * lobe(SP.WL[i], c, w) + a2 * lobe(SP.WL[i], c2, 40) + (c > 640 ? a * 0.8 * (SP.WL[i] > c ? 1 - lobe(SP.WL[i], c, w) : 0) : 0);
    } else {
      const type = k === 23 ? "white" : k < 2 ? "sky" : k % 6 === 0 ? "bark" : "leaf";
      for (let i = 0; i < SP.NB; i++) {
        const l = SP.WL[i];
        if (type === "white") s[i] = 0.85;
        else if (type === "sky") s[i] = 0.25 + 0.35 * lobe(l, 450, 60) * (0.8 + 0.4 * r.unif());
        else if (type === "bark") s[i] = 0.05 + 0.25 * (l - 380) / 350 * (0.7 + 0.3 * r.unif());
        else s[i] = 0.03 + (0.12 + 0.15 * r.unif()) * lobe(l, 545, 32) + (0.35 + 0.2 * r.unif()) / (1 + Math.exp(-(l - 705) / 12));
      }
    }
    for (let i = 0; i < SP.NB; i++) s[i] = Math.min(0.95, Math.max(0.02, s[i]));
    out.push(s);
  }
  return out;
}

export function illuminant(id: Illum, T = 2800): Float64Array {
  return id === "d65" ? SP.D65_SPD : id === "blackbody" ? SP.blackbody(T) : id === "fluorescent" ? SP.fluorescent() : id === "rgbled" ? SP.rgbLED() : SP.metamericLED();
}

/** Linear RGB of a light spectrum for an observer (before normalisation). */
function observe(light: ArrayLike<number>, obs: WbObserver): RGB {
  if (obs === "eye") return SP.xyzToLinearSrgbCal(SP.spectrumToXyz(light));
  const v = SP.apply(CAM, light), w = SP.apply(CAM, SP.D65_SPD);
  return [v[0] / w[0], v[1] / w[1], v[2] / w[2]];
}
const CAM = SP.sensitivity("camera");

const lumOf = (c: RGB) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/** Linear RGB of each patch under illuminant E, exposed so that a perfect white reflector
 *  just reaches 1 in its largest channel (an exposure that keeps the colour cast and does
 *  not clip the chart). */
export function renderPatches(refl: Float64Array[], E: ArrayLike<number>, obs: WbObserver): RGB[] {
  const white = observe(E, obs), k = 1 / Math.max(...white);
  return refl.map(r => { const c = observe(Float64Array.from(r, (v, i) => v * E[i]), obs); return [c[0] * k, c[1] * k, c[2] * k] as RGB; });
}

/** The pixels of the chart image: patch p fills a block; an optional specular highlight on
 *  patch `hiPatch` (light 3 x the white, clipped at 1 in each channel). Returns per-pixel
 *  patch index too (-1 = gap). */
export interface ChartImage { w: number; h: number; r: Float32Array; g: Float32Array; b: Float32Array; patch: Int16Array }
export const CELL = 22, GAP = 3;
export function chartImage(patches: RGB[], highlight: boolean, whiteLin: RGB, hiPatch = 8): ChartImage {
  const w = COLS * (CELL + GAP) + GAP, h = ROWS * (CELL + GAP) + GAP, n = w * h;
  const img: ChartImage = { w, h, r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n), patch: new Int16Array(n).fill(-1) };
  for (let k = 0; k < NP; k++) {
    const cx = k % COLS, cy = Math.floor(k / COLS), x0 = GAP + cx * (CELL + GAP), y0 = GAP + cy * (CELL + GAP);
    for (let y = y0; y < y0 + CELL; y++) for (let x = x0; x < x0 + CELL; x++) {
      const p = y * w + x;
      let c = patches[k];
      if (highlight && k === hiPatch && (x - x0 - CELL / 2) ** 2 + (y - y0 - CELL / 2) ** 2 < 30) c = [Math.min(1, 3 * whiteLin[0]), Math.min(1, 3 * whiteLin[1]), Math.min(1, 3 * whiteLin[2])];
      img.r[p] = Math.min(1, c[0]); img.g[p] = Math.min(1, c[1]); img.b[p] = Math.min(1, c[2]); img.patch[p] = k;
    }
  }
  return img;
}

/** Estimate of the illuminant's white (linear RGB, up to scale) from the image. */
export function estimateWhite(img: ChartImage, how: Estimate, card: number): RGB {
  const n = img.w * img.h;
  if (how === "greyworld") {
    let r = 0, g = 0, b = 0, m = 0;
    for (let p = 0; p < n; p++) if (img.patch[p] >= 0) { r += img.r[p]; g += img.g[p]; b += img.b[p]; m++; }
    return [r / m, g / m, b / m];
  }
  if (how === "whitepatch") {
    let r = 0, g = 0, b = 0;
    for (let p = 0; p < n; p++) { r = Math.max(r, img.r[p]); g = Math.max(g, img.g[p]); b = Math.max(b, img.b[p]); }
    return [r, g, b];
  }
  if (how === "card") {
    let r = 0, g = 0, b = 0, m = 0;
    for (let p = 0; p < n; p++) if (img.patch[p] === card) { r += img.r[p]; g += img.g[p]; b += img.b[p]; m++; }
    return [r / m, g / m, b / m];
  }
  return [1, 1, 1];
}

/** 3 x 3 least-squares fit M minimising Σ |M c_i - t_i|² over the patches. */
export function fitMatrix(src: RGB[], dst: RGB[]): number[][] {
  // normal equations: M = (Σ t cᵀ)(Σ c cᵀ)⁻¹
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], T = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < src.length; i++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { C[a][b] += src[i][a] * src[i][b]; T[a][b] += dst[i][a] * src[i][b]; }
  return SP.mat3mul(T, SP.inv3(C));
}

/** Apply a correction to linear RGB colours. */
export function correct(cols: RGB[], how: Estimate, space: Space, white: RGB, fit?: number[][]): RGB[] {
  if (how === "none") return cols.map(c => [...c] as RGB);
  if (how === "lstsq" && fit) return cols.map(c => SP.mat3v(fit, c));
  if (space === "encoded") {
    const g = white.map(v => linearToSrgb(1) / linearToSrgb(Math.max(1e-6, v)));
    return cols.map(c => c.map((v, k) => srgbToLinear(Math.max(0, linearToSrgb(Math.max(0, v)) * g[k]))) as RGB);
  }
  const M = adaptationMatrix(white, [1, 1, 1], space);
  return cols.map(c => SP.mat3v(M, c));
}

/** Scale colours so that their mean luminance equals `target` (shared exposure). */
export function expose(cols: RGB[], target: number): RGB[] {
  const m = cols.reduce((s, c) => s + lumOf(c), 0) / cols.length;
  const k = m > 0 ? target / m : 1;
  return cols.map(c => [c[0] * k, c[1] * k, c[2] * k] as RGB);
}

export const labOf = (c: RGB) => xyzToLab(linearRgbToXyz([Math.max(0, c[0]), Math.max(0, c[1]), Math.max(0, c[2])]));

/** ΔE*ab per patch between two sets of linear RGB colours. */
export function deltaEs(a: RGB[], b: RGB[]): number[] {
  return a.map((c, i) => { const x = labOf(c), y = labOf(b[i]); return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); });
}

export interface WbResult { captured: RGB[]; corrected: RGB[]; reference: RGB[]; white: RGB; dE: number[]; img: ChartImage }

/** The whole pipeline for one setting. */
export function runWB(o: { kind: ChartKind; illum: Illum; T: number; obs: WbObserver; how: Estimate; space: Space; card: number; highlight: boolean }): WbResult {
  const refl = reflectances(o.kind), E = illuminant(o.illum, o.T);
  const reference = renderPatches(refl, SP.D65_SPD, o.obs);
  const raw = renderPatches(refl, E, o.obs);
  const whiteLin = (() => { const w = observe(E, o.obs), k = 1 / Math.max(...w); return [w[0] * k, w[1] * k, w[2] * k] as RGB; })();
  const clipped = raw.map(c => c.map(v => Math.min(1, v)) as RGB);
  const img = chartImage(clipped, o.highlight, whiteLin);
  const white = estimateWhite(img, o.how, o.card);
  const fit = o.how === "lstsq" ? fitMatrix(clipped, reference) : undefined;
  const target = reference.reduce((s, c) => s + lumOf(c), 0) / reference.length;
  const corrected = expose(correct(clipped, o.how, o.space, white, fit), target);
  const captured = expose(clipped, target);
  return { captured, corrected, reference, white, dE: deltaEs(corrected, reference), img };
}
