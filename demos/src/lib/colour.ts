// Colour for the course demos: standard conversions and the course's colour maps.
// Owner: colour day (plan/colour-day.md); every demo takes its maps from here so that the
// whole course shows the same, validated maps.
//
// Conventions
//  - RGB triples are numbers in [0, 1]. "sRGB" means display-encoded (what a canvas stores,
//    divided by 255); "linear" means linear light. Convert with srgbToLinear/linearToSrgb.
//  - XYZ is relative to the D65 white with Y = 1 for white; Lab uses the same white.
//  - HSV: hue in degrees [0, 360), NaN when the hue is undefined (greys); s, v in [0, 1].
//  - A ColorMap maps t in [0, 1] (clamped) to an sRGB triple in [0, 1]. For per-pixel
//    work use a lookup table (colormapLUT / scalarToImageData), not the function.
// No DOM at import time (the canvas helpers only touch the DOM when called).

export type RGB = [number, number, number];
export type ColorMap = (t: number) => RGB;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

// ---------------------------------------------------------------------------------------
// sRGB transfer function

/** sRGB decoding (display value in [0,1] -> linear light), IEC 61966-2-1. */
export function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

/** sRGB encoding (linear light -> display value). */
export function linearToSrgb(v: number): number {
  return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
}

/** Decoding table for 8-bit values: SRGB8_TO_LINEAR[byte] = linear light in [0, 1]. */
export const SRGB8_TO_LINEAR: Float32Array = Float32Array.from({ length: 256 }, (_, i) => srgbToLinear(i / 255));

/** Encode linear light in [0,1] to an 8-bit sRGB value (rounded, clamped). */
export function linearToSrgb8(v: number): number {
  return Math.round(255 * linearToSrgb(clamp01(v)));
}

// ---------------------------------------------------------------------------------------
// Linear RGB <-> XYZ (sRGB primaries, D65), XYZ <-> Lab

/** D65 reference white (X, Y, Z) with Y = 1. */
export const D65: RGB = [0.95047, 1.0, 1.08883];

/** Linear sRGB -> XYZ (rows), sRGB primaries and D65 white. */
export const M_RGB_TO_XYZ: number[][] = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
];
/** XYZ -> linear sRGB (rows), the inverse of M_RGB_TO_XYZ. */
export const M_XYZ_TO_RGB: number[][] = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.9692660, 1.8760108, 0.0415560],
  [0.0556434, -0.2040259, 1.0572252],
];

const mat3v = (M: number[][], v: RGB): RGB => [
  M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
  M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
  M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
];

export const linearRgbToXyz = (rgb: RGB): RGB => mat3v(M_RGB_TO_XYZ, rgb);
export const xyzToLinearRgb = (xyz: RGB): RGB => mat3v(M_XYZ_TO_RGB, xyz);

/** Relative luminance Y of a linear RGB triple. */
export const luminance = (rgb: RGB): number => M_RGB_TO_XYZ[1][0] * rgb[0] + M_RGB_TO_XYZ[1][1] * rgb[1] + M_RGB_TO_XYZ[1][2] * rgb[2];

const LAB_E = 216 / 24389, LAB_K = 24389 / 27;

/** XYZ -> CIE L*a*b* (relative to `white`, default D65). L in [0, 100]. */
export function xyzToLab(xyz: RGB, white: RGB = D65): RGB {
  const f = (t: number) => (t > LAB_E ? Math.cbrt(t) : (LAB_K * t + 16) / 116);
  const fx = f(xyz[0] / white[0]), fy = f(xyz[1] / white[1]), fz = f(xyz[2] / white[2]);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** CIE L*a*b* -> XYZ (relative to `white`, default D65). */
export function labToXyz(lab: RGB, white: RGB = D65): RGB {
  const fy = (lab[0] + 16) / 116, fx = fy + lab[1] / 500, fz = fy - lab[2] / 200;
  const inv = (f: number) => (f ** 3 > LAB_E ? f ** 3 : (116 * f - 16) / LAB_K);
  const yr = lab[0] > LAB_K * LAB_E ? fy ** 3 : lab[0] / LAB_K;
  return [inv(fx) * white[0], yr * white[1], inv(fz) * white[2]];
}

/** Display sRGB in [0,1] -> Lab (D65). */
export const srgbToLab = (rgb: RGB): RGB => xyzToLab(linearRgbToXyz([srgbToLinear(rgb[0]), srgbToLinear(rgb[1]), srgbToLinear(rgb[2])]));

/** Lab (D65) -> display sRGB, clamped into [0,1] (out-of-gamut colours are clipped). */
export function labToSrgb(lab: RGB): RGB {
  const l = xyzToLinearRgb(labToXyz(lab));
  return [clamp01(linearToSrgb(clamp01(l[0]))), clamp01(linearToSrgb(clamp01(l[1]))), clamp01(linearToSrgb(clamp01(l[2])))];
}

/** Is a Lab colour inside the sRGB gamut (with a small tolerance)? */
export function labInGamut(lab: RGB, tol = 1e-4): boolean {
  return xyzToLinearRgb(labToXyz(lab)).every(v => v >= -tol && v <= 1 + tol);
}

/** Colour difference ΔE*ab (CIE 1976): Euclidean distance in Lab. */
export function deltaE76(lab1: RGB, lab2: RGB): number {
  return Math.hypot(lab1[0] - lab2[0], lab1[1] - lab2[1], lab1[2] - lab2[2]);
}

/** Lab -> LCh (chroma, hue in degrees [0, 360)). */
export function labToLch(lab: RGB): RGB {
  const h = Math.atan2(lab[2], lab[1]) * 180 / Math.PI;
  return [lab[0], Math.hypot(lab[1], lab[2]), h < 0 ? h + 360 : h];
}
/** LCh -> Lab. */
export function lchToLab(lch: RGB): RGB {
  const h = lch[2] * Math.PI / 180;
  return [lch[0], lch[1] * Math.cos(h), lch[1] * Math.sin(h)];
}

// ---------------------------------------------------------------------------------------
// HSV (on whatever RGB it is given; usually display sRGB)

/** RGB -> HSV. Hue in degrees [0, 360), NaN for greys (undefined hue). */
export function rgbToHsv(rgb: RGB): RGB {
  const [r, g, b] = rgb, mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = NaN;
  if (d > 0) {
    if (mx === r) h = 60 * (((g - b) / d) % 6);
    else if (mx === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h < 0) h += 360;
  }
  return [h, mx > 0 ? d / mx : 0, mx];
}

/** HSV -> RGB. An undefined (NaN) hue is treated as 0, which is harmless when s = 0. */
export function hsvToRgb(hsv: RGB): RGB {
  const h = Number.isFinite(hsv[0]) ? ((hsv[0] % 360) + 360) % 360 : 0, s = hsv[1], v = hsv[2];
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  const k = Math.floor(h / 60);
  const [r, g, b] = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][k % 6];
  return [r + m, g + m, b + m];
}

// ---------------------------------------------------------------------------------------
// Small helpers

/** "#1f77b4" or "#abc" -> sRGB triple in [0,1]. */
export function hexToRgb(hex: string): RGB {
  let h = hex.replace("#", "").trim();
  if (h.length === 3) h = h.split("").map(c => c + c).join("");
  const n = parseInt(h, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** sRGB triple in [0,1] -> "#rrggbb". */
export function rgbToHex(rgb: RGB): string {
  return "#" + rgb.map(v => Math.round(255 * clamp01(v)).toString(16).padStart(2, "0")).join("");
}

/** sRGB triple in [0,1] -> "rgb(r,g,b)" or "rgba(...)" for CSS/SVG/canvas. */
export function toCss(rgb: RGB, alpha = 1): string {
  const [r, g, b] = rgb.map(v => Math.round(255 * clamp01(v)));
  return alpha >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
}

/** Black or white, whichever reads better on top of the sRGB colour `bg`. */
export function textColorOn(bg: RGB): string {
  const Y = luminance([srgbToLinear(bg[0]), srgbToLinear(bg[1]), srgbToLinear(bg[2])]);
  return Y > 0.33 ? "#111" : "#fff";
}

// ---------------------------------------------------------------------------------------
// Colour maps

/** A map through sRGB anchor colours (equally spaced), interpolated in Lab so lightness
 *  changes smoothly; precomputed into a 1024-entry table. */
export function mapFromAnchors(hexes: string[]): ColorMap {
  const labs = hexes.map(h => srgbToLab(hexToRgb(h)));
  const N = 1024, tab = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    const u = i / (N - 1) * (labs.length - 1), k = Math.min(labs.length - 2, Math.floor(u)), f = u - k;
    const a = labs[k], b = labs[k + 1];
    const c = labToSrgb([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])]);
    tab.set(c, 3 * i);
  }
  return (t: number) => {
    const u = clamp01(Number.isFinite(t) ? t : 0) * (N - 1), i = Math.min(N - 2, Math.floor(u)), f = u - i;
    return [0, 1, 2].map(c => tab[3 * i + c] + f * (tab[3 * i + 3 + c] - tab[3 * i + c])) as RGB;
  };
}

/** Grey ramp (display values, black to white). */
export const grey: ColorMap = t => { const v = clamp01(Number.isFinite(t) ? t : 0); return [v, v, v]; };

/** Perceptual sequential map (the viridis anchors): dark blue -> green -> yellow, lightness
 *  increases monotonically; readable in greyscale and for common colour-vision deficiencies. */
export const viridis: ColorMap = mapFromAnchors(["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#fde725"]);

/** Perceptual sequential "heat" map (the inferno anchors): black -> purple -> orange -> pale yellow. */
export const inferno: ColorMap = mapFromAnchors(["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]);

/** Diverging map blue -> white -> red; t = 0.5 is white (zero). Use `divergingT` for signed data. */
export const diverging: ColorMap = mapFromAnchors(["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]);

/** Map a signed value to t for the diverging map: 0 -> 0.5, ±maxAbs -> 0 / 1. */
export const divergingT = (v: number, maxAbs: number): number => 0.5 + 0.5 * v / (maxAbs || 1);

/** All named continuous maps (for data-colormap attributes and menus). */
export const COLORMAPS: Record<"grey" | "viridis" | "inferno" | "diverging", ColorMap> = { grey, viridis, inferno, diverging };
export type ColorMapName = keyof typeof COLORMAPS;

/** Qualitative palette for a few categories (series, classes); the first five equal the
 *  --qb-c1..--qb-c5 variables of demos/demo.css. */
export const QUALITATIVE: string[] = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];

/** Colour-blind-safe qualitative palette (Okabe & Ito), for when the classes must be told apart by everyone. */
export const OKABE_ITO: string[] = ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#000000"];

/** Category colour k (cycles through QUALITATIVE) as an sRGB triple. */
export const categoryColor = (k: number): RGB => hexToRgb(QUALITATIVE[((k % QUALITATIVE.length) + QUALITATIVE.length) % QUALITATIVE.length]);

/**
 * Label palette for label images: label 0 is the background (black, or `background`),
 * labels 1, 2, ... get hues spaced by the golden angle in LCh, with alternating
 * lightness, so neighbouring label numbers always look different. Deterministic.
 */
export function labelColor(k: number, background: RGB = [0, 0, 0]): RGB {
  if (k <= 0) return background;
  const h = (20 + (k - 1) * 137.50776) % 360;
  const L = [68, 82, 56][(k - 1) % 3], C = [62, 48, 58][(k - 1) % 3];
  // reduce chroma until inside the gamut, so hue and lightness are kept
  let lab = lchToLab([L, C, h]);
  for (let c = C; c > 0 && !labInGamut(lab); c -= 4) lab = lchToLab([L, c, h]);
  return labToSrgb(lab);
}

/** The first n label colours (index 0 = background). */
export function labelPalette(n: number, background: RGB = [0, 0, 0]): RGB[] {
  return Array.from({ length: n }, (_, k) => labelColor(k, background));
}

/** Lightness profile L*(t) of a map at n points (to check or plot perceptual uniformity). */
export function lightnessProfile(map: ColorMap, n = 64): number[] {
  return Array.from({ length: n }, (_, i) => srgbToLab(map(i / (n - 1)))[0]);
}

// ---------------------------------------------------------------------------------------
// Lookup tables and canvas helpers

/** RGBA lookup table with n entries (bytes, alpha 255) for a map. */
export function colormapLUT(map: ColorMap, n = 256): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const c = map(n === 1 ? 0 : i / (n - 1));
    lut[4 * i] = 255 * c[0]; lut[4 * i + 1] = 255 * c[1]; lut[4 * i + 2] = 255 * c[2]; lut[4 * i + 3] = 255;
  }
  return lut;
}

/** RGBA lookup table for the label palette: entry k = colour of label k (k < n). */
export function labelLUT(n = 256, background: RGB = [0, 0, 0], backgroundAlpha = 255): Uint8ClampedArray {
  const lut = colormapLUT(t => labelColor(Math.round(t * (n - 1)), background), n);
  lut[3] = backgroundAlpha;
  return lut;
}

export interface ScalarImageOptions {
  /** Colour map (default grey) or a precomputed RGBA LUT. */
  map?: ColorMap | Uint8ClampedArray;
  /** Value shown as the first / last colour (default: the data range). */
  lo?: number;
  hi?: number;
  /** RGBA used for NaN values (default transparent). */
  nan?: [number, number, number, number];
}

const lutCache = new WeakMap<ColorMap, Uint8ClampedArray>();
function lutOf(map: ColorMap | Uint8ClampedArray | undefined): Uint8ClampedArray {
  if (map instanceof Uint8ClampedArray) return map;
  const m = map ?? grey;
  let l = lutCache.get(m);
  if (!l) { l = colormapLUT(m, 256); lutCache.set(m, l); }
  return l;
}

/** Min and max of the finite values. */
export function finiteRange(v: ArrayLike<number>): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v.length; i++) { const x = v[i]; if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; } }
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** Fill an RGBA byte array (w*h*4) from scalar values through a colour map. */
export function scalarToRGBA(values: ArrayLike<number>, opts: ScalarImageOptions = {}, out?: Uint8ClampedArray): Uint8ClampedArray {
  const lut = lutOf(opts.map), n = lut.length / 4;
  let { lo, hi } = opts;
  if (lo === undefined || hi === undefined) { const r = finiteRange(values); lo ??= r[0]; hi ??= r[1]; }
  const k = hi > lo ? (n - 1) / (hi - lo) : 0, nan = opts.nan ?? [0, 0, 0, 0];
  const o = out ?? new Uint8ClampedArray(4 * values.length);
  for (let p = 0; p < values.length; p++) {
    const v = values[p];
    if (!Number.isFinite(v)) { o[4 * p] = nan[0]; o[4 * p + 1] = nan[1]; o[4 * p + 2] = nan[2]; o[4 * p + 3] = nan[3]; continue; }
    let i = Math.round((v - lo) * k);
    i = i < 0 ? 0 : i > n - 1 ? n - 1 : i;
    o[4 * p] = lut[4 * i]; o[4 * p + 1] = lut[4 * i + 1]; o[4 * p + 2] = lut[4 * i + 2]; o[4 * p + 3] = lut[4 * i + 3];
  }
  return o;
}

/** ImageData of a w x h scalar image through a colour map (needs a DOM). */
export function scalarToImageData(values: ArrayLike<number>, w: number, h: number, opts: ScalarImageOptions = {}): ImageData {
  const id = new ImageData(w, h);
  scalarToRGBA(values, opts, id.data);
  return id;
}

/** Draw a scalar image into a canvas (resizes the canvas to w x h pixels). */
export function drawScalarImage(canvas: HTMLCanvasElement, values: ArrayLike<number>, w: number, h: number, opts: ScalarImageOptions = {}): void {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.getContext("2d")!.putImageData(scalarToImageData(values, w, h, opts), 0, 0);
}

/** Draw a label image (0 = background) with the label palette. */
export function drawLabelImage(canvas: HTMLCanvasElement, labels: ArrayLike<number>, w: number, h: number,
  background: RGB = [0, 0, 0], backgroundAlpha = 255): void {
  let mx = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] > mx) mx = labels[i];
  const lut = labelLUT(mx + 1, background, backgroundAlpha);
  drawScalarImage(canvas, labels, w, h, { map: lut, lo: 0, hi: Math.max(1, mx) });
}
