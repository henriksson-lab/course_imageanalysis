// Colour conversions the colour day needs beyond lib/colour.ts (which this folder must not
// edit): YCbCr, CIEDE2000, chromatic adaptation (von Kries in LMS, Bradford), dichromat
// simulation, and a few extra colour maps (jet, rainbow, cividis, magma) for the colour-map
// unit. No DOM. Used by the colour day's demos (colourcloud, colourseg, colourmatrix,
// whitebalance, colourmap); candidates to move into lib/colour.ts.
import { RGB, M_RGB_TO_XYZ, M_XYZ_TO_RGB, mapFromAnchors, ColorMap } from "../lib/colour";
import { M_XYZ_TO_LMS, inv3, mat3mul, mat3v } from "../lib/spectra";

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

// ---------------------------------------------------------------------------------------
// YCbCr (full range, as in JPEG/JFIF), on encoded values 0..255

/** Luma weights (BT.601) on encoded R'G'B'. */
export const LUMA601: RGB = [0.299, 0.587, 0.114];
/** Luminance weights on linear RGB (sRGB/BT.709 primaries). */
export const LUMINANCE709: RGB = [0.2126, 0.7152, 0.0722];

/** Encoded R'G'B' (0..255) -> Y'CbCr (0..255, chroma centred on 128). */
export function rgbToYcbcr(r: number, g: number, b: number): RGB {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return [y, 128 + 0.564 * (b - y), 128 + 0.713 * (r - y)];
}

/** Y'CbCr -> encoded R'G'B' (0..255, not clamped). */
export function ycbcrToRgb(y: number, cb: number, cr: number): RGB {
  const r = y + (cr - 128) / 0.713, b = y + (cb - 128) / 0.564;
  const g = (y - 0.299 * r - 0.114 * b) / 0.587;
  return [r, g, b];
}

// ---------------------------------------------------------------------------------------
// CIEDE2000 (Sharma, Wu & Dalal 2005)

export function deltaE2000(lab1: RGB, lab2: RGB): number {
  const [L1, a1, b1] = lab1, [L2, a2, b2] = lab2;
  const rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2), Cm = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hp = (a: number, b: number) => { if (a === 0 && b === 0) return 0; const h = Math.atan2(b, a) / rad; return h < 0 ? h + 360 : h; };
  const h1p = hp(a1p, b1), h2p = hp(a2p, b2);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) { dhp = h2p - h1p; if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360; }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dhp * rad / 2);
  const Lpm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2;
  let hpm = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) <= 180) hpm = (h1p + h2p) / 2;
    else hpm = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2;
  }
  const T = 1 - 0.17 * Math.cos((hpm - 30) * rad) + 0.24 * Math.cos(2 * hpm * rad) + 0.32 * Math.cos((3 * hpm + 6) * rad) - 0.2 * Math.cos((4 * hpm - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hpm - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7));
  const Sl = 1 + 0.015 * (Lpm - 50) ** 2 / Math.sqrt(20 + (Lpm - 50) ** 2);
  const Sc = 1 + 0.045 * Cpm, Sh = 1 + 0.015 * Cpm * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

// ---------------------------------------------------------------------------------------
// Chromatic adaptation

/** Bradford "sharpened cone" matrix XYZ -> RGB_B (rows). */
export const M_BRADFORD: number[][] = [
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
];

const RGB2XYZ = M_RGB_TO_XYZ, XYZ2RGB = M_XYZ_TO_RGB;

/** Linear-RGB 3 x 3 matrix that maps the white `src` (linear RGB) to `dst` by a diagonal
 *  scaling in the basis `basis` (rows: XYZ -> cone-like space): "rgb" scales the camera
 *  channels directly, "lms" uses Hunt–Pointer–Estévez cones (von Kries), "bradford" the
 *  sharpened Bradford basis. */
export function adaptationMatrix(src: RGB, dst: RGB, basis: "rgb" | "lms" | "bradford"): number[][] {
  if (basis === "rgb") return [[dst[0] / src[0], 0, 0], [0, dst[1] / src[1], 0], [0, 0, dst[2] / src[2]]];
  const B = basis === "lms" ? M_XYZ_TO_LMS : M_BRADFORD;
  const T = mat3mul(B, RGB2XYZ), Ti = mat3mul(XYZ2RGB, inv3(B));
  const s = mat3v(T, src), d = mat3v(T, dst);
  const D = [[d[0] / s[0], 0, 0], [0, d[1] / s[1], 0], [0, 0, d[2] / s[2]]];
  return mat3mul(Ti, mat3mul(D, T));
}

// ---------------------------------------------------------------------------------------
// Dichromat simulation (Viénot, Brettel & Mollon 1999 style): in LMS the missing cone's
// response is replaced by a combination of the other two, chosen so that two anchor
// colours (white, and blue for protan/deutan, red for tritan) look unchanged. In linear
// RGB this is one 3 x 3 matrix of rank 2: a projection onto a plane.

export type Dichromacy = "protan" | "deutan" | "tritan";

function dichromatMatrix(kind: Dichromacy): number[][] {
  const T = mat3mul(M_XYZ_TO_LMS, RGB2XYZ), Ti = inv3(T);
  const w = mat3v(T, [1, 1, 1]), anchor = mat3v(T, kind === "tritan" ? [1, 0, 0] : [0, 0, 1]);
  const miss = kind === "protan" ? 0 : kind === "deutan" ? 1 : 2;
  const [i, j] = [0, 1, 2].filter(k => k !== miss);
  // solve a*w[i] + b*w[j] = w[miss], a*anchor[i] + b*anchor[j] = anchor[miss]
  const det = w[i] * anchor[j] - w[j] * anchor[i];
  const a = (w[miss] * anchor[j] - w[j] * anchor[miss]) / det;
  const b = (w[i] * anchor[miss] - w[miss] * anchor[i]) / det;
  const D = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  D[miss] = [0, 0, 0]; D[miss][i] = a; D[miss][j] = b;
  return mat3mul(Ti, mat3mul(D, T));
}

/** Linear-RGB simulation matrices (rank 2). */
export const DICHROMAT: Record<Dichromacy, number[][]> = {
  protan: dichromatMatrix("protan"),
  deutan: dichromatMatrix("deutan"),
  tritan: dichromatMatrix("tritan"),
};

const dec = (v: number) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const enc = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

/** Simulate a dichromat on a display sRGB triple (0..1): decode, apply the matrix, encode. */
export function simulateDichromat(rgb: RGB, kind: Dichromacy): RGB {
  const l = mat3v(DICHROMAT[kind], [dec(rgb[0]), dec(rgb[1]), dec(rgb[2])]);
  return [enc(clamp01(l[0])), enc(clamp01(l[1])), enc(clamp01(l[2]))];
}

/** 256-entry tables: encoded byte -> linear, for fast per-pixel simulation. */
export function simulateDichromatBytes(data: Uint8ClampedArray, kind: Dichromacy): void {
  const M = DICHROMAT[kind], L = new Float32Array(256);
  for (let i = 0; i < 256; i++) L[i] = dec(i / 255);
  for (let p = 0; p < data.length; p += 4) {
    const r = L[data[p]], g = L[data[p + 1]], b = L[data[p + 2]];
    data[p] = 255 * enc(clamp01(M[0][0] * r + M[0][1] * g + M[0][2] * b));
    data[p + 1] = 255 * enc(clamp01(M[1][0] * r + M[1][1] * g + M[1][2] * b));
    data[p + 2] = 255 * enc(clamp01(M[2][0] * r + M[2][1] * g + M[2][2] * b));
  }
}

// ---------------------------------------------------------------------------------------
// Extra colour maps (for the colour-map unit: good and bad examples)

/** The classic "jet" rainbow (blue -> cyan -> yellow -> red): non-monotone lightness. */
export const jet: ColorMap = t => {
  const u = clamp01(Number.isFinite(t) ? t : 0);
  return [clamp01(1.5 - Math.abs(4 * u - 3)), clamp01(1.5 - Math.abs(4 * u - 2)), clamp01(1.5 - Math.abs(4 * u - 1))];
};

/** A hue-circle rainbow (red -> yellow -> green -> cyan -> blue -> magenta). */
export const rainbow: ColorMap = t => {
  const h = 300 * clamp01(Number.isFinite(t) ? t : 0) / 60, x = 1 - Math.abs((h % 2) - 1);
  const k = Math.min(4, Math.floor(h));
  return ([[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1]][k]) as RGB;
};

/** cividis anchors: blue -> grey -> yellow, designed to look the same to deuteranopes. */
export const cividis: ColorMap = mapFromAnchors(["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"]);

/** magma anchors: black -> purple -> orange -> pale yellow. */
export const magma: ColorMap = mapFromAnchors(["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"]);
