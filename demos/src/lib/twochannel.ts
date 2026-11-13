// Two-channel fluorescence forward model (owner: colocalisation day; FRET/FLIM extend it).
// No DOM. Two kinds of scenes:
//
//  - pixelScene(): the 128 × 128 two-cell scene of the `coloc` demo (moved here from
//    coloc/model.ts `makeChannels`, bit-identical for the same options and seed). Channel A:
//    blurred blobs + diffuse cell signal; B: a fraction f of the same blobs + independent ones
//    (or large blobs containing A's, or blobs that avoid A's), + bleed-through of A, shifted;
//    both + offset + Gaussian noise. New options (all off by default): blur, structure size,
//    a smooth background shared by both channels, a sub-pixel shift in x and y.
//  - spot scenes at physical scale: spotPairScene() gives true coordinates (A spots, a share
//    of them with a B partner at a given distance, extra independent B spots), spotImage()
//    renders them with a Gaussian PSF (pixel-integrated), addNoise() adds Poisson + offset.
//
// Corrections: compensate (undo bleed-through), subtractBackground (white top-hat),
// registerShift (phase correlation + upsampled peak), shiftImage (Fourier shift).
// Display: mergeGreenMagenta → RGBA bytes.
import { gaussianBlur, fromArray } from "./imageops";
import { rng, gauss, poisson, Rng } from "./random";
import { normCdf } from "./stats";
import { whiteTopHat, seDisk } from "./morph";
import { correlate2d, upsampledPeak2d, findPeaks, fourierShift2d } from "./registration";

type A = ArrayLike<number>;

export const S = 128;

// ---------------------------------------------------------------------------------------
// The pixel scene (coloc demo)

export interface ChannelOptions {
  /** Fraction of B's structures shared with A (0..1). */
  f?: number;
  /** "mix": B = f · shared + (1 − f) · independent; "inside": A's spots sit inside B's large
   *  blobs; "exclude": B's blobs are placed away from A's (segregated). */
  layout?: "mix" | "inside" | "exclude";
  offset?: number;
  noise?: number;
  /** Diffuse signal inside the cells, both channels. */
  diffuse?: number;
  /** Fraction of A that leaks into B. */
  bleed?: number;
  /** Sideways shift of B in pixels (rounded; the original integer shift). */
  shift?: number;
  /** Sub-pixel shift of B (x, y), applied by a Fourier shift after the integer one. */
  shiftX?: number;
  shiftY?: number;
  /** Blur (PSF σ) in pixels (default 1). */
  blur?: number;
  /** Scale of the structure radii (default 1: radii 1.3–2.5 px). */
  size?: number;
  /** Amplitude of a smooth uneven background added to both channels (default 0). */
  uneven?: number;
  seed?: number;
}

export interface Channels { w: number; h: number; a: Float64Array; b: Float64Array; cells: Uint8Array }

type Blob = [number, number, number];

function blobs(r: Rng, inside: Uint8Array, n: number, rMin: number, rMax: number, centres?: Blob[], avoid?: Blob[], avoidDist = 0): { img: Float64Array; list: Blob[] } {
  const img = new Float64Array(S * S), list: Blob[] = centres ? [...centres] : [];
  let guard = 0;
  while (!centres && list.length < n && guard++ < 5000) {
    const x = 4 + (S - 8) * r.unif(), y = 4 + (S - 8) * r.unif(), rad = rMin + (rMax - rMin) * r.unif();
    if (!inside[Math.round(y) * S + Math.round(x)]) continue;
    if (avoid && avoid.some(([ax, ay]) => (ax - x) ** 2 + (ay - y) ** 2 < avoidDist * avoidDist)) continue;
    list.push([x, y, rad]);
  }
  for (const [x, y, rad] of list) {
    for (let yy = Math.max(0, Math.floor(y - 3 * rad)); yy <= Math.min(S - 1, Math.ceil(y + 3 * rad)); yy++)
      for (let xx = Math.max(0, Math.floor(x - 3 * rad)); xx <= Math.min(S - 1, Math.ceil(x + 3 * rad)); xx++)
        img[yy * S + xx] += Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * rad * rad));
  }
  return { img, list };
}

/** The two elliptical cells of the 128 × 128 scene (1 inside). */
export function cellMask(): Uint8Array {
  const m = new Uint8Array(S * S);
  const cells = [[42, 50, 34, 26, 0.5], [90, 82, 30, 36, -0.3]];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) for (const [cx, cy, a, b, t] of cells) {
    const u = (x - cx) * Math.cos(t) + (y - cy) * Math.sin(t), v = -(x - cx) * Math.sin(t) + (y - cy) * Math.cos(t);
    if ((u / a) ** 2 + (v / b) ** 2 <= 1) m[y * S + x] = 1;
  }
  return m;
}

/** A smooth background field (three broad bumps and a ramp), max ≈ 1, seeded. */
export function unevenBackground(w: number, h: number, seed = 9): Float64Array {
  const r = rng(seed), out = new Float64Array(w * h);
  const bumps = Array.from({ length: 3 }, () => [w * r.unif(), h * r.unif(), (0.25 + 0.2 * r.unif()) * w]);
  let mx = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0.3 * x / w;
    for (const [cx, cy, s] of bumps) v += Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
    out[y * w + x] = v; mx = Math.max(mx, v);
  }
  for (let i = 0; i < out.length; i++) out[i] /= mx;
  return out;
}

/**
 * The coloc demo's two channels. With the original options (f, layout mix/inside, offset,
 * noise, diffuse, bleed, integer shift, seed) the result is bit-identical to the built
 * coloc/model.ts makeChannels.
 */
export function pixelScene(o: ChannelOptions = {}): Channels {
  const r = rng(o.seed ?? 4);
  const cells = cellMask();
  const f = o.f ?? 0.5, off = o.offset ?? 10, sd = o.noise ?? 4, diff = o.diffuse ?? 0, bleed = o.bleed ?? 0, shift = Math.round(o.shift ?? 0);
  const sz = o.size ?? 1, blur = o.blur ?? 1;
  let A: Float64Array, Bs: Float64Array;
  const layout = o.layout ?? "mix";
  if (layout === "inside") {
    const big = blobs(r, cells, 7, 7, 9);
    const small: Blob[] = [];
    big.list.slice(0, 4).forEach(([x, y]) => { for (let k = 0; k < 3; k++) small.push([x + 3 * r.norm(), y + 3 * r.norm(), 1.4]); });
    A = blobs(r, cells, 0, 0, 0, small).img.map(v => 160 * v);
    Bs = big.img.map(v => 110 * Math.min(1, v));
  } else if (layout === "exclude") {
    const sa = blobs(r, cells, 40, 1.3 * sz, 2.5 * sz);
    const sb = blobs(r, cells, 40, 1.3 * sz, 2.5 * sz, undefined, sa.list, 7 * sz);
    A = sa.img.map(v => 150 * v);
    Bs = sb.img.map(v => 150 * v);
  } else {
    const shared = blobs(r, cells, 40, 1.3 * sz, 2.5 * sz);
    const indep = blobs(r, cells, 40, 1.3 * sz, 2.5 * sz);
    A = shared.img.map(v => 150 * v);
    Bs = Float64Array.from(shared.img, (v, i) => 150 * (f * v + (1 - f) * indep.img[i]));
  }
  const blurA = gaussianBlur(fromArray(S, S, A), blur).data, blurB = gaussianBlur(fromArray(S, S, Bs), blur).data;
  const a = new Float64Array(S * S);
  let b = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) a[i] = blurA[i] + diff * cells[i];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const xs = Math.min(S - 1, Math.max(0, x - shift)), j = y * S + xs;
    b[y * S + x] = blurB[j] + diff * cells[j] + bleed * a[j];
  }
  if (o.shiftX || o.shiftY) b = Float64Array.from(fourierShift2d(b, S, S, o.shiftX ?? 0, o.shiftY ?? 0));
  if (o.uneven) {
    const g = unevenBackground(S, S);
    for (let i = 0; i < S * S; i++) { a[i] += o.uneven * g[i]; b[i] += o.uneven * g[i]; }
  }
  const r2 = rng((o.seed ?? 4) + 101);
  for (let i = 0; i < S * S; i++) {
    a[i] = Math.max(0, a[i] + off + gauss(r2, 0, sd));
    b[i] = Math.max(0, b[i] + off + gauss(r2, 0, sd));
  }
  return { w: S, h: S, a, b, cells };
}

// ---------------------------------------------------------------------------------------
// Spot scenes at physical scale

/** PSF σ in nm for a Gaussian approximation of the Airy disc: 0.21 λ/NA. */
export const psfSigmaNm = (NA: number, lambdaNm: number): number => 0.21 * lambdaNm / NA;
/** FWHM of the Gaussian approximation: 2√(2 ln 2) σ ≈ 0.49 λ/NA. */
export const psfFwhmNm = (NA: number, lambdaNm: number): number => 2 * Math.sqrt(2 * Math.LN2) * psfSigmaNm(NA, lambdaNm);

export interface SpotSceneOptions {
  w: number; h: number;
  /** Region in which spots are placed (1 = allowed); default the whole image minus a margin. */
  mask?: A;
  /** Number of A spots. */
  nA: number;
  /** Share of A spots that have a B partner. */
  pairFraction: number;
  /** True distance A → partner B, in pixels (random direction). */
  distance: number;
  /** Independent B spots on top of the partners. */
  extraB: number;
  /** Margin from the image border in pixels (default 4). */
  margin?: number;
  seed?: number;
}

export interface SpotScene {
  ax: Float64Array; ay: Float64Array; bx: Float64Array; by: Float64Array;
  /** Index of A's partner in B, or −1. */
  partner: Int32Array;
}

function placeIn(r: Rng, o: SpotSceneOptions): [number, number] {
  const m = o.margin ?? 4;
  for (let k = 0; k < 10000; k++) {
    const x = m + (o.w - 1 - 2 * m) * r.unif(), y = m + (o.h - 1 - 2 * m) * r.unif();
    if (!o.mask || o.mask[Math.round(y) * o.w + Math.round(x)]) return [x, y];
  }
  return [o.w / 2, o.h / 2];
}

/** True coordinates (pixels) of A spots, their partners in B and extra B spots. Seeded. */
export function spotPairScene(o: SpotSceneOptions): SpotScene {
  const r = rng(o.seed ?? 11);
  const nA = o.nA, ax = new Float64Array(nA), ay = new Float64Array(nA), partner = new Int32Array(nA).fill(-1);
  const bxs: number[] = [], bys: number[] = [];
  for (let i = 0; i < nA; i++) {
    const [x, y] = placeIn(r, o);
    ax[i] = x; ay[i] = y;
    const paired = r.unif() < o.pairFraction, th = 2 * Math.PI * r.unif();
    if (paired) { partner[i] = bxs.length; bxs.push(x + o.distance * Math.cos(th)); bys.push(y + o.distance * Math.sin(th)); }
  }
  for (let k = 0; k < o.extraB; k++) { const [x, y] = placeIn(r, o); bxs.push(x); bys.push(y); }
  return { ax, ay, bx: Float64Array.from(bxs), by: Float64Array.from(bys), partner };
}

/** Uniformly random points inside a mask (rejection sampling), as in spotPairScene. */
export function randomPointsInMask(n: number, mask: A | undefined, w: number, h: number, r: Rng, margin = 0): { x: Float64Array; y: Float64Array } {
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) [x[i], y[i]] = placeIn(r, { w, h, mask, margin, nA: 0, pairFraction: 0, distance: 0, extraB: 0 });
  return { x, y };
}

/**
 * Expected photons per pixel of Gaussian spots (σ in pixels) at (xs, ys), each with `photons`
 * in total, integrated over the pixel area (pixel centres at integers). shift moves all spots.
 */
export function spotImage(xs: A, ys: A, w: number, h: number, sigma: number, photons: number, shiftX = 0, shiftY = 0): Float64Array {
  const out = new Float64Array(w * h), R = Math.ceil(4 * sigma + 1);
  const gx = new Float64Array(2 * R + 2), gy = new Float64Array(2 * R + 2);
  for (let k = 0; k < xs.length; k++) {
    const cx = xs[k] + shiftX, cy = ys[k] + shiftY;
    const x0 = Math.round(cx) - R, y0 = Math.round(cy) - R;
    for (let j = 0; j <= 2 * R; j++) {
      gx[j] = normCdf((x0 + j + 0.5 - cx) / sigma) - normCdf((x0 + j - 0.5 - cx) / sigma);
      gy[j] = normCdf((y0 + j + 0.5 - cy) / sigma) - normCdf((y0 + j - 0.5 - cy) / sigma);
    }
    for (let j = 0; j <= 2 * R; j++) {
      const y = y0 + j;
      if (y < 0 || y >= h) continue;
      for (let i = 0; i <= 2 * R; i++) {
        const x = x0 + i;
        if (x < 0 || x >= w) continue;
        out[y * w + x] += photons * gx[i] * gy[j];
      }
    }
  }
  return out;
}

/** Poisson noise on an expected image, plus a camera offset and Gaussian read noise. */
export function addNoise(expected: A, seed: number, offset = 0, readSd = 0): Float64Array {
  const r = rng(seed), out = new Float64Array(expected.length);
  for (let i = 0; i < expected.length; i++) out[i] = poisson(r, Math.max(0, expected[i])) + offset + (readSd ? gauss(r, 0, readSd) : 0);
  return out;
}

/** A grid of two-colour beads (centres in pixels), nx × ny, inside the image. */
export function beadPositions(w: number, h: number, nx = 5, ny = 5, seed = 3): { x: Float64Array; y: Float64Array } {
  const r = rng(seed), x: number[] = [], y: number[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    x.push((i + 0.5) * w / nx + 3 * (r.unif() - 0.5));
    y.push((j + 0.5) * h / ny + 3 * (r.unif() - 0.5));
  }
  return { x: Float64Array.from(x), y: Float64Array.from(y) };
}

// ---------------------------------------------------------------------------------------
// Corrections

/** Undo bleed-through of A into B: B − s·A (s estimated from a single-stained control). */
export function compensate(a: A, b: A, s: number): Float64Array {
  return Float64Array.from(b as ArrayLike<number>, (v, i) => v - s * a[i]);
}

/** Slope through the origin of B on A (single-stained A control: B is pure leakage). */
export function leakSlope(a: A, b: A, offsetA = 0, offsetB = 0): number {
  let ab = 0, aa = 0;
  for (let i = 0; i < a.length; i++) { const u = a[i] - offsetA, v = b[i] - offsetB; ab += u * v; aa += u * u; }
  return aa > 0 ? ab / aa : 0;
}

/** Background subtraction by a white top-hat with a disk of radius r (morphology day). */
export function subtractBackground(img: A, w: number, h: number, r: number): Float64Array {
  return Float64Array.from(whiteTopHat(img, w, h, seDisk(r)));
}

/**
 * Shift of b relative to a (b(x) ≈ a(x − t)) by cross-correlation of the two w × h images
 * (powers of two; p = 0, which is more robust to shot noise on isolated beads than phase
 * correlation, p = 1) with an upsampled (1/U px) peak. Returns t = (tx, ty).
 */
export function registerShift(a: A, b: A, w: number, h: number, U = 20, p = 0): { tx: number; ty: number } {
  const c = correlate2d(b, a, w, h, { p });
  const pk = findPeaks(c, 1, 1)[0];
  const up = upsampledPeak2d(c.R, w, h, pk.tx, pk.ty, U);
  return { tx: up.tx, ty: up.ty };
}

/** out(x) = img(x − (sx, sy)), circular, band-limited (powers of two). */
export function shiftImage(img: A, w: number, h: number, sx: number, sy: number): Float64Array {
  return Float64Array.from(fourierShift2d(img, w, h, sx, sy));
}

// ---------------------------------------------------------------------------------------
// Display

/**
 * Green/magenta merge as RGBA bytes: A → green, B → red + blue, each scaled from its range
 * [lo, hi] to 0..255 (white where both are bright).
 */
export function mergeGreenMagenta(a: A, b: A, rangeA: [number, number], rangeB: [number, number], out?: Uint8ClampedArray): Uint8ClampedArray {
  const n = a.length, o = out ?? new Uint8ClampedArray(4 * n);
  const sa = 255 / Math.max(1e-9, rangeA[1] - rangeA[0]), sb = 255 / Math.max(1e-9, rangeB[1] - rangeB[0]);
  for (let i = 0; i < n; i++) {
    const g = (a[i] - rangeA[0]) * sa, m = (b[i] - rangeB[0]) * sb;
    o[4 * i] = m; o[4 * i + 1] = g; o[4 * i + 2] = m; o[4 * i + 3] = 255;
  }
  return o;
}
