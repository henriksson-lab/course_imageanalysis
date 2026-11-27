// Model of the fdmatch demo (shape-matching day, unit fourier-descriptors/): the outlines,
// the presets and one `compute()` that the page, the test and the figure scripts share. No DOM.
//
// Outline A is a shape; outline B is A turned by α (clockwise on screen), scaled by s,
// started m samples later, optionally mirrored and noisy, or a different shape. Both are
// sampled at N points (arc length, or equal steps in vertex index), optionally after being
// rasterised to a pixel mask and traced. Then: descriptors, the match over the starting point
// (one FFT), and the magnitude signature.
import {
  Pt, Coeffs, Match, descriptors, matchOutlines, magnitudeDistance, resampleArcLength, resampleByIndex,
  traceBoundary, clockwise, restart, mirrorTraced, similarity, polygonLength, sampleMean,
} from "../lib/contour";
import { family, meanParams, polygonCoverage } from "../lib/shapes";
import { rng } from "../lib/random";

export type ShapeId = "ellipse" | "square" | "leaf" | "hand" | "nucleus" | "shellA" | "shellB";

/** A source outline: a closed polygon (resampled), or a Fourier series sampled evenly in its
 *  parameter (so its |c_k| are exact). */
export type Source = { kind: "poly"; poly: Pt[] } | { kind: "series"; terms: [number, number, number][] };

const dense = (f: (t: number) => Pt, n = 720): Pt[] => Array.from({ length: n }, (_, i) => f(i / n));

function familyOutline(id: string, p?: number[]): Pt[] {
  const f = family(id), s = f.make(p ?? meanParams(f));
  const pts: Pt[] = [];
  for (let i = 0; i < s.length; i += 2) pts.push([s[i], s[i + 1]]);
  return normalised(clockwise(pts));
}

/** Centred on the mean of 512 arc-length samples, largest radius 1. */
function normalised(poly: Pt[]): Pt[] {
  const z = resampleArcLength(poly, 512);
  let cx = 0, cy = 0;
  for (const p of z) { cx += p[0]; cy += p[1]; }
  cx /= z.length; cy /= z.length;
  let r = 0;
  for (const p of z) r = Math.max(r, Math.hypot(p[0] - cx, p[1] - cy));
  return poly.map(p => [(p[0] - cx) / r, (p[1] - cy) / r] as Pt);
}

/** Two outlines with the same |c_k| for every k but different shapes (B: c_3 has the
 *  opposite sign; no rotation and no starting point can do that). */
export const SHELL_A: [number, number, number][] = [[1, 1, 0], [-1, 0.25, 0], [-2, 0.2, 0], [3, 0.12, 0]];
export const SHELL_B: [number, number, number][] = [[1, 1, 0], [-1, 0.25, 0], [-2, 0.2, 0], [3, -0.12, 0]];

export const SHAPES: Record<ShapeId, { label: string; src: Source }> = {
  ellipse: { label: "ellipse 2 × 1", src: { kind: "poly", poly: dense(t => [2 * Math.cos(2 * Math.PI * t), Math.sin(2 * Math.PI * t)]) } },
  square: { label: "square (corners 1, i, −1, −i)", src: { kind: "poly", poly: [[1, 0], [0, 1], [-1, 0], [0, -1]] } },
  leaf: { label: "leaf", src: { kind: "poly", poly: familyOutline("leaf", [0.36, 0.45, 0.2]) } },
  hand: { label: "hand", src: { kind: "poly", poly: familyOutline("hand") } },
  nucleus: { label: "nucleus", src: { kind: "poly", poly: familyOutline("nucleus", [0.2, 0.3, 0.15]) } },
  shellA: { label: "shape S", src: { kind: "series", terms: SHELL_A } },
  shellB: { label: "shape S′ (c₃ flipped)", src: { kind: "series", terms: SHELL_B } },
};

/** Synthetic nuclei for "which two are the same shape?": kidney outlines (lib/shapes.ts nucleus
 *  family: elongation, dent, bend), each at its own size, angle and position in the mask.
 *  Nuclei 2 and 5 have the same shape. */
export const NUCLEI: { p: number[]; size: number; angle: number }[] = [
  { p: [0.15, 0.22, 0.0], size: 15, angle: 20 },
  { p: [0.25, 0.30, 0.20], size: 14, angle: 110 },
  { p: [0.05, 0.12, -0.1], size: 16, angle: -40 },
  { p: [0.30, 0.10, 0.05], size: 15, angle: 160 },
  { p: [0.25, 0.30, 0.20], size: 18, angle: -75 },
  { p: [0.20, 0.38, 0.25], size: 15, angle: 60 },
];

export interface Settings {
  mode: "pair" | "nuclei";
  a: ShapeId; b: ShapeId | "same";
  n: number; K: number;
  /** B relative to A: angle (degrees, clockwise on screen), scale, start m samples later, mirror. */
  angle: number; scale: number; start: number; mirrorB: boolean;
  /** Noise on B's samples, as a fraction of A's largest radius. */
  noise: number;
  /** Rasterise to a pixel mask and trace; px = largest radius of A in pixels. */
  pixels: boolean; px: number;
  arclen: boolean; allowMirror: boolean; magOnly: boolean;
  nucA: number; nucB: number;
}

export const DEFAULTS: Settings = {
  mode: "pair", a: "leaf", b: "same", n: 64, K: 10, angle: 40, scale: 1.2, start: 20, mirrorB: false,
  noise: 0.01, pixels: false, px: 14, arclen: true, allowMirror: false, magOnly: false, nucA: 2, nucB: 5,
};

export interface Preset { id: string; label: string; s: Partial<Settings> }
export const PRESETS: Preset[] = [
  { id: "leaf", label: "leaf: turned, scaled, new start", s: {} },
  { id: "rotstart", label: "rotation vs new starting point", s: { a: "ellipse", n: 4, K: 2, angle: 0, scale: 1, start: 1, noise: 0 } },
  { id: "square", label: "a square sampled at its corners is a circle", s: { a: "square", n: 4, K: 2, angle: 0, scale: 1, start: 0, noise: 0 } },
  { id: "staircase", label: "staircase noise without resampling", s: { a: "nucleus", pixels: true, px: 14, arclen: false, angle: 30, scale: 1.3, start: 0, noise: 0, K: 16 } },
  { id: "samemag", label: "same magnitudes, different shape", s: { a: "shellA", b: "shellB", angle: 0, scale: 1, start: 0, noise: 0, K: 4, magOnly: true } },
  { id: "mirror", label: "mirror image", s: { a: "hand", angle: 25, scale: 1, start: 7, noise: 0, mirrorB: true, K: 12 } },
  { id: "nuclei", label: "nuclei: which two are the same shape?", s: { mode: "nuclei", pixels: true, nucA: 1, nucB: 2, K: 12 } },
];

const RAD = Math.PI / 180;

/** Pixel mask of a polygon (pixel centres at integers), centred in a square grid, and its traced boundary. */
export function rasterTrace(poly: Pt[], px: number): { trace: Pt[]; mask: Uint8Array; w: number } {
  let r = 0;
  for (const p of poly) r = Math.max(r, Math.hypot(p[0], p[1]));
  const w = Math.ceil(2 * r * px) + 8, c = (w - 1) / 2;
  const sh = new Float64Array(2 * poly.length);
  poly.forEach((p, i) => { sh[2 * i] = c + px * p[0]; sh[2 * i + 1] = c + px * p[1]; });
  const cov = polygonCoverage(sh, w, w), mask = new Uint8Array(w * w);
  for (let i = 0; i < w * w; i++) mask[i] = cov[i] > 0.5 ? 1 : 0;
  return { trace: traceBoundary(mask, w, w), mask, w };
}

function seriesSamples(terms: [number, number, number][], n: number): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    let x = 0, y = 0;
    for (const [k, re, im] of terms) {
      const a = 2 * Math.PI * k * i / n, cs = Math.cos(a), sn = Math.sin(a);
      x += re * cs - im * sn; y += re * sn + im * cs;
    }
    return [x, y] as Pt;
  });
}

export interface Sampled {
  /** The N samples (B: after the new start and the noise). */
  z: Pt[];
  /** What was sampled: the transformed polygon, or the traced pixel boundary. */
  path: Pt[];
  /** Pixel mask (pixels mode only). */
  mask?: { data: Uint8Array; w: number };
}

/** Sample a source after the pose x ↦ s e^{iα} x (mirror first: y → −y, walked clockwise again). */
export function sampleSource(src: Source, st: Pick<Settings, "n" | "arclen" | "pixels" | "px">, s = 1, alphaDeg = 0, mirror = false): Sampled {
  const a = alphaDeg * RAD;
  if (src.kind === "series" && !st.pixels) {
    let z = seriesSamples(src.terms, st.n);
    if (mirror) z = mirrorTraced(z);
    z = similarity(z, s, a);
    return { z, path: similarity(mirror ? mirrorTraced(seriesSamples(src.terms, 512)) : seriesSamples(src.terms, 512), s, a) };
  }
  let poly = src.kind === "poly" ? src.poly : seriesSamples(src.terms, 512);
  if (mirror) poly = mirrorTraced(poly);
  poly = similarity(poly, s, a);
  let mask: Sampled["mask"];
  if (st.pixels) {
    const r = rasterTrace(poly, st.px);
    poly = r.trace; mask = { data: r.mask, w: r.w };
  }
  const z = st.arclen ? resampleArcLength(poly, st.n) : resampleByIndex(poly, st.n);
  return { z, path: poly, mask };
}

/** Gaussian noise of standard deviation `sd` on every sample (seeded). */
export function addNoise(z: Pt[], sd: number, seed = 7): Pt[] {
  if (!sd) return z;
  const r = rng(seed);
  return z.map(p => [p[0] + sd * r.norm(), p[1] + sd * r.norm()] as Pt);
}

export interface Result {
  A: Sampled; B: Sampled;
  cA: Coeffs; cB: Coeffs;
  match: Match;
  magDist: number;
  /** For the nuclei mode: score and misfit of every pair (i < j), and the nuclei outlines. */
  pairs?: { i: number; j: number; score: number; misfit: number; mirrored: boolean }[];
  nuclei?: Sampled[];
}

const nucleiCache = new Map<string, Sampled[]>();
/** The six nuclei, rasterised and traced (pixel units), sampled at N points. */
export function nucleiOutlines(n: number, arclen: boolean): Sampled[] {
  const key = n + ":" + arclen;
  let v = nucleiCache.get(key);
  if (!v) {
    v = NUCLEI.map(q => sampleSource({ kind: "poly", poly: familyOutline("nucleus", q.p) }, { n, arclen, pixels: true, px: q.size }, 1, q.angle));
    nucleiCache.set(key, v);
  }
  return v;
}

/** Misfit √(1 − score²): the rms distance left after the best fit, relative to the size of B. */
export const misfit = (score: number): number => Math.sqrt(Math.max(0, 1 - score * score));

export function compute(st: Settings): Result {
  const K = Math.max(1, Math.min(st.K, Math.floor(st.n / 2)));
  if (st.mode === "nuclei") {
    const nuc = nucleiOutlines(st.n, st.arclen);
    const ia = Math.max(0, Math.min(nuc.length - 1, st.nucA - 1)), ib = Math.max(0, Math.min(nuc.length - 1, st.nucB - 1));
    const A = nuc[ia], B = nuc[ib], cA = descriptors(A.z), cB = descriptors(B.z);
    const pairs: Result["pairs"] = [];
    for (let i = 0; i < nuc.length; i++) for (let j = i + 1; j < nuc.length; j++) {
      const m = matchOutlines(nuc[i].z, nuc[j].z, { mirror: st.allowMirror });
      pairs.push({ i: i + 1, j: j + 1, score: m.score, misfit: misfit(m.score), mirrored: m.mirrored });
    }
    return { A, B, cA, cB, match: matchOutlines(A.z, B.z, { mirror: st.allowMirror }), magDist: magnitudeDistance(cA, cB, K), pairs, nuclei: nuc };
  }
  const srcA = SHAPES[st.a].src, srcB = st.b === "same" ? srcA : SHAPES[st.b].src;
  const A = sampleSource(srcA, st);
  const B0 = sampleSource(srcB, st, st.scale, st.angle, st.mirrorB);
  const [mx, my] = sampleMean(A.z);
  let r = 0;
  for (const p of A.z) r = Math.max(r, Math.hypot(p[0] - mx, p[1] - my));
  const zB = addNoise(restart(B0.z, st.start), st.noise * r);
  const B: Sampled = { z: zB, path: B0.path, mask: B0.mask };
  const cA = descriptors(A.z), cB = descriptors(B.z);
  return { A, B, cA, cB, match: matchOutlines(A.z, B.z, { mirror: st.allowMirror }), magDist: magnitudeDistance(cA, cB, K) };
}

/** Length of the sampled path (for the readout). */
export const pathLength = (s: Sampled): number => polygonLength(s.path);
