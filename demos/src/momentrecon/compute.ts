// Pure computations of the momentrecon demo (no DOM): an image rebuilt from its moments up to
// order N with four bases. Shared with the scratch scripts that compute the numbers quoted in
// the orthogonal-moments and zernike exercises.
//   monomials: least squares through the Gram system of the geometric moments (lib/moments.ts
//              monomialReconstruct; ill-conditioned, computed only up to NMON);
//   Legendre:  orthogonal projection with Legendre polynomials sampled at pixel centres
//              ("point"), averaged over each pixel ("integrated"), or the discrete orthogonal
//              polynomials ("discrete", exact on the grid);
//   Zernike:   on the inscribed disc (0 outside);
//   Fourier:   the same number of coefficients, lowest frequencies first (a round low-pass).
// Errors: relative RMS against the clean image over the whole square.
import {
  legendreMoments, orthoBasis, reconstructSeparable, idx, monomialReconstruct, discGrid, zernikeMoments,
  zernikeReconstruct, relError, countUpTo, unitFrame, zeroOneFrame, polygonCoverageEvenOdd, OrthoVariant, PixelModel,
} from "../lib/moments";
import { fftReal2d, ifftReal2d } from "../lib/fft";
import { rng } from "../lib/random";
import { FAMILIES, meanParams } from "../lib/shapes";

export const W = 64;
export const NMAX = 40;
export const NMON = 32;

export type Pt = [number, number];

export const LETTER_E: Pt[] = [[14, 10], [48, 10], [48, 18], [23, 18], [23, 28], [43, 28], [43, 36], [23, 36], [23, 46], [48, 46], [48, 54], [14, 54]];
export const SQUARE: Pt[] = [[16, 16], [48, 16], [48, 48], [16, 48]];

/** The kidney-shaped nucleus of lib/shapes.ts (mean parameters), about 46 pixels across. */
export function nucleusPoly(): Pt[] {
  const fam = FAMILIES.find(f => f.id === "nucleus")!, p = meanParams(fam);
  p[1] = 0.3;  // a clear dent
  const s = fam.make(p), out: Pt[] = [];
  for (let i = 0; i < s.length / 2; i++) out.push([32 + 30 * s[2 * i], 32 + 30 * s[2 * i + 1]]);
  return out;
}

/** Anti-aliased image (0..1) of a polygon on the 64 × 64 grid (pixel (x, y) covers
 *  [x, x+1] × [y, y+1]). */
export function polyImage(p: Pt[]): Float64Array {
  return polygonCoverageEvenOdd([p], W, W, 4);
}

export function addNoise(f: Float64Array, sigma: number, seed = 17): Float64Array {
  if (!(sigma > 0)) return f;
  const r = rng(seed);
  return f.map(v => v + sigma * r.norm());
}

export type Variant = "point" | "integrated" | "discrete";

export interface AllRecons {
  leg: Float64Array[];
  zer: Float64Array[];
  four: Float64Array[];
  errLeg: number[]; errZer: number[]; errFour: number[];
}

/** Fourier low-pass keeping the K lowest frequencies (sorted by radius, then angle). */
export function fourierOrder(): number[] {
  const ids = Array.from({ length: W * W }, (_, i) => i);
  const fq = (k: number) => (k < W / 2 ? k : k - W);
  const key = (i: number) => { const u = fq(i % W), v = fq(Math.floor(i / W)); return [u * u + v * v, Math.atan2(v, u)]; };
  const keys = ids.map(key);
  return ids.sort((a, b) => keys[a][0] - keys[b][0] || keys[a][1] - keys[b][1]);
}

export function computeAll(f: Float64Array, clean: Float64Array, variant: Variant): AllRecons {
  // Legendre / discrete
  const ov: OrthoVariant = variant === "point" ? "sampled" : variant === "integrated" ? "integrated" : "discrete";
  const a = legendreMoments(f, W, W, NMAX, ov), B = orthoBasis(W, NMAX, ov);
  const leg: Float64Array[] = [];
  for (let N = 0; N <= NMAX; N++) leg.push(reconstructSeparable((p, q) => a[idx(NMAX, p, q)], B.S, B.S, W, W, N));
  // Zernike on the inscribed disc, all orders at once (incremental)
  const g = discGrid(W, W), Z = zernikeMoments(f, g, NMAX), zer: Float64Array[] = [];
  zernikeReconstruct(Z, g, NMAX, (_n, img) => zer.push(img));
  // Fourier
  const F = fftReal2d(f, W, W), order = fourierOrder(), four: Float64Array[] = [];
  for (let N = 0; N <= NMAX; N++) {
    const K = countUpTo(N), re = new Float64Array(W * W), im = new Float64Array(W * W);
    for (let k = 0; k < K; k++) { const i = order[k]; re[i] = F.re[i]; im[i] = F.im[i]; }
    four.push(ifftReal2d(re, im, W, W));
  }
  return {
    leg, zer, four,
    errLeg: leg.map(r => relError(r, clean)), errZer: zer.map(r => relError(r, clean)), errFour: four.map(r => relError(r, clean)),
  };
}

/** Monomial reconstruction with coordinates in [−1, 1] ("sym") or [0, 1] ("unit"). */
export function monomial(f: Float64Array, N: number, variant: Variant, domain: "sym" | "unit" = "sym"): Float64Array {
  const px: PixelModel = variant === "integrated" ? "square" : "point";
  return monomialReconstruct(f, W, W, N, px, domain === "sym" ? unitFrame(W, W) : zeroOneFrame(W, W));
}
