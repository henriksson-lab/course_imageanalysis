// Model of the gac demo (no DOM; used by the chanvese demo and the tests): the synthetic test
// image, the edge-stopping function g and one step of the edge-based level set flow
//   φ_t = −g (c − ε κ) |∇φ| + a ∇g·∇φ          (φ < 0 inside)
// c = balloon (> 0 inflates), ε = curvature weight, a = advection towards the edges
// (geodesic active contours: a = ε; a = 0 is the "braking balloon" of Caselles et al. 1993 /
// Malladi, Sethian & Vemuri 1995).
import * as LS from "../lib/levelset";
import { Grey, fromArray, gradient } from "../lib/imageops";
import { rng, gauss } from "../lib/random";

export const N = 128;

/** Objects of the synthetic image (pixel coordinates). */
export const BLOBS = { A: { x: 28, y: 28, r: 13 }, B: { x: 28, y: 100, r: 13 }, D: { x: 58, y: 64, r: 11 } };
export const CSHAPE = { x: 100, y: 64, R: 22, r: 11, gap: 6 };
export const BG = 60, FG = 190, WEAK = 85;

/** Is (x, y) inside the C shape (a ring open towards +x)? */
export function inC(x: number, y: number): boolean {
  const d = Math.hypot(x - CSHAPE.x, y - CSHAPE.y);
  return d <= CSHAPE.R && d >= CSHAPE.r ? !(x > CSHAPE.x && Math.abs(y - CSHAPE.y) < CSHAPE.gap) : false;
}

/** Noise-free synthetic image (grey 0..255) with its object mask (1 = object). Blob B fades to
 *  WEAK on its right side: a weak edge. */
export function synthClean(): { img: Float64Array; mask: Uint8Array } {
  const img = new Float64Array(N * N), mask = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = BG, m = 0;
    const { A, B, D } = BLOBS;
    // anti-aliased discs: coverage from the distance to the rim
    const cov = (cx: number, cy: number, r: number) => Math.max(0, Math.min(1, r - Math.hypot(x - cx, y - cy) + 0.5));
    const a = Math.max(cov(A.x, A.y, A.r), cov(D.x, D.y, D.r));
    if (a > 0) v = BG + a * (FG - BG);
    const b = cov(B.x, B.y, B.r);
    if (b > 0) {
      const top = FG - (FG - WEAK) * Math.max(0, Math.min(1, (x - (B.x - 4)) / (B.r + 4)));
      v = BG + b * (top - BG);
    }
    if (inC(x, y)) v = FG;
    if (a >= 0.5 || b >= 0.5 || inC(x, y)) m = 1;
    img[y * N + x] = v; mask[y * N + x] = m;
  }
  return { img, mask };
}

/** Add Gaussian noise of standard deviation sd (fixed seed). */
export function addNoise(img: ArrayLike<number>, sd: number, seed = 7): Float64Array {
  const r = rng(seed);
  return Float64Array.from(img, v => v + (sd > 0 ? gauss(r, 0, sd) : 0));
}

export interface EdgeMap { g: Float64Array; gx: Float64Array; gy: Float64Array; mag: Float64Array }

/** g = 1 / (1 + |∇(G_σ * I)|² / k²) and its gradient. σ = 0: plain central differences. */
export function edgeMap(img: ArrayLike<number>, sigma: number, k: number): EdgeMap {
  const G: Grey = fromArray(N, N, img);
  const { mag } = gradient(G, sigma);
  const m = Float64Array.from(mag.data);
  const e = LS.edgeStop(m, N, N, k);
  return { ...e, mag: m };
}

export interface GacParams { c: number; eps: number; a: number }

/** Speed terms of the flow for an edge map. */
export function gacTerms(E: EdgeMap, P: GacParams): LS.SpeedTerms {
  return {
    F: P.c, eps: P.eps, g: E.g,
    vx: P.a > 0 ? Float64Array.from(E.gx, v => -P.a * v) : undefined,
    vy: P.a > 0 ? Float64Array.from(E.gy, v => -P.a * v) : undefined,
  };
}

/** Union of circles as a signed distance function. */
export function circlesPhi(cs: { x: number; y: number; r: number }[]): Float64Array {
  if (!cs.length) return new Float64Array(N * N).fill(N);
  return LS.fromFn(N, N, (x, y) => Math.min(...cs.map(c => Math.hypot(x - c.x, y - c.y) - c.r)));
}

/** A rectangle 4 pixels inside the image border. */
export function boxPhi(): Float64Array { return LS.fromFn(N, N, (x, y) => LS.sdfBox(x, y, 63.5, 63.5, 59.5, 59.5)); }
