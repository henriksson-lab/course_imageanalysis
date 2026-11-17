// Model of the maskfill demo (no DOM; tested in demos/test/maskfill-test.ts).
//
// Masked image modelling in miniature (unit pretext/). A 120 × 120 test image is cut into
// 6 × 6 patches (20 × 20); a fraction of the inner patches is hidden (or, for comparison, a
// fraction of single pixels). Two ways to fill in what is hidden:
//   interp   local interpolation: a Gaussian-weighted mean of the visible pixels nearby
//            (normalised convolution, σ = 3 px); knows nothing about images;
//   pca      a linear model of 18 × 18 windows (3 × 3 patches), learnt WITHOUT labels from
//            OTHER images of the same kind: mean window + k principal components. For each
//            hidden patch, fit the k coefficients to the visible pixels of its window (least
//            squares with a ridge: coefficient a has the prior variance of component a, the
//            residual variance is lambda), and read the hidden pixels off the fitted window.
//            k = 0 is the mean window.
// Two scenes: synthetic nuclei (lib/synth.ts; model learnt on seed 1, test crop from seed 2)
// and the course micrograph (model learnt on its upper half, test crop from its lower half).
// Error: PSNR (peak 255) over the hidden pixels only.
import { solve } from "../lib/linalg";
import { nuclei } from "../lib/synth";
import { rng } from "../lib/ui";

export const P = 6, G = 20, N = P * G, WIN = 3 * P, DW = WIN * WIN, KMAX = 40;

export interface PcaModel { mean: Float64Array; comps: Float64Array[]; values: number[] }

/** All 18 × 18 windows (stride 6) of a w × h image. */
export function trainingWindows(img: ArrayLike<number>, w: number, h: number): Float64Array[] {
  const out: Float64Array[] = [];
  for (let y0 = 0; y0 + WIN <= h; y0 += 6) for (let x0 = 0; x0 + WIN <= w; x0 += 6) {
    const v = new Float64Array(DW);
    for (let y = 0; y < WIN; y++) for (let x = 0; x < WIN; x++) v[y * WIN + x] = img[(y0 + y) * w + x0 + x];
    out.push(v);
  }
  return out;
}

/** Mean and the top KMAX principal components (subspace iteration on the covariance). */
export function fitPca(windows: Float64Array[]): PcaModel {
  const n = windows.length, mean = new Float64Array(DW);
  for (const w of windows) for (let i = 0; i < DW; i++) mean[i] += w[i] / n;
  const C = new Float64Array(DW * DW), d = new Float64Array(DW);
  for (const w of windows) {
    for (let i = 0; i < DW; i++) d[i] = w[i] - mean[i];
    for (let i = 0; i < DW; i++) { const di = d[i]; if (di === 0) continue; const row = i * DW; for (let j = i; j < DW; j++) C[row + j] += di * d[j]; }
  }
  for (let i = 0; i < DW; i++) for (let j = i; j < DW; j++) { C[i * DW + j] /= n; C[j * DW + i] = C[i * DW + j]; }
  const r = rng(3);
  let Q = Array.from({ length: KMAX }, () => Float64Array.from({ length: DW }, () => r.norm()));
  const orth = (V: Float64Array[]) => {
    for (let a = 0; a < V.length; a++) {
      for (let b = 0; b < a; b++) { let s = 0; for (let i = 0; i < DW; i++) s += V[a][i] * V[b][i]; for (let i = 0; i < DW; i++) V[a][i] -= s * V[b][i]; }
      let s = 0; for (let i = 0; i < DW; i++) s += V[a][i] ** 2; s = Math.sqrt(s) || 1; for (let i = 0; i < DW; i++) V[a][i] /= s;
    }
  };
  orth(Q);
  for (let it = 0; it < 30; it++) {
    Q = Q.map(q => { const o = new Float64Array(DW); for (let i = 0; i < DW; i++) { let s = 0; const row = i * DW; for (let j = 0; j < DW; j++) s += C[row + j] * q[j]; o[i] = s; } return o; });
    orth(Q);
  }
  // Rayleigh quotients, then sort by them (subspace iteration keeps the order only roughly)
  const values = Q.map(q => { let s = 0; for (let i = 0; i < DW; i++) { let t = 0; for (let j = 0; j < DW; j++) t += C[i * DW + j] * q[j]; s += q[i] * t; } return s; });
  const order = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map(p => p[1]);
  return { mean, comps: order.map(i => Q[i]), values: order.map(i => values[i]) };
}

export type MaskKind = "patches" | "pixels";

/** Hidden pixels (1) of the N × N crop: a fraction of the inner patches, or of the inner pixels. */
export function makeMask(kind: MaskKind, ratio: number, seed: number): Uint8Array {
  const r = rng(seed), m = new Uint8Array(N * N);
  if (kind === "patches") {
    for (let gy = 1; gy < G - 1; gy++) for (let gx = 1; gx < G - 1; gx++) {
      if (r.unif() >= ratio) continue;
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) m[(gy * P + y) * N + gx * P + x] = 1;
    }
  } else {
    for (let y = P; y < N - P; y++) for (let x = P; x < N - P; x++) if (r.unif() < ratio) m[y * N + x] = 1;
  }
  return m;
}

/** Fill by normalised convolution: Gaussian-weighted mean of the visible pixels (σ in px). */
export function fillInterp(img: ArrayLike<number>, mask: Uint8Array, sigma = 3): Float64Array {
  const R = Math.ceil(3 * sigma), k = Float64Array.from({ length: 2 * R + 1 }, (_, i) => Math.exp(-((i - R) ** 2) / (2 * sigma * sigma)));
  const num = new Float64Array(N * N), den = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) if (!mask[i]) { num[i] = img[i]; den[i] = 1; }
  const blur = (a: Float64Array) => {
    const t = new Float64Array(N * N), o = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let s = 0; for (let d = -R; d <= R; d++) { const xx = x + d; if (xx >= 0 && xx < N) s += k[d + R] * a[y * N + xx]; } t[y * N + x] = s; }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let s = 0; for (let d = -R; d <= R; d++) { const yy = y + d; if (yy >= 0 && yy < N) s += k[d + R] * t[yy * N + x]; } o[y * N + x] = s; }
    return o;
  };
  const bn = blur(num), bd = blur(den), out = Float64Array.from(img);
  for (let i = 0; i < N * N; i++) if (mask[i]) out[i] = bd[i] > 1e-9 ? bn[i] / bd[i] : 128;
  return out;
}

/** Fill with the PCA model (k components, ridge lambda): see the header. */
export function fillPca(img: ArrayLike<number>, mask: Uint8Array, model: PcaModel, k: number, lambda = 100): Float64Array {
  const out = Float64Array.from(img);
  for (let gy = 1; gy < G - 1; gy++) for (let gx = 1; gx < G - 1; gx++) {
    const y0 = (gy - 1) * P, x0 = (gx - 1) * P;
    // does the centre patch have hidden pixels?
    let any = false;
    for (let y = 0; y < P && !any; y++) for (let x = 0; x < P; x++) if (mask[(gy * P + y) * N + gx * P + x]) { any = true; break; }
    if (!any) continue;
    const c = new Float64Array(k);
    if (k > 0) {
      const A = new Float64Array(k * k), b = new Float64Array(k);
      for (let y = 0; y < WIN; y++) for (let x = 0; x < WIN; x++) {
        const gi = (y0 + y) * N + x0 + x;
        if (mask[gi]) continue;
        const wi = y * WIN + x, res = img[gi] - model.mean[wi];
        for (let a = 0; a < k; a++) {
          const ua = model.comps[a][wi];
          b[a] += ua * res;
          for (let q = a; q < k; q++) A[a * k + q] += ua * model.comps[q][wi];
        }
      }
      for (let a = 0; a < k; a++) { A[a * k + a] += lambda / model.values[a]; for (let q = 0; q < a; q++) A[a * k + q] = A[q * k + a]; }
      c.set(solve(A, b, k) ?? new Float64Array(k));
    }
    for (let y = P; y < 2 * P; y++) for (let x = P; x < 2 * P; x++) {
      const gi = (y0 + y) * N + x0 + x;
      if (!mask[gi]) continue;
      const wi = y * WIN + x;
      let v = model.mean[wi];
      for (let a = 0; a < k; a++) v += c[a] * model.comps[a][wi];
      out[gi] = v;
    }
  }
  return out;
}

/** PSNR (peak 255) over the hidden pixels only. */
export function hiddenPsnr(a: ArrayLike<number>, truth: ArrayLike<number>, mask: Uint8Array): number {
  let s = 0, n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { s += (a[i] - truth[i]) ** 2; n++; }
  return n ? 10 * Math.log10(255 * 255 / (s / n)) : Infinity;
}

/** A 120 × 120 crop at (x0, y0) of a w-wide image. */
export function testCrop(img: ArrayLike<number>, w: number, x0: number, y0: number): Float64Array {
  const out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) out[y * N + x] = img[(y + y0) * w + x + x0];
  return out;
}

export type Scene = "nuclei" | "cell";
/** Training windows and the test crop of a scene (`cell`: the 256 × 256 micrograph, 0..255). */
export function scene(kind: Scene, cell?: ArrayLike<number>): { windows: Float64Array[]; test: Float64Array } {
  if (kind === "nuclei") {
    const tr = nuclei({ seed: 1, count: 40 }).image.data, te = nuclei({ seed: 2, count: 40 }).image.data;
    return { windows: trainingWindows(tr, 256, 256), test: testCrop(te, 256, 60, 60) };
  }
  return { windows: trainingWindows(cell!, 256, 128), test: testCrop(cell!, 256, 68, 132) };
}
