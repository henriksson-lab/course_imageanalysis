// Model of the latentdiff demo (no DOM; tested in demos/test/diffusion-test.ts).
// Latent diffusion in the 2D latent space of the cell-shape VAE of the VAE day (vaeimages,
// shape_weights.ts, used unchanged). The training set's codes (encoder mean and sd of 800
// synthetic masks, make_codes.py) define the aggregate posterior q(z) = (1/N) Σ N(μ_i, diag σ_i²),
// whose noised denoiser is exact (a mixture). Samples: reverse diffusion in the latent plane,
// optionally conditioned on a kind with classifier-free guidance, then the decoder.
// Contrast: the VAE prior N(0, I) decoded directly.
import { SHAPE_VAE } from "../vaeimages/shape_weights";
import { deq } from "../lib/q8";
import { rng } from "../lib/ui";
import { sigmaGrid, sampleRun, guidedDenoiser, type Denoiser } from "../lib/diffusion";
import { CODE_MU, CODE_SD, CODE_KIND } from "./latent_codes";

export const KINDS = ["round", "elongated", "dividing", "protrusion"];
export const NCODES = CODE_KIND.length;

// ---- decoder 2 → 192 ReLU → 784 sigmoid (as vaeimages/imagevae.ts; dequantised with lib/q8)
let dec: { d1: ReturnType<typeof deq>; d2: ReturnType<typeof deq> } | null = null;
export function decode(z: ArrayLike<number>): Float32Array {
  dec ??= { d1: deq(SHAPE_VAE.d1), d2: deq(SHAPE_VAE.d2) };
  const { d1, d2 } = dec, h = new Float32Array(d1.nOut), out = new Float32Array(784);
  for (let o = 0; o < d1.nOut; o++) { let s = d1.b[o]; for (let i = 0; i < d1.nIn; i++) s += d1.W[o * d1.nIn + i] * z[i]; h[o] = Math.max(0, s); }
  for (let o = 0; o < 784; o++) { let s = d2.b[o]; const r = o * d2.nIn; for (let i = 0; i < d2.nIn; i++) s += d2.W[r + i] * h[i]; out[o] = 1 / (1 + Math.exp(-s)); }
  return out;
}

/**
 * Exact denoiser of the aggregate posterior (2D, diagonal components), restricted to `kind`
 * (null: all). `sdScale` multiplies the encoder sds (tiny: a memorising model).
 */
export function codeDenoiser(kind: number | null, sdScale = 1): Denoiser {
  const idx: number[] = [];
  for (let i = 0; i < NCODES; i++) if (kind === null || CODE_KIND[i] === kind) idx.push(i);
  const n = idx.length, mx_ = new Float64Array(n), my_ = new Float64Array(n), vx = new Float64Array(n), vy = new Float64Array(n), lw = new Float64Array(n);
  idx.forEach((i, a) => { mx_[a] = CODE_MU[2 * i]; my_[a] = CODE_MU[2 * i + 1]; vx[a] = (sdScale * CODE_SD[2 * i]) ** 2; vy[a] = (sdScale * CODE_SD[2 * i + 1]) ** 2; });
  return (x, s, out) => {
    const s2 = s * s, X = x[0], Y = x[1];
    let m = -Infinity;
    for (let a = 0; a < n; a++) {
      const cx = vx[a] + s2, cy = vy[a] + s2, dx = X - mx_[a], dy = Y - my_[a];
      const l = -0.5 * (dx * dx / cx + dy * dy / cy + Math.log(cx * cy));
      lw[a] = l; if (l > m) m = l;
    }
    let z = 0, ox = 0, oy = 0;
    for (let a = 0; a < n; a++) {
      const d = lw[a] - m; if (d < -30) continue;
      const p = Math.exp(d), cx = vx[a] + s2, cy = vy[a] + s2;
      z += p; ox += p * (mx_[a] + vx[a] / cx * (X - mx_[a])); oy += p * (my_[a] + vy[a] / cy * (Y - my_[a]));
    }
    out[0] = ox / z; out[1] = oy / z;
  };
}

export type Source = "diffusion" | "prior";
export interface Opts { source: Source; kind: number | null; w: number; steps: number; eta: number; sdScale: number; n: number; nPaths: number; seed: number }

export interface Result { Z: Float64Array; paths: Float64Array[] }

/** n latent samples (and the first nPaths paths, as x,y pairs per step). */
export function sampleLatents(o: Opts): Result {
  const r = rng(o.seed), rn = rng(o.seed + 1), Z = new Float64Array(2 * o.n), paths: Float64Array[] = [];
  if (o.source === "prior") {
    for (let i = 0; i < 2 * o.n; i++) Z[i] = r.norm();
    return { Z, paths };
  }
  const un = codeDenoiser(null, o.sdScale);
  const D = o.kind === null ? un : guidedDenoiser(codeDenoiser(o.kind, o.sdScale), un, o.w);
  const sig = sigmaGrid(o.steps, "time", "cosine"), x = new Float64Array(2);
  for (let p = 0; p < o.n; p++) {
    x[0] = Math.sqrt(sig[0] ** 2 + 1) * r.norm(); x[1] = Math.sqrt(sig[0] ** 2 + 1) * r.norm();
    const path = p < o.nPaths ? new Float64Array(2 * sig.length) : null;
    // paths are drawn in the variance-preserving form x/√(1 + σ²), so they start inside the plot
    sampleRun(x, D, { sigmas: sig, eta: o.eta }, rn, path ? (i, v) => { const k = 1 / Math.sqrt(1 + sig[i] ** 2); path[2 * i] = k * v[0]; path[2 * i + 1] = k * v[1]; } : undefined);
    if (path) paths.push(path);
    Z[2 * p] = x[0]; Z[2 * p + 1] = x[1];
  }
  return { Z, paths };
}

/** Kind of the nearest training code, and the distance to it, for every sample. */
export function nearestCode(Z: Float64Array): { kind: Int32Array; dist: Float64Array } {
  const n = Z.length / 2, kind = new Int32Array(n), dist = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    let b = Infinity, bi = 0;
    for (let i = 0; i < NCODES; i++) { const d = (Z[2 * p] - CODE_MU[2 * i]) ** 2 + (Z[2 * p + 1] - CODE_MU[2 * i + 1]) ** 2; if (d < b) { b = d; bi = i; } }
    kind[p] = CODE_KIND[bi]; dist[p] = Math.sqrt(b);
  }
  return { kind, dist };
}

/** Fraction of samples per kind (by nearest code). */
export function kindShares(kind: Int32Array): number[] {
  const c = [0, 0, 0, 0]; for (const k of kind) c[k]++;
  return c.map(v => v / kind.length);
}

/** Median nearest-code distance of the codes themselves (leave-one-out), for reference. */
export function codeSpacing(): number {
  const d: number[] = [];
  for (let i = 0; i < NCODES; i++) { let b = Infinity; for (let j = 0; j < NCODES; j++) if (j !== i) b = Math.min(b, (CODE_MU[2 * i] - CODE_MU[2 * j]) ** 2 + (CODE_MU[2 * i + 1] - CODE_MU[2 * j + 1]) ** 2); d.push(Math.sqrt(b)); }
  d.sort((a, b) => a - b);
  return d[d.length >> 1];
}

/** Ink outside the [0.2, 0.8] band: a crude "how crisp is the decoded outline" (fraction of grey pixels). */
export function greyFraction(img: Float32Array): number {
  let g = 0, ink = 0; for (const v of img) { if (v > 0.2 && v < 0.8) g++; if (v > 0.5) ink++; } return g / Math.max(1, ink);
}
