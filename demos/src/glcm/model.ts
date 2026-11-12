// Texture patches for the glcm demo (no DOM): 64 × 64, values 0..255, seeded.
import { gaussianBlur, fromArray } from "../lib/imageops";
import { rng, shuffle } from "../lib/random";

export const P = 64;

export type PatchId = "stripes" | "checker" | "noise" | "smooth" | "blobs" | "pairA" | "pairB" | "cell";
export const PATCHES: { id: PatchId; label: string }[] = [
  { id: "stripes", label: "stripes" },
  { id: "checker", label: "checkerboard" },
  { id: "noise", label: "white noise" },
  { id: "smooth", label: "smoothed noise" },
  { id: "blobs", label: "blobs" },
  { id: "pairA", label: "pair: coarse" },
  { id: "pairB", label: "pair: shuffled" },
  { id: "cell", label: "micrograph" },
];

export interface PatchParams {
  /** Stripe period (px) and angle (degrees, counter-clockwise on screen, of the stripe normal). */
  period?: number; angle?: number;
  /** Correlation length (Gaussian σ, px) of smoothed noise. */
  corr?: number;
  /** Checker square size. */
  square?: number;
  /** Contrast factor around 128 (1 = full range) and brightness offset. */
  contrast?: number; brightness?: number;
  seed?: number;
}

/** Rescale to 0..255 by the min and max. */
function stretch(a: Float64Array): Float64Array {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return a.map(v => (hi > lo ? 255 * (v - lo) / (hi - lo) : 128));
}

function smoothNoise(seed: number, sigma: number): Float64Array {
  const r = rng(seed);
  const raw = Float64Array.from({ length: P * P }, () => r.norm());
  return stretch(Float64Array.from(gaussianBlur(fromArray(P, P, raw), sigma, "wrap").data));
}

/** Histogram-equalise to 256 equally filled levels (so the "pair" has a flat histogram). */
function equalise(a: Float64Array): Float64Array {
  const idx = Array.from(a.keys()).sort((i, j) => a[i] - a[j]);
  const out = new Float64Array(a.length);
  idx.forEach((i, k) => { out[i] = Math.floor(k * 256 / a.length); });
  return out;
}

/** A 64 × 64 patch (values 0..255). `cell` needs the micrograph crop passed in `cellCrop`. */
export function makePatch(id: PatchId, p: PatchParams, cellCrop?: ArrayLike<number>): Float64Array {
  const seed = p.seed ?? 1;
  let a: Float64Array;
  switch (id) {
    case "stripes": {
      const per = p.period ?? 8, t = (p.angle ?? 0) * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
      a = Float64Array.from({ length: P * P }, (_, i) => {
        const x = i % P, y = Math.floor(i / P);
        return 127.5 + 127.5 * Math.cos(2 * Math.PI * (x * c - y * s) / per);
      });
      break;
    }
    case "checker": {
      const q = p.square ?? 4;
      a = Float64Array.from({ length: P * P }, (_, i) => ((Math.floor((i % P) / q) + Math.floor(Math.floor(i / P) / q)) % 2 ? 255 : 0));
      break;
    }
    case "noise": { const r = rng(seed); a = Float64Array.from({ length: P * P }, () => 255 * r.unif()); break; }
    case "smooth": a = smoothNoise(seed, p.corr ?? 3); break;
    case "blobs": {
      const r = rng(seed + 50);
      a = new Float64Array(P * P);
      for (let k = 0; k < 22; k++) {
        const cx = P * r.unif(), cy = P * r.unif(), rad = 3 + 4 * r.unif();
        for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
          const dx = Math.min(Math.abs(x - cx), P - Math.abs(x - cx)), dy = Math.min(Math.abs(y - cy), P - Math.abs(y - cy));
          a[y * P + x] += Math.exp(-(dx * dx + dy * dy) / (2 * rad * rad));
        }
      }
      a = stretch(a);
      break;
    }
    case "pairA": a = equalise(smoothNoise(seed + 10, p.corr ?? 3)); break;
    case "pairB": a = shuffle(rng(seed + 20), equalise(smoothNoise(seed + 10, p.corr ?? 3))); break;
    case "cell": a = Float64Array.from(cellCrop ?? new Float64Array(P * P).fill(128)); break;
  }
  const k = p.contrast ?? 1, b = p.brightness ?? 0;
  if (k !== 1 || b !== 0) a = a.map(v => Math.max(0, Math.min(255, 128 + k * (v - 128) + b)));
  return a;
}

/** Disc mask of radius r centred in the patch. */
export function discMask(r: number): Uint8Array {
  const c = (P - 1) / 2;
  return Uint8Array.from({ length: P * P }, (_, i) => (Math.hypot(i % P - c, Math.floor(i / P) - c) <= r ? 1 : 0));
}
