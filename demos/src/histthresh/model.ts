// Computation of the histthresh demo (no DOM), so scripts and tests can reproduce its numbers.
import { Grey, gaussianBlur } from "../lib/imageops";
import { median, seSquare } from "../lib/morph";
import { histogram, intHistogram, Histogram } from "../lib/histogram";
import { otsu, otsuMulti3 } from "../lib/threshold";
import { rng } from "../lib/random";
import { blobMask, renderScene, quantise8 } from "./scenes";

export const N = 256;

/** Ground truth of the synthetic blobs: 25 % object pixels. */
export const blobTruth = (): Uint8Array => blobMask(N, N, 0.25, 6, 3);

export interface ImageSettings { source: "blobs" | "cell"; contrast: number; noise: number; median: boolean; blur: number }

/** The analysed image: blobs (or the micrograph) + noise, then median 3 × 3, then Gaussian blur, rounded to 8 bits. */
export function makeImage(s: ImageSettings, truth: Uint8Array, cell: Grey | null): Grey {
  let base: Grey;
  if (s.source === "blobs" || !cell) base = renderScene(truth, { contrast: s.contrast, noise: s.noise, seed: 5 });
  else {
    const r = rng(11);
    base = quantise8({ width: N, height: N, data: Float32Array.from(cell.data, v => v + s.noise * r.norm()) });
  }
  if (s.median) base = { width: N, height: N, data: median(base.data, N, N, seSquare(1)) };
  if (s.blur > 0) base = gaussianBlur(base, s.blur);
  return quantise8(base);
}

export interface Analysis {
  h256: Histogram;            // one bin per grey level
  hb: Histogram;              // `bins` bins over [0, 256)
  w: number;                  // bin width
  otsu: ReturnType<typeof otsu>;
  otsuT: number;              // Otsu's threshold in grey levels (mask I > otsuT)
  multi: { t1v: number; t2v: number };
}

/** Histograms and the Otsu thresholds (computed on the binned histogram, mapped back to grey levels). */
export function analyse(img: Grey, bins: number): Analysis {
  const h256 = intHistogram(img.data, 0, 255);
  const hb = histogram(img.data, { bins, min: 0, max: 256 });
  const w = 256 / bins;
  const centres = Float64Array.from(hb.counts, (_, i) => (i + 0.5) * w - 0.5);
  const o = otsu(hb.counts, centres);
  const m = otsuMulti3(hb.counts, centres);
  return { h256, hb, w, otsu: o, otsuT: Math.round((o.t + 1) * w - 1), multi: { t1v: Math.round((m.t1 + 1) * w - 1), t2v: Math.round((m.t2 + 1) * w - 1) } };
}

/** Smallest t with P(t) ≥ 1 − fraction (keep the brightest `fraction`). */
export function percentileT(h256: Histogram, fraction: number): number {
  let s = 0;
  for (let v = 0; v < 256; v++) { s += h256.counts[v]; if (s >= (1 - fraction) * h256.total) return v; }
  return 255;
}
