// Computation of the localthresh demo (no DOM), so scripts and tests can reproduce its numbers.
import { Grey, createGrey, boxFilter } from "../lib/imageops";
import { intHistogram } from "../lib/histogram";
import { otsu, localThreshold, LocalMethod } from "../lib/threshold";
import { rng } from "../lib/random";
import { discMask, renderScene, illumination, quantise8 } from "../histthresh/scenes";

export const N = 256;
export type Method = "global" | "divide" | LocalMethod;
export interface Settings { source: "discs" | "cell"; method: Method; r: number; k: number; C: number; bg: number; light: number; vignette: number; noise: number }

/** Ground truth: 26 dark discs, radii 4 to 22. */
export const discTruth = (): Uint8Array => discMask(N, 26, 4, 22, 5);

/** Smooth background estimate: three box-filter passes (≈ Gaussian of scale σ). */
export function smoothBackground(img: Grey, sigma: number): Grey {
  const r = Math.max(1, Math.round(sigma));
  return boxFilter(boxFilter(boxFilter(img, r, "mirror"), r, "mirror"), r, "mirror");
}

/** Dark discs on a bright background (or the micrograph) times the illumination, plus noise; 8 bits. */
export function makeImage(s: Settings, truth: Uint8Array, cell: Grey | null): Grey {
  if (s.source === "discs" || !cell) return renderScene(truth, { background: 170, contrast: -100, noise: s.noise, gradient: s.light, vignette: s.vignette, edgeBlur: 1, seed: 9 });
  const L = illumination(N, N, s.light, s.vignette), r = rng(17), out = createGrey(N, N);
  for (let i = 0; i < N * N; i++) out.data[i] = cell.data[i] * L.data[i] + s.noise * r.norm();
  return quantise8(out);
}

/** Threshold surface T and mask [I < T] of a method. */
export function segment(img: Grey, s: Settings): { T: Grey; mask: Uint8Array; info: string } {
  let T: Grey, info: string;
  if (s.method === "global") {
    const o = otsu(intHistogram(img.data, 0, 255).counts);
    T = createGrey(N, N, o.t + 0.5);
    info = `Otsu t* = ${o.t}`;
  } else if (s.method === "divide") {
    const B = smoothBackground(img, s.bg), ratio = createGrey(N, N);
    for (let i = 0; i < N * N; i++) ratio.data[i] = Math.max(0, Math.min(255, Math.round(128 * img.data[i] / Math.max(1, B.data[i]))));
    const o = otsu(intHistogram(ratio.data, 0, 255).counts), tr = (o.t + 0.5) / 128;
    T = createGrey(N, N);
    for (let i = 0; i < N * N; i++) T.data[i] = tr * B.data[i];
    info = `Otsu on I/B: t* = ${tr.toFixed(2)} × background`;
  } else {
    T = localThreshold(img, s.method, { r: s.r, k: s.k, C: s.C, R: 128 });
    info = `window ${2 * s.r + 1} × ${2 * s.r + 1}`;
  }
  const mask = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) mask[i] = img.data[i] < T.data[i] ? 1 : 0;
  return { T, mask, info };
}
