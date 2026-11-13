// Denoising a path-traced image (pathtracer demo). The Gaussian, bilateral and non-local means
// filters come from lib/denoise.ts (deconvolution day), applied per colour channel to the
// displayed (sRGB) values. The joint (cross) bilateral filter is local: its range weights come
// from the noise-free guide images a renderer has for free (first-hit normal, albedo, depth), not
// from the noisy colours. (Request to the owner of lib/denoise.ts: add `jointBilateral` there.)
import { gaussianDenoise, bilateral, nonLocalMeans } from "../lib/denoise";
import { linearToSrgb } from "../lib/colour";

export type Denoiser = "none" | "gaussian" | "bilateral" | "nlm" | "joint";

/** Linear RGB → display values in [0, 1] (clipped, sRGB-encoded), as three channel planes. */
export function toDisplayPlanes(rgb: Float32Array, n: number): Float64Array[] {
  const out = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c][i] = linearToSrgb(Math.max(0, Math.min(1, rgb[3 * i + c])));
  return out;
}

/** Joint bilateral: w = exp(−r²/2σs²) · exp(−‖Δnormal‖²/2σn²) · exp(−‖Δalbedo‖²/2σa²). */
export function jointBilateral(img: Float64Array, w: number, h: number, normal: Float32Array, albedo: Float32Array, sigmaS: number, sigmaN = 0.1, sigmaA = 0.1): Float64Array {
  const out = new Float64Array(w * h), R = Math.ceil(2 * sigmaS);
  const ks = -1 / (2 * sigmaS * sigmaS), kn = -1 / (2 * sigmaN * sigmaN), ka = -1 / (2 * sigmaA * sigmaA);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    let s = 0, ws = 0;
    for (let dy = -R; dy <= R; dy++) {
      const yy = y + dy; if (yy < 0 || yy >= h) continue;
      for (let dx = -R; dx <= R; dx++) {
        const xx = x + dx; if (xx < 0 || xx >= w) continue;
        const q = yy * w + xx;
        let dn = 0, da = 0;
        for (let c = 0; c < 3; c++) { dn += (normal[3 * p + c] - normal[3 * q + c]) ** 2; da += (albedo[3 * p + c] - albedo[3 * q + c]) ** 2; }
        const wt = Math.exp((dx * dx + dy * dy) * ks + dn * 4 * kn + da * ka);
        s += wt * img[q]; ws += wt;
      }
    }
    out[p] = s / ws;
  }
  return out;
}

/** Denoise display planes; strength = σ in pixels (Gaussian, bilateral, joint) or the NLM h. */
export function denoise(kind: Denoiser, planes: Float64Array[], w: number, h: number, strength: number, aux: { normal: Float32Array; albedo: Float32Array }): Float64Array[] {
  if (kind === "none") return planes;
  return planes.map(p => {
    switch (kind) {
      case "gaussian": return gaussianDenoise(p, w, h, strength);
      case "bilateral": return bilateral(p, w, h, strength, 0.12);
      case "nlm": return nonLocalMeans(p, w, h, 2, 5, 0.06 * strength);
      default: return jointBilateral(p, w, h, aux.normal, aux.albedo, strength);
    }
  });
}
