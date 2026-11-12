// Synthetic fluorescence-like images for the unetseg demo (no DOM): 64 × 64 nuclei of varying
// size and brightness, touching pairs, an illumination gradient and noise, with the true mask.
// Drawn with lib/synth.ts `nuclei`. The SAME function makes the U-Net's training set
// (make_data.ts → raw bytes → train_unet.py), so "new image" in the demo draws from exactly
// the training distribution; the domain-shift settings move away from it in known ways.
import { nuclei } from "../lib/synth";
import { rng } from "../lib/random";

export const N = 64;

/** Departures from the training distribution (all 0 / 1 = typical training image). */
export interface Shift {
  /** Extra Gaussian noise sd added on top (grey levels; training noise is 3–10). */
  noise: number;
  /** Illumination gradient across the image in grey levels (training: 0–100, random). Negative = random. */
  gradient: number;
  /** Multiply the whole image (training: 1). */
  brightness: number;
  /** Multiply the nucleus radii (training: 1, radii 4–9 px). */
  size: number;
}
export const NO_SHIFT: Shift = { noise: 0, gradient: -1, brightness: 1, size: 1 };

export interface Sample { image: Float32Array; mask: Uint8Array; labels: Int32Array; params: Record<string, number> }

/** One image (grey levels, about 0..255) and its mask; the seed fixes everything. */
export function sample(seed: number, s: Shift = NO_SHIFT): Sample {
  const r = rng(seed * 7919 + 13);
  const u = (a: number, b: number) => a + (b - a) * r.unif();
  const count = 3 + Math.floor(6 * r.unif());
  const fg = u(90, 200), bg = u(20, 50), grad = s.gradient >= 0 ? s.gradient : u(0, 100), noise = u(3, 10);
  const flip = r.unif() < 0.5, transpose = r.unif() < 0.5;
  const nu = nuclei({
    width: N, height: N, count, radius: [4 * s.size, 9 * s.size], elongation: 0.5, touching: 0.35,
    foreground: fg, background: bg, gradient: grad, texture: 8, blur: 1, noise, seed: seed + 1,
  });
  const image = new Float32Array(N * N), mask = new Uint8Array(N * N), labels = new Int32Array(N * N);
  const nr = rng(seed * 31 + 5);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let sx = flip ? N - 1 - x : x, sy = y;
    if (transpose) [sx, sy] = [sy, sx];
    const i = y * N + x, j = sy * N + sx;
    image[i] = s.brightness * nu.image.data[j] + (s.noise > 0 ? s.noise * nr.norm() : 0);
    labels[i] = nu.labels[j];
    mask[i] = nu.labels[j] > 0 ? 1 : 0;
  }
  return { image, mask, labels, params: { count: nu.count, fg, bg, grad, noise } };
}
