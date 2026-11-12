// Data of the hysteresis demo (no DOM), so scripts and tests can reproduce its numbers.
import { Grey, createGrey, sobel } from "../lib/imageops";
import { rng } from "../lib/random";
import { quantise8 } from "../histthresh/scenes";

export const N = 128;

/** Edge strength (Sobel magnitude) of a 128 × 128 image, scaled so 1/1.6 of the maximum is 255. */
export function edgeStrength(c: Grey): Grey {
  const m = sobel(c).mag;
  let mx = 0;
  for (const v of m.data) mx = Math.max(mx, v);
  return quantise8({ width: N, height: N, data: Float32Array.from(m.data, v => 255 * Math.min(1, 1.6 * v / mx)) });
}

/** Thin bright curves whose brightness fades from one end to the other, plus noise. */
export function fadingFilaments(seed: number): Grey {
  const r = rng(seed), val = new Float32Array(N * N), sg = 0.85;
  for (let k = 0; k < 7; k++) {
    const L = 50 + 60 * r.unif(), ang = Math.PI * r.unif(), cx = 20 + (N - 40) * r.unif(), cy = 20 + (N - 40) * r.unif();
    const dx = Math.cos(ang) * L / 2, dy = Math.sin(ang) * L / 2, off = 0.3 * L * (2 * r.unif() - 1);
    const p0 = [cx - dx, cy - dy], p1 = [cx - Math.sin(ang) * off, cy + Math.cos(ang) * off], p2 = [cx + dx, cy + dy];
    const a0 = 150 + 30 * r.unif(), a1 = 35 + 15 * r.unif();
    const steps = Math.ceil(L * 3);
    for (let s = 0; s <= steps; s++) {
      const t = s / steps, u = 1 - t;
      const px = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], py = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1];
      const amp = a0 + (a1 - a0) * t;
      for (let y = Math.max(0, Math.floor(py) - 3); y <= Math.min(N - 1, Math.floor(py) + 3); y++)
        for (let x = Math.max(0, Math.floor(px) - 3); x <= Math.min(N - 1, Math.floor(px) + 3); x++) {
          const v = amp * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sg * sg));
          if (v > val[y * N + x]) val[y * N + x] = v;
        }
    }
  }
  const img = createGrey(N, N);
  for (let i = 0; i < N * N; i++) img.data[i] = 30 + val[i] + 13 * r.norm();
  return quantise8(img);
}

/** A 1D profile: a structure fading from bright to faint, a faint separate bump, noise. */
export function profile1d(seed: number): Float64Array {
  const r = rng(seed), n = 200, y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = 30;
    if (i >= 25 && i <= 115) v += 130 - 85 * (i - 25) / 90;          // fades from 160 to 75
    if (i >= 150 && i <= 175) v += 45;                                 // faint, no seed
    y[i] = v + 12 * r.norm();
  }
  return y;
}

