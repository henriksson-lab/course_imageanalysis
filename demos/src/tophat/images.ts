// Seeded test image for the top-hat demo (no DOM): small bright spots and small dark specks
// on a sloping, vignetted background with noise (synthetic), with the true positions.
import { Grey, createGrey } from "../lib/imageops";
import { spots, addGaussianNoise } from "../lib/synth";
import { rng } from "../lib/random";

export interface SpotImage { image: Grey; background: Grey; bright: { x: number; y: number }[]; dark: { x: number; y: number }[] }

export function spotsOnSlope(seed = 2): SpotImage {
  const s = spots({ count: 40, sigma: 1.6, amplitude: [45, 90], background: 60, slope: 90, bump: 50, noise: 0, seed });
  const w = s.image.width, h = s.image.height, img = createGrey(w, h), bg = createGrey(w, h);
  const vig = (x: number, y: number) => 1 - 0.35 * (((x - w / 2) ** 2 + (y - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, v = vig(x, y);
    img.data[i] = s.image.data[i] * v; bg.data[i] = s.background.data[i] * v;
  }
  // dark specks (dips of a Gaussian profile), away from the bright spots
  const r = rng(seed + 31), dark: { x: number; y: number }[] = [], sg = 1.5;
  for (let tries = 0; dark.length < 25 && tries < 4000; tries++) {
    const x = 6 + r.unif() * (w - 12), y = 6 + r.unif() * (h - 12);
    if (s.spots.some(p => Math.hypot(p.x - x, p.y - y) < 10) || dark.some(p => Math.hypot(p.x - x, p.y - y) < 10)) continue;
    dark.push({ x, y });
    const a = 40 + 25 * r.unif();
    for (let yy = Math.floor(y) - 6; yy <= Math.floor(y) + 6; yy++) for (let xx = Math.floor(x) - 6; xx <= Math.floor(x) + 6; xx++)
      img.data[yy * w + xx] -= a * Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sg * sg));
  }
  const noisy = addGaussianNoise(img, 4, seed + 77);
  for (let i = 0; i < noisy.data.length; i++) noisy.data[i] = Math.max(0, Math.min(255, noisy.data[i]));
  return { image: noisy, background: bg, bright: s.spots.map(p => ({ x: p.x, y: p.y })), dark };
}
