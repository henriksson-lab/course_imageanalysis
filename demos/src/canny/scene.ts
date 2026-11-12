// Synthetic test scene for the canny demo (no DOM), drawn for any size N (designed at 128):
// a rectangle, a disc, a staircase (two upward steps 4 px apart: close parallel edges), a
// T-junction and a faint
// rectangle whose contrast fades from 45 at the top to 8 at the bottom; Gaussian noise.
import { Grey, fromFunction } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";

export function cleanScene(N = 128): Grey {
  const s = N / 128;
  return fromFunction(N, N, (X, Y) => {
    const x = (X + 0.5) / s, y = (Y + 0.5) / s;
    if (x >= 8 && x < 48 && y >= 8 && y < 44) return 190;                         // rectangle
    if (Math.hypot(x - 86, y - 27) < 17) return 180;                               // disc
    if (x >= 8 && x < 20 && y >= 56 && y < 120) return x < 12 ? 120 : 190;          // staircase: steps at x = 8 and 12
    if (x >= 26 && x < 84 && y >= 56 && y < 120) return y < 88 ? (x < 55 ? 140 : 210) : 20;   // T-junction at (55, 88)
    if (x >= 96 && x < 122 && y >= 52 && y < 120) return 60 + 45 - 37 * (y - 52) / 68;        // faint, fading rectangle
    return 60;
  });
}

export function makeScene(kind: "synthetic", N = 128, noise = 8, seed = 1): { image: Grey } {
  const img = cleanScene(N);
  return { image: noise > 0 ? addGaussianNoise(img, noise, seed) : img };
}
