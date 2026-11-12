// Synthetic textures for the structtensor demo (no DOM), N × N, values around 128:
//  stripes (angle θ on screen, counter-clockwise, of the stripes' gradient; frequency f cycles/px),
//  rotating (the stripe angle turns from 0° at the left to 180° at the right),
//  fingerprint (concentric rings), boundary (two orientations meeting at x = N/2),
//  checker (a checkerboard on the left, a square and a triangle on the right),
//  square (one bright square: edges, corners and a flat inside), noise (isotropic, smoothed).
import { Grey, fromFunction, gaussianBlur } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";
import { rng } from "../lib/random";

export const TEXTURES = ["stripes", "rotating", "fingerprint", "boundary", "checker", "square", "noise"] as const;
export type TextureId = typeof TEXTURES[number];

const wave = (x: number, y: number, th: number, f: number) => Math.sin(2 * Math.PI * f * (Math.cos(th) * x - Math.sin(th) * y));

function inTriangle(x: number, y: number, a: number[], b: number[], c: number[]): boolean {
  const s = (p: number[], q: number[]) => (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
  const d1 = s(a, b), d2 = s(b, c), d3 = s(c, a);
  return (d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0);
}

export function cleanTexture(id: string, N = 128, angleDeg = 30, freq = 0.12): Grey {
  const th = angleDeg * Math.PI / 180, s = N / 128;
  switch (id) {
    case "rotating":
      return fromFunction(N, N, (x, y) => {
        // phase whose gradient direction turns with x: use local angle a(x) and integrate along y-dominant waves
        const a = Math.PI * x / N, cx = x - N / 2, cy = y - N / 2;
        return 128 + 80 * Math.sin(2 * Math.PI * freq * (Math.cos(a) * cx - Math.sin(a) * cy));
      });
    case "fingerprint":
      return fromFunction(N, N, (x, y) => 128 + 80 * Math.cos(2 * Math.PI * freq * Math.hypot(x - N / 2 + 0.5, y - N / 2 + 0.5)));
    case "boundary":
      return fromFunction(N, N, (x, y) => 128 + 80 * wave(x, y, x < N / 2 ? th : th + Math.PI / 2, freq));
    case "checker": {
      const q = 16 * s;
      return fromFunction(N, N, (x, y) => {
        if (x < N / 2) return (Math.floor(x / q) + Math.floor(y / q)) % 2 ? 200 : 60;
        const X = x / s, Y = y / s;
        if (X >= 76 && X < 116 && Y >= 12 && Y < 52) return 200;
        if (inTriangle(X, Y, [96, 66], [120, 116], [72, 116])) return 200;
        return 60;
      });
    }
    case "square":
      return fromFunction(N, N, (x, y) => { const X = x / s, Y = y / s; return X >= 32 && X < 96 && Y >= 32 && Y < 96 ? 200 : 60; });
    case "noise": {
      const r = rng(17), g = fromFunction(N, N, () => r.norm());
      const b = gaussianBlur(g, 1.5);
      let sd = 0; for (const v of b.data) sd += v * v; sd = Math.sqrt(sd / b.data.length);
      for (let i = 0; i < b.data.length; i++) b.data[i] = 128 + 50 * b.data[i] / sd;
      return b;
    }
    default:
      return fromFunction(N, N, (x, y) => 128 + 80 * wave(x - N / 2, y - N / 2, th, freq));
  }
}

export function makeTexture(id: string, N = 128, noise = 0, seed = 1, angleDeg = 30, freq = 0.12): Grey {
  const img = cleanTexture(id, N, angleDeg, freq);
  return noise > 0 ? addGaussianNoise(img, noise, seed) : img;
}
