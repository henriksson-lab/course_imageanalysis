// Synthetic 128 × 128 test images for the snake demo (no DOM), values 0..1, fixed seeds.
// Each is drawn from a signed distance (anti-aliased over one pixel), blurred slightly, plus
// Gaussian noise. "weak" has an edge whose contrast fades to almost nothing on one side.
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { rng } from "../lib/random";

export const N = 128;

export type ImageId = "disk" | "square" | "u" | "blobs" | "weak" | "cell";

const BG = 0.2, FG = 0.8;

function fromSdf(sdf: (x: number, y: number) => number, value: (x: number, y: number) => number = () => FG): Grey {
  const g = createGrey(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = Math.max(0, Math.min(1, 0.5 - sdf(x, y)));
    g.data[y * N + x] = BG + (value(x, y) - BG) * a;
  }
  return g;
}

const boxSdf = (x: number, y: number, cx: number, cy: number, hw: number, hh: number, r = 0) => {
  const qx = Math.abs(x - cx) - hw + r, qy = Math.abs(y - cy) - hh + r;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
};

/** The clean image (before noise). */
export function cleanImage(id: Exclude<ImageId, "cell">): Grey {
  const c = 63.5;
  let g: Grey;
  switch (id) {
    case "disk": g = fromSdf((x, y) => Math.hypot(x - c, y - c) - 30); break;
    case "square": g = fromSdf((x, y) => boxSdf(x, y, c, c, 30, 30, 3)); break;
    case "u": g = fromSdf((x, y) => Math.max(boxSdf(x, y, c, c + 2, 36, 34, 4), -boxSdf(x, y, c, c - 22, 10, 36, 3))); break;
    case "blobs": g = fromSdf((x, y) => Math.min(Math.hypot(x - 41, y - c) - 20, Math.hypot(x - 87, y - c) - 20)); break;
    case "weak": g = fromSdf((x, y) => Math.hypot(x - c, y - c) - 30, (x) => {
      // contrast fades on the right: full at x ≤ 64, almost none at x ≥ 92
      const t = Math.max(0, Math.min(1, (x - 64) / 28));
      return FG - (FG - BG) * 0.85 * t;
    }); break;
  }
  return gaussianBlur(g, 1);
}

/** Add Gaussian noise of SD `sd` (fixed seed). */
export function addNoise(img: Grey, sd: number, seed = 11): Grey {
  const r = rng(seed), out = createGrey(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = img.data[i] + sd * r.norm();
  return out;
}
