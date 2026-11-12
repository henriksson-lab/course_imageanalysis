// Synthetic blob image for the blobscale demo (no DOM): flat discs of radii 2–20 px and
// Gaussian spots, a touching pair, an elongated blob, on an uneven background (a slope and
// a broad bright patch), with seeded Gaussian noise. Ground truth: centre, radius and kind.
// The expected detection scale is σ = r/√2 for a disc and σ = s for a Gaussian spot of width s
// (drawn radius √2·s).
import { Grey, fromFunction } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";

export interface Blob { x: number; y: number; r: number; kind: "disc" | "gauss" | "ellipse"; amp: number; b?: number }

/** Blobs at 256 × 256 (positions scale with N). For "gauss", r = √2·s (the radius a detector should report). */
export const BLOBS: Blob[] = [
  { x: 24, y: 24, r: 2, kind: "disc", amp: 90 },
  { x: 56, y: 24, r: 3, kind: "disc", amp: 80 },
  { x: 96, y: 26, r: 4, kind: "disc", amp: 70 },
  { x: 146, y: 30, r: 6, kind: "disc", amp: 90 },
  { x: 205, y: 36, r: 9, kind: "disc", amp: 60 },
  { x: 40, y: 82, r: 12, kind: "disc", amp: 80 },
  { x: 120, y: 92, r: 16, kind: "disc", amp: 50 },
  { x: 200, y: 110, r: 20, kind: "disc", amp: 70 },
  { x: 30, y: 150, r: 2 * Math.SQRT2, kind: "gauss", amp: 110 },
  { x: 72, y: 152, r: 4 * Math.SQRT2, kind: "gauss", amp: 90 },
  { x: 130, y: 160, r: 7 * Math.SQRT2, kind: "gauss", amp: 80 },
  // touching pair
  { x: 44, y: 215, r: 10, kind: "disc", amp: 70 },
  { x: 64, y: 215, r: 10, kind: "disc", amp: 70 },
  // elongated: semi-axes 26 × 5, horizontal
  { x: 140, y: 225, r: 5, b: 26, kind: "ellipse", amp: 80 },
  { x: 215, y: 200, r: 5, kind: "disc", amp: 40 },
];

export function makeBlobImage(kind: "synthetic", N = 256, noise = 6, seed = 1, background = true): { image: Grey; blobs: Blob[] } {
  const s = N / 256;
  const blobs = BLOBS.map(b => ({ ...b, x: b.x * s, y: b.y * s, r: b.r * s, b: b.b !== undefined ? b.b * s : undefined }));
  const img = fromFunction(N, N, (x, y) => {
    let v = background ? 30 + 40 * x / N + 45 * Math.exp(-((x - 0.7 * N) ** 2 + (y - 0.75 * N) ** 2) / (2 * (0.2 * N) ** 2)) : 40;
    for (const b of blobs) {
      const dx = x - b.x, dy = y - b.y;
      if (b.kind === "gauss") { const sg = b.r / Math.SQRT2; v += b.amp * Math.exp(-(dx * dx + dy * dy) / (2 * sg * sg)); }
      else if (b.kind === "ellipse") { if ((dx / b.b!) ** 2 + (dy / b.r) ** 2 < 1) v += b.amp; }
      else if (dx * dx + dy * dy < b.r * b.r) v += b.amp;
    }
    return v;
  });
  return { image: noise > 0 ? addGaussianNoise(img, noise, seed) : img, blobs };
}
