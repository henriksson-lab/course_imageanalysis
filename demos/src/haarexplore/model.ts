// Presets, the cartoon-face scene and the class-painting / one-threshold logic of the
// haarexplore demo (no DOM, so the exercise numbers can be computed in node).
import { rng } from "../lib/random";
import { background, drawFace } from "../boostdetect/synth";

export const N = 256;
export const SCALES = [1, 2, 4, 8, 16];

export type ImageId = "face" | "cell" | "cells";

/** A 256 × 256 scene with one large cartoon face (seeded). */
export function faceScene(): Float64Array {
  const R = rng(3), img = background(N, N, R, "faces", 0.6);
  drawFace(img, N, N, 128, 132, 150, { skin: 185, hair: 60, light: 0.15 });
  for (let i = 0; i < img.length; i++) img[i] = Math.max(0, Math.min(255, img[i] + 4 * R.norm()));
  return img;
}

export interface TPreset { id: string; label: string; image: ImageId; tpl: string; x: number; y: number; sx: number; sy: number; map: boolean }
export const T_PRESETS: TPreset[] = [
  { id: "eyes", label: "eyes are darker than cheeks", image: "face", tpl: "edge-y", x: 83, y: 106, sx: 90, sy: 15, map: false },
  { id: "nose", label: "eyes darker than the bridge of the nose", image: "face", tpl: "line-x", x: 89, y: 108, sx: 26, sy: 20, map: false },
  { id: "edge", label: "vertical edge on a cell boundary", image: "cell", tpl: "edge-x", x: 88, y: 96, sx: 8, sy: 24, map: true },
  { id: "blobs", label: "centre − surround finds cells of its size", image: "cells", tpl: "centre", x: 36, y: 96, sx: 16, sy: 16, map: true },
];

export interface SPreset { id: string; label: string; image: ImageId; fx: number; fy: number; a: [number, number][][]; b: [number, number][][] }
export const S_PRESETS: SPreset[] = [
  {
    id: "nucleus", label: "nucleus or cytoplasm?", image: "cell", fx: 0 * 5 + 2, fy: 1 * 5 + 1,
    a: [[[128, 86], [158, 112], [140, 150]], [[118, 120], [170, 128]]],
    b: [[[70, 18], [88, 40]], [[136, 186], [146, 212]], [[86, 70], [94, 96]]],
  },
  {
    id: "cells", label: "cell or background?", image: "cells", fx: 2 * 5 + 2, fy: 1 * 5 + 1,
    a: [[[36, 36], [44, 44]], [[106, 30], [114, 38]], [[56, 116], [64, 124]], [[146, 116], [154, 124]], [[126, 196], [134, 204]]],
    b: [[[70, 70], [100, 80]], [[180, 80], [200, 90]], [[90, 160], [110, 170]], [[170, 230], [200, 236]]],
  },
];


/** Paint the strokes of a stack preset into a label image (1 = A, 2 = B), brush radius 3. */
export function dab(labels: Uint8Array, x: number, y: number, lab: number): void {
  const r = 3;
  for (let yy = Math.max(0, Math.round(y - r)); yy <= Math.min(N - 1, Math.round(y + r)); yy++)
    for (let xx = Math.max(0, Math.round(x - r)); xx <= Math.min(N - 1, Math.round(x + r)); xx++)
      if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) labels[yy * N + xx] = lab;
}

export function paintStrokes(labels: Uint8Array, strokes: [number, number][][], lab: number): void {
  for (const s of strokes) for (let k = 0; k < s.length; k++) {
    const [x0, y0] = s[Math.max(0, k - 1)], [x1, y1] = s[k];
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
    for (let j = 0; j <= n; j++) dab(labels, x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * j / n, lab);
  }
}

/** Counts of A and B left/right of a threshold on one feature map, and the share one threshold gets right. */
export function separation(X: ArrayLike<number>, labels: Uint8Array, thr: number): { aL: number; aR: number; bL: number; bR: number; share: number } {
  let aL = 0, aR = 0, bL = 0, bR = 0;
  for (let i = 0; i < labels.length; i++) {
    if (!labels[i]) continue;
    const left = X[i] < thr;
    if (labels[i] === 1) { if (left) aL++; else aR++; } else { if (left) bL++; else bR++; }
  }
  const n = aL + aR + bL + bR;
  return { aL, aR, bL, bR, share: n ? Math.max(aL + bR, aR + bL) / n : 0 };
}

/** Best share over all thresholds (the stump of the viola-jones unit, on one feature). */
export function bestSeparation(X: ArrayLike<number>, labels: Uint8Array): { thr: number; share: number } {
  const v: [number, number][] = [];
  for (let i = 0; i < labels.length; i++) if (labels[i]) v.push([X[i], labels[i]]);
  v.sort((a, b) => a[0] - b[0]);
  const nA = v.filter(p => p[1] === 1).length, n = v.length;
  let aL = 0, bL = 0, best = { thr: v.length ? v[0][0] - 1 : 0, share: Math.max(nA, n - nA) / Math.max(1, n) };
  for (let k = 0; k < n; k++) {
    if (v[k][1] === 1) aL++; else bL++;
    if (k + 1 < n && v[k + 1][0] === v[k][0]) continue;
    const share = Math.max(aL + (n - nA - bL), (nA - aL) + bL) / n;
    if (share > best.share) best = { thr: k + 1 < n ? 0.5 * (v[k][0] + v[k + 1][0]) : v[k][0] + 1, share };
  }
  return best;
}
