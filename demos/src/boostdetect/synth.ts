// Seeded synthetic data for the detection demos (boostdetect, haarexplore). No DOM.
//  - cartoon faces (head, hair, eyes, brows, nose, mouth, optional glasses) with random
//    pose, lighting and contrast; no photographs of real people;
//  - bright "cells" with a darker nucleus, random size and texture;
//  - background scenes (gradients, blobs, bars, debris) without targets;
//  - 24 × 24 training patches and 256 × 256 test scenes.
// Images are Float64Array, row-major, grey levels about 0..255.
import { rng } from "../lib/random";

export type Rng = ReturnType<typeof rng>;
export type Target = "faces" | "cells";
export const WIN = 24;

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

/** Paint an anti-aliased ellipse (centre, half-axes, angle) blending `value` with opacity `alpha`. */
export function ellipse(img: Float64Array, w: number, h: number, cx: number, cy: number, a: number, b: number,
  angle: number, value: number, alpha = 1): void {
  const R = Math.max(a, b) + 1.5, c = Math.cos(angle), s = Math.sin(angle), m = Math.min(a, b);
  const x0 = Math.max(0, Math.floor(cx - R)), x1 = Math.min(w - 1, Math.ceil(cx + R));
  const y0 = Math.max(0, Math.floor(cy - R)), y1 = Math.min(h - 1, Math.ceil(cy + R));
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const dx = x - cx, dy = y - cy, u = c * dx + s * dy, v = -s * dx + c * dy;
    const d = Math.sqrt((u / a) ** 2 + (v / b) ** 2);
    const cov = clamp((1 - d) * m + 0.5, 0, 1) * alpha;
    if (cov > 0) { const i = y * w + x; img[i] = img[i] * (1 - cov) + value * cov; }
  }
}

export interface FaceOptions { glasses?: boolean; angle?: number; skin?: number; hair?: number; light?: number }

/**
 * A cartoon face of height s centred at (cx, cy). The parts are placed relative to s and
 * rotated by `angle` (radians) about the centre.
 */
export function drawFace(img: Float64Array, w: number, h: number, cx: number, cy: number, s: number, o: FaceOptions = {}): void {
  const th = o.angle ?? 0, c = Math.cos(th), sn = Math.sin(th);
  const skin = o.skin ?? 185, hair = o.hair ?? 60;
  const P = (dx: number, dy: number): [number, number] => [cx + (c * dx - sn * dy) * s, cy + (sn * dx + c * dy) * s];
  const E = (dx: number, dy: number, a: number, b: number, v: number, al = 1) => { const [x, y] = P(dx, dy); ellipse(img, w, h, x, y, a * s, b * s, th, v, al); };
  E(0, -0.08, 0.42, 0.52, hair);           // hair
  E(0, 0.04, 0.37, 0.46, skin);            // head
  if (o.glasses) {
    for (const sx of [-1, 1]) { E(0.16 * sx, -0.07, 0.12, 0.09, 40); E(0.16 * sx, -0.07, 0.095, 0.068, skin + 15); }
    E(0, -0.08, 0.05, 0.016, 40);
  }
  for (const sx of [-1, 1]) {
    E(0.16 * sx, -0.07, 0.07, 0.04, skin - 120);  // eye
    E(0.16 * sx, -0.165, 0.09, 0.022, skin - 105); // brow
  }
  E(0, 0.07, 0.035, 0.085, skin - 35);      // nose
  E(0, 0.25, 0.13, 0.035, skin - 85);       // mouth
  if (o.light) {
    // lighting: brightness gradient across the face
    const R = 0.55 * s;
    for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(h - 1, cy + R); y++)
      for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(w - 1, cx + R); x++)
        img[y * w + x] *= 1 + o.light * (x - cx) / s;
  }
}

/** A bright cell of radius r with a darker nucleus and a little texture. */
export function drawCell(img: Float64Array, w: number, h: number, cx: number, cy: number, r: number, R: Rng, bright = 190): void {
  ellipse(img, w, h, cx, cy, r, r * (0.85 + 0.3 * R.unif()), R.unif() * Math.PI, bright);
  ellipse(img, w, h, cx + (R.unif() - 0.5) * 0.3 * r, cy + (R.unif() - 0.5) * 0.3 * r, 0.45 * r, 0.4 * r, R.unif() * Math.PI, bright - 70);
  for (let k = 0; k < 5; k++) {
    const a = R.unif() * 2 * Math.PI, d = (0.55 + 0.3 * R.unif()) * r;
    ellipse(img, w, h, cx + d * Math.cos(a), cy + d * Math.sin(a), 0.12 * r, 0.12 * r, 0, bright - 40, 0.7);
  }
}

/** A background without targets: gradient, blobs, bars and debris. */
export function background(w: number, h: number, R: Rng, target: Target, clutter = 1): Float64Array {
  const img = new Float64Array(w * h);
  const g0 = 60 + 100 * R.unif(), gx = (R.unif() - 0.5) * 80, gy = (R.unif() - 0.5) * 80;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = g0 + gx * x / w + gy * y / h;
  const n = Math.round(clutter * w * h / 900);
  for (let k = 0; k < n; k++) {
    const t = R.unif(), x = R.unif() * w, y = R.unif() * h, v = 20 + 220 * R.unif();
    if (target === "faces") {
      if (t < 0.4) ellipse(img, w, h, x, y, 2 + 18 * R.unif(), 2 + 18 * R.unif(), R.unif() * Math.PI, v, 0.9);
      else if (t < 0.75) ellipse(img, w, h, x, y, 3 + 25 * R.unif(), 1 + 2 * R.unif(), R.unif() * Math.PI, v);  // bars
      else { // dark pairs (eye-like) or ring-like clutter
        const d = 3 + 6 * R.unif();
        ellipse(img, w, h, x - d, y, 1.5 + 2 * R.unif(), 1 + R.unif(), 0, v * 0.4);
        ellipse(img, w, h, x + d, y, 1.5 + 2 * R.unif(), 1 + R.unif(), 0, v * 0.4);
      }
    } else {
      if (t < 0.35) ellipse(img, w, h, x, y, 1 + 3 * R.unif(), 1 + 3 * R.unif(), R.unif() * Math.PI, v);           // debris
      else if (t < 0.65) ellipse(img, w, h, x, y, 6 + 14 * R.unif(), 1.5 + 2 * R.unif(), R.unif() * Math.PI, v);   // elongated
      else if (t < 0.85) ellipse(img, w, h, x, y, 14 + 12 * R.unif(), 14 + 12 * R.unif(), 0, 60 + 80 * R.unif(), 0.6); // large dim
      else ellipse(img, w, h, x, y, 3 + 2 * R.unif(), 3 + 2 * R.unif(), 0, 200);                                     // small bright
    }
  }
  return img;
}

export function addNoise(img: Float64Array, R: Rng, sd: number): void {
  for (let i = 0; i < img.length; i++) img[i] = clamp(img[i] + sd * R.norm(), 0, 255);
}

/** Random contrast and brightness change about the mean. */
function jitterContrast(img: Float64Array, R: Rng): void {
  let m = 0;
  for (const v of img) m += v;
  m /= img.length;
  const c = 0.5 + 0.9 * R.unif(), b = (R.unif() - 0.5) * 60;
  for (let i = 0; i < img.length; i++) img[i] = clamp(m + b + c * (img[i] - m), 0, 255);
}

export interface PatchOptions { glasses?: "random" | "never" | "always"; angle?: number }

/** One positive 24 × 24 patch. */
export function positivePatch(R: Rng, target: Target, o: PatchOptions = {}): Float64Array {
  const img = background(WIN, WIN, R, target, 0.4);
  const cx = WIN / 2 + (R.unif() - 0.5) * 2, cy = WIN / 2 + (R.unif() - 0.5) * 2;
  if (target === "faces") {
    const glasses = o.glasses === "always" ? true : o.glasses === "never" ? false : R.unif() < 0.3;
    drawFace(img, WIN, WIN, cx, cy + 1, WIN * (1.0 + 0.15 * R.unif()), {
      glasses, angle: o.angle ?? (R.unif() - 0.5) * 0.12, skin: 140 + 80 * R.unif(), hair: 20 + 200 * R.unif() * R.unif(), light: (R.unif() - 0.5) * 0.6,
    });
  } else {
    drawCell(img, WIN, WIN, cx, cy, 7 + 3 * R.unif(), R, 150 + 90 * R.unif());
  }
  jitterContrast(img, R);
  addNoise(img, R, 3);
  return img;
}

/** One negative patch: a crop (24 to 72 pixels, resampled to 24) of a scene without targets. */
export function negativePatch(R: Rng, target: Target): Float64Array {
  const S = Math.round(WIN * (1 + 2 * R.unif()));
  const big = background(S, S, R, target, 1.2);
  if (target === "cells" && R.unif() < 0.35) {
    // a cell cut by the window border, or one far too small / large: hard negatives
    const t = R.unif(), r = S / WIN;
    if (t < 0.5) drawCell(big, S, S, (R.unif() < 0.5 ? 0 : S) + (R.unif() - 0.5) * 4 * r, S * R.unif(), (7 + 3 * R.unif()) * r, R);
    else drawCell(big, S, S, S / 2, S / 2, (t < 0.75 ? 2.5 : 16) * r, R);
  }
  if (target === "faces" && R.unif() < 0.15) {
    // half a face at the border (hard negative)
    drawFace(big, S, S, R.unif() < 0.5 ? 0 : S, S / 2, S * 1.05, { skin: 140 + 80 * R.unif() });
  }
  const out = new Float64Array(WIN * WIN), f = S / WIN;
  for (let y = 0; y < WIN; y++) for (let x = 0; x < WIN; x++) {
    let s = 0, n = 0;
    for (let yy = Math.floor(y * f); yy < Math.min(S, Math.ceil((y + 1) * f)); yy++)
      for (let xx = Math.floor(x * f); xx < Math.min(S, Math.ceil((x + 1) * f)); xx++) { s += big[yy * S + xx]; n++; }
    out[y * WIN + x] = s / n;
  }
  jitterContrast(out, R);
  addNoise(out, R, 3);
  return out;
}

export interface Dataset { patches: Float64Array[]; y: Int8Array; nPos: number }

/** A seeded training or test set: nPos positives first, then nNeg negatives. */
export function makeDataset(target: Target, nPos: number, nNeg: number, seed: number, o: PatchOptions = {}): Dataset {
  const R = rng(seed), patches: Float64Array[] = [];
  for (let i = 0; i < nPos; i++) patches.push(positivePatch(R, target, o));
  for (let i = 0; i < nNeg; i++) patches.push(negativePatch(R, target));
  const y = Int8Array.from({ length: nPos + nNeg }, (_, i) => (i < nPos ? 1 : -1));
  return { patches, y, nPos };
}

export interface SceneTarget { x: number; y: number; size: number; kind: string }
export interface SceneOptions { rotated?: boolean; glasses?: "random" | "never" | "always" }

/** A 256 × 256 test scene with targets at several sizes (sizes 24..48), and their boxes. */
export function testScene(target: Target, seed: number, o: SceneOptions = {}): { img: Float64Array; targets: SceneTarget[] } {
  const W = 256, R = rng(seed);
  const img = background(W, W, R, target, 0.9);
  const targets: SceneTarget[] = [];
  const spots: [number, number, number][] = target === "faces"
    ? [[52, 54, 44], [150, 46, 30], [222, 70, 26], [70, 168, 34], [168, 150, 48], [226, 214, 28], [118, 222, 26]]
    : [[40, 40, 26], [110, 34, 30], [196, 46, 24], [60, 120, 36], [150, 120, 28], [222, 128, 32], [46, 210, 30], [130, 200, 40], [214, 214, 26]];
  spots.forEach(([x, y, size], k) => {
    if (target === "faces") {
      const rot = o.rotated && k === 4;
      const glasses = o.glasses === "always" ? true : o.glasses === "never" ? false : k % 3 === 1;
      drawFace(img, W, W, x, y, size * 1.05, { glasses, angle: rot ? 0.6 : 0, skin: 150 + 60 * R.unif(), hair: 30 + 150 * R.unif(), light: (R.unif() - 0.5) * 0.4 });
      targets.push({ x: x - size / 2, y: y - size / 2, size, kind: rot ? "rotated" : glasses ? "glasses" : "face" });
    } else {
      drawCell(img, W, W, x, y, size * 0.36, R, 160 + 70 * R.unif());
      targets.push({ x: x - size / 2, y: y - size / 2, size, kind: "cell" });
    }
  });
  addNoise(img, R, 3);
  return { img, targets };
}
