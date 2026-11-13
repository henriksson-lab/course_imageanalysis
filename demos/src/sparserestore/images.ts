// Test images and masks for the sparse restoration demos (sparserestore, patchdict); no DOM except
// textMask. Values 0..255.
import { Vec } from "../lib/sparse";
import { rng } from "../lib/random";

/** Piecewise-constant shapes: a disc, a rectangle, a triangle and a ring on a grey background, plus a
 *  small faint square (contrast 12) and a thin line. `shift` moves the scene (for a training image). */
export function shapesImage(n: number, shift = 0): Vec {
  const im = new Float64Array(n * n), s = n / 128;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = (x + shift) / s, v = (y + shift * 0.7) / s;
    let val = 70;
    if ((u - 40) ** 2 + (v - 42) ** 2 < 22 ** 2) val = 190;
    if (u > 72 && u < 116 && v > 20 && v < 56) val = 130;
    if (v > 70 && v < 118 && u > 14 && u < 64 && (u - 14) > (118 - v) * 0.5 && (64 - u) > (118 - v) * 0.5) val = 220;
    const r = Math.hypot(u - 92, v - 92);
    if (r > 14 && r < 24) val = 30;
    if (u > 86 && u < 98 && v > 86 && v < 98) val = 82;     // faint square inside the ring (contrast 12)
    if (Math.abs(u - v - 2) < 0.8 && u > 66 && u < 124 && v < 66) val = 240;   // thin diagonal line
    im[y * n + x] = val;
  }
  return im;
}

/** Texture: oriented stripes of several frequencies in four quadrants, low contrast, plus a disc. */
export function textureImage(n: number, shift = 0): Vec {
  const im = new Float64Array(n * n), s = n / 128;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = (x + shift) / s, v = (y + shift) / s;
    const q = (u < 64 ? 0 : 1) + (v < 64 ? 0 : 2);
    const ang = [0.3, 1.2, 2.0, 2.7][q], f = [0.25, 0.35, 0.5, 0.7][q];
    let val = 120 + 30 * Math.sin(f * (u * Math.cos(ang) + v * Math.sin(ang)));
    if ((u - 64) ** 2 + (v - 64) ** 2 < 16 ** 2) val = 200;
    im[y * n + x] = val;
  }
  return im;
}

/** Random missing pixels: 1 = known. */
export function randomMask(n: number, missing: number, seed = 3): Uint8Array {
  const r = rng(seed), m = new Uint8Array(n * n);
  for (let i = 0; i < m.length; i++) m[i] = r.unif() < missing ? 0 : 1;
  return m;
}

/** Thin scratches (2 px wide curves) plus one large square hole (n/8 wide). */
export function scratchMask(n: number): Uint8Array {
  const m = new Uint8Array(n * n).fill(1), s = n / 128;
  const curves = [(t: number) => [10 + 108 * t, 30 + 60 * t + 10 * Math.sin(6 * t)], (t: number) => [20 + 90 * t, 110 - 80 * t * t], (t: number) => [64 + 20 * Math.sin(3 * t), 5 + 115 * t]];
  for (const c of curves) for (let k = 0; k <= 2000; k++) {
    const [u, v] = c(k / 2000);
    for (let dy = -1; dy <= 0; dy++) for (let dx = -1; dx <= 0; dx++) {
      const x = Math.round(u * s) + dx, y = Math.round(v * s) + dy;
      if (x >= 0 && y >= 0 && x < n && y < n) m[y * n + x] = 0;
    }
  }
  const h0 = Math.round(84 * s), w = Math.round(16 * s);
  for (let y = h0; y < h0 + w; y++) for (let x = Math.round(20 * s); x < Math.round(20 * s) + w; x++) m[y * n + x] = 0;
  return m;
}

/** Text overlay mask ("SPARSE 2026" in two lines, bold), rendered with a canvas: 1 = known. */
export function textMask(n: number): Uint8Array {
  const c = document.createElement("canvas");
  c.width = n; c.height = n;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, n, n);
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${Math.round(n * 0.16)}px sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("SPARSE", n / 2, n * 0.33);
  ctx.fillText("PRIORS", n / 2, n * 0.7);
  const d = ctx.getImageData(0, 0, n, n).data, m = new Uint8Array(n * n);
  for (let i = 0; i < n * n; i++) m[i] = d[4 * i] > 60 ? 0 : 1;
  return m;
}
