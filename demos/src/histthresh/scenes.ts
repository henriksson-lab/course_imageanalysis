// Synthetic scenes with ground truth for the thresholding demos (histthresh, mixture,
// localthresh, threshroc), plus drawing helpers. No DOM except drawOverlay.
// (Generators belong in lib/synth.ts; kept here because that file is owned elsewhere.)
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { rng } from "../lib/random";
import { discs } from "../lib/synth";

/** Round and clip to 8-bit grey levels 0..255 (a new image). */
export function quantise8(img: Grey): Grey {
  const out = createGrey(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = Math.max(0, Math.min(255, Math.round(img.data[i])));
  return out;
}

/** Value below which a fraction q of the entries lie (sorts a copy). */
function quantile(a: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(a).sort();
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * s.length)))];
}

/**
 * Blob-shaped truth mask with an exact object fraction: smooth random noise (Gaussian
 * blur of scale `scale`) thresholded at its (1 − fraction) quantile.
 */
export function blobMask(w: number, h: number, fraction: number, scale: number, seed: number): Uint8Array {
  const r = rng(seed), f = createGrey(w, h);
  for (let i = 0; i < f.data.length; i++) f.data[i] = r.norm();
  const g = gaussianBlur(f, scale, "wrap");
  const thr = quantile(g.data, 1 - fraction);
  return Uint8Array.from(g.data, v => (v > thr ? 1 : 0));
}

export interface SceneOptions {
  size?: number;          // width = height (default 256)
  background?: number;    // grey level of the background (default 60)
  contrast?: number;      // object − background (default 100)
  noise?: number;         // Gaussian noise sd (default 8)
  edgeBlur?: number;      // optical blur σ of the edges: partial-volume pixels (default 1)
  gradient?: number;      // illumination: multiplicative ramp from 1 − g/2 (left) to 1 + g/2 (right) (default 0)
  vignette?: number;      // illumination: darkening towards the corners, 0..1 (default 0)
  seed?: number;
}

/** Illumination field: left-to-right ramp times a radial vignette. */
export function illumination(w: number, h: number, gradient: number, vignette: number): Grey {
  const L = createGrey(w, h), cx = (w - 1) / 2, cy = (h - 1) / 2, rmax2 = cx * cx + cy * cy;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const ramp = 1 - gradient / 2 + gradient * x / Math.max(1, w - 1);
    const vig = 1 - vignette * ((x - cx) ** 2 + (y - cy) ** 2) / rmax2;
    L.data[y * w + x] = ramp * vig;
  }
  return L;
}

/**
 * Render a truth mask as an image: (background + contrast · blurred mask) · illumination
 * + Gaussian noise, rounded to 8 bits. The noise uses its own seed, so changing the
 * noise level keeps the same noise pattern (scaled).
 */
export function renderScene(truth: Uint8Array, o: SceneOptions = {}): Grey {
  const n = o.size ?? 256, bg = o.background ?? 60, c = o.contrast ?? 100, sd = o.noise ?? 8;
  let soft: Grey = { width: n, height: n, data: Float32Array.from(truth) };
  if ((o.edgeBlur ?? 1) > 0) soft = gaussianBlur(soft, o.edgeBlur ?? 1);
  const L = illumination(n, n, o.gradient ?? 0, o.vignette ?? 0);
  const r = rng((o.seed ?? 1) * 7919 + 13), img = createGrey(n, n);
  for (let i = 0; i < img.data.length; i++) img.data[i] = (bg + c * soft.data[i]) * L.data[i] + sd * r.norm();
  return quantise8(img);
}

/** Non-overlapping discs of several sizes (radii in [rMin, rMax]); truth mask. */
export function discMask(n: number, count: number, rMin: number, rMax: number, seed: number): Uint8Array {
  const r = rng(seed), list: { x: number; y: number; r: number }[] = [];
  for (let tries = 0; list.length < count && tries < 20000; tries++) {
    // larger discs first, so they find room
    const k = list.length / Math.max(1, count - 1);
    const rad = rMax - (rMax - rMin) * k * (0.85 + 0.15 * r.unif());
    const x = rad + 3 + (n - 2 * rad - 6) * r.unif(), y = rad + 3 + (n - 2 * rad - 6) * r.unif();
    if (list.some(d => Math.hypot(d.x - x, d.y - y) < d.r + rad + 4)) continue;
    list.push({ x, y, r: rad });
  }
  const d = discs(n, n, list);
  return Uint8Array.from(d.labels, v => (v > 0 ? 1 : 0));
}

/** RGB colour of an overlay layer. */
export type RGB = [number, number, number];

/**
 * DOM: draw a grey image (0..255) into a canvas, tinting the pixels of each layer's mask
 * with its colour (alpha 0..1), and optionally outlining a mask in a colour.
 */
export function drawOverlay(canvas: HTMLCanvasElement, img: Grey, layers: { mask: ArrayLike<number>; rgb: RGB; alpha?: number }[] = [],
  outline?: { mask: ArrayLike<number>; rgb: RGB }): void {
  const w = img.width, h = img.height;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d")!, id = ctx.createImageData(w, h), d = id.data;
  for (let i = 0; i < w * h; i++) {
    const g = Math.max(0, Math.min(255, img.data[i]));
    let r = g, gg = g, b = g;
    for (const L of layers) if (L.mask[i]) {
      const a = L.alpha ?? 0.5;
      r = (1 - a) * r + a * L.rgb[0]; gg = (1 - a) * gg + a * L.rgb[1]; b = (1 - a) * b + a * L.rgb[2];
    }
    if (outline && outline.mask[i]) {
      const x = i % w, y = (i - x) / w, m = outline.mask;
      if ((x > 0 && !m[i - 1]) || (x < w - 1 && !m[i + 1]) || (y > 0 && !m[i - w]) || (y < h - 1 && !m[i + w])) [r, gg, b] = outline.rgb;
    }
    d[4 * i] = r; d[4 * i + 1] = gg; d[4 * i + 2] = b; d[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

/** Inject a demo's CSS once. */
export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}

/** Make a whole SVG plot a drag area that reports the data x of the pointer. */
export function dragX(svg: SVGSVGElement, toX: (ev: PointerEvent) => number, onX: (x: number) => void): void {
  svg.style.touchAction = "none";
  svg.style.cursor = "ew-resize";
  let down = false;
  svg.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; svg.setPointerCapture(ev.pointerId); onX(toX(ev)); });
  svg.addEventListener("pointermove", ev => { if (down) onX(toX(ev)); });
  const up = () => { down = false; };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);
}
