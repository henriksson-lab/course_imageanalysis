// Test images, an 8 × 8 block DCT and the drawing of a subband pyramid, shared by the
// dwt2d and wavedenoise demos (wavelets day). No DOM except drawPyramid / drawImage.
// The block DCT is a local stand-in until lib/dct.ts (owner: compression day) exists.
import { Band, subbandRect } from "../lib/wavelet";
import { diverging, divergingT } from "../lib/colour";

export const N = 256;
export type ImageName = "cell" | "disc" | "square" | "bars" | "checker" | "gradient" | "shapes";
export const IMAGE_LABELS: Record<ImageName, string> = {
  cell: "micrograph", disc: "disc", square: "square", bars: "bars", checker: "checker", gradient: "gradient", shapes: "shapes",
};

/** Antialiased synthetic image (4 × 4 supersampling), values 0..255. */
export function synthImage(name: Exclude<ImageName, "cell">, n = N): Float64Array {
  const out = new Float64Array(n * n), S = 4;
  const f = (x: number, y: number): number => {
    // x, y in pixel units, pixel centres at integers
    const u = x / n, v = y / n;
    switch (name) {
      case "disc": return (u - 0.52) ** 2 + (v - 0.47) ** 2 < 0.3 ** 2 ? 200 : 50;
      case "square": return x >= 77.5 && x < 178.5 && y >= 69.5 && y < 182.5 ? 200 : 50;
      case "bars": {
        if (x >= 40.5 && x < 61.5 && y >= 25.5 && y < 230.5) return 210;            // vertical bar
        if (y >= 40.5 && y < 61.5 && x >= 90.5 && x < 230.5) return 210;            // horizontal bar
        const d = (x - y - 30) / Math.SQRT2;                                       // diagonal bar
        if (Math.abs(d) < 11 && x > 100 && y > 80 && x < 235 && y < 235) return 210;
        return 40;
      }
      case "checker": return ((Math.floor((x + 0.5) / 32) + Math.floor((y + 0.5) / 32)) % 2) ? 200 : 50;
      case "gradient": return 128 + 90 * Math.sin(2 * Math.PI * (0.6 * u + 0.3 * v)) * Math.cos(2 * Math.PI * 0.5 * v);
      case "shapes": {
        let val = 60;
        if ((u - 0.3) ** 2 + (v - 0.32) ** 2 < 0.17 ** 2) val = 190;
        if (x >= 150.5 && x < 225.5 && y >= 40.5 && y < 120.5) val = 140;
        if (Math.abs(u - 0.62) + Math.abs(v - 0.7) < 0.2) val = 220;
        if ((u - 0.22) ** 2 / 0.12 ** 2 + (v - 0.75) ** 2 / 0.06 ** 2 < 1) val = 110;
        return val;
      }
    }
  };
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let s = 0;
    for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) s += f(x - 0.5 + (i + 0.5) / S, y - 0.5 + (j + 0.5) / S);
    out[y * n + x] = s / (S * S);
  }
  return out;
}

// ---- 8 × 8 block DCT (orthonormal DCT-II per block) ----

const B = 8;
const C8 = (() => {
  const c = new Float64Array(B * B);
  for (let k = 0; k < B; k++) for (let i = 0; i < B; i++) c[k * B + i] = (k === 0 ? Math.sqrt(1 / B) : Math.sqrt(2 / B)) * Math.cos(Math.PI * (2 * i + 1) * k / (2 * B));
  return c;
})();

/** Orthonormal 8 × 8 block DCT of an n × n image (n divisible by 8); coefficients stored per block in place of the pixels. */
export function blockDct(img: ArrayLike<number>, n: number, inverse = false): Float64Array {
  const out = new Float64Array(n * n), tmp = new Float64Array(B * B);
  for (let by = 0; by < n; by += B) for (let bx = 0; bx < n; bx += B) {
    // rows
    for (let y = 0; y < B; y++) for (let k = 0; k < B; k++) {
      let s = 0;
      for (let i = 0; i < B; i++) s += (inverse ? C8[i * B + k] : C8[k * B + i]) * img[(by + y) * n + bx + i];
      tmp[y * B + k] = s;
    }
    // columns
    for (let x = 0; x < B; x++) for (let k = 0; k < B; k++) {
      let s = 0;
      for (let i = 0; i < B; i++) s += (inverse ? C8[i * B + k] : C8[k * B + i]) * tmp[i * B + x];
      out[(by + k) * n + bx + x] = s;
    }
  }
  return out;
}

// ---- drawing ----

/** Draw a grey image (values lo..hi) into a canvas of n × n pixels. */
export function drawImage(cv: HTMLCanvasElement, img: ArrayLike<number>, n: number, lo = 0, hi = 255): void {
  if (cv.width !== n) { cv.width = n; cv.height = n; }
  const ctx = cv.getContext("2d")!, id = ctx.createImageData(n, n), k = 255 / (hi - lo || 1);
  for (let p = 0; p < n * n; p++) {
    const g = (img[p] - lo) * k;
    id.data[4 * p] = id.data[4 * p + 1] = id.data[4 * p + 2] = g; id.data[4 * p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

/** Draw a signed image with the diverging map (0 = white, ±range = full colour). */
export function drawSigned(cv: HTMLCanvasElement, img: ArrayLike<number>, n: number, range: number): void {
  if (cv.width !== n) { cv.width = n; cv.height = n; }
  const ctx = cv.getContext("2d")!, id = ctx.createImageData(n, n);
  for (let p = 0; p < n * n; p++) {
    const [r, g, b] = diverging(Math.max(0, Math.min(1, divergingT(img[p], range))));
    id.data[4 * p] = 255 * r; id.data[4 * p + 1] = 255 * g; id.data[4 * p + 2] = 255 * b; id.data[4 * p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

export type PyramidDisplay = "band" | "log";

/**
 * Draw a Mallat-layout coefficient image: LL in grey, detail bands with the diverging map,
 * each band normalised to its own largest |value| ("band"), or log10(1 + |w|) in grey for
 * all ("log"). Bands listed in `dropped` (keys "level:band") are dimmed; thin lines separate the bands.
 */
export function drawPyramid(cv: HTMLCanvasElement, c: ArrayLike<number>, n: number, levels: number, display: PyramidDisplay,
  dropped: Set<string> = new Set(), keptMask?: Uint8Array): void {
  if (cv.width !== n) { cv.width = n; cv.height = n; }
  const ctx = cv.getContext("2d")!, id = ctx.createImageData(n, n), d = id.data;
  const put = (p: number, r: number, g: number, b: number) => { d[4 * p] = r; d[4 * p + 1] = g; d[4 * p + 2] = b; d[4 * p + 3] = 255; };
  const bands: { level: number; band: Band }[] = [{ level: levels, band: "LL" }];
  for (let j = levels; j >= 1; j--) for (const b of ["V", "H", "D"] as Band[]) bands.push({ level: j, band: b });
  let logMax = 0;
  if (display === "log") for (let p = 0; p < n * n; p++) logMax = Math.max(logMax, Math.log10(1 + Math.abs(c[p])));
  for (const { level, band } of bands) {
    const r = subbandRect(level, band, n, n);
    let lo = Infinity, hi = -Infinity, mx = 0;
    for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
      const v = c[y * n + x]; lo = Math.min(lo, v); hi = Math.max(hi, v); mx = Math.max(mx, Math.abs(v));
    }
    const dim = dropped.has(`${level}:${band}`);
    for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
      const p = y * n + x, v = c[p];
      let R: number, G: number, Bc: number;
      if (display === "log") { const g = 255 * Math.log10(1 + Math.abs(v)) / (logMax || 1); R = G = Bc = g; }
      else if (band === "LL") { const g = 255 * (v - lo) / (hi - lo || 1); R = G = Bc = g; }
      else { const [a, b, e] = diverging(divergingT(v, mx || 1)); R = 255 * a; G = 255 * b; Bc = 255 * e; }
      if (dim || (keptMask && !keptMask[p])) { R = 0.25 * R + 0.75 * 90; G = 0.25 * G + 0.75 * 90; Bc = 0.25 * Bc + 0.75 * 90; }
      put(p, R, G, Bc);
    }
  }
  // separating lines
  for (let j = 1; j <= levels; j++) {
    const w = n >> j, h = n >> j;
    for (let t = 0; t < 2 * h; t++) put(t * n + w, 255, 200, 0);
    for (let t = 0; t < 2 * w; t++) put(h * n + t, 255, 200, 0);
  }
  ctx.putImageData(id, 0, 0);
}
