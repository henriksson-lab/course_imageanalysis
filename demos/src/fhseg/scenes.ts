// Synthetic test scenes of the graph-based segmentation day, with ground truth (no DOM).
// Shared by randwalk, fhseg, mergetree and slic (and the test). All seeded.
import { rng } from "../lib/random";
import { nuclei } from "../lib/synth";

export interface Scene { w: number; h: number; I: Float64Array; truth: Int32Array; rgb?: [Float64Array, Float64Array, Float64Array] }

function smoothstep(t: number): number { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); }

function addNoise(I: Float64Array, sd: number, seed: number): Float64Array {
  const r = rng(seed), out = new Float64Array(I.length);
  for (let i = 0; i < I.length; i++) out[i] = Math.min(255, Math.max(0, I[i] + sd * r.norm()));
  return out;
}

/** 3 × 3 box blur (once), to soften synthetic edges a little. */
function soften(I: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(I.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0, c = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      s += I[yy * w + xx]; c++;
    }
    out[y * w + x] = s / c;
  }
  return out;
}

/** Random-walker scenes, 128 × 128.
 *  objects: a bright disc (180) with a thin arm (3 px wide) to the right, and a second disc (160)
 *           touching it below with a weak boundary; background 60. truth 1 = upper disc + arm, 2 = lower disc.
 *  gap:     a dark wall (20) splits a mid-grey field (110) into left and right halves, with a 3-px gap.
 *  ramp:    left 80, right 170, joined by a smooth ramp 40 px wide: different brightness, no edge. */
export function walkerScene(kind: "objects" | "gap" | "ramp", noise: number, seed = 3): Scene {
  const w = 128, h = 128, I = new Float64Array(w * h), truth = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (kind === "objects") {
      let v = 60, t = 0;
      const inLow = (x - 52) ** 2 + (y - 88) ** 2 < 22 ** 2;
      const inUp = (x - 52) ** 2 + (y - 46) ** 2 < 24 ** 2;
      const inArm = x >= 70 && x < 116 && Math.abs(y - 40) <= 1;
      if (inLow) { v = 160; t = 2; }
      if (inUp || inArm) { v = 180; t = 1; }
      I[p] = v; truth[p] = t;
    } else if (kind === "gap") {
      const wall = Math.abs(x - 64) <= 1 && !(y >= 62 && y <= 64);
      I[p] = wall ? 20 : 110;
      truth[p] = x < 64 ? 1 : x > 64 ? 2 : (y < 64 ? 1 : 2);
    } else {
      I[p] = 80 + 90 * smoothstep((x - 44) / 40);
      truth[p] = x < 64 ? 1 : 2;
    }
  }
  return { w, h, I: addNoise(kind === "ramp" ? soften(I, w, h) : I, noise, seed), truth };
}

/** FH / superpixel test scene, 256 × 256 grey, truth labels 1..8:
 *  1 background (flat 70) with five small bright spots (8, radius 2.5, value 230);
 *  2 a textured disc (150 ± 40, smooth bumps: neighbour differences up to about 20);
 *  3 and 4 two flat squares (120 and 140: a faint step of 20 between them);
 *  5 a horizontal ramp 40 → 200 (bottom band; it equals the background near x = 58);
 *  6 and 7 two bright halves (200) of a square, split by a dark line (70) with a 3-px gap. */
export function fhScene(noise: number, seed = 5): Scene {
  const w = 256, h = 256, I = new Float64Array(w * h), truth = new Int32Array(w * h);
  const spots = [[30, 150], [110, 140], [240, 140], [20, 20], [125, 175]];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    let v = 70, t = 1;
    if ((x - 70) ** 2 + (y - 70) ** 2 < 46 ** 2) { v = 150 + 40 * Math.sin(0.5 * x) * Math.sin(0.5 * y); t = 2; }
    else if (x >= 136 && x < 186 && y >= 24 && y < 116) { v = 120; t = 3; }
    else if (x >= 186 && x < 236 && y >= 24 && y < 116) { v = 140; t = 4; }
    else if (y >= 196 && y < 240 && x >= 16 && x < 240) { v = 40 + 160 * (x - 16) / 224; t = 5; }
    else if (x >= 150 && x < 230 && y >= 132 && y < 182) {
      if (x >= 189 && x <= 190 && (y < 156 || y > 158)) { v = 70; t = 1; } else { v = 200; t = x < 190 ? 6 : 7; }
    } else for (const [sx, sy] of spots) if ((x - sx) ** 2 + (y - sy) ** 2 <= 6.25) { v = 230; t = 8; }
    I[p] = v; truth[p] = t;
  }
  return { w, h, I: addNoise(I, noise, seed), truth };
}

/** Colour scene, 256 × 256 RGB 0..255 with truth 1..6: blobs of different colours, two of them
 *  with the same lightness (only hue separates them), one textured. I = luminance-like grey. */
export function colourScene(noise: number, seed = 7): Scene {
  const w = 256, h = 256, n = w * h;
  const R = new Float64Array(n), G = new Float64Array(n), B = new Float64Array(n), truth = new Int32Array(n);
  const cols: [number, number, number][] = [[225, 220, 205], [200, 60, 50], [60, 140, 60], [70, 90, 190], [210, 170, 40], [150, 80, 160]];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    let t = 1;
    if ((x - 75) ** 2 / 2500 + (y - 80) ** 2 / 1600 < 1) t = 2;
    else if ((x - 180) ** 2 + (y - 70) ** 2 < 45 ** 2) t = 3;
    else if ((x - 170) ** 2 / 3600 + (y - 185) ** 2 / 1225 < 1) t = 4;
    else if (x > 30 && x < 110 && y > 150 && y < 230 && (x - 70) ** 2 + (y - 190) ** 2 > 18 ** 2) t = 5;
    else if ((x - 70) ** 2 + (y - 190) ** 2 <= 18 ** 2) t = 6;
    truth[p] = t;
    const c = cols[t - 1], tex = t === 3 ? 25 * Math.sin(x * 0.9) * Math.sin(y * 0.9) : 0;
    R[p] = c[0] + tex; G[p] = c[1] + tex; B[p] = c[2] + tex;
  }
  const r = rng(seed);
  for (let p = 0; p < n; p++) {
    R[p] = Math.min(255, Math.max(0, R[p] + noise * r.norm()));
    G[p] = Math.min(255, Math.max(0, G[p] + noise * r.norm()));
    B[p] = Math.min(255, Math.max(0, B[p] + noise * r.norm()));
  }
  const I = new Float64Array(n);
  for (let p = 0; p < n; p++) I[p] = 0.299 * R[p] + 0.587 * G[p] + 0.114 * B[p];
  return { w, h, I, truth, rgb: [R, G, B] };
}

/** Nuclei scene for hierarchies, 256 × 256: touching nuclei (lib/synth), truth = nucleus labels (0 = background). */
export function nucleiScene(noise: number, seed = 4): Scene {
  const s = nuclei({ width: 256, height: 256, count: 18, radius: [10, 17], touching: 0.45, noise, seed, texture: 8 });
  return { w: 256, h: 256, I: Float64Array.from(s.image.data), truth: Int32Array.from(s.labels) };
}

/** A bright disc (190, truth 1) whose boundary with the background (70) has one weak spot (a
 *  gentle ramp over a 30° arc), and a darker disc (110, truth 2) with a sharp boundary. 256 × 256. */
export function leakScene(noise: number, seed = 6): Scene {
  const w = 256, h = 256, I = new Float64Array(w * h), truth = new Int32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    const ra = Math.hypot(x - 80, y - 128), ang = Math.abs(Math.atan2(y - 128, x - 80));
    const rb = Math.hypot(x - 192, y - 128);
    let v = 70, t = 0;
    if (ang < Math.PI / 12 && ra > 45 && ra < 75) v = 190 - 120 * (ra - 45) / 30;
    else if (ra <= 50) { v = 190; t = 1; }
    if (ra <= 50) t = 1;
    if (rb <= 40) { v = 110; t = 2; }
    I[p] = v; truth[p] = t;
  }
  return { w, h, I: addNoise(soften(I, w, h), noise, seed), truth };
}
