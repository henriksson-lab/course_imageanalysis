// Seeded synthetic colour scenes with ground truth, for the colour day's demos
// (colourcloud, colourseg, colourmatrix). No DOM. Every scene is planar LINEAR RGB in [0, 1]
// (light), W x H, row-major; demos encode to sRGB only when drawing. Captions must say
// "synthetic".
//
//  objects : shaded coloured objects on a table, with a cast shadow, a grey card, a dark
//            object and a specular highlight; truth = object classes.
//  he      : brightfield H&E-like tissue, rendered by Beer–Lambert from two stain
//            concentration maps (hematoxylin in nuclei, eosin in stroma); truth = nuclei.
//  chart   : every hue at every saturation and value (fills the RGB gamut).
//  isolum  : red text on a green background of the same luminance.
import { rng } from "../lib/random";
import { srgbToLinear, linearToSrgb } from "../lib/colour";

export interface Scene {
  w: number; h: number;
  r: Float32Array; g: Float32Array; b: Float32Array;  // linear light, [0, 1]
  truth?: Uint8Array;                                   // class per pixel (0 = background)
  classes?: string[];                                   // class names, index = class
  credit: string;
}

export type SceneId = "objects" | "he" | "chart" | "isolum";
export const SCENE_LABELS: Record<SceneId, string> = {
  objects: "objects in a shadow", he: "stained tissue (H&E-like)", chart: "all hues", isolum: "isoluminant text",
};

const enc = linearToSrgb, dec = srgbToLinear;
const blank = (w: number, h: number): Scene => ({ w, h, r: new Float32Array(w * h), g: new Float32Array(w * h), b: new Float32Array(w * h), credit: "" });
const smooth = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/** Object classes of the "objects" scene. */
export const OBJECT_CLASSES = ["table", "red apple", "green leaf", "yellow fruit", "blue cup", "grey card", "dark stone"];

export function objectsScene(n = 160): Scene {
  const S = blank(n, n), truth = new Uint8Array(n * n);
  const k = n / 160;
  // reflectances (linear), from display colours
  const R = (hex: number[]) => hex.map(v => dec(v / 255));
  const refl: number[][] = [R([196, 184, 164]), R([200, 28, 30]), R([70, 150, 50]), R([235, 200, 40]), R([40, 80, 190]), R([118, 118, 118]), R([42, 40, 38])];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = x / k, v = y / k, p = y * n + x;
    let cls = 0, shade = 1, spec = 0;
    // objects (later ones on top)
    const dA = Math.hypot(u - 52, v - 62) / 30;
    if (dA < 1) { cls = 1; const nz = Math.sqrt(1 - dA * dA); const nx = (u - 52) / 30, ny = (v - 62) / 30; shade = 0.25 + 0.75 * Math.max(0, -0.5 * nx - 0.55 * ny + 0.67 * nz); spec = Math.max(0, -0.5 * nx - 0.55 * ny + 0.67 * nz) ** 40; }
    const eL = ((u - 118) * Math.cos(0.5) + (v - 42) * Math.sin(0.5)) ** 2 / 30 ** 2 + (-(u - 118) * Math.sin(0.5) + (v - 42) * Math.cos(0.5)) ** 2 / 13 ** 2;
    if (eL < 1) { cls = 2; shade = 0.85 + 0.15 * Math.sin(u * 0.4); }
    const eY = (u - 112) ** 2 / 26 ** 2 + (v - 118) ** 2 / 17 ** 2;
    if (eY < 1) { cls = 3; shade = 0.45 + 0.55 * Math.sqrt(1 - eY); }
    if (u > 18 && u < 58 && v > 112 && v < 150) { cls = 4; shade = 0.55 + 0.45 * Math.cos((u - 38) / 20 * 1.3); }
    if (u > 132 && u < 154 && v > 70 && v < 96) { cls = 5; shade = 1; }
    if (Math.hypot(u - 80, v - 14) < 10) { cls = 6; shade = 1; }
    // illumination: a gradient from the left, and a soft cast shadow over the lower left
    let light = 1.0 - 0.35 * (u / 160);
    const sh = smooth((u + 0.55 * v - 118) / 8);          // 1 outside the shadow
    const shV = smooth((v - 70) / 8);                      // shadow only below y ~ 70
    light *= 1 - 0.72 * shV * (1 - sh);
    const c = refl[cls];
    S.r[p] = Math.min(1, c[0] * shade * light + spec * light);
    S.g[p] = Math.min(1, c[1] * shade * light + spec * light);
    S.b[p] = Math.min(1, c[2] * shade * light + spec * light);
    truth[p] = cls;
  }
  S.truth = truth; S.classes = OBJECT_CLASSES;
  S.credit = "Synthetic scene (seeded): shaded objects, a cast shadow over the lower left, a grey card.";
  return S;
}

/** Ruifrok–Johnston optical-density vectors of hematoxylin and eosin (R, G, B). */
export const OD_H = [0.65, 0.70, 0.29], OD_E = [0.07, 0.99, 0.11];

export function heScene(n = 160, seed = 7): Scene {
  const S = blank(n, n), truth = new Uint8Array(n * n), r = rng(seed);
  const cH = new Float32Array(n * n), cE = new Float32Array(n * n);
  // stroma: eosin with smooth variation; two lumens (white space)
  const waves = Array.from({ length: 5 }, () => [r.unif() * 0.08, r.unif() * 0.08, r.unif() * 6]);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let e = 0.42;
    for (const [a, b, ph] of waves) e += 0.06 * Math.sin(a * x + b * y + ph);
    const l1 = Math.hypot(x - 0.28 * n, y - 0.72 * n) / (0.15 * n), l2 = ((x - 0.78 * n) / (0.14 * n)) ** 2 + ((y - 0.25 * n) / (0.09 * n)) ** 2;
    const lumen = Math.max(1 - smooth((l1 - 0.85) / 0.3), 1 - smooth((Math.sqrt(l2) - 0.85) / 0.3));
    cE[y * n + x] = Math.max(0, e * (1 - lumen));
  }
  // nuclei: seeded ellipses with dense, slightly textured hematoxylin
  for (let k = 0; k < 34; k++) {
    const cx = r.unif() * n, cy = r.unif() * n, a = (4 + 3 * r.unif()) * n / 160, b = a * (0.6 + 0.35 * r.unif()), th = r.unif() * Math.PI, dens = 0.75 + 0.45 * r.unif();
    if (cE[Math.floor(cy) * n + Math.floor(cx)] < 0.15) continue; // not in a lumen
    for (let y = Math.max(0, Math.floor(cy - a - 2)); y < Math.min(n, cy + a + 2); y++) for (let x = Math.max(0, Math.floor(cx - a - 2)); x < Math.min(n, cx + a + 2); x++) {
      const dx = x - cx, dy = y - cy, u = dx * Math.cos(th) + dy * Math.sin(th), v = -dx * Math.sin(th) + dy * Math.cos(th);
      const d = Math.sqrt((u / a) ** 2 + (v / b) ** 2);
      if (d < 1.15) {
        const w = 1 - smooth((d - 0.85) / 0.3), p = y * n + x;
        cH[p] = Math.max(cH[p], w * dens * (0.85 + 0.15 * Math.sin(1.7 * x + 2.3 * y)));
        if (d < 1) truth[p] = 1;
      }
    }
  }
  for (let p = 0; p < n * n; p++) {
    const od = [0, 1, 2].map(c => cH[p] * OD_H[c] + cE[p] * OD_E[c]);
    S.r[p] = 0.94 * 10 ** -od[0]; S.g[p] = 0.92 * 10 ** -od[1]; S.b[p] = 0.95 * 10 ** -od[2];
  }
  S.truth = truth; S.classes = ["stroma / space", "nucleus"];
  S.credit = "Synthetic H&E-like image (seeded), rendered by Beer–Lambert from two stain maps.";
  return S;
}

export function chartScene(n = 160): Scene {
  const S = blank(n, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const hue = 360 * (x + 0.5) / n, t = (y + 0.5) / n;
    // top half: saturation 0 -> 1 at value 1; bottom half: value 1 -> 0 at saturation 1
    const s = t < 0.5 ? t / 0.5 : 1, v = t < 0.5 ? 1 : 1 - (t - 0.5) / 0.5;
    const c = hsv(hue, s, v), p = y * n + x;
    S.r[p] = dec(c[0]); S.g[p] = dec(c[1]); S.b[p] = dec(c[2]);
  }
  S.credit = "Synthetic chart: hue across, saturation (top half) and value (bottom half) down.";
  return S;
}

export function isolumScene(n = 160): Scene {
  const S = blank(n, n), truth = new Uint8Array(n * n);
  // equal luminance: 0.2126 R_lin = 0.7152 G_lin with G_lin = 0.25
  const gLin = 0.25, rLin = 0.7152 * gLin / 0.2126;
  // a block letter "A" and a bar, as a mask
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = x / n, v = y / n;
    const leg1 = Math.abs((u - 0.5) - (v - 0.15) * -0.32) < 0.06 && v > 0.15 && v < 0.85;
    const leg2 = Math.abs((u - 0.5) - (v - 0.15) * 0.32) < 0.06 && v > 0.15 && v < 0.85;
    const bar = v > 0.55 && v < 0.65 && Math.abs(u - 0.5) < 0.17;
    const on = leg1 || leg2 || bar;
    const p = y * n + x;
    truth[p] = on ? 1 : 0;
    S.r[p] = on ? rLin : 0; S.g[p] = on ? 0 : gLin; S.b[p] = 0;
  }
  S.truth = truth; S.classes = ["green", "red letter"];
  S.credit = "Synthetic test: red letter on green, both with the same luminance Y.";
  return S;
}

export function makeScene(id: SceneId, n = 160): Scene {
  return id === "objects" ? objectsScene(n) : id === "he" ? heScene(n) : id === "chart" ? chartScene(n) : isolumScene(n);
}

/** Add sensor-like Gaussian noise of standard deviation `sd` to the ENCODED values (as in
 *  an 8-bit image), returning a new linear scene. Fixed seed. */
export function withNoise(s: Scene, sd: number, seed = 11): Scene {
  if (sd <= 0) return s;
  const r = rng(seed), o: Scene = { ...s, r: new Float32Array(s.r.length), g: new Float32Array(s.g.length), b: new Float32Array(s.b.length) };
  const f = (v: number) => dec(Math.min(1, Math.max(0, enc(v) + sd * r.norm())));
  for (let p = 0; p < s.r.length; p++) { o.r[p] = f(s.r[p]); o.g[p] = f(s.g[p]); o.b[p] = f(s.b[p]); }
  return o;
}

/** Multiply the light by a left-to-right gradient from 1 down to 1 - amount (shading). */
export function withShading(s: Scene, amount: number): Scene {
  if (amount <= 0) return s;
  const o: Scene = { ...s, r: new Float32Array(s.r.length), g: new Float32Array(s.g.length), b: new Float32Array(s.b.length) };
  for (let y = 0; y < s.h; y++) for (let x = 0; x < s.w; x++) {
    const f = 1 - amount * x / (s.w - 1), p = y * s.w + x;
    o.r[p] = s.r[p] * f; o.g[p] = s.g[p] * f; o.b[p] = s.b[p] * f;
  }
  return o;
}

function hsv(h: number, s: number, v: number): [number, number, number] {
  const c = v * s, hp = (h % 360) / 60, x = c * (1 - Math.abs((hp % 2) - 1)), m = v - c;
  const k = Math.floor(hp) % 6;
  const t = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][k];
  return [t[0] + m, t[1] + m, t[2] + m];
}

/** Draw a linear scene into RGBA bytes (sRGB-encoded), optionally scaled and with a mask
 *  tint. `linearDisplay` writes linear values straight to the screen (too dark). */
export function sceneToRGBA(s: { w: number; h: number; r: ArrayLike<number>; g: ArrayLike<number>; b: ArrayLike<number> }, out?: Uint8ClampedArray, linearDisplay = false): Uint8ClampedArray {
  const o = out ?? new Uint8ClampedArray(4 * s.w * s.h);
  const T = new Uint8ClampedArray(4096);
  for (let i = 0; i < 4096; i++) T[i] = Math.round(255 * (linearDisplay ? i / 4095 : enc(i / 4095)));
  const q = (v: number) => T[v <= 0 ? 0 : v >= 1 ? 4095 : Math.round(v * 4095)];
  for (let p = 0; p < s.w * s.h; p++) { o[4 * p] = q(s.r[p]); o[4 * p + 1] = q(s.g[p]); o[4 * p + 2] = q(s.b[p]); o[4 * p + 3] = 255; }
  return o;
}
