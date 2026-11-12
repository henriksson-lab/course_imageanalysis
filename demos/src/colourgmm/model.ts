// Model of the colourgmm demo (no DOM): synthetic colour images with ground truth, the feature
// vectors of their pixels (RGB, Lab, Lab + position), and scoring (cluster → class mapping,
// Dice per class). The fitting itself is lib/kmeans.ts and lib/gmm.ts.
//
// Images (all synthetic, seeded):
//   scene   – a red ball lit from the left (a strong shadow gradient across it), a green
//             block and a blue ellipse on a beige table; classes 0 table, 1 red, 2 green, 3 blue;
//   odd     – the same scene plus one small cyan object (class 4) that no class describes;
//   tissue  – a stained tissue section: pink matrix, purple nuclei, white lumina (0, 1, 2).
import { srgbToLab, type RGB } from "../lib/colour";
import { rng } from "../lib/random";
import { ellipseDistance, type Ellipse } from "../lib/synth";
import { gaussianBlur, createGrey } from "../lib/imageops";

export const N = 128;
export type ImageId = "scene" | "odd" | "tissue";
export type Features = "rgb" | "lab" | "labxy";

export interface ColourImage {
  /** sRGB in [0, 1], 3 numbers per pixel. */
  rgb: Float32Array;
  truth: Int32Array;
  classNames: string[];
  /** Index of the class with the shadow gradient (for the readout), or -1. */
  shadowed: number;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export function makeImage(id: ImageId, noise = 0.025, seed = 3): ColourImage {
  const rgb = new Float32Array(3 * N * N), truth = new Int32Array(N * N), r = rng(seed);
  if (id === "tissue") {
    // low-frequency stain-density variation
    const tex = createGrey(N, N);
    for (let i = 0; i < N * N; i++) tex.data[i] = r.norm();
    const smooth = gaussianBlur(tex, 6);
    let sd = 0;
    for (const v of smooth.data) sd += v * v;
    sd = Math.sqrt(sd / (N * N)) || 1;
    const nuclei: Ellipse[] = [];
    for (let k = 0; k < 34; k++) nuclei.push({ x: 4 + 120 * r.unif(), y: 4 + 120 * r.unif(), a: 4.5 + 2.5 * r.unif(), b: 3 + 1.5 * r.unif(), angle: Math.PI * r.unif() });
    const lumina: Ellipse[] = [{ x: 36, y: 40, a: 18, b: 11, angle: 0.5 }, { x: 96, y: 92, a: 14, b: 9, angle: -0.4 }];
    const cols: RGB[] = [[0.92, 0.62, 0.76], [0.38, 0.22, 0.56], [0.96, 0.94, 0.95]];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const p = y * N + x;
      let c = 0;
      if (lumina.some(e => ellipseDistance(e, x, y) <= 1)) c = 2;
      else if (nuclei.some(e => ellipseDistance(e, x, y) <= 1)) c = 1;
      truth[p] = c;
      const dens = c === 2 ? 0 : 0.08 * smooth.data[p] / sd;
      for (let q = 0; q < 3; q++) rgb[3 * p + q] = clamp01(cols[c][q] * (1 - dens) + noise * r.norm());
    }
    return { rgb, truth, classNames: ["matrix", "nuclei", "lumen"], shadowed: -1 };
  }
  const ball: Ellipse = { x: 42, y: 46, a: 30, b: 30, angle: 0 };
  const blue: Ellipse = { x: 90, y: 98, a: 27, b: 18, angle: 0.3 };
  const odd: Ellipse = { x: 26, y: 106, a: 6, b: 6, angle: 0 };
  const cols: RGB[] = [[0.86, 0.82, 0.72], [0.92, 0.2, 0.14], [0.2, 0.56, 0.48], [0.24, 0.46, 0.7], [0.95, 0.85, 0.1]];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x;
    let c = 0;
    if (ellipseDistance(ball, x, y) <= 1) c = 1;
    else if (x >= 80 && x < 120 && y >= 12 && y < 52) c = 2;
    else if (ellipseDistance(blue, x, y) <= 1) c = 3;
    else if (id === "odd" && ellipseDistance(odd, x, y) <= 1) c = 4;
    truth[p] = c;
    // light from the left: the ball darkens strongly towards its right side
    const shade = c === 1 ? 1 - 0.85 * clamp01((x - ball.x + ball.a) / (2 * ball.a)) : 1;
    for (let q = 0; q < 3; q++) rgb[3 * p + q] = clamp01(cols[c][q] * shade + noise * r.norm());
  }
  return { rgb, truth, classNames: id === "odd" ? ["table", "red ball", "green block", "blue ellipse", "odd object"] : ["table", "red ball", "green block", "blue ellipse"], shadowed: 1 };
}

export const FEATURE_NAMES: Record<Features, string[]> = { rgb: ["R", "G", "B"], lab: ["L*", "a*", "b*"], labxy: ["L*", "a*", "b*", "x", "y"] };

/** Feature vectors (n·d): RGB scaled to 0..100, Lab, or Lab plus position scaled by `wxy`
 *  (x, y in 0..100 times wxy). Also returns the Lab values for display. */
export function features(img: ColourImage, f: Features, wxy = 0.5): { X: Float64Array; d: number; lab: Float64Array } {
  const n = N * N, lab = new Float64Array(3 * n);
  for (let p = 0; p < n; p++) {
    const L = srgbToLab([img.rgb[3 * p], img.rgb[3 * p + 1], img.rgb[3 * p + 2]]);
    lab[3 * p] = L[0]; lab[3 * p + 1] = L[1]; lab[3 * p + 2] = L[2];
  }
  const d = f === "labxy" ? 5 : 3, X = new Float64Array(n * d);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < 3; q++) X[p * d + q] = f === "rgb" ? 100 * img.rgb[3 * p + q] : lab[3 * p + q];
    if (d === 5) { X[p * d + 3] = wxy * 100 * (p % N) / N; X[p * d + 4] = wxy * 100 * Math.floor(p / N) / N; }
  }
  return { X, d, lab };
}

/** Map every cluster to the truth class it overlaps most; Dice per truth class of the union
 *  of its clusters. */
export function score(labels: ArrayLike<number>, truth: ArrayLike<number>, K: number, C: number): { map: number[]; dice: number[]; mean: number } {
  const ov = Array.from({ length: K }, () => new Float64Array(C));
  for (let p = 0; p < labels.length; p++) ov[labels[p]][truth[p]]++;
  const map = ov.map(row => { let b = 0; for (let c = 1; c < C; c++) if (row[c] > row[b]) b = c; return b; });
  const tp = new Float64Array(C), np = new Float64Array(C), nt = new Float64Array(C);
  for (let p = 0; p < labels.length; p++) {
    const pc = map[labels[p]], t = truth[p];
    np[pc]++; nt[t]++;
    if (pc === t) tp[t]++;
  }
  const dice = Array.from({ length: C }, (_, c) => (np[c] + nt[c] > 0 ? 2 * tp[c] / (np[c] + nt[c]) : 1));
  return { map, dice, mean: dice.reduce((a, b) => a + b, 0) / C };
}
