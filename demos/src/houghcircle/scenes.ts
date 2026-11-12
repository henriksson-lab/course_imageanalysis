// Seeded synthetic scenes for the houghcircle demo (no DOM; also used by demos/test/hough-test.ts).
// N x N, grey 0..255, with the true circles (pixel coordinates). `clean` marks an isolated,
// fully visible disc that every reasonable setting should find.
import { Grey, createGrey, fromArray } from "../lib/imageops";
import { discs, nuclei, addGaussianNoise } from "../lib/synth";
import { fillPolygons } from "../lib/hough";

export const N = 160;
export type CircleSceneId = "discs" | "plate" | "rings" | "nuclei";
export interface TrueCircle { x: number; y: number; r: number; clean: boolean; note?: string }
export interface CircleScene { img: Grey; truth: TrueCircle[]; polarity: "bright" | "dark" }

export function makeCircleScene(id: CircleSceneId, noise = 6, seed = 1): CircleScene {
  let img: Grey, truth: TrueCircle[] = [], polarity: "bright" | "dark" = "bright";
  if (id === "discs") {
    truth = [
      { x: 35, y: 35, r: 12, clean: true },
      { x: 96, y: 32, r: 18, clean: true },
      { x: 140, y: 30, r: 8, clean: true },
      { x: 28, y: 96, r: 13, clean: false, note: "touching" },
      { x: 53, y: 96, r: 12, clean: false, note: "touching" },
      { x: 62, y: 140, r: 7, clean: true },
      { x: 118, y: 128, r: 20, clean: false, note: "occluded" },
    ];
    const el = { x: 108, y: 82, a: 17, b: 10, angle: 0.5 };
    const d = discs(N, N, [...truth.map(t => ({ x: t.x, y: t.y, r: t.r })), el], { background: 50 });
    // occluder: a background-coloured band across the lower-right disc
    const occ = fillPolygons(N, N, [{ poly: [[96, 136], [160, 120], [160, 132], [100, 146]], value: 1 }], 0);
    for (let i = 0; i < occ.length; i++) d.image.data[i] = d.image.data[i] * (1 - occ[i]) + 50 * occ[i];
    img = d.image;
  } else if (id === "plate") {
    polarity = "dark";
    img = createGrey(N, N, 175);
    const shapes: { x: number; y: number; r: number }[] = [];
    for (let row = 0; row < 6; row++) for (let col = 0; col < 7; col++) {
      const x = 17 + col * 21, y = 27 + row * 21;
      shapes.push({ x, y, r: 8 });
      truth.push({ x, y, r: 8, clean: true });
    }
    const d = discs(N, N, shapes, { background: 175, values: shapes.map((_, k) => (k % 9 === 4 ? 120 : 70)) });
    img = d.image;
  } else if (id === "rings") {
    img = createGrey(N, N, 40);
    const cx = 80, cy = 80, radii = [16, 29, 43, 58], amp = [150, 90, 120, 60];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const rr = Math.hypot(x - cx, y - cy);
      let v = 40 + 30 * Math.exp(-rr / 25);
      radii.forEach((R, k) => { v += amp[k] * Math.exp(-0.5 * ((rr - R) / 1.6) ** 2); });
      img.data[y * N + x] = v;
    }
    truth = radii.map(R => ({ x: cx, y: cy, r: R, clean: true }));
  } else {
    const nu = nuclei({ width: N, height: N, count: 12, radius: [8, 13], elongation: 0.25, touching: 0.3, noise: 0, seed: 4 });
    img = nu.image;
    truth = nu.ellipses.map(e => ({ x: e.x, y: e.y, r: Math.sqrt(e.a * e.b), clean: e.a / e.b < 1.15, note: e.a / e.b >= 1.15 ? "elongated" : undefined }));
  }
  if (noise > 0) img = addGaussianNoise(img, noise, seed + 101);
  return { img: fromArray(N, N, img.data), truth, polarity };
}
