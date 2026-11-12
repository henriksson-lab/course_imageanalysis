// Scenes and the simulated detector of the nmsboxes demo (no DOM, so the tests use it).
// A scene is a 128 × 128 synthetic fluorescence image of elliptic cells (lib/synth.ts) with
// the true boxes. The "detector" is simulated: several jittered boxes per true cell with a
// score that grows with their IoU, plus a few false positives; and a centre heat map.
import { nuclei, rasteriseEllipses, addGaussianNoise, type Ellipse } from "../lib/synth";
import { createGrey, gaussianBlur, type Grey } from "../lib/imageops";
import { rng } from "../lib/ui";
import { type Box, boxIou, boxFromCentre, boxW, boxH } from "../lib/boxes";

export const N = 128;

export type SceneId = "sparse" | "crowd" | "rod";

export interface Scene { image: Grey; ellipses: Ellipse[]; truth: Box[]; labels: Int32Array }

/** Axis-aligned bounding box of an ellipse (half-widths √(a²cos² + b²sin²), …). */
export function ellipseBox(e: Ellipse): Box {
  const c = Math.cos(e.angle), s = Math.sin(e.angle);
  const hx = Math.sqrt(e.a * e.a * c * c + e.b * e.b * s * s), hy = Math.sqrt(e.a * e.a * s * s + e.b * e.b * c * c);
  return { x0: e.x - hx, y0: e.y - hy, x1: e.x + hx, y1: e.y + hy };
}

function render(ellipses: Ellipse[], seed: number): Scene {
  const labels = rasteriseEllipses(N, N, ellipses);
  const img = createGrey(N, N, 40);
  const r = rng(seed);
  const level = ellipses.map(() => 150 + 60 * r.unif());
  for (let i = 0; i < N * N; i++) if (labels[i] > 0) img.data[i] = level[labels[i] - 1];
  const out = addGaussianNoise(gaussianBlur(img, 1), 8, seed + 11);
  return { image: out, ellipses, truth: ellipses.map(ellipseBox), labels };
}

export function makeScene(id: SceneId): Scene {
  if (id === "sparse") {
    const nu = nuclei({ width: N, height: N, count: 9, radius: [7, 11], touching: 0, elongation: 0.5, seed: 4 });
    return render(nu.ellipses, 4);
  }
  if (id === "crowd") {
    // two palisades of touching elongated cells at ±45°: their boxes overlap their neighbours'
    const el: Ellipse[] = [];
    const r = rng(21);
    for (let k = 0; k < 5; k++) {
      el.push({ x: 20 + 9 * k + 0.6 * r.norm(), y: 54 - 9 * k + 0.6 * r.norm(), a: 19 + 2 * r.unif(), b: 6, angle: Math.PI / 4 + 0.05 * r.norm() });
      el.push({ x: 64 + 9 * k + 0.6 * r.norm(), y: 72 + 9 * k + 0.6 * r.norm(), a: 19 + 2 * r.unif(), b: 6, angle: -Math.PI / 4 + 0.05 * r.norm() });
    }
    el.push({ x: 30, y: 100, a: 11, b: 10, angle: 0 }, { x: 101, y: 30, a: 10, b: 9, angle: 0.5 });
    return render(el, 21);
  }
  // rod: a long diagonal rod, a tiny cell and a few ordinary cells
  const el: Ellipse[] = [
    { x: 62, y: 62, a: 34, b: 3.5, angle: Math.PI / 4 },
    { x: 100, y: 28, a: 3.2, b: 3.2, angle: 0 },
    { x: 28, y: 98, a: 11, b: 9, angle: 0.4 },
    { x: 98, y: 96, a: 12, b: 10, angle: -0.3 },
    { x: 26, y: 26, a: 10, b: 8, angle: 1.0 },
  ];
  return render(el, 33);
}

export interface Candidates { boxes: Box[]; scores: number[]; source: number[] }   // source: truth index or −1

/**
 * The simulated detector: per true box `per` candidates with centre jitter (sd `jit` × size)
 * and size jitter (log-normal, sd `jit`), score = 0.3 + 0.65·IoU(candidate, truth) + noise;
 * plus `nFalse` false positives with scores 0.1–0.5. Seeded; scores rounded to 0.01.
 */
export function simulateDetector(truth: Box[], seed: number, per = 5, jit = 0.12, nFalse = 3): Candidates {
  const r = rng(seed), boxes: Box[] = [], scores: number[] = [], source: number[] = [];
  truth.forEach((t, k) => {
    const w = boxW(t), h = boxH(t), cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2;
    for (let m = 0; m < per; m++) {
      const b = m === 0 ? boxFromCentre(cx + 0.04 * w * r.norm(), cy + 0.04 * h * r.norm(), w * Math.exp(0.03 * r.norm()), h * Math.exp(0.03 * r.norm()))
        : boxFromCentre(cx + jit * w * r.norm(), cy + jit * h * r.norm(), w * Math.exp(jit * r.norm()), h * Math.exp(jit * r.norm()));
      const s = Math.min(0.99, Math.max(0.05, 0.3 + 0.65 * boxIou(b, t) + 0.04 * r.norm()));
      boxes.push(b); scores.push(Math.round(s * 100) / 100); source.push(k);
    }
  });
  for (let m = 0; m < nFalse; m++) {
    const w = 10 + 12 * r.unif(), h = 10 + 12 * r.unif();
    boxes.push(boxFromCentre(10 + (N - 20) * r.unif(), 10 + (N - 20) * r.unif(), w, h));
    scores.push(Math.round((0.1 + 0.4 * r.unif()) * 100) / 100); source.push(-1);
  }
  return { boxes, scores, source };
}

/**
 * Simulated centre heat map: a Gaussian bump of width σ at every true centre (max over
 * objects, as the training target), plus smooth seeded noise of amplitude `noise`, stronger on the objects (× (0.2 + bump)).
 */
export function centreHeatmap(truth: Box[], sigma: number, noise: number, seed: number): Float32Array {
  const H = new Float32Array(N * N);
  for (const t of truth) {
    const cx = (t.x0 + t.x1) / 2 - 0.5, cy = (t.y0 + t.y1) / 2 - 0.5;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const v = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma));
      if (v > H[y * N + x]) H[y * N + x] = v;
    }
  }
  if (noise > 0) {
    const g = createGrey(N, N), r = rng(seed);
    for (let i = 0; i < g.data.length; i++) g.data[i] = r.norm();
    const b = gaussianBlur(g, 2);
    let sd = 0; for (const v of b.data) sd += v * v; sd = Math.sqrt(sd / b.data.length);
    for (let i = 0; i < H.length; i++) H[i] += noise * (0.2 + H[i]) * b.data[i] / sd;   // larger where there is signal
  }
  return H;
}

/** Boxes from heat-map peaks: centre at the peak, size of the nearest true box (a simulated "size head"). */
export function boxesFromPeaks(peaks: number[], truth: Box[]): Box[] {
  return peaks.map(p => {
    const x = (p % N) + 0.5, y = Math.floor(p / N) + 0.5;
    let best = truth[0], bd = Infinity;
    for (const t of truth) { const d = Math.hypot((t.x0 + t.x1) / 2 - x, (t.y0 + t.y1) / 2 - y); if (d < bd) { bd = d; best = t; } }
    return boxFromCentre(x, y, boxW(best), boxH(best));
  });
}
