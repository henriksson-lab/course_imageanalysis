// Data for the ncut demo: point-cloud presets and graphs from points or a small image.
// No DOM: also used to compute the numbers quoted in the exercise solutions.
import { rng } from "../lib/ui";
import { WGraph } from "./spectral";

export type Pt = [number, number];
export const IMG = 32;

export interface PointPreset { id: string; label: string; sigma: number; pts: () => Pt[] }

function blob(r: ReturnType<typeof rng>, n: number, cx: number, cy: number, sx: number, sy = sx): Pt[] {
  return Array.from({ length: n }, () => [cx + sx * r.norm(), cy + sy * r.norm()] as Pt);
}

function ring(r: ReturnType<typeof rng>, n: number, rad: number, jit: number): Pt[] {
  return Array.from({ length: n }, (_, k) => {
    const a = 2 * Math.PI * k / n + 0.1 * r.norm();
    const q = rad + jit * r.norm();
    return [0.5 + q * Math.cos(a), 0.5 + q * Math.sin(a)] as Pt;
  });
}

export const POINT_PRESETS: PointPreset[] = [
  { id: "two", label: "two blobs", sigma: 0.08, pts: () => { const r = rng(4); return [...blob(r, 25, 0.3, 0.45, 0.08), ...blob(r, 25, 0.72, 0.58, 0.08)]; } },
  { id: "three", label: "three blobs", sigma: 0.08, pts: () => { const r = rng(6); return [...blob(r, 20, 0.25, 0.3, 0.07), ...blob(r, 20, 0.75, 0.3, 0.07), ...blob(r, 20, 0.5, 0.75, 0.07)]; } },
  { id: "outlier", label: "blobs + outlier", sigma: 0.08, pts: () => { const r = rng(4); return [...blob(r, 25, 0.3, 0.45, 0.08), ...blob(r, 25, 0.72, 0.58, 0.08), [0.93, 0.08]]; } },
  { id: "rings", label: "two rings", sigma: 0.05, pts: () => { const r = rng(9); return [...ring(r, 20, 0.1, 0.01), ...ring(r, 44, 0.36, 0.015)]; } },
  { id: "long", label: "one long blob", sigma: 0.12, pts: () => { const r = rng(12); return blob(r, 50, 0.5, 0.5, 0.25, 0.06); } },
];

/** Complete graph on the points with Gaussian affinity w = exp(-d^2 / 2 sigma^2); tiny weights dropped. */
export function pointGraph(pts: Pt[], sigma: number): WGraph {
  const G: WGraph = { n: pts.length, p: [], q: [], w: [] };
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
    const d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
    const w = Math.exp(-d2 / (2 * sigma * sigma));
    if (w > 1e-12) { G.p.push(i); G.q.push(j); G.w.push(w); }
  }
  return G;
}

/** Pixel graph of an IMG x IMG image: neighbours within distance 2,
 *  w = exp(-(I_p - I_q)^2 / 2 sigma^2) * exp(-d^2 / 2 * 1.5^2). */
export function imageGraph(I: ArrayLike<number>, sigma: number): WGraph {
  const G: WGraph = { n: IMG * IMG, p: [], q: [], w: [] };
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) for (let dy = 0; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if ((dy === 0 && dx <= 0) || dx * dx + dy * dy > 4) continue;
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || xx >= IMG || yy >= IMG) continue;
    const a = y * IMG + x, b = yy * IMG + xx;
    G.p.push(a); G.q.push(b);
    G.w.push(Math.exp(-((I[a] - I[b]) ** 2) / (2 * sigma * sigma)) * Math.exp(-(dx * dx + dy * dy) / (2 * 1.5 * 1.5)));
  }
  return G;
}

/** Synthetic 32 x 32 image: a bright disk and a dimmer square on a dark background, with noise. */
export function diskImage(): Float64Array {
  const r = rng(21);
  return Float64Array.from({ length: IMG * IMG }, (_, p) => {
    const x = p % IMG, y = Math.floor(p / IMG);
    const v = (x - 11) ** 2 + (y - 12) ** 2 <= 49 ? 190 : (x >= 19 && x <= 28 && y >= 17 && y <= 27) ? 130 : 60;
    return Math.min(255, Math.max(0, v + 15 * r.norm()));
  });
}

/** 32 x 32 crop of the 256 x 256 micrograph around the nucleus: 160 x 160 pixels from
 *  (x 64, y 16), 5 x 5 block averages. */
export function cellCrop(big: ArrayLike<number>): Float64Array {
  const out = new Float64Array(IMG * IMG);
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
    let s = 0;
    for (let v = 0; v < 5; v++) for (let u = 0; u < 5; u++) s += big[(16 + 5 * y + v) * 256 + 64 + 5 * x + u];
    out[y * IMG + x] = s / 25;
  }
  return out;
}
