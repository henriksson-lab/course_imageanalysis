// The crowdfit pipeline without DOM (shared by the demo and the day's scripts): two spots
// a distance d apart → matched filter → maxima → one- and two-spot least-squares fits → penalty.
import { fromArray, gaussianBlur } from "../lib/imageops";
import { localMaxima2D } from "../lib/edges";
import { median } from "../lib/stats";
import { pixelGaussian, fitSpots, selectSpotCount, FitResult } from "../lib/spotfit";
import { rng } from "../lib/ui";

export const W = 31, H = 19, CX = 15, CY = 9, BG = 20, A1 = 60, K_DET = 4;
export interface State { sigma: number; dr: number; q: number; noise: number; c: number; seed: number }

export interface Analysis { img: Float64Array; filt: Float64Array; cands: { x: number; y: number; value: number }[]; f1: FitResult; f2: FitResult; sel: ReturnType<typeof selectSpotCount>; final: number }

/** Simulate the pair and run the pipeline: matched filter → maxima → fits → selection. */
export function analyse(s: State, seed: number): Analysis {
  const d = s.dr * s.sigma, tot = 2 * Math.PI * s.sigma * s.sigma;
  const m1 = pixelGaussian(CX - d / 2, CY, s.sigma, A1 * tot, BG, W, H), m2 = pixelGaussian(CX + d / 2, CY, s.sigma, s.q * A1 * tot, 0, W, H);
  const r = rng(seed), img = Float64Array.from(m1, (v, i) => v + m2[i] + s.noise * r.norm());
  const g = gaussianBlur(fromArray(W, H, img), s.sigma, "mirror");
  const med = median(g.data), gain = 1 / (2 * Math.sqrt(Math.PI) * s.sigma);
  const filt = Float64Array.from(g.data, v => v - med);
  const cands = localMaxima2D({ width: W, height: H, data: Float32Array.from(filt) }, K_DET * Math.max(0.3, s.noise) * gain, 2, 1);
  const c0 = cands[0] ?? { x: CX, y: CY };
  const sumAll = img.reduce((a, v) => a + v, 0) - BG * W * H;
  const f1 = fitSpots(img, W, H, s.sigma, [{ x: c0.x, y: c0.y, N: Math.max(50, sumAll) }], BG);
  // second spot where the residual is largest; the first one mirrored about the one-spot fit
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < W * H; i++) { const v = img[i] - f1.model[i]; if (v > bv) { bv = v; bi = i; } }
  const sx = bi % W, sy = Math.floor(bi / W), p = f1.spots[0];
  let ox = sx - p.x, oy = sy - p.y; const L = Math.hypot(ox, oy);
  if (L < 0.5) { ox = 0.5; oy = 0; }
  const f2 = fitSpots(img, W, H, s.sigma, [{ x: p.x - ox, y: p.y - oy, N: p.N / 2 }, { x: p.x + ox, y: p.y + oy, N: p.N / 2 }], f1.B);
  const sel = selectSpotCount(f1.rss, f2.rss, Math.max(0.3, s.noise), s.c);
  const final = cands.length >= 2 ? cands.length : sel.keepTwo ? 2 : 1;
  return { img, filt, cands, f1, f2, sel, final };
}

