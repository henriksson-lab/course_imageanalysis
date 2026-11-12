// Synthetic images for the hessian demo (no DOM), designed at 128 × 128 (positions scale with N):
//  quadratic: f = a x² + b x y + c y² around the centre (x right, y DOWN, unit = 8 pixels);
//  scene: a bright blob, a thin and a thick bright line, a dark line, a saddle (two bright
//         blobs and two dark ones around a point), an edge;
//  vessels: curved bright lines of two thicknesses (a stand-in for a vessel tree).
import { Grey, fromFunction } from "../lib/imageops";
import { addGaussianNoise, filaments } from "../lib/synth";

export const QUAD_UNIT = 8, QUAD_GAIN = 1.5;

/** f = a x² + b x y + c y² with x, y in units of QUAD_UNIT pixels from the centre (y down), as grey levels 128 + QUAD_GAIN·f. */
export function quadratic(N: number, a: number, b: number, c: number): Grey {
  const m = (N - 1) / 2, u = QUAD_UNIT * N / 128;
  return fromFunction(N, N, (X, Y) => { const x = (X - m) / u, y = (Y - m) / u; return 128 + QUAD_GAIN * (a * x * x + b * x * y + c * y * y); });
}

/** Factor that turns the pixel Hessian of the quadratic image back into the Hessian of f in its own units ([[2a, b], [b, 2c]]). */
export const quadUnits = (N: number): number => (QUAD_UNIT * N / 128) ** 2 / QUAD_GAIN;

/** Distance from (x, y) to the segment (x0, y0)–(x1, y1). */
function segDist(x: number, y: number, x0: number, y0: number, x1: number, y1: number): number {
  const dx = x1 - x0, dy = y1 - y0, t = Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(x - x0 - t * dx, y - y0 - t * dy);
}

export function cleanHessScene(N = 128): Grey {
  const s = N / 128;
  return fromFunction(N, N, (X, Y) => {
    const x = X / s, y = Y / s;
    let v = 100;
    v += 80 * Math.exp(-((x - 28) ** 2 + (y - 28) ** 2) / (2 * 5 * 5));              // bright blob, s = 5
    v += 80 * Math.exp(-(segDist(x, y, 52, 10, 120, 44) ** 2) / (2 * 1.5 * 1.5));       // thin bright line, s = 1.5
    v += 60 * Math.exp(-(segDist(x, y, 10, 70, 60, 70) ** 2) / (2 * 4.5 * 4.5));        // thick bright line, s = 4.5
    v -= 60 * Math.exp(-(segDist(x, y, 72, 60, 72, 118) ** 2) / (2 * 2 * 2));           // dark line, s = 2
    // saddle around (104, 92): bright left/right, dark above/below
    for (const [cx, cy, sg] of [[92, 92, 1], [116, 92, 1], [104, 80, -1], [104, 104, -1]])
      v += sg * 50 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 5 * 5));
    if (y > 100 && x < 50) v += x < 25 ? 0 : 60;                                       // an edge (step)
    return v;
  });
}

export function vessels(N = 128, seed = 3): Grey {
  const thin = filaments({ width: N, height: N, count: 7, thickness: 1.2 * N / 128, amplitude: [70, 100], background: 0, noise: 0, seed, curvature: 0.3 });
  const thick = filaments({ width: N, height: N, count: 3, thickness: 3.5 * N / 128, amplitude: [60, 80], background: 0, noise: 0, seed: seed + 10, curvature: 0.2 });
  const out = fromFunction(N, N, (x, y) => 60 + Math.max(thin.image.data[y * N + x], thick.image.data[y * N + x]));
  return out;
}

export function makeHessScene(id: "scene" | "vessels", N = 128, noise = 4, seed = 1): { image: Grey } {
  const img = id === "vessels" ? vessels(N) : cleanHessScene(N);
  return { image: noise > 0 ? addGaussianNoise(img, noise, seed) : img };
}
