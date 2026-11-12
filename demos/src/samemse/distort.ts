// Five distortions of a grey image (0..255), each tuned by bisection on its parameter so that
// the MSE against the original equals a target (no DOM; used by the samemse demo and by the
// node scripts behind the exercise numbers). Every result is rounded and clamped to 0..255.
import { fromArray, gaussianBlur } from "../lib/imageops";
import { mse } from "../lib/metrics";
import { scaleTable, LUMA_Q, codeChannel } from "../lib/jpeg";
import { rng } from "../lib/random";

export type DistId = "shift" | "contrast" | "noise" | "blur" | "jpeg";
export const DISTORTIONS: DistId[] = ["shift", "contrast", "noise", "blur", "jpeg"];

const clampRound = (v: number) => Math.min(255, Math.max(0, Math.round(v)));

/** The distorted image for parameter p (meaning per distortion; see `describe`). */
export function distort(id: DistId, img: Float64Array, w: number, h: number, p: number): Float64Array {
  const n = w * h;
  if (id === "shift") {
    let mean = 0; for (const v of img) mean += v; mean /= n;
    const s = mean > 128 ? -p : p;                       // move away from the nearer end
    return img.map(v => clampRound(v + s));
  }
  if (id === "contrast") {
    let mean = 0; for (const v of img) mean += v; mean /= n;
    return img.map(v => clampRound(mean + (1 - p) * (v - mean)));   // contrast reduced by p
  }
  if (id === "noise") {
    const r = rng(42), out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = clampRound(img[i] + p * r.norm());
    return out;
  }
  if (id === "blur") return gaussianBlur(fromArray(w, h, img), p).data.map(clampRound) as unknown as Float64Array;
  return codeChannel(img, w, h, scaleTable(LUMA_Q, p)).decoded;     // JPEG with the table scaled by p %
}

/** Search range of the parameter. */
const RANGE: Record<DistId, [number, number]> = { shift: [0, 120], contrast: [0, 1], noise: [0, 100], blur: [0.05, 12], jpeg: [1, 5000] };

/** Bisection for the parameter that gives MSE = target (MSE grows with the parameter). */
export function tune(id: DistId, img: Float64Array, w: number, h: number, target: number, iters = 22): { p: number; out: Float64Array; mse: number } {
  let [lo, hi] = RANGE[id];
  const f = (p: number) => { const out = distort(id, img, w, h, p); return { p, out, mse: mse(out, img) }; };
  let best = f(hi);
  if (best.mse < target) return best;
  for (let k = 0; k < iters; k++) {
    const mid = id === "jpeg" ? Math.sqrt(lo * hi) : (lo + hi) / 2;
    const r = f(mid);
    if (Math.abs(r.mse - target) < Math.abs(best.mse - target)) best = r;
    if (r.mse < target) lo = mid; else hi = mid;
  }
  return best;
}

/** A short description of the parameter value. */
export function describe(id: DistId, p: number, img: Float64Array): string {
  let mean = 0; for (const v of img) mean += v; mean /= img.length;
  if (id === "shift") return `${mean > 128 ? "darker" : "brighter"} by ${p.toFixed(1)}`;
  if (id === "contrast") return `contrast × ${(1 - p).toFixed(3)}`;
  if (id === "noise") return `noise sd ${p.toFixed(1)}`;
  if (id === "blur") return `blur σ = ${p.toFixed(2)}`;
  return `JPEG, table × ${(p / 100).toFixed(2)}`;
}

export const LABEL: Record<DistId, string> = { shift: "brightness shift", contrast: "contrast change", noise: "Gaussian noise", blur: "blur", jpeg: "JPEG" };
