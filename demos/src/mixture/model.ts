// Computation of the mixture demo (no DOM), so scripts and tests can reproduce its numbers.
import { Grey, createGrey } from "../lib/imageops";
import { intHistogram } from "../lib/histogram";
import { rng } from "../lib/random";
import { otsu, isodata, kittler, triangle, kapur, gaussianCrossings, thresholdError, normalPdf } from "../lib/threshold";
import { blobMask } from "../histthresh/scenes";

export type Method = "otsu" | "isodata" | "kittler" | "triangle" | "kapur" | "bayes";
export interface Params { mu0: number; mu1: number; s0: number; s1: number; w1: number; n: number }
export interface Result { t: number; curve: Float64Array; extra?: number[] }

export interface Model {
  img: Grey; truth: Uint8Array; hist: Float64Array;
  results: Map<Method, Result>;
  isObject(m: Method, v: number): boolean;
  modelError(m: Method): number;
  wrongPixels(m: Method): number;   // fraction of pixels of the image classified wrongly
}

/** Draw the image from the mixture and run every method on its histogram. */
export function mixtureModel(p: Params): Model {
  const n = p.n;
  const truth = blobMask(n, n, p.w1, Math.max(1.5, n / 24), 7);
  const r = rng(1234 + n);
  const img = createGrey(n, n);
  for (let i = 0; i < n * n; i++) {
    const v = truth[i] ? p.mu1 + p.s1 * r.norm() : p.mu0 + p.s0 * r.norm();
    img.data[i] = Math.max(0, Math.min(255, Math.round(v)));
  }
  const hist = intHistogram(img.data, 0, 255).counts;
  const results = new Map<Method, Result>();
  results.set("otsu", otsu(hist));
  const iso = isodata(hist, 127.5);
  results.set("isodata", { t: iso.t, curve: Float64Array.from(iso.curve, (g, t) => g - (t + 0.5)) });
  results.set("kittler", kittler(hist));
  results.set("triangle", triangle(hist));
  results.set("kapur", kapur(hist));
  const cr = gaussianCrossings(p.mu0, p.s0, p.mu1, p.s1, p.w1).filter(c => c > -50 && c < 305);
  const err = Float64Array.from({ length: 256 }, (_, t) => thresholdError(t + 0.5, p.mu0, p.s0, p.mu1, p.s1, p.w1));
  const between = cr.filter(c => c > Math.min(p.mu0, p.mu1) && c < Math.max(p.mu0, p.mu1));
  const tb = between.length ? Math.round(between[0] - 0.5) : err.indexOf(Math.min(...err));
  results.set("bayes", { t: tb, curve: err, extra: cr });

  const isObject = (m: Method, v: number): boolean => {
    if (m === "bayes") return p.w1 * normalPdf(v, p.mu1, p.s1) > (1 - p.w1) * normalPdf(v, p.mu0, p.s0);
    return v > results.get(m)!.t;
  };
  const modelError = (m: Method): number => {
    if (m !== "bayes") return thresholdError(results.get(m)!.t + 0.5, p.mu0, p.s0, p.mu1, p.s1, p.w1);
    let e = 0;
    for (let x = -200; x <= 455; x += 0.25) e += Math.min((1 - p.w1) * normalPdf(x, p.mu0, p.s0), p.w1 * normalPdf(x, p.mu1, p.s1)) * 0.25;
    return e;
  };
  const wrongPixels = (m: Method): number => {
    let w = 0;
    for (let i = 0; i < n * n; i++) if ((isObject(m, img.data[i]) ? 1 : 0) !== truth[i]) w++;
    return w / (n * n);
  };
  return { img, truth, hist, results, isObject, modelError, wrongPixels };
}
