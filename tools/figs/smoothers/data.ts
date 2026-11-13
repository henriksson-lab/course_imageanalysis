// Image data for tools/figs/smoothers/figs.py (run by it; prints JSON). Uses the methods of the
// smoothercompare demo (demos/src/smoothercompare/methods.ts) at the same best-PSNR settings.
import { METHODS, noisyOf, noiseSd, scores, cardTruth } from "../../../demos/src/smoothercompare/methods";
import { testCard, psnr255 } from "../../../demos/src/lib/pde";
const r = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 10) / 10);
const BEST15: Record<string, number> = { gauss: 0.541, median: 1, pm: 0.268, tv: 9.34, bilateral: 28.6, nlm: 19.7, wavelet: 16.5, wiener: 3.66 };
const BEST40: Record<string, number> = { gauss: 0.903, median: 1, pm: 0.47, tv: 29.9, bilateral: 88.5, nlm: 38.3, wavelet: 56.7, wiener: 0.738 };
const out: Record<string, unknown> = {};
const truth = cardTruth(); out.truth = r(truth); out.features = testCard().features;
const y = noisyOf(truth, { image: "card", noise: "gauss", level: 15 }), sd = noiseSd(truth, y);
out.noisy = r(y); out.noisyPsnr = psnr255(y, truth);
out.res = METHODS.map(m => { const t0 = Date.now(); const u = m.run(y, BEST15[m.id], sd); return { id: m.id, label: m.label, param: m.param, p: BEST15[m.id], prior: m.prior, u: r(u), sc: scores(u, truth), ms: Date.now() - t0 }; });
const y40 = noisyOf(truth, { image: "card", noise: "gauss", level: 40 }), sd40 = noiseSd(truth, y40);
out.noisy40 = r(y40);
// thresholding at 120 after smoothing; settings that leave a similar small noise in flat regions
const SEG: [string, number][] = [["gauss", 2], ["pm", 2], ["tv", 45]];
const inRegion = (x: number, yy: number) => x < 64 && yy < 96 && !(x >= 38 && x < 56 && yy >= 66 && yy < 88);
const wrong = (u: ArrayLike<number>) => { let w = 0; for (let yy = 0; yy < 128; yy++) for (let x = 0; x < 128; x++) if (inRegion(x, yy)) { const i = yy * 128 + x; if ((u[i] > 120) !== (truth[i] > 120)) w++; } return w; };
const flatSd = (u: ArrayLike<number>) => { let s = 0, s2 = 0, n = 0; for (let j = 12; j < 34; j++) for (let i = 12; i < 34; i++) { const d = u[j * 128 + i]; s += d; s2 += d * d; n++; } return Math.sqrt(s2 / n - (s / n) ** 2); };
out.segNoisy = { wrong: wrong(y40), sd: flatSd(y40) };
out.seg = SEG.map(([id, p]) => { const m = METHODS.find(q => q.id === id)!; const u = m.run(y40, p, sd40); return { id, label: m.label, param: m.param, p, u: r(u), wrong: wrong(u), sd: flatSd(u) }; });
void BEST40;
console.log(JSON.stringify(out));
