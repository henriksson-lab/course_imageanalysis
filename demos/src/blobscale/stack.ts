// The scale-space stack of the blobscale demo (no DOM), made fast by working on a smaller
// image at coarse scales: after a blur of σ there is no detail finer than about σ, so for
// σ ≥ 3 the image is shrunk ×2 (σ ≥ 6: ×4) by area averaging, filtered with σ/d (minus
// the small blur of the averaging itself), and the response enlarged back bilinearly. The
// normalised LoG σ²∇²L is scale-invariant, so it needs no extra factor. Cost ≈ 1/8 of the
// direct stack; the test compares both.
import { Grey, resize, gaussianBlur, gaussianKernel, correlateRows, correlateCols, combine } from "../lib/imageops";

export function factorFor(sigma: number): number { return sigma >= 8 ? 4 : sigma >= 4 ? 2 : 1; }

/** σ²(Lxx + Lyy) with kernels cut at 3σ (the lib's logFilter cuts at 4σ; the difference is < 1 %). */
function normLoG3(g: Grey, s: number): Grey {
  const r = Math.max(1, Math.ceil(3 * s)), k0 = gaussianKernel(s, 0, r), k2 = gaussianKernel(s, 2, r);
  const xx = correlateCols(correlateRows(g, k2), k0), yy = correlateCols(correlateRows(g, k0), k2);
  return combine(xx, yy, (a, b) => s * s * (a + b));
}

/** −σ²∇²L (sign = −1, bright blobs positive) or +σ²∇²L (sign = +1, dark blobs) for each σ; DoG/(k−1) if k > 0. */
export function responseStack(img: Grey, sigmas: number[], sign: 1 | -1, dogK = 0): Grey[] {
  const small = new Map<number, Grey>();
  const shrink = (d: number) => {
    if (d === 1) return img;
    let g = small.get(d);
    if (!g) { g = resize(img, Math.ceil(img.width / d), Math.ceil(img.height / d), "area"); small.set(d, g); }
    return g;
  };
  return sigmas.map(s => {
    const d = factorFor(s), g = shrink(d);
    const sd = Math.sqrt(Math.max(0.25, s * s - (d * d - 1) / 12)) / d;
    let r: Grey;
    if (dogK > 0) r = combine(gaussianBlur(g, dogK * sd), gaussianBlur(g, sd), (a, b) => sign * (a - b) / (dogK - 1));
    else { r = normLoG3(g, sd); for (let i = 0; i < r.data.length; i++) r.data[i] *= sign; }
    return d === 1 ? r : resize(r, img.width, img.height, "bilinear");
  });
}
