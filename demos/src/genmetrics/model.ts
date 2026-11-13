// Model of the genmetrics demo (no DOM; tested in demos/test/diffusion-test.ts).
// The 2D Gaussian-mixture clouds of diffusion2d (imported from its model, unchanged): 1 000
// held-out real points, 1 000 training points, and 1 000 generated points from a "generator":
// the exact diffusion sampler (N steps, η), one wide Gaussian matched in mean and covariance
// ("between the modes"), mode dropping, a memoriser (copies of 50 training points), samples
// shrunk towards their mode ("too narrow"), or (guidance mode) the guided sampler of one class.
import { dataSet, type DataId } from "../diffusion2d/model";
import { sampleDiag, restrictMixture, type DiagMixture } from "../lib/mixture";
import { rng } from "../lib/ui";
import { mixtureDenoiser, guidedDenoiser, sigmaGrid, sampleRun } from "../lib/diffusion";
import { meanCov, frechetGaussian, precisionRecall, nearestDistances } from "../lib/genmetrics";
import { median } from "../lib/stats";

export { type DataId };
export type GenId = "sampler" | "between" | "drop" | "memo" | "narrow";
export const GEN_IDS: GenId[] = ["sampler", "between", "drop", "memo", "narrow"];
export const GEN_LABEL: Record<GenId, string> = { sampler: "diffusion sampler", between: "between the modes", drop: "mode dropping", memo: "memoriser", narrow: "too narrow" };
export const N = 1000;

/** Real (held-out), training and class labels for a data set; `cls` restricts to one group (guidance mode). */
export function realSets(id: DataId, cls: number | null): { m: DiagMixture; real: Float64Array; train: Float64Array; full: DiagMixture } {
  const ds = dataSet(id), full = ds.m;
  const m = cls === null ? full : restrictMixture(full, ds.groups[cls]);
  return { m, full, real: sampleDiag(m, N, rng(31)).X, train: sampleDiag(m, N, rng(32)).X };
}

/** Exact diffusion sampler (cosine schedule, uniform in time), optionally guided towards group `cls` with weight w. */
export function sampler(id: DataId, steps: number, eta: number, seed = 33, cls: number | null = null, w = 0, n = N): Float64Array {
  const ds = dataSet(id);
  const D = cls === null ? mixtureDenoiser(ds.m) : guidedDenoiser(mixtureDenoiser(ds.m, ds.groups[cls]), mixtureDenoiser(ds.m), w);
  const sig = sigmaGrid(steps, "time", "cosine"), r = rng(seed), rn = rng(seed + 1), out = new Float64Array(2 * n), x = new Float64Array(2);
  for (let p = 0; p < n; p++) {
    x[0] = Math.sqrt(sig[0] ** 2 + 1) * r.norm(); x[1] = Math.sqrt(sig[0] ** 2 + 1) * r.norm();
    sampleRun(x, D, { sigmas: sig, eta }, rn);
    out[2 * p] = x[0]; out[2 * p + 1] = x[1];
  }
  return out;
}

export function generate(id: DataId, gen: GenId, steps: number, eta: number, train: Float64Array): Float64Array {
  const ds = dataSet(id), m = ds.m;
  if (gen === "sampler") return sampler(id, steps, eta);
  if (gen === "between") {
    const { mu, S } = meanCov(sampleDiag(m, 20000, rng(34)).X, 2);
    // Cholesky of the 2 × 2 covariance
    const l11 = Math.sqrt(S[0]), l21 = S[2] / l11, l22 = Math.sqrt(Math.max(0, S[3] - l21 * l21));
    const r = rng(35), out = new Float64Array(2 * N);
    for (let p = 0; p < N; p++) { const a = r.norm(), b = r.norm(); out[2 * p] = mu[0] + l11 * a; out[2 * p + 1] = mu[1] + l21 * a + l22 * b; }
    return out;
  }
  if (gen === "drop") {
    // drop the first quarter of the components (at least one)
    const k = Math.max(1, Math.floor(m.k / 4));
    return sampleDiag(restrictMixture(m, Array.from({ length: m.k - k }, (_, j) => j + k)), N, rng(36)).X;
  }
  if (gen === "memo") {
    const r = rng(37), out = new Float64Array(2 * N);
    for (let p = 0; p < N; p++) { const q = p % 50; out[2 * p] = train[2 * q] + 0.01 * r.norm(); out[2 * p + 1] = train[2 * q + 1] + 0.01 * r.norm(); }
    return out;
  }
  // too narrow: samples pulled towards their component mean (factor 0.35)
  const s = sampleDiag(m, N, rng(38)), out = new Float64Array(2 * N);
  for (let p = 0; p < N; p++) for (let i = 0; i < 2; i++) { const mu = m.mu[2 * s.comp[p] + i]; out[2 * p + i] = mu + 0.35 * (s.X[2 * p + i] - mu); }
  return out;
}

export interface Metrics { precision: number; recall: number; frechet: number; nnTrain: number; nnReal: number; inReal: Uint8Array; rReal: Float64Array }

/** Precision/recall (k = 3), Fréchet distance, median nearest-training distance of the samples (and of the held-out real points, for reference). */
export function metrics(real: Float64Array, fake: Float64Array, train: Float64Array): Metrics {
  const pr = precisionRecall(real, fake, 2, 3);
  const a = meanCov(real, 2), b = meanCov(fake, 2);
  return {
    precision: pr.precision, recall: pr.recall, frechet: frechetGaussian(a.mu, a.S, b.mu, b.S, 2),
    nnTrain: median(nearestDistances(fake, train, 2)), nnReal: median(nearestDistances(real, train, 2)),
    inReal: pr.inReal, rReal: pr.rReal,
  };
}
