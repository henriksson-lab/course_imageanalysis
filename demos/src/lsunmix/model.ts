// The lsunmix demo's model, without DOM (so the exercise numbers can be computed in node):
// the four components' spectra, the true abundance maps of the day's cell, the lambda
// stack and the four unmixing methods.
import * as U from "../lib/unmix";

export const L0 = 480, L1 = 700, N_PIX = 128, P = N_PIX * N_PIX;
export type Method = "one" | "ls" | "wls" | "nnls";

export function dyesFor(sep: number): U.Dye[] {
  return [
    { name: "nucleus", peak: 515, width: 13, tail: 2.2 },
    { name: "mitochondria", peak: 515 + sep, width: 13, tail: 2.2 },
    { name: "membrane", peak: 625, width: 13, tail: 1.8 },
    { name: "autofluorescence", peak: 600, width: 70, tail: 1.3 },
  ];
}

/** True abundance maps (4 × P) of the day's cell. */
export function truthMaps(): Float32Array {
  const sc = U.cellScene(N_PIX, 3), A = new Float32Array(4 * P);
  for (let p = 0; p < P; p++) {
    A[p] = 0.9 * sc.nucleus[p];
    A[P + p] = sc.mito[p];
    A[2 * P + p] = 0.8 * sc.membrane[p];
    A[3 * P + p] = 0.6 * sc.auto[p];
  }
  return A;
}

/** Channel bands, the full 4-column mixing matrix and the noisy stack. */
export function simulate(A: Float32Array, n: number, sep: number, photons: number, seed: number): { bands: U.Band[]; M: Float64Array; Y: Float32Array } {
  const bands = U.contiguousBands(L0, L1, n), M = U.channelMatrix(dyesFor(sep), bands);
  return { bands, M, Y: U.forward(M, n, 4, A, P, { photons, seed }) };
}

/** Unmix with the first K columns of M. Returns abundances (4 × P; rows ≥ K zero), the
 *  model matrix and the residual stack (in counts). */
export function unmix(Y: Float32Array, M: Float64Array, n: number, K: number, method: Method, photons: number): { Ah: Float32Array; Mm: Float64Array; R: Float32Array } {
  const Mm = new Float64Array(n * K);
  for (let c = 0; c < n; c++) for (let j = 0; j < K; j++) Mm[c * K + j] = M[c * 4 + j];
  const Ah = new Float32Array(4 * P);
  if (n < K) { Ah.fill(NaN); return { Ah, Mm, R: new Float32Array(n * P) }; }
  if (method === "one") {
    for (let j = 0; j < K; j++) {
      let best = 0;
      for (let c = 1; c < n; c++) if (Mm[c * K + j] > Mm[best * K + j]) best = c;
      const s = 1 / (photons * Mm[best * K + j]);
      for (let p = 0; p < P; p++) Ah[j * P + p] = Y[best * P + p] * s;
    }
  } else {
    const Pi = U.pseudoInverse(Mm, n, K);
    if (!Pi) { Ah.fill(NaN); return { Ah, Mm, R: new Float32Array(n * P) }; }
    const ls = U.applyToStack(Pi, K, n, Y, P, 0, 1 / photons);
    Ah.set(ls);
    if (method === "wls" || method === "nnls") {
      const y = new Float64Array(n), w = new Float64Array(n), G = U.gram(Mm, n, K);
      for (let p = 0; p < P; p++) {
        for (let c = 0; c < n; c++) y[c] = Y[c * P + p] / photons;
        if (method === "nnls") {
          const a = U.nnlsGram(G, U.mty(Mm, n, K, y), K);
          for (let j = 0; j < K; j++) Ah[j * P + p] = a[j];
        } else {
          // weights 1/expected count, the expected count from the LS fit (Poisson: variance = mean)
          for (let c = 0; c < n; c++) { let e = 0; for (let j = 0; j < K; j++) e += Mm[c * K + j] * ls[j * P + p]; w[c] = 1 / Math.max(photons * e, 1); }
          const a = U.wlsPixel(Mm, n, K, y, w);
          if (a) for (let j = 0; j < K; j++) Ah[j * P + p] = a[j];
        }
      }
    }
  }
  return { Ah, Mm, R: U.residualStack(Mm, n, K, Ah, Y, P, photons) };
}

/** Residual per pixel in units of the expected Poisson noise: sqrt(Σ_c r_c²/max(y_c, 1) / n). */
export function residualChi(R: Float32Array, Y: Float32Array, n: number): Float32Array {
  const chi = new Float32Array(P);
  for (let p = 0; p < P; p++) { let s = 0; for (let c = 0; c < n; c++) s += R[c * P + p] ** 2 / Math.max(Y[c * P + p], 1); chi[p] = Math.sqrt(s / n); }
  return chi;
}

/** RMSE of component k against the truth, over all pixels. */
export function rmse(Ah: Float32Array, A: Float32Array, k: number): number {
  let s = 0;
  for (let p = 0; p < P; p++) s += (Ah[k * P + p] - A[k * P + p]) ** 2;
  return Math.sqrt(s / P);
}

/** Share (0..1) of negative values among the K abundance maps. */
export function negativeShare(Ah: Float32Array, K: number): number {
  let neg = 0;
  for (let i = 0; i < K * P; i++) if (Ah[i] < 0) neg++;
  return neg / (K * P);
}
