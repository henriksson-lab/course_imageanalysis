// The endmembers demo's data and methods, without DOM (numbers in the exercise come from here).
import * as U from "../lib/unmix";
import { dyesFor, truthMaps, L0, L1, N_PIX, P } from "../lsunmix/model";

export const NCH = 16, SEP = 50;

export interface EmData { K: number; Mtrue: Float64Array; cellIdx: number[]; Yc: Float32Array; sv: Float64Array }

/** The lambda stack of the cell's first K components (16 channels, Poisson noise, in
 *  abundance units), cell pixels only. Without pure pixels every component gets 30 % of
 *  the mean of the others. */
export function buildData(K: number, pure: boolean, photons: number): EmData {
  const Atrue = truthMaps(), bands = U.contiguousBands(L0, L1, NCH), Mfull = U.channelMatrix(dyesFor(SEP), bands);
  const A = new Float32Array(K * P);
  for (let k = 0; k < K; k++) for (let p = 0; p < P; p++) A[k * P + p] = Atrue[k * P + p];
  if (!pure) for (let p = 0; p < P; p++) {
    let tot = 0;
    for (let k = 0; k < K; k++) tot += Atrue[k * P + p];
    for (let k = 0; k < K; k++) A[k * P + p] = 0.7 * Atrue[k * P + p] + 0.3 * (tot - Atrue[k * P + p]) / (K - 1);
  }
  const Mtrue = new Float64Array(NCH * K);
  for (let c = 0; c < NCH; c++) for (let k = 0; k < K; k++) Mtrue[c * K + k] = Mfull[c * 4 + k];
  const Yn = U.forward(Mtrue, NCH, K, A, P, { photons, seed: 1 });
  const cell = U.cellScene(N_PIX, 3).cell, cellIdx: number[] = [];
  for (let p = 0; p < P; p++) if (cell[p]) cellIdx.push(p);
  const n = cellIdx.length, Yc = new Float32Array(NCH * n);
  for (let c = 0; c < NCH; c++) cellIdx.forEach((p, i) => { Yc[c * n + i] = Yn[c * P + p] / photons; });
  return { K, Mtrue, cellIdx, Yc, sv: U.stackSingularValues(Yc, NCH, n) };
}

/** Vertex finding on the rank-K projection; found spectra (sum 1) from the projected pixels. */
export function vertexSpectra(d: EmData, steps = d.K, Yk = U.projectRank(d.Yc, NCH, d.cellIdx.length, d.K)): { verts: number[]; F: Float64Array; Yk: Float32Array } {
  const n = d.cellIdx.length, K = d.K, verts = U.vertexFind(Yk, NCH, n, K, 0.3, steps), F = new Float64Array(NCH * K);
  verts.forEach((i, k) => { let s = 0; for (let c = 0; c < NCH; c++) s += Yk[c * n + i]; for (let c = 0; c < NCH; c++) F[c * K + k] = Yk[c * n + i] / s; });
  return { verts, F, Yk };
}

/** The first three true spectra, each scaled to sum 1 (the "known" spectra). */
export function knownSpectra(d: EmData): Float64Array[] {
  return [0, 1, 2].map(k => { const v = new Float64Array(NCH); let s = 0; for (let c = 0; c < NCH; c++) { v[c] = d.Mtrue[c * d.K + k]; s += v[c]; } return v.map(x => x / s); });
}
