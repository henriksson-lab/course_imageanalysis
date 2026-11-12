// The stainsep demo's constants, without DOM (numbers in the exercise come from here).
import * as U from "../lib/unmix";

export type V3 = [number, number, number];
export const N = 160, P = N * N, I0_TRUE: V3 = [242, 238, 246];
// the stains of THIS (synthetic) slide: close to, but not exactly, the published vectors
export const TRUE_V: V3[] = [U.unit3([0.60, 0.74, 0.30]), U.unit3([0.12, 0.93, 0.34]), U.unit3([0.30, 0.55, 0.78])];
export const RUIF: V3[] = [U.unit3(U.STAINS.H), U.unit3(U.STAINS.E), U.unit3(U.STAINS.DAB)];
export const WRONG_E: V3 = U.unit3([0.30, 0.88, 0.37]), WRONG_DAB: V3 = U.unit3([0.45, 0.62, 0.64]);
export const PARALLEL_E: V3 = U.unit3([0.45, 0.80, 0.40]), PARALLEL_DAB: V3 = U.unit3([0.52, 0.68, 0.52]);
