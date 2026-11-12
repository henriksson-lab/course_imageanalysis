// Default training settings of the boostdetect demo (shared by the demo and precompute.ts).
export const DEFAULTS = {
  nPos: 300, nNeg: 600, nTestPos: 200, nTestNeg: 400,
  monoRounds: 50,
  nStages: 8, dStage: 0.995, fStage: 0.5, maxStumps: 30, nMine: 500,
};
/** Training-set variants: target, seed of the training set, glasses in the training faces. */
export const VARIANTS = [
  { id: "faces", target: "faces", seed: 11, testSeed: 12, noGlasses: false },
  { id: "faces-noglasses", target: "faces", seed: 11, testSeed: 12, noGlasses: true },
  { id: "cells", target: "cells", seed: 21, testSeed: 22, noGlasses: false },
];
export function variantId(target: string, noGlasses: boolean): string {
  return target === "faces" && noGlasses ? "faces-noglasses" : target;
}
