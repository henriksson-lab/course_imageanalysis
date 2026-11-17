// Presets of the blindspot demo (no DOM, so the tests and the figure scripts can use them).
import type { Settings } from "./model";

export type Img = "cell" | "test";
export interface Preset { id: string; label: string; img: Img; s: Settings }
export const DEFAULT: Settings = { noise: "white", sigma: 25, r: 2, train: "n2v", mix: false, cmp: "gauss", gsig: 0.8, nlmH: 25 };
export const PRESETS: Preset[] = [
  { id: "n2v", label: "Noise2Void: one noisy image", img: "cell", s: { ...DEFAULT } },
  { id: "ident", label: "no blind spot: it learns to copy", img: "cell", s: { ...DEFAULT, train: "ident" } },
  { id: "n2n", label: "Noise2Noise: two noisy images", img: "cell", s: { ...DEFAULT, train: "n2n" } },
  { id: "mix", label: "put the pixel back", img: "cell", s: { ...DEFAULT, mix: true } },
  { id: "nlm", label: "against non-local means", img: "test", s: { ...DEFAULT, mix: true, cmp: "nlm" } },
  { id: "corr", label: "correlated noise: it learns the noise", img: "cell", s: { ...DEFAULT, noise: "corr" } },
  { id: "stripes", label: "stripes", img: "cell", s: { ...DEFAULT, noise: "stripes" } },
];
