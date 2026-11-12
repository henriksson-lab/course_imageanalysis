// Images, presets and modes of the hessian demo, as registries so other days can extend the
// demo without touching its main loop (the skeleton day adds a vesselness mode this way):
//
//   // demos/src/hessian/vessel.ts (owned by the skeleton day)
//   import { registerImage, registerMode } from "./modes";
//   registerImage({ id: "tree", label: "vessel tree", size: 256, make: (N, noise) => ... });
//   registerMode({ id: "vessel", label: "vesselness", images: ["tree", "cell"],
//     defaults: { image: "tree", tab: "vessel", multi: true }, presets: [ ... ] });
//   // and one line in main.ts:  import "./vessel";
//
// A mode is chosen with data-mode="<id>" (default "hessian"); it selects which image buttons
// and presets are shown and the starting state. Everything else (tabs, sliders, glyph,
// scatter) is shared. No DOM in this file except through `make` returning a Promise.
import { Grey } from "../lib/imageops";
import { loadCell } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { quadratic, quadUnits, makeHessScene } from "./scene";

export type Tab = "image" | "classes" | "l1" | "l2" | "trace" | "det" | "vessel" | "scale";
export const TABS: [Tab, string][] = [
  ["image", "image"], ["classes", "classes"], ["l1", "λ1"], ["l2", "λ2"], ["trace", "trace (LoG)"],
  ["det", "determinant"], ["vessel", "vesselness"], ["scale", "best σ"],
];

export interface HessState {
  mode: string;
  image: string;
  qa: number; qb: number; qc: number;          // quadratic image f = a x² + b x y + c y²
  noise: number;
  sigma: number;                               // single scale
  multi: boolean; sigmaMax: number;            // multi-scale: σ from 1 to sigmaMax (8 levels)
  t: number;                                   // flat threshold on |λ2| (σ²-normalised units)
  beta: number; c: number; bright: boolean;    // vesselness
  tab: Tab;
  px: number; py: number;                      // selected pixel (in units of a 128-pixel image; scaled)
}

export const BASE_STATE: HessState = {
  mode: "hessian", image: "scene", qa: -1, qb: 0, qc: -1, noise: 3, sigma: 3, multi: false, sigmaMax: 8,
  t: 4, beta: 0.5, c: 15, bright: true, tab: "classes", px: 28, py: 28,
};

export interface HessImage {
  id: string;
  label: string;
  /** Pixel size N of the square image (128 for synthetic ones, 256 for the micrograph). */
  size: number;
  /** The image (values about 0..255) for the state (noise, quadratic coefficients). */
  make(N: number, st: HessState): Grey | Promise<Grey>;
  credit?: string;
  /** Factor applied to the pixel Hessian at scale σ (default σ², scale normalisation). */
  units?(N: number, sigma: number): number;
}

export interface HessPreset { id: string; label: string; s: Partial<HessState> }

export interface HessMode {
  id: string;
  label: string;
  /** Image ids offered as buttons, in order. */
  images: string[];
  defaults: Partial<HessState>;
  presets: HessPreset[];
}

export const IMAGES = new Map<string, HessImage>();
export const MODES = new Map<string, HessMode>();
export function registerImage(im: HessImage): void { IMAGES.set(im.id, im); }
export function registerMode(m: HessMode): void { MODES.set(m.id, m); }

let cell256: Grey | null = null;
registerImage({ id: "quadratic", label: "f = ax² + bxy + cy²", size: 128, make: (N, st) => quadratic(N, st.qa, st.qb, st.qc), units: N => quadUnits(N) });
registerImage({ id: "scene", label: "test scene", size: 128, make: (N, st) => makeHessScene("scene", N, st.noise, 1).image });
registerImage({ id: "lines", label: "curved lines", size: 128, make: (N, st) => makeHessScene("vessels", N, st.noise, 1).image });
registerImage({
  id: "cell", label: "micrograph", size: 256,
  make: async (N, st) => {
    if (!cell256) cell256 = await loadCell(256);
    const g = N === 256 ? cell256 : await loadCell(N);
    return st.noise > 0 ? addGaussianNoise(g, st.noise, 1) : g;
  },
  credit: "cell",
});

registerMode({
  id: "hessian", label: "Hessian", images: ["quadratic", "scene", "lines", "cell"], defaults: {},
  presets: [
    { id: "hill", label: "hilltop: both negative", s: { image: "quadratic", qa: -1, qb: 0, qc: -1, noise: 0, tab: "image", px: 64, py: 64, sigma: 2, t: 0.5 } },
    { id: "ridge", label: "ridge: one eigenvalue", s: { image: "quadratic", qa: -1, qb: 0, qc: 0, noise: 0, tab: "image", px: 64, py: 64, sigma: 2, t: 0.5 } },
    { id: "saddle", label: "saddle", s: { image: "quadratic", qa: 1, qb: 0, qc: -1, noise: 0, tab: "image", px: 64, py: 64, sigma: 2, t: 0.5 } },
    { id: "edge", label: "an edge is not a ridge", s: { image: "scene", tab: "classes", px: 25, py: 114, sigma: 3 } },
    { id: "trace", label: "LoG cannot tell blob from ridge; det can", s: { image: "scene", tab: "trace", px: 28, py: 28, sigma: 3 } },
    { id: "vessel", label: "vesselness: blob suppressed", s: { image: "scene", tab: "vessel", px: 28, py: 28, sigma: 3 } },
    { id: "thick", label: "one σ misses the thick line", s: { image: "scene", tab: "vessel", px: 30, py: 70, sigma: 1.5 } },
    { id: "multi", label: "multi-scale", s: { image: "scene", tab: "vessel", multi: true, px: 30, py: 70 } },
    { id: "cell", label: "membranes in the micrograph", s: { image: "cell", noise: 0, tab: "vessel", bright: false, sigma: 2, c: 20, px: 47, py: 63 } },
  ],
});
