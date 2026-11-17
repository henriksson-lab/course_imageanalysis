// Presets of the maskfill demo (no DOM, so the tests and figure scripts can use them).
import type { MaskKind, Scene } from "./model";

export interface Settings { scene: Scene; kind: MaskKind; ratio: number; k: number; seed: number }
export interface Preset { id: string; label: string; s: Settings }
export const DEFAULT: Settings = { scene: "nuclei", kind: "patches", ratio: 0.75, k: 32, seed: 7 };
export const PRESETS: Preset[] = [
  { id: "p25", label: "25 % of the patches hidden", s: { ...DEFAULT, ratio: 0.25 } },
  { id: "p75", label: "75 % hidden", s: { ...DEFAULT } },
  { id: "p90", label: "90 % hidden", s: { ...DEFAULT, ratio: 0.9 } },
  { id: "pixels", label: "75 % of the pixels: easy", s: { ...DEFAULT, kind: "pixels" } },
  { id: "k0", label: "k = 0: the average window", s: { ...DEFAULT, k: 0 } },
  { id: "cell", label: "micrograph: texture", s: { ...DEFAULT, scene: "cell" } },
];
