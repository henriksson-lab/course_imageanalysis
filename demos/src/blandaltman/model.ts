// Model of the blandaltman demo (no DOM): 40 images with true counts 10–150, a manual count
// (annotator A) and an automated count from a model with a scale factor, a constant offset,
// an undercount that grows with density (merges) and random scatter. Or, in "areas" mode,
// one point per object: true radius 2.5–15 px, the automated area from an outline that is
// `outline` px too large or too small. A second annotator B replaces the method when
// `annotator` is on.
import { rng, gauss } from "../lib/random";

export interface Settings {
  scale: number;       // auto ≈ scale × manual
  offset: number;      // + constant
  density: number;     // undercount fraction at 150 objects (grows linearly with the count)
  scatter: number;     // SD of random errors (counts)
  areas: boolean;      // object areas instead of counts per image
  outline: number;     // areas: outline offset in px (+ = too large)
  annotator: boolean;  // compare annotator B with annotator A instead of the method
}

export const DEFAULTS: Settings = { scale: 1, offset: 0, density: 0.1, scatter: 3, areas: false, outline: 0.5, annotator: false };

export interface Data { manual: Float64Array; auto: Float64Array; xlabel: string; ylabel: string; unit: string }

const N_IMG = 40, N_OBJ = 150;

export function makeData(s: Settings): Data {
  if (s.areas) {
    const r = rng(31), manual = new Float64Array(N_OBJ), auto = new Float64Array(N_OBJ);
    for (let i = 0; i < N_OBJ; i++) {
      const rad = 2.5 + 12.5 * r.unif();
      const ea = gauss(r, 0, 0.25), eb = gauss(r, 0, 0.25), em = gauss(r, 0, 0.25);
      manual[i] = Math.PI * (rad + ea) ** 2;
      auto[i] = s.annotator ? Math.PI * (rad + eb) ** 2 : Math.PI * Math.max(0.5, rad + s.outline + em) ** 2;
    }
    return { manual, auto, xlabel: "annotator A area (px²)", ylabel: s.annotator ? "annotator B area (px²)" : "automated area (px²)", unit: "px²" };
  }
  const r = rng(17), manual = new Float64Array(N_IMG), auto = new Float64Array(N_IMG);
  for (let i = 0; i < N_IMG; i++) {
    const t = Math.round(10 + 140 * r.unif());
    const za = r.norm(), zb = r.norm(), zm = r.norm();
    manual[i] = Math.max(0, Math.round(t + 1.5 * za));
    if (s.annotator) auto[i] = Math.max(0, Math.round(t + 1.5 * zb));
    else auto[i] = Math.max(0, Math.round(s.scale * t * (1 - s.density * t / 150) + s.offset + s.scatter * zm));
  }
  return { manual, auto, xlabel: "manual count (annotator A)", ylabel: s.annotator ? "annotator B count" : "automated count", unit: "objects" };
}

export interface Preset { id: string; label: string; settings: Partial<Settings> }

export const PRESETS: Preset[] = [
  { id: "default", label: "a typical method", settings: {} },
  { id: "corr", label: "r = 0.99, still 20 % off", settings: { scale: 0.8, density: 0, scatter: 2 } },
  { id: "prop", label: "merges at high density", settings: { density: 0.3, scatter: 2 } },
  { id: "scatter", label: "scatter only", settings: { density: 0, scatter: 9 } },
  { id: "areas", label: "small objects: area bias", settings: { areas: true, outline: 1 } },
  { id: "annotator", label: "two annotators", settings: { annotator: true } },
];
