// The ray of the raycomposite demo (no DOM; used by main.ts and by the exercise scripts).
import { TF } from "../lib/volrender";
import { trilinear, Volume } from "../lib/volume";

/** The sagittal slice x = 32 mm: horizontal axis z (0..64 mm), vertical axis y (down). */
export const X0 = 32;

export const WALL: TF = { points: [[110, 0], [135, 0.9], [255, 0.9]], map: "bone" };
export const FOG: TF = { points: [[40, 0], [60, 0.05], [255, 0.05]], map: "bone" };
export const LAYERS: TF = { points: [[60, 0], [70, 0.04], [110, 0.04], [140, 0], [165, 0.15], [200, 0.15], [222, 0.9]], map: "inferno" };

export interface State { a: [number, number]; b: [number, number]; ds: number; tf: TF; order: "ftb" | "btf"; mip: boolean; early: boolean; correction: boolean }
export const DEF: State = { a: [62, 10], b: [8, 50], ds: 1, tf: LAYERS, order: "ftb", mip: false, early: false, correction: true };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "wall", label: "opaque wall hides what is behind", s: { tf: WALL, early: true } },
  { id: "fog", label: "fog", s: { tf: FOG } },
  { id: "layers", label: "two layers", s: { tf: LAYERS } },
  { id: "mip", label: "MIP vs DVR on the same ray", s: { tf: LAYERS, mip: true } },
];


/** Value at slice position (z mm, y mm), trilinear in the volume (0 outside). */
export const at = (vol: Volume, z: number, y: number): number => trilinear(vol, X0, y, z / vol.spacing[2], 0);

/** Samples every Δs from the eye end a to b, the first at Δs/2: distances, values, positions. */
export function raySamples(vol: Volume, a: [number, number], b: [number, number], ds: number): { t: number[]; v: number[]; pts: [number, number][] } {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const t: number[] = [], v: number[] = [], pts: [number, number][] = [];
  for (let s = 0.5 * ds; s < L; s += ds) {
    const z = a[0] + (b[0] - a[0]) * s / L, y = a[1] + (b[1] - a[1]) * s / L;
    t.push(s); v.push(at(vol, z, y)); pts.push([z, y]);
  }
  return { t, v, pts };
}
