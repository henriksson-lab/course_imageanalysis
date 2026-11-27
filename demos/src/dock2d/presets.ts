// Presets of the dock2d demo (no DOM; the test checks that each does what its label says).
import { Settings, TRUE_POSE } from "./model";

export interface Preset { id: string; label: string; s: Partial<Settings> }

export const PRESETS: Preset[] = [
  { id: "key", label: "key and lock", s: {} },
  { id: "angle", label: "the right angle, the wrong angle", s: { x: TRUE_POSE.x, y: TRUE_POSE.y - 1, angle: 110 } },
  { id: "mild", label: "ρ too mild: the ligand sinks in", s: { rho: -0.1 } },
  { id: "thin", label: "surface too thin: no contact signal", s: { thickness: 0 } },
  { id: "coarse", label: "coarse angle step misses the fit", s: { step: 30 } },
  { id: "bump", label: "a flexible bump: the true fit clashes", s: { bump: true, x: TRUE_POSE.x, y: TRUE_POSE.y, angle: TRUE_POSE.angle } },
];
