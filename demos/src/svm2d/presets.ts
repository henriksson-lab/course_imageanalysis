// Presets of the svm2d demo (no DOM; also used by the tests and the solution scripts).
// Points are in world coordinates; class 1 = +1 (red), class 0 = −1 (blue).
import type { LPoint } from "../lib/datasets2d";
import { blobs, moons, rings, xor, withLabelNoise } from "../lib/datasets2d";
import type { KernelType } from "../lib/svmkernels";

export interface SVMPreset {
  id: string;
  label: string;
  /** Which unit's list it belongs to by default. */
  group: "linear" | "kernel" | "cv";
  range: [number, number];
  kernel: KernelType;
  /** log10 C, log10 γ, polynomial degree. */
  logC: number;
  logGamma: number;
  degree: number;
  points: (seed?: number) => LPoint[];
  /** Generator of a test set (cross-validation mode); default: points(seed + 100). */
  note: string;
}

const toRange = (pts: LPoint[], lo: number, hi: number): LPoint[] => pts.map(p => ({ x: lo + (hi - lo) * p.x, y: lo + (hi - lo) * p.y, c: p.c }));

/** The by-hand example of the max-margin unit. */
export const BY_HAND_POINTS: LPoint[] = [{ x: 0, y: 0, c: 0 }, { x: 0, y: 2, c: 0 }, { x: 2, y: 1, c: 1 }, { x: 3, y: 0, c: 1 }];

export const SVM_PRESETS: SVMPreset[] = [
  // ---- linear (max-margin unit)
  { id: "byhand", label: "the by-hand example", group: "linear", range: [-1, 4], kernel: "linear", logC: 3, logGamma: 0, degree: 2, points: () => BY_HAND_POINTS.map(p => ({ ...p })), note: "Max-margin line x₁ = 1, street width 2; three support vectors. Drag (3, 0) around: nothing changes until it enters the street." },
  {
    id: "nonsv", label: "move a non-support vector", group: "linear", range: [-1, 4], kernel: "linear", logC: 3, logGamma: 0, degree: 2,
    points: () => [...BY_HAND_POINTS, { x: -0.5, y: 3.2, c: 0 }, { x: -0.6, y: 0.8, c: 0 }, { x: 3.4, y: 2.4, c: 1 }, { x: 2.6, y: 3.3, c: 1 }].map(p => ({ ...p })),
    note: "Drag any point without a ring: the line does not move. Drag a ringed point (a support vector): it does.",
  },
  {
    id: "outlier", label: "one outlier: hard vs soft", group: "linear", range: [0, 10], kernel: "linear", logC: 3, logGamma: 0, degree: 2,
    points: () => [...toRange(blobs([[0.3, 0.4], [0.7, 0.6]], { sd: 0.07, n: 40, seed: 21 }).points, 0, 10), { x: 4.3, y: 4.6, c: 1 }],
    note: "With C = 1000 the street squeezes past the outlier. Lower C: the outlier is given up and the street widens.",
  },
  {
    id: "overlap", label: "overlapping classes: sweep C", group: "linear", range: [0, 10], kernel: "linear", logC: 0, logGamma: 0, degree: 2,
    points: () => toRange(blobs([[0.4, 0.42], [0.6, 0.58]], { sd: 0.1, n: 60, seed: 22 }).points, 0, 10),
    note: "No line separates these. Sweep C: small C = wide street, many support vectors; large C = narrow street.",
  },
  {
    id: "narrow", label: "C large = narrow street", group: "linear", range: [0, 10], kernel: "linear", logC: 3, logGamma: 0, degree: 2,
    points: () => [...toRange(blobs([[0.3, 0.35], [0.7, 0.65]], { sd: 0.07, n: 40, seed: 23 }).points, 0, 10), { x: 5.6, y: 4.2, c: 0 }, { x: 4.4, y: 5.9, c: 1 }],
    note: "Two points near the middle force a narrow street at large C. Lower C: the street widens and lets them inside.",
  },
  // ---- kernel functions (kernel-trick unit)
  {
    id: "ring", label: "ring: linear fails, RBF works", group: "kernel", range: [-5, 5], kernel: "linear", logC: 1, logGamma: -0.5, degree: 2,
    points: s => toRange(rings({ n: 80, seed: s ?? 31, radii: [0.12, 0.34] }).points, -5, 5),
    note: "Switch from linear to RBF (or polynomial, degree 2): a closed boundary around the inner class.",
  },
  {
    id: "xor", label: "XOR: degree 2", group: "kernel", range: [-5, 5], kernel: "poly", logC: 1, logGamma: -0.5, degree: 2,
    points: s => toRange(xor({ n: 80, seed: s ?? 32 }).points, -5, 5),
    note: "Degree 2 contains x₁x₂, the product that XOR needs. Compare with the linear kernel function.",
  },
  {
    id: "islands", label: "γ too large: islands", group: "kernel", range: [-5, 5], kernel: "rbf", logC: 2, logGamma: 1.3, degree: 2,
    points: s => toRange(withLabelNoise(moons({ n: 80, seed: s ?? 33, noise: 0.04 }), 0.08, 3).points, -5, 5),
    note: "Length scale 1/√γ ≈ 0.2: every training point gets its own island. Lower γ.",
  },
  {
    id: "smallgamma", label: "γ too small: a line again", group: "kernel", range: [-5, 5], kernel: "rbf", logC: 0, logGamma: -2, degree: 2,
    points: s => toRange(moons({ n: 80, seed: s ?? 33, noise: 0.04 }).points, -5, 5),
    note: "Length scale 1/√γ = 10, wider than the data: the boundary is almost straight. Raise γ.",
  },
  {
    id: "both", label: "C and γ both large", group: "kernel", range: [-5, 5], kernel: "rbf", logC: 3, logGamma: 0.7, degree: 2,
    points: s => toRange(withLabelNoise(moons({ n: 80, seed: s ?? 34, noise: 0.05 }), 0.1, 4).points, -5, 5),
    note: "Zero training errors, and a boundary that bends around every mislabelled point.",
  },
  // ---- cross-validation (pixel-classify unit)
  {
    id: "cv", label: "noisy moons: train vs test", group: "cv", range: [-5, 5], kernel: "rbf", logC: 0, logGamma: 0, degree: 2,
    points: s => toRange(withLabelNoise(moons({ n: 100, seed: s ?? 35, noise: 0.05 }), 0.1, s ?? 35).points, -5, 5),
    note: "Hollow points: a test set from the same source. Training error falls with C or γ; test error has a minimum.",
  },
  {
    id: "cvblobs", label: "overlapping blobs: train vs test", group: "cv", range: [-5, 5], kernel: "rbf", logC: 0, logGamma: -1, degree: 2,
    points: s => toRange(blobs([[0.4, 0.42], [0.6, 0.58]], { sd: 0.11, n: 100, seed: s ?? 36 }).points, -5, 5),
    note: "The best boundary is a straight line: a large γ only fits noise.",
  },
];

export const presetPoints = (p: SVMPreset, seed?: number): LPoint[] => p.points(seed);

/** Test set of a preset (the same generator, another seed). */
export const presetTest = (p: SVMPreset): LPoint[] => p.points(1000 + p.id.length * 7);
