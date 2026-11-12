// Painted-stroke presets of the pixel-classification demo (no DOM; also used by the tests
// and the solutions scripts). Strokes are polylines in the coordinates of a 256 × 256 image
// with a brush radius; label values: 1 = background, 2 = nucleus, 3 = granular cell (test
// scene) or cytoplasm (micrograph).
import { paintSegment } from "../lib/paint";
import { Scene } from "./scene";
import { ellipseDistance, Ellipse } from "../lib/synth";

export interface Stroke { label: number; pts: [number, number][]; r: number }

/** Held-out region (x from this fraction of the width to the right edge). */
export const HOLDOUT_X = 0.625;

export type Families = string[];
export const ALL_FAMILIES = ["intensity", "edges", "blobs", "ridges", "texture"];
export const SMALL = [0.7, 1, 1.6], LARGE = [3.5, 5, 10];

export interface PixelPreset {
  id: string;
  label: string;
  image: "synth" | "cell";
  strokes: (sc: Scene) => Stroke[];
  families: Families;
  small: boolean; large: boolean;
  balance: boolean;
  heldout: boolean;
  view: "seg" | "prob" | "unc" | "feature" | "truth";
}

/** Rasterise strokes into a label map of size w × h (stroke coordinates are for 256 × 256). */
export function strokesToMap(strokes: Stroke[], w: number, h: number): Uint8Array {
  const m = new Uint8Array(w * h), s = w / 256;
  for (const st of strokes) {
    const p = st.pts;
    if (p.length === 1) paintSegment(m, w, h, p[0][0] * s, p[0][1] * s, p[0][0] * s, p[0][1] * s, st.r * s, st.label);
    for (let i = 0; i + 1 < p.length; i++) paintSegment(m, w, h, p[i][0] * s, p[i][1] * s, p[i + 1][0] * s, p[i + 1][1] * s, st.r * s, st.label);
  }
  return m;
}

/** A stroke along the long axis of an ellipse (the middle `frac` of it). */
function through(e: Ellipse, label: number, frac = 0.6, r = 2): Stroke {
  const c = Math.cos(e.angle) * e.a * frac, s = Math.sin(e.angle) * e.a * frac;
  return { label, pts: [[e.x - c, e.y - s], [e.x + c, e.y + s]], r };
}

/** Horizontal background strokes of length `len` that keep `margin` px from every object, left of xmax. */
function backgroundLines(sc: Scene, count: number, len: number, xmin: number, xmax: number, r = 2, margin = 8): Stroke[] {
  const out: Stroke[] = [];
  const free = (x: number, y: number) => sc.ellipses.every(e => ellipseDistance({ ...e, a: e.a + margin, b: e.b + margin }, x, y) > 1);
  for (let y = 20; y < 236 && out.length < count; y += 37) {
    for (let x = xmin; x + len < xmax && out.length < count; x += 6) {
      let ok = true;
      for (let t = 0; t <= len && ok; t += 2) if (!free(x + t, y)) ok = false;
      if (ok) { out.push({ label: 1, pts: [[x, y], [x + len, y]], r }); break; }
    }
  }
  return out;
}

const xLimit = 256 * HOLDOUT_X;
const left = (sc: Scene, kind: string) => sc.ellipses.filter((e, i) => sc.kinds[i] === kind && e.x + e.a < xLimit - 2);
const right = (sc: Scene, kind: string) => sc.ellipses.filter((e, i) => sc.kinds[i] === kind && e.x - e.a > xLimit + 2);

/** The base set: two background strokes, a bright and a dim nucleus, a granular cell (all left of the held-out region). */
function base(sc: Scene, o: { dim?: boolean; granular?: boolean } = {}): Stroke[] {
  const s: Stroke[] = [...backgroundLines(sc, 2, 34, 4, xLimit - 4)];
  s.push(through(left(sc, "bright")[0], 2));
  if (o.dim !== false && left(sc, "dim").length) s.push(through(left(sc, "dim")[0], 2));
  if (o.granular !== false) s.push(through(left(sc, "granular")[0], 3));
  return s;
}

/** Test strokes inside the held-out region (excluded from training, used to score it). */
function heldoutStrokes(sc: Scene): Stroke[] {
  const s: Stroke[] = [...backgroundLines(sc, 2, 30, xLimit + 4, 252)];
  for (const k of ["bright", "dim"]) { const e = right(sc, k)[0]; if (e) s.push(through(e, 2)); }
  const g = right(sc, "granular")[0]; if (g) s.push(through(g, 3));
  return s;
}

// micrograph strokes (256 × 256 coordinates, drawn by hand on cell256.png)
const CELL_NUC: Stroke[] = [{ label: 2, pts: [[128, 85], [150, 110], [172, 135]], r: 3 }];
const CELL_BG: Stroke[] = [
  { label: 1, pts: [[18, 60], [40, 95], [52, 128]], r: 3 },
  { label: 1, pts: [[205, 215], [240, 230]], r: 3 },
  { label: 1, pts: [[200, 20], [235, 30]], r: 3 },
];
const CELL_CYTO: Stroke[] = [
  { label: 3, pts: [[66, 10], [82, 28], [96, 44]], r: 2 },
  { label: 3, pts: [[128, 196], [142, 212]], r: 2 },
];

export const PIXEL_PRESETS: PixelPreset[] = [
  { id: "two", label: "two strokes", image: "synth", strokes: sc => [backgroundLines(sc, 1, 34, 4, xLimit - 4)[0], through(left(sc, "bright")[0], 2)], families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg" },
  { id: "intensity", label: "intensity only", image: "synth", strokes: sc => base(sc), families: ["intensity"], small: true, large: true, balance: false, heldout: false, view: "seg" },
  { id: "full", label: "full stack", image: "synth", strokes: sc => base(sc), families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg" },
  { id: "small", label: "small scales only", image: "synth", strokes: sc => base(sc), families: ALL_FAMILIES, small: true, large: false, balance: false, heldout: false, view: "seg" },
  { id: "bright", label: "only bright nuclei painted", image: "synth", strokes: sc => base(sc, { dim: false }), families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg" },
  {
    id: "imbalance", label: "background 20× larger", image: "synth", families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg",
    strokes: sc => [
      ...backgroundLines(sc, 6, 60, 4, 252, 6, 7),
      { ...through(left(sc, "bright")[0], 2, 0.3, 1) }, { ...through(left(sc, "dim")[0] ?? left(sc, "bright")[1], 2, 0.3, 1) }, { ...through(left(sc, "granular")[0], 3, 0.3, 1) },
    ],
  },
  { id: "heldout", label: "OOB vs held-out region", image: "synth", strokes: sc => [...base(sc, { dim: false }), ...heldoutStrokes(sc)], families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: true, view: "seg" },
  { id: "uncertain", label: "uncertainty: where to paint", image: "synth", strokes: sc => base(sc, { granular: false }), families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "unc" },
  { id: "cell", label: "micrograph: nucleus", image: "cell", strokes: () => [...CELL_NUC, ...CELL_BG], families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg" },
  { id: "cellcyto", label: "micrograph: + cytoplasm", image: "cell", strokes: () => [...CELL_NUC, ...CELL_BG, ...CELL_CYTO], families: ALL_FAMILIES, small: true, large: true, balance: false, heldout: false, view: "seg" },
];
