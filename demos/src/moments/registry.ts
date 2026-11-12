// Mode registry of the moments demo (the course's shape-editing shell). See the comment at
// the top of main.ts for how another day adds a mode (the moments elective's "higher" and
// "invariants" modes, the measurements day's feature views).
import type { ShapeCanvas, Poly } from "../lib/shapecanvas";

/** The digitised shape, as every mode sees it. */
export interface Digitised {
  /** The polygons in canvas units (y down). */
  polys: Poly[];
  /** Binary mask on the current pixel grid (w × h), after the mode-independent shell steps. */
  mask: Uint8Array;
  w: number;
  h: number;
  /** Size of one pixel in canvas units, in x and in y (y differs for anisotropic pixels). */
  px: number;
  py: number;
  /** True after a pointer drag ends or a button was used (expensive extras may run). */
  final: boolean;
}

export interface MomentsPreset {
  id: string;
  label: string;
  /** Shapes in canvas units (the canvas is `size` units wide; see main.ts). */
  shapes: (size: number) => Poly[];
  /** Pixel size in canvas units (default 1). */
  pixel?: number;
  /** Set the mode's own widgets. */
  apply?(): void;
}

export interface MomentsContext {
  host: HTMLElement;
  canvas: ShapeCanvas;
  /** SVG group above the shapes, in canvas units (one unit = one canvas pixel). */
  overlay: SVGGElement;
  /** Column for the mode's readout and plots. */
  out: HTMLElement;
  /** Column for the mode's controls (above the shell's shape and preset buttons). */
  controls: HTMLElement;
  /** Size of the canvas in canvas units. */
  size: number;
  /** Recompute (call after a mode setting changed). */
  refresh(): void;
  /** Change the pixel size (slider value) from the mode. */
  setPixel(p: number): void;
  /** Pixel height / pixel width (1 = square pixels, 2 = rows twice as far apart). */
  setAnisotropy(a: number): void;
}

export interface MomentsModeSpec {
  presets: MomentsPreset[];
  /** Called after every change of the shape, the pixel size or a setting. */
  update(d: Digitised): void;
  /** Optional: change the mask before update (e.g. a ± mask error); default identity. */
  transformMask?(d: Digitised): Digitised;
}

export interface MomentsMode {
  id: string;
  build(ctx: MomentsContext): MomentsModeSpec;
}

const MODES = new Map<string, MomentsMode>();

export function registerMode(m: MomentsMode): void { MODES.set(m.id, m); }
export function getMode(id: string): MomentsMode | undefined { return MODES.get(id); }
