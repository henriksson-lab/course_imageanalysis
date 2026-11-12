// Mode registry of the morphgrid demo. See the comment at the top of main.ts for how a
// day adds a mode. This file has no mode of its own.
import type { GridStepper, GridStepperOptions } from "../lib/gridui";
import type { ColorMap } from "../lib/colour";

/** An input preset of a mode: a grid (text rows, as gridFromText reads them, or values)
 *  plus optional settings (an algorithm to select, anything else in `apply`). */
export interface ModePreset {
  id: string;
  label: string;
  /** Text rows ("0110", or "3 1 2" with spaces) or explicit values with their size. */
  grid: string[] | { rows: number; cols: number; values: ArrayLike<number> };
  /** Id of the algorithm to select. */
  algorithm?: string;
  /** Set the mode's own widgets (SE, toggles) before the frames are generated. */
  apply?(): void;
  /** Frame to show after loading (default 0; "end" = the last frame). */
  frame?: number | "end";
}

/** What a mode gets when it is built. */
export interface ModeContext {
  /** The demo element; per-embed settings are in host.dataset (data-*). */
  host: HTMLElement;
  /** An empty column left of the grids for the mode's widgets (SE editor, toggles). It is
   *  hidden if the mode leaves it empty. */
  side: HTMLElement;
  /** The stepper (only valid after build() has returned, i.e. inside callbacks). */
  stepper(): GridStepper;
  /** Regenerate the frames after a setting changed and go back to step 0. */
  restart(): void;
}

/** What a mode returns: the stepper options (without `presets`, which the shell draws
 *  with a reset button) plus its presets. */
export interface ModeSpec extends Omit<GridStepperOptions, "presets" | "rows" | "cols" | "initial"> {
  presets: ModePreset[];
  /** Colour map of the main grid for "int"/"float" display (t = (v − lo)/(hi − lo)). */
  map?: ColorMap;
  /** Called once after the stepper exists (e.g. to add controls to stepper.panel). */
  ready?(s: GridStepper): void;
}

export interface MorphGridMode {
  /** Value of data-mode that selects this mode. */
  id: string;
  build(ctx: ModeContext): ModeSpec;
}

const MODES = new Map<string, MorphGridMode>();

/** Register a mode (call at module load, from a file imported by main.ts). */
export function registerMode(m: MorphGridMode): void {
  MODES.set(m.id, m);
}

export function getMode(id: string): MorphGridMode | undefined {
  return MODES.get(id);
}

export function modeIds(): string[] {
  return [...MODES.keys()];
}
