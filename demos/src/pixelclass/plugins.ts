// Plug-in types for the pixelclass shell (no DOM at import). See the header of main.ts for how
// other days add a classifier, a tab or a feature family.
import type { Classifier, Model, TrainingSet } from "../lib/classifier";
import type { FeatureStack } from "../lib/featurestack";

/** What the shell offers to a plug-in tab. */
export interface ShellContext {
  /** The feature stack with the currently selected families and scales (null while computing). */
  stack(): FeatureStack | null;
  /** The current training set (painted pixels outside the held-out region; groups = stroke ids). */
  training(): TrainingSet | null;
  /** The classifier in use and its parameter values (slider units). */
  classifier(): Classifier;
  params(): Record<string, number>;
  /** Set parameter values (slider units) and retrain. */
  setParams(p: Record<string, number>): void;
  /** The trained model (null before training). */
  model(): Model | null;
  /** Class names of the current image. */
  classNames(): string[];
  /** Image size. */
  width(): number;
  height(): number;
  /** Retrain with the current settings. */
  retrain(): void;
  /** Called after every training (register to update the tab). */
  onTrained(fn: () => void): void;
}

/** An extra section of the control panel (e.g. the SVM day's cross-validation tab). */
export interface PixelclassTab {
  id: string;
  label: string;
  mount(parent: HTMLElement, ctx: ShellContext): void;
}

const TABS = new Map<string, PixelclassTab>();
export function registerPixelclassTab(t: PixelclassTab): void { TABS.set(t.id, t); }
export function getPixelclassTab(id: string): PixelclassTab | undefined { return TABS.get(id); }
