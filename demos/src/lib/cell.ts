// The course micrograph (demos/src/fourier2d/cell256.png, 256 × 256 grey), inlined into the
// demo bundle as a data URL so it also works from file://. Kept apart from imageops.ts
// because the PNG import needs esbuild's image loader, which the node tests do not use.
// DOM only.

// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";
import { Grey, loadGrey } from "./imageops";

export const CELL_URL: string = cellUrl;

/** Credit line to show under any demo that displays the micrograph. */
export const CELL_CREDIT =
  "Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons";

/** The micrograph as a Grey with values 0..255; `size` shrinks it by area averaging (e.g. 128). */
export function loadCell(size?: number): Promise<Grey> {
  return loadGrey(CELL_URL, size);
}
