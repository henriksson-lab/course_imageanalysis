// The three test images of the live-wire and fast-marching demos (DOM: the micrograph is
// decoded from the inlined PNG). All 256 × 256, grey levels 0..255, cached.
import { Grey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { nucleusImage, NUCLEUS } from "../polarseg/model";
import { neuriteImage, Neurite } from "./model";

export type ImageId = "cell" | "nucleus" | "neurite";
export const IMAGE_LABELS: [ImageId, string][] = [["cell", "micrograph"], ["nucleus", "nuclei"], ["neurite", "neuron"]];
export { CELL_CREDIT };

const cache = new Map<ImageId, Promise<Grey>>();
let neurite: Neurite | null = null;

export function neuron(): Neurite {
  if (!neurite) neurite = neuriteImage();
  return neurite;
}

export function getImage(id: ImageId): Promise<Grey> {
  let p = cache.get(id);
  if (!p) {
    p = id === "cell" ? loadCell() : Promise.resolve(id === "nucleus" ? nucleusImage() : neuron().image);
    cache.set(id, p);
  }
  return p;
}

/** A point on the boundary of the main synthetic nucleus at angle θ (degrees, clockwise on screen from →). */
export function nucleusPoint(deg: number): [number, number] {
  const N = NUCLEUS, a = deg * Math.PI / 180, da = a - N.angle;
  const r = 1 / Math.sqrt((Math.cos(da) / N.a) ** 2 + (Math.sin(da) / N.b) ** 2);
  return [Math.round(N.cx + r * Math.cos(a)), Math.round(N.cy + r * Math.sin(a))];
}
