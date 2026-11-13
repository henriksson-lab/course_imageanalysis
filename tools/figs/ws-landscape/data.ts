// Data for the ws-landscape figures, from the demos' own code (run by figs.py via esbuild + node).
import { writeFileSync } from "fs";
import { twoCells } from "../../../demos/src/flood1d/f1d";
import { minimaMarkers, floodWatershed } from "../../../demos/src/lib/watershed";
import { nuclei } from "../../../demos/src/lib/synth";
import { label2d } from "../../../demos/src/lib/label";

const flood = (f: number[]) => {
  const mk = minimaMarkers(f, f.length, 1, 4);
  return { f, markers: Array.from(mk.data), nMinima: mk.count, labels: Array.from(floodWatershed(f, f.length, 1, mk.data, { conn: 4 }).labels) };
};
// the distws demo's nuclei image (N = 180, seed 4), threshold 115
const N = 180, nuc = nuclei({ width: N, height: N, count: 15, radius: [10, 15], touching: 0.6, seed: 4, noise: 6 });
const mask = Uint8Array.from(nuc.image.data, v => (v > 115 ? 1 : 0));
const cc = label2d(mask, N, N, 8);
writeFileSync(process.argv[2], JSON.stringify({
  clean: flood(twoCells()), noisy: flood(twoCells(0.35)),
  nuc: { n: N, image: Array.from(nuc.image.data), labels: Array.from(cc.data), count: cc.count, truth: nuc.count },
}));
