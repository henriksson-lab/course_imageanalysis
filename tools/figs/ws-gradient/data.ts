// Data for the ws-gradient figures, computed with the watershed demo's own pipeline
// (demos/src/watershed/pipeline.ts and images.ts). Args: out.json cell.json
// (cell.json: the micrograph cell256.png as 256*256 grey values 0..255, written by figs.py).
import { writeFileSync, readFileSync } from "fs";
import { synth, N, CLICKS_LEAK } from "../../../demos/src/watershed/images";
import { landscape, runWatershed } from "../../../demos/src/watershed/pipeline";
import { fromArray } from "../../../demos/src/lib/imageops";

const img = synth(8);
const r8 = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 10) / 10);
const out: any = { n: N, image: r8(img.data) };
const land1 = landscape(img, "gradient", 1), land15 = landscape(img, "gradient", 1.5), land4 = landscape(img, "gradient", 4);
const raw = runWatershed(land1, N, N, "minima", 0, []), sm = runWatershed(land4, N, N, "minima", 0, []);
out.raw = { labels: Array.from(raw.labels), n: raw.nRegions };
out.smooth = { labels: Array.from(sm.labels), n: sm.nRegions };
out.grad15 = r8(land15);
out.hsweep = [] as [number, number][];
for (let h = 0; h <= 8.001; h += 0.25) out.hsweep.push([h, runWatershed(land15, N, N, "hmin", h, []).nRegions]);
const h3 = runWatershed(land15, N, N, "hmin", 3, []);
out.h3 = { labels: Array.from(h3.labels), n: h3.nRegions };
const leakClicks = CLICKS_LEAK.map(([x, y, label]) => ({ x, y, label, r: label === 1 ? 5 : 3 }));
const leak = runWatershed(land15, N, N, "clicked", 0, leakClicks);
out.leak = { labels: Array.from(leak.labels), clicks: leakClicks };
// the micrograph preset "nucleus + background clicks"
const cellVals: number[] = JSON.parse(readFileSync(process.argv[3], "utf8"));
const cell = fromArray(N, N, cellVals);
const cc: [number, number, number][] = [[160, 150, 2], [95, 60, 1], [165, 40, 1], [230, 120, 1], [215, 240, 1], [110, 230, 1], [80, 150, 1]];
const cellClicks = cc.map(([x, y, label]) => ({ x, y, label, r: label === 1 ? 5 : 3 }));
const cr = runWatershed(landscape(cell, "gradient", 2), N, N, "clicked", 0, cellClicks);
out.cell = { labels: Array.from(cr.labels), clicks: cellClicks };
writeFileSync(process.argv[2], JSON.stringify(out));
