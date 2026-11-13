// Arrival times for fig-arrival (fastmarching/lec.qmd), computed with the demo's fast marching
// (demos/src/lib/shortestpath.ts). Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/fastmarching/data.ts --bundle --platform=node --outfile=<tmp>/fm-data.js
//   node <tmp>/fm-data.js <tmp>/fm-data.json
import { writeFileSync } from "fs";
import { fastMarching, seedNeighbourhood } from "../../../demos/src/lib/shortestpath";

const W = 240, H = 150;
const P = new Float64Array(W * H);
const disc = { x: 120, y: 75, r: 32 }, slow = 4;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
  P[y * W + x] = Math.hypot(x - disc.x, y - disc.y) <= disc.r ? slow : 1;
const seed = 75 * W + 30;
const { T } = fastMarching(W, H, P, seedNeighbourhood(W, H, P, [seed], 3));
writeFileSync(process.argv[2], JSON.stringify({ W, H, disc, slow, seed: [30, 75], P: Array.from(P), T: Array.from(T) }));
