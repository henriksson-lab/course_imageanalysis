// Data for the figures of mst-seg/, computed with demos/src/lib/mst.ts. Run from the repository root:
//   python3 -I tools/figs/random-walker/figs.py <rawdir> --export-cell
//   node_modules/.bin/esbuild tools/figs/mst-seg/compute.ts --bundle --platform=node --log-level=warning > <rawdir>/mst.js && node <rawdir>/mst.js <rawdir>
//   python3 -I tools/figs/mst-seg/figs.py <rawdir>
import * as M from "../../../demos/src/lib/mst";
import { gaussianBlur, fromArray } from "../../../demos/src/lib/imageops";
import { fhScene } from "../../../demos/src/fhseg/scenes";

declare const process: { argv: string[] };
declare function require(m: string): { readFileSync(p: string, e: string): string; writeFileSync(p: string, d: string): void };
const fs = require("fs");
const raw = process.argv[2];
const out: Record<string, unknown> = {};
const cell = Float64Array.from(JSON.parse(fs.readFileSync(`${raw}/cell256.json`, "utf8")) as number[]);
const blur = (a: Float64Array, s: number) => Float64Array.from(gaussianBlur(fromArray(256, 256, a), s).data);

// MST of a 16 × 16 crop (no smoothing)
{
  const x0 = 120, y0 = 120, n = 16, crop = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) crop[y * n + x] = cell[(y0 + y) * 256 + x0 + x];
  const G = M.gridDissim([crop], n, n, 4), K = M.kruskal(G);
  out.crop = { I: Array.from(crop), p: Array.from(G.p), q: Array.from(G.q), d: Array.from(G.d), tree: Array.from(K.inTree) };
}
// micrograph: single linkage at three thresholds, FH at three k (σ = 0.8, 8 neighbours)
{
  const G = M.gridDissim([blur(cell, 0.8)], 256, 256, 8), ord = M.sortEdges(G);
  const sl: Record<string, unknown> = {}, fh: Record<string, unknown> = {};
  for (const t of [3, 6, 12]) { const r = M.singleLinkage(G, t); sl[t] = { labels: Array.from(r.labels), count: r.count }; }
  for (const k of [50, 300, 1500]) { const r = M.felzenszwalb(G, k, 20, ord); fh[k] = { labels: Array.from(r.labels), count: r.count }; }
  out.cell = { I: Array.from(cell, v => Math.round(v)), sl, fh };
}
// the test scene: single linkage t = 5 and 9, FH k = 3000 and 10000
{
  const s = fhScene(4);
  const G = M.gridDissim([blur(s.I, 0.8)], 256, 256, 8), ord = M.sortEdges(G);
  const res: Record<string, unknown> = {};
  for (const t of [5, 9]) { const r = M.singleLinkage(G, t); res["sl" + t] = { labels: Array.from(r.labels), count: r.count }; }
  for (const k of [300, 3000, 10000]) { const r = M.felzenszwalb(G, k, 0, ord); res["fh" + k] = { labels: Array.from(r.labels), count: r.count }; }
  out.scene = { I: Array.from(s.I, v => Math.round(v)), truth: Array.from(s.truth), ...res };
  console.log("scene counts", Object.entries(res).map(([k, v]) => `${k}:${(v as { count: number }).count}`).join(" "));
}
// the 1D example
{
  const b = [0, 6, 0, 6, 20, 20, 20, 20, 24, 24, 24, 24];
  const G = M.gridDissim([b], 12, 1);
  out.oned = { b, sl5: Array.from(M.singleLinkage(G, 5).labels), sl7: Array.from(M.singleLinkage(G, 7).labels), fh8: Array.from(M.felzenszwalb(G, 8).labels) };
}
fs.writeFileSync(`${raw}/mst.json`, JSON.stringify(out));
console.log("cell counts", Object.entries(out.cell as Record<string, Record<string, { count: number }>>).filter(([k]) => k !== "I").map(([k, v]) => k + " " + Object.entries(v).map(([a, b]) => `${a}:${b.count}`).join(",")).join(" | "));
