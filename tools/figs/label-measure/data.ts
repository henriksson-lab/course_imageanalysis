// Data for the label-measure figures, from the regionfilter demo's pipeline (synthetic
// "mixed population", seed 3, 256 x 256). Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { mixedPopulation, runPipeline, areaOpening, PipelineOptions } from "../../../demos/src/regionfilter/pipeline";

const BASE: PipelineOptions = { threshold: 100, dark: false, openRadius: 0, fill: true, conn: 8, border: "keep", frameMargin: 20 };
const syn = mixedPopulation(3, 256);
const W = 256, H = 256;
const out: Record<string, unknown> = { image: Array.from(syn.image.data), truth: Array.from(syn.truth), kinds: syn.kinds };
const run = (o: Partial<PipelineOptions>) => {
  const r = runPipeline(syn.image, { ...BASE, ...o });
  // kind of each measured object: the kinds of the true objects it overlaps (by pixel count)
  const rows = r.rows.map(row => {
    const cnt = new Map<number, number>();
    for (let i = 0; i < W * H; i++) if (r.labels[i] === row.label && syn.truth[i]) cnt.set(syn.truth[i], (cnt.get(syn.truth[i]) ?? 0) + 1);
    const ks = [...cnt.entries()].sort((a, b) => b[1] - a[1]);
    const kind = ks.length ? syn.kinds[ks[0][0] - 1] : "none";
    return { label: row.label, area: row.area, eccentricity: row.eccentricity, solidity: row.solidity, formFactor: row.formFactor, counted: row.counted, touchesBorder: row.touchesBorder, kind, ntruth: ks.length, cx: row.cx, cy: row.cy };
  });
  return { mask: Array.from(r.mask), labels: Array.from(r.labels), count: r.count, rows };
};
out.keep = run({});
out.frame = run({ border: "frame" });
out.open2 = run({ openRadius: 2 });
const m0 = (out.keep as any).mask as number[];
out.area40 = Array.from(areaOpening(m0, W, H, 40, 8));
fs.writeFileSync(process.argv[2], JSON.stringify(out));
