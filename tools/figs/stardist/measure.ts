// Explore numbers for stardist/exercise.qmd, measured with the real network (the StarDist Wasm
// module and the 2D_versatile_fluo weights, exactly as the stardistnet demo runs them: input
// normalised by the 1st and 99.8th percentile, the demo's presets for the thresholds).
// For images with ground truth: F1 at τ = 0.5 and 0.7 (lib/eval.ts, optimal matching), and
// where the splits and merges are. Writes the label image of each run (trained thresholds) as
// DIR/<name>.pred (int32) for the figures.
//   python3 -I tools/figs/stardist/measure_prep.py DIR
//   node_modules/.bin/esbuild tools/figs/stardist/measure.ts --bundle --platform=node --log-level=warning | node - DIR
declare const require: (m: string) => any;
declare const process: { argv: string[]; cwd(): string };
import { iouTable, match, objectScores, splitsMerges } from "../../../demos/src/lib/eval";
const fs = require("fs"), path = require("path"), url = require("url");

const PRESETS: [string, number | null, number | null][] = [["trained", null, null], ["low", 0.15, null], ["high", 0.85, null], ["weak", null, 0.9], ["strong", null, 0.02]];

async function main(): Promise<void> {
  const dir = process.argv[2];
  const wdir = path.join(process.cwd(), "demos/dist/wasm/stardist"), mdir = path.join(process.cwd(), "demos/dist/models/stardist/2D_versatile_fluo");
  const g: any = await import(url.pathToFileURL(path.join(wdir, "stardist.js")).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(path.join(wdir, "stardist_bg.wasm"))) });
  g.load(fs.readFileSync(path.join(mdir, "config.json"), "utf8"), fs.readFileSync(path.join(mdir, "thresholds.json"), "utf8"),
    new Uint8Array(fs.readFileSync(path.join(mdir, "weights_best.h5"))));
  const P0 = g.default_prob_thresh(), N0 = g.default_nms_thresh();
  const out: any = { probThresh: P0, nmsThresh: N0, images: [] };
  for (const e of JSON.parse(fs.readFileSync(path.join(dir, "index.json"), "utf8"))) {
    const buf = fs.readFileSync(path.join(dir, e.name + ".f32"));
    const grey = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4).slice();
    const x = g.normalize_percentile(grey, 1, 99.8);
    const t0 = Date.now();
    g.predict(x, e.w, e.h, 1);
    const ms = Date.now() - t0;
    const prob = g.prob(), dist = g.dist(), ow = g.out_width(), oh = g.out_height();
    let truth: Int32Array | null = null;
    if (e.labels) { const lb = fs.readFileSync(path.join(dir, e.name + ".lab")); truth = new Int32Array(lb.buffer, lb.byteOffset, lb.length / 4).slice(); }
    const rec: any = { name: e.name, w: e.w, h: e.h, ms, nTrue: e.labels ?? null, presets: {} };
    let pmax = 0; for (const v of prob) pmax = Math.max(pmax, v);
    rec.maxProb = +pmax.toFixed(3);
    for (const [pid, pp, nn] of PRESETS) {
      g.instances(prob, dist, ow, oh, e.w, e.h, 32, 2, pp ?? P0, nn ?? N0);
      const r: any = { candidates: g.dec_candidates(), count: g.dec_count() };
      if (truth) {
        const lab = g.dec_labels();
        const T = iouTable(truth, lab);
        const s5 = objectScores(match(T, 0.5)), s7 = objectScores(match(T, 0.7)), sm = splitsMerges(T);
        Object.assign(r, { tp: s5.tp, fp: s5.fp, fn: s5.fn, f1: +s5.f1.toFixed(3), f1_07: +s7.f1.toFixed(3), splits: sm.splits.length, merges: sm.merges.length });
      }
      if (pid === "trained") fs.writeFileSync(path.join(dir, e.name + ".pred"), new Uint8Array(new Int32Array(g.dec_labels()).buffer));
      if (truth && pid === "trained") {
        // where the splits and merges are (bounding boxes of the true / predicted objects involved)
        const sm = splitsMerges(iouTable(truth, g.dec_labels()));
        const bbox = (lab: ArrayLike<number>, ids: number[]) => {
          let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
          for (let i = 0; i < lab.length; i++) if (ids.includes(lab[i])) { const x = i % e.w, y = (i - x) / e.w; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
          return [x0, y0, x1, y1];
        };
        rec.splitBoxes = sm.splits.map(q => bbox(truth!, [q.t + 1]));
        rec.mergeBoxes = sm.merges.map(q => bbox(g.dec_labels(), [q.p + 1]));
      }
      rec.presets[pid] = r;
    }
    out.images.push(rec);
    console.error(e.name, ms, "ms", JSON.stringify(rec.presets.trained));
  }
  console.log(JSON.stringify(out, null, 1));
}
main();
