// Measures the comparison of the cellnets-compare unit in node, with the same code as the
// cellcompare demo: demos/src/cellcompare/compare.ts (images, classical pipeline, scoring),
// the StarDist / Cellpose / SAM Wasm modules (demos/dist/wasm/, `make wasm`) and the weights
// and annotated images of `make models`. For each of the seven images and each method:
// F1(tau), TP/FP/FN at 0.5 and 0.7, splits, merges, time; also the dark-nuclei images with the
// grey levels inverted, and the classical pipeline tuned on each image (grid search: an
// optimistic score). Writes tools/figs/cellnets-compare/results.json and the label images to
// .cache/cellnets-compare/<image>[-inv]-<method>.bin (Int32, row-major) for lec_figs.py.
// Run (from the repository root):
//   node_modules/.bin/esbuild tools/figs/cellnets-compare/measure.ts --bundle --platform=node \
//     --log-level=warning --outfile=.cache/cellnets-compare/measure.cjs && node .cache/cellnets-compare/measure.cjs [ids…] [--no-sam]
import * as C from "../../../demos/src/cellcompare/compare";
import * as P from "../../../demos/src/cellposenet/pipeline";
import * as S from "../../../demos/src/samnet/post";
import { decodePng } from "../../../demos/src/cellposenet/png";

declare const require: (m: string) => any;
declare const process: { argv: string[]; cwd(): string; exit(c: number): never };
const fs = require("fs"), path = require("path"), url = require("url"), zlib = require("zlib"), os = require("os");

const ROOT = process.cwd();
const MODELS = path.join(ROOT, "demos/dist/models");
// CC_OUT / CC_RESULTS override the output places (e.g. when the repository's disk is full).
const env = (globalThis as any).process.env;
const OUT = env.CC_OUT ?? path.join(ROOT, ".cache/cellnets-compare");
const RESULTS = env.CC_RESULTS ?? path.join(ROOT, "tools/figs/cellnets-compare/results.json");
fs.mkdirSync(OUT, { recursive: true });

const args = process.argv.slice(2);
const noSam = args.includes("--no-sam");
const only = args.filter((a) => !a.startsWith("--"));

async function png(key: string) {
  return decodePng(new Uint8Array(fs.readFileSync(path.join(MODELS, key))), async (z) => new Uint8Array(zlib.inflateSync(z)));
}

async function glue(name: string): Promise<any> {
  const g: any = await import(url.pathToFileURL(path.join(ROOT, `demos/dist/wasm/${name}/${name}.js`)).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(path.join(ROOT, `demos/dist/wasm/${name}/${name}_bg.wasm`))) });
  return g;
}

const t = () => Date.now();

/** F1 at IoU > 0.5 for the grid search: pairs above 0.5 are unique, so no assignment is needed
 *  (the reported score of the chosen setting is then computed with C.score as for the others). */
function fastF1(truth: Int32Array, pred: Int32Array): number {
  const T = C.table(truth, pred);
  let tp = 0;
  for (let k = 0; k < T.iou.length; k++) if (T.iou[k] > 0.5) tp++;
  return T.nt + T.np ? (2 * tp) / (T.nt + T.np) : 1;
}

async function main(): Promise<void> {
  const prev = fs.existsSync(RESULTS) ? JSON.parse(fs.readFileSync(RESULTS, "utf8")) : { runs: {} };
  const results: any = { machine: `${os.cpus()[0]?.model ?? "?"}, node ${(globalThis as any).process?.version}, one thread`, date: new Date().toISOString().slice(0, 10), taus: C.TAUS, classicalDefault: C.CLASSICAL_DEFAULT, runs: prev.runs ?? {} };

  const sd = await glue("stardist");
  const sdDir = path.join(MODELS, "stardist/2D_versatile_fluo");
  sd.load(fs.readFileSync(path.join(sdDir, "config.json"), "utf8"), fs.readFileSync(path.join(sdDir, "thresholds.json"), "utf8"),
    new Uint8Array(fs.readFileSync(path.join(sdDir, "weights_best.h5"))));
  const cp = await glue("cellpose");
  cp.load_network(new Uint8Array(fs.readFileSync(path.join(MODELS, "cellpose/nuclei.safetensors"))));
  let sam: any = null;
  if (!noSam) {
    sam = await glue("sam");
    sam.load(new Uint8Array(fs.readFileSync(path.join(MODELS, "sam2/sam2.1_hiera_tiny.safetensors"))));
  }

  for (const s of C.SAMPLES) {
    if (only.length && !only.includes(s.id)) continue;
    const im = await png(s.file);
    const lp = await png(s.labels);
    const truth = Int32Array.from(lp.data.filter((_, i) => i % lp.channels === 0));
    const w = im.w, h = im.h;
    const grey0 = C.greyFromPixels(im.data, w, h, im.channels, im.depth);
    for (const inv of s.dark ? [false, true] : [false]) {
      const key = s.id + (inv ? "-inv" : "");
      const grey = inv ? C.invertGrey(grey0) : grey0;
      const run: any = results.runs[key] ?? { image: s.id, inverted: inv, w, h, nTrue: C.compact(truth).reduce((m, v) => Math.max(m, v), 0), methods: {} };
      run.methods = run.methods ?? {};
      const save = (m: string, lab: ArrayLike<number>, ms: number, extra: any = {}) => {
        const L = C.compact(lab);
        fs.writeFileSync(path.join(OUT, `${key}-${m}.bin`), Buffer.from(L.buffer));
        run.methods[m] = { ...C.score(truth, L), ms: Math.round(ms), ...extra };
        const r = run.methods[m];
        console.log(`${key.padEnd(15)} ${m.padEnd(16)} n=${r.nPred} F1@.5 ${r.f1[0].toFixed(3)} @.7 ${r.f1[4].toFixed(3)} split ${r.splits} merge ${r.merges} ${r.ms} ms`);
      };
      fs.writeFileSync(path.join(OUT, `${key}-truth.bin`), Buffer.from(C.compact(truth).buffer));

      // classical, default knobs
      const cl = C.classical(grey, w, h, C.CLASSICAL_DEFAULT);
      save("classical", cl.labels, cl.ms, { thr: cl.thr });
      // classical tuned on this image (optimistic)
      let best: { f: number; p: C.ClassicalParams; lab: Int32Array } | null = null;
      for (const sigma of [0, 1, 2]) for (const hh of [0.5, 1, 2, 4]) for (const minSize of [0, 15, 50]) for (const method of ["otsu", "manual"] as const)
        for (const thr of method === "otsu" ? [0] : [0.1, 0.2, 0.35]) {
          const p: C.ClassicalParams = { sigma, method, thr, h: hh, minSize };
          const r = C.classical(grey, w, h, p);
          const f = fastF1(truth, r.labels);
          if (!best || f > best.f + 1e-12) best = { f, p, lab: r.labels };
        }
      save("tuned", best!.lab, 0, { params: best!.p });

      // StarDist: percentile normalisation 1 / 99.8, trained thresholds
      {
        const t0 = t();
        const x = sd.normalize_percentile(new Float32Array(grey), 1, 99.8);
        sd.predict(x, w, h, 1);
        sd.instances(sd.prob(), sd.dist(), sd.out_width(), sd.out_height(), w, h, sd.n_rays(), sd.grid(), sd.default_prob_thresh(), sd.default_nms_thresh());
        save("stardist", sd.dec_labels(), t() - t0);
      }
      // Cellpose nuclei: one channel, normalize99, diameter of the sample, Cellpose 3 defaults
      {
        const t0 = t();
        const ch: [Float32Array, Float32Array] = [P.normalize99(grey), new Float32Array(w * h)];
        const rescale = P.MODELS.nuclei.diamMean / s.diameter;
        const net = await P.runNetwork(ch, w, h, rescale, async (inp, th, tw) => {
          const y = cp.run_tile(inp, th, tw);
          return { y, style: cp.last_style() };
        });
        cp.compute_masks(net.dy, net.dx, net.cellprob, h, w, P.defaultNiter(rescale), 0, 0.4, 15, 0.4);
        save("cellpose", cp.last_masks(), t() - t0, { diameter: s.diameter });
      }
      // SAM automatic: 8 x 8 grid, default filters
      if (sam) {
        const t0 = t();
        sam.encode(C.rgbFromGrey(grey), w, h);
        const tEnc = t() - t0;
        const grid = S.pointGrid(8).map(([x, y]) => [x * w, y * h] as [number, number]);
        const cands: S.Candidate[] = [];
        for (let i = 0; i < grid.length; i += 4) {
          const pts = grid.slice(i, i + 4);
          const logits: Float32Array = sam.decode_points(new Float32Array(pts.flat()));
          const [count, masks] = sam.last_shape();
          cands.push(...S.batchCandidates(logits, sam.last_iou(), count, masks, pts, i, w, h, 1));
        }
        const { kept } = S.filterCandidates(cands, S.AMG_DEFAULT);
        save("sam", S.labelImage(kept, w, h), t() - t0, { encodeMs: tEnc });
      }
      results.runs[key] = run;
      fs.writeFileSync(RESULTS, JSON.stringify(results, null, 1));
    }
  }
  console.log("wrote " + path.relative(ROOT, RESULTS));
}

main().catch((e) => { console.error(e); process.exit(1); });
