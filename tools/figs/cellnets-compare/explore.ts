// Readouts of the classical pipeline for the Explore solutions of cellnets-compare/exercise.qmd
// (the same code as the cellcompare demo's live classical pipeline): F1 at 0.5 / 0.7, splits,
// merges for a few slider settings. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/cellnets-compare/explore.ts --bundle --platform=node --log-level=warning | node
import * as C from "../../../demos/src/cellcompare/compare";
import { decodePng } from "../../../demos/src/cellposenet/png";
declare const require: (m: string) => any;
const fs = require("fs"), zlib = require("zlib");
const png = (k: string) => decodePng(new Uint8Array(fs.readFileSync("demos/dist/models/" + k)), async (z) => new Uint8Array(zlib.inflateSync(z)));

async function run(id: string, inv: boolean, settings: Partial<C.ClassicalParams>[]): Promise<void> {
  const s = C.SAMPLES.find((x) => x.id === id)!;
  const im = await png(s.file), lp = await png(s.labels);
  const truth = Int32Array.from(lp.data.filter((_, i) => i % lp.channels === 0));
  let g = C.greyFromPixels(im.data, im.w, im.h, im.channels, im.depth);
  if (inv) g = C.invertGrey(g);
  for (const p of settings) {
    const q = { ...C.CLASSICAL_DEFAULT, ...p };
    const r = C.classical(g, im.w, im.h, q);
    const sc = C.score(truth, r.labels);
    console.log(`${id}${inv ? " (inverted)" : ""} ${JSON.stringify(p)}: n ${sc.nPred}/${sc.nTrue}, thr ${r.thr.toFixed(2)}, F1 ${sc.f1[0].toFixed(3)} / ${sc.f1[4].toFixed(3)}, TP/FP/FN ${sc.tp5}/${sc.fp5}/${sc.fn5}, split ${sc.splits}, merge ${sc.merges}, ${Math.round(r.ms)} ms`);
  }
}

(async () => {
  await run("sparse", false, [{}, { sigma: 0 }, { sigma: 2 }, { h: 0.5 }, { h: 2 }, { minSize: 30 }, { method: "manual", thr: 0.2 }, { method: "manual", thr: 0.35 }]);
  await run("dense", false, [{}, { h: 0.5 }, { h: 2 }, { h: 3 }, { h: 4 }, { sigma: 2 }, { sigma: 2, h: 0.5 }, { sigma: 0, h: 0.5 }]);
  await run("histology", false, [{}]);
  await run("histology", true, [{}, { sigma: 2 }, { h: 2 }]);
  await run("brightfield", false, [{}]);
  await run("brightfield", true, [{}, { minSize: 50 }]);
  await run("u2os", false, [{}]);
})();
