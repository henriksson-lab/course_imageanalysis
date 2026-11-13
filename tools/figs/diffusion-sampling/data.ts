// Data for the slide figures of diffusion-sampling/lec.qmd, from the samplerpaths demo's own model
// (same data, seeds and options as its "forget" preset). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/diffusion-sampling/data.ts --bundle --platform=node --log-level=warning | node > <tmp>/ds.json
import * as M from "../../../demos/src/samplerpaths/model";

const m = M.data("three");
const tr = M.truth(m, 2000);
const out: Record<string, unknown> = {};
for (const eta of [0, 1]) {
  const r = M.run(m, { steps: 20, eta, spacing: "time", twoStage: false, langevin: 0, seed: 1, n: 2000, nPaths: 30, startScale: 0.2 });
  const sm = M.summary(m, r.ends, tr);
  out["eta" + eta] = { taus: Array.from(r.taus), paths: r.paths.map(p => Array.from(p)), ends: Array.from(r.ends), sd: sm.sd, frechet: sm.frechet };
}
out.mix = m;
console.log(JSON.stringify(out));
