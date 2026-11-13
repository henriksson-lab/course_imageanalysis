// Point data for the slide figures of the diffusion/ lecture (modern deep learning day),
// computed with the diffusion2d demo's own model code (same seeds as the demo, so the numbers
// match its readouts). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/diffusion/lec_data.ts --bundle --platform=node --log-level=warning | node > /tmp/lec_data.json
// then: python3 -I tools/figs/diffusion/lec_figs.py /tmp/lec_data.json
import * as M from "../../../demos/src/diffusion2d/model";
import { SCORE_NETS } from "../../../demos/src/diffusion2d/score_weights";
import { alphaBarAt } from "../../../demos/src/lib/mixture";

const out: Record<string, unknown> = {};
const arr = (X: Float64Array) => Array.from(X, v => +v.toFixed(4));
function run(name: string, data: M.DataId, steps: number, sampler: M.Sampler, opts: { learned?: "under" | "trained"; cond?: number | null; w?: number } = {}): void {
  const ds = M.dataSet(data);
  const score = opts.learned ? M.learnedScore(SCORE_NETS[data][opts.learned]) : M.exactScore(ds, opts.cond ?? null, opts.w ?? 0);
  const r = M.sample(score, 500, steps, sampler, "cosine", 11);
  const X = r.frames[r.frames.length - 1];
  // per-blob counts for the ring (component with the largest responsibility)
  out[name] = { data, steps, sampler, X: arr(X), train: arr(M.forwardPoints(ds, 500, 0, "cosine")), mu: Array.from(ds.m.mu), summary: M.summarise(ds, X), path: r.frames.map(f => [+f[0].toFixed(4), +f[1].toFixed(4)]) };
}
run("few", "two", 2, "ddim");
run("many", "two", 100, "ddim");
run("copy", "twenty", 200, "ddpm");
run("rare", "rare", 100, "ddpm", { learned: "under" });
run("rare_exact", "rare", 100, "ddpm");
run("cond", "ring", 100, "ddpm", { cond: 0, w: 0 });
run("guide", "ring", 100, "ddpm", { cond: 1, w: 4 });
run("guide0", "ring", 100, "ddpm", { cond: 1, w: 0 });
const moons = M.dataSet("moons");
out.forward = Object.fromEntries([0, 0.2, 0.4, 0.6, 1].map(t => [t, { X: arr(M.forwardPoints(moons, 500, t, "cosine")), ab: alphaBarAt(t, "cosine") }]));
const ts = Array.from({ length: 101 }, (_, i) => i / 100);
out.sched = { t: ts, cosine: ts.map(t => alphaBarAt(t, "cosine")), linear: ts.map(t => alphaBarAt(t, "linear")) };
console.log(JSON.stringify(out));
