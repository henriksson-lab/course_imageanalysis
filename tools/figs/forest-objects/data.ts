// Data for the extra slide figures of forest-objects/ (the feature-importance demo's own
// population and analysis, so the numbers match the demo). Run:
// FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/forest-objects/data.ts, then
// python3 -I tools/figs/forest-objects/figs.py <dir>. Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import { population, FEATURES } from "../../../demos/src/featimp/population";
import { analyse } from "../../../demos/src/featimp/analysis";

const TMP = process.env.FIGTMP || os.tmpdir();
const ALL = FEATURES.map(f => f.id);
const pop = population({ seed: 1 });
const pres: Record<string, { use: string[]; dup: boolean; group: boolean }> = {
  all: { use: ALL, dup: false, group: false },
  duplicate: { use: ["area", "int", "sol", "noise", "coin"], dup: true, group: false },
  block: { use: ["area", "perim", "eqd", "int", "sol", "noise", "coin"], dup: false, group: true },
};
const out: Record<string, unknown> = { features: FEATURES, cols: pop.cols.map(c => Array.from(c)), y: Array.from(pop.y) };
for (const [k, p] of Object.entries(pres)) {
  const r = analyse(p.use, p.dup, p.group, 3);
  console.log(k, r.names.map((n, j) => `${n} ${(100 * r.mdi[j]).toFixed(0)}% / ${(100 * r.perm[j]).toFixed(1)}`).join(", "),
    `OOB ${(100 * r.oob).toFixed(1)}`, r.groupPerm !== null ? `group ${(100 * r.groupPerm).toFixed(1)}` : "");
  out[k] = { names: r.names, mdi: Array.from(r.mdi), perm: Array.from(r.perm), oob: r.oob, groupPerm: r.groupPerm };
}
const corr = (a: number[], b: number[]) => {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
};
const c = (i: number, j: number) => corr(Array.from(pop.cols[i]), Array.from(pop.cols[j]));
console.log(`correlations: area-perim ${c(0, 1).toFixed(3)}, area-eqd ${c(0, 2).toFixed(3)}, perim-eqd ${c(1, 2).toFixed(3)}`);
fs.writeFileSync(TMP + "/forest-objects.json", JSON.stringify(out));
console.log("wrote", TMP + "/forest-objects.json");
