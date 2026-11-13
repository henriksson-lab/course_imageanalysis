// Exports the endmembers demo's data (same model: 16 channels, 400 photons, seed 1) for
// figs.py: singular values, PCA axes, true spectra, and the normalised cloud projected on
// its two main principal directions with true and found (vertex finding) spectra.
//   node_modules/.bin/esbuild tools/figs/endmembers/export.ts --bundle --platform=node --outfile=<tmp>/x.js
//   node <tmp>/x.js <outdir>      (writes <outdir>/endmembers.json)
import { writeFileSync } from "fs";
import * as U from "../../../demos/src/lib/unmix";
import { NCH, buildData, vertexSpectra } from "../../../demos/src/endmembers/model";
import { L0, L1 } from "../../../demos/src/lsunmix/model";

const out = process.argv[2] ?? ".", ph = 400;
const res: Record<string, unknown> = { cen: U.contiguousBands(L0, L1, NCH).map(b => (b.lo + b.hi) / 2) };
const r4 = (v: number) => Math.round(v * 1e5) / 1e5;
{
  const d = buildData(4, true, ph), n = d.cellIdx.length;
  res.sv = Array.from(d.sv, r4);
  console.log("sv ratios", Array.from(d.sv.slice(0, 8), v => (v / d.sv[0]).toFixed(3)).join(" "));
  const pca = U.stackPCA(d.Yc, NCH, n);
  res.pca = [0, 1, 2, 3].map(k => Array.from({ length: NCH }, (_, c) => r4(pca.vectors[c * NCH + k])));
  res.Mtrue = [0, 1, 2, 3].map(k => { const v = Array.from({ length: NCH }, (_, c) => d.Mtrue[c * 4 + k]); const s = v.reduce((a, b) => a + b, 0); return v.map(x => r4(x / s)); });
}
for (const pure of [true, false]) {
  const d = buildData(3, pure, ph), n = d.cellIdx.length;
  const Yn2 = U.normaliseSum(d.Yc, NCH, n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < NCH; c++) s += d.Yc[c * n + i]; w[i] = s > 0.15 && Number.isFinite(Yn2[i]) ? 1 : 0; if (!w[i]) for (let c = 0; c < NCH; c++) Yn2[c * n + i] = 0; }
  const pca = U.stackPCA(Yn2, NCH, n, w), mean = pca.mean;
  const pc = [0, 1].map(j => Float64Array.from({ length: NCH }, (_, c) => pca.vectors[c * NCH + j]));
  const proj = (v: ArrayLike<number>) => { let s = 0; for (let c = 0; c < NCH; c++) s += v[c]; return pc.map(p => { let x = 0; for (let c = 0; c < NCH; c++) x += (v[c] / (s || 1) - mean[c]) * p[c]; return r4(x); }); };
  const cloud: number[][] = [];
  for (let i = 0; i < n; i += 2) if (w[i]) cloud.push(proj(Array.from({ length: NCH }, (_, c) => Yn2[c * n + i])));
  const tv = [0, 1, 2].map(k => proj(Array.from({ length: NCH }, (_, c) => d.Mtrue[c * 3 + k])));
  const r = vertexSpectra(d, 3), fv = [0, 1, 2].map(k => proj(Array.from({ length: NCH }, (_, c) => r.F[c * 3 + k])));
  const m = U.matchColumns(U.normaliseSum(d.Mtrue, NCH, 3), r.F, NCH, 3);
  console.log(pure ? "pure" : "nopure", "angles", m.angleDeg.map(a => a.toFixed(1)).join(" "));
  res[pure ? "pure" : "nopure"] = { cloud, tv, fv, perm: m.perm, angles: m.angleDeg };
}
writeFileSync(out + "/endmembers.json", JSON.stringify(res));
