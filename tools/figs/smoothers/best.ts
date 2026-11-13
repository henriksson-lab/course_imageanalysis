// Grid search for the best-PSNR settings used by the smoothercompare demo (table BEST in
// demos/src/smoothercompare/main.ts). Run from the repository root (takes a few minutes):
//   python3 -I -c "import sys; sys.path.insert(0, 'tools/figs/heat-equation'); import pslib, json; json.dump(pslib.cell128().ravel().tolist(), open('/tmp/cell128.json', 'w'))"
//   node_modules/.bin/esbuild tools/figs/smoothers/best.ts --bundle --platform=node --log-level=error | node - /tmp/cell128.json
import { METHODS, noisyOf, noiseSd, scores, cardTruth, Scenario } from "../../../demos/src/smoothercompare/methods";
import { psnr255 } from "../../../demos/src/lib/pde";
declare const require: (m: string) => any; declare const process: { argv: string[] };
const fs = require("fs");
const cell = Float64Array.from(JSON.parse(fs.readFileSync(process.argv[2], "utf8")) as number[]);
const scen: [string, Scenario][] = [["card-gauss-15", { image: "card", noise: "gauss", level: 15 }], ["card-gauss-40", { image: "card", noise: "gauss", level: 40 }],
  ["card-saltpepper-5", { image: "card", noise: "saltpepper", level: 5 }], ["card-poisson-30", { image: "card", noise: "poisson", level: 30 }], ["cell-gauss-15", { image: "cell", noise: "gauss", level: 15 }]];
for (const [name, sc] of scen) {
  const truth = sc.image === "card" ? cardTruth() : cell, y = noisyOf(truth, sc), sd = noiseSd(truth, y);
  const best: Record<string, number> = {};
  for (const m of METHODS) {
    const vals: number[] = m.id === "median" ? [1, 2, 3] : Array.from({ length: 24 }, (_, k) => (m.log ? m.lo * Math.pow(m.hi / m.lo, k / 23) : m.lo + (m.hi - m.lo) * k / 23));
    let bp = vals[0], bv = -Infinity;
    for (const p of vals) { const v = psnr255(m.run(y, p, sd), truth); if (v > bv) { bv = v; bp = p; } }
    if (m.id !== "median") {   // refine around the best grid value
      const span = m.log ? 1.3 : (m.hi - m.lo) / 23;
      for (let k = -4; k <= 4; k++) { const p = m.log ? bp * Math.pow(span, k / 4) : bp + span * k / 4; if (p < m.lo || p > m.hi) continue; const v = psnr255(m.run(y, p, sd), truth); if (v > bv) { bv = v; bp = p; } }
    }
    best[m.id] = Number(bp.toPrecision(3));
    const s = scores(m.run(y, bp, sd), truth);
    console.log(`${name} ${m.id}: ${best[m.id]}  PSNR ${s.psnr.toFixed(2)} SSIM ${s.ssim.toFixed(3)}`);
  }
  console.log(`"${name}": ${JSON.stringify(best)}`);
}
