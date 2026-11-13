// Data for the relations figures, from the neighbours demo's own code (same seeded
// populations, definitions, border rules and parent assignment). Run via figs.py.
import * as fs from "fs";
import { makePopulation, analyse, summarise, W, H, GUARD, Pattern, Def, Border } from "../../../demos/src/neighbours/model";
import { expandLabels, assignParents, touching } from "../../../demos/src/lib/neighbours";
import { edt } from "../../../demos/src/lib/distance";

const out: Record<string, unknown> = { W, H, GUARD };
const pops: Record<string, unknown> = {};
for (const [key, pattern, n] of [["poisson", "poisson", 40], ["clustered", "clustered", 40], ["regular", "regular", 40], ["border", "poisson", 60]] as [string, Pattern, number][]) {
  const pop = makePopulation({ pattern, n });
  const vor = expandLabels(pop.nuclei, W, H);
  const res: Record<string, unknown> = { count: pop.count, xs: Array.from(pop.xs), ys: Array.from(pop.ys), nuclei: Array.from(pop.nuclei), voronoi: Array.from(vor) };
  const defs: [string, Def, number, Border][] = key === "border"
    ? [["expanded", "expanded", 10, "all"], ["guard", "expanded", 10, "guard"]]
    : [["adjacent", "adjacent", 10, "all"], ["within10", "within", 10, "all"], ["within20", "within", 20, "all"], ["within30", "within", 30, "all"], ["expanded", "expanded", 10, "all"]];
  for (const [name, def, r, border] of defs) {
    const a = analyse(pop, vor, def, r, border);
    const s = summarise(pop, a, border);
    res[name] = { nb: a.nb, measured: a.measured, touchesBorder: a.touchesBorder, meanCount: s.meanCount, interior: s.interior, atBorder: s.atBorder, R: s.R, expected: s.expected, observed: s.observed,
      first: s.nn.first, second: s.nn.second, firstDist: s.nn.firstDist, secondDist: s.nn.secondDist, angle: s.nn.angle, nMeasured: s.counts.length,
      percent: a.percent ? Array.from(a.percent) : null };
  }
  if (key === "poisson") {
    res.cells = Array.from(pop.cells);
    res.spots = Array.from(pop.spots); res.nSpots = pop.nSpots; res.borderSpot = pop.borderSpot;
    const pn = assignParents(pop.spots, pop.nuclei, W, H, pop.nSpots, pop.count, "overlap");
    const pc = assignParents(pop.spots, pop.cells, W, H, pop.nSpots, pop.count, "overlap");
    res.parN = { parent: Array.from(pn.parent), nParents: Array.from(pn.nParents), fraction: Array.from(pn.fraction) };
    res.parC = { parent: Array.from(pc.parent), nParents: Array.from(pc.nParents), fraction: Array.from(pc.fraction) };
    const T = touching(pop.cells, W, H, pop.count, 8);
    res.cellPercent = Array.from(T.percent);
    // distance to the nearest nucleus (for the expansion figure)
    const bg = Uint8Array.from(pop.nuclei, l => (l ? 0 : 1));
    res.dist = Array.from(edt(bg, W, H), v => Math.round(v * 10) / 10);
  }
  pops[key] = res;
}
out.pops = pops;
fs.writeFileSync(process.argv[2], JSON.stringify(out));
