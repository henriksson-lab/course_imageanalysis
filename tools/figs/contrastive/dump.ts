// Data for the figures of the contrastive unit, computed with the contrastive demo's own code.
// Run from the repository root (DIR = any scratch folder):
//   node_modules/.bin/esbuild tools/figs/contrastive/dump.ts --bundle --platform=node --outfile=DIR/ct.js && node DIR/ct.js > DIR/ct.json
//   python3 -I tools/figs/contrastive/figs.py DIR/ct.json
import { augment, makeCells, scores, Trainer, type Settings } from "../../../demos/src/contrastive/model";
import { rng } from "../../../demos/src/lib/ui";

const cells = makeCells(1);
const BASE: Settings = { tau: 0.2, aug: { strength: 1, geom: true, bright: true, noise: true }, negatives: true };
const out: any = { cells: cells.map(c => ({ shape: c.shape, bright: c.bright })) };
// views: cells 1 (elongated), 2 (pair), 3 (round): the clean picture and 4 views each
const r = rng(5);
out.views = [1, 2, 3].map(i => ({ clean: Array.from(cells[i].img), views: [0, 1, 2, 3].map(() => Array.from(augment(cells[i].img, BASE.aug, r))) }));
const cases: [string, Settings][] = [
  ["all", BASE], ["nobright", { ...BASE, aug: { ...BASE.aug, bright: false } }],
  ["none", { ...BASE, aug: { ...BASE.aug, strength: 0 } }], ["noneg", { ...BASE, negatives: false }],
  ["tau1", { ...BASE, tau: 1 }], ["tau005", { ...BASE, tau: 0.05 }],
];
out.runs = {};
for (const [id, s] of cases) {
  const t = new Trainer(cells);
  for (let i = 0; i < 600; i++) t.train(s);
  const sc = scores(t.net, cells);
  out.runs[id] = { z: sc.z, shape: sc.shape, bright: sc.bright, collapse: sc.collapse, losses: t.losses };
}
console.log(JSON.stringify(out));
