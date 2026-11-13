// Data for the regionprops figures, from the course's demo code (moments demo, 64 x 64 grid,
// pixel size 1, as in the demo's presets). Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { rasterise, ellipsePoly, SHAPE_PRESETS, rotatePoly, Poly } from "../../../demos/src/lib/shapecanvas";
import { label2d } from "../../../demos/src/lib/label";
import { regionProps, maskProps, bitQuads, eulerNumber, countHoles } from "../../../demos/src/moments/regionprops";
import { disc, rect } from "../../../demos/src/moments/shapes";

const S = 64;
const out: Record<string, unknown> = {};
const shape = (polys: Poly[]) => {
  const m = rasterise(polys, S, S);
  const r = maskProps(m, S, S)!;
  return { mask: Array.from(m), props: r };
};
// presets of the moments mode (mode-moments.ts)
out.ellipse = shape([ellipsePoly(32, 32, 22, 10, 96, 0.5)]);
out.square = shape([rotatePoly(rect(18, 18, 46, 46), 0.35)]);
out.L = shape(SHAPE_PRESETS.find(p => p.id === "L")!.make(S));
out.ring = shape([disc(32, 32, 22), disc(32, 32, 11)]);
// "two objects, one hole": a label image with two rows
{
  const m = rasterise([disc(22, 24, 15), disc(22, 24, 7), ellipsePoly(49, 46, 10, 6, 48, -0.6)], S, S);
  const lab = label2d(m, S, S, 8);
  out.two = { labels: Array.from(lab.data), rows: regionProps(lab.data, S, S, lab.count) };
}
// Euler number examples on small grids (strings, # = object)
const ex: Record<string, string[]> = {
  disc: ["..###..", ".#####.", "#######", "#######", "#######", ".#####.", "..###.."],
  ring: ["..###..", ".#####.", "##...##", "##...##", "##...##", ".#####.", "..###.."],
  eight: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
  two: ["##.....", "##.....", ".......", "....###", "....###", "....###", "......."],
};
const eul: Record<string, unknown> = {};
for (const [k, rows] of Object.entries(ex)) {
  const h = rows.length, w = rows[0].length;
  const m = Uint8Array.from(rows.join(""), c => (c === "#" ? 1 : 0));
  eul[k] = { w, h, mask: Array.from(m), quads: bitQuads(m, w, h), e8: eulerNumber(m, w, h, 8), objects: label2d(m, w, h, 8).count, holes: countHoles(m, w, h, 8) };
}
out.euler = eul;
fs.writeFileSync(process.argv[2], JSON.stringify(out));
