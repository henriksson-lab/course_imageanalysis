// Reconstruction error against N for the micrograph crop of the momentrecon demo (loadCell(96),
// crop at (24, 12), 64 x 64), with the three pixel variants. Arguments: output JSON path and a
// JSON file with the 256 x 256 grey values of demos/src/fourier2d/cell256.png (decoded by figs.py).
import * as fs from "fs";
import { createGrey, resize, cropGrey } from "../../../demos/src/lib/imageops";
import { computeAll } from "../../../demos/src/momentrecon/compute";

const raw: number[] = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const g = createGrey(256, 256);
for (let i = 0; i < raw.length; i++) g.data[i] = raw[i];
const c = cropGrey(resize(g, 96, 96, "area"), 24, 12, 64, 64);
const f = Float64Array.from(c.data, v => v / 255);
const out: Record<string, number[]> = { image: Array.from(f) };
for (const v of ["point", "integrated", "discrete"] as const) out[v] = computeAll(f, f, v).errLeg;
fs.writeFileSync(process.argv[2], JSON.stringify(out));
