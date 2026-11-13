// Data for the zernike figures, from the zernikebasis demo's code (64 x 64 test images,
// inscribed disc, n <= 8) and lib/moments.ts (radial polynomials). Run via figs.py;
// argument: output JSON path.
import * as fs from "fs";
import { shapeImage, zernikeOf, S } from "../../../demos/src/zernikebasis/compute";
import { discGrid, zernikeRadialFactorial, zernikeRadial } from "../../../demos/src/lib/moments";

const set = (f: Float64Array) => { const z = zernikeOf(f); return { nm: z.nm, re: Array.from(z.re), im: Array.from(z.im) }; };
const b0 = shapeImage("blob", 0), b60 = shapeImage("blob", 60 * Math.PI / 180);
const six = shapeImage("six", 0), nine = shapeImage("six", Math.PI);
// factorial formula vs recurrence for R_{n,0}
const rhos = Array.from({ length: 101 }, (_, i) => i / 100);   // the grid of the numbers quoted in the unit
const radial: Record<string, number[]> = {};
for (const n of [30, 40, 50]) radial[n] = rhos.map(r => Math.abs(zernikeRadialFactorial(n, 0, r) - zernikeRadial(n, 0, r)));
fs.writeFileSync(process.argv[2], JSON.stringify({
  S, blob0: Array.from(b0), blob60: Array.from(b60), z0: set(b0), z60: set(b60),
  six: Array.from(six), nine: Array.from(nine), zsix: set(six), znine: set(nine),
  discCount: discGrid(64, 64).pix.length, rhos, radial,
}));
