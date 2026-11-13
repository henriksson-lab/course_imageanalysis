// Data for the orthogonal-moments figures, from the momentrecon demo's code (64 x 64 inputs,
// Legendre sampled at pixel centres, Fourier low-pass with the same number of coefficients).
// Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { computeAll, polyImage, nucleusPoly, SQUARE } from "../../../demos/src/momentrecon/compute";

const nuc = polyImage(nucleusPoly());
const a = computeAll(nuc, nuc, "point");
const sq = polyImage(SQUARE);
const b = computeAll(sq, sq, "point");
const pick = (arr: Float64Array[], Ns: number[]) => Ns.map(N => Array.from(arr[N]));
fs.writeFileSync(process.argv[2], JSON.stringify({
  nucleus: Array.from(nuc), nucLeg: pick(a.leg, [2, 8, 20]), nucErr: [2, 8, 20].map(N => a.errLeg[N]),
  square: Array.from(sq), sqLeg: Array.from(b.leg[16]), sqFour: Array.from(b.four[16]), sqErr: [b.errLeg[16], b.errFour[16]],
}));
