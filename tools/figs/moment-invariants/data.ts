// Data for the moment-invariants figures, from the moments demo's code ("invariants" mode:
// the L of the presets, binary rasterisation on the 64 x 64 grid). Run via figs.py;
// argument: output JSON path.
import * as fs from "fs";
import { rasterise, rotatePoly } from "../../../demos/src/lib/shapecanvas";
import { rawMoments, centralFromRaw, normalisedMoments, huInvariants, logFeature } from "../../../demos/src/lib/moments";
import { L_SHAPE } from "../../../demos/src/moments/mode-invariants";

const S = 64, N = 4;
function hu(mask: ArrayLike<number>) {
  const m = rawMoments(mask, S, S, N), mu = centralFromRaw(m, N), eta = normalisedMoments(mu, N);
  return huInvariants(eta, N);
}
const base = rasterise([L_SHAPE], S, S);
const rot = rasterise([rotatePoly(L_SHAPE.map(p => [...p] as [number, number]), 50 * Math.PI / 180)], S, S);
const h0 = hu(base), h1 = hu(rot);
fs.writeFileSync(process.argv[2], JSON.stringify({
  L: L_SHAPE, mask: Array.from(base), hu: h0, log: h0.map(logFeature), huRot: h1, logRot: h1.map(logFeature),
}));
