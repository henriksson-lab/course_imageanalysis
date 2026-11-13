// Dumps a marching-cubes mesh for tools/figs/marchingcubes/figs.py. Run from the repo root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/marchingcubes/data.ts --outfile=<tmp>/mc-data.js && node <tmp>/mc-data.js > <tmp>/mc-data.json
// A sphere of radius 4.3 voxels (signed distance, c = 0) on a 12³ grid, extracted with the
// course's marchingCubes (lib/marching.ts), and the same with 6 tetrahedra per cube.
import { sphereSDF } from "../../../demos/src/lib/volume";
import { marchingCubes, TETS6, CUBE_CORNERS } from "../../../demos/src/lib/marching";
const v = sphereSDF(12, 4.3);
const mc = marchingCubes(v.data, v.nx, v.ny, v.nz, 0);
const mt = marchingCubes(v.data, v.nx, v.ny, v.nz, 0, { algorithm: "mt6" });
const dump = (m: { positions: Float32Array; indices: Uint32Array }) => ({ p: Array.from(m.positions), t: Array.from(m.indices) });
console.log(JSON.stringify({ mc: dump(mc), mt: dump(mt), tets6: TETS6, corners: CUBE_CORNERS }));
