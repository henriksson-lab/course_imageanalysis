// Dumps marching-cubes meshes of a sphere and a torus for tools/figs/surface-measure/figs.py.
// Run from the repo root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/surface-measure/data.ts --outfile=<tmp>/sm-data.js && node <tmp>/sm-data.js > <tmp>/sm-data.json
// Signed-distance volumes (lib/volume.ts), c = 0, the course's marchingCubes (lib/marching.ts).
import { sphereSDF, torusSDF } from "../../../demos/src/lib/volume";
import { marchingCubes } from "../../../demos/src/lib/marching";
const dump = (m: { positions: Float32Array; indices: Uint32Array }) => ({ p: Array.from(m.positions), t: Array.from(m.indices) });
const s = sphereSDF(14, 5.2), t = torusSDF(18, 5.2, 2.2);
console.log(JSON.stringify({
  sphere: dump(marchingCubes(s.data, s.nx, s.ny, s.nz, 0)),
  torus: dump(marchingCubes(t.data, t.nx, t.ny, t.nz, 0)),
}));
