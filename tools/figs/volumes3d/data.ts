// Dumps slices of the volview nuclei stack for tools/figs/volumes3d/figs.py. Run from the repo root:
//   node_modules/.bin/esbuild --bundle --platform=node tools/figs/volumes3d/data.ts --outfile=<tmp>/vol-data.js && node <tmp>/vol-data.js > <tmp>/vol-data.json
// Same volume, origin and oblique plane as the "oblique" preset of demos/src/volview/main.ts.
import { nucleiStack, sliceAxis, reslice, tiltedPlane, Plane, Vec3, NUCLEI_OBJECTS, Volume, planePoint } from "../../../demos/src/lib/volume";
const ORIGIN: Vec3 = [10, 20, 4];
const base = nucleiStack();
const v: Volume = { ...base, spacing: [0.2, 0.2, 0.5], origin: ORIGIN };
const C = NUCLEI_OBJECTS[2].centre;
const xz = sliceAxis(v, "y", 53);
const P = tiltedPlane([0, 0, 0], -35 * Math.PI / 180, 0, 64, 64, 0.22);
const plane: Plane = { ...P, centre: [C[0] + ORIGIN[0], C[1] + ORIGIN[1], C[2] + ORIGIN[2]] };
const ob = reslice(v, plane, NaN);
const ends = [planePoint(plane, -0.5, 31.5), planePoint(plane, 63.5, 31.5)].map(p => [p[0] - ORIGIN[0], p[2] - ORIGIN[2]]);
console.log(JSON.stringify({ nx: v.nx, nz: v.nz, spacing: v.spacing, xz: { w: xz.width, h: xz.height, d: Array.from(xz.data) },
  ob: { w: ob.width, h: ob.height, step: 0.22, d: Array.from(ob.data, x => isNaN(x) ? null : x) }, C, u: P.u, ends }));
