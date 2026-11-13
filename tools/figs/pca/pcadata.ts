// Dump the pca2d demo's data sets as JSON (bundled with esbuild, run with node), so the
// figures in pca/ use exactly the points the demo shows.
import { makeData, KINDS } from "../../../demos/src/pca2d/data";
const out: Record<string, unknown> = {};
for (const k of KINDS) out[k] = makeData(k);
console.log(JSON.stringify(out));
