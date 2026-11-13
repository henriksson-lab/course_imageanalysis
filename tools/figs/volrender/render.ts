// Renders for the volrender/ figures with the course's own CPU ray caster (demos/src/lib/volrender.ts
// castImage, the same algorithm as the volview demo) and the demo's presets. Writes JSON for figs.py.
// Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/volrender/render.ts --bundle --platform=node --outfile=$TMP/v.js
//   node $TMP/v.js $TMP/data
import * as fs from "fs";
import * as M from "../../../demos/src/lib/mat4";
import { OrbitCamera } from "../../../demos/src/lib/gl3d";
import { headPhantom, castImage, TF } from "../../../demos/src/lib/volrender";
import { DEF, PRESETS, VState } from "../../../demos/src/volview/render";

const out = process.argv[2] ?? ".";
const vol = headPhantom();
fs.writeFileSync(`${out}/phantom.json`, JSON.stringify({ nx: vol.nx, ny: vol.ny, nz: vol.nz, spacing: vol.spacing, data: Array.from(vol.data, x => +x.toFixed(2)) }));

function render(id: string, size: number, over: Partial<VState> = {}) {
  const st: VState = { ...DEF, ...PRESETS.find(p => p.id === id)!.s, ...over };
  const cam = new OrbitCamera({ distance: 2.2, azimuth: 0.5, elevation: 0.25, fovy: 35 * Math.PI / 180, orthographic: st.ortho });
  if (typeof st.view === "string") cam.setView(st.view); else [cam.azimuth, cam.elevation] = st.view;
  const PV = M.multiply(cam.projection(1), cam.view());
  const r = castImage(vol, st.tf, PV, size, size, { ...st, eps: st.eps });
  const img: number[][] = [];
  // castImage rows go from NDC y = −1 (bottom) upwards; flip to top-first
  for (let j = size - 1; j >= 0; j--) {
    const row: number[] = [];
    for (let i = 0; i < size; i++) {
      const res = r.results[j * size + i];
      if (!res) { row.push(-1, -1, -1, 1); continue; }
      if (st.mode === "dvr") row.push(+res.C[0].toFixed(4), +res.C[1].toFixed(4), +res.C[2].toFixed(4), +res.T.toFixed(4));
      else if (st.mode === "iso") row.push(res.hit, 0, 0, 0);
      else row.push(res.value, res.value, res.value, 0);
    }
    img.push(row);
  }
  const used = Array.from(r.used).reduce((a, b) => a + b, 0), full = Array.from(r.full).reduce((a, b) => a + b, 0);
  fs.writeFileSync(`${out}/vr-${id}${over.step ? "-s" + over.step : ""}.json`, JSON.stringify({ size, mode: st.mode, img, tf: st.tf }));
  console.log(id, st.mode, "step", st.step, "samples used/full", used, full);
}
const N = 160;
render("mip", N); render("minip", N); render("avg", N);
render("soft", N); render("skull", N);
render("wood", N); render("wood", N, { step: 0.5 });
render("vessels", N);
const soft = PRESETS.find(p => p.id === "soft")!.s.tf as TF, skull = PRESETS.find(p => p.id === "skull")!.s.tf as TF;
fs.writeFileSync(`${out}/tfs.json`, JSON.stringify({ soft, skull }));
