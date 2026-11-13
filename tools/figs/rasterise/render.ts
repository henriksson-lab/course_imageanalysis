// Renders for the rasterise/ figures with the course's own software rasteriser
// (demos/src/lib/raster.ts, demos/src/shading/*), written as JSON (RGB floats, rows from the TOP)
// for figs.py to turn into PNGs. Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/rasterise/render.ts --bundle --platform=node --outfile=<tmp>/r.js && node <tmp>/r.js <outdir>
import * as fs from "fs";
import * as M from "../../../demos/src/lib/mat4";
import { createTarget, drawClipTriangle, ClipVertex, rasteriseTriangle, Target } from "../../../demos/src/lib/raster";
import { icosphere } from "../../../demos/src/shading/meshes";
import { reflectance, V3, unit } from "../../../demos/src/shading/light";
import { PRESETS, winTris, shade } from "../../../demos/src/raster/presets";

const out = process.argv[2] ?? ".";
const dump = (name: string, t: Target) => {
  const rows: number[][] = [];
  for (let j = t.h - 1; j >= 0; j--) { const r: number[] = []; for (let i = 0; i < t.w; i++) for (let k = 0; k < 3; k++) r.push(+t.color[3 * (j * t.w + i) + k].toFixed(4)); rows.push(r); }
  fs.writeFileSync(`${out}/${name}.json`, JSON.stringify({ w: t.w, h: t.h, rows }));
};

// 1. shading: the demo's "Gouraud misses the highlight" preset, flat / Gouraud / per pixel
const RES = 360, BASE: V3 = [0.32, 0.55, 0.92];
const eye: V3 = [0, 0, 3.6];
const PV = M.multiply(M.perspective(34 * Math.PI / 180, 1, 0.5, 20), M.lookAt(eye, [0, 0, 0]));
const st = { level: 1, ka: 0.1, kd: 0.8, ks: 0.8, shin: 80, light: [-0.76, 0] };
const L: V3 = unit([st.light[0], st.light[1], Math.sqrt(1 - st.light[0] ** 2 - st.light[1] ** 2)]);
const m = icosphere(st.level);
const mat = { ka: st.ka, kd: st.kd, ks: st.ks, shininess: st.shin };
const shadeAt = (pw: V3, nn: V3): [number, number, number] => {
  const v = unit([eye[0] - pw[0], eye[1] - pw[1], eye[2] - pw[2]]);
  const r = reflectance(unit(nn), L, v, "blinn", mat);
  return [Math.min(1, BASE[0] * r.diffuse + r.specular), Math.min(1, BASE[1] * r.diffuse + r.specular), Math.min(1, BASE[2] * r.diffuse + r.specular)];
};
const nv = m.positions.length / 3;
const W: V3[] = [], N: V3[] = [];
for (let v = 0; v < nv; v++) { const p: V3 = [m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]]; W.push(p); N.push(unit(p)); }
let maxVertSpec = 0;
for (const where of ["flat", "gouraud", "pixel"]) {
  const vcol = W.map((p, v) => shadeAt(p, N[v]));
  const t = createTarget(RES, RES, [1, 1, 1]);
  for (let f = 0; f < m.indices.length; f += 3) {
    const id = [m.indices[f], m.indices[f + 1], m.indices[f + 2]];
    let attrs: number[][];
    if (where === "flat") {
      const [a, b, c] = id.map(i => W[i]);
      const fn = unit(M.cross(M.sub(b, a), M.sub(c, a)) as V3);
      const cen: V3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
      const col = shadeAt(cen, fn); attrs = id.map(() => col);
    } else if (where === "gouraud") attrs = id.map(i => vcol[i]);
    else attrs = id.map(i => [...N[i], ...W[i]]);
    const tri: ClipVertex[] = id.map((i, k) => ({ p: M.transformVec4(PV, [...W[i], 1]), attr: attrs[k] }));
    drawClipTriangle(t, tri, { cull: true, perspective: true }, a => (where === "pixel" ? shadeAt([a[3], a[4], a[5]], [a[0], a[1], a[2]]) : [a[0], a[1], a[2]]));
  }
  dump("shade-" + where, t);
}
for (let v = 0; v < nv; v++) { const vv = unit([eye[0] - W[v][0], eye[1] - W[v][1], eye[2] - W[v][2]]); maxVertSpec = Math.max(maxVertSpec, reflectance(N[v], L, vv, "blinn", mat).specular); }
console.log("icosphere level 1:", m.indices.length / 3, "triangles; max vertex specular", maxVertSpec.toExponential(2));

// 2. the raster demo's floor at 4x the demo resolution (64x48 * 4), affine vs perspective-correct
const pp = PRESETS.find(p => p.id === "perspective")!;
for (const persp of [false, true]) {
  const s = pp.make();
  const k = 4, w = 64 * k, h = 48 * k;
  const t = createTarget(w, h, [1, 1, 1]);
  winTris(s).forEach((tri, i) => rasteriseTriangle(t, tri.map(v => ({ ...v, x: v.x * k, y: v.y * k })), { perspective: persp }, a => shade(a), i));
  dump(persp ? "floor-persp" : "floor-affine", t);
}

// 3. the raster demo's "two triangles intersect", at 8x resolution: colour and depth buffer
{
  const s = PRESETS.find(p => p.id === "intersect")!.make();
  const k = 8, w = 32 * k, h = 24 * k;
  const t = createTarget(w, h, [1, 1, 1]);
  winTris(s).forEach((tri, i) => rasteriseTriangle(t, tri.map(v => ({ ...v, x: v.x * k, y: v.y * k })), {}, a => shade(a), i));
  dump("intersect-colour", t);
  const d = createTarget(w, h, [1, 1, 1]);
  for (let i = 0; i < w * h; i++) { const z = t.depth[i]; const g = isFinite(z) ? (z + 1) / 2 : 1; d.color[3 * i] = d.color[3 * i + 1] = d.color[3 * i + 2] = g; }
  dump("intersect-depth", d);
  const zs = Array.from(t.depth).filter(isFinite);
  console.log("intersect depth range", Math.min(...zs).toFixed(3), Math.max(...zs).toFixed(3));
}
