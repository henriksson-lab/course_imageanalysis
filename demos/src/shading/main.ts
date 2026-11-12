// shading: local illumination and texture filtering, drawn by the course's own software
// rasteriser (lib/raster.ts) so that every stage is visible (3D-graphics day, units
// `rasterise/` and `gpuformats/`).
//
// Tab "shading" (default): an icosphere (levels 0–3), a torus or a low-poly gem; drag to rotate;
// light direction on the disc gizmo (the hemisphere facing you). Where the lighting is
// evaluated: flat (one colour per triangle), Gouraud (per vertex, colours interpolated) or
// per pixel (normals interpolated); model Lambert, Phong or Blinn–Phong with k_a, k_d, k_s and
// the shininess. "Squash" scales the object along its own y axis; its normals are transformed
// by M⁻ᵀ (correct) or by M (wrong), and the readout gives the worst angle error.
// Tab "texture" (data-tab="texture"): a checkerboard (or text) on a receding floor, sampled
// nearest, bilinear, from the nearest mipmap level, or trilinear; "colour the mip level"; a
// zoom on the far region; click a pixel to read its footprint and mip level.
// Auto-mounts on <div data-demo="shading">. Optional: data-tab="shading|texture",
// data-preset (shading: gouraud|facets|squash|grazing; texture: alias|bilinear|mip|levels|seams).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import * as M from "../lib/mat4";
import { createTarget, drawClipTriangle, ClipVertex, targetToRGBA, Target } from "../lib/raster";
import { icosphere, torus, creaseBlob, vertexNormals, TriMesh } from "./meshes";
import { reflectance, Model, V3, unit, dot3 } from "./light";
import { buildMips, Mip, sampleMip, footprintLevel, checkerTexture, MIP_TINT } from "./texture";

const NAME = "shading";

const STYLE = `
.qb-demo[data-demo="${NAME}"] .sh-view { flex: 1 1 420px; min-width: 260px; max-width: 500px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] canvas.sh-cv { width: 100%; display: block; touch-action: none; border-radius: 4px; background: #111; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 470px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .sh-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .sh-lab { font-size: 15px; color: var(--qb-muted); min-width: 4.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] svg.sh-gizmo { width: 120px; height: 120px; touch-action: none; cursor: crosshair; flex: 0 0 auto; }
.qb-demo[data-demo="${NAME}"] .sh-tex { flex: 1 1 560px; max-width: 660px; }
.qb-demo[data-demo="${NAME}"] canvas.sh-px { image-rendering: pixelated; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] canvas.sh-cv.sh-zoom { width: 46%; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; gap: .8em; }
.reveal .qb-demo[data-demo="${NAME}"] .sh-view { flex: 0 0 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .sh-tex { flex: 0 0 600px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 470px; max-width: 560px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .25em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

/** A row of mutually exclusive buttons. */
function choice<T extends string>(parent: Element, label: string, items: [T, string][], get: () => T, set: (v: T) => void): () => void {
  const row = htmlEl("div", "sh-row", "", parent);
  htmlEl("span", "sh-lab", label, row);
  const bs = items.map(([v, t]) => button(row, t, () => { set(v); sync(); }));
  const sync = () => bs.forEach((b, i) => b.classList.toggle("active", items[i][0] === get()));
  sync();
  return sync;
}

// =========================================================================================
// Shading tab

type Where = "flat" | "gouraud" | "pixel";
type MeshId = "sphere" | "torus" | "gem";
interface SState { mesh: MeshId; level: number; where: Where; model: Model; ka: number; kd: number; ks: number; shin: number; squash: number; byM: boolean; normals: boolean; light: [number, number]; yaw: number; pitch: number }
const SDEF: SState = { mesh: "sphere", level: 2, where: "pixel", model: "blinn", ka: 0.1, kd: 0.8, ks: 0.6, shin: 40, squash: 1, byM: false, normals: false, light: [-0.45, 0.5], yaw: 0.3, pitch: 0.25 };
const SPRESETS: { id: string; label: string; s: Partial<SState> }[] = [
  { id: "gouraud", label: "Gouraud misses the highlight", s: { mesh: "sphere", level: 1, where: "gouraud", model: "blinn", shin: 80, ks: 0.8, light: [-0.76, 0], yaw: 0, pitch: 0 } },
  { id: "facets", label: "facets vs smooth", s: { mesh: "gem", where: "flat", model: "blinn", shin: 30, normals: true } },
  { id: "squash", label: "squashed sphere: wrong normals", s: { mesh: "sphere", level: 3, where: "pixel", squash: 0.4, byM: true, normals: true, light: [-0.2, 0.75], yaw: 0, pitch: 0.5 } },
  { id: "grazing", label: "Phong vs Blinn–Phong at grazing light", s: { mesh: "sphere", level: 3, where: "pixel", model: "phong", shin: 20, ks: 0.7, light: [0.96, 0.15], yaw: 0, pitch: 0 } },
];

const RES = 360;
const BASE: V3 = [0.32, 0.55, 0.92];

function mountShading(host: HTMLElement): void {
  const p0 = SPRESETS.find(p => p.id === host.dataset.preset);
  let st: SState = { ...SDEF, ...(p0?.s ?? {}) };
  const view = htmlEl("div", "sh-view", "", host);
  const cv = htmlEl("canvas", "sh-cv", "", view);
  cv.style.aspectRatio = "1";
  const off = document.createElement("canvas");
  off.width = RES; off.height = RES;
  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, SPRESETS.map(p => ({ label: p.label, apply: () => { st = { ...SDEF, ...p.s }; syncAll(); } })));
  const syncs: (() => void)[] = [];
  syncs.push(choice(panel, "mesh", [["sphere", "sphere"], ["torus", "torus"], ["gem", "low-poly gem"]], () => st.mesh, v => { st.mesh = v; draw(); }));
  syncs.push(choice(panel, "where", [["flat", "flat"], ["gouraud", "Gouraud"], ["pixel", "per pixel"]], () => st.where, v => { st.where = v; draw(); }));
  syncs.push(choice(panel, "model", [["lambert", "Lambert"], ["phong", "Phong"], ["blinn", "Blinn–Phong"]], () => st.model, v => { st.model = v; draw(); }));
  syncs.push(choice(panel, "normals", [["inv", "by M⁻ᵀ"], ["m", "by M (wrong)"]] as ["inv" | "m", string][], () => (st.byM ? "m" : "inv"), v => { st.byM = v === "m"; draw(); }));
  const lvS = slider(panel, "sphere level", 0, 3, 1, st.level, v => { st.level = v; draw(); });
  const kaS = slider(panel, "k_a (ambient)", 0, 0.5, 0.05, st.ka, v => { st.ka = v; draw(); });
  const kdS = slider(panel, "k_d (diffuse)", 0, 1, 0.05, st.kd, v => { st.kd = v; draw(); });
  const ksS = slider(panel, "k_s (specular)", 0, 1, 0.05, st.ks, v => { st.ks = v; draw(); });
  const shS = slider(panel, "shininess", 1, 200, 1, st.shin, v => { st.shin = v; draw(); });
  const sqS = slider(panel, "squash along y", 0.3, 1, 0.05, st.squash, v => { st.squash = v; draw(); });
  const row = htmlEl("div", "sh-row", "", panel);
  const giz = svgEl("svg", { class: "sh-gizmo", viewBox: "-60 -60 120 120" }, row);
  svgEl("circle", { r: 54, fill: "#223", stroke: "#889", "stroke-width": 2 }, giz);
  svgEl("text", { x: 0, y: -40, "text-anchor": "middle", "font-size": 15, fill: "#ccd" }, giz).textContent = "light";
  const ldot = svgEl("circle", { r: 9, fill: "#ffd84a", stroke: "#000", "stroke-width": 2 }, giz);
  const rcol = htmlEl("div", "", "", row);
  const nC = checkbox(rcol, "show normals", st.normals, v => { st.normals = v; draw(); });
  button(rcol, "reset", () => { st = { ...SDEF, ...(p0?.s ?? {}) }; pbtns.forEach(b => b.classList.remove("active")); syncAll(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag the object to rotate it; drag the yellow dot to move the light (the disc is the half of the sky facing you).", panel);

  function syncAll(): void {
    syncs.forEach(f => f());
    lvS.set(st.level); kaS.set(st.ka); kdS.set(st.kd); ksS.set(st.ks); shS.set(st.shin); sqS.set(st.squash); nC.checked = st.normals;
    draw();
  }

  const meshCache = new Map<string, { m: TriMesh; n: number[] }>();
  function getMesh(): { m: TriMesh; n: number[] } {
    const key = st.mesh === "sphere" ? "s" + st.level : st.mesh;
    let e = meshCache.get(key);
    if (!e) {
      const m = st.mesh === "sphere" ? icosphere(st.level) : st.mesh === "torus" ? torus() : creaseBlob();
      e = { m, n: st.mesh === "sphere" ? m.positions.slice() : vertexNormals(m) };   // sphere: exact normals
      meshCache.set(key, e);
    }
    return e;
  }

  const eye: V3 = [0, 0, 3.6];
  const V = M.lookAt(eye, [0, 0, 0]), P = M.perspective(34 * Math.PI / 180, 1, 0.5, 20);
  const PV = M.multiply(P, V);

  function draw(): void {
    const t0 = performance.now();
    lvS.input.disabled = st.mesh !== "sphere";
    const L: V3 = unit([st.light[0], st.light[1], Math.sqrt(Math.max(0.003, 1 - st.light[0] ** 2 - st.light[1] ** 2))]);
    ldot.setAttribute("cx", String(st.light[0] * 54)); ldot.setAttribute("cy", String(-st.light[1] * 54));
    const { m, n } = getMesh();
    const Mm = M.mul(M.rotationX(st.pitch), M.rotationY(st.yaw), M.scaling(1, st.squash, 1));
    const NM = st.byM ? M.upper3(Mm) : M.normalMatrix(Mm), NC = M.normalMatrix(Mm);
    const nv = m.positions.length / 3;
    const W: V3[] = [], N: V3[] = [];
    let worst = 0;
    for (let v = 0; v < nv; v++) {
      const p: V3 = [m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]];
      W.push(M.transformPoint(Mm, p) as V3);
      const nn: V3 = [n[3 * v], n[3 * v + 1], n[3 * v + 2]];
      const a = unit(M.mat3Vec(NM, nn) as V3), b = unit(M.mat3Vec(NC, nn) as V3);
      N.push(a);
      worst = Math.max(worst, Math.acos(Math.min(1, Math.max(-1, dot3(a, b)))));
    }
    const mat = { ka: st.ka, kd: st.kd, ks: st.ks, shininess: st.shin };
    const shadeAt = (pw: V3, nn: V3): [number, number, number] => {
      const vdir = unit([eye[0] - pw[0], eye[1] - pw[1], eye[2] - pw[2]]);
      const r = reflectance(unit(nn), L, vdir, st.model, mat);
      return [Math.min(1, BASE[0] * r.diffuse + r.specular), Math.min(1, BASE[1] * r.diffuse + r.specular), Math.min(1, BASE[2] * r.diffuse + r.specular)];
    };
    const vcol = st.where === "gouraud" ? W.map((p, v) => shadeAt(p, N[v])) : [];
    const t = createTarget(RES, RES, [0.07, 0.07, 0.09]);
    const clip = (v: number) => M.transformVec4(PV, [...W[v], 1]);
    for (let f = 0; f < m.indices.length; f += 3) {
      const id = [m.indices[f], m.indices[f + 1], m.indices[f + 2]];
      let attrs: number[][];
      if (st.where === "flat") {
        const [a, b, c] = id.map(i => W[i]);
        const fn = unit(M.cross(M.sub(b, a), M.sub(c, a)) as V3);
        const cen: V3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
        const col = shadeAt(cen, fn);
        attrs = id.map(() => col);
      } else if (st.where === "gouraud") attrs = id.map(i => vcol[i]);
      else attrs = id.map(i => [...N[i], ...W[i]]);
      const tri: ClipVertex[] = id.map((i, k) => ({ p: clip(i), attr: attrs[k] }));
      drawClipTriangle(t, tri, { cull: true, perspective: true }, a => (st.where === "pixel" ? shadeAt([a[3], a[4], a[5]], [a[0], a[1], a[2]]) : [a[0], a[1], a[2]]));
    }
    blit(t, nv, m.indices.length / 3, worst, performance.now() - t0, L);
  }

  function blit(t: Target, nv: number, nt: number, worst: number, ms: number, L: V3): void {
    off.getContext("2d")!.putImageData(new ImageData(targetToRGBA(t) as Uint8ClampedArray<ArrayBuffer>, RES, RES), 0, 0);
    const dpr = window.devicePixelRatio || 1, css = cv.clientWidth || 440;
    cv.width = Math.round(css * dpr); cv.height = Math.round(css * dpr);
    const g = cv.getContext("2d")!;
    g.imageSmoothingEnabled = true;
    g.drawImage(off, 0, 0, cv.width, cv.height);
    if (st.normals) {
      const { m, n } = getMesh();
      const Mm = M.mul(M.rotationX(st.pitch), M.rotationY(st.yaw), M.scaling(1, st.squash, 1));
      const NM = st.byM ? M.upper3(Mm) : M.normalMatrix(Mm);
      const PV2 = M.multiply(P, V), s = cv.width / 2;
      g.strokeStyle = "#ffe14a"; g.lineWidth = 1.5 * dpr; g.beginPath();
      const step = m.positions.length / 3 > 700 ? 3 : 1;
      for (let v = 0; v < m.positions.length / 3; v += step) {
        const p = M.transformPoint(Mm, [m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]]);
        const nn = unit(M.mat3Vec(NM, [n[3 * v], n[3 * v + 1], n[3 * v + 2]]) as V3);
        if (dot3(nn, unit(M.sub([0, 0, 3.6], p) as V3)) <= 0) continue;
        const a = M.transformPoint(PV2, p), b = M.transformPoint(PV2, M.add(p, M.scale3(nn, 0.18)));
        g.moveTo((a[0] + 1) * s, (1 - a[1]) * s); g.lineTo((b[0] + 1) * s, (1 - b[1]) * s);
      }
      g.stroke();
    }
    const deg = (x: number) => (x * 180 / Math.PI).toFixed(1);
    ro.textContent = [
      `${nt} triangles, ${nv} vertices`,
      `light l = (${L.map(x => x.toFixed(2)).join(", ")})`,
      st.squash < 1 ? `normals by ${st.byM ? "M: off by up to " + deg(worst) + "°" : "M⁻ᵀ: perpendicular to the surface"}` : "no squash: M and M⁻ᵀ agree",
      `drawn in ${ms.toFixed(0)} ms (software rasteriser)`,
    ].join("\n");
  }

  // rotate by drag
  let drag: [number, number] | null = null;
  cv.addEventListener("pointerdown", ev => { drag = [ev.clientX, ev.clientY]; cv.setPointerCapture(ev.pointerId); ev.preventDefault(); });
  cv.addEventListener("pointermove", ev => {
    if (!drag) return;
    const r = cv.getBoundingClientRect();
    st.yaw += (ev.clientX - drag[0]) / r.width * 3; st.pitch += (ev.clientY - drag[1]) / r.height * 3;
    drag = [ev.clientX, ev.clientY]; draw();
  });
  const up = () => { drag = null; };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  // light gizmo
  let ldrag = false;
  const setL = (ev: PointerEvent) => {
    const [x, y] = svgPoint(giz, ev);
    let lx = x / 54, ly = -y / 54; const r = Math.hypot(lx, ly);
    if (r > 0.995) { lx *= 0.995 / r; ly *= 0.995 / r; }
    st.light = [lx, ly]; draw();
  };
  giz.addEventListener("pointerdown", ev => { ldrag = true; giz.setPointerCapture(ev.pointerId); setL(ev); ev.preventDefault(); });
  giz.addEventListener("pointermove", ev => { if (ldrag) setL(ev); });
  giz.addEventListener("pointerup", () => { ldrag = false; });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);
  if (p0) pbtns[SPRESETS.indexOf(p0)].classList.add("active");
  syncAll();
}

// =========================================================================================
// Texture tab

type Filter = "nearest" | "bilinear" | "mipnearest" | "trilinear";
interface TState { filter: Filter; tint: boolean; tex: "checker" | "text"; tilt: number }
const TDEF: TState = { filter: "nearest", tint: false, tex: "checker", tilt: 12 };
const TPRESETS: { id: string; label: string; s: Partial<TState> }[] = [
  { id: "alias", label: "minification aliasing", s: { filter: "nearest" } },
  { id: "bilinear", label: "bilinear is not enough", s: { filter: "bilinear" } },
  { id: "mip", label: "mipmaps fix it", s: { filter: "trilinear" } },
  { id: "levels", label: "colour the mip level", s: { filter: "trilinear", tint: true } },
  { id: "seams", label: "nearest level: seams", s: { filter: "mipnearest", tint: true } },
];
const TW = 320, TH = 200;

/** A texture with letters, drawn with the canvas (DOM only). */
function textTexture(n: number): Uint8ClampedArray {
  const c = document.createElement("canvas"); c.width = c.height = n;
  const g = c.getContext("2d")!;
  g.fillStyle = "#f4f0e6"; g.fillRect(0, 0, n, n);
  g.fillStyle = "#1a2a6c"; g.font = `bold ${n / 4.4}px sans-serif`; g.textAlign = "center"; g.textBaseline = "middle";
  const words = ["IMAGE", "PIXEL", "VOXEL", "MIP"];
  words.forEach((w, i) => g.fillText(w, n / 2, (i + 0.5) * n / 4));
  return g.getImageData(0, 0, n, n).data;
}

function mountTexture(host: HTMLElement): void {
  const p0 = TPRESETS.find(p => p.id === host.dataset.preset) ?? TPRESETS[0];
  let st: TState = { ...TDEF, ...p0.s };
  const left = htmlEl("div", "sh-tex", "", host);
  const cv = htmlEl("canvas", "sh-cv sh-px", "", left);
  cv.width = TW; cv.height = TH; cv.style.aspectRatio = `${TW} / ${TH}`;
  const zrow = htmlEl("div", "sh-row", "", left);
  const zc = htmlEl("canvas", "sh-cv sh-px sh-zoom", "", zrow);
  zc.width = 80; zc.height = 40; zc.style.aspectRatio = "2";
  htmlEl("div", "qb-note", "Zoom ×4 on the far region (the box in the picture).", zrow);
  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, TPRESETS.map(p => ({ label: p.label, apply: () => { st = { ...TDEF, ...p.s }; sync(); } })));
  const syncs: (() => void)[] = [];
  syncs.push(choice(panel, "filter", [["nearest", "nearest"], ["bilinear", "bilinear"], ["mipnearest", "mipmap, nearest level"], ["trilinear", "trilinear"]], () => st.filter, v => { st.filter = v; draw(); }));
  syncs.push(choice(panel, "texture", [["checker", "checkerboard"], ["text", "text"]], () => st.tex, v => { st.tex = v; draw(); }));
  const tC = checkbox(panel, "colour the mip level", st.tint, v => { st.tint = v; draw(); });
  const tiltS = slider(panel, "camera height", 4, 40, 1, st.tilt, v => { st.tilt = v; draw(); }, v => (v / 10).toFixed(1));
  button(panel, "reset", () => { st = { ...TDEF, ...p0.s }; pbtns.forEach((b, i) => b.classList.toggle("active", TPRESETS[i] === p0)); sync(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const legend = htmlEl("div", "sh-row", "", panel);
  MIP_TINT.slice(0, 9).forEach((c, i) => { const s = htmlEl("span", "", String(i), legend); s.style.cssText = `background:rgb(${c.map(x => 255 * x).join(",")});color:#000;padding:0 .45em;border-radius:3px;font-size:15px`; });
  htmlEl("div", "qb-note", "Click a pixel: its footprint in texels and the mip level used. The legend shows the colour of each level.", panel);

  const mips = new Map<string, Mip[]>();
  const getMips = () => {
    let m = mips.get(st.tex);
    if (!m) { m = buildMips(st.tex === "checker" ? checkerTexture(256, 16) : textTexture(256), 256); mips.set(st.tex, m); }
    return m;
  };
  let pick: [number, number] | null = [TW / 2, 40];
  let uvBuf = new Float32Array(2 * TW * TH), cover = new Uint8Array(TW * TH);

  function sync(): void { syncs.forEach(f => f()); tC.checked = st.tint; tiltS.set(st.tilt); draw(); }

  function draw(): void {
    const mp = getMips(), N = 256;
    // pass 1: texture coordinates per pixel (perspective-correct), by the rasteriser
    const h = st.tilt / 10;
    const V = M.lookAt([0, h, 2], [0, 0, -10]), P = M.perspective(50 * Math.PI / 180, TW / TH, 0.1, 200);
    const PV = M.multiply(P, V), t = createTarget(TW, TH);
    const X = 8, Z0 = 1.5, Z1 = -80, REP = 2;   // world units per texture repeat
    const corners: [number, number][] = [[-X, Z0], [X, Z0], [X, Z1], [-X, Z1]];
    const cvx: ClipVertex[] = corners.map(([x, z]) => ({ p: M.transformVec4(PV, [x, 0, z, 1]), attr: [(x + X) / REP, (Z0 - z) / REP] }));
    uvBuf = new Float32Array(2 * TW * TH); cover = new Uint8Array(TW * TH);
    for (const tri of [[0, 1, 2], [0, 2, 3]]) drawClipTriangle(t, tri.map(i => cvx[i]), { perspective: true }, (a, i, j) => { const p = j * TW + i; uvBuf[2 * p] = a[0]; uvBuf[2 * p + 1] = a[1]; cover[p] = 1; return [0, 0, 0]; });
    // pass 2: footprint (finite differences, as a GPU does on 2×2 pixel blocks) and sampling
    const img = new Uint8ClampedArray(4 * TW * TH);
    const uvAt = (i: number, j: number) => { const p = j * TW + i; return [uvBuf[2 * p] * N, uvBuf[2 * p + 1] * N]; };
    for (let j = 0; j < TH; j++) for (let i = 0; i < TW; i++) {
      const p = j * TW + i, q = 4 * ((TH - 1 - j) * TW + i);
      img[q + 3] = 255;
      if (!cover[p]) { img[q] = 150; img[q + 1] = 190; img[q + 2] = 235; continue; }
      const lam = levelAt(i, j, uvAt);
      const [u, v] = uvAt(i, j);
      const c = sampleMip(mp, u / N, v / N, st.filter, lam, st.tint);
      img[q] = c[0]; img[q + 1] = c[1]; img[q + 2] = c[2];
    }
    cv.getContext("2d")!.putImageData(new ImageData(img as Uint8ClampedArray<ArrayBuffer>, TW, TH), 0, 0);
    // zoom box near the horizon
    const hz = horizonRow(), zx = TW / 2 - 40, zy = Math.max(0, Math.min(TH - 40, hz + 2));
    const g = cv.getContext("2d")!;
    zc.getContext("2d")!.putImageData(g.getImageData(zx, zy, 80, 40), 0, 0);
    g.strokeStyle = "#ff3df5"; g.lineWidth = 1; g.strokeRect(zx + 0.5, zy + 0.5, 79, 39);
    // readout
    const lines = [`${TW} × ${TH} pixels; texture ${N} × ${N}, ${mp.length} mip levels`];
    if (pick) {
      const i = Math.round(pick[0]), jj = TH - 1 - Math.round(pick[1]);
      if (i >= 0 && i < TW && jj >= 0 && jj < TH && cover[jj * TW + i]) {
        const lam = levelAt(i, jj, uvAt), rho = 2 ** lam;
        g.strokeStyle = "#ffe600"; g.strokeRect(i - 2.5, TH - 1 - jj - 2.5, 6, 6);
        lines.push(`pixel (${i}, ${TH - 1 - jj}): footprint ${rho.toFixed(1)} texels`, `level λ = log₂ ${rho.toFixed(1)} = ${lam.toFixed(2)}${st.filter === "trilinear" ? ` → mix of ${Math.max(0, Math.floor(lam))} and ${Math.min(mp.length - 1, Math.max(0, Math.floor(lam)) + 1)}` : st.filter === "mipnearest" ? ` → level ${Math.min(mp.length - 1, Math.max(0, Math.round(lam)))}` : " (ignored: level 0)"}`);
      } else lines.push("click the floor to read a pixel");
    }
    ro.textContent = lines.join("\n");
  }

  function levelAt(i: number, j: number, uvAt: (i: number, j: number) => number[]): number {
    const i2 = i + 1 < TW && cover[j * TW + i + 1] ? i + 1 : i - 1, j2 = j + 1 < TH && cover[(j + 1) * TW + i] ? j + 1 : j - 1;
    const a = uvAt(i, j), bx = uvAt(Math.max(0, i2), j), by = uvAt(i, Math.max(0, j2));
    return footprintLevel(a[0] - bx[0], a[1] - bx[1], a[0] - by[0], a[1] - by[1]);
  }
  function horizonRow(): number { for (let r = 0; r < TH; r++) if (cover[(TH - 1 - r) * TW + TW / 2]) return r; return 0; }

  cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    pick = [(ev.clientX - r.left) / r.width * TW, (ev.clientY - r.top) / r.height * TH];
    draw();
  });
  pbtns[TPRESETS.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => {
  injectStyle();
  if (host.dataset.tab === "texture") mountTexture(host); else mountShading(host);
});
