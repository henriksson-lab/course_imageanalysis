// Marching cubes on a whole volume, recomputed in TypeScript on every change (a 48³ volume
// takes a few ms), rendered with lib/gl3d.ts; drag to rotate. Volumes: two blobs that merge as
// c drops (metaballs), a torus, a noisy sphere, a tilted ellipsoid sampled with anisotropic
// voxels, and the synthetic nuclei stack of `volumes3d/`. Controls: iso-value c (slider and
// histogram), Gaussian blur σ, noise, normals (face / from the image gradient), wireframe,
// z spacing (correct / ignored), algorithm (marching cubes / tetrahedra), face rule (cracks =
// boundary edges in red), and an xy slice plane with its marching-squares contour. Readouts:
// vertices, triangles, area, volume, components, Euler characteristic, cracks.
// The volume is padded with background so that every surface closes.
// Auto-mounts on <div data-demo="marchcubes">. Optional: data-volume="blobs|torus|sphere|ellipsoid|nuclei",
// data-preset="merge|torus|noise|blur|normals|zspacing|cracks|tets|nuclei".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons } from "../lib/gl3d";
import * as M4 from "../lib/mat4";
import {
  Volume, metaballs, volumeFromFunction, softStep, ellipsoidVolume, nucleiStack, rotY, addNoise3d, gaussianBlur3d,
  padVolume, gradientNormals, volumeRange, sliceAxis, Vec3, cloneVolume,
} from "../lib/volume";
import { marchingCubes, marchingSquares, padGrid, FaceRule, Algorithm } from "../lib/marching";
import { meshArea, meshVolume, meshTopology, boundarySegments } from "../lib/mesh";
import { histC } from "../isocontour/histc";

const NAME = "marchcubes";

export type VolumeId = "blobs" | "torus" | "sphere" | "ellipsoid" | "nuclei";
const VOLUME_NAMES: Record<VolumeId, string> = { blobs: "two blobs", torus: "torus", sphere: "sphere", ellipsoid: "ellipsoid (Δz = 2.5)", nuclei: "nuclei stack" };

/** The clean (noise-free) volumes, values about 0..1, and their true spacing. */
export function makeVolume(id: VolumeId): Volume {
  const n = 48, mid = (n - 1) / 2;
  switch (id) {
    case "blobs": return metaballs(n, [[mid - 10, mid, mid], [mid + 10, mid, mid]], 6);
    case "torus": return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => softStep(5.5 - Math.hypot(Math.hypot(x - mid, y - mid) - 13, z - mid), 1));
    case "sphere": return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => softStep(15 - Math.hypot(x - mid, y - mid, z - mid), 1));
    case "ellipsoid": {
      const v = ellipsoidVolume(n, n, 20, [1, 1, 2.5], [{ centre: [mid, mid, 23.75], radii: [17, 9, 12], rot: rotY(35 * Math.PI / 180) }], { edge: 1 });
      return v;
    }
    case "nuclei": {
      const v = nucleiStack({ noise: 0 });
      for (let i = 0; i < v.data.length; i++) v.data[i] /= 200;
      return v;
    }
  }
}

export interface MCState {
  volume: VolumeId; c: number; sigma: number; noise: number; normals: "face" | "gradient"; wire: boolean;
  zIgnored: boolean; alg: Algorithm; rule: FaceRule; slice: boolean; k: number;
}
interface Preset extends Partial<MCState> { id: string; label: string }
const DEFAULT: MCState = { volume: "blobs", c: 0.6, sigma: 0, noise: 0, normals: "gradient", wire: false, zIgnored: false, alg: "mc", rule: "decider", slice: false, k: 24 };
const PRESETS: Preset[] = [
  { id: "merge", label: "two blobs merge", volume: "blobs", c: 0.6 },
  { id: "torus", label: "a torus has a hole", volume: "torus", c: 0.5 },
  { id: "noise", label: "noise: many tiny surfaces", volume: "sphere", c: 0.5, noise: 0.3 },
  { id: "blur", label: "blur first", volume: "sphere", c: 0.5, noise: 0.3, sigma: 1.5 },
  { id: "normals", label: "faceted vs gradient normals", volume: "sphere", c: 0.5, normals: "face", wire: true },
  { id: "zspacing", label: "wrong z spacing", volume: "ellipsoid", c: 0.5, zIgnored: true },
  { id: "cracks", label: "cracks from inconsistent faces", volume: "sphere", c: 0.5, noise: 0.3, sigma: 0.5, rule: "complement" },
  { id: "slice", label: "a slice is a contour", volume: "torus", c: 0.5, slice: true, k: 24 },
  { id: "nuclei", label: "nuclei stack", volume: "nuclei", c: 0.5, sigma: 0.1 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .mq-view { flex: 1 1 420px; min-width: 260px; max-width: 520px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .3em; }
[data-demo="${NAME}"] .mq-side { flex: 0 1 220px; min-width: 160px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .mq-slice { position: relative; }
[data-demo="${NAME}"] .mq-slice canvas, [data-demo="${NAME}"] .mq-slice svg { width: 100%; display: block; image-rendering: pixelated; }
[data-demo="${NAME}"] .mq-slice svg { position: absolute; left: 0; top: 0; height: 100%; }
[data-demo="${NAME}"] .mq-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .qb-slider { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .mq-view { max-width: 470px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? { ...PRESETS[0], volume: (host.dataset.volume as VolumeId) ?? "blobs" };
  const st: MCState = { ...DEFAULT, ...p0 };

  const vbox = htmlEl("div", "mq-view", "", host);
  const cam = new OrbitCamera({ distance: 2.2, azimuth: 0.6, elevation: 0.35 });
  const view = new GLView(vbox, { aspect: 1.1, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }], fallback: "The static figure shows a typical result." });
  viewButtons(vbox, view, 0, ["front", "right", "top", "iso"], true);
  const side = htmlEl("div", "mq-side", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  const surf = view.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.55, 0.72, 0.92], shading: "smooth", lighting: "blinn-phong", wireColor: [0.15, 0.2, 0.35] });
  const crackObj = view.addLines({ positions: [] }, { width: 4, color: [0.9, 0.1, 0.1] });
  const boxObj = view.addLines({ positions: [] }, { width: 1.5, color: [0.55, 0.55, 0.6] });
  const sliceObj = view.addSlice({ rgba: new Uint8ClampedArray(4), w: 1, h: 1 }, { opacity: 0.75 });
  const contour3d = view.addLines({ positions: [] }, { width: 4, color: [1, 0.75, 0] });

  // side: histogram and the slice with its contour
  const H = histC(side, { lo: 0, hi: 1, c: st.c, width: 300, height: 170, xlabel: "value (drag: c)", onChange: v => { st.c = Math.round(v * 100) / 100; cS.set(st.c); schedule("mesh"); } });
  const slCap = htmlEl("div", "mq-lab", "", side);
  const sl = htmlEl("div", "mq-slice", "", side);
  const slCv = htmlEl("canvas", "", "", sl);
  const slSvg = svgEl("svg", { preserveAspectRatio: "none" }, sl);
  const slPath = svgEl("path", { fill: "none", stroke: "#ffc000", "stroke-width": 2.5, "vector-effect": "non-scaling-stroke" }, slSvg);

  htmlEl("div", "mq-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "mq-lab", "volume", panel);
  const volRow = htmlEl("div", "qb-presets", "", panel);
  const volBtns = (Object.keys(VOLUME_NAMES) as VolumeId[]).map(id => button(volRow, VOLUME_NAMES[id], () => { st.volume = id; st.k = Math.min(st.k, 47); schedule("volume"); }));
  const cS = slider(panel, "iso-value c", 0.02, 1.2, 0.01, st.c, v => { st.c = v; H.setC(v); schedule("mesh"); });
  const bS = slider(panel, "blur σ", 0, 3, 0.1, st.sigma, v => { st.sigma = v; schedule("volume"); });
  const nS = slider(panel, "noise", 0, 0.5, 0.01, st.noise, v => { st.noise = v; schedule("volume"); });
  htmlEl("div", "mq-lab", "normals · algorithm · face rule", panel);
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const nBtns = (["face", "gradient"] as const).map(m => button(r1, m === "face" ? "face normals" : "gradient normals", () => { st.normals = m; schedule("normals"); }));
  const wChk = checkbox(r1, "wireframe", st.wire, v => { st.wire = v; schedule("normals"); });
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const aBtns = (["mc", "mt6", "mt5"] as Algorithm[]).map(a => button(r2, a === "mc" ? "cubes" : a === "mt6" ? "6 tetrahedra" : "5 tetrahedra", () => { st.alg = a; schedule("mesh"); }));
  const r3 = htmlEl("div", "qb-presets", "", panel);
  const fBtns = (["decider", "separate", "complement"] as FaceRule[]).map(f => button(r3, f === "complement" ? "complement (15 cases)" : f, () => { st.rule = f; schedule("mesh"); }));
  const zChk = checkbox(panel, "ignore the z spacing", st.zIgnored, v => { st.zIgnored = v; schedule("mesh"); });
  const sChk = checkbox(panel, "show an xy slice", st.slice, v => { st.slice = v; schedule("slice"); });
  const kS = slider(panel, "slice k", 0, 47, 1, st.k, v => { st.k = v; schedule("slice"); });
  button(panel, "reset", () => load(p0 as Preset));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag to rotate. Red lines: boundary edges (cracks). Yellow: the marching-squares contour of the slice at the same c.", panel);

  function load(p: Preset): void {
    Object.assign(st, DEFAULT, p);
    cS.set(st.c); bS.set(st.sigma); nS.set(st.noise); kS.set(st.k); wChk.checked = st.wire; zChk.checked = st.zIgnored; sChk.checked = st.slice; H.setC(st.c);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    schedule("volume");
  }

  // ---- pipeline: volume → mesh → normals; slice ----
  let vol: Volume = makeVolume(st.volume), padded: Volume = vol, mesh = marchingCubes(new Float32Array(8), 2, 2, 2, 1);
  let level = 0, pending = false;
  const order = { volume: 3, mesh: 2, normals: 1, slice: 0 } as const;
  function schedule(what: keyof typeof order): void {
    level = Math.max(level, order[what]);
    syncButtons();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; const l = level; level = 0; compute(l); });
  }
  function syncButtons(): void {
    volBtns.forEach((b, i) => b.classList.toggle("active", Object.keys(VOLUME_NAMES)[i] === st.volume));
    nBtns.forEach((b, i) => b.classList.toggle("active", ["face", "gradient"][i] === st.normals));
    aBtns.forEach((b, i) => b.classList.toggle("active", ["mc", "mt6", "mt5"][i] === st.alg));
    fBtns.forEach((b, i) => b.classList.toggle("active", ["decider", "separate", "complement"][i] === st.rule));
    for (const b of fBtns) b.disabled = st.alg !== "mc";
  }

  let tMesh = 0;
  function compute(l: number): void {
    if (l >= 3) {
      let v = makeVolume(st.volume);
      if (st.noise > 0) v = addNoise3d(v, st.noise, 5);
      if (st.sigma > 0) v = gaussianBlur3d(v, st.sigma * v.spacing[0]);
      vol = v;
      const [lo, hi] = volumeRange(vol);
      padded = padVolume(vol, 1, Math.min(0, lo));
      H.setValues(vol.data, Math.min(0, lo), Math.max(1.05, hi));
      kS.input.max = String(vol.nz - 1);
      if (st.k > vol.nz - 1) { st.k = Math.floor((vol.nz - 1) / 2); kS.set(st.k); }
    }
    const sp: Vec3 = st.zIgnored ? [vol.spacing[0], vol.spacing[1], vol.spacing[0]] : vol.spacing;
    const pv = cloneVolume(padded, padded.data); pv.spacing = sp;
    if (l >= 2) {
      const t0 = performance.now();
      mesh = marchingCubes(pv.data, pv.nx, pv.ny, pv.nz, st.c, { algorithm: st.alg, faceRule: st.rule, spacing: sp, origin: [-sp[0], -sp[1], -sp[2]] });
      tMesh = performance.now() - t0;
      crackObj.setData({ positions: boundarySegments(mesh) });
    }
    // display transform: centre, longest side 1, flip y and z (image y down, z into the screen)
    const ext: Vec3 = [vol.nx * sp[0], vol.ny * sp[1], vol.nz * sp[2]], L = Math.max(...ext);
    const ctr: Vec3 = [(vol.nx - 1) * sp[0] / 2, (vol.ny - 1) * sp[1] / 2, (vol.nz - 1) * sp[2] / 2];
    const model = M4.mul(M4.scaling(1 / L, -1 / L, -1 / L), M4.translation(-ctr[0], -ctr[1], -ctr[2]));
    for (const o of [surf, crackObj, boxObj, sliceObj, contour3d]) o.setModel(model);
    const lo: Vec3 = [-sp[0] / 2, -sp[1] / 2, -sp[2] / 2], hi: Vec3 = [ext[0] - sp[0] / 2, ext[1] - sp[1] / 2, ext[2] - sp[2] / 2];
    const bl: number[] = [];
    for (const [a, b] of [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) {
      const P = (k: number) => [k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]];
      bl.push(...P(a), ...P(b));
    }
    boxObj.setData({ positions: bl });
    if (l >= 1) {
      // positions are in padded voxel units shifted by one voxel: gradient on the padded volume
      const normals = st.normals === "gradient" ? gradientNormals(pv, mesh.voxelPositions) : undefined;
      // the display flips y and z (a rotation), so normals transform with the model matrix
      surf.setData({ positions: mesh.positions.length ? mesh.positions : [0, 0, 0], indices: mesh.indices, normals });
      surf.set({ shading: st.normals === "gradient" ? "smooth" : "flat", wireframe: st.wire ? "overlay" : "off" });
    }
    // slice
    const s = sliceAxis(vol, "z", st.k);
    const [vlo, vhi] = volumeRange(vol);
    const rgba = new Uint8ClampedArray(4 * s.data.length);
    for (let i = 0; i < s.data.length; i++) { const g = 255 * (s.data[i] - vlo) / (vhi - vlo || 1); rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = g; rgba[4 * i + 3] = 255; }
    slCv.width = s.width; slCv.height = s.height;
    slCv.getContext("2d")!.putImageData(new ImageData(rgba, s.width, s.height), 0, 0);
    slCv.style.aspectRatio = String(ext[0] / ext[1]);
    slSvg.setAttribute("viewBox", `0 0 ${s.width} ${s.height}`);
    const P = padGrid(s.data, s.width, s.height, Math.min(0, vlo) - 1);
    const C = marchingSquares(P.values, P.w, P.h, st.c);
    let d = "";
    const seg3: number[] = [], z = st.k * sp[2];
    for (const Ln of C.lines) {
      d += Ln.ids.map((id, i) => `${i ? "L" : "M"}${(C.points[2 * id] - 0.5).toFixed(2)},${(C.points[2 * id + 1] - 0.5).toFixed(2)}`).join("") + (Ln.closed ? "Z" : "");
      const ids = Ln.closed ? [...Ln.ids, Ln.ids[0]] : Ln.ids;
      for (let i = 0; i + 1 < ids.length; i++) seg3.push((C.points[2 * ids[i]] - 1) * sp[0], (C.points[2 * ids[i] + 1] - 1) * sp[1], z, (C.points[2 * ids[i + 1]] - 1) * sp[0], (C.points[2 * ids[i + 1] + 1] - 1) * sp[1], z);
    }
    slPath.setAttribute("d", d);
    slCap.textContent = `slice k = ${st.k} and its contour`;
    sliceObj.setVisible(st.slice); contour3d.setVisible(st.slice);
    if (st.slice) {
      sliceObj.setImage({ rgba, w: s.width, h: s.height });
      sliceObj.set({ corners: [[lo[0], hi[1], z], [hi[0], hi[1], z], [hi[0], lo[1], z], [lo[0], lo[1], z]] });
      contour3d.setData({ positions: seg3 });
    }
    // readout
    const T = meshTopology(mesh), unit = st.volume === "nuclei" ? "µm" : "vx";
    ro.textContent = [
      `vertices ${String(T.vertices).padStart(7)}   triangles ${String(T.faces).padStart(7)}`,
      `area     ${meshArea(mesh).toFixed(1).padStart(9)} ${unit}²`,
      `volume   ${meshVolume(mesh).toFixed(1).padStart(9)} ${unit}³`,
      `components ${T.components}   χ = V − E + F = ${T.euler}`,
      `cracks (boundary edges) ${T.boundaryEdges}`,
      `extraction ${tMesh.toFixed(0)} ms`,
    ].join("\n");
  }

  load(p0 as Preset);
}

autoMount(NAME, mount);
