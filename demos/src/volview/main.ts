// volview: the course's volume viewer (owner: 3D-graphics day; first built for the marching-
// cubes day's unit `volumes3d/`). One demo, several modes chosen by data-mode; each mode is a
// function in MODES below, so the 3D-graphics day adds its rendering modes (MIP/DVR/iso
// ray casting, transfer functions) as new entries without touching this one.
//
// Mode "volumes3d" (default): a synthetic nuclei stack with anisotropic voxels
// (lib/volume.ts nucleiStack: 64 × 64 × 20 voxels of 0.2 × 0.2 × 0.5 µm; nuclei A and B touch
// in x-y at different depths, C is an elongated cell tilted in x-z, D a small nucleus).
//  - 3D view (WebGL2, lib/gl3d.ts): the box of the volume with the xy and xz slices and the
//    oblique plane, in world (physical) coordinates; drag to rotate.
//  - 2D panels: the xy slice (z = k), the xz slice (y = j), the oblique slice (world point →
//    voxel by A⁻¹ → trilinear) and the maximum-intensity projection along z, each drawn with
//    its physical aspect ratio. Click the xy or xz slice to read a voxel and its world position.
//  - Δz in the header (true 0.5 µm): set it to 0.2 to "forget" the spacing and squash the cell;
//    "resample to isotropic" resamples to 0.2 µm voxels (trilinear).
//  - The voxel → world matrix A, and object counts at a threshold: in the MIP (2D labelling)
//    and in 3D (3D labelling, 26-connected).
// Auto-mounts on <div data-demo="volview">. Optional: data-mode="volumes3d",
// data-preset="slices|mip|squash|oblique|iso".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons, boxLines } from "../lib/gl3d";
import type { Vec3 as GVec3 } from "../lib/mat4";
import {
  Volume, nucleiStack, sliceAxis, mip, reslice, tiltedPlane, Plane, planePoint, resampleIsotropic, voxelToWorld,
  applyAffine, formatAffine, trilinear, NUCLEI_OBJECTS, Vec3, Slice2D,
} from "../lib/volume";
import { label2d, label3d } from "../lib/label";

const NAME = "volview";
const ORIGIN: Vec3 = [10, 20, 4];
const LO = 0, HI = 210, THR = 100;

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .vv-3d { flex: 1 1 330px; min-width: 240px; max-width: 400px; }
.qb-demo[data-demo="${NAME}"] .vv-grid { flex: 1 1 400px; min-width: 260px; max-width: 470px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .6em; align-items: end; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] canvas.vv-img { width: 100%; display: block; image-rendering: pixelated; border-radius: 2px; cursor: crosshair; touch-action: none; background: #234; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 290px; max-width: 360px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .vv-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .vv-3d { max-width: 340px; }
.reveal .qb-demo[data-demo="${NAME}"] .vv-grid { max-width: 430px; }
.reveal .qb-demo[data-demo="${NAME}"] { gap: .6em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .vv-3d { flex: 0 0 300px; }
.reveal .qb-demo[data-demo="${NAME}"] .vv-grid { flex: 0 0 370px; gap: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 420px; max-width: 440px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .vv-lab { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Grey RGBA bytes of a slice (NaN = outside: dark blue), with optional red marks. */
function toRGBA(s: Slice2D, marks: [number, number][] = []): Uint8ClampedArray {
  const out = new Uint8ClampedArray(4 * s.data.length);
  for (let i = 0; i < s.data.length; i++) {
    const v = s.data[i];
    if (v !== v) { out[4 * i] = 25; out[4 * i + 1] = 40; out[4 * i + 2] = 60; }
    else { const g = Math.max(0, Math.min(255, 255 * (v - LO) / (HI - LO))); out[4 * i] = g; out[4 * i + 1] = g; out[4 * i + 2] = g; }
    out[4 * i + 3] = 255;
  }
  for (const [x, y] of marks) {
    const i = y * s.width + x;
    if (i >= 0 && i < s.data.length) { out[4 * i] = 255; out[4 * i + 1] = 40; out[4 * i + 2] = 40; }
  }
  return out;
}

function draw(cv: HTMLCanvasElement, s: Slice2D, aspect: number, marks: [number, number][] = []): void {
  cv.width = s.width; cv.height = s.height;
  cv.style.aspectRatio = String(aspect);
  const ctx = cv.getContext("2d")!;
  ctx.putImageData(new ImageData(toRGBA(s, marks) as Uint8ClampedArray<ArrayBuffer>, s.width, s.height), 0, 0);
}

interface Preset { id: string; label: string; k: number; j: number; dz: number; iso: boolean; tiltY: number; tiltX: number; centre: Vec3 | null; shift: number }
const C_OBJ = NUCLEI_OBJECTS[2];
const PRESETS: Preset[] = [
  { id: "slices", label: "slices", k: 6, j: 31, dz: 0.5, iso: false, tiltY: 30, tiltX: 20, centre: null, shift: 0 },
  { id: "mip", label: "two nuclei merge in the MIP, not in 3D", k: 6, j: 31, dz: 0.5, iso: false, tiltY: 0, tiltX: 90, centre: null, shift: 0 },
  { id: "squash", label: "ignored z spacing squashes the cell", k: 10, j: 53, dz: 0.2, iso: false, tiltY: 0, tiltX: 0, centre: null, shift: 0 },
  { id: "oblique", label: "an oblique slice through the tilted cell", k: 10, j: 53, dz: 0.5, iso: false, tiltY: -35, tiltX: 0, centre: C_OBJ.centre, shift: 0 },
  { id: "iso", label: "resample to isotropic", k: 10, j: 53, dz: 0.5, iso: true, tiltY: 0, tiltX: 0, centre: null, shift: 0 },
];

function mountVolumes3d(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const base = nucleiStack();            // true spacing 0.2, 0.2, 0.5 µm
  const st = { ...p0, centre: p0.centre as Vec3 | null, pick: null as [number, number, number] | null };

  // ---- layout ----
  const v3 = htmlEl("div", "vv-3d", "", host);
  const cam = new OrbitCamera({ distance: 2.3, azimuth: 0.7, elevation: 0.45 });
  const view = new GLView(v3, { aspect: 1, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }], fallback: "The 2D panels show the same slices." });
  viewButtons(v3, view, 0, ["front", "right", "top", "iso"], false);
  const grid = htmlEl("div", "vv-grid", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", grid); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "vv-img", "", f); return { c, cv }; };
  const fXY = fig("xy slice"), fXZ = fig("xz slice"), fOB = fig("oblique slice"), fMIP = fig("MIP along z");
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "vv-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const kS = slider(panel, "z slice k", 0, 19, 1, st.k, v => { st.k = v; update(); });
  const jS = slider(panel, "y slice j", 0, 63, 1, st.j, v => { st.j = v; update(); });
  const tyS = slider(panel, "tilt about y (°)", -90, 90, 5, st.tiltY, v => { st.tiltY = v; update(); });
  const txS = slider(panel, "tilt about x (°)", -90, 90, 5, st.tiltX, v => { st.tiltX = v; update(); });
  const shS = slider(panel, "plane shift (µm)", -6, 6, 0.2, st.shift, v => { st.shift = v; update(); });
  const dzS = slider(panel, "Δz in the header (µm)", 0.1, 1, 0.05, st.dz, v => { st.dz = v; st.iso = false; update(); });
  const isoBtn = button(panel, "resample to isotropic", () => { st.iso = !st.iso; update(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click the xy or xz slice to read a voxel. Red: the clicked voxel. True spacing 0.2 × 0.2 × 0.5 µm.", panel);

  // ---- 3D objects ----
  const boxObj = view.addLines({ positions: [] }, { width: 2, color: [0.35, 0.35, 0.4] });
  const sXY = view.addSlice({ rgba: new Uint8ClampedArray(4), w: 1, h: 1 }, { filter: "nearest" });
  const sXZ = view.addSlice({ rgba: new Uint8ClampedArray(4), w: 1, h: 1 }, { filter: "nearest" });
  const sOB = view.addSlice({ rgba: new Uint8ClampedArray(4), w: 1, h: 1 }, { filter: "linear", opacity: 0.9 });
  const pickPts = view.addPoints({ positions: [] }, { size: 12, color: [0.9, 0.1, 0.1] });

  function load(p: Preset): void {
    Object.assign(st, p, { centre: p.centre, pick: null });
    kS.set(st.k); jS.set(st.j); tyS.set(st.tiltY); txS.set(st.tiltX); shS.set(st.shift); dzS.set(st.dz);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update();
  }

  /** The volume as the software sees it: header spacing, optionally resampled. */
  function current(): Volume {
    const v: Volume = { ...base, spacing: [0.2, 0.2, st.dz], origin: [...ORIGIN] as Vec3 };
    return st.iso ? resampleIsotropic(v) : v;
  }

  function update(): void {
    isoBtn.classList.toggle("active", st.iso);
    const v = current(), [dx, dy, dz] = v.spacing;
    const ext: Vec3 = [v.nx * dx, v.ny * dy, v.nz * dz];
    const kk = st.iso ? Math.round(st.k * 0.5 / dz) : st.k;   // same physical depth after resampling
    kS.input.max = String(base.nz - 1);
    // physical (µm, relative to the origin, voxel centres at i·Δ) → display world: centred, longest side 1, y and z flipped
    const L = Math.max(...ext), ctr: Vec3 = [(v.nx - 1) * dx / 2, (v.ny - 1) * dy / 2, (v.nz - 1) * dz / 2];
    const W = (p: Vec3): GVec3 => [(p[0] - ctr[0]) / L, -(p[1] - ctr[1]) / L, -(p[2] - ctr[2]) / L];
    // box (voxel edges)
    const lo: Vec3 = [-dx / 2, -dy / 2, -dz / 2], hi: Vec3 = [ext[0] - dx / 2, ext[1] - dy / 2, ext[2] - dz / 2];
    const bl = boxLines(W(lo), W(hi));
    boxObj.setData(bl);
    // xy slice
    const xy = sliceAxis(v, "z", kk), z0 = kk * dz;
    const pk = st.pick;
    const kOf = (k: number) => (st.iso ? Math.round(k * 0.5 / dz) : k);
    const markXY: [number, number][] = pk && kOf(pk[2]) === kk ? [[pk[0], pk[1]]] : [];
    draw(fXY.cv, xy, ext[0] / ext[1], markXY);
    fXY.c.textContent = `xy: k = ${kk}, z = ${z0.toFixed(1)} µm`;
    sXY.setImage({ rgba: toRGBA(xy, markXY), w: xy.width, h: xy.height });
    sXY.set({ corners: [W([lo[0], hi[1], z0]), W([hi[0], hi[1], z0]), W([hi[0], lo[1], z0]), W([lo[0], lo[1], z0])] });
    // xz slice
    const xz = sliceAxis(v, "y", st.j), y0 = st.j * dy;
    const markXZ: [number, number][] = pk && pk[1] === st.j ? [[pk[0], kOf(pk[2])]] : [];
    draw(fXZ.cv, xz, ext[0] / ext[2], markXZ);
    fXZ.c.textContent = `xz: j = ${st.j}, y = ${y0.toFixed(1)} µm`;
    sXZ.setImage({ rgba: toRGBA(xz, markXZ), w: xz.width, h: xz.height });
    sXZ.set({ corners: [W([lo[0], y0, hi[2]]), W([hi[0], y0, hi[2]]), W([hi[0], y0, lo[2]]), W([lo[0], y0, lo[2]])] });
    // oblique slice: plane in physical coordinates relative to the origin; reslice works in world coordinates
    const cRel: Vec3 = st.centre ? [...st.centre] as Vec3 : [ctr[0], ctr[1], ctr[2]];
    const P = tiltedPlane([0, 0, 0], st.tiltY * Math.PI / 180, st.tiltX * Math.PI / 180, 64, 64, 0.22);
    const n: Vec3 = [P.u[1] * P.w[2] - P.u[2] * P.w[1], P.u[2] * P.w[0] - P.u[0] * P.w[2], P.u[0] * P.w[1] - P.u[1] * P.w[0]];
    const cw: Vec3 = [cRel[0] + st.shift * n[0], cRel[1] + st.shift * n[1], cRel[2] + st.shift * n[2]];
    const plane: Plane = { ...P, centre: [cw[0] + ORIGIN[0], cw[1] + ORIGIN[1], cw[2] + ORIGIN[2]] };
    const ob = reslice(v, plane, NaN);
    draw(fOB.cv, ob, 1);
    sOB.setImage({ rgba: toRGBA(ob), w: ob.width, h: ob.height });
    const rel = (i: number, j: number): Vec3 => { const q = planePoint(plane, i, j); return [q[0] - ORIGIN[0], q[1] - ORIGIN[1], q[2] - ORIGIN[2]]; };
    sOB.set({ corners: [W(rel(-0.5, 63.5)), W(rel(63.5, 63.5)), W(rel(63.5, -0.5)), W(rel(-0.5, -0.5))] });
    fOB.c.textContent = `oblique slice (${st.tiltY}°, ${st.tiltX}°)`;
    // MIP
    const m = mip(v, "z");
    draw(fMIP.cv, m, ext[0] / ext[1]);
    // picked voxel in 3D
    if (pk) {
      const q = W([pk[0] * dx, pk[1] * dy, pk[2] * 0.5 * (st.iso ? 1 : dz / 0.5)]);
      pickPts.setData({ positions: q });
    } else pickPts.setData({ positions: [] });
    readout(v);
    view.render();
  }

  // counts of objects at a threshold (fixed: they do not depend on the header)
  const mipMask = Array.from(mip(base, "z").data, a => (a > THR ? 1 : 0));
  const nMip = label2d(mipMask, base.nx, base.ny, 8).count;
  const nVol = label3d(Array.from(base.data, a => (a > THR ? 1 : 0)), base.nx, base.ny, base.nz, 26).count;

  function readout(v: Volume): void {
    const A = voxelToWorld(v);
    const lines = [`voxel → world A (µm):`, formatAffine(A, 2), `${v.nx} × ${v.ny} × ${v.nz} voxels, Δ = (${v.spacing.map(s => s.toFixed(2)).join(", ")})`];
    if (st.pick) {
      const [i, j, k] = st.pick;
      const kk = st.iso ? Math.round(k * 0.5 / v.spacing[2]) : k;
      const val = trilinear(v, i, j, kk);
      const w = applyAffine(A, [i, j, kk]);
      lines.push(`voxel (${i}, ${j}, ${kk}): value ${val.toFixed(0)}`, `world (${w.map(a => a.toFixed(2)).join(", ")}) µm`);
    }
    lines.push(`objects above ${THR}: MIP ${nMip}, 3D ${nVol}`);
    ro.textContent = lines.join("\n");
  }

  // clicking a slice picks a voxel (stored in the original 64 × 64 × 20 indices)
  const pickFrom = (cv: HTMLCanvasElement, f: (u: number, w: number) => void) => cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    f((ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height);
    update();
  });
  pickFrom(fXY.cv, (u, w) => { st.pick = [Math.floor(u * base.nx), Math.floor(w * base.ny), st.iso ? Math.round(st.k) : st.k]; });
  pickFrom(fXZ.cv, (u, w) => { st.pick = [Math.floor(u * base.nx), st.j, Math.min(base.nz - 1, Math.floor(w * base.nz))]; });

  load(p0);
}

/** Modes of the viewer; the 3D-graphics day adds its own. */
const MODES: Record<string, (host: HTMLElement) => void> = { volumes3d: mountVolumes3d };

autoMount(NAME, host => {
  injectStyle();
  (MODES[host.dataset.mode ?? "volumes3d"] ?? mountVolumes3d)(host);
});
