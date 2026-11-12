// connparadox: the digital topology paradox (labelling day, unit label-connectivity/).
// 2D: a 12 × 12 binary grid (click to edit); choose the connectivity of the objects and of
// the background separately. Object components are coloured and numbered, enclosed
// background regions are tinted; the verdict says whether the pair is consistent and what
// goes wrong otherwise (8/8: a closed curve that does not separate; 4/4: a broken curve
// that does). 3D: a 5 × 5 × 5 voxel set with 6/18/26 for objects and background, drawn in
// WebGL2 (lib/gl3d.ts) and edited slice by slice.
//
// Auto-mounts on <div data-demo="connparadox">. Options: data-preset (id below),
// data-tab ("2d" or "3d", default from the preset).
import { autoMount, htmlEl, button } from "../lib/ui";
import { GridView, CellMark, gridFromText } from "../lib/gridui";
import { labelColor, RGB } from "../lib/colour";
import { GLView, OrbitCamera, viewButtons, cubeMesh, boxLines, MeshObject, LinesObject } from "../lib/gl3d";
import { Conn3 } from "../lib/label";
import { analyse2d, analyse3d, verdict2d, verdict3d, PARADOX_GRIDS, preset3d } from "./topology";

const NAME = "connparadox";
const N3 = 5;
const ENCL = ["#d62728", "#9467bd", "#8c564b", "#e377c2"];

interface Preset { id: string; label: string; tab: "2d" | "3d"; grid?: string; fg: number; bg: number; vox?: string }
const PRESETS: Preset[] = [
  { id: "leaky", label: "8/8: closed but leaky", tab: "2d", grid: "diamond", fg: 8, bg: 8 },
  { id: "broken", label: "4/4: no closed curve", tab: "2d", grid: "diamond", fg: 4, bg: 4 },
  { id: "works", label: "8/4 works", tab: "2d", grid: "diamond", fg: 8, bg: 4 },
  { id: "dual", label: "4/8 works too", tab: "2d", grid: "diamond", fg: 4, bg: 8 },
  { id: "line", label: "a diagonal line: two halves?", tab: "2d", grid: "staircase", fg: 8, bg: 8 },
  { id: "square", label: "a square frame", tab: "2d", grid: "square", fg: 8, bg: 4 },
  { id: "corner3d", label: "3D: corner-touching voxels", tab: "3d", vox: "corner", fg: 26, bg: 6 },
  { id: "edge3d", label: "3D: edge-touching voxels", tab: "3d", vox: "edge", fg: 18, bg: 6 },
  { id: "shell3d", label: "3D: a closed shell", tab: "3d", vox: "shell", fg: 26, bg: 6 },
  { id: "leaky3d", label: "3D: shell with a corner missing", tab: "3d", vox: "leaky", fg: 26, bg: 26 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .cp-main { flex: 1 1 520px; min-width: 0; display: flex; gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .cp-grid { flex: 1 1 440px; min-width: 0; max-width: 470px; }
[data-demo="${NAME}"] .cp-3d { flex: 1 1 420px; min-width: 0; max-width: 580px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .cp-slices { display: grid; grid-template-columns: repeat(5, 1fr); gap: .3em; }
[data-demo="${NAME}"] .cp-slices .qb-grid-svg { max-width: 110px !important; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 330px; max-width: 420px; gap: .45em; }
[data-demo="${NAME}"] .cp-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .cp-verdict { font-weight: 600; padding: .35em .5em; border-radius: 6px; border: 2px solid var(--qb-c3); }
[data-demo="${NAME}"] .cp-verdict.bad { border-color: var(--qb-c2); color: var(--qb-c2); }
.reveal [data-demo="${NAME}"] { font-size: 19px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .cp-lab { font-size: 15px; }
.reveal [data-demo="${NAME}"] .cp-grid { max-width: 500px; }
@media (max-width: 900px) { [data-demo="${NAME}"], [data-demo="${NAME}"] .cp-main { flex-wrap: wrap; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function choice<T extends number>(parent: HTMLElement, label: string, values: T[], cur: T, on: (v: T) => void): { set(v: T): void } {
  htmlEl("div", "cp-lab", label, parent);
  const row = htmlEl("div", "qb-presets", "", parent);
  const bs = values.map(v => button(row, String(v), () => { set(v); on(v); }));
  const set = (v: T) => bs.forEach((b, k) => b.classList.toggle("active", values[k] === v));
  set(cur);
  return { set };
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let tab: "2d" | "3d" = (host.dataset.tab === "3d" || host.dataset.tab === "2d") ? host.dataset.tab : p0.tab;
  let fg2: 4 | 8 = 8, bg2: 4 | 8 = 4, fg3: Conn3 = 26, bg3: Conn3 = 6;
  let mask2: Uint8Array = new Uint8Array(144), vox: Uint8Array = new Uint8Array(N3 ** 3);

  const main = htmlEl("div", "cp-main", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  // ---- 2D ----
  const gbox = htmlEl("div", "cp-grid", "", main);
  const grid = new GridView(gbox, 12, 12, undefined, {
    cell: 36, display: "label", text: v => (v ? String(v) : ""), edit: old => (old ? 0 : 1), title: "click to edit",
    onEdit: v => { mask2 = Uint8Array.from(v, x => (x ? 1 : 0)); update(); },
  });

  // ---- 3D ----
  const box3 = htmlEl("div", "cp-3d", "", main);
  const cam = new OrbitCamera({ distance: 10.5, azimuth: -2.3, elevation: 0.55 });
  const view = new GLView(box3, { aspect: 1.6, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }], fallback: "Edit the slices below; the counts and the verdict still work." });
  const cubes: MeshObject = view.addMesh({ positions: [0, 0, 0], indices: [] }, { vertexColors: true, shading: "flat", lighting: "lambert" });
  const cav: MeshObject = view.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.85, 0.15, 0.15], shading: "flat", lighting: "lambert", opacity: 0.85 });
  const frame: LinesObject = view.addLines(boxLines([-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]), { width: 1.5, color: [0.55, 0.55, 0.6] });
  void frame;
  htmlEl("div", "cp-lab", "slices z = 0 … 4 (click to edit)", box3);
  const slices = htmlEl("div", "cp-slices", "", box3);
  const sliceViews = Array.from({ length: N3 }, (_, z) => new GridView(slices, N3, N3, undefined, {
    cell: 24, display: "label", numbers: false, edit: old => (old ? 0 : 1), title: `z = ${z}`,
    onEdit: v => { for (let i = 0; i < N3 * N3; i++) vox[z * N3 * N3 + i] = v[i] ? 1 : 0; update(); },
  }));
  viewButtons(box3, view, 0, ["front", "top", "iso"], true);

  // ---- panel ----
  const tabs = htmlEl("div", "qb-presets", "", panel);
  const t2 = button(tabs, "2D grid", () => { tab = "2d"; update(); });
  const t3 = button(tabs, "3D voxels", () => { tab = "3d"; update(); });
  const sel2 = htmlEl("div", "", "", panel), sel3 = htmlEl("div", "", "", panel);
  const cfg2 = choice(sel2, "objects (foreground)", [4, 8] as (4 | 8)[], fg2, v => { fg2 = v; update(); });
  const cbg2 = choice(sel2, "background", [4, 8] as (4 | 8)[], bg2, v => { bg2 = v; update(); });
  const cfg3 = choice(sel3, "objects (foreground)", [6, 18, 26] as Conn3[], fg3, v => { fg3 = v; update(); });
  const cbg3 = choice(sel3, "background", [6, 18, 26] as Conn3[], bg3, v => { bg3 = v; update(); });
  const read = htmlEl("div", "qb-readout", "", panel);
  const verd = htmlEl("div", "cp-verdict", "", panel);
  htmlEl("div", "cp-lab", "examples", panel);
  const prow = htmlEl("div", "qb-presets", "", panel);
  const pbtn = PRESETS.map(p => button(prow, p.label, () => load(p)));
  button(prow, "clear", () => { if (tab === "2d") mask2.fill(0); else vox.fill(0); update(); });
  button(prow, "⟲ reset", () => load(p0));

  function load(p: Preset): void {
    tab = p.tab;
    if (p.tab === "2d") { mask2 = Uint8Array.from(gridFromText(PARADOX_GRIDS[p.grid!]).values, v => (v ? 1 : 0)); fg2 = p.fg as 4 | 8; bg2 = p.bg as 4 | 8; cfg2.set(fg2); cbg2.set(bg2); }
    else { vox = preset3d(p.vox!); fg3 = p.fg as Conn3; bg3 = p.bg as Conn3; cfg3.set(fg3); cbg3.set(bg3); }
    pbtn.forEach((b, k) => b.classList.toggle("active", PRESETS[k] === p));
    update();
  }

  function update(): void {
    t2.classList.toggle("active", tab === "2d"); t3.classList.toggle("active", tab === "3d");
    gbox.style.display = tab === "2d" ? "" : "none"; box3.style.display = tab === "3d" ? "" : "none";
    sel2.style.display = tab === "2d" ? "" : "none"; sel3.style.display = tab === "3d" ? "" : "none";
    if (tab === "2d") {
      const a = analyse2d(mask2, 12, 12, fg2, bg2);
      const marks: CellMark[] = [];
      const enclIds = [...new Set(Array.from(a.bgLabels).filter(l => a.isEnclosed[l]))];
      for (let i = 0; i < 144; i++) {
        const l = a.bgLabels[i];
        if (l && a.isEnclosed[l]) marks.push({ index: i, fill: true, color: ENCL[enclIds.indexOf(l) % ENCL.length] });
      }
      grid.show({ values: a.fgLabels, marks });
      const v = verdict2d(mask2, 12, 12, fg2, bg2);
      read.textContent = `objects (${fg2}-connected):     ${a.fg}\nbackground regions (${bg2}-conn.): ${a.bg}\n  of which enclosed (tinted):  ${a.enclosed}`;
      verd.textContent = v.text; verd.classList.toggle("bad", !v.ok);
    } else {
      const a = analyse3d(vox, N3, fg3, bg3);
      for (let z = 0; z < N3; z++) sliceViews[z].show({
        values: a.fgLabels.slice(z * 25, z * 25 + 25),
        marks: Array.from({ length: 25 }, (_, i) => i).filter(i => { const l = a.bgLabels[z * 25 + i]; return l && a.isEnclosed[l]; }).map(i => ({ index: i, fill: true, color: ENCL[0] })),
      });
      const P: number[] = [], C: number[] = [], I: number[] = [], P2: number[] = [], I2: number[] = [];
      const addCube = (x: number, y: number, z: number, s: number, col: RGB | null) => {
        const cx = x - 2, cy = 2 - y, cz = z - 2;
        const m = cubeMesh([cx - s, cy - s, cz - s], [cx + s, cy + s, cz + s]);
        const PP = col ? P : P2, II = col ? I : I2, base = PP.length / 3;
        for (let k = 0; k < m.positions.length; k++) PP.push(m.positions[k]);
        for (let k = 0; k < m.indices!.length; k++) II.push(base + m.indices![k]);
        if (col) for (let k = 0; k < m.positions.length / 3; k++) C.push(...col);
      };
      for (let z = 0; z < N3; z++) for (let y = 0; y < N3; y++) for (let x = 0; x < N3; x++) {
        const i = (z * N3 + y) * N3 + x;
        if (vox[i]) addCube(x, y, z, 0.47, labelColor(a.fgLabels[i]));
        else if (a.isEnclosed[a.bgLabels[i]]) addCube(x, y, z, 0.22, null);
      }
      cubes.setData({ positions: Float32Array.from(P.length ? P : [0, 0, 0]), indices: Uint32Array.from(I), colors: Float32Array.from(C.length ? C : [0, 0, 0]) });
      cav.setData({ positions: Float32Array.from(P2.length ? P2 : [0, 0, 0]), indices: Uint32Array.from(I2) });
      const v = verdict3d(vox, N3, fg3, bg3);
      read.textContent = `objects (${fg3}-connected):     ${a.fg}\nbackground regions (${bg3}-conn.): ${a.bg}\n  of which cavities (red):     ${a.enclosed}`;
      verd.textContent = v.text; verd.classList.toggle("bad", !v.ok);
    }
  }

  if (tab !== p0.tab) { const q = PRESETS.find(p => p.tab === tab)!; load(p0); load(q); }
  else load(p0);
  // load the other tab's first preset into its buffer too, so switching tabs shows something
  if (tab === "2d") vox = preset3d("corner");
  else mask2 = Uint8Array.from(gridFromText(PARADOX_GRIDS.diamond).values, v => (v ? 1 : 0));
}

autoMount(NAME, mount);
