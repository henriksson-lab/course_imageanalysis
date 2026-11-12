// Measuring surfaces. Three tabs (data-tab):
//  - "resolution": a disk (2D) and a ball (3D) of radius R pixels/voxels, digitised as a
//    partial-volume grey image (each pixel ≈ the fraction of it inside: a ramp across the
//    boundary) or as a binary mask. Perimeter by pixel edges vs marching squares vs 2πR; area
//    by voxel faces vs marching cubes vs 4πR²; volume by voxel count vs marching cubes vs ⁴⁄₃πR³;
//    and a log–log plot of the relative errors against R.
//  - "isovalue": the volume enclosed by the marching-cubes surface as a function of c, for a
//    blurred ball (blur σ adjustable) and for the nuclei stack of `volumes3d/`, with the true
//    volume; a slice with its contour at the current c.
//  - "decimate": a marching-cubes mesh of a tilted ellipsoid from a grey image, a binary mask,
//    or a slightly blurred mask (terraces), decimated by vertex clustering (cell size slider),
//    with triangle count, area and volume (lib/mesh.ts).
// Auto-mounts on <div data-demo="meshmeasure">. Optional: data-tab="resolution|isovalue|decimate", data-preset=….
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { GLView, OrbitCamera, viewButtons } from "../lib/gl3d";
import * as M4 from "../lib/mat4";
import { marchingSquares, marchingCubes, padGrid } from "../lib/marching";
import { diskImage, diskMeasures, ballMeasures, blurredBall, volumeVsC } from "./measure";
import { meshArea, meshVolume, meshTopology, vertexCluster } from "../lib/mesh";
import { Volume, gaussianBlur3d, padVolume, nucleiStack, NUCLEI_OBJECTS, ellipsoidVolume, rotY, thresholdVolume, sliceAxis } from "../lib/volume";

const NAME = "meshmeasure";
type Tab = "resolution" | "isovalue" | "decimate";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .mm-tabs { flex: 1 1 100%; display: flex; gap: .4em; flex-wrap: wrap; align-items: center; }
.qb-demo[data-demo="${NAME}"] .mm-body { flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .mm-col { flex: 1 1 300px; min-width: 240px; max-width: 420px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .mm-col.wide { max-width: 520px; }
.qb-demo[data-demo="${NAME}"] .mm-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .mm-slice { position: relative; max-width: 260px; }
.qb-demo[data-demo="${NAME}"] .mm-slice canvas, [data-demo="${NAME}"] .mm-slice svg { width: 100%; display: block; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .mm-slice svg { position: absolute; left: 0; top: 0; height: 100%; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .mm-col { flex: 1 1 300px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .mm-body { flex-wrap: nowrap; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .mm-slice { max-width: 200px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

// ------------------------------------------------------------------ the demo

interface Preset { id: string; label: string; tab: Tab; R?: number; binary?: boolean; sigma?: number; c?: number; input?: "grey" | "binary" | "blurred"; cell?: number; which?: "ball" | "nuclei" }
const PRESETS: Preset[] = [
  { id: "faces", label: "voxel faces overestimate area", tab: "resolution", R: 6, binary: false },
  { id: "converge", label: "marching cubes converges", tab: "resolution", R: 16, binary: false },
  { id: "plateau", label: "a plateau in volume vs c", tab: "isovalue", which: "ball", sigma: 0.5, c: 0.5 },
  { id: "steep", label: "heavy blur: c decides the volume", tab: "isovalue", which: "ball", sigma: 2.5, c: 0.5 },
  { id: "terraces", label: "terraces from a binary mask", tab: "decimate", input: "binary", cell: 0 },
  { id: "decimate", label: "decimate 10×", tab: "decimate", input: "grey", cell: 3 },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const pAttr = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = (pAttr?.tab ?? host.dataset.tab ?? "resolution") as Tab;
  const st = { tab: tab0, R: 6, binary: false, sigma: 0.5, c: 0.5, which: "ball" as "ball" | "nuclei", input: "grey" as "grey" | "binary" | "blurred", cell: 0 };

  const tabs = htmlEl("div", "mm-tabs", "", host);
  const tabBtns = (["resolution", "isovalue", "decimate"] as Tab[]).map(t => button(tabs, t === "resolution" ? "resolution" : t === "isovalue" ? "iso-value" : "decimate", () => { st.tab = t; show(); }));
  const presetRow = htmlEl("div", "qb-presets", "", tabs);
  const presetBtns = presets(presetRow, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const body = htmlEl("div", "mm-body", "", host);
  const panes: Record<Tab, HTMLElement> = {
    resolution: htmlEl("div", "mm-body", "", body), isovalue: htmlEl("div", "mm-body", "", body), decimate: htmlEl("div", "mm-body", "", body),
  };
  const built: Partial<Record<Tab, { refresh: () => void }>> = {};

  function load(p: Preset): void {
    Object.assign(st, { tab: p.tab }, p.R !== undefined ? { R: p.R } : {}, p.binary !== undefined ? { binary: p.binary } : {},
      p.sigma !== undefined ? { sigma: p.sigma } : {}, p.c !== undefined ? { c: p.c } : {}, p.which ? { which: p.which } : {},
      p.input ? { input: p.input } : {}, p.cell !== undefined ? { cell: p.cell } : {});
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    show(true);
  }

  function show(refresh = false): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["resolution", "isovalue", "decimate"][i] === st.tab));
    presetBtns.forEach((b, i) => { b.style.display = PRESETS[i].tab === st.tab ? "" : "none"; });
    for (const t of Object.keys(panes) as Tab[]) panes[t].style.display = t === st.tab ? "" : "none";
    if (!built[st.tab]) built[st.tab] = (st.tab === "resolution" ? buildResolution : st.tab === "isovalue" ? buildIso : buildDecimate)(panes[st.tab]);
    else if (refresh) built[st.tab]!.refresh();
  }

  // ---------------- resolution ----------------
  function buildResolution(pane: HTMLElement) {
    const left = htmlEl("div", "mm-col", "", pane);
    const svg = svgEl("svg", { viewBox: "0 0 400 400", width: "100%" }, left);
    const mid = htmlEl("div", "mm-col", "", pane);
    const panel = htmlEl("div", "mm-col", "", pane);
    const rS = slider(panel, "radius R (pixels / voxels)", 2, 16, 1, st.R, v => { st.R = v; refresh(); });
    const row = htmlEl("div", "qb-presets", "", panel);
    const bBtns = [button(row, "grey (partial volume)", () => { st.binary = false; refresh(); }), button(row, "binary mask", () => { st.binary = true; refresh(); })];
    const ro = htmlEl("div", "qb-readout", "", panel);
    htmlEl("div", "qb-note", "Left: the disk (true circle dashed), pixels above 1/2 shaded, the pixel-edge boundary in red and the marching-squares contour in blue. Plot: relative error |measured − true| / true against R (log–log); the dot marks the current R.", panel);
    const plot = new Plot(mid, { width: 420, height: 400, x: [2, 16], y: [0.0003, 30], xlog: true, ylog: true, xticks: [2, 4, 8, 16], xfmt: v => String(Math.round(v)), xlabel: "radius R (voxels)", ylabel: "relative error" });
    // precompute the curves (both image kinds)
    const Rs = Array.from({ length: 15 }, (_, i) => i + 2);
    const curves: Record<string, { faces: number[]; mcA: number[]; vox: number[]; mcV: number[]; edges: number[]; ms: number[] }> = {};
    for (const bin of [false, true]) {
      const k = String(bin), d = { faces: [] as number[], mcA: [] as number[], vox: [] as number[], mcV: [] as number[], edges: [] as number[], ms: [] as number[] };
      for (const R of Rs) {
        const b = ballMeasures(R, bin), q = diskMeasures(R, bin);
        const e = (a: number, t: number) => Math.max(1e-4, Math.abs(a / t - 1));
        d.faces.push(e(b.faces, b.trueA)); d.mcA.push(e(b.mcArea, b.trueA)); d.vox.push(e(b.voxels, b.trueV)); d.mcV.push(e(b.mcVolume, b.trueV));
        d.edges.push(e(q.edges, q.trueL)); d.ms.push(e(q.msLength, q.trueL));
      }
      curves[k] = d;
    }
    const names: [keyof typeof curves["true"], string, number, string?][] = [
      ["faces", "area: voxel faces", 1], ["mcA", "area: marching cubes", 0], ["vox", "volume: voxel count", 3, "6 4"], ["mcV", "volume: marching cubes", 2, "6 4"],
      ["edges", "2D perimeter: pixel edges", 4, "2 3"],
    ];
    const lines = names.map(([k, label, ci, dash]) => plot.line(Rs, curves["false"][k], { color: seriesColor(ci), label, dash, width: 2.5 }));
    const dots = names.map(([, , ci]) => plot.scatter([st.R], [0.5], { color: seriesColor(ci), r: 6 }));
    plot.legend(undefined, "tr");

    function refresh(): void {
      rS.set(st.R);
      bBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === st.binary));
      const cv = curves[String(st.binary)];
      names.forEach(([k], i) => { lines[i].set(Rs, cv[k]); dots[i].set([st.R], [cv[k][st.R - 2]]); });
      // 2D drawing
      svg.replaceChildren();
      const { v, n, c } = diskImage(st.R, st.binary), s = 380 / n, o = 10;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const g = v[y * n + x];
        if (g > 0) svgEl("rect", { x: o + x * s, y: o + y * s, width: s, height: s, fill: g > 0.5 ? `rgba(255,127,14,${0.25 + 0.5 * g})` : `rgba(255,127,14,${0.5 * g})` }, svg);
      }
      if (n <= 40) for (let k = 0; k <= n; k++) {
        svgEl("line", { x1: o + k * s, y1: o, x2: o + k * s, y2: o + n * s, stroke: "#ccc", "stroke-width": 0.8 }, svg);
        svgEl("line", { x1: o, y1: o + k * s, x2: o + n * s, y2: o + k * s, stroke: "#ccc", "stroke-width": 0.8 }, svg);
      }
      // pixel-edge boundary
      let d = "";
      const ins = (x: number, y: number) => x >= 0 && y >= 0 && x < n && y < n && v[y * n + x] > 0.5;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (ins(x, y)) {
        const X = o + x * s, Y = o + y * s;
        if (!ins(x + 1, y)) d += `M${X + s},${Y}v${s}`;
        if (!ins(x - 1, y)) d += `M${X},${Y}v${s}`;
        if (!ins(x, y + 1)) d += `M${X},${Y + s}h${s}`;
        if (!ins(x, y - 1)) d += `M${X},${Y}h${s}`;
      }
      svgEl("path", { d, stroke: "#d62728", "stroke-width": 3, fill: "none" }, svg);
      const C = marchingSquares(v, n, n, 0.5);
      let dm = "";
      for (const L of C.lines) dm += L.ids.map((id, i) => `${i ? "L" : "M"}${(o + (C.points[2 * id] + 0.5) * s).toFixed(1)},${(o + (C.points[2 * id + 1] + 0.5) * s).toFixed(1)}`).join("") + (L.closed ? "Z" : "");
      svgEl("path", { d: dm, stroke: "#1f77b4", "stroke-width": 3, fill: "none" }, svg);
      svgEl("circle", { cx: o + (c[0] + 0.5) * s, cy: o + (c[1] + 0.5) * s, r: st.R * s, fill: "none", stroke: "#222", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
      const q = diskMeasures(st.R, st.binary), b = ballMeasures(st.R, st.binary);
      const pc = (a: number, t: number) => `${a.toFixed(1).padStart(8)} (${((a / t - 1) * 100 >= 0 ? "+" : "")}${((a / t - 1) * 100).toFixed(1)}%)`;
      ro.textContent = [
        `2D, R = ${st.R}         value      error`,
        `perimeter 2πR      ${(q.trueL).toFixed(1).padStart(8)}`,
        `  pixel edges      ${pc(q.edges, q.trueL)}`,
        `  marching squares ${pc(q.msLength, q.trueL)}`,
        `3D`,
        `area 4πR²          ${b.trueA.toFixed(1).padStart(8)}`,
        `  voxel faces      ${pc(b.faces, b.trueA)}`,
        `  marching cubes   ${pc(b.mcArea, b.trueA)}`,
        `volume ⁴⁄₃πR³      ${b.trueV.toFixed(1).padStart(8)}`,
        `  voxel count      ${pc(b.voxels, b.trueV)}`,
        `  marching cubes   ${pc(b.mcVolume, b.trueV)}`,
      ].join("\n");
    }
    refresh();
    return { refresh };
  }

  // ---------------- iso-value ----------------
  function buildIso(pane: HTMLElement) {
    const left = htmlEl("div", "mm-col wide", "", pane);
    const panel = htmlEl("div", "mm-col", "", pane);
    const plot = new Plot(left, { width: 520, height: 360, x: [0, 1], y: [0, 1], xlabel: "iso-value c", ylabel: "enclosed volume" });
    const curve = plot.line([0], [0], { color: seriesColor(0), width: 3, label: "marching cubes" });
    const truth = plot.hline(1, { color: seriesColor(2), text: "true volume" });
    const cur = plot.vline(st.c, { color: seriesColor(1) });
    const row = htmlEl("div", "qb-presets", "", panel);
    const wBtns = [button(row, "blurred ball (R = 8)", () => { st.which = "ball"; recompute(); }), button(row, "nuclei stack", () => { st.which = "nuclei"; recompute(); })];
    const sS = slider(panel, "blur σ (voxels)", 0, 3, 0.25, st.sigma, v => { st.sigma = v; recompute(); });
    const cS = slider(panel, "iso-value c", 0.05, 0.95, 0.01, st.c, v => { st.c = v; refresh(); });
    const ro = htmlEl("div", "qb-readout", "", panel);
    htmlEl("div", "mm-lab", "middle slice and its contour at c", panel);
    const sl = htmlEl("div", "mm-slice", "", panel);
    const cv = htmlEl("canvas", "", "", sl);
    const ssvg = svgEl("svg", { preserveAspectRatio: "none" }, sl);
    const spath = svgEl("path", { fill: "none", stroke: "#ffc000", "stroke-width": 2.5, "vector-effect": "non-scaling-stroke" }, ssvg);
    let vol: Volume, cs: number[] = [], vs: number[] = [], trueV = 1, unit = "vx³";
    plot.svg.style.cursor = "ew-resize";
    plot.svg.addEventListener("pointerdown", ev => { const [x] = plot.toData(ev); st.c = Math.max(0.05, Math.min(0.95, Math.round(x * 100) / 100)); refresh(); });

    function recompute(): void {
      if (st.which === "ball") {
        vol = blurredBall(st.sigma); trueV = 4 / 3 * Math.PI * 512; unit = "vx³";
      } else {
        const v = nucleiStack();
        for (let i = 0; i < v.data.length; i++) v.data[i] = (v.data[i] - 10) / 180;
        vol = padVolume(st.sigma > 0 ? gaussianBlur3d(v, st.sigma * 0.2) : v, 1, 0);
        trueV = NUCLEI_OBJECTS.reduce((s, e) => s + 4 / 3 * Math.PI * e.radii[0] * e.radii[1] * e.radii[2], 0); unit = "µm³";
      }
      cs = Array.from({ length: 37 }, (_, i) => 0.05 + i * 0.025);
      vs = volumeVsC(vol, cs);
      plot.setDomain([0, 1], [0, Math.max(...vs, trueV) * 1.1]);
      curve.set(cs, vs); truth.set(trueV);
      refresh();
    }
    function refresh(): void {
      wBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.which === "nuclei")));
      sS.set(st.sigma); cS.set(st.c); cur.set(st.c);
      const V = meshVolume(marchingCubes(vol.data, vol.nx, vol.ny, vol.nz, st.c, { spacing: vol.spacing }));
      // slope: change of volume for Δc = ±0.05
      const Vp = meshVolume(marchingCubes(vol.data, vol.nx, vol.ny, vol.nz, Math.min(0.99, st.c + 0.05), { spacing: vol.spacing }));
      const Vm = meshVolume(marchingCubes(vol.data, vol.nx, vol.ny, vol.nz, Math.max(0.01, st.c - 0.05), { spacing: vol.spacing }));
      ro.textContent = [
        `volume at c = ${st.c.toFixed(2)}: ${V.toFixed(1)} ${unit}`,
        `true volume:        ${trueV.toFixed(1)} ${unit}  (${((V / trueV - 1) * 100).toFixed(1)}%)`,
        `c ± 0.05 changes it by ${(((Vm - Vp) / 2) / V * 100).toFixed(1)}%`,
      ].join("\n");
      const k = Math.floor(vol.nz / 2), s = sliceAxis(vol, "z", st.which === "nuclei" ? 7 : k);
      cv.width = s.width; cv.height = s.height;
      const rgba = new Uint8ClampedArray(4 * s.data.length);
      for (let i = 0; i < s.data.length; i++) { const g = 255 * Math.max(0, Math.min(1, s.data[i])); rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = g; rgba[4 * i + 3] = 255; }
      cv.getContext("2d")!.putImageData(new ImageData(rgba, s.width, s.height), 0, 0);
      ssvg.setAttribute("viewBox", `0 0 ${s.width} ${s.height}`);
      const P = padGrid(s.data, s.width, s.height, -1), C = marchingSquares(P.values, P.w, P.h, st.c);
      spath.setAttribute("d", C.lines.map(L => L.ids.map((id, i) => `${i ? "L" : "M"}${(C.points[2 * id] - 0.5).toFixed(2)},${(C.points[2 * id + 1] - 0.5).toFixed(2)}`).join("") + (L.closed ? "Z" : "")).join(""));
    }
    recompute();
    return { refresh: recompute };
  }

  // ---------------- decimate ----------------
  function buildDecimate(pane: HTMLElement) {
    const left = htmlEl("div", "mm-col wide", "", pane);
    const cam = new OrbitCamera({ distance: 2.0, azimuth: 0.5, elevation: 0.3 });
    const view = new GLView(left, { aspect: 1.25, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }] });
    viewButtons(left, view, 0, ["front", "right", "top", "iso"], true);
    const panel = htmlEl("div", "mm-col", "", pane);
    htmlEl("div", "mm-lab", "input", panel);
    const row = htmlEl("div", "qb-presets", "", panel);
    const iBtns = (["grey", "binary", "blurred"] as const).map(m => button(row, m === "grey" ? "grey-level image" : m === "binary" ? "binary mask" : "blurred mask (σ = 1)", () => { st.input = m; rebuild(); }));
    const cS = slider(panel, "cluster cell size (voxels)", 0, 8, 0.25, st.cell, v => { st.cell = v; decimate(); });
    const ro = htmlEl("div", "qb-readout", "", panel);
    htmlEl("div", "qb-note", "Vertex clustering: snap every vertex to a grid of the given cell size, merge the vertices in each cell, drop collapsed triangles. Face normals, so terraces and facets show.", panel);
    const surf = view.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.6, 0.75, 0.92], shading: "flat", lighting: "blinn-phong", wireframe: "overlay", wireColor: [0.2, 0.25, 0.4], wireWidth: 0.9 });
    const n = 40, sp: [number, number, number] = [1, 1, 1];
    const grey = ellipsoidVolume(n, n, n, sp, [{ centre: [19.5, 19.5, 19.5], radii: [16, 9, 11], rot: rotY(0.6) }], { edge: 0.7 });
    let full = marchingCubes(new Float32Array(8), 2, 2, 2, 1);
    const k = 1.7 / n, model = M4.mul(M4.scaling(k, -k, -k), M4.translation(-n / 2, -n / 2, -n / 2));
    surf.setModel(model);
    let A0 = 1, V0 = 1, T0 = 1;
    function rebuild(): void {
      iBtns.forEach((b, i) => b.classList.toggle("active", ["grey", "binary", "blurred"][i] === st.input));
      let v = grey;
      if (st.input !== "grey") v = thresholdVolume(grey, 0.5);
      if (st.input === "blurred") v = gaussianBlur3d(v, 1);
      const p = padVolume(v, 1, 0);
      full = marchingCubes(p.data, p.nx, p.ny, p.nz, 0.5, { origin: [-1, -1, -1] });
      A0 = meshArea(full); V0 = meshVolume(full); T0 = full.indices.length / 3;
      decimate();
    }
    function decimate(): void {
      cS.set(st.cell);
      const m = st.cell > 0 ? vertexCluster(full, st.cell, [-0.5, -0.5, -0.5]) : full;
      surf.setData({ positions: m.positions.length ? m.positions : [0, 0, 0], indices: m.indices });
      const T = meshTopology(m), A = meshArea(m), V = meshVolume(m), tri = m.indices.length / 3;
      const trueV = 4 / 3 * Math.PI * 16 * 9 * 11;
      ro.textContent = [
        `triangles ${String(tri).padStart(6)}  (${(T0 / Math.max(1, tri)).toFixed(1)}× fewer)`,
        `vertices  ${String(T.vertices).padStart(6)}`,
        `area      ${A.toFixed(1).padStart(8)} vx²  (${((A / A0 - 1) * 100).toFixed(1)}%)`,
        `volume    ${V.toFixed(1).padStart(8)} vx³  (${((V / V0 - 1) * 100).toFixed(1)}%)`,
        `true volume ${trueV.toFixed(1)} vx³`,
        `χ = ${T.euler}, boundary edges ${T.boundaryEdges}, non-manifold ${T.nonManifoldEdges}`,
      ].join("\n");
    }
    rebuild();
    return { refresh: rebuild };
  }

  if (pAttr) load(pAttr); else show();
}

autoMount(NAME, mount);
