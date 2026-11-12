// The kernel trick as a picture: lift 2D points into 3D, where a plane separates them.
// Left: points of two classes in the plane (click to add, drag to move). Right (WebGL2): the
// same points lifted by a feature map φ, on the lifted surface (translucent), and the plane
// found by a linear SVM in 3D (translucent, orange). The plane's cut through the surface (black
// curve, right) is the 2D boundary (left: f(φ(x)) = 0 and the sides it separates): a straight
// cut in 3D is a curved (conic) boundary in 2D. Maps: (x₁, x₂, x₁² + x₂²) (paraboloid),
// (x₁, x₂, x₁x₂) (saddle), and (x₁², √2 x₁x₂, x₂²), the map of the kernel function (x·z)².
// Drag the 3D view to orbit (buttons for standard views; no keyboard).
//
// Auto-mounts on <div data-demo="svmlift">. Options:
//   data-preset  ring | xor | blobs | tworings (default ring)
//   data-map     paraboloid | saddle | quadratic (default from the preset)
import { htmlEl, slider, presets, checkbox, autoMount, Slider } from "../lib/ui";
import { pointPlayground } from "../lib/playground";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import { GLView, OrbitCamera, viewButtons, boxLines, MeshObject, PointsObject, LinesObject } from "../lib/gl3d";
import { trainSVM, SVMModel } from "../lib/svm";
import { paraboloidMap, quadMap } from "../lib/svmkernels";
import { rings, xor, blobs, LPoint } from "../lib/datasets2d";
import { marchingSquares } from "../lib/marching";
import { worldOf, sampleGrid, contourPath, pathEl, signColour } from "../svm2d/field";

const NAME = "svmlift";
type MapId = "paraboloid" | "saddle" | "quadratic";
const MAPS: Record<MapId, { label: string; f: (x: number, y: number) => [number, number, number]; axes: string }> = {
  paraboloid: { label: "x₁² + x₂² (paraboloid)", f: paraboloidMap, axes: "(x₁, x₂, x₁² + x₂²)" },
  saddle: { label: "x₁x₂ (saddle)", f: (x, y) => [x, y, x * y], axes: "(x₁, x₂, x₁x₂)" },
  quadratic: { label: "(x₁², √2 x₁x₂, x₂²)", f: quadMap, axes: "(x₁², √2 x₁x₂, x₂²)" },
};

interface Preset { id: string; label: string; map: MapId; points: () => LPoint[]; note: string }
const to11 = (pts: LPoint[]): LPoint[] => pts.map(p => ({ x: 2 * p.x - 1, y: 2 * p.y - 1, c: p.c }));
const PRESETS: Preset[] = [
  { id: "ring", label: "ring → plane", map: "paraboloid", points: () => to11(rings({ n: 70, seed: 41, radii: [0.13, 0.36], noise: 0.03 }).points).map(p => ({ ...p, c: 1 - p.c })), note: "On the paraboloid the inner points sit low, the ring high: a flat plane separates them. Its cut through the paraboloid is a circle." },
  { id: "xor", label: "XOR needs x₁x₂", map: "saddle", points: () => to11(xor({ n: 70, seed: 42 }).points), note: "Lifted by x₁x₂, the two diagonals go up and down: the plane x₁x₂ = 0 separates them. Try the paraboloid: it does not help here." },
  { id: "blobs", label: "two blobs: lifting does not hurt", map: "paraboloid", points: () => to11(blobs([[0.32, 0.4], [0.68, 0.6]], { sd: 0.08, n: 50, seed: 43 }).points), note: "Separable in 2D already. In 3D the plane is nearly vertical and its cut is almost a straight line." },
  {
    id: "tworings", label: "two rings: one plane is not enough", map: "paraboloid",
    points: () => to11(rings({ n: 110, seed: 44, radii: [0.1, 0.26, 0.42], noise: 0.02 }).points).map(p => ({ ...p, c: p.c === 1 ? 1 : 0 })),
    note: "Blue inside, red ring, blue again outside: on the paraboloid red is a band in the middle height. No single plane cuts out a band: this map is not rich enough.",
  },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .sl-left { flex: 0 1 380px; min-width: 220px; }
[data-demo="${NAME}"] .sl-left .qb-plot { max-width: none; }
[data-demo="${NAME}"] .sl-mid { flex: 1 1 380px; min-width: 240px; max-width: 520px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; gap: .35em; }
[data-demo="${NAME}"] .sl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .sl-cap { font-size: 15px; text-align: center; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .sl-left { flex-basis: 340px; }
.reveal [data-demo="${NAME}"] .sl-mid { flex-basis: 440px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { max-height: 560px; overflow-y: auto; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const rgb = (hex: string): [number, number, number] => hexToRgb(hex) as [number, number, number];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const colors = [QUALITATIVE[0], QUALITATIVE[1]];
  const st = { map: ((host.dataset.map as MapId) in MAPS ? host.dataset.map : p0.map) as MapId, logC: 2, plane: true, surface: true, note: p0.note };

  const left = htmlEl("div", "sl-left", "", host);
  const mid = htmlEl("div", "sl-mid", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "sl-lab", "add points of class", panel);
  const classBox = htmlEl("div", "", "", panel);
  const pg = pointPlayground(left, { k: 2, colors, names: ["class −1", "class +1"], controlsParent: classBox, resolution: 80, r: 7, onChange: (_, fin) => update(fin) });
  const W = worldOf(pg, -1, 1);
  htmlEl("div", "sl-cap", "2D: the plane's cut, seen from above", left);

  const cam = new OrbitCamera({ distance: 3.1, azimuth: 0.7, elevation: 0.42 });
  const view = new GLView(mid, { aspect: 1.15, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: cam }], fallback: "The 2D view on the left still shows the boundary." });
  viewButtons(mid, view, 0, ["front", "right", "top", "iso"], true);
  const cap3 = htmlEl("div", "sl-cap", "", mid);

  htmlEl("div", "sl-lab", "feature map: third coordinate", panel);
  const mapBtns = presets(panel, (Object.keys(MAPS) as MapId[]).map(id => ({ label: MAPS[id].label, apply: () => { st.map = id; update(true); } })));
  const cS: Slider = slider(panel, "C", -1, 3, 0.1, st.logC, v => { st.logC = v; update(true); }, v => Math.pow(10, v).toPrecision(2));
  checkbox(panel, "show the plane", st.plane, v => { st.plane = v; update(true); });
  checkbox(panel, "show the lifted surface", st.surface, v => { st.surface = v; update(true); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  // GL objects
  let surf: MeshObject | null = null, plane: MeshObject | null = null, pts3: PointsObject | null = null, cut: LinesObject | null = null, box: LinesObject | null = null;
  const gBg = pg.overlay.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "g"));

  /** Bounding box of φ over [−1, 1]² and the map to GL coordinates (φ1 → x, φ3 → up, φ2 → −z), scaled to about ±0.8. */
  function frame(m: MapId) {
    const f = MAPS[m].f, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let j = 0; j <= 20; j++) for (let i = 0; i <= 20; i++) { const v = f(-1 + i / 10, -1 + j / 10); for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], v[k]); hi[k] = Math.max(hi[k], v[k]); } }
    const c = lo.map((l, k) => (l + hi[k]) / 2), s = lo.map((l, k) => 1.6 / Math.max(1e-9, hi[k] - l));
    const gl = (v: [number, number, number]): [number, number, number] => [(v[0] - c[0]) * s[0], (v[2] - c[2]) * s[2] * 0.8, -(v[1] - c[1]) * s[1]];
    return { lo, hi, gl };
  }

  function update(final: boolean): void { requestAnimationFrame(() => draw(final)); }

  function draw(_final: boolean): void {
    mapBtns.forEach((b, i) => b.classList.toggle("active", (Object.keys(MAPS) as MapId[])[i] === st.map));
    const P = W.points(), M = MAPS[st.map], F = frame(st.map);
    const n = P.length, X = new Float64Array(3 * n);
    P.forEach((p, i) => X.set(M.f(p.x, p.y), 3 * i));
    const nPos = P.filter(p => p.c === 1).length;
    let model: SVMModel | null = null;
    if (nPos > 0 && nPos < n) model = trainSVM(X, n, 3, P.map(p => (p.c === 1 ? 1 : -1)), { C: Math.pow(10, st.logC), tol: 1e-4 });
    const fx = (x: number, y: number) => { if (!model) return 0; const v = M.f(x, y); return model.w![0] * v[0] + model.w![1] * v[1] + model.w![2] * v[2] + model.b; };
    // 2D
    gBg.replaceChildren();
    if (model) {
      const R = 80, grid = sampleGrid(W, R, fx);
      pg.setBackgroundColour(signColour(W, colors, fx, { street: false }));
      pathEl(gBg as SVGGElement, contourPath(W, pg, grid, R, 0), "var(--qb-fg, #222)", 3.2);
    } else pg.setBackgroundColour(null);
    // 3D: points
    const pos = new Float32Array(3 * n), col = new Float32Array(3 * n);
    P.forEach((p, i) => { pos.set(F.gl(M.f(p.x, p.y)), 3 * i); col.set(rgb(colors[p.c]), 3 * i); });
    if (!pts3) pts3 = view.addPoints({ positions: pos, colors: col }, { size: 9 });
    else pts3.setData({ positions: pos, colors: col });
    // surface
    const G = 28, sp: number[] = [], si: number[] = [];
    for (let j = 0; j <= G; j++) for (let i = 0; i <= G; i++) sp.push(...F.gl(M.f(-1 + 2 * i / G, -1 + 2 * j / G)));
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const a = j * (G + 1) + i; si.push(a, a + 1, a + G + 2, a, a + G + 2, a + G + 1); }
    const sd = st.surface ? { positions: sp, indices: si } : { positions: [0, 0, 0], indices: [] as number[] };
    if (!surf) surf = view.addMesh(sd, { color: [0.75, 0.75, 0.78], opacity: 0.35, lighting: "lambert", shading: "smooth", wireframe: "overlay", wireColor: [0.55, 0.55, 0.6], wireWidth: 0.6 });
    else surf.setData(sd);
    // plane: solve for the coordinate with the largest |w_k| over a grid of the other two
    let pd = { positions: [0, 0, 0] as number[], indices: [] as number[] };
    if (model && st.plane) {
      const w = model.w!, k = [0, 1, 2].reduce((a, b) => (Math.abs(w[b]) > Math.abs(w[a]) ? b : a), 0);
      const [u, v] = [0, 1, 2].filter(q => q !== k), Q = 30, pp: number[] = [], ok: boolean[] = [], ii: number[] = [];
      const pad = F.hi.map((h, q) => 0.04 * (h - F.lo[q]));
      for (let j = 0; j <= Q; j++) for (let i = 0; i <= Q; i++) {
        const c3: [number, number, number] = [0, 0, 0];
        c3[u] = F.lo[u] + (F.hi[u] - F.lo[u]) * i / Q; c3[v] = F.lo[v] + (F.hi[v] - F.lo[v]) * j / Q;
        c3[k] = Math.abs(w[k]) > 1e-12 ? -(model.b + w[u] * c3[u] + w[v] * c3[v]) / w[k] : 0;
        ok.push(c3[k] >= F.lo[k] - pad[k] && c3[k] <= F.hi[k] + pad[k]);
        pp.push(...F.gl(c3));
      }
      for (let j = 0; j < Q; j++) for (let i = 0; i < Q; i++) {
        const a = j * (Q + 1) + i, b = a + 1, c = a + Q + 2, d = a + Q + 1;
        if (ok[a] && ok[b] && ok[c]) ii.push(a, b, c);
        if (ok[a] && ok[c] && ok[d]) ii.push(a, c, d);
      }
      if (ii.length) pd = { positions: pp, indices: ii };
    }
    if (!plane) plane = view.addMesh(pd, { color: [0.95, 0.65, 0.15], opacity: 0.45, lighting: "unlit", shading: "flat" });
    else plane.setData(pd);
    // the cut: contour f = 0 lifted onto the surface
    const lines: number[] = [];
    if (model) {
      const R = 80, grid = new Float64Array(R * R);
      for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) grid[j * R + i] = fx(-1 + 2 * i / (R - 1), -1 + 2 * j / (R - 1));
      const C = marchingSquares(grid, R, R, 0);
      for (let s = 0; s < C.segments.length; s += 2) for (const id of [C.segments[s], C.segments[s + 1]]) {
        const x = -1 + 2 * C.points[2 * id] / (R - 1), y = -1 + 2 * C.points[2 * id + 1] / (R - 1);
        lines.push(...F.gl(M.f(x, y)));
      }
    }
    const ld = { positions: lines.length ? lines : [0, 0, 0, 0, 0, 0] };
    if (!cut) cut = view.addLines(ld, { color: [0.1, 0.1, 0.1], width: 4 });
    else cut.setData(ld);
    if (!box) box = view.addLines(boxLines([-0.8, -0.64, -0.8], [0.8, 0.64, 0.8]), { width: 1.2, color: [0.6, 0.6, 0.65] });
    // readout
    cap3.textContent = `3D: φ(x) = ${M.axes}; drag to rotate`;
    if (model) {
      const err = P.filter(p => (p.c === 1 ? 1 : -1) * fx(p.x, p.y) <= 0).length;
      const w = model.w!;
      ro.textContent = `plane in 3D: ${w[0].toFixed(2)} φ₁ ${w[1] >= 0 ? "+" : "−"} ${Math.abs(w[1]).toFixed(2)} φ₂ ${w[2] >= 0 ? "+" : "−"} ${Math.abs(w[2]).toFixed(2)} φ₃ ${model.b >= 0 ? "+" : "−"} ${Math.abs(model.b).toFixed(2)} = 0\ntraining errors: ${err} of ${n}\nsupport vectors: ${model.sv.length}`;
    } else ro.textContent = "add points of both classes";
    note.textContent = st.note;
  }

  function load(p: Preset): void {
    st.map = (host.dataset.map as MapId) in MAPS && p === p0 ? (host.dataset.map as MapId) : p.map;
    st.note = p.note;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    W.set(p.points());
  }
  cS.set(st.logC);
  load(p0);
}

autoMount(NAME, mount);
