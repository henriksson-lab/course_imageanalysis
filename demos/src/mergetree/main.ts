// Hierarchical segmentation (graph-based segmentation day, unit 4). Initial regions
// (watershed basins of the gradient, FH at small k, or SLIC superpixels) are merged one pair at a
// time into a tree. Hierarchies: altitude (lowest pass on the gradient = single linkage = the
// watershed hierarchy), dynamics, area, volume (extinction values), or greedy merging of the
// region adjacency graph by difference of means. Panels: the partition at the current level,
// the ultrametric contour map (every boundary at the level where it disappears), and the
// dendrogram (y = merge order) with a draggable cut line. Click a region: its node is
// highlighted; "merge up" / "split" change the cut on that branch only.
// <div data-demo="mergetree" data-preset="altitude|dynamics|area|ucm|leak|mean|branch|cell"></div>
import { autoMount, htmlEl, svgEl, slider, button, presets, Slider } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gradient, fromArray } from "../lib/imageops";
import { inferno, labelColor } from "../lib/colour";
import { hMinimaMarkers, watershed } from "../lib/watershed";
import { rag, Rag, attributeTree, meanDiffTree, MergeTree, labelsFromNodes, ucmImage, dendrogramLayout, leavesOf } from "../lib/hierarchy";
import { gridDissim, felzenszwalb } from "../lib/mst";
import { slic } from "../lib/superpixels";
import { iouTable, match } from "../lib/eval";
import { nucleiScene, fhScene, leakScene } from "../fhseg/scenes";

const NAME = "mergetree";
const N = 256;
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 330px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .mt-part { cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .mt-dendro { flex: 1 1 100%; order: 5; }
.qb-demo[data-demo="${NAME}"] .mt-dendro svg { width: 100%; height: 150px; display: block; touch-action: none; cursor: ns-resize; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 380px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .mt-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .mt-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 16px; fill: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] figure { flex: 0 0 300px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 450px; max-width: 450px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .mt-dendro svg { height: 150px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .02em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .1em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .mt-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Img = "nuclei" | "scene" | "leak" | "cell";
type Init = "basins" | "fh" | "slic";
type Hier = "altitude" | "dynamics" | "area" | "volume" | "mean";
interface Settings { img: Img; init: Init; hier: Hier; regions: number; view: "mean" | "ucm" }
interface Preset { id: string; label: string; s: Partial<Settings>; select?: [number, number] }
const DEF: Settings = { img: "nuclei", init: "basins", hier: "altitude", regions: 20, view: "mean" };
const PRESETS: Preset[] = [
  { id: "altitude", label: "altitude: the watershed hierarchy", s: { hier: "altitude", regions: 20 } },
  { id: "dynamics", label: "dynamics vs altitude", s: { hier: "dynamics", regions: 20 } },
  { id: "area", label: "area: small regions first", s: { hier: "area", regions: 20 } },
  { id: "ucm", label: "UCM thresholds", s: { img: "cell", hier: "altitude", regions: 12 } },
  { id: "leak", label: "cheapest edge leaks…", s: { img: "leak", hier: "altitude", regions: 2 } },
  { id: "mean", label: "… mean difference does not", s: { img: "leak", hier: "mean", regions: 2 } },
  { id: "branch", label: "per-branch cut: one nucleus, one cluster", s: { hier: "dynamics", regions: 12 }, select: [0, 0] },
  { id: "cell", label: "micrograph", s: { img: "cell", hier: "dynamics", regions: 15 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: Settings = { ...DEF, ...p0.s };

  const f1 = htmlEl("figure", "", "", host);
  const cap1 = htmlEl("figcaption", "", "", f1);
  const cvP = htmlEl("canvas", "mt-part", "", f1); cvP.width = N; cvP.height = N;
  const f2 = htmlEl("figure", "", "", host);
  htmlEl("figcaption", "", "ultrametric contour map", f2);
  const cvU = htmlEl("canvas", "", "", f2); cvU.width = N; cvU.height = N;
  const credit = htmlEl("div", "mt-credit", CELL_CREDIT, f2);
  const panel = htmlEl("div", "qb-panel", "", host);
  const dd = htmlEl("div", "mt-dendro", "", host);
  const svg = svgEl("svg", { viewBox: "0 0 1000 150", preserveAspectRatio: "none" }, dd);
  htmlEl("div", "mt-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const ib = (["nuclei", "scene", "leak", "cell"] as Img[]).map(i => button(r1, i === "nuclei" ? "nuclei" : i === "scene" ? "test scene" : i === "leak" ? "weak spot" : "micrograph", () => { st.img = i; build(); }));
  const nb = (["basins", "fh", "slic"] as Init[]).map(i => button(r1, i === "basins" ? "leaves: basins" : i === "fh" ? "FH" : "SLIC", () => { st.init = i; build(); }));
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const HS: [Hier, string][] = [["altitude", "altitude"], ["dynamics", "dynamics"], ["area", "area"], ["volume", "volume"], ["mean", "RAG: mean diff."]];
  const hb = HS.map(([h, l]) => button(r2, l, () => { st.hier = h; makeTree(); }));
  const rS: Slider = slider(panel, "regions", 1, 200, 1, st.regions, v => { st.regions = v; custom = null; draw(); });
  const r3 = htmlEl("div", "qb-presets", "", panel);
  button(r3, "merge up", () => branch(+1));
  button(r3, "split", () => branch(-1));
  button(r3, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag on the dendrogram to move the cut. Click a region to select its branch; merge up / split change the cut on that branch only.", panel);

  function sync(): void {
    ib.forEach((b, i) => b.classList.toggle("active", ["nuclei", "scene", "leak", "cell"][i] === st.img));
    nb.forEach((b, i) => b.classList.toggle("active", ["basins", "fh", "slic"][i] === st.init));
    hb.forEach((b, i) => b.classList.toggle("active", HS[i][0] === st.hier));
  }
  let pendingSelect: [number, number] | null = null;
  function load(p: Preset): void {
    Object.assign(st, DEF, p.s);
    rS.set(st.regions);
    pendingSelect = p.select ?? null;
    pb.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    build();
  }

  // ---- data ----
  let cell: Float64Array | null = null, I: Float64Array = new Float64Array(N * N), truth: Int32Array | null = null;
  let leaves: Int32Array<ArrayBufferLike> = new Int32Array(N * N), R: Rag | null = null, T: MergeTree | null = null, grad: Float64Array = new Float64Array(N * N);
  let custom: number[] | null = null, selNode = -1, buildMs = 0;
  function build(): void {
    sync();
    if (st.img === "cell" && !cell) { loadCell(N).then(g => { cell = Float64Array.from(g.data); build(); }); return; }
    const t0 = performance.now();
    if (st.img === "cell") { I = cell!; truth = null; }
    else if (st.img === "nuclei") { const s = nucleiScene(6); I = s.I; truth = s.truth; }
    else if (st.img === "leak") { const s = leakScene(5); I = s.I; truth = s.truth; }
    else { const s = fhScene(4); I = s.I; truth = s.truth; }
    credit.style.visibility = st.img === "cell" ? "visible" : "hidden";
    grad = Float64Array.from(gradient(fromArray(N, N, I), 1.5).mag.data);
    if (st.init === "basins") {
      const mk = hMinimaMarkers(grad, N, N, st.img === "cell" ? 3 : 0.7, 8);
      leaves = watershed(grad, N, N, mk.data, { conn: 8 });
    } else if (st.init === "fh") {
      leaves = felzenszwalb(gridDissim([I], N, N, 8), 150, 40).labels;
    } else leaves = slic([I], N, N, { K: 300, m: 25, iters: 10 }).labels;
    R = rag(leaves, N, N, I, grad);
    buildMs = performance.now() - t0;
    makeTree();
  }
  function makeTree(): void {
    sync();
    if (!R) return;
    T = st.hier === "mean" ? meanDiffTree(R) : attributeTree(R, st.hier).tree;
    custom = null; selNode = -1;
    rS.input.max = String(Math.min(400, R.n));
    if (pendingSelect && truth) {
      // select the first truth nucleus that is merged with another at this level (a cluster)
      pendingSelect = null;
    }
    draw();
  }
  function levelFor(regions: number): number {
    const T_ = T!, m = T_.merges, k = Math.max(0, Math.min(m, T_.n - regions));
    return k === 0 ? -Infinity : T_.level[T_.n + k - 1];
  }
  function currentNodes(): number[] {
    if (custom) return custom;
    const T_ = T!, k = Math.max(0, Math.min(T_.merges, T_.n - st.regions));
    // the first k merges (merge order) form the cut
    const out: number[] = [], applied = new Uint8Array(T_.n + T_.merges);
    for (let i = 0; i < k; i++) applied[T_.n + i] = 1;
    for (let v = 0; v < T_.n + T_.merges; v++) {
      if (v >= T_.n && !applied[v]) continue;
      const p = T_.parent[v];
      if (p < 0 || !applied[p]) out.push(v);
    }
    return out;
  }
  function branch(dir: number): void {
    if (!T || selNode < 0) return;
    const nodes = currentNodes().slice();
    if (dir > 0) {
      const p = T.parent[selNode];
      if (p < 0) return;
      const under = new Set(leavesOf(T, p));
      const keep = nodes.filter(v => !leavesOf(T!, v).every(l => under.has(l)));
      keep.push(p); custom = keep; selNode = p;
    } else {
      if (selNode < T.n) return;
      const i = selNode - T.n;
      custom = nodes.filter(v => v !== selNode).concat([T.left[i], T.right[i]]);
      selNode = T.left[i];
    }
    draw();
  }

  function draw(): void {
    if (!T || !R) return;
    const nodes = currentNodes();
    const leafLab = labelsFromNodes(T, nodes);
    const lab = Int32Array.from(leaves, v => (v > 0 ? leafLab[v - 1] : 0));
    const nreg = nodes.length;
    // mean colours + boundaries
    const sum = new Float64Array(nreg + 1), cnt = new Float64Array(nreg + 1);
    for (let p = 0; p < N * N; p++) { sum[lab[p]] += I[p]; cnt[lab[p]]++; }
    const selLeaves = selNode >= 0 ? new Set(leavesOf(T, selNode).map(l => l + 1)) : null;
    const ctx = cvP.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const x = p % N, y = (p - x) / N, l = lab[p];
      const edge = (x + 1 < N && lab[p + 1] !== l) || (y + 1 < N && lab[p + N] !== l);
      const m = sum[l] / (cnt[l] || 1);
      let c = [m, m, m];
      if (selLeaves && selLeaves.has(leaves[p])) { const q = labelColor(2); c = [0.5 * m + 0.5 * 255 * q[0], 0.5 * m + 0.5 * 255 * q[1], 0.5 * m + 0.5 * 255 * q[2]]; }
      if (edge) c = [255, 230, 0];
      id.data[4 * p] = c[0]; id.data[4 * p + 1] = c[1]; id.data[4 * p + 2] = c[2]; id.data[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    // UCM (by merge order, so all hierarchies share a scale)
    const rankT: MergeTree = { ...T, level: Float64Array.from(T.level, (_, v) => (v < T!.n ? 0 : v - T!.n + 1)) };
    const u = ucmImage(rankT, leaves, N, N);
    const cu = cvU.getContext("2d")!, iu = cu.createImageData(N, N);
    const umax = Math.max(1, T.merges);
    for (let p = 0; p < N * N; p++) {
      const v = u[p] > 0 ? Math.pow(u[p] / umax, 2) : 0;
      const c = inferno(v);
      iu.data[4 * p] = 255 * c[0]; iu.data[4 * p + 1] = 255 * c[1]; iu.data[4 * p + 2] = 255 * c[2]; iu.data[4 * p + 3] = 255;
    }
    cu.putImageData(iu, 0, 0);
    // dendrogram
    drawDendro(nodes);
    cap1.textContent = `${custom ? "per-branch cut" : "cut"}: ${nreg} regions`;
    const lv = levelFor(st.regions);
    const lines = [`${R.n} leaves (${st.init === "basins" ? "watershed basins" : st.init === "fh" ? "FH regions" : "SLIC superpixels"}), ${R.a.length} RAG edges; built in ${buildMs.toFixed(0)} ms`,
      `hierarchy: ${HS.find(h => h[0] === st.hier)![1]}; ${custom ? "custom cut" : `level ${Number.isFinite(lv) ? fmt(lv) : "−"}`}: ${nreg} regions`];
    if (truth) {
      const nt = Math.max(...truth);
      const M = match(iouTable(truth, lab, nt, nreg), 0.5, "greedy");
      lines.push(`truth objects matched (IoU ≥ 0.5): ${M.tp} of ${nt}`);
    }
    if (selNode >= 0) lines.push(`selected node: ${leavesOf(T, selNode).length} leaves, level ${fmt(T.level[selNode])}`);
    ro.textContent = lines.join("\n");
  }
  const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(2) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));

  function drawDendro(nodes: number[]): void {
    const T_ = T!;
    svg.innerHTML = "";
    const { x } = dendrogramLayout(T_);
    const n = T_.n, m = T_.merges, W = 1000, Hh = 150, top = 8, bot = 142;
    const X = (v: number) => 10 + (x[v] + 0.5) / n * (W - 20);
    const Y = (v: number) => (v < n ? bot : bot - (v - n + 1) / Math.max(1, m) * (bot - top));
    const sel = selNode >= 0 ? new Set<number>() : null;
    if (sel) { const st2 = [selNode]; while (st2.length) { const u = st2.pop()!; sel.add(u); if (u >= n) st2.push(T_.left[u - n], T_.right[u - n]); } }
    let d = "", ds = "";
    for (let i = 0; i < m; i++) {
      const v = n + i, a = T_.left[i], b = T_.right[i];
      const seg = `M${X(a).toFixed(1)},${Y(a).toFixed(1)}V${Y(v).toFixed(1)}H${X(b).toFixed(1)}V${Y(b).toFixed(1)}`;
      if (sel && sel.has(v)) ds += seg; else d += seg;
    }
    svgEl("path", { d, fill: "none", stroke: "var(--qb-fg, #333)", "stroke-width": 1.2, "vector-effect": "non-scaling-stroke" }, svg);
    if (ds) svgEl("path", { d: ds, fill: "none", stroke: "#d62728", "stroke-width": 2.5, "vector-effect": "non-scaling-stroke" }, svg);
    for (const v of nodes) svgEl("circle", { cx: X(v), cy: Y(v), r: 3, fill: "#ff9f1c" }, svg);
    if (!custom) {
      const k = Math.max(0, Math.min(m, n - st.regions)), yc = bot - (k + 0.5) / Math.max(1, m) * (bot - top);
      svgEl("line", { x1: 0, x2: W, y1: yc, y2: yc, stroke: "#1f77b4", "stroke-width": 2, "stroke-dasharray": "8 5", "vector-effect": "non-scaling-stroke" }, svg);
    }
    void Hh;
  }
  svg.addEventListener("pointerdown", ev => {
    svg.setPointerCapture(ev.pointerId);
    const move = (e: PointerEvent) => {
      if (!T) return;
      const r = svg.getBoundingClientRect(), fy = (e.clientY - r.top) / r.height * 150;
      const k = Math.round((142 - fy) / 134 * T.merges);
      st.regions = Math.max(1, Math.min(T.n, T.n - Math.max(0, Math.min(T.merges, k))));
      rS.set(Math.min(st.regions, Number(rS.input.max)));
      custom = null; draw();
    };
    move(ev);
    const up = () => { svg.removeEventListener("pointermove", move); svg.removeEventListener("pointerup", up); };
    svg.addEventListener("pointermove", move); svg.addEventListener("pointerup", up);
  });
  cvP.addEventListener("pointerdown", ev => {
    if (!T) return;
    const r = cvP.getBoundingClientRect();
    const px = Math.floor((ev.clientX - r.left) / r.width * N), py = Math.floor((ev.clientY - r.top) / r.height * N);
    const leaf = leaves[Math.min(N - 1, py) * N + Math.min(N - 1, px)] - 1;
    if (leaf < 0) return;
    const nodes = currentNodes();
    selNode = nodes.find(v => leavesOf(T!, v).includes(leaf)) ?? -1;
    draw();
  });

  load(p0);
}

autoMount(NAME, mount);
