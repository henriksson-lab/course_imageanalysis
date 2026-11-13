// Spanning trees on a small grid of numbers (graph-based segmentation day, units 2 and 3).
// Cells are pixels; 4-neighbour edges carry a weight shown between the cells: the
// dissimilarity |f_p − f_q| or the altitude max(f_p, f_q) (watershed). Step through:
//  - Kruskal: edges in sorted order, "join" or "same component: skip" (union–find);
//  - single linkage at t: Kruskal that only uses edges with d < t;
//  - Felzenszwalb–Huttenlocher with k: merge if d ≤ min(Int(C) + k/|C|) over both components;
//  - Prim from markers: grow trees from the marked cells, always along the cheapest edge
//    leaving them; an edge between two different trees is a cut edge (the watershed);
//  - minimax path: the minimum spanning tree and the path between two clicked cells, with its
//    heaviest edge.
// Click a cell: upper half +1, lower half −1 (edit), or toggle a marker / pick a path end.
// <div data-demo="graphgrid" data-preset="fh1d|ramp|texture|fhk|prim|plateau|kruskal|minimax"></div>
import { autoMount, htmlEl, svgEl, slider, button, presets, Slider } from "../lib/ui";
import { DGraph, gridDissim, sortEdges, kruskal, fhTrace, primForest, treePath, DissimMode } from "../lib/mst";
import { UnionFind } from "../lib/label";
import { labelColor, toCss } from "../lib/colour";

const NAME = "graphgrid";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .7em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .gg-fig { flex: 1 1 560px; max-width: 640px; }
.qb-demo[data-demo="${NAME}"] .gg-fig svg { width: 100%; height: auto; display: block; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .gg-list { flex: 0 1 190px; font-family: var(--qb-mono, monospace); font-size: 17px; white-space: pre; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .gg-list .cur { background: #ffe08a; }
.qb-demo[data-demo="${NAME}"] .gg-list .skip { color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .gg-list .cut { color: #d62728; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 320px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; min-height: 4.2em; }
.qb-demo[data-demo="${NAME}"] .gg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] svg text { font-size: 20px; }
.qb-demo[data-demo="${NAME}"] svg text.gg-val { font-size: 22px; font-weight: 600; }
.qb-demo[data-demo="${NAME}"] svg text.gg-w { font-size: 18px; }
/* Pages: grid, edge list and controls side by side, the presets as a row under them, so the
   demo fits a 1366x768 window. */
@media (min-width: 641px) {
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) { flex-wrap: wrap; gap: .5em .7em; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .gg-fig { flex: 1 1 400px; max-width: min(560px, calc((100vh - 290px) * 1.25)); min-width: 0; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .gg-fig svg { max-height: calc(100vh - 290px); }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .gg-list { flex: 0 0 165px; font-size: 15px; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .qb-panel { flex: 0 1 300px; min-width: 0; gap: .25em; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .qb-readout { font-size: 15px; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .qb-slider { grid-template-columns: auto minmax(3em, 1fr) 2.2em; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .qb-slider input[type=range] { min-width: 3em; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .qb-note { font-size: 13px; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .gg-pre { flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
  .qb-demo[data-demo="${NAME}"]:not(.reveal *) .gg-pre .gg-lab { margin: 0; }
}
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .gg-fig { flex: 0 0 540px; }
.reveal .qb-demo[data-demo="${NAME}"] .gg-list { flex: 0 0 180px; font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 360px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; padding: .15em .4em; }
.reveal .qb-demo[data-demo="${NAME}"].gg-1d { flex-wrap: wrap; row-gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"].gg-1d .gg-fig { flex: 0 0 100%; max-width: 1000px; }
.reveal .qb-demo[data-demo="${NAME}"].gg-1d .gg-list { flex: 0 0 300px; font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .gg-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Algo = "kruskal" | "single" | "fh" | "prim" | "path";
type EdgeState = 0 | 1 | 2 | 3;   // 0 untouched, 1 in the tree / joined, 2 skipped / kept apart, 3 cut (between trees)
interface Frame { es: Uint8Array; comp: Int32Array; cur: number; log: string; list: { e: number; cls: string; text: string }[] }
interface Preset { id: string; label: string; rows: number; cols: number; values: number[]; algo: Algo; mode: DissimMode; t?: number; k?: number; markers?: [number, number][]; path?: [number, number] }

const P: Preset[] = [
  { id: "fh1d", label: "the 1D example (FH, k = 8)", rows: 1, cols: 12, values: [0, 6, 0, 6, 20, 20, 20, 20, 24, 24, 24, 24], algo: "fh", mode: "absdiff", k: 8 },
  { id: "kruskal", label: "Kruskal on a small image", rows: 4, cols: 5, values: [1, 2, 2, 8, 9, 1, 3, 7, 8, 8, 2, 2, 9, 9, 7, 1, 4, 8, 6, 7], algo: "kruskal", mode: "absdiff" },
  { id: "ramp", label: "chaining: a ramp (single linkage)", rows: 4, cols: 8, values: [0, 0, 3, 6, 9, 12, 15, 15, 0, 0, 3, 6, 9, 12, 15, 15, 0, 0, 3, 6, 9, 12, 15, 15, 0, 0, 3, 6, 9, 12, 15, 15], algo: "single", mode: "absdiff", t: 4 },
  { id: "texture", label: "texture next to a faint step (single linkage)", rows: 4, cols: 8, values: [0, 6, 0, 6, 20, 20, 24, 24, 6, 0, 6, 0, 20, 20, 24, 24, 0, 6, 0, 6, 20, 20, 24, 24, 6, 0, 6, 0, 20, 20, 24, 24], algo: "single", mode: "absdiff", t: 7 },
  { id: "fhk", label: "the same with FH (k = 8)", rows: 4, cols: 8, values: [0, 6, 0, 6, 20, 20, 24, 24, 6, 0, 6, 0, 20, 20, 24, 24, 0, 6, 0, 6, 20, 20, 24, 24, 6, 0, 6, 0, 20, 20, 24, 24], algo: "fh", mode: "absdiff", k: 8 },
  { id: "prim", label: "Prim from two markers (watershed)", rows: 4, cols: 4, values: [1, 3, 9, 4, 2, 5, 8, 6, 3, 7, 2, 7, 4, 9, 6, 0], algo: "prim", mode: "altitude", markers: [[0, 0], [3, 3]] },
  { id: "plateau", label: "a plateau: ties", rows: 3, cols: 7, values: [0, 5, 5, 5, 5, 5, 0, 1, 5, 5, 5, 5, 5, 1, 0, 5, 5, 5, 5, 5, 0], algo: "prim", mode: "altitude", markers: [[1, 0], [1, 6]] },
  { id: "minimax", label: "minimax path in the MST", rows: 4, cols: 5, values: [1, 2, 2, 8, 9, 1, 3, 7, 8, 8, 2, 2, 9, 9, 7, 1, 4, 8, 6, 7], algo: "path", mode: "absdiff", path: [0, 19] },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = P.find(p => p.id === host.dataset.preset) ?? P[0];
  let rows = 0, cols = 0, vals: number[] = [], algo: Algo = "kruskal", mode: DissimMode = "absdiff", t = 4, k = 8;
  let markers = new Int32Array(0), pathEnds: number[] = [], frames: Frame[] = [], fi = 0, G: DGraph = { n: 0, p: new Int32Array(0), q: new Int32Array(0), d: new Float64Array(0) };
  let clickMode: "edit" | "marker" | "path" = "edit", timer = 0;

  const fig = htmlEl("div", "gg-fig", "", host);
  const svg = svgEl("svg", {}, fig);
  const listEl = htmlEl("div", "gg-list", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  // On a page the presets go in a row under the grid (fits the window); on a slide in the panel.
  const preBox = host.closest(".reveal") ? panel : htmlEl("div", "gg-pre", "", host);
  htmlEl("div", "gg-lab", "presets", preBox);
  const pb = presets(preBox, P.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "gg-lab", "algorithm", panel);
  const arow = htmlEl("div", "qb-presets", "", panel);
  const ALGOS: [Algo, string][] = [["kruskal", "Kruskal"], ["single", "single linkage"], ["fh", "FH"], ["prim", "Prim from markers"], ["path", "minimax path"]];
  const ab = ALGOS.map(([a, l]) => button(arow, l, () => { algo = a; if (a === "prim") clickMode = "marker"; else if (a === "path") clickMode = "path"; else clickMode = "edit"; rebuild(); }));
  const mrow = htmlEl("div", "qb-presets", "", panel);
  const mb = (["absdiff", "altitude"] as DissimMode[]).map(m => button(mrow, m === "absdiff" ? "|f_p − f_q|" : "max(f_p, f_q)", () => { mode = m; rebuild(); }));
  const tS: Slider = slider(panel, "t (single linkage)", 1, 30, 1, t, v => { t = v; if (algo === "single") rebuild(); });
  const kS: Slider = slider(panel, "k (FH)", 0, 60, 1, k, v => { k = v; if (algo === "fh") rebuild(); });
  const crow = htmlEl("div", "qb-presets", "", panel);
  button(crow, "⏮", () => go(0)); button(crow, "◀", () => go(fi - 1)); button(crow, "▶ step", () => go(fi + 1));
  const playB = button(crow, "play", () => play()); button(crow, "⏭", () => go(frames.length - 1));
  const erow = htmlEl("div", "qb-presets", "", panel);
  const cb = (["edit", "marker", "path"] as const).map(m => button(erow, m === "edit" ? "click: edit ±1" : m === "marker" ? "click: markers" : "click: path ends", () => { clickMode = m; sync(); }));
  button(erow, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Numbers in the cells are pixel values; numbers between cells are edge weights. Thick dark: tree edges (joined); faint: skipped (would close a cycle, or rejected by the test); red: cut between two trees. Colours: current components.", panel);

  function sync(): void {
    pb.forEach(b => b.classList.remove("active"));
    ab.forEach((b, i) => b.classList.toggle("active", ALGOS[i][0] === algo));
    mb.forEach((b, i) => b.classList.toggle("active", ["absdiff", "altitude"][i] === mode));
    cb.forEach((b, i) => b.classList.toggle("active", ["edit", "marker", "path"][i] === clickMode));
  }
  function load(p: Preset): void {
    rows = p.rows; cols = p.cols; vals = p.values.slice(); algo = p.algo; mode = p.mode;
    t = p.t ?? t; k = p.k ?? k; tS.set(t); kS.set(k);
    markers = new Int32Array(rows * cols);
    (p.markers ?? []).forEach(([r, c], i) => { markers[r * cols + c] = i + 1; });
    pathEnds = p.path ? p.path.slice() : [];
    clickMode = algo === "prim" ? "marker" : algo === "path" ? "path" : "edit";
    rebuild();
    pb.forEach((b, i) => b.classList.toggle("active", P[i] === p));
    go(p === p0 && host.dataset.step === "end" ? frames.length - 1 : 0);
  }
  const name = (v: number) => (rows === 1 ? "abcdefghijklmnop"[v] : `${Math.floor(v / cols) + 1},${v % cols + 1}`);
  const ename = (e: number) => `${name(G.p[e])}–${name(G.q[e])}`;
  const f2 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

  function rebuild(): void {
    stop();
    G = gridDissim([vals], cols, rows, 4, mode);
    frames = [];
    const m = G.p.length, n = rows * cols;
    const compOf = (uf: UnionFind) => Int32Array.from({ length: n }, (_, v) => uf.find(v));
    const order = sortEdges(G);
    const listAll = (es: Uint8Array, cur: number, verdict: (e: number) => string) => Array.from(order, e => ({ e, cls: e === cur ? "cur" : es[e] === 2 ? "skip" : es[e] === 3 ? "cut" : "", text: `${ename(e).padEnd(rows === 1 ? 4 : 8)}${String(f2(G.d[e])).padStart(3)} ${verdict(e)}` }));
    if (algo === "kruskal" || algo === "single") {
      const K = kruskal(G, order, true), uf = new UnionFind(n), es = new Uint8Array(m), verd = new Map<number, string>();
      frames.push({ es: es.slice(), comp: compOf(uf), cur: -1, log: algo === "single" ? `Single linkage at t = ${t}: join across every edge with d < ${t}.` : "Kruskal: take the edges in order of weight.", list: listAll(es, -1, () => "") });
      for (const s of K.steps!) {
        if (algo === "single" && G.d[s.edge] >= t) {
          frames.push({ es: es.slice(), comp: compOf(uf), cur: s.edge, log: `${ename(s.edge)}: d = ${f2(G.d[s.edge])} ≥ t: stop. ${countComps(uf, n)} regions.`, list: listAll(es, s.edge, e => verd.get(e) ?? "") });
          break;
        }
        es[s.edge] = s.join ? 1 : 2;
        verd.set(s.edge, s.join ? "join" : "skip");
        if (s.join) uf.union(G.p[s.edge], G.q[s.edge]);
        frames.push({ es: es.slice(), comp: compOf(uf), cur: s.edge, log: `${ename(s.edge)} (d = ${f2(G.d[s.edge])}): ${s.join ? "different components: join" : "same component: skip (it would close a cycle)"}. ${countComps(uf, n)} components.`, list: listAll(es, s.edge, e => verd.get(e) ?? "") });
      }
      if (algo === "kruskal") frames[frames.length - 1].log += `  Tree weight ${f2(K.total)}.`;
    } else if (algo === "fh") {
      const tr = fhTrace(G, k, order), es = new Uint8Array(m), verd = new Map<number, string>();
      frames.push({ es: es.slice(), comp: Int32Array.from({ length: n }, (_, v) => v), cur: -1, log: `FH with k = ${k}: merge across d if d ≤ Int(C) + k/|C| for both components.`, list: listAll(es, -1, () => "") });
      for (const s of tr) {
        es[s.edge] = s.merge ? 1 : 2;
        verd.set(s.edge, s.same ? "same" : s.merge ? "merge" : "keep");
        const thr = (i: number, z: number, th: number) => `${f2(i)} + ${k}/${z} = ${th.toFixed(2)}`;
        const log = s.same ? `${ename(s.edge)} (d = ${f2(s.d)}): same component: skip.`
          : `${ename(s.edge)} (d = ${f2(s.d)}): thresholds ${thr(s.intA, s.sizeA, s.thrA)} and ${thr(s.intB, s.sizeB, s.thrB)} → ${s.merge ? "merge" : "keep apart"}.`;
        frames.push({ es: es.slice(), comp: s.comp, cur: s.edge, log, list: listAll(es, s.edge, e => verd.get(e) ?? "") });
      }
      const last = frames[frames.length - 1];
      last.log += `  ${new Set(Array.from(last.comp)).size} regions.`;
    } else if (algo === "prim") {
      const pr = primForest(G, markers, true), es = new Uint8Array(m), lab = Int32Array.from(markers);
      const queued: number[] = [];
      const queueList = (cur: number) => queued.filter(e => es[e] === 0 || e === cur).sort((a, b) => G.d[a] - G.d[b] || queued.indexOf(a) - queued.indexOf(b)).map(e => ({ e, cls: e === cur ? "cur" : "", text: `${ename(e).padEnd(rows === 1 ? 4 : 8)}${String(f2(G.d[e])).padStart(3)}` }));
      const nb = (v: number) => { for (let e = 0; e < m; e++) if ((G.p[e] === v || G.q[e] === v) && !queued.includes(e)) queued.push(e); };
      for (let v = 0; v < n; v++) if (markers[v]) nb(v);
      const mk = Array.from(markers).some(v => v > 0);
      frames.push({ es: es.slice(), comp: lab.slice(), cur: -1, log: mk ? "Prim from all markers: the queue holds the edges leaving the labelled cells, cheapest first." : "Click cells to place markers (two or more).", list: queueList(-1) });
      for (const s of pr.steps!) {
        es[s.edge] = s.kind === "grow" ? 1 : s.kind === "cut" ? 3 : 2;
        const list = queueList(s.edge);
        if (s.kind === "grow") { lab[s.to] = s.label; nb(s.to); }
        const log = s.kind === "grow" ? `${ename(s.edge)} (${f2(s.d)}): ${name(s.to)} joins tree ${"AB CDEFG"[s.label - 1] ?? s.label}.`
          : s.kind === "cut" ? `${ename(s.edge)} (${f2(s.d)}): joins two different trees: a cut edge (watershed).` : `${ename(s.edge)} (${f2(s.d)}): both ends already in the same tree: skip.`;
        frames.push({ es: es.slice(), comp: lab.slice(), cur: s.edge, log, list });
      }
    } else {
      const K = kruskal(G, order), es = new Uint8Array(m);
      for (let e = 0; e < m; e++) es[e] = K.inTree[e] ? 1 : 2;
      let log = "Click two cells for the path between them in the minimum spanning tree.";
      let cur = -1;
      if (pathEnds.length === 2) {
        const pth = treePath(G, K.inTree, pathEnds[0], pathEnds[1]);
        if (pth) {
          for (const e of pth.edges) es[e] = 3;
          cur = pth.maxEdge;
          log = `Tree path ${name(pathEnds[0])} → ${name(pathEnds[1])}: edges ${pth.edges.map(e => f2(G.d[e])).join(", ")}. Heaviest: ${f2(G.d[pth.maxEdge])} = the minimax distance (no path has a lower highest edge).`;
        }
      }
      frames.push({ es, comp: new Int32Array(n), cur, log, list: Array.from(order, e => ({ e, cls: es[e] === 3 ? "cut" : es[e] === 2 ? "skip" : "", text: `${ename(e).padEnd(rows === 1 ? 4 : 8)}${String(f2(G.d[e])).padStart(3)}` })) });
    }
    fi = algo === "path" ? 0 : Math.min(fi, frames.length - 1);
    if (algo !== "path") fi = 0;
    sync();
    render();
  }
  function countComps(uf: UnionFind, n: number): number { const s = new Set<number>(); for (let v = 0; v < n; v++) s.add(uf.find(v)); return s.size; }
  function go(i: number): void { fi = Math.max(0, Math.min(frames.length - 1, i)); render(); }
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; playB.textContent = "play"; } }
  function play(): void {
    if (timer) { stop(); return; }
    if (fi >= frames.length - 1) fi = 0;
    playB.textContent = "pause";
    timer = window.setInterval(() => { if (fi >= frames.length - 1) stop(); else go(fi + 1); }, 700);
  }

  function render(): void {
    const F = frames[fi];
    host.classList.toggle("gg-1d", rows === 1);
    const C = rows === 1 ? 46 : 58, Gp = rows === 1 ? 26 : 30, pad = 6;
    const W = cols * C + (cols - 1) * Gp + 2 * pad, H = rows * C + (rows - 1) * Gp + 2 * pad;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const X = (c: number) => pad + c * (C + Gp), Y = (r: number) => pad + r * (C + Gp);
    // edges
    for (let e = 0; e < G.p.length; e++) {
      const a = G.p[e], b = G.q[e], ra = Math.floor(a / cols), ca = a % cols, rb = Math.floor(b / cols), cb2 = b % cols;
      const x1 = X(ca) + C / 2, y1 = Y(ra) + C / 2, x2 = X(cb2) + C / 2, y2 = Y(rb) + C / 2;
      const s = F.es[e], cur = e === F.cur;
      const stroke = s === 3 ? "#d62728" : s === 1 ? "var(--qb-fg, #222)" : "var(--qb-grid, #ccc)";
      svgEl("line", { x1, y1, x2, y2, stroke: cur ? "#ff9f1c" : stroke, "stroke-width": cur ? 14 : s === 1 || s === 3 ? 9 : 3, "stroke-linecap": "round", opacity: s === 2 && !cur ? 0.5 : 1 }, svg);
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      svgEl("rect", { x: mx - 13, y: my - 12, width: 26, height: 24, rx: 5, fill: "var(--qb-bg, #fff)", stroke: cur ? "#ff9f1c" : "none", "stroke-width": 2 }, svg);
      const tx = svgEl("text", { x: mx, y: my + 6, "text-anchor": "middle", class: "gg-w", fill: s === 3 ? "#d62728" : "var(--qb-fg, #222)" }, svg);
      tx.textContent = f2(G.d[e]);
    }
    // cells
    const n = rows * cols;
    const compIds = Array.from(new Set(Array.from(F.comp)));
    for (let v = 0; v < n; v++) {
      const r = Math.floor(v / cols), c = v % cols;
      let fill = "var(--qb-panel, #f4f4f4)";
      if (algo === "prim") { if (F.comp[v] > 0) fill = toCss(F.comp[v] === 1 ? [1, 0.75, 0.5] : F.comp[v] === 2 ? [0.6, 0.78, 0.95] : labelColor(F.comp[v] + 2)); }
      else if (algo !== "path") { const cid = F.comp[v], size = Array.from(F.comp).filter(x => x === cid).length; if (size > 1) fill = toCss(mix(labelColor(compIds.indexOf(cid) + 1))); }
      const g = svgEl("g", { style: "cursor:pointer" }, svg);
      const mkd = markers[v] > 0 && algo === "prim", pe = algo === "path" && pathEnds.includes(v);
      svgEl("rect", { x: X(c), y: Y(r), width: C, height: C, rx: 6, fill, stroke: mkd || pe ? "#000" : "var(--qb-axis, #666)", "stroke-width": mkd || pe ? 4 : 1.5 }, g);
      const tv = svgEl("text", { x: X(c) + C / 2, y: Y(r) + C / 2 + 8, "text-anchor": "middle", class: "gg-val" }, g);
      tv.textContent = String(vals[v]);
      if (mkd) { const tm = svgEl("text", { x: X(c) + 8, y: Y(r) + 18, class: "gg-w" }, g); tm.textContent = "AB CDEFG"[markers[v] - 1] ?? "?"; }
      g.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        if (clickMode === "edit") {
          const rc = (ev.currentTarget as SVGGElement).getBoundingClientRect();
          vals[v] = Math.max(0, Math.min(99, vals[v] + (ev.clientY < rc.top + rc.height / 2 ? 1 : -1)));
        } else if (clickMode === "marker") {
          if (markers[v]) markers[v] = 0; else { const used = new Set(Array.from(markers)); let l = 1; while (used.has(l)) l++; markers[v] = l; }
          if (algo !== "prim") algo = "prim";
        } else {
          pathEnds.push(v); if (pathEnds.length > 2) pathEnds = [v];
          algo = "path";
        }
        const keep = fi;
        rebuild();
        if (algo === "path" || clickMode === "edit") go(Math.min(keep, frames.length - 1));
      });
    }
    // edge list / queue
    listEl.innerHTML = "";
    htmlEl("div", "", algo === "prim" ? "queue (cheapest first)" : "edges by weight", listEl).style.color = "var(--qb-muted)";
    const L = F.list, ci = Math.max(0, L.findIndex(x => x.cls === "cur"));
    const from = Math.max(0, Math.min(L.length - 14, ci - 6));
    if (from > 0) htmlEl("div", "skip", "  …", listEl);
    for (const it of L.slice(from, from + 14)) htmlEl("div", it.cls, it.text, listEl);
    if (from + 14 < L.length) htmlEl("div", "skip", "  …", listEl);
    ro.textContent = `step ${fi} / ${frames.length - 1}\n${F.log}`;
  }
  const mix = (c: [number, number, number]): [number, number, number] => [0.55 + 0.45 * c[0], 0.55 + 0.45 * c[1], 0.55 + 0.45 * c[2]];

  load(p0);
}

autoMount(NAME, mount);
