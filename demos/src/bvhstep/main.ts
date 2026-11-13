// bvhstep: acceleration structures for ray tracing, in 2D (ray-tracing day, unit bvh/). A scene
// of N discs and segments (seeded layouts: uniform, clustered, "teapot in a stadium", one long
// wall). Structure: none (brute force), BVH split at the median, BVH by the surface area
// heuristic (perimeter in 2D), or a uniform grid. The boxes are drawn level by level (or the
// grid). Drag the ray's origin (O) and aim (◆); step through the traversal: visited boxes or
// cells in order, tested primitives outlined, the hit marked. Counters: box tests and primitive
// tests against brute force. Plot: average tests per ray against N (1000 random rays per point)
// for brute force, the grid and the SAH BVH, with O(N) and O(log N) reference lines.
// Options: data-preset (brute, median, sah, grid, stadium, miss, early).
import { htmlEl, slider, button, presets, autoMount } from "../lib/ui";
import { rng } from "../lib/random";
import * as B from "../lib/bvh";
import { Layout, Struct, Prim, State, makeScene, primBox, primHit } from "./scene";
import { Plot, seriesColor } from "../lib/plot";

const NAME = "bvhstep";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .7em; }
${S} .bv-c1 { flex: 0 1 470px; min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
${S} .bv-c2 { flex: 1 1 300px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; }
${S} .bv-c3 { flex: 1 1 300px; max-width: 420px; }
${S} canvas.bv-cv { width: 100%; aspect-ratio: 1; display: block; border: 1px solid var(--qb-grid); border-radius: 4px; touch-action: none; cursor: crosshair; }
${S} .bv-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-family: system-ui, sans-serif; white-space: pre; line-height: 1.3; }
/* pages: the scene on the left, controls and the plot under them in a column on the right */
${S} { display: grid; gap: .6em 1em; align-items: start; justify-content: start;
  grid-template-columns: minmax(240px, min(460px, calc(100vh - 250px))) minmax(0, 420px); grid-template-rows: auto 1fr; }
${S} .bv-c1 { grid-row: 1 / span 2; }
${S} .bv-c2, ${S} .bv-c3 { max-width: none; }
${S} .bv-c2 .qb-btn { font-size: 14px; padding: .15em .5em; }
${S} .bv-c3 { max-width: 380px; }
@media (max-width: 640px) { ${S} { grid-template-columns: minmax(0, 1fr); } ${S} .bv-c1 { grid-row: auto; } }
.reveal ${S} { display: flex; flex-wrap: nowrap; font-size: 18px; gap: .5em; }
.reveal ${S} .bv-c1 { flex: 0 0 440px; }
.reveal ${S} .bv-c2 { flex: 0 0 330px; }
.reveal ${S} .bv-c3 { flex: 0 0 360px; max-width: 420px; }
.reveal ${S} .qb-panel { flex: 0 0 330px; gap: .2em; padding: .3em .45em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-presets { gap: .2em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

const DEF: State = { n: 64, layout: "uniform", struct: "sah", cells: 8, level: 3, o: [6, 12], aim: [60, 62], step: -1 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "brute", label: "brute force", s: { struct: "none", n: 64 } },
  { id: "median", label: "median BVH", s: { struct: "median", n: 64, level: 2, step: 0 } },
  { id: "sah", label: "SAH vs median, clustered", s: { struct: "sah", layout: "clustered", n: 256, level: 2 } },
  { id: "grid", label: "grid on a uniform scene", s: { struct: "grid", layout: "uniform", n: 256, cells: 12 } },
  { id: "stadium", label: "teapot in a stadium", s: { struct: "grid", layout: "stadium", n: 1024, cells: 16, o: [8, 10], aim: [54, 46] } },
  { id: "miss", label: "a ray that misses", s: { struct: "sah", n: 256, o: [1.5, 1.5], aim: [0.5, 1] } },
  { id: "early", label: "near-first stops early", s: { struct: "sah", n: 1024, o: [3, 50], aim: [50, 50], level: 4 } },
];

const NS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const fresh = (s: Partial<State> = {}): State => ({ ...DEF, ...s, o: [...(s.o ?? DEF.o)] as [number, number], aim: [...(s.aim ?? DEF.aim)] as [number, number] });
  let st = fresh(p0?.s);

  const c1 = htmlEl("div", "bv-c1", "", host);
  const cv = htmlEl("canvas", "bv-cv", "", c1);
  const r1 = htmlEl("div", "bv-row", "", c1);
  button(r1, "step", () => { st.step = st.step < 0 ? 1 : st.step + 1; draw(); });
  button(r1, "all", () => { st.step = -1; draw(); });
  button(r1, "restart", () => { st.step = 0; draw(); });
  const ro = htmlEl("div", "qb-readout", "", c1);

  const c2 = htmlEl("div", "bv-c2", "", host);
  const pb = presets(c2, [...PRESETS.map(p => ({ label: p.label, apply: () => { st = fresh(p.s); sync(); } })), { label: "reset", apply: () => { st = fresh(); sync(); pb.forEach(b => b.classList.remove("active")); } }]);
  const sr = htmlEl("div", "bv-row", "", c2);
  const structBtns = (["none", "median", "sah", "grid"] as Struct[]).map(s => ({ s, b: button(sr, s === "none" ? "none" : s === "median" ? "BVH median" : s === "sah" ? "BVH SAH" : "grid", () => { st.struct = s; st.step = -1; draw(); }) }));
  const lr = htmlEl("div", "bv-row", "", c2);
  const layBtns = (["uniform", "clustered", "stadium", "wall"] as Layout[]).map(l => ({ l, b: button(lr, l === "stadium" ? "teapot/stadium" : l === "wall" ? "long wall" : l, () => { st.layout = l; st.step = -1; rebuild(); curve(); draw(); }) }));
  const nS = slider(c2, "N objects", 3, 12.3, 0.1, Math.log2(st.n), v => { st.n = Math.round(2 ** v); st.step = -1; rebuild(); draw(); }, v => String(Math.round(2 ** v)));
  const lvS = slider(c2, "box level", 0, 14, 1, st.level, v => { st.level = v; draw(); });
  const cS = slider(c2, "grid cells", 2, 48, 1, st.cells, v => { st.cells = v; rebuild(); draw(); });
  const c3 = htmlEl("div", "bv-c3", "", host);
  const fig = htmlEl("div", "", "", c3);
  const plot = new Plot(fig, { width: 350, height: 250, x: [6, 5000], y: [1, 30000], xlog: true, ylog: true, xlabel: "N objects", ylabel: "tests per ray", font: 20, margin: { l: 76, b: 58, t: 8, r: 12 } });
  const lBrute = plot.line([], [], { color: seriesColor(1), label: "brute force" });
  const lGrid = plot.line([], [], { color: seriesColor(2), label: "grid" });
  const lBvh = plot.line([], [], { color: seriesColor(0), label: "BVH (SAH)" });
  const lLog = plot.line([], [], { color: "var(--qb-muted)", dash: "6 5", width: 2, label: "4 log₂ N" });
  const cur = plot.vline(64, { color: "#e0a800", width: 2 });
  plot.legend(undefined, "tl");

  let prims: Prim[] = [], boxes: B.AABB[] = [], grid: B.Grid | null = null, bvhDepth = 0;
  function rebuild(): void {
    prims = makeScene(st.n, st.layout); boxes = prims.map(primBox);
    const bm = B.buildBVH(boxes, { method: "median" }), bs = B.buildBVH(boxes, { method: "sah" });
    bvhCache = { median: bm, sah: bs };
    bvhDepth = Math.max(B.bvhStats(bm).depth, B.bvhStats(bs).depth);
    grid = B.buildGrid(boxes, [0, 0], [100, 100], [st.cells, st.cells]);
    lvS.input.max = String(bvhDepth);
  }
  let bvhCache: { median: B.BVH; sah: B.BVH } | null = null;

  function curve(): void {
    const r = rng(99), gx: number[] = [], gy: number[] = [], bx: number[] = [], by: number[] = [];
    for (const n of NS) {
      const pr = makeScene(n, st.layout), bx_ = pr.map(primBox), bv = B.buildBVH(bx_, { method: "sah" });
      const k = Math.max(2, Math.round(Math.sqrt(n / 2))), gr = B.buildGrid(bx_, [0, 0], [100, 100], [k, k]);
      let sb = 0, sg = 0;
      const R = 1000;
      for (let i = 0; i < R; i++) {
        const a = 2 * Math.PI * r.unif(), o = [50 + 70 * Math.cos(a), 50 + 70 * Math.sin(a)], t = [100 * r.unif(), 100 * r.unif()];
        const L = Math.hypot(t[0] - o[0], t[1] - o[1]), d = [(t[0] - o[0]) / L, (t[1] - o[1]) / L];
        const hit = (p: number, tm: number) => primHit(pr[p], o, d, tm);
        const b = B.traverseBVH(bv, o, d, hit); sb += b.boxTests + b.primTests;
        const g = B.traverseGrid(gr, o, d, hit); sg += g.cells + g.primTests;
      }
      bx.push(n); by.push(sb / R); gx.push(n); gy.push(sg / R);
    }
    lBrute.set(NS, NS); lGrid.set(gx, gy); lBvh.set(bx, by); lLog.set(NS, NS.map(n => 4 * Math.log2(n)));
  }

  function sync(): void { nS.set(Math.log2(st.n)); lvS.set(st.level); cS.set(st.cells); rebuild(); curve(); draw(); }

  const toPx = (v: number) => v / 100 * cv.width, toPy = (v: number) => (1 - v / 100) * cv.height;
  function draw(): void {
    structBtns.forEach(({ s, b }) => b.classList.toggle("active", s === st.struct));
    layBtns.forEach(({ l, b }) => b.classList.toggle("active", l === st.layout));
    cS.input.closest("label")!.style.opacity = st.struct === "grid" ? "1" : "0.4";
    lvS.input.closest("label")!.style.opacity = st.struct === "median" || st.struct === "sah" ? "1" : "0.4";
    cur.set(st.n);
    const tree = st.struct === "median" ? bvhCache!.median : bvhCache!.sah;
    const dpr = window.devicePixelRatio || 1, cw = cv.clientWidth || 440;
    cv.width = Math.round(cw * dpr); cv.height = Math.round(cw * dpr);
    const g = cv.getContext("2d")!, lw = dpr;
    g.fillStyle = "#fcfcfd"; g.fillRect(0, 0, cv.width, cv.height);
    // the ray and its trace
    const o = st.o, L = Math.hypot(st.aim[0] - o[0], st.aim[1] - o[1]) || 1, d = [(st.aim[0] - o[0]) / L, (st.aim[1] - o[1]) / L];
    const hit = (p: number, tm: number) => primHit(prims[p], o, d, tm);
    let res: B.TraceResult & { cells?: number };
    if (st.struct === "none") res = B.bruteForce(prims.length, hit);
    else if (st.struct === "grid") res = B.traverseGrid(grid!, o, d, hit, Infinity, true);
    else res = B.traverseBVH(tree, o, d, hit, Infinity, true);
    const ev = st.struct === "none" ? prims.map((_, i) => ({ kind: "prim" as const, id: i, t: hit(i, Infinity) })) : res.trace;
    const k = st.step < 0 ? ev.length : Math.min(st.step, ev.length);
    const shown = ev.slice(0, k);
    // structure
    if (st.struct === "grid") {
      const c = st.cells;
      g.strokeStyle = "rgba(0,0,0,0.18)"; g.lineWidth = lw;
      for (let i = 0; i <= c; i++) { g.beginPath(); g.moveTo(toPx(100 * i / c), 0); g.lineTo(toPx(100 * i / c), cv.height); g.stroke(); g.beginPath(); g.moveTo(0, toPy(100 * i / c)); g.lineTo(cv.width, toPy(100 * i / c)); g.stroke(); }
      shown.forEach((e, idx) => {
        if (e.kind !== "cell") return;
        const x0 = 100 * e.cell[0] / c, y0 = 100 * e.cell[1] / c;
        g.fillStyle = idx === k - 1 ? "rgba(255,200,0,0.45)" : "rgba(255,200,0,0.2)";
        g.fillRect(toPx(x0), toPy(y0 + 100 / c), toPx(100 / c), toPx(100 / c));
      });
    } else if (st.struct !== "none") {
      tree.nodes.forEach(nd => {
        if (nd.depth !== st.level && !(nd.left < 0 && nd.depth < st.level)) return;
        g.strokeStyle = "rgba(31,119,180,0.55)"; g.lineWidth = 1.5 * lw;
        g.strokeRect(toPx(nd.min[0]), toPy(nd.max[1]), toPx(nd.max[0] - nd.min[0]), toPx(nd.max[1] - nd.min[1]));
      });
      shown.forEach((e, idx) => {
        if (e.kind !== "node") return;
        const nd = tree.nodes[e.id];
        g.strokeStyle = e.hit ? (idx === k - 1 ? "#e08a00" : "rgba(224,138,0,0.8)") : "rgba(214,39,40,0.8)";
        g.lineWidth = (idx === k - 1 ? 3.5 : 2.2) * lw; g.setLineDash(e.hit ? [] : [5 * lw, 4 * lw]);
        g.strokeRect(toPx(nd.min[0]), toPy(nd.max[1]), toPx(nd.max[0] - nd.min[0]), toPx(nd.max[1] - nd.min[1]));
        g.setLineDash([]);
      });
    }
    // primitives
    const tested = new Set<number>();
    shown.forEach(e => { if (e.kind === "prim") tested.add(e.id); });
    prims.forEach((p, i) => {
      const t = tested.has(i);
      g.strokeStyle = t ? "#000" : "#5b6b7b"; g.fillStyle = t ? "#9ecae1" : "#c9d3dd"; g.lineWidth = (t ? 2.2 : 1) * lw;
      if (p.kind === 0) { g.beginPath(); g.arc(toPx(p.a[0]), toPy(p.a[1]), Math.max(1, toPx(p.a[2])), 0, 7); g.fill(); g.stroke(); }
      else { g.lineWidth = (t ? 3.5 : 2) * lw; g.beginPath(); g.moveTo(toPx(p.a[0]), toPy(p.a[1])); g.lineTo(toPx(p.a[2]), toPy(p.a[3])); g.stroke(); }
    });
    // the ray
    const done = st.step < 0 || k >= ev.length;
    const tEnd = done && res.prim >= 0 ? res.t : 200;
    g.strokeStyle = "#d49b00"; g.lineWidth = 3 * lw;
    g.beginPath(); g.moveTo(toPx(o[0]), toPy(o[1])); g.lineTo(toPx(o[0] + tEnd * d[0]), toPy(o[1] + tEnd * d[1])); g.stroke();
    if (done && res.prim >= 0) { g.fillStyle = "#d62728"; g.beginPath(); g.arc(toPx(o[0] + res.t * d[0]), toPy(o[1] + res.t * d[1]), 7 * lw, 0, 7); g.fill(); }
    // handles
    const hd = (x: number, y: number, txt: string, diamond: boolean) => {
      g.fillStyle = diamond ? "#ffcc00" : "#fff"; g.strokeStyle = "#000"; g.lineWidth = 1.5 * lw;
      g.beginPath();
      if (diamond) { const s = 9 * lw; g.moveTo(toPx(x), toPy(y) - s); g.lineTo(toPx(x) + s, toPy(y)); g.lineTo(toPx(x), toPy(y) + s); g.lineTo(toPx(x) - s, toPy(y)); g.closePath(); }
      else g.arc(toPx(x), toPy(y), 12 * lw, 0, 7);
      g.fill(); g.stroke();
      if (txt) { g.fillStyle = "#000"; g.font = `bold ${16 * lw}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(txt, toPx(x), toPy(y) + lw); }
    };
    hd(o[0], o[1], "O", false); hd(st.aim[0], st.aim[1], "", true);
    // readout
    let boxT = 0, primT = 0, cells = 0;
    shown.forEach(e => { if (e.kind === "node") boxT++; else if (e.kind === "prim") primT++; else cells++; });
    const lines = [
      st.struct === "grid" ? `cells visited ${cells}, object tests ${primT}` : st.struct === "none" ? `object tests ${primT}` : `box tests ${boxT}, object tests ${primT}`,
      `total ${boxT + primT + cells} vs brute force ${prims.length}${done ? (res.prim >= 0 ? `; hit at t = ${res.t.toFixed(1)}` : "; no hit") : ` (step ${k} of ${ev.length})`}`,
    ];
    if (st.struct === "median" || st.struct === "sah") { const s = B.bvhStats(tree); lines.push(`tree depth ${s.depth}, ${s.leaves} leaves, SAH cost ${B.bvhCost(tree).toFixed(1)}`); }
    ro.textContent = lines.join("\n");
  }

  // dragging O and the aim
  let drag: "o" | "aim" | null = null;
  const toW = (ev: PointerEvent): [number, number] => { const r = cv.getBoundingClientRect(); return [Math.max(0, Math.min(100, (ev.clientX - r.left) / r.width * 100)), Math.max(0, Math.min(100, (1 - (ev.clientY - r.top) / r.height) * 100))]; };
  cv.addEventListener("pointerdown", ev => {
    const p = toW(ev), dO = Math.hypot(p[0] - st.o[0], p[1] - st.o[1]), dA = Math.hypot(p[0] - st.aim[0], p[1] - st.aim[1]);
    drag = dO < dA ? "o" : "aim";
    cv.setPointerCapture(ev.pointerId); ev.preventDefault();
    st[drag] = [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]; st.step = -1; draw();
  });
  cv.addEventListener("pointermove", ev => { if (!drag) return; const p = toW(ev); st[drag] = [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]; draw(); });
  const up = () => { drag = null; };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);

  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
