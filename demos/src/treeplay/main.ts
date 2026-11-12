// Decision-tree playground. Left: points of 2–3 classes in a plane (two features, the unit
// square); the tree's partition is painted behind them and its split lines drawn on top.
// Right: the tree diagram (click a node to select it) and, for the selected node, the
// split scan: impurity decrease against the threshold for a split on x and on y (click a
// curve to force that split), or the training and test error against the maximum depth.
// Buttons grow the tree one split at a time (best node first) or fully; sliders set the
// maximum depth and the minimum leaf size; criterion Gini / entropy / misclassification.
// data-mode="regression": a noisy intensity profile across a cell edge, fitted by a
// regression tree (steps). Trees come from lib/forest.ts.
// Auto-mounts on <div data-demo="treeplay">. Options: data-preset="blobs|diagonal|xor|island|overlap|three|regression",
// data-view="scan|depth".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, Slider } from "../lib/ui";
import { pointPlayground, Playground } from "../lib/playground";
import { PRESETS, Preset, edgeProfile } from "./data";
import { Plot, seriesColor } from "../lib/plot";
import { QUALITATIVE } from "../lib/colour";
import * as FO from "../lib/forest";

const NAME = "treeplay";
const DEPTH_MAX = 12;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .tp-left { flex: 0 1 400px; min-width: 240px; }
[data-demo="${NAME}"] .tp-left .qb-plot { max-width: none; }
[data-demo="${NAME}"] .tp-mid { flex: 1 1 360px; min-width: 260px; max-width: 470px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .tp-tree { width: 100%; display: block; }
[data-demo="${NAME}"] .tp-tree text { font-size: 16px; }
[data-demo="${NAME}"] .tp-tree .tp-node { cursor: pointer; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; gap: .4em; }
[data-demo="${NAME}"] .tp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-line; }
[data-demo="${NAME}"] .tp-cap { font-size: 15px; color: var(--qb-muted); text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .tp-left { flex-basis: 380px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const CRITS: { id: FO.Criterion; label: string }[] = [
  { id: "gini", label: "Gini" }, { id: "entropy", label: "entropy" }, { id: "misclass", label: "misclass." },
];

const pct = (v: number) => (100 * v).toFixed(1) + " %";

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    mode: p0.mode, splits: p0.splits, depth: p0.depth, minLeaf: 1, crit: "gini" as FO.Criterion,
    forced: {} as Record<string, { feature: number; threshold: number }>,
    selected: p0.select ?? "", view: (host.dataset.view as "scan" | "depth") ?? p0.view, showTest: p0.showTest,
    test: p0.test(), preset: p0,
  };
  let tree: FO.Tree | null = null;
  const prof = edgeProfile(21), profTest = edgeProfile(22);

  // ---- layout
  const left = htmlEl("div", "tp-left", "", host);
  const mid = htmlEl("div", "tp-mid", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "tp-lab", "presets", panel);
  presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const classBox = htmlEl("div", "", "", panel);
  const classLab = htmlEl("div", "tp-lab", "add points of class", classBox);
  const pg: Playground = pointPlayground(left, {
    data: st.mode === "class" ? p0.data() : { points: [], k: 2 }, k: 3, resolution: 96, controlsParent: classBox,
    names: ["class 1", "class 2", "class 3"], r: 7,
    onChange: () => { rebuild(); },
  });
  const regBox = htmlEl("div", "", "", left);
  const regPlot = new Plot(regBox, { width: 480, height: 440, x: [0, 1], y: [0, 220], xlabel: "position across the edge", ylabel: "intensity", ariaLabel: "intensity profile with a regression tree fit" });
  const regPts = regPlot.scatter(prof.x, prof.y, { r: 5, color: seriesColor(0), label: "training" });
  const regTestPts = regPlot.scatter(profTest.x, profTest.y, { r: 4, color: "var(--qb-muted)", opacity: 0.6, label: "test" });
  const regFit = regPlot.line([0], [0], { color: seriesColor(1), width: 3.5, label: "tree" });
  const cap = htmlEl("div", "tp-cap", "", left);

  htmlEl("div", "tp-lab", "grow", panel);
  const growRow = htmlEl("div", "qb-presets", "", panel);
  button(growRow, "next split", () => { if (!Number.isFinite(st.splits)) st.splits = tree ? tree.splitOrder.length : 0; st.splits++; st.selected = ""; rebuild(true); });
  button(growRow, "grow fully", () => { st.splits = Infinity; rebuild(); });
  button(growRow, "reset", () => { st.splits = 0; st.forced = {}; st.selected = ""; rebuild(); });
  const depthS: Slider = slider(panel, "max depth", 1, DEPTH_MAX, 1, st.depth, v => { st.depth = v; rebuild(); });
  const leafS: Slider = slider(panel, "min leaf size", 1, 20, 1, st.minLeaf, v => { st.minLeaf = v; rebuild(); });
  htmlEl("div", "tp-lab", "impurity", panel);
  const critBtns = presets(panel, CRITS.map(c => ({ label: c.label, apply: () => { st.crit = c.id; rebuild(); } })));
  const testCb = checkbox(panel, "show test points", st.showTest, v => { st.showTest = v; rebuild(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click a node to see its split scan; click a curve to force that split.", panel);

  // tree diagram
  const TW = 460, TH = 250;
  const treeSvg = svgEl("svg", { viewBox: `0 0 ${TW} ${TH}`, class: "tp-tree", role: "img", "aria-label": "tree diagram" }, mid);
  const viewRow = htmlEl("div", "qb-presets", "", mid);
  const viewBtns = presets(viewRow, [
    { label: "split scan", apply: () => { st.view = "scan"; drawSide(); } },
    { label: "error vs depth", apply: () => { st.view = "depth"; drawSide(); } },
  ]);
  const sideBox = htmlEl("div", "", "", mid);
  let side: Plot | null = null;

  // ---- model
  function trainTable(): { B: FO.Binned; target: FO.Target; T: FO.Table } {
    if (st.mode === "reg") {
      const T = FO.tableFromColumns([prof.x]);
      return { B: FO.binTable(T, 4096), target: FO.regTarget(prof.y), T };
    }
    const pts = pg.points;
    const T = FO.tableFromRows(pts.map(p => [p.x, p.y]));
    return { B: FO.binTable(T, 4096), target: FO.classTarget(pts.map(p => p.c), Math.max(2, ...pts.map(p => p.c + 1))), T };
  }

  function grow(maxDepth: number, maxSplits: number): FO.Tree | null {
    const { B, target } = trainTable();
    if (B.n === 0) return null;
    return FO.growTree(B, target, null, { criterion: st.mode === "reg" ? "sse" : st.crit, maxDepth, minLeaf: st.minLeaf, maxSplits, forced: st.forced });
  }

  function errorOn(t: FO.Tree, pts: { x: number; y: number; c: number }[]): number {
    if (!pts.length) return NaN;
    let e = 0;
    for (const p of pts) {
      const v = FO.predictPoint(t, [p.x, p.y]);
      let best = 0; for (let c = 1; c < v.length; c++) if (v[c] > v[best]) best = c;
      if (best !== p.c) e++;
    }
    return e / pts.length;
  }
  const mseOn = (t: FO.Tree, xs: number[], ys: number[]) => xs.reduce((a, x, i) => a + (FO.predictPoint(t, [x])[0] - ys[i]) ** 2, 0) / xs.length;

  function rebuild(selectLast = false): void {
    tree = grow(st.depth, st.splits);
    if (tree && selectLast && tree.splitOrder.length) st.selected = tree.path[tree.splitOrder[tree.splitOrder.length - 1]];
    if (tree && !tree.path.includes(st.selected)) st.selected = "";
    drawLeft(); drawTree(); drawSide(); readout();
  }

  // ---- left: plane or profile
  function regionOf(t: FO.Tree): Map<number, [number, number, number, number]> {
    const m = new Map<number, [number, number, number, number]>();
    const go = (q: number, b: [number, number, number, number]) => {
      m.set(q, b);
      if (t.feature[q] < 0) return;
      const th = t.threshold[q];
      if (t.feature[q] === 0) { go(t.left[q], [b[0], Math.min(b[1], th), b[2], b[3]]); go(t.right[q], [Math.max(b[0], th), b[1], b[2], b[3]]); }
      else { go(t.left[q], [b[0], b[1], b[2], Math.min(b[3], th)]); go(t.right[q], [b[0], b[1], Math.max(b[2], th), b[3]]); }
    };
    go(0, [0, 1, 0, 1]);
    return m;
  }

  const testG = svgEl("g", {}, pg.overlay);
  const linesG = svgEl("g", {}, pg.overlay);
  function drawLeft(): void {
    const reg = st.mode === "reg";
    pg.svg.parentElement!.style.display = reg ? "none" : "";
    regBox.style.display = reg ? "" : "none";
    classBox.style.display = reg ? "none" : "";
    testCb.parentElement!.style.display = "";
    if (reg) {
      const xs = Array.from({ length: 401 }, (_, i) => i / 400);
      regFit.set(xs, tree ? xs.map(x => FO.predictPoint(tree!, [x])[0]) : xs.map(() => 0));
      regTestPts.g.style.display = st.showTest ? "" : "none";
      regPts.render();
      cap.textContent = "a regression tree: each leaf predicts the mean of its points";
      regPlot.legend(undefined, "tl");
      return;
    }
    const t = tree;
    pg.setBackground(t ? (x, y) => FO.predictPoint(t, [x, y]) : null);
    linesG.replaceChildren(); testG.replaceChildren();
    if (st.showTest) for (const p of st.test.points) {
      const c = svgEl("circle", { cx: pg.sx(p.x).toFixed(1), cy: pg.sy(p.y).toFixed(1), r: 3 }, testG);
      c.style.setProperty("fill", "none"); c.style.setProperty("stroke", pg.color(p.c)); c.style.setProperty("stroke-width", "1.3"); c.style.setProperty("opacity", "0.7");
    }
    if (!t) return;
    const reg2 = regionOf(t), last = t.splitOrder[t.splitOrder.length - 1];
    for (let q = 0; q < t.nNodes; q++) {
      const b = reg2.get(q)!;
      if (t.path[q] === st.selected) {
        const r = svgEl("rect", { x: pg.sx(b[0]), y: pg.sy(b[3]), width: pg.sx(b[1]) - pg.sx(b[0]), height: pg.sy(b[2]) - pg.sy(b[3]) }, linesG);
        r.style.setProperty("fill", "none"); r.style.setProperty("stroke", "var(--qb-accent)"); r.style.setProperty("stroke-width", "4"); r.style.setProperty("stroke-dasharray", "8 5");
      }
      if (t.feature[q] < 0) continue;
      const th = t.threshold[q];
      const ln = t.feature[q] === 0
        ? svgEl("line", { x1: pg.sx(th), x2: pg.sx(th), y1: pg.sy(b[2]), y2: pg.sy(b[3]) }, linesG)
        : svgEl("line", { x1: pg.sx(b[0]), x2: pg.sx(b[1]), y1: pg.sy(th), y2: pg.sy(th) }, linesG);
      ln.style.setProperty("stroke", "var(--qb-fg)");
      ln.style.setProperty("stroke-width", q === last ? "3.5" : "2");
    }
    cap.textContent = "click to add points · drag to move · the tree colours each rectangle";
  }

  // ---- tree diagram
  const fname = (j: number) => (st.mode === "reg" ? "x" : ["x", "y"][j]);
  function drawTree(): void {
    treeSvg.replaceChildren();
    const t = tree;
    if (!t) return;
    const { leaves, depth } = FO.treeShape(t);
    const compact = leaves > 8;
    const xs = new Float64Array(t.nNodes);
    let li = 0;
    const place = (q: number): number => {
      if (t.feature[q] < 0) { xs[q] = 40 + (TW - 80) * (leaves > 1 ? li++ / (leaves - 1) : 0.5); if (leaves === 1) li++; return xs[q]; }
      xs[q] = 0.5 * (place(t.left[q]) + place(t.right[q]));
      return xs[q];
    };
    place(0);
    const dy = depth > 0 ? Math.min(62, (TH - 44) / depth) : 0;
    const yOf = (q: number) => 22 + t.depth[q] * dy;
    const g = svgEl("g", {}, treeSvg);
    for (let q = 0; q < t.nNodes; q++) if (t.feature[q] >= 0) for (const c of [t.left[q], t.right[q]]) {
      const l = svgEl("line", { x1: xs[q], y1: yOf(q), x2: xs[c], y2: yOf(c) - (compact ? 0 : 13) }, g);
      l.style.setProperty("stroke", "var(--qb-axis)"); l.style.setProperty("stroke-width", "1.5");
    }
    for (let q = 0; q < t.nNodes; q++) {
      const ng = svgEl("g", { class: "tp-node" }, treeSvg);
      const sel = t.path[q] === st.selected;
      const leaf = t.feature[q] < 0;
      let fill = "var(--qb-bg)";
      if (leaf && st.mode === "class") {
        let best = 0; for (let c = 1; c < t.k; c++) if (t.value[q * t.k + c] > t.value[q * t.k + best]) best = c;
        fill = QUALITATIVE[best];
      }
      if (compact) {
        const c = svgEl("circle", { cx: xs[q], cy: yOf(q), r: leaf ? 6 : 5 }, ng);
        c.style.setProperty("fill", leaf ? fill : "var(--qb-panel)");
        c.style.setProperty("stroke", sel ? "var(--qb-accent)" : "var(--qb-axis)");
        c.style.setProperty("stroke-width", sel ? "4" : "1.2");
      } else {
        const text = leaf
          ? (st.mode === "reg" ? `mean ${t.value[q].toFixed(0)}` : Array.from({ length: t.k }, (_, c) => Math.round(t.value[q * t.k + c] * t.weight[q])).join(" | "))
          : `${fname(t.feature[q])} ≤ ${t.threshold[q].toFixed(2)}`;
        const wbox = Math.max(56, text.length * 8.6 + 12);
        const r = svgEl("rect", { x: xs[q] - wbox / 2, y: yOf(q) - 13, width: wbox, height: 26, rx: 6 }, ng);
        r.style.setProperty("fill", leaf && st.mode === "class" ? fill : "var(--qb-panel, #f4f4f4)");
        r.style.setProperty("fill-opacity", leaf && st.mode === "class" ? "0.3" : "1");
        r.style.setProperty("stroke", sel ? "var(--qb-accent)" : "var(--qb-axis)");
        r.style.setProperty("stroke-width", sel ? "4" : "1.2");
        const tx = svgEl("text", { x: xs[q], y: yOf(q) + 5.5, "text-anchor": "middle" }, ng);
        tx.textContent = text;
      }
      ng.addEventListener("pointerdown", ev => { ev.preventDefault(); st.selected = t.path[q]; drawLeft(); drawTree(); drawSide(); });
    }
  }

  // ---- right: split scan or error vs depth
  function samplesAt(t: FO.Tree, path: string): Int32Array {
    const { T } = trainTable(), out: number[] = [];
    for (let i = 0; i < T.n; i++) {
      let q = 0, ok = path === "";
      while (!ok && t.feature[q] >= 0) {
        q = T.X[t.feature[q] * T.n + i] <= t.threshold[q] ? t.left[q] : t.right[q];
        if (t.path[q] === path) ok = true;
        if (!path.startsWith(t.path[q])) break;
      }
      if (ok) out.push(i);
    }
    return Int32Array.from(out);
  }

  function drawSide(): void {
    viewBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.view === "scan")));
    sideBox.replaceChildren();
    side = null;
    const t = tree;
    if (!t) return;
    const reg = st.mode === "reg";
    if (st.view === "scan") {
      const q = t.path.indexOf(st.selected);
      const idx = samplesAt(t, st.selected);
      const { B, target } = trainTable();
      const scan: FO.SplitScan[] = [];
      const feats = reg ? [0] : [0, 1];
      FO.bestSplit(B, idx, 0, idx.length, target, null, feats, { criterion: reg ? "sse" : st.crit, minLeaf: st.minLeaf, scan });
      let ymax = 0;
      scan.forEach(s => s.gains.forEach(g => (ymax = Math.max(ymax, g))));
      ymax = ymax > 0 ? ymax * 1.15 : (reg ? 100 : 0.1);
      const p = new Plot(sideBox, { width: 460, height: 230, x: [0, 1], y: [0, ymax], xlabel: `threshold (node: ${st.selected === "" ? "root" : st.selected}, ${idx.length} points)`, ylabel: reg ? "SSE decrease" : "impurity decrease", margin: { l: 88 }, ariaLabel: "split scan" });
      side = p;
      scan.forEach(s => {
        const col = seriesColor(s.feature === 0 ? 0 : 2);
        p.line(s.thresholds, s.gains, { color: col, width: 2.5, label: `split on ${fname(s.feature)}` });
        let bi = -1; s.gains.forEach((g, i) => { if (bi < 0 || g > s.gains[bi]) bi = i; });
        if (bi >= 0) p.scatter([s.thresholds[bi]], [s.gains[bi]], { color: col, r: 6 });
      });
      if (q >= 0 && t.feature[q] >= 0) p.vline(t.threshold[q], { color: "var(--qb-fg)", dash: "6 4", text: "chosen" });
      if (scan.length) p.legend(undefined, "tl");
      p.svg.style.cursor = "pointer";
      p.svg.addEventListener("pointerdown", ev => {
        const [x, y] = p.toData(ev);
        let best: { f: number; t: number; d: number } | null = null;
        for (const s of scan) s.thresholds.forEach((th, i) => {
          const dd = Math.hypot((th - x) / 1, (s.gains[i] - y) / ymax);
          if (!best || dd < best.d) best = { f: s.feature, t: th, d: dd };
        });
        if (!best) return;
        const b = best as { f: number; t: number; d: number };
        const wasLeaf = q >= 0 && t.feature[q] < 0;
        st.forced[st.selected] = { feature: b.f, threshold: b.t };
        if (wasLeaf && Number.isFinite(st.splits)) st.splits++;
        if (st.depth <= st.selected.length) { st.depth = st.selected.length + 1; depthS.set(st.depth); }
        rebuild();
      });
    } else {
      const ds = Array.from({ length: DEPTH_MAX }, (_, i) => i + 1);
      const tr: number[] = [], te: number[] = [];
      for (const D of ds) {
        const tt = grow(D, Infinity)!;
        if (reg) { tr.push(mseOn(tt, prof.x, prof.y)); te.push(mseOn(tt, profTest.x, profTest.y)); }
        else { tr.push(errorOn(tt, pg.points)); te.push(errorOn(tt, st.test.points)); }
      }
      const ymax = Math.max(...tr, ...te.filter(Number.isFinite), reg ? 10 : 0.05) * 1.1;
      const p = new Plot(sideBox, { width: 460, height: 230, x: [1, DEPTH_MAX], y: [0, ymax], xlabel: "max depth (grown fully)", ylabel: reg ? "mean squared error" : "error", margin: { l: 70 }, yfmt: reg ? undefined : v => (100 * v).toFixed(0) + "%", ariaLabel: "error against depth" });
      side = p;
      p.line(ds, tr, { color: seriesColor(0), label: "training" });
      p.scatter(ds, tr, { color: seriesColor(0), r: 4 });
      if (st.showTest || reg) { p.line(ds, te, { color: seriesColor(1), dash: "7 4", label: "test" }); p.scatter(ds, te, { color: seriesColor(1), r: 4 }); }
      p.vline(st.depth, { color: "var(--qb-accent)", dash: "4 4" });
      p.legend(undefined, "bl");
    }
  }

  function readout(): void {
    const t = tree;
    if (!t) { ro.textContent = "no points"; return; }
    const { leaves, depth } = FO.treeShape(t);
    if (st.mode === "reg") {
      ro.textContent = `${t.splitOrder.length} splits · ${leaves} leaves · depth ${depth}\ntraining MSE ${mseOn(t, prof.x, prof.y).toFixed(1)}` + (st.showTest ? `\ntest MSE ${mseOn(t, profTest.x, profTest.y).toFixed(1)}` : "");
      return;
    }
    const n = pg.points.length, eTr = errorOn(t, pg.points);
    let s = `${t.splitOrder.length} splits · ${leaves} leaves · depth ${depth}\ntraining error ${Math.round(eTr * n)}/${n} = ${pct(eTr)}`;
    if (st.showTest) s += `\ntest error ${pct(errorOn(t, st.test.points))} (${st.test.points.length} pts)`;
    const q = t.path.indexOf(st.selected);
    if (q >= 0) s += `\nnode: ${t.count[q]} pts, impurity ${t.impurity[q].toFixed(3)}`;
    ro.textContent = s;
  }

  function load(p: Preset): void {
    st.preset = p; st.mode = p.mode; st.splits = p.splits; st.depth = p.depth; st.minLeaf = 1; st.crit = "gini";
    st.forced = {}; st.selected = p.select ?? ""; st.view = p.view; st.showTest = p.showTest; st.test = p.test();
    depthS.set(st.depth); leafS.set(1); testCb.checked = st.showTest;
    critBtns.forEach((b, i) => b.classList.toggle("active", i === 0));
    if (p.mode === "class") pg.setPoints(p.data()); else rebuild();
  }

  critBtns[0].classList.add("active");
  load(p0);
}

autoMount(NAME, mount);
