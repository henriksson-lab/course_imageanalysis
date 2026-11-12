// Neighbours and parents in a population of cells. A seeded synthetic population: nuclei
// (Poisson, clustered or regular), cell bodies grown 9 px from the nuclei, and spots.
// Neighbour mode: the neighbour graph for one of three definitions (adjacent cell bodies /
// nuclei within r / expand until adjacent), a click shows an object's row; side panel:
// histograms of the number of neighbours and of the nearest-neighbour distance (with the
// Poisson expectation), the Clark–Evans ratio, and a border rule. Parent mode: spots
// assigned to nuclei or cells by largest overlap or by centroid, with spots outside any
// parent and spots touching two parents highlighted.
// Auto-mounts on <div data-demo="neighbours">. Options: data-preset="<id>", data-mode.
import { htmlEl, slider, presets, checkbox, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { labelColor } from "../lib/colour";
import { expandLabels, poissonNNDensity, assignParents } from "../lib/neighbours";
import { W, H, Pattern, makePopulation, Population, Def, Border, GUARD, Analysis, analyse as analyseModel, summarise } from "./model";

const NAME = "neighbours";

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .nb-img { flex: 0 1 470px; min-width: 260px; margin: 0; }
[data-demo="${NAME}"] .nb-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: pointer; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .nb-side { flex: 1 1 260px; min-width: 220px; max-width: 330px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 250px; gap: .3em; }
[data-demo="${NAME}"] .nb-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .nb-row { font-size: 15px; white-space: pre-wrap; background: var(--qb-panel); border-radius: 4px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 15px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .nb-img { flex-basis: 430px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { mode: "neighbours" | "parents"; pattern: Pattern; n: number; def: Def; r: number; border: Border; parentOf: "nuclei" | "cells"; method: "overlap" | "centroid"; selected: number; graph: boolean }
const DEFAULT: State = { mode: "neighbours", pattern: "poisson", n: 40, def: "adjacent", r: 10, border: "all", parentOf: "nuclei", method: "overlap", selected: 0, graph: true };

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "adjacent", label: "adjacent vs expanded", s: { def: "adjacent" } },
  { id: "within", label: "within r: grows with r", s: { def: "within", r: 10 } },
  { id: "clustered", label: "clustered: R < 1", s: { pattern: "clustered", def: "expanded" } },
  { id: "regular", label: "regular: R > 1", s: { pattern: "regular", def: "expanded" } },
  { id: "border", label: "border objects have fewer neighbours", s: { def: "expanded", n: 60 } },
  { id: "spots", label: "spots per nucleus", s: { mode: "parents", parentOf: "nuclei" } },
  { id: "between", label: "a spot between two cells", s: { mode: "parents", parentOf: "cells", selected: -1 } },
];


function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 21 : 18;

  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(host.dataset.mode === "parents" ? { mode: "parents" as const } : {}), ...(p0?.s ?? {}) };

  const fig = htmlEl("figure", "nb-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  const S = 2;
  cv.width = W * S; cv.height = H * S;
  const side = htmlEl("div", "nb-side", "", host);
  const rowBox = htmlEl("div", "nb-row", "", side);
  const f1 = plotFigure(side, "", "qb-figure");
  const f2 = plotFigure(side, "", "qb-figure");
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "nb-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  htmlEl("div", "nb-lab", "view", panel);
  const modeB = presets(panel, [
    { label: "neighbours", apply: () => { st.mode = "neighbours"; st.selected = 0; unp(); draw(); } },
    { label: "parents and children", apply: () => { st.mode = "parents"; st.selected = 0; unp(); draw(); } },
  ]);
  htmlEl("div", "nb-lab", "placement of the nuclei", panel);
  const patB = presets(panel, (["poisson", "clustered", "regular"] as Pattern[]).map(p => ({ label: p === "poisson" ? "random (Poisson)" : p, apply: () => { st.pattern = p; unp(); rebuild(); } })));
  const nS = slider(panel, "number", 20, 80, 5, st.n, v => { st.n = v; unp(); rebuild(); });
  const nbCtl = htmlEl("div", "qb-panel", "", panel);
  nbCtl.style.flex = "none";
  htmlEl("div", "nb-lab", "neighbour definition", nbCtl);
  const defB = presets(nbCtl, [
    { label: "adjacent cells", apply: () => { st.def = "adjacent"; unp(); draw(); } },
    { label: "within r", apply: () => { st.def = "within"; unp(); draw(); } },
    { label: "expand until adjacent", apply: () => { st.def = "expanded"; unp(); draw(); } },
  ]);
  const rS = slider(nbCtl, "r (px)", 1, 40, 1, st.r, v => { st.r = v; if (st.def === "within") { unp(); draw(); } });
  htmlEl("div", "nb-lab", "objects at the image border", nbCtl);
  const borB = presets(nbCtl, [
    { label: "keep all", apply: () => { st.border = "all"; unp(); draw(); } },
    { label: "remove them", apply: () => { st.border = "remove"; unp(); draw(); } },
    { label: "guard zone", apply: () => { st.border = "guard"; unp(); draw(); } },
  ]);
  const grC = checkbox(nbCtl, "show the neighbour graph", st.graph, v => { st.graph = v; draw(); });
  const parCtl = htmlEl("div", "qb-panel", "", panel);
  parCtl.style.flex = "none";
  htmlEl("div", "nb-lab", "parents of the spots", parCtl);
  const parB = presets(parCtl, [
    { label: "nuclei", apply: () => { st.parentOf = "nuclei"; unp(); draw(); } },
    { label: "cells", apply: () => { st.parentOf = "cells"; unp(); draw(); } },
  ]);
  const metB = presets(parCtl, [
    { label: "largest overlap", apply: () => { st.method = "overlap"; unp(); draw(); } },
    { label: "label at centroid", apply: () => { st.method = "centroid"; unp(); draw(); } },
  ]);
  button(panel, "reset", () => { Object.assign(st, DEFAULT); sync(); unp(); rebuild(); });
  const note = htmlEl("div", "qb-note", "Click an object to see its row of the table.", panel);

  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    nS.set(st.n); rS.set(st.r); grC.checked = st.graph;
    modeB.forEach((b, i) => b.classList.toggle("active", ["neighbours", "parents"][i] === st.mode));
    patB.forEach((b, i) => b.classList.toggle("active", ["poisson", "clustered", "regular"][i] === st.pattern));
    defB.forEach((b, i) => b.classList.toggle("active", ["adjacent", "within", "expanded"][i] === st.def));
    borB.forEach((b, i) => b.classList.toggle("active", ["all", "remove", "guard"][i] === st.border));
    parB.forEach((b, i) => b.classList.toggle("active", ["nuclei", "cells"][i] === st.parentOf));
    metB.forEach((b, i) => b.classList.toggle("active", ["overlap", "centroid"][i] === st.method));
    nbCtl.style.display = st.mode === "neighbours" ? "" : "none";
    parCtl.style.display = st.mode === "parents" ? "" : "none";
  }

  let pop: Population = makePopulation({ pattern: st.pattern, n: st.n });
  let voronoi = expandLabels(pop.nuclei, W, H);
  function rebuild(): void {
    pop = makePopulation({ pattern: st.pattern, n: st.n });
    voronoi = expandLabels(pop.nuclei, W, H);
    if (st.selected < 0) st.selected = pop.borderSpot;
    draw();
  }

  const analyse = (def: Def): Analysis => analyseModel(pop, voronoi, def, st.r, st.border);

  function draw(): void {
    sync();
    const ctx = cv.getContext("2d")!;
    const id = ctx.createImageData(W, H);
    const nbA = st.mode === "neighbours" ? analyse(st.def) : null;
    let par: ReturnType<typeof assignParents> | null = null;
    const parentLab = st.parentOf === "nuclei" ? pop.nuclei : pop.cells;
    if (st.mode === "parents") par = assignParents(pop.spots, parentLab, W, H, pop.nSpots, pop.count, st.method);
    const regions = st.mode === "parents" ? pop.cells : nbA!.objects;
    for (let i = 0; i < W * H; i++) {
      let rgb: [number, number, number] = [250, 250, 250];
      const reg = regions[i];
      if (reg) {
        const c = labelColor(reg);
        const keep = st.mode === "parents" || nbA!.measured[reg];
        const k = keep ? 0.45 : 0.15;
        rgb = [250 * (1 - k) + 255 * c[0] * k, 250 * (1 - k) + 255 * c[1] * k, 250 * (1 - k) + 255 * c[2] * k];
      }
      if (pop.nuclei[i]) rgb = [70, 85, 120];
      const sp = pop.spots[i];
      if (sp) {
        if (!par) rgb = [255, 210, 0];
        else rgb = par.nParents[sp] > 1 ? [230, 0, 200] : par.parent[sp] ? [40, 190, 60] : [255, 120, 0];
      }
      id.data[4 * i] = rgb[0]; id.data[4 * i + 1] = rgb[1]; id.data[4 * i + 2] = rgb[2]; id.data[4 * i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = W; tmp.height = H;
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, W * S, H * S);
    // region outlines
    ctx.fillStyle = "rgba(60,60,60,0.55)";
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, l = regions[i];
      if (l && ((x + 1 < W && regions[i + 1] !== l) || (y + 1 < H && regions[i + W] !== l))) ctx.fillRect(x * S, y * S, S, S);
    }
    if (st.border === "guard" && st.mode === "neighbours") {
      ctx.strokeStyle = "#d62728"; ctx.setLineDash([8, 6]); ctx.lineWidth = 2;
      ctx.strokeRect(GUARD * S, GUARD * S, (W - 2 * GUARD) * S, (H - 2 * GUARD) * S);
      ctx.setLineDash([]);
    }
    const sel = st.selected;
    if (nbA) {
      const P = (k: number): [number, number] => [(pop.xs[k - 1] + 0.5) * S, (pop.ys[k - 1] + 0.5) * S];
      if (st.graph) {
        ctx.strokeStyle = "rgba(20,20,20,0.75)"; ctx.lineWidth = 2;
        ctx.beginPath();
        nbA.nb.forEach((list, a) => { for (const b of list) if (b > a) { const [x1, y1] = P(a), [x2, y2] = P(b); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); } });
        ctx.stroke();
      }
      if (sel > 0 && sel <= pop.count) {
        ctx.strokeStyle = "#d62728"; ctx.lineWidth = 4;
        ctx.beginPath();
        for (const b of nbA.nb[sel] ?? []) { const [x1, y1] = P(sel), [x2, y2] = P(b); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
        ctx.stroke();
        const [x, y] = P(sel);
        ctx.beginPath(); ctx.arc(x, y, 16, 0, 2 * Math.PI); ctx.stroke();
        if (st.def === "within") { ctx.setLineDash([5, 5]); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, (st.r + 6) * S, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); }
      }
    } else if (sel > 0) {
      let sx = 0, sy = 0, n = 0;
      for (let i = 0; i < W * H; i++) if (pop.spots[i] === sel) { sx += i % W; sy += Math.floor(i / W); n++; }
      if (n) { ctx.strokeStyle = "#d62728"; ctx.lineWidth = 4; ctx.beginPath(); ctx.arc((sx / n + 0.5) * S, (sy / n + 0.5) * S, 14, 0, 2 * Math.PI); ctx.stroke(); }
    }
    if (nbA) drawNeighbourStats(nbA); else drawParentStats(par!);
  }

  function drawNeighbourStats(a: Analysis): void {
    const c = pop.count, sm = summarise(pop, a, st.border), counts = sm.counts, N = sm.nn, nnd = sm.nnd;
    const defName = { adjacent: "adjacent cells (grown 9 px)", within: `nuclei within r = ${st.r} px`, expanded: "expand until adjacent" }[st.def];
    cap.textContent = `${defName}: ${counts.length} objects measured`;
    // histogram of neighbour counts
    f1.replaceChildren();
    const maxK = Math.max(8, ...counts);
    const hk = new Array(maxK + 1).fill(0);
    for (const k of counts) hk[k]++;
    const p1 = new Plot(f1, { width: 330, height: 200, x: [-0.6, maxK + 0.6], y: [0, Math.max(...hk) * 1.15 + 0.5], xlabel: "number of neighbours", font: FS, margin: { l: 2.4 * FS } });
    p1.bars(hk.map((_, k) => k), hk, { barWidth: 0.8, color: "var(--qb-c1)" });
    // NN distances
    f2.replaceChildren();
    const xmax = 60, bw = 4, nb = xmax / bw;
    const hd = new Array(nb).fill(0);
    for (const d of nnd) if (d < xmax) hd[Math.floor(d / bw)]++;
    const rho = sm.rho;
    const p2 = new Plot(f2, { width: 330, height: 200, x: [0, xmax], y: [0, Math.max(...hd, nnd.length * bw * poissonNNDensity(1 / Math.sqrt(2 * Math.PI * rho), rho)) * 1.15 + 0.5], xlabel: "nearest-neighbour distance (px)", font: FS, margin: { l: 2.4 * FS } });
    p2.histogram(Array.from({ length: nb + 1 }, (_, k) => k * bw), hd, { color: "var(--qb-c3)" });
    p2.fn(r => nnd.length * bw * poissonNNDensity(r, rho), { color: "var(--qb-c2)", width: 2.5, dash: "6 4", label: "random" });
    p2.legend(undefined, "tr");
    const all3 = (["adjacent", "within", "expanded"] as Def[]).map(d => summarise(pop, analyse(d), st.border).meanCount.toFixed(2));
    const f2d = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
    ro.textContent = `mean number of neighbours: ${sm.meanCount.toFixed(2)}\n  adjacent ${all3[0]} · within r ${all3[1]} · expanded ${all3[2]}\n`
      + `  interior objects ${f2d(sm.interior)} · at the border ${f2d(sm.atBorder)}\n`
      + `mean NN distance ${sm.observed.toFixed(1)} px; random: ${sm.expected.toFixed(1)} px\nClark–Evans R = ${sm.R.toFixed(2)}`;
    // selected row
    const s = st.selected;
    if (s > 0 && s <= c) {
      const k = s - 1;
      rowBox.textContent = `object ${s}${a.measured[s] ? "" : " (not measured)"}\nneighbours: ${a.nb[s].length}` + (a.percent ? `, touching ${a.percent[s].toFixed(0)} %` : "")
        + `\nclosest: ${N.first[k] + 1} at ${N.firstDist[k].toFixed(1)} px\nsecond: ${N.second[k] + 1} at ${N.secondDist[k].toFixed(1)} px\nangle between them: ${N.angle[k].toFixed(0)}°`;
    } else rowBox.textContent = "Click an object to see its row.";
  }

  function drawParentStats(par: ReturnType<typeof assignParents>): void {
    const ns = pop.nSpots, c = pop.count;
    let none = 0, amb = 0;
    for (let s = 1; s <= ns; s++) { if (!par.parent[s]) none++; if (par.nParents[s] > 1) amb++; }
    cap.textContent = `spots → ${st.parentOf} by ${st.method === "overlap" ? "largest overlap" : "label at centroid"}`;
    f1.replaceChildren();
    const cc = Array.from(par.childCount).slice(1);
    const maxK = Math.max(5, ...cc), hk = new Array(maxK + 1).fill(0);
    for (const k of cc) hk[k]++;
    const p1 = new Plot(f1, { width: 330, height: 220, x: [-0.6, maxK + 0.6], y: [0, Math.max(...hk) * 1.15], xlabel: `spots per ${st.parentOf === "nuclei" ? "nucleus" : "cell"}`, font: FS, margin: { l: 2.4 * FS } });
    p1.bars(hk.map((_, k) => k), hk, { barWidth: 0.8, color: "var(--qb-c3)" });
    f2.replaceChildren();
    const mean = cc.reduce((x, y) => x + y, 0) / c;
    ro.textContent = `${ns} spots: ${ns - none} with a parent, ${none} without (orange), ${amb} touching two parents (magenta)\nmean ${mean.toFixed(2)} spots per ${st.parentOf === "nuclei" ? "nucleus" : "cell"} (${c} parents)`;
    const s = st.selected;
    if (s > 0 && s <= ns) rowBox.textContent = `spot ${s}: parent ${par.parent[s] || "none"}, ${Math.round(100 * par.fraction[s])} % of its pixels inside it, overlaps ${par.nParents[s]} parent(s)`;
    else rowBox.textContent = "Click a spot to see its parent.";
  }

  cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * W), y = Math.floor((ev.clientY - r.top) / r.height * H);
    if (st.mode === "parents") {
      let best = 0, bd = 8;
      for (let yy = Math.max(0, y - 6); yy < Math.min(H, y + 7); yy++) for (let xx = Math.max(0, x - 6); xx < Math.min(W, x + 7); xx++) {
        const s = pop.spots[yy * W + xx], d = Math.hypot(xx - x, yy - y);
        if (s && d < bd) { bd = d; best = s; }
      }
      st.selected = best;
    } else {
      let best = 0, bd = Infinity;
      for (let k = 0; k < pop.count; k++) { const d = Math.hypot(pop.xs[k] - x, pop.ys[k] - y); if (d < bd) { bd = d; best = k + 1; } }
      st.selected = bd < 25 ? best : 0;
    }
    draw();
  });

  void note;
  sync();
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  rebuild();
}

autoMount(NAME, mount);
