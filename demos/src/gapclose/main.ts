// Gaps, divisions and lineage trees. A seeded simulation of dividing cells in a 400 × 70 field
// (gamma cycle times, daughters on both sides, area halved), with missed detections, false
// detections, segmentation splits (a fragment cut off for one frame) and cells entering at the
// left border. Stage 1 links frame to frame (global assignment); stage 2 joins segment ends
// and starts in one more assignment: gap closing (up to G missed frames), splitting (a
// division) and merging (lib/linking.ts linkSegments). Left: the space–time view (x across,
// time down), segments coloured, gap joins dashed, splits orange, merges purple, wrong links
// red. Middle: the lineage of one founder cell, true and found. Right: cycle-time histogram
// (complete cycles vs all, incl. censored ones). Readouts: link P/R/F1, division triplets,
// complete tracks, why tracks start and end.
//
// <div data-demo="gapclose" data-preset="blink|nogap|bigG|division|bubble|krule|border|censor"
//      data-panels="space-time,tree,cycles"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { labelColor, toCss } from "../lib/colour";
import { drawLineage, layoutX, subtree } from "../lib/lineage";
import { CellMovie } from "../lib/tracksim";
import { T, W, N0, State, DEFAULT, PRESETS, K_FRAMES, simulate, analyse, Analysis, rootOf, cycles } from "./model";

const NAME = "gapclose";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .gc-st { flex: 1 1 520px; min-width: 280px; max-width: 600px; }
${S} .gc-tree { flex: 0 1 280px; min-width: 200px; }
${S} .gc-cyc { flex: 1 1 340px; max-width: 420px; }
${S} .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .25em; }
${S} .gc-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 17px; white-space: pre-wrap; }
${S} .qb-presets { flex-basis: 100%; }
${S} .qb-btn.gc-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
${S} .gc-key { font-size: 15px; color: var(--qb-muted); }
.reveal ${S} { font-size: 18px; gap: .4em; }
.reveal ${S} .gc-st { flex: 0 0 440px; }
.reveal ${S} .gc-tree { flex: 0 0 240px; }
.reveal ${S} .qb-panel { flex: 0 0 410px; max-width: 420px; gap: .05em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .15em .4em; }
.reveal ${S} .gc-key { display: none; }
`;

const C_GAP = "#0072b2", C_SPLIT = "#e69f00", C_MERGE = "#cc79a7", C_BAD = "#d62728";

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  const panelsAttr = (host.dataset.panels ?? "space-time,tree,cycles").split(",").map(s => s.trim());
  let st: State = { ...DEFAULT };
  const stBox = htmlEl("div", "gc-st", "", host);
  const stPlot = new Plot(stBox, { width: 560, height: 430, x: [0, W], y: [T - 0.5, -0.5], xlabel: "x position (px)", ylabel: "frame", font: 26, xticks: [0, 100, 200, 300, 400] });
  const treeBox = htmlEl("div", "gc-tree", "", host);
  const trPlot = new Plot(treeBox, { width: 280, height: 430, x: [0, 1], y: [T - 0.5, -0.5], ylabel: "frame", font: 24, xticks: [], grid: false, margin: { t: 34, b: 12 } });
  const panel = htmlEl("div", "qb-panel", "", host);
  const cycBox = htmlEl("div", "gc-cyc", "", host);
  const cyPlot = new Plot(cycBox, { width: 420, height: 250, x: [0, 60], y: [0, 10], xlabel: "cycle time (frames)", ylabel: "cells", font: 22 });
  if (!panelsAttr.includes("space-time")) stBox.style.display = "none";
  if (!panelsAttr.includes("tree")) treeBox.style.display = "none";
  if (!panelsAttr.includes("cycles")) cycBox.style.display = "none";

  const read = htmlEl("div", "qb-readout", "", panel);
  const sl = {
    miss: slider(panel, "miss probability", 0, 0.3, 0.01, st.miss, v => { st.miss = v; recompute(true); }, v => v.toFixed(2)),
    falseRate: slider(panel, "false det./frame", 0, 3, 0.1, st.falseRate, v => { st.falseRate = v; recompute(true); }, v => v.toFixed(1)),
    splitProb: slider(panel, "segm. split prob.", 0, 0.1, 0.005, st.splitProb, v => { st.splitProb = v; recompute(true); }, v => v.toFixed(3)),
    enterRate: slider(panel, "entering/frame", 0, 0.2, 0.01, st.enterRate, v => { st.enterRate = v; recompute(true); }, v => v.toFixed(2)),
    maxGap: slider(panel, "max gap G (frames)", 0, 10, 1, st.maxGap, v => { st.maxGap = v; recompute(); }),
    gate: slider(panel, "join gate (px)", 2, 20, 0.5, st.gate, v => { st.gate = v; recompute(); }, v => v.toFixed(1)),
    bd: slider(panel, "b as distance (px)", 2, 20, 0.5, st.bd, v => { st.bd = v; recompute(); }, v => v.toFixed(1)),
  };
  const chk = htmlEl("div", "gc-row", "", panel);
  const cS = checkbox(chk, "splitting", st.splitting, v => { st.splitting = v; recompute(); });
  const cM = checkbox(chk, "merging", st.merging, v => { st.merging = v; recompute(); });
  const cK = checkbox(chk, `k-frame rule (k = ${K_FRAMES})`, st.kRule, v => { st.kRule = v; recompute(); });
  const cC = checkbox(chk, "candidates", st.candidates, v => { st.candidates = v; draw(); });
  const fRow = htmlEl("div", "gc-row", "lineage of cell:", panel);
  const fBtns = Array.from({ length: N0 }, (_, i) => button(fRow, String(i + 1), () => { st.founder = i + 1; draw(); }));
  htmlEl("div", "gc-key", "lines: segments (colour); dashed blue: gap closed; orange: split (division); purple: merge; red: wrong link; dotted grey: candidates", panel);
  presets(host, [...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s) })), { label: "reset", apply: () => apply({}) }]);

  let mv: CellMovie = simulate(st), an: Analysis = analyse(mv, st);

  function apply(p: Partial<State>): void {
    st = { ...DEFAULT, ...p };
    if (st.founder > N0) st.founder = 1;
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].set(st[k]);
    cS.checked = st.splitting; cM.checked = st.merging; cK.checked = st.kRule; cC.checked = st.candidates;
    recompute(true);
  }
  function recompute(resim = false): void {
    if (resim) mv = simulate(st);
    an = analyse(mv, st);
    draw();
  }

  function draw(): void {
    fBtns.forEach((b, i) => b.classList.toggle("gc-on", st.founder === i + 1));
    const root = rootOf(mv, an.chainOf, an.branches, st.founder - 1);
    const sel = new Set(root >= 0 ? subtree(an.branches, root).map(b => b.id) : []);
    // space–time view
    stPlot.clear();
    stPlot.mark(g => {
      const X = stPlot.x, Y = stPlot.y;
      const line = (a: number, b: number, color: string, w: number, dash?: string, op = 1) => {
        const l = svgEl("line", { x1: X(mv.dets[a].x), y1: Y(mv.dets[a].t), x2: X(mv.dets[b].x), y2: Y(mv.dets[b].t) }, g);
        l.style.stroke = color; l.style.strokeWidth = String(w); l.style.opacity = String(op);
        if (dash) l.style.strokeDasharray = dash;
      };
      if (st.candidates) for (const c of an.seg.candidates) line(c.a, c.b, "#888", 1.2, "2 3", 0.8);
      for (const c of an.seg.removed) line(c.a, c.b, "#888", 2, "2 3");
      an.edges.forEach((e, i) => {
        const inSel = sel.has(an.chainOf[e.b]) || sel.has(an.chainOf[e.a]);
        const bad = an.wrong[i];
        const base = e.kind === "split" ? C_SPLIT : e.kind === "merge" ? C_MERGE : e.kind === "gap" ? C_GAP : toCss(labelColor(1 + (an.chainOf[e.b] % 97)));
        line(e.a, e.b, bad ? C_BAD : base, inSel ? 3.4 : bad || e.kind !== "link" ? 2.6 : 1.8, e.kind === "gap" ? "5 3" : undefined, inSel || sel.size === 0 ? 1 : 0.55);
      });
      mv.dets.forEach((d, k) => {
        const on = sel.has(an.chainOf[k]);
        const c = svgEl("circle", { cx: X(d.x), cy: Y(d.t), r: on ? 2.2 : 1.3 }, g);
        c.style.fill = "#444"; c.style.opacity = on ? "0.9" : "0.45";
      });
    }, []);
    // lineage trees: true (left) and found (right)
    trPlot.clear();
    const troot = rootOf(mv, an.truthChainOf, an.truthBranches, st.founder - 1);
    const lx = troot >= 0 ? layoutX(an.truthBranches, troot) : new Map<number, number>();
    const rx = root >= 0 ? layoutX(an.branches, root) : new Map<number, number>();
    const L1 = Math.max(1, ...[...lx.values()].map(v => v + 1)), L2 = Math.max(1, ...[...rx.values()].map(v => v + 1));
    trPlot.setDomain([-0.8, L1 + L2 + 0.3], undefined);
    trPlot.mark(g => {
      if (troot >= 0) drawLineage(g, an.truthBranches, troot, x => trPlot.x(x), t => trPlot.y(t - 0.5), { color: "#555", width: 3 });
      if (root >= 0) drawLineage(g, an.branches, root, x => trPlot.x(x + L1 + 0.5), t => trPlot.y(t - 0.5), { color: "var(--qb-c1, #1f77b4)", wrong: C_BAD, width: 3 });
    }, []);
    trPlot.mark(g => {
      for (const [x, s] of [[(L1 - 1) / 2, "true"], [L1 + 0.5 + (L2 - 1) / 2, "found"]] as [number, string][]) {
        const t = svgEl("text", { x: trPlot.x(x), y: 24, "text-anchor": "middle" }, g);
        t.textContent = s; t.style.fontSize = "24px"; t.style.fontWeight = "600";
      }
    }, [], {}, "line", trPlot.over);
    // cycle times
    const cy = cycles(an);
    const edges = Array.from({ length: 16 }, (_, i) => i * 4);
    const hist = (v: number[]) => { const c = new Array(15).fill(0); for (const x of v) c[Math.min(14, Math.floor(x / 4))]++; return c; };
    const ha = hist(cy.all), hc = hist(cy.complete);
    cyPlot.clear();
    cyPlot.setDomain([0, 60], [0, Math.max(4, ...ha) + 1]);
    cyPlot.histogram(edges, ha, { color: "#999", label: "all (incl. censored)" });
    cyPlot.histogram(edges, hc, { color: "var(--qb-c1, #1f77b4)", label: "complete cycles" });
    const mean = (v: number[]) => (v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN);
    cyPlot.vline(st.cycleMean, { color: "#222", dash: "6 4", width: 2, label: "model mean" });
    cyPlot.legend(undefined, "tr");
    // readout
    const fmtR = (o: Record<string, number>) => Object.entries(o).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(", ");
    read.textContent = [
      `links  P ${an.links.precision.toFixed(3)}  R ${an.links.recall.toFixed(3)}  F1 ${an.links.f1.toFixed(3)}`,
      `divisions  TP ${an.div.tp}  FP ${an.div.fp}  FN ${an.div.fn}`,
      `perfect tracks  ${an.complete.complete} / ${an.complete.total}`,
      `starts: ${fmtR(an.reasons.start)}`,
      `ends: ${fmtR(an.reasons.end)}`,
      `cycles: complete ${cy.complete.length}, mean ${cy.complete.length ? mean(cy.complete).toFixed(1) : "–"}; model mean ${st.cycleMean}`,
    ].join("\n");
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply(pre ? pre.s : {});
});
