// arbor: measuring neurons and vessel networks (skeleton day, unit skel-neurovessel). A
// seeded synthetic neuron (soma + branching dendrites) or vessel network (tree with Murray
// radii, tortuosity and loops) from lib/trees.ts, with its ground-truth graph. Steps as tabs:
// image → vesselness (multi-scale Frangi of the edges day, lib/edges.ts) → mask (hysteresis
// threshold of the vesselness, area filter, small holes filled) → skeleton (pink: pruned) →
// graph. Overlays: radius (DT on the skeleton), tortuosity per edge, Sholl circles (radius
// slider, profile plot), ground truth (cyan centreline; red rings: missed branch points,
// orange squares: spurious ones). The table compares the measurements with the truth.
// The pipeline itself is in pipeline.ts (no DOM).
//
// <div data-demo="arbor" data-preset="neuron|vessel|noise|scale|junction|count|tortuous"></div>
import { autoMount, htmlEl, slider, presets, button, svgEl } from "../lib/ui";
import { inferno, viridis, toCss, diverging, scalarToRGBA } from "../lib/colour";
import { Plot, seriesColor } from "../lib/plot";
import { displayRange } from "../lib/imageops";
import { pathLength, chord, LengthEstimator } from "../lib/skeleton";
import { ImgView, injectStyle } from "../canny/imgview";
import { choice, CHOICE_CSS } from "../medialaxis/choice";
import {
  N, Generated, Segmented, generate, vesselness, segment, measure, truthMeasures, matchPoints, sholl, SHOLL_RADII,
  Tab, Overlay, State, Preset, NEURON, VESSEL, SEG_N, SEG_V, PRESETS,
} from "./pipeline";

const NAME = "arbor";
const STYLE = CHOICE_CSS + `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; align-items: flex-start; }
[data-demo="${NAME}"] .ab-main { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .ab-mid { flex: 0 1 300px; min-width: 230px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
[data-demo="${NAME}"] .ab-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
[data-demo="${NAME}"] .ab-tabs .qb-btn { padding: .12em .45em; }
[data-demo="${NAME}"] .ab-legend { font-size: 14px; color: var(--qb-muted); min-height: 1.2em; }
[data-demo="${NAME}"] .ab-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
[data-demo="${NAME}"] table.ab-table { border-collapse: collapse; font-size: 14px; width: 100%; font-variant-numeric: tabular-nums; }
[data-demo="${NAME}"] table.ab-table th, [data-demo="${NAME}"] table.ab-table td { padding: .08em .3em; text-align: right; border-bottom: 1px solid var(--qb-grid, #ddd); }
[data-demo="${NAME}"] table.ab-table td:first-child, [data-demo="${NAME}"] table.ab-table th:first-child { text-align: left; }
[data-demo="${NAME}"] details summary { cursor: pointer; font-size: 15px; color: var(--qb-muted); }
[data-demo="${NAME}"] details > div { display: flex; flex-direction: column; gap: .25em; margin-top: .25em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .ab-main { flex: 0 0 420px; }
.reveal [data-demo="${NAME}"] .ab-mid { flex: 0 0 300px; gap: .1em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 340px; gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .04em .35em; }
.reveal [data-demo="${NAME}"] table.ab-table { font-size: 13px; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const fromPreset = (p: Preset): State => ({
    gen: { ...NEURON, ...(p.s.gen ?? {}) }, seg: { ...SEG_N, ...(p.s.seg ?? {}) }, tab: p.s.tab ?? "graph", overlay: p.s.overlay ?? "none",
    est: p.s.est ?? "corrected", shollR: p.s.shollR ?? 50,
  });
  const st: State = fromPreset(p0);

  // ---- layout ----
  const main = htmlEl("div", "ab-main", "", host);
  const tabsDiv = htmlEl("div", "ab-tabs", "", main);
  const TABS: [Tab, string][] = [["image", "image"], ["vessel", "vesselness"], ["mask", "mask"], ["skeleton", "skeleton"], ["graph", "graph"]];
  const tabBtns = TABS.map(([t, l]) => button(tabsDiv, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  const legend = htmlEl("div", "ab-legend", "", main);
  htmlEl("div", "ev-credit", "synthetic image (lib/trees.ts), ground truth known", main);
  const mid = htmlEl("div", "ab-mid", "", host);
  const plot = new Plot(mid, { width: 300, height: 190, x: [0, SHOLL_RADII[SHOLL_RADII.length - 1]], y: [0, 10], xlabel: "Sholl radius r (px)", ylabel: "crossings N(r)", font: 14, margin: { l: 48, r: 8, t: 6, b: 40 } });
  const profTruth = plot.step([], [], { color: "#17becf", width: 2, dash: "5 4", label: "truth" }, "mid");
  const profMeas = plot.step([], [], { color: seriesColor(0), width: 2.5, label: "measured" }, "mid");
  const rMark = plot.vline(st.shollR, { color: "#d62728", width: 1.5 });
  plot.legend(undefined, "tr");
  const shollS = slider(mid, "Sholl r", 4, 120, 2, st.shollR, v => { st.shollR = v; rMark.set(v); if (st.overlay !== "sholl") { st.overlay = "sholl"; ovC.set("sholl"); } draw(); });
  const estC = choice<LengthEstimator>(mid, "length", [["count", "pixel steps"], ["weighted", "1 / √2"], ["corrected", "corrected"]], st.est, v => { st.est = v; table(); });
  const tbl = htmlEl("div", "", "", mid);
  const busy = htmlEl("div", "ab-busy", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "examples", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const ovC = choice<Overlay>(panel, "overlay", [["none", "none"], ["radius", "radius"], ["tortuosity", "tortuosity"], ["sholl", "Sholl"], ["truth", "truth"]], st.overlay, v => { st.overlay = v; draw(); });
  const sMinS = slider(panel, "σ min", 1, 6, 0.5, st.seg.sMin, v => { st.seg.sMin = v; if (st.seg.sMax < v) { st.seg.sMax = v; sMaxS.set(v); } runV(); });
  const sMaxS = slider(panel, "σ max", 1, 8, 0.5, st.seg.sMax, v => { st.seg.sMax = v; if (st.seg.sMin > v) { st.seg.sMin = v; sMinS.set(v); } runV(); });
  const tHS = slider(panel, "threshold high", 0.05, 0.95, 0.05, st.seg.tHigh, v => { st.seg.tHigh = v; if (st.seg.tLow > v) { st.seg.tLow = v; tLS.set(v); } runSeg(); });
  const tLS = slider(panel, "threshold low", 0.05, 0.95, 0.05, st.seg.tLow, v => { st.seg.tLow = v; if (st.seg.tHigh < v) { st.seg.tHigh = v; tHS.set(v); } runSeg(); });
  const kS = slider(panel, "prune k", 0, 30, 1, st.seg.k, v => { st.seg.k = v; runSeg(); }, v => `${v} px`);
  const det = htmlEl("details", "", "", panel);
  htmlEl("summary", "", "generator: kind, seed, noise, …", det);
  const dbox = htmlEl("div", "", "", det);
  const kindC = choice<"neuron" | "vessel">(dbox, "kind", [["neuron", "neuron"], ["vessel", "vessels"]], st.gen.kind, v => { st.gen = { ...(v === "neuron" ? NEURON : VESSEL), noise: st.gen.noise, blur: st.gen.blur }; st.seg = { ...(v === "neuron" ? SEG_N : SEG_V) }; sync(); runGen(); });
  const seedS = slider(dbox, "seed", 1, 30, 1, st.gen.seed, v => { st.gen.seed = v; runGen(); });
  const brS = slider(dbox, "branching", 0, 0.05, 0.002, st.gen.branching, v => { st.gen.branching = v; runGen(); });
  const toS = slider(dbox, "tortuosity", 0, 2, 0.1, st.gen.tortuosity, v => { st.gen.tortuosity = v; runGen(); });
  const noS = slider(dbox, "noise", 0, 30, 1, st.gen.noise, v => { st.gen.noise = v; runGen(); });
  const blS = slider(dbox, "blur", 0, 3, 0.25, st.gen.blur, v => { st.gen.blur = v; runGen(); });
  const spS = slider(dbox, "bright spots", 0, 20, 1, st.gen.spots, v => { st.gen.spots = v; runGen(); });
  button(panel, "⟲ reset", () => { Object.assign(st, fromPreset(p0)); pbtns.forEach((b, k) => b.classList.toggle("active", PRESETS[k] === p0)); sync(); runGen(); });

  function sync(): void {
    sMinS.set(st.seg.sMin); sMaxS.set(st.seg.sMax); tHS.set(st.seg.tHigh); tLS.set(st.seg.tLow); kS.set(st.seg.k);
    kindC.set(st.gen.kind); seedS.set(st.gen.seed); brS.set(st.gen.branching); toS.set(st.gen.tortuosity); noS.set(st.gen.noise); blS.set(st.gen.blur); spS.set(st.gen.spots);
    ovC.set(st.overlay); estC.set(st.est); shollS.set(st.shollR); rMark.set(st.shollR);
  }
  function load(p: Preset): void { Object.assign(st, fromPreset(p)); sync(); runGen(); }

  // ---- computation (debounced: the vesselness takes ~0.2 s) ----
  let gen: Generated | null = null, V: Float32Array = new Float32Array(N * N), bestSigma: Float32Array = new Float32Array(N * N), seg: Segmented | null = null;
  let timer = 0, level = 0;
  function run(): void {
    const L = level; level = 0;
    if (L >= 3 || !gen) gen = generate(st.gen);
    if (L >= 2) { const r = vesselness(gen.image, st.seg); V = r.V; bestSigma = r.bestSigma; }
    seg = segment(V, st.seg);
    busy.textContent = "";
    draw();
  }
  function schedule(l: number, now = false): void {
    level = Math.max(level, l);
    clearTimeout(timer);
    if (now) { run(); return; }
    busy.textContent = "computing …";
    timer = window.setTimeout(run, 60);
  }
  const runGen = () => schedule(3), runV = () => schedule(2), runSeg = () => schedule(1);

  // ---- drawing ----
  function draw(): void {
    if (!gen || !seg) return;
    tabBtns.forEach((b, k) => b.classList.toggle("active", TABS[k][0] === st.tab));
    const px = new Uint8ClampedArray(N * N * 4), img = gen.image;
    const { lo, hi } = displayRange(img, { mode: "auto" });
    const g01 = (i: number) => Math.max(0, Math.min(1, (img.data[i] - lo) / (hi - lo || 1)));
    if (st.tab === "image") px.set(scalarToRGBA(img.data, { lo, hi }));
    else if (st.tab === "vessel") px.set(scalarToRGBA(V, { map: inferno, lo: 0, hi: 1 }));
    else for (let i = 0; i < N * N; i++) {
      let c: number[];
      if (st.tab === "mask") c = seg.mask[i] ? [235, 235, 235] : [40 * g01(i) + 15, 40 * g01(i) + 15, 40 * g01(i) + 20];
      else if (st.tab === "skeleton") c = seg.skel[i] ? [255, 255, 255] : seg.skel0[i] ? [255, 120, 160] : seg.mask[i] ? [70, 75, 90] : [15, 15, 20];
      else { const v = 25 + 70 * g01(i); c = seg.skel[i] ? [255, 255, 255] : [v, v, v]; }
      px.set([c[0], c[1], c[2], 255], 4 * i);
    }
    if (st.overlay === "radius" && st.tab !== "image" && st.tab !== "vessel") {
      for (let i = 0; i < N * N; i++) if (seg.skel[i]) { const c = viridis(Math.min(1, seg.dt[i] / 7)); px.set([255 * c[0], 255 * c[1], 255 * c[2], 255], 4 * i); }
    }
    view.rgba(px, N, N);
    view.setCaption(({ image: "synthetic image", vessel: `vesselness (σ ${st.seg.sMin}–${st.seg.sMax})`, mask: "mask: hysteresis threshold of the vesselness", skeleton: "skeleton (pink: pruned spurs)", graph: "skeleton graph" } as Record<Tab, string>)[st.tab]);
    overlay();
    const sh = sholl(gen, seg);
    profTruth.set(SHOLL_RADII, sh.truth); profMeas.set(SHOLL_RADII, sh.measured);
    plot.setDomain(undefined, [0, Math.max(4, ...sh.truth, ...sh.measured) + 1]);
    table();
  }

  function overlay(): void {
    view.clear();
    if (!gen || !seg) return;
    const G = seg.graph, ov = st.overlay;
    let leg = "";
    const showGraph = st.tab === "graph";
    if (ov === "tortuosity" || showGraph) {
      for (const e of G.edges) {
        const c = chord(e.path, N), t = c >= 3 ? pathLength(e.path, N, "corrected") / c : 1;
        const col = ov === "tortuosity" ? toCss(diverging(Math.min(1, 0.5 + (t - 1) * 1.5))) : "#ffd400";
        view.add("polyline", { points: e.path.map(i => `${(i % N) + 0.5},${Math.floor(i / N) + 0.5}`).join(" "), fill: "none", stroke: col, "stroke-width": ov === "tortuosity" ? 3 : 2, opacity: 0.9 });
      }
      if (ov === "tortuosity") leg = "edge colour: tortuosity (white 1 = straight, red ≥ 1.33)";
    }
    if (showGraph) for (const n of G.nodes) {
      const col = n.kind === "end" ? "#1f77b4" : n.kind === "junction" ? "#d62728" : "#9467bd";
      view.add("circle", { cx: n.x + 0.5, cy: n.y + 0.5, r: n.kind === "junction" ? 3.2 : 2.5, fill: col, stroke: "#fff", "stroke-width": 1 });
    }
    if (ov === "radius") leg = "skeleton colour: radius DT (dark 0 → yellow ≥ 7 px)";
    if (ov === "sholl") {
      const [cx, cy] = gen.centre;
      for (const r of [st.shollR]) view.add("circle", { cx, cy, r, fill: "none", stroke: "#d62728", "stroke-width": 2, "stroke-dasharray": "5 4" });
      view.add("circle", { cx, cy, r: 2.5, fill: "#d62728" });
      leg = "red circle: Sholl radius r; plot: crossings at every r";
    }
    if (ov === "truth") {
      for (const e of gen.truth.edges) view.add("polyline", { points: e.line.map(([x, y]) => `${x},${y}`).join(" "), fill: "none", stroke: "#17becf", "stroke-width": 1.5, opacity: 0.9 });
      const m = measure(gen, seg), t = truthMeasures(gen), mt = matchPoints(t.branchXY, m.branchXY);
      for (const [x, y] of mt.missed) view.add("circle", { cx: x, cy: y, r: 6, fill: "none", stroke: "#ff3030", "stroke-width": 2.5 });
      for (const [x, y] of mt.spurious) view.add("rect", { x: x + 0.5 - 5, y: y + 0.5 - 5, width: 10, height: 10, fill: "none", stroke: "#ff9900", "stroke-width": 2.5 });
      leg = `cyan: true centreline · red ○ missed (${mt.missed.length}) · orange □ spurious (${mt.spurious.length})`;
    }
    legend.textContent = leg;
    void svgEl;
  }

  function table(): void {
    if (!gen || !seg) return;
    const m = measure(gen, seg), t = truthMeasures(gen), mt = matchPoints(t.branchXY, m.branchXY);
    let rs = 0, rn = 0;
    for (let i = 0; i < N * N; i++) if (seg.skel[i] && bestSigma[i] > 0) { rs += bestSigma[i]; rn++; }
    const f = (v: number, d = 0) => v.toFixed(d);
    const rows: [string, string, string][] = [
      [`total length (${st.est === "count" ? "steps" : st.est === "weighted" ? "1/√2" : "corr."})`, f(m.length[st.est]), f(t.length)],
      ["end points", f(m.ends), f(t.ends)],
      ["branch points", f(m.branchPoints), f(t.branchPoints)],
      ["  missed / spurious", `${mt.missed.length} / ${mt.spurious.length}`, ""],
      ["branch points / 10⁴ px²", f(m.density, 2), f(t.density, 2)],
      ["mean radius (DT)", f(m.radius, 2), f(t.radius, 2)],
      ["mean best σ", rn ? f(rs / rn, 2) : "–", ""],
      ["mean tortuosity", f(m.tortuosity, 3), f(t.tortuosity, 3)],
      ["loops E − V + C", f(m.loops), f(t.loops)],
    ];
    if (gen.somaR > 0) rows.push(["primary branches", f(m.primaries), f(t.primaries)]);
    const sh = sholl(gen, seg), k = SHOLL_RADII.indexOf(st.shollR);
    if (st.overlay === "sholl" && k >= 0) rows.push([`Sholl N(${st.shollR})`, String(sh.measured[k]), String(sh.truth[k])]);
    tbl.innerHTML = `<table class="ab-table"><tr><th></th><th>measured</th><th>truth</th></tr>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</table>`;
  }

  pbtns[PRESETS.indexOf(p0)]?.classList.add("active");
  schedule(3, true);
}

autoMount(NAME, mount);
