// Evaluating a spot detector by matching points. Tab "image": a seeded synthetic 256 × 256
// image (spots of σ = 1.5 px, amplitudes 8–80, noise 10; density 10–1000 spots) and the
// detections of a zero-mean Gaussian or LoG detector (threshold k noise sd, window r = 2),
// optionally refined by a least-squares Gaussian fit (7 × 7 patch) and optionally shifted by
// 0.5 px (a biased detector). Truth = rings, detections = dots, matched pairs joined by
// lines coloured by distance; one-to-one matching within a gate radius, greedy or optimal.
// Plots: F1 against the gate, the precision–recall curve as the threshold sweeps (current k
// as a dot, AP and the best-F1 threshold), F1 and RMSE against density (one seeded image per
// density). Tab "by hand": draggable truths and detections, greedy and optimal side by side.
//
// <div data-demo="spotmatch" data-tab="0|1" data-preset="greedy|small|large|pr|density|fit|bias" data-presets="..."></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { localMaxima2D } from "../lib/edges";
import { prFromScores } from "../lib/eval";
import { spotImage, spotFilter, estimateNoise, noiseGain, matchPoints, SpotTruth, Pt, PointMatch } from "../lib/spotdetect";
import { fitSpots } from "../lib/spotfit";
import { ImgView, injectStyle, label, shownIds, tabRow, TP_COL, FP_COL, FN_COL } from "../spotdetect/common";

const NAME = "spotmatch";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .sm-main { flex: 0 1 min(410px, calc(100vh - 250px)); min-width: 300px; }
.qb-demo[data-demo="${NAME}"] .sm-mid { flex: 0 1 280px; min-width: 200px; display: flex; flex-direction: column; gap: .1em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; max-width: 370px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider { grid-template-columns: auto 1fr 3em; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider input[type=range] { min-width: 4em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .sm-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.qb-demo[data-demo="${NAME}"] .sm-hand svg { touch-action: none; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .sm-main { flex: 0 0 420px; }
.reveal .qb-demo[data-demo="${NAME}"] .sm-mid { flex: 0 0 320px; gap: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

const N = 256, PSF = 1.5, K_MIN = 2, GATES = Array.from({ length: 20 }, (_, i) => 0.25 * (i + 1)), DENS = [25, 50, 100, 200, 400, 700, 1000];
type Det = "zmgauss" | "log";
interface State { tab: number; det: Det; k: number; gate: number; optimal: boolean; fit: boolean; bias: boolean; count: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { tab: 0, det: "zmgauss", k: 4, gate: 2, optimal: true, fit: false, bias: false, count: 80 };
const PRESETS: Preset[] = [
  { id: "greedy", label: "greedy loses a match", s: { tab: 1, gate: 2.5 } },
  { id: "small", label: "gate too small: good spots counted wrong", s: { gate: 0.5 } },
  { id: "large", label: "gate too large: everything matches", s: { gate: 5, k: 3, count: 200 } },
  { id: "pr", label: "PR curve: pick the threshold", s: { k: 3 } },
  { id: "density", label: "density: recall collapses", s: { count: 1000 } },
  { id: "fit", label: "fitting rescues localisation, not merges", s: { count: 300, fit: true } },
  { id: "bias", label: "a biased detector (offset 0.5 px)", s: { bias: true, fit: true } },
];
const HAND0 = { T: [[0, 0], [2.5, 0]] as Pt[], D: [[1, 0], [-2, 0]] as Pt[] };

interface Cand { x: number; y: number; score: number }

/** Detect: matched filter or LoG, maxima (r = 2), scores in noise sd, optional fit and offset. */
function detect(img: Grey, det: Det, fit: boolean, kMin: number): Cand[] {
  const resp = spotFilter(img, det, PSF), noise = estimateNoise(img), rn = noise * noiseGain(det, PSF);
  const m = localMaxima2D(resp, kMin * rn, 2, 3);
  return m.map(p => {
    let x = p.x, y = p.y;
    if (fit) {
      const R = 3, w = 2 * R + 1, patch = new Float64Array(w * w);
      let lo = Infinity;
      for (let j = 0; j < w; j++) for (let i = 0; i < w; i++) { const v = img.data[(p.y - R + j) * N + (p.x - R + i)]; patch[j * w + i] = v; lo = Math.min(lo, v); }
      let s = 0; for (const v of patch) s += v - lo;
      const f = fitSpots(patch, w, w, PSF, [{ x: R, y: R, N: Math.max(10, s) }], lo, { maxIter: 15 });
      const sx = f.spots[0].x, sy = f.spots[0].y;
      if (Math.abs(sx - R) < 1.5 && Math.abs(sy - R) < 1.5) { x = p.x - R + sx; y = p.y - R + sy; }
    }
    return { x, y, score: p.value / rn };
  });
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = host.dataset.tab ? Number(host.dataset.tab) : undefined;
  const start = (): State => ({ ...DEFAULT, ...(tab0 !== undefined ? { tab: tab0 } : {}), ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "sm-main", "", host);
  const setTab = tabRow(main, ["image", "by hand"], i => { st.tab = i; draw(); });
  const view = new ImgView(main, "");
  const handDiv = htmlEl("div", "sm-hand", "", main);
  const hp1 = new Plot(handDiv, { width: 430, height: 230, x: [-4, 5], y: [-2.2, 2.2], equal: true, font: 18, title: "greedy (closest pair first)", margin: { l: 40, r: 8, t: 40, b: 34 } });
  const mid = htmlEl("div", "sm-mid", "", host);
  const hp2 = new Plot(mid, { width: 320, height: 200, x: [-4, 5], y: [-2.2, 2.2], equal: true, font: 18, title: "optimal (most matches)", margin: { l: 40, r: 8, t: 40, b: 34 } });
  const gp = new Plot(mid, { width: 320, height: 165, x: [0, 5], y: [0, 1.02], font: 17, margin: { l: 56, r: 8, t: 10, b: 44 }, xlabel: "gate radius (px)", yticks: [0, 0.5, 1] });
  const gF1 = gp.line([], [], { color: seriesColor(0), width: 3 });
  const gNow = gp.vline(2, { color: "var(--qb-c4)", width: 2 });
  gp.text(0.15, 0.12, "F1", { size: 16 });
  const pr = new Plot(mid, { width: 320, height: 180, x: [0, 1], y: [0, 1.02], font: 17, margin: { l: 56, r: 8, t: 10, b: 44 }, xlabel: "recall", ylabel: "precision", yticks: [0, 0.5, 1], xticks: [0, 0.5, 1] });
  const prL = pr.line([], [], { color: seriesColor(1), width: 3 });
  const prDot = pr.scatter([], [], { color: "var(--qb-c4)", r: 7 });
  const dp = new Plot(mid, { width: 320, height: 175, x: [0, 1050], y: [0, 1.02], font: 17, margin: { l: 56, r: 8, t: 10, b: 44 }, xlabel: "spots per image", xticks: [0, 500, 1000], yticks: [0, 0.5, 1] });
  const dF1 = dp.line([], [], { color: seriesColor(0), width: 3 });
  const dR = dp.line([], [], { color: seriesColor(2), width: 3, dash: "6 4" });
  const dNow = dp.vline(80, { color: "var(--qb-c4)", width: 2 });
  dp.legend([{ label: "F1", color: seriesColor(0) }, { label: "RMSE (px)", color: seriesColor(2), dash: "6 4" }], "tr");
  const busy = htmlEl("div", "sm-busy", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const detLab = label(panel, "detector");
  const detBtns = presets(panel, [
    { label: "matched filter (zero-mean Gaussian)", apply: () => { st.det = "zmgauss"; makeDets(); } },
    { label: "LoG", apply: () => { st.det = "log"; makeDets(); } },
  ]);
  const gS = slider(panel, "gate radius (px)", 0.25, 5, 0.25, st.gate, v => { st.gate = v; draw(); });
  const kDiv = htmlEl("div", "", "", panel);
  const kS = slider(kDiv, "threshold k", K_MIN, 10, 0.25, st.k, v => { st.k = v; draw(); });
  const cS = slider(kDiv, "spots", 10, 1000, 10, st.count, v => { st.count = v; makeImage(); });
  const row = htmlEl("div", "", "", panel);
  row.style.display = "flex"; row.style.flexWrap = "wrap"; row.style.gap = "0 1em";
  const optC = checkbox(row, "optimal matching", st.optimal, v => { st.optimal = v; draw(); });
  const fitC = checkbox(row, "fit positions", st.fit, v => { st.fit = v; makeDets(); });
  const biasC = checkbox(row, "offset 0.5 px", st.bias, v => { st.bias = v; draw(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); hand.T = HAND0.T.map(p => [...p] as Pt); hand.D = HAND0.D.map(p => [...p] as Pt); sync(); makeImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void { Object.assign(st, DEFAULT, tab0 !== undefined ? { tab: tab0 } : {}, p.s); sync(); makeImage(); }
  function sync(): void {
    gS.set(st.gate); kS.set(st.k); cS.set(st.count); optC.checked = st.optimal; fitC.checked = st.fit; biasC.checked = st.bias;
    detBtns.forEach((b, i) => b.classList.toggle("active", i === (st.det === "zmgauss" ? 0 : 1)));
  }

  // ---- hand-made scene ----
  const hand = { T: HAND0.T.map(p => [...p] as Pt), D: HAND0.D.map(p => [...p] as Pt) };
  const handMarks = [hp1, hp2].map(p => p.mark<[PointMatch]>((g, m) => drawHand(p, g, m), [matchPoints(hand.T, hand.D, st.gate)]));
  function drawHand(p: Plot, g: SVGGElement, m: PointMatch): void {
    const el = (tag: string, a: Record<string, string | number>) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v)); g.appendChild(e); return e; };
    const sx = (v: number) => p.x(v), sy = (v: number) => p.y(v), u = Math.abs(p.x(1) - p.x(0));
    hand.T.forEach(t => el("circle", { cx: sx(t[0]), cy: sy(t[1]), r: st.gate * u, fill: "rgba(120,120,120,0.08)", stroke: "#aaa", "stroke-dasharray": "4 4" }));
    for (const q of m.pairs) el("line", { x1: sx(hand.T[q.t][0]), y1: sy(hand.T[q.t][1]), x2: sx(hand.D[q.d][0]), y2: sy(hand.D[q.d][1]), stroke: TP_COL, "stroke-width": 4 });
    const mt = new Set(m.pairs.map(q => q.t)), md = new Set(m.pairs.map(q => q.d));
    hand.T.forEach((t, i) => { const c = el("circle", { cx: sx(t[0]), cy: sy(t[1]), r: 13, fill: "#fff", stroke: mt.has(i) ? TP_COL : FN_COL, "stroke-width": 4 }); c.setAttribute("data-k", "T" + i); });
    hand.D.forEach((d, i) => { const c = el("circle", { cx: sx(d[0]), cy: sy(d[1]), r: 9, fill: md.has(i) ? TP_COL : FP_COL, stroke: "#000", "stroke-width": 1.5 }); c.setAttribute("data-k", "D" + i); });
    hand.T.forEach((t, i) => { const e = el("text", { x: sx(t[0]), y: sy(t[1]) - 18, "text-anchor": "middle", "font-size": 17 }); e.textContent = "T" + (i + 1); });
    hand.D.forEach((d, i) => { const e = el("text", { x: sx(d[0]), y: sy(d[1]) + 30, "text-anchor": "middle", "font-size": 17 }); e.textContent = "D" + (i + 1); });
  }
  for (const p of [hp1, hp2]) {
    let drag: string | null = null;
    p.svg.style.touchAction = "none";
    p.svg.addEventListener("pointerdown", ev => {
      const [x, y] = p.toData(ev);
      let best: string | null = null, bd = 0.6;
      hand.T.forEach((t, i) => { const d = Math.hypot(t[0] - x, t[1] - y); if (d < bd) { bd = d; best = "T" + i; } });
      hand.D.forEach((t, i) => { const d = Math.hypot(t[0] - x, t[1] - y); if (d < bd) { bd = d; best = "D" + i; } });
      if (best) { drag = best; p.svg.setPointerCapture(ev.pointerId); ev.preventDefault(); }
    });
    p.svg.addEventListener("pointermove", ev => {
      if (!drag) return;
      const [x, y] = p.toData(ev), i = Number(drag.slice(1)), q = Math.round(x * 4) / 4, r = Math.round(y * 4) / 4;
      (drag[0] === "T" ? hand.T : hand.D)[i] = [Math.max(-3.8, Math.min(4.8, q)), Math.max(-2, Math.min(2, r))];
      draw();
    });
    p.svg.addEventListener("pointerup", () => { drag = null; });
  }

  // ---- image ----
  let img: Grey | null = null, truth: SpotTruth[] = [], cands: Cand[] = [];
  function makeImage(): void {
    const s = spotImage({ size: N, count: st.count, sigma: PSF, amplitude: [8, 80], background: 40, noise: 10, seed: 21 });
    img = s.image; truth = s.truth;
    makeDets();
  }
  function makeDets(): void {
    if (!img) return;
    sync();
    const t0 = performance.now();
    cands = detect(img, st.det, st.fit, K_MIN);
    busy.textContent = `detection${st.fit ? " + fitting" : ""}: ${Math.round(performance.now() - t0)} ms`;
    draw();
    scheduleDensity();
  }
  const T = (): Pt[] => truth.map(t => [t.x, t.y]);
  const off = (c: Cand): Pt => [c.x + (st.bias ? 0.5 : 0), c.y + (st.bias ? 0.5 : 0)];

  let densKey = "", densTimer = 0;
  function scheduleDensity(): void {
    const key = `${st.det}:${st.fit}:${st.k}:${st.gate}:${st.optimal}:${st.bias}`;
    if (key === densKey) return;
    densKey = key; dF1.set([], []); dR.set([], []);
    clearTimeout(densTimer);
    densTimer = window.setTimeout(() => {
      const f: number[] = [], r: number[] = [];
      for (const n of DENS) {
        const s = spotImage({ size: N, count: n, sigma: PSF, amplitude: [8, 80], background: 40, noise: 10, seed: 300 + n });
        const c = detect(s.image, st.det, st.fit, st.k).map(off);
        const m = matchPoints(s.truth.map(t => [t.x, t.y] as Pt), c, st.gate, st.optimal ? "optimal" : "greedy");
        f.push(m.f1); r.push(m.rmse);
      }
      if (densKey === key) { dF1.set(DENS, f); dR.set(DENS, r); busy.textContent = `F1 and RMSE against density: ${DENS.length} seeded images`; }
    }, 60);
  }

  function draw(): void {
    setTab(st.tab);
    const isImg = st.tab === 0;
    view.fig.style.display = isImg ? "" : "none";
    handDiv.style.display = isImg ? "none" : "";
    hp2.svg.style.display = isImg ? "none" : "";
    for (const p of [gp, pr, dp]) p.svg.style.display = isImg ? "" : "none";
    busy.style.display = kDiv.style.display = detLab.style.display = isImg ? "" : "none";
    detBtns.forEach(b => { (b.parentElement as HTMLElement).style.display = isImg ? "" : "none"; });
    fitC.parentElement!.style.display = biasC.parentElement!.style.display = isImg ? "" : "none";
    optC.parentElement!.style.display = isImg ? "" : "none";
    if (!isImg) return drawHandTab();
    if (!img) return;
    const dets = cands.filter(c => c.score > st.k), D = dets.map(off), Tr = T();
    const m = matchPoints(Tr, D, st.gate, st.optimal ? "optimal" : "greedy");
    view.grey(img, { mode: "auto" });
    view.clear();
    view.setCaption(`${truth.length} true spots, ${dets.length} detections`);
    const mt = new Set(m.pairs.map(p => p.t)), md = new Set(m.pairs.map(p => p.d));
    for (const p of m.pairs) {
      const t = Tr[p.t], d = D[p.d], h = Math.max(0, 120 - 120 * p.dist / Math.max(0.5, st.gate));
      view.add("line", { x1: t[0] + 0.5, y1: t[1] + 0.5, x2: d[0] + 0.5, y2: d[1] + 0.5, stroke: `hsl(${h},90%,50%)`, "stroke-width": 2.5 });
    }
    Tr.forEach((t, i) => view.add("circle", { cx: t[0] + 0.5, cy: t[1] + 0.5, r: 3.2, fill: "none", stroke: mt.has(i) ? "#fff" : FN_COL, "stroke-width": mt.has(i) ? 1.2 : 2.2 }));
    D.forEach((d, j) => view.add("circle", { cx: d[0] + 0.5, cy: d[1] + 0.5, r: 1.1, fill: md.has(j) ? TP_COL : FP_COL }));
    // F1 vs gate
    gF1.set(GATES, GATES.map(g => matchPoints(Tr, D, g, st.optimal ? "optimal" : "greedy").f1)); gNow.set(st.gate);
    // PR curve: candidates ≥ K_MIN by score, each takes the nearest free truth within the gate
    const order = cands.map((_, i) => i).sort((a, b) => cands[b].score - cands[a].score);
    const used = new Uint8Array(Tr.length), isTP = new Uint8Array(cands.length);
    for (const i of order) {
      const [x, y] = off(cands[i]);
      let best = -1, bd = st.gate;
      Tr.forEach((t, j) => { if (!used[j]) { const d = Math.hypot(t[0] - x, t[1] - y); if (d <= bd) { bd = d; best = j; } } });
      if (best >= 0) { used[best] = 1; isTP[i] = 1; }
    }
    const P = prFromScores(cands.map(c => c.score), isTP, Tr.length);
    prL.set(P.recall, P.precision);
    let kIdx = -1; for (let q = 0; q < P.order.length; q++) if (cands[P.order[q]].score > st.k) kIdx = q;
    prDot.set(kIdx >= 0 ? [P.recall[kIdx]] : [], kIdx >= 0 ? [P.precision[kIdx]] : []);
    let bestF = 0, bestK = NaN;
    for (let q = 0; q < P.order.length; q++) { const tp = P.recall[q] * Tr.length, f = 2 * tp / (q + 1 + Tr.length); if (f > bestF) { bestF = f; bestK = cands[P.order[q]].score; } }
    dNow.set(st.count);
    scheduleDensity();
    ro.textContent =
      `TP ${m.tp}   FP ${m.fp}   FN ${m.fn}\nprecision ${m.precision.toFixed(2)}   recall ${m.recall.toFixed(2)}\nF1 ${m.f1.toFixed(2)}   Jaccard ${m.jaccard.toFixed(2)}\n` +
      `RMSE ${Number.isFinite(m.rmse) ? m.rmse.toFixed(2) : "–"} px   mean offset (${m.bias.map(v => Number.isFinite(v) ? v.toFixed(2) : "–").join(", ")})\n` +
      `AP ${P.ap.toFixed(2)}; best F1 ${bestF.toFixed(2)} at k ≈ ${bestK.toFixed(1)}`;
  }

  function drawHandTab(): void {
    const g = matchPoints(hand.T, hand.D, st.gate, "greedy"), o = matchPoints(hand.T, hand.D, st.gate, "optimal");
    handMarks[0].set(g); handMarks[1].set(o);
    const line = (n: string, m: PointMatch) => `${n}: TP ${m.tp} FP ${m.fp} FN ${m.fn}  F1 ${m.f1.toFixed(2)}  RMSE ${Number.isFinite(m.rmse) ? m.rmse.toFixed(2) : "–"}`;
    ro.textContent = `gate ${st.gate}; drag the points\n${line("greedy", g)}\n${line("optimal", o)}\n` +
      `pairs (greedy): ${g.pairs.map(p => `D${p.d + 1}–T${p.t + 1} ${p.dist.toFixed(2)}`).join(", ") || "none"}\n` +
      `pairs (optimal): ${o.pairs.map(p => `D${p.d + 1}–T${p.t + 1} ${p.dist.toFixed(2)}`).join(", ") || "none"}`;
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
