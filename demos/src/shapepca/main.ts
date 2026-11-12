// Point distribution model: a synthetic training set of landmark shapes (hand, nucleus,
// leaf; lib/shapes.ts, fixed seed), aligned by generalised Procrustes, PCA of the landmark
// vectors. Left: the training set (raw or aligned, with every landmark's scatter) and the
// variance per mode. Centre: the model shape x̄ + Φb, with sliders for b_1..b_5 in units of
// √λ_k (limited to ±3 unless "unlimited"). Mode "project": drag the landmarks of a new shape
// and see its closest model shape (and the clamped one): the projection step of ASM.
// Auto-mounts on <div data-demo="shapepca">. Options: data-preset="<id>", data-family, data-mode="model|project".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, Slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Shape, synthesise, projectB, clampB, procrustes, invertFit, modesFor, family, sampleParams, nPoints, centroidSize } from "../lib/shapes";
import { rng } from "../lib/random";
import { buildModel, varianceShares, ShapeModel, ModelOptions } from "./model";

const NAME = "shapepca";
const K = 5;

interface Preset { id: string; label: string; o: Partial<ModelOptions>; mode?: "model" | "project"; anim?: number; b?: number[]; unlimited?: boolean; view?: "raw" | "aligned" }

const BASE: ModelOptions = { family: "hand", count: 40, noise: 0.01, align: true, corrupt: 0 };
const PRESETS: Preset[] = [
  { id: "spread", label: "first mode of the hand: finger spread", o: {}, anim: 0 },
  { id: "noalign", label: "without alignment: pose becomes a mode", o: { align: false }, anim: 0, view: "raw" },
  { id: "three", label: "3 training shapes", o: { count: 3 } },
  { id: "corrupt", label: "bad correspondence", o: { corrupt: 0.2 }, anim: 0 },
  { id: "implausible", label: "unlimited: implausible shapes", o: {}, b: [0, 0, 5, 0, 0], unlimited: true },
  { id: "project", label: "drag a finger: the model follows", o: {}, mode: "project" },
  { id: "nucleus", label: "nuclei", o: { family: "nucleus" }, anim: 0 },
  { id: "leaf", label: "leaves", o: { family: "leaf" }, anim: 1 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .sp-left { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .sp-main { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .sp-main svg { touch-action: none; }
[data-demo="${NAME}"] .sp-cap { font-size: 16px; text-align: center; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .sp-noaxes .qb-axes { display: none; }
[data-demo="${NAME}"] .sp-noaxes { background: var(--qb-panel); border-radius: 4px; }
[data-demo="${NAME}"] .sp-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
[data-demo="${NAME}"] .sp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .sp-left { flex-basis: 270px; }
.reveal .qb-demo[data-demo="${NAME}"] .sp-main { flex-basis: 440px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function onScreen(el: HTMLElement): boolean {
  if (el.closest("section[aria-hidden='true'], section.past, section.future")) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { o: { ...BASE, family: host.dataset.family ?? BASE.family } as ModelOptions, mode: (host.dataset.mode === "project" ? "project" : "model") as "model" | "project", unlimited: false, view: "aligned" as "raw" | "aligned", kKeep: 3 };
  let M: ShapeModel = buildModel(st.o);
  const b = new Float64Array(K);          // in units of √λ
  let free: Shape | null = null;          // project mode: the shape being dragged
  let anim: number | null = null, t0 = 0;

  // ---- layout ----
  const left = htmlEl("div", "sp-left", "", host);
  const setCap = htmlEl("div", "sp-cap", "", left);
  const pSet = new Plot(left, { width: 320, height: 300, x: [-2.6, 2.6], y: [-2.6, 2.6], equal: true, grid: false, xticks: [], yticks: [], margin: { l: 4, r: 4, t: 4, b: 4 }, cls: "qb-plot-svg sp-noaxes" });
  const gSet = svgEl("g", {}, pSet.layer);
  const pVar = new Plot(left, { width: 320, height: 210, x: [0.4, 8.6], y: [0, 100], xlabel: "mode k", ylabel: "% variance", xticks: [1, 2, 3, 4, 5, 6, 7, 8], yticks: [0, 50, 100], font: 17 });
  const mBars = pVar.bars([], [], { color: seriesColor(0) });
  const mCum = pVar.line([], [], { color: seriesColor(1), width: 2.5 });
  const mCumDots = pVar.scatter([], [], { color: seriesColor(1), r: 4 });
  pVar.hline(95, { color: "var(--qb-muted, #888)", width: 1.5, text: "95 %" });

  const main = htmlEl("div", "sp-main", "", host);
  const mainCap = htmlEl("div", "sp-cap", "", main);
  const pMain = new Plot(main, { width: 440, height: 440, x: [-2.8, 2.8], y: [-2.8, 2.8], equal: true, grid: false, xticks: [], yticks: [], margin: { l: 4, r: 4, t: 4, b: 4 }, cls: "qb-plot-svg sp-noaxes" });
  const gMain = svgEl("g", {}, pMain.over);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sp-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const modeRow = htmlEl("div", "sp-row", "", panel);
  const modeBtns = (["model", "project"] as const).map(m => button(modeRow, m === "model" ? "model: sliders" : "project: drag", () => { st.mode = m; stopAnim(); if (m === "project") newFree(); draw(); }));
  const bS: Slider[] = [];
  for (let k = 0; k < K; k++) bS.push(slider(panel, `b${k + 1}/√λ${k + 1}`, -6, 6, 0.1, 0, v => { stopAnim(); b[k] = v; clearActive(); draw(); }));
  const animRow = htmlEl("div", "sp-row", "", panel);
  const animBtns = [0, 1, 2].map(k => button(animRow, `animate b${k + 1}`, () => startAnim(k)));
  button(animRow, "stop", () => stopAnim());
  button(animRow, "b = 0", () => { stopAnim(); b.fill(0); draw(); });
  const famRow = htmlEl("div", "sp-row", "", panel);
  const famBtns = ["hand", "nucleus", "leaf"].map(f => button(famRow, f, () => { st.o.family = f; rebuild(); }));
  const viewBtns = (["raw", "aligned"] as const).map(v => button(famRow, v, () => { st.view = v; draw(); }));
  const cntS = slider(panel, "shapes", 3, 60, 1, st.o.count, v => { st.o.count = v; rebuild(); });
  const noiseS = slider(panel, "landmark noise", 0, 0.06, 0.005, st.o.noise, v => { st.o.noise = v; rebuild(); });
  const keepS = slider(panel, "modes kept", 1, K, 1, st.kKeep, v => { st.kKeep = v; draw(); });
  const opt = htmlEl("div", "sp-row", "", panel);
  const alignBox = checkbox(opt, "align (GPA)", st.o.align, v => { st.o.align = v; rebuild(); });
  const corruptBox = checkbox(opt, "bad correspondence", st.o.corrupt > 0, v => { st.o.corrupt = v ? 0.2 : 0; rebuild(); });
  const unlimBox = checkbox(opt, "unlimited", st.unlimited, v => { st.unlimited = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function clearActive(): void { presetBtns.forEach(x => x.classList.remove("active")); }

  function load(p: Preset): void {
    stopAnim();
    st.o = { ...BASE, ...p.o };
    st.mode = p.mode ?? "model"; st.unlimited = !!p.unlimited; st.view = p.view ?? "aligned";
    b.fill(0);
    if (p.b) p.b.forEach((v, k) => (b[k] = v));
    presetBtns.forEach((x, i) => x.classList.toggle("active", PRESETS[i] === p));
    rebuild(false);
    if (p.anim !== undefined) startAnim(p.anim);
  }

  function rebuild(clear = true): void {
    if (clear) clearActive();
    M = buildModel(st.o);
    st.kKeep = Math.min(K, Math.max(1, modesFor(M.model, 0.95)));
    if (st.mode === "project") newFree();
    draw();
  }

  /** A new shape (unseen parameters), in the model frame, slightly perturbed: the start of project mode. */
  function newFree(): void {
    const f = family(st.o.family), r = rng(99);
    const s = f.make(sampleParams(f, r));
    const D = Math.sqrt(nPoints(s)), k = D / centroidSize(s);
    const y = s.map((v, i) => (i % 2 ? -v : v) * k);
    free = procrustes(y, M.model.mean).aligned;
  }

  // ---- animation ----
  function startAnim(k: number): void {
    if (st.mode !== "model") { st.mode = "model"; }
    stopAnim();
    b.fill(0);
    t0 = performance.now();
    const id = (anim = k);
    const frame = (t: number) => {
      if (anim !== id) return;
      if (!onScreen(host)) { anim = null; draw(); return; }
      b.fill(0); b[k] = 3 * Math.sin((t - t0) / 600);
      draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
  function stopAnim(): void { anim = null; }

  // ---- drawing ----
  function poly(g: SVGGElement, P: Plot, s: Shape, col: string, w: number, op = 1, dash = "", fill = "none"): void {
    let d = "";
    for (let i = 0; i < s.length; i += 2) d += (i ? "L" : "M") + P.x(s[i]).toFixed(1) + "," + P.y(s[i + 1]).toFixed(1);
    svgEl("path", { d: d + "Z", fill, "fill-opacity": 0.12, stroke: col, "stroke-width": w, opacity: op, "stroke-dasharray": dash, "stroke-linejoin": "round" }, g);
  }

  function setDomainFor(P: Plot, shapes: Shape[]): void {
    let m = 0;
    for (const s of shapes) for (const v of s) m = Math.max(m, Math.abs(v));
    const R = Math.max(1.5, m * 1.08);
    P.setDomain([-R, R], [-R, R]);
  }

  function drawSet(): void {
    gSet.replaceChildren();
    const shapes = st.view === "raw" ? M.raw : M.aligned;
    setDomainFor(pSet, shapes);
    const n = nPoints(shapes[0]);
    shapes.forEach((s, j) => poly(gSet, pSet, s, M.corrupted[j] ? seriesColor(1) : "var(--qb-muted, #888)", 1.2, 0.55));
    if (st.view === "aligned") {
      for (const s of shapes) for (let i = 0; i < n; i++) svgEl("circle", { cx: pSet.x(s[2 * i]), cy: pSet.y(s[2 * i + 1]), r: 2, fill: seriesColor(i % 5), opacity: 0.8 }, gSet);
      poly(gSet, pSet, M.model.mean, "var(--qb-fg, #222)", 2.5);
    }
    const nBad = M.corrupted.filter(Boolean).length;
    setCap.textContent = `${shapes.length} training shapes, ${st.view}${st.view === "aligned" && st.o.align ? ` (GPA: ${M.gpaIter} it.)` : st.view === "aligned" ? " (not aligned)" : ""}${nBad ? `, ${nBad} misnumbered (red)` : ""}`;
  }

  function drawVar(): void {
    const k = Math.min(8, M.model.values.length), v = varianceShares(M.model, k), ks = v.share.map((_, i) => i + 1);
    mBars.set(ks, v.share); mCum.set(ks, v.cum); mCumDots.set(ks, v.cum);
  }

  function drawMain(): void {
    gMain.replaceChildren();
    const m = M.model, lim = st.unlimited ? 6 : 3;
    const sd = Array.from({ length: K }, (_, k) => Math.sqrt(m.values[k]));
    for (let k = 0; k < K; k++) {
      const off = k >= m.values.length || sd[k] < 1e-9 * Math.sqrt(m.total);
      bS[k].input.disabled = off || st.mode === "project";
      b[k] = Math.max(-lim, Math.min(lim, b[k]));
      bS[k].set(Number(b[k].toFixed(2)));
      bS[k].input.min = String(-lim); bS[k].input.max = String(lim);
    }
    setDomainFor(pMain, [m.mean.map(v => v * 1.35)]);
    poly(gMain, pMain, m.mean, "var(--qb-muted, #888)", 2, 1, "6 5");
    if (st.mode === "model") {
      const shape = synthesise(m, Array.from(b, (v, k) => v * sd[k]));
      poly(gMain, pMain, shape, seriesColor(0), 3.5, 1, "", seriesColor(0));
      const n = nPoints(shape);
      for (let i = 0; i < n; i++) svgEl("circle", { cx: pMain.x(shape[2 * i]), cy: pMain.y(shape[2 * i + 1]), r: M.fam.anchors.includes(i) ? 6 : 4, fill: seriesColor(0) }, gMain);
      const big = Array.from(b).some(v => Math.abs(v) > 3);
      mainCap.textContent = `x̄ + Φb (dashed: the mean)${big ? ": beyond ±3√λ" : ""}`;
      ro.textContent = readout();
    } else drawProject();
  }

  function drawProject(): void {
    if (!free) newFree();
    const m = M.model, k = st.kKeep, f = free!;
    // pose: align the free shape to the mean; project in the model frame; map back
    const fit = procrustes(f, m.mean);
    const bRaw = projectB(m, fit.aligned, k), bCl = clampB(m, bRaw, 3);
    const back = (bb: Float64Array) => invertFit(fit, synthesise(m, bb));
    const proj = back(bRaw), clamped = back(bCl);
    poly(gMain, pMain, f, seriesColor(3), 2.5, 1, "", seriesColor(3));
    poly(gMain, pMain, proj, seriesColor(0), 2.5, 1, "7 4");
    poly(gMain, pMain, clamped, seriesColor(2), 3.5);
    const n = nPoints(f);
    for (let i = 0; i < n; i++) {
      const h = svgEl("circle", { cx: pMain.x(f[2 * i]), cy: pMain.y(f[2 * i + 1]), r: 7, fill: seriesColor(3), stroke: "var(--qb-bg, #fff)", "stroke-width": 1.5 }, gMain);
      h.style.cursor = "grab";
      h.addEventListener("pointerdown", ev => {
        ev.preventDefault(); clearActive();
        const mv = (e: PointerEvent) => { const [x, y] = pMain.toData(e); free![2 * i] = x; free![2 * i + 1] = y; drawMain(); };
        const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
        window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
      });
    }
    for (let j = 0; j < K; j++) bS[j].set(j < k ? Number(Math.max(-6, Math.min(6, bRaw[j] / Math.sqrt(m.values[j]))).toFixed(2)) : 0);
    mainCap.textContent = `orange: your shape · dashed blue: model (k = ${k}) · green: clamped`;
    const lines = [`k = ${k} modes kept`];
    for (let j = 0; j < k; j++) lines.push(`b${j + 1}/√λ${j + 1} = ${(bRaw[j] / Math.sqrt(m.values[j])).toFixed(2).padStart(6)}${Math.abs(bRaw[j]) > 3 * Math.sqrt(m.values[j]) ? "  → clamped to ±3" : ""}`);
    ro.textContent = lines.join("\n");
  }

  function readout(): string {
    const m = M.model, v = varianceShares(m, Math.min(K, m.values.length));
    const lines = [`mode   % var   cum %`];
    for (let k = 0; k < v.share.length; k++) lines.push(`${String(k + 1).padStart(3)}  ${v.share[k].toFixed(1).padStart(6)}  ${v.cum[k].toFixed(1).padStart(6)}`);
    lines.push(`95 % needs ${modesFor(m, 0.95)} modes`);
    return lines.join("\n");
  }

  function draw(): void {
    modeBtns.forEach((x, i) => x.classList.toggle("active", ["model", "project"][i] === st.mode));
    famBtns.forEach((x, i) => x.classList.toggle("active", ["hand", "nucleus", "leaf"][i] === st.o.family));
    viewBtns.forEach((x, i) => x.classList.toggle("active", ["raw", "aligned"][i] === st.view));
    animBtns.forEach((x, i) => x.classList.toggle("active", anim === i));
    cntS.set(st.o.count); noiseS.set(st.o.noise); keepS.set(st.kKeep);
    alignBox.checked = st.o.align; corruptBox.checked = st.o.corrupt > 0; unlimBox.checked = st.unlimited;
    if (anim === null) { drawSet(); drawVar(); }
    drawMain();
  }

  load(p0);
  if (host.dataset.mode === "project") { st.mode = "project"; newFree(); draw(); }
}

autoMount(NAME, mount);
