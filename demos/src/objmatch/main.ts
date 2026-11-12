// Object matching: predicted objects against ground-truth nuclei. Predictions come from an
// error generator applied to the truth (pipeline mode: boundary jitter, outline bias, merges,
// splits, missed dim nuclei, specks) or are edited by hand as ellipses (edit mode: drag,
// resize and rotate with handles, add, delete, split, merge). All numbers are computed on
// pixels: IoU table, one-to-one matching at τ (greedy or optimal), TP/FP/FN objects,
// precision/recall/F1, F1 vs τ with the pixel Dice for reference, PR curve and AP, splits and
// merges, and the agreement of a second annotator as a ceiling.
// Auto-mounts on <div data-demo="objmatch">. Options: data-preset="default|dice|sloppy|split|
// greedy|specks|annotator", data-panels="image,table,curve,controls" (any subset),
// data-tab="f1|pr".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, Slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { toBytes } from "../lib/imageops";
import { boundaryMask } from "../lib/eval";
import * as M from "./model";

const NAME = "objmatch";
const C_TP = [44, 160, 44], C_FP = [214, 39, 40], C_FN = [255, 127, 14];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .om-img { flex: 0 1 330px; min-width: 240px; }
[data-demo="${NAME}"] .om-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .om-stage canvas, [data-demo="${NAME}"] .om-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
[data-demo="${NAME}"] .om-stage canvas { image-rendering: pixelated; }
[data-demo="${NAME}"] .om-stage svg { touch-action: none; cursor: crosshair; overflow: visible; }
[data-demo="${NAME}"] .om-key { font-size: 14px; display: flex; flex-wrap: wrap; gap: .2em .8em; margin-top: .25em; }
[data-demo="${NAME}"] .om-key span::before { content: ""; display: inline-block; width: .9em; height: .9em; margin-right: .3em; vertical-align: -.1em; border-radius: 2px; background: var(--k); }
[data-demo="${NAME}"] .om-table { flex: 0 1 270px; min-width: 200px; }
[data-demo="${NAME}"] .om-table .qb-note { margin-top: .3em; }
[data-demo="${NAME}"] .om-curve { flex: 1 1 340px; max-width: 420px; min-width: 260px; }
[data-demo="${NAME}"] .om-tabs { display: flex; gap: .35em; justify-content: center; margin-bottom: .2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; margin-top: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; gap: .35em; }
[data-demo="${NAME}"] .om-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .om-hidden { display: none !important; }
[data-demo="${NAME}"] .om-edit .qb-btn:disabled { opacity: .4; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .7em; }
.reveal [data-demo="${NAME}"] .om-img { flex-basis: 330px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; line-height: 1.25; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; flex-basis: 300px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .om-curve { max-width: 400px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);
const clone = (objs: M.PredObj[]): M.PredObj[] => objs.map(o => ({ ...o, parts: o.parts.map(p => ({ ...p, e: { ...p.e } })) }));

function centroids(lab: Int32Array, n: number): { x: number; y: number; area: number }[] {
  const s = Array.from({ length: n }, () => ({ x: 0, y: 0, area: 0 }));
  for (let i = 0; i < lab.length; i++) { const l = lab[i]; if (l > 0 && l <= n) { s[l - 1].x += i % M.W; s[l - 1].y += Math.floor(i / M.W); s[l - 1].area++; } }
  return s.map(c => (c.area ? { x: c.x / c.area, y: c.y / c.area, area: c.area } : { x: NaN, y: NaN, area: 0 }));
}

function mount(host: HTMLElement): void {
  injectStyle();
  const panels = new Set((host.dataset.panels ?? "image,table,curve,controls").split(/[\s,]+/));
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  const scene = M.makeScene(1);
  const onSlide = !!host.closest(".reveal"), font = onSlide ? 26 : 18;
  const imgBytes = toBytes(scene.image, { mode: "clip", lo: 20, hi: 220 });
  const st = {
    settings: { ...p0.settings } as M.Settings, tau: p0.tau, method: p0.method as "greedy" | "optimal",
    mode: "pipeline" as "pipeline" | "edit", objs: [] as M.PredObj[], sel: -1, annotator: !!p0.annotator,
    tab: (host.dataset.tab === "pr" ? "pr" : p0.tab ?? "f1") as "f1" | "pr", focus: { x: 128, y: 128 },
  };
  const firstPair = scene.pairs[0];
  const e0 = scene.ellipses[firstPair[0]], e1 = scene.ellipses[firstPair[1]];
  st.focus = { x: (e0.x + e1.x) / 2, y: (e0.y + e1.y) / 2 };

  // ---- layout ----
  const imgFig = htmlEl("figure", "om-img", "", host);
  htmlEl("figcaption", "", "synthetic nuclei: truth (white) and prediction", imgFig);
  const stage = htmlEl("div", "om-stage", "", imgFig);
  const cv = htmlEl("canvas", "", "", stage);
  cv.width = M.W; cv.height = M.W;
  const svg = svgEl("svg", { viewBox: `0 0 ${M.W} ${M.W}` }, stage);
  const key = htmlEl("div", "om-key", "", imgFig);
  for (const [t, c] of [["TP", C_TP], ["FP", C_FP], ["FN (missed truth)", C_FN]] as const) htmlEl("span", "", t, key).style.setProperty("--k", `rgb(${c.join(",")})`);
  const ro = htmlEl("div", "qb-readout", "", imgFig);

  const tabFig = htmlEl("figure", "om-table", "", host);
  htmlEl("figcaption", "", "IoU table near the marker", tabFig);
  const tableSvg = svgEl("svg", { width: "100%", class: "qb-plot-svg" }, tabFig);
  const methRow = htmlEl("div", "qb-presets", "", tabFig);
  methRow.style.justifyContent = "center";
  const methBtns = (["greedy", "optimal"] as const).map(m => button(methRow, m === "greedy" ? "greedy" : "optimal", () => { st.method = m; update(); }));
  htmlEl("div", "qb-note", "Rows: true objects G, columns: predictions P. Boxed: matched pairs; grey: IoU below τ. Click the image to move the marker.", tabFig);

  const curveFig = htmlEl("figure", "om-curve", "", host);
  const tabs = htmlEl("div", "om-tabs", "", curveFig);
  const tabBtns = [button(tabs, "F1 vs τ", () => { st.tab = "f1"; drawCurve(); }), button(tabs, "PR curve at τ", () => { st.tab = "pr"; drawCurve(); })];
  const plotHost = htmlEl("div", "", "", curveFig);
  let plot: Plot | null = null;

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "om-lab", "presets", panel);
  const presetBtns = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const modeRow = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = [button(modeRow, "error generator", () => setMode("pipeline")), button(modeRow, "edit by hand", () => setMode("edit"))];
  const pipe = htmlEl("div", "", "", panel);
  pipe.style.cssText = "display:flex;flex-direction:column;gap:.25em";
  const sl: Record<keyof M.Settings, Slider> = {
    jitter: slider(pipe, "jitter (px)", 0, 4, 0.2, st.settings.jitter, v => setS("jitter", v)),
    dilate: slider(pipe, "outline bias (px)", -3, 3, 0.5, st.settings.dilate, v => setS("dilate", v)),
    merge: slider(pipe, "merge touching", 0, 1, 0.05, st.settings.merge, v => setS("merge", v)),
    split: slider(pipe, "split", 0, 0.5, 0.02, st.settings.split, v => setS("split", v)),
    miss: slider(pipe, "miss dim", 0, 1, 0.05, st.settings.miss, v => setS("miss", v)),
    specks: slider(pipe, "specks", 0, 20, 1, st.settings.specks, v => setS("specks", v)),
  };
  const edit = htmlEl("div", "qb-presets om-edit", "", panel);
  const editBtns = {
    add: button(edit, "add", () => { st.objs.push({ parts: [{ e: { x: st.focus.x, y: st.focus.y, a: 11, b: 9, angle: 0 } }], score: 0.8, kind: "ok" }); st.sel = st.objs.length - 1; update(); }),
    del: button(edit, "delete", () => { if (st.sel >= 0) { st.objs.splice(st.sel, 1); st.sel = -1; update(); } }),
    split: button(edit, "split", () => splitSel()),
    merge: button(edit, "merge with nearest", () => mergeSel()),
  };
  const tauS = slider(panel, "τ (IoU)", 0.1, 0.95, 0.05, st.tau, v => { st.tau = v; update(); });
  const annoC = checkbox(panel, "annotator B (ceiling)", st.annotator, v => { st.annotator = v; drawCurve(); });
  button(panel, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  if (!panels.has("image")) imgFig.classList.add("om-hidden");
  if (!panels.has("table")) tabFig.classList.add("om-hidden");
  if (!panels.has("curve")) curveFig.classList.add("om-hidden");
  if (!panels.has("controls")) panel.classList.add("om-hidden");

  function setS(k: keyof M.Settings, v: number): void {
    st.settings[k] = v;
    presetBtns.forEach(b => b.classList.remove("active"));
    if (st.mode === "pipeline") regen();
  }
  function regen(): void { st.objs = M.generate(scene, st.settings); st.sel = -1; update(); }
  function setMode(m: "pipeline" | "edit"): void {
    st.mode = m;
    if (m === "pipeline") regen(); else { st.objs = clone(st.objs); update(); }
  }
  function load(p: M.Preset): void {
    st.settings = { ...p.settings }; st.tau = p.tau; st.method = p.method; st.annotator = !!p.annotator;
    if (p.tab) st.tab = p.tab;
    (Object.keys(sl) as (keyof M.Settings)[]).forEach(k => sl[k].set(st.settings[k]));
    tauS.set(st.tau); annoC.checked = st.annotator;
    presetBtns.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    if (p.craft === "greedy") {
      st.mode = "edit"; st.objs = M.greedyTrap(scene); st.sel = -1;
      const [i1, i2] = scene.pairs[0];
      st.focus = { x: (scene.ellipses[i1].x + scene.ellipses[i2].x) / 2, y: (scene.ellipses[i1].y + scene.ellipses[i2].y) / 2 };
      update();
    } else { st.mode = "pipeline"; regen(); }
  }
  function splitSel(): void {
    const o = st.objs[st.sel];
    if (!o || o.parts.length !== 1 || o.parts[0].half) return;
    const e = o.parts[0].e;
    st.objs.splice(st.sel, 1, { parts: [{ e: { ...e }, half: 1, gap: 2 }], score: o.score, kind: "split" }, { parts: [{ e: { ...e }, half: -1, gap: 2 }], score: o.score * 0.95, kind: "split" });
    update();
  }
  function mergeSel(): void {
    const o = st.objs[st.sel];
    if (!o) return;
    const c = (q: M.PredObj) => { const n = q.parts.length; return { x: q.parts.reduce((s, p) => s + p.e.x, 0) / n, y: q.parts.reduce((s, p) => s + p.e.y, 0) / n }; };
    const a = c(o);
    let best = -1, bd = Infinity;
    st.objs.forEach((q, j) => { if (j !== st.sel) { const b = c(q), d = Math.hypot(a.x - b.x, a.y - b.y); if (d < bd) { bd = d; best = j; } } });
    if (best < 0) return;
    const merged: M.PredObj = { parts: [...o.parts, ...st.objs[best].parts], score: Math.max(o.score, st.objs[best].score), kind: "merge" };
    st.objs = st.objs.filter((_, j) => j !== st.sel && j !== best);
    st.objs.push(merged); st.sel = st.objs.length - 1;
    update();
  }

  // ---- evaluation and drawing ----
  let ev: M.Evaluation, predLab: Int32Array = new Int32Array(0), predC: { x: number; y: number; area: number }[] = [];
  const truthC = centroids(scene.truth.labels, scene.truth.count);
  const truthB = boundaryMask(scene.truth.labels.map(v => (v ? 1 : 0)), M.W, M.W);
  // boundary of each truth object separately (touching objects share a visible line)
  const truthEdge = new Uint8Array(M.W * M.W);
  for (let y = 0; y < M.W; y++) for (let x = 0; x < M.W; x++) {
    const i = y * M.W + x, l = scene.truth.labels[i];
    if (!l) continue;
    if (truthB[i] || (x < M.W - 1 && scene.truth.labels[i + 1] !== l) || (y < M.W - 1 && scene.truth.labels[i + M.W] !== l) || (x > 0 && scene.truth.labels[i - 1] !== l) || (y > 0 && scene.truth.labels[i - M.W] !== l)) truthEdge[i] = 1;
  }
  const evB = M.evaluate(scene.truth.labels, scene.truth.count, scene.annotatorB.labels, scene.annotatorB.count, new Float64Array(scene.annotatorB.count).fill(1), 0.5, "optimal");

  function update(): void {
    predLab = M.rasterise(st.objs);
    ev = M.evaluate(scene.truth.labels, scene.truth.count, predLab, st.objs.length, st.objs.map(o => o.score), st.tau, st.method);
    predC = centroids(predLab, st.objs.length);
    modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.mode === "pipeline")));
    methBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.method === "greedy")));
    Object.values(sl).forEach(s => (s.input.disabled = st.mode !== "pipeline"));
    pipe.style.opacity = st.mode === "pipeline" ? "1" : "0.45";
    Object.values(editBtns).forEach(b => (b.disabled = st.mode !== "edit"));
    note.textContent = st.mode === "edit" ? "Edit: click an object to select it; drag it to move; drag the round handles to resize and rotate." : "Error generator: the sliders turn the truth into predictions. Click the image to move the table marker.";
    drawImage(); drawOverlay(); drawTable(); drawCurve(); drawReadout();
  }

  function drawImage(): void {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(M.W, M.W), d = id.data;
    const status = new Int8Array(st.objs.length + 1);   // 1 TP, 2 FP
    for (let j = 0; j < st.objs.length; j++) status[j + 1] = ev.match.matchP[j] >= 0 ? 1 : 2;
    for (let i = 0; i < M.W * M.W; i++) {
      let r = imgBytes[i], g = r, b = r;
      const pl = predLab[i];
      if (pl) {
        const c = status[pl] === 1 ? C_TP : C_FP, a = 0.45;
        r = r * (1 - a) + c[0] * a; g = g * (1 - a) + c[1] * a; b = b * (1 - a) + c[2] * a;
        const x = i % M.W, y = (i - x) / M.W;
        const edge = (x > 0 && predLab[i - 1] !== pl) || (x < M.W - 1 && predLab[i + 1] !== pl) || (y > 0 && predLab[i - M.W] !== pl) || (y < M.W - 1 && predLab[i + M.W] !== pl);
        if (edge) { r = c[0] * 0.8; g = c[1] * 0.8; b = c[2] * 0.8; }
      }
      if (truthEdge[i]) {
        const t = scene.truth.labels[i] - 1, x = i % M.W, y = (i - x) / M.W;
        if (ev.match.matchT[t] < 0) { if (((x + y) >> 1) % 2 === 0) { r = C_FN[0]; g = C_FN[1]; b = C_FN[2]; } else { r = 255; g = 255; b = 255; } }
        else { r = 255; g = 255; b = 255; }
      }
      d[4 * i] = r; d[4 * i + 1] = g; d[4 * i + 2] = b; d[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  let tableT: number[] = [], tableP: number[] = [];
  function chooseTableObjects(): void {
    const dist = (c: { x: number; y: number }) => Math.hypot(c.x - st.focus.x, c.y - st.focus.y);
    tableT = truthC.map((c, i) => [i, dist(c)] as [number, number]).sort((a, b) => a[1] - b[1]).slice(0, 4).map(v => v[0]).sort((a, b) => a - b);
    const cand = new Map<number, number>();
    for (const t of tableT) for (let j = 0; j < st.objs.length; j++) if (ev.table.iou[t * ev.table.np + j] > 0) cand.set(j, dist(predC[j]));
    predC.forEach((c, j) => { if (!cand.has(j) && c.area && dist(c) < 30) cand.set(j, dist(c)); });
    tableP = [...cand.entries()].sort((a, b) => a[1] - b[1]).slice(0, 5).map(v => v[0]).sort((a, b) => a - b);
  }

  function drawOverlay(): void {
    svg.replaceChildren();
    chooseTableObjects();
    // focus marker
    const fm = svgEl("circle", { cx: st.focus.x, cy: st.focus.y, r: 4, fill: "none", stroke: "#ffd400", "stroke-width": 1.6 }, svg);
    fm.style.pointerEvents = "none";
    const lab = (x: number, y: number, s: string, col: string) => {
      const t = svgEl("text", { x, y, "text-anchor": "middle", "font-size": 9.5, "font-weight": 700, class: "qb-halo" }, svg);
      t.textContent = s; t.style.fill = col; t.style.stroke = "#000"; t.style.strokeWidth = "2.5px"; t.style.paintOrder = "stroke"; t.style.pointerEvents = "none";
    };
    if (panels.has("table")) {
      tableT.forEach((t, k) => lab(truthC[t].x - 5, truthC[t].y - 2, "G" + (k + 1), "#fff"));
      tableP.forEach((j, k) => lab(predC[j].x + 5, predC[j].y + 9, "P" + (k + 1), "#ffd400"));
    }
    for (const s of ev.splits) lab(truthC[s.t].x, truthC[s.t].y - 14, "split", "#ff9f40");
    for (const m of ev.merges) if (predC[m.p]) lab(predC[m.p].x, predC[m.p].y - 14, "merge", "#ff9f40");
    if (st.mode === "edit" && st.sel >= 0 && st.objs[st.sel]) {
      const p = st.objs[st.sel].parts[0], e = p.e, c = Math.cos(e.angle), s = Math.sin(e.angle);
      svgEl("line", { x1: e.x, y1: e.y, x2: e.x + e.a * c, y2: e.y + e.a * s, stroke: "#ffd400", "stroke-width": 1.2 }, svg).style.pointerEvents = "none";
      svgEl("line", { x1: e.x, y1: e.y, x2: e.x - e.b * s, y2: e.y + e.b * c, stroke: "#ffd400", "stroke-width": 1.2 }, svg).style.pointerEvents = "none";
      const h = (x: number, y: number, name: string) => {
        const q = svgEl("circle", { cx: x, cy: y, r: 4.5, fill: "#fff", stroke: "#1f77b4", "stroke-width": 1.6 }, svg);
        q.dataset.h = name; q.style.cursor = "grab";
      };
      h(e.x + e.a * c, e.y + e.a * s, "a");
      h(e.x - e.b * s, e.y + e.b * c, "b");
    }
  }

  // pointer: select / move / handles in edit mode; move the table marker otherwise
  let drag: { kind: "move" | "a" | "b"; x0: number; y0: number; start: M.Part[] } | null = null;
  svg.addEventListener("pointerdown", ev0 => {
    ev0.preventDefault();
    const [x, y] = svgPoint(svg, ev0);
    const h = (ev0.target as SVGElement).dataset?.h;
    if (st.mode === "edit") {
      if (h && st.sel >= 0) drag = { kind: h as "a" | "b", x0: x, y0: y, start: st.objs[st.sel].parts.map(p => ({ ...p, e: { ...p.e } })) };
      else {
        const xi = Math.round(x), yi = Math.round(y);
        const l = xi >= 0 && yi >= 0 && xi < M.W && yi < M.W ? predLab[yi * M.W + xi] : 0;
        st.sel = l - 1;
        if (l) drag = { kind: "move", x0: x, y0: y, start: st.objs[st.sel].parts.map(p => ({ ...p, e: { ...p.e } })) };
        else st.focus = { x, y };
        update();
      }
      if (drag) svg.setPointerCapture(ev0.pointerId);
    } else { st.focus = { x, y }; drawOverlay(); drawTable(); }
  });
  svg.addEventListener("pointermove", ev1 => {
    if (!drag || st.sel < 0) return;
    const [x, y] = svgPoint(svg, ev1), o = st.objs[st.sel];
    if (drag.kind === "move") o.parts = drag.start.map(p => ({ ...p, e: { ...p.e, x: p.e.x + x - drag!.x0, y: p.e.y + y - drag!.y0 } }));
    else {
      const e = drag.start[0].e, dx = x - e.x, dy = y - e.y, r = Math.max(2, Math.hypot(dx, dy));
      if (drag.kind === "a") o.parts = drag.start.map(p => ({ ...p, e: { ...p.e, a: r, angle: Math.atan2(dy, dx) } }));
      else o.parts = drag.start.map(p => ({ ...p, e: { ...p.e, b: r } }));
    }
    update();
  });
  const up = () => { drag = null; };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);

  function drawTable(): void {
    tableSvg.replaceChildren();
    const cell = 46, lw = 34, th = 26, nr = tableT.length, nc = tableP.length;
    const Wd = lw + Math.max(1, nc) * cell + 4, Hd = th + nr * cell + 4;
    tableSvg.setAttribute("viewBox", `0 0 ${Wd} ${Hd}`);
    tableSvg.style.maxWidth = `${Math.round(Wd * 1.25)}px`;
    const T = ev.table;
    tableP.forEach((j, c) => { const t = svgEl("text", { x: lw + (c + 0.5) * cell, y: th - 7, "text-anchor": "middle", "font-size": 17 }, tableSvg); t.textContent = "P" + (c + 1); });
    tableT.forEach((i, r) => {
      const t = svgEl("text", { x: lw - 5, y: th + (r + 0.5) * cell + 6, "text-anchor": "end", "font-size": 17 }, tableSvg); t.textContent = "G" + (r + 1);
      tableP.forEach((j, c) => {
        const v = T.iou[i * T.np + j], ok = v >= st.tau && v > 0, matched = ev.match.matchT[i] === j;
        const rc = svgEl("rect", { x: lw + c * cell + 1, y: th + r * cell + 1, width: cell - 2, height: cell - 2, rx: 3 }, tableSvg);
        rc.style.fill = ok ? `rgba(44,160,44,${0.15 + 0.6 * v})` : "var(--qb-panel)";
        if (matched) { rc.style.stroke = "var(--qb-fg)"; rc.style.strokeWidth = "3"; }
        const tx = svgEl("text", { x: lw + (c + 0.5) * cell, y: th + (r + 0.5) * cell + 6, "text-anchor": "middle", "font-size": 16 }, tableSvg);
        tx.textContent = v >= 0.005 ? f2(v) : v > 0 ? "<.01" : "·";
        tx.style.fill = ok ? "var(--qb-fg)" : "var(--qb-muted)";
        if (matched) tx.style.fontWeight = "700";
      });
    });
  }

  function drawCurve(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "f1")));
    plotHost.replaceChildren();
    if (st.tab === "f1") {
      plot = new Plot(plotHost, { x: [0.1, 0.95], y: [0, 1.02], xlabel: "IoU threshold τ", ylabel: "object F1", font, width: onSlide ? 460 : 560, height: onSlide ? 500 : 380, xticks: [0.1, 0.3, 0.5, 0.7, 0.9] });
      if (st.annotator) plot.band(M.TAUS, evB.f1, M.TAUS.map(() => 1.02), { color: "#888", fillOpacity: 0.25, label: "above annotator B" });
      if (st.annotator) plot.line(M.TAUS, evB.f1, { color: "#888", width: 2, label: "annotator B vs A" });
      plot.hline(ev.dice, { color: seriesColor(4), width: 2, text: "pixel Dice " + f2(ev.dice) });
      plot.line(M.TAUS, ev.f1, { color: seriesColor(0), label: "F1 (prediction)" });
      const k = M.TAUS.findIndex(t => Math.abs(t - st.tau) < 1e-6);
      plot.scatter([st.tau], [k >= 0 ? ev.f1[k] : ev.scores.f1], { color: seriesColor(1), r: 8 });
      plot.legend(undefined, "bl");
    } else {
      plot = new Plot(plotHost, { x: [0, 1], y: [0, 1.02], xlabel: "recall", ylabel: "precision", font, width: onSlide ? 460 : 560, height: onSlide ? 500 : 380, title: `τ = ${f2(st.tau)}: AP ${f3(ev.pr.ap)}` });
      const R = [0, ...ev.pr.recall], I = [ev.pr.interp[0] ?? 1, ...ev.pr.interp];
      plot.band(R, R.map(() => 0), I, { color: seriesColor(0), fillOpacity: 0.15 });
      plot.line(R, I, { color: seriesColor(0), label: "interpolated" });
      const cols = Array.from(ev.pr.order, j => (st.objs[j] && ev.match.matchP[j] >= 0 ? "rgb(44,160,44)" : "rgb(214,39,40)"));
      plot.line(ev.pr.recall, ev.pr.precision, { color: "var(--qb-muted)", width: 1.5, dash: "4 3" });
      plot.scatter(ev.pr.recall, ev.pr.precision, { colors: cols, r: 5 });
      plot.legend(undefined, "bl");
    }
  }

  function drawReadout(): void {
    const s = ev.scores;
    ro.textContent =
      `τ = ${f2(st.tau)} (${st.method})\n${scene.truth.count} true, ${st.objs.length} predicted\n` +
      `TP ${s.tp}  FP ${s.fp}  FN ${s.fn}\nsplits ${ev.splits.length}  merges ${ev.merges.length}\n` +
      `precision ${f3(s.precision)}  recall ${f3(s.recall)}\n` +
      `F1 ${f3(s.f1)}  TP/(TP+FP+FN) ${f3(s.accuracy)}\n` +
      `pixel Dice ${f3(ev.dice)}\n` +
      `AP at τ ${f3(ev.pr.ap)}\nAP over 0.5–0.95 ${f3(ev.apMean)}`;
  }

  load(p0);
}

autoMount(NAME, mount);
