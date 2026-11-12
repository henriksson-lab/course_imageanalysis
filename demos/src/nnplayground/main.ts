// nnplayground: a small neural network that classifies 2D points, trained live.
// Left: the points (editable) on the decision regions (colour = class, strength = confidence).
// Right: the network (hidden layers, units, activation), one thumbnail per hidden unit showing
// where *that* unit is active, the loss curve(s), and the training controls.
// A network with no hidden layer and two classes is a single neuron: its decision line can be
// dragged by hand (the round handle moves the line, the arrow tip turns it and sets |w|).
//
// <div data-demo="nnplayground" data-preset="xor2"></div>
//   data-preset   a preset id (see PRESETS)
//   data-train    "basic" (default: step / play / reset) or "full" (adds learning rate, batch size,
//                 validation split, weight decay, dropout: the backprop unit)
//   data-presets  "unit1" (default) or "unit2": which preset buttons are shown
import { autoMount, button, htmlEl, presets as presetRow, slider, svgEl, svgPoint, type Slider } from "../lib/ui";
import { pointPlayground, type Playground } from "../lib/playground";
import { dataset, blobs, type Dataset2D } from "../lib/datasets2d";
import { Plot, type Mark } from "../lib/plot";
import type { Act } from "../lib/nn";
import { Trainer, DEFAULT, isVal, setNeuron, type Config } from "./model";

const NAME = "nnplayground";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .np-left { flex: 0 1 470px; min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .np-left .qb-playground { max-width: none; }
[data-demo="${NAME}"] .np-right { flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .np-row { display: flex; flex-wrap: wrap; gap: .35em .9em; align-items: center; }
[data-demo="${NAME}"] .np-row .qb-slider { flex: 1 1 210px; }
[data-demo="${NAME}"] .np-thumbs { display: flex; flex-direction: column; gap: .2em; min-height: 40px; }
[data-demo="${NAME}"] .np-layer { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
[data-demo="${NAME}"] .np-layer span { font-size: 16px; width: 5.2em; }
[data-demo="${NAME}"] .np-layer canvas { width: 46px; height: 46px; border: 1px solid var(--qb-grid); image-rendering: pixelated; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
[data-demo="${NAME}"] .qb-note { font-size: 15px; }
[data-demo="${NAME}"] .np-loss { max-width: 560px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .np-left { flex-basis: 430px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// ---- data sets ----
interface DataSpec { id: string; label: string; make: (n: number | undefined, noise: number | undefined) => Dataset2D }
const DATA: DataSpec[] = [
  { id: "blobs", label: "two blobs", make: (n, noise) => dataset("blobs", { n, noise, seed: 1 }) },
  { id: "corners", label: "XOR", make: (n, noise) => blobs([[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]], { classes: [0, 0, 1, 1], sd: noise ?? 0.07, n: n ?? 100, seed: 5 }) },
  { id: "rings", label: "circle in ring", make: (n, noise) => dataset("rings", { n, noise, seed: 1 }) },
  { id: "moons", label: "two moons", make: (n, noise) => dataset("moons", { n, noise, seed: 1 }) },
  { id: "spirals", label: "spiral", make: (n, noise) => dataset("spirals", { n, noise, seed: 1 }) },
  { id: "blobs3", label: "three classes", make: (n, noise) => dataset("blobs3", { n, noise, seed: 1 }) },
];
const DEFAULT_NOISE: Record<string, number> = { blobs: 0.09, corners: 0.07, rings: 0.025, moons: 0.04, spirals: 0.015, blobs3: 0.08 };

interface Preset { id: string; label: string; data: string; n?: number; noise?: number; cfg: Partial<Config>; run?: number; set?: "unit1" | "unit2"; hand?: [number, number, number] }
const PRESETS: Preset[] = [
  { id: "hand", label: "one neuron, by hand", data: "blobs", cfg: { hidden: [] }, hand: [1, 0, 0], set: "unit1" },
  { id: "neuron", label: "one neuron, separable", data: "blobs", cfg: { hidden: [] }, set: "unit1" },
  { id: "xor0", label: "one neuron on XOR fails", data: "corners", cfg: { hidden: [] }, set: "unit1" },
  { id: "xor2", label: "2 hidden units solve XOR", data: "corners", cfg: { hidden: [2], act: "relu", seed: 1 }, set: "unit1" },
  { id: "linear", label: "no nonlinearity: still a line", data: "rings", cfg: { hidden: [8], act: "id" }, set: "unit1" },
  { id: "rings", label: "circle: 3 ReLU units", data: "rings", cfg: { hidden: [3], act: "relu" }, set: "unit1" },
  { id: "spiral", label: "spiral needs depth", data: "spirals", cfg: { hidden: [8, 8], act: "relu", lr: 0.3, seed: 1 }, set: "unit1" },
  { id: "three", label: "3 classes with softmax", data: "blobs3", cfg: { hidden: [] }, set: "unit1" },
  { id: "overfit", label: "overfitting: 8+8 units, 60 noisy points", data: "moons", n: 60, noise: 0.1, cfg: { hidden: [8, 8], lr: 0.1, valFrac: 0.5 }, set: "unit2" },
  { id: "decay", label: "weight decay smooths it", data: "moons", n: 60, noise: 0.1, cfg: { hidden: [8, 8], lr: 0.1, valFrac: 0.5, decay: 0.003 }, set: "unit2" },
  { id: "dropout", label: "dropout", data: "moons", n: 60, noise: 0.1, cfg: { hidden: [8, 8], lr: 0.1, valFrac: 0.5, dropout: 0.4 }, set: "unit2" },
  { id: "moredata", label: "more data cures it", data: "moons", n: 300, noise: 0.1, cfg: { hidden: [8, 8], lr: 0.1, valFrac: 0.5 }, set: "unit2" },
  { id: "minibatch", label: "mini-batches of 8", data: "moons", n: 200, noise: 0.1, cfg: { hidden: [8, 8], lr: 0.1, valFrac: 0.5, batch: 8 }, set: "unit2" },
  { id: "lrbig", label: "learning rate too large", data: "moons", n: 200, noise: 0.1, cfg: { hidden: [8, 8], lr: 1.5, valFrac: 0.5 }, set: "unit2" },
  { id: "dead", label: "dead network", data: "moons", n: 200, noise: 0.1, cfg: { hidden: [8, 8], lr: 5, valFrac: 0.5 }, set: "unit2" },
];

const BATCHES = [1, 2, 4, 8, 16, 32, 0];
const ACTS: { id: Act; label: string }[] = [{ id: "relu", label: "ReLU" }, { id: "tanh", label: "tanh" }, { id: "sigmoid", label: "sigmoid" }, { id: "id", label: "none" }];
const THUMB = 32;

function mount(host: HTMLElement): void {
  injectStyle();
  const full = host.dataset.train === "full";
  const set = host.dataset.presets === "unit2" || (host.dataset.presets === undefined && full) ? "unit2" : "unit1";
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.set === set)!;

  let st = { data: start.data, n: start.n as number | undefined, noise: start.noise ?? DEFAULT_NOISE[start.data], cfg: { ...DEFAULT, ...start.cfg } as Config };
  let trainer!: Trainer;
  let playing = 0;

  // ---- layout ----
  const left = htmlEl("div", "np-left", "", host);
  const right = htmlEl("div", "np-right", "", host);
  const ctlUnder = htmlEl("div", "np-row", "", left);
  const pg: Playground = pointPlayground(left, { data: DATA[0].make(undefined, undefined), presets: false, controlsParent: ctlUnder, size: 480, r: 7,
    onChange: (_pts, final) => { if (final && trainer) rebuild(); } });
  left.appendChild(ctlUnder);   // controls below the plot
  ctlUnder.querySelector(".qb-note")?.remove();

  // data row
  const dataRow = htmlEl("div", "np-row", "", right);
  const dataBtns = DATA.map(d => button(dataRow, d.label, () => { st.data = d.id; st.n = undefined; st.noise = DEFAULT_NOISE[d.id]; sNoise.set(st.noise); newData(); }));
  const nRow = htmlEl("div", "np-row", "", right);
  const sNoise = slider(nRow, "noise", 0, 0.2, 0.005, st.noise, v => { st.noise = v; newData(); });

  // architecture row
  const archRow = htmlEl("div", "np-row", "", right);
  const sLayers = slider(archRow, "hidden layers", 0, 2, 1, st.cfg.hidden.length, v => { const u = st.cfg.hidden[0] ?? sUnits.get(); st.cfg.hidden = Array(v).fill(u); rebuild(); });
  const sUnits = slider(archRow, "units per layer", 1, 8, 1, st.cfg.hidden[0] ?? 2, v => { st.cfg.hidden = st.cfg.hidden.map(() => v); rebuild(); });
  const actRow = htmlEl("div", "np-row", "", right);
  htmlEl("span", "", "activation φ:", actRow);
  const actBtns = ACTS.map(a => button(actRow, a.label, () => { st.cfg.act = a.id; rebuild(); }));

  // training (full mode)
  let sLr: Slider | null = null, sBatch: Slider | null = null, sVal: Slider | null = null, sDecay: Slider | null = null, sDrop: Slider | null = null;
  if (full) {
    const r1 = htmlEl("div", "np-row", "", right);
    sLr = slider(r1, "learning rate η", -2, 1, 0.05, Math.log10(st.cfg.lr), v => { st.cfg.lr = 10 ** v; trainer.opt.lr = st.cfg.lr; }, v => (10 ** v).toPrecision(2));
    sBatch = slider(r1, "batch size", 0, BATCHES.length - 1, 1, BATCHES.indexOf(st.cfg.batch), v => { st.cfg.batch = BATCHES[v]; rebuild(); }, v => (BATCHES[v] === 0 ? "all" : String(BATCHES[v])));
    const r2 = htmlEl("div", "np-row", "", right);
    sVal = slider(r2, "validation share", 0, 0.5, 0.1, st.cfg.valFrac, v => { st.cfg.valFrac = v; rebuild(); });
    sDecay = slider(r2, "weight decay λ", -4, -1, 0.25, st.cfg.decay > 0 ? Math.log10(st.cfg.decay) : -4, v => { st.cfg.decay = v <= -4 ? 0 : 10 ** v; trainer.cfg.decay = st.cfg.decay; }, v => (v <= -4 ? "0" : (10 ** v).toPrecision(2)));
    sDrop = slider(r2, "dropout", 0, 0.5, 0.1, st.cfg.dropout, v => { st.cfg.dropout = v; trainer.cfg.dropout = v; });
  }

  // thumbnails
  const thumbs = htmlEl("div", "np-thumbs", "", right);
  let canvases: HTMLCanvasElement[][] = [];

  // loss plot
  const lossDiv = htmlEl("div", "np-loss", "", right);
  const plot = new Plot(lossDiv, { width: 560, height: 190, x: [0, 100], y: [0.001, 3], ylog: true, xlabel: "training step", ylabel: "loss", margin: { l: 70, b: 52, t: 8 } });
  const trainLine: Mark<any> = plot.line([], [], { color: "var(--qb-c1)", width: 2.5, label: "training" });
  const valLine: Mark<any> = plot.line([], [], { color: "var(--qb-c2)", width: 2.5, label: "validation" });

  // buttons and readout
  const btnRow = htmlEl("div", "np-row", "", right);
  button(btnRow, "step", () => { stop(); trainer.train(1); draw(); });
  const playBtn = button(btnRow, "play", () => (playing ? stop() : play()));
  button(btnRow, "reset", () => { stop(); rebuild(); });
  button(btnRow, "new start", () => { stop(); st.cfg.seed++; rebuild(); });
  const readout = htmlEl("div", "qb-readout", "", right);
  const presetItems = PRESETS.filter(p => p.set === set);
  const presetBtns = presetRow(right, presetItems.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  // ---- the neuron's line (no hidden layer, two classes) ----
  const lineG = svgEl("g", {}, pg.top);
  const sx = pg.sx, sy = pg.sy, ARROW = 0.08;
  const neuron = () => {
    const W = trainer.net.W[0], b = trainer.net.b[0][0];
    return { w1: W[0], w2: W[1], b };
  };
  // the line in data coordinates: 2 w·(p − c) + b = 0, c = (0.5, 0.5); anchor = closest point to c
  function anchor(): [number, number, number, number] {
    const { w1, w2, b } = neuron(), n2 = w1 * w1 + w2 * w2 || 1e-9;
    return [0.5 - b * w1 / (2 * n2), 0.5 - b * w2 / (2 * n2), w1, w2];
  }
  const lineEl = svgEl("line", { "stroke-width": 3.5, stroke: "#333" }, lineG);
  const arrowEl = svgEl("line", { "stroke-width": 3.5, stroke: "#333", "marker-end": "" }, lineG);
  const anchorEl = svgEl("circle", { r: 11, class: "handle", stroke: "#333" }, lineG);
  const tipEl = svgEl("circle", { r: 9, fill: "#333", stroke: "#fff", "stroke-width": 2 }, lineG);
  const wLabel = svgEl("text", { "font-size": 18, class: "qb-halo" }, lineG);
  function drawLine(): void {
    const show = trainer.net.W.length === 1 && trainer.k === 2;
    lineG.style.display = show ? "" : "none";
    if (!show) return;
    const [ax, ay, w1, w2] = anchor(), nn = Math.hypot(w1, w2) || 1e-9, tx = -w2 / nn, ty = w1 / nn;
    lineEl.setAttribute("x1", String(sx(ax - 2 * tx))); lineEl.setAttribute("y1", String(sy(ay - 2 * ty)));
    lineEl.setAttribute("x2", String(sx(ax + 2 * tx))); lineEl.setAttribute("y2", String(sy(ay + 2 * ty)));
    const px = ax + ARROW * w1, py = ay + ARROW * w2;
    arrowEl.setAttribute("x1", String(sx(ax))); arrowEl.setAttribute("y1", String(sy(ay)));
    arrowEl.setAttribute("x2", String(sx(px))); arrowEl.setAttribute("y2", String(sy(py)));
    anchorEl.setAttribute("cx", String(sx(ax))); anchorEl.setAttribute("cy", String(sy(ay)));
    tipEl.setAttribute("cx", String(sx(px))); tipEl.setAttribute("cy", String(sy(py)));
    wLabel.setAttribute("x", String(sx(px) + 12)); wLabel.setAttribute("y", String(sy(py) - 8));
    wLabel.textContent = "w";
  }
  function dragHandle(el: SVGElement, which: "anchor" | "tip"): void {
    el.style.cursor = "grab"; el.style.touchAction = "none";
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ev.stopPropagation(); stop();
      el.setPointerCapture(ev.pointerId);
      const move = (e: PointerEvent) => {
        const [qx, qy] = svgPoint(pg.svg, e), x = sx.invert(qx), y = sy.invert(qy);
        let [ax, ay, w1, w2] = anchor();
        if (which === "anchor") { ax = x; ay = y; }
        else { w1 = (x - ax) / ARROW; w2 = (y - ay) / ARROW; if (Math.hypot(w1, w2) < 0.05) return; }
        setNeuron(trainer, w1, w2, -2 * (w1 * (ax - 0.5) + w2 * (ay - 0.5)));
        trainer.history = []; trainer.step = 0; trainer.record();
        draw();
      };
      const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
      el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
    });
  }
  dragHandle(anchorEl, "anchor"); dragHandle(tipEl, "tip");

  // validation markers
  const valG = svgEl("g", {}, pg.top);
  pg.top.insertBefore(valG, lineG);

  // ---- behaviour ----
  function currentData(): Dataset2D { return DATA.find(d => d.id === st.data)!.make(st.n, st.noise); }
  function newData(): void { stop(); pg.setPoints(currentData()); }

  function rebuild(): void {
    stop();
    const pts = pg.points;
    const k = Math.max(2, ...pts.map(p => p.c + 1), DATA.find(d => d.id === st.data)?.id === "blobs3" ? 3 : 2);
    trainer = new Trainer(pts, k, { ...st.cfg, hidden: [...st.cfg.hidden] });
    buildThumbs();
    syncControls();
    draw();
  }

  function buildThumbs(): void {
    thumbs.replaceChildren();
    canvases = trainer.cfg.hidden.map((u, l) => {
      const row = htmlEl("div", "np-layer", "", thumbs);
      htmlEl("span", "", `layer ${l + 1}`, row);
      return Array.from({ length: u }, () => { const c = htmlEl("canvas", "", "", row); c.width = THUMB; c.height = THUMB; return c; });
    });
    if (!canvases.length) htmlEl("div", "qb-note", trainer.k === 2 ? "No hidden layer: the network is one neuron. Drag the round handle to move its line, the dark tip to turn it (w)." : "No hidden layer: one neuron per class, softmax on top.", thumbs);
  }

  function drawThumbs(): void {
    if (!canvases.length) return;
    const vals = canvases.map(c => c.map(() => new Float32Array(THUMB * THUMB)));
    for (let j = 0; j < THUMB; j++) for (let i = 0; i < THUMB; i++) {
      const h = trainer.hiddenOutputs((i + 0.5) / THUMB, 1 - (j + 0.5) / THUMB);
      h.forEach((layer, l) => layer.forEach((v, u) => { vals[l][u][j * THUMB + i] = v; }));
    }
    canvases.forEach((row, l) => row.forEach((cv, u) => {
      const v = vals[l][u];
      let m = 1e-9;
      for (let q = 0; q < v.length; q++) m = Math.max(m, Math.abs(v[q]));
      const ctx = cv.getContext("2d")!, id = ctx.createImageData(THUMB, THUMB);
      for (let q = 0; q < v.length; q++) {
        const t = v[q] / m;   // -1..1
        const [r, g, b] = t >= 0 ? [255 - 25 * t, 255 - 135 * t, 255 - 230 * t] : [255 + 225 * t, 255 + 120 * t, 255 + 40 * t];
        id.data[4 * q] = r; id.data[4 * q + 1] = g; id.data[4 * q + 2] = b; id.data[4 * q + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
    }));
  }

  function drawValMarks(): void {
    valG.replaceChildren();
    if (trainer.cfg.valFrac <= 0) return;
    pg.points.forEach((p, i) => {
      if (isVal(i, trainer.cfg.valFrac)) svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 2.8, fill: "#fff" }, valG);
    });
  }

  function draw(): void {
    pg.setBackground((x, y) => trainer.probs(x, y));
    drawThumbs();
    drawLine();
    drawValMarks();
    const H = trainer.history, last = H[H.length - 1];
    const xs = H.map(h => h.step), cl = (v: number) => Math.max(0.001, Math.min(3, v));
    plot.setDomain([0, Math.max(100, trainer.step)], [0.001, 3]);
    trainLine.set(xs, H.map(h => cl(h.train)));
    valLine.set(trainer.Xv.length ? xs : [], trainer.Xv.length ? H.map(h => cl(h.val)) : []);
    let txt = `step ${trainer.step}` + (full ? `   epoch ${last.epoch.toFixed(1)}` : "") + `\ntraining loss ${last.train.toFixed(3)}, accuracy ${(100 * last.trainAcc).toFixed(0)} %`;
    if (trainer.Xv.length) {
      let best = 0; H.forEach((h, i) => { if (h.val < H[best].val) best = i; });
      txt += `\nvalidation loss ${last.val.toFixed(3)}, accuracy ${(100 * last.valAcc).toFixed(0)} %  (lowest ${H[best].val.toFixed(3)} at step ${H[best].step})`;
    }
    if (trainer.net.W.length === 1 && trainer.k === 2) { const { w1, w2, b } = neuron(); txt += `\nneuron: w = (${w1.toFixed(2)}, ${w2.toFixed(2)}), b = ${b.toFixed(2)}`; }
    readout.textContent = txt;
  }

  function syncControls(): void {
    sLayers.set(st.cfg.hidden.length);
    if (st.cfg.hidden.length) sUnits.set(st.cfg.hidden[0]);
    actBtns.forEach((b, i) => b.classList.toggle("active", ACTS[i].id === st.cfg.act));
    dataBtns.forEach((b, i) => b.classList.toggle("active", DATA[i].id === st.data));
    sNoise.set(st.noise);
    if (full) {
      sLr!.set(Math.log10(st.cfg.lr)); sBatch!.set(BATCHES.indexOf(st.cfg.batch)); sVal!.set(st.cfg.valFrac);
      sDecay!.set(st.cfg.decay > 0 ? Math.log10(st.cfg.decay) : -4); sDrop!.set(st.cfg.dropout);
    }
  }

  function play(): void {
    playBtn.textContent = "pause";
    const tick = () => {
      if (!host.isConnected) { playing = 0; return; }
      trainer.train(trainer.X.length > 150 && trainer.cfg.batch === 0 ? 4 : 8);
      draw();
      if (trainer.step >= 20000) { stop(); return; }
      playing = requestAnimationFrame(tick);
    };
    playing = requestAnimationFrame(tick);
  }
  function stop(): void { if (playing) cancelAnimationFrame(playing); playing = 0; playBtn.textContent = "play"; }

  function applyPreset(p: Preset): void {
    stop();
    st = { data: p.data, n: p.n, noise: p.noise ?? DEFAULT_NOISE[p.data], cfg: { ...DEFAULT, ...p.cfg, hidden: [...(p.cfg.hidden ?? DEFAULT.hidden)] } };
    pg.setPoints(currentData());   // → onChange → rebuild()
    if (!trainer) rebuild();
    if (p.hand) { setNeuron(trainer, ...p.hand); trainer.history = []; trainer.record(); draw(); }
    presetBtns.forEach((b, i) => b.classList.toggle("active", presetItems[i] === p));
  }

  applyPreset(start);
}

autoMount(NAME, mount);
