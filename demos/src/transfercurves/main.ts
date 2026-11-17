// transfercurves: four ways to use a pretrained network on a new task, trained live side by side.
// A small network (100 → 32 → 16 → 4) is pretrained on a source task with many labels (four
// generic shapes); the new task has three cell phenotypes and only a few labels. The four
// runs: frozen backbone + new head (a linear probe on the 16 features), last layers tuned,
// all layers tuned, and the same network trained from scratch. Plot: validation accuracy (or
// loss) against the training step; table: weights trained, accuracy, and how much of the old
// task the backbone still knows (catastrophic forgetting).
//
// <div data-demo="transfercurves" data-preset="few"></div>
//   data-preset  few | many | farfew | farmany | lowlr (see PRESETS)
import { autoMount, button, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, seriesColor, type Mark } from "../lib/plot";
import { drawScalarImage } from "../lib/colour";
import {
  makeSet, pretrained, Run, subset, STRATEGIES, STRATEGY_LABEL, oldTaskAccuracy, LOOK_A, LOOK_FAR,
  TARGET_CLASSES, P, LABELS, STEPS, type DataSet, type Look,
} from "./toy";

const NAME = "transfercurves";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .tc-left { flex: 1 1 520px; min-width: 0; max-width: 620px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; }
[data-demo="${NAME}"] table.tc-table { border-collapse: collapse; font-size: 14px; width: 100%; }
[data-demo="${NAME}"] table.tc-table td, [data-demo="${NAME}"] table.tc-table th { padding: 1px 4px; text-align: right; white-space: nowrap; }
[data-demo="${NAME}"] table.tc-table td:first-child, [data-demo="${NAME}"] table.tc-table th:first-child { text-align: left; }
[data-demo="${NAME}"] table.tc-table th { font-weight: 600; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .tc-swatch { display: inline-block; width: 14px; height: 4px; margin-right: 6px; vertical-align: middle; }
[data-demo="${NAME}"] .tc-patches { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
[data-demo="${NAME}"] .tc-patches figure { margin: 0; text-align: center; font-size: 14px; }
[data-demo="${NAME}"] .tc-patches canvas { width: 40px; height: 40px; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .tc-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .tc-left { flex-basis: 640px; max-width: 640px; }
.reveal [data-demo="${NAME}"] table.tc-table { font-size: 15px; }
.reveal [data-demo="${NAME}"] .tc-patches canvas { width: 34px; height: 34px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; n: number; far: boolean; lr: number }
const PRESETS: Preset[] = [
  { id: "few", label: "9 labels", n: 9, far: false, lr: 0.003 },
  { id: "many", label: "300 labels", n: 300, far: false, lr: 0.003 },
  { id: "lowlr", label: "9 labels, smaller rate", n: 9, far: false, lr: 0.0005 },
  { id: "farfew", label: "other modality, 9 labels", n: 9, far: true, lr: 0.003 },
  { id: "farmany", label: "other modality, 300 labels", n: 300, far: true, lr: 0.003 },
];
const COLOURS = [seriesColor(0), seriesColor(2), seriesColor(3), seriesColor(1)];
const WEIGHTS: Record<string, number> = {};

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let st = { n: start.n, far: start.far, lr: start.lr, showLoss: false };

  const left = htmlEl("div", "tc-left", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  const plot = new Plot(left, { width: 600, height: 300, x: [0, STEPS], y: [0.2, 1], xlabel: "training step", ylabel: "validation accuracy", margin: { t: 10, r: 16 } });
  const chance = plot.hline(1 / 3, { color: "var(--qb-muted)", width: 1.5, dash: "4 4", text: "guessing" });
  const lines: Mark<any>[] = STRATEGIES.map((s, i) => plot.line([], [], { color: COLOURS[i], width: 3, label: STRATEGY_LABEL[s] }));
  const table = htmlEl("table", "tc-table", "", left);
  const status = htmlEl("div", "qb-note", "", left);

  // ---- panel ----
  presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...st, n: p.n, far: p.far, lr: p.lr }; sN.set(LABELS.indexOf(p.n)); sLr.set(Math.log10(p.lr)); setDomainBtns(); reset(true); } })));
  const sN = slider(panel, "labelled images", 0, LABELS.length - 1, 1, LABELS.indexOf(st.n), v => { st.n = LABELS[v]; reset(false); }, v => String(LABELS[v]));
  const sLr = slider(panel, "learning rate", -4, -2, 0.05, Math.log10(st.lr), v => { st.lr = 10 ** v; reset(false); }, v => (10 ** v).toPrecision(2));
  const domRow = htmlEl("div", "tc-row", "", panel);
  htmlEl("span", "", "new images:", domRow);
  const bSame = button(domRow, "same microscope", () => { st.far = false; setDomainBtns(); reset(false); });
  const bFar = button(domRow, "other modality", () => { st.far = true; setDomainBtns(); reset(false); });
  function setDomainBtns(): void { bSame.classList.toggle("active", !st.far); bFar.classList.toggle("active", st.far); }
  setDomainBtns();
  const patches = htmlEl("div", "tc-patches", "", panel);
  const ctl = htmlEl("div", "tc-row", "", panel);
  const playBtn = button(ctl, "train", () => (playing ? stop() : play()));
  button(ctl, "reset", () => reset(false));
  const lossBtn = button(ctl, "show loss", () => { st.showLoss = !st.showLoss; lossBtn.classList.toggle("active", st.showLoss); setAxes(); draw(); });
  htmlEl("div", "qb-note", "Validation: 300 other labelled images of the new task. Old task: the backbone with its old head on 400 images of the four shapes.", panel);

  // ---- state ----
  let runs: Run[] = [];
  let val: DataSet | null = null, pool: DataSet | null = null;
  const srcTest = makeSet("source", 400, 202);
  let playing = 0;
  let poolLook: Look | null = null;

  function setAxes(): void {
    if (st.showLoss) plot.setDomain([0, STEPS], [0, 1.6]); else plot.setDomain([0, STEPS], [0.2, 1]);
    chance.set(st.showLoss ? Math.log(3) : 1 / 3);
    (plot as any).opts.ylabel = st.showLoss ? "validation loss" : "validation accuracy";
    plot.setDomain();
  }

  function drawPatches(look: Look): void {
    patches.innerHTML = "";
    const ex = makeSet("target", 6, 77, look);
    for (let c = 0; c < 3; c++) {
      const fig = htmlEl("figure", "", "", patches);
      const row = htmlEl("div", "", "", fig);
      for (const i of [c, c + 3]) {
        const cv = htmlEl("canvas", "", "", row);
        drawScalarImage(cv, ex.X[i], P, P, { lo: 0, hi: 1 });
      }
      htmlEl("figcaption", "", TARGET_CLASSES[c], fig);
    }
  }

  function reset(autoplay: boolean): void {
    stop();
    const look = st.far ? LOOK_FAR : LOOK_A;
    if (look !== poolLook) {
      pool = makeSet("target", 300, 303, look);
      val = makeSet("target", 300, 404, look);
      poolLook = look;
      drawPatches(look);
    }
    const train = subset(pool!, st.n);
    runs = STRATEGIES.map(s => new Run(s, train, val!, st.lr));
    for (const r of runs) {
      WEIGHTS[r.strategy] = r.net.W.reduce((a, W, l) => a + (r.opt.rates[l] > 0 ? W.length + r.net.b[l].length : 0), 0);
    }
    draw();
    if (autoplay) play();
  }

  function draw(): void {
    runs.forEach((r, i) => lines[i].set(r.curve.map(c => c.step), r.curve.map(c => (st.showLoss ? Math.min(1.6, c.valLoss) : c.valAcc))));
    const rows = runs.map((r, i) => {
      const c = r.last(), b = r.best();
      const old = r.steps === STEPS || !playing ? (100 * oldTaskAccuracy(r.net, srcTest)).toFixed(0) + " %" : "…";
      return `<tr><td><span class="tc-swatch" style="background:${COLOURS[i]}"></span>${STRATEGY_LABEL[r.strategy]}</td><td>${WEIGHTS[r.strategy]}</td><td>${c.trainLoss.toFixed(3)}</td><td>${(100 * c.valAcc).toFixed(1)} %</td><td>${(100 * b.valAcc).toFixed(1)} % (${b.step})</td><td>${r.strategy === "scratch" ? "–" : old}</td></tr>`;
    });
    table.innerHTML = `<tr><th>strategy</th><th>weights</th><th>train loss</th><th>val. acc.</th><th>best (step)</th><th>old task</th></tr>` + rows.join("");
    status.textContent = `step ${runs[0]?.steps ?? 0} of ${STEPS}; ${st.n} labelled images (${st.n / 3} per class), learning rate ${st.lr.toPrecision(2)}`;
  }

  function frame(): void {
    if (!playing) return;
    const t0 = performance.now();
    while (performance.now() - t0 < 25 && runs[0].steps < STEPS) for (const r of runs) r.run(1, 5);
    draw();
    if (runs[0].steps >= STEPS) { stop(); draw(); return; }
    playing = requestAnimationFrame(frame);
  }
  function play(): void { if (runs[0].steps >= STEPS) reset(false); playing = requestAnimationFrame(frame); playBtn.textContent = "pause"; }
  function stop(): void { if (playing) cancelAnimationFrame(playing); playing = 0; playBtn.textContent = "train"; }

  status.textContent = "pretraining the network on the old task (four shapes, 2400 labelled images)…";
  setTimeout(() => { pretrained(); setAxes(); reset(false); }, 30);
}

autoMount(NAME, mount);
