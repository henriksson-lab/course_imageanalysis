// optrace: an optimiser race on a loss surface over two weights (a, b). Plain gradient descent
// (red), momentum (orange) and Adam (purple) start from the same point and step together; each
// has its own learning rate. Surfaces: a long valley, a saddle, a curved valley, a round bowl.
// Gradient noise imitates mini-batches; a learning-rate schedule and a scale factor on the loss
// (the same problem in other units) show what Adam's per-weight step sizes buy.
//
// <div data-demo="optrace" data-preset="valley"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, type Mark } from "../lib/plot";
import { viridis } from "../lib/colour";
import { SURFACES, OPTS, OPT_LABEL, SCHEDULE_LABEL, Walker, lrAt, surface, DEFAULT_RACE, type OptId, type RaceOptions, type Schedule, type Surface } from "./model";

const NAME = "optrace";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .or-left { flex: 1 1 400px; min-width: 0; max-width: 450px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .or-map { cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 340px; max-width: 520px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .or-key { display: flex; flex-wrap: wrap; gap: .2em .8em; }
[data-demo="${NAME}"] .or-key span { font-weight: 600; }
[data-demo="${NAME}"] .or-ctl, [data-demo="${NAME}"] .or-out { display: flex; flex-direction: column; gap: .35em; min-width: 0; }
.reveal [data-demo="${NAME}"] { width: 100%; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-direction: row; max-width: none; flex: 1 1 auto; max-height: none; overflow: visible; }
.reveal [data-demo="${NAME}"] .or-ctl { flex: 0 0 340px; }
.reveal [data-demo="${NAME}"] .or-out { flex: 1 1 300px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .or-left { max-width: 420px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COLOR: Record<OptId, string> = { gd: "#d62728", momentum: "#ff7f0e", adam: "#9467bd" };
interface Preset { id: string; label: string; surf: string; eta: [number, number, number]; schedule?: Schedule; noise?: number; scale?: number; bias?: boolean; steps?: number }
const PRESETS: Preset[] = [
  { id: "valley", label: "long valley, each at its best η", surf: "valley", eta: [0.19, 0.03, 0.1] },
  { id: "same", label: "long valley, all η = 0.03", surf: "valley", eta: [0.03, 0.03, 0.03] },
  { id: "edge", label: "plain descent over the edge (η = 0.21)", surf: "valley", eta: [0.21, 0.03, 0.1] },
  { id: "scale", label: "the loss × 100", surf: "valley", eta: [0.19, 0.03, 0.1], scale: 100 },
  { id: "saddle", label: "saddle", surf: "saddle", eta: [0.03, 0.03, 0.03] },
  { id: "curved", label: "curved valley", surf: "curved", eta: [0.03, 0.01, 0.3] },
  { id: "noisy", label: "noisy gradients, constant η", surf: "bowl", eta: [0.1, 0.01, 0.1], noise: 1 },
  { id: "cosine", label: "noisy gradients, warm-up + cosine", surf: "bowl", eta: [0.1, 0.01, 0.1], noise: 1, schedule: "warmup" },
  { id: "nobias", label: "Adam without bias correction", surf: "valley", eta: [0.19, 0.03, 0.1], bias: false },
];
const NG = 110;
const fmtEta = (v: number) => { const e = 10 ** v; return e >= 0.1 ? e.toFixed(2) : e >= 0.01 ? e.toFixed(3) : e.toPrecision(2); };

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let surf: Surface = surface(start.surf);
  const eta: Record<OptId, number> = { gd: 0.19, momentum: 0.03, adam: 0.1 };
  const show: Record<OptId, boolean> = { gd: true, momentum: true, adam: true };
  let opts: RaceOptions = { ...DEFAULT_RACE };
  let from: [number, number] = surf.start;
  let walkers: Walker[] = [];
  let timer = 0;

  // ---- left: the map and the presets
  const left = htmlEl("div", "or-left", "", host);
  const mapDiv = htmlEl("div", "or-map", "", left);
  const plot = new Plot(mapDiv, { width: 500, height: 480, x: [-2, 2], y: [-2, 2], xlabel: "weight a", ylabel: "weight b", grid: false, margin: { l: 64, r: 10, t: 8, b: 56 } });
  let heat: Mark<any> | null = null;
  const minMark = plot.scatter([], [], { color: "#fff", r: 7, stroke: "#000" });
  const lines: Record<OptId, Mark<any>> = {} as any, dots: Record<OptId, Mark<any>> = {} as any, heads: Record<OptId, Mark<any>> = {} as any;
  for (const o of OPTS) {
    lines[o] = plot.line([], [], { color: COLOR[o], width: 2.5 });
    dots[o] = plot.scatter([], [], { color: COLOR[o], r: 2.5 });
    heads[o] = plot.scatter([], [], { color: COLOR[o], r: 7, stroke: "#fff" });
  }
  const startMark = plot.scatter([], [], { color: "#000", r: 6, stroke: "#fff" });
  const presetBtns = presetRow(left, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  // ---- right: controls, loss plot, readout
  const panelBox = htmlEl("div", "qb-panel", "", host);
  const panel = htmlEl("div", "or-ctl", "", panelBox);
  const outCol = htmlEl("div", "or-out", "", panelBox);
  const key = htmlEl("div", "or-key", "", panel);
  for (const o of OPTS) {
    const lab = htmlEl("label", "qb-check", "", key);
    const cb = htmlEl("input", "", "", lab); cb.type = "checkbox"; cb.checked = true;
    cb.addEventListener("change", () => { show[o] = cb.checked; cb.blur(); draw(); });
    const sp = htmlEl("span", "", OPT_LABEL[o], lab); sp.style.color = COLOR[o];
  }
  const surfRow = htmlEl("div", "qb-presets", "", panel);
  const surfBtns = SURFACES.map(S => button(surfRow, S.label, () => { setSurface(S, S.start); restart(); }));
  const sl: Record<OptId, ReturnType<typeof slider>> = {} as any;
  for (const o of OPTS) sl[o] = slider(panel, `η ${o === "gd" ? "plain" : o}`, -3, 0, 0.01, Math.log10(eta[o]), v => { eta[o] = 10 ** v; restart(); }, fmtEta);
  const schedRow = htmlEl("div", "qb-presets", "", panel);
  const schedBtns = (Object.keys(SCHEDULE_LABEL) as Schedule[]).map(s => button(schedRow, SCHEDULE_LABEL[s], () => { opts.schedule = s; restart(); }));
  const sNoise = slider(panel, "gradient noise", 0, 2, 0.05, 0, v => { opts.noise = v; restart(); }, v => v.toFixed(2));
  const sScale = slider(panel, "loss ×", 0, 2, 1, 0, v => { opts.scale = 10 ** v; restart(); }, v => String(10 ** v));
  const cBias = checkbox(panel, "Adam: bias correction", true, v => { opts.biasCorrection = v; restart(); });
  const runRow = htmlEl("div", "qb-presets", "", panel);
  button(runRow, "run 200 steps", () => run());
  button(runRow, "10 steps", () => { stop(); stepAll(10); draw(); });
  button(runRow, "restart", () => restart());
  const lossDiv = htmlEl("div", "", "", outCol);
  const lplot = new Plot(lossDiv, { width: 500, height: 150, x: [0, 200], y: [1e-4, 100], ylog: true, yticks: [1e-4, 1e-2, 1, 100], xlabel: "step", ylabel: "loss − min", margin: { l: 66, b: 46, t: 6 } });
  const llines: Record<OptId, Mark<any>> = {} as any;
  for (const o of OPTS) llines[o] = lplot.line([], [], { color: COLOR[o], width: 2.5 });
  const readout = htmlEl("div", "qb-readout", "", outCol);
  htmlEl("div", "qb-note", "Click the map to start all three there.", outCol);

  function setSurface(S: Surface, at: [number, number]): void {
    surf = S; from = at;
    const lo = S.min;
    const V = new Float64Array(NG * NG);
    for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) {
      const a = -2 + 4 * (i + 0.5) / NG, b = 2 - 4 * (j + 0.5) / NG;
      V[j * NG + i] = Math.floor(Math.log10(S.f(a, b) - lo + 0.01) * 4) / 4;
    }
    heat?.remove();
    heat = plot.heatmap(V, NG, NG, { map: (t: number) => viridis(0.15 + 0.85 * t), maxCells: 100 });
    plot.layer.insertBefore(heat.g, plot.layer.firstChild);
    minMark.set(S.minima.map(m => m[0]), S.minima.map(m => m[1]));
    surfBtns.forEach((b, i) => b.classList.toggle("active", SURFACES[i] === S));
  }

  function restart(): void {
    stop();
    walkers = OPTS.map(o => new Walker(surf, o, { ...opts, eta: eta[o] }, from));
    schedBtns.forEach((b, i) => b.classList.toggle("active", Object.keys(SCHEDULE_LABEL)[i] === opts.schedule));
    draw();
  }
  function stepAll(n: number): void { for (const w of walkers) w.run(n); }

  function draw(): void {
    startMark.set([from[0]], [from[1]]);
    walkers.forEach((w, k) => {
      const o = OPTS[k], on = show[o];
      const xs = on ? w.path.map(p => Math.max(-2.2, Math.min(2.2, p[0]))) : [], ys = on ? w.path.map(p => Math.max(-2.2, Math.min(2.2, p[1]))) : [];
      lines[o].set(xs, ys); dots[o].set(xs, ys);
      heads[o].set(on && !w.diverged ? [xs[xs.length - 1]] : [], on && !w.diverged ? [ys[ys.length - 1]] : []);
      llines[o].set(on ? w.losses.map((_, i) => i) : [], on ? w.losses.map(l => Math.min(100, Math.max(1e-4, l - surf.min))) : []);
    });
    lplot.setDomain([0, opts.steps], [1e-4, 100]);
    const lines2 = walkers.map((w, k) => {
      const o = OPTS[k], l = w.losses[w.losses.length - 1] - surf.min, s = w.stepsTo();
      const state = w.diverged ? "diverged" : `loss − min ${l < 1e-4 ? l.toExponential(1) : l.toFixed(4)}`;
      return `${OPT_LABEL[o]} (η ${eta[o].toPrecision(2)}): ${state}` + (s >= 0 && !w.diverged ? `\n   within 0.001 from step ${s}` : "");
    });
    const t = walkers[0]?.t ?? 0;
    lines2.push(`step ${t} of ${opts.steps}; learning rate now ×${(lrAt(opts.schedule, 1, Math.min(t, opts.steps - 1), opts.steps)).toFixed(3)} of η`);
    readout.textContent = lines2.join("\n");
  }

  function run(): void {
    stop();
    if ((walkers[0]?.t ?? 0) >= opts.steps) restart();
    const tick = () => {
      if (!host.isConnected) return;
      stepAll(2); draw();
      if (walkers.some(w => w.t < opts.steps && !w.diverged)) timer = window.setTimeout(tick, 30);
    };
    tick();
  }
  function stop(): void { clearTimeout(timer); }

  plot.svg.addEventListener("pointerdown", ev => {
    const [a, b] = plot.toData(ev);
    if (Math.abs(a) > 2 || Math.abs(b) > 2) return;
    from = [a, b]; restart();
  });

  function applyPreset(p: Preset): void {
    OPTS.forEach((o, k) => { eta[o] = p.eta[k]; sl[o].set(Math.log10(p.eta[k])); });
    opts = { ...DEFAULT_RACE, schedule: p.schedule ?? "constant", noise: p.noise ?? 0, scale: p.scale ?? 1, biasCorrection: p.bias ?? true, steps: p.steps ?? 200 };
    sNoise.set(opts.noise); sScale.set(Math.log10(opts.scale)); cBias.checked = opts.biasCorrection;
    setSurface(surface(p.surf), surface(p.surf).start);
    restart();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  applyPreset(start);
}

autoMount(NAME, mount);
