// A GAN in one dimension, trained live: a line generator x = a·z + b against a discriminator
// D(x) = σ(c0 + c1·x + c2·x²). Top: the data density (grey) and the generator's density N(b, a²)
// (blue). Below: the discriminator D(x) (red; dashed: the best one, p/(p + q)). Right: the path
// of the generator's parameters (b, a) towards the data's (star), and the losses.
//
// <div data-demo="gan1d" data-preset="converge|optimal|minimax|fast"></div>
// <div data-demo="gan1d" data-mode="training" data-preset="slow|fast|drop|between|balanced"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";
import { Gan1d, dataPdf, optimalD, precisionRecall1d, DATA, type DataId, type Loss, type Options } from "./model";

const NAME = "gan1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const P = `${S}:not(.reveal *)`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .g1-main { flex: 0 1 470px; min-width: 260px; display: flex; flex-direction: column; }
${S} .g1-side { flex: 0 1 270px; min-width: 220px; display: flex; flex-direction: column; }
${S} .qb-panel { flex: 1 1 280px; gap: .35em; max-width: 360px; }
${S} .qb-readout { font-size: 14px; }
@media (min-width: 641px) {
  ${P} { flex-wrap: nowrap; }
  ${P} .g1-main { flex: 0 1 min(470px, calc(100vh - 230px)); min-width: 0; }
  ${P} .g1-side { flex: 0 1 250px; min-width: 0; }
  ${P} .qb-panel { flex: 1 1 300px; min-width: 0; }
  ${P} .qb-note { font-size: 13px; }
}
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .g1-main { flex: 0 0 440px; }
.reveal ${S} .g1-side { flex: 0 0 280px; }
.reveal ${S} .qb-panel { flex: 1 1 380px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 17px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State extends Options { data: DataId; optimal: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { data: "one", loss: "ns", lrG: 0.03, lrD: 0.5, kD: 5, batch: 128, optimal: false };
const GAN_PRESETS: Preset[] = [
  { id: "converge", label: "the game settles", s: {} },
  { id: "optimal", label: "always the best discriminator", s: { optimal: true } },
  { id: "minimax", label: "original loss: a slow start", s: { loss: "mm" } },
  { id: "fast", label: "generator too fast", s: { lrG: 0.3, kD: 1 } },
];
const TRAIN_PRESETS: Preset[] = [
  { id: "slow", label: "discriminator too slow: collapse", s: { lrD: 0.05, kD: 1 } },
  { id: "fast", label: "generator too fast: overshoot", s: { lrG: 0.3, kD: 1 } },
  { id: "drop", label: "two bumps, best discriminator: one dropped", s: { data: "two", optimal: true } },
  { id: "between", label: "two bumps, learned discriminator: in between", s: { data: "two" } },
  { id: "balanced", label: "balanced: it settles", s: {} },
];
const MAX_STEPS = 2000;
const XR: [number, number] = [-3, 4];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const slide = !!host.closest(".reveal"), F = slide ? 22 : 17;
  const training = host.dataset.mode === "training";
  const PS = training ? TRAIN_PRESETS : GAN_PRESETS;
  const p0 = PS.find(p => p.id === host.dataset.preset) ?? PS[0];
  const st: State = { ...BASE, ...p0.s };

  const main = htmlEl("div", "g1-main", "", host);
  const dp = new Plot(main, { width: 470, height: 230, x: XR, y: [0, 1.5], xticks: [-3, -2, -1, 0, 1, 2, 3, 4], yticks: [0, 0.5, 1, 1.5], ylabel: "density", font: F, margin: { b: 1.4 * F } });
  const fx = Array.from({ length: 281 }, (_, i) => XR[0] + (XR[1] - XR[0]) * i / 280);
  const dataBand = dp.band(fx, fx.map(() => 0), fx.map(() => 0), { color: "#888", fillOpacity: 0.35 });
  const genLine = dp.line([], [], { color: seriesColor(0), width: 3.5 });
  dp.legend([{ label: "data", color: "#888", kind: "box" }, { label: "generator", color: seriesColor(0) }], "tl");
  const cp = new Plot(main, { width: 470, height: 170, x: XR, y: [0, 1], xticks: [-3, -2, -1, 0, 1, 2, 3, 4], yticks: [0, 0.5, 1], xlabel: "x", ylabel: "D(x)", font: F });
  cp.hline(0.5, { color: "var(--qb-muted)", dash: "4 4", width: 1.5 });
  const dOpt = cp.line([], [], { color: seriesColor(1), width: 2, dash: "7 5" });
  const dCur = cp.line([], [], { color: seriesColor(1), width: 3.5 });

  const side = htmlEl("div", "g1-side", "", host);
  const pp = new Plot(side, { width: 270, height: 240, x: [-1.5, 1.5], y: [0, 1.5], xlabel: "b (centre)", ylabel: "a (spread)", xticks: [-1, 0, 1], yticks: [0, 0.5, 1, 1.5], font: F });
  const path = pp.line([], [], { color: seriesColor(0), width: 2 });
  const now = pp.scatter([], [], { color: seriesColor(0), r: 6 });
  const target = pp.scatter([], [], { color: "#333", r: 7 });
  let lp: Plot | null = null, bars: ReturnType<Plot["bars"]> | null = null;
  if (training) {
    lp = new Plot(side, { width: 270, height: 170, x: [-0.6, 1.6], y: [0, 1], xticks: [], yticks: [0, 0.5, 1], font: F, margin: { b: 1.6 * F } });
    bars = lp.bars([0, 1], [0, 0], { barWidth: 0.6, colors: [seriesColor(2), seriesColor(4)] });
    lp.text(0, -0.14, "precision", { anchor: "middle" }); lp.text(1, -0.14, "recall", { anchor: "middle" });
  } else {
    lp = new Plot(side, { width: 270, height: 170, x: [0, MAX_STEPS], y: [0, 3], xlabel: "step", ylabel: "loss", xticks: [0, 1000, 2000], yticks: [0, 1, 2, 3], font: F, margin: { r: 1.6 * F } });
    lp.hline(2 * Math.LN2, { color: "var(--qb-muted)", dash: "4 4", width: 1.5 });
  }
  const lD = training ? null : lp.line([], [], { color: seriesColor(1), width: 2 });
  const lG = training ? null : lp.line([], [], { color: seriesColor(0), width: 2 });
  if (!training) lp.legend([{ label: "D", color: seriesColor(1) }, { label: "G", color: seriesColor(0) }], "tr");

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const dataC = choiceRow(panel, "data", [{ v: "one" as DataId, label: "one bump" }, { v: "two" as DataId, label: "two bumps" }], st.data, v => { st.data = v; restart(); });
  const lossC = choiceRow(panel, "generator loss", [{ v: "ns" as Loss, label: "non-saturating" }, { v: "mm" as Loss, label: "minimax" }], st.loss, v => { st.loss = v; restart(); });
  const lg10 = (v: number) => Math.log10(v), p10 = (u: number) => +(10 ** u).toPrecision(2);
  const sG = slider(panel, "learning rate G", -2.5, 0, 0.05, lg10(st.lrG), u => { st.lrG = p10(u); restart(); }, u => String(p10(u)));
  const sD = slider(panel, "learning rate D", -2, 0, 0.05, lg10(st.lrD), u => { st.lrD = p10(u); restart(); }, u => String(p10(u)));
  const sK = slider(panel, "D steps per G step", 1, 10, 1, st.kD, v => { st.kD = v; restart(); });
  const optC = checkbox(panel, "always the best discriminator", st.optimal, v => { st.optimal = v; restart(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const runB = button(row, "run", () => toggle());
  button(row, "10 steps", () => { stop(); for (let i = 0; i < 10; i++) step(); draw(); });
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Generator x = a·z + b with z ~ N(0, 1); discriminator D(x) = σ(c0 + c1x + c2x²). Batches of 128; plain gradient descent. Changing a setting restarts from a = 1, b = −1.", panel);

  let g = new Gan1d(st.data, st, 1), timer = 0;

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    dataC.set(st.data); lossC.set(st.loss); sG.set(lg10(st.lrG)); sD.set(lg10(st.lrD)); sK.set(st.kD); optC.checked = st.optimal;
    pbtns.forEach((b, i) => b.classList.toggle("active", PS[i] === p));
    restart();
  }
  function restart(): void {
    stop();
    g = new Gan1d(st.data, st, 1);
    g.optimal = st.optimal;
    const d = DATA[st.data];
    target.set(d.mu.length === 1 ? [d.mu[0]] : [], d.mu.length === 1 ? [d.sd[0]] : []);
    dataBand.set(fx, fx.map(() => 0), fx.map(x => dataPdf(st.data, x)));
    draw();
  }
  function step(): void { if (g.step < MAX_STEPS && Number.isFinite(g.a)) g.train(); }
  function stop(): void { if (timer) cancelAnimationFrame(timer); timer = 0; runB.textContent = "run"; }
  function toggle(): void {
    if (timer) { stop(); return; }
    if (g.step >= MAX_STEPS) restart();
    runB.textContent = "pause";
    const tick = () => {
      for (let i = 0; i < 4; i++) step();
      draw();
      if (g.step >= MAX_STEPS) { stop(); return; }
      timer = requestAnimationFrame(tick);
    };
    timer = requestAnimationFrame(tick);
  }
  function draw(): void {
    genLine.set(fx, fx.map(x => Math.min(1.6, g.genPdf(x))));
    dCur.set(fx, fx.map(x => g.disc(x)));
    dOpt.set(fx, fx.map(x => optimalD(dataPdf(st.data, x), g.genPdf(x))));
    const h = g.history;
    path.set([-1, ...h.map(q => q.b)], [1, ...h.map(q => Math.abs(q.a))]);
    now.set([g.b], [Math.abs(g.a)]);
    const last = h[h.length - 1];
    let extra = "";
    if (training && bars) {
      const pr = precisionRecall1d(g);
      bars.set([0, 1], [pr.precision, pr.recall]);
      extra = `\nprecision ${fmt(pr.precision, 2)}  recall ${fmt(pr.recall, 2)}`;
    } else if (lD && lG) {
      const every = Math.max(1, Math.floor(h.length / 400)), hs = h.filter((_, i) => i % every === 0);
      lD.set(hs.map(q => q.step), hs.map(q => q.lossD));
      lG.set(hs.map(q => q.step), hs.map(q => Math.min(3, q.lossG)));
    }
    ro.textContent = `step ${g.step}   a ${fmt(Math.abs(g.a), 2)}   b ${fmt(g.b, 2)}` +
      (st.data === "one" ? `   (data: a 0.50, b 1.00)` : "") +
      `\nD loss ${last && Number.isFinite(last.lossD) ? fmt(last.lossD, 2) : "–"}   G loss ${last ? fmt(last.lossG, 2) : "–"}` + extra;
  }

  apply(p0);
}

autoMount(NAME, mount);
