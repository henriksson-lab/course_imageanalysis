// Drifting (diffusion-models day, unit drifting/). Left: the real points (grey) and the generated
// points (blue), each with its drift V (a short line: where the next step moves it). Right: the
// mean |V| (log scale) and the coverage against the step, with the coverage of a new draw of as
// many real points for comparison. Two modes: free points (the points themselves are the
// synthetic data set; augmented views drawn faintly) and a generator (an MLP 2 → 32 → 32 → 2 that
// maps noise to a point in one pass, trained with the stop-gradient loss; the plot shows 400 of
// its samples). Sliders: τ, number of points, views per point and their jitter; a switch for the
// push away from the other generated points.
//
// <div data-demo="drifting" data-preset="drift|match|far|small|large|attract|few|views|gen|genattract"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { injectStyle, choiceRow, helperCss } from "../scorematch1d/dview";
import * as M from "./model";

const NAME = "drifting";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; align-items: flex-start; }
${S} .dr-main { flex: 0 1 420px; min-width: 260px; }
${S} .dr-side { flex: 0 1 240px; min-width: 200px; display: flex; flex-direction: column; gap: .2em; }
${S} .qb-panel { flex: 1 1 280px; min-width: 260px; max-width: 360px; gap: .25em; }
${S} .qb-readout { font-size: 15px; white-space: pre; line-height: 1.3; }
${S} .dr-off { opacity: .4; pointer-events: none; }
.reveal ${S} { font-size: 17px; gap: .5em; flex-wrap: nowrap; }
.reveal ${S} .dr-main { flex: 0 0 450px; }
.reveal ${S} .dr-side { flex: 0 0 250px; }
.reveal ${S} .qb-panel { flex: 1 1 420px; max-width: 460px; gap: .12em; }
.reveal ${S} .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal ${S} .qb-readout { font-size: 15px; line-height: 1.2; padding: .15em .4em; }
.reveal ${S} .qb-note { display: none; }
`;

const tauOf = (u: number) => 0.02 * 250 ** (u / 100);                // 0.02 … 5, log scale
const uOfTau = (t: number) => 100 * Math.log(t / 0.02) / Math.log(250);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let c: M.Config = M.presetConfig(p0.id);
  let drift: M.Drift | null = null, gen: M.Generator | null = null;
  let genCov: number[] = [], refCov = 0, anim = 0;
  const real = () => M.makeData(c.data);

  const main = htmlEl("div", "dr-main", "", host);
  const plot = new Plot(main, { width: 500, height: 500, x: [-2.5, 2.5], y: [-2.5, 2.5], equal: true, xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2], ariaLabel: "real and generated points" });
  const realMark = plot.scatter([], [], { color: "#b8b8b8", r: 3.2, stroke: "none" });
  const viewMark = plot.scatter([], [], { color: seriesColor(0), r: 2.2, opacity: 0.3, stroke: "none" });
  const vMark = plot.line([], [], { color: "#e08a1e", width: 2 });
  const genMark = plot.scatter([], [], { color: seriesColor(0), r: 4, stroke: "none" });

  const side = htmlEl("div", "dr-side", "", host);
  const pN = new Plot(side, { width: 300, height: 190, x: [0, M.POINT_STEPS], y: [1e-4, 3], ylog: true, title: "mean |V|", xticks: 3, margin: { l: 62 } });
  const nLine = pN.line([], [], { color: "#e08a1e" });
  const pC = new Plot(side, { width: 300, height: 210, x: [0, M.POINT_STEPS], y: [0, 1], title: "coverage", xlabel: "step", xticks: 3, yticks: [0, 0.5, 1], margin: { l: 62 } });
  const refLine = pC.hline(0, { color: "var(--qb-muted)", dash: "5 4", width: 2 });
  const cLine = pC.line([], [], { color: seriesColor(0) });
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p.id) })));
  const modeRow = choiceRow<M.Config["mode"]>(panel, "", [{ v: "points", label: "free points" }, { v: "generator", label: "generator" }], c.mode, v => { c.mode = v; restart(); });
  const dataRow = choiceRow<M.DataId>(panel, "data", M.DATA_IDS.map(d => ({ v: d, label: M.DATA_LABEL[d] })), c.data, v => { c.data = v; restart(); });
  const initRow = choiceRow<M.InitId>(panel, "start", M.INIT_IDS.map(d => ({ v: d, label: M.INIT_LABEL[d] })), c.init, v => { c.init = v; restart(); });
  const tauS = slider(panel, "temperature τ", 0, 100, 1, uOfTau(c.tau), u => { c.tau = tauOf(u); restart(); }, u => tauOf(u).toPrecision(2));
  const nS = slider(panel, "points", 5, 300, 5, c.n, v => { c.n = v; restart(); });
  const vS = slider(panel, "views per point", 1, 16, 1, c.views, v => { c.views = v; restart(); });
  const jS = slider(panel, "jitter of the views", 0, 0.4, 0.01, c.jitter, v => { c.jitter = v; restart(); });
  const repelBox = checkbox(panel, "push away from the other generated points (V⁻)", c.repel > 0, v => { c.repel = v ? 1 : 0; restart(); });
  const runRow = htmlEl("div", "qb-presets", "", panel);
  button(runRow, "1 step", () => { stop(); advance(1); draw(); });
  button(runRow, "run", () => play());
  button(runRow, "pause", () => stop());
  button(runRow, "restart", () => restart());
  button(runRow, "reset", () => applyPreset(p0.id, false));
  htmlEl("div", "qb-note", "Grey: the real points. Blue: the generated points; the orange line on each is its drift V (the next step moves it half of that). Faint blue: the augmented views. Coverage: the fraction of real points whose 5-nearest-neighbour ball holds a generated point; dashed: a new draw of as many real points.", panel);

  function applyPreset(id: string, run = true): void {
    c = M.presetConfig(id);
    tauS.set(uOfTau(c.tau)); nS.set(c.n); vS.set(c.views); jS.set(c.jitter); repelBox.checked = c.repel > 0;
    modeRow.set(c.mode); dataRow.set(c.data); initRow.set(c.init);
    pbtns.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i].id === id));
    restart();
    if (run) play();
  }

  function restart(): void {
    stop();
    const pts = c.mode === "points";
    [initRow.row, nS.input.parentElement!, vS.input.parentElement!, jS.input.parentElement!].forEach(e => (e as HTMLElement).classList.toggle("dr-off", !pts));
    (jS.input.parentElement as HTMLElement).classList.toggle("dr-off", !pts || c.views === 1);
    const R = real();
    if (pts) {
      drift = M.newDrift(c); gen = null;
      refCov = M.coverage(R, M.makeData(c.data, c.n, 77));
      pN.setDomain([0, M.POINT_STEPS], [1e-4, 3]); pC.setDomain([0, M.POINT_STEPS], [0, 1]);
      pN.setLabels({ title: "mean |V|" });
    } else {
      gen = M.newGenerator(c); drift = null;
      genCov = [gen.coverageNow()];
      refCov = M.coverage(R, M.makeData(c.data, 400, 77));
      pN.setDomain([0, M.GEN_STEPS], [1e-5, 1]); pC.setDomain([0, M.GEN_STEPS], [0, 1]);
      pN.setLabels({ title: "loss = mean |V|² / 2" });
    }
    refLine.set(refCov);
    realMark.set(xs(R), ys(R));
    draw();
  }

  const xs = (P: M.Pts) => Array.from({ length: P.length / 2 }, (_, i) => P[2 * i]);
  const ys = (P: M.Pts) => Array.from({ length: P.length / 2 }, (_, i) => P[2 * i + 1]);

  function advance(k: number): void {
    for (let s = 0; s < k; s++) {
      if (drift && drift.step < M.POINT_STEPS) drift.advance();
      if (gen && gen.step < M.GEN_STEPS) { gen.train(); if (gen.step % 50 === 0) genCov.push(gen.coverageNow()); }
    }
  }

  function draw(): void {
    if (drift) {
      const X = drift.x, V = drift.lastV, n = X.length / 2;
      genMark.set(xs(X), ys(X));
      const lx: number[] = [], ly: number[] = [];
      for (let i = 0; i < n; i++) lx.push(X[2 * i], X[2 * i] + V[2 * i], NaN), ly.push(X[2 * i + 1], X[2 * i + 1] + V[2 * i + 1], NaN);
      vMark.set(lx, ly);
      if (c.views > 1) { const W = drift.lastViews; viewMark.set(xs(W), ys(W)); } else viewMark.set([], []);
      const steps = drift.normHist.map((_, i) => i);
      nLine.set(steps, drift.normHist.map(v => Math.max(v, 1e-4)));
      cLine.set(steps, drift.covHist);
      const s = drift.step;
      ro.textContent = `step ${s} / ${M.POINT_STEPS}\nmean |V| ${drift.normHist[s].toFixed(3)}\ncoverage ${drift.covHist[s].toFixed(3)}\nnew draw of ${c.n} real: ${refCov.toFixed(3)}`;
    } else if (gen) {
      const X = gen.sample();
      genMark.set(xs(X), ys(X)); vMark.set([], []); viewMark.set([], []);
      const L = gen.lossHist;
      // a running mean over 20 steps keeps the batch noise readable
      const sm = L.map((_, i) => { let a = 0, k = 0; for (let j = Math.max(0, i - 19); j <= i; j++) { a += L[j]; k++; } return Math.max(a / k, 1e-5); });
      nLine.set(L.map((_, i) => i + 1), sm);
      cLine.set(genCov.map((_, i) => 50 * i), genCov);
      ro.textContent = `step ${gen.step} / ${M.GEN_STEPS}\nloss ${(sm[sm.length - 1] ?? 0).toFixed(4)}\ncoverage ${genCov[genCov.length - 1].toFixed(3)} (400 samples)\nnew draw of 400 real: ${refCov.toFixed(3)}`;
    }
  }

  function play(): void {
    stop();
    const tick = () => {
      advance(drift ? 1 : 10);
      draw();
      const done = drift ? drift.step >= M.POINT_STEPS : gen ? gen.step >= M.GEN_STEPS : true;
      anim = done ? 0 : requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }
  function stop(): void { if (anim) cancelAnimationFrame(anim); anim = 0; }

  applyPreset(p0.id, false);   // shown at step 0; "run" or a preset starts it
}

autoMount(NAME, mount);
