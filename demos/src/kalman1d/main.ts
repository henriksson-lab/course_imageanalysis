// The Kalman filter in 1D, step by step. An object moves along a line (constant velocity, an
// optional turn at frame 20); every frame gives a noisy measurement (seeded). Predict: the
// estimate moves with the model (random walk or constant velocity) and its uncertainty grows
// (process noise Q). Update: a weighted average of prediction and measurement (variance R)
// with gain K = P⁻/(P⁻ + R). Top: the current step as Gaussians (prediction, measurement,
// estimate) and the gain as a point between them. Bottom: position vs time with the truth, the
// measurements, the estimate and its ±2σ band; right: K over time and its steady state.
// "Skip measurement" turns a frame into a missed detection (predict only).
//
// <div data-demo="kalman1d" data-preset="meas|model|rwlag|cv|missed|golden"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import * as KF from "../lib/kalman";
import { T, Model, State, DEFAULT, PRESETS, C_PRED, C_MEAS, C_EST, C_TRUE, truth, StepRec, run } from "./model";

const NAME = "kalman1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .k1-plots { flex: 1 1 560px; max-width: 680px; display: flex; flex-direction: column; gap: .1em; min-width: 280px; }
${S} .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .3em; }
${S} .k1-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 18px; white-space: pre-wrap; }
${S} .qb-presets { flex-basis: 100%; }
${S} .qb-btn.k1-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
.reveal ${S} { font-size: 19px; }
.reveal ${S} .k1-plots { flex: 0 0 600px; }
.reveal ${S} .qb-panel { flex: 0 0 470px; max-width: 480px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const gaussPdf = (x: number, m: number, v: number) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);

autoMount(NAME, host => {
  injectStyle();
  let st: State = { ...DEFAULT };
  const plots = htmlEl("div", "k1-plots", "", host);
  const gp = new Plot(plots, { width: 640, height: 190, x: [0, 10], y: [0, 1], xlabel: "position", ylabel: "density", font: 22, yticks: [], margin: { l: 40 } });
  const tp = new Plot(plots, { width: 640, height: 235, x: [0, T - 1], y: [0, 26], xlabel: "frame", ylabel: "position", font: 22 });
  const panel = htmlEl("div", "qb-panel", "", host);
  const read = htmlEl("div", "qb-readout", "", panel);
  const kp = new Plot(panel, { width: 420, height: 170, x: [0, T - 1], y: [0, 1], xlabel: "frame", ylabel: "gain K", font: 22, yticks: [0, 0.5, 1] });
  const stepRow = htmlEl("div", "k1-row", "", panel);
  button(stepRow, "predict", () => { if (st.phase === 1 && st.k < T - 1) { st.k++; st.phase = 0; } else if (st.phase === 0) { /* already predicted */ } draw(); });
  button(stepRow, "update", () => { st.phase = 1; draw(); });
  button(stepRow, "step", () => { advance(); draw(); });
  let timer = 0;
  const playBtn = button(stepRow, "play", () => {
    if (timer) { stop(); return; }
    if (st.k >= T - 1 && st.phase === 1) { st.k = 1; st.phase = 0; }
    playBtn.textContent = "pause";
    timer = window.setInterval(() => { advance(); draw(); if (st.k >= T - 1 && st.phase === 1) stop(); }, 350);
  });
  const stop = () => { window.clearInterval(timer); timer = 0; playBtn.textContent = "play"; };
  button(stepRow, "skip measurement", () => {
    const i = st.missed.indexOf(st.k);
    if (i >= 0) st.missed.splice(i, 1); else st.missed.push(st.k);
    recompute();
  });
  const modelRow = htmlEl("div", "k1-row", "", panel);
  const mBtns: Record<Model, HTMLButtonElement> = {
    rw: button(modelRow, "random walk", () => { st.model = "rw"; recompute(); }),
    cv: button(modelRow, "constant velocity", () => { st.model = "cv"; recompute(); }),
  };
  const turnC = checkbox(modelRow, "turn at frame 20", st.turn, v => { st.turn = v; recompute(); });
  const rS = slider(panel, "R (meas. var.)", -1.5, 1.3, 0.05, Math.log10(st.R), v => { st.R = +(10 ** v).toPrecision(2); recompute(); }, v => (+(10 ** v).toPrecision(2)).toString());
  const qS = slider(panel, "Q (process var.)", -3, 0.6, 0.05, Math.log10(st.Q), v => { st.Q = +(10 ** v).toPrecision(2); recompute(); }, v => (+(10 ** v).toPrecision(2)).toString());
  presets(host, [...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s) })), { label: "reset", apply: () => apply({}) }]);

  let rec: StepRec[] = run(st);
  function advance(): void {
    if (st.phase === 0) st.phase = 1;
    else if (st.k < T - 1) { st.k++; st.phase = 0; }
  }
  function apply(p: Partial<State>): void {
    stop();
    st = { ...DEFAULT, ...p, missed: [...(p.missed ?? [])] };
    rS.set(Math.log10(st.R)); qS.set(Math.log10(st.Q)); turnC.checked = st.turn;
    recompute();
  }
  function recompute(): void { rec = run(st); draw(); }

  function draw(): void {
    const k = st.k, r = rec[k], upd = st.phase === 1 && !r.missed;
    mBtns.rw.classList.toggle("k1-on", st.model === "rw"); mBtns.cv.classList.toggle("k1-on", st.model === "cv");
    // Gaussians
    const sdMax = Math.sqrt(Math.max(r.pPred, st.R));
    const lo = Math.min(r.xPred, r.z) - 3.2 * sdMax, hi = Math.max(r.xPred, r.z) + 3.2 * sdMax;
    const peak = Math.max(gaussPdf(0, 0, r.pPred), r.missed ? 0 : gaussPdf(0, 0, st.R), upd ? gaussPdf(0, 0, r.p) : 0);
    gp.clear();
    gp.setDomain([lo, hi], [0, peak * 1.38]);
    gp.fn(x => gaussPdf(x, r.xPred, r.pPred), { color: C_PRED, width: 3, label: "prediction" });
    if (!r.missed) gp.fn(x => gaussPdf(x, r.z, st.R), { color: C_MEAS, width: 3, label: "measurement" });
    if (upd) {
      gp.fn(x => gaussPdf(x, r.x, r.p), { color: C_EST, width: 3.5, label: "estimate" });
      const yb = peak * 1.05;
      gp.line([r.xPred, r.z], [yb, yb], { color: "#888", width: 2 });
      gp.scatter([r.x], [yb], { color: C_EST, r: 7 });
      gp.text(Math.min(r.xPred, r.z), yb + peak * 0.1, `K = ${r.K.toFixed(2)}`, { anchor: "start", size: 22 });
    }
    gp.vline(truth(k, st.turn), { color: C_TRUE, dash: "4 4", width: 2, label: "truth" });
    gp.legend(undefined, "tr");
    // time plot
    const upto = st.phase === 1 ? k : k - 1;
    const ts = Array.from({ length: upto + 1 }, (_, t) => t);
    tp.clear();
    const allY = rec.flatMap((q, t) => [q.z, truth(t, st.turn)]);
    tp.setDomain([0, T - 1], [Math.floor(Math.min(...allY) - 2), Math.ceil(Math.max(...allY) + 2)]);
    const tt = Array.from({ length: k + 1 }, (_, t) => t);
    tp.line(tt, tt.map(t => truth(t, st.turn)), { color: C_TRUE, width: 2, dash: "6 4", label: "truth" });
    tp.band(ts, ts.map(t => rec[t].x - 2 * Math.sqrt(rec[t].p)), ts.map(t => rec[t].x + 2 * Math.sqrt(rec[t].p)), { color: C_EST, fill: C_EST, fillOpacity: 0.13 });
    const mt = Array.from({ length: k + 1 }, (_, t) => t).filter(t => !rec[t].missed);
    tp.scatter(mt, mt.map(t => rec[t].z), { color: C_MEAS, r: 4.5, label: "measurement" });
    tp.line(ts, ts.map(t => rec[t].x), { color: C_EST, width: 3, label: "estimate ± 2σ" });
    if (st.phase === 0) tp.scatter([k], [r.xPred], { color: C_PRED, r: 7 });
    for (const t of st.missed) if (t <= k) tp.text(t, Math.min(...allY) - 1, "×", { anchor: "middle", size: 22 });
    tp.vline(k, { color: "#bbb", width: 1.5 });
    tp.legend(undefined, "tl");
    // gain plot
    kp.clear();
    const ks = Array.from({ length: k }, (_, t) => t + 1).filter(t => (t < k || st.phase === 1) && !rec[t].missed);
    let kInf = NaN;
    if (st.model === "rw") kInf = KF.steadyRW(st.Q, st.R).K;
    else { const w = run({ ...st, missed: [] }); kInf = w[T - 1].K; }
    kp.hline(kInf, { color: "#888", dash: "5 4", width: 2 });
    kp.line(ks, ks.map(t => rec[t].K), { color: C_EST, width: 2.5 });
    kp.scatter(ks, ks.map(t => rec[t].K), { color: C_EST, r: 3 });
    // readout
    const lines = [
      `frame ${k}: ${st.phase === 0 ? "after predict" : r.missed ? "missed: no update" : "after update"}`,
      `prediction ${r.xPred.toFixed(2)},  P⁻ = ${r.pPred.toFixed(2)}`,
      r.missed ? "measurement: none" : `measurement ${r.z.toFixed(2)},  R = ${st.R}`,
    ];
    if (upd) lines.push(`K = ${r.K.toFixed(3)}  →  estimate ${r.x.toFixed(2)},  P = ${r.p.toFixed(3)}`);
    if (st.model === "cv" && st.phase === 1) lines.push(`velocity estimate ${r.v.toFixed(2)} per frame`);
    lines.push(`steady-state K ≈ ${kInf.toFixed(3)} (dashed)`);
    read.textContent = lines.join("\n");
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply(pre ? pre.s : {});
});
