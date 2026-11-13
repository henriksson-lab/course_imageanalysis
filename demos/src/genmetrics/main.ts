// Judging generated samples as a distribution. 2D clouds (the data sets of diffusion2d):
// 1 000 held-out real points (grey) and 1 000 generated points (blue) from a generator menu.
// Fitted Gaussians as ellipses; optional k-NN balls around the real points; bars for precision
// and recall; Fréchet distance and median nearest-training distance; a curve of the metrics
// against the number of steps (or, in guidance mode, against the guidance weight w).
//
// <div data-demo="genmetrics" data-preset="between|drop|memo|steps"></div>
// <div data-demo="genmetrics" data-mode="guidance" data-preset="w0|w2|w8"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { meanCov } from "../lib/genmetrics";
import * as M from "./model";
import { injectStyle, choiceRow, helperCss, fmt } from "../scorematch1d/dview";

const NAME = "genmetrics";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .gm-main { flex: 0 1 460px; min-width: 260px; }
${S} .gm-side { flex: 0 1 320px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 280px; gap: .35em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .gm-main { flex: 0 0 430px; }
.reveal ${S} .gm-side { flex: 0 0 320px; }
.reveal ${S} .qb-panel { flex: 1 1 360px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State { data: M.DataId; gen: M.GenId; steps: number; eta: number; balls: boolean; w: number }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { data: "two", gen: "sampler", steps: 100, eta: 0, balls: false, w: 0 };
const PRESETS: Preset[] = [
  { id: "between", label: "between the modes: low precision", s: { data: "two", gen: "between" } },
  { id: "drop", label: "mode dropping: recall drops", s: { data: "ring", gen: "drop" } },
  { id: "memo", label: "memoriser: Fréchet and precision look fine", s: { data: "moons", gen: "memo" } },
  { id: "steps", label: "more steps: all metrics improve, then flatten", s: { data: "moons", gen: "sampler", steps: 3 } },
];
const GPRESETS: Preset[] = [
  { id: "w0", label: "w = 0: the class as it is", s: { data: "ring", w: 0 } },
  { id: "w2", label: "w = 2: recall drops", s: { data: "ring", w: 2 } },
  { id: "w8", label: "w = 8: only the most typical", s: { data: "ring", w: 8 } },
];
const DATA: M.DataId[] = ["two", "ring", "rare", "moons"];
const DATA_LABEL: Record<string, string> = { two: "two blobs", ring: "ring of 8", rare: "rare blob", moons: "two moons" };
const stepsOf = (u: number) => Math.max(1, Math.round(200 ** (u / 100)));
const uOf = (n: number) => 100 * Math.log(n) / Math.log(200);
const CURVE_STEPS = [1, 2, 3, 5, 10, 20, 50, 100];
const CURVE_W = [0, 0.5, 1, 2, 4, 8];
const CLS = 1; // guidance mode: the "upper half" group of the ring

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  // larger labels on slides (the plots are scaled down there)
  const F = host.closest(".reveal") ? 23 : 18;
  const guide = host.dataset.mode === "guidance";
  const PS = guide ? GPRESETS : PRESETS;
  const p0 = PS.find(p => p.id === host.dataset.preset) ?? PS[0];
  const st: State = { ...BASE, ...p0.s };

  const main = htmlEl("div", "gm-main", "", host);
  const pl = new Plot(main, { width: 470, height: 470, x: [-3, 3], y: [-3, 3], equal: true, xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], font: F });
  const balls = pl.mark<[Float64Array, Float64Array]>((g, X, R) => {
    for (let p = 0; p < R.length; p++) {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", String(pl.x(X[2 * p]))); c.setAttribute("cy", String(pl.y(X[2 * p + 1]))); c.setAttribute("r", String(Math.abs(pl.x(R[p]) - pl.x(0))));
      c.style.fill = "#999"; c.style.fillOpacity = "0.08"; c.style.stroke = "none";
      g.appendChild(c);
    }
  }, [new Float64Array(0), new Float64Array(0)]);
  const fakeM = pl.scatter([], [], { color: seriesColor(0), r: 2.8, opacity: 0.5, stroke: "none" });
  const realM = pl.scatter([], [], { color: "#333", r: 1.8, opacity: 0.8, stroke: "none" });
  const ellR = pl.ellipse([0, 0], [[1, 0], [0, 1]], 2, { color: "#555", dash: "7 5", fillOpacity: 0 });
  const ellF = pl.ellipse([0, 0], [[1, 0], [0, 1]], 2, { color: seriesColor(0), fillOpacity: 0 });
  pl.legend([{ label: guide ? "real (class)" : "real", color: "#333", kind: "dot" }, { label: "generated", color: seriesColor(0), kind: "dot" }], "tl");

  const side = htmlEl("div", "gm-side", "", host);
  const bp = new Plot(side, { width: 320, height: 190, x: [-0.6, 1.6], y: [0, 1], xticks: [], yticks: [0, 0.5, 1], font: F, margin: { b: 1.6 * F + 6, l: 2.8 * F } });
  const bars = bp.bars([0, 1], [0, 0], { barWidth: 0.6, colors: [seriesColor(2), seriesColor(4)] });
  bp.text(0, -0.13, "precision", { anchor: "middle" }); bp.text(1, -0.13, "recall", { anchor: "middle" });
  const cp = new Plot(side, { width: 320, height: 230, x: guide ? [0, 8] : [1, 100], xlog: !guide, y: [0, 1], xlabel: guide ? "guidance w" : "steps", yticks: [0, 0.5, 1], font: F, margin: { l: 2.8 * F } });
  const cPrec = cp.line([], [], { color: seriesColor(2), width: 3 }), cRec = cp.line([], [], { color: seriesColor(4), width: 3 });
  const cNow = cp.vline(NaN, { color: "var(--qb-muted)", width: 2 });
  cp.legend([{ label: "precision", color: seriesColor(2) }, { label: "recall", color: seriesColor(4) }], "br");

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const dataC = choiceRow(panel, "data", (guide ? ["ring"] as M.DataId[] : DATA).map(v => ({ v, label: DATA_LABEL[v] })), st.data, v => { st.data = v; recompute(); });
  const genC = choiceRow(panel, "generator", M.GEN_IDS.map(v => ({ v, label: M.GEN_LABEL[v] })), st.gen, v => { st.gen = v; recompute(); });
  const stepS = slider(panel, "steps", 0, 100, 1, uOf(st.steps), u => { st.steps = stepsOf(u); if (st.gen === "sampler" || guide) later(); }, u => String(stepsOf(u)));
  const etaS = slider(panel, "η", 0, 1, 0.1, st.eta, v => { st.eta = v; if (st.gen === "sampler" || guide) later(); });
  const wS = slider(panel, "guidance w", 0, 8, 0.5, st.w, v => { st.w = v; later(); });
  if (guide) { genC.row.style.display = "none"; } else { (wS.input.parentElement as HTMLElement).style.display = "none"; }
  const row = htmlEl("div", "qb-presets", "", panel);
  const ballC = checkbox(row, "k-NN balls", st.balls, v => { st.balls = v; drawBalls(); });
  button(row, guide ? "curve against w" : "curve against steps", () => curve());
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", guide
    ? "Real: 1 000 points of the class 'upper half' of the ring. Generated: guided diffusion towards that class with weight w. Ellipses: the fitted Gaussians (2 sd)."
    : "Precision: generated points inside the union of k-NN balls (k = 3) of the real points. Recall: real points inside the balls of the generated points. Ellipses: the fitted Gaussians (2 sd) used by the Fréchet distance.", panel);

  let sets = M.realSets(st.data, guide ? CLS : null), fake: Float64Array = new Float64Array(0), met: M.Metrics | null = null;

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    dataC.set(st.data); genC.set(st.gen); stepS.set(uOf(st.steps)); etaS.set(st.eta); wS.set(st.w); ballC.checked = st.balls;
    pbtns.forEach((b, i) => b.classList.toggle("active", PS[i] === p));
    cPrec.set([], []); cRec.set([], []);
    recompute();
  }

  let timer = 0;
  /** Recompute after the slider has rested for a moment (a run takes 0.2–0.5 s). */
  function later(): void { clearTimeout(timer); timer = window.setTimeout(recompute, 180); }

  function recompute(): void {
    sets = M.realSets(st.data, guide ? CLS : null);
    fake = guide ? M.sampler(st.data, st.steps, st.eta, 33, CLS, st.w) : M.generate(st.data, st.gen, st.steps, st.eta, sets.train);
    met = M.metrics(sets.real, fake, sets.train);
    const xs = (X: Float64Array, o: number) => Float64Array.from({ length: X.length / 2 }, (_, i) => X[2 * i + o]);
    realM.set(xs(sets.real, 0), xs(sets.real, 1));
    fakeM.set(xs(fake, 0), xs(fake, 1));
    const a = meanCov(sets.real, 2), b = meanCov(fake, 2);
    ellR.set([a.mu[0], a.mu[1]], [[a.S[0], a.S[1]], [a.S[2], a.S[3]]]);
    ellF.set([b.mu[0], b.mu[1]], [[b.S[0], b.S[1]], [b.S[2], b.S[3]]]);
    bars.set([0, 1], [met.precision, met.recall]);
    cNow.set(guide ? st.w : st.gen === "sampler" ? st.steps : NaN);
    drawBalls();
    ro.textContent =
      `precision ${fmt(met.precision, 2)}   recall ${fmt(met.recall, 2)}\n` +
      `Fréchet distance ${fmt(met.frechet, 3)}\n` +
      `median distance to the nearest training point:\n  generated ${fmt(met.nnTrain, 3)}   held-out real ${fmt(met.nnReal, 3)}`;
  }

  function drawBalls(): void {
    if (st.balls && met) balls.set(sets.real, met.rReal); else balls.set(new Float64Array(0), new Float64Array(0));
  }

  function curve(): void {
    const xsv = guide ? CURVE_W : CURVE_STEPS, P: number[] = [], R: number[] = [];
    for (const v of xsv) {
      const f = guide ? M.sampler(st.data, st.steps, st.eta, 33, CLS, v) : M.sampler(st.data, v, st.eta);
      const m = M.metrics(sets.real, f, sets.train);
      P.push(m.precision); R.push(m.recall);
    }
    cPrec.set(xsv, P); cRec.set(xsv, R);
  }

  apply(p0);
}

autoMount(NAME, mount);
