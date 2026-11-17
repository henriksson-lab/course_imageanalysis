// Image-to-image translation in one pixel: an input value x (label-free) and a target y
// (a stain), measured in pairs (grey dots). For x > 0.5 a marker that the input does not show
// adds `gap` to the target in a share q of the cells. What a translator trained perfectly with
// each loss outputs: squared error → the conditional mean (orange), absolute error → the
// median (green), a conditional GAN → a sample of p(y | x) (blue dots, one per held-out cell);
// unpaired training (purple) can only match the two distributions. Below: how much 16 samples
// per cell disagree. Readout: errors against held-out truth, realism, the share of marker-on cells.
//
// <div data-demo="translate1d" data-preset="visible|hidden|samples|treated|unpaired"></div>
import { autoMount, htmlEl, slider, presets, checkbox, rng } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { injectStyle, helperCss, fmt } from "../scorematch1d/dview";
import * as T from "./model";

const NAME = "translate1d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const P = `${S}:not(.reveal *)`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .t1-main { flex: 0 1 480px; min-width: 260px; display: flex; flex-direction: column; }
${S} .qb-panel { flex: 1 1 300px; gap: .35em; max-width: 420px; }
${S} .qb-readout { font-size: 14px; }
@media (min-width: 641px) {
  ${P} { flex-wrap: nowrap; }
  ${P} .t1-main { flex: 0 1 min(480px, calc(100vh - 170px)); min-width: 0; }
  ${P} .qb-note { font-size: 13px; }
}
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .6em; }
.reveal ${S} .t1-main { flex: 0 0 470px; }
.reveal ${S} .qb-panel { flex: 1 1 520px; max-width: none; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 17px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State { gap: number; qTrain: number; qTest: number; mean: boolean; median: boolean; samples: boolean; unpaired: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { gap: 0.3, qTrain: 0.5, qTest: 0.5, mean: true, median: false, samples: false, unpaired: false };
const PRESETS: Preset[] = [
  { id: "visible", label: "target visible in the input: all agree", s: { gap: 0, mean: true, median: true, samples: true } },
  { id: "hidden", label: "hidden marker: the mean is no real cell", s: { mean: true, median: true } },
  { id: "samples", label: "sharp samples: coin flips", s: { mean: false, samples: true } },
  { id: "treated", label: "treated test cells: the effect vanishes", s: { qTrain: 0.2, qTest: 0.8, mean: false, samples: true } },
  { id: "unpaired", label: "unpaired: matching distributions, not pairs", s: { mean: false, unpaired: true } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const F = host.closest(".reveal") ? 22 : 17;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };

  const main = htmlEl("div", "t1-main", "", host);
  const pl = new Plot(main, { width: 480, height: 360, x: [0, 1], y: [0, 1.1], xticks: [0, 0.2, 0.4, 0.6, 0.8, 1], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1], xlabel: "input x (label-free)", ylabel: "target y (stain)", font: F });
  const trainM = pl.scatter([], [], { color: "#888", r: 2.6, opacity: 0.6, stroke: "none" });
  const sampM = pl.scatter([], [], { color: seriesColor(0), r: 3.2, opacity: 0.8, stroke: "none" });
  const upInc = pl.line([], [], { color: seriesColor(4), width: 3 });
  const upDec = pl.line([], [], { color: seriesColor(4), width: 3, dash: "7 5" });
  const medM = pl.line([], [], { color: seriesColor(2), width: 3.5 });
  const meanM = pl.line([], [], { color: seriesColor(3), width: 3.5 });
  pl.legend([{ label: "training pairs", color: "#888", kind: "dot" }, { label: "mean (squared error)", color: seriesColor(3) }, { label: "median (absolute error)", color: seriesColor(2) },
    { label: "samples (conditional GAN)", color: seriesColor(0), kind: "dot" }, { label: "unpaired maps", color: seriesColor(4) }], "br");
  const sp = new Plot(main, { width: 480, height: 130, x: [0, 1], y: [0, 0.2], xticks: [0, 0.5, 1], yticks: [0, 0.1, 0.2], ylabel: "spread", font: F });
  const spread = sp.line([], [], { color: seriesColor(0), width: 3 });

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const sGap = slider(panel, "marker adds (gap)", 0, 0.4, 0.05, st.gap, v => { st.gap = v; update(); });
  const sTr = slider(panel, "marker-on share, training", 0, 1, 0.1, st.qTrain, v => { st.qTrain = v; update(); });
  const sTe = slider(panel, "marker-on share, test", 0, 1, 0.1, st.qTest, v => { st.qTest = v; update(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const cMean = checkbox(row, "mean", st.mean, v => { st.mean = v; draw(); });
  const cMed = checkbox(row, "median", st.median, v => { st.median = v; draw(); });
  const cSam = checkbox(row, "samples", st.samples, v => { st.samples = v; draw(); });
  const cUp = checkbox(row, "unpaired", st.unpaired, v => { st.unpaired = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "400 training pairs, 400 held-out test pairs (noise sd 0.03). Each translator is shown as it would be after perfect training on the training pairs. A prediction counts as realistic if a real cell could have that value.", panel);

  let train: T.Pair[] = [], test: T.Pair[] = [];
  const gx = Array.from({ length: 101 }, (_, i) => i / 100);

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    sGap.set(st.gap); sTr.set(st.qTrain); sTe.set(st.qTest);
    cMean.checked = st.mean; cMed.checked = st.median; cSam.checked = st.samples; cUp.checked = st.unpaired;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update();
  }
  function update(): void {
    train = T.pairs(T.N_TRAIN, st.gap, st.qTrain, 1);
    test = T.pairs(T.N_TEST, st.gap, st.qTest, 2);
    trainM.set(train.map(p => p.x), train.map(p => p.y));
    meanM.set(gx, gx.map(x => T.condMean(train, x)));
    medM.set(gx, gx.map(x => T.condMedian(train, x)));
    const r = rng(9);
    sampM.set(test.map(p => p.x), test.map(p => T.condSample(train, p.x, r)));
    const xs = train.map(p => p.x), ys = train.map(p => p.y);
    upInc.set(gx, gx.map(x => T.quantileMap(xs, ys, x)));
    upDec.set(gx, gx.map(x => T.quantileMap(xs, ys, x, true)));
    const r2 = rng(10), sx = gx.filter((_, i) => i % 2 === 0);
    spread.set(sx, sx.map(x => {
      let s1 = 0, s2 = 0;
      for (let k = 0; k < 16; k++) { const y = T.condSample(train, x, r2); s1 += y; s2 += y * y; }
      return Math.sqrt(Math.max(0, s2 / 16 - (s1 / 16) ** 2));
    }));
    const s = T.scores(train, test, st.gap);
    ro.textContent =
      `error against held-out truth (RMSE):\n  mean ${fmt(s.rmseMean, 3)}  median ${fmt(s.rmseMedian, 3)}  samples ${fmt(s.rmseSample, 3)}  unpaired ${fmt(s.rmseUnpaired, 3)}\n` +
      `realistic predictions:\n  mean ${fmt(100 * s.realMean, 0)} %  median ${fmt(100 * s.realMedian, 0)} %  samples ${fmt(100 * s.realSample, 0)} %\n` +
      `marker-on share (x > 0.5): truth ${fmt(100 * s.onTruth, 0)} %, from samples ${fmt(100 * s.onSample, 0)} %`;
    draw();
  }
  function draw(): void {
    meanM.g.style.display = st.mean ? "" : "none";
    medM.g.style.display = st.median ? "" : "none";
    sampM.g.style.display = st.samples ? "" : "none";
    upInc.g.style.display = upDec.g.style.display = st.unpaired ? "" : "none";
  }

  apply(p0);
}

autoMount(NAME, mount);
