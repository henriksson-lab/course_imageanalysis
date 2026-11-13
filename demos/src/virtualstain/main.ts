// Virtual staining with an exact generative prior: predict a fluorescence channel (target)
// from a transmitted-light-like channel (input). 300 synthetic training pairs, 32 × 32; the
// input is measured with noise, the target not at all, so the posterior over the target is
// exact. Panels: input, truth (toggle), posterior mean, four samples, disagreement (pixel sd
// over 32 samples). Marker: the nucleus (visible in the input), random spots (not visible),
// or a treatment that doubles the stain (the prior has only untreated cells).
//
// <div data-demo="virtualstain" data-preset="nucleus|spots|treated|noisy"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import * as M from "./model";
import { injectStyle, choiceRow, helperCss, imgPanel, fmt } from "../scorematch1d/dview";

const NAME = "virtualstain";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = helperCss(NAME) + `
${S} { gap: .7em; }
${S} .vs-grid { flex: 1 1 520px; min-width: 0; max-width: 640px; display: grid; grid-template-columns: repeat(4, 1fr); gap: .3em .5em; }
${S} .qb-panel { flex: 1 1 300px; gap: .4em; }
${S} .qb-readout { font-size: 15px; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; gap: .5em; }
.reveal ${S} .vs-grid { flex: 0 0 600px; }
.reveal ${S} .qb-panel { flex: 1 1 440px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .05em .4em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .45em; line-height: 1.25; }
.reveal ${S} .qb-note { display: none; }
`;

interface State { marker: M.Marker; noise: number; truth: boolean; cell: number; seed: number }
interface Preset { id: string; label: string; s: Partial<State> }
const BASE: State = { marker: "nucleus", noise: 0.1, truth: true, cell: 3, seed: 5 };
const PRESETS: Preset[] = [
  { id: "nucleus", label: "nucleus: virtual staining works", s: { marker: "nucleus" } },
  { id: "spots", label: "spots: it paints the average", s: { marker: "spots" } },
  { id: "treated", label: "a treatment effect disappears", s: { marker: "treated" } },
  { id: "noisy", label: "noisier input: the nucleus too", s: { marker: "nucleus", noise: 0.4 } },
];
// fluorescence-like colour map: black → green → white
const green = (t: number): [number, number, number] => [255 * Math.max(0, 2 * t - 1), 255 * Math.min(1, 1.3 * t), 255 * Math.max(0, 2 * t - 1)];
const heat = (t: number): [number, number, number] => [255 * Math.min(1, 2 * t), 255 * Math.max(0, 2 * t - 1), 40];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, ...p0.s };
  const trains: Partial<Record<M.Marker, ReturnType<typeof M.trainingSet>>> = {};
  const train = (m: M.Marker) => (trains[m === "treated" ? "nucleus" : m] ??= M.trainingSet(m === "treated" ? "nucleus" : m));

  const grid = htmlEl("div", "vs-grid", "", host);
  const pIn = imgPanel(grid, "input (measured)"), pTr = imgPanel(grid, "truth (target)"), pMean = imgPanel(grid, "posterior mean"), pDis = imgPanel(grid, "disagreement");
  const pS = [0, 1, 2, 3].map(i => imgPanel(grid, `sample ${i + 1}`));

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const mkC = choiceRow(panel, "marker", [{ v: "nucleus" as M.Marker, label: "nucleus" }, { v: "spots" as M.Marker, label: "random spots" }, { v: "treated" as M.Marker, label: "treated cell" }], st.marker, v => { st.marker = v; draw(); });
  const nS = slider(panel, "input noise", 0.02, 0.6, 0.02, st.noise, v => { st.noise = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const trC = checkbox(row, "show truth", st.truth, v => { st.truth = v; draw(); });
  button(row, "another cell", () => { st.cell++; draw(); });
  button(row, "resample", () => { st.seed++; draw(); });
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Training set: 75 cell geometries (nucleus position and size), each with 4 independent draws of the marker: 300 pairs (synthetic). The prior is one narrow Gaussian around each pair, so the posterior is exact: a weighted average of training targets whose inputs match the measurement.", panel);

  function apply(p: Preset): void {
    Object.assign(st, BASE, p.s);
    mkC.set(st.marker); nS.set(st.noise); trC.checked = st.truth;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  function draw(): void {
    const c = M.truthCell(st.marker, st.cell), y = M.measureInput(c, st.noise);
    const post = M.posterior(train(st.marker), y, st.noise), mean = M.meanTarget(post), S = M.sampleTargets(post, 32, st.seed), dis = M.disagreement(S);
    const W = M.W;
    pIn.draw(y, W, W, 0, 1);
    if (st.truth) pTr.draw(c.target, W, W, 0, 0.9, green); else pTr.draw(new Float64Array(W * W).fill(0.5), W, W, 0, 1);
    pTr.cap.textContent = st.truth ? "truth (target)" : "truth (hidden)";
    pMean.draw(mean, W, W, 0, 0.9, green);
    pDis.draw(dis, W, W, 0, 0.4, heat);
    pS.forEach((p, i) => p.draw(S[i], W, W, 0, 0.9, green));
    const rt = M.readouts(c.target), rm = M.readouts(mean), rs = S.map(s => M.readouts(s));
    const tw = M.topWeights(post, 4);
    let eff = 0; for (const w of post.w) eff += w * w;
    const line = (n: string, r: { area: number; spots: number; intensity: number }) => `${n.padEnd(8)} ${String(r.area).padStart(4)} ${String(r.spots).padStart(5)} ${fmt(r.intensity, 2).padStart(9)}`;
    ro.textContent =
      `         area  spots  intensity\n` +
      (st.truth ? line("truth", rt) + "\n" : "") +
      line("mean", rm) + "\n" +
      rs.slice(0, 4).map((r, i) => line(`sample ${i + 1}`, r)).join("\n") + "\n" +
      `top training weights: ${tw.map(t => fmt(t.w, 2)).join(", ")}\n` +
      `effective number of training pairs: ${fmt(1 / eff, 1)}`;
  }

  apply(p0);
}

autoMount(NAME, mount);
