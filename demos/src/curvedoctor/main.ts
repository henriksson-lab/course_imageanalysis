// curvedoctor: eight recorded training runs of a small network on a three-armed spiral (three
// classes); some healthy, most with one thing wrong. Play a run's curves (training blue,
// validation red, ln 3 dashed), diagnose it from the buttons, read the explanation and the
// settings, then apply the fix and compare. The runs were recorded with the demo's own model
// (model.ts) by tools/figs/training-curves/dump.ts into curves.json, so nothing trains in the
// browser.
//
// <div data-demo="curvedoctor" data-run="1"></div>
import { autoMount, button, htmlEl } from "../lib/ui";
import { Plot, type Mark } from "../lib/plot";
import { CASES, DIAGNOSES, BASE, LN3, type CaseConfig, type Diagnosis } from "./model";
import curves from "./curves.json";

const NAME = "curvedoctor";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .cd-left { flex: 1 1 500px; min-width: 0; max-width: 600px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 460px; gap: .4em; }
[data-demo="${NAME}"] .cd-diag .qb-btn { text-align: left; }
[data-demo="${NAME}"] .cd-diag .qb-btn.right { background: #2ca02c; border-color: #2ca02c; color: #fff; }
[data-demo="${NAME}"] .cd-diag .qb-btn.wrong { background: #d62728; border-color: #d62728; color: #fff; }
[data-demo="${NAME}"] .cd-explain { font-size: 15px; min-height: 3em; white-space: pre-line; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .cd-explain { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Rec { train: number[]; val: number[]; trainAcc: number; valAcc: number; recalls: number[]; nTrain: number }
interface Recorded { id: string; diagnosis: Diagnosis; fixLabel: string; explain: string; run: Rec; fix: Rec | null }
const RUNS = curves as unknown as Recorded[];

/** The settings of a case in words. */
function describe(p: Partial<CaseConfig>): string {
  const c = { ...BASE, ...p };
  const net = c.hidden.length ? `2-${c.hidden.join("-")}-3` : "2-3 (no hidden layer)";
  return `network ${net}, Adam ${c.lr}, batches of ${c.batch}, ${c.epochs} epochs, weight decay ${c.decay}\n` +
    `training points per class ${c.counts.join(" / ")}` + (c.zeroInputs ? ", inputs all 0" : "") +
    (c.shuffleLabels ? ", labels shuffled" : "") + (c.weighted ? ", class weights" : "");
}

function mount(host: HTMLElement): void {
  injectStyle();
  let cur = Math.max(0, RUNS.findIndex(r => r.id === host.dataset.run));
  let shown = 0, timer = 0, answered = false, fixShown = false;

  const left = htmlEl("div", "cd-left", "", host);
  const plot = new Plot(left, { width: 600, height: 340, x: [0, 200], y: [0, 1.6], xlabel: "epoch", ylabel: "loss (cross-entropy)", margin: { l: 70, r: 12, t: 10, b: 52 } });
  plot.hline(LN3, { color: "#777", dash: "6 5" });
  const fixT = plot.line([], [], { color: "#2ca02c", width: 2.5, label: "fixed: training" });
  const fixV = plot.line([], [], { color: "#ff7f0e", width: 2.5, label: "fixed: validation" });
  const tLine = plot.line([], [], { color: "#1f77b4", width: 3, label: "training" });
  const vLine = plot.line([], [], { color: "#d62728", width: 3, label: "validation" });
  const runRow = htmlEl("div", "qb-presets", "", left);
  const runBtns = RUNS.map((r, i) => button(runRow, `run ${r.id}`, () => select(i)));
  const readout = htmlEl("div", "qb-readout", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "play", () => play());
  button(ctl, "show all", () => { stop(); shown = Infinity; draw(); });
  htmlEl("div", "qb-big", "Your diagnosis:", panel);
  const diagBox = htmlEl("div", "qb-presets cd-diag", "", panel);
  const diagBtns = DIAGNOSES.map(([d, label]) => button(diagBox, label, () => answer(d)));
  const explain = htmlEl("div", "cd-explain", "", panel);
  const fixBtn = button(panel, "apply the fix", () => { fixShown = !fixShown; draw(); });

  function select(i: number): void {
    stop();
    cur = i; shown = 0; answered = false; fixShown = false;
    runBtns.forEach((b, k) => b.classList.toggle("active", k === i));
    diagBtns.forEach(b => b.classList.remove("right", "wrong"));
    explain.textContent = "Play the run, then choose a diagnosis.";
    play();
  }
  function answer(d: Diagnosis): void {
    const R = RUNS[cur], c = CASES.find(k => k.id === R.id)!;
    answered = true;
    diagBtns.forEach((b, k) => {
      b.classList.remove("right", "wrong");
      if (DIAGNOSES[k][0] === R.diagnosis) b.classList.add("right");
      else if (DIAGNOSES[k][0] === d) b.classList.add("wrong");
    });
    explain.textContent = (d === R.diagnosis ? "Right. " : "Not quite. ") + R.explain + "\nSettings: " + describe(c.cfg);
    draw();
  }

  function draw(): void {
    const R = RUNS[cur], run = R.run, n = run.train.length - 1, k = Math.min(n, shown);
    const ep = Array.from({ length: k + 1 }, (_, i) => i), clip = (v: number) => Math.min(1.6, v);
    plot.setDomain([0, n], [0, 1.6]);
    tLine.set(ep, run.train.slice(0, k + 1).map(clip));
    vLine.set(ep, run.val.slice(0, k + 1).map(clip));
    const f = fixShown && R.fix ? R.fix : null;
    fixT.set(f ? f.train.map((_, i) => i) : [], f ? f.train.map(clip) : []);
    fixV.set(f ? f.val.map((_, i) => i) : [], f ? f.val.map(clip) : []);
    plot.legend([
      { label: "training", color: "#1f77b4" }, { label: "validation", color: "#d62728" },
      ...(f ? [{ label: "fixed: training", color: "#2ca02c" }, { label: "fixed: validation", color: "#ff7f0e" }] : []),
      { label: "ln 3: guessing", color: "#777", dash: "6 5" },
    ], "tr");
    const done = k >= n;
    let best = 0;
    for (let i = 0; i <= k; i++) if (run.val[i] < run.val[best]) best = i;
    const pct = (v: number) => `${(100 * v).toFixed(0)} %`;
    let txt = `epoch ${k} of ${n}: training loss ${run.train[k].toFixed(3)}, validation loss ${run.val[k].toFixed(3)}\n` +
      `lowest validation loss so far: ${run.val[best].toFixed(3)} at epoch ${best}`;
    if (done) txt += `\nfinal accuracy: training ${pct(run.trainAcc)}, validation ${pct(run.valAcc)}\nvalidation recall per class: ${run.recalls.map(pct).join(", ")}`;
    if (f) txt += `\nfixed (${R.fixLabel}): validation ${pct(f.valAcc)}, recall ${f.recalls.map(pct).join(", ")}`;
    readout.textContent = txt;
    fixBtn.disabled = !answered || !R.fix;
    fixBtn.textContent = !answered ? "apply the fix (after your diagnosis)" : R.fix ? (fixShown ? "hide the fix" : `apply the fix: ${R.fixLabel}`) : "nothing to fix";
  }

  function play(): void {
    stop();
    shown = 0;
    const n = RUNS[cur].run.train.length - 1, per = Math.max(1, Math.round(n / 80));
    const tick = () => {
      if (!host.isConnected) return;
      shown += per; draw();
      if (shown < n) timer = window.setTimeout(tick, 30);
    };
    tick();
  }
  function stop(): void { clearTimeout(timer); }

  select(cur);
}

autoMount(NAME, mount);
