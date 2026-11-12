// Pixel-level evaluation of a threshold against ground truth. A synthetic image of
// bright blobs (object fraction, contrast, noise and an illumination ramp adjustable;
// the truth is known by construction). Left: the image coloured by the four cases (TP
// green, FP red, FN blue, TN grey). Middle: the confusion matrix and the derived rates.
// Right: the ROC curve (TPR vs FPR) and the precision–recall curve traced as t sweeps,
// with the current t as a moving dot and the Otsu, minimum-error and max-Dice thresholds
// marked. "local" evaluates a local threshold instead (one point, no curve).
// Auto-mounts on <div data-demo="threshroc">. Options: data-preset="balanced|rare|otsu|illum|noise".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { confusion, Counts, dice, jaccard, precision, recall, fpr, accuracy } from "../lib/metrics";
import { injectStyle } from "../histthresh/scenes";
import { SweepPoint } from "./roc";
import { N, rocModel, RocModel } from "./model";

const NAME = "threshroc";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .tr-img { flex: 0 1 280px; min-width: 200px; }
[data-demo="${NAME}"] .tr-img canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .tr-key span { display: inline-block; width: .9em; height: .9em; border-radius: 2px; margin: 0 .2em 0 .45em; vertical-align: -1px; }
[data-demo="${NAME}"] .tr-mid { flex: 1 1 420px; min-width: 280px; max-width: 540px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .tr-plots { display: flex; gap: .4em; }
[data-demo="${NAME}"] .tr-plots > svg { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .tr-tab { display: flex; gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] table.tr-cm { border-collapse: collapse; font-size: 15px; }
[data-demo="${NAME}"] .tr-cm td, [data-demo="${NAME}"] .tr-cm th { border: 1px solid var(--qb-grid); padding: .15em .45em; text-align: center; }
[data-demo="${NAME}"] .tr-cm th { font-weight: 600; background: var(--qb-panel); }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; flex: 0 0 auto; white-space: pre; }
[data-demo="${NAME}"] .tr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .tr-cm { font-size: 14px; }
`;

const TPc = [44, 160, 44], FPc = [214, 39, 40], FNc = [31, 119, 220];

interface State { fraction: number; contrast: number; noise: number; light: number; smooth: boolean; local: boolean; t: number }
const BASE: State = { fraction: 0.25, contrast: 50, noise: 20, light: 0, smooth: false, local: false, t: 100 };
const PRESETS: { id: string; label: string; s: Partial<State>; at?: "otsu" | "dice" }[] = [
  { id: "balanced", label: "balanced classes", s: { fraction: 0.5 }, at: "otsu" },
  { id: "rare", label: "3 % objects: accuracy lies", s: { fraction: 0.03, t: 254 } },
  { id: "otsu", label: "Otsu vs max Dice", s: { fraction: 0.1, contrast: 60, noise: 15 }, at: "otsu" },
  { id: "illum", label: "uneven illumination", s: { light: 1.2 }, at: "dice" },
  { id: "noise", label: "noise: smooth first", s: { noise: 40, contrast: 40 }, at: "dice" },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE };

  const fig = htmlEl("figure", "tr-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  const key = htmlEl("div", "tr-key qb-note", "", fig);
  key.innerHTML = `<span style="background:rgb(${TPc})"></span>TP<span style="background:rgb(${FPc})"></span>FP<span style="background:rgb(${FNc})"></span>FN<span style="background:#999"></span>TN`;
  const mid = htmlEl("div", "tr-mid", "", host);
  const pl = htmlEl("div", "tr-plots", "", mid);
  const roc = new Plot(pl, { width: 270, height: 270, x: [0, 1], y: [0, 1], xlabel: "FPR", ylabel: "TPR (recall)", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], equal: true });
  const pr = new Plot(pl, { width: 270, height: 270, x: [0, 1], y: [0, 1], xlabel: "recall", ylabel: "precision", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], equal: true });
  const tab = htmlEl("div", "tr-tab", "", mid);
  const cm = htmlEl("table", "tr-cm", "", tab);
  const ro = htmlEl("div", "qb-readout", "", tab);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "tr-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const tS = slider(panel, "t", 0, 254, 1, st.t, v => { st.t = v; draw(); });
  const tRow = htmlEl("div", "qb-presets", "", panel);
  button(tRow, "t = Otsu", () => { st.t = marks.otsu; draw(); });
  button(tRow, "t = max Dice", () => { st.t = marks.dice; draw(); });
  htmlEl("div", "tr-lab", "image", panel);
  const fS = slider(panel, "objects", 0.01, 0.6, 0.01, st.fraction, v => { st.fraction = v; regen(); }, v => `${Math.round(100 * v)} %`);
  const cS = slider(panel, "contrast", 5, 150, 5, st.contrast, v => { st.contrast = v; regen(); });
  const nS = slider(panel, "noise σ", 0, 60, 1, st.noise, v => { st.noise = v; regen(); });
  const lS = slider(panel, "light ramp", 0, 1.4, 0.05, st.light, v => { st.light = v; regen(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const smC = checkbox(row, "smooth first", st.smooth, v => { st.smooth = v; regen(); });
  const loC = checkbox(row, "local threshold", st.local, v => { st.local = v; draw(); });
  button(panel, "reset", () => load(p0));

  let pendingAt: "otsu" | "dice" | undefined;
  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, p.s);
    pendingAt = p.at;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    fS.set(st.fraction); cS.set(st.contrast); nS.set(st.noise); lS.set(st.light); smC.checked = st.smooth; loC.checked = st.local;
    regen();
  }

  let M: RocModel = rocModel(st);
  let img: Grey = M.img, truth: Uint8Array = M.truth, pts: SweepPoint[] = M.pts, A = M.auc, localMask: Uint8Array = M.localMask;
  const marks = { ...M.marks };
  function regen(): void {
    M = rocModel(st);
    ({ img, truth, pts, localMask } = M); A = M.auc;
    Object.assign(marks, M.marks);
    if (pendingAt) st.t = marks[pendingAt];
    pendingAt = undefined;
    draw();
  }

  const at = (t: number) => M.at(t);

  function draw(): void {
    tS.set(st.t);
    const mask = st.local ? localMask : Uint8Array.from(img.data, v => (v > st.t ? 1 : 0));
    const c: Counts = confusion(mask, truth);
    // overlay
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const g = img.data[i], p = mask[i], q = truth[i];
      const col = p && q ? TPc : p ? FPc : q ? FNc : null;
      const a = 0.62;
      id.data[4 * i] = col ? (1 - a) * g + a * col[0] : g;
      id.data[4 * i + 1] = col ? (1 - a) * g + a * col[1] : g;
      id.data[4 * i + 2] = col ? (1 - a) * g + a * col[2] : g;
      id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    cap.textContent = st.local ? "local: I > local mean + 5" : `mask: I > ${st.t}`;

    // table
    const n = N * N, pc = (v: number) => `${v}<br><small>${(100 * v / n).toFixed(1)} %</small>`;
    cm.innerHTML = `<tr><th></th><th>pred. object</th><th>pred. backgr.</th></tr>
      <tr><th>true object</th><td style="background:rgba(${TPc},.25)">TP ${pc(c.tp)}</td><td style="background:rgba(${FNc},.22)">FN ${pc(c.fn)}</td></tr>
      <tr><th>true backgr.</th><td style="background:rgba(${FPc},.22)">FP ${pc(c.fp)}</td><td>TN ${pc(c.tn)}</td></tr>`;
    ro.textContent = [
      `precision ${precision(c).toFixed(3)}`, `recall    ${recall(c).toFixed(3)}`, `FPR       ${fpr(c).toFixed(3)}`,
      `accuracy  ${accuracy(c).toFixed(3)}`, `Dice      ${dice(c).toFixed(3)}`, `Jaccard   ${jaccard(c).toFixed(3)}`, `AUC       ${A.toFixed(3)}`,
    ].join("\n");

    // curves
    for (const [P, X, Y] of [[roc, "fpr", "tpr"], [pr, "tpr", "precision"]] as const) {
      P.clear();
      if (P === roc) P.line([0, 1], [0, 1], { color: "var(--qb-grid, #ccc)", width: 1.5, dash: "4 4" });
      P.line(pts.map(p => p[X]), pts.map(p => p[Y]), { color: seriesColor(0), width: 3 });
      const mk: [string, number, string][] = [["Otsu", marks.otsu, seriesColor(4)], ["min-err", marks.kittler, "#8c564b"], ["max Dice", marks.dice, seriesColor(2)]];
      for (const [lab, t, col] of mk) {
        const q = at(t);
        P.scatter([q[X]], [q[Y]], { color: col, r: 7 });
      }
      if (st.local) {
        const q = { fpr: fpr(c), tpr: recall(c), precision: precision(c) };
        P.scatter([q[X]], [q[Y]], { color: "#222", r: 9 });
      } else {
        const q = at(st.t);
        P.scatter([q[X]], [q[Y]], { color: seriesColor(3), r: 9, stroke: "var(--qb-fg, #222)" });
      }
    }
    roc.legend([{ label: "Otsu", color: seriesColor(4), kind: "dot" }, { label: "min-error", color: "#8c564b", kind: "dot" }, { label: "max Dice", color: seriesColor(2), kind: "dot" }, { label: st.local ? "local" : "current t", color: st.local ? "#222" : seriesColor(3), kind: "dot" }], "br");
  }

  load(p0);
}

autoMount(NAME, mount);
