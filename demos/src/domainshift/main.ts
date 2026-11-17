// domainshift: a classifier trained on microscope A, used on microscope B. Each cell is two
// numbers (mean intensity, size); B measures intensity on another scale (the shift slider).
// Left: A's cells (hollow) and B's cells (filled), the decision line learnt on A (dashed) and the
// line of the chosen remedy, drawn in B's measurements. Right: accuracy on B against the shift
// for the four remedies, and a shift score computed without labels.
//
// <div data-demo="domainshift" data-preset="shift"></div>
//   data-preset  none | shift | normalise | balance | finetune
import { autoMount, button, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, seriesColor, type Mark } from "../lib/plot";
import { trainA, evaluateB, REMEDIES, REMEDY_LABEL, type Remedy, type W } from "./model";

const NAME = "domainshift";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .ds-left { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .ds-right { flex: 1 1 400px; min-width: 0; max-width: 500px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .ds-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .5em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; s: number; frac: number; remedy: Remedy }
const PRESETS: Preset[] = [
  { id: "none", label: "same microscope", s: 0, frac: 0.5, remedy: "none" },
  { id: "shift", label: "dimmer microscope B", s: 0.8, frac: 0.5, remedy: "none" },
  { id: "normalise", label: "normalise per microscope", s: 0.8, frac: 0.5, remedy: "normalise" },
  { id: "balance", label: "fewer mitoses in B", s: 0.8, frac: 0.2, remedy: "normalise" },
  { id: "finetune", label: "fine-tune on 10 B cells", s: 0.8, frac: 0.2, remedy: "finetune" },
];
const C0 = "#2a6f97", C1 = "#e08a1e";
const REM_COLOUR: Record<Remedy, string> = { none: seriesColor(1), normalise: seriesColor(0), augment: seriesColor(2), finetune: seriesColor(4) };
const SHIFTS = Array.from({ length: 21 }, (_, i) => i / 20);

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  let st = { s: start.s, frac: start.frac, remedy: start.remedy };
  const T = trainA();

  const left = htmlEl("div", "ds-left", "", host);
  const sc = new Plot(left, { width: 440, height: 420, x: [0, 1], y: [0, 1], xlabel: "mean intensity", ylabel: "size", margin: { t: 10, r: 12 } });
  const aMarks = [0, 1].map(c => sc.scatter([], [], { color: "none", stroke: c ? C1 : C0, r: 4, opacity: 0.55 } as any));
  const bMarks = [0, 1].map(c => sc.scatter([], [], { color: c ? C1 : C0, r: 4, opacity: 0.85 }));
  const tuneMark: Mark<any> = sc.scatter([], [], { color: "var(--qb-fg)", r: 7 });
  const lineA: Mark<any> = sc.line([], [], { color: "var(--qb-fg)", width: 2.5, dash: "7 5" });
  const lineR: Mark<any> = sc.line([], [], { color: REM_COLOUR.none, width: 3.5 });
  htmlEl("div", "qb-note", "", left).innerHTML =
    `<span style="color:${C0}">●</span> interphase <span style="color:${C1}">●</span> mitotic; hollow: microscope A (training), filled: microscope B. Dashed: the line learnt on A; solid: the chosen remedy's line.`;

  const right = htmlEl("div", "ds-right", "", host);
  presetRow(right, PRESETS.map(p => ({ label: p.label, apply: () => { st = { s: p.s, frac: p.frac, remedy: p.remedy }; sS.set(p.s); sF.set(p.frac); setBtns(); recompute(); } })));
  const sS = slider(right, "shift of B", 0, 1, 0.05, st.s, v => { st.s = v; draw(); });
  const sF = slider(right, "mitotic share in B", 0.1, 0.5, 0.05, st.frac, v => { st.frac = v; recompute(); });
  const remRow = htmlEl("div", "ds-row", "", right);
  htmlEl("span", "", "remedy:", remRow);
  const remBtns = REMEDIES.map(r => button(remRow, REMEDY_LABEL[r], () => { st.remedy = r; setBtns(); draw(); }));
  function setBtns(): void { remBtns.forEach((b, i) => b.classList.toggle("active", REMEDIES[i] === st.remedy)); }
  const readout = htmlEl("div", "qb-readout", "", right);
  const ac = new Plot(right, { width: 500, height: 220, x: [0, 1], y: [0.5, 1], xlabel: "shift of microscope B", ylabel: "accuracy on B", margin: { t: 10, r: 12 } });
  const curves = REMEDIES.map(r => ac.line([], [], { color: REM_COLOUR[r], width: 2.8 }));
  ac.legend(REMEDIES.map(r => ({ label: REMEDY_LABEL[r], color: REM_COLOUR[r] })), "bl");
  const marker: Mark<any> = ac.vline(st.s, { color: "var(--qb-fg)", width: 2 });

  let table: ReturnType<typeof evaluateB>[] = [];
  function recompute(): void {
    table = SHIFTS.map(s => evaluateB(T, s, st.frac));
    REMEDIES.forEach((r, i) => curves[i].set(SHIFTS, table.map(e => e.acc[r])));
    draw();
  }

  // the line w0 + w1 f1 + w2 f2 = 0 across the plot
  function lineOf(w: W): [number[], number[]] {
    if (Math.abs(w[2]) < 1e-9) { const x = -w[0] / w[1]; return [[x, x], [0, 1]]; }
    const xs = [0, 1], ys = xs.map(x => -(w[0] + w[1] * x) / w[2]);
    return [xs, ys];
  }

  function draw(): void {
    const e = evaluateB(T, st.s, st.frac);
    for (const c of [0, 1]) {
      const a = T.A.filter(q => q.c === c), b = e.B.filter(q => q.c === c);
      aMarks[c].set(a.map(q => q.f[0]), a.map(q => q.f[1]));
      bMarks[c].set(b.map(q => q.f[0]), b.map(q => q.f[1]));
    }
    tuneMark.set(st.remedy === "finetune" ? e.tune.map(q => q.f[0]) : [], st.remedy === "finetune" ? e.tune.map(q => q.f[1]) : []);
    const [ax, ay] = lineOf(T.wNone); lineA.set(ax, ay);
    let w: W = T.wNone;
    if (st.remedy === "augment") w = T.wAug;
    if (st.remedy === "finetune") w = e.wTune;
    if (st.remedy === "normalise") {
      // z_k = (f_k − μ_k)/σ_k with B's own statistics, written back in B's raw measurements
      const n = T.wNorm, m = e.stB.mean, s = e.stB.sd;
      w = [n[0] - n[1] * m[0] / s[0] - n[2] * m[1] / s[1], n[1] / s[0], n[2] / s[1]];
    }
    const [rx, ry] = lineOf(w); lineR.set(rx, ry);
    lineR.style.color = REM_COLOUR[st.remedy];
    lineR.g.querySelectorAll("path,polyline,line").forEach(p => (p as SVGElement).style.stroke = REM_COLOUR[st.remedy]);
    marker.set(st.s);
    readout.textContent =
      `accuracy on B\n${REMEDIES.map(r => `  ${(REMEDY_LABEL[r] + ":").padEnd(27)} ${(100 * e.acc[r]).toFixed(1)} %`).join("\n")}\n` +
      `shift score (no labels): ${e.score.toFixed(2)} SD`;
  }

  setBtns();
  recompute();
}

autoMount(NAME, mount);
