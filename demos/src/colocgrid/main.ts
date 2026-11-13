// Colocalisation by hand: two small editable channels A and B (3 × 3 or 6 × 6; click the
// upper half of a cell for +1, the lower half for −1), the scatter plot of the pixel pairs
// with the two thresholds drawn as lines, and a live readout of Pearson r, Spearman,
// the overlap coefficient, k1, k2 and Manders M1, M2. Cells above their own threshold are
// tinted; cells above both thresholds (the ones M1 and M2 count) are outlined.
// Auto-mounts on <div data-demo="colocgrid">. Options: data-preset="<id>".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { GridView, CellMark } from "../lib/gridui";
import { pearsonMasked, spearmanMasked, overlapCoefficient, kCoefficients, manders } from "../lib/coloc";

const NAME = "colocgrid";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cg-grid { flex: 0 1 250px; min-width: 150px; }
.qb-demo[data-demo="${NAME}"] .cg-grid figcaption { font-size: 18px; text-align: center; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] .cg-sc { flex: 0 1 320px; min-width: 220px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .cg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .6em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cg-grid { flex: 0 0 225px; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .cg-sc { flex: 0 0 290px; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; gap: .2em; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; line-height: 1.25; padding: .3em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .cg-lab { font-size: 18px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; n: number; a: number[]; b: number[]; tA: number; tB: number }

const B1A = [0, 2, 8, 1, 6, 9, 0, 1, 3], B1B = [1, 1, 6, 0, 5, 8, 1, 2, 0];
// 6 × 6 channels for the larger presets
const NOISE_A = [1, 2, 0, 1, 3, 2, 2, 1, 1, 0, 2, 3, 0, 3, 2, 1, 1, 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 1, 2, 2, 1, 0, 3, 1, 0];
const NOISE_B = [2, 0, 1, 3, 1, 0, 1, 3, 0, 2, 2, 1, 3, 1, 1, 0, 2, 3, 0, 2, 1, 3, 0, 2, 1, 2, 3, 0, 2, 1, 0, 1, 2, 1, 3, 2];
const OUT_A = NOISE_A.slice(), OUT_B = NOISE_B.slice();
OUT_A[14] = 20; OUT_B[14] = 20;
const LEFT = Array.from({ length: 36 }, (_, i) => (i % 6 < 3 ? 8 : 1) + (NOISE_A[i] % 2));
const RIGHT = Array.from({ length: 36 }, (_, i) => (i % 6 >= 3 ? 8 : 1) + (NOISE_B[i] % 2));

const PRESETS: Preset[] = [
  { id: "b1", label: "exercise B1", n: 3, a: B1A, b: B1B, tA: 2, tB: 2 },
  { id: "offset", label: "add an offset to B (B3)", n: 3, a: B1A, b: B1B.map(v => v + 10), tA: 2, tB: 2 },
  { id: "inside", label: "A inside B (B4)", n: 3, a: [0, 0, 0, 0, 9, 0, 0, 6, 0], b: [0, 5, 5, 0, 5, 5, 0, 5, 5], tA: 0, tB: 0 },
  { id: "outlier", label: "one bright outlier carries r", n: 6, a: OUT_A, b: OUT_B, tA: 0, tB: 0 },
  { id: "anti", label: "anti-correlated: r < 0", n: 6, a: LEFT, b: RIGHT, tA: 4, tB: 4 },
  { id: "nothr", label: "no thresholds: M1 = M2 = 1", n: 6, a: NOISE_A.map(v => v + 4), b: NOISE_B.map(v => v + 4), tA: 0, tB: 0 },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 24 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let n = p0.n, tA = p0.tA, tB = p0.tB;

  const fa = plotFigure(host, "channel A (green)", "cg-grid");
  const fb = plotFigure(host, "channel B (magenta)", "cg-grid");
  const edit = (vals: Int32Array) => { for (let i = 0; i < vals.length; i++) vals[i] = Math.max(0, Math.min(99, vals[i])); };
  const gA = new GridView(fa, n, n, p0.a, { cell: 40, display: "int", lo: 0, edit: "plusminus", onEdit: v => { edit(v); gA.setValues(v); unp(); update(); } });
  const gB = new GridView(fb, n, n, p0.b, { cell: 40, display: "int", lo: 0, edit: "plusminus", onEdit: v => { edit(v); gB.setValues(v); unp(); update(); } });
  const sf = plotFigure(host, "scatter plot of the pixel pairs", "cg-sc");
  const pl = new Plot(sf, { width: 340, height: 340, x: [0, 10], y: [0, 10], xlabel: "A", ylabel: "B", font: FS, margin: { l: 2.6 * FS, b: 2.6 * FS, t: 10, r: 12 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const sA = slider(panel, "tA", 0, 20, 0.5, tA, v => { tA = v; unp(); update(); });
  const sB = slider(panel, "tB", 0, 30, 0.5, tB, v => { tB = v; unp(); update(); });
  htmlEl("div", "cg-lab", "presets (click a cell: upper half +1, lower half −1)", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => { load(PRESETS[0]); pb.forEach((b, i) => b.classList.toggle("active", i === 0)); });
  pb[PRESETS.indexOf(p0)].classList.add("active");

  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function load(p: Preset): void {
    n = p.n; tA = p.tA; tB = p.tB;
    gA.resize(n, n, p.a); gB.resize(n, n, p.b);
    sA.set(tA); sB.set(tB);
    update();
  }

  function update(): void {
    const a = Array.from(gA.values), b = Array.from(gB.values);
    const f = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "–");
    const k = kCoefficients(a, b), m = manders(a, b, tA, tB);
    ro.textContent = `Pearson r      ${f(pearsonMasked(a, b))}\nSpearman       ${f(spearmanMasked(a, b))}\noverlap        ${f(overlapCoefficient(a, b))}\nk1 = ΣAB/ΣA²   ${f(k.k1)}\nk2 = ΣAB/ΣB²   ${f(k.k2)}\nM1 (A > ${tA}, B > ${tB})  ${f(m.m1)}\nM2             ${f(m.m2)}`;
    const marksA: CellMark[] = [], marksB: CellMark[] = [];
    for (let i = 0; i < a.length; i++) {
      const ia = a[i] > tA, ib = b[i] > tB;
      if (ia && ib) { marksA.push({ index: i, color: "var(--qb-c2)" }); marksB.push({ index: i, color: "var(--qb-c2)" }); }
      else {
        if (ia) marksA.push({ index: i, fill: true, color: "#2ca02c" });
        if (ib) marksB.push({ index: i, fill: true, color: "#c040c0" });
      }
    }
    gA.setMarks(marksA); gB.setMarks(marksB);
    const hi = Math.max(10, ...a, ...b, tA, tB) * 1.08;
    pl.clear();
    pl.setDomain([-0.04 * hi, hi], [-0.04 * hi, hi]);
    pl.vline(tA, { color: "#2ca02c", width: 3 });
    pl.hline(tB, { color: "#c040c0", width: 3 });
    const cols = a.map((v, i) => (v > tA && b[i] > tB ? "var(--qb-c2)" : "var(--qb-c1)"));
    pl.scatter(a, b, { colors: cols, r: 7, opacity: 0.75 });
    pl.text(tA, hi * 0.97, " tA", { color: "#2ca02c", anchor: "start", size: FS * 0.85 });
    pl.text(hi * 0.97, tB, "tB", { color: "#c040c0", anchor: "end", size: FS * 0.85 });
  }

  update();
}

autoMount(NAME, mount);
