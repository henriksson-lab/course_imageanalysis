// Feature importance of a random forest on a table of object measurements. Data: a seeded
// synthetic population of 400 segmented cells, two phenotypes (interphase / mitotic), with
// eight features: area, perimeter and equivalent diameter (three versions of "size"),
// intensity, texture, solidity, and two pure-noise features. Left: a scatter plot of two
// chosen features, coloured by class. Middle: for every feature in use, the impurity
// importance (mean decrease in Gini, as a share of the total) and the permutation
// importance (how much the out-of-bag error rises when the column is shuffled). Toggles:
// include/exclude features, add an exact duplicate of area, permute the size block
// (area, perimeter, eq. diameter, duplicate) together as one group.
// Auto-mounts on <div data-demo="featimp">. Options: data-preset="texture|duplicate|block|blockgroup|noise".
import { htmlEl, slider, presets, checkbox, autoMount, svgEl, releaseFocus, Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { QUALITATIVE } from "../lib/colour";
import { population, FEATURES, CLASS_NAMES } from "./population";
import { analyse, ImportanceResult } from "./analysis";

const NAME = "featimp";

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; align-items: flex-start; }
[data-demo="${NAME}"] .fi-left { flex: 0 1 370px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .fi-mid { flex: 1 1 420px; min-width: 280px; max-width: 520px; }
[data-demo="${NAME}"] .fi-bars { width: 100%; display: block; }
[data-demo="${NAME}"] .fi-bars text { font-size: 16px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 250px; gap: .35em; }
[data-demo="${NAME}"] .fi-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .fi-feats { display: grid; grid-template-columns: 1fr 1fr; gap: .1em .6em; font-size: .9em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-line; }
[data-demo="${NAME}"] select { font: inherit; font-size: .9em; }
[data-demo="${NAME}"] .fi-axes { display: flex; gap: .6em; flex-wrap: wrap; font-size: .9em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; use: string[]; dup: boolean; group: boolean }
const ALL = FEATURES.map(f => f.id);
const PRESETS: Preset[] = [
  { id: "texture", label: "texture decides", use: ALL, dup: false, group: false },
  { id: "duplicate", label: "duplicate area: credit splits", use: ["area", "int", "sol", "noise", "coin"], dup: true, group: false },
  { id: "block", label: "size block: permutation hides it", use: ["area", "perim", "eqd", "int", "sol", "noise", "coin"], dup: false, group: false },
  { id: "blockgroup", label: "permute the block together", use: ["area", "perim", "eqd", "int", "sol", "noise", "coin"], dup: false, group: true },
  { id: "noise", label: "noise: impurity vs permutation", use: ALL, dup: false, group: false },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { use: [...p0.use], dup: p0.dup, group: p0.group, mtry: 3, xf: 0, yf: 4 };
  const pop = population({ seed: 1 });

  // ---- left: scatter
  const left = htmlEl("div", "fi-left", "", host);
  const axes = htmlEl("div", "fi-axes", "", left);
  const mkSel = (label: string, val: number, on: (v: number) => void) => {
    const lab = htmlEl("label", "", label + " ", axes);
    const sel = htmlEl("select", "", "", lab);
    FEATURES.forEach((f, i) => { const o = htmlEl("option", "", f.short, sel); o.value = String(i); });
    sel.value = String(val);
    sel.addEventListener("change", () => { on(Number(sel.value)); sel.blur(); });
    releaseFocus(sel);
    return sel;
  };
  mkSel("x:", st.xf, v => { st.xf = v; drawScatter(); });
  mkSel("y:", st.yf, v => { st.yf = v; drawScatter(); });
  const sc = new Plot(left, { width: 400, height: 360, x: [0, 1], y: [0, 1], margin: { l: 70 }, ariaLabel: "scatter plot of two features" });
  const colors = Array.from(pop.y, c => QUALITATIVE[c]);
  const dots = sc.scatter([0], [0], { r: 4, opacity: 0.75 });
  function drawScatter(): void {
    const xs = pop.cols[st.xf], ys = pop.cols[st.yf];
    const ext = (a: Float64Array): [number, number] => { let lo = Infinity, hi = -Infinity; for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); } const pad = 0.05 * (hi - lo || 1); return [lo - pad, hi + pad]; };
    sc.setDomain(ext(xs), ext(ys));
    sc.setLabels({ xlabel: FEATURES[st.xf].label, ylabel: FEATURES[st.yf].label });
    dots.set(xs, ys, colors);
  }
  sc.legend(CLASS_NAMES.map((n, c) => ({ label: n, color: QUALITATIVE[c], kind: "dot" as const })), "tr");

  // ---- middle: importance bars
  const mid = htmlEl("div", "fi-mid", "", host);
  const BW = 520;
  const bars = svgEl("svg", { viewBox: `0 0 ${BW} 340`, class: "fi-bars", role: "img", "aria-label": "feature importances" }, mid);

  // ---- panel
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fi-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "fi-lab", "features used", panel);
  const featBox = htmlEl("div", "fi-feats", "", panel);
  const featCbs = FEATURES.map(f => checkbox(featBox, f.short, st.use.includes(f.id), v => {
    st.use = v ? [...st.use, f.id] : st.use.filter(x => x !== f.id);
    run();
  }));
  const dupCb = checkbox(panel, "add an exact copy of area", st.dup, v => { st.dup = v; run(); });
  const grpCb = checkbox(panel, "permute the size block together", st.group, v => { st.group = v; run(); });
  const mS: Slider = slider(panel, "m", 1, 9, 1, st.mtry, v => { st.mtry = v; run(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function drawBars(r: ImportanceResult): void {
    bars.replaceChildren();
    const rows = r.names.length + (r.groupPerm !== null ? 1 : 0), rh = 32, top = 46;
    const H = top + rows * rh + 10;
    bars.setAttribute("viewBox", `0 0 ${BW} ${H}`);
    const nameW = 130, colW = (BW - nameW - 20) / 2;
    const maxM = Math.max(0.05, ...r.mdi), maxP = Math.max(0.02, ...r.perm, r.groupPerm ?? 0);
    const head = (x: number, t: string, t2: string) => {
      const a = svgEl("text", { x, y: 18, "text-anchor": "start" }, bars); a.textContent = t; a.style.setProperty("font-weight", "700");
      const b = svgEl("text", { x, y: 37, "text-anchor": "start" }, bars); b.textContent = t2; b.style.setProperty("fill", "var(--qb-muted)"); b.style.setProperty("font-size", "14px");
    };
    head(nameW + 6, "impurity", "share of Gini decrease");
    head(nameW + colW + 16, "permutation", "rise of OOB error");
    const row = (i: number, name: string, m: number | null, p: number, block: boolean) => {
      const y = top + i * rh;
      const t = svgEl("text", { x: nameW - 6, y: y + 20, "text-anchor": "end" }, bars); t.textContent = name;
      if (block) t.style.setProperty("font-style", "italic");
      if (m !== null) {
        const wm = Math.max(0, m / maxM) * (colW - 60);
        const rm = svgEl("rect", { x: nameW + 6, y: y + 6, width: wm, height: rh - 12 }, bars); rm.style.setProperty("fill", QUALITATIVE[0]);
        const tm = svgEl("text", { x: nameW + 10 + wm, y: y + 21 }, bars); tm.textContent = (100 * m).toFixed(0) + "%"; tm.style.setProperty("font-size", "14px");
      }
      const x0 = nameW + colW + 16, wp = Math.max(0, p / maxP) * (colW - 70);
      const rp = svgEl("rect", { x: x0, y: y + 6, width: wp, height: rh - 12 }, bars); rp.style.setProperty("fill", QUALITATIVE[3]);
      const tp = svgEl("text", { x: x0 + wp + 4, y: y + 21 }, bars); tp.textContent = (p >= 0 ? "+" : "") + (100 * p).toFixed(1); tp.style.setProperty("font-size", "14px");
    };
    r.names.forEach((n, j) => row(j, n, r.mdi[j], r.perm[j], false));
    if (r.groupPerm !== null) row(r.names.length, "size block", null, r.groupPerm, true);
    const ax = svgEl("line", { x1: nameW + colW + 16, x2: nameW + colW + 16, y1: top, y2: H - 6 }, bars);
    ax.style.setProperty("stroke", "var(--qb-axis)");
    const ax2 = svgEl("line", { x1: nameW + 6, x2: nameW + 6, y1: top, y2: H - 6 }, bars);
    ax2.style.setProperty("stroke", "var(--qb-axis)");
  }

  function run(): void {
    if (!st.use.length) { bars.replaceChildren(); ro.textContent = "choose at least one feature"; return; }
    const d = st.use.length + (st.dup && st.use.includes("area") ? 1 : 0);
    mS.input.max = String(d);
    if (st.mtry > d) { st.mtry = d; mS.set(d); }
    const r = analyse(st.use, st.dup, st.group, st.mtry);
    drawBars(r);
    ro.textContent = `100 trees, m = ${Math.min(st.mtry, d)} of d = ${d}\nOOB error ${(100 * r.oob).toFixed(1)} %\npermutation: points of OOB error`;
  }

  function load(p: Preset): void {
    st.use = [...p.use]; st.dup = p.dup; st.group = p.group; st.mtry = 3; mS.set(3);
    FEATURES.forEach((f, i) => (featCbs[i].checked = st.use.includes(f.id)));
    dupCb.checked = st.dup; grpCb.checked = st.group;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    run();
  }

  drawScatter();
  load(p0);
}

autoMount(NAME, mount);
