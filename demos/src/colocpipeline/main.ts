// What the microscope does to colocalisation: the two-channel forward model and the
// corrections in pipeline order. Left: the truth (true spot positions; yellow links = pairs);
// middle: the measured merge; right: the corrected merge. Confound sliders: true pair
// distance, share of paired spots, a background shared by both channels, bleed-through,
// lateral chromatic shift, photons. Corrections: subtract background (white top-hat),
// compensate (B − ŝA), register (shift from a two-colour bead image). Table: Pearson r,
// Manders M1/M2 with Costes thresholds, ICQ and the object fraction within 130 nm (with its
// chance level) for an ideal microscope (same PSF, photons and noise, no confounds), the
// measurement and the corrected images. Tab "resolution sweep": r and the object fraction
// against the true pair distance, with the analytic Gaussian overlap.
// Images and r update live; the full table 120 ms after the last change (≈ 35 ms per column).
// Auto-mounts on <div data-demo="colocpipeline">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { S, mergeGreenMagenta } from "../lib/twochannel";
import { pearsonMasked } from "../lib/coloc";
import { quantile } from "../lib/stats";
import { PipeOptions, DEFAULT_PIPE, runPipeline, PipeResult, readout, Readout, Stage, mask, sweep, PIXEL_NM, OFFSET, fwhmA } from "./model";

const NAME = "colocpipeline";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cp-left { flex: 1 1 560px; min-width: 280px; max-width: 640px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .cp-tabs { display: flex; flex-wrap: wrap; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .cp-imgs { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .cp-imgs figure { flex: 1 1 0; min-width: 0; margin: 0; }
.qb-demo[data-demo="${NAME}"] .cp-imgs canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] table.cp-tab { border-collapse: collapse; font-size: 16px; font-variant-numeric: tabular-nums; width: 100%; }
.qb-demo[data-demo="${NAME}"] table.cp-tab td, .qb-demo[data-demo="${NAME}"] table.cp-tab th { padding: .1em .45em; text-align: right; border-bottom: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] table.cp-tab td:first-child, .qb-demo[data-demo="${NAME}"] table.cp-tab th:first-child { text-align: left; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .cp-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .cp-note { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .6em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-left { flex: 0 0 560px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] table.cp-tab { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 560px; gap: .12em; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-sliders { display: grid; grid-template-columns: 1fr 1fr; column-gap: .8em; row-gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-lab, .reveal .qb-demo[data-demo="${NAME}"] .cp-note { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Tab = "pipe" | "sweep";
interface State extends PipeOptions { tab: Tab }
const DEFAULT: State = { ...DEFAULT_PIPE, tab: "pipe" } as State;

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "clean", label: "clean", s: {} },
  { id: "redpsf", label: "all paired at 0 nm: the red PSF is wider", s: { pair: 1, extraB: 0 } },
  { id: "shift", label: "100 nm chromatic shift", s: { shiftXNm: 100 } },
  { id: "register", label: "register with beads", s: { shiftXNm: 100, register: true } },
  { id: "bleed", label: "30 % bleed-through: compensate", s: { bleed: 0.3, subtract: true, compensate: true, sHat: 0.3 } },
  { id: "over", label: "over-compensation: anti-colocalisation", s: { pair: 0, bleed: 0.3, subtract: true, compensate: true, sHat: 0.6 } },
  { id: "uneven", label: "uneven background shared by both", s: { uneven: 40 } },
  { id: "all", label: "everything at once (now correct step by step)", s: { uneven: 40, bleed: 0.3, shiftXNm: 100 } },
  { id: "sweep", label: "0 nm and 50 nm look the same", s: { tab: "sweep" } },
];

const KEYS = ["dNm", "pair", "uneven", "bleed", "shiftXNm", "shiftYNm", "photons", "tophatR", "sHat"] as const;

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 24 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  if (!p0 && host.dataset.tab) st.tab = host.dataset.tab as Tab;

  const left = htmlEl("div", "cp-left", "", host);
  const tabsEl = htmlEl("div", "cp-tabs", "", left);
  const tabBtns = ([["pipe", "pipeline"], ["sweep", "resolution sweep"]] as [Tab, string][]).map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(true); }));
  const imgs = htmlEl("div", "cp-imgs", "", left);
  const mk = (cap: string) => { const f = htmlEl("figure", "", "", imgs); htmlEl("figcaption", "", cap, f); const c = htmlEl("canvas", "", "", f); c.width = S; c.height = S; return c; };
  const cvT = mk("truth"), cvM = mk("measured"), cvC = mk("corrected");
  const table = htmlEl("table", "cp-tab", "", left);
  const sf = plotFigure(left, "", "qb-figure");
  const pl = new Plot(sf, { width: 560, height: 330, x: [0, 500], y: [0, 1.05], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1], xlabel: "true pair distance (nm)", ylabel: "r  /  object fraction", font: FS, margin: { l: 3.6 * FS, b: 2.8 * FS, t: 12, r: 18 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  const slBox = htmlEl("div", "cp-sliders", "", panel);
  const sl = {
    dNm: slider(slBox, "pair distance (nm)", 0, 500, 10, st.dNm, v => set("dNm", v)),
    pair: slider(slBox, "share paired", 0, 1, 0.05, st.pair ?? 0.5, v => set("pair", v)),
    uneven: slider(slBox, "shared background", 0, 80, 2, st.uneven, v => set("uneven", v)),
    bleed: slider(slBox, "bleed A→B", 0, 0.6, 0.05, st.bleed, v => set("bleed", v)),
    shiftXNm: slider(slBox, "shift x (nm)", -200, 200, 5, st.shiftXNm, v => set("shiftXNm", v)),
    shiftYNm: slider(slBox, "shift y (nm)", -200, 200, 5, st.shiftYNm, v => set("shiftYNm", v)),
    photons: slider(slBox, "photons/spot", 100, 3000, 50, st.photons, v => set("photons", v)),
    tophatR: slider(slBox, "top-hat radius", 2, 10, 1, st.tophatR, v => set("tophatR", v)),
    sHat: slider(slBox, "ŝ (estimated)", 0, 1, 0.05, st.sHat, v => set("sHat", v)),
  };
  const corrLab = htmlEl("div", "cp-lab", "corrections, in pipeline order", panel);
  const corr = htmlEl("div", "qb-presets", "", panel);
  const cSub = checkbox(corr, "1 subtract background", st.subtract, v => { st.subtract = v; unp(); draw(true); });
  const cCom = checkbox(corr, "2 compensate", st.compensate, v => { st.compensate = v; unp(); draw(true); });
  const cReg = checkbox(corr, "3 register", st.register, v => { st.register = v; unp(); draw(true); });
  const note = htmlEl("div", "cp-note", "", panel);
  htmlEl("div", "cp-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [];
  button(panel, "reset", () => applyPreset(null));

  function set(k: typeof KEYS[number], v: number): void { (st as unknown as Record<string, number>)[k] = v; unp(); draw(false); }
  function buildPresetButtons(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => (p.s.tab ?? "pipe") === st.tab) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => b.dataset.id = list[i].id);
  }
  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    Object.assign(st, DEFAULT, p?.s ?? {});
    for (const k of KEYS) sl[k].set((st as unknown as Record<string, number>)[k]);
    cSub.checked = st.subtract; cCom.checked = st.compensate; cReg.checked = st.register;
    draw(true);
    pb.forEach(b => b.classList.toggle("active", !!p && b.dataset.id === p.id));
  }

  // caches
  let idealKey = "", idealRead: Readout | null = null, measKey = "", measRead: Readout | null = null;
  let res: PipeResult | null = null, timer = 0;
  let sweepRes: ReturnType<typeof sweep> | null = null;

  const rng = (img: ArrayLike<number>, off: number): [number, number] => [off, off + Math.max(5, quantile(Array.from(img, v => v - off), 0.998))];
  function drawMerge(cv: HTMLCanvasElement, s: Stage, off: number): void {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(S, S);
    mergeGreenMagenta(s.a, s.b, rng(s.a, off), rng(s.b, off), id.data);
    ctx.putImageData(id, 0, 0);
  }
  function drawTruth(r: PipeResult): void {
    const ctx = cvT.getContext("2d")!, m = mask();
    const id = ctx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) { const v = m[i] ? 40 : 15; id.data[4 * i] = v; id.data[4 * i + 1] = v; id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    const t = r.truth;
    ctx.lineWidth = 0.8;
    t.partner.forEach((j, i) => { if (j >= 0) { ctx.strokeStyle = "#ffd400"; ctx.beginPath(); ctx.moveTo(t.ax[i] + 0.5, t.ay[i] + 0.5); ctx.lineTo(t.bx[j] + 0.5, t.by[j] + 0.5); ctx.stroke(); } });
    ctx.fillStyle = "#4dff4d";
    t.ax.forEach((x, i) => { ctx.beginPath(); ctx.arc(x + 0.5, t.ay[i] + 0.5, 1.3, 0, 2 * Math.PI); ctx.fill(); });
    ctx.fillStyle = "#ff5cff";
    t.bx.forEach((x, i) => { ctx.beginPath(); ctx.arc(x + 0.5, t.by[i] + 0.5, 1.0, 0, 2 * Math.PI); ctx.fill(); });
  }

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  function fillTable(cols: (Readout | null)[], live: number[]): void {
    const row = (name: string, f: (r: Readout) => string, liveVal?: (k: number) => string) =>
      `<tr><td>${name}</td>${cols.map((c, k) => `<td>${c ? f(c) : liveVal ? liveVal(k) : "…"}</td>`).join("")}</tr>`;
    table.innerHTML = `<tr><th></th><th>ideal</th><th>measured</th><th>corrected</th></tr>` +
      row("Pearson r", c => fmt(c.r), k => fmt(live[k])) +
      row("M1, M2 (Costes)", c => `${fmt(c.m1)}, ${fmt(c.m2)}${c.costesOk ? "" : "*"}`) +
      row("ICQ", c => fmt(c.icq)) +
      row("A with B ≤ 130 nm", c => `${fmt(100 * c.fObs, 0)} %`) +
      row("… by chance", c => `${fmt(100 * c.fChance, 0)} %`);
  }

  function draw(full: boolean): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["pipe", "sweep"][i] === st.tab));
    buildPresetButtons();
    const pipe = st.tab === "pipe";
    imgs.style.display = pipe ? "" : "none";
    table.style.display = pipe ? "" : "none";
    sf.style.display = pipe ? "none" : "";
    corr.style.display = pipe ? "" : "none";
    corrLab.style.display = pipe ? "" : "none";
    slBox.style.display = pipe ? "" : "none";
    if (!pipe) { drawSweep(); return; }
    res = runPipeline(st, false);
    drawTruth(res);
    drawMerge(cvM, res.measured, OFFSET);
    const offC = st.subtract ? 0 : OFFSET;
    drawMerge(cvC, res.corrected, offC);
    const m = mask();
    const live = [pearsonMasked(res.ideal.a, res.ideal.b, m), pearsonMasked(res.measured.a, res.measured.b, m), pearsonMasked(res.corrected.a, res.corrected.b, m)];
    const iKey = `${st.dNm},${st.pair},${st.photons}`, mKey = `${iKey},${st.uneven},${st.bleed},${st.shiftXNm},${st.shiftYNm}`;
    const anyCorr = st.subtract || st.compensate || st.register;
    const shiftTxt = res.estShiftNm ? `bead shift estimate: (${fmt(res.estShiftNm[0], 0)}, ${fmt(res.estShiftNm[1], 0)}) nm, true (${st.shiftXNm}, ${st.shiftYNm})` : "";
    note.textContent = [`true bleed s = ${st.bleed}`, shiftTxt, "* Costes undefined: thresholds 3 sd above the offset"].filter(Boolean).join(" · ");
    const finish = () => {
      if (!res) return;
      if (iKey !== idealKey || !idealRead) { idealRead = readout(res.ideal, OFFSET); idealKey = iKey; }
      if (mKey !== measKey || !measRead) { measRead = readout(res.measured, OFFSET); measKey = mKey; }
      const corrRead = anyCorr ? readout(res.corrected, offC) : measRead;
      fillTable([idealRead, measRead, corrRead], live);
    };
    clearTimeout(timer);
    if (full) finish();
    else {
      fillTable([iKey === idealKey ? idealRead : null, mKey === measKey ? measRead : null, null], live);
      timer = window.setTimeout(finish, 120);
    }
  }

  function drawSweep(): void {
    const ds = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500];
    sweepRes ??= sweep(ds);
    const s = sweepRes;
    pl.clear();
    pl.setDomain([0, 500], [0, 1.55]);
    pl.band([0, 0.7 * fwhmA], [0, 0], [1.55, 1.55], { fill: "var(--qb-c4)", fillOpacity: 0.12, color: "none" });
    pl.fn(d => s.analytic(d), { color: "var(--qb-c1)", width: 2.5, dash: "6 4" });
    pl.line(s.d, s.r, { color: "var(--qb-c1)", width: 3 });
    pl.scatter(s.d, s.r, { color: "var(--qb-c1)", r: 5 });
    pl.line(s.d, s.frac, { color: "var(--qb-c2)", width: 3 });
    pl.scatter(s.d, s.frac, { color: "var(--qb-c2)", r: 5 });
    pl.vline(fwhmA, { color: "var(--qb-axis)", width: 2, dash: "3 3" });
    pl.legend([{ label: "pixel r (simulated)", color: "var(--qb-c1)" }, { label: "Gaussian overlap", color: "var(--qb-c1)", dash: "6 4" }, { label: "A with B ≤ 130 nm", color: "var(--qb-c2)" }, { label: "< 0.7 FWHM", color: "var(--qb-c4)", kind: "box" }, { label: "FWHM of A", color: "var(--qb-axis)", dash: "3 3" }], "tr");
    note.textContent = `A at 520 nm, B at 670 nm, NA 1.4: FWHM of A ≈ ${fmt(fwhmA, 0)} nm; pixel ${PIXEL_NM} nm.`;
  }

  for (const k of KEYS) sl[k].set((st as unknown as Record<string, number>)[k]);
  draw(true);
  if (p0) pb.forEach(b => b.classList.toggle("active", b.dataset.id === p0.id));
}

autoMount(NAME, mount);
