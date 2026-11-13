// Three images, two controls: correcting sensitised emission. A seeded field of cells of
// five classes (donor only, acceptor only, free donor + acceptor without FRET, a donor–
// acceptor fusion with efficiency E, and "binding" cells where half the donors are paired),
// with per-cell expression levels, Poisson noise and a background offset. Tiles: the three
// raw images IDD, IDA (the FRET channel), IAA. Result: Fc = IDA − d IDD − a IAA, or an index
// (Fc/IAA, Fc/√(IDD IAA)) or the efficiency E_app = Fc/(Fc + G IDD), on a diverging map with
// 0 = white, per-cell values printed on the cells (ratio of sums over the cell). Scatter: the
// current control (IDA against IDD for donor-only pixels, or IDA against IAA for acceptor-
// only pixels) with the line of the current coefficient. "Measure from the controls" sets d
// and a to the slopes through the origin. data-layout="full" adds a per-cell table.
//
// <div data-demo="threecube" data-preset="raw|bleed|both|measured|nofret|dsmall|lowE|ratio|shift|nobg"
//      data-layout="full" data-photons="300"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { diverging, scalarToRGBA, ColorMap } from "../lib/colour";
import * as F from "../lib/fret";
import { Tile, tileCss, injectStyle, fmt, debounce, filterPresets, hideControl } from "./view";

const NAME = "threecube";
const Q = `.qb-demo[data-demo="${NAME}"]`;
const CSS = tileCss(NAME) + `
${Q} { gap: .7em; align-items: flex-start; }
${Q} .tc-main { flex: 0 1 340px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
${Q} .tc-tiles { display: flex; flex-direction: row; gap: .3em; align-items: flex-end; }
${Q} .tc-tiles .fr-fig { flex: 1 1 0; }
${Q} .tc-tiles .fr-fig figcaption { font-size: 14px; }
${Q} .tc-res { display: flex; flex-direction: column; gap: .25em; }
${Q} .tc-res .fr-wrap { max-width: calc(100vh - 340px); margin: 0 auto; }
${Q} .tc-sc { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${Q} .tc-sc .qb-readout { font-size: 13px; }
${Q} .qb-panel { flex: 0 1 310px; max-width: 340px; }
${Q} .tc-more > summary { cursor: pointer; color: var(--qb-muted); font-size: 14px; }
${Q} .tc-more[open] { display: flex; flex-direction: column; gap: .35em; }
${Q} .tc-table { width: 100%; border-collapse: collapse; font-size: 15px; }
${Q} .tc-table td, ${Q} .tc-table th { padding: .1em .4em; border-bottom: 1px solid var(--qb-grid); text-align: right; }
${Q} .tc-table td:first-child, ${Q} .tc-table th:first-child { text-align: left; }
${Q} .tc-full { flex: 1 1 100%; }
${Q} .tc-full > summary { cursor: pointer; font-size: 15px; }
${Q} .tc-modes { display: flex; flex-wrap: wrap; gap: .25em; }
.reveal ${Q} { flex-wrap: nowrap; }
.reveal ${Q} .tc-main { display: contents; }
.reveal ${Q} .tc-tiles { flex: 0 0 150px; flex-direction: column; gap: .25em; align-items: stretch; }
.reveal ${Q} .tc-tiles .fr-fig figcaption { font-size: 18px; }
.reveal ${Q} .tc-res { flex: 0 0 300px; }
.reveal ${Q} .tc-res .fr-wrap { max-width: none; }
.reveal ${Q} .tc-sc { flex: 0 0 300px; }
.reveal ${Q} .tc-sc .qb-readout { font-size: 15px; }
.reveal ${Q} .qb-panel { flex: 0 0 340px; }
.reveal ${Q} .tc-more > summary { font-size: 17px; }
`;

type Mode = "Fc" | "perA" | "index" | "E";
const MODES: [Mode, string, number][] = [["Fc", "Fc", 1], ["perA", "Fc / IAA", 1], ["index", "Fc / √(IDD·IAA)", 1], ["E", "E_app", 0.5]];
interface State {
  d: number; a: number; G: number; bg: number;
  subD: boolean; subA: boolean; shift: boolean;
  photons: number; E: number; extra: number;
  mode: Mode; control: "donor" | "acceptor";
}
const OFFSET = 100;
const DEFAULT: State = { d: 0, a: 0, G: 2, bg: OFFSET, subD: true, subA: true, shift: false, photons: 300, E: 0.3, extra: 0, mode: "Fc", control: "donor" };
const PRESETS: { id: string; label: string; s: Partial<State>; measure?: boolean }[] = [
  { id: "raw", label: "raw FRET channel: everything lights up", s: { d: 0, a: 0 } },
  { id: "bleed", label: "remove bleed-through only", s: { d: 0.5, a: 0, control: "donor" } },
  { id: "both", label: "remove cross-excitation too", s: { d: 0.5, a: 0.1, control: "acceptor" } },
  { id: "measured", label: "measured from the controls", s: { control: "donor" }, measure: true },
  { id: "nofret", label: "the no-FRET cells are zero", s: { mode: "E" }, measure: true },
  { id: "dsmall", label: "d too small: false FRET follows the donor", s: { d: 0.4, a: 0.1, mode: "Fc", control: "donor" } },
  { id: "lowE", label: "low E, few photons: the difference is noise", s: { E: 0.05, photons: 40, mode: "E" }, measure: true },
  { id: "ratio", label: "indices change with D:A, E does not", s: { extra: 2, mode: "perA" }, measure: true },
  { id: "shift", label: "one pixel shift: edges", s: { shift: true, mode: "Fc" }, measure: true },
  { id: "nobg", label: "forgot the background", s: { bg: 0, mode: "E" }, measure: true },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const full = host.dataset.layout === "full";
  const photons0 = Number(host.dataset.photons) || DEFAULT.photons;
  const pre = PRESETS.find(p => p.id === host.dataset.preset);
  const base = (): State => ({ ...DEFAULT, photons: photons0 });
  const st: State = { ...base(), ...(pre?.s ?? {}) };

  // ---- layout ----
  const main = htmlEl("div", "tc-main", "", host);
  const tiles = htmlEl("div", "tc-tiles", "", main);
  const tDD = new Tile(tiles, "IDD"), tDA = new Tile(tiles, "IDA (FRET channel)"), tAA = new Tile(tiles, "IAA");
  const res = htmlEl("div", "tc-res", "", main);
  const modeRow = htmlEl("div", "tc-modes", "", res);
  const modeBtns = MODES.map(([m, lab]) => button(modeRow, lab, () => { st.mode = m; manual(); }));
  const tRes = new Tile(res, "");
  const scBox = htmlEl("div", "tc-sc", "", host);
  const ctlRow = htmlEl("div", "tc-modes", "", scBox);
  const ctlBtns = [button(ctlRow, "donor-only control", () => { st.control = "donor"; manual(); }), button(ctlRow, "acceptor-only control", () => { st.control = "acceptor"; manual(); })];
  const sc = new Plot(scBox, { width: host.closest(".reveal") ? 300 : 340, height: host.closest(".reveal") ? 290 : 330, x: [0, 1], y: [0, 1], xlabel: "IDD", ylabel: "IDA", xticks: 4, yticks: 4, margin: { l: 70, r: 12, t: 10, b: 56 } });
  const pts = sc.scatter([], [], { color: "#1f77b4", r: 2.6, opacity: 0.35 });
  const fitLine = sc.line([0, 1], [0, 0], { color: "#d62728", width: 4 });
  const scNote = htmlEl("div", "qb-readout", "", scBox);

  const panel = htmlEl("div", "qb-panel", "", host);
  const PL = filterPresets(host, PRESETS);
  const presetBtns = presets(panel, PL.map(p => ({ label: p.label, apply: () => { Object.assign(st, base(), p.s); openMore(p.s); newData(); if (p.measure) measure(); sync(); draw(); } })));
  const dS = slider(panel, "bleed-through d", 0, 1, 0.01, st.d, v => { st.d = v; manual(); });
  const aS = slider(panel, "cross-excitation a", 0, 0.4, 0.005, st.a, v => { st.a = v; manual(); });
  const row = htmlEl("div", "tc-modes", "", panel);
  const cD = checkbox(row, "subtract d·IDD", st.subD, v => { st.subD = v; manual(); });
  const cA = checkbox(row, "subtract a·IAA", st.subA, v => { st.subA = v; manual(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "measure d, a from the controls", () => { measure(); sync(); manual(); });
  button(brow, "reset", () => { Object.assign(st, base()); presetBtns.forEach(b => b.classList.remove("active")); newData(); sync(); draw(); });
  const more = htmlEl("details", "tc-more", "", panel);
  htmlEl("summary", "", "more options (G, background, data)", more);
  const gS = slider(more, "G factor", 0.5, 4, 0.05, st.G, v => { st.G = v; manual(); });
  const bS = slider(more, "background subtracted", 0, 150, 1, st.bg, v => { st.bg = v; manual(); });
  const regen = debounce(() => { newData(); draw(); }, 120);
  const phS = slider(more, "photons", 10, 1000, 10, st.photons, v => { st.photons = v; presetOff(); regen(); });
  const eS = slider(more, "true E (fusion)", 0, 0.8, 0.01, st.E, v => { st.E = v; presetOff(); regen(); });
  const xS = slider(more, "extra free acceptor", 0, 3, 0.1, st.extra, v => { st.extra = v; presetOff(); regen(); });
  const cS = checkbox(more, "shift IDA by 1 px", st.shift, v => { st.shift = v; presetOff(); newData(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", scBox);
  if (host.dataset.compact !== undefined) [phS, eS, xS].forEach(hideControl);
  let table: HTMLTableElement | null = null;
  if (full) {
    const tb = htmlEl("details", "tc-full", "", host);
    htmlEl("summary", "", "per-cell table", tb);
    table = htmlEl("table", "tc-table", "", tb);
  }

  // presets that change a setting kept under "more options" open that section
  function openMore(s: Partial<State>): void { if (["G", "bg", "photons", "E", "extra", "shift"].some(k => k in s)) more.open = true; }
  if (pre) openMore(pre.s);
  const presetOff = () => presetBtns.forEach(b => b.classList.remove("active"));
  const manual = () => { presetOff(); draw(); };
  function sync(): void {
    dS.set(st.d); aS.set(st.a); gS.set(st.G); bS.set(st.bg); phS.set(st.photons); eS.set(st.E); xS.set(st.extra);
    cD.checked = st.subD; cA.checked = st.subA; cS.checked = st.shift;
  }

  // ---- data ----
  let field: F.Field, M: { IDD: Float32Array; IDA: Float32Array; IAA: Float32Array };
  let donorMask: Uint8Array, accMask: Uint8Array, cellPix: number[][] = [], cent: [number, number][] = [];
  function newData(): void {
    field = F.fretField({ seed: 5, E: st.E, extraAcceptor: st.extra });
    M = F.measureField(field, { photons: st.photons, offset: OFFSET, seed: 3, shift: st.shift ? 1 : 0 });
    const n = field.w * field.h;
    donorMask = new Uint8Array(n); accMask = new Uint8Array(n);
    cellPix = Array.from({ length: field.count }, () => []);
    for (let i = 0; i < n; i++) {
      const L = field.labels[i];
      if (!L) continue;
      cellPix[L - 1].push(i);
      if (field.cls[L - 1] === "donor") donorMask[i] = 1;
      if (field.cls[L - 1] === "acceptor") accMask[i] = 1;
    }
    cent = cellPix.map(p => { let sx = 0, sy = 0; for (const i of p) { sx += i % field.w; sy += Math.floor(i / field.w); } return [sx / p.length + 0.5, sy / p.length + 0.5]; });
  }
  function measure(): void {
    const b = st.bg;
    const sub = (v: Float32Array) => Float32Array.from(v, x => x - b);
    const IDD = sub(M.IDD), IDA = sub(M.IDA), IAA = sub(M.IAA);
    st.d = Math.round(F.slopeThroughOriginMasked(IDD, IDA, donorMask) * 1000) / 1000;
    st.a = Math.round(F.slopeThroughOriginMasked(IAA, IDA, accMask) * 1000) / 1000;
    st.subD = st.subA = true;
  }

  // ---- drawing ----
  const greyTile = (t: Tile, v: Float32Array, hi: number) => t.grey(v, field.w, field.h, 0, hi);
  function value(mode: Mode, Fc: number, IDD: number, IAA: number): number {
    if (mode === "Fc") return Fc;
    if (mode === "perA") return F.perAcceptorIndex(Fc, IAA);
    if (mode === "index") return F.nfretIndex(Fc, IDD, IAA);
    return F.eFret(Fc, IDD, st.G);
  }
  function draw(): void {
    const n = field.w * field.h, b = st.bg, d = st.subD ? st.d : 0, a = st.subA ? st.a : 0;
    const hi = Math.max(1, 1.6 * st.photons + OFFSET);
    greyTile(tDD, M.IDD, hi); greyTile(tDA, M.IDA, hi); greyTile(tAA, M.IAA, hi);
    // per-pixel map
    const v = new Float32Array(n), thr = 0.12 * st.photons;
    for (let i = 0; i < n; i++) {
      const IDD = M.IDD[i] - b, IDA = M.IDA[i] - b, IAA = M.IAA[i] - b;
      if (IDD + IAA < thr + (OFFSET - b) * 0) { v[i] = NaN; continue; }
      const Fc = F.correctFc(IDA, IDD, IAA, d, a);
      v[i] = value(st.mode, Fc, IDD, IAA);
    }
    const m = MODES.find(x => x[0] === st.mode)!;
    const lim = st.mode === "Fc" ? 0.6 * st.photons : m[2];
    const map: ColorMap = diverging;
    tRes.rgba(scalarToRGBA(v, { map, lo: -lim, hi: lim, nan: [40, 40, 40, 255] }), field.w, field.h);
    tRes.setCaption(`${m[1]}  (white = 0, red > 0, blue < 0; ±${fmt(lim, st.mode === "Fc" ? 0 : 1)})`);
    // per-cell values (sums over the cell)
    tRes.clear();
    const rows: string[] = [];
    for (let k = 0; k < field.count; k++) {
      let sDD = 0, sDA = 0, sAA = 0;
      for (const i of cellPix[k]) { sDD += M.IDD[i] - b; sDA += M.IDA[i] - b; sAA += M.IAA[i] - b; }
      const Fc = F.correctFc(sDA, sDD, sAA, d, a), val = value(st.mode, Fc, sDD, sAA);
      const cls = field.cls[k];
      const show = cls === "acceptor" && (st.mode === "E" || st.mode === "index") ? "—" : st.mode === "Fc" ? fmt(Fc / cellPix[k].length, 0) : fmt(val, 2);
      tRes.label(cent[k][0], cent[k][1], show, 9);
      if (table) {
        const truth = field.trueE[k] * field.trueF[k];
        rows.push(`<tr><td>${k + 1}: ${F.CLASS_NAMES[cls]}</td><td>${fmt(Fc / cellPix[k].length, 1)}</td><td>${fmt(F.perAcceptorIndex(Fc, sAA), 3)}</td><td>${fmt(F.nfretIndex(Fc, sDD, sAA), 3)}</td><td>${cls === "acceptor" ? "—" : fmt(F.eFret(Fc, sDD, st.G), 3)}</td><td>${cls === "acceptor" ? "—" : fmt(truth, 3)}</td><td>${cls === "acceptor" || cls === "donor" ? "—" : fmt(sAA / Math.max(1, sDD), 2)}</td></tr>`);
      }
    }
    if (table) table.innerHTML = `<tr><th>cell</th><th>mean Fc</th><th>Fc/IAA</th><th>Fc/√(IDD·IAA)</th><th>E_app</th><th>true f·E</th><th>ΣIAA/ΣIDD</th></tr>` + rows.join("");
    // scatter of the current control
    const mask = st.control === "donor" ? donorMask : accMask, X = st.control === "donor" ? M.IDD : M.IAA;
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) if (mask[i] && i % 2 === 0) { xs.push(X[i] - b); ys.push(M.IDA[i] - b); }
    const xmax = Math.max(10, ...xs) * 1.05, ymax = Math.max(10, ...ys, 0.6 * xmax) * 1.05;
    sc.setDomain([Math.min(0, -0.05 * xmax), xmax], [Math.min(0, -0.05 * ymax), ymax]);
    sc.setLabels({ xlabel: st.control === "donor" ? "IDD (donor-only cells)" : "IAA (acceptor-only cells)", ylabel: "IDA" });
    pts.set(xs, ys);
    const k = st.control === "donor" ? d : a;
    fitLine.set([0, xmax], [0, k * xmax]);
    const slope = F.slopeThroughOriginMasked(Float32Array.from(X, x => x - b), Float32Array.from(M.IDA, x => x - b), mask);
    scNote.textContent = `slope through 0: ${fmt(slope, 3)}  (line: ${st.control === "donor" ? "d" : "a"} = ${fmt(k, 3)})`;
    modeBtns.forEach((bt, i) => bt.classList.toggle("active", MODES[i][0] === st.mode));
    ctlBtns.forEach((bt, i) => bt.classList.toggle("active", (i === 0) === (st.control === "donor")));
    // summary readout: mean E_app of the no-FRET and fusion cells
    const classMean = (c: F.CellClass) => {
      let sDD = 0, sDA = 0, sAA = 0;
      for (let kk = 0; kk < field.count; kk++) if (field.cls[kk] === c) for (const i of cellPix[kk]) { sDD += M.IDD[i] - b; sDA += M.IDA[i] - b; sAA += M.IAA[i] - b; }
      return F.eFret(F.correctFc(sDA, sDD, sAA, d, a), sDD, st.G);
    };
    ro.textContent = [
      `E_app, ratio of sums per class:`,
      `  donor only ${fmt(classMean("donor"), 3)}`,
      `  no FRET    ${fmt(classMean("free"), 3)}`,
      `  fusion     ${fmt(classMean("fusion"), 3)} (true ${fmt(st.E, 2)})`,
      `  binding    ${fmt(classMean("binding"), 3)} (true 0.5×0.4 = 0.20)`,
    ].join("\n");
  }

  newData();
  if (pre?.measure) measure();
  sync();
  const pi = PL.findIndex(p => p.id === host.dataset.preset);
  if (pi >= 0) presetBtns[pi].classList.add("active");
  draw();
  void svgEl; void ({} as Slider);
}

autoMount(NAME, mount);
