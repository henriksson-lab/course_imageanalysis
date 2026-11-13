// Ripley's K, the L function and the pair-correlation function g. The pattern comes from the
// shared source (csrplay/source.ts) or is edited by hand. A probe sits on one point: the disc of
// radius r (what K counts) and the ring r ± h (what g counts), with the counts and the CSR
// expectation. Tabs: K, L − r, g (with the Thomas formula for Thomas patterns), and the bias of
// the four edge corrections averaged over 50 CSR patterns in this window.
// Auto-mounts on <div data-demo="ripleyk">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount, svgEl, checkbox } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rng } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { Spec, DEFAULT_SPEC, makePattern, makeWindow, PatternView, sourceControls, injectStyle, sourceCss, SIDE, WinKind } from "../csrplay/source";

const NAME = "ripleyk";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = sourceCss(NAME) + `
${P} .rk-view { flex: 0 1 360px; min-width: 240px; }
${P} .rk-main { flex: 1 1 360px; min-width: 260px; max-width: 520px; display: flex; flex-direction: column; gap: .3em; }
${P} .qb-panel { flex: 1 1 280px; gap: .35em; }
${P} .rk-cap { font-size: .9em; text-align: center; }
.reveal ${P} .rk-view { flex: 0 0 370px; min-width: 0; }
.reveal ${P} .rk-main { flex: 0 0 420px; min-width: 0; }
.reveal ${P} .qb-panel { flex: 1 1 320px; min-width: 0; gap: .15em; }
`;

type Tab = "K" | "L" | "g" | "bias";
type Corr = PP.KCorr;
const CORRS: Corr[] = ["none", "border", "translation", "isotropic"];
interface State { spec: Spec; tab: Tab; corr: Corr; r: number; h: number; edit: boolean; probe: number }
const DEFAULT: State = { spec: { ...DEFAULT_SPEC }, tab: "L", corr: "translation", r: 10, h: 2, edit: false, probe: -1 };
const RMAX = 40, RADII = PP.linspace(0, RMAX, 81);

const PRESETS: { id: string; label: string; s: Partial<State>; spec?: Partial<Spec> }[] = [
  { id: "csr", label: "CSR: L − r ≈ 0", s: { tab: "L" }, spec: { model: "csr" } },
  { id: "thomas", label: "Thomas: g falls to 1 at the cluster size", s: { tab: "g", r: 8 }, spec: { model: "thomas", k: 20, mu: 8, sigma: 4 } },
  { id: "remember", label: "K remembers small clusters", s: { tab: "L", r: 30 }, spec: { model: "thomas", k: 25, mu: 6, sigma: 1.5 } },
  { id: "hardcore", label: "hard core: g = 0 below h", s: { tab: "g", r: 5 }, spec: { model: "hardcore", n: 200, h: 8 } },
  { id: "colonies", label: "colonies with a hard core: two scales", s: { tab: "g", r: 4, h: 1.5 }, spec: { model: "colonies" } },
  { id: "pairs", label: "tight pairs: g at 3 µm only", s: { tab: "g", r: 3, h: 1 }, spec: { model: "pairs", pairD: 3 } },
  { id: "none", label: "no correction: looks regular", s: { tab: "L", corr: "none", r: 30 }, spec: { model: "csr" } },
  { id: "bias", label: "the four corrections, averaged", s: { tab: "bias" }, spec: { model: "csr" } },
  { id: "biasmask", label: "corrections in a tissue mask", s: { tab: "bias" }, spec: { model: "csr", win: "tissue" } },
  { id: "edit", label: "build your own pattern", s: { tab: "K", edit: true, r: 15, corr: "isotropic" }, spec: { model: "csr", n: 20 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 22 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, spec: { ...DEFAULT_SPEC } };
  if (p0) { Object.assign(st, p0.s); Object.assign(st.spec, p0.spec ?? {}); }
  else if (host.dataset.tab) st.tab = host.dataset.tab as Tab;

  const viewBox = htmlEl("div", "rk-view", "", host);
  const cap = htmlEl("div", "rk-cap", "", viewBox);
  const view = new PatternView(viewBox);
  const main = htmlEl("div", "rk-main", "", host);
  const tabsEl = htmlEl("div", "ss-tabs", "", main);
  const TABS: [Tab, string][] = [["K", "K"], ["L", "L − r"], ["g", "g"], ["bias", "bias of corrections"]];
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(); }));
  const fig = plotFigure(main, "", "qb-figure");
  const pl = new Plot(fig, { width: 460, height: 370, x: [0, RMAX], y: [0, 1], font: FS, margin: { l: 4 * FS, b: 2.7 * FS, t: 10, r: 14 } });
  const corrRow = htmlEl("div", "qb-presets", "", main);
  const corrBtns = CORRS.map(c => button(corrRow, c, () => { st.corr = c; unp(); draw(); }));

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const rSl = slider(panel, "r (µm)", 0, RMAX, 0.5, st.r, v => { st.r = v; unp(); drawLight(); });
  const hSl = slider(panel, "ring half-width h", 0.5, 6, 0.5, st.h, v => { st.h = v; unp(); draw(); });
  const editBox = checkbox(panel, "edit: click adds, drag moves, shift-click removes", st.edit, v => { st.edit = v; draw(); });
  const src = sourceControls(panel, st.spec, () => { unp(); regenerate(); draw(); }, { windows: ["square", "small", "tissue"] as WinKind[] });
  htmlEl("div", "ss-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [], activeId = p0?.id ?? "";
  function buildPresets(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => p.s.tab === st.tab || (st.tab === "K" && p.id === "edit")) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => { b.dataset.id = list[i].id; b.classList.toggle("active", list[i].id === activeId); });
  }
  button(panel, "reset", () => applyPreset(null));

  let pts = makePattern(st.spec);
  let cache: { pd: PP.PairData; K: Float64Array; g: Float64Array } | null = null;
  const biasCache = new Map<string, Record<Corr, Float64Array>>();
  function regenerate(): void { pts = makePattern(st.spec); cache = null; st.probe = -1; }
  function unp(): void { activeId = ""; pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    const { spec } = st;
    Object.assign(st, DEFAULT, p?.s ?? {}, { spec });
    Object.assign(spec, DEFAULT_SPEC, p?.spec ?? {});
    src.sync(); rSl.set(st.r); hSl.set(st.h); editBox.checked = st.edit;
    activeId = p?.id ?? "";
    regenerate(); draw();
  }

  // ---- pointer: probe or edit
  let dragging = -1;
  view.svg.addEventListener("pointerdown", ev => {
    const [x, y] = view.toWorld(ev), win = view.win;
    const near = PatternView.nearest(pts, x, y, 5 * win.w / SIDE + 1);
    if (st.edit) {
      if (near >= 0 && ev.shiftKey) { pts = PP.filterPoints(pts, i => i !== near); st.probe = -1; }
      else if (near >= 0) { dragging = near; view.svg.setPointerCapture(ev.pointerId); st.probe = near; }
      else if (PP.inside(win, x, y) && pts.x.length < 300) { pts = PP.makePoints([...pts.x, x], [...pts.y, y]); st.probe = pts.x.length - 1; }
      cache = null; unp(); draw();
    } else if (near >= 0) { st.probe = near; dragging = -2; view.svg.setPointerCapture(ev.pointerId); drawLight(); }
  });
  view.svg.addEventListener("pointermove", ev => {
    if (dragging === -1) return;
    const [x, y] = view.toWorld(ev);
    if (dragging >= 0) {
      if (!PP.inside(view.win, x, y)) return;
      pts.x[dragging] = x; pts.y[dragging] = y; cache = null; draw();
    } else {
      const near = PatternView.nearest(pts, x, y, 12);
      if (near >= 0 && near !== st.probe) { st.probe = near; drawLight(); }
    }
  });
  const up = () => { dragging = -1; };
  view.svg.addEventListener("pointerup", up);
  view.svg.addEventListener("pointercancel", up);

  function curves(win: PP.Window): { K: Float64Array; g: Float64Array } {
    if (!cache || cache.pd.corr !== st.corr || cache.g.length === 0 || (cache as unknown as { h: number }).h !== st.h) {
      const pd = PP.pairData(pts, win, RMAX + 6, st.corr);
      const pdG = st.corr === "border" ? PP.pairData(pts, win, RMAX + 6, "translation") : pd;
      cache = { pd, K: PP.KfromPairs(pd, RADII), g: PP.pcfFromPairs(pdG, RADII, st.h) };
      (cache as unknown as { h: number }).h = st.h;
    }
    return cache;
  }

  function biasCurves(win: PP.Window, n: number): Record<Corr, Float64Array> {
    const key = `${st.spec.win}|${n}`;
    const hit = biasCache.get(key);
    if (hit) return hit;
    const r = rng(2024), T = 50, acc = {} as Record<Corr, Float64Array>;
    for (const c of CORRS) acc[c] = new Float64Array(RADII.length);
    for (let t = 0; t < T; t++) {
      const q = PP.binomialProcess(r, n, win);
      for (const c of CORRS) {
        const K = PP.KfromPairs(PP.pairData(q, win, RMAX, c), RADII);
        for (let k = 1; k < RADII.length; k++) acc[c][k] += (K[k] / (Math.PI * RADII[k] ** 2) - 1) * 100 / T;
      }
    }
    // below 4 µm there are too few pairs for a stable percentage
    for (const c of CORRS) for (let k = 0; k < RADII.length && RADII[k] < 4; k++) acc[c][k] = NaN;
    biasCache.set(key, acc);
    return acc;
  }

  const fmt = (v: number, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const idxOf = (r: number) => Math.round(r / (RMAX / (RADII.length - 1)));
  let rLine: ReturnType<Plot["vline"]> | null = null;

  function ensureProbe(): void {
    const win = view.win, n = pts.x.length;
    if (st.probe >= 0 && st.probe < n) return;
    // default probe: the point closest to the window centre
    let best = Infinity;
    for (let i = 0; i < n; i++) { const q = (pts.x[i] - win.w / 2) ** 2 + (pts.y[i] - win.h / 2) ** 2; if (q < best) { best = q; st.probe = i; } }
  }

  /** Probe circles and readout (cheap: on r moves and probe changes). */
  function drawLight(): void {
    const win = view.win, n = pts.x.length, rho = n / win.area, s = win.w / SIDE;
    view.over.replaceChildren();
    let txt = `n ${n}, ρ = ${(1000 * rho).toFixed(2)} per 1000 µm²\ncorrection: ${st.corr}\n`;
    if (n >= 2) {
      ensureProbe();
      const i = st.probe, r = st.r, h = st.h;
      let inDisc = 0, inRing = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = Math.hypot(pts.x[j] - pts.x[i], pts.y[j] - pts.y[i]);
        if (d <= r) inDisc++;
        if (d > r - h && d <= r + h) inRing++;
      }
      const ringArea = Math.PI * ((r + h) ** 2 - Math.max(0, r - h) ** 2);
      svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: Math.min(r + h, 400), fill: "none", stroke: "var(--qb-c3)", "stroke-width": 0.8 * s, "stroke-dasharray": `${2 * s} ${1.5 * s}` }, view.over);
      if (r - h > 0) svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: r - h, fill: "none", stroke: "var(--qb-c3)", "stroke-width": 0.8 * s, "stroke-dasharray": `${2 * s} ${1.5 * s}` }, view.over);
      svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: Math.max(0.01, r), fill: "var(--qb-c4)", "fill-opacity": 0.12, stroke: "var(--qb-c4)", "stroke-width": 1.4 * s }, view.over);
      svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: 3.2 * s, fill: "var(--qb-c2)" }, view.over);
      const frac = PP.circleFraction(win, pts.x[i], pts.y[i], Math.max(r, 0.01));
      txt += `probe: within r = ${r}: ${inDisc}\n  CSR expects ρπr² = ${fmt(rho * Math.PI * r * r)}\n  ring r ± h: ${inRing}, CSR ${fmt(rho * ringArea)}\n  circle inside window: ${Math.round(100 * frac)} %\n`;
      if (st.tab !== "bias" && cache) {
        const k = idxOf(r), K = cache.K[k], L = Math.sqrt(Math.max(0, K) / Math.PI);
        txt += `K̂(r) ${fmt(K, 0)}  πr² ${fmt(Math.PI * r * r, 0)}\nL − r ${fmt(L - r, 2)}   ĝ(r) ${fmt(cache.g[k], 2)}`;
      }
    }
    ro.textContent = txt;
    if (rLine) rLine.set(st.r);
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    corrBtns.forEach((b, i) => b.classList.toggle("active", CORRS[i] === st.corr));
    buildPresets();
    const win = makeWindow(st.spec.win);
    view.setWindow(win);
    view.under.replaceChildren(); view.mid.replaceChildren();
    ensureProbe();
    view.setPoints(pts, i => (i === st.probe ? "var(--qb-c2)" : "var(--qb-c1)"));
    cap.textContent = `${pts.x.length} points` + (st.edit ? " · editing" : " · click a point to move the probe");
    pl.clear(); rLine = null;
    const n = pts.x.length;
    corrRow.style.display = st.tab === "bias" ? "none" : "";
    if (st.tab === "bias") {
      const b = biasCurves(win, Math.max(n, 10));
      const all = CORRS.flatMap(c => Array.from(b[c]).filter(Number.isFinite));
      pl.setDomain([0, RMAX], [Math.min(-5, ...all) * 1.1, Math.max(5, ...all) * 1.1]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: "mean bias of K̂ (%)" });
      pl.hline(0, { color: "var(--qb-muted)", dash: "6 4", width: 2 });
      CORRS.forEach((c, k) => pl.line(RADII, b[c], { color: ["var(--qb-c2)", "var(--qb-c4)", "var(--qb-c1)", "var(--qb-c3)"][k], label: c }));
      pl.legend(undefined, "bl");
      drawLight();
      ro.textContent += `\n50 CSR patterns, n = ${Math.max(n, 10)}, this window`;
      return;
    }
    if (n < 2) { drawLight(); return; }
    const { K, g } = curves(win);
    const thom = st.spec.model === "thomas" && !st.edit;
    const kappa = st.spec.k / (SIDE * SIDE), sig = st.spec.sigma;
    if (st.tab === "K") {
      pl.setDomain([0, RMAX], [0, Math.max(Math.PI * RMAX * RMAX, ...Array.from(K)) * 1.05]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: "K(r)  (µm²)" });
      pl.fn(r => Math.PI * r * r, { color: "var(--qb-muted)", dash: "7 5", width: 2.5, label: "CSR: πr²" });
      // probe's own count / ρ
      const i = st.probe, rho = n / win.area;
      if (i >= 0 && i < n) {
        const ds: number[] = [];
        for (let j = 0; j < n; j++) if (j !== i) ds.push(Math.hypot(pts.x[j] - pts.x[i], pts.y[j] - pts.y[i]));
        const pc = Array.from(RADII, r => ds.filter(d => d <= r).length / rho);
        pl.step(RADII, pc, { color: "var(--qb-c2)", width: 1.5, label: "probe count / ρ" });
      }
      pl.line(RADII, K, { color: "var(--qb-c1)", label: "K̂ observed" });
      pl.legend(undefined, "tl");
    } else if (st.tab === "L") {
      const L = PP.LminusR(K, RADII), fin = Array.from(L).filter(Number.isFinite);
      const lim = Math.max(2, ...fin.map(Math.abs)) * 1.15;
      pl.setDomain([0, RMAX], [-lim, lim]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: "L(r) − r  (µm)" });
      pl.hline(0, { color: "var(--qb-muted)", dash: "7 5", width: 2.5, label: "CSR" });
      pl.line(RADII, L, { color: "var(--qb-c1)", label: "L̂ − r observed" });
      pl.legend(undefined, "tr");
    } else {
      const fin = Array.from(g).filter(Number.isFinite);
      pl.setDomain([0, RMAX], [0, Math.min(30, Math.max(2, ...fin, thom ? PP.thomasPcf(st.h, kappa, sig) : 0) * 1.1)]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: "g(r)" });
      pl.hline(1, { color: "var(--qb-muted)", dash: "7 5", width: 2.5, label: "CSR: g = 1" });
      if (thom) pl.fn(r => PP.thomasPcf(r, kappa, sig), { color: "var(--qb-c3)", dash: "3 3", width: 2.5, label: "Thomas formula" });
      pl.line(RADII, g, { color: "var(--qb-c5)", label: `ĝ observed (h = ${st.h})` });
      pl.legend(undefined, "tr");
    }
    rLine = pl.vline(st.r, { color: "var(--qb-c2)", width: 1.5, dash: "3 3" });
    drawLight();
  }

  draw();
}

autoMount(NAME, mount);
