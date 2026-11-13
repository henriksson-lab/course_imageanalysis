// Two cell types and an uneven density. A synthetic tissue (mask with a vessel) of tumour cells
// (orange) and immune cells (blue) from several generators. Tabs: cross-L (immune around
// tumour), L − r of all cells, the mark connection function p01(r), and the inhomogeneous L with
// the kernel estimate ρ̂ (bandwidth slider) drawn under the tissue. Envelope from the chosen null
// (relocate immune cells in the window, toroidal shift, permute labels, CSR, inhomogeneous Poisson
// from ρ̂), with the global MAD test and its p. Window: the tissue mask or the whole image.
// Auto-mounts on <div data-demo="celltypes">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rng } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { OKABE_ITO } from "../lib/colour";
import { PatternView, injectStyle, sourceCss, heatUrl } from "../csrplay/source";
import { Gen, GEN_LABELS, TissueOptions, DEFAULT_TISSUE, makeTissue, tissueWindow, imageWindow, Curve, Null, curveOf, simulateNull, referenceOf, RADII, RMAX, VESSEL, rhoGrid, GRID_N } from "./model";

const NAME = "celltypes";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = sourceCss(NAME) + `
${P} .ct-view { flex: 0 1 360px; min-width: 240px; }
${P} .ct-main { flex: 1 1 380px; min-width: 260px; max-width: 540px; display: flex; flex-direction: column; gap: .3em; }
${P} .qb-panel { flex: 1 1 280px; gap: .35em; }
${P} .ct-cap { font-size: .9em; text-align: center; }
.reveal ${P} .ct-view { flex: 0 0 360px; min-width: 0; }
.reveal ${P} .ct-main { flex: 0 0 440px; min-width: 0; }
.reveal ${P} .qb-panel { flex: 1 1 320px; min-width: 0; gap: .15em; }
`;

const TABS: [Curve, string][] = [["cross", "cross-L"], ["L", "L of all cells"], ["mcf", "mark connection"], ["inhom", "inhomogeneous L"]];
const NULL_LABELS: Record<Null, string> = { relocate: "relocate immune cells", shift: "toroidal shift", permute: "permute labels", csr: "CSR (all cells)", inhomPoisson: "Poisson from ρ̂" };
const NULLS_FOR: Record<Curve, Null[]> = { cross: ["relocate", "shift", "permute"], L: ["csr"], mcf: ["permute"], inhom: ["inhomPoisson"] };
const C0 = OKABE_ITO[0], C1 = OKABE_ITO[4];

interface State { t: TissueOptions; tab: Curve; nul: Null; win: "tissue" | "image"; bw: number }
const DEFAULT: State = { t: { ...DEFAULT_TISSUE }, tab: "cross", nul: "relocate", win: "tissue", bw: 40 };

const PRESETS: { id: string; label: string; s: Partial<State>; t?: Partial<TissueOptions> }[] = [
  { id: "attract", label: "attraction: cross-L above", s: { tab: "cross", nul: "relocate" }, t: { gen: "attraction" } },
  { id: "indep", label: "independent: inside", s: { tab: "cross", nul: "relocate" }, t: { gen: "independent" } },
  { id: "segregate", label: "segregated domains: below", s: { tab: "cross", nul: "permute" }, t: { gen: "segregation" } },
  { id: "packedReloc", label: "packed tissue: relocation says repulsion", s: { tab: "cross", nul: "relocate" }, t: { gen: "packed" } },
  { id: "packedPerm", label: "packed tissue: labels say random", s: { tab: "cross", nul: "permute" }, t: { gen: "packed" } },
  { id: "gradientReloc", label: "gradient: relocation says attraction", s: { tab: "cross", nul: "relocate" }, t: { gen: "gradient" } },
  { id: "gradientPerm", label: "random labelling ignores the gradient", s: { tab: "cross", nul: "permute" }, t: { gen: "gradient" } },
  { id: "gradientL", label: "density gradient: plain L says clustered", s: { tab: "L", nul: "csr" }, t: { gen: "gradient" } },
  { id: "image", label: "the image as the window", s: { tab: "L", nul: "csr", win: "image" }, t: { gen: "random" } },
  { id: "mask", label: "the tissue mask as the window", s: { tab: "L", nul: "csr", win: "tissue" }, t: { gen: "random" } },
  { id: "inhom", label: "inhomogeneous L: no interaction", s: { tab: "inhom", nul: "inhomPoisson", bw: 40 }, t: { gen: "gradient" } },
  { id: "bwSmall", label: "bandwidth too small: clusters explained away", s: { tab: "inhom", nul: "inhomPoisson", bw: 5 }, t: { gen: "labelled" } },
  { id: "bwLarge", label: "large bandwidth: clusters stay", s: { tab: "inhom", nul: "inhomPoisson", bw: 40 }, t: { gen: "labelled" } },
  { id: "mcf", label: "mark connection: segregation", s: { tab: "mcf", nul: "permute" }, t: { gen: "segregation" } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 22 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, t: { ...DEFAULT_TISSUE } };
  if (p0) { Object.assign(st, p0.s); Object.assign(st.t, p0.t ?? {}); }
  else if (host.dataset.tab) { st.tab = host.dataset.tab as Curve; st.nul = NULLS_FOR[st.tab][0]; }

  // the 39 null simulations take a few hundred ms: slider moves redraw once the slider rests
  let timer = 0;
  const later = (f: () => void) => { clearTimeout(timer); timer = window.setTimeout(f, 180); };

  const viewBox = htmlEl("div", "ct-view", "", host);
  const cap = htmlEl("div", "ct-cap", "", viewBox);
  const view = new PatternView(viewBox);
  const main = htmlEl("div", "ct-main", "", host);
  const tabsEl = htmlEl("div", "ss-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; st.nul = NULLS_FOR[t][0]; unp(); draw(); }));
  const fig = plotFigure(main, "", "qb-figure");
  const pl = new Plot(fig, { width: 480, height: 360, x: [0, RMAX], y: [-5, 5], font: FS, margin: { l: 3.8 * FS, b: 2.7 * FS, t: 10, r: 14 } });
  const nullRow = htmlEl("div", "qb-presets", "", main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const genSel = htmlEl("select", "ss-select", "", panel);
  for (const g of Object.keys(GEN_LABELS) as Gen[]) { const o = htmlEl("option", "", GEN_LABELS[g], genSel); o.value = g; }
  genSel.addEventListener("change", () => { st.t.gen = genSel.value as Gen; genSel.blur(); unp(); regenerate(); draw(); });
  const winSel = htmlEl("select", "ss-select", "", panel);
  for (const [v, l] of [["tissue", "window: tissue mask"], ["image", "window: whole image"]]) { const o = htmlEl("option", "", l, winSel); o.value = v; }
  winSel.addEventListener("change", () => { st.win = winSel.value as State["win"]; winSel.blur(); unp(); draw(); });
  const sl = {
    share1: slider(panel, "immune share", 0.1, 0.5, 0.05, st.t.share1, v => { st.t.share1 = v; unp(); later(() => { regenerate(); draw(); }); }),
    strength: slider(panel, "strength", 0, 1, 0.05, st.t.strength, v => { st.t.strength = v; unp(); later(() => { regenerate(); draw(); }); }),
    bw: slider(panel, "bandwidth σ (µm)", 4, 60, 1, st.bw, v => { st.bw = v; unp(); later(draw); }),
  };
  const seedRow = htmlEl("div", "ss-seedrow", "", panel);
  const seedOut = htmlEl("span", "ss-seed", "", seedRow);
  button(seedRow, "new seed", () => { st.t.seed++; unp(); regenerate(); draw(); });
  htmlEl("div", "ss-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [], activeId = p0?.id ?? "";
  function buildPresets(): void {
    presetRow.replaceChildren();
    if (onSlide) {
      // on a slide the presets of the tab go into a drop-down (there are too many for buttons)
      const list = PRESETS.filter(p => p.s.tab === st.tab);
      const sel = htmlEl("select", "ss-select", "", presetRow);
      htmlEl("option", "", "choose a preset …", sel).value = "";
      for (const p of list) { const o = htmlEl("option", "", p.label, sel); o.value = p.id; }
      sel.value = list.some(p => p.id === activeId) ? activeId : "";
      sel.addEventListener("change", () => { const p = PRESETS.find(q => q.id === sel.value); sel.blur(); if (p) applyPreset(p); });
      pb = [];
      return;
    }
    pb = presets(presetRow, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => { b.dataset.id = PRESETS[i].id; b.classList.toggle("active", PRESETS[i].id === activeId); });
  }
  button(panel, "reset", () => applyPreset(null));

  let pts = makeTissue(st.t);
  function regenerate(): void { pts = makeTissue(st.t); }
  function unp(): void { activeId = ""; pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    const { t } = st;
    Object.assign(st, DEFAULT, p?.s ?? {}, { t });
    Object.assign(t, DEFAULT_TISSUE, p?.t ?? {});
    sl.share1.set(t.share1); sl.strength.set(t.strength); sl.bw.set(st.bw);
    activeId = p?.id ?? "";
    regenerate(); draw();
  }

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    buildPresets();
    genSel.value = st.t.gen; winSel.value = st.win;
    seedOut.textContent = "seed " + st.t.seed;
    sl.strength.input.closest("label")!.style.display = st.t.gen === "attraction" || st.t.gen === "segregation" ? "" : "none";
    sl.bw.input.closest("label")!.style.display = st.tab === "inhom" ? "" : "none";
    nullRow.replaceChildren();
    htmlEl("span", "ss-lab", "null: ", nullRow);
    for (const nl of NULLS_FOR[st.tab]) { const b = button(nullRow, NULL_LABELS[nl], () => { st.nul = nl; unp(); draw(); }); b.classList.toggle("active", nl === st.nul); }

    const tissue = tissueWindow(), win = st.win === "tissue" ? tissue : imageWindow();
    view.setWindow(tissue);
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    let grid: Float64Array | undefined;
    if (st.tab === "inhom") {
      grid = rhoGrid(pts, win, st.bw);
      const fin = Array.from(grid).filter(Number.isFinite);
      svgEl("image", { x: 0, y: 0, width: 200, height: 200, href: heatUrl(grid, GRID_N, GRID_N, 0, Math.max(...fin), 190), preserveAspectRatio: "none" }, view.under);
    }
    if (st.win === "image") svgEl("rect", { x: 0, y: 0, width: 200, height: 200, fill: "none", stroke: "var(--qb-c2)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, view.mid);
    svgEl("circle", { cx: VESSEL[0], cy: VESSEL[1], r: VESSEL[2], fill: "none", stroke: "#c0392b", "stroke-width": 1.4 }, view.mid);
    view.setPoints(pts, i => (pts.marks![i] === 1 ? C1 : C0), 2.4);
    let n1 = 0;
    for (const m of pts.marks!) if (m === 1) n1++;
    cap.textContent = `${pts.x.length - n1} tumour (orange), ${n1} immune (blue); vessel red`;

    // observed and null curves
    const obs = curveOf(pts, win, st.tab, st.bw), ref = referenceOf(pts, st.tab);
    const r = rng(77 + st.t.seed), s = 39, sims: Float64Array[] = [];
    for (let k = 0; k < s; k++) sims.push(curveOf(simulateNull(r, pts, win, st.nul, st.bw, grid), win, st.tab, st.bw));
    const idx = PP.rangeIdx(RADII, [2, RMAX]);
    // the reference is the mean of the null curves: under permuted labels (or a hard core) it is not the CSR value
    const env = PP.pointwiseEnvelope(sims), mad = PP.madTest(obs, sims, env.mean, idx);
    pl.clear();
    const all = [...Array.from(obs), ...Array.from(mad.lo), ...Array.from(mad.hi), ...Array.from(env.lo), ...Array.from(env.hi)].filter(Number.isFinite);
    if (st.tab === "mcf") pl.setDomain([0, RMAX], [0, Math.min(1, Math.max(...all) * 1.15)]);
    else { const m = Math.max(2, ...all.map(Math.abs)) * 1.1; pl.setDomain([0, RMAX], [-m, m]); }
    const ylab = st.tab === "cross" ? "L₀₁(r) − r  (µm)" : st.tab === "L" ? "L(r) − r  (µm)" : st.tab === "mcf" ? "p₀₁(r)" : "L_inhom(r) − r  (µm)";
    pl.setLabels({ xlabel: "r (µm)", ylabel: ylab });
    const xs = idx.map(k => RADII[k]);
    pl.band(xs, idx.map(k => mad.lo[k]), idx.map(k => mad.hi[k]), { color: "var(--qb-c3)", fillOpacity: 0.18, label: "global (MAD) envelope" });
    pl.band(xs, idx.map(k => env.lo[k]), idx.map(k => env.hi[k]), { color: "var(--qb-muted)", fillOpacity: 0.3, label: "pointwise, 39 sims" });
    pl.line(RADII, ref, { color: "var(--qb-muted)", dash: "2 4", width: 1.5 });
    pl.line(RADII, env.mean, { color: "var(--qb-muted)", dash: "7 5", width: 2, label: "null mean" });
    pl.line(RADII, obs, { color: "var(--qb-c1)", label: "observed" });
    pl.legend(undefined, "tr");
    const verdict = mad.p > 0.05 ? "consistent with the null" : st.tab === "mcf" ? "differs from random labels" : (obs[idx[Math.floor(idx.length / 4)]] > ref[0] ? "above: more pairs than the null" : "below: fewer pairs than the null");
    let txt = `null: ${NULL_LABELS[st.nul]}\nwindow: ${st.win === "tissue" ? "tissue mask" : "whole image"} (${Math.round(win.area)} µm²)\nMAD T = ${fmt(mad.T0, 3)}, rank ${mad.rank} of ${s + 1}\nglobal p = ${fmt(mad.p, 3)}\n→ ${verdict}`;
    if (st.tab === "mcf") txt += `\nrandom labels: 2·p₀·p₁ = ${fmt(ref[0], 3)}`;
    if (st.nul === "shift" && st.win === "tissue") txt += `\nshifted cells outside the\nmask are lost: invalid here`;
    ro.textContent = txt;
  }

  draw();
}

autoMount(NAME, mount);
