// Is the colocalisation more than chance? The two-cell scene of the coloc demo
// (lib/twochannel.ts pixelScene) with four tabs, one question each:
//  - shift curve: Pearson r against a shift of B (−20..20 px) along x and along y, the blur
//    FWHM marked, the peak with a sub-pixel parabola, the half width;
//  - randomisation: B cut into blocks (1..16 px) and shuffled 200 times (seeded); histogram of
//    the shuffled r with the observed r, the permutation p-value, and the 1-px shuffle null in
//    grey for comparison;
//  - PDM / ICQ: the product of the differences from the mean as a signed image, Li's ICQ with
//    its (naive) sign-test p, and the ICQ of 200 block-shuffled B as an honest reference;
//  - Costes trace: r of the pixels below the thresholds against tA, with the stop marked.
// Sliders: shared fraction f, structure size, blur, sub-pixel shift of B, bleed-through, a
// smooth background shared by both channels, noise; region inside the cells or whole image.
// Auto-mounts on <div data-demo="colocnull">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { diverging } from "../lib/colour";
import { pixelScene, Channels, S, mergeGreenMagenta } from "../lib/twochannel";
import { pearsonMasked, shiftCurve, curvePeak, halfWidth, randomisationTest, pdm, icq, costes, manders, CostesResult } from "../lib/coloc";
import { sd, mean, quantile } from "../lib/stats";

const NAME = "colocnull";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cn-img { flex: 0 1 300px; min-width: 200px; margin: 0; }
.qb-demo[data-demo="${NAME}"] .cn-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] .cn-main { flex: 1 1 400px; min-width: 260px; max-width: 520px; }
.qb-demo[data-demo="${NAME}"] .cn-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .cn-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .5em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cn-img { flex: 0 0 250px; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .cn-main { flex: 0 0 430px; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; gap: .12em; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; line-height: 1.2; padding: .25em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .cn-lab { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Tab = "shift" | "rand" | "pdm" | "costes";
const TABS: [Tab, string][] = [["shift", "shift curve"], ["rand", "randomisation"], ["pdm", "PDM / ICQ"], ["costes", "Costes trace"]];
const BLOCKS = [1, 2, 4, 8, 16];
const NSHUF = 200;

interface State {
  f: number; size: number; blur: number; shiftX: number; bleed: number; uneven: number; noise: number; diffuse: number;
  layout: "mix" | "exclude"; seed: number; roi: "cells" | "all"; block: number; tab: Tab;
}
const DEFAULT: State = { f: 0.5, size: 1, blur: 1, shiftX: 0, bleed: 0, uneven: 0, noise: 4, diffuse: 0, layout: "mix", seed: 4, roi: "cells", block: 8, tab: "shift" };

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "coloc", label: "colocalised: narrow peak at 0", s: { f: 1, tab: "shift" } },
  { id: "hump", label: "large-scale correlation: broad hump", s: { f: 0, diffuse: 30, roi: "all", tab: "shift" } },
  { id: "shifted", label: "a shifted channel: peak at +2.4 px", s: { f: 1, shiftX: 2.4, tab: "shift" } },
  { id: "pixel", label: "pixel shuffle: everything is significant", s: { f: 0, blur: 2, seed: 15, block: 1, tab: "rand" } },
  { id: "blocks", label: "blocks the size of the blur: honest null", s: { f: 0, blur: 2, seed: 15, block: 8, tab: "rand" } },
  { id: "bleed", label: "bleed-through passes the test", s: { f: 0, bleed: 0.4, block: 8, tab: "rand" } },
  { id: "independent", label: "independent: ICQ is not 0", s: { f: 0, tab: "pdm" } },
  { id: "segregated", label: "segregated: ICQ < 0", s: { f: 0, layout: "exclude", tab: "pdm" } },
  { id: "weak", label: "weak correlation: Costes stops high", s: { f: 0.15, noise: 12, tab: "costes" } },
  { id: "background", label: "shared background: Costes walks to the bottom", s: { f: 0, uneven: 40, roi: "all", tab: "costes" } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 24 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  if (!p0 && host.dataset.tab) st.tab = host.dataset.tab as Tab;

  let ch: Channels = pixelScene(st);
  let nullR: { block: Float64Array; px: Float64Array; obs: number; p: number; pPx: number } | null = null;
  let nullIcq: { vals: Float64Array; obs: number; p: number } | null = null;

  const imgFig = htmlEl("figure", "cn-img", "", host);
  const imgCap = htmlEl("figcaption", "", "", imgFig);
  const cv = htmlEl("canvas", "", "", imgFig);
  cv.width = S; cv.height = S;

  const main = htmlEl("div", "cn-main", "", host);
  const tabs = htmlEl("div", "cn-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabs, l, () => { st.tab = t; unp(); draw(); }));
  const pf = plotFigure(main, "", "qb-figure");
  const pl = new Plot(pf, { width: 460, height: 340, x: [-20, 20], y: [-0.2, 1], font: FS, margin: { l: 3.2 * FS, b: 2.8 * FS, t: 12, r: 12 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const runRow = htmlEl("div", "qb-presets", "", panel);
  const runBtn = button(runRow, `shuffle B ×${NSHUF}`, () => { runNull(); draw(); });
  const bS = slider(panel, "block (px)", 0, BLOCKS.length - 1, 1, BLOCKS.indexOf(st.block), v => { st.block = BLOCKS[v]; nullR = null; nullIcq = null; unp(); draw(); }, v => String(BLOCKS[v]));
  htmlEl("div", "cn-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [];
  const fS = slider(panel, "shared f", 0, 1, 0.05, st.f, v => { st.f = v; changed(); });
  const zS = slider(panel, "structure size", 0.5, 3, 0.1, st.size, v => { st.size = v; changed(); });
  const blS = slider(panel, "blur σ (px)", 0.5, 4, 0.1, st.blur, v => { st.blur = v; changed(); });
  const shS = slider(panel, "shift of B (px)", -4, 4, 0.1, st.shiftX, v => { st.shiftX = v; changed(); });
  const leS = slider(panel, "bleed A→B", 0, 0.8, 0.05, st.bleed, v => { st.bleed = v; changed(); });
  const unS = slider(panel, "shared background", 0, 80, 2, st.uneven, v => { st.uneven = v; changed(); });
  const noS = slider(panel, "noise σ", 0, 20, 1, st.noise, v => { st.noise = v; changed(); });
  const roiB = presets(panel, [
    { label: "inside cells", apply: () => { st.roi = "cells"; nullR = null; nullIcq = null; unp(); draw(); } },
    { label: "whole image", apply: () => { st.roi = "all"; nullR = null; nullIcq = null; unp(); draw(); } },
  ]);
  button(panel, "reset", () => { applyPreset(null); });

  function buildPresetButtons(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => p.s.tab === st.tab) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => b.dataset.id = list[i].id);
  }
  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function changed(): void { unp(); rebuild(); draw(); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    Object.assign(st, DEFAULT, p?.s ?? {});
    if (!p) st.tab = "shift";
    sync(); rebuild();
    if (p && (st.tab === "rand" || st.tab === "pdm")) runNull();
    draw();
    pb.forEach(b => b.classList.toggle("active", !!p && b.dataset.id === p.id));
  }
  function sync(): void {
    fS.set(st.f); zS.set(st.size); blS.set(st.blur); shS.set(st.shiftX); leS.set(st.bleed); unS.set(st.uneven); noS.set(st.noise);
    bS.set(BLOCKS.indexOf(st.block));
  }
  const roi = () => (st.roi === "cells" ? ch.cells : undefined);
  function rebuild(): void {
    ch = pixelScene({ f: st.f, size: st.size, blur: st.blur, shiftX: st.shiftX, bleed: st.bleed, uneven: st.uneven, noise: st.noise, diffuse: st.diffuse, layout: st.layout, seed: st.seed });
    nullR = null; nullIcq = null;
  }
  function runNull(): void {
    const m = roi();
    if (st.tab === "pdm") {
      const t = randomisationTest(ch.a, ch.b, S, S, { block: st.block, n: NSHUF, seed: 77, mask: m, stat: (a, b, mm) => icq(a, b, mm).icq });
      nullIcq = { vals: t.null, obs: t.observed, p: t.p };
    } else {
      const t = randomisationTest(ch.a, ch.b, S, S, { block: st.block, n: NSHUF, seed: 77, mask: m });
      const t1 = randomisationTest(ch.a, ch.b, S, S, { block: 1, n: NSHUF, seed: 78, mask: m });
      nullR = { block: t.null, px: t1.null, obs: t.observed, p: t.p, pPx: t1.p };
    }
  }

  function drawImage(kind: "merge" | "pdm"): void {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(S, S);
    if (kind === "merge") {
      let ma = 1, mb = 1;
      for (let i = 0; i < S * S; i++) { ma = Math.max(ma, ch.a[i]); mb = Math.max(mb, ch.b[i]); }
      mergeGreenMagenta(ch.a, ch.b, [0, 0.8 * ma], [0, 0.8 * mb], id.data);
      imgCap.textContent = st.roi === "cells" ? "A green, B magenta" : "whole image";
    } else {
      const d = pdm(ch.a, ch.b, roi());
      const lim = quantile(Array.from(d, Math.abs), 0.99) || 1;
      for (let i = 0; i < S * S; i++) {
        const inside = !roi() || ch.cells[i];
        const [r, g, b] = inside ? diverging(0.5 + 0.5 * Math.max(-1, Math.min(1, d[i] / lim))) : [0.55, 0.55, 0.55];
        id.data[4 * i] = 255 * r; id.data[4 * i + 1] = 255 * g; id.data[4 * i + 2] = 255 * b; id.data[4 * i + 3] = 255;
      }
      imgCap.textContent = "PDM: red +, blue −";
    }
    if (kind === "merge" && st.roi === "cells") {
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const i = y * S + x;
        if (ch.cells[i] && ((x + 1 < S && !ch.cells[i + 1]) || (x > 0 && !ch.cells[i - 1]) || (y + 1 < S && !ch.cells[i + S]) || (y > 0 && !ch.cells[i - S]))) {
          id.data[4 * i] = 255; id.data[4 * i + 1] = 212; id.data[4 * i + 2] = 0;
        }
      }
    }
    ctx.putImageData(id, 0, 0);
  }

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const hist = (vals: ArrayLike<number>, lo: number, hi: number, nb: number) => {
    const bw = (hi - lo) / nb, c = new Array(nb).fill(0);
    for (let i = 0; i < vals.length; i++) { const k = Math.floor((vals[i] - lo) / bw); if (k >= 0 && k < nb) c[k]++; }
    return { edges: Array.from({ length: nb + 1 }, (_, k) => lo + k * bw), counts: c };
  };

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    buildPresetButtons();
    roiB.forEach((b, i) => b.classList.toggle("active", ["cells", "all"][i] === st.roi));
    const showRun = st.tab === "rand" || st.tab === "pdm";
    runRow.style.display = showRun ? "" : "none";
    bS.input.closest("label")!.style.display = showRun ? "" : "none";
    runBtn.textContent = `shuffle ${st.block}-px blocks of B ×${NSHUF}`;
    drawImage(st.tab === "pdm" ? "pdm" : "merge");
    const m = roi();
    const r0 = pearsonMasked(ch.a, ch.b, m);
    pl.clear();
    let txt = "";
    if (st.tab === "shift") {
      const cx = shiftCurve(ch.a, ch.b, S, S, 20, m, "x"), cy = shiftCurve(ch.a, ch.b, S, S, 20, m, "y");
      const fw = 2.3548 * st.blur;
      pl.setDomain([-20, 20], [Math.min(-0.2, ...cx.r, ...cy.r) - 0.02, 1.02]);
      pl.setLabels({ xlabel: "shift of B (px)", ylabel: "Pearson r" });
      pl.band([-fw / 2, fw / 2], [-1, -1], [2, 2], { fill: "var(--qb-c4)", fillOpacity: 0.15, color: "none" });
      pl.hline(0, { color: "var(--qb-axis)", width: 1 });
      pl.line(cy.shifts, cy.r, { color: "var(--qb-c5)", width: 2.5, dash: "6 4", label: "shift in y" });
      pl.line(cx.shifts, cx.r, { color: "var(--qb-c1)", width: 3, label: "shift in x" });
      const px = curvePeak(cx.shifts, cx.r), py = curvePeak(cy.shifts, cy.r);
      pl.vline(px.at, { color: "var(--qb-c2)", width: 2 });
      pl.legend([{ label: "shift in x", color: "var(--qb-c1)" }, { label: "shift in y", color: "var(--qb-c5)", dash: "6 4" }, { label: "blur FWHM", color: "var(--qb-c4)", kind: "box" }], "tr");
      txt = `r at 0           ${fmt(r0)}\npeak in x at     ${fmt(px.at, 2)} px\npeak in y at     ${fmt(py.at, 2)} px\npeak r (x)       ${fmt(px.value)}\nhalf width (x)   ${fmt(halfWidth(cx.shifts, cx.r), 1)} px\nr at Δx = 1, 2, 3 ${fmt(cx.r[21], 2)}, ${fmt(cx.r[22], 2)}, ${fmt(cx.r[23], 2)}\nr at Δx = ±20    ${fmt(cx.r[0], 2)} / ${fmt(cx.r[40], 2)}\nblur FWHM        ${fmt(fw, 1)} px`;
    } else if (st.tab === "rand") {
      pl.setLabels({ xlabel: "Pearson r with B shuffled", ylabel: "count" });
      if (nullR) {
        const all = [...nullR.block, ...nullR.px, nullR.obs];
        const lo0 = Math.min(...all) - 0.03, hi0 = Math.max(...all) + 0.03, lo = lo0, hi = hi0 + 0.3 * (hi0 - lo0), nb = 40;
        const hb = hist(nullR.block, lo, hi, nb), hp = hist(nullR.px, lo, hi, nb);
        pl.setDomain([lo, hi], [0, Math.max(...hb.counts, ...hp.counts) * 1.15]);
        pl.histogram(hp.edges, hp.counts, { color: "#999", fillOpacity: 0.45, label: "1-px shuffle" });
        pl.histogram(hb.edges, hb.counts, { color: "var(--qb-c1)", fillOpacity: 0.6, label: `${st.block}-px blocks` });
        pl.vline(nullR.obs, { color: "var(--qb-c2)", width: 3, text: "observed" });
        pl.legend([{ label: "1-px shuffle", color: "#999", kind: "box" }, { label: `${st.block}-px blocks`, color: "var(--qb-c1)", kind: "box" }], "tl");
        const ex = nullR.block.filter(v => v >= nullR!.obs).length, exPx = nullR.px.filter(v => v >= nullR!.obs).length;
        txt = `observed r        ${fmt(nullR.obs)}\n${st.block}-px blocks: sd ${fmt(sd(nullR.block))}\n  ${ex} of ${NSHUF} ≥ observed\n  p = ${fmt(nullR.p)}\n1-px shuffle: sd ${fmt(sd(nullR.px))}\n  ${exPx} of ${NSHUF} ≥ observed\n  p = ${fmt(nullR.pPx)}`;
      } else {
        pl.setDomain([-0.3, 1], [0, 10]);
        pl.vline(r0, { color: "var(--qb-c2)", width: 3, text: "observed" });
        txt = `observed r        ${fmt(r0)}\n\npress "shuffle" to build\nthe null distribution`;
      }
    } else if (st.tab === "pdm") {
      const q = icq(ch.a, ch.b, m);
      pl.setLabels({ xlabel: "ICQ with B shuffled", ylabel: "count" });
      if (nullIcq) {
        const lo0 = Math.min(...nullIcq.vals, q.icq, 0) - 0.03, hi0 = Math.max(...nullIcq.vals, q.icq, 0) + 0.03, lo = lo0, hi = hi0 + 0.3 * (hi0 - lo0);
        const hb = hist(nullIcq.vals, lo, hi, 40);
        pl.setDomain([lo, hi], [0, Math.max(...hb.counts) * 1.15]);
        pl.histogram(hb.edges, hb.counts, { color: "var(--qb-c1)", fillOpacity: 0.6 });
        pl.vline(0, { color: "var(--qb-axis)", width: 1.5, dash: "4 4" });
        pl.vline(q.icq, { color: "var(--qb-c2)", width: 3, text: "observed" });
      } else {
        pl.setDomain([-0.5, 0.5], [0, 10]);
        pl.vline(0, { color: "var(--qb-axis)", width: 1.5, dash: "4 4" });
        pl.vline(q.icq, { color: "var(--qb-c2)", width: 3, text: "observed" });
      }
      txt = `Pearson r         ${fmt(r0)}\nPDM > 0: ${q.nPos} of ${q.n} px\nICQ               ${fmt(q.icq)}\nsign test p       ${q.p < 1e-4 ? q.p.toExponential(1) : fmt(q.p, 4)}\n  (pixels assumed independent)`;
      if (nullIcq) txt += `\nshuffled ICQ (${st.block}-px blocks):\n  mean ${fmt(mean(nullIcq.vals))}, sd ${fmt(sd(nullIcq.vals))}\n  p = ${fmt(nullIcq.p)}`;
      else txt += `\n\npress "shuffle" for the ICQ\nof block-shuffled B`;
    } else {
      const c: CostesResult = costes(ch.a, ch.b, m, 100);
      pl.setLabels({ xlabel: "threshold tA (tB on the line)", ylabel: "r of pixels below" });
      if (c.ok) {
        const xs = c.trace.map(t => t.tA), ys = c.trace.map(t => t.r);
        pl.setDomain([0, Math.max(...xs) * 1.03], [Math.min(-0.1, ...ys) - 0.02, Math.max(0.2, ...ys) * 1.05]);
        pl.hline(0, { color: "var(--qb-axis)", width: 1.5, dash: "4 4" });
        pl.line(xs, ys, { color: "var(--qb-c1)", width: 3 });
        pl.vline(c.tA, { color: "var(--qb-c2)", width: 3, text: "stop" });
        const md = manders(ch.a, ch.b, c.tA, c.tB, m);
        let nAb = 0, n = 0;
        for (let i = 0; i < S * S; i++) if (!m || m[i]) { n++; if (ch.a[i] > c.tA && ch.b[i] > c.tB) nAb++; }
        txt = `Pearson r         ${fmt(r0)}\nCostes stops at\n  tA = ${fmt(c.tA, 1)}, tB = ${fmt(c.tB, 1)}\n  after ${c.trace.length} steps\nM1 = ${fmt(md.m1)}, M2 = ${fmt(md.m2)}\nabove both: ${nAb} of ${n} px\n  (${fmt(100 * nAb / n, 0)} %)`;
      } else {
        pl.setDomain([0, 250], [-0.2, 1]);
        txt = `Pearson r         ${fmt(r0)}\nCostes not defined:\nno rising line (r ≤ 0)`;
      }
    }
    ro.textContent = txt;
  }

  sync();
  draw();
  if (p0 && (st.tab === "rand" || st.tab === "pdm")) { runNull(); draw(); }
  if (p0) pb.forEach(b => b.classList.toggle("active", b.dataset.id === p0.id));
}

autoMount(NAME, mount);
