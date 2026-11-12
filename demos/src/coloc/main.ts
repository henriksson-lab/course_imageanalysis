// Colocalisation of two channels. Left: the merged image (A green, B magenta; white where
// both are bright) with the region of interest; middle: the 2D histogram (scatter plot) of
// (A, B) over the region with the orthogonal regression line and two draggable threshold
// lines tA, tB; right: Pearson r (all pixels and above both thresholds), the overlap
// coefficient, Manders M1 and M2, a Costes button that animates the automatic threshold
// search, and a randomisation test (block scramble of B, 50 repeats).
// Auto-mounts on <div data-demo="coloc">. Options: data-preset="<id>".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { inferno } from "../lib/colour";
import { histogram2d } from "../lib/histogram";
import { S, makeChannels, ChannelOptions, Channels, colocStats, randomisation, startThresholds } from "./model";

const NAME = "coloc";
const BINS = 64;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .co-img { flex: 0 1 340px; min-width: 220px; margin: 0; }
[data-demo="${NAME}"] .co-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .co-sc { flex: 0 1 360px; min-width: 240px; }
[data-demo="${NAME}"] .co-sc svg { touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .co-side { flex: 1 1 220px; min-width: 200px; max-width: 300px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 250px; gap: .3em; }
[data-demo="${NAME}"] .co-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 15px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .co-img { flex-basis: 300px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .co-sc { flex-basis: 330px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State extends Required<Omit<ChannelOptions, "seed">> { roi: "all" | "cells"; thr: "costes" | "zero" }
const DEFAULT: State = { f: 0.5, layout: "mix", offset: 10, noise: 4, diffuse: 0, bleed: 0, shift: 0, roi: "cells", thr: "costes" };

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "perfect", label: "perfect colocalisation", s: { f: 1 } },
  { id: "independent", label: "independent: r ≈ 0", s: { f: 0 } },
  { id: "background", label: "background inflates r", s: { f: 0, diffuse: 30, roi: "all" } },
  { id: "nothreshold", label: "M1 ≈ M2 ≈ 1 without thresholds", s: { f: 0, thr: "zero" } },
  { id: "inside", label: "A inside B: M1 ≫ M2", s: { layout: "inside" } },
  { id: "bleed", label: "bleed-through fakes it", s: { f: 0, bleed: 0.4 } },
  { id: "shift", label: "shift by 3 px: r drops", s: { f: 1, shift: 3 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 25 : 18;

  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  let ch: Channels = makeChannels(st);
  let tA = 0, tB = 0;
  let animating = 0;
  let rand: number[] | null = null;

  const imgFig = htmlEl("figure", "co-img", "", host);
  const imgCap = htmlEl("figcaption", "", "A green, B magenta", imgFig);
  const cv = htmlEl("canvas", "", "", imgFig);
  cv.width = S * 3; cv.height = S * 3;
  const scFig = plotFigure(host, "2D histogram over the region", "co-sc qb-figure");
  const pl = new Plot(scFig, { width: 380, height: 380, x: [0, 200], y: [0, 200], xlabel: "channel A", ylabel: "channel B", font: FS, grid: false, margin: { l: 3.9 * FS, b: 3.4 * FS, t: 8, r: 10 } });
  const side = htmlEl("div", "co-side", "", host);
  const ro = htmlEl("div", "qb-readout", "", side);
  const rFig = plotFigure(side, "", "qb-figure");

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "co-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(true); } })));
  const fS = slider(panel, "shared f", 0, 1, 0.05, st.f, v => { st.f = v; unp(); rebuild(false); });
  const offS = slider(panel, "offset", 0, 60, 1, st.offset, v => { st.offset = v; unp(); rebuild(false); });
  const nS = slider(panel, "noise σ", 0, 20, 1, st.noise, v => { st.noise = v; unp(); rebuild(false); });
  const dS = slider(panel, "diffuse", 0, 60, 2, st.diffuse, v => { st.diffuse = v; unp(); rebuild(false); });
  const bS = slider(panel, "bleed A→B", 0, 0.8, 0.05, st.bleed, v => { st.bleed = v; unp(); rebuild(false); });
  const sS = slider(panel, "shift B (px)", 0, 6, 1, st.shift, v => { st.shift = v; unp(); rebuild(false); });
  htmlEl("div", "co-lab", "structures · region", panel);
  const layB = presets(panel, [
    { label: "mixture", apply: () => { st.layout = "mix"; unp(); rebuild(false); } },
    { label: "A inside B", apply: () => { st.layout = "inside"; unp(); rebuild(false); } },
  ]);
  const roiB = presets(panel, [
    { label: "inside cells", apply: () => { st.roi = "cells"; unp(); rand = null; update(); } },
    { label: "whole image", apply: () => { st.roi = "all"; unp(); rand = null; update(); } },
  ]);
  htmlEl("div", "co-lab", "thresholds (or drag the lines)", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "Costes", () => runCostes(true));
  button(row, "zero", () => { tA = 0; tB = 0; costesMsg = ""; update(); });
  button(row, "randomise B ×50", () => randomise());
  button(panel, "reset", () => { Object.assign(st, DEFAULT); sync(); unp(); rebuild(true); });

  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    fS.set(st.f); offS.set(st.offset); nS.set(st.noise); dS.set(st.diffuse); bS.set(st.bleed); sS.set(st.shift);
    layB.forEach((b, i) => b.classList.toggle("active", ["mix", "inside"][i] === st.layout));
    roiB.forEach((b, i) => b.classList.toggle("active", ["cells", "all"][i] === st.roi));
  }
  const roi = () => (st.roi === "cells" ? ch.cells : undefined);

  function rebuild(resetThr: boolean): void {
    ch = makeChannels(st);
    rand = null;
    sync();
    if (resetThr) { costesMsg = ""; if (st.thr === "zero") { tA = 0; tB = 0; } else runCostes(false); }
    update();
  }

  let costesMsg = "";
  function runCostes(animate: boolean): void {
    const st0 = startThresholds(ch, roi()), c = st0.costes;
    costesMsg = c.ok ? "" : "Costes not defined (r ≤ 0):\nthresholds at the 75th percentiles";
    if (!animate || !c.ok) { tA = st0.tA; tB = st0.tB; if (animate) update(); return; }
    const id = ++animating;
    let k = 0;
    const step = () => {
      if (id !== animating) return;
      const t = c.trace[Math.min(k, c.trace.length - 1)];
      tA = t.tA; tB = t.tB;
      update(`Costes step ${k + 1}: r below = ${t.r.toFixed(3)}`);
      k += 2;
      if (k < c.trace.length + 1) setTimeout(step, 40);
      else { tA = c.tA; tB = c.tB; update(); }
    };
    step();
  }

  function randomise(): void {
    rand = randomisation(ch, roi());
    update();
  }

  function drawImage(): void {
    const ctx = cv.getContext("2d")!;
    const id = ctx.createImageData(S, S);
    let ma = 1, mb = 1;
    for (let i = 0; i < S * S; i++) { ma = Math.max(ma, ch.a[i]); mb = Math.max(mb, ch.b[i]); }
    for (let i = 0; i < S * S; i++) {
      const g = Math.min(255, 255 * ch.a[i] / (0.8 * ma)), m = Math.min(255, 255 * ch.b[i] / (0.8 * mb));
      id.data[4 * i] = m; id.data[4 * i + 1] = g; id.data[4 * i + 2] = m; id.data[4 * i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = S; tmp.height = S;
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, S * 3, S * 3);
    if (st.roi === "cells") {
      ctx.fillStyle = "rgba(255,212,0,0.9)";
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const i = y * S + x;
        if (ch.cells[i] && ((x + 1 < S && !ch.cells[i + 1]) || (x > 0 && !ch.cells[i - 1]) || (y + 1 < S && !ch.cells[i + S]) || (y > 0 && !ch.cells[i - S]))) ctx.fillRect(x * 3, y * 3, 3, 3);
      }
    }
    imgCap.textContent = st.roi === "cells" ? "A green, B magenta; yellow: region (cells)" : "A green, B magenta; region: whole image";
  }

  let hiA = 200, hiB = 200;
  function update(msg = ""): void {
    drawImage();
    const m = roi();
    hiA = 10; hiB = 10;
    for (let i = 0; i < S * S; i++) if (!m || m[i]) { hiA = Math.max(hiA, ch.a[i]); hiB = Math.max(hiB, ch.b[i]); }
    hiA *= 1.02; hiB *= 1.02;
    const H2 = histogram2d(ch.a, ch.b, { binsX: BINS, binsY: BINS, minX: 0, maxX: hiA, minY: 0, maxY: hiB, mask: m });
    const logc = Array.from(H2.counts, v => (v > 0 ? Math.log10(v) : NaN));
    pl.clear();
    pl.setDomain([0, hiA], [0, hiB]);
    pl.heatmap(logc, BINS, BINS, { map: inferno, lo: -0.3, hi: Math.max(1, ...logc.filter(Number.isFinite)), flipY: false, maxCells: 100 });
    const cs = colocStats(ch, m, tA, tB);
    pl.fn(x => cs.slope * x + cs.intercept, { color: "#7fd3ff", width: 2.5, dash: "7 5" });
    pl.vline(tA, { color: "#2ca02c", width: 3 });
    pl.hline(tB, { color: "#d62728", width: 3 });
    const r = cs.r;
    const f = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "–");
    let txt = `Pearson r          ${f(r)}\nr above both t     ${f(cs.rAbove)}\noverlap coeff.     ${f(cs.overlap)}\nManders M1 (A)     ${f(cs.m1)}\nManders M2 (B)     ${f(cs.m2)}\n\ntA = ${tA.toFixed(1)}   tB = ${tB.toFixed(1)}\nslope of B on A    ${f(cs.slope, 2)}`;
    if (rand) {
      const exceed = rand.filter(v => v >= r).length;
      txt += `\nscrambled B: r from ${Math.min(...rand).toFixed(2)} to ${Math.max(...rand).toFixed(2)}\n${exceed} of 50 ≥ observed r`;
    }
    if (msg) txt += "\n" + msg;
    else if (costesMsg) txt += "\n" + costesMsg;
    ro.textContent = txt;
    rFig.replaceChildren();
    if (rand) {
      const lo = Math.min(-0.2, ...rand, r) - 0.05, hi = Math.max(0.2, ...rand, r) + 0.05, nb = 24, bw = (hi - lo) / nb;
      const cnt = new Array(nb).fill(0);
      for (const v of rand) cnt[Math.min(nb - 1, Math.floor((v - lo) / bw))]++;
      const rp = new Plot(rFig, { width: 300, height: 190, x: [lo, hi], y: [0, Math.max(...cnt) * 1.2], xlabel: "r with B scrambled", font: FS, margin: { l: 2.3 * FS } });
      rp.histogram(Array.from({ length: nb + 1 }, (_, k) => lo + k * bw), cnt, { color: "var(--qb-c5)" });
      rp.vline(r, { color: "var(--qb-c2)", width: 3, text: "observed" });
    }
  }

  // dragging the threshold lines
  let drag: "a" | "b" | null = null;
  pl.svg.addEventListener("pointerdown", ev => {
    const [x, y] = pl.toData(ev);
    const dx = Math.abs(pl.x(x) - pl.x(tA)), dy = Math.abs(pl.y(y) - pl.y(tB));
    drag = dx < dy ? "a" : "b";
    pl.svg.setPointerCapture(ev.pointerId);
    move(ev);
  });
  const move = (ev: PointerEvent) => {
    if (!drag) return;
    animating++;
    const [x, y] = pl.toData(ev);
    if (drag === "a") tA = Math.max(0, Math.min(hiA, x)); else tB = Math.max(0, Math.min(hiB, y));
    update();
  };
  pl.svg.addEventListener("pointermove", move);
  pl.svg.addEventListener("pointerup", () => { drag = null; });
  pl.svg.addEventListener("pointercancel", () => { drag = null; });

  sync();
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  rebuild(true);
}

autoMount(NAME, mount);
