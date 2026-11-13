// When do two spots become one? Two pixel-integrated Gaussian spots (PSF width σ) a distance d
// apart, amplitude ratio q, background 20 and seeded Gaussian noise, in a 31 × 19 patch.
// Panels: the profile through both spots (raw and after the matched filter), the image with
// the candidates (maxima of the matched-filtered image above 4 noise sd, window r = 2) and the
// fitted spots; the residuals of the one-spot and the two-spot least-squares fit; RSS₁, RSS₂
// gain and the penalty 3·c·σ_n² with the verdict; the fraction of 20 seeded repeats that end
// with two spots against d/σ, for maxima only and for maxima + two-spot fitting.
//
// <div data-demo="crowdfit" data-preset="three|two|filter|split|unequal|penalty" data-presets="..."></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { fromArray, gaussianBlur } from "../lib/imageops";
import { pixelGaussian } from "../lib/spotfit";
import { W, H, CX, CY, BG, A1, State, analyse } from "./model";
import { ImgView, injectStyle, label, shownIds, TP_COL, FP_COL, CAND_COL } from "../spotdetect/common";

const NAME = "crowdfit";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .cf-main { flex: 0 1 430px; min-width: 300px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .cf-mid { flex: 0 1 320px; min-width: 200px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .cf-res { display: grid; grid-template-columns: 1fr 1fr; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .cf-res .ev-wrap { aspect-ratio: 31 / 19; }
.qb-demo[data-demo="${NAME}"] .cf-res figcaption { font-size: 16px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 290px; max-width: 360px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .cf-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cf-main { flex: 0 0 420px; }
.reveal .qb-demo[data-demo="${NAME}"] .cf-mid { flex: 0 0 320px; gap: .05em; }
.reveal .qb-demo[data-demo="${NAME}"] .cf-res figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { sigma: 1.5, dr: 3, q: 1, noise: 6, c: 2, seed: 1 };
const PRESETS: Preset[] = [
  { id: "three", label: "d = 3σ: two maxima", s: { dr: 3 } },
  { id: "two", label: "d = 2σ: one maximum", s: { dr: 2 } },
  { id: "filter", label: "the filter merges them earlier", s: { dr: 2.6 } },
  { id: "split", label: "the fit splits what the maxima cannot", s: { dr: 1.8 } },
  { id: "unequal", label: "unequal pair: maxima lose the dim one", s: { dr: 2.5, q: 0.3 } },
  { id: "penalty", label: "penalty too small: noise gets a spot", s: { dr: 0, c: 0 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "cf-main", "", host);
  const prof = new Plot(main, { width: 430, height: 190, x: [0, W - 1], y: [0, 1], font: 17, margin: { l: 52, r: 8, t: 24, b: 30 }, xticks: [] });
  const pRaw = prof.line([], [], { color: "#555", width: 2 });
  const pClean = prof.line([], [], { color: "#555", width: 1.5, dash: "3 3" });
  const pFilt = prof.line([], [], { color: seriesColor(0), width: 3 });
  const pTxt = prof.text(0.5, 1, "", { size: 16 });
  const view = new ImgView(main, "");
  view.wrap.style.aspectRatio = `${W} / ${H}`;
  const mid = htmlEl("div", "cf-mid", "", host);
  const res = htmlEl("div", "cf-res", "", mid);
  const r1 = new ImgView(res, "residual, 1 spot"), r2 = new ImgView(res, "residual, 2 spots");
  const bar = new Plot(mid, { width: 320, height: 150, x: [0, 2], y: [0, 1], font: 17, margin: { l: 64, r: 8, t: 12, b: 34 }, xticks: [0.5, 1.5], xfmt: v => (v < 1 ? "RSS₁ − RSS₂" : "penalty"), yticks: 3 });
  const bars = bar.bars([0.5, 1.5], [0, 0], { colors: [TP_COL, FP_COL], barWidth: 0.6 });
  const sp = new Plot(mid, { width: 320, height: 190, x: [0, 6], y: [0, 1.02], font: 17, margin: { l: 56, r: 8, t: 12, b: 50 }, xlabel: "d / σ", yticks: [0, 0.5, 1] });
  const sMax = sp.line([], [], { color: CAND_COL, width: 3 });
  const sFit = sp.line([], [], { color: TP_COL, width: 3 });
  sp.vline(2, { color: "#888", width: 1.5, text: "2" }); sp.vline(2 * Math.SQRT2, { color: "#888", width: 1.5, text: "2√2" });
  const sNow = sp.vline(3, { color: "var(--qb-c4)", width: 2 });
  const busy = htmlEl("div", "cf-busy", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const dS = slider(panel, "separation d/σ", 0, 6, 0.1, st.dr, v => { st.dr = v; run(); });
  const sS = slider(panel, "PSF σ (px)", 1, 2.5, 0.1, st.sigma, v => { st.sigma = v; run(true); });
  const qS = slider(panel, "amplitude ratio", 0.1, 1, 0.05, st.q, v => { st.q = v; run(true); });
  const nS = slider(panel, "noise sd", 0.5, 10, 0.5, st.noise, v => { st.noise = v; run(true); });
  const cS = slider(panel, "penalty c", 0, 10, 0.5, st.c, v => { st.c = v; run(true); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "new noise", () => { st.seed++; run(); });
  button(brow, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); run(true); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); run(true); }
  function sync(): void { dS.set(st.dr); sS.set(st.sigma); qS.set(st.q); nS.set(st.noise); cS.set(st.c); }

  let curveKey = "";
  function run(curves = false): void {
    const A = analyse(st, 1000 + st.seed), d = st.dr * st.sigma;
    // profile through the spots' row
    const xs = Array.from({ length: W }, (_, i) => i), row = xs.map(i => A.img[CY * W + i]);
    const tot = 2 * Math.PI * st.sigma ** 2;
    const clean = pixelGaussian(CX - d / 2, CY, st.sigma, A1 * tot, BG, W, H), c2 = pixelGaussian(CX + d / 2, CY, st.sigma, st.q * A1 * tot, 0, W, H);
    const cleanRow = xs.map(i => clean[CY * W + i] + c2[CY * W + i]);
    const fr = xs.map(i => A.filt[CY * W + i] + BG);
    const hi = Math.max(...row, ...cleanRow) * 1.25;
    prof.setDomain(undefined, [0, hi]);
    pRaw.set(xs, row); pClean.set(xs, cleanRow); pFilt.set(xs, fr);
    const cnt = (v: number[]) => { let n = 0; for (let i = 1; i < v.length - 1; i++) if (v[i] > v[i - 1] && v[i] >= v[i + 1] && v[i] > BG + 5) n++; return n; };
    const cleanF = Array.from(gaussianBlur(fromArray(W, H, Float64Array.from(clean, (v, i) => v + c2[i])), st.sigma, "mirror").data.slice(CY * W, CY * W + W));
    pTxt.set(0.5, hi * 0.92, `noise-free maxima: raw ${cnt(cleanRow)}, filtered ${cnt(cleanF)}`);
    // image
    view.grey(fromArray(W, H, A.img), { mode: "auto" });
    view.clear();
    view.setCaption(`d = ${d.toFixed(1)} px (${st.dr.toFixed(1)} σ): ${A.cands.length} candidate${A.cands.length === 1 ? "" : "s"} → ${A.final} spot${A.final === 1 ? "" : "s"}`);
    for (const x of [CX - d / 2, CX + d / 2]) view.add("circle", { cx: x + 0.5, cy: CY + 0.5, r: 0.35, fill: "#ffd400", stroke: "#000", "stroke-width": 1 });
    for (const c of A.cands) view.add("circle", { cx: c.x + 0.5, cy: c.y + 0.5, r: 1.6, fill: "none", stroke: CAND_COL, "stroke-width": 2.5 });
    const fits = A.final >= 2 && A.cands.length < 2 ? A.f2.spots : A.final === 1 ? A.f1.spots : [];
    for (const f of fits) {
      const x = f.x + 0.5, y = f.y + 0.5;
      view.add("path", { d: `M${x - 0.8},${y - 0.8}L${x + 0.8},${y + 0.8}M${x - 0.8},${y + 0.8}L${x + 0.8},${y - 0.8}`, stroke: TP_COL, "stroke-width": 3 });
    }
    // residuals (same scale)
    const res1 = Float32Array.from(A.img, (v, i) => v - A.f1.model[i]), res2 = Float32Array.from(A.img, (v, i) => v - A.f2.model[i]);
    let m = 1; for (const v of res1) m = Math.max(m, Math.abs(v));
    r1.wrap.style.aspectRatio = r2.wrap.style.aspectRatio = `${W} / ${H}`;
    r1.grey({ width: W, height: H, data: res1 }, { mode: "clip", lo: -m, hi: m });
    r2.grey({ width: W, height: H, data: res2 }, { mode: "clip", lo: -m, hi: m });
    // RSS bars
    const top = Math.max(A.sel.penalty, A.sel.gain, 1) * 1.15;
    bar.setDomain(undefined, [0, top]);
    bars.set([0.5, 1.5], [Math.max(0, A.sel.gain), A.sel.penalty]);
    sNow.set(st.dr);
    ro.textContent =
      `candidates (filter + maxima): ${A.cands.length}\n` +
      `RSS₁ ${A.f1.rss.toFixed(0)}, RSS₂ ${A.f2.rss.toFixed(0)}\n` +
      `gain ${A.sel.gain.toFixed(0)} vs penalty 3·c·σn² = ${A.sel.penalty.toFixed(0)}\n` +
      `fit verdict: ${A.sel.keepTwo ? "2 spots" : "1 spot"}` +
      (A.sel.keepTwo ? `\nfitted at x = ${A.f2.spots.map(s => s.x.toFixed(2)).join(", ")} (truth ${(CX - d / 2).toFixed(2)}, ${(CX + d / 2).toFixed(2)})` : `\nfitted at x = ${A.f1.spots[0].x.toFixed(2)}`);
    // split fraction against d (seeded repeats)
    const key = `${st.sigma}:${st.q}:${st.noise}:${st.c}`;
    if (curves || key !== curveKey) {
      curveKey = key; sMax.set([], []); sFit.set([], []);
      busy.textContent = "simulating 20 pairs per separation …";
      setTimeout(() => {
        if (curveKey !== key) return;
        const t0 = performance.now(), ds = Array.from({ length: 12 }, (_, i) => 0.5 + 0.5 * i), fm: number[] = [], ff: number[] = [];
        for (const dr of ds) {
          let a = 0, b = 0;
          for (let r = 0; r < 20; r++) { const R = analyse({ ...st, dr }, 50 + r); if (R.cands.length === 2) a++; if (R.final === 2) b++; }
          fm.push(a / 20); ff.push(b / 20);
        }
        sMax.set(ds, fm); sFit.set(ds, ff);
        busy.textContent = `two spots found (blue: maxima only, green: + fitting), 20 repeats, ${Math.round(performance.now() - t0)} ms`;
      }, 20);
    }
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  run(true);
}

autoMount(NAME, mount);
