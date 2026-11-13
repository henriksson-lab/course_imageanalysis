// Sub-pixel localisation of one spot. A 13 × 13 pixel patch with one simulated spot
// (pixel-integrated Gaussian of width s = 100 nm, or an Airy spot with the same Gaussian
// width), N photons, b background photons per pixel, pixel size a, seeded Poisson noise.
// Tabs:
//  - one spot: click/drag to move the true centre; estimates of the brightest pixel, the
//    centroid (window 2m+1, optionally minus the background), the three-point parabola and
//    log-parabola, the least-squares Gaussian fit (with a step-through of its iterations) and
//    the Poisson maximum-likelihood fit; the residual image of the least-squares fit;
//  - bias: estimate − truth against the true offset within a pixel (noise-free);
//  - precision: 500 seeded repeats (scatter, nm) and RMS error against N for the centroid,
//    least squares and maximum likelihood, with the Cramér–Rao bound s²/N, Thompson's
//    formula and Mortensen's least-squares and maximum-likelihood formulas.
//
// <div data-demo="spotfit" data-preset="peak|centroidbg|locking|logexact|fit|thousand|background|pixel|airy|mle"
//      data-presets="..." data-tab="0|1|2"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { fromArray } from "../lib/imageops";
import { median } from "../lib/stats";
import {
  pixelGaussian, pixelAiry, poissonImage, peakPixel, centroid, threePoint, gauss3Point, parabola3Point, fitSpots, FitResult,
  crlbGaussian, thompsonVar, mortensenLSQ, mortensenMLE,
} from "../lib/spotfit";
import { ImgView, injectStyle, label, shownIds, logSlider, tabRow } from "../spotdetect/common";

const NAME = "spotfit";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .sf-main { flex: 0 1 min(410px, calc(100vh - 250px)); min-width: 300px; }
.qb-demo[data-demo="${NAME}"] .sf-mid { flex: 0 1 280px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; max-width: 370px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider { grid-template-columns: auto 1fr 3.2em; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider input[type=range] { min-width: 4em; }
.qb-demo[data-demo="${NAME}"] .sf-mid > .qb-readout { font-size: 14px; }
.qb-demo[data-demo="${NAME}"] .sf-mid .sf-legend { font-size: 15px; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .sf-legend { font-size: 16px; line-height: 1.5; }
.qb-demo[data-demo="${NAME}"] .sf-legend span { display: inline-block; margin-right: .7em; white-space: nowrap; }
.qb-demo[data-demo="${NAME}"] .sf-legend b { display: inline-block; width: .9em; height: .9em; border-radius: 50%; vertical-align: -0.1em; margin-right: .25em; }
.qb-demo[data-demo="${NAME}"] .sf-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .sf-main { flex: 0 0 420px; }
.reveal .qb-demo[data-demo="${NAME}"] .sf-mid { flex: 0 0 300px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 370px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .sf-legend { font-size: 18px; }
`;

const G = 13, C = 6, S_NM = 100;
const EST = [
  { id: "peak", label: "brightest pixel", col: "#999999" },
  { id: "cen", label: "centroid", col: seriesColor(0) },
  { id: "par", label: "parabola", col: seriesColor(3) },
  { id: "logp", label: "log-parabola", col: seriesColor(4) },
  { id: "lsq", label: "least squares", col: seriesColor(2) },
  { id: "mle", label: "max. likelihood", col: seriesColor(1) },
] as const;
type EstId = typeof EST[number]["id"];

interface State { tab: number; N: number; b: number; a: number; airy: boolean; m: number; subBg: boolean; x0: number; y0: number; seed: number; step: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { tab: 0, N: 1000, b: 5, a: 100, airy: false, m: 2, subBg: false, x0: C + 0.3, y0: C - 0.2, seed: 1, step: -1 };
const PRESETS: Preset[] = [
  { id: "peak", label: "pixel peak: ±0.5 px", s: { tab: 0, b: 0 } },
  { id: "centroidbg", label: "centroid with background: pulled to the centre", s: { tab: 1, b: 20, m: 3 } },
  { id: "locking", label: "parabola: pixel locking", s: { tab: 1, b: 0 } },
  { id: "logexact", label: "log-parabola is exact for a Gaussian", s: { tab: 1, b: 0, subBg: true } },
  { id: "fit", label: "fit all pixels", s: { tab: 0, step: 0 } },
  { id: "thousand", label: "1 000 photons: a few nm", s: { tab: 2, N: 1000, b: 0 } },
  { id: "background", label: "background dominates dim spots", s: { tab: 2, N: 300, b: 20 } },
  { id: "pixel", label: "pixel size: too big and too small", s: { tab: 2, N: 1000, b: 5, a: 250 } },
  { id: "airy", label: "Airy spot, Gaussian model: small bias", s: { tab: 1, b: 0, airy: true } },
  { id: "mle", label: "max. likelihood vs least squares, 100 photons", s: { tab: 2, N: 100, b: 2 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = host.dataset.tab ? Number(host.dataset.tab) : undefined;
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}), ...(tab0 !== undefined && !p0 ? { tab: tab0 } : {}) });
  const st = start();

  const main = htmlEl("div", "sf-main", "", host);
  const setTab = tabRow(main, ["one spot", "bias", "precision"], i => { st.tab = i; run(); });
  const view = new ImgView(main, "");
  const biasP = new Plot(main, { width: 430, height: 400, x: [-0.5, 0.5], y: [-0.3, 0.3], font: 18, xlabel: "true offset from pixel centre (px)", ylabel: "estimate − truth (px)", margin: { l: 78, r: 10, t: 14, b: 64 } });
  const biasLines = EST.slice(1, 5).map(e => biasP.line([], [], { color: e.col, width: 3 }));
  biasP.hline(0, { color: "#999", width: 1.5 });
  const scP = new Plot(main, { width: 430, height: 400, x: [-50, 50], y: [-50, 50], font: 18, equal: true, xlabel: "x error (nm)", ylabel: "y error (nm)", margin: { l: 78, r: 10, t: 14, b: 64 } });
  const scGrid = scP.mark<[number]>((g, a) => {
    for (let k = -6; k <= 6; k++) {
      const v = (k + 0.5) * a;
      for (const [x1, y1, x2, y2] of [[v, -400, v, 400], [-400, v, 400, v]]) {
        const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
        l.setAttribute("x1", String(scP.x(x1))); l.setAttribute("x2", String(scP.x(x2))); l.setAttribute("y1", String(scP.y(y1))); l.setAttribute("y2", String(scP.y(y2)));
        l.style.stroke = "#bbb"; l.style.strokeWidth = "1.5"; g.appendChild(l);
      }
    }
  }, [100]);
  const scDots = (["cen", "lsq", "mle"] as EstId[]).map(id => scP.scatter([], [], { color: EST.find(e => e.id === id)!.col, r: 2.6, opacity: 0.55 }));
  const mid = htmlEl("div", "sf-mid", "", host);
  const resid = new ImgView(mid, "residual: data − least-squares model");
  const precP = new Plot(mid, { width: 320, height: 330, x: [50, 20000], y: [0.5, 200], xlog: true, ylog: true, font: 17, xlabel: "photons N", ylabel: "RMS error (nm)", margin: { l: 66, r: 8, t: 10, b: 60 } });
  const crlbL = precP.line([], [], { color: "#222", width: 2.5 });
  const thomL = precP.line([], [], { color: "#888", width: 2.5, dash: "6 4" });
  const mlsL = precP.line([], [], { color: EST[4].col, width: 2, dash: "3 3" });
  const mmlL = precP.line([], [], { color: EST[5].col, width: 2, dash: "3 3" });
  const precDots = (["cen", "lsq", "mle"] as EstId[]).map(id => precP.scatter([], [], { color: EST.find(e => e.id === id)!.col, r: 5.5 }));
  const nMark = precP.vline(1000, { color: "var(--qb-c4)", width: 2 });
  const legend = htmlEl("div", "sf-legend", "", mid);
  const busy = htmlEl("div", "sf-busy", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const nS = logSlider(panel, "photons N", 50, 20000, st.N, v => { st.N = Math.round(v); run(); }, v => v.toFixed(0));
  const bS = slider(panel, "background b", 0, 50, 1, st.b, v => { st.b = v; run(); });
  const aS = slider(panel, "pixel size (nm)", 50, 400, 10, st.a, v => { st.a = v; run(); });
  const mS = slider(panel, "centroid window m", 1, 6, 1, st.m, v => { st.m = v; run(); });
  const row = htmlEl("div", "", "", panel);
  row.style.display = "flex"; row.style.flexWrap = "wrap"; row.style.gap = "0 1em";
  const airyC = checkbox(row, "Airy spot", st.airy, v => { st.airy = v; run(); });
  const bgC = checkbox(row, "subtract b first", st.subBg, v => { st.subBg = v; run(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  const stepB = button(brow, "step", () => { st.tab = 0; st.step = st.step < 0 ? 0 : st.step + 1; run(); });
  button(brow, "all steps", () => { st.step = -1; run(); });
  button(brow, "new noise", () => { st.seed++; run(); });
  button(brow, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); run(); });
  // on a page under the residual (keeps the panel short); on a slide in the panel
  const ro = htmlEl("div", "qb-readout", "", host.closest(".reveal") ? panel : mid);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); run(); }
  function sync(): void { nS.setValue(st.N); bS.set(st.b); aS.set(st.a); mS.set(st.m); airyC.checked = st.airy; bgC.checked = st.subBg; }

  // drag the true centre (fractional pixel coordinates)
  const toPix = (ev: PointerEvent): [number, number] => {
    const r = view.wrap.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * G - 0.5, (ev.clientY - r.top) / r.height * G - 0.5];
  };
  let dragging = false;
  view.wrap.style.cursor = "crosshair"; view.wrap.style.touchAction = "none";
  view.wrap.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; view.wrap.setPointerCapture(ev.pointerId); move(ev); });
  view.wrap.addEventListener("pointermove", ev => { if (dragging) move(ev); });
  view.wrap.addEventListener("pointerup", () => { dragging = false; });
  function move(ev: PointerEvent): void {
    const [x, y] = toPix(ev);
    st.x0 = Math.max(C - 2, Math.min(C + 2, Math.round(x * 100) / 100)); st.y0 = Math.max(C - 2, Math.min(C + 2, Math.round(y * 100) / 100));
    run();
  }

  const sPx = () => S_NM / st.a;
  const model = (x0: number, y0: number) => (st.airy ? pixelAiry : pixelGaussian)(x0, y0, sPx(), st.N, st.b, G);

  function estimate(img: ArrayLike<number>, which: EstId[]): Partial<Record<EstId, [number, number]>> & { fit?: FitResult } {
    const out: Partial<Record<EstId, [number, number]>> & { fit?: FitResult } = {};
    const [px, py] = peakPixel(img, G);
    const bg = st.subBg ? st.b : 0;
    if (which.includes("peak")) out.peak = [px, py];
    if (which.includes("cen")) out.cen = centroid(img, G, G, px, py, st.m, bg);
    if (which.includes("par")) out.par = threePoint(img, G, G, px, py, parabola3Point, bg);
    if (which.includes("logp")) out.logp = threePoint(img, G, G, px, py, gauss3Point, bg);
    if (which.includes("lsq") || which.includes("mle")) {
      const ring: number[] = [];
      for (let i = 0; i < G; i++) { ring.push(img[i], img[(G - 1) * G + i], img[i * G], img[i * G + G - 1]); }
      const B0 = Math.max(0.5, median(ring)), tot = Array.from(img).reduce((s, v) => s + v, 0);
      const init = [{ x: px, y: py, N: Math.max(10, tot - B0 * G * G) }];
      if (which.includes("lsq")) { const f = fitSpots(img, G, G, sPx(), init, B0); out.lsq = [f.spots[0].x, f.spots[0].y]; out.fit = f; }
      if (which.includes("mle")) { const f = fitSpots(img, G, G, sPx(), init, B0, { mle: true }); out.mle = [f.spots[0].x, f.spots[0].y]; }
    }
    return out;
  }

  let precKey = "", precCurve: { N: number; rms: number[] }[] = [];
  function run(): void {
    sync();
    setTab(st.tab);
    view.fig.style.display = st.tab === 0 ? "" : "none";
    resid.fig.style.display = st.tab === 0 ? "" : "none";
    biasP.svg.style.display = st.tab === 1 ? "" : "none";
    scP.svg.style.display = st.tab === 2 ? "" : "none";
    precP.svg.style.display = st.tab === 2 ? "" : "none";
    stepB.disabled = st.tab !== 0;
    const shownEst = st.tab === 0 ? EST : st.tab === 1 ? EST.slice(1, 5) : EST.filter(e => ["cen", "lsq", "mle"].includes(e.id));
    legend.innerHTML = shownEst.map(e => `<span><b style="background:${e.col}"></b>${e.label}</span>`).join("") +
      (st.tab === 2 ? `<span><b style="background:#222"></b>bound s²/N</span><span><b style="background:#888"></b>Thompson</span><span>dashed: Mortensen</span>` : "");
    if (st.tab === 0) oneSpot();
    else if (st.tab === 1) bias();
    else precision();
  }

  function oneSpot(): void {
    const mu = model(st.x0, st.y0), img = poissonImage(mu, 100 + st.seed);
    const e = estimate(img, EST.map(x => x.id));
    view.grey(fromArray(G, G, img), { mode: "auto" });
    view.clear();
    view.setCaption(`${st.N} photons, b = ${st.b}, pixel ${st.a} nm (s = ${sPx().toFixed(2)} px)`);
    const fit = e.fit!, its = fit.iterations, k = st.step < 0 ? its.length - 1 : Math.min(st.step, its.length - 1);
    for (let i = 0; i <= k; i++) {
      const p = its[i].spots[0];
      view.add("circle", { cx: p.x + 0.5, cy: p.y + 0.5, r: sPx(), fill: "none", stroke: EST[4].col, "stroke-width": i === k ? 2.5 : 1, opacity: i === k ? 1 : 0.5 });
    }
    // markers
    EST.forEach(est => {
      const p = e[est.id]; if (!p) return;
      if (est.id === "peak") view.add("rect", { x: p[0], y: p[1], width: 1, height: 1, fill: "none", stroke: est.col, "stroke-width": 3 });
      else view.add("circle", { cx: p[0] + 0.5, cy: p[1] + 0.5, r: 0.17, fill: est.col, stroke: "#fff", "stroke-width": 1.5 });
    });
    const tx = st.x0 + 0.5, ty = st.y0 + 0.5;
    view.add("path", { d: `M${tx - 0.45},${ty}H${tx + 0.45}M${tx},${ty - 0.45}V${ty + 0.45}`, stroke: "#fff", "stroke-width": 3, fill: "none" });
    view.add("path", { d: `M${tx - 0.45},${ty}H${tx + 0.45}M${tx},${ty - 0.45}V${ty + 0.45}`, stroke: "#000", "stroke-width": 1.2, fill: "none" });
    // residual
    const res = Float32Array.from(img, (v, q) => v - fit.model[q]);
    let mx = 1; for (const v of res) mx = Math.max(mx, Math.abs(v));
    resid.grey({ width: G, height: G, data: res }, { mode: "clip", lo: -mx, hi: mx });
    const nm = (p?: [number, number]) => p ? `${(Math.hypot(p[0] - st.x0, p[1] - st.y0) * st.a).toFixed(1)} nm` : "–";
    ro.textContent = EST.map(est => `${est.label.padEnd(16)} ${nm(e[est.id])}`).join("\n") +
      `\nfit step ${k} of ${its.length - 1}: RSS ${its[k].rss.toFixed(0)}`;
  }

  function bias(): void {
    const xs = Array.from({ length: 41 }, (_, i) => -0.5 + i / 40);
    const ys: number[][] = [[], [], [], []];
    for (const t of xs) {
      const mu = model(C + t, C), e = estimate(mu, ["cen", "par", "logp", "lsq"]);
      ys[0].push(e.cen![0] - C - t); ys[1].push(e.par![0] - C - t); ys[2].push(e.logp![0] - C - t); ys[3].push(e.lsq![0] - C - t);
    }
    let m = 0.05; for (const y of ys) for (const v of y) m = Math.max(m, Math.abs(v));
    biasP.setDomain(undefined, [-1.15 * m, 1.15 * m]);
    ys.forEach((y, i) => biasLines[i].set(xs, y));
    const at = (i: number, t: number) => ys[i][Math.round((t + 0.5) * 40)];
    ro.textContent = `noise-free, ${st.airy ? "Airy" : "Gaussian"} spot, b = ${st.b}${st.subBg ? " (subtracted)" : ""}\nbias at offset 0.3 px:\n` +
      ["centroid", "parabola", "log-parabola", "least squares"].map((n, i) => `${n.padEnd(14)} ${at(i, 0.3) >= 0 ? " " : ""}${at(i, 0.3).toFixed(3)} px`).join("\n");
  }

  function precision(): void {
    const reps = 500, errs: [number, number][][] = [[], [], []];
    for (let r = 0; r < reps; r++) {
      const img = poissonImage(model(st.x0, st.y0), 5000 + r), e = estimate(img, ["cen", "lsq", "mle"]);
      (["cen", "lsq", "mle"] as EstId[]).forEach((id, i) => errs[i].push([(e[id]![0] - st.x0) * st.a, (e[id]![1] - st.y0) * st.a]));
    }
    const rms = errs.map(E => Math.sqrt(E.reduce((s, [x]) => s + x * x, 0) / E.length));
    let lim = 10; for (const v of rms) lim = Math.max(lim, 3.2 * Math.min(v, 150));
    scP.setDomain([-lim, lim], [-lim, lim]);
    scGrid.set(st.a);
    errs.forEach((E, i) => scDots[i].set(E.map(p => p[0]), E.map(p => p[1])));
    // RMS vs N (computed once per b, a, model, truth)
    const key = `${st.b}:${st.a}:${st.airy}:${st.x0}:${st.y0}:${st.m}:${st.subBg}`;
    const Ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    const sA = sPx();
    const curve = (f: (N: number) => number) => Ns.map(N => Math.sqrt(f(N)) * st.a);
    crlbL.set(Ns, curve(N => crlbGaussian(sA, N)));
    thomL.set(Ns, curve(N => thompsonVar(sA, 1, st.b, N)));
    mlsL.set(Ns, curve(N => mortensenLSQ(sA, 1, st.b, N)));
    mmlL.set(Ns, curve(N => mortensenMLE(sA, 1, st.b, N)));
    nMark.set(st.N);
    if (key !== precKey) {
      precKey = key; precDots.forEach(d => d.set([], []));
      busy.textContent = "computing RMS error against N …";
      setTimeout(() => {
        if (precKey !== key) return;
        const t0 = performance.now();
        precCurve = Ns.map(N => {
          const E: number[][] = [[], [], []];
          for (let r = 0; r < 120; r++) {
            const mu = (st.airy ? pixelAiry : pixelGaussian)(st.x0, st.y0, sA, N, st.b, G);
            const img = poissonImage(mu, 9000 + r), e = estimate(img, ["cen", "lsq", "mle"]);
            (["cen", "lsq", "mle"] as EstId[]).forEach((id, i) => E[i].push(Math.hypot(e[id]![0] - st.x0, 0) * st.a));
          }
          return { N, rms: E.map(v => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length)) };
        });
        precDots.forEach((d, i) => d.set(Ns, precCurve.map(c => Math.max(0.5, Math.min(200, c.rms[i])))));
        busy.textContent = `RMS error (x) from 120 seeded repeats per N, ${Math.round(performance.now() - t0)} ms`;
      }, 20);
    } else precDots.forEach((d, i) => d.set(Ns, precCurve.map(c => Math.max(0.5, Math.min(200, c.rms[i])))));
    const bound = Math.sqrt(crlbGaussian(sA, st.N)) * st.a, th = Math.sqrt(thompsonVar(sA, 1, st.b, st.N)) * st.a;
    ro.textContent = `${reps} repeats, N = ${st.N}, b = ${st.b}, pixel ${st.a} nm\nRMS error per axis (x):\n` +
      ["centroid", "least squares", "max. likelihood"].map((n, i) => `${n.padEnd(15)} ${rms[i].toFixed(1)} nm`).join("\n") +
      `\nbound s/√N ${bound.toFixed(1)} nm, Thompson ${th.toFixed(1)} nm`;
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  run();
}

autoMount(NAME, mount);
