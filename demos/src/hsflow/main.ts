// Horn–Schunck optical flow: data + λ·smoothness, solved by sweeps that average the
// neighbours and step towards each pixel's constraint line. The sweeps are animated, so
// the flow can be seen filling in flat regions. Same frame pairs as lkflow (lib/flow.ts):
// the micrograph moved by a known flow, a textured ring around a flat disc (filling-in),
// and a textured cell moving over a still substrate (a motion boundary). Tabs: flow,
// arrows, endpoint error, Lucas–Kanade for comparison, the warping residual (no truth
// needed), the frames. A button computes the mean EPE against λ.
//
// <div data-demo="hsflow" data-preset="small|large|fill|boundary|lkhs|choose|residual"
//      data-tab="flow|arrows|epe|lk|resid|frames"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox } from "../lib/ui";
import { Grey, cropGrey } from "../lib/imageops";
import {
  FieldKind, makePair, FramePair, HornSchunck, hsPyramid, lucasKanade, Flow, endpointError, flowToRGBA, warpResidual,
  ringScene, cellScene, compositePair, maxSpeed, interiorMean,
} from "../lib/flow";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Plot } from "../lib/plot";
import { inferno } from "../lib/colour";
import { ImgView } from "../canny/imgview";
import { injectCss, flowWheel, drawQuiver, WHEEL_CSS, fmtSpeed } from "../lkflow/flowui";

const NAME = "hsflow", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .9em; flex-wrap: wrap; }
${S} .hs-main { flex: 0 1 400px; min-width: 240px; }
${S} .hs-mid { flex: 0 1 320px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .3em; }
${S} .hs-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
${S} .hs-tabs .qb-btn { padding: .12em .45em; }
${S} .qb-readout { font-size: 15px; }
${S} .hs-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .hs-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .hs-main { flex: 0 0 390px; }
.reveal ${S} .hs-mid { flex: 0 0 310px; }
.reveal ${S} .qb-panel { flex: 1 1 370px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .hs-lab { font-size: 16px; }
.reveal ${S} .ev-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
${WHEEL_CSS(S)}
`;

type Scene = "cell" | "ring" | "boundary";
type Tab = "flow" | "arrows" | "epe" | "lk" | "resid" | "frames";
const TABS: [Tab, string][] = [["flow", "flow"], ["arrows", "arrows"], ["epe", "error (EPE)"], ["lk", "Lucas–Kanade"], ["resid", "residual"], ["frames", "frame 1"]];
const MOTIONS: [FieldKind, string][] = [["translation", "shift"], ["rotation", "rotation"], ["swirl", "swirl"], ["regions", "disc moves"]];
interface State { scene: Scene; motion: FieldKind; size: number; noise: number; lambda: number; robust: boolean; pyramid: boolean; target: number; tab: Tab }
const DEFAULT: State = { scene: "cell", motion: "translation", size: 1.5, noise: 4, lambda: 300, robust: false, pyramid: false, target: 300, tab: "flow" };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "small", label: "λ small: noisy, holes", s: { lambda: 1, tab: "flow" } },
  { id: "large", label: "λ large: boundaries smeared", s: { scene: "boundary", motion: "regions", lambda: 10000, target: 1500, tab: "arrows" } },
  { id: "fill", label: "filling-in: watch it spread", s: { scene: "ring", noise: 2, lambda: 100, target: 2000, tab: "flow" } },
  { id: "boundary", label: "boundary: quadratic vs robust", s: { scene: "boundary", motion: "regions", lambda: 300, robust: true, target: 500, tab: "epe" } },
  { id: "lkhs", label: "LK vs HS", s: { scene: "ring", noise: 2, lambda: 100, target: 2000, tab: "lk" } },
  { id: "choose", label: "choose λ by the EPE", s: { tab: "epe" } },
  { id: "residual", label: "residual without truth: the flat trap", s: { scene: "ring", noise: 2, lambda: 100, target: 60, tab: "resid" } },
  { id: "pyr", label: "4 px rotation: pyramid", s: { motion: "rotation", size: 4, pyramid: true, tab: "arrows" } },
];
const LMIN = 0.1, LMAX = 1e5;
const lamOf = (t: number) => Number((LMIN * (LMAX / LMIN) ** (t / 100)).toPrecision(2));
const tOfLam = (l: number) => 100 * Math.log(l / LMIN) / Math.log(LMAX / LMIN);

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = TABS.find(t => t[0] === host.dataset.tab)?.[0];
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}), ...(tab0 ? { tab: tab0 } : {}) });
  const st = start();
  const N = 128;

  const main = htmlEl("div", "hs-main", "", host);
  const tabs = htmlEl("div", "hs-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabs, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);

  const mid = htmlEl("div", "hs-mid", "", host);
  htmlEl("div", "hs-lab", "mean endpoint error against λ", mid);
  const plot = new Plot(mid, { width: 320, height: 230, x: [LMIN, LMAX], y: [0, 1], xlog: true, xticks: [0.1, 10, 1000, 100000], xfmt: v => (v >= 1000 ? `10${["³", "⁴", "⁵"][Math.round(Math.log10(v)) - 3]}` : String(v)), xlabel: "λ", ylabel: "mean EPE (px)", font: 18, margin: { l: 66, r: 12, t: 10, b: 58 } });
  const curve = plot.line([], [], { color: "var(--qb-c1)", width: 3 });
  const curveDots = plot.scatter([], [], { color: "var(--qb-c1)", r: 4 });
  const lkLine = plot.hline(0, { color: "var(--qb-c3)", dash: "6 4", width: 2, text: "LK" });
  const lamMark = plot.vline(st.lambda, { color: "var(--qb-c2)", width: 2.5 });
  const curveBtn = button(mid, "compute EPE for 11 values of λ", () => sweepLambda());
  const ro = htmlEl("div", "qb-readout", "", mid);
  const wheel = flowWheel(mid, 72);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hs-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  htmlEl("div", "hs-lab", "scene and motion", panel);
  const sceneBtns = presets(panel, [
    { label: "micrograph", apply: () => { st.scene = "cell"; if (st.motion === "regions") st.motion = "translation"; rebuild(); } },
    { label: "ring + flat", apply: () => { st.scene = "ring"; if (st.motion === "regions") st.motion = "translation"; rebuild(); } },
    { label: "cell on substrate", apply: () => { st.scene = "boundary"; st.motion = "regions"; rebuild(); } },
  ]);
  const motionBtns = presets(panel, MOTIONS.map(([k, l]) => ({ label: l, apply: () => { st.motion = k; if (k === "regions") st.scene = "boundary"; else if (st.scene === "boundary") st.scene = "cell"; rebuild(); } })));
  const lamS = slider(panel, "λ", 0, 100, 1, tOfLam(st.lambda), t => { st.lambda = lamOf(t); restart(); }, t => String(lamOf(t)));
  const sizeS = slider(panel, "largest motion (px)", 0, 6, 0.25, st.size, v => { st.size = v; rebuild(); });
  const noiseS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; rebuild(); });
  const row = htmlEl("div", "hs-row", "", panel);
  const robC = checkbox(row, "robust smoothness", st.robust, v => { st.robust = v; restart(); });
  const pyrC = checkbox(row, "pyramid (3 levels)", st.pyramid, v => { st.pyramid = v; restart(); });
  const row2 = htmlEl("div", "hs-row", "", panel);
  const playB = button(row2, "play", () => { if (timer) stop(); else play(); });
  button(row2, "+1 sweep", () => { stop(); hs?.step(1); draw(); });
  button(row2, "+100", () => { stop(); hs?.step(100); draw(); });
  button(row2, "restart", () => { stop(); restart(false); });
  button(panel, "reset", () => { stop(); Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });

  function sync(): void {
    lamS.set(tOfLam(st.lambda)); sizeS.set(st.size); noiseS.set(st.noise); robC.checked = st.robust; pyrC.checked = st.pyramid;
  }

  // ---- data ----
  let cell: Grey | null = null, ring: Grey | null = null, sc: ReturnType<typeof cellScene> | null = null;
  let pair: FramePair | null = null, hs: HornSchunck | null = null, lk: Flow | null = null, lkEpe = NaN;
  function rebuild(): void {
    stop();
    const m = st.size;
    if (st.scene === "boundary") {
      sc ??= cellScene(N, N, 5);
      pair = compositePair(sc.bg, sc.fg, sc.r, m * Math.cos(Math.PI / 6), -m * Math.sin(Math.PI / 6), st.noise, 11);
    } else {
      let b: Grey | null;
      if (st.scene === "ring") b = ring ??= ringScene(N, N, 3);
      else { if (!cell) { loadCell().then(g => { cell = cropGrey(g, 64, 64, N, N); rebuild(); }); return; } b = cell; }
      const kind = st.motion === "regions" ? "translation" : st.motion;
      const p = kind === "translation" ? { s: m, angle: 30 } : kind === "swirl" ? { s: m, r: 30 } : { s: m / 64 };
      pair = makePair(b, kind, p, st.noise, 11);
    }
    const l = lucasKanade(pair.I1, pair.I2, { sigma: 1, rho: 3 });
    lk = l; lkEpe = endpointError(l, pair.truth, 8).mean;
    curve.set([], []); curveDots.set([], []);
    lkLine.set(lkEpe);
    restart();
  }
  /** New solver for the current λ; `auto`: run to the target number of sweeps (animated). */
  function restart(auto = true): void {
    if (!pair) return;
    stop();
    if (st.pyramid) {
      // coarse levels solved at once; the finest level is animated from their result
      const coarse = hsPyramid(pair.I1, pair.I2, { lambda: st.lambda, sigma: 1, robust: st.robust, levels: 3, iters: 150 });
      hs = new HornSchunck(pair.I1, pair.I2, { lambda: st.lambda, sigma: 1, robust: st.robust, init: { w: coarse.w, h: coarse.h, u: coarse.u, v: coarse.v } });
      hs.iterations = 0;
    } else hs = new HornSchunck(pair.I1, pair.I2, { lambda: st.lambda, sigma: 1, robust: st.robust });
    lamMark.set(st.lambda);
    draw();
    if (auto) play();
  }

  let timer = 0;
  function stop(): void { if (timer) { cancelAnimationFrame(timer); timer = 0; } playB.textContent = "play"; }
  function play(): void {
    if (!hs) return;
    playB.textContent = "pause";
    const goal = hs.iterations < st.target ? st.target : hs.iterations + 400;
    const tick = () => {
      if (!hs || !host.isConnected) { stop(); return; }
      hs.step(Math.min(hs.iterations < 100 ? 4 : 15, goal - hs.iterations));
      draw();
      if (hs.iterations >= goal) { stop(); return; }
      timer = requestAnimationFrame(tick);
    };
    timer = requestAnimationFrame(tick);
  }

  let sweeping = false;
  function sweepLambda(): void {
    if (!pair || sweeping) return;
    sweeping = true; curveBtn.disabled = true;
    const lams = Array.from({ length: 11 }, (_, k) => Number((LMIN * 10 ** (k * 0.6)).toPrecision(2))).filter(l => l <= LMAX);
    const xs: number[] = [], ys: number[] = [];
    let k = 0;
    const next = () => {
      if (!pair || k >= lams.length || !host.isConnected) { sweeping = false; curveBtn.disabled = false; curveBtn.textContent = "compute EPE for 11 values of λ"; return; }
      const h = new HornSchunck(pair.I1, pair.I2, { lambda: lams[k], sigma: 1, robust: st.robust });
      h.step(st.target);
      xs.push(lams[k]); ys.push(endpointError(h, pair.truth, 8).mean);
      curve.set(xs, ys); curveDots.set(xs, ys);
      plot.setDomain(undefined, [0, Math.max(0.5, ...ys.filter(Number.isFinite).map(v => Math.min(v, 3)), lkEpe) * 1.1]);
      curveBtn.textContent = `computing… λ = ${lams[k]}`;
      k++;
      setTimeout(next, 0);
    };
    next();
  }

  function draw(): void {
    if (!pair || !hs) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "ring", "boundary"][i] === st.scene));
    motionBtns.forEach((b, i) => b.classList.toggle("active", MOTIONS[i][0] === st.motion));
    credit.style.display = st.scene === "cell" ? "" : "none";
    const vmax = Math.max(0.5, maxSpeed(pair.truth, 4)), epe = endpointError(hs, pair.truth, 8);
    wheel.el.style.display = st.tab === "flow" || st.tab === "lk" ? "" : "none";
    wheel.setScale(vmax);
    view.clear();
    let resMean = NaN;
    switch (st.tab) {
      case "flow": view.rgba(flowToRGBA(hs, vmax), N, N); view.setCaption(`Horn–Schunck flow after ${hs.iterations} sweeps`); break;
      case "lk": view.rgba(flowToRGBA(lk!, vmax), N, N); view.setCaption("Lucas–Kanade (σ = 1, ρ = 3) for comparison"); break;
      case "arrows":
        view.grey(pair.I1, { mode: "clip", lo: -60, hi: 330 }); view.setCaption("HS flow (black) and truth (green)");
        { const s = 6 / vmax; drawQuiver(view, pair.truth, 12, s, "#18a558", 2.2); drawQuiver(view, hs, 12, s, "#111", 1.6); }
        break;
      case "epe": view.map(epe.map, N, N, inferno, 0, vmax); view.setCaption(`endpoint error (black 0 … yellow ${fmtSpeed(vmax)} px)`); break;
      case "resid": {
        const r = warpResidual(pair.I1, pair.I2, hs);
        resMean = interiorMean(Float32Array.from(r.data, Math.abs), N, N, 8);
        view.grey(r, { mode: "mid", lo: -40, hi: 40 }); view.setCaption("warping residual: frame 2 warped back − frame 1");
        break;
      }
      case "frames": view.grey(pair.I1, { mode: "clip", lo: 0, hi: 255 }); view.setCaption("frame 1"); break;
    }
    const E = hs.energy();
    ro.textContent = `sweeps ${hs.iterations}\nmean EPE: HS ${epe.mean.toFixed(2)}, LK ${lkEpe.toFixed(2)} px` +
      (Number.isFinite(resMean) ? `\nmean |residual| ${resMean.toFixed(1)}` : "") +
      `\nenergy ${E.total.toExponential(2)}`;
  }

  rebuild();
}

autoMount(NAME, mount);
