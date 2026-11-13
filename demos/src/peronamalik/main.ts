// Edge-stopping diffusion (Perona–Malik). Tabs:
//  1D: a profile (step, soft step, ramp, a narrow spike, seeded noise) under Perona–Malik steps,
//      with the heat equation for the same time and the truth for comparison. Below: the flux
//      φ(s) = s·g(s) against the size s of the differences, the histogram of the profile's current
//      differences underneath, the backward range (φ falling) shaded; drag on it to set K.
//  2D: the test card (or the micrograph) with seeded noise: the current image, the conductivity
//      map g(|∇u|) (white = heat flows, black = insulates), PSNR against time (best stopping time
//      marked) and the profile along row 112 (step, ramp, 1-px line, soft step).
// Controls: g1 = e^{−(s/K)²}, g2 = 1/(1 + (s/K)²) or linear; K; pre-smoothing σ (Catté; 0 =
// classical); Δt as a multiple of the limit; noise; play / step / restart.
//
// <div data-demo="peronamalik" data-tab="1d|2d" data-preset="linear|edges|ksmall|klarge|sharpen|stairs|spike|presmooth|forever|cell"></div>
import { autoMount, htmlEl, button, slider, presets as presetRow } from "../lib/ui";
import { Plot } from "../lib/plot";
import { CELL_CREDIT } from "../lib/cell";
import { nonlinearStep, heatStep, conductivityMap, flux, fluxPeak, testCard, addNoise, psnr255, Diffusivity } from "../lib/pde";
import { ImgView, injectStyle } from "../canny/imgview";
import { tabRow, logSlider, choice, cell128, toGrey, PS_CSS, animator } from "../heatflow/common";
import { rng } from "../lib/random";

const NAME = "peronamalik";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pm-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .pm-row { display: flex; gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .pm-row > * { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .pm-flux { cursor: ew-resize; touch-action: none; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .pm-main { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .2em .45em; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .pm-row { flex-wrap: wrap; } }
`;

type Tab = "1d" | "2d";
interface State { tab: Tab; kind: Diffusivity; K: number; pre: number; mult: number; noise: number; image: "card" | "cell"; run: number }
const DEFAULT: State = { tab: "2d", kind: "g2", K: 10, pre: 0, mult: 0.9, noise: 12, image: "card", run: 0 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "linear", label: "linear: plain blur", s: { tab: "2d", kind: "linear", run: 40 } },
  { id: "edges", label: "edges stop the heat", s: { tab: "2d", kind: "g2", K: 10, run: 30 } },
  { id: "ksmall", label: "K too small: noise kept", s: { tab: "2d", kind: "g2", K: 2, run: 60 } },
  { id: "klarge", label: "K too large: linear again", s: { tab: "2d", kind: "g2", K: 150, run: 60 } },
  { id: "sharpen", label: "soft edge sharpens (backward)", s: { tab: "1d", kind: "g2", K: 8, noise: 0, run: 150 } },
  { id: "stairs", label: "ramp becomes a staircase", s: { tab: "1d", kind: "g2", K: 3, noise: 2, run: 600 } },
  { id: "spike", label: "a spike survives forever", s: { tab: "1d", kind: "g1", K: 10, noise: 0, run: 150 } },
  { id: "presmooth", label: "pre-smoothing fixes the spike", s: { tab: "1d", kind: "g1", K: 10, noise: 0, pre: 1.5, run: 150 } },
  { id: "forever", label: "run forever: everything flattens", s: { tab: "2d", kind: "g2", K: 10, run: 3000 } },
  { id: "cell", label: "nuclei in the micrograph", s: { tab: "2d", image: "cell", noise: 0, kind: "g2", K: 6, pre: 1, run: 80 } },
];
const N1 = 128;

function truth1d(): Float64Array {
  const u = new Float64Array(N1);
  for (let i = 0; i < N1; i++) {
    let v = 60;
    if (i >= 16 && i < 40) v = 180;                                   // step up and down
    if (i >= 40 && i < 64) v = 60 + 120 / (1 + Math.exp(-(i - 52) / 1.5));   // soft step (max slope 20)
    if (i >= 64 && i < 70) v = 180;
    if (i >= 70 && i < 100) v = 180 - 120 * (i - 70) / 30;            // ramp, slope 4
    if (i === 112) v = 100;                                           // narrow spike of 40
    u[i] = v;
  }
  return u;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab ? { tab: host.dataset.tab as Tab } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "pm-main", "", host);
  const setTab = tabRow(main, [["1d", "1D profile"], ["2d", "2D image"]], id => { st.tab = id as Tab; anim.stop(); playB.textContent = "play"; restart(); });
  // ---- 1D ----
  const v1 = htmlEl("div", "", "", main);
  const prof = new Plot(v1, { width: 770, height: 270, x: [0, N1 - 1], y: [30, 210], xlabel: "position x", ylabel: "value", margin: { t: 6 } });
  const truthL = prof.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, dash: "4 4", label: "truth" });
  const heatL = prof.line([], [], { color: "#ff9f1c", width: 2.5, label: "heat, same time" });
  const pmL = prof.line([], [], { color: "var(--qb-c1)", width: 3.5, label: "Perona–Malik" });
  prof.legend(undefined, "tr");
  const fluxDiv = htmlEl("div", "pm-flux", "", v1);
  const fl = new Plot(fluxDiv, { width: 770, height: 220, x: [0, 40], y: [0, 1], xlabel: "size of a difference |d| (drag to set K)", ylabel: "flux φ", yticks: [], margin: { t: 6, l: 60 } });
  const backBand = fl.band([], [], [], { color: "#e8402a", fill: "#e8402a", fillOpacity: 0.12 });
  const histM = fl.bars([], [], { color: "var(--qb-c3, #2ca02c)", opacity: 0.45 });
  const fluxL = fl.line([], [], { color: "var(--qb-c1)", width: 3.5 });
  const kLine = fl.vline(10, { color: "#e8402a", width: 3, text: "K" });
  const ro1 = htmlEl("div", "qb-readout", "", v1);
  // ---- 2D ----
  const v2 = htmlEl("div", "", "", main);
  const row2 = htmlEl("div", "pm-row", "", v2);
  const viewU = new ImgView(row2, "current image"), viewG = new ImgView(row2, "conductivity g");
  const psDiv = htmlEl("div", "", "", row2);
  const ps = new Plot(psDiv, { width: 260, height: 260, x: [0, 100], y: [20, 40], xlabel: "time t", ylabel: "PSNR (dB)", font: 17, margin: { l: 66 } });
  const psL = ps.line([], [], { color: "var(--qb-c1)", width: 3 });
  const psBest = ps.scatter([], [], { color: "#e8402a", r: 6 });
  const pr2 = new Plot(v2, { width: 770, height: 170, x: [0, 127], y: [30, 210], xlabel: "row 112: step, ramp, 1-px line, soft step", ylabel: "value", font: 16, margin: { t: 4, l: 64 } });
  const pr2T = pr2.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, dash: "4 4" });
  const pr2U = pr2.line([], [], { color: "var(--qb-c1)", width: 3 });
  const ro2 = htmlEl("div", "qb-readout", "", v2);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, v2);

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const setKind = choice(panel, "g", [["linear", "linear"], ["g1", "g₁ = e^−(s/K)²"], ["g2", "g₂ = 1/(1+(s/K)²)"]], id => { st.kind = id as Diffusivity; draw(); });
  const kS = logSlider(panel, "K", 1, 200, st.K, v => { st.K = Math.round(v * 10) / 10; draw(); }, v => (Math.round(v * 10) / 10).toFixed(1));
  const preS = slider(panel, "pre-smooth σ", 0, 3, 0.25, st.pre, v => { st.pre = v; draw(); });
  const multS = slider(panel, "Δt / limit", 0.1, 1.5, 0.05, st.mult, v => { st.mult = v; draw(); });
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; restart(); });
  const setImg = choice(panel, "image", [["card", "test card"], ["cell", "micrograph"]], id => { st.image = id as State["image"]; restart(); });
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "step", () => { anim.stop(); playB.textContent = "play"; doSteps(1); draw(); });
  button(ctl, "+50", () => { anim.stop(); playB.textContent = "play"; doSteps(50); draw(); });
  const playB = button(ctl, "play", () => { if (anim.running()) { anim.stop(); playB.textContent = "play"; } else { anim.start(); playB.textContent = "pause"; } });
  button(ctl, "restart", () => { anim.stop(); playB.textContent = "play"; restart(); });
  button(panel, "reset all", () => { anim.stop(); playB.textContent = "play"; Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); restart(true); });

  const anim = animator(() => { doSteps(st.tab === "1d" ? 5 : 3); draw(); if (steps >= 20000) { playB.textContent = "play"; return false; } return true; });

  // ---- state ----
  let cellImg: Float64Array | null = null;
  const card = testCard();
  let t1: Float64Array = truth1d(), u1: Float64Array = new Float64Array(N1), h1: Float64Array = new Float64Array(N1);
  let t2: Float64Array = card.truth, u2: Float64Array = new Float64Array(0);
  let steps = 0, t = 0, psT: number[] = [], psV: number[] = [];
  const dt = () => st.mult * (st.tab === "1d" ? 0.5 : 0.25);

  function restart(runPreset = false): void {
    steps = 0; t = 0; psT = []; psV = [];
    const r = rng(11);
    t1 = truth1d();
    u1 = Float64Array.from(t1, v => v + st.noise * 0.35 * r.norm());   // 1D noise: a third of the 2D level
    h1 = Float64Array.from(u1);
    t2 = st.image === "cell" && cellImg ? cellImg : card.truth;
    u2 = st.noise > 0 ? addNoise(t2, st.noise, 4) : Float64Array.from(t2);
    psT.push(0); psV.push(psnr255(u2, t2));
    if (runPreset && st.run) doSteps(st.run);
    draw();
  }
  function doSteps(n: number): void {
    const d = dt();
    for (let k = 0; k < n; k++) {
      if (st.tab === "1d") { u1 = nonlinearStep(u1, N1, 1, d, st.kind, st.K, st.pre); h1 = heatStep(h1, N1, 1, d); }
      else {
        u2 = nonlinearStep(u2, 128, 128, d, st.kind, st.K, st.pre);
        if (steps % 4 === 3 || n === 1) { psT.push(t + d); psV.push(psnr255(u2, t2)); }
      }
      steps++; t += d;
    }
  }
  function load(p: Preset): void {
    anim.stop(); playB.textContent = "play";
    Object.assign(st, DEFAULT, p.s);
    sync();
    if (st.image === "cell" && !cellImg) { pending = true; return; }
    restart(true);
  }
  let pending = false;
  function sync(): void { setTab(st.tab); setKind(st.kind); kS.setVal(st.K); preS.set(st.pre); multS.set(st.mult); noiseS.set(st.noise); setImg(st.image); }

  // drag K on the flux plot
  let dragging = false;
  const setKFrom = (ev: PointerEvent) => { const [x] = fl.toData(ev); st.K = Math.max(1, Math.min(200, Math.round(x * 10) / 10)); kS.setVal(st.K); draw(); };
  fl.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; fl.svg.setPointerCapture(ev.pointerId); setKFrom(ev); });
  fl.svg.addEventListener("pointermove", ev => { if (dragging) setKFrom(ev); });
  fl.svg.addEventListener("pointerup", () => { dragging = false; });

  function draw(): void {
    setTab(st.tab);
    v1.style.display = st.tab === "1d" ? "" : "none";
    v2.style.display = st.tab === "2d" ? "" : "none";
    setImg.el.style.display = st.tab === "2d" ? "" : "none";
    const xs = Array.from({ length: N1 }, (_, i) => i);
    if (st.tab === "1d") {
      truthL.set(xs, t1); heatL.set(xs, h1); pmL.set(xs, u1);
      // flux plot: s up to max(3K, 40)
      const smax = Math.max(40, Math.min(200, 3 * st.K));
      const ss: number[] = [], ph: number[] = [];
      for (let k = 0; k <= 200; k++) { const s = smax * k / 200; ss.push(s); ph.push(flux(st.kind, s, st.K)); }
      const pmax = Math.max(...ph) || 1;
      // histogram of current |differences| (of the pre-smoothed profile if σ > 0)
      const nb = 40, bw = smax / nb, cnt = new Array(nb).fill(0);
      for (let i = 0; i + 1 < N1; i++) { const d = Math.abs(u1[i + 1] - u1[i]); cnt[Math.min(nb - 1, Math.floor(d / bw))]++; }
      const cmax = Math.max(...cnt.slice(0)) || 1;
      fl.setDomain([0, smax], [0, pmax * 1.12]);
      histM.set(cnt.map((_, i) => (i + 0.5) * bw), cnt.map(c => Math.min(1, Math.sqrt(c / cmax)) * pmax * 0.6));
      fluxL.set(ss, ph);
      const pk = fluxPeak(st.kind, st.K);
      backBand.set(pk < smax ? [pk, smax] : [], [0, 0], [pmax * 1.12, pmax * 1.12]);
      kLine.set(st.K);
      let back = 0; for (let i = 0; i + 1 < N1; i++) if (Math.abs(u1[i + 1] - u1[i]) > pk) back++;
      ro1.textContent = `${steps} steps, Δt = ${dt().toFixed(3)}, time t = ${t.toFixed(1)}   ·   flux peaks at |d| = ${Number.isFinite(pk) ? pk.toFixed(1) : "∞ (linear)"}; ` +
        `${back} differences are in the backward range (shaded): there the edge gets steeper.` + (st.pre > 0 ? `  g is computed from the profile blurred with σ = ${st.pre}.` : "");
    } else {
      viewU.grey(toGrey(u2, 128, 128), { mode: "clip", lo: 0, hi: 255 });
      const g = conductivityMap(u2, 128, 128, st.kind, st.K, st.pre);
      viewG.grey(toGrey(g, 128, 128), { mode: "clip", lo: 0, hi: 1 });
      viewU.setCaption(`after ${steps} steps (t = ${t.toFixed(1)})`); viewG.setCaption("conductivity g (white = flows)");
      const tmax = Math.max(10, t * 1.05), vmin = Math.min(...psV), vmax = Math.max(...psV);
      ps.setDomain([0, tmax], [Math.floor(vmin - 1), Math.ceil(vmax + 1)]);
      psL.set(psT, psV);
      const bi = psV.indexOf(vmax);
      psBest.set([psT[bi]], [vmax]);
      const row = 112 * 128;
      pr2T.set(xs, t2.subarray(row, row + 128)); pr2U.set(xs, u2.subarray(row, row + 128));
      credit.style.display = st.image === "cell" ? "" : "none";
      ro2.textContent = `PSNR now ${psV[psV.length - 1].toFixed(2)} dB, best ${vmax.toFixed(2)} dB at t = ${psT[bi].toFixed(1)} (red dot)` + (st.image === "cell" ? "  ·  truth = the micrograph without added noise" : "");
    }
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  cell128().then(g => { cellImg = Float64Array.from(g.data); if (pending || st.image === "cell") { pending = false; restart(true); } });
  restart(true);
}

autoMount(NAME, mount);
