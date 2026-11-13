// Smoothing as heat flow. Tabs:
//  1D: a 128-sample profile under explicit (or implicit) heat steps, with the Gaussian blur at
//      σ = √(2t) dotted on top, the total ("mass"), and the amplification factor A(ω) of one
//      step for the current r = Δt/Δx² (band |A| ≤ 1; the checkerboard frequency ω = π marked).
//  2D: an image (micrograph or synthetic test card) after explicit steps, next to the Gaussian
//      at σ = √(2t) and their difference (mid-grey = 0); steps, time, σ, ms per step.
//  by hand: a 7 × 7 grid of numbers, one explicit 5-point step per click; click a cell to see
//      its update written out.
// Δt is set as a multiple of the stability limit (1/2 in 1D, 1/4 in 2D).
//
// <div data-demo="heatflow" data-tab="1d|2d|hand" data-preset="gauss|sigma10|limit|unstable|implicit|zero|mirror|hand"></div>
import { autoMount, htmlEl, button, presets as presetRow, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { CELL_CREDIT } from "../lib/cell";
import { heatStep, implicitStep1d, implicitStep2d, amplification, gaussianHeat, sigmaOfTime, testCard, addNoise, HeatBorder } from "../lib/pde";
import { ImgView, injectStyle } from "../canny/imgview";
import { tabRow, logSlider, choice, cell128, toGrey, PS_CSS, animator } from "./common";
import { rng } from "../lib/random";

const NAME = "heatflow";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .hf-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .hf-row { display: flex; gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .hf-row > * { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .hf-amp { flex: 0 0 300px; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .hf-hand { display: flex; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .hf-hand svg { flex: 0 0 400px; max-width: 400px; }
.qb-demo[data-demo="${NAME}"] .hf-explain { font-size: 18px; white-space: pre-wrap; line-height: 1.35; font-family: inherit; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; gap: .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .hf-main { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .2em .45em; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .hf-row { flex-wrap: wrap; } }
`;

type Tab = "1d" | "2d" | "hand";
type Scheme = "explicit" | "implicit";
interface State { tab: Tab; mult: number; scheme: Scheme; border: HeatBorder; profile: "mixed" | "step" | "cell"; image: "card" | "cell"; run: number }
const DEFAULT: State = { tab: "1d", mult: 0.9, scheme: "explicit", border: "mirror", profile: "mixed", image: "cell", run: 0 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "gauss", label: "heat = Gaussian", s: { tab: "1d", mult: 0.9, run: 20 } },
  { id: "sigma10", label: "σ = 10 takes 200 steps", s: { tab: "2d", mult: 1, run: 200, image: "cell" } },
  { id: "limit", label: "at the limit", s: { tab: "1d", mult: 1, run: 40 } },
  { id: "unstable", label: "1.2 × the limit: the checkerboard grows", s: { tab: "2d", mult: 1.2, run: 8, image: "card" } },
  { id: "implicit", label: "implicit: any Δt is stable", s: { tab: "1d", scheme: "implicit", mult: 8, run: 5 } },
  { id: "zero", label: "zero border: heat leaks out", s: { tab: "1d", border: "zero", profile: "step", mult: 0.9, run: 150 } },
  { id: "mirror", label: "mirror border: the mean is kept", s: { tab: "1d", border: "mirror", profile: "step", mult: 0.9, run: 150 } },
  { id: "hand", label: "by hand: one step on a 7 × 7 grid", s: { tab: "hand", mult: 0.8 } },
];
const N1 = 128;

function profileOf(kind: State["profile"], cellRow: Float64Array | null): Float64Array {
  const u = new Float64Array(N1), r = rng(7);
  for (let i = 0; i < N1; i++) {
    if (kind === "cell" && cellRow) { u[i] = cellRow[i]; continue; }
    if (kind === "step") { u[i] = i >= 96 ? 200 : 40; continue; }
    let v = 40;
    if (i >= 12 && i < 40) v = 200;            // plateau with steps
    if (i === 55 || i === 56) v = 220;         // narrow peak
    if (i >= 70 && i < 102) v = 40 + 160 * (i - 70) / 32;   // ramp
    if (i >= 102) v = 200;
    u[i] = v + 8 * r.norm();
  }
  return u;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab ? { tab: host.dataset.tab as Tab } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "hf-main", "", host);
  const setTab = tabRow(main, [["1d", "1D profile"], ["2d", "2D image"], ["hand", "by hand"]], id => { st.tab = id as Tab; anim.stop(); resetRun(); });
  // ---- 1D view ----
  const v1 = htmlEl("div", "", "", main);
  const prof = new Plot(v1, { width: 760, height: 300, x: [0, N1 - 1], y: [-20, 260], xlabel: "position x (pixels)", ylabel: "value", margin: { t: 8 } });
  const origL = prof.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, label: "start" });
  const curL = prof.line([], [], { color: "var(--qb-c1)", width: 3.5, label: "after the steps" });
  const gaussL = prof.line([], [], { color: "#e8402a", width: 3, dash: "2 6", label: "Gaussian σ = √(2t)" });
  prof.legend(undefined, "tl");
  const row1 = htmlEl("div", "hf-row", "", v1);
  const ampDiv = htmlEl("div", "hf-amp", "", row1);
  const amp = new Plot(ampDiv, { width: 300, height: 190, x: [0, Math.PI], y: [-1.6, 1.1], xlabel: "frequency ω", ylabel: "A(ω)", xticks: [0, Math.PI / 2, Math.PI], xfmt: v => (v === 0 ? "0" : v < 2 ? "π/2" : "π"), font: 17 });
  amp.band([0, Math.PI], [-1, -1], [1, 1], { color: "#2ca02c", fill: "#2ca02c", fillOpacity: 0.12 });
  const ampL = amp.line([], [], { color: "var(--qb-c1)", width: 3 });
  const ampPi = amp.scatter([Math.PI], [0], { color: "#e8402a", r: 6 });
  const ro1 = htmlEl("div", "qb-readout", "", row1);
  // ---- 2D view ----
  const v2 = htmlEl("div", "", "", main);
  const row2 = htmlEl("div", "hf-row", "", v2);
  const viewE = new ImgView(row2, "explicit steps"), viewG = new ImgView(row2, "Gaussian"), viewD = new ImgView(row2, "difference (grey = 0)");
  const ro2 = htmlEl("div", "qb-readout", "", v2);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, v2);
  // ---- by hand ----
  const v3 = htmlEl("div", "hf-hand", "", main);
  const G = 7, CS = 56;
  const gsvg = svgEl("svg", { viewBox: `0 0 ${G * CS + 4} ${G * CS + 4}` }, v3);
  const explain = htmlEl("div", "hf-explain qb-readout", "", v3);

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const multS = logSlider(panel, "Δt / limit", 0.1, 10, st.mult, v => { st.mult = Math.round(v * 100) / 100; draw(); }, v => (Math.round(v * 100) / 100).toFixed(2));
  const setScheme = choice(panel, "scheme", [["explicit", "explicit"], ["implicit", "implicit"]], id => { st.scheme = id as Scheme; draw(); });
  const setBorder = choice(panel, "border", [["mirror", "mirror"], ["zero", "zero"]], id => { st.border = id as HeatBorder; resetRun(); });
  const setProf = choice(panel, "profile", [["mixed", "mixed"], ["step", "step"], ["cell", "micrograph row"]], id => { st.profile = id as State["profile"]; resetRun(); });
  const setImg = choice(panel, "image", [["cell", "micrograph"], ["card", "test card"]], id => { st.image = id as State["image"]; resetRun(); });
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "step", () => { anim.stop(); doSteps(1); draw(); });
  button(ctl, "10 steps", () => { anim.stop(); doSteps(10); draw(); });
  const playB = button(ctl, "play", () => { if (anim.running()) { anim.stop(); playB.textContent = "play"; } else { anim.start(); playB.textContent = "pause"; } });
  button(ctl, "restart", () => { anim.stop(); resetRun(); });
  button(panel, "reset all", () => { anim.stop(); Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); resetRun(true); });

  const anim = animator(() => { doSteps(st.tab === "2d" ? 2 : 1); draw(); if (steps >= 2000 || maxAbs() > 1e6) { playB.textContent = "play"; return false; } return true; });

  // ---- state of a run ----
  let cellImg: Float64Array | null = null, cellRow: Float64Array | null = null;
  let u1: Float64Array = new Float64Array(N1), u1s: Float64Array = new Float64Array(N1), u2: Float64Array = new Float64Array(0), u2s: Float64Array = new Float64Array(0);
  const W2 = 128;
  let hand: Float64Array = new Float64Array(G * G), handPrev: Float64Array = new Float64Array(G * G), handSel = 24;
  let steps = 0, t = 0, ms = 0;
  const limit = () => (st.tab === "1d" ? 0.5 : 0.25);
  const dt = () => st.mult * limit();
  const maxAbs = () => { const a = st.tab === "1d" ? u1 : u2; let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };

  function handStart(): void {
    hand = new Float64Array(G * G);
    hand[3 * G + 3] = 100; hand[2 * G + 3] = 20; hand[3 * G + 2] = 20;
    handPrev = Float64Array.from(hand); handSel = 3 * G + 3;
  }
  function resetRun(initial = false): void {
    steps = 0; t = 0;
    u1 = profileOf(st.profile, cellRow); u1s = Float64Array.from(u1);
    if (st.image === "cell" && cellImg) { u2 = Float64Array.from(cellImg); }
    else { const c = testCard(); u2 = addNoise(c.truth, 12, 5); }
    u2s = Float64Array.from(u2);
    handStart();
    if (initial && st.run) { doSteps(st.run); }
    draw();
  }
  function doSteps(n: number): void {
    const d = dt();
    const t0 = performance.now();
    for (let k = 0; k < n; k++) {
      if (st.tab === "1d") u1 = st.scheme === "explicit" ? heatStep(u1, N1, 1, d, st.border) : implicitStep1d(u1, d, st.border);
      else if (st.tab === "2d") u2 = st.scheme === "explicit" ? heatStep(u2, W2, W2, d, st.border) : implicitStep2d(u2, W2, W2, d);
      else { handPrev = hand; hand = heatStep(hand, G, G, d, "zero"); }
      steps++; t += d;
    }
    ms = (performance.now() - t0) / Math.max(1, n);
  }
  function load(p: Preset): void {
    anim.stop(); playB.textContent = "play";
    Object.assign(st, DEFAULT, p.s);
    sync();
    if (st.tab === "2d" && st.image === "cell" && !cellImg) { pending = true; return; }
    resetRun(true);
  }
  let pending = false;
  function sync(): void { multS.setVal(st.mult); setScheme(st.scheme); setBorder(st.border); setProf(st.profile); setImg(st.image); setTab(st.tab); }

  // by-hand grid clicks
  gsvg.addEventListener("pointerdown", ev => {
    const r = gsvg.getBoundingClientRect(), x = Math.floor((ev.clientX - r.left) / r.width * G), y = Math.floor((ev.clientY - r.top) / r.height * G);
    if (x >= 0 && y >= 0 && x < G && y < G) { handSel = y * G + x; draw(); }
  });

  function draw(): void {
    setTab(st.tab);
    v1.style.display = st.tab === "1d" ? "" : "none";
    v2.style.display = st.tab === "2d" ? "" : "none";
    v3.style.display = st.tab === "hand" ? "" : "none";
    setProf.el.style.display = st.tab === "1d" ? "" : "none";
    setImg.el.style.display = st.tab === "2d" ? "" : "none";
    setScheme.el.style.display = setBorder.el.style.display = st.tab === "hand" ? "none" : "";
    const sig = sigmaOfTime(t), d = dt(), r = d;
    if (st.tab === "1d") {
      const xs = Array.from({ length: N1 }, (_, i) => i);
      origL.set(xs, u1s);
      gaussL.set(xs, gaussianHeat(u1s, N1, 1, sig, st.border));
      const lo = Math.min(-20, ...u1), hi = Math.max(260, ...u1);
      const clip = (v: number) => Math.max(-1e4, Math.min(1e4, v));
      prof.setDomain(undefined, [Math.max(lo, -400), Math.min(hi, 700)]);
      curL.set(xs, Array.from(u1, clip));
      const om: number[] = [], A: number[] = [];
      for (let k = 0; k <= 100; k++) { const w = Math.PI * k / 100; om.push(w); A.push(amplification(r, w, st.scheme)); }
      amp.setDomain(undefined, [Math.min(-1.6, ...A) - 0.05, 1.15]);
      ampL.set(om, A); ampPi.set([Math.PI], [amplification(r, Math.PI, st.scheme)]);
      let mass = 0, mass0 = 0; for (let i = 0; i < N1; i++) { mass += u1[i]; mass0 += u1s[i]; }
      const Api = amplification(r, Math.PI, st.scheme);
      ro1.textContent = `${steps} steps of Δt = ${d.toFixed(3)} (r = ${r.toFixed(3)})\ntime t = ${t.toFixed(2)}, σ = √(2t) = ${sig.toFixed(2)}\n` +
        `total: ${mass.toFixed(0)} (start ${mass0.toFixed(0)})\nA(π) = ${Api.toFixed(2)}: ` +
        (Math.abs(Api) <= 1 + 1e-9 ? "stable" : "the checkerboard grows ×" + Math.abs(Api).toFixed(2) + " per step");
    } else if (st.tab === "2d") {
      const g = gaussianHeat(u2s, W2, W2, sig, st.border);
      viewE.grey(toGrey(u2, W2, W2), { mode: "clip", lo: 0, hi: 255 });
      viewG.grey(toGrey(g, W2, W2), { mode: "clip", lo: 0, hi: 255 });
      const diff = new Float64Array(W2 * W2); let md = 0;
      for (let i = 0; i < diff.length; i++) { diff[i] = u2[i] - g[i]; md = Math.max(md, Math.abs(diff[i])); }
      viewD.grey(toGrey(diff, W2, W2), { mode: "clip", lo: -20, hi: 20 });
      viewE.setCaption(`${st.scheme} steps: ${steps}`); viewG.setCaption(`Gaussian σ = ${sig.toFixed(2)}`); viewD.setCaption("difference (grey = 0, ±20)");
      credit.style.display = st.image === "cell" ? "" : "none";
      ro2.textContent = `${steps} steps × Δt = ${d.toFixed(3)} → t = ${t.toFixed(1)}, σ = √(2t) = ${sig.toFixed(2)}   ·   largest difference ${md < 1e5 ? md.toFixed(2) : md.toExponential(1)}   ·   ${ms.toFixed(2)} ms per step`;
    } else drawHand(d);
  }

  function drawHand(d: number): void {
    gsvg.replaceChildren();
    const hi = 100;
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
      const p = y * G + x, v = hand[p];
      const gv = Math.round(255 - 200 * Math.max(0, Math.min(1, v / hi)));
      const neg = v < -1e-9;
      const rc = svgEl("rect", { x: 2 + x * CS, y: 2 + y * CS, width: CS - 2, height: CS - 2, rx: 4 }, gsvg);
      rc.style.fill = neg ? "#f4b6a8" : `rgb(${gv},${gv},${Math.min(255, gv + 30)})`;
      rc.style.stroke = "#c8c8c8"; rc.style.strokeWidth = "1";
      const sel = handSel, sx = sel % G, sy = (sel - sx) / G;
      const isSel = p === sel, isNb = Math.abs(x - sx) + Math.abs(y - sy) === 1;
      if (isSel || isNb) { rc.style.stroke = isSel ? "#e8402a" : "#ff9f1c"; rc.style.strokeWidth = "4"; }
      const tx = svgEl("text", { x: 2 + x * CS + CS / 2 - 1, y: 2 + y * CS + CS / 2 + 7, "text-anchor": "middle", "font-size": 19 }, gsvg);
      tx.textContent = Math.abs(v) < 0.05 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
      tx.style.fill = gv < 120 ? "#fff" : "#111";
    }
    const p = handSel, x = p % G, y = (p - x) / G, u = handPrev;
    const nb = [x > 0 ? u[p - 1] : 0, x < G - 1 ? u[p + 1] : 0, y > 0 ? u[p - G] : 0, y < G - 1 ? u[p + G] : 0];
    const s = nb.reduce((a, b) => a + b, 0);
    let mass = 0; for (const v of hand) mass += v;
    explain.textContent = steps === 0
      ? `Click "step". Each pixel moves towards\nits four neighbours:\n\nu ← u + Δt · (sum of 4 neighbours − 4u)\n\nΔt = ${d.toFixed(3)} (limit 0.25)\nOutside the grid: 0.\nClick a cell to see its update.`
      : `step ${steps}, Δt = ${d.toFixed(3)}\ncell (${x}, ${y}): old ${u[p].toFixed(2)}\nneighbours ${nb.map(v => v.toFixed(2)).join(" + ")} = ${s.toFixed(2)}\n` +
        `new = ${u[p].toFixed(2)} + ${d.toFixed(3)} · (${s.toFixed(2)} − 4 · ${u[p].toFixed(2)})\n    = ${hand[p].toFixed(2)}\n\n` +
        `centre weight 1 − 4Δt = ${(1 - 4 * d).toFixed(2)}` + (1 - 4 * d < 0 ? "  < 0: overshoot!" : "") + `\ntotal ${mass.toFixed(1)}`;
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  cell128().then(g => {
    cellImg = Float64Array.from(g.data);
    cellRow = Float64Array.from(g.data.subarray(64 * 128, 65 * 128));
    if (pending || st.image === "cell" || st.profile === "cell") { pending = false; resetRun(true); }
  });
  resetRun(true);
}

autoMount(NAME, mount);
