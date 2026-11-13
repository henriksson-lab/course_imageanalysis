// Total variation flow. Tabs:
//  1D: plateaus of width 2, 4 and 8 and a ramp (optional seeded noise). TV flow u_t =
//      (u_x/|u_x|_ε)_x by explicit steps (Δt = ε/2.5), the ROF result with λ = t (exact, Condat)
//      and the heat equation at the same time. Below: each plateau's height against time with
//      the predicted straight line (slope = 2/width).
//  discs: discs of radius 3 … 12, a square and a star on a flat background under TV flow,
//      computed as repeated ROF steps (one implicit TV-flow step per frame); the contrast of
//      each disc against time with the lines c − 2t/r.
//  image: the test card or the micrograph with noise: TV (ROF), Perona–Malik and the heat
//      equation, each tuned (λ, time, σ) to remove exactly as much as the noise, SD(noisy −
//      result) = σn (the discrepancy principle); row profiles.
//
// <div data-demo="tvflow" data-tab="1d|discs|image" data-preset="plateaus|discs|corners|ramp|lambda|eps|compare"></div>
import { autoMount, htmlEl, button, slider, presets as presetRow } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { CELL_CREDIT } from "../lib/cell";
import { tvFlowStep, heatStep, gaussianHeat, nonlinearStep, testCard, addNoise, psnr255 } from "../lib/pde";
import { tv1d, tv2d } from "../lib/smooth";
import { ImgView, injectStyle } from "../canny/imgview";
import { tabRow, logSlider, cell128, toGrey, PS_CSS, animator, choice } from "../heatflow/common";
import { rng } from "../lib/random";

const NAME = "tvflow";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .tv-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .tv-row { display: flex; gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .tv-row > * { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .tv-main { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .2em .45em; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .tv-row { flex-wrap: wrap; } }
`;

type Tab = "1d" | "discs" | "image";
interface State { tab: Tab; eps: number; noise1: number; until: number; image: "card" | "cell"; noise2: number }
const DEFAULT: State = { tab: "1d", eps: 0.5, noise1: 0, until: 0, image: "card", noise2: 15 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "plateaus", label: "plateaus sink at rate 2/width", s: { tab: "1d", until: 40 } },
  { id: "discs", label: "small discs vanish first", s: { tab: "discs", until: 90 } },
  { id: "corners", label: "corners round, edges stay", s: { tab: "discs", until: 50 } },
  { id: "ramp", label: "ramp staircases", s: { tab: "1d", noise1: 4, until: 15 } },
  { id: "lambda", label: "flow time ≈ λ", s: { tab: "1d", until: 30 } },
  { id: "eps", label: "small ε: thousands of steps", s: { tab: "1d", eps: 0.05, until: 20 } },
  { id: "compare", label: "TV vs PM vs heat", s: { tab: "image" } },
  { id: "cell", label: "… on the micrograph", s: { tab: "image", image: "cell", noise2: 15 } },
];
const N1 = 128;
const PLAT: [number, number][] = [[10, 2], [24, 4], [40, 8]];   // start, width (height +60)

function profile1d(noise: number): { truth: Float64Array; y: Float64Array } {
  const t = new Float64Array(N1).fill(60);
  for (const [s, w] of PLAT) for (let i = s; i < s + w; i++) t[i] = 120;
  for (let i = 64; i < 96; i++) t[i] = 60 + 120 * (i - 64) / 32;
  for (let i = 96; i < 112; i++) t[i] = 180;
  const r = rng(21), y = Float64Array.from(t, v => v + noise * r.norm());
  return { truth: t, y };
}

const DISCS: [number, number, number][] = [[18, 18, 3], [44, 18, 4.5], [76, 20, 6], [108, 24, 9], [24, 60, 12]];
function discsImage(): Float64Array {
  const n = 128, u = new Float64Array(n * n).fill(60);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const p = y * n + x;
    for (const [cx, cy, r] of DISCS) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) u[p] = 100;
    if (x >= 56 && x < 88 && y >= 48 && y < 80) u[p] = 100;              // square, side 32
    const dx = x - 106, dy = y - 100, a = Math.atan2(dy, dx), rr = Math.hypot(dx, dy);
    if (rr <= 9 + 7 * Math.cos(5 * a)) u[p] = 100;                        // a five-pointed star
    if (x >= 10 && x < 70 && y >= 96 && y < 104) u[p] = 100;             // a long bar, width 8
  }
  return u;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab ? { tab: host.dataset.tab as Tab } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "tv-main", "", host);
  const setTab = tabRow(main, [["1d", "1D plateaus"], ["discs", "discs and shapes"], ["image", "image: TV, PM, heat"]], id => { anim.stop(); playB.textContent = "play"; st.tab = id as Tab; st.until = 0; restart(); });
  // ---- 1D ----
  const v1 = htmlEl("div", "", "", main);
  const prof = new Plot(v1, { width: 770, height: 270, x: [0, N1 - 1], y: [40, 200], xlabel: "position x", ylabel: "value", margin: { t: 6 } });
  const p1T = prof.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, dash: "4 4", label: "start" });
  const p1H = prof.line([], [], { color: "#ff9f1c", width: 2.5, label: "heat, same t" });
  const p1R = prof.line([], [], { color: "#e8402a", width: 3, dash: "7 5", label: "ROF, λ = t" });
  const p1F = prof.line([], [], { color: "var(--qb-c1)", width: 3.5, label: "TV flow" });
  prof.legend(undefined, "tr");
  const row1 = htmlEl("div", "tv-row", "", v1);
  const hp = new Plot(row1, { width: 420, height: 220, x: [0, 40], y: [60, 125], xlabel: "time t", ylabel: "plateau height", font: 17 });
  const hpL = PLAT.map((_, k) => hp.line([], [], { color: seriesColor(k), width: 3, label: `width ${PLAT[k][1]}` }));
  const hpP = PLAT.map((_, k) => hp.line([], [], { color: seriesColor(k), width: 1.5, dash: "3 4" }));
  hp.legend(undefined, "tr");
  const ro1 = htmlEl("div", "qb-readout", "", row1);
  // ---- discs ----
  const v2 = htmlEl("div", "", "", main);
  const row2 = htmlEl("div", "tv-row", "", v2);
  const viewD = new ImgView(row2, "TV flow");
  const cp = new Plot(row2, { width: 420, height: 330, x: [0, 100], y: [0, 42], xlabel: "time t", ylabel: "contrast of each disc", font: 17 });
  const cpL = DISCS.map((d, k) => cp.line([], [], { color: seriesColor(k), width: 3, label: `r = ${d[2]}` }));
  const cpP = DISCS.map((d, k) => cp.line([0, 40 * d[2] / 2], [40, 0], { color: seriesColor(k), width: 1.5, dash: "3 4" }));
  void cpP;
  cp.legend(undefined, "tr");
  const ro2 = htmlEl("div", "qb-readout", "", v2);
  // ---- image ----
  const v3 = htmlEl("div", "", "", main);
  const row3 = htmlEl("div", "tv-row", "", v3);
  const viewT = new ImgView(row3, "TV"), viewP = new ImgView(row3, "Perona–Malik"), viewH = new ImgView(row3, "heat");
  const pr3 = new Plot(v3, { width: 770, height: 170, x: [0, 127], y: [30, 210], xlabel: "row 112 (test card) / row 64 (micrograph)", ylabel: "value", font: 16, margin: { t: 4, l: 64 } });
  const pr3S = pr3.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, dash: "4 4" });
  const pr3H = pr3.line([], [], { color: "#ff9f1c", width: 2.5 });
  const pr3P = pr3.line([], [], { color: "#2ca02c", width: 2.5 });
  const pr3T = pr3.line([], [], { color: "var(--qb-c1)", width: 3 });
  pr3.legend([{ label: "truth", color: "#999", dash: "4 4" }, { label: "TV", color: seriesColor(0) }, { label: "PM", color: "#2ca02c" }, { label: "heat", color: "#ff9f1c" }], "tr");
  const ro3 = htmlEl("div", "qb-readout", "", v3);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, v3);

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const epsS = logSlider(panel, "ε (1D flow)", 0.05, 5, st.eps, v => { st.eps = Math.round(v * 100) / 100; restart(); }, v => (Math.round(v * 100) / 100).toFixed(2));
  const n1S = slider(panel, "noise (1D)", 0, 10, 1, st.noise1, v => { st.noise1 = v; restart(); });
  const n2S = slider(panel, "noise (image)", 0, 30, 1, st.noise2, v => { st.noise2 = v; });
  n2S.input.addEventListener("change", () => computeImage());
  const setImg = choice(panel, "image", [["card", "test card"], ["cell", "micrograph"]], id => { st.image = id as State["image"]; computeImage(); });
  const ctl = htmlEl("div", "qb-presets", "", panel);
  const stepB = button(ctl, "t + 5", () => { anim.stop(); playB.textContent = "play"; advanceTo(time + 5); draw(); });
  const playB = button(ctl, "play", () => { if (anim.running()) { anim.stop(); playB.textContent = "play"; } else { anim.start(); playB.textContent = "pause"; } });
  const restartB = button(ctl, "restart", () => { anim.stop(); playB.textContent = "play"; st.until = 0; restart(); });
  button(panel, "reset all", () => { anim.stop(); playB.textContent = "play"; Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); restart(); });

  const anim = animator(() => { advanceTo(time + (st.tab === "1d" ? 0.5 : 1)); draw(); if (time >= (st.tab === "1d" ? 200 : 400)) { playB.textContent = "play"; return false; } return true; });

  // ---- state ----
  let time = 0, steps = 0;
  let y1: Float64Array = new Float64Array(N1), t1: Float64Array = new Float64Array(N1), f1: Float64Array = new Float64Array(N1);
  let hist: { t: number; h: number[] }[] = [];
  let d0: Float64Array = discsImage(), d: Float64Array = Float64Array.from(d0);
  let dhist: { t: number; c: number[] }[] = [];
  let cellImg: Float64Array | null = null;

  function plateauHeights(u: ArrayLike<number>): number[] { return PLAT.map(([s, w]) => { let m = 0; for (let i = s; i < s + w; i++) m += u[i]; return m / w; }); }
  function discContrast(u: ArrayLike<number>): number[] { const med = Float64Array.from(u).sort()[8192]; return DISCS.map(([cx, cy]) => u[Math.round(cy) * 128 + Math.round(cx)] - med); }

  function restart(): void {
    time = 0; steps = 0;
    const p = profile1d(st.noise1); t1 = p.truth; y1 = p.y; f1 = Float64Array.from(y1);
    hist = [{ t: 0, h: plateauHeights(f1) }];
    d0 = discsImage(); d = Float64Array.from(d0); dhist = [{ t: 0, c: discContrast(d) }];
    if (st.until > 0 && st.tab !== "image") advanceTo(st.until);
    draw();
    if (st.tab === "image") computeImage();
  }
  function advanceTo(T: number): void {
    if (st.tab === "1d") {
      const dt = st.eps / 2.5;
      while (time + 1e-9 < T) {
        const h = Math.min(dt, T - time);
        f1 = tvFlowStep(f1, N1, 1, h, st.eps); time += h; steps++;
        if (hist.length === 0 || time - hist[hist.length - 1].t >= 0.25) hist.push({ t: time, h: plateauHeights(f1) });
      }
    } else if (st.tab === "discs") {
      while (time + 1e-9 < T) {
        const h = Math.min(2, T - time);
        d = tv2d(d, 128, 128, h, 60); time += h; steps++;
        dhist.push({ t: time, c: discContrast(d) });
      }
    }
  }
  let imgRes: { y: Float64Array; truth: Float64Array; tv: Float64Array; pm: Float64Array; heat: Float64Array; sd: number[]; ps: number[]; sig: number; pmT: number; lam: number } | null = null;
  function computeImage(): void {
    if (st.tab !== "image") return;
    if (st.image === "cell" && !cellImg) { ro3.textContent = "loading…"; return; }
    ro3.textContent = "computing…";
    setTimeout(() => {
      const truth = st.image === "cell" ? cellImg! : testCard().truth, sn = st.noise2;
      const y = addNoise(truth, sn, 6);
      // the discrepancy principle: each method removes as much as the noise, SD(y − u) = σn
      const res = (u: ArrayLike<number>) => { let s = 0, s2 = 0; for (let i = 0; i < y.length; i++) { const d = y[i] - u[i]; s += d; s2 += d * d; } return Math.sqrt(s2 / y.length - (s / y.length) ** 2); };
      let lo = 0, hi = 4 * sn + 10, tv: Float64Array = y;
      for (let k = 0; k < 8; k++) { const m = (lo + hi) / 2; tv = tv2d(y, 128, 128, m, 80); if (res(tv) < sn) lo = m; else hi = m; }
      const lam = (lo + hi) / 2; tv = tv2d(y, 128, 128, lam, 200);
      let a = 0, b = 10, heat: Float64Array = y;
      for (let k = 0; k < 22; k++) { const m = (a + b) / 2; heat = gaussianHeat(y, 128, 128, m); if (res(heat) < sn) a = m; else b = m; }
      const sig = (a + b) / 2; heat = gaussianHeat(y, 128, 128, sig);
      let pm: Float64Array = Float64Array.from(y), pmT = 0;
      const K = Math.max(4, sn);
      for (let k = 0; k < 4000 && res(pm) < sn; k++) { pm = nonlinearStep(pm, 128, 128, 0.05, "g2", K, 1); pmT += 0.05; }
      imgRes = { y, truth, tv, pm, heat, sd: [res(tv), res(pm), res(heat)], ps: [psnr255(tv, truth), psnr255(pm, truth), psnr255(heat, truth), psnr255(y, truth)], sig, pmT, lam };
      draw();
    }, 20);
  }
  function load(p: Preset): void {
    anim.stop(); playB.textContent = "play";
    Object.assign(st, DEFAULT, p.s);
    sync(); restart();
  }
  function sync(): void { setTab(st.tab); epsS.setVal(st.eps); n1S.set(st.noise1); n2S.set(st.noise2); setImg(st.image); }

  function draw(): void {
    setTab(st.tab);
    v1.style.display = st.tab === "1d" ? "" : "none";
    v2.style.display = st.tab === "discs" ? "" : "none";
    v3.style.display = st.tab === "image" ? "" : "none";
    for (const el of [epsS.input.parentElement!, n1S.input.parentElement!]) el.style.display = st.tab === "1d" ? "" : "none";
    for (const el of [n2S.input.parentElement!, setImg.el]) el.style.display = st.tab === "image" ? "" : "none";
    for (const el of [stepB, playB, restartB]) el.style.display = st.tab === "image" ? "none" : "";
    const xs = Array.from({ length: N1 }, (_, i) => i);
    if (st.tab === "1d") {
      p1T.set(xs, y1); p1F.set(xs, f1);
      p1R.set(xs, tv1d(y1, time));
      p1H.set(xs, gaussianHeat(y1, N1, 1, Math.sqrt(2 * time)));
      const T = Math.max(10, time * 1.05);
      hp.setDomain([0, T], [55, 125]);
      PLAT.forEach(([, w], k) => {
        hpL[k].set(hist.map(e => e.t), hist.map(e => e.h[k]));
        hpP[k].set([0, Math.min(T, 60 * w / 2)], [120, 120 - 2 * Math.min(T, 60 * w / 2) / w]);
      });
      const h = plateauHeights(f1);
      ro1.textContent = `time t = ${time.toFixed(1)} after ${steps} explicit steps (Δt = ε/2.5 = ${(st.eps / 2.5).toFixed(3)})\n` +
        `plateau heights: ${h.map(v => v.toFixed(1)).join(", ")}\npredicted (120 − 2t/width, ≥ the surroundings): ${PLAT.map(([, w]) => Math.max(60, 120 - 2 * time / w).toFixed(1)).join(", ")}\n` +
        `dashed red: ROF with λ = t, computed in one go`;
    } else if (st.tab === "discs") {
      viewD.grey(toGrey(d, 128, 128), { mode: "clip", lo: 50, hi: 110 });
      viewD.setCaption(`TV flow, t = ${time.toFixed(0)} (${steps} implicit steps)`);
      const T = Math.max(40, time * 1.05);
      cp.setDomain([0, T], [-2, 42]);
      DISCS.forEach((_, k) => cpL[k].set(dhist.map(e => e.t), dhist.map(e => e.c[k])));
      const c = discContrast(d);
      ro2.textContent = `contrast now: ${DISCS.map(([, , r], k) => `r = ${r}: ${c[k].toFixed(1)}`).join(" · ")}\n` +
        `dashed: 40 − 2t/r, gone at t = 20 r: ${DISCS.map(([, , r]) => (20 * r).toFixed(0)).join(", ")}`;
    } else if (imgRes) {
      const R = imgRes;
      viewT.grey(toGrey(R.tv, 128, 128), { mode: "clip", lo: 0, hi: 255 });
      viewP.grey(toGrey(R.pm, 128, 128), { mode: "clip", lo: 0, hi: 255 });
      viewH.grey(toGrey(R.heat, 128, 128), { mode: "clip", lo: 0, hi: 255 });
      viewT.setCaption(`TV, λ = ${R.lam.toFixed(1)}`); viewP.setCaption(`Perona–Malik, t = ${R.pmT.toFixed(1)}`); viewH.setCaption(`heat, σ = ${R.sig.toFixed(2)}`);
      const row = (st.image === "cell" ? 64 : 112) * 128;
      pr3S.set(xs, R.truth.subarray(row, row + 128)); pr3T.set(xs, R.tv.subarray(row, row + 128));
      pr3P.set(xs, R.pm.subarray(row, row + 128)); pr3H.set(xs, R.heat.subarray(row, row + 128));
      credit.style.display = st.image === "cell" ? "" : "none";
      ro3.textContent = `each tuned to remove as much as the noise: SD(noisy − result) = ${st.noise2} (TV ${R.sd[0].toFixed(1)}, PM ${R.sd[1].toFixed(1)}, heat ${R.sd[2].toFixed(1)})\nPSNR: TV ${R.ps[0].toFixed(1)}, PM ${R.ps[1].toFixed(1)}, heat ${R.ps[2].toFixed(1)} dB (noisy ${R.ps[3].toFixed(1)})`;
    }
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  cell128().then(g => { cellImg = Float64Array.from(g.data); if (st.tab === "image" && st.image === "cell") computeImage(); });
  restart();
}

autoMount(NAME, mount);
