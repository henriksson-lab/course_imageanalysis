// Diffusion in 2D (modern deep learning day, unit diffusion/). Data = Gaussian mixtures, so the
// score of the noised data is exact. Forward tab: 500 data points melt into N(0, I) as the
// time τ goes from 0 to 1 (schedule plot beside). Reverse tab: 500 points start as noise and
// are denoised step by step (DDPM or DDIM, 2–1000 steps); arrows = the denoising direction at
// the current noise level; one trajectory highlighted. Score: exact, learned (a small network,
// two snapshots, train_score.py), conditioned on a group with guidance weight w.
// Auto-mounts on <div data-demo="diffusion2d">. Options: data-preset="noise|few|rare|copy|cond|guide|forward".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { alphaBarAt, type Schedule } from "../lib/mixture";
import * as M from "./model";
import { SCORE_NETS } from "./score_weights";

const NAME = "diffusion2d";
const NPTS = 500;
const DATA: M.DataId[] = ["two", "ring", "rare", "moons", "twenty"];
const DATA_LABEL: Record<M.DataId, string> = { two: "two blobs", ring: "ring of 8", rare: "rare blob", moons: "two moons", twenty: "20 points" };
type Source = "exact" | "under" | "trained";

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .df-main { flex: 0 1 470px; min-width: 260px; }
[data-demo="${NAME}"] .df-side { flex: 0 1 250px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .3em; }
[data-demo="${NAME}"] .df-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .df-hide { display: none !important; }
[data-demo="${NAME}"] .df-dis { opacity: .4; pointer-events: none; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .df-main { flex: 0 0 420px; }
.reveal [data-demo="${NAME}"] .df-side { flex: 0 0 230px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 430px; max-width: 480px; }
.reveal [data-demo="${NAME}"] .df-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; line-height: 1.2; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State {
  tab: "forward" | "reverse"; data: M.DataId; tau: number; sched: Schedule;
  steps: number; sampler: M.Sampler; source: Source; cond: number | null; w: number; shown: number;
}
interface Preset { id: string; label: string; s: Partial<State>; run?: boolean }
const BASE: State = { tab: "reverse", data: "two", tau: 0.5, sched: "cosine", steps: 100, sampler: "ddpm", source: "exact", cond: null, w: 0, shown: 0 };
const PRESETS: Preset[] = [
  { id: "forward", label: "forward: data melts into noise", s: { tab: "forward", data: "moons", tau: 0.4 } },
  { id: "noise", label: "noise in, data out", s: { data: "moons", steps: 100 }, run: true },
  { id: "few", label: "too few steps: between the blobs", s: { data: "two", steps: 2, sampler: "ddim" }, run: true },
  { id: "rare", label: "learned score: the rare blob", s: { data: "rare", source: "under", steps: 100 }, run: true },
  { id: "copy", label: "perfect score of 20 points: copies", s: { data: "twenty", steps: 200 }, run: true },
  { id: "cond", label: "conditioning on one blob", s: { data: "ring", cond: 0, w: 0, steps: 100 }, run: true },
  { id: "guide", label: "strong guidance", s: { data: "ring", cond: 1, w: 4, steps: 100 }, run: true },
];
const stepsOf = (u: number) => Math.round(2 * 500 ** (u / 100));      // 2 … 1000
const uOf = (n: number) => 100 * Math.log(n / 2) / Math.log(500);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st: State = { ...BASE };
  let ds = M.dataSet(st.data), run: M.Run | null = null, anim = 0;

  const main = htmlEl("div", "df-main", "", host);
  const plot = new Plot(main, { width: 520, height: 520, x: [-3, 3], y: [-3, 3], equal: true, xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], ariaLabel: "points" });
  const side = htmlEl("div", "df-side", "", host);
  const sp = new Plot(side, { width: 300, height: 230, x: [0, 1], y: [0, 1], xlabel: "time τ", title: "schedule", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1] });
  const sigLine = sp.line([], [], { color: seriesColor(0), label: "√ā (signal)" });
  const noiLine = sp.line([], [], { color: seriesColor(1), label: "√(1−ā) (noise)" });
  sp.legend(undefined, "tr");
  const tMark = sp.vline(0.5, { color: "var(--qb-muted)", dash: "4 3", width: 2 });
  const ro = htmlEl("div", "qb-readout", "", side);

  const arrows: { remove(): void }[] = [];
  const dataMark = plot.scatter([], [], { color: "#bbb", r: 3 });
  const ptsMark = plot.scatter([], [], { color: seriesColor(0), r: 3.2, opacity: 0.75 });
  const trajMark = plot.line([], [], { color: seriesColor(3), width: 1.8 });
  const trajDot = plot.scatter([], [], { color: seriesColor(3), r: 7 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "df-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  const tabRow = htmlEl("div", "qb-presets", "", panel);
  const tabBtns = (["forward", "reverse"] as const).map(t => button(tabRow, t === "forward" ? "forward (noising)" : "reverse (sampling)", () => { st.tab = t; stop(); refresh(); }));
  htmlEl("div", "df-lab", "data", panel);
  const dRow = htmlEl("div", "qb-presets", "", panel);
  const dBtns = DATA.map(d => button(dRow, DATA_LABEL[d], () => { st.data = d; st.cond = null; newRun(); }));
  const fwdBox = htmlEl("div", "", "", panel);
  const tauS = slider(fwdBox, "time τ", 0, 1, 0.01, st.tau, v => { st.tau = v; refresh(); });
  const revBox = htmlEl("div", "", "", panel);
  const stepS = slider(revBox, "steps", 0, 100, 1, uOf(st.steps), u => { st.steps = stepsOf(u); newRun(); }, u => String(stepsOf(u)));
  const r1 = htmlEl("div", "qb-presets", "", revBox);
  const smpBtns = (["ddpm", "ddim"] as M.Sampler[]).map(s => button(r1, s === "ddpm" ? "stochastic" : "deterministic", () => { st.sampler = s; newRun(); }));
  const schBtns = (["cosine", "linear"] as Schedule[]).map(s => button(r1, s, () => { st.sched = s; newRun(); }));
  const r2 = htmlEl("div", "qb-presets", "", revBox);
  const srcBtns = (["exact", "under", "trained"] as Source[]).map(s => button(r2, s === "exact" ? "exact score" : s === "under" ? "learned: short" : "learned: long", () => { st.source = s; newRun(); }));
  const r3 = htmlEl("div", "qb-presets", "", revBox);
  const condBtns = [null, 0, 1].map(c => button(r3, "", () => { st.cond = c; newRun(); }));
  const wS = slider(revBox, "guidance w", 0, 6, 0.25, st.w, v => { st.w = v; newRun(); });
  const r4 = htmlEl("div", "qb-presets", "", revBox);
  button(r4, "step", () => { stop(); if (run) { st.shown = Math.min(run.N, st.shown + 1); draw(); } });
  button(r4, "run", () => play());
  button(r4, "back to noise", () => { stop(); st.shown = 0; draw(); });
  button(panel, "reset", () => applyPreset(p0));
  htmlEl("div", "qb-note", "Grey: the training data. Blue: the points being sampled. Arrows: where the denoiser pushes at the current noise level. Orange: one point's path.", panel);

  function scoreFn(): M.ScoreFn {
    if (st.source !== "exact" && SCORE_NETS[st.data]) return M.learnedScore(SCORE_NETS[st.data][st.source]);
    return M.exactScore(ds, st.cond, st.w);
  }

  function applyPreset(p: Preset): void {
    stop();
    Object.assign(st, BASE, p.s);
    tauS.set(st.tau); stepS.set(uOf(st.steps)); wS.set(st.w);
    newRun();
    if (p.run && run) { st.shown = run.N; draw(); }
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function newRun(): void {
    stop();
    ds = M.dataSet(st.data);
    if (st.source !== "exact" && !SCORE_NETS[st.data]) st.source = "exact";
    if (st.source !== "exact") st.cond = null;
    run = M.sample(scoreFn(), NPTS, st.steps, st.sampler, st.sched, 11);
    st.shown = 0;
    refresh();
  }

  function refresh(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["forward", "reverse"][i] === st.tab));
    dBtns.forEach((b, i) => b.classList.toggle("active", DATA[i] === st.data));
    smpBtns.forEach((b, i) => b.classList.toggle("active", ["ddpm", "ddim"][i] === st.sampler));
    schBtns.forEach((b, i) => b.classList.toggle("active", ["cosine", "linear"][i] === st.sched));
    srcBtns.forEach((b, i) => { b.classList.toggle("active", ["exact", "under", "trained"][i] === st.source); b.classList.toggle("df-dis", i > 0 && !SCORE_NETS[st.data]); });
    condBtns.forEach((b, i) => { b.textContent = i === 0 ? "no condition" : ds.groupNames[i - 1]; b.classList.toggle("active", st.cond === [null, 0, 1][i]); b.classList.toggle("df-dis", st.source !== "exact"); });
    (wS.input.parentElement as HTMLElement).classList.toggle("df-dis", st.cond === null || st.source !== "exact");
    fwdBox.classList.toggle("df-hide", st.tab !== "forward");
    revBox.classList.toggle("df-hide", st.tab !== "reverse");
    const ts = Array.from({ length: 101 }, (_, i) => i / 100);
    sigLine.set(ts, ts.map(t => Math.sqrt(alphaBarAt(t, st.sched))));
    noiLine.set(ts, ts.map(t => Math.sqrt(1 - alphaBarAt(t, st.sched))));
    const dpts = M.forwardPoints(ds, NPTS, 0, st.sched);
    dataMark.set(ev(dpts), od(dpts));
    draw();
  }

  const ev = (X: Float64Array) => Array.from({ length: X.length / 2 }, (_, i) => X[2 * i]);
  const od = (X: Float64Array) => Array.from({ length: X.length / 2 }, (_, i) => X[2 * i + 1]);

  function drawArrows(tau: number): void {
    for (const a of arrows) a.remove();
    arrows.length = 0;
    if (tau <= 0) return;
    const ab = alphaBarAt(tau, st.sched), f = scoreFn(), s = new Float64Array(2);
    for (let gx = -2.75; gx <= 2.76; gx += 0.5) for (let gy = -2.75; gy <= 2.76; gy += 0.5) {
      f(gx, gy, tau, ab, s);
      // one Tweedie step: from x to the denoised guess E[x_0 | x_τ] (scaled back to x_τ's level), capped
      let dx = (1 - ab) * s[0], dy = (1 - ab) * s[1];
      const L = Math.hypot(dx, dy), cap = 0.45;
      if (L > cap) { dx *= cap / L; dy *= cap / L; }
      if (L < 0.02) continue;
      arrows.push(plot.arrow(gx, gy, gx + dx, gy + dy, { color: "var(--qb-muted)", width: 1.6 }));
    }
  }

  function draw(): void {
    if (st.tab === "forward") {
      const X = M.forwardPoints(ds, NPTS, st.tau, st.sched);
      ptsMark.set(ev(X), od(X));
      trajMark.set([], []); trajDot.set([], []);
      tMark.set(st.tau);
      drawArrows(0);
      const ab = alphaBarAt(st.tau, st.sched);
      ro.textContent = `τ = ${st.tau.toFixed(2)}\nsignal √ā = ${Math.sqrt(ab).toFixed(3)}\nnoise √(1−ā) = ${Math.sqrt(1 - ab).toFixed(3)}\nSNR ā/(1−ā) = ${(ab / (1 - ab)).toPrecision(3)}`;
      return;
    }
    if (!run) return;
    const X = run.frames[st.shown], tau = run.taus[st.shown];
    ptsMark.set(ev(X), od(X));
    trajMark.set(run.frames.slice(0, st.shown + 1).map(F => F[0]), run.frames.slice(0, st.shown + 1).map(F => F[1]));
    trajDot.set([X[0]], [X[1]]);
    tMark.set(tau);
    drawArrows(tau);
    let txt = `step ${st.shown} / ${run.N}   τ = ${tau.toFixed(3)}`;
    if (st.shown === run.N) {
      const s = M.summarise(ds, X);
      const tw = ds.groups.map(g => g.reduce((a, j) => a + ds.m.w[j], 0));
      txt += `\n${ds.groupNames.map((n, k) => `${n}: ${(100 * s.perGroup[k]).toFixed(0)} % (data ${(100 * tw[k]).toFixed(0)} %)`).join("\n")}` +
        `\noff the data: ${(100 * s.off).toFixed(0)} %`;
      if (st.data === "ring") {
        const cnt = new Array(8).fill(0);
        for (let p = 0; p < X.length / 2; p++) {
          let b = 0, bd = Infinity;
          for (let j = 0; j < 8; j++) { const d = Math.hypot(X[2 * p] - ds.m.mu[2 * j], X[2 * p + 1] - ds.m.mu[2 * j + 1]); if (d < bd) { bd = d; b = j; } }
          cnt[b]++;
        }
        txt += `\nper blob (from 0°, anticlockwise):\n  ${cnt.join(" ")}`;
      }
      if (st.data === "twenty") txt += `\nwithin 0.05 of a training point: ${(100 * s.nearTrain).toFixed(0)} %`;
    }
    ro.textContent = txt;
  }

  function stop(): void { if (anim) cancelAnimationFrame(anim); anim = 0; }
  function play(): void {
    stop();
    if (!run) return;
    if (st.shown >= run.N) st.shown = 0;
    const per = Math.max(1, Math.round(run.N / 60));
    const tick = () => {
      st.shown = Math.min(run!.N, st.shown + per); draw();
      if (st.shown < run!.N) anim = requestAnimationFrame(tick); else anim = 0;
    };
    tick();
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
