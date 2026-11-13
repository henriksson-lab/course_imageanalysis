// A simulated movie of moving particles, linked frame to frame. Particles (seeded) diffuse
// and/or move in a straight line, enter and leave at the border, with localisation noise and an
// optional stage drift. Methods: nearest neighbour, greedy, global assignment with a gate and
// an end/start cost b (lib/linking.ts). The field shows the current frame with track tails
// (colour = track) and wrong links as red crosses. Readouts: displacement/spacing ratio, link
// error rate, perfect tracks. Side plot: link error rate vs the ratio for all three methods on
// a sweep, with the nearest-neighbour estimate a/(1+a), a = π ρ²/4.
//
// <div data-demo="linkmovie" data-preset="easy|half|one|drift|blow|bhigh" data-sweep="on|off"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { labelColor, toCss } from "../lib/colour";
import * as LK from "../lib/linking";
import { simParticles, SimMovie, meanSpacing } from "../lib/tracksim";
import { Method, Motion, State, T, W, GATE_OFF, DEFAULT, PRESETS, METHODS, MCOL, DRIFT, ratioOf, simulate, link } from "./model";

const NAME = "linkmovie";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .7em; }
${S} .lm-field { flex: 1 1 440px; min-width: 260px; max-width: 520px; }
${S} .lm-field svg { width: 100%; display: block; background: #fbfbfd; border: 1px solid #ccc; border-radius: 4px; }
${S} .qb-panel { flex: 1 1 300px; max-width: 360px; gap: .3em; }
${S} .lm-side { flex: 1 1 300px; max-width: 380px; display: flex; flex-direction: column; gap: .3em; }
${S} .lm-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 17px; white-space: pre-wrap; }
${S} .qb-presets { flex-basis: 100%; }
${S} .qb-btn.lm-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
.reveal ${S} { font-size: 18px; gap: .5em; }
.reveal ${S} .lm-field { flex: 0 0 430px; }
.reveal ${S} .qb-panel { flex: 0 0 330px; gap: .1em; }
.reveal ${S} .lm-side { flex: 0 0 330px; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  let st: State = { ...DEFAULT };
  const showSweep = host.dataset.sweep !== "off";
  const fieldBox = htmlEl("div", "lm-field", "", host);
  const svg = svgEl("svg", { viewBox: `-4 -4 ${W + 8} ${W + 8}`, role: "img", "aria-label": "simulated particles and their tracks" }, fieldBox);
  const gTails = svgEl("g", {}, svg), gDots = svgEl("g", {}, svg), gBad = svgEl("g", {}, svg);
  const frameLab = svgEl("text", { x: 4, y: 14, class: "qb-halo" }, svg);
  frameLab.style.fontSize = "11px";
  const panel = htmlEl("div", "qb-panel", "", host);
  const side = htmlEl("div", "lm-side", "", host);
  const read = htmlEl("div", "qb-readout", "", side);
  let sweepPlot: Plot | null = null;
  if (showSweep) sweepPlot = new Plot(side, { width: 380, height: 290, x: [0, 1.3], y: [0, 0.6], xlabel: "displacement / spacing", ylabel: "link error", font: 24, xticks: [0, 0.5, 1], yticks: [0, 0.2, 0.4, 0.6] });

  const methRow = htmlEl("div", "lm-row", "", panel);
  const methBtns = METHODS.map(([m, l]) => button(methRow, l, () => { st.method = m; recompute(); }));
  const motRow = htmlEl("div", "lm-row", "", panel);
  const motBtns = (["rw", "directed", "mixed"] as Motion[]).map(m => button(motRow, m === "rw" ? "random walk" : m, () => { st.motion = m; recompute(true); }));
  const sl = {
    n: slider(panel, "particles", 10, 200, 5, st.n, v => { st.n = v; recompute(true); }),
    D: slider(panel, "D (px²/s)", 0.5, 40, 0.1, st.D, v => { st.D = v; recompute(true); }, v => v.toFixed(1)),
    dt: slider(panel, "Δt (s)", 0.1, 2, 0.05, st.dt, v => { st.dt = v; recompute(true); }, v => v.toFixed(2)),
    sigma: slider(panel, "noise σ (px)", 0, 3, 0.1, st.sigma, v => { st.sigma = v; recompute(true); }, v => v.toFixed(1)),
    gate: slider(panel, "gate (px)", 2, GATE_OFF, 1, st.gate, v => { st.gate = v; recompute(); }, v => (v >= GATE_OFF ? "off" : String(v))),
    bd: slider(panel, "b as distance (px)", 1, 60, 0.5, st.bd, v => { st.bd = v; recompute(); }, v => v.toFixed(1)),
  };
  const chk = htmlEl("div", "lm-row", "", panel);
  const driftC = checkbox(chk, "stage drift", st.drift, v => { st.drift = v; recompute(true); });
  const corrC = checkbox(chk, "correct drift", st.correct, v => { st.correct = v; recompute(); });
  const tRow = htmlEl("div", "lm-row", "", panel);
  let timer = 0;
  const playBtn = button(tRow, "play", () => {
    if (timer) { stop(); return; }
    playBtn.textContent = "pause";
    timer = window.setInterval(() => { st.t = (st.t + 1) % T; tS.set(st.t); draw(); }, 140);
  });
  const stop = () => { window.clearInterval(timer); timer = 0; playBtn.textContent = "play"; };
  const tS = slider(tRow, "frame", 0, T - 1, 1, st.t, v => { st.t = v; draw(); });
  presets(host, [...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s) })), { label: "reset", apply: () => apply({}) }]);

  let mv: SimMovie = simulate(st), edges: LK.Edge[] = [], wrong: boolean[] = [], trackId: ArrayLike<number> = new Int32Array(0);
  let stats = { err: 0, prec: 0, perfect: 0, total: 0 };

  function apply(p: Partial<State>): void {
    stop();
    st = { ...DEFAULT, ...p };
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].set(st[k]);
    driftC.checked = st.drift; corrC.checked = st.correct; tS.set(st.t);
    recomputeNow(true);
  }

  let sweepKey = "", sweepTimer = 0, pending = 0, pendingSim = false, lastMs = 0;
  /** Slider moves: recompute at once if it is fast, else shortly after the slider stops. */
  function recompute(resim = false): void {
    pendingSim = pendingSim || resim;
    window.clearTimeout(pending);
    if (lastMs < 30) { const r = pendingSim; pendingSim = false; recomputeNow(r); }
    else pending = window.setTimeout(() => { const r = pendingSim; pendingSim = false; recomputeNow(r); }, 120);
  }
  function recomputeNow(resim: boolean): void {
    const t0 = performance.now();
    if (resim) mv = simulate(st);
    edges = link(mv, st, st.method);
    const ev = LK.evalLinks(edges, mv.truth);
    wrong = ev.wrong;
    trackId = LK.chains(mv.dets.length, edges).id;
    const ct = LK.completeTracks(mv.dets.length, edges, mv.truth);
    stats = { err: 1 - ev.recall, prec: ev.precision, perfect: ct.total ? ct.complete / ct.total : 1, total: ct.total };
    lastMs = performance.now() - t0;
    methBtns.forEach((b, i) => b.classList.toggle("lm-on", METHODS[i][0] === st.method));
    motBtns.forEach((b, i) => b.classList.toggle("lm-on", (["rw", "directed", "mixed"] as Motion[])[i] === st.motion));
    draw();
    if (sweepPlot) {
      const key = `${st.n}|${st.sigma}`;
      if (key !== sweepKey) { sweepKey = key; window.clearTimeout(sweepTimer); sweepTimer = window.setTimeout(drawSweep, 60); }
      else drawCurrent();
    }
  }

  // the sweep: random walk at the same density and noise, D varied to cover ratios 0.1 … 1.2;
  // gate 3.5 and b 2.5 rms steps (sensible settings for each ratio)
  const RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2];
  let sweepRes: Record<Method, number[]> = { nn: [], greedy: [], global: [] };
  function drawSweep(): void {
    if (!sweepPlot) return;
    sweepRes = { nn: [], greedy: [], global: [] };
    const sp = meanSpacing(st.n / (W * W));
    for (const r of RATIOS) {
      const D = (r * sp) ** 2 / 4;
      const rms = r * sp;
      const s2: State = { ...st, D, dt: 1, motion: "rw", drift: false, correct: false, gate: 3.5 * rms, bd: 2.5 * rms };
      const m2 = simParticles({ n: st.n, T: 12, W, H: W, motion: "rw", D, dt: 1, speed: 0, sigma: st.sigma, seed: 77 });
      for (const [m] of METHODS) sweepRes[m].push(1 - LK.evalLinks(link(m2, s2, m), m2.truth).recall);
    }
    drawCurrent();
  }
  function drawCurrent(): void {
    const p = sweepPlot!;
    p.clear();
    p.fn(r => { const a = Math.PI * r * r / 4; return a / (1 + a); }, { color: "#555", dash: "6 4", width: 2.5, label: "a/(1+a)" });
    for (const [m, l] of METHODS) {
      p.line(RATIOS, sweepRes[m], { color: MCOL[m], width: 2, opacity: 0.6 });
      p.scatter(RATIOS, sweepRes[m], { color: MCOL[m], r: 4, label: l });
    }
    const r = ratioOf(st);
    p.scatter([Math.min(1.29, r)], [Math.min(0.59, stats.err)], { color: MCOL[st.method], r: 9, stroke: "#000" });
    p.legend(undefined, "tl");
  }

  function draw(): void {
    gTails.replaceChildren(); gDots.replaceChildren(); gBad.replaceChildren();
    const t = st.t, tail = 10;
    const col = (k: number) => toCss(labelColor(1 + (trackId[k] % 97)));
    edges.forEach((e, i) => {
      const da = mv.dets[e.a], db = mv.dets[e.b];
      if (db.t > t || db.t <= t - tail) return;
      const l = svgEl("line", { x1: da.x, y1: da.y, x2: db.x, y2: db.y }, gTails);
      l.style.stroke = wrong[i] ? "#d62728" : col(e.b); l.style.strokeWidth = wrong[i] ? "1.6" : "1.4";
      l.style.opacity = String(0.35 + 0.65 * (1 - (t - db.t) / tail));
      if (wrong[i]) {
        const mx = (da.x + db.x) / 2, my = (da.y + db.y) / 2, s = 2.4;
        const c = svgEl("path", { d: `M${mx - s},${my - s}L${mx + s},${my + s}M${mx - s},${my + s}L${mx + s},${my - s}` }, gBad);
        c.style.stroke = "#d62728"; c.style.strokeWidth = "1.8";
      }
    });
    mv.dets.forEach((d, k) => {
      if (d.t !== t) return;
      const c = svgEl("circle", { cx: d.x, cy: d.y, r: 2.6 }, gDots);
      c.style.fill = col(k); c.style.stroke = "#222"; c.style.strokeWidth = "0.6";
    });
    frameLab.textContent = `frame ${t}`;
    const ratio = ratioOf(st);
    read.textContent = [
      `displacement / spacing  ${ratio.toFixed(2)}`,
      `link error rate  ${(100 * stats.err).toFixed(1)} %`,
      `wrong links made  ${(100 * (1 - stats.prec)).toFixed(1)} %`,
      `perfect tracks  ${(100 * stats.perfect).toFixed(0)} % of ${stats.total}`,
      ...(st.drift ? [`drift ${Math.hypot(...DRIFT).toFixed(1)} px/frame${st.correct ? ", corrected" : ""}`] : []),
    ].join("\n");
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply(pre ? pre.s : {});
});
