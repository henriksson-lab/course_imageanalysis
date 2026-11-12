// Every state of a tiny Ising lattice (2 x 2: 16 states, 3 x 3: 512 states). Click spins to
// set a state; sliders T, J, h; free or periodic boundary. Readouts for the current state:
// E, e^{-E/T}, Z, p. Plot: the probability of each energy level (count of states x weight) or
// of each magnetisation m. Sample mode: run a Gibbs or Metropolis chain on the lattice; the
// empirical histogram fills in over the exact bars.
// Auto-mounts on <div data-demo="isingenum">. Options: data-preset="hand|cold|hot|field|anti|sample|stuck",
// data-mode="exact|sample", data-tab="energy|m".
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { rng, randInt } from "../lib/random";
import { Plot, seriesColor } from "../lib/plot";
import { lattice, enumerate, energy, stateIndex, Enumeration, Lattice, conditionalPlus, localField, acceptance } from "../lib/mrf";

const NAME = "isingenum";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .ie-left { flex: 0 1 260px; min-width: 200px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .ie-plot { flex: 1 1 380px; min-width: 260px; max-width: 520px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .ie-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .ie-cell { cursor: pointer; }
[data-demo="${NAME}"] .qb-btn:disabled { opacity: .4; cursor: default; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .ie-left { flex-basis: 240px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Settings { size: 2 | 3; periodic: boolean; T: number; J: number; h: number; mode: "exact" | "sample"; sampler: "gibbs" | "metropolis"; tab: "energy" | "m"; start: number }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { size: 3, periodic: true, T: 2, J: 1, h: 0, mode: "exact", sampler: "gibbs", tab: "energy", start: 0b101010101 };
const PRESETS: Preset[] = [
  { id: "hand", label: "2×2 by hand", s: { ...base, size: 2, periodic: false, T: 1, start: 0b1111 } },
  { id: "cold", label: "T → 0: two ground states", s: { ...base, T: 0.3, start: 511 } },
  { id: "hot", label: "T → ∞: all equally likely", s: { ...base, T: 100, start: 0b100110101 } },
  { id: "field", label: "a field breaks the tie", s: { ...base, T: 1.5, h: 0.3, tab: "m", start: 511 } },
  { id: "anti", label: "J < 0: checkerboards win", s: { ...base, periodic: false, T: 1, J: -1, start: 0b101010101 } },
  { id: "sample", label: "sample: converges", s: { ...base, T: 3, mode: "sample", start: 0b100110101 } },
  { id: "stuck", label: "sample: low T, stuck", s: { ...base, T: 1, mode: "sample", sampler: "metropolis", tab: "m", start: 511 } },
];

const fmtNum = (v: number): string => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(3).replace("e+", "e");
  return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : v.toPrecision(3);
};
const Tof = (u: number) => Math.round(10 ** u * 100) / 100;
const uOf = (T: number) => Math.log10(T);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const init: Settings = { ...p0.s };
  if (host.dataset.mode === "sample" || host.dataset.mode === "exact") init.mode = host.dataset.mode;
  if (host.dataset.tab === "m" || host.dataset.tab === "energy") init.tab = host.dataset.tab;
  const st: Settings = { ...init };

  let L: Lattice = lattice(st.size, st.size, st.periodic && st.size > 2);
  let spins = new Int8Array(L.n);
  let en: Enumeration;
  let r = rng(1);
  let counts = new Float64Array(512), nSteps = 0, lastSite = -1, lastAccepted = true;
  let playing = false, speed = 1;

  // ---- layout ----
  const left = htmlEl("div", "ie-left", "", host);
  const figL = htmlEl("figure", "", "", left);
  const capL = htmlEl("figcaption", "", "click a spin to flip it", figL);
  const svg = svgEl("svg", { viewBox: "0 0 300 300", width: "100%" }, figL);
  const ro = htmlEl("div", "qb-readout", "", left);

  const figP = htmlEl("figure", "ie-plot", "", host);
  const tabRow = htmlEl("div", "qb-presets", "", figP);
  const tabBtns = [
    button(tabRow, "energy levels", () => { st.tab = "energy"; drawPlot(true); }),
    button(tabRow, "magnetisation m", () => { st.tab = "m"; drawPlot(true); }),
  ];
  const plotBox = htmlEl("div", "", "", figP);
  let plot: Plot;

  const panel = htmlEl("div", "qb-panel", "", host);
  const sT = slider(panel, "T", -1, 2, 0.01, uOf(st.T), u => { st.T = Tof(u); recompute(); }, u => String(Tof(u)));
  const sJ = slider(panel, "J", -2, 2, 0.1, st.J, v => { st.J = v; recompute(); });
  const sH = slider(panel, "h", -1, 1, 0.05, st.h, v => { st.h = v; recompute(); });
  const rowA = htmlEl("div", "qb-presets", "", panel);
  const sizeBtns = ([2, 3] as const).map(k => button(rowA, `${k}×${k}`, () => { st.size = k; st.start = k === 2 ? 0b1111 : 511; rebuild(); }));
  const bndBtns = [false, true].map(pv => button(rowA, pv ? "periodic" : "free", () => { st.periodic = pv; rebuild(); }));
  const rowM = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = (["exact", "sample"] as const).map(m => button(rowM, m === "exact" ? "exact" : "sample", () => { st.mode = m; resetChain(); syncButtons(); }));
  const samplerBtns = (["gibbs", "metropolis"] as const).map(m => button(rowM, m === "gibbs" ? "Gibbs" : "Metropolis", () => { st.sampler = m; resetChain(); syncButtons(); }));
  const rowS = htmlEl("div", "qb-presets", "", panel);
  const stepBtn = button(rowS, "1 step", () => { step(1); draw(); });
  const playBtn = button(rowS, "play", () => { playing = !playing; syncButtons(); if (playing) loop(); });
  const sSpeed = slider(panel, "steps/frame", 0, 4, 1, 1, v => { speed = 10 ** v; }, v => String(10 ** v));
  speed = 10;
  htmlEl("div", "ie-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  button(panel, "reset", () => { load(init); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });

  function syncButtons(): void {
    sizeBtns.forEach((b, i) => b.classList.toggle("active", [2, 3][i] === st.size));
    bndBtns.forEach((b, i) => { b.classList.toggle("active", [false, true][i] === (st.periodic && st.size > 2)); b.disabled = st.size === 2; });
    modeBtns.forEach((b, i) => b.classList.toggle("active", ["exact", "sample"][i] === st.mode));
    samplerBtns.forEach((b, i) => { b.classList.toggle("active", ["gibbs", "metropolis"][i] === st.sampler); b.style.display = st.mode === "sample" ? "" : "none"; });
    stepBtn.style.display = playBtn.style.display = st.mode === "sample" ? "" : "none";
    (sSpeed.input.parentElement as HTMLElement).style.display = st.mode === "sample" ? "" : "none";
    playBtn.textContent = playing ? "pause" : "play";
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["energy", "m"][i] === st.tab));
    capL.textContent = st.mode === "sample" ? "the chain's current state" : "click a spin to flip it";
  }

  function load(s: Settings): void {
    Object.assign(st, s);
    sT.set(uOf(st.T)); sJ.set(st.J); sH.set(st.h);
    playing = false;
    rebuild();
  }

  function rebuild(): void {
    L = lattice(st.size, st.size, st.periodic && st.size > 2);
    spins = new Int8Array(L.n);
    for (let i = 0; i < L.n; i++) spins[i] = (st.start >> i) & 1 ? 1 : -1;
    plotBox.replaceChildren();
    plot = new Plot(plotBox, { width: 520, height: 400, x: [0, 1], y: [0, 1], xlabel: "energy E", ylabel: "probability" });
    recompute();
  }

  function resetChain(): void {
    r = rng(1);
    counts = new Float64Array(1 << L.n); nSteps = 0; lastSite = -1;
    draw();
  }

  function recompute(): void {
    en = enumerate(L, st.J, st.h, st.T);
    resetChain();
  }

  // ---- sampling: one step = one site, chosen at random ----
  function step(k: number): void {
    for (let i = 0; i < k; i++) {
      const p = randInt(r, L.n);
      const u = localField(L, spins, p, st.J, st.h);
      if (st.sampler === "gibbs") {
        const v = r.unif() < conditionalPlus(u, st.T) ? 1 : -1;
        lastAccepted = v !== spins[p];
        spins[p] = v;
      } else {
        lastAccepted = r.unif() < acceptance(2 * spins[p] * u, st.T);
        if (lastAccepted) spins[p] = -spins[p];
      }
      lastSite = p;
      counts[stateIndex(spins)]++; nSteps++;
    }
  }

  let raf = 0;
  function loop(): void {
    cancelAnimationFrame(raf);
    const tick = () => {
      if (!playing || !host.isConnected) return;
      if (isVisible()) { step(speed); draw(); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  let visible = true;
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(host);
  const isVisible = () => visible;

  // ---- drawing ----
  function drawLattice(): void {
    svg.replaceChildren();
    const n = st.size, pad = 14, cs = (300 - 2 * pad) / n;
    const cx = (i: number) => pad + (i % n + 0.5) * cs, cy = (i: number) => pad + (Math.floor(i / n) + 0.5) * cs;
    // bonds: green if the pair agrees (lowers E when J > 0), red if not
    for (let k = 0; k < L.ep.length; k++) {
      const a = L.ep[k], b = L.eq[k];
      const agree = spins[a] === spins[b];
      const col = (agree === st.J >= 0) ? "var(--qb-c3)" : "var(--qb-c2)";
      const wrap = Math.abs(a % n - b % n) > 1 || Math.abs(Math.floor(a / n) - Math.floor(b / n)) > 1;
      if (wrap) {
        // periodic bond: two stubs leaving the lattice
        const horiz = Math.floor(a / n) === Math.floor(b / n);
        for (const [q, dir] of [[a, 1], [b, -1]] as [number, number][]) {
          const x = cx(q), y = cy(q);
          const x2 = horiz ? x + dir * cs * 0.5 : x, y2 = horiz ? y : y + dir * cs * 0.5;
          svgEl("line", { x1: x, y1: y, x2, y2, "stroke-width": 5, "stroke-dasharray": "5 4", style: `stroke:${col}` }, svg);
        }
      } else svgEl("line", { x1: cx(a), y1: cy(a), x2: cx(b), y2: cy(b), "stroke-width": 6, style: `stroke:${col}` }, svg);
    }
    for (let i = 0; i < L.n; i++) {
      const g = svgEl("g", { class: "ie-cell" }, svg);
      const up = spins[i] > 0;
      const hl = st.mode === "sample" && i === lastSite;
      svgEl("circle", { cx: cx(i), cy: cy(i), r: cs * 0.32, style: `fill:${up ? "#ff7f0e" : "#1f77b4"};stroke:${hl ? (lastAccepted ? "var(--qb-fg)" : "var(--qb-muted)") : "none"};stroke-width:${hl ? 5 : 0};${hl && !lastAccepted ? "stroke-dasharray:6 4" : ""}` }, g);
      const t = svgEl("text", { x: cx(i), y: cy(i) + cs * 0.12, "text-anchor": "middle", "font-size": cs * 0.36, "font-weight": 700, style: "fill:#fff" }, g);
      t.textContent = up ? "+" : "−";
      g.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        spins[i] = -spins[i] as 1 | -1;
        st.start = stateIndex(spins);
        if (st.mode === "sample") resetChain(); else draw();
      });
    }
  }

  function drawReadout(): void {
    const E = energy(L, spins, st.J, st.h);
    const k = stateIndex(spins);
    const w = Math.exp(-E.total / st.T);
    const lines = [
      `E = ${fmtNum(E.total)}`,
      `  = ${fmtNum(E.pair)} (pairs) + ${fmtNum(E.field)} (field)`,
      `e^(−E/T) = ${fmtNum(w)}`,
      `Z = ${fmtNum(en.Z)}  (${1 << L.n} states)`,
      `p(this state) = ${fmtNum(en.p[k])}`,
    ];
    if (st.mode === "sample") lines.push(`steps: ${nSteps}  seen: ${fmtNum(nSteps ? counts[k] / nSteps : 0)}`);
    ro.textContent = lines.join("\n");
  }

  function drawPlot(rescale = false): void {
    if (rescale) syncButtons();
    plot.clear();
    plot.svg.querySelector(":scope > g.qb-legend")?.remove();
    const E = energy(L, spins, st.J, st.h).total;
    if (st.tab === "energy") {
      const xs = en.levels.map(l => l.E), ps = en.levels.map(l => l.p);
      const lo = Math.min(...xs), hi = Math.max(...xs), span = Math.max(1, hi - lo);
      let gap = span; for (let i = 1; i < xs.length; i++) gap = Math.min(gap, xs[i] - xs[i - 1]);
      const bw = Math.min(gap * 0.8, span / 6);
      const ymax = Math.min(1, Math.max(...ps) * 1.18 + 0.02);
      plot.setLabels({ xlabel: "energy E" });
      plot.setDomain([lo - bw - span * 0.04, hi + bw + span * 0.04], [0, ymax]);
      const cur = Math.round(E * 1e9) / 1e9;
      plot.bars(xs, ps, { barWidth: bw, colors: xs.map(x => (Math.abs(x - cur) < 1e-6 ? seriesColor(3) : seriesColor(0))) });
      if (xs.length <= 16) xs.forEach((x, i) => plot.text(x, ps[i] + ymax * 0.015, String(en.levels[i].count), { anchor: "middle", size: 15 }));
      if (st.mode === "sample" && nSteps > 0) {
        const emp = new Map<number, number>();
        for (let k = 0; k < counts.length; k++) if (counts[k]) { const e = Math.round(en.E[k] * 1e9) / 1e9; emp.set(e, (emp.get(e) ?? 0) + counts[k] / nSteps); }
        plot.scatter(xs, xs.map(x => emp.get(x) ?? 0), { color: seriesColor(1), r: 7, label: "chain" });
        plot.legend([{ label: "exact", color: seriesColor(0), kind: "box" }, { label: "chain", color: seriesColor(1), kind: "dot" }]);
      }
    } else {
      const xs = en.mags.map(m => m.m), ps = en.mags.map(m => m.p);
      const ymax = Math.min(1, Math.max(...ps) * 1.18 + 0.02);
      plot.setLabels({ xlabel: "magnetisation m" });
      plot.setDomain([-1.2, 1.2], [0, ymax]);
      const n = L.n, cur = spins.reduce((a, b) => a + b, 0) / n;
      plot.bars(xs, ps, { barWidth: 2 / n * 0.8, colors: xs.map(x => (Math.abs(x - cur) < 1e-9 ? seriesColor(3) : seriesColor(0))) });
      if (st.mode === "sample" && nSteps > 0) {
        const emp = new Map<number, number>();
        for (let k = 0; k < counts.length; k++) if (counts[k]) {
          let s = 0; for (let i = 0; i < n; i++) s += (k >> i) & 1 ? 1 : -1;
          const m = s / n; emp.set(m, (emp.get(m) ?? 0) + counts[k] / nSteps);
        }
        plot.scatter(xs, xs.map(x => emp.get(x) ?? 0), { color: seriesColor(1), r: 7 });
        plot.legend([{ label: "exact", color: seriesColor(0), kind: "box" }, { label: "chain", color: seriesColor(1), kind: "dot" }]);
      }
    }
  }

  function draw(): void {
    syncButtons();
    drawLattice();
    drawReadout();
    drawPlot();
  }

  rebuild();
  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
