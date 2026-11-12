// A large 2D Ising lattice (periodic, 64/128/256 squared), animated with Metropolis or Gibbs
// (heat-bath) sweeps in checkerboard order. Sliders T (T_c = 2.27 marked), h, J; start random
// or all +1; play / one sweep / reset; cooling: anneal from the current T down to 0.5 over a
// chosen number of sweeps, or quench. Side panel: magnetisation and energy per spin against
// the sweep number, or a T-scan on its own 64 x 64 lattice (J = 1, h = 0) that plots <|m|>
// against T with Onsager's exact curve.
// Auto-mounts on <div data-demo="isinglattice">. Options: data-preset="hot|coarsen|critical|field|quench|slow|slowdown",
// data-size="64|128|256", data-tab="traces|scan".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { rng, Rng } from "../lib/random";
import { Plot, seriesColor } from "../lib/plot";
import { lattice, Lattice, metropolisSweep, gibbsSweep, magnetisation, energy, randomSpins, onsagerM, TC } from "../lib/mrf";

const NAME = "isinglattice";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .9em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] .il-main { flex: 0 1 370px; min-width: 220px; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .il-side { flex: 1 1 330px; min-width: 240px; max-width: 440px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .il-scan { display: flex; gap: .5em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .il-scan canvas { width: 96px; flex: 0 0 96px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .il-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .12em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .il-main { flex-basis: 360px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; }
`;

/** lib/plot.ts numbers its clip paths per bundle (qb-clip-1, ...), so two demos on one page
 *  can share an id and one plot clips to another's box. Give this plot's clip path a unique id. */
function uniqueClip(p: Plot): Plot {
  const cp = p.svg.querySelector("clipPath");
  if (cp) {
    const w = window as unknown as { __qbClipSeq?: number };
    w.__qbClipSeq = (w.__qbClipSeq ?? 0) + 1;
    const id = `${NAME}-clip-${w.__qbClipSeq}`;
    cp.id = id;
    p.layer.setAttribute("clip-path", `url(#${id})`);
  }
  return p;
}

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Algo = "metropolis" | "gibbs";
interface Settings { size: number; T: number; J: number; h: number; algo: Algo; start: "random" | "plus"; coolTo?: number; coolSweeps: number; play: boolean }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { size: 256, T: 2, J: 1, h: 0, algo: "metropolis", start: "random", coolSweeps: 1000, play: true };
const PRESETS: Preset[] = [
  { id: "hot", label: "hot: noise", s: { ...base, T: 5 } },
  { id: "coarsen", label: "cold: domains coarsen", s: { ...base, T: 1.5 } },
  { id: "critical", label: "at T_c: all scales", s: { ...base, T: 2.27 } },
  { id: "field", label: "a small field decides", s: { ...base, T: 1.8, h: 0.05 } },
  { id: "quench", label: "quench", s: { ...base, size: 128, T: 0.5 } },
  { id: "slow", label: "slow cooling", s: { ...base, size: 128, T: 3.5, coolTo: 0.5, coolSweeps: 300 } },
  { id: "slowdown", label: "critical slowing down", s: { ...base, size: 128, T: 2.3, start: "plus" } },
];
const SIZES = [64, 128, 256];
const COOL = [100, 300, 1000, 3000, 10000];
const TRACE = 400;   // sweeps shown in the trace plots

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const init: Settings = { ...p0.s };
  const ds = Number(host.dataset.size);
  if (SIZES.includes(ds)) init.size = ds;
  const st: Settings = { ...init };

  let L: Lattice, s: Int8Array, r: Rng;
  let sweep = 0, playing = false, perFrame = 2;
  let cooling: { from: number; rate: number } | null = null;
  const tr = { k: [] as number[], m: [] as number[], e: [] as number[] };

  // ---- layout ----
  const main = htmlEl("figure", "il-main", "", host);
  const cap = htmlEl("figcaption", "", "", main);
  const cv = htmlEl("canvas", "", "", main);
  const ro = htmlEl("div", "qb-readout", "", main);
  ro.style.marginTop = ".3em";

  const side = htmlEl("div", "il-side", "", host);
  const tabRow = htmlEl("div", "qb-presets", "", side);
  let tab: "traces" | "scan" = host.dataset.tab === "scan" ? "scan" : "traces";
  const tabBtns = [button(tabRow, "m and E per sweep", () => setTab("traces")), button(tabRow, "T-scan", () => setTab("scan"))];
  const traceBox = htmlEl("div", "", "", side);
  const pm = uniqueClip(new Plot(traceBox, { width: 440, height: 210, x: [0, TRACE], y: [-1, 1], ylabel: "m", yticks: [-1, -0.5, 0, 0.5, 1], margin: { b: 30 } }));
  const pe = uniqueClip(new Plot(traceBox, { width: 440, height: 230, x: [0, TRACE], y: [-2, 0], xlabel: "sweep", ylabel: "E / n" }));
  pm.hline(0, { color: "var(--qb-muted)", width: 1.5 });
  const lm = pm.line([], [], { color: seriesColor(0) });
  const le = pe.line([], [], { color: seriesColor(1) });
  const scanBox = htmlEl("div", "", "", side);
  const ps = uniqueClip(new Plot(scanBox, { width: 440, height: 340, x: [0.5, 4.5], y: [0, 1.05], xlabel: "temperature T", ylabel: "⟨|m|⟩" }));
  ps.fn(T => onsagerM(T), { color: seriesColor(0), width: 3, label: "exact (Onsager)" }, 400);
  ps.vline(TC, { color: "var(--qb-muted)", dash: "6 4", width: 2, text: "T_c" });
  const scanDots = ps.scatter([], [], { color: seriesColor(1), r: 6, label: "64 × 64 lattice" });
  ps.legend(undefined, "bl");
  const scanRow = htmlEl("div", "il-scan", "", scanBox);
  const scv = htmlEl("canvas", "", "", scanRow);
  const scanCtl = htmlEl("div", "", "", scanRow);
  const scanBtn = button(scanCtl, "run T-scan", () => toggleScan());
  const scanRo = htmlEl("div", "qb-note", "64 × 64, J = 1, h = 0; from T = 4 down to 1: 100 sweeps to settle, 200 to measure at each T.", scanCtl);
  scanRo.style.marginTop = ".3em";

  const panel = htmlEl("div", "qb-panel", "", host);
  const sT = slider(panel, "T", 0.5, 5, 0.01, st.T, v => { st.T = v; cooling = null; drawReadout(); });
  const sH = slider(panel, "h", -0.3, 0.3, 0.01, st.h, v => { st.h = v; drawReadout(); });
  const sJ = slider(panel, "J", 0.2, 2, 0.05, st.J, v => { st.J = v; drawReadout(); });
  const rowA = htmlEl("div", "qb-presets", "", panel);
  const algoBtns = (["metropolis", "gibbs"] as Algo[]).map(a => button(rowA, a === "metropolis" ? "Metropolis" : "Gibbs", () => { st.algo = a; syncButtons(); }));
  button(rowA, "T = T_c", () => { st.T = Math.round(TC * 100) / 100; sT.set(st.T); cooling = null; drawReadout(); });
  const rowB = htmlEl("div", "qb-presets", "", panel);
  const playBtn = button(rowB, "play", () => { playing = !playing; syncButtons(); if (playing) loop(); });
  button(rowB, "1 sweep", () => { doSweep(); draw(); });
  button(rowB, "random start", () => { st.start = "random"; restart(); });
  button(rowB, "all +1", () => { st.start = "plus"; restart(); });
  const rowC = htmlEl("div", "qb-presets", "", panel);
  const sizeBtns = SIZES.map(n => button(rowC, `${n}²`, () => { st.size = n; restart(); }));
  const sSpeed = slider(panel, "sweeps/frame", 1, 8, 1, perFrame, v => { perFrame = v; });
  const sCool = slider(panel, "cooling sweeps", 0, COOL.length - 1, 1, COOL.indexOf(st.coolSweeps), v => { st.coolSweeps = COOL[v]; }, v => String(COOL[v]));
  const rowD = htmlEl("div", "qb-presets", "", panel);
  button(rowD, "cool to 0.5", () => startCooling(0.5));
  button(rowD, "quench to 0.5", () => { cooling = null; st.T = 0.5; sT.set(0.5); drawReadout(); });
  htmlEl("div", "il-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  button(panel, "reset", () => { load(init); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });

  function setTab(t: "traces" | "scan"): void { tab = t; syncButtons(); }

  function syncButtons(): void {
    algoBtns.forEach((b, i) => b.classList.toggle("active", ["metropolis", "gibbs"][i] === st.algo));
    sizeBtns.forEach((b, i) => b.classList.toggle("active", SIZES[i] === st.size));
    playBtn.textContent = playing ? "pause" : "play";
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["traces", "scan"][i] === tab));
    traceBox.style.display = tab === "traces" ? "" : "none";
    scanBox.style.display = tab === "scan" ? "" : "none";
    scanBtn.textContent = scan.running ? "stop" : scan.Ts.length ? "run again" : "run T-scan";
  }

  function load(p: Settings): void {
    Object.assign(st, p);
    sT.set(st.T); sH.set(st.h); sJ.set(st.J); sCool.set(COOL.indexOf(st.coolSweeps));
    restart();
    if (p.coolTo !== undefined) startCooling(p.coolTo);
    playing = p.play;
    syncButtons();
    if (playing) loop();
  }

  function restart(): void {
    L = lattice(st.size, st.size, true);
    r = rng(1);
    s = st.start === "plus" ? new Int8Array(L.n).fill(1) : randomSpins(L.n, r);
    cv.width = cv.height = st.size;
    sweep = 0; cooling = null;
    tr.k = []; tr.m = []; tr.e = [];
    record();
    draw();
  }

  function startCooling(Tend: number): void {
    if (st.T <= Tend) return;
    cooling = { from: st.T, rate: (Tend / st.T) ** (1 / st.coolSweeps) };
    drawReadout();
  }

  function record(): void {
    tr.k.push(sweep); tr.m.push(magnetisation(s)); tr.e.push(energy(L, s, st.J, st.h).total / L.n);
    if (tr.k.length > TRACE + 1) { tr.k.shift(); tr.m.shift(); tr.e.shift(); }
  }

  function doSweep(): void {
    if (cooling) {
      st.T = Math.max(0.5, st.T * cooling.rate);
      if (st.T <= 0.5 + 1e-9) { st.T = 0.5; cooling = null; }
      sT.set(Math.round(st.T * 100) / 100);
    }
    if (st.algo === "metropolis") metropolisSweep(L, s, st.J, st.h, st.T, r);
    else gibbsSweep(L, s, st.J, st.h, st.T, r);
    sweep++;
    record();
  }

  // ---- animation (paused while off screen) ----
  let visible = true, raf = 0;
  new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && (playing || scan.running)) loop(); }).observe(host);
  function loop(): void {
    cancelAnimationFrame(raf);
    const tick = () => {
      if (!host.isConnected || !visible || !(playing || scan.running)) return;
      if (playing) { for (let i = 0; i < perFrame; i++) doSweep(); draw(); }
      if (scan.running) scanStep();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  // ---- T-scan on its own 64 x 64 lattice ----
  const scan = {
    running: false, Ts: [] as number[], ms: [] as number[],
    L: lattice(64, 64, true), s: new Int8Array(0) as Int8Array, r: rng(2), i: 0, k: 0, acc: 0,
  };
  const SCAN_T = Array.from({ length: 31 }, (_, i) => Math.round((4 - 0.1 * i) * 10) / 10);
  scv.width = scv.height = 64;
  function toggleScan(): void {
    if (scan.running) { scan.running = false; syncButtons(); return; }
    scan.Ts = []; scan.ms = []; scan.i = 0; scan.k = 0; scan.acc = 0; scan.r = rng(2);
    scan.s = randomSpins(scan.L.n, scan.r);
    scan.running = true;
    scanDots.set([], []);
    syncButtons();
    loop();
  }
  function scanStep(): void {
    for (let n = 0; n < 40 && scan.running; n++) {
      const T = SCAN_T[scan.i];
      if (st.algo === "metropolis") metropolisSweep(scan.L, scan.s, 1, 0, T, scan.r);
      else gibbsSweep(scan.L, scan.s, 1, 0, T, scan.r);
      scan.k++;
      if (scan.k > 100) scan.acc += Math.abs(magnetisation(scan.s));
      if (scan.k === 300) {
        scan.Ts.push(T); scan.ms.push(scan.acc / 200);
        scan.k = 0; scan.acc = 0; scan.i++;
        if (scan.i >= SCAN_T.length) { scan.running = false; syncButtons(); }
      }
    }
    scanDots.set(scan.Ts, scan.ms);
    paint(scv, scan.s, 64);
  }

  // ---- drawing ----
  function paint(c: HTMLCanvasElement, sp: Int8Array, n: number): void {
    const ctx = c.getContext("2d")!;
    const id = ctx.createImageData(n, n), d = id.data;
    for (let p = 0; p < n * n; p++) {
      const up = sp[p] > 0;
      d[4 * p] = up ? 255 : 31; d[4 * p + 1] = up ? 127 : 119; d[4 * p + 2] = up ? 14 : 180; d[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  function drawReadout(): void {
    const m = tr.m[tr.m.length - 1], e = tr.e[tr.e.length - 1];
    const Tc = TC * st.J;
    ro.textContent = [
      `sweep ${sweep}  T = ${st.T.toFixed(2)}${cooling ? " (cooling)" : ""}`,
      `T/T_c = ${(st.T / Tc).toFixed(2)}`,
      `m = ${m.toFixed(3)}   E/n = ${e.toFixed(3)}`,
    ].join("\n");
    cap.textContent = `${st.size} × ${st.size}, periodic: orange +1, blue −1`;
  }

  function draw(): void {
    paint(cv, s, st.size);
    drawReadout();
    const k0 = tr.k[0], k1 = Math.max(k0 + TRACE, tr.k[tr.k.length - 1]);
    pm.setDomain([k1 - TRACE, k1]);
    pe.setDomain([k1 - TRACE, k1], [-2 * st.J - Math.abs(st.h) - 0.05, 0.1]);
    lm.set(tr.k, tr.m);
    le.set(tr.k, tr.e);
  }

  sSpeed.set(perFrame);
  load(init);
  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
  if (tab === "scan") syncButtons();
}

autoMount(NAME, mount);
