// The roofline: attainable operations per second against arithmetic intensity (operations per
// byte moved), for generic devices (order-of-magnitude numbers, lib/gpusim.ts), with image
// operations as points. Tab "end to end": upload + launches + kernels + download on the GPU
// against the same work on a CPU, over image size; the crossover is marked.
//
// <div data-demo="roofline" data-tab="roofline|e2e" data-preset="..." data-device="cpu|gpu|dc"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, Slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import * as G from "../lib/gpusim";

const NAME = "roofline";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .rf-main { flex: 1 1 640px; min-width: 0; }
${S} .qb-panel { flex: 0 1 330px; gap: .4em; }
${S} .rf-tabs { display: flex; gap: .3em; margin-bottom: .2em; }
${S} .rf-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .qb-readout { font-size: 15px; }
${S} .rf-hidden { display: none !important; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .rf-main { flex: 0 0 680px; }
.reveal ${S} .qb-panel { flex: 1 1 360px; gap: .2em; }
.reveal ${S} .qb-readout { font-size: 15px; columns: 2; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .rf-lab { font-size: 15px; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

type E2EOp = "scale" | "conv";
interface State {
  tab: "roofline" | "e2e"; dev: string; k: number; N: number; C: number; f16: boolean; cnn: boolean; sel: G.OpId;
  logSize: number; m: number; stay: boolean; e2eOp: E2EOp;
}
const DEFAULT: State = { tab: "roofline", dev: "gpu", k: 3, N: 1024, C: 64, f16: false, cnn: false, sel: "conv", logSize: 12, m: 1, stay: true, e2eOp: "conv" };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "add", label: "adding images: far left", s: { tab: "roofline", sel: "add" } },
  { id: "ridge", label: "k = 11 hits the ridge", s: { tab: "roofline", sel: "conv", k: 11 } },
  { id: "matmul", label: "matrix products live under the roof", s: { tab: "roofline", sel: "matmul", N: 1024 } },
  { id: "f16", label: "float16 moves the point right", s: { tab: "roofline", sel: "conv", k: 5, f16: true } },
  { id: "cnn", label: "a network layer: three roofs", s: { tab: "roofline", cnn: true, sel: "convlayer", C: 64 } },
  { id: "small", label: "one filter on a small image: transfer is everything", s: { tab: "e2e", e2eOp: "conv", k: 3, logSize: 7, m: 1 } },
  { id: "pipeline", label: "a 10-step pipeline that stays on the GPU", s: { tab: "e2e", e2eOp: "conv", k: 9, logSize: 11, m: 10, stay: true } },
];
const SHOWN: G.OpId[] = ["add", "pipe4", "fused4", "sep", "conv", "convnaive", "fft", "matmul", "convlayer"];
const label = (id: G.OpId) => G.OPS.find(o => o.id === id)!;

const fmtRate = (r: number) => r >= 1e12 ? `${+(r / 1e12).toPrecision(3)} T op/s` : `${+(r / 1e9).toPrecision(3)} G op/s`;
function fmtTime(t: number): string {
  if (t >= 1) return `${t.toPrecision(3)} s`;
  if (t >= 1e-3) return `${+(t * 1e3).toPrecision(3)} ms`;
  return `${+(t * 1e6).toPrecision(3)} µs`;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab === "e2e" ? { tab: "e2e" as const } : {}), ...(host.dataset.device ? { dev: host.dataset.device } : {}), ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "rf-main", "", host);
  const tabs = htmlEl("div", "rf-tabs", "", main);
  const tabBtns = [button(tabs, "roofline", () => { st.tab = "roofline"; draw(); }), button(tabs, "end to end", () => { st.tab = "e2e"; draw(); })];
  const rfBox = htmlEl("div", "", "", main), e2eBox = htmlEl("div", "", "", main);
  const plot = new Plot(rfBox, { width: 680, height: 400, xlog: true, ylog: true, x: [0.01, 3000], y: [0.01, 3000], xlabel: "arithmetic intensity (operations per byte)", ylabel: "attainable (10¹² op/s)" });
  const tplot = new Plot(e2eBox, { width: 680, height: 300, xlog: true, ylog: true, x: [64, 8192], y: [1e-6, 10], xlabel: "image side n (n × n float32 pixels)", ylabel: "time (s)", xticks: [64, 256, 1024, 4096], xfmt: v => String(v) });
  const bars = new Plot(e2eBox, { width: 680, height: 130, x: [0, 1], y: [0, 2], xlabel: "", grid: false, xticks: [], yticks: [], margin: { l: 70, r: 10, t: 6, b: 10 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  const rfCtl = htmlEl("div", "", "", panel), e2eCtl = htmlEl("div", "", "", panel);
  const presetRow = (box: HTMLElement, tab: State["tab"]) => {
    htmlEl("div", "rf-lab", "presets", box);
    return presets(box, PRESETS.filter(p => p.s.tab === tab).map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  };
  const pBtns = [...presetRow(rfCtl, "roofline"), ...presetRow(e2eCtl, "e2e")];
  htmlEl("div", "rf-lab", "device (order of magnitude)", rfCtl);
  const devBtns = presets(rfCtl, G.DEVICES.map(d => ({ label: d.label, apply: () => { st.dev = d.id; draw(); } })));
  htmlEl("div", "rf-lab", "operation (readout)", rfCtl);
  const opBtns = presets(rfCtl, SHOWN.map(id => ({ label: label(id).short, apply: () => { st.sel = id; st.cnn = false; draw(); } })));
  const kS = slider(panel, "filter size k", 1, 31, 2, st.k, v => { st.k = v; draw(); });
  const NS = slider(rfCtl, "matrix N", 6, 13, 1, Math.log2(st.N), v => { st.N = 2 ** v; draw(); }, v => String(2 ** v));
  const CS = slider(rfCtl, "channels C", 2, 9, 1, Math.log2(st.C), v => { st.C = 2 ** v; draw(); }, v => String(2 ** v));
  const f16C = checkbox(rfCtl, "float16 data (half the bytes)", st.f16, v => { st.f16 = v; draw(); });
  htmlEl("div", "rf-lab", "operation", e2eCtl);
  const e2eBtns = presets(e2eCtl, [{ label: "multiply by a constant", apply: () => { st.e2eOp = "scale"; draw(); } }, { label: "k×k filter", apply: () => { st.e2eOp = "conv"; draw(); } }]);
  const sizeS = slider(e2eCtl, "image side n", 6, 13, 1, st.logSize, v => { st.logSize = v; draw(); }, v => String(2 ** v));
  const mS = slider(e2eCtl, "chained operations", 1, 20, 1, st.m, v => { st.m = v; draw(); });
  const stayC = checkbox(e2eCtl, "data stays on the GPU between operations", st.stay, v => { st.stay = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", main);

  function sync(): void {
    const ss: [Slider, number][] = [[kS, st.k], [NS, Math.log2(st.N)], [CS, Math.log2(st.C)], [sizeS, st.logSize], [mS, st.m]];
    for (const [s, v] of ss) s.set(v);
    f16C.checked = st.f16; stayC.checked = st.stay;
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "roofline")));
    rfBox.classList.toggle("rf-hidden", st.tab !== "roofline"); rfCtl.classList.toggle("rf-hidden", st.tab !== "roofline");
    e2eBox.classList.toggle("rf-hidden", st.tab !== "e2e"); e2eCtl.classList.toggle("rf-hidden", st.tab !== "e2e");
    devBtns.forEach((b, i) => b.classList.toggle("active", G.DEVICES[i].id === st.dev));
    if (st.tab === "roofline") drawRoofline(); else drawE2E();
  }

  const P = 4096 * 4096;
  function cost(id: G.OpId, bytes: number): G.OpCost {
    return G.opCost(id, { k: st.k, N: st.N, C: st.C, pixels: id === "convlayer" ? 256 * 256 : P, bytes });
  }

  function drawRoofline(): void {
    opBtns.forEach((b, i) => b.classList.toggle("active", SHOWN[i] === st.sel));
    NS.input.closest("label")!.classList.toggle("rf-hidden", st.sel !== "matmul");
    CS.input.closest("label")!.classList.toggle("rf-hidden", st.sel !== "convlayer" && !st.cnn);
    const d = G.device(st.dev), T = 1e12;
    plot.clear();
    // roofs
    const roof = (peak: number, col: string, dash: string | undefined, lab: string, below = false) => {
      plot.fn(I => G.attainable(I, peak, d.bw) / T, { color: col, width: 3.5, dash });
      if (below) plot.text(2500, peak / T * 1.35, lab, { anchor: "end", color: col });
      else plot.text(0.012, peak / T * 1.3, lab, { anchor: "start", color: col });
    };
    roof(d.peak, "var(--qb-fg)", undefined, `float32 peak ${fmtRate(d.peak)}`);
    if (st.cnn && d.matrix) { roof(2 * d.peak, seriesColor(4), "8 5", "float16 lanes ×2", true); roof(d.matrix, seriesColor(2), "3 4", "matrix units"); }
    const I0 = G.ridge(d.peak, d.bw);
    plot.vline(I0, { text: `ridge ${I0.toPrecision(2)}`, color: "var(--qb-muted)", width: 1.5 });
    if (!st.cnn) plot.text(40, 0.02, `slope: bandwidth ${d.bw >= 1e12 ? d.bw / 1e12 + " TB/s" : d.bw / 1e9 + " GB/s"}`, { color: "var(--qb-muted)" });
    // points
    const pts: { id: G.OpId; I: number; rate: number; t: number; lab: string; col: string; c: G.OpCost; roofPeak: number }[] = [];
    const bytes = st.f16 ? 2 : 4;
    if (st.cnn) {
      const variants: [string, number, number, string][] = [["layer float32", 4, d.peak, seriesColor(0)], ["layer float16", 2, d.matrix ? 2 * d.peak : d.peak, seriesColor(4)], ["layer, matrix units", 2, d.matrix || d.peak, seriesColor(2)]];
      for (const [lab, b, pk, col] of variants) {
        const c = cost("convlayer", b), I = G.intensity(c);
        pts.push({ id: "convlayer", I, rate: G.attainable(I, pk, d.bw), t: G.kernelTime(c.W, c.Q, pk, d.bw), lab, col, c, roofPeak: pk });
      }
    } else {
      SHOWN.forEach((id, i) => {
        const c = cost(id, bytes), I = G.intensity(c);
        pts.push({ id, I, rate: G.attainable(I, d.peak, d.bw), t: G.kernelTime(c.W, c.Q, d.peak, d.bw), lab: label(id).short, col: id === st.sel ? seriesColor(1) : seriesColor(0), c, roofPeak: d.peak });
        void i;
      });
    }
    [...pts].sort((a, b) => a.I - b.I).forEach((p, i) => {
      plot.scatter([p.I], [p.rate / T], { color: p.col, r: p.id === st.sel ? 9 : 7 });
      if (st.cnn) return;
      const up = i % 2 === 0;
      plot.text(up ? p.I * 0.85 : p.I * 1.15, p.rate / T * (up ? 1.9 : 0.42), p.lab, { color: p.col, size: 17, anchor: up ? "end" : "start" });
    });
    if (st.cnn) plot.legend(pts.map(p => ({ label: p.lab, color: p.col, kind: "dot" as const })), "br");
    else plot.svg.querySelector(":scope > g.qb-legend")?.remove();
    const sel = st.cnn ? pts[2] : pts.find(p => p.id === st.sel)!;
    const bound = sel.I < sel.roofPeak / d.bw ? "memory-bound" : "compute-bound";
    const what = sel.id === "matmul" ? `N = ${st.N}` : sel.id === "convlayer" ? `C = ${st.C}, 256×256 pixels` : `4096×4096 image`;
    ro.textContent =
      `${label(sel.id).label}${st.cnn ? " (matrix units)" : ""}, ${what}${st.f16 && !st.cnn ? ", float16" : ""}\n` +
      `  work W         ${sel.c.W.toPrecision(3)} op\n` +
      `  traffic Q      ${sel.c.Q.toPrecision(3)} bytes\n` +
      `  intensity I    ${sel.I.toPrecision(3)} op/byte\n` +
      `  ${bound} (ridge ${(sel.roofPeak / d.bw).toPrecision(3)})\n` +
      `  attainable     ${fmtRate(sel.rate)} (${(100 * sel.rate / sel.roofPeak).toFixed(sel.rate / sel.roofPeak < 0.01 ? 2 : 0)} % of peak)\n` +
      `  time           ${fmtTime(sel.t)}`;
  }

  function drawE2E(): void {
    e2eBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.e2eOp === "scale")));
    const dev = G.device(st.dev === "cpu" ? "gpu" : st.dev), cpu = G.device("cpu");
    const id: G.OpId = st.e2eOp === "scale" ? "scale" : "conv";
    const gpuT = (n: number) => { const c = G.opCost(id, { k: st.k, pixels: n * n }); return G.endToEnd(dev, 4 * n * n, st.m, c.W, c.Q, st.stay); };
    const cpuT = (n: number) => { const c = G.opCost(id, { k: st.k, pixels: n * n }); return st.m * G.kernelTime(c.W, c.Q, cpu.peak, cpu.bw); };
    tplot.clear();
    tplot.fn(n => cpuT(n), { color: seriesColor(0), label: "laptop CPU" });
    tplot.fn(n => gpuT(n).total, { color: seriesColor(1), label: `${dev.label}, end to end` });
    tplot.fn(n => gpuT(n).compute, { color: seriesColor(1), dash: "6 5", width: 2, label: "GPU kernels only" });
    // crossover
    let cross = NaN;
    for (let e = 4; e <= 14; e += 0.01) { const n = 2 ** e; if (gpuT(n).total < cpuT(n)) { cross = n; break; } }
    const n = 2 ** st.logSize;
    tplot.vline(n, { color: "var(--qb-muted)", width: 1.5, text: `n = ${n}` });
    if (Number.isFinite(cross) && cross > 64) tplot.scatter([cross], [cpuT(cross)], { color: "var(--qb-c3)", r: 9 });
    tplot.legend(undefined, "tl");
    // stacked bars at the chosen size
    const g = gpuT(n), c = cpuT(n), tot = Math.max(g.total, c);
    bars.clear();
    bars.setDomain([0, tot * 1.02], [0, 2]);
    const parts: [number, string, string][] = [[g.upload, "#9467bd", "upload"], [g.launch, "#7f7f7f", "launch"], [g.compute, seriesColor(1), "kernels"], [g.download, "#c5b0d5", "download"]];
    let x = 0;
    for (const [v, col] of parts) { if (v > 0) bars.mark(gg => { const r = document.createElementNS("http://www.w3.org/2000/svg", "rect"); r.setAttribute("x", String(bars.x(x))); r.setAttribute("y", String(bars.y(1.85))); r.setAttribute("width", String(Math.max(0.5, bars.x(x + v) - bars.x(x)))); r.setAttribute("height", String(bars.y(1.15) - bars.y(1.85))); r.style.fill = col; gg.appendChild(r); }, []); x += v; }
    bars.mark(gg => { const r = document.createElementNS("http://www.w3.org/2000/svg", "rect"); r.setAttribute("x", String(bars.x(0))); r.setAttribute("y", String(bars.y(0.85))); r.setAttribute("width", String(Math.max(0.5, bars.x(c) - bars.x(0)))); r.setAttribute("height", String(bars.y(0.15) - bars.y(0.85))); r.style.fill = seriesColor(0); gg.appendChild(r); }, []);
    const tl = (y: number, s: string) => { const t = document.createElementNS("http://www.w3.org/2000/svg", "text"); t.setAttribute("x", "62"); t.setAttribute("y", String(bars.y(y) + 6)); t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "18"); t.textContent = s; bars.svg.appendChild(t); return t; };
    bars.svg.querySelectorAll(":scope > text.rf-bl").forEach(e => e.remove());
    tl(1.5, "GPU").classList.add("rf-bl"); tl(0.5, "CPU").classList.add("rf-bl");
    const what = id === "scale" ? "multiply by a constant" : `${st.k}×${st.k} filter`;
    ro.textContent =
      `${st.m} × ${what}, ${n}×${n} float32 (${(4 * n * n / 1e6).toPrecision(3)} MB)\n` +
      `GPU: upload   ${fmtTime(g.upload)}\n` +
      `     launch   ${fmtTime(g.launch)}\n` +
      `     kernels  ${fmtTime(g.compute)}\n` +
      `     download ${fmtTime(g.download)}\n` +
      `     total    ${fmtTime(g.total)}\n` +
      `CPU: total    ${fmtTime(c)}\n` +
      (g.total < c ? `GPU ${(c / g.total).toFixed(1)}× faster` : `CPU ${(g.total / c).toFixed(1)}× faster`) + "\n" +
      (Number.isFinite(cross) ? `GPU wins from n ≈ ${Math.round(cross)} (green dot)` : "the GPU never wins here");
  }

  sync();
  draw();
}

autoMount(NAME, mount);
