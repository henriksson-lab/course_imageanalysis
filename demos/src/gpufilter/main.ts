// The same filter on the CPU and on the GPU, measured in this browser. A direct k x k box or
// Gaussian filter (replicate border) on an n x n float32 image: on the CPU with
// lib/imageops.ts correlate2d (TypeScript, one thread), on the GPU with a fragment shader that
// runs once per output pixel (lib/glcompute.ts), the result read back so the timing is honest.
// Tab "race": run both, two times and the ratio. Tab "breakdown": GPU time split into upload,
// compute and download; time against image size for both (log axes), built up as sizes are
// measured; agreement of the two results (float32, or float16 textures).
// Without WebGL2 float render targets: a message and a recorded table.
//
// <div data-demo="gpufilter" data-tab="race|breakdown" data-preset="big|small|k31|f16"
//      data-size="2048" data-k="9"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, Slider } from "../lib/ui";
import { Grey, correlate2d, resize, gaussianKernel, cropGrey, drawGrey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Plot, seriesColor } from "../lib/plot";
import { ComputeContext, Target, median, FloatPrecision } from "../lib/glcompute";

const NAME = "gpufilter";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { flex-wrap: nowrap; gap: 1em; }
${S} .gf-main { flex: 1 1 640px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
${S} .qb-panel { flex: 0 1 330px; gap: .4em; }
${S} .gf-tabs { display: flex; gap: .3em; }
${S} .gf-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
${S} .gf-note { font-size: 14px; color: var(--qb-muted); }
${S} .qb-readout { font-size: 15px; }
${S} .gf-hidden { display: none !important; }
${S} .gf-race { display: grid; grid-template-columns: 1fr 1fr; gap: .6em; }
${S} .gf-card { background: var(--qb-panel); border-radius: 8px; padding: .5em .7em; }
${S} .gf-card .gf-t { font-size: 34px; font-weight: 700; font-family: ui-monospace, monospace; }
${S} .gf-ratio { font-size: 24px; font-weight: 600; text-align: center; }
${S} .gf-imgs { display: flex; gap: .6em; align-items: flex-start; }
${S} .gf-imgs figure { margin: 0; flex: 0 1 220px; }
${S} .gf-imgs canvas { width: 100%; image-rendering: pixelated; display: block; border-radius: 3px; }
${S} .gf-imgs figcaption { font-size: 15px; text-align: center; }
${S} .gf-msg { padding: .6em; border: 2px solid var(--qb-c2); border-radius: 6px; background: var(--qb-panel); }
${S} table.gf-rec { border-collapse: collapse; font-size: 15px; }
${S} table.gf-rec td, ${S} table.gf-rec th { border-bottom: 1px solid var(--qb-grid); padding: .1em .5em; text-align: right; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 0 0 340px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .4em; }
.reveal ${S} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${S} .gf-lab, .reveal ${S} .gf-note { font-size: 15px; }
.reveal ${S} .gf-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_src;
uniform float u_w[31];
uniform int u_r;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 hi = textureSize(u_src, 0) - 1;
  float s = 0.0;
  for (int dy = -u_r; dy <= u_r; dy++) {
    float wy = u_w[dy + u_r];
    for (int dx = -u_r; dx <= u_r; dx++) {
      s += wy * u_w[dx + u_r] * texelFetch(u_src, clamp(p + ivec2(dx, dy), ivec2(0), hi), 0).r;
    }
  }
  o = vec4(s, 0.0, 0.0, 1.0);
}`;
const TOUCH = `#version 300 es
precision highp float;
uniform highp sampler2D u_src;
out vec4 o;
void main() { o = vec4(texelFetch(u_src, ivec2(0), 0).r, 0.0, 0.0, 1.0); }`;

type Kind = "box" | "gauss";
interface State { tab: "race" | "breakdown"; n: number; k: number; kind: Kind; f16: boolean }
const DEFAULT: State = { tab: "race", n: 2048, k: 9, kind: "box", f16: false };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "big", label: "2048², 9×9: the GPU wins", s: { n: 2048, k: 9 } },
  { id: "small", label: "128², 3×3: no gain", s: { n: 128, k: 3 } },
  { id: "k31", label: "1024², 31×31: lots of arithmetic", s: { n: 1024, k: 31 } },
  { id: "f16", label: "float16 textures: small differences", s: { n: 512, k: 9, f16: true, tab: "breakdown" } },
];
const SIZES = [128, 256, 512, 1024, 2048];
const CPU_CAP = 4e8;   // multiply-adds above which the CPU time is extrapolated, not run

/** Recorded on the author's computer (see the label in the table); shown without WebGL2. */
const RECORDED: { n: number; k: number; cpu: number; gpu: number }[] = [
  { n: 128, k: 3, cpu: 1.6, gpu: 1.9 }, { n: 128, k: 9, cpu: 13.7, gpu: 7.3 }, { n: 256, k: 9, cpu: 25.4, gpu: 8.5 },
  { n: 512, k: 9, cpu: 79, gpu: 14.5 }, { n: 1024, k: 9, cpu: 328, gpu: 47.6 }, { n: 2048, k: 9, cpu: 1230, gpu: 145 },
  { n: 2048, k: 3, cpu: 220, gpu: 132 }, { n: 1024, k: 31, cpu: 4514, gpu: 76.5 },
];
const RECORDED_ON = "a workstation with a 2019-generation desktop GPU, box filter, float32; 2048² 9×9 split as upload 25 ms + compute 14 ms + download 106 ms; the 1024² 31×31 CPU time is extrapolated";

interface Meas { cpu: number; cpuExtrap: boolean; gpu: number; up: number; comp: number; down: number; diff: number | null }

function fmtMs(t: number): string { return t >= 100 ? t.toFixed(0) + " ms" : t >= 10 ? t.toFixed(1) + " ms" : t.toFixed(2) + " ms"; }

function kernel1d(kind: Kind, k: number): Float64Array {
  if (kind === "box") return new Float64Array(k).fill(1 / k);
  return gaussianKernel(Math.max(0.5, k / 6), 0, (k - 1) / 2);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => {
    const s: State = { ...DEFAULT, ...(host.dataset.tab === "breakdown" ? { tab: "breakdown" as const } : {}), ...(p0?.s ?? {}) };
    if (host.dataset.size && SIZES.includes(Number(host.dataset.size))) s.n = Number(host.dataset.size);
    if (host.dataset.k) s.k = Math.max(1, Math.min(31, Number(host.dataset.k) | 1));
    return s;
  };
  const st = start();

  const main = htmlEl("div", "gf-main", "", host);
  const tabs = htmlEl("div", "gf-tabs", "", main);
  const tabBtns = [button(tabs, "race", () => { st.tab = "race"; draw(); }), button(tabs, "breakdown", () => { st.tab = "breakdown"; draw(); })];
  const msg = htmlEl("div", "gf-msg gf-hidden", "", main);
  const raceBox = htmlEl("div", "", "", main);
  const race = htmlEl("div", "gf-race", "", raceBox);
  const cpuCard = htmlEl("div", "gf-card", "", race), gpuCard = htmlEl("div", "gf-card", "", race);
  htmlEl("div", "", "CPU (one thread of the browser)", cpuCard);
  const cpuT = htmlEl("div", "gf-t", "–", cpuCard);
  const cpuSub = htmlEl("div", "gf-note", "", cpuCard);
  htmlEl("div", "", "GPU (fragment shader, read back)", gpuCard);
  const gpuT = htmlEl("div", "gf-t", "–", gpuCard);
  const gpuSub = htmlEl("div", "gf-note", "", gpuCard);
  const ratio = htmlEl("div", "gf-ratio", "", raceBox);
  const imgs = htmlEl("div", "gf-imgs", "", raceBox);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", imgs); const c = htmlEl("canvas", "", "", f); htmlEl("figcaption", "", cap, f); return c; };
  const inC = fig("input (256² shown)"), outC = fig("GPU output (top-left 256²)");
  outC.width = outC.height = 256;
  const brBox = htmlEl("div", "", "", main);
  const plot = new Plot(brBox, { width: 640, height: 300, xlog: true, ylog: true, x: [100, 2600], y: [0.01, 5000], xlabel: "image side n (n × n pixels)", ylabel: "time (ms)", xticks: SIZES, xfmt: v => String(v) });
  const bars = new Plot(brBox, { width: 640, height: 110, x: [0, 1], y: [0, 1], grid: false, xticks: [], yticks: [], margin: { l: 10, r: 10, t: 6, b: 8 } });
  htmlEl("div", "gf-note", "Measured on this computer; your numbers will differ. CPU = one thread of the browser, not an optimised multi-core program: a tuned CPU code is 10–50× faster.", main);
  htmlEl("div", "gf-note gf-credit", CELL_CREDIT, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "gf-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  htmlEl("div", "gf-lab", "image size n × n", panel);
  const sizeBtns = presets(panel, SIZES.map(n => ({ label: String(n), apply: () => { st.n = n; draw(); } })));
  const kS = slider(panel, "kernel k × k", 1, 31, 2, st.k, v => { st.k = v; draw(); });
  const kindBtns = presets(panel, [{ label: "box", apply: () => { st.kind = "box"; draw(); } }, { label: "Gaussian", apply: () => { st.kind = "gauss"; draw(); } }]);
  const f16C = checkbox(panel, "float16 textures on the GPU", st.f16, v => { st.f16 = v; draw(); });
  const runRow = htmlEl("div", "qb-presets", "", panel);
  const runB = button(runRow, "run both", () => runOne(st.n));
  const allB = button(runRow, "measure all sizes", () => runAll());
  button(panel, "reset", () => { Object.assign(st, start()); results.clear(); pBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  const cc = new ComputeContext(host);
  const results = new Map<string, Meas>();
  const key = (n: number) => `${st.kind}|${st.k}|${st.f16 ? 16 : 32}|${n}`;
  let base: Grey | null = null, busy = false;
  const images = new Map<number, Grey>();

  function sync(): void { kS.set(st.k); f16C.checked = st.f16; }
  function image(n: number): Grey {
    let g = images.get(n);
    if (!g) { g = n === 256 ? base! : resize(base!, n, n, n < 256 ? "area" : "bilinear"); images.set(n, g); }
    return g;
  }

  function measureCPU(img: Grey, w1: Float64Array): { t: number; extrap: boolean; out: Grey | null } {
    const k = w1.length, k2 = new Float64Array(k * k);
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) k2[i * k + j] = w1[i] * w1[j];
    const n = img.width, work = n * n * k * k;
    if (work > CPU_CAP) {
      const m = 256, crop = cropGrey(img, 0, 0, m, m);
      const t0 = performance.now(); correlate2d(crop, k2, k, k, "replicate"); const t1 = performance.now();
      return { t: (t1 - t0) * (n * n) / (m * m), extrap: true, out: null };
    }
    correlate2d(cropGrey(img, 0, 0, Math.min(n, 64), Math.min(n, 64)), k2, k, k, "replicate"); // warm-up (JIT)
    const runs = work > 5e7 ? 1 : 3, ts: number[] = [];
    let out: Grey | null = null;
    for (let r = 0; r < runs; r++) { const t0 = performance.now(); out = correlate2d(img, k2, k, k, "replicate"); ts.push(performance.now() - t0); }
    return { t: median(ts), extrap: false, out };
  }

  function measureGPU(img: Grey, w1: Float64Array, prec: FloatPrecision): { gpu: number; up: number; comp: number; down: number; out: Float32Array } | null {
    const gl = cc.get();
    if (!gl) return null;
    const n = img.width, prog = cc.program(FRAG), touch = cc.program(TOUCH);
    const w = new Float32Array(31); w.set(w1);
    const r = (w1.length - 1) / 2;
    const dst = cc.target(n, n, prec), tiny: Target = cc.target(1, 1, "f32");
    const ups: number[] = [], comps: number[] = [], downs: number[] = [];
    let out: Float32Array = new Float32Array(0);
    const runs = n * n * w1.length ** 2 > 2e9 ? 2 : 5;
    for (let it = 0; it <= runs; it++) {            // run 0 = warm-up
      const t0 = performance.now();
      const src = cc.texture(img.data, n, n, prec);
      cc.run(touch, tiny, { u_src: src }); cc.finish(tiny);
      const t1 = performance.now();
      cc.run(prog, dst, { u_src: src }, (g, p) => { g.uniform1fv(g.getUniformLocation(p, "u_w"), w); g.uniform1i(g.getUniformLocation(p, "u_r"), r); });
      cc.finish(dst);
      const t2 = performance.now();
      out = cc.read(dst);
      const t3 = performance.now();
      cc.free(src);
      if (it > 0) { ups.push(t1 - t0); comps.push(t2 - t1); downs.push(t3 - t2); }
    }
    cc.free(dst); cc.free(tiny);
    const up = median(ups), comp = median(comps), down = median(downs);
    return { gpu: up + comp + down, up, comp, down, out };
  }

  function runOne(n: number): void {
    if (busy || !base) return;
    busy = true; runB.disabled = allB.disabled = true;
    ro.textContent = `measuring ${n}×${n}, ${st.k}×${st.k} …`;
    window.setTimeout(() => {
      try {
        const img = image(n), w1 = kernel1d(st.kind, st.k);
        const g = measureGPU(img, w1, st.f16 ? "f16" : "f32");
        const c = measureCPU(img, w1);
        if (!g) { showFallback(); return; }
        let diff: number | null = null;
        if (c.out) { diff = 0; for (let i = 0; i < g.out.length; i++) diff = Math.max(diff, Math.abs(g.out[i] - c.out.data[i])); }
        results.set(key(n), { cpu: c.t, cpuExtrap: c.extrap, gpu: g.gpu, up: g.up, comp: g.comp, down: g.down, diff });
        if (n === st.n) showOutput(g.out, n);
      } catch (e) {
        ro.textContent = "GPU error: " + (e as Error).message;
      } finally {
        busy = false; runB.disabled = allB.disabled = false;
        draw();
      }
    }, 30);
  }

  function runAll(): void {
    const todo = [...SIZES];
    const next = () => {
      if (busy) { window.setTimeout(next, 60); return; }
      const n = todo.shift();
      if (n === undefined) return;
      runOne(n); window.setTimeout(next, 60);
    };
    next();
  }

  function showOutput(out: Float32Array, n: number): void {
    const g: Grey = { width: n, height: n, data: out };
    const shown = n > 256 ? cropGrey(g, 0, 0, 256, 256) : g;
    drawGrey(outC, shown, { mode: "clip", lo: 0, hi: 255 });
  }

  function showFallback(): void {
    msg.classList.remove("gf-hidden");
    let html = `<b>${cc.error ?? "WebGL2 is not available."}</b>`;
    if (RECORDED.length) {
      html += `<br>Recorded instead (${RECORDED_ON}; not this computer):<table class="gf-rec"><tr><th>n</th><th>k</th><th>CPU</th><th>GPU</th><th>ratio</th></tr>` +
        RECORDED.map(r => `<tr><td>${r.n}</td><td>${r.k}</td><td>${fmtMs(r.cpu)}</td><td>${fmtMs(r.gpu)}</td><td>${(r.cpu / r.gpu).toFixed(1)}×</td></tr>`).join("") + "</table>";
    }
    msg.innerHTML = html;
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "race")));
    raceBox.classList.toggle("gf-hidden", st.tab !== "race");
    brBox.classList.toggle("gf-hidden", st.tab !== "breakdown");
    sizeBtns.forEach((b, i) => b.classList.toggle("active", SIZES[i] === st.n));
    kindBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.kind === "box")));
    const m = results.get(key(st.n));
    const work = st.n * st.n * st.k * st.k;
    cpuT.textContent = m ? fmtMs(m.cpu) + (m.cpuExtrap ? "*" : "") : "–";
    cpuSub.textContent = m?.cpuExtrap ? `* extrapolated from a 256² crop (${(work / 1e6).toFixed(0)} M multiply–adds is too slow to run)` : `${(work / 1e6).toFixed(1)} M multiply–adds`;
    gpuT.textContent = m ? fmtMs(m.gpu) : "–";
    gpuSub.textContent = m ? `upload ${fmtMs(m.up)} + compute ${fmtMs(m.comp)} + download ${fmtMs(m.down)}` : "press “run both”";
    ratio.textContent = m ? (m.gpu < m.cpu ? `GPU ${(m.cpu / m.gpu).toFixed(1)}× faster` : `CPU ${(m.gpu / m.cpu).toFixed(1)}× faster`) : "";
    // breakdown plot from all results with this kernel
    plot.clear();
    const pts = SIZES.map(n => [n, results.get(key(n))] as const).filter(([, r]) => r) as [number, Meas][];
    if (pts.length) {
      plot.line(pts.map(p => p[0]), pts.map(p => p[1].cpu), { color: seriesColor(0), label: "CPU" });
      plot.scatter(pts.map(p => p[0]), pts.map(p => p[1].cpu), { color: seriesColor(0), colors: pts.map(p => p[1].cpuExtrap ? "#fff" : seriesColor(0)), stroke: seriesColor(0) });
      plot.line(pts.map(p => p[0]), pts.map(p => p[1].gpu), { color: seriesColor(1), label: "GPU total" });
      plot.scatter(pts.map(p => p[0]), pts.map(p => p[1].gpu), { color: seriesColor(1) });
      plot.line(pts.map(p => p[0]), pts.map(p => p[1].comp), { color: seriesColor(1), dash: "6 5", width: 2, label: "GPU compute only" });
      plot.legend(undefined, "tl");
    } else plot.text(140, 1, "press “measure all sizes”", { size: 18 });
    bars.clear();
    if (m) {
      const tot = Math.max(m.gpu, 1e-9);
      bars.setDomain([0, tot], [0, 1]);
      let x = 0;
      const parts: [number, string, string][] = [[m.up, "#9467bd", "upload"], [m.comp, seriesColor(1), "compute"], [m.down, "#c5b0d5", "download"]];
      for (const [v, col, lab] of parts) {
        bars.band([x, x + v], [0.35, 0.35], [0.95, 0.95], { color: col, fillOpacity: 0.9 });
        if (v / tot > 0.12) bars.text(x + v / 2, 0.1, `${lab} ${fmtMs(v)}`, { anchor: "middle", size: 17 });
        x += v;
      }
    }
    const cross = pts.find(([, r]) => r.gpu < r.cpu);
    ro.textContent =
      `${st.n}×${st.n}, ${st.kind === "box" ? "box" : "Gaussian"} ${st.k}×${st.k}, ${st.f16 ? "float16" : "float32"} textures\n` +
      (m ? (m.diff === null ? "agreement: not checked (CPU extrapolated)\n" : `max |GPU − CPU| = ${m.diff.toPrecision(2)} (pixel values 0–255)\n`) : "") +
      (pts.length > 1 ? (cross ? `GPU faster from n = ${cross[0]} (of the sizes measured)` : "GPU not faster at any size measured") : "");
  }

  sync();
  draw();
  loadCell(256).then(g => {
    base = g;
    drawGrey(inC, g, { mode: "clip", lo: 0, hi: 255 });
    if (!cc.get()) showFallback();
  });
}

autoMount(NAME, mount);
