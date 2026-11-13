// pathtracer: a progressive path tracer in the browser (ray-tracing day, unit pathtrace/). A
// closed box (white floor, ceiling and back, red and green side walls, a square ceiling light)
// with a diffuse, a mirror and a glass ball. Every pass adds one path per pixel to a running
// mean (WebGL2; CPU fallback at 128 × 96). Target samples per pixel 1 … 4096; max bounces
// (1 = direct light only); Russian roulette; sampling uniform / cosine / cosine + light
// sampling; light size (smaller = more fireflies without light sampling); "Whitted" mode for
// comparison. Views: image, per-pixel standard deviation, error against a reference, normal,
// albedo, denoised. "Compute reference" renders 4096 samples per pixel with a different seed.
// Plot: RMS error against samples per pixel (log–log) with the 1/√N line. Denoise panel
// (CPU, on the displayed values): Gaussian, bilateral, non-local means, joint bilateral guided by
// normal and albedo; RMSE and PSNR of noisy and denoised against the reference.
// Drag to turn the camera a little. Options: data-preset, data-refspp (reference samples), data-cpu (force the CPU tracer).
import { htmlEl, slider, button, presets, checkbox, autoMount } from "../lib/ui";
import * as RT from "../lib/raytrace";
import { GLPathTracer, View } from "../lib/glpathtrace";
import { Plot, seriesColor } from "../lib/plot";
import { linearToSrgb8 } from "../lib/colour";
import { rmse, psnr } from "../lib/metrics";
import { Denoiser, denoise, toDisplayPlanes } from "./denoise";

const NAME = "pathtracer";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .pt-c1 { flex: 0 1 420px; min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
${S} .pt-c2, ${S} .pt-c3 { flex: 1 1 290px; max-width: 350px; display: flex; flex-direction: column; gap: .3em; }
${S} .pt-view { position: relative; width: 100%; aspect-ratio: 4 / 3; background: #111; border-radius: 4px; overflow: hidden; touch-action: none; cursor: grab; }
${S} .pt-view canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; }
${S} .pt-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-family: system-ui, sans-serif; white-space: pre; line-height: 1.3; }
${S} .pt-msg { font-size: .85em; color: var(--qb-c2); }
.reveal ${S} { flex-wrap: nowrap; font-size: 18px; gap: .45em; }
.reveal ${S} .pt-c1 { flex: 0 0 400px; }
.reveal ${S} .pt-c2 { flex: 0 0 340px; }
.reveal ${S} .pt-c3 { flex: 0 0 330px; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-presets { gap: .2em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

type ViewX = View | "denoised";
interface State {
  target: number; maxB: number; rr: boolean; sampling: RT.Sampling; whitted: boolean; light: number;
  diffuse: boolean; mirror: boolean; glass: boolean; view: ViewX; denoiser: Denoiser; strength: number;
  az: number; el: number; needRef: boolean;
}
const DEF: State = { target: 256, maxB: 5, rr: true, sampling: "light", whitted: false, light: 0.5, diffuse: true, mirror: true, glass: true, view: "image", denoiser: "none", strength: 2, az: 0, el: 0, needRef: false };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "one", label: "1 sample per pixel", s: { target: 1 } },
  { id: "noise", label: "16 → 64 → 256: noise halves", s: { target: 256, view: "std", needRef: true } },
  { id: "direct", label: "direct light only", s: { maxB: 1, target: 256 } },
  { id: "whitted", label: "Whitted vs path traced", s: { whitted: true, target: 64 } },
  { id: "uniform", label: "uniform vs cosine", s: { sampling: "uniform", target: 16 } },
  { id: "fireflies", label: "tiny light: fireflies", s: { sampling: "cosine", light: 0.1, target: 64 } },
  { id: "nee", label: "light sampling removes them", s: { sampling: "light", light: 0.1, target: 64 } },
  { id: "caustic", label: "glass ball caustic", s: { sampling: "light", mirror: false, diffuse: false, target: 1024 } },
  { id: "denoise", label: "denoised 16 vs 1024 spp", s: { target: 16, view: "denoised", denoiser: "joint", strength: 3, needRef: true } },
];
const TARGETS = [1, 4, 16, 64, 256, 1024, 4096];
const GW = 400, GH = 300, CW = 128, CH = 96;

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  let st: State = { ...DEF, ...(p0?.s ?? {}) };
  const REF_SPP = Math.max(16, Number(host.dataset.refspp ?? 4096));

  // ---- column 1: the image
  const c1 = htmlEl("div", "pt-c1", "", host);
  const view = htmlEl("div", "pt-view", "", c1);
  const glc = htmlEl("canvas", "", "", view);
  const over = htmlEl("canvas", "", "", view);
  const r1 = htmlEl("div", "pt-row", "", c1);
  const playB = button(r1, "pause", () => { playing = !playing; playB.textContent = playing ? "pause" : "play"; kick(); });
  button(r1, "restart", () => { restart(); });
  const sppL = htmlEl("span", "", "", r1); sppL.style.fontWeight = "600";
  const r2 = htmlEl("div", "pt-row", "", c1);
  const viewBtns = (["image", "std", "error", "normal", "albedo", "denoised"] as ViewX[]).map(v => ({ v, b: button(r2, v === "std" ? "std dev" : v, () => { st.view = v; if (v === "denoised") runDenoise(); show(); }) }));
  const msg = htmlEl("div", "pt-msg", "", c1);

  // ---- column 2: presets and the renderer
  const c2 = htmlEl("div", "pt-c2", "", host);
  const pb = presets(c2, [...PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...DEF, ...p.s }; sync(true); } })), { label: "reset", apply: () => { st = { ...DEF }; sync(true); pb.forEach(b => b.classList.remove("active")); } }]);
  const tS = slider(c2, "target spp", 0, 6, 1, TARGETS.indexOf(st.target), v => { st.target = TARGETS[v]; kick(); }, v => String(TARGETS[v]));
  const bS = slider(c2, "max bounces", 1, 8, 1, st.maxB, v => { st.maxB = v; restart(); });
  const lS = slider(c2, "light size", 0.05, 1, 0.05, st.light, v => { st.light = v; newScene(); });
  const sr = htmlEl("div", "pt-row", "", c2);
  const sampBtns = (["uniform", "cosine", "light"] as RT.Sampling[]).map(s => ({ s, b: button(sr, s === "light" ? "cosine + light" : s, () => { st.sampling = s; restart(); }) }));
  const cr = htmlEl("div", "pt-row", "", c2);
  const rrC = checkbox(cr, "roulette", st.rr, v => { st.rr = v; restart(); });
  const whC = checkbox(cr, "Whitted", st.whitted, v => { st.whitted = v; restart(); });
  const or = htmlEl("div", "pt-row", "", c2);
  const dC = checkbox(or, "diffuse", st.diffuse, v => { st.diffuse = v; newScene(); });
  const mC = checkbox(or, "mirror", st.mirror, v => { st.mirror = v; newScene(); });
  const gC = checkbox(or, "glass", st.glass, v => { st.glass = v; newScene(); });

  // ---- column 3: reference, error plot, denoising
  const c3 = htmlEl("div", "pt-c3", "", host);
  const refRow = htmlEl("div", "pt-row", "", c3);
  const refB = button(refRow, "compute reference", () => { startReference(); });
  const refL = htmlEl("span", "", "", refRow);
  const fig = htmlEl("div", "", "", c3);
  const plot = new Plot(fig, { width: 330, height: 230, x: [1, 4096], y: [1e-3, 1], xlog: true, ylog: true, xlabel: "samples per pixel", ylabel: "RMS error", font: 20, margin: { l: 74, b: 58, t: 8, r: 12 } });
  const eLine = plot.line([], [], { color: seriesColor(0), label: "measured" });
  const eRef = plot.line([], [], { color: "var(--qb-muted)", dash: "6 5", width: 2, label: "1/√N" });
  plot.legend(undefined, "bl");
  const dr = htmlEl("div", "pt-row", "", c3);
  const denBtns = (["none", "gaussian", "bilateral", "nlm", "joint"] as Denoiser[]).map(d => ({ d, b: button(dr, d === "nlm" ? "non-local means" : d === "joint" ? "joint bilateral" : d, () => { st.denoiser = d; st.view = "denoised"; runDenoise(); show(); }) }));
  const dS = slider(c3, "strength", 0.5, 6, 0.5, st.strength, v => { st.strength = v; if (st.view === "denoised") { runDenoise(); show(); } });
  const ro = htmlEl("div", "qb-readout", "", c3);

  // ---- renderer
  let gpt: GLPathTracer | null = null, cpu: RT.Progressive | null = null, W = GW, H = GH;
  let cpuRow = 0;
  let playing = true, raf = 0, visible = false, passes = 1, lastT = 0;
  let refMean: Float32Array | null = null, refKey = "", refRunning = false;
  const refCache = new Map<string, Float32Array>();
  let curve: [number, number][] = [];
  let denoised: { rgb: Uint8ClampedArray; r1: number; r2: number; p1: number; p2: number } | null = null;

  const camera = (): RT.Camera => {
    const d = 2.9, az = st.az, el = st.el;
    return { eye: [d * Math.sin(az) * Math.cos(el), 1 + d * Math.sin(el), d * Math.cos(az) * Math.cos(el)], target: [0, 1, 0], fov: 40 };
  };
  const sceneKey = () => [st.light, st.diffuse, st.mirror, st.glass, st.az.toFixed(3), st.el.toFixed(3)].join("|");
  const options = (): RT.PathOptions => ({ maxBounces: st.maxB, russianRoulette: st.rr, sampling: st.sampling, whitted: st.whitted });
  const scene = () => RT.boxScene({ lightSize: st.light, diffuse: st.diffuse, mirror: st.mirror, glass: st.glass });

  function init(): void {
    if (gpt || cpu) return;
    gpt = host.dataset.cpu ? null : GLPathTracer.create(glc, GW, GH);
    if (!gpt) {
      W = CW; H = CH;
      glc.style.display = "none";
      msg.textContent = "WebGL2 with float render targets is not available: CPU path tracer at 128 × 96 (slow, no reference).";
      cpu = new RT.Progressive(scene(), camera(), CW, CH, options(), 1);
      refB.disabled = true;
    } else if (gpt.floatType === "RGBA16F") msg.textContent = "Half-float accumulation (RGBA16F): long runs lose precision.";
    newScene();
  }

  function newScene(): void {
    if (!gpt && !cpu) return;
    const sc = scene(), cam = camera();
    if (gpt) { gpt.setScene(sc, cam); gpt.setOptions(options()); }
    else cpu = new RT.Progressive(sc, cam, CW, CH, options(), 1);
    const k = sceneKey();
    refMean = refCache.get(k) ?? null; refKey = k;
    if (gpt && refMean) gpt.uploadRefMean(refMean);
    else if (gpt) gpt.hasRef = false;
    restart();
  }

  function restart(): void {
    if (gpt) gpt.setOptions(options());
    else if (cpu) { cpu.opt = options(); cpu.reset(); cpuRow = 0; }
    curve = []; denoised = null;
    syncButtons(); kick();
  }

  function spp(): number { return gpt ? gpt.spp : cpu ? cpu.spp : 0; }

  function startReference(): void {
    if (!gpt) return;
    if (refCache.has(sceneKey())) { refMean = refCache.get(sceneKey())!; gpt.uploadRefMean(refMean); restart(); return; }
    gpt.resetReference(); refRunning = true; kick();
  }

  function measure(): void {
    if (!refMean || !gpt) return;
    const n = spp();
    if (n < 1 || (n & (n - 1)) !== 0 || curve.some(c => c[0] === n)) return;
    curve.push([n, rmse(gpt.readMean(), refMean)]);
  }

  function frame(t: number): void {
    raf = 0;
    if (!visible) return;
    const dt = lastT ? t - lastT : 16; lastT = t;
    if (gpt) {
      if (refRunning) {
        const k = Math.min(REF_SPP - gpt.refSpp, Math.max(1, passes - 1));
        gpt.stepReference(k);
        if (playing && gpt.spp < st.target) gpt.step(1);
        if (gpt.refSpp >= REF_SPP) {
          refRunning = false; refMean = gpt.readReference(); refCache.set(refKey, refMean); gpt.uploadRefMean(refMean); restart();
        }
      } else if (playing && gpt.spp < st.target) {
        // power-of-two sample counts are measured exactly
        let k = Math.min(st.target - gpt.spp, passes);
        if (refMean) { let nx = 1; while (nx <= gpt.spp) nx *= 2; k = Math.min(k, nx - gpt.spp); }
        gpt.step(k);
        measure();
        if (st.view === "denoised" && gpt.spp >= st.target) runDenoise();
      }
      adapt(dt);
    } else if (cpu && playing && cpu.spp < st.target) {
      // spread a sample over several frames (rows), about 20 ms of work per frame
      const t0 = performance.now();
      while (cpu.spp < st.target && performance.now() - t0 < 20) { const j1 = Math.min(CH, cpuRow + 8); cpu.addSample(cpuRow, j1); cpuRow = j1 >= CH ? 0 : j1; }
      if (st.view === "denoised" && cpu.spp >= st.target) runDenoise();
    }
    show();
    if ((playing && spp() < st.target) || refRunning) kick();
  }
  function adapt(dt: number): void {
    if (dt < 22) passes = Math.min(64, passes + 1);
    else if (dt > 40) passes = Math.max(1, Math.floor(passes / 2));
  }
  function kick(): void { if (!raf && visible) raf = requestAnimationFrame(frame); }

  let auxCache: { normal: Float32Array; albedo: Float32Array } | null = null;
  function aux(): { normal: Float32Array; albedo: Float32Array } {
    if (gpt) return gpt.readAux();
    if (!auxCache) { const a = RT.auxBuffers(scene(), camera(), CW, CH); auxCache = a; }
    return auxCache;
  }
  function mean(): Float32Array { return gpt ? gpt.readMean() : cpu!.mean(); }

  function runDenoise(): void {
    if (!gpt && !cpu) return;
    const m = mean(), n = W * H, planes = toDisplayPlanes(m, n);
    const out = denoise(st.denoiser, planes, W, H, st.strength, aux());
    const rgb = new Uint8ClampedArray(4 * n);
    for (let i = 0; i < n; i++) { rgb[4 * i] = 255 * out[0][i]; rgb[4 * i + 1] = 255 * out[1][i]; rgb[4 * i + 2] = 255 * out[2][i]; rgb[4 * i + 3] = 255; }
    let r1 = NaN, r2 = NaN, p1 = NaN, p2 = NaN;
    if (refMean) {
      const ref = toDisplayPlanes(refMean, n);
      const flat = (a: Float64Array[]) => { const f = new Float64Array(3 * n); for (let c = 0; c < 3; c++) f.set(a[c], c * n); return f; };
      const R = flat(ref), N0 = flat(planes), D0 = flat(out);
      r1 = rmse(N0, R); r2 = rmse(D0, R); p1 = psnr(N0, R, 1); p2 = psnr(D0, R, 1);
    }
    denoised = { rgb, r1, r2, p1, p2 };
  }

  function show(): void {
    syncButtons();
    const n = spp();
    sppL.textContent = refRunning && gpt ? `reference ${gpt.refSpp} / ${REF_SPP}` : `${n} spp`;
    over.style.display = "none";
    if (gpt) {
      if (st.view !== "denoised") gpt.display(st.view, st.view === "error" ? 6 : 12);
      else if (denoised) paintOver(denoised.rgb);
    } else if (cpu) {
      if (st.view === "denoised" && denoised) paintOver(denoised.rgb);
      else {
        const m = cpu.mean(), rgb = new Uint8ClampedArray(4 * CW * CH);
        const a = st.view === "normal" || st.view === "albedo" ? aux()[st.view] : null;
        for (let i = 0; i < CW * CH; i++) for (let c = 0; c < 3; c++) rgb[4 * i + c] = a ? 255 * a[3 * i + c] : linearToSrgb8(m[3 * i + c]);
        for (let i = 0; i < CW * CH; i++) rgb[4 * i + 3] = 255;
        paintOver(rgb);
      }
    }
    // plot
    if (curve.length) {
      const xs = curve.map(c => c[0]), ys = curve.map(c => c[1]);
      const lo = Math.min(...ys) / 3, hi = Math.max(...ys) * 2;
      plot.setDomain([1, 4096], [Math.max(1e-4, lo), hi]);
      eLine.set(xs, ys);
      eRef.set(TARGETS, TARGETS.map(N => ys[0] * Math.sqrt(xs[0] / N)));
    } else { eLine.set([], []); eRef.set([], []); }
    refL.textContent = refMean ? "reference ready" : refRunning ? "rendering…" : "";
    const L: string[] = [];
    if (refMean && curve.length) { const c = curve[curve.length - 1]; L.push(`RMS error at ${c[0]} spp: ${c[1].toFixed(4)}`); }
    if (st.view === "denoised" && denoised) {
      if (refMean) { L.push(`noisy:    RMSE ${denoised.r1.toFixed(4)}, PSNR ${denoised.p1.toFixed(1)} dB`); L.push(`denoised: RMSE ${denoised.r2.toFixed(4)}, PSNR ${denoised.p2.toFixed(1)} dB`); }
      else L.push("compute the reference to score the denoiser");
    }
    if (!L.length) L.push(refMean ? "error view: |mean − reference|" : "no reference yet");
    ro.textContent = L.join("\n");
  }
  function paintOver(rgba: Uint8ClampedArray): void {
    over.style.display = "block"; over.width = W; over.height = H;
    over.getContext("2d")!.putImageData(new ImageData(rgba as unknown as Uint8ClampedArray<ArrayBuffer>, W, H), 0, 0);
  }

  function syncButtons(): void {
    viewBtns.forEach(({ v, b }) => b.classList.toggle("active", v === st.view));
    sampBtns.forEach(({ s, b }) => b.classList.toggle("active", s === st.sampling));
    denBtns.forEach(({ d, b }) => b.classList.toggle("active", d === st.denoiser));
  }

  function sync(scene_: boolean): void {
    tS.set(TARGETS.indexOf(st.target)); bS.set(st.maxB); lS.set(st.light); dS.set(st.strength);
    rrC.checked = st.rr; whC.checked = st.whitted; dC.checked = st.diffuse; mC.checked = st.mirror; gC.checked = st.glass;
    auxCache = null;
    if (scene_) newScene(); else restart();
    if (st.needRef && gpt && !refCache.has(sceneKey())) startReference();
  }

  // drag to turn the camera a little (it stays inside the box)
  let drag: { x: number; y: number; az: number; el: number } | null = null;
  view.addEventListener("pointerdown", ev => { drag = { x: ev.clientX, y: ev.clientY, az: st.az, el: st.el }; view.setPointerCapture(ev.pointerId); ev.preventDefault(); });
  view.addEventListener("pointermove", ev => {
    if (!drag) return;
    const r = view.getBoundingClientRect();
    st.az = Math.max(-0.35, Math.min(0.35, drag.az - (ev.clientX - drag.x) / r.width));
    st.el = Math.max(-0.2, Math.min(0.25, drag.el + (ev.clientY - drag.y) / r.height * 0.6));
    auxCache = null; newScene();
  });
  const up = () => { drag = null; };
  view.addEventListener("pointerup", up); view.addEventListener("pointercancel", up);

  // run only while on screen; the WebGL context is created when first visible
  const start = () => {
    visible = true;
    if (!gpt && !cpu) { init(); if (st.needRef && gpt) startReference(); }
    kick();
  };
  if (typeof IntersectionObserver === "undefined") start();
  else new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) start(); else visible = false; }, { rootMargin: "50px" }).observe(view);

  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  tS.set(TARGETS.indexOf(st.target)); bS.set(st.maxB); lS.set(st.light); dS.set(st.strength);
  rrC.checked = st.rr; whC.checked = st.whitted; dC.checked = st.diffuse; mC.checked = st.mirror; gC.checked = st.glass;
  syncButtons();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
