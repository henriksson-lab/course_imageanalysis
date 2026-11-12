// Bayesian denoising of a binary image with an Ising (MRF) prior. A 128 x 128 binary image
// (test shapes, or the micrograph thresholded), flip noise eps or Gaussian noise sigma_n; the
// prior weight lambda (= 2J). Methods: threshold, ICM, Gibbs sampling at T = 1 (posterior
// marginals -> MPM), simulated annealing, mean field, and the exact MAP by one minimum cut.
// Panels: noisy input, current estimate, errors against the truth (or the marginal p(object));
// energy against sweeps for every method run so far, with the exact minimum dashed.
// Auto-mounts on <div data-demo="mrfdenoise">. Options: data-preset="icm|slow|quench|mpm|large|zero|notbest|gauss".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import {
  N, PRESETS, Preset, Problem, Run, Method, METHODS, METHOD_LABEL, Noise, ImageKind, RunOptions, DEFAULT_RUN,
  shapesTruth, cellTruth, makeProblem, wrong,
} from "./model";

const NAME = "mrfdenoise";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; white-space: nowrap; }
[data-demo="${NAME}"] .md-left { flex: 1 1 560px; max-width: 640px; min-width: 280px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .md-imgs { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .md-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .md-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .md-tabs .qb-btn { font-size: 13px; padding: .05em .45em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .12em; }
.reveal [data-demo="${NAME}"] .md-left { flex-basis: 600px; max-width: 600px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .5em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COOL = [10, 30, 100, 300, 1000, 3000];
const COLOR: Record<Method, string> = {
  threshold: seriesColor(9), icm: seriesColor(1), gibbs: seriesColor(2), anneal: seriesColor(3), meanfield: seriesColor(4), exact: seriesColor(0),
};

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    image: p0.image as ImageKind, noise: { ...p0.noise } as Noise, lambda: p0.lambda,
    method: p0.method as Method, run: { ...p0.run } as RunOptions, third: "errors" as "truth" | "errors" | "marginal",
  };
  let P: Problem | null = null;
  let runs = new Map<Method, Run>();
  let playing = false, perFrame = 4;
  let cellGrey: Float64Array | null = null;

  // ---- layout ----
  const left = htmlEl("div", "md-left", "", host);
  const imgs = htmlEl("div", "md-imgs", "", left);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", imgs); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "", "", f); cv.width = cv.height = N; return { c, cv, f }; };
  const fIn = fig("noisy input"), fEst = fig("estimate"), fErr = fig("");
  const tabs = htmlEl("div", "qb-presets md-tabs", "", fErr.c);
  fErr.c.style.display = "flex"; fErr.c.style.justifyContent = "center";
  const thirdBtns = [button(tabs, "truth", () => { st.third = "truth"; draw(); }), button(tabs, "errors", () => { st.third = "errors"; draw(); }), button(tabs, "p(object)", () => { st.third = "marginal"; draw(); })];
  const plot = new Plot(left, { width: 600, height: 250, x: [1, 3000], xlog: true, y: [0, 1], xlabel: "sweep", ylabel: "energy", margin: { l: 90 }, yfmt: v => v.toFixed(0) });
  const credit = htmlEl("div", "md-credit", CELL_CREDIT, left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const rowI = htmlEl("div", "qb-presets", "", panel);
  const imgBtns = (["shapes", "cell"] as ImageKind[]).map(k => button(rowI, k === "shapes" ? "test shapes" : "micrograph", () => { st.image = k; rebuild(); }));
  const nzBtns = (["flip", "gauss"] as const).map(k => button(rowI, k === "flip" ? "flip noise" : "Gaussian", () => { st.noise.kind = k; syncNoise(); rebuild(); }));
  const sEps = slider(panel, "ε", 0.02, 0.45, 0.01, st.noise.eps, v => { st.noise.eps = v; rebuild(); });
  const sSd = slider(panel, "σn", 0.1, 1.5, 0.05, st.noise.sd, v => { st.noise.sd = v; rebuild(); });
  const sLam = slider(panel, "λ", 0, 4, 0.05, st.lambda, v => { st.lambda = v; rebuild(); });
  htmlEl("div", "md-lab", "method", panel);
  const rowM = htmlEl("div", "qb-presets", "", panel);
  const methodBtns = METHODS.map(m => button(rowM, METHOD_LABEL[m], () => selectMethod(m)));
  const sT0 = slider(panel, "T₀", 1, 8, 0.5, st.run.T0, v => { st.run.T0 = v; restartMethod(); });
  const sCool = slider(panel, "cooling sweeps", 0, COOL.length - 1, 1, COOL.indexOf(st.run.coolSweeps), v => { st.run.coolSweeps = COOL[v]; restartMethod(); }, v => String(COOL[v]));
  const rowP = htmlEl("div", "qb-presets", "", panel);
  const playBtn = button(rowP, "play", () => { playing = !playing; sync(); if (playing) loop(); });
  button(rowP, "1 sweep", () => { cur()?.step(); draw(); });
  button(rowP, "restart", () => { restartMethod(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "md-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => { load(p0); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });

  const cur = () => runs.get(st.method);

  function syncNoise(): void {
    (sEps.input.parentElement as HTMLElement).style.display = st.noise.kind === "flip" ? "" : "none";
    (sSd.input.parentElement as HTMLElement).style.display = st.noise.kind === "gauss" ? "" : "none";
  }
  function sync(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["shapes", "cell"][i] === st.image));
    nzBtns.forEach((b, i) => b.classList.toggle("active", ["flip", "gauss"][i] === st.noise.kind));
    methodBtns.forEach((b, i) => b.classList.toggle("active", METHODS[i] === st.method));
    const ann = st.method === "anneal";
    (sT0.input.parentElement as HTMLElement).style.display = ann ? "" : "none";
    (sCool.input.parentElement as HTMLElement).style.display = ann ? "" : "none";
    playBtn.textContent = playing ? "pause" : "play";
    const hasMarg = st.method === "gibbs" || st.method === "meanfield";
    thirdBtns[2].disabled = !hasMarg;
    thirdBtns[2].style.opacity = hasMarg ? "1" : "0.4";
    const third = st.third === "marginal" && !hasMarg ? "errors" : st.third;
    thirdBtns.forEach((b, i) => b.classList.toggle("active", ["truth", "errors", "marginal"][i] === third));
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    syncNoise();
  }

  function load(p: Preset): void {
    st.image = p.image; st.noise = { ...p.noise }; st.lambda = p.lambda; st.method = p.method; st.run = { ...p.run };
    sEps.set(st.noise.eps); sSd.set(st.noise.sd); sLam.set(st.lambda); sT0.set(st.run.T0); sCool.set(COOL.indexOf(st.run.coolSweeps));
    rebuild(() => {
      for (const m of p.also ?? []) { const R = new Run(m, P!, st.run); R.finish(); runs.set(m, R); }
      selectMethod(p.method);
      playing = true; sync(); loop();
    });
  }

  // ---- problem and runs ----
  let pending = false;
  function rebuild(then?: () => void): void {
    sync();
    if (st.image === "cell" && !cellGrey) {
      loadCell(N).then(g => { cellGrey = Float64Array.from(g.data); rebuild(then); });
      return;
    }
    if (pending && !then) return;
    pending = true;
    // coalesce slider drags: the min-cut (~50 ms) runs once per frame at most
    requestAnimationFrame(() => {
      pending = false;
      const truth = st.image === "cell" ? cellTruth(cellGrey!) : shapesTruth();
      P = makeProblem(truth, st.noise, st.lambda);
      runs = new Map();
      runs.set(st.method, new Run(st.method, P, st.run));
      if (then) then(); else draw();
    });
  }

  function selectMethod(m: Method): void {
    st.method = m;
    if (P && !runs.has(m)) runs.set(m, new Run(m, P, st.run));
    draw();
  }

  function restartMethod(): void {
    if (!P) return;
    runs.set(st.method, new Run(st.method, P, st.run));
    draw();
  }

  // ---- animation ----
  let visible = true, raf = 0;
  new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && playing) loop(); }).observe(host);
  function loop(): void {
    cancelAnimationFrame(raf);
    const tick = () => {
      if (!host.isConnected || !visible || !playing) return;
      const R = cur();
      if (R && !R.done) {
        const t0 = performance.now();
        for (let i = 0; i < perFrame && !R.done; i++) R.step();
        // keep the frame short: adapt the number of sweeps per frame
        const dt = performance.now() - t0;
        perFrame = Math.max(1, Math.min(20, Math.round(perFrame * (dt > 25 ? 0.7 : dt < 12 ? 1.3 : 1))));
        draw();
      } else { playing = false; sync(); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  // ---- drawing ----
  function paintBinary(cv: HTMLCanvasElement, f: (p: number) => [number, number, number]): void {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N), d = id.data;
    for (let p = 0; p < N * N; p++) { const [r, g, b] = f(p); d[4 * p] = r; d[4 * p + 1] = g; d[4 * p + 2] = b; d[4 * p + 3] = 255; }
    ctx.putImageData(id, 0, 0);
  }

  function draw(): void {
    sync();
    if (!P) return;
    const Q = P, R = cur();
    // noisy input
    if (st.noise.kind === "flip") paintBinary(fIn.cv, p => (Q.y[p] ? [30, 30, 30] : [250, 250, 250]));
    else paintBinary(fIn.cv, p => { const v = Math.max(0, Math.min(255, 250 - 220 * Q.y[p])); return [v, v, v]; });
    if (!R) return;
    const est = R.estimate;
    fEst.c.textContent = `${METHOD_LABEL[R.method]}${R.method === "gibbs" || R.method === "meanfield" ? " → MPM" : ""}`;
    paintBinary(fEst.cv, p => (est[p] > 0 ? [30, 30, 30] : [250, 250, 250]));
    const showMarg = st.third === "marginal" && R.marginal;
    if (showMarg) {
      const mg = R.marginal!;
      paintBinary(fErr.cv, p => { const v = Math.round(250 - 220 * mg[p]); return [v, v, v]; });
    } else if (st.third === "truth") {
      paintBinary(fErr.cv, p => (Q.truth[p] ? [30, 30, 30] : [250, 250, 250]));
    } else {
      paintBinary(fErr.cv, p => {
        const e = (est[p] > 0 ? 1 : 0), t = Q.truth[p];
        if (e === t) return t ? [150, 150, 150] : [250, 250, 250];
        return t ? [31, 119, 180] : [214, 39, 40];   // blue: missed object, red: false object
      });
    }
    // readout
    const nw = wrong(Q, est);
    const E = R.energy;
    ro.textContent = [
      `${METHOD_LABEL[R.method]}: ${R.sweeps} sweeps${R.done ? " (done)" : ""}${R.method === "anneal" ? `, T = ${R.T.toFixed(2)}` : ""}`,
      `energy ${E.toFixed(1)}  (exact min ${Q.Emin.toFixed(1)})`,
      `wrong pixels ${nw}  (${(100 * nw / (N * N)).toFixed(2)} %)`,
      `J = λ/2 = ${(Q.lambda / 2).toFixed(3)}`,
    ].join("\n");
    drawPlot();
  }

  function drawPlot(): void {
    const Q = P!;
    plot.clear();
    let hi = Q.Emin;
    for (const R of runs.values()) for (let i = 1; i < R.hist.E.length; i++) hi = Math.max(hi, R.hist.E[i]);
    hi = Math.max(Math.min(hi, Q.Emin * 1.3 + 10), Q.Emin + 0.02 * Math.abs(Q.Emin) + 10);
    const lo = Q.Emin - 0.04 * (hi - Q.Emin) - 1;
    let kmax = 30;
    for (const R of runs.values()) kmax = Math.max(kmax, R.sweeps);
    plot.setDomain([1, Math.max(30, 10 ** Math.ceil(Math.log10(kmax + 1)))], [lo, hi + 0.04 * (hi - lo)]);
    plot.hline(Q.Emin, { color: COLOR.exact, dash: "7 5", width: 2.5 });
    const items: { label: string; color: string; kind?: "line" | "dot" | "box"; dash?: string }[] = [{ label: "exact min", color: COLOR.exact, dash: "7 5" }];
    for (const m of METHODS) {
      const R = runs.get(m);
      if (!R || m === "exact" || m === "threshold") continue;
      const ks = R.hist.k.slice(1), Es = R.hist.E.slice(1);
      plot.line(ks, Es, { color: COLOR[m], width: m === st.method ? 3.5 : 2.5 });
      items.push({ label: METHOD_LABEL[m], color: COLOR[m] });
    }
    plot.legend(items, "tr");
  }

  load(p0);
  presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
