// Point patterns and complete spatial randomness. Seeded patterns from the models of the day
// (CSR, Thomas clusters, hard core, lattice, gradient, colonies with a hard core, tight pairs),
// optionally with a detection artefact (close spots merge, objects split). Tabs:
//  - quadrats: grid with counts over the pattern, histogram of the counts against the Poisson
//    pmf of the same mean, VMR and the χ² test, and VMR against the quadrat size;
//  - intensity: kernel estimate of ρ(x) (edge-corrected) with a bandwidth slider, and the
//    column profile against the true ρ;
//  - which is random?: three patterns (CSR, weak clusters, weak hard core), shuffled per seed;
//  - from an image: spots rendered at the pattern's positions (PSF σ), detected again; the
//    detected pattern against the true one.
// Auto-mounts on <div data-demo="csrplay">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rng, shuffle } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { nnDistances } from "../lib/pointpattern";
import { clarkEvans } from "../lib/neighbours";
import { spotImage, detectSpots } from "../lib/spotdetect";
import { Spec, DEFAULT_SPEC, makePattern, makeWindow, PatternView, sourceControls, injectStyle, sourceCss, heatUrl, viewText, describe, gradientRho, SIDE } from "./source";

const NAME = "csrplay";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = sourceCss(NAME) + `
${P} .cp-view { flex: 0 1 360px; min-width: 240px; }
${P} .cp-main { flex: 1 1 360px; min-width: 260px; max-width: 520px; display: flex; flex-direction: column; gap: .3em; }
${P} .qb-panel { flex: 1 1 300px; gap: .35em; }
${P} .cp-trio { display: flex; gap: .5em; }
${P} .cp-trio > div { flex: 1 1 0; display: flex; flex-direction: column; align-items: center; gap: .2em; }
${P} .cp-cap { font-size: .9em; text-align: center; }
.reveal ${P} .cp-view { flex: 0 0 360px; min-width: 0; }
.reveal ${P} .cp-main { flex: 0 0 410px; min-width: 0; }
.reveal ${P} .qb-panel { flex: 1 1 330px; min-width: 0; gap: .15em; }
`;

type Tab = "quadrats" | "intensity" | "random" | "image";
const TABS: [Tab, string][] = [["quadrats", "quadrats"], ["intensity", "intensity"], ["random", "which is random?"], ["image", "from an image"]];

interface State { spec: Spec; tab: Tab; q: number; bw: number; psf: number; guess: number }
const DEFAULT: State = { spec: { ...DEFAULT_SPEC }, tab: "quadrats", q: 8, bw: 15, psf: 1.5, guess: -1 };

const PRESETS: { id: string; label: string; s: Partial<State>; spec?: Partial<Spec> }[] = [
  { id: "csr", label: "CSR: variance = mean", s: { tab: "quadrats", q: 8 }, spec: { model: "csr" } },
  { id: "clusters", label: "clusters: VMR > 1", s: { tab: "quadrats", q: 8 }, spec: { model: "thomas", k: 20, mu: 8, sigma: 4 } },
  { id: "regular", label: "regular: VMR < 1", s: { tab: "quadrats", q: 8 }, spec: { model: "hardcore", n: 150, h: 12 } },
  { id: "scale", label: "quadrat size changes the verdict", s: { tab: "quadrats", q: 40 }, spec: { model: "colonies" } },
  { id: "merged", label: "merged spots look regular", s: { tab: "quadrats", q: 10 }, spec: { model: "csr", artefact: "merge", mergeD: 8 } },
  { id: "split", label: "split nuclei look clustered", s: { tab: "quadrats", q: 10 }, spec: { model: "csr", artefact: "split", splitFrac: 0.3 } },
  { id: "gradient", label: "gradient: clustered by quadrats", s: { tab: "quadrats", q: 4 }, spec: { model: "inhom", grad: 0.8 } },
  { id: "bandwidth", label: "intensity bandwidth", s: { tab: "intensity", bw: 15 }, spec: { model: "inhom", grad: 0.8 } },
  { id: "random", label: "which one is random?", s: { tab: "random" } },
  { id: "image", label: "detected from an image", s: { tab: "image", psf: 3 }, spec: { model: "csr", n: 150 } },
];

const QSIDES = [2, 4, 5, 8, 10, 20, 25, 40];   // quadrats per side: 100 … 5 µm

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 22 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, spec: { ...DEFAULT_SPEC } };
  if (p0) { Object.assign(st, p0.s); Object.assign(st.spec, p0.spec ?? {}); }
  else if (host.dataset.tab) st.tab = host.dataset.tab as Tab;

  // ---- layout
  const viewBox = htmlEl("div", "cp-view", "", host);
  const viewCap = htmlEl("div", "cp-cap", "", viewBox);
  const view = new PatternView(viewBox);

  const main = htmlEl("div", "cp-main", "", host);
  const tabsEl = htmlEl("div", "ss-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(); }));
  const trio = htmlEl("div", "cp-trio", "", main);
  const f1 = plotFigure(main, "", "qb-figure"), f2 = plotFigure(main, "", "qb-figure");
  const pl1 = new Plot(f1, { width: 460, height: 250, x: [0, 10], y: [0, 1], font: FS, margin: { l: 4.2 * FS, b: 2.7 * FS, t: 10, r: 14 } });
  const pl2 = new Plot(f2, { width: 460, height: 230, x: [4, 120], y: [0, 1], font: FS, xlog: true, margin: { l: 3.4 * FS, b: 2.7 * FS, t: 10, r: 14 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const tabCtl = htmlEl("div", "", "", panel);
  const sl = {
    q: slider(tabCtl, "quadrats per side", 2, 40, 1, st.q, v => { st.q = v; unp(); draw(); }),
    bw: slider(tabCtl, "bandwidth σ µm", 3, 60, 1, st.bw, v => { st.bw = v; unp(); draw(); }),
    psf: slider(tabCtl, "PSF σ µm", 0.8, 5, 0.1, st.psf, v => { st.psf = v; unp(); draw(); }),
  };
  const guessRow = htmlEl("div", "qb-presets", "", tabCtl);
  const guessBtns = ["A", "B", "C"].map((l, i) => button(guessRow, "random is " + l, () => { st.guess = i; draw(); }));
  button(guessRow, "new trio", () => { st.spec.seed++; st.guess = -1; src.sync(); draw(); });
  const src = sourceControls(panel, st.spec, () => { unp(); st.guess = -1; regenerate(); draw(); }, { artefacts: true });
  htmlEl("div", "ss-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [];
  let activeId = p0?.id ?? "";
  function buildPresets(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => p.s.tab === st.tab) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => { b.dataset.id = list[i].id; b.classList.toggle("active", list[i].id === activeId); });
  }
  button(panel, "reset", () => applyPreset(null));

  let pts = makePattern(st.spec);
  function regenerate(): void { pts = makePattern(st.spec); }
  function unp(): void { activeId = ""; pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    const { spec } = st;
    Object.assign(st, DEFAULT, p?.s ?? {}, { spec });
    Object.assign(spec, DEFAULT_SPEC, p?.spec ?? {});
    src.sync();
    st.guess = -1;
    sl.q.set(st.q); sl.bw.set(st.bw); sl.psf.set(st.psf);
    activeId = p?.id ?? "";
    regenerate(); draw();
  }

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const pfmt = (v: number) => (v < 0.001 ? "< 0.001" : v.toFixed(3));

  // ---- tabs
  function drawQuadrats(): string {
    const win = makeWindow(st.spec.win), q = PP.quadratCounts(pts, win, st.q, st.q);
    view.mid.replaceChildren();
    const qw = win.w / st.q, lw = 0.6;
    for (let k = 1; k < st.q; k++) {
      svgEl("line", { x1: k * qw, x2: k * qw, y1: 0, y2: win.h, stroke: "var(--qb-c4)", "stroke-width": lw }, view.mid);
      svgEl("line", { y1: k * qw, y2: k * qw, x1: 0, x2: win.w, stroke: "var(--qb-c4)", "stroke-width": lw }, view.mid);
    }
    if (st.q <= 10) for (let gy = 0; gy < st.q; gy++) for (let gx = 0; gx < st.q; gx++) {
      const c = q.counts[gy * st.q + gx];
      if (Number.isFinite(c)) viewText(view.over, (gx + 0.5) * qw, (gy + 0.5) * qw + 0.35 * Math.min(qw * 0.45, 16), String(c), Math.min(qw * 0.45, 16), "var(--qb-c2)");
    }
    // histogram of counts against Poisson(mean)
    const maxC = Math.max(...Array.from(q.counts).filter(Number.isFinite), 1);
    const K = Math.max(maxC, Math.ceil(q.mean + 4 * Math.sqrt(q.mean)) + 1);
    const hist = new Array(K + 1).fill(0);
    for (const c of q.counts) if (Number.isFinite(c)) hist[c]++;
    const pmf = (k: number) => Math.exp(k * Math.log(q.mean) - q.mean - lgamma(k + 1)) * q.m;
    const xs = Array.from({ length: K + 1 }, (_, k) => k), pm = xs.map(pmf);
    pl1.setDomain([-0.6, K + 0.6], [0, Math.max(...hist, ...pm) * 1.25 + 0.5]);
    pl1.setLabels({ xlabel: "points per quadrat", ylabel: "quadrats" });
    pl1.bars(xs, hist, { color: "var(--qb-c1)", barWidth: 0.8, label: "observed" });
    pl1.scatter(xs, pm, { color: "var(--qb-c2)", r: 5, label: "Poisson, same mean" });
    pl1.legend(undefined, "tr");
    // VMR against quadrat size
    const sizes = QSIDES.map(k => win.w / k), vmrs = QSIDES.map(k => PP.quadratCounts(pts, win, k, k).vmr);
    const vmax = Math.max(2, ...vmrs.filter(Number.isFinite)) * 1.15;
    pl2.setDomain([4, 120], [0, vmax]);
    pl2.setLabels({ xlabel: "quadrat side (µm, log scale)", ylabel: "VMR" });
    pl2.hline(1, { color: "var(--qb-muted)", dash: "6 4", width: 2 });
    pl2.line(sizes, vmrs, { color: "var(--qb-c1)" });
    pl2.scatter(sizes, vmrs, { color: "var(--qb-c1)", r: 5 });
    pl2.scatter([win.w / st.q], [q.vmr], { color: "var(--qb-c2)", r: 8 });
    const verdict = q.pClustered < 0.05 ? "clustered (p < 0.05)" : q.pRegular < 0.05 ? "regular (p < 0.05)" : "consistent with CSR";
    return `quadrats ${q.m} of ${fmt(qw, 1)} µm\nmean ${fmt(q.mean)}  variance ${fmt(q.variance)}\nVMR ${fmt(q.vmr)}   X² ${fmt(q.X2, 1)}, df ${q.df}\np clustered ${pfmt(q.pClustered)}\np regular   ${pfmt(q.pRegular)}\n→ ${verdict}`;
  }

  function drawIntensity(): string {
    const win = makeWindow(st.spec.win), N = 64;
    const map = PP.kernelIntensity(pts, win, st.bw, N, N, true);
    const fin = Array.from(map).filter(Number.isFinite), hi = Math.max(...fin), lo = 0;
    view.under.replaceChildren();
    svgEl("image", { x: 0, y: 0, width: win.w, height: win.h, href: heatUrl(map.map(v => v * 1000), N, N, lo * 1000, hi * 1000), preserveAspectRatio: "none" }, view.under);
    // profile along x (column means) × 1000
    const prof: number[] = [], xs: number[] = [];
    for (let gx = 0; gx < N; gx++) {
      let s = 0, c = 0;
      for (let gy = 0; gy < N; gy++) { const v = map[gy * N + gx]; if (Number.isFinite(v)) { s += v; c++; } }
      xs.push((gx + 0.5) * win.w / N); prof.push(c ? 1000 * s / c : NaN);
    }
    const rho = gradientRho(st.spec, win), avg = 1000 * pts.x.length / win.area;
    const truth = st.spec.model === "inhom" ? (x: number) => 1000 * rho(x, 0) : () => avg;
    const ymax = Math.max(...prof.filter(Number.isFinite), truth(0), truth(win.w)) * 1.2;
    pl1.setDomain([0, win.w], [0, ymax]);
    pl1.setLabels({ xlabel: "x (µm)", ylabel: "ρ̂ per 1000 µm²" });
    pl1.line(xs, prof, { color: "var(--qb-c1)", label: "estimate (column mean)" });
    pl1.fn(truth, { color: "var(--qb-c2)", dash: "7 5", label: st.spec.model === "inhom" ? "true ρ(x)" : "n / area" });
    pl1.legend(undefined, "tl");
    f2.style.display = "none";
    return `n / area = ${fmt(avg, 2)} per 1000 µm²\nmap: ${fmt(1000 * Math.min(...fin), 2)} … ${fmt(1000 * hi, 2)}\nbandwidth σ = ${st.bw} µm\nsmall σ: a bump per point\nlarge σ: a constant`;
  }

  function trioPatterns(): { pts: PP.Points[]; order: number[] } {
    const win = makeWindow("square"), seed = st.spec.seed;
    const cl = PP.thomas(rng(500 + seed), 40 / (SIDE * SIDE), 4, 7, win), n = Math.max(60, cl.x.length);
    const list = [PP.binomialProcess(rng(600 + seed), n, win), cl, PP.ssi(rng(700 + seed), n, 8, win)];
    const order = shuffle(rng(800 + seed), [0, 1, 2]);
    return { pts: order.map(k => list[k]), order };
  }

  function drawRandom(): string {
    const { pts: three, order } = trioPatterns();
    trio.replaceChildren();
    const names = ["random (CSR)", "weakly clustered", "weakly regular"];
    three.forEach((p, i) => {
      const box = htmlEl("div", "", "", trio);
      const v = new PatternView(box);
      v.setPoints(p, undefined, 2.6);
      const cap = htmlEl("div", "cp-cap", ["A", "B", "C"][i] + (st.guess >= 0 ? ": " + names[order[i]] : ""), box);
      if (st.guess >= 0 && order[i] === 0) cap.style.fontWeight = "700";
    });
    guessBtns.forEach((b, i) => b.classList.toggle("active", i === st.guess));
    const truthIdx = order.indexOf(0);
    if (st.guess < 0) return "Which of A, B, C is\ncompletely random?\nPick one, then compare.";
    return (st.guess === truthIdx ? "Right: " : "No: ") + ["A", "B", "C"][truthIdx] + " is random.\nRandom patterns look\nclumpier than we expect;\nthe regular one looks\n'more random' to most.";
  }

  function drawImage(): string {
    const win = makeWindow("square");
    const list = Array.from(pts.x, (x, i) => ({ x: x - 0.5, y: pts.y[i] - 0.5, amplitude: 80 }));
    const im = spotImage({ size: SIDE, sigma: st.psf, list, noise: 6, background: 30, seed: 3 });
    const det = detectSpots(im.image, { filter: "log", sigma: st.psf, radius: Math.max(1, Math.round(st.psf)), k: 5 }).dets;
    const dp = PP.makePoints(det.map(d => d.x + 0.5), det.map(d => d.y + 0.5));
    // image under the view
    let lo = Infinity, hi = -Infinity;
    for (const v of im.image.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const c = document.createElement("canvas"); c.width = SIDE; c.height = SIDE;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(SIDE, SIDE);
    for (let i = 0; i < SIDE * SIDE; i++) { const g = Math.max(0, Math.min(255, 255 * (im.image.data[i] - lo) / (hi - lo) * 1.4)); id.data[4 * i] = g; id.data[4 * i + 1] = g; id.data[4 * i + 2] = g; id.data[4 * i + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    view.under.replaceChildren();
    svgEl("image", { x: 0, y: 0, width: SIDE, height: SIDE, href: c.toDataURL(), preserveAspectRatio: "none", style: "image-rendering: pixelated" }, view.under);
    view.setPoints(pts, () => "var(--qb-c4)", 1.3);
    for (let i = 0; i < dp.x.length; i++) svgEl("circle", { cx: dp.x[i], cy: dp.y[i], r: 3.4, fill: "none", stroke: "var(--qb-c3)", "stroke-width": 1.1 }, view.over);
    // NN distance histograms: true vs detected
    const dt = nnDistances(pts).d, dd = nnDistances(dp).d, bw = 2, nb = 15;
    const ht = new Array(nb).fill(0), hd = new Array(nb).fill(0);
    for (const v of dt) { const k = Math.floor(v / bw); if (k < nb) ht[k]++; }
    for (const v of dd) { const k = Math.floor(v / bw); if (k < nb) hd[k]++; }
    const edges = Array.from({ length: nb + 1 }, (_, k) => k * bw);
    pl1.setDomain([0, nb * bw], [0, Math.max(...ht, ...hd) * 1.25 + 1]);
    pl1.setLabels({ xlabel: "nearest-neighbour distance (µm)", ylabel: "points" });
    pl1.histogram(edges, ht, { color: "var(--qb-c4)", fillOpacity: 0.45, label: "true positions" });
    pl1.histogram(edges, hd, { color: "var(--qb-c3)", fillOpacity: 0.45, label: "detected" });
    pl1.legend(undefined, "tr");
    f2.style.display = "none";
    const Rt = clarkEvans(dt, win.area).R, Rd = clarkEvans(dd, win.area).R;
    const qt = PP.quadratCounts(pts, win, 10, 10).vmr, qd = PP.quadratCounts(dp, win, 10, 10).vmr;
    return `true points  ${pts.x.length}\ndetected     ${dp.x.length}\nClark–Evans R: true ${fmt(Rt)}, detected ${fmt(Rd)}\nVMR (20 µm):   true ${fmt(qt)}, detected ${fmt(qd)}\norange · true, green ○ detected`;
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    buildPresets();
    const win = makeWindow(st.tab === "image" || st.tab === "random" ? "square" : st.spec.win);
    if (st.tab === "image" && st.spec.win !== "square") { st.spec.win = "square"; regenerate(); }
    view.setWindow(win);
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    pl1.clear(); pl2.clear();
    f1.style.display = ""; f2.style.display = "";
    const rnd = st.tab === "random";
    trio.style.display = rnd ? "" : "none";
    viewBox.style.display = rnd ? "none" : "";
    main.style.flexBasis = rnd ? (onSlide ? "780px" : "760px") : "";
    main.style.maxWidth = rnd ? "none" : "";
    f1.style.display = rnd ? "none" : ""; f2.style.display = rnd ? "none" : "";
    sl.q.input.closest("label")!.style.display = st.tab === "quadrats" ? "" : "none";
    sl.bw.input.closest("label")!.style.display = st.tab === "intensity" ? "" : "none";
    sl.psf.input.closest("label")!.style.display = st.tab === "image" ? "" : "none";
    guessRow.style.display = rnd ? "" : "none";
    src.el.style.display = rnd ? "none" : "";
    if (!rnd) view.setPoints(pts, () => (st.tab === "intensity" ? "#fff" : "var(--qb-c1)"));
    let txt = "";
    if (st.tab === "quadrats") txt = drawQuadrats();
    else if (st.tab === "intensity") txt = drawIntensity();
    else if (st.tab === "random") txt = drawRandom();
    else txt = drawImage();
    viewCap.textContent = onSlide ? `${pts.x.length} points` + (st.spec.artefact !== "none" ? " (detected)" : "") : describe(st.spec, pts);
    ro.textContent = txt;
  }

  draw();
}

function lgamma(x: number): number {
  let s = 0;
  for (let k = 2; k < x; k++) s += Math.log(k);
  return s;
}

autoMount(NAME, mount);
