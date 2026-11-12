// Bootstrap and the variance of an average. Left: n numbered training points; "draw a
// bootstrap sample" draws n of them with replacement and shows how often each was drawn
// (grey = never: out of bag); "draw 1000" collects the out-of-bag fraction of many samples
// in a histogram around (1 − 1/n)^n ≈ 1/e. Right: B predictors with variance σ² and pairwise
// correlation ρ; the variance of their average against B, simulated (dots, seeded) and by
// the formula ρσ² + (1 − ρ)σ²/B, with its floor ρσ².
// Auto-mounts on <div data-demo="bootstrap">. Options: data-preset="n20|small|rho0|rho05",
// data-panel="both|bootstrap|variance".
import { htmlEl, slider, presets, button, autoMount, svgEl, Slider, rng } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { bootstrapCounts, oobFractionSamples, pNotDrawn, varianceOfAverage, simulateAverageVariance } from "./sim";

const NAME = "bootstrap";

const STYLE = `
[data-demo="${NAME}"] { gap: 1.2em; align-items: flex-start; }
[data-demo="${NAME}"] .bs-col { flex: 1 1 420px; min-width: 260px; max-width: 560px; display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .bs-pts { width: 100%; display: block; }
[data-demo="${NAME}"] .bs-pts text { font-size: 15px; }
[data-demo="${NAME}"] .bs-lab { font-size: 15px; color: var(--qb-muted); font-weight: 600; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-line; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const BS = [1, 2, 3, 5, 10, 20, 50, 100, 200];

function mount(host: HTMLElement): void {
  injectStyle();
  const st = { n: 20, sigma2: 1, rho: 0.3, sampleNo: 0, counts: null as Int32Array | null, many: null as Float64Array | null };
  const pre = host.dataset.preset ?? "n20";
  if (pre === "small") st.n = 5;
  if (pre === "rho0") st.rho = 0;
  if (pre === "rho05") st.rho = 0.5;
  const which = host.dataset.panel ?? "both";
  let r = rng(1);

  // ---- left: bootstrap samples
  const colA = htmlEl("div", "bs-col", "", host);
  htmlEl("div", "bs-lab", "a bootstrap sample: draw n of the n points, with replacement", colA);
  const W = 520;
  const ptsSvg = svgEl("svg", { viewBox: `0 0 ${W} 150`, class: "bs-pts", role: "img", "aria-label": "training points and how often each was drawn" }, colA);
  const rowA = htmlEl("div", "qb-presets", "", colA);
  button(rowA, "draw a bootstrap sample", () => { st.sampleNo++; st.counts = bootstrapCounts(st.n, r); drawA(); });
  button(rowA, "draw 1000", () => { st.many = oobFractionSamples(st.n, 1000, 100 + st.n); drawA(); });
  button(rowA, "reset", () => { r = rng(1); st.sampleNo = 1; st.counts = bootstrapCounts(st.n, r); st.many = null; drawA(); });
  const nS: Slider = slider(colA, "points n", 5, 100, 1, st.n, v => { st.n = v; r = rng(1); st.sampleNo = 0; st.counts = null; st.many = null; drawA(); });
  const hist = new Plot(colA, { width: W, height: 210, x: [0, 1], y: [0, 300], xlabel: "fraction out of bag", ylabel: "samples", margin: { l: 70 }, ariaLabel: "histogram of the out-of-bag fraction" });
  const roA = htmlEl("div", "qb-readout", "", colA);

  function drawA(): void {
    ptsSvg.replaceChildren();
    const n = st.n, perRow = n <= 20 ? 10 : 20, rows = Math.ceil(n / perRow), cell = W / perRow;
    const rad = Math.min(17, cell * 0.38), H = rows * (cell + 18) + 6;
    ptsSvg.setAttribute("viewBox", `0 0 ${W} ${Math.max(60, H)}`);
    for (let i = 0; i < n; i++) {
      const cx = (i % perRow + 0.5) * cell, cy = Math.floor(i / perRow) * (cell + 18) + cell / 2 + 4;
      const c = st.counts ? st.counts[i] : 1;
      const circ = svgEl("circle", { cx, cy, r: rad }, ptsSvg);
      circ.style.setProperty("fill", !st.counts ? "var(--qb-c1, #1f77b4)" : c === 0 ? "var(--qb-grid, #ddd)" : "var(--qb-c1, #1f77b4)");
      circ.style.setProperty("fill-opacity", !st.counts ? "0.35" : c === 0 ? "1" : String(Math.min(1, 0.35 + 0.25 * c)));
      circ.style.setProperty("stroke", "var(--qb-axis)");
      if (n <= 40) {
        const t = svgEl("text", { x: cx, y: cy + 5, "text-anchor": "middle" }, ptsSvg);
        t.textContent = String(i + 1);
      }
      if (st.counts) {
        const t2 = svgEl("text", { x: cx, y: cy + rad + 15, "text-anchor": "middle" }, ptsSvg);
        t2.textContent = c === 0 ? "–" : "×" + c;
        t2.style.setProperty("font-weight", c > 1 ? "700" : "400");
      }
    }
    hist.clear();
    const p = pNotDrawn(n);
    if (st.many) {
      const nb = 25, edges = Array.from({ length: nb + 1 }, (_, i) => i / nb * 0.8), cnt = new Array(nb).fill(0);
      for (const v of st.many) cnt[Math.min(nb - 1, Math.floor(v / 0.8 * nb))]++;
      hist.setDomain([0, 0.8], [0, Math.max(...cnt) * 1.15]);
      hist.histogram(edges, cnt, { color: seriesColor(0), fillOpacity: 0.5 });
      hist.vline(p, { color: seriesColor(1), text: "(1−1/n)ⁿ" });
    } else hist.setDomain([0, 0.8], [0, 300]);
    let s = `(1 − 1/n)ⁿ = ${p.toFixed(3)}   (1/e = 0.368)`;
    if (st.counts) {
      let z = 0, two = 0; for (const c of st.counts) { if (!c) z++; if (c > 1) two++; }
      s = `sample ${st.sampleNo}: ${z} of ${st.n} out of bag (${(100 * z / st.n).toFixed(0)} %), ${two} drawn more than once\n` + s;
    }
    if (st.many) s += `\nmean over 1000 samples: ${(100 * st.many.reduce((a, b) => a + b, 0) / st.many.length).toFixed(1)} %`;
    roA.textContent = s;
  }

  // ---- right: variance of an average
  const colB = htmlEl("div", "bs-col", "", host);
  htmlEl("div", "bs-lab", "averaging B predictors with correlation ρ", colB);
  const vp = new Plot(colB, { width: W, height: 300, x: [1, 200], y: [0.005, 2], xlog: true, ylog: true, xlabel: "number of predictors B", ylabel: "variance of the average", margin: { l: 80 }, ariaLabel: "variance of an average against B" });
  const fLine = vp.line([1], [1], { color: seriesColor(0), label: "ρσ² + (1−ρ)σ²/B" });
  const floor = vp.line([1, 200], [1, 1], { color: seriesColor(2), dash: "7 5", label: "floor ρσ²" });
  const dots = vp.scatter([1], [1], { color: seriesColor(1), r: 6, label: "simulated" });
  vp.legend(undefined, "bl");
  const s2S: Slider = slider(colB, "σ²", 0.1, 2, 0.1, st.sigma2, v => { st.sigma2 = v; drawB(); });
  const rS: Slider = slider(colB, "ρ", 0, 1, 0.05, st.rho, v => { st.rho = v; drawB(); });
  presets(colB, [
    { label: "ρ = 0", apply: () => { st.rho = 0; rS.set(0); drawB(); } },
    { label: "ρ = 0.3", apply: () => { st.rho = 0.3; rS.set(0.3); drawB(); } },
    { label: "ρ = 0.5: the floor", apply: () => { st.rho = 0.5; rS.set(0.5); drawB(); } },
    { label: "small n", apply: () => { st.n = 5; nS.set(5); r = rng(1); st.sampleNo = 0; st.counts = null; st.many = null; drawA(); } },
  ]);
  const roB = htmlEl("div", "qb-readout", "", colB);

  function drawB(): void {
    const xs = Array.from({ length: 120 }, (_, i) => Math.pow(200, i / 119));
    fLine.set(xs, xs.map(B => varianceOfAverage(st.sigma2, st.rho, B)));
    const fl = st.rho * st.sigma2;
    floor.set([1, 200], [Math.max(fl, 1e-9), Math.max(fl, 1e-9)]);
    floor.g.style.display = fl > 0 ? "" : "none";
    const sim = simulateAverageVariance(st.sigma2, st.rho, BS, 2000, 5);
    dots.set(BS, Array.from(sim).map(v => Math.max(v, 1e-4)));
    vp.setDomain([1, 200], [Math.min(0.002, Math.max(1e-4, fl * 0.5)) , Math.max(2, st.sigma2 * 1.5)]);
    const f = (B: number) => varianceOfAverage(st.sigma2, st.rho, B).toFixed(3);
    roB.textContent = `formula: B = 1: ${f(1)}, 10: ${f(10)}, 100: ${f(100)}, ∞: ${fl.toFixed(3)}\nsimulated (2000 runs): B = 10: ${sim[4].toFixed(3)}, 100: ${sim[7].toFixed(3)}`;
  }

  st.sampleNo = 1; st.counts = bootstrapCounts(st.n, r);
  if (which === "bootstrap") colB.style.display = "none";
  if (which === "variance") colA.style.display = "none";
  drawA(); drawB();
}

autoMount(NAME, mount);
