// Two Gaussians and the thresholds of five methods. The image is drawn from a mixture
// ω0 N(μ0, σ0²) + ω1 N(μ1, σ1²): a blob-shaped mask with object fraction ω1 decides the
// class of each pixel, its grey level is sampled from that class (seeded). Top: the
// histogram of the image (density), the weighted class densities ω_k f_k and markers for
// the thresholds of Otsu, isodata, minimum error (Kittler–Illingworth), triangle, Kapur
// and the true Bayes boundary (both crossings when there are two). The selected method's
// error is shaded (missed object: orange tail left of t; false object: blue tail right
// of t) and its criterion curve is drawn below on the same t axis. Right: the image with
// the selected method's mask. Smaller images give noisier histograms.
// Auto-mounts on <div data-demo="mixture">. Options: data-preset="equal|weights|variances|small|overlap", data-method="otsu|isodata|kittler|triangle|kapur|bayes".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { normalPdf } from "../lib/threshold";
import { drawOverlay, injectStyle } from "../histthresh/scenes";
import { mixtureModel, Model, Method } from "./model";

const NAME = "mixture";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .mx-plots { flex: 1 1 420px; min-width: 280px; max-width: 520px; display: flex; flex-direction: column; }
[data-demo="${NAME}"] .mx-mid { flex: 0 1 270px; min-width: 220px; display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .mx-mid canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] table.mx-t { border-collapse: collapse; font-size: 14px; width: 100%; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] .mx-t td, [data-demo="${NAME}"] .mx-t th { padding: .05em .3em; text-align: right; white-space: nowrap; }
[data-demo="${NAME}"] .mx-t th { font-family: inherit; font-weight: 600; }
[data-demo="${NAME}"] .mx-t td:first-child, [data-demo="${NAME}"] .mx-t th:first-child { text-align: left; }
[data-demo="${NAME}"] .mx-t tr.sel td { font-weight: 700; background: var(--qb-panel); }
[data-demo="${NAME}"] .mx-t .sw { display: inline-block; width: .8em; height: .8em; border-radius: 2px; margin-right: .3em; vertical-align: -1px; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .3em; }
[data-demo="${NAME}"] .mx-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .mx-mid { flex-basis: 260px; }
.reveal [data-demo="${NAME}"] .mx-plots { max-width: 500px; }
`;

const METHODS: { id: Method; label: string; color: string; crit: string }[] = [
  { id: "otsu", label: "Otsu", color: "#1f77b4", crit: "between-class variance σ_B²(t)" },
  { id: "isodata", label: "isodata", color: "#17becf", crit: "(μ0 + μ1)/2 − t: fixed points at 0" },
  { id: "kittler", label: "min-error", color: "#2ca02c", crit: "J(t) (minimise)" },
  { id: "triangle", label: "triangle", color: "#8c564b", crit: "distance below the line" },
  { id: "kapur", label: "Kapur", color: "#e377c2", crit: "entropy H0 + H1" },
  { id: "bayes", label: "Bayes", color: "#d62728", crit: "error rate of [I > t] (model)" },
];

import type { Params } from "./model";
const PRESETS: { id: string; label: string; p: Params; m: Method }[] = [
  { id: "equal", label: "equal, well separated", p: { mu0: 80, mu1: 170, s0: 15, s1: 15, w1: 0.5, n: 128 }, m: "otsu" },
  { id: "weights", label: "unequal weights", p: { mu0: 70, mu1: 130, s0: 20, s1: 20, w1: 0.1, n: 128 }, m: "otsu" },
  { id: "variances", label: "unequal variances", p: { mu0: 80, mu1: 150, s0: 12, s1: 40, w1: 0.4, n: 128 }, m: "kittler" },
  { id: "small", label: "small object (3 %)", p: { mu0: 60, mu1: 100, s0: 12, s1: 20, w1: 0.03, n: 128 }, m: "triangle" },
  { id: "overlap", label: "heavy overlap", p: { mu0: 100, mu1: 130, s0: 20, s1: 20, w1: 0.4, n: 128 }, m: "bayes" },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const pr0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { ...pr0.p, method: (host.dataset.method as Method) ?? pr0.m };

  const plots = htmlEl("div", "mx-plots", "", host);
  const top = new Plot(plots, { width: 520, height: 300, x: [0, 255], y: [0, 0.03], xlabel: "grey level", ylabel: "density", xticks: [0, 50, 100, 150, 200, 250] });
  const bot = new Plot(plots, { width: 520, height: 180, x: [0, 255], y: [0, 1], ylabel: "criterion", xticks: [0, 50, 100, 150, 200, 250], yticks: 3 });
  const mid = htmlEl("div", "mx-mid", "", host);
  const fig = htmlEl("figure", "", "", mid);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  const table = htmlEl("table", "mx-t", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "mx-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "mx-lab", "method", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = METHODS.map(m => button(mRow, m.label, () => { st.method = m.id; draw(); }));
  const sl = {
    mu0: slider(panel, "μ₀", 0, 255, 1, st.mu0, v => { st.mu0 = v; regen(); }),
    s0: slider(panel, "σ₀", 3, 60, 1, st.s0, v => { st.s0 = v; regen(); }),
    mu1: slider(panel, "μ₁", 0, 255, 1, st.mu1, v => { st.mu1 = v; regen(); }),
    s1: slider(panel, "σ₁", 3, 60, 1, st.s1, v => { st.s1 = v; regen(); }),
    w1: slider(panel, "ω₁", 0.01, 0.9, 0.01, st.w1, v => { st.w1 = v; regen(); }),
    n: slider(panel, "image", 5, 8, 1, Math.log2(st.n), v => { st.n = 2 ** v; regen(); }, v => `${2 ** v}²`),
  };
  button(panel, "reset", () => load(pr0));

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, p.p); st.method = p.m;
    sl.mu0.set(st.mu0); sl.s0.set(st.s0); sl.mu1.set(st.mu1); sl.s1.set(st.s1); sl.w1.set(st.w1); sl.n.set(Math.log2(st.n));
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    regen();
  }

  let model: Model = mixtureModel(st);
  function regen(): void { model = mixtureModel(st); draw(); }

  function draw(): void {
    const { img, truth, hist, results, isObject: isObj, modelError } = model;
    const isObject = (m: Method, v: number) => isObj(m, v);
    mBtns.forEach((b, i) => b.classList.toggle("active", METHODS[i].id === st.method));
    const N = st.n * st.n, xs = Array.from({ length: 256 }, (_, i) => i);
    const f0 = xs.map(x => (1 - st.w1) * normalPdf(x, st.mu0, st.s0)), f1 = xs.map(x => st.w1 * normalPdf(x, st.mu1, st.s1));
    const dens = Array.from(hist, c => c / N);
    let ymax = 0;
    for (let i = 0; i < 256; i++) ymax = Math.max(ymax, f0[i] + f1[i], Math.min(dens[i], 2 * (f0[i] + f1[i]) + 0.002));
    top.clear();
    top.setDomain([0, 255], [0, ymax * 1.12]);
    const edges = Array.from({ length: 257 }, (_, i) => i - 0.5);
    top.histogram(edges, dens, { color: "var(--qb-muted, #999)", fillOpacity: 0.35, width: 0.5 });
    // shaded error of the selected method
    const sel = METHODS.find(m => m.id === st.method)!;
    const miss = xs.map((x, i) => (isObject(st.method, x) ? 0 : f1[i])), fals = xs.map((x, i) => (isObject(st.method, x) ? f0[i] : 0));
    const zeros = xs.map(() => 0);
    top.band(xs, zeros, miss, { color: "#ff7f0e", fillOpacity: 0.45 });
    top.band(xs, zeros, fals, { color: "#1f77b4", fillOpacity: 0.45 });
    top.line(xs, f0, { color: "#1f77b4", width: 2.5 });
    top.line(xs, f1, { color: "#ff7f0e", width: 2.5 });
    top.line(xs, f0.map((v, i) => v + f1[i]), { color: "var(--qb-fg, #222)", width: 1.5, dash: "4 3" });
    for (const m of METHODS) {
      const r = results.get(m.id)!, on = m.id === st.method;
      const ts = m.id === "bayes" ? (r.extra ?? []).filter(c => c >= 0 && c <= 255) : [r.t + 0.5];
      for (const tv of ts) top.vline(tv, { color: m.color, width: on ? 4 : 2, dash: on ? "" : "5 4", text: on ? m.label : undefined });
    }

    // criterion of the selected method
    const r = results.get(st.method)!;
    const cx: number[] = [], cy: number[] = [];
    r.curve.forEach((v, i) => { if (Number.isFinite(v)) { cx.push(i + 0.5); cy.push(v); } });
    let lo = Math.min(...cy), hi = Math.max(...cy);
    if (st.method === "kittler") hi = Math.min(hi, lo + 3);
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    bot.clear();
    bot.setLabels({ xlabel: "threshold t   ·   " + sel.crit });
    bot.setDomain([0, 255], [lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo)]);
    bot.line(cx, cy, { color: sel.color, width: 3 });
    if (st.method === "isodata") bot.hline(0, { width: 1.5, dash: "3 3" });
    bot.vline(r.t + 0.5, { color: sel.color, width: 2.5, dash: "5 4" });

    // image with the mask
    const n = st.n, mask = new Uint8Array(n * n);
    let wrong = 0;
    for (let i = 0; i < n * n; i++) { mask[i] = isObject(st.method, img.data[i]) ? 1 : 0; if (mask[i] !== truth[i]) wrong++; }
    drawOverlay(cv, img, [{ mask, rgb: [255, 127, 14], alpha: 0.5 }]);
    cap.textContent = `${sel.label}: wrong pixels ${(100 * wrong / (n * n)).toFixed(1)} %`;

    // table
    table.replaceChildren();
    const hr = htmlEl("tr", "", "", table);
    for (const c of ["method", "t", "error"]) htmlEl("th", "", c, hr);
    for (const m of METHODS) {
      const tr = htmlEl("tr", m.id === st.method ? "sel" : "", "", table);
      const td = htmlEl("td", "", "", tr);
      const sw = htmlEl("span", "sw", "", td); sw.style.background = m.color;
      td.append(m.label);
      const rr = results.get(m.id)!;
      const tl = m.id === "bayes" && rr.extra && rr.extra.length > 1 ? rr.extra.filter(c => c >= 0 && c <= 255).map(c => c.toFixed(0)).join(", ") : m.id === "bayes" && rr.extra?.length ? rr.extra[0].toFixed(1) : String(rr.t);
      htmlEl("td", "", tl, tr);
      htmlEl("td", "", (100 * modelError(m.id)).toFixed(1) + " %", tr);
      tr.style.cursor = "pointer";
      tr.addEventListener("click", () => { st.method = m.id; draw(); });
    }
  }

  load(pr0);
  if (host.dataset.method) { st.method = host.dataset.method as Method; draw(); }
}

autoMount(NAME, mount);
