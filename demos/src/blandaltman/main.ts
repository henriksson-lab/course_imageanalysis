// Agreement of an automated measurement with a manual one. Left: automated against manual
// with the identity line and the correlation r; below it the ratio auto/manual. Right: the
// Bland–Altman plot, difference against mean, with the bias, the limits of agreement
// (bias ± 1.96 SD) and a fitted line (proportional bias). Sliders: scale, constant offset,
// undercount growing with density, scatter; "areas" switches to one point per object with an
// outline offset in pixels; "annotator B" compares two manual annotations.
// Auto-mounts on <div data-demo="blandaltman">. Options: data-preset="default|corr|prop|scatter|areas|annotator".
import { htmlEl, slider, presets, button, checkbox, autoMount, Slider } from "../lib/ui";
import { Plot, seriesColor, plotFigure, extent } from "../lib/plot";
import { blandAltman, mean } from "../lib/stats";
import * as M from "./model";

const NAME = "blandaltman";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .ba-col { flex: 0 1 330px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ba-main { flex: 1 1 380px; max-width: 470px; min-width: 260px; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .35em; }
[data-demo="${NAME}"] .ba-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .ba-col { flex-basis: 300px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// lib/plot.ts numbers its clip paths per bundle ("qb-clip-1", …), so two demos on one page
// share ids and clip each other's plots. Give every plot of this demo a unique clip id.
let clipCount = 0;
function uniqueClip<T extends { svg: SVGSVGElement; layer: SVGGElement }>(p: T): T {
  const cp = p.svg.querySelector("clipPath");
  if (cp) { cp.id = `${NAME}-clip-${++clipCount}`; p.layer.setAttribute("clip-path", `url(#${cp.id})`); }
  return p;
}

const f1 = (v: number) => v.toFixed(1), f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let st: M.Settings = { ...M.DEFAULTS, ...p0.settings };
  const slide = !!host.closest(".reveal"), fs = slide ? 28 : 18;

  const col = htmlEl("div", "ba-col", "", host);
  const scF = plotFigure(col, "");
  const raF = plotFigure(col, "");
  const baF = plotFigure(host, "Bland–Altman plot", "ba-main qb-plot");
  const ro = htmlEl("div", "qb-readout", "", baF);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ba-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const sl: Partial<Record<keyof M.Settings, Slider>> = {
    scale: slider(panel, "scale", 0.7, 1.2, 0.02, st.scale, v => set("scale", v)),
    offset: slider(panel, "offset", -15, 15, 1, st.offset, v => set("offset", v)),
    density: slider(panel, "density undercount", 0, 0.4, 0.02, st.density, v => set("density", v)),
    scatter: slider(panel, "scatter (SD)", 0, 12, 0.5, st.scatter, v => set("scatter", v)),
    outline: slider(panel, "outline offset (px)", -1.5, 1.5, 0.1, st.outline, v => set("outline", v)),
  };
  const areaC = checkbox(panel, "areas (one point per object)", st.areas, v => set("areas", v));
  const annC = checkbox(panel, "annotator B instead of the method", st.annotator, v => set("annotator", v));
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Counts: 40 images. Areas: 150 objects of radius 2.5–15 px. Synthetic data.", panel);

  let sc: Plot, ra: Plot, ba: Plot;

  function set<K extends keyof M.Settings>(k: K, v: M.Settings[K]): void {
    st[k] = v;
    pbtn.forEach(b => b.classList.remove("active"));
    draw();
  }
  function load(p: M.Preset): void {
    st = { ...M.DEFAULTS, ...p.settings };
    (Object.keys(sl) as (keyof M.Settings)[]).forEach(k => sl[k]!.set(st[k] as number));
    areaC.checked = st.areas; annC.checked = st.annotator;
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    draw();
  }

  function draw(): void {
    const d = M.makeData(st), a = blandAltman(d.manual, d.auto);
    for (const k of ["scale", "offset", "density", "scatter"] as const) sl[k]!.input.disabled = st.areas || st.annotator;
    sl.outline!.input.disabled = !st.areas || st.annotator;
    // scatter
    const lim = extent([d.manual, d.auto], 0.05);
    lim[0] = Math.min(0, lim[0]);
    scF.replaceChildren(); raF.replaceChildren(); baF.querySelector("svg")?.remove();
    sc = new Plot(scF, { x: lim, y: lim, xlabel: slide ? d.xlabel.replace(" (annotator A)", "") : d.xlabel, ylabel: st.annotator ? "annotator B" : "automated", title: `r = ${f3(a.r)}`, width: 400, height: slide ? 440 : 400, font: fs });
    uniqueClip(sc);
    sc.line(lim, lim, { color: "var(--qb-muted)", width: 2, dash: "6 4" });
    sc.scatter(d.manual, d.auto, { color: seriesColor(0), r: st.areas ? 4 : 6, fillOpacity: 0.8 });
    // ratio
    const ratio = Array.from(d.manual, (m, i) => (m > 0 ? d.auto[i] / m : NaN));
    const rr = extent([ratio, [1]], 0.1);
    ra = new Plot(raF, { x: extent([a.means], 0.05), y: [Math.min(rr[0], 0.8), Math.max(rr[1], 1.2)], xlabel: "mean of the two", ylabel: "ratio", width: 400, height: slide ? 330 : 260, font: fs });
    uniqueClip(ra);
    ra.hline(1, { color: "var(--qb-muted)", width: 1.5 });
    ra.hline(mean(ratio.filter(Number.isFinite)), { color: seriesColor(1), width: 2, text: "mean " + f2(mean(ratio.filter(Number.isFinite))) });
    ra.scatter(a.means, ratio, { color: seriesColor(0), r: st.areas ? 3.5 : 5, fillOpacity: 0.8 });
    // Bland–Altman
    const xr = extent([a.means], 0.05), dr = extent([a.diffs, [a.lo, a.hi, 0]], 0.12);
    ba = new Plot(baF, { x: xr, y: dr, xlabel: "mean of the two (" + d.unit + ")", ylabel: "difference: " + (st.annotator ? "B − A" : "auto − manual"), height: 400, font: slide ? 22 : 18 });
    uniqueClip(ba);
    ba.hline(0, { color: "var(--qb-muted)", width: 1.5 });
    ba.hline(a.hi, { color: seriesColor(3), text: "+1.96 SD " + f1(a.hi) });
    ba.hline(a.lo, { color: seriesColor(3), text: "−1.96 SD " + f1(a.lo) });
    ba.hline(a.bias, { color: seriesColor(1), dash: "", width: 2.5, text: "bias " + f1(a.bias) });
    ba.line(xr, xr.map(x => a.intercept + a.slope * x), { color: seriesColor(2), width: 2, dash: "2 4" });
    ba.scatter(a.means, a.diffs, { color: seriesColor(0), r: st.areas ? 4 : 6, fillOpacity: 0.8 });
    baF.appendChild(ro);
    ro.textContent =
      `n = ${d.manual.length}   r = ${f3(a.r)}\n` +
      `bias ${f2(a.bias)}   SD ${f2(a.sd)}\n` +
      `limits of agreement ${f1(a.lo)} to ${f1(a.hi)}\n` +
      `slope of difference on mean ${f3(a.slope)} (green)`;
  }

  load(p0);
}

autoMount(NAME, mount);
