// Loss functions of a classifier, as functions of the margin value m = y·f.
// Top left: the 0–1, hinge, logistic (in bits) and squared losses against m (toggles).
// Bottom left: a 1D data set (blue = −1, red = +1) and the classifier f(x) = a (x − t): drag
// anywhere in the plot to move the threshold t, the slider sets the slope a; the band
// t ± 1/a is the street (|f| < 1). Each point carries a bar: its loss under the chosen loss
// function. Top right: the average loss of every enabled loss function against t, with the
// current t marked: the 0–1 loss is flat with steps, the others slope smoothly. Buttons move
// t to the minimum of a loss (slope fixed). Readout: the total of each loss.
//
// Auto-mounts on <div data-demo="hingeloss">. Options:
//   data-preset  outlier | far | steps | table (default outlier)
import { htmlEl, slider, presets, checkbox, button, autoMount, Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { QUALITATIVE } from "../lib/colour";
import { PRESETS, Preset, LOSSES, avgLoss, argminT, margins } from "./losses";

const NAME = "hingeloss";

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .hl-plots { flex: 1 1 560px; min-width: 0; max-width: 760px; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .8em; }
[data-demo="${NAME}"] .hl-plots > .hl-wide { grid-column: 1 / span 2; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; gap: .35em; }
[data-demo="${NAME}"] .hl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .hl-wide svg { touch-action: none; cursor: ew-resize; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .hl-plots { flex-basis: 760px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { max-height: 560px; overflow-y: auto; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { pts: p0.pts, t: p0.t, a: p0.a, on: [...p0.on], show: p0.show, note: p0.note };
  const plots = htmlEl("div", "hl-plots", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  const pA = new Plot(htmlEl("div", "", "", plots), { width: 420, height: 300, x: [-3, 3], y: [0, 4], xlabel: "margin value y·f", ylabel: "loss", xticks: [-3, -2, -1, 0, 1, 2, 3], ariaLabel: "loss functions against y f" });
  const pC = new Plot(htmlEl("div", "", "", plots), { width: 420, height: 300, x: [0, 10], y: [0, 2], xlabel: "threshold t", ylabel: "average loss", ariaLabel: "average loss against the threshold" });
  const wide = htmlEl("div", "hl-wide", "", plots);
  const pB = new Plot(wide, { width: 860, height: 250, x: [0, 10], y: [-0.7, 4], xlabel: "feature x   (drag to move t)", ylabel: "loss", yticks: [0, 1, 2, 3, 4], ariaLabel: "data on a line with each point's loss" });

  htmlEl("div", "hl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "hl-lab", "loss functions shown", panel);
  const cbBox = htmlEl("div", "", "", panel);
  const cbs = LOSSES.map((L, i) => checkbox(cbBox, L.name, st.on[i], v => { st.on[i] = v; draw(); }));
  cbs.forEach((cb, i) => { (cb.parentElement as HTMLElement).style.color = LOSSES[i].color; (cb.parentElement as HTMLElement).style.fontWeight = "600"; });
  htmlEl("div", "hl-lab", "bars under each point", panel);
  const showBtns = presets(panel, LOSSES.map((L, i) => ({ label: L.name, apply: () => { st.show = i; draw(); } })));
  const aS: Slider = slider(panel, "slope a", 0.2, 3, 0.1, st.a, v => { st.a = v; draw(); });
  const tS: Slider = slider(panel, "threshold t", 0, 10, 0.05, st.t, v => { st.t = v; draw(); });
  htmlEl("div", "hl-lab", "move t to the minimum of", panel);
  const minRow = htmlEl("div", "qb-presets", "", panel);
  LOSSES.forEach((L, i) => button(minRow, L.name, () => { st.t = argminT(st.pts, i, st.a); tS.set(st.t); draw(); }));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);


  // drag on the data plot moves t
  let dragging = false;
  pB.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; pB.svg.setPointerCapture(ev.pointerId); moveT(ev); });
  pB.svg.addEventListener("pointermove", ev => { if (dragging) moveT(ev); });
  const up = () => { dragging = false; };
  pB.svg.addEventListener("pointerup", up); pB.svg.addEventListener("pointercancel", up);
  function moveT(ev: PointerEvent): void { const [x] = pB.toData(ev); st.t = Math.max(0, Math.min(10, Math.round(x * 20) / 20)); tS.set(st.t); draw(); }

  const ms = Array.from({ length: 601 }, (_, k) => -3 + k / 100);
  const ts = Array.from({ length: 401 }, (_, k) => k / 40);

  function draw(): void {
    const M = margins(st.pts, st.t, st.a);
    showBtns.forEach((b, i) => b.classList.toggle("active", i === st.show));
    // A: loss curves
    pA.clear();
    LOSSES.forEach((L, i) => {
      if (!st.on[i]) return;
      if (i === 0) pA.line([-3, 0, 0, 3], [1, 1, 0, 0], { color: L.color, width: 3.5, label: L.name });
      else pA.fn(L.f, { color: L.color, width: 3.5, label: L.name }, 300);
    });
    const Ls = LOSSES[st.show];
    pA.scatter(M.map(m => Math.max(-3, Math.min(3, m))), M.map(m => Math.min(4, Ls.f(m))), { colors: st.pts.map(p => QUALITATIVE[p.y > 0 ? 1 : 0]), r: 6, stroke: "#fff" });
    pA.vline(1, { color: "var(--qb-muted)", dash: "3 4", width: 1.5 });
    pA.legend(undefined, "tr");
    // C: average loss vs t
    pC.clear();
    let ymax = 1.2;
    const curves: [number, number[]][] = [];
    LOSSES.forEach((L, i) => { if (!st.on[i]) return; const v = ts.map(t => avgLoss(st.pts, i, t, st.a)); curves.push([i, v]); ymax = Math.max(ymax, 2.2 * Math.min(...v)); });
    pC.setDomain([0, 10], [0, Math.min(12, Math.ceil(ymax))]);
    curves.forEach(([i, v]) => pC.line(ts, v, { color: LOSSES[i].color, width: 3 }));
    pC.vline(st.t, { color: "var(--qb-fg)", width: 2, dash: "5 4" });
    // B: data, street, bars
    pB.clear();
    const lo = st.t - 1 / st.a, hi = st.t + 1 / st.a;
    pB.band([lo, hi], [-0.7, -0.7], [4, 4], { color: "#bbb", fill: "#bbb", fillOpacity: 0.25 });
    pB.hline(0, { color: "var(--qb-axis)", width: 1.5 });
    pB.bars(st.pts.map(p => p.x), M.map(m => Math.min(4, Ls.f(m))), { colors: st.pts.map(p => QUALITATIVE[p.y > 0 ? 1 : 0]), barWidth: 0.16, opacity: 0.55 });
    pB.scatter(st.pts.map(p => p.x), st.pts.map(() => -0.35), { colors: st.pts.map(p => QUALITATIVE[p.y > 0 ? 1 : 0]), r: 8, stroke: "#fff" });
    pB.vline(st.t, { color: "var(--qb-fg)", width: 3, text: "t" });
    // readout
    const lines = LOSSES.map(L => `${L.name.padEnd(9)} ${M.reduce((s, m) => s + L.f(m), 0).toFixed(2).padStart(6)}`);
    ro.textContent = `f(x) = ${st.a.toFixed(1)} (x − ${st.t.toFixed(2)})\nstreet: ${lo.toFixed(2)} < x < ${hi.toFixed(2)}\ntotal loss of the ${M.length} points:\n` + lines.join("\n") + (st.pts.length <= 6 ? "\ny·f: " + M.map(m => m.toFixed(2)).join(", ") : "");
    note.textContent = st.note;
  }

  function load(p: Preset): void {
    st.pts = p.pts; st.t = p.t; st.a = p.a; st.on = [...p.on]; st.show = p.show; st.note = p.note;
    cbs.forEach((cb, i) => (cb.checked = st.on[i]));
    aS.set(st.a); tS.set(st.t);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }
  load(p0);
}

autoMount(NAME, mount);
