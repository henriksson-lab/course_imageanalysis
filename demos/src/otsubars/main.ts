// Otsu by hand: an 8-level histogram as draggable bars (heights 0–9). A threshold t
// splits the levels into class 0 (0..t, blue) and class 1 (t+1..7, orange). The table
// lists ω0, μ0, μ1 and σ_B² for every t (current row highlighted, best starred); a
// stacked bar shows σ_T² = σ_W² + σ_B² (the total stays fixed while t moves). The
// "isodata step" button moves the isodata threshold to the midpoint of the class means.
// Auto-mounts on <div data-demo="otsubars">. Options: data-preset="example|equal|tiny|stuck", data-h="2 5 3 1 1 2 4 2".
import { htmlEl, slider, presets, button, autoMount, numList, svgEl } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { splitStats, otsu } from "../lib/threshold";
import { injectStyle } from "../histthresh/scenes";

const NAME = "otsubars";
const L = 8;
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .ob-left { flex: 1 1 400px; min-width: 280px; max-width: 470px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ob-left svg.qb-plot-svg { touch-action: none; cursor: ns-resize; }
[data-demo="${NAME}"] .ob-mid { flex: 0 1 330px; min-width: 260px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] table.ob-table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 15px; width: 100%; }
[data-demo="${NAME}"] .ob-table th, [data-demo="${NAME}"] .ob-table td { padding: .12em .45em; text-align: right; border-bottom: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .ob-table th { font-family: inherit; font-weight: 600; }
[data-demo="${NAME}"] .ob-table tr.cur td { background: rgba(255,127,14,.22); font-weight: 700; }
[data-demo="${NAME}"] .ob-table tr { cursor: pointer; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 260px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .ob-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .ob-table { font-size: 16px; }
`;

const PRESETS: { id: string; label: string; h: number[]; t: number; iso: number }[] = [
  { id: "example", label: "the worked example", h: [2, 5, 3, 1, 1, 2, 4, 2], t: 5, iso: 5.5 },
  { id: "equal", label: "two equal peaks", h: [1, 4, 6, 2, 2, 6, 4, 1], t: 1, iso: 1.5 },
  { id: "tiny", label: "a tiny object class", h: [3, 8, 9, 6, 3, 1, 0, 2], t: 5, iso: 5.5 },
  { id: "stuck", label: "isodata stuck", h: [9, 0, 0, 8, 7, 0, 0, 5], t: 6, iso: 6.5 },
];

const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const custom = host.dataset.h ? numList(host.dataset.h, p0.h).slice(0, L) : null;
  const st = { h: (custom ?? p0.h).slice(), t: p0.t, iso: p0.iso, trace: [p0.iso] as number[], showIso: false };

  const left = htmlEl("div", "ob-left", "", host);
  const plot = new Plot(left, { width: 470, height: 330, x: [-0.6, 7.6], y: [0, 10], xlabel: "grey level i", ylabel: "count hᵢ", xticks: [0, 1, 2, 3, 4, 5, 6, 7], yticks: [0, 2, 4, 6, 8, 10] });
  const stack = svgEl("svg", { viewBox: "0 0 470 74", width: "100%" }, left);
  const mid = htmlEl("div", "ob-mid", "", host);
  const table = htmlEl("table", "ob-table", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ob-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const tS = slider(panel, "t", 0, 6, 1, st.t, v => { st.t = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "isodata step", () => isoStep());
  button(row, "t = Otsu", () => { st.t = otsu(st.h).t; draw(); });
  button(row, "reset", () => load(p0, custom));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag a bar up or down to change its count. Click a table row to choose t.", panel);

  function load(p: typeof PRESETS[number], h?: number[] | null): void {
    st.h = (h ?? p.h).slice(); st.t = p.t; st.iso = p.iso; st.trace = [p.iso]; st.showIso = false;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p && !h));
    draw();
  }

  function isoStep(): void {
    const k = Math.min(L - 1, Math.max(-1, Math.floor(st.iso)));
    const s = splitStats(st.h, k);
    st.showIso = true;
    if (!(s.w0 > 0 && s.w1 > 0)) { draw(); return; }
    st.iso = (s.mu0 + s.mu1) / 2;
    st.trace.push(st.iso);
    st.t = Math.min(6, Math.max(0, Math.floor(st.iso)));
    draw();
  }

  // drag bars
  const svg = plot.svg;
  svg.style.touchAction = "none";
  let dragging = -1;
  const setFrom = (ev: PointerEvent) => {
    const [x, y] = plot.toData(ev);
    if (dragging < 0) dragging = Math.round(x);
    if (dragging < 0 || dragging >= L) return;
    st.h[dragging] = Math.max(0, Math.min(9, Math.round(y)));
    presetBtns.forEach(b => b.classList.remove("active"));
    st.trace = [st.iso];
    draw();
  };
  svg.addEventListener("pointerdown", ev => { ev.preventDefault(); svg.setPointerCapture(ev.pointerId); dragging = -1; setFrom(ev); });
  svg.addEventListener("pointermove", ev => { if (dragging >= 0) setFrom(ev); });
  const up = () => { dragging = -1; };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);

  function draw(): void {
    tS.set(st.t);
    const xs = Array.from({ length: L }, (_, i) => i);
    const s = splitStats(st.h, st.t), o = otsu(st.h);
    plot.clear();
    plot.bars(xs, st.h, { barWidth: 0.82, colors: xs.map(i => (i <= st.t ? "var(--qb-c1, #1f77b4)" : "var(--qb-c4, #ff7f0e)")) });
    for (let i = 0; i < L; i++) plot.text(i, st.h[i] + 0.25, String(st.h[i]), { anchor: "middle" });
    plot.vline(st.t + 0.5, { color: "var(--qb-fg, #222)", width: 3, dash: "", text: "t = " + st.t });
    if (Number.isFinite(s.mu0)) plot.scatter([s.mu0], [0.35], { color: "var(--qb-c1, #1f77b4)", r: 9 });
    if (Number.isFinite(s.mu1)) plot.scatter([s.mu1], [0.35], { color: "var(--qb-c4, #ff7f0e)", r: 9 });
    if (st.showIso) plot.vline(st.iso, { color: seriesColor(4), width: 2.5, dash: "4 4", text: "", });
    plot.text(-0.45, 9.3, "● class means", { size: 16 });

    // table
    table.replaceChildren();
    const hr = htmlEl("tr", "", "", table);
    for (const c of ["t", "ω₀", "μ₀", "μ₁", "σ<sub>B</sub>²"]) htmlEl("th", "", "", hr).innerHTML = c;
    for (let t = 0; t < L - 1; t++) {
      const q = splitStats(st.h, t), tr = htmlEl("tr", t === st.t ? "cur" : "", "", table);
      const ok = q.w0 > 0 && q.w1 > 0;
      htmlEl("td", "", String(t) + (t === o.t ? " ★" : ""), tr);
      htmlEl("td", "", q.w0.toFixed(2), tr);
      htmlEl("td", "", f2(q.mu0), tr);
      htmlEl("td", "", f2(q.mu1), tr);
      htmlEl("td", "", ok ? q.varB.toFixed(2) : "–", tr);
      tr.addEventListener("click", () => { st.t = t; draw(); });
    }

    // stacked bar σ_T² = σ_W² + σ_B²
    stack.replaceChildren();
    const W = 450, x0 = 10, tot = s.varT || 1, wW = W * s.varW / tot, wB = W * s.varB / tot;
    const rW = svgEl("rect", { x: x0, y: 30, width: wW, height: 34 }, stack);
    rW.style.fill = "var(--qb-grid, #ccc)"; rW.style.stroke = "var(--qb-axis, #444)";
    const rB = svgEl("rect", { x: x0 + wW, y: 30, width: wB, height: 34 }, stack);
    rB.style.fill = "var(--qb-c5, #9467bd)"; rB.style.fillOpacity = "0.75"; rB.style.stroke = "var(--qb-axis, #444)";
    const tt = svgEl("text", { x: x0, y: 22 }, stack);
    tt.textContent = `total ${s.varT.toFixed(2)} = within ${s.varW.toFixed(2)} + between ${s.varB.toFixed(2)}`;
    tt.style.fontSize = "18px";
    const lab = (x: number, w: number, str: string) => { if (w < 70) return; const e = svgEl("text", { x: x + w / 2, y: 53, "text-anchor": "middle" }, stack); e.textContent = str; e.style.fontSize = "17px"; };
    lab(x0, wW, "within"); lab(x0 + wW, wB, "between");

    const iso = st.trace.map(v => v.toFixed(2)).join(" → ");
    ro.textContent = `N = ${st.h.reduce((a, b) => a + b, 0)}   μ = ${s.mu.toFixed(2)}\nOtsu t* = ${o.t}  (σ_B² = ${o.t >= 0 ? o.curve[o.t].toFixed(2) : "–"})\nisodata: ${iso}`;
  }

  load(p0, custom);
}

autoMount(NAME, mount);
