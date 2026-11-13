// FRET as a ruler. Left: a donor (cyan) and an acceptor (yellow) drawn to scale on a 1 nm
// grid; drag the acceptor, drag the tips of the two dipole arrows to rotate them. Top
// middle: excitation (dashed) and emission (solid) spectra of both, the overlap of donor
// emission and acceptor excitation shaded; the acceptor's spectra shift along λ with a
// slider. Bottom middle: E(r) = 1/(1 + (r/R0)^6) with the sensitive window 0.5–1.5 R0 and
// the current molecule; in population mode 200 seeded molecules with a distance
// distribution (normal, or two distances) and optionally random static orientations, with
// the histogram of r, the mean E and the apparent distance against the mean distance.
// Panel: presets, sliders, readouts and two bars (donor emission, quenched; acceptor
// emission, sensitised; both under donor excitation, relative to the donor alone).
//
// <div data-demo="fretruler" data-preset="half|twice|halfR0|window|overlap|perp|k4|mixture|spread"
//      data-mode="single|population"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, svgPoint, rng, Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { emission } from "../lib/unmix";
import * as F from "../lib/fret";
import { injectStyle, fmt, filterPresets, hideControl } from "../threecube/view";

const NAME = "fretruler";
const Q = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${Q} { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
${Q} .rl-mol { flex: 0 1 min(360px, calc(100vh - 240px)); min-width: 240px; }
${Q} .rl-mol svg { width: 100%; display: block; touch-action: none; }
${Q} .rl-plots { flex: 1 1 300px; min-width: 240px; max-width: 380px; display: flex; flex-direction: column; gap: .2em; }
${Q} .qb-panel { flex: 0 0 280px; gap: .3em; font-size: 15px; }
${Q} .qb-readout { font-size: 14px; }
${Q} .rl-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
.reveal ${Q} { font-size: 18px; flex-wrap: nowrap; margin: 0; }
.reveal ${Q} .rl-mol { flex: 0 0 360px; }
.reveal ${Q} .rl-plots { flex: 0 0 380px; }
.reveal ${Q} .qb-panel { flex: 0 0 340px; gap: .15em; }
.reveal ${Q} .qb-btn { font-size: 18px; padding: .02em .3em; }
.reveal ${Q} .qb-readout { font-size: 18px; padding: .2em .4em; line-height: 1.22; }
.reveal ${Q} .rl-lab { display: none; }
`;

const D0 = F.CYAN_DONOR, A0 = F.YELLOW_ACCEPTOR, N_REFR = 1.4, NMOL = 200;
const VIEW = 12;                       // nm across the molecule panel
const DONOR_POS: [number, number] = [3, 6];
const R_CONTACT = 2.4;                 // two barrels side by side

interface State {
  ax: number; ay: number;              // acceptor position, nm
  thD: number; thA: number;            // dipole angles (rad, counter-clockwise on screen)
  tumbling: boolean;                   // κ² = 2/3
  shift: number;                       // acceptor spectra shift (nm)
  QD: number;
  pop: boolean; dist: "normal" | "two"; mean: number; width: number; randomOrient: boolean;
}
const R00 = (() => F.forsterRadius(D0.Q, 2 / 3, N_REFR, F.overlapIntegral(D0.em, A0.ex, A0.eps)))();
const DEFAULT: State = { ax: DONOR_POS[0] + R00, ay: 6, thD: Math.PI / 2, thA: Math.PI / 2, tumbling: true, shift: 0, QD: D0.Q, pop: false, dist: "normal", mean: R00, width: 0.8, randomOrient: false };
const at = (rr: number): Partial<State> => ({ ax: DONOR_POS[0] + rr, ay: 6 });
const r1 = (v: number) => Math.round(v * 100) / 100;
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "half", label: "r = R₀: half", s: { ...at(R00) } },
  { id: "twice", label: "twice R₀: almost nothing", s: { ...at(2 * R00) } },
  { id: "halfR0", label: "half R₀: almost everything", s: { ...at(0.5 * R00) } },
  { id: "window", label: "the sensitive window", s: { ...at(1.25 * R00) } },
  { id: "overlap", label: "poor overlap: a smaller R₀", s: { ...at(R00), shift: 60 } },
  { id: "perp", label: "perpendicular dipoles: κ² = 0", s: { ...at(R00), tumbling: false, thD: Math.PI / 2, thA: 0 } },
  { id: "k4", label: "κ² = 4: same r, more FRET", s: { ...at(1.3 * R00), tumbling: false, thD: 0, thA: 0 } },
  { id: "mixture", label: "two distances: mean E ≠ E(mean r)", s: { pop: true, dist: "two", mean: r1(0.75 * R00), width: r1(0.25 * R00) } },
  { id: "spread", label: "a spread of distances", s: { pop: true, dist: "normal", mean: r1(R00), width: 1.2 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const pre = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.mode === "population" ? { pop: true } : {}), ...(pre?.s ?? {}) });
  const st: State = start();

  // ---- molecule panel ----
  const molBox = htmlEl("div", "rl-mol", "", host);
  const S = 400 / VIEW;                // px per nm
  const svg = svgEl("svg", { viewBox: "0 0 400 400" }, molBox);
  const X = (v: number) => v * S, Y = (v: number) => v * S;
  for (let i = 0; i <= VIEW; i++) {
    const l1 = svgEl("line", { x1: X(i), y1: 0, x2: X(i), y2: 400 }, svg), l2 = svgEl("line", { x1: 0, y1: Y(i), x2: 400, y2: Y(i) }, svg);
    for (const l of [l1, l2]) { l.style.stroke = "var(--qb-grid)"; l.style.strokeWidth = "1"; }
  }
  const scaleBar = svgEl("g", {}, svg);
  svgEl("line", { x1: X(0.5), y1: Y(11.4), x2: X(2.5), y2: Y(11.4), stroke: "#555", "stroke-width": 4 }, scaleBar);
  const sbt = svgEl("text", { x: X(1.5), y: Y(11.4) - 8, "text-anchor": "middle", "font-size": 18 }, scaleBar); sbt.textContent = "2 nm";
  const popG = svgEl("g", {}, svg);
  const windowRing = svgEl("circle", { cx: X(DONOR_POS[0]), cy: Y(DONOR_POS[1]), r: 1, fill: "none", "stroke-width": 2, "stroke-dasharray": "6 5" }, svg);
  windowRing.style.stroke = "#888";
  const r0Ring = svgEl("circle", { cx: X(DONOR_POS[0]), cy: Y(DONOR_POS[1]), r: 1, fill: "none", "stroke-width": 2.5 }, svg);
  r0Ring.style.stroke = "#7b3294";
  const r0Lab = svgEl("text", { "font-size": 18, "text-anchor": "middle" }, svg);
  r0Lab.style.fill = "#7b3294";
  const link = svgEl("line", { "stroke-width": 3, "stroke-dasharray": "4 4" }, svg);
  link.style.stroke = "#444";
  const mkMol = (fill: string, label: string) => {
    const g = svgEl("g", {}, svg);
    const c = svgEl("circle", { r: S * R_CONTACT / 2, fill, stroke: "#333", "stroke-width": 2, "fill-opacity": 0.85 }, g);
    const t = svgEl("text", { "text-anchor": "middle", "dominant-baseline": "central", "font-size": 22, "font-weight": 700 }, g);
    t.textContent = label; t.style.fill = "#111";
    const arr = svgEl("line", { "stroke-width": 5, stroke: "#111", "stroke-linecap": "round" }, g);
    const tip = svgEl("circle", { r: 11, fill: "#fff", stroke: "#111", "stroke-width": 3 }, g);
    tip.style.cursor = "grab";
    return { g, c, t, arr, tip };
  };
  const donor = mkMol("#3fc1d9", "D"), acc = mkMol("#f2d22e", "A");
  acc.c.style.cursor = "grab";
  const molNote = htmlEl("div", "rl-lab", "drag the acceptor; drag the white tips to turn the dipoles", molBox);
  void molNote;

  // ---- plots ----
  const plots = htmlEl("div", "rl-plots", "", host);
  const RV = !!host.closest(".reveal");
  const sp = new Plot(plots, { width: RV ? 380 : 400, height: RV ? 200 : 230, x: [380, 620], y: [0, 1.05], xlabel: "wavelength λ (nm)", ylabel: "relative", xticks: [400, 450, 500, 550, 600], yticks: [0, 0.5, 1], margin: { l: 58, r: 10, t: 8, b: 52 } });
  const L = Float64Array.from({ length: 241 }, (_, i) => 380 + i);
  const zero = new Float64Array(L.length);
  const ovl = sp.band(L, zero, zero, { color: "#7b3294", fillOpacity: 0.35 });
  const DCOL = "#1b9fc0", ACOL = "#d39e00";
  const dEx = sp.line(L, zero, { color: DCOL, width: 3, dash: "7 5" }), dEm = sp.line(L, zero, { color: DCOL, width: 4 });
  const aEx = sp.line(L, zero, { color: ACOL, width: 3, dash: "7 5" }), aEm = sp.line(L, zero, { color: ACOL, width: 4 });
  sp.legend([{ label: "excitation", color: "#555", dash: "7 5" }, { label: "emission", color: "#555" }, { label: "overlap", color: "#7b3294", kind: "box" }], "tr");

  const ep = new Plot(plots, { width: RV ? 380 : 400, height: RV ? 240 : 270, x: [0, 12], y: [0, 1.02], xlabel: "distance r (nm)", ylabel: "efficiency E", xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [0, 0.5, 1], margin: { l: 58, r: 10, t: 8, b: 52 } });
  const win = ep.band([0, 1], [0, 0], [1.02, 1.02], { color: "#7b3294", fillOpacity: 0.12 });
  const hist = ep.mark<[number[], number[]]>((g, edges, counts) => {
    const mx = Math.max(1, ...counts);
    for (let i = 0; i < counts.length; i++) {
      if (!counts[i]) continue;
      const x0 = ep.x(edges[i]), x1 = ep.x(edges[i + 1]), y0 = ep.y(0), y1 = ep.y(0.35 * counts[i] / mx);
      const r = svgEl("rect", { x: x0, y: y1, width: Math.max(0.5, x1 - x0 - 1), height: y0 - y1 }, g);
      r.style.fill = "#999"; r.style.opacity = "0.55";
    }
  }, [[], []]);
  const curve = ep.fn(r => F.efficiency(r, R00), { color: "var(--qb-c1)", width: 3.5 });
  const dots = ep.scatter([], [], { color: "#d62728", r: 3.5, opacity: 0.45 });
  const vMean = ep.vline(-1, { color: "#2ca02c", width: 3 }), vApp = ep.vline(-1, { color: "#d62728", width: 3, dash: "6 4", text: "apparent" });
  const hMean = ep.hline(-1, { color: "#d62728", width: 2, dash: "3 4" });
  const point = ep.scatter([0], [0], { color: "#111", r: 9 });

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const PL = filterPresets(host, PRESETS);
  const presetBtns = presets(panel, PL.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); draw(); } })));
  const shiftS = slider(panel, "acceptor shift (nm)", -40, 80, 1, st.shift, v => { st.shift = v; manual(); });
  const qS = slider(panel, "donor yield Q_D", 0.05, 1, 0.01, st.QD, v => { st.QD = v; manual(); });
  const tumC = checkbox(panel, "freely tumbling dyes: κ² = 2/3", st.tumbling, v => { st.tumbling = v; manual(); });
  const popC = checkbox(panel, "population of 200 molecules", st.pop, v => { st.pop = v; manual(); });
  const popBox = htmlEl("div", "", "", panel);
  popBox.style.display = "contents";
  const meanS: Slider = slider(popBox, "mean r (nm)", 1, 10, 0.05, st.mean, v => { st.mean = v; manual(); });
  const widS: Slider = slider(popBox, "width (nm)", 0, 3, 0.05, st.width, v => { st.width = v; manual(); });
  const twoC = checkbox(popBox, "two distances (mean ± width)", st.dist === "two", v => { st.dist = v ? "two" : "normal"; manual(); });
  const oriC = checkbox(popBox, "random static orientations", st.randomOrient, v => { st.randomOrient = v; manual(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const bars = new Plot(plots, { width: RV ? 380 : 340, height: RV ? 100 : 120, x: [0, 1.25], y: [0, 2], yticks: [], xticks: [0, 0.5, 1], margin: { l: 112, r: 10, t: 4, b: 30 } });
  const barMark = bars.mark<[number, number]>((g, dv, av) => {
    const rows: [string, number, string][] = [["donor", dv, "#3fc1d9"], ["acceptor", av, "#e0b800"]];
    rows.forEach(([lab, v, c], i) => {
      const y0 = bars.y(1.9 - i), y1 = bars.y(1.1 - i);
      const r = svgEl("rect", { x: bars.x(0), y: y0, width: Math.max(0, bars.x(Math.min(1.25, v)) - bars.x(0)), height: y1 - y0 }, g);
      r.style.fill = c;
      const t = svgEl("text", { x: bars.box.l - 8, y: (y0 + y1) / 2 + 6, "text-anchor": "end", "font-size": 18 }, g);
      t.textContent = lab;
    });
  }, [1, 0], {}, "box", bars.over);

  const manual = () => { presetBtns.forEach(b => b.classList.remove("active")); draw(); };
  function sync(): void {
    shiftS.set(st.shift); qS.set(st.QD); tumC.checked = st.tumbling; popC.checked = st.pop;
    meanS.set(st.mean); widS.set(st.width); twoC.checked = st.dist === "two"; oriC.checked = st.randomOrient;
  }

  // ---- interaction ----
  const toNm = (ev: PointerEvent): [number, number] => { const [px, py] = svgPoint(svg, ev); return [px / S, py / S]; };
  const dragOn = (el: SVGElement, f: (x: number, y: number) => void) => {
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ev.stopPropagation(); el.setPointerCapture(ev.pointerId);
      const mv = (e: PointerEvent) => { f(...toNm(e)); };
      const up = () => { el.removeEventListener("pointermove", mv); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); };
      el.addEventListener("pointermove", mv); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    });
  };
  dragOn(acc.c, (x, y) => {
    let dx = x - DONOR_POS[0], dy = y - DONOR_POS[1];
    const d = Math.hypot(dx, dy);
    if (d < R_CONTACT) { const k = R_CONTACT / (d || 1); dx = (d ? dx : 1) * k; dy = dy * k; }
    st.ax = Math.min(VIEW - 0.3, Math.max(0.3, DONOR_POS[0] + dx)); st.ay = Math.min(VIEW - 0.3, Math.max(0.3, DONOR_POS[1] + dy));
    manual();
  });
  const ARM = 1.6;
  dragOn(donor.tip, (x, y) => { st.thD = Math.atan2(-(y - DONOR_POS[1]), x - DONOR_POS[0]); st.tumbling = false; tumC.checked = false; manual(); });
  dragOn(acc.tip, (x, y) => { st.thA = Math.atan2(-(y - st.ay), x - st.ax); st.tumbling = false; tumC.checked = false; manual(); });

  // ---- model ----
  const accEx = (): F.Absorber => ({ ...A0.ex, peak: A0.ex.peak + st.shift });
  const accEm = () => ({ ...A0.em, peak: A0.em.peak + st.shift });
  const kOf = (): number => {
    const dir: F.V3 = [st.ax - DONOR_POS[0], -(st.ay - DONOR_POS[1]), 0];
    return F.kappa2([Math.cos(st.thD), Math.sin(st.thD), 0], [Math.cos(st.thA), Math.sin(st.thA), 0], dir);
  };

  function population(R0: number, k2fixed: number, J: number): { r: number[]; E: number[]; ang: number[] } {
    const g = rng(20261113), go = rng(77), r: number[] = [], E: number[] = [], ang: number[] = [];
    const unitv = (): F.V3 => { for (;;) { const v: F.V3 = [2 * go.unif() - 1, 2 * go.unif() - 1, 2 * go.unif() - 1]; const n = Math.hypot(...v); if (n > 0.05 && n <= 1) return v; } };
    for (let i = 0; i < NMOL; i++) {
      const z = g.norm(), u = g.unif();
      let ri = st.dist === "two" ? st.mean + (i % 2 ? 1 : -1) * st.width : st.mean + st.width * z;
      ri = Math.max(1, ri);
      const R0i = st.randomOrient ? F.forsterRadius(st.QD, F.kappa2(unitv(), unitv(), [1, 0, 0]), N_REFR, J) : (st.tumbling ? R0 : F.forsterRadius(st.QD, k2fixed, N_REFR, J));
      r.push(ri); E.push(R0i > 0 ? F.efficiency(ri, R0i) : 0); ang.push(2 * Math.PI * u);
    }
    return { r, E, ang };
  }

  function draw(): void {
    const J = F.overlapIntegral(D0.em, accEx(), A0.eps);
    const R0iso = F.forsterRadius(st.QD, 2 / 3, N_REFR, J);
    const k2 = st.tumbling ? 2 / 3 : kOf();
    const R0 = F.forsterRadius(st.QD, k2, N_REFR, J);
    const rr = Math.hypot(st.ax - DONOR_POS[0], st.ay - DONOR_POS[1]);
    const E = R0 > 0 ? F.efficiency(rr, R0) : 0;

    // spectra
    const exD = L.map(l => F.absorption(D0.ex, l)), emD = L.map(l => emission(D0.em, l));
    const exA = L.map(l => F.absorption(accEx(), l)), emA = L.map(l => emission(accEm(), l));
    dEx.set(L, exD); dEm.set(L, emD); aEx.set(L, exA); aEm.set(L, emA);
    ovl.set(L, zero, L.map((_, i) => Math.min(emD[i], exA[i])));

    // molecules
    const place = (m: typeof donor, x: number, y: number, th: number, show: boolean) => {
      svgSet(m.c, { cx: X(x), cy: Y(y) }); svgSet(m.t, { x: X(x), y: Y(y) });
      const tx = x + ARM * Math.cos(th), ty = y - ARM * Math.sin(th), bx = x - ARM * Math.cos(th), by = y + ARM * Math.sin(th);
      svgSet(m.arr, { x1: X(bx), y1: Y(by), x2: X(tx), y2: Y(ty) }); svgSet(m.tip, { cx: X(tx), cy: Y(ty) });
      m.arr.style.display = m.tip.style.display = show ? "" : "none";
    };
    const showArrows = !st.tumbling && !st.pop;
    place(donor, DONOR_POS[0], DONOR_POS[1], st.thD, showArrows);
    place(acc, st.ax, st.ay, st.thA, showArrows);
    acc.g.style.display = st.pop ? "none" : "";
    svgSet(link, { x1: X(DONOR_POS[0]), y1: Y(DONOR_POS[1]), x2: X(st.ax), y2: Y(st.ay) });
    link.style.display = st.pop ? "none" : "";
    svgSet(r0Ring, { r: R0 * S }); svgSet(windowRing, { r: 1.5 * R0 * S });
    svgSet(r0Lab, { x: X(DONOR_POS[0]), y: Y(DONOR_POS[1]) - R0 * S - 8 }); r0Lab.textContent = `R₀ = ${fmt(R0, 1)} nm`;

    // efficiency plot
    curve.set(r => (R0 > 0 ? F.efficiency(r, R0) : 0));
    win.set([0.5 * R0, 1.5 * R0], [0, 0], [1.02, 1.02]);
    popG.replaceChildren();
    const lines: string[] = [];
    if (st.pop) {
      const p = population(R0, k2, J);
      const edges = Array.from({ length: 49 }, (_, i) => i * 0.25), counts = new Array(48).fill(0);
      p.r.forEach(v => { const b = Math.floor(v / 0.25); if (b >= 0 && b < 48) counts[b]++; });
      hist.set(edges, counts);
      dots.set(p.r, p.E);
      const mE = p.E.reduce((a, b) => a + b, 0) / NMOL, mr = p.r.reduce((a, b) => a + b, 0) / NMOL;
      const app = F.distanceFromE(mE, R0iso);
      vMean.set(mr); vApp.set(app); hMean.set(mE);
      for (const m of [vMean, vApp, hMean]) m.g.style.display = "";
      point.set([], []);
      p.r.forEach((ri, i) => {
        const x = DONOR_POS[0] + ri * Math.cos(p.ang[i]), y = DONOR_POS[1] - ri * Math.sin(p.ang[i]);
        if (x < 0 || x > VIEW || y < 0 || y > VIEW) return;
        const c = svgEl("circle", { cx: X(x), cy: Y(y), r: 5, stroke: "#333", "stroke-width": 0.8 }, popG);
        c.style.fill = `rgb(${Math.round(255 * (0.25 + 0.7 * p.E[i]))},${Math.round(200 * (0.25 + 0.7 * p.E[i]))},40)`;
        c.style.opacity = "0.8";
      });
      barMark.set(1 - mE, mE * A0.Q / st.QD);
      lines.push(`R₀ = ${fmt(R0, 2)} nm (κ² = 2/3: ${fmt(R0iso, 2)})`);
      lines.push(`mean r (green) = ${fmt(mr, 2)} nm`);
      lines.push(`E(mean r) = ${fmt(F.efficiency(mr, R0), 3)}`);
      lines.push(`mean E (red dashes) = ${fmt(mE, 3)}`);
      lines.push(`apparent r (red) = ${fmt(app, 2)} nm`);
    } else {
      hist.set([], []); dots.set([], []);
      for (const m of [vMean, vApp, hMean]) m.g.style.display = "none";
      point.set([rr], [E]);
      barMark.set(1 - E, E * A0.Q / st.QD);
      lines.push(`r = ${fmt(rr, 2)} nm   κ² = ${fmt(k2, 2)}`);
      lines.push(`R₀ = ${fmt(R0, 2)} nm (κ² = 2/3: ${fmt(R0iso, 2)})`);
      lines.push(`E = ${fmt(E, 3)}   r/R₀ = ${fmt(R0 > 0 ? rr / R0 : Infinity, 2)}`);
      lines.push(`r_app (κ² 2/3) = ${fmt(F.distanceFromE(E, R0iso), 2)} nm`);
    }
    if (host.dataset.compact === undefined) lines.push(`J = ${(J / 1e15).toFixed(2)}×10¹⁵ M⁻¹cm⁻¹nm⁴`);
    ro.textContent = lines.join("\n");
    popBox.querySelectorAll<HTMLElement>(":scope > *").forEach(l => { l.style.display = st.pop ? "" : "none"; });
  }

  sync();
  draw();
  const pi = PL.findIndex(p => p.id === host.dataset.preset);
  if (pi >= 0) presetBtns[pi].classList.add("active");
  void hideControl;
}

function svgSet(e: Element, a: Record<string, number>): void { for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v)); }

autoMount(NAME, mount);
