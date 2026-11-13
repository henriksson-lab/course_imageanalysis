// Nearest neighbours and empty space: the G, F and J functions and the edge effect. The
// pattern comes from the shared source (csrplay/source.ts). Tabs:
//  - G: every point with an arrow to its nearest neighbour; Ĝ(r) against the CSR curve;
//  - F: a grid of test locations with arrows to the nearest point over the distance map; F̂(r);
//  - J: Ĵ = (1 − Ĝ)/(1 − F̂) against 1.
// Correction none / border (reduced sample, guard zone drawn); window square / small / band /
// tissue mask; click a point to see its distance and whether the border rule keeps it at r.
// "average of 100 CSR patterns" plots the mean curve of CSR patterns with the same n in the same
// window, with and without the correction (the bias).
// Auto-mounts on <div data-demo="gfj">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount, svgEl, checkbox } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rng } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { clarkEvans } from "../lib/neighbours";
import { Spec, DEFAULT_SPEC, makePattern, makeWindow, PatternView, sourceControls, injectStyle, sourceCss, heatUrl, WinKind } from "../csrplay/source";

const NAME = "gfj";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = sourceCss(NAME) + `
${P} .gf-view { flex: 0 1 360px; min-width: 240px; }
${P} .gf-main { flex: 1 1 360px; min-width: 260px; max-width: 520px; display: flex; flex-direction: column; gap: .3em; }
${P} .qb-panel { flex: 1 1 280px; gap: .35em; }
${P} .gf-cap { font-size: .9em; text-align: center; }
.reveal ${P} .gf-view { flex: 0 0 370px; min-width: 0; }
.reveal ${P} .gf-main { flex: 0 0 420px; min-width: 0; }
.reveal ${P} .qb-panel { flex: 1 1 320px; min-width: 0; gap: .15em; }
`;

type Tab = "G" | "F" | "J";
type Corr = PP.NNCorr;
interface State { spec: Spec; tab: Tab; corr: Corr; r: number; avg: boolean; sel: number }
const DEFAULT: State = { spec: { ...DEFAULT_SPEC }, tab: "G", corr: "border", r: 6, avg: false, sel: -1 };
const RMAX = 30, RADII = PP.linspace(0, RMAX, 61);

const PRESETS: { id: string; label: string; s: Partial<State>; spec?: Partial<Spec> }[] = [
  { id: "csr", label: "CSR: G = F = theory", s: { tab: "G" }, spec: { model: "csr" } },
  { id: "clusters", label: "clusters: G up, F down", s: { tab: "F" }, spec: { model: "thomas", k: 20, mu: 8, sigma: 4 } },
  { id: "regular", label: "regular: the opposite", s: { tab: "G" }, spec: { model: "hardcore", n: 150, h: 10 } },
  { id: "J", label: "J needs no ρ", s: { tab: "J" }, spec: { model: "thomas", k: 20, mu: 8, sigma: 4 } },
  { id: "pairs", label: "tight pairs", s: { tab: "G", r: 3 }, spec: { model: "pairs", pairD: 3 } },
  { id: "small", label: "small window: no correction", s: { tab: "G", corr: "none", avg: true, r: 10 }, spec: { model: "csr", win: "small" } },
  { id: "border", label: "border correction", s: { tab: "G", corr: "border", avg: true, r: 10 }, spec: { model: "csr", win: "small" } },
  { id: "band", label: "thin tissue band", s: { tab: "G", corr: "none", avg: true, r: 8 }, spec: { model: "csr", win: "band", n: 400 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 22 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, spec: { ...DEFAULT_SPEC } };
  if (p0) { Object.assign(st, p0.s); Object.assign(st.spec, p0.spec ?? {}); }
  else if (host.dataset.tab) st.tab = host.dataset.tab as Tab;

  const viewBox = htmlEl("div", "gf-view", "", host);
  const cap = htmlEl("div", "gf-cap", "", viewBox);
  const view = new PatternView(viewBox);
  const main = htmlEl("div", "gf-main", "", host);
  const tabsEl = htmlEl("div", "ss-tabs", "", main);
  const TABS: [Tab, string][] = [["G", "G: nearest neighbour"], ["F", "F: empty space"], ["J", "J"]];
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(); }));
  const fig = plotFigure(main, "", "qb-figure");
  const pl = new Plot(fig, { width: 460, height: 380, x: [0, RMAX], y: [0, 1], font: FS, margin: { l: 3.6 * FS, b: 2.7 * FS, t: 10, r: 14 } });
  const corrRow = htmlEl("div", "qb-presets", "", main);
  const corrBtns = (["none", "border"] as Corr[]).map(c => button(corrRow, c === "none" ? "no correction" : "border (reduced sample)", () => { st.corr = c; unp(); draw(); }));
  const avgBox = checkbox(corrRow, "average of 100 CSR patterns", st.avg, v => { st.avg = v; unp(); draw(); });

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const rSl = slider(panel, "r (µm)", 0, RMAX, 0.5, st.r, v => { st.r = v; unp(); draw(); });
  const src = sourceControls(panel, st.spec, () => { unp(); st.sel = -1; regenerate(); draw(); }, { windows: ["square", "small", "band", "tissue"] as WinKind[] });
  htmlEl("div", "ss-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [], activeId = p0?.id ?? "";
  function buildPresets(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => p.s.tab === st.tab) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => { b.dataset.id = list[i].id; b.classList.toggle("active", list[i].id === activeId); });
  }
  button(panel, "reset", () => applyPreset(null));

  let pts = makePattern(st.spec);
  let avgCache: { key: string; none: Float64Array; border: Float64Array } | null = null;
  function regenerate(): void { pts = makePattern(st.spec); }
  function unp(): void { activeId = ""; pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    const { spec } = st;
    Object.assign(st, DEFAULT, p?.s ?? {}, { spec });
    Object.assign(spec, DEFAULT_SPEC, p?.spec ?? {});
    src.sync(); rSl.set(st.r); avgBox.checked = st.avg;
    activeId = p?.id ?? "";
    regenerate(); draw();
  }

  view.svg.addEventListener("pointerdown", ev => {
    const [x, y] = view.toWorld(ev);
    st.sel = PatternView.nearest(pts, x, y, 8 * view.win.w / 200 + 2);
    draw();
  });

  /** Mean Ĝ or F̂ of 100 CSR patterns with the same n in this window, both corrections. */
  function averageCurves(win: PP.Window, tab: "G" | "F"): { none: Float64Array; border: Float64Array } {
    const key = `${st.spec.win}|${pts.x.length}|${tab}`;
    if (avgCache && avgCache.key === key) return avgCache;
    const r = rng(99), none = new Float64Array(RADII.length), border = new Float64Array(RADII.length), cnt = new Float64Array(RADII.length);
    const T = 100;
    for (let t = 0; t < T; t++) {
      const q = PP.binomialProcess(r, pts.x.length, win);
      const a = tab === "G" ? PP.Gest(q, win, RADII, "none") : PP.Fest(q, win, RADII, "none", win.w / 40);
      const b = tab === "G" ? PP.Gest(q, win, RADII, "border") : PP.Fest(q, win, RADII, "border", win.w / 40);
      for (let k = 0; k < RADII.length; k++) { none[k] += a[k] / T; if (Number.isFinite(b[k])) { border[k] += b[k]; cnt[k]++; } }
    }
    for (let k = 0; k < RADII.length; k++) border[k] = cnt[k] > T / 2 ? border[k] / cnt[k] : NaN;
    avgCache = { key, none, border };
    return avgCache;
  }

  function guardZone(win: PP.Window, r: number): void {
    if (st.corr !== "border" || r <= 0) return;
    if (!win.mask) {
      const d = `M0,0H${win.w}V${win.h}H0Z M${r},${r}V${win.h - r}H${win.w - r}V${r}Z`;
      svgEl("path", { d, fill: "var(--qb-c2)", "fill-opacity": 0.13, "fill-rule": "evenodd" }, view.under);
      svgEl("rect", { x: r, y: r, width: Math.max(0, win.w - 2 * r), height: Math.max(0, win.h - 2 * r), fill: "none", stroke: "var(--qb-c2)", "stroke-dasharray": "3 2", "stroke-width": 0.6 * win.w / 200 }, view.under);
    } else {
      const v = new Float64Array(win.w * win.h);
      for (let i = 0; i < v.length; i++) v[i] = win.mask[i] && win.bdist![i] < r ? 1 : NaN;
      const c = document.createElement("canvas"); c.width = win.w; c.height = win.h;
      const ctx = c.getContext("2d")!, id = ctx.createImageData(win.w, win.h);
      for (let i = 0; i < v.length; i++) if (v[i] === 1) { id.data[4 * i] = 214; id.data[4 * i + 1] = 39; id.data[4 * i + 2] = 40; id.data[4 * i + 3] = 45; }
      ctx.putImageData(id, 0, 0);
      svgEl("image", { x: 0, y: 0, width: win.w, height: win.h, href: c.toDataURL(), preserveAspectRatio: "none" }, view.under);
    }
  }

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const at = (c: ArrayLike<number>) => c[Math.round(st.r / (RMAX / (RADII.length - 1)))];

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    corrBtns.forEach((b, i) => b.classList.toggle("active", (["none", "border"] as Corr[])[i] === st.corr));
    buildPresets();
    const win = makeWindow(st.spec.win), n = pts.x.length, rho = n / win.area, s = win.w / 200;
    view.setWindow(win);
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    const { d, idx } = PP.nnDistances(pts);
    const b = Float64Array.from(pts.x, (x, i) => PP.borderDistance(win, x, pts.y[i]));
    guardZone(win, st.r);
    const step = win.w / (st.spec.win === "small" ? 6 : 16);
    if (st.tab === "F") {
      // distance map under the view, grid locations with arrows
      const N = 80, dm = new Float64Array(N * N);
      const loc = PP.makePoints(Array.from({ length: N * N }, (_, k) => ((k % N) + 0.5) * win.w / N), Array.from({ length: N * N }, (_, k) => (Math.floor(k / N) + 0.5) * win.h / N));
      const e = PP.emptySpaceDistances(loc, pts);
      for (let k = 0; k < N * N; k++) dm[k] = PP.inside(win, loc.x[k], loc.y[k]) ? e[k] : NaN;
      view.under.replaceChildren();
      svgEl("image", { x: 0, y: 0, width: win.w, height: win.h, href: heatUrl(dm, N, N, 0, Math.max(RMAX, 1), 200), preserveAspectRatio: "none" }, view.under);
      guardZone(win, st.r);
      const g = PP.gridLocations(win, step), ge = PP.emptySpaceDistances(g, pts);
      for (let k = 0; k < g.x.length; k++) {
        let best = Infinity, bi = 0;
        for (let i = 0; i < n; i++) { const q = (pts.x[i] - g.x[k]) ** 2 + (pts.y[i] - g.y[k]) ** 2; if (q < best) { best = q; bi = i; } }
        const used = st.corr === "none" || PP.borderDistance(win, g.x[k], g.y[k]) >= st.r;
        const col = ge[k] <= st.r ? "#ffd400" : "#fff";
        svgEl("line", { x1: g.x[k], y1: g.y[k], x2: pts.x[bi], y2: pts.y[bi], stroke: col, "stroke-width": 0.7 * s, opacity: used ? 1 : 0.35 }, view.mid);
        svgEl("rect", { x: g.x[k] - 1.6 * s, y: g.y[k] - 1.6 * s, width: 3.2 * s, height: 3.2 * s, fill: col, stroke: "#000", "stroke-width": 0.4 * s, opacity: used ? 1 : 0.35 }, view.mid);
      }
    } else {
      for (let i = 0; i < n; i++) {
        if (idx[i] < 0) continue;
        const used = st.corr === "none" || b[i] >= st.r;
        svgEl("line", { x1: pts.x[i], y1: pts.y[i], x2: pts.x[idx[i]], y2: pts.y[idx[i]], stroke: d[i] <= st.r ? "var(--qb-c4)" : "var(--qb-muted)", "stroke-width": 0.9 * s, opacity: used ? 1 : 0.3 }, view.mid);
      }
    }
    view.setPoints(pts, i => (st.corr === "border" && b[i] < st.r ? "var(--qb-muted)" : "var(--qb-c1)"));
    if (st.sel >= 0 && st.sel < n) {
      const i = st.sel;
      svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: Math.max(0.01, st.r), fill: "none", stroke: "var(--qb-c2)", "stroke-width": 1.2 * s }, view.over);
      if (idx[i] >= 0) svgEl("line", { x1: pts.x[i], y1: pts.y[i], x2: pts.x[idx[i]], y2: pts.y[idx[i]], stroke: "var(--qb-c2)", "stroke-width": 1.6 * s }, view.over);
      svgEl("circle", { cx: pts.x[i], cy: pts.y[i], r: 3.4 * s, fill: "var(--qb-c2)" }, view.over);
    }
    cap.textContent = `${n} points · ${win.mask ? "mask" : `${win.w} × ${win.h} µm`}`;

    // curves
    pl.clear();
    const G = PP.Gest(pts, win, RADII, st.corr), F = PP.Fest(pts, win, RADII, st.corr, win.w / 50), J = PP.Jfun(G, F);
    const csr = (r: number) => PP.csrG(r, rho);
    if (st.tab === "J") {
      const jm = Math.min(4, Math.max(2, ...Array.from(J).filter(Number.isFinite)) * 1.1);
      pl.setDomain([0, RMAX], [0, jm]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: "J(r)" });
      pl.hline(1, { color: "var(--qb-muted)", dash: "7 5", width: 2.5, label: "CSR: J = 1" });
      pl.line(RADII, J, { color: "var(--qb-c5)", label: "Ĵ observed" });
    } else {
      pl.setDomain([0, RMAX], [0, 1.02]);
      pl.setLabels({ xlabel: "r (µm)", ylabel: st.tab === "G" ? "G(r)" : "F(r)" });
      pl.fn(csr, { color: "var(--qb-muted)", dash: "7 5", width: 2.5, label: "CSR, same ρ" });
      if (st.avg) {
        const a = averageCurves(win, st.tab);
        pl.line(RADII, a.none, { color: "var(--qb-c2)", width: 2.5, label: "CSR mean, no corr." });
        pl.line(RADII, a.border, { color: "var(--qb-c3)", width: 2.5, label: "CSR mean, border" });
      }
      pl.line(RADII, st.tab === "G" ? G : F, { color: st.tab === "G" ? "var(--qb-c1)" : "var(--qb-c4)", label: (st.tab === "G" ? "Ĝ" : "F̂") + " observed" });
    }
    pl.vline(st.r, { color: "var(--qb-c2)", width: 1.5, dash: "3 3" });
    pl.legend(undefined, "br");
    const kept = Array.from(b).filter(v => v >= st.r).length, near = n - kept;
    const ce = clarkEvans(d, win.area).R;
    let txt = `n ${n}, ρ = ${(1000 * rho).toFixed(2)} per 1000 µm²\nClark–Evans R ${fmt(ce, 2)}\nat r = ${st.r} µm:  CSR ${fmt(csr(st.r))}\n  Ĝ ${fmt(at(G))}  F̂ ${fmt(at(F))}  Ĵ ${fmt(at(J), 2)}\n≥ r from the border: ${kept} of ${n}\n(${Math.round(100 * near / Math.max(1, n))} % of the points are closer)`;
    if (st.sel >= 0 && st.sel < n) txt += `\nselected: NN ${fmt(d[st.sel], 1)} µm, border ${fmt(b[st.sel], 1)} µm\n  ${st.corr === "none" ? "used" : b[st.sel] >= st.r ? "kept by the border rule" : "dropped by the border rule"}`;
    ro.textContent = txt;
  }

  draw();
}

autoMount(NAME, mount);
