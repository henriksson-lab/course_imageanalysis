// Is it more than chance? Simulation envelopes and global tests. Tabs:
//  - envelopes: the observed L − r (or g) of a pattern from the shared source, the pointwise
//    min/max envelope of s CSR simulations (same n, same window) and the global (MAD) envelope;
//    readouts: where the curve leaves the pointwise band, the MAD statistic, its rank and p;
//  - how often is CSR rejected?: runs CSR patterns and counts how often the pointwise envelope is
//    left somewhere vs how often the MAD test rejects; the precomputed curve against the number of radii;
//  - conditions: 2 conditions × experiments × images, one L(r0) − r0 per image, as a SuperPlot;
//    the t-test with images as n against experiments as n.
// Auto-mounts on <div data-demo="envelopetest">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount, checkbox } from "../lib/ui";
import { Plot, plotFigure, seriesColor } from "../lib/plot";
import { rng } from "../lib/random";
import * as PP from "../lib/pointpattern";
import { Spec, DEFAULT_SPEC, makePattern, makeWindow, PatternView, sourceControls, injectStyle, sourceCss, Model, WinKind } from "../csrplay/source";
import { RADII, RMAX, Fn, curve, reference, nullCurves, rejectionTrial, PRECOMP_M, PRECOMP_PW39, PRECOMP_PW19, PRECOMP_MAD, DEFAULT_COND, CondOptions, simulateConditions, conditionTests } from "./model";

const NAME = "envelopetest";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = sourceCss(NAME) + `
${P} .et-view { flex: 0 1 300px; min-width: 200px; }
${P} .et-main { flex: 1 1 400px; min-width: 260px; max-width: 560px; display: flex; flex-direction: column; gap: .3em; }
${P} .qb-panel { flex: 1 1 280px; gap: .35em; }
${P} .et-cap { font-size: .9em; text-align: center; }
.reveal ${P} .et-view { flex: 0 0 300px; min-width: 0; }
.reveal ${P} .et-main { flex: 0 0 480px; min-width: 0; }
.reveal ${P} .qb-panel { flex: 1 1 320px; min-width: 0; gap: .15em; }
`;

type Tab = "env" | "rate" | "cond";
const TABS: [Tab, string][] = [["env", "envelopes"], ["rate", "how often is CSR rejected?"], ["cond", "conditions"]];
const SIMS = [19, 39, 99, 199];
interface State { spec: Spec; tab: Tab; fn: Fn; s: number; r1: number; r2: number; pw: boolean; glob: boolean; cond: CondOptions }
const DEFAULT: State = { spec: { ...DEFAULT_SPEC }, tab: "env", fn: "L", s: 39, r1: 1, r2: 50, pw: true, glob: false, cond: { ...DEFAULT_COND } };

// seeds chosen with tools/figs/envelopes/seeds.ts: CSR seed 6 leaves the 39-simulation pointwise
// envelope at 6 of 50 radii (8–12 and 29 µm) while the MAD test gives p = 0.25; seed 4 stays inside
const TRAP_SEED = 6;
const PRESETS: { id: string; label: string; s: Partial<State>; spec?: Partial<Spec>; cond?: Partial<CondOptions> }[] = [
  { id: "csr", label: "CSR: inside the band", s: { tab: "env" }, spec: { model: "csr", seed: 4 } },
  { id: "weak", label: "weak clusters: escapes at small r", s: { tab: "env", glob: true }, spec: { model: "thomas", k: 75, mu: 2, sigma: 3 } },
  { id: "trap", label: "pointwise trap: a random pattern escapes", s: { tab: "env" }, spec: { model: "csr", seed: TRAP_SEED } },
  { id: "global", label: "global envelope: no false alarm", s: { tab: "env", glob: true }, spec: { model: "csr", seed: TRAP_SEED } },
  { id: "s199", label: "199 simulations", s: { tab: "env", s: 199, glob: true }, spec: { model: "csr", seed: TRAP_SEED } },
  { id: "hardcore", label: "hard core: below the band", s: { tab: "env", glob: true, fn: "L" }, spec: { model: "hardcore", n: 150, h: 7 } },
  { id: "range", label: "choose the r range first", s: { tab: "env", glob: true, r1: 20, r2: 50 }, spec: { model: "thomas", k: 75, mu: 2, sigma: 3 } },
  { id: "rate", label: "how often is CSR rejected?", s: { tab: "rate" } },
  { id: "images", label: "images as n: tiny p", s: { tab: "cond" }, cond: { effect: 0, dishSd: 0.15 } },
  { id: "effect", label: "a real effect, experiments as n", s: { tab: "cond" }, cond: { effect: 0.3, dishSd: 0.15, seed: 2 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 22 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, spec: { ...DEFAULT_SPEC }, cond: { ...DEFAULT_COND } };
  if (p0) { Object.assign(st, p0.s); Object.assign(st.spec, p0.spec ?? {}); Object.assign(st.cond, p0.cond ?? {}); }
  else if (host.dataset.tab) st.tab = host.dataset.tab as Tab;

  const viewBox = htmlEl("div", "et-view", "", host);
  const cap = htmlEl("div", "et-cap", "", viewBox);
  const view = new PatternView(viewBox);
  const main = htmlEl("div", "et-main", "", host);
  const tabsEl = htmlEl("div", "ss-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(); }));
  const fig = plotFigure(main, "", "qb-figure");
  const pl = new Plot(fig, { width: 500, height: 370, x: [0, RMAX], y: [-5, 5], font: FS, margin: { l: 3.8 * FS, b: 2.7 * FS, t: 10, r: 14 } });
  const envRow = htmlEl("div", "qb-presets", "", main);
  const fnBtns = (["L", "g"] as Fn[]).map(f => button(envRow, f === "L" ? "L − r" : "g", () => { st.fn = f; unp(); sims = null; draw(); }));
  const sBtns = SIMS.map(s => button(envRow, `s = ${s}`, () => { st.s = s; unp(); draw(); }));

  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const envCtl = htmlEl("div", "", "", panel);
  const pwBox = checkbox(envCtl, "pointwise envelope", st.pw, v => { st.pw = v; unp(); draw(); });
  const glBox = checkbox(envCtl, "global (MAD) envelope", st.glob, v => { st.glob = v; unp(); draw(); });
  const r1Sl = slider(envCtl, "r from (µm)", 0, 50, 1, st.r1, v => { st.r1 = Math.min(v, st.r2 - 1); unp(); draw(); });
  const r2Sl = slider(envCtl, "r to (µm)", 1, 50, 1, st.r2, v => { st.r2 = Math.max(v, st.r1 + 1); unp(); draw(); });
  const src = sourceControls(envCtl, st.spec, () => { unp(); regenerate(); draw(); }, { models: ["csr", "thomas", "hardcore", "colonies", "inhom"] as Model[], windows: ["square", "tissue"] as WinKind[] });
  const rateCtl = htmlEl("div", "", "", panel);
  const runBtn = button(rateCtl, "run 200 CSR patterns", () => startRun());
  const condCtl = htmlEl("div", "", "", panel);
  const cs = {
    effect: slider(condCtl, "treatment effect", 0, 0.5, 0.05, st.cond.effect, v => { st.cond.effect = v; unp(); condCache = null; draw(); }),
    dishSd: slider(condCtl, "dish-to-dish sd", 0, 0.3, 0.01, st.cond.dishSd, v => { st.cond.dishSd = v; unp(); condCache = null; draw(); }),
    images: slider(condCtl, "images per dish", 1, 8, 1, st.cond.images, v => { st.cond.images = v; unp(); condCache = null; draw(); }),
  };
  button(condCtl, "new experiment (seed)", () => { st.cond.seed++; condCache = null; unp(); draw(); });
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
  let sims: { key: string; curves: Float64Array[] } | null = null;
  let condCache: ReturnType<typeof simulateConditions> | null = null;
  function regenerate(): void { pts = makePattern(st.spec); sims = null; }
  function unp(): void { activeId = ""; pb.forEach(b => b.classList.remove("active")); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    const { spec, cond } = st;
    Object.assign(st, DEFAULT, p?.s ?? {}, { spec, cond });
    Object.assign(spec, DEFAULT_SPEC, p?.spec ?? {});
    Object.assign(cond, DEFAULT_COND, p?.cond ?? {});
    src.sync(); pwBox.checked = st.pw; glBox.checked = st.glob; r1Sl.set(st.r1); r2Sl.set(st.r2);
    cs.effect.set(st.cond.effect); cs.dishSd.set(st.cond.dishSd); cs.images.set(st.cond.images);
    activeId = p?.id ?? ""; condCache = null;
    regenerate(); draw();
  }

  // ---- rejection-rate run (incremental, requestAnimationFrame batches)
  let run = { done: 0, pw39: 0, pw19: 0, mad: 0, active: false };
  function startRun(): void {
    run = { done: 0, pw39: 0, pw19: 0, mad: 0, active: true };
    const r = rng(1234), T = 200;
    const step = () => {
      if (!run.active) return;
      const t0 = performance.now();
      while (run.done < T && performance.now() - t0 < 25) {
        const x = rejectionTrial(r, 100, 200, 25);
        run.done++; if (x.pw39) run.pw39++; if (x.pw19) run.pw19++; if (x.mad) run.mad++;
      }
      if (run.done >= T) run.active = false;
      if (st.tab === "rate") draw();
      if (run.active) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");

  function drawEnv(): string {
    const win = makeWindow(st.spec.win), n = pts.x.length;
    view.setWindow(win);
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    view.setPoints(pts);
    cap.textContent = `${n} points, ${st.spec.win === "square" ? "200 × 200 µm" : "tissue mask"}`;
    const key = `${st.fn}|${st.spec.win}|${n}|${st.spec.seed}`;
    if (!sims || sims.key !== key || sims.curves.length < st.s) sims = { key, curves: nullCurves(n, win, 199, st.fn, 31 * st.spec.seed + 5) };
    const sc = sims.curves.slice(0, st.s), obs = curve(pts, win, st.fn), ref = reference(st.fn);
    const idx = PP.rangeIdx(RADII, [st.r1, st.r2]).filter(k => RADII[k] > 0);
    const env = PP.pointwiseEnvelope(sc), mad = PP.madTest(obs, sc, ref, idx);
    const esc = PP.escapes(obs, env, idx);
    const all = [...Array.from(obs), ...Array.from(env.lo), ...Array.from(env.hi), ...Array.from(mad.lo), ...Array.from(mad.hi)].filter(Number.isFinite);
    pl.opts.xticks = undefined; pl.opts.xfmt = undefined;
    if (st.fn === "L") { const m = Math.max(2, ...all.map(Math.abs)) * 1.1; pl.setDomain([0, RMAX], [-m, m]); }
    else pl.setDomain([0, RMAX], [0, Math.min(8, Math.max(2, ...all) * 1.05)]);
    pl.setLabels({ xlabel: "r (µm)", ylabel: st.fn === "L" ? "L(r) − r  (µm)" : "g(r)" });
    const xs = idx.map(k => RADII[k]);
    if (st.glob) pl.band(xs, idx.map(k => mad.lo[k]), idx.map(k => mad.hi[k]), { color: "var(--qb-c3)", fillOpacity: 0.18, label: "global (MAD) envelope" });
    if (st.pw) pl.band(xs, idx.map(k => env.lo[k]), idx.map(k => env.hi[k]), { color: "var(--qb-muted)", fillOpacity: 0.35, label: `pointwise, s = ${st.s}` });
    pl.line(RADII, ref, { color: "var(--qb-muted)", dash: "7 5", width: 2 });
    pl.line(RADII, obs, { color: "var(--qb-c1)", label: "observed" });
    if (st.pw && esc.length) pl.scatter(esc.map(k => RADII[k]), esc.map(k => obs[k]), { color: "var(--qb-c2)", r: 5 });
    if (st.r1 > 0) pl.vline(st.r1, { color: "var(--qb-c4)", width: 1.5, dash: "3 3" });
    if (st.r2 < RMAX) pl.vline(st.r2, { color: "var(--qb-c4)", width: 1.5, dash: "3 3" });
    pl.legend(undefined, st.fn === "g" ? "tr" : "tr");
    const escTxt = esc.length ? `${esc.length} of ${idx.length} radii (${RADII[esc[0]]}–${RADII[esc[esc.length - 1]]} µm)` : "nowhere";
    return `s = ${st.s} CSR simulations, n = ${n}\nr range ${st.r1}–${st.r2} µm (${idx.length} radii)\npointwise band left at:\n  ${escTxt}\n  level per r: 2/(s+1) = ${fmt(2 / (st.s + 1), 3)}\nMAD T = ${fmt(mad.T0)} µm, rank ${mad.rank} of ${st.s + 1}\n  global p = ${fmt(mad.p, 3)}${mad.p <= 0.05 ? "  → reject CSR" : ""}`;
  }

  function drawRate(): string {
    view.setWindow(makeWindow("square"));
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    view.setPoints(PP.binomialProcess(rng(1234 + run.done), 100, makeWindow("square")));
    cap.textContent = "one of the CSR patterns (n = 100)";
    pl.opts.xticks = undefined; pl.opts.xfmt = undefined;
    pl.setDomain([0, 52], [0, 0.8]);
    pl.setLabels({ xlabel: "number of radii tested", ylabel: "share of CSR rejected" });
    pl.hline(0.05, { color: "var(--qb-muted)", dash: "6 4", width: 2, text: "nominal 5 %" });
    pl.line(PRECOMP_M, PRECOMP_PW19, { color: "var(--qb-c4)", label: "pointwise, s = 19" });
    pl.line(PRECOMP_M, PRECOMP_PW39, { color: "var(--qb-c2)", label: "pointwise, s = 39" });
    pl.line(PRECOMP_M, PRECOMP_MAD, { color: "var(--qb-c3)", label: "MAD test, s = 39" });
    pl.scatter(PRECOMP_M, PRECOMP_PW19, { color: "var(--qb-c4)", r: 4 });
    pl.scatter(PRECOMP_M, PRECOMP_PW39, { color: "var(--qb-c2)", r: 4 });
    pl.scatter(PRECOMP_M, PRECOMP_MAD, { color: "var(--qb-c3)", r: 4 });
    if (run.done > 0) {
      const d = run.done;
      pl.scatter([25, 25, 25], [run.pw19 / d, run.pw39 / d, run.mad / d], { colors: ["var(--qb-c4)", "var(--qb-c2)", "var(--qb-c3)"], r: 8, stroke: "#000" });
    }
    pl.legend(undefined, "tl");
    const pct = (k: number) => run.done ? (100 * k / run.done).toFixed(1) + " %" : "–";
    return `lines: precomputed (1000 patterns)\ndots: this run, 25 radii\n  patterns ${run.done} of 200${run.active ? " …" : ""}\n  pointwise s = 19: ${pct(run.pw19)}\n  pointwise s = 39: ${pct(run.pw39)}\n  MAD test:         ${pct(run.mad)}\nEvery pattern is CSR: every\nrejection is a false alarm.`;
  }

  function drawCond(): string {
    if (!condCache) condCache = simulateConditions(st.cond);
    const { images, patterns } = condCache, E = st.cond.experiments;
    const win = makeWindow("square");
    view.setWindow(win);
    view.under.replaceChildren(); view.mid.replaceChildren(); view.over.replaceChildren();
    const lastTreated = images.map((s, i) => [s, i] as const).filter(([s]) => s.cond === 1).pop()![1];
    view.setPoints(patterns[lastTreated]);
    cap.textContent = "one treated image (n = 120)";
    const vals = images.map(s => s.value), lo = Math.min(...vals), hi = Math.max(...vals), pad = 0.15 * (hi - lo + 0.5);
    pl.opts.xticks = [0, 1]; pl.opts.xfmt = v => (v < 0.5 ? "control" : "treated");
    pl.setDomain([-0.6, 1.6], [lo - pad, hi + pad]);
    pl.setLabels({ xlabel: "", ylabel: `L(${st.cond.r0}) − ${st.cond.r0} per image (µm)` });
    const tests = conditionTests(images, E);
    images.forEach((s, i) => {
      const jit = ((i * 0.6180339) % 1 - 0.5) * 0.35;
      pl.scatter([s.cond + jit], [s.value], { color: seriesColor(s.exp), r: 5 });
    });
    for (const c of [0, 1]) tests.means[c].forEach((m, e) => pl.scatter([c + (e - (E - 1) / 2) * 0.08], [m], { color: seriesColor(e), r: 11, stroke: "#000" }));
    const nI = images.length / 2;
    return `small dots: images (${nI} per condition)\nbig dots: experiment means (${E})\nWelch t-test, images as n:\n  p = ${fmt(tests.imagesP, 4)}${tests.imagesP < 0.05 ? "  (significant)" : ""}\nexperiments as n:\n  p = ${fmt(tests.expP, 4)}${tests.expP < 0.05 ? "  (significant)" : ""}\ntrue treatment effect: ${st.cond.effect === 0 ? "none" : "+" + st.cond.effect + " clustered share"}`;
  }

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    fnBtns.forEach((b, i) => b.classList.toggle("active", (["L", "g"] as Fn[])[i] === st.fn));
    sBtns.forEach((b, i) => b.classList.toggle("active", SIMS[i] === st.s));
    buildPresets();
    envRow.style.display = st.tab === "env" ? "" : "none";
    envCtl.style.display = st.tab === "env" ? "" : "none";
    rateCtl.style.display = st.tab === "rate" ? "" : "none";
    condCtl.style.display = st.tab === "cond" ? "" : "none";
    runBtn.textContent = run.active ? "running …" : "run 200 CSR patterns";
    pl.clear();
    let txt = "";
    if (st.tab === "env") txt = drawEnv();
    else if (st.tab === "rate") txt = drawRate();
    else txt = drawCond();
    ro.textContent = txt;
  }

  draw();
}

autoMount(NAME, mount);
