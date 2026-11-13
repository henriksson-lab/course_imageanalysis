// Matching descriptors. Two views (view 2 = view 1 under a known rotation and zoom, with a
// lighting change, blur and noise), DoG keypoints with gradient-histogram, binary or patch
// descriptors. Matches are lines: green = correct (the known transform puts the point within
// ε px of its match), red = wrong. Filters: ratio test d1/d2 < t, a distance threshold,
// mutual nearest neighbours, one-to-one assignment. Panels: the matches; histograms of d1
// and of d1/d2 for correct and wrong nearest-neighbour matches (click to set the threshold);
// the precision–recall curve as the ratio threshold sweeps (NN only, ratio, ratio + mutual).
// Search: brute force or a k-d tree with a leaf budget (approximate).
//
// <div data-demo="featurematch" data-preset="nn|dist|ratio|mutual|grid|binary|kd|cell"
//      data-layout="full|tabs" data-tab="matches|hist|pr" data-scene="scene|grid|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { Grey } from "../lib/imageops";
import * as K from "../lib/keypoints";
import * as D from "../lib/descriptors";
import * as M from "../lib/featurematch";
import { CELL_CREDIT } from "../lib/cell";
import { injectStyle } from "../canny/imgview";
import { getScene, makeViews, detect, describe, scheduler, SceneKind, PairView, GREEN, RED, YELLOW, CYAN, COMMON_CSS, ViewPairResult, DescKind } from "../repeatability/kpcommon";

const NAME = "featurematch";
const STYLE = COMMON_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .fm-main { flex: 1 1 560px; min-width: 300px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .fm-plots { display: flex; flex-wrap: wrap; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .fm-plots > svg { flex: 1 1 200px; min-width: 180px; max-width: 300px; }
.qb-demo[data-demo="${NAME}"] .fm-tabs { display: flex; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .fm-key { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .fm-main { flex: 0 0 740px; }
.reveal .qb-demo[data-demo="${NAME}"] .fm-plots > svg { max-width: 360px; flex: 1 1 240px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 370px; gap: .1em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .ev-credit { display: none; }
`;

type Tab = "matches" | "hist" | "pr";
interface State {
  scene: SceneKind; desc: DescKind; ratio: number; dist: number; mutual: boolean; one: boolean;
  n: number; angle: number; zoom: number; gain: number; blur: number; noise: number; eps: number;
  kd: boolean; leaves: number; tab: Tab; sel: number;
}
const DEFAULT: State = { scene: "scene", desc: "hist", ratio: 0.8, dist: 2, mutual: false, one: false, n: 200, angle: 30, zoom: 1.3, gain: 1.2, blur: 0, noise: 3, eps: 3, kd: false, leaves: 8, tab: "matches", sel: -1 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "nn", label: "nearest only: a mess", s: { ratio: 1 } },
  { id: "dist", label: "distance threshold: can't separate", s: { ratio: 1, dist: 0.45, tab: "hist" } },
  { id: "ratio", label: "ratio 0.8", s: { ratio: 0.8 } },
  { id: "mutual", label: "ratio + mutual", s: { ratio: 0.8, mutual: true } },
  { id: "grid", label: "repetitive grid", s: { scene: "grid", ratio: 1, angle: 10, zoom: 1 } },
  { id: "binary", label: "binary vs histogram under blur", s: { desc: "binary", blur: 1.5 } },
  { id: "kd", label: "approximate search: 4× fewer distances", s: { kd: true, leaves: 8 } },
  { id: "cell", label: "the micrograph tiles", s: { scene: "cell", angle: 20, zoom: 1.2, noise: 0 } },
];
const DIST_OFF = 2;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const full = host.dataset.layout === "full" || !host.closest(".reveal");
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const base = (): Partial<State> => ({ ...(host.dataset.scene ? { scene: host.dataset.scene as SceneKind } : {}), ...(host.dataset.tab ? { tab: host.dataset.tab as Tab } : {}) });
  const start = (): State => ({ ...DEFAULT, ...base(), ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "fm-main", "", host);
  const tabs = htmlEl("div", "fm-tabs", "", main);
  const tabBtns = (["matches", "hist", "pr"] as Tab[]).map(t => button(tabs, t === "matches" ? "matches" : t === "hist" ? "distances and ratios" : "precision–recall", () => { st.tab = t; layout(); }));
  if (full) tabs.style.display = "none";
  const pv = new PairView(main, "");
  const key = htmlEl("div", "fm-key", "", main);
  key.innerHTML = `<b style="color:${GREEN}">━</b> correct &nbsp; <b style="color:${RED}">━</b> wrong (by the known transform); click a keypoint of view 1`;
  const plots = htmlEl("div", "fm-plots", "", main);
  const pd1 = new Plot(plots, { width: 300, height: 230, x: [0, 1.2], y: [0, 1], font: 17, xlabel: "nearest distance d1", ylabel: "matches", margin: { l: 56, r: 8, t: 26, b: 48 }, yticks: 3, xticks: 4, title: "" });
  const pr = new Plot(plots, { width: 300, height: 230, x: [0.3, 1], y: [0, 1], font: 17, xlabel: "ratio d1/d2", ylabel: "matches", margin: { l: 56, r: 8, t: 26, b: 48 }, yticks: 3, xticks: [0.4, 0.6, 0.8, 1] });
  const ppr = new Plot(plots, { width: 300, height: 230, x: [0.3, 1], y: [0, 1], font: 17, xlabel: "recall", ylabel: "precision", margin: { l: 56, r: 8, t: 26, b: 48 }, xticks: 3, yticks: 3 });
  const hC1 = pd1.histogram([], [], { color: GREEN, fillOpacity: 0.55 }), hW1 = pd1.histogram([], [], { color: RED, fillOpacity: 0.45 });
  const vD = pd1.vline(DIST_OFF, { color: "#333", width: 2.5, dash: "6 4" });
  const hC2 = pr.histogram([], [], { color: GREEN, fillOpacity: 0.55 }), hW2 = pr.histogram([], [], { color: RED, fillOpacity: 0.45 });
  const vR = pr.vline(0.8, { color: "#333", width: 2.5, dash: "6 4" });
  const cRatio = ppr.line([], [], { color: "#1f77b4", width: 3, label: "ratio" }), cMut = ppr.line([], [], { color: "#9467bd", width: 3, label: "ratio + mutual" });
  const pNN = ppr.scatter([], [], { color: "#ff7f0e", r: 7, label: "NN only" }), pCur = ppr.scatter([], [], { color: "#111", r: 6 });
  ppr.legend(undefined, "bl");
  pr.svg.style.cursor = "pointer"; pd1.svg.style.cursor = "pointer";
  pr.svg.addEventListener("pointerdown", ev => { const [x] = pr.toData(ev); st.ratio = Math.max(0.3, Math.min(1, Math.round(x * 100) / 100)); rS.set(st.ratio); filterAndDraw(); });
  pd1.svg.addEventListener("pointerdown", ev => { const [x] = pd1.toData(ev); st.dist = Math.max(0.05, Math.round(x * 100) / 100); dS.set(st.dist); filterAndDraw(); });
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const busy = htmlEl("div", "kp-busy", "", main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, base(), p.s); sync(); load(); } })));
  const descBtns = presets(panel, [
    { label: "histogram", apply: () => { st.desc = "hist"; recompute(); } },
    { label: "binary", apply: () => { st.desc = "binary"; recompute(); } },
    { label: "patch", apply: () => { st.desc = "patch"; recompute(); } },
  ]);
  const rS = slider(panel, "ratio t", 0.5, 1, 0.01, st.ratio, v => { st.ratio = v; filterAndDraw(); }, v => v >= 1 ? "off" : v.toFixed(2));
  const dS = slider(panel, "distance ≤", 0.05, DIST_OFF, 0.05, st.dist, v => { st.dist = v; filterAndDraw(); }, v => v >= DIST_OFF ? "off" : v.toFixed(2));
  const checks = htmlEl("div", "qb-presets", "", panel);
  const mC = checkbox(checks, "mutual", st.mutual, v => { st.mutual = v; filterAndDraw(); });
  const oC = checkbox(checks, "one-to-one", st.one, v => { st.one = v; filterAndDraw(); });
  const kC = checkbox(checks, "k-d tree", st.kd, v => { st.kd = v; recompute(); });
  const lS = slider(panel, "leaf budget", 1, 64, 1, st.leaves, v => { st.leaves = v; recompute(); });
  const aS = slider(panel, "rotation°", -180, 180, 5, st.angle, v => { st.angle = v; recompute(); });
  const zS = slider(panel, "zoom", 0.6, 2, 0.05, st.zoom, v => { st.zoom = v; recompute(); });
  const bS = slider(panel, "blur σ", 0, 3, 0.25, st.blur, v => { st.blur = v; recompute(); });
  const nS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; recompute(); });
  const kS = slider(panel, "keypoints", 50, 400, 25, st.n, v => { st.n = v; recompute(); });
  const eS = slider(panel, "ε (px)", 1, 6, 0.5, st.eps, v => { st.eps = v; recompute(); });
  const sceneBtns = presets(panel, [
    { label: "scene", apply: () => { st.scene = "scene"; load(); } },
    { label: "repetitive", apply: () => { st.scene = "grid"; load(); } },
    { label: "micrograph", apply: () => { st.scene = "cell"; load(); } },
  ]);
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); load(); });
  const ro = htmlEl("div", "qb-readout", "", full ? panel : main);

  function sync(): void {
    rS.set(st.ratio); dS.set(st.dist); mC.checked = st.mutual; oC.checked = st.one; kC.checked = st.kd; lS.set(st.leaves);
    aS.set(st.angle); zS.set(st.zoom); bS.set(st.blur); nS.set(st.noise); kS.set(st.n); eS.set(st.eps);
  }
  function layout(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (["matches", "hist", "pr"] as Tab[])[i] === st.tab));
    if (full) { pv.fig.style.display = ""; key.style.display = ""; plots.style.display = ""; pd1.svg.style.display = pr.svg.style.display = ppr.svg.style.display = ""; return; }
    pv.fig.style.display = key.style.display = st.tab === "matches" ? "" : "none";
    plots.style.display = st.tab === "matches" ? "none" : "";
    pd1.svg.style.display = pr.svg.style.display = st.tab === "hist" ? "" : "none";
    ppr.svg.style.display = st.tab === "pr" ? "" : "none";
  }

  let scene: Grey | null = null, views: ViewPairResult | null = null;
  let k1: K.Keypoint[] = [], k2: K.Keypoint[] = [], D1: D.Desc[] = [], D2: D.Desc[] = [];
  let m12: M.Match[] = [], m21: M.Match[] = [], correct: boolean[] = [], possible = 0, searchMs = 0, exactFrac = 1, oneLinks: M.Match[] = [];
  const run = scheduler(on => { busy.textContent = on ? "computing…" : ""; });
  function load(): void { getScene(st.scene).then(g => { scene = g; recompute(); }); }

  function recompute(): void {
    run(() => {
      if (!scene) return;
      views = makeViews(st.scene, scene, { zoom: st.zoom, angle: st.angle }, { gain: st.gain, blur: st.blur, noise: st.noise });
      const a = detect(views.v1, "dog", st.n), b = detect(views.v2, "dog", st.n);
      const r1 = describe(a, st.desc), r2 = describe(b, st.desc);
      k1 = r1.kps; D1 = r1.desc; k2 = r2.kps; D2 = r2.desc;
      const metric: D.Metric = st.desc === "binary" ? "hamming" : "l2";
      let t = performance.now();
      const brute = M.matchDescriptors(D1, D2, metric);
      const tBrute = performance.now() - t;
      if (st.kd && st.desc !== "binary") {
        t = performance.now();
        m12 = M.matchKd(D1, D2, st.leaves).matches;
        searchMs = performance.now() - t;
        let same = 0; m12.forEach((m, k) => { if (m.j === brute[k].j) same++; });
        exactFrac = m12.length ? same / m12.length : 1;
      } else { m12 = brute; searchMs = tBrute; exactFrac = 1; }
      m21 = M.matchDescriptors(D2, D1, metric);
      correct = M.correctFlags(m12, k1, k2, views.map12, st.eps);
      possible = M.possibleCount(k1, k2, views.map12, st.eps);
      oneLinks = st.one ? M.oneToOne(D1, D2, metric) : [];
      if (st.sel >= k1.length) st.sel = -1;
      filterAndDraw();
    });
  }

  const scale = () => st.desc === "binary" ? 256 : st.desc === "patch" ? 2 : 1.4;
  function accepted(ratio: number, mutual: boolean): M.Match[] {
    const dmax = st.dist >= DIST_OFF ? Infinity : st.dist * (st.desc === "binary" ? 128 : 1);
    let ms = (st.one ? oneLinks.map(l => ({ ...l, j2: m12[l.i]?.j2 ?? -1, d2: m12[l.i]?.d2 ?? Infinity })) : m12).filter(m => m.d1 <= dmax);
    if (ratio < 1) ms = ms.filter(m => M.ratioOf(m) < ratio);
    if (mutual) ms = M.mutualCheck(ms, m21);
    return ms;
  }

  function filterAndDraw(): void {
    if (!views) return;
    layout();
    descBtns.forEach((b, i) => b.classList.toggle("active", ["hist", "binary", "patch"][i] === st.desc));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["scene", "grid", "cell"][i] === st.scene));
    credit.style.display = st.scene === "cell" ? "" : "none";
    const acc = accepted(st.ratio, st.mutual);
    const q = M.evaluateMatches(acc, k1, k2, views.map12, st.eps, possible);
    // matches
    pv.images(views.v1, views.v2);
    pv.clear();
    for (const k of k1) pv.dot(pv.at(1, k.x, k.y), 1.8, "rgba(0,0,0,.5)", "rgba(255,255,255,.8)", 0.8);
    for (const k of k2) pv.dot(pv.at(2, k.x, k.y), 1.8, "rgba(0,0,0,.5)", "rgba(255,255,255,.8)", 0.8);
    acc.forEach((m, k) => pv.line(pv.at(1, k1[m.i].x, k1[m.i].y), pv.at(2, k2[m.j].x, k2[m.j].y), q.correct[k] ? GREEN : RED, 1.6, 0.8));
    let selTxt = "";
    if (st.sel >= 0) {
      const m = m12.find(x => x.i === st.sel);
      const kp = k1[st.sel];
      pv.dot(pv.at(1, kp.x, kp.y), 7, YELLOW, "none", 3);
      const [px, py] = views.map12(kp.x, kp.y);
      pv.add("path", { d: `M${pv.at(2, px, py)[0] - 7},${pv.at(2, px, py)[1]}h14M${pv.at(2, px, py)[0]},${pv.at(2, px, py)[1] - 7}v14`, stroke: YELLOW, "stroke-width": 3 });
      if (m) {
        pv.dot(pv.at(2, k2[m.j].x, k2[m.j].y), 8, correct[m12.indexOf(m)] ? GREEN : RED, "none", 3);
        if (m.j2 >= 0) pv.dot(pv.at(2, k2[m.j2].x, k2[m.j2].y), 8, CYAN, "none", 3);
        selTxt = `\nselected: d1 = ${m.d1.toFixed(3)}, d2 = ${m.d2.toFixed(3)}, ratio ${(M.ratioOf(m)).toFixed(2)}`;
      }
    }
    // histograms of the nearest-neighbour matches, correct vs wrong
    const sc = scale(), d1c: number[] = [], d1w: number[] = [], rc: number[] = [], rw: number[] = [];
    m12.forEach((m, k) => { (correct[k] ? d1c : d1w).push(m.d1 / (st.desc === "binary" ? 128 : 1)); (correct[k] ? rc : rw).push(M.ratioOf(m)); });
    const hist = (v: number[], lo: number, hi: number, nb: number) => { const c = new Array(nb).fill(0); for (const x of v) { const b = Math.floor((x - lo) / (hi - lo) * nb); if (b >= 0 && b < nb) c[b]++; else if (b >= nb) c[nb - 1]++; } return c; };
    const edges = (lo: number, hi: number, nb: number) => Array.from({ length: nb + 1 }, (_, k) => lo + (hi - lo) * k / nb);
    const xmax = st.desc === "binary" ? 1 : sc, cA = hist(d1c, 0, xmax, 24), cB = hist(d1w, 0, xmax, 24), rA = hist(rc, 0.3, 1, 21), rB = hist(rw, 0.3, 1, 21);
    pd1.setDomain([0, xmax], [0, Math.max(1, ...cA, ...cB) * 1.1]); hC1.set(edges(0, xmax, 24), cA); hW1.set(edges(0, xmax, 24), cB);
    vD.set(st.dist >= DIST_OFF ? NaN : st.dist);
    const xl = st.desc === "binary" ? "nearest distance d1 / 128" : "nearest distance d1";
    if (pd1.opts.xlabel !== xl) pd1.setLabels({ xlabel: xl });
    pr.setDomain([0.3, 1], [0, Math.max(1, ...rA, ...rB) * 1.1]); hC2.set(edges(0.3, 1, 21), rA); hW2.set(edges(0.3, 1, 21), rB);
    vR.set(st.ratio >= 1 ? NaN : st.ratio);
    // precision–recall
    const ts = Array.from({ length: 36 }, (_, k) => 0.3 + k * 0.02);
    const c1 = M.prCurve(m12, correct, possible, ts);
    const mut = M.mutualCheck(m12, m21), mutCorrect = M.correctFlags(mut, k1, k2, views.map12, st.eps);
    const c2 = M.prCurve(mut, mutCorrect, possible, ts);
    cRatio.set(c1.map(p => p.recall), c1.map(p => p.precision)); cMut.set(c2.map(p => p.recall), c2.map(p => p.precision));
    const nn = M.evaluateMatches(m12, k1, k2, views.map12, st.eps, possible);
    pNN.set([nn.recall], [nn.precision]); pCur.set([q.recall], [q.precision]);
    ro.textContent = `matches accepted: ${q.accepted}, correct ${q.nCorrect}\nprecision ${(100 * q.precision).toFixed(0)} %, recall ${(100 * q.recall).toFixed(0)} % (of ${possible} possible)\n` +
      `search: ${st.kd && st.desc !== "binary" ? `k-d tree, ${st.leaves} leaves` : "brute force"} ${searchMs.toFixed(0)} ms` + (st.kd && st.desc !== "binary" ? `, exact NN ${(100 * exactFrac).toFixed(0)} %` : "") + selTxt;
  }

  pv.onPick((v, x, y) => {
    if (v !== 1) return;
    let best = -1, bd = 8;
    k1.forEach((k, i) => { const d = Math.hypot(k.x - x, k.y - y); if (d < bd) { bd = d; best = i; } });
    st.sel = best; filterAndDraw();
  });

  sync(); layout();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  load();
}

autoMount(NAME, mount);
