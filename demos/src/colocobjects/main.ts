// Object-based colocalisation. A synthetic two-channel spot scene at physical scale (pixel
// 65 nm, NA 1.4, A at 520 nm, B at 670 nm, two cells): spots are detected in both channels
// and compared as points. Tabs:
//  - distances: merge with the detected spots (green A, magenta B) and the links within
//    dMax; histogram of A → B distances with the chance density for randomly placed B spots;
//    readout: fraction within dMax, chance fraction, corrected fraction, pixel Pearson r;
//    nearest neighbour or one-to-one pairing;
//  - cross-L: L(r) − r against r with the envelope of 39 random relocations of B in the cells;
//  - objects: larger discs, thresholded and labelled: overlap by any pixel, by > 50 % and
//    centroid-in-object;
//  - below the resolution: measured distances of truly paired spots against the true
//    distance and the mean predicted from the localisation errors.
// Auto-mounts on <div data-demo="colocobjects">. Options: data-preset="<id>", data-tab.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { S, mergeGreenMagenta } from "../lib/twochannel";
import { poissonNNDensity } from "../lib/neighbours";
import { quantile } from "../lib/stats";
import { ObjOptions, DEFAULT_OBJ, makeObjScene, ObjScene, distances, pixelR, crossL, pairedDistances, makeBlobScene, PIXEL_NM, OFFSET } from "./model";

const NAME = "colocobjects";
const Z = 3;

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cb-left { flex: 0 1 280px; min-width: 180px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .cb-img { margin: 0; }
.qb-demo[data-demo="${NAME}"] .cb-img canvas { width: 100%; max-width: min(280px, calc(100vh - 460px)); min-width: 160px; aspect-ratio: 1; display: block; margin: 0 auto; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
.qb-demo[data-demo="${NAME}"] .cb-main { flex: 1 1 380px; min-width: 260px; max-width: min(430px, calc((100vh - 250px) * 1.4)); }
.qb-demo[data-demo="${NAME}"] .cb-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 320px; gap: .25em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider { gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre-wrap; line-height: 1.3; }
.qb-demo[data-demo="${NAME}"] .cb-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .5em; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cb-left { flex: 0 0 260px; min-width: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .cb-img canvas { max-width: 260px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .cb-main { flex: 0 0 430px; min-width: 0; max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; gap: .12em; min-width: 0; font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; line-height: 1.2; padding: .25em .45em; white-space: pre; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .cb-lab { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Tab = "dist" | "crossl" | "objects" | "below";
const TABS: [Tab, string][] = [["dist", "distances"], ["crossl", "cross-L"], ["objects", "objects"], ["below", "below the resolution"]];

interface State extends ObjOptions { dMax: number; oneToOne: boolean; rA: number; rB: number; tab: Tab }
const DEFAULT: State = { ...DEFAULT_OBJ, dMax: 130, oneToOne: false, rA: 3, rB: 3, tab: "dist" } as State;

const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "pairs0", label: "pairs at 0 nm", s: { tab: "dist" } },
  { id: "sparse", label: "independent, sparse: chance ≈ 2.5 %", s: { pair: 0, extraB: 12, tab: "dist" } },
  { id: "dense", label: "independent, dense: chance ≈ 20 %", s: { pair: 0, extraB: 120, tab: "dist" } },
  { id: "shift", label: "chromatic shift 100 nm", s: { shiftNm: 100, tab: "dist" } },
  { id: "missed", label: "few photons: missed detections", s: { pair: 1, extraB: 0, photons: 80, tab: "dist" } },
  { id: "attract", label: "attraction: L above the envelope", s: { tab: "crossl" } },
  { id: "indepL", label: "independent: L inside the envelope", s: { pair: 0, extraB: 60, tab: "crossl" } },
  { id: "big", label: "big objects: overlap or centroid?", s: { rA: 6, rB: 4, dNm: 325, tab: "objects" } },
  { id: "small", label: "small A inside big B", s: { rA: 4, rB: 6, dNm: 325, tab: "objects" } },
  { id: "d60", label: "pairs at 60 nm: pixels say yes, centroids say 60", s: { pair: 1, dNm: 60, extraB: 0, photons: 2000, tab: "below" } },
  { id: "few", label: "few photons: the distance bias", s: { pair: 1, dNm: 0, extraB: 0, photons: 150, tab: "below" } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const onSlide = !!host.closest(".reveal");
  const FS = onSlide ? 24 : 18;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  if (!p0 && host.dataset.tab) st.tab = host.dataset.tab as Tab;
  let sc: ObjScene = makeObjScene(st);
  let clCache: ReturnType<typeof crossL> | null = null;

  const left = htmlEl("div", "cb-left", "", host);
  const imgFig = htmlEl("figure", "cb-img", "", left);
  const imgCap = htmlEl("figcaption", "", "", imgFig);
  const cv = htmlEl("canvas", "", "", imgFig);
  cv.width = S * Z; cv.height = S * Z;
  const ro = htmlEl("div", "qb-readout", "", left);

  const main = htmlEl("div", "cb-main", "", host);
  const tabsEl = htmlEl("div", "cb-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsEl, l, () => { st.tab = t; unp(); draw(); }));
  const pf = plotFigure(main, "", "qb-figure");
  const pl = new Plot(pf, { width: 460, height: 330, x: [0, 800], y: [0, 10], font: FS, margin: { l: 3.5 * FS, b: 2.8 * FS, t: 12, r: 26 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  if (onSlide) panel.prepend(ro);   // slides: the wide panel holds the readout (pages: under the image)
  const pairB = presets(panel, [
    { label: "nearest neighbour", apply: () => { st.oneToOne = false; draw(); } },
    { label: "one-to-one", apply: () => { st.oneToOne = true; draw(); } },
  ]);
  htmlEl("div", "cb-lab", "presets", panel);
  const presetRow = htmlEl("div", "", "", panel);
  let pb: HTMLButtonElement[] = [];
  const sl = {
    dMax: slider(panel, "dMax (nm)", 30, 400, 10, st.dMax, v => { st.dMax = v; unp(); draw(); }),
    nA: slider(panel, "A spots", 10, 150, 5, st.nA, v => { st.nA = v; changed(); }),
    pair: slider(panel, "share paired", 0, 1, 0.05, st.pair, v => { st.pair = v; changed(); }),
    dNm: slider(panel, "true distance (nm)", 0, 500, 10, st.dNm, v => { st.dNm = v; changed(); }),
    extraB: slider(panel, "extra B spots", 0, 200, 2, st.extraB, v => { st.extraB = v; changed(); }),
    photons: slider(panel, "photons per spot", 50, 3000, 10, st.photons, v => { st.photons = v; changed(); }),
    shiftNm: slider(panel, "chromatic shift (nm)", 0, 200, 5, st.shiftNm, v => { st.shiftNm = v; changed(); }),
    rA: slider(panel, "radius A (px)", 1, 10, 0.5, st.rA, v => { st.rA = v; unp(); draw(); }),
    rB: slider(panel, "radius B (px)", 1, 10, 0.5, st.rB, v => { st.rB = v; unp(); draw(); }),
  };
  button(panel, "reset", () => applyPreset(null));

  const VISIBLE: Record<Tab, (keyof typeof sl)[]> = {
    dist: ["dMax", "nA", "pair", "dNm", "extraB", "photons", "shiftNm"],
    crossl: ["nA", "pair", "dNm", "extraB", "photons", "shiftNm"],
    objects: ["pair", "dNm", "rA", "rB"],
    below: ["pair", "dNm", "photons", "shiftNm"],
  };

  function buildPresetButtons(): void {
    presetRow.replaceChildren();
    const list = onSlide ? PRESETS.filter(p => p.s.tab === st.tab) : PRESETS;
    pb = presets(presetRow, list.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
    pb.forEach((b, i) => b.dataset.id = list[i].id);
  }
  function unp(): void { pb.forEach(b => b.classList.remove("active")); }
  function changed(): void { unp(); sc = makeObjScene(st); clCache = null; draw(); }
  function applyPreset(p: typeof PRESETS[number] | null): void {
    Object.assign(st, DEFAULT, p?.s ?? {});
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].set(st[k] as number);
    sc = makeObjScene(st); clCache = null;
    draw();
    pb.forEach(b => b.classList.toggle("active", !!p && b.dataset.id === p.id));
  }

  function baseImage(a: ArrayLike<number>, b: ArrayLike<number>, ra: [number, number], rb: [number, number]): void {
    const ctx = cv.getContext("2d")!, tmp = document.createElement("canvas");
    tmp.width = S; tmp.height = S;
    const id = tmp.getContext("2d")!.createImageData(S, S);
    mergeGreenMagenta(a, b, ra, rb, id.data);
    tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, S * Z, S * Z);
  }
  const range = (img: Float64Array): [number, number] => [OFFSET, OFFSET + Math.max(5, quantile(Array.from(img, v => v - OFFSET), 0.998))];

  function drawSpots(res: ReturnType<typeof distances> | null): void {
    baseImage(sc.a, sc.b, range(sc.a), range(sc.b));
    const ctx = cv.getContext("2d")!;
    const P = (v: number) => (v + 0.5) * Z;
    if (res) {
      ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2.5;
      sc.detA.forEach((s, i) => {
        const j = res.partner[i];
        if (j < 0 || !(res.dist[i] <= st.dMax / PIXEL_NM)) return;
        const q = sc.detB[j];
        ctx.beginPath(); ctx.moveTo(P(s.x), P(s.y)); ctx.lineTo(P(q.x), P(q.y)); ctx.stroke();
      });
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#5dff5d";
    for (const s of sc.detA) { ctx.beginPath(); ctx.arc(P(s.x), P(s.y), 7, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.strokeStyle = "#ff6bff";
    for (const s of sc.detB) { ctx.beginPath(); ctx.arc(P(s.x), P(s.y), 10, 0, 2 * Math.PI); ctx.stroke(); }
    imgCap.textContent = "detected spots: A ○ green, B ○ magenta";
  }

  const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const pct = (v: number) => (Number.isFinite(v) ? (100 * v).toFixed(1) + " %" : "–");

  function draw(): void {
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    buildPresetButtons();
    pairB.forEach((b, i) => b.classList.toggle("active", [false, true][i] === st.oneToOne));
    pairB[0].parentElement!.style.display = st.tab === "dist" ? "" : "none";
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].input.closest("label")!.style.display = VISIBLE[st.tab].includes(k) ? "" : "none";
    pl.clear();
    let txt = "";
    const nAt = sc.truth.ax.length, nBt = sc.truth.bx.length;
    if (st.tab === "dist") {
      const res = distances(sc, st.dMax, st.oneToOne);
      drawSpots(res);
      const dn = Array.from(res.dist).filter(Number.isFinite).map(v => v * PIXEL_NM);
      const bw = 25, hi = 800, nb = hi / bw, cnt = new Array(nb).fill(0);
      for (const v of dn) { const k = Math.floor(v / bw); if (k < nb) cnt[k]++; }
      const rhoNm = res.rhoB / (PIXEL_NM * PIXEL_NM), nAd = sc.detA.length;
      const chance = (d: number) => nAd * bw * poissonNNDensity(d, rhoNm);
      pl.setDomain([0, hi], [0, Math.max(4, ...cnt, chance(1 / Math.sqrt(2 * Math.PI * rhoNm))) * 1.6]);
      pl.setLabels({ xlabel: "A → B distance (nm)", ylabel: "A spots" });
      pl.histogram(Array.from({ length: nb + 1 }, (_, k) => k * bw), cnt, { color: "var(--qb-c1)", fillOpacity: 0.6 });
      pl.fn(chance, { color: "var(--qb-c4)", width: 3 });
      pl.vline(st.dMax, { color: "var(--qb-c2)", width: 3 });
      pl.legend([{ label: "observed", color: "var(--qb-c1)", kind: "box" }, { label: "chance (random B)", color: "var(--qb-c4)" }, { label: "dMax", color: "var(--qb-c2)" }], "tr");
      txt = `detected A ${sc.detA.length} of ${nAt}\ndetected B ${sc.detB.length} of ${nBt}\nwithin ${st.dMax} nm  ${pct(res.fObs)}\nby chance       ${pct(res.fChance)}\ncorrected       ${pct(res.fCorr)}\npixel r (cells) ${fmt(pixelR(sc))}`;
    } else if (st.tab === "crossl") {
      drawSpots(null);
      const radii = Array.from({ length: 21 }, (_, k) => 25 * (k + 1));
      clCache ??= crossL(sc, radii);
      const c = clCache;
      const all = [...c.obs, ...c.lo, ...c.hi];
      const lo = Math.min(...all) - 10, hi = Math.max(...all) + 10;
      pl.setDomain([0, radii[radii.length - 1]], [lo, hi + 0.45 * (hi - lo)]);
      pl.setLabels({ xlabel: "radius r (nm)", ylabel: "L(r) − r (nm)" });
      pl.band(radii, c.lo, c.hi, { fill: "#999", fillOpacity: 0.3, color: "none" });
      pl.hline(0, { color: "var(--qb-axis)", width: 1, dash: "4 4" });
      pl.line(radii, c.obs, { color: "var(--qb-c2)", width: 3 });
      pl.legend([{ label: "observed", color: "var(--qb-c2)" }, { label: "39 random B in the cells", color: "#999", kind: "box" }], "tr");
      const above = radii.filter((_, k) => c.obs[k] > c.hi[k]), below = radii.filter((_, k) => c.obs[k] < c.lo[k]);
      txt = `A ${sc.detA.length}, B ${sc.detB.length} detected\nabove the envelope at\n  ${above.length ? above[0] + "–" + above[above.length - 1] + " nm" : "no radius"}\nbelow the envelope at\n  ${below.length ? below[0] + "–" + below[below.length - 1] + " nm" : "no radius"}\nL − r at 100 nm   ${fmt(c.obs[3], 0)} nm`;
    } else if (st.tab === "objects") {
      const bs = makeBlobScene({ rA: st.rA, rB: st.rB, dPx: st.dNm / PIXEL_NM, pair: st.pair });
      const a = Float64Array.from(bs.labA, v => (v ? 1 : 0)), b = Float64Array.from(bs.labB, v => (v ? 1 : 0));
      baseImage(a, b, [0, 1], [0, 1]);
      const ctx = cv.getContext("2d")!;
      for (const q of bs.rel) {
        ctx.fillStyle = q.centroidIn ? "#ffd400" : "#ffffff";
        ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc((q.cx + 0.5) * Z, (q.cy + 0.5) * Z, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
      imgCap.textContent = "A objects green, B magenta; dot: A centroid (yellow = inside B)";
      const vals = [bs.anyFrac, bs.halfFrac, bs.centroidFrac];
      pl.setDomain([-0.5, 2.5], [0, 1.05]);
      pl.setLabels({ xlabel: "", ylabel: "share of A objects" });
      pl.bars([0, 1, 2], vals, { colors: ["var(--qb-c1)", "var(--qb-c3)", "var(--qb-c4)"], barWidth: 0.6 });
      ["any pixel", "> 50 % of area", "centroid in B"].forEach((l, i) => pl.text(i, vals[i] + 0.04, l, { anchor: "middle", size: FS * 0.8 }));
      txt = `A objects ${bs.nA}, B objects ${bs.nB}\noverlap, any pixel  ${pct(bs.anyFrac)}\noverlap > 50 %      ${pct(bs.halfFrac)}\ncentroid in B       ${pct(bs.centroidFrac)}\ndistance ${fmt(st.dNm / PIXEL_NM, 1)} px`;
    } else {
      drawSpots(null);
      const pd = pairedDistances(sc);
      const bw = 5, hi = Math.max(200, st.dNm + 150), nb = Math.ceil(hi / bw), cnt = new Array(nb).fill(0);
      for (const v of pd.measured) { const k = Math.floor(v / bw); if (k < nb) cnt[k]++; }
      const mean = pd.measured.reduce((x, y) => x + y, 0) / Math.max(1, pd.measured.length);
      pl.setDomain([0, hi], [0, Math.max(3, ...cnt) * 1.7]);
      pl.setLabels({ xlabel: "measured distance of pairs (nm)", ylabel: "pairs" });
      pl.histogram(Array.from({ length: nb + 1 }, (_, k) => k * bw), cnt, { color: "var(--qb-c1)", fillOpacity: 0.6 });
      pl.vline(st.dNm, { color: "var(--qb-c3)", width: 3 });
      pl.vline(mean, { color: "var(--qb-c2)", width: 3 });
      pl.vline(pd.predicted, { color: "var(--qb-c4)", width: 2.5, dash: "6 4" });
      pl.legend([{ label: "true distance", color: "var(--qb-c3)" }, { label: "measured mean", color: "var(--qb-c2)" }, { label: "predicted mean", color: "var(--qb-c4)", dash: "6 4" }], "tr");
      txt = `pairs measured  ${pd.measured.length}\ntrue distance   ${st.dNm} nm\nmeasured mean   ${fmt(mean, 0)} nm\npredicted mean  ${fmt(pd.predicted, 0)} nm\nprecision A, B  ${fmt(pd.sigA, 0)}, ${fmt(pd.sigB, 0)} nm\npixel r (cells) ${fmt(pixelR(sc))}`;
    }
    ro.textContent = txt;
  }

  for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].set(st[k] as number);
  draw();
  if (p0) pb.forEach(b => b.classList.toggle("active", b.dataset.id === p0.id));
}

autoMount(NAME, mount);
