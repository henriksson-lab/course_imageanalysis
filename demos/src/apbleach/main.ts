// Acceptor photobleaching. A seeded field with two FRET cells (donor + acceptor, efficiency E,
// a fraction f of donors paired) and one acceptor-only control cell. Drag on the "donor
// before" tile to draw the bleach region (default: the right half of cell 1). Each bleach
// pulse destroys 30 % of the acceptors left in the region; the donor there dequenches. The
// donor may also bleach slowly everywhere (per pulse), the cell may drift between before and
// after, and bleached acceptors may turn into a donor-coloured species (photoconversion).
// Tiles: donor before, donor after, acceptor after, and the E map 1 − pre/post (pixel-wise,
// optional smoothing of both images first). Plot: donor and acceptor in the region and the
// donor in the unbleached control part of the FRET cells, per pulse (expected values,
// relative to before). Corrections: incomplete bleaching (from the measured acceptor loss),
// donor bleaching (from the control region), alignment (integer shift by phase correlation).
//
// <div data-demo="apbleach" data-preset="ideal|early|donorbleach|moved|few|half|photo"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { diverging, scalarToRGBA } from "../lib/colour";
import { blurMap } from "../lib/unmix";
import { correlate2d, findPeaks } from "../lib/registration";
import * as F from "../lib/fret";
import { Tile, tileCss, injectStyle, fmt, debounce, filterPresets, hideControl } from "../threecube/view";

const NAME = "apbleach";
const Q = `.qb-demo[data-demo="${NAME}"]`;
const CSS = tileCss(NAME) + `
${Q} { gap: .7em; align-items: flex-start; }
${Q} .ab-tiles { flex: 0 1 340px; min-width: 240px; display: grid; grid-template-columns: 1fr 1fr; gap: .3em; }
${Q} .ab-mid { flex: 1 1 340px; min-width: 240px; max-width: 400px; display: flex; flex-direction: column; gap: .3em; }
${Q} .qb-panel { flex: 1 1 280px; max-width: 360px; }
.reveal ${Q} { flex-wrap: nowrap; }
.reveal ${Q} .ab-tiles { flex: 0 0 360px; }
.reveal ${Q} .ab-mid { flex: 0 0 370px; }
.reveal ${Q} .qb-panel { flex: 0 0 350px; }
`;

interface State {
  E: number; f: number; pulses: number; donorLoss: number; drift: number; photons: number; photo: number;
  smooth: number; corrIncomplete: boolean; corrDonor: boolean; align: boolean;
  rect: [number, number, number, number] | null;
}
const DEFAULT: State = { E: 0.3, f: 1, pulses: 15, donorLoss: 0, drift: 0, photons: 600, photo: 0, smooth: 0, corrIncomplete: false, corrDonor: false, align: false, rect: null };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "ideal", label: "ideal", s: {} },
  { id: "early", label: "stop early: incomplete bleaching", s: { pulses: 4 } },
  { id: "donorbleach", label: "donor bleaching fakes less FRET", s: { donorLoss: 0.015 } },
  { id: "moved", label: "the cell moved: edges", s: { drift: 3 } },
  { id: "few", label: "few photons: noisy map, region fine", s: { photons: 40 } },
  { id: "half", label: "half the donors paired", s: { f: 0.5 } },
  { id: "photo", label: "photoconversion: the control lights up", s: { photo: 0.15 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const pre = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(pre?.s ?? {}) };

  const tiles = htmlEl("div", "ab-tiles", "", host);
  const tPre = new Tile(tiles, "donor before"), tPost = new Tile(tiles, "donor after");
  const tAcc = new Tile(tiles, "acceptor after"), tE = new Tile(tiles, "E map, ±0.6");
  const mid = htmlEl("div", "ab-mid", "", host);
  const tp = new Plot(mid, { width: host.closest(".reveal") ? 370 : 380, height: host.closest(".reveal") ? 280 : 290, x: [0, 15], y: [0, 2], xlabel: "bleach pulse", ylabel: "relative to before", yticks: [0, 0.5, 1, 1.5, 2], margin: { l: 64, r: 12, t: 10, b: 54 } });
  const lD = tp.line([], [], { color: "#1b9fc0", width: 4, label: "donor, region" });
  const lA = tp.line([], [], { color: "#d39e00", width: 4, label: "acceptor, region" });
  const lC = tp.line([], [], { color: "#555", width: 3, dash: "7 5", label: "donor, control" });
  tp.legend(undefined, "tr");
  const ro = htmlEl("div", "qb-readout", "", mid);
  htmlEl("div", "fr-lab", "drag on \"donor before\" to draw a new bleach region", tiles).style.gridColumn = "1 / 3";

  const panel = htmlEl("div", "qb-panel", "", host);
  const PL = filterPresets(host, PRESETS);
  const presetBtns = presets(panel, PL.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  const regen = debounce(() => rebuild(), 80);
  const ch = (k: keyof State) => (v: number) => { (st as unknown as Record<string, number>)[k] = v; off(); regen(); };
  const eS = slider(panel, "true E", 0, 0.8, 0.01, st.E, ch("E"));
  const fS = slider(panel, "fraction paired f", 0, 1, 0.05, st.f, ch("f"));
  const pS = slider(panel, "pulses", 0, 20, 1, st.pulses, ch("pulses"));
  const dS = slider(panel, "donor loss / pulse", 0, 0.05, 0.001, st.donorLoss, ch("donorLoss"));
  const xS = slider(panel, "drift (px)", 0, 6, 1, st.drift, ch("drift"));
  const phS = slider(panel, "photons", 10, 1000, 10, st.photons, ch("photons"));
  const pcS = slider(panel, "photoconversion", 0, 0.3, 0.01, st.photo, ch("photo"));
  const smS = slider(panel, "smooth σ (px)", 0, 4, 0.5, st.smooth, v => { st.smooth = v; off(); draw(); });
  const c1 = checkbox(panel, "correct incomplete bleaching", st.corrIncomplete, v => { st.corrIncomplete = v; off(); draw(); });
  const c2 = checkbox(panel, "correct donor bleaching (control)", st.corrDonor, v => { st.corrDonor = v; off(); draw(); });
  const c3 = checkbox(panel, "align after to before", st.align, v => { st.align = v; off(); draw(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "+1 pulse", () => { st.pulses = Math.min(20, st.pulses + 1); pS.set(st.pulses); off(); rebuild(); });
  button(brow, "reset", () => { Object.assign(st, DEFAULT); presetBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  const off = () => presetBtns.forEach(b => b.classList.remove("active"));
  if (host.dataset.compact !== undefined) [dS, xS, phS, pcS, smS].forEach(hideControl);
  function sync(): void {
    eS.set(st.E); fS.set(st.f); pS.set(st.pulses); dS.set(st.donorLoss); xS.set(st.drift); phS.set(st.photons); pcS.set(st.photo); smS.set(st.smooth);
    c1.checked = st.corrIncomplete; c2.checked = st.corrDonor; c3.checked = st.align;
  }

  // draw a new bleach region on the "before" tile
  let drag: [number, number] | null = null;
  tPre.onDrag((x, y) => { drag = [x, y]; }, (x, y) => {
    if (!drag) return;
    const r: [number, number, number, number] = [Math.floor(Math.min(drag[0], x)), Math.floor(Math.min(drag[1], y)), Math.ceil(Math.max(drag[0], x)), Math.ceil(Math.max(drag[1], y))];
    if (r[2] - r[0] >= 2 && r[3] - r[1] >= 2) { st.rect = r; off(); regen(); }
  });

  let sc: F.BleachScene, rect: [number, number, number, number];
  function rebuild(): void {
    sc = F.bleachScene({ E: st.E, f: st.f, pulses: st.pulses, donorLoss: st.donorLoss, photons: st.photons, drift: [st.drift, Math.round(st.drift * 2 / 3)], rect: st.rect ?? undefined, photoconvert: st.photo, seed: 4 });
    rect = st.rect ?? defaultRect();
    draw();
  }
  function defaultRect(): [number, number, number, number] {
    // recover the default region's bounding box from the scene's region mask (cell 1 part)
    let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
    for (let i = 0; i < sc.region.length; i++) if (sc.region[i] && sc.labels[i] !== sc.accOnlyCell) { const x = i % sc.w, y = (i - x) / sc.w; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
    return x1 < 0 ? [0, 0, 0, 0] : [x0, y0, x1, y1];
  }

  function draw(): void {
    const w = sc.w, h = sc.h, n = w * h;
    let dPost = sc.donorPost, aPost = sc.accPost;
    let shift: [number, number] = [0, 0];
    if (st.align) {
      const c = correlate2d(sc.donorPre, dPost, w, h);
      const pk = findPeaks(c, 1)[0];
      // pick the sign that matches best
      const cands: [number, number][] = [[pk.tx, pk.ty], [-pk.tx, -pk.ty]];
      let best = Infinity;
      for (const [tx, ty] of cands) {
        const s = F.shiftMap(dPost, w, h, tx, ty);
        let e = 0; for (let i = 0; i < n; i++) e += (s[i] - sc.donorPre[i]) ** 2;
        if (e < best) { best = e; shift = [tx, ty]; }
      }
      dPost = F.shiftMap(dPost, w, h, shift[0], shift[1]); aPost = F.shiftMap(aPost, w, h, shift[0], shift[1]);
    }
    const P = st.smooth > 0 ? blurMap(sc.donorPre, w, h, st.smooth) : sc.donorPre;
    const Qp = st.smooth > 0 ? blurMap(dPost, w, h, st.smooth) : dPost;
    // region sums (FRET cells only) and the control region
    let sPre = 0, sPost = 0, sApre = 0, sApost = 0, cPre = 0, cPost = 0, accOnlyPre = 0, accOnlyPost = 0, nAcc = 0;
    const fretPix = (i: number) => sc.labels[i] && sc.labels[i] !== sc.accOnlyCell;
    for (let i = 0; i < n; i++) {
      if (sc.region[i] && fretPix(i)) { sPre += sc.donorPre[i]; sPost += dPost[i]; sApre += sc.accPre[i]; sApost += aPost[i]; }
      if (sc.control[i]) { cPre += sc.donorPre[i]; cPost += dPost[i]; }
      if (sc.region[i] && sc.labels[i] === sc.accOnlyCell) { accOnlyPre += sc.donorPre[i]; accOnlyPost += dPost[i]; nAcc++; }
    }
    const b = sApre > 0 ? Math.min(1, Math.max(0, 1 - sApost / sApre)) : 1;
    const cf = st.corrDonor && cPost > 0 ? cPre / cPost : 1;
    const corr = (E: number) => (st.corrIncomplete && b > 0 ? F.correctIncomplete(E, b) : E);
    const Ereg = corr(F.eFromBleach(sPre, sPost * cf));
    // pixel map
    const Emap = new Float32Array(n), thr = 0.15 * st.photons;
    let sumE = 0, nE = 0;
    for (let i = 0; i < n; i++) {
      if (!(P[i] > thr || Qp[i] > thr)) { Emap[i] = NaN; continue; }
      const e = corr(F.eFromBleach(P[i], Qp[i] * cf));
      Emap[i] = e;
      if (sc.region[i] && fretPix(i) && Number.isFinite(e)) { sumE += e; nE++; }
    }
    const hi = 1.2 * st.photons;
    tPre.grey(sc.donorPre, w, h, 0, hi); tPost.grey(dPost, w, h, 0, hi); tAcc.grey(aPost, w, h, 0, hi);
    tE.rgba(scalarToRGBA(Emap, { map: diverging, lo: -0.6, hi: 0.6, nan: [40, 40, 40, 255] }), w, h);
    for (const t of [tPre, tPost, tAcc, tE]) {
      t.clear();
      t.add("rect", { x: rect[0], y: rect[1], width: rect[2] - rect[0], height: rect[3] - rect[1], fill: "none", stroke: "#ffd400", "stroke-width": 2.5, "stroke-dasharray": "5 3" });
    }
    tE.setCaption("E map, ±0.6");
    // time course (expected, relative to before; donor includes the correction factor? no: raw)
    const k = Array.from({ length: st.pulses + 1 }, (_, i) => i);
    const ser = sc.series, rel = (a: number[]) => a.map(v => (a[0] ? v / a[0] : 0));
    tp.setDomain([0, Math.max(1, st.pulses)], [0, 2]);
    lD.set(k, rel(ser.regionDonor)); lA.set(k, rel(ser.regionAcc)); lC.set(k, rel(ser.controlDonor));
    const lines = [
      `region E (ratio of sums): ${fmt(Ereg, 3)}`,
      `region, mean of pixel E: ${fmt(nE ? sumE / nE : NaN, 3)}`,
      `true E = ${fmt(st.E, 2)}, f·E = ${fmt(st.f * st.E, 3)}`,
      `acceptor left in region: ${fmt(100 * (1 - b), 1)} %`,
      `control after/before: ${fmt(cPost / cPre, 3)}`,
      `acc.-only cell, donor: ${fmt(accOnlyPre / Math.max(1, nAcc), 0)} → ${fmt(accOnlyPost / Math.max(1, nAcc), 0)}`,
    ];
    if (st.align) lines.push(`estimated shift: (${shift[0]}, ${shift[1]}) px`);
    ro.textContent = lines.join("\n");
    void svgEl;
  }

  sync();
  rebuild();
  const pi = PL.findIndex(p => p.id === host.dataset.preset);
  if (pi >= 0) presetBtns[pi].classList.add("active");
}

autoMount(NAME, mount);
