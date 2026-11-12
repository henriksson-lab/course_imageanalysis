// The generalised Hough transform (Ballard 1981). Left: the template (drag its vertices or
// its reference point +), with the R-table drawn as arrows from edge points to the
// reference point, coloured by direction bin, and the table itself (bin -> vectors).
// Middle: a scene with copies of the template (translated, optionally rotated and
// scaled), clutter, occlusion and noise; its Canny edge points; detections drawn as the
// template outline at the found pose. Right: the accumulator (maximum over the searched
// rotations and scales). Click an edge point of the scene: its votes for the searched
// pose of the strongest detection appear as dots.
//
// <div data-demo="ght" data-preset="translate|occluded|rotmiss|rotsearch|scale|similar|circle|square"></div>
import { autoMount, htmlEl, slider, presets, button, svgEl, draggable, svgPoint, Slider } from "../lib/ui";
import { Grey } from "../lib/imageops";
import { canny, edgeList, EdgePoint } from "../lib/edges";
import { OrientedPt, buildRTable, ghtVote, ghtVotesOf, peaks3, placePolygon, RTable, Peak3, dirBin } from "../lib/hough";
import { TEMPLATES, Template, templateEdgePoints, makeGhtScene, Copy, N, Poly, CANNY } from "./scenes";
import { ImgPanel, injectStyle, PANEL_CSS, greyRGBA, heatRGBA, COL, fmtInt, maxOf } from "../houghlines/view";
import { maxOverSlices } from "../lib/hough";

const NAME = "ght";
const STYLE = PANEL_CSS(NAME) + `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .gh-left { flex: 1 1 820px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .gh-figs { display: flex; gap: .6em; }
[data-demo="${NAME}"] .gh-figs > * { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .gh-tpl svg { width: 100%; aspect-ratio: 1; display: block; background: #20232a; border-radius: 3px; touch-action: none; }
[data-demo="${NAME}"] .gh-bottom { display: flex; gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .gh-bottom .qb-presets { flex: 1 1 40%; }
[data-demo="${NAME}"] .gh-table { flex: 1 1 28%; font-size: 13px; white-space: pre; overflow: hidden; max-height: 11em; }
[data-demo="${NAME}"] .gh-bottom .qb-readout.ro { flex: 1 1 32%; font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; min-width: 230px; gap: .3em; }
[data-demo="${NAME}"] .qb-slider { grid-template-columns: 6.6em 1fr 3.6em; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .gh-left { flex: 0 0 830px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 300px; gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .06em .4em; }
@media (max-width: 860px) { [data-demo="${NAME}"], [data-demo="${NAME}"] .gh-bottom { flex-wrap: wrap; } }
`;

interface State {
  tpl: string; nBins: number; nRot: number; nScale: number; thr: number; noise: number; occlusion: number; clutter: number;
  copies: Copy[];
}
interface Preset { id: string; label: string; s: Partial<State> }
const D2R = Math.PI / 180;
const DEFAULT: State = {
  tpl: "key", nBins: 36, nRot: 1, nScale: 1, thr: 0.5, noise: 6, occlusion: 0, clutter: 8,
  copies: [{ x: 48, y: 50, phi: 0, s: 1 }, { x: 112, y: 112, phi: 0, s: 1 }],
};
const PRESETS: Preset[] = [
  { id: "translate", label: "translation only", s: {} },
  { id: "occluded", label: "occluded copy still found", s: { occlusion: 0.5 } },
  { id: "rotmiss", label: "rotated: missed", s: { copies: [{ x: 48, y: 50, phi: 0, s: 1 }, { x: 112, y: 112, phi: 60 * D2R, s: 1 }] } },
  { id: "rotsearch", label: "rotation search: found", s: { nRot: 36, copies: [{ x: 48, y: 50, phi: 0, s: 1 }, { x: 112, y: 112, phi: 60 * D2R, s: 1 }] } },
  { id: "scale", label: "scale search", s: { nScale: 5, copies: [{ x: 48, y: 50, phi: 0, s: 0.8 }, { x: 110, y: 110, phi: 0, s: 1.2 }] } },
  { id: "similar", label: "similar shape: false positive", s: { thr: 0.3, copies: [{ x: 48, y: 50, phi: 0, s: 1 }, { x: 110, y: 112, phi: 0, s: 1, similar: true }] } },
  { id: "circle", label: "circle template = circle Hough", s: { tpl: "circle", nBins: 36, thr: 0.5, copies: [{ x: 45, y: 48, phi: 0, s: 1 }, { x: 115, y: 110, phi: 0, s: 1 }] } },
  { id: "square", label: "square, 4 bins", s: { tpl: "square", nBins: 4, thr: 0.5, clutter: 0, copies: [{ x: 50, y: 50, phi: 0, s: 1 }, { x: 110, y: 110, phi: 0, s: 1 }] } },
];
const ROTS = [1, 4, 8, 12, 18, 36, 72];
const SCALES = [1, 3, 5];
const scaleList = (n: number) => (n === 1 ? [1] : Array.from({ length: n }, (_, i) => 0.8 + 0.4 * i / (n - 1)));
const rotList = (n: number) => Array.from({ length: n }, (_, i) => 2 * Math.PI * i / n);
const BIN_COL = (b: number, n: number) => `hsl(${Math.round(360 * b / n)}, 85%, 60%)`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const fresh = (p?: Preset): State => {
    const s = { ...DEFAULT, ...(p?.s ?? {}) };
    s.copies = s.copies.map(c => ({ ...c }));
    return s;
  };
  let st: State = fresh(p0);
  let tpl: Template = cloneTpl(st.tpl), ref = { x: 0, y: 0 };

  // ---- layout ----
  const left = htmlEl("div", "gh-left", "", host);
  const figs = htmlEl("div", "gh-figs", "", left);
  const tFig = htmlEl("figure", "hv-fig gh-tpl", "", figs);
  htmlEl("figcaption", "", "template and R-table", tFig);
  const T = 34;
  const tsvg = svgEl("svg", { viewBox: `${-T} ${-T} ${2 * T} ${2 * T}` }, tFig);
  const tArrows = svgEl("g", {}, tsvg), tPoly = svgEl("g", {}, tsvg), tHandles = svgEl("g", {}, tsvg);
  const pS = new ImgPanel(figs, "scene: edges and detections", N, N);
  const pA = new ImgPanel(figs, "accumulator (max over poses)", N, N);
  const bottom = htmlEl("div", "gh-bottom", "", left);
  const presetBtns = presets(bottom, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const tableDiv = htmlEl("div", "qb-readout gh-table", "", bottom);
  const ro = htmlEl("div", "qb-readout ro", "", bottom);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hv-lab", "template (drag vertices and +)", panel);
  const tplBtns = presets(panel, TEMPLATES.map(t => ({ label: t.label, apply: () => { st.tpl = t.id; tpl = cloneTpl(t.id); ref = { x: 0, y: 0 }; makeScene(); } })));
  const sl: Record<string, Slider> = {};
  sl.nBins = slider(panel, "direction bins", 4, 72, 1, st.nBins, v => { st.nBins = v; vote(); });
  sl.nRot = slider(panel, "rotations", 0, ROTS.length - 1, 1, ROTS.indexOf(st.nRot), v => { st.nRot = ROTS[v]; vote(); }, v => (ROTS[v] === 1 ? "off" : `${ROTS[v]}`));
  sl.nScale = slider(panel, "scales", 0, SCALES.length - 1, 1, SCALES.indexOf(st.nScale), v => { st.nScale = SCALES[v]; vote(); }, v => (SCALES[v] === 1 ? "off" : `${SCALES[v]}`));
  sl.thr = slider(panel, "threshold", 0.1, 1, 0.05, st.thr, v => { st.thr = v; detect(); });
  sl.noise = slider(panel, "noise", 0, 40, 1, st.noise, v => { st.noise = v; makeScene(); });
  sl.occlusion = slider(panel, "occlusion", 0, 1, 0.05, st.occlusion, v => { st.occlusion = v; makeScene(); });
  sl.clutter = slider(panel, "clutter", 0, 30, 1, st.clutter, v => { st.clutter = v; makeScene(); });
  button(panel, "reset", () => { st = fresh(p0); tpl = cloneTpl(st.tpl); ref = { x: 0, y: 0 }; presetBtns.forEach(b => b.classList.remove("active")); sync(); makeScene(); });
  htmlEl("div", "qb-note", "Score: edge points voting within ±1 px of a cell, as a fraction of the template's edge points (found by the same edge detector). Rotations over 360°; scales 0.8–1.2.", panel);

  function load(p: Preset): void { st = fresh(p); tpl = cloneTpl(st.tpl); ref = { x: 0, y: 0 }; sync(); makeScene(); }
  function sync(): void {
    sl.nBins.set(st.nBins); sl.nRot.set(ROTS.indexOf(st.nRot)); sl.nScale.set(SCALES.indexOf(st.nScale)); sl.thr.set(st.thr);
    sl.noise.set(st.noise); sl.occlusion.set(st.occlusion); sl.clutter.set(st.clutter);
  }

  // ---- computation ----
  let img: Grey | null = null, pts: EdgePoint[] = [], edgeIdx: number[] = [], table: RTable | null = null;
  let acc: Float32Array | null = null, maxMap: Float32Array | null = null, nSl = 1, votes = 0, lookups = 0, ms = 0, found: Peak3[] = [];
  let phis = [0], scales = [1], tplN = 1, pick: EdgePoint | null = null, tp: OrientedPt[] = [];
  function makeScene(): void {
    tplBtns.forEach((b, i) => b.classList.toggle("active", TEMPLATES[i].id === st.tpl));
    const base = TEMPLATES.find(t => t.id === st.tpl)!;
    img = makeGhtScene(base, { copies: st.copies, clutter: st.clutter, noise: st.noise, occlusion: st.occlusion }).img;
    const c = canny(img, CANNY.sigma, CANNY.lo, CANNY.hi);
    pts = edgeList(c.hyst.edges, c.gx, c.gy, c.mag);
    edgeIdx = pts.map(p => p.y * N + p.x);
    pick = null;
    vote();
  }
  let pending = false;
  function vote(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      tp = templateEdgePoints(tpl);
      tplN = tp.length;
      table = buildRTable(tp, ref, st.nBins);
      phis = rotList(st.nRot); scales = scaleList(st.nScale);
      const t0 = performance.now();
      const G = ghtVote(pts, table, N, N, { phis, scales, tol: 1 });
      ms = performance.now() - t0;
      nSl = phis.length * scales.length; votes = G.votes; lookups = G.lookups;
      acc = G.acc;
      const sc = 1 / tplN;
      for (let i = 0; i < acc.length; i++) acc[i] *= sc;
      maxMap = maxOverSlices(acc, N * N, nSl).max;
      drawTemplate();
      detect();
    });
  }
  function detect(): void {
    if (!acc) return;
    found = peaks3(acc, N, N, nSl, st.thr, 6, nSl, 20);
    draw();
  }

  // ---- template panel ----
  function drawTemplate(): void {
    tArrows.replaceChildren(); tPoly.replaceChildren(); tHandles.replaceChildren();
    for (let i = 0; i < tp.length; i += 2) {
      const p = tp[i];
      const b = dirBin(p.theta!, st.nBins), col = BIN_COL(b, st.nBins);
      const l = svgEl("line", { x1: p.x, y1: p.y, x2: ref.x, y2: ref.y, stroke: col, "stroke-width": 0.35, opacity: 0.75 }, tArrows);
      void l;
      svgEl("circle", { cx: p.x, cy: p.y, r: 0.9, fill: col }, tArrows);
    }
    for (const poly of [tpl.poly, ...tpl.holes]) svgEl("polygon", { points: poly.map(q => q.join(",")).join(" "), fill: "none", stroke: "#fff", "stroke-width": 0.6 }, tPoly);
    const toW = (ev: PointerEvent) => svgPoint(tsvg, ev);
    [tpl.poly, ...tpl.holes].forEach(poly => poly.forEach((v, k) => {
      if (poly.length > 24) return;   // the circle: no vertex handles
      const h = svgEl("circle", { cx: v[0], cy: v[1], r: 1.6, fill: "#fff", stroke: COL.orange, "stroke-width": 0.6 }, tHandles);
      draggable(h, toW, (x, y) => { poly[k] = [clampT(x), clampT(y)]; h.setAttribute("cx", String(poly[k][0])); h.setAttribute("cy", String(poly[k][1])); vote(); });
    }));
    const rh = svgEl("g", { transform: `translate(${ref.x},${ref.y})` }, tHandles);
    svgEl("circle", { r: 3, fill: "rgba(0,0,0,0.01)" }, rh);
    svgEl("path", { d: "M-2.6,0H2.6M0,-2.6V2.6", stroke: COL.yellow, "stroke-width": 1.1 }, rh);
    draggable(rh as unknown as SVGElement, toW, (x, y) => { ref = { x: clampT(x), y: clampT(y) }; rh.setAttribute("transform", `translate(${ref.x},${ref.y})`); vote(); });
    // the table
    if (table) {
      const lines: string[] = [`R-table: ${table.size} entries, ${table.nBins} bins`];
      const small = table.nBins <= 8;
      for (let b = 0; b < table.nBins; b++) {
        const e = table.bins[b];
        if (!e.length && !small) continue;
        const deg = Math.round(360 * b / table.nBins);
        if (small) {
          const uniq = new Map<string, number>();
          for (const v of e) { const k = `(${v[0].toFixed(0)},${v[1].toFixed(0)})`; uniq.set(k, (uniq.get(k) ?? 0) + 1); }
          const list = [...uniq.keys()];
          lines.push(`${String(deg).padStart(3)}°: ${e.length} → ${list.slice(0, 3).join(" ")}${list.length > 3 ? " …" : ""}`);
        } else lines.push(`${String(deg).padStart(3)}°: ${e.length} vectors`);
      }
      tableDiv.textContent = lines.join("\n");
    }
  }
  const clampT = (v: number) => Math.round(Math.max(-T + 2, Math.min(T - 2, v)) * 2) / 2;

  // ---- interaction ----
  pS.onPick((x, y) => {
    let best: EdgePoint | null = null, bd = 16;
    for (const p of pts) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = p; } }
    pick = best; draw();
  });

  // ---- drawing ----
  function draw(): void {
    if (!img || !acc || !maxMap) return;
    const rgb = greyRGBA(img.data, 0, 255, 0.55);
    for (const i of edgeIdx) { rgb[4 * i] = 255; rgb[4 * i + 1] = 159; rgb[4 * i + 2] = 28; }
    pS.putRGBA(rgb);
    pS.clearOverlay();
    const poseOf = (f: Peak3) => ({ phi: phis[f.s % phis.length], s: scales[Math.floor(f.s / phis.length)] });
    for (const f of found) {
      const { phi, s } = poseOf(f);
      for (const poly of [tpl.poly, ...tpl.holes]) pS.polyline(placePolygon(poly, ref, phi, s, { x: f.x, y: f.y }) as [number, number][], COL.cyan, 2.5, true);
      pS.dot(f.x, f.y, 2.5, COL.yellow);
    }
    if (pick && table) {
      const pose = found.length ? poseOf(found[0]) : { phi: 0, s: 1 };
      pS.circle(pick.x, pick.y, 3.5, COL.magenta, 3);
      for (const [x, y] of ghtVotesOf(pick, table, pose.phi, pose.s)) pS.dot(x, y, 1.8, COL.magenta);
    }
    pA.putRGBA(heatRGBA(maxMap, Math.max(0.3, maxOf(maxMap))));
    pA.clearOverlay();
    for (const f of found) pA.circle(f.x, f.y, 6, COL.cyan, 2.5);
    const lines = [
      `scene edge points ${fmtInt(pts.length)}`,
      `poses ${phis.length} × ${scales.length} = ${nSl} slices of ${N}×${N}`,
      `lookups ${fmtInt(lookups)}   votes ${fmtInt(votes)}`,
      `voting ${ms.toFixed(0)} ms   detected ${found.length}`,
    ];
    for (const f of found.slice(0, 4)) { const { phi, s } = poseOf(f); lines.push(`  (${f.x}, ${f.y}) ${Math.round(phi / D2R)}°, ×${s.toFixed(1)}: ${f.value.toFixed(2)}`); }
    if (pick) lines.push(`point (${pick.x}, ${pick.y}): bin ${dirBin(pick.theta - (found.length ? poseOf(found[0]).phi : 0), st.nBins)}`);
    ro.textContent = lines.join("\n");
  }

  sync();
  makeScene();
}

function cloneTpl(id: string): Template {
  const t = TEMPLATES.find(q => q.id === id) ?? TEMPLATES[0];
  return { ...t, poly: t.poly.map(p => [p[0], p[1]] as [number, number]), holes: t.holes.map(h => h.map(p => [p[0], p[1]] as [number, number])) };
}

void ({} as Poly);
autoMount(NAME, mount);
