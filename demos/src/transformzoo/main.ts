// The transform zoo. A reference square with a grid and six numbered landmarks; drag the
// target points (orange). The chosen model (translation, rigid, similarity, affine,
// homography) is fitted by least squares to the first n pairs and drawn as the warped grid,
// with residual arrows from each mapped landmark to its target. Too few pairs: "needs n
// pairs". "Add an outlier" moves one target far away: the least-squares fit skews.
//
// <div data-demo="transformzoo" data-preset="one|two|three|four|more|outlier" data-model="affine"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, draggable, svgPoint } from "../lib/ui";
import { injectStyle } from "../canny/imgview";
import { Kind, KINDS, fitTransform, apply, dof, minimalSize, rmsError, P2 } from "../lib/transforms2d";

const NAME = "transformzoo";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .tz-svg { flex: 1 1 460px; max-width: 560px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .tz-svg svg { width: 100%; display: block; background: var(--qb-panel); border-radius: 6px; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 380px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 20px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .tz-svg { flex: 0 0 560px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 440px; max-width: 440px; gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 17px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; }
`;

const SRC: P2[] = [{ x: 100, y: 100 }, { x: 300, y: 100 }, { x: 300, y: 300 }, { x: 100, y: 300 }, { x: 210, y: 160 }, { x: 150, y: 250 }];
/** Default targets: a mild perspective map of the square plus small bumps, so every model leaves some residual. */
function defaultTargets(): P2[] {
  const H = [1.05, 0.22, 40, -0.12, 0.95, 60, 0.0016, -0.0005, 1];
  const bump = [[3, -2], [-2, 4], [4, 3], [-3, -2], [2, 3], [-3, 1]];
  return SRC.map((p, i) => { const [x, y] = apply(H, p.x, p.y); return { x: x + bump[i][0], y: y + bump[i][1] }; });
}
const KIND_LABEL: Record<Kind, string> = { translation: "translation", rigid: "rigid", similarity: "similarity", affine: "affine", homography: "homography" };

interface State { kind: Kind; n: number; outlier: boolean; tgt: P2[] }
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "one", label: "one pair: a translation", s: { kind: "translation", n: 1 } },
  { id: "two", label: "two pairs: similarity", s: { kind: "similarity", n: 2 } },
  { id: "three", label: "three: affine (parallel lines stay parallel)", s: { kind: "affine", n: 3 } },
  { id: "four", label: "four: homography (lines meet)", s: { kind: "homography", n: 4 } },
  { id: "more", label: "more pairs than needed: least squares", s: { kind: "affine", n: 6 } },
  { id: "outlier", label: "one outlier ruins it", s: { kind: "affine", n: 6, outlier: true } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ kind: (host.dataset.model as Kind) ?? "affine", n: 6, outlier: false, tgt: defaultTargets(), ...(p0?.s ?? {}) });
  const st: State = start();

  const wrap = htmlEl("div", "tz-svg", "", host);
  const svg = svgEl("svg", { viewBox: "0 0 480 420" }, wrap);
  const gGrid = svgEl("g", {}, svg), gWarp = svgEl("g", {}, svg), gRes = svgEl("g", {}, svg), gPts = svgEl("g", {}, svg), gMsg = svgEl("g", {}, svg);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, start(), { outlier: false }, p.s, { tgt: defaultTargets() }); sync(); draw(); } })));
  htmlEl("div", "qb-note", "model", panel);
  const kindBtns = presets(panel, KINDS.map(k => ({ label: KIND_LABEL[k], apply: () => { st.kind = k; draw(); } })));
  const nS = slider(panel, "pairs used", 1, 6, 1, st.n, v => { st.n = v; draw(); });
  const oC = checkbox(panel, "add an outlier (last pair used)", st.outlier, v => { st.outlier = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { nS.set(st.n); oC.checked = st.outlier; }

  const OUT = { x: 95, y: -75 };
  const targets = (): P2[] => st.tgt.map((p, i) => st.outlier && i === st.n - 1 ? { x: p.x + OUT.x, y: p.y + OUT.y } : p);

  // the reference square and its grid (faint)
  const gridLines: P2[][] = [];
  for (let k = 0; k <= 4; k++) {
    const t = 100 + 50 * k;
    gridLines.push(Array.from({ length: 21 }, (_, j) => ({ x: t, y: 100 + 10 * j })));
    gridLines.push(Array.from({ length: 21 }, (_, j) => ({ x: 100 + 10 * j, y: t })));
  }
  for (const L of gridLines) { const pl = svgEl("polyline", { points: L.map(p => `${p.x},${p.y}`).join(" "), fill: "none", "stroke-width": 1.5 }, gGrid); pl.style.stroke = "#b8bec6"; }

  // handles
  const handles = SRC.map((_, i) => {
    const g = svgEl("g", {}, gPts);
    const c = svgEl("circle", { r: 13, class: "handle" }, g);
    c.style.stroke = "#ff7f0e"; c.style.fill = "#fff";
    const t = svgEl("text", { "text-anchor": "middle", "font-size": 16, "font-weight": 700, "pointer-events": "none" }, g);
    t.textContent = String(i + 1);
    draggable(c, ev => svgPoint(svg, ev), (x, y) => {
      const ox = st.outlier && i === st.n - 1 ? OUT.x : 0, oy = st.outlier && i === st.n - 1 ? OUT.y : 0;
      st.tgt[i] = { x: Math.max(5, Math.min(475, x)) - ox, y: Math.max(5, Math.min(415, y)) - oy }; draw();
    });
    return { g, c, t };
  });
  const srcDots = SRC.map((p, i) => {
    const c = svgEl("circle", { cx: p.x, cy: p.y, r: 6 }, gGrid); c.style.fill = "#7a828c";
    const t = svgEl("text", { x: p.x - 12, y: p.y - 9, "font-size": 15, "text-anchor": "middle" }, gGrid); t.textContent = String(i + 1); t.style.fill = "#7a828c";
    return c;
  });
  void srcDots;

  function draw(): void {
    kindBtns.forEach((b, i) => b.classList.toggle("active", KINDS[i] === st.kind));
    const tg = targets(), n = st.n, need = minimalSize(st.kind);
    handles.forEach((h, i) => {
      h.c.setAttribute("cx", String(tg[i].x)); h.c.setAttribute("cy", String(tg[i].y));
      h.t.setAttribute("x", String(tg[i].x)); h.t.setAttribute("y", String(tg[i].y + 6));
      h.g.style.opacity = i < n ? "1" : "0.3";
      h.c.style.stroke = st.outlier && i === n - 1 ? "#d62728" : "#ff7f0e";
    });
    gWarp.replaceChildren(); gRes.replaceChildren(); gMsg.replaceChildren();
    const T = n >= need ? fitTransform(st.kind, SRC.slice(0, n), tg.slice(0, n)) : null;
    if (!T) {
      const t = svgEl("text", { x: 240, y: 395, "text-anchor": "middle", "font-size": 22, "font-weight": 600 }, gMsg);
      t.textContent = n < need ? `${KIND_LABEL[st.kind]} needs ${need} pair${need > 1 ? "s" : ""}` : "degenerate: no unique fit";
      t.style.fill = "#d62728";
      ro.textContent = `${KIND_LABEL[st.kind]}: ${dof(st.kind)} numbers, needs ${need} pair${need > 1 ? "s" : ""}\npairs used: ${n}`;
      return;
    }
    for (const L of gridLines) {
      const pl = svgEl("polyline", { points: L.map(p => apply(T, p.x, p.y).map(v => v.toFixed(1)).join(",")).join(" "), fill: "none", "stroke-width": 2.2 }, gWarp);
      pl.style.stroke = "#1f77b4"; pl.style.opacity = "0.8";
    }
    SRC.forEach((p, i) => {
      if (i >= n) return;
      const [x, y] = apply(T, p.x, p.y);
      const d = svgEl("circle", { cx: x, cy: y, r: 5 }, gRes); d.style.fill = "#1f77b4";
      if (Math.hypot(tg[i].x - x, tg[i].y - y) > 1.5) {
        const l = svgEl("line", { x1: x, y1: y, x2: tg[i].x, y2: tg[i].y, "stroke-width": 3 }, gRes); l.style.stroke = "#d62728";
      }
    });
    const rms = rmsError(T, SRC.slice(0, n), tg.slice(0, n));
    const extra = 2 * n - dof(st.kind);
    ro.textContent = `${KIND_LABEL[st.kind]}: ${dof(st.kind)} numbers, needs ${need} pair${need > 1 ? "s" : ""}\npairs used: ${n} → ${2 * n} equations` +
      (extra > 0 ? `, ${extra} more than unknowns\nleast-squares fit, RMS residual ${rms.toFixed(1)}` : extra === 0 ? `\nexact fit (residual ${rms.toFixed(1)})` : `\nunder-determined (one of many fits)`);
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  draw();
}

autoMount(NAME, mount);
