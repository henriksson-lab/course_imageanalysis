// momentphase (moments elective, unit moment-invariants/): complex moments as arrows.
// A shape on the shared shape canvas (lib/shapecanvas.ts) is rotated by α (slider or "play")
// about its centroid. Six small complex planes show c20, c11, c30, c21, c12 of the normalised
// central moments (y up, so α turns counter-clockwise on screen) and the product c30·c12³.
// Rotating by α turns c_pq by (p − q)α: c11 stays, c20 turns twice as fast as c21, c12 turns
// backwards, and the product (3α − 3α = 0) stays put; the trail shows the path of each tip.
// "mirror ↕" reflects the shape top–bottom, which conjugates every c_pq (arrows reflect in
// the real axis). Moments of the anti-aliased mask (4 × 4 samples per pixel), so the arrows
// follow the rotation closely. Options: data-preset (blob, mirror, tri, rect), data-alpha.
import { autoMount, htmlEl, button, slider } from "../lib/ui";
import { shapeCanvas, rotatePoly, mirrorPoly, polygonCentroid, Poly, Pt } from "../lib/shapecanvas";
import { svgEl } from "../lib/ui";
import { blob, trefoil, PAIRS, phaseMoments, SIZE } from "./compute";

const NAME = "momentphase";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; gap: 1em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .mp-left { flex: 0 1 360px; min-width: 240px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .mp-dials { flex: 1 1 560px; display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: .4em; }
.qb-demo[data-demo="${NAME}"] .mp-dial { text-align: center; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .mp-dial b { font-size: 18px; }
.qb-demo[data-demo="${NAME}"] .mp-num { font-family: ui-monospace, monospace; font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const left = htmlEl("div", "mp-left", "", host);
  const dialsBox = htmlEl("div", "mp-dials", "", host);

  let base: Poly[] = [blob(32, 32, 14)], alpha = 0, playing = false, raf = 0;
  const sc = shapeCanvas(left, {
    size: SIZE, controls: false, shapes: base,
    onChange: (polys, _m, final) => { base = polys.map(p => rotatePoly(p, -alpha, centre(polys))); update(final); },
  });
  const centre = (polys: Poly[]): Pt => polygonCentroid(polys[0]);
  const aS = slider(left, "rotate α", -180, 180, 1, 0, v => { stop(); setAlpha(v * Math.PI / 180); }, v => `${v}°`);
  const row = htmlEl("div", "qb-presets", "", left);
  const playBtn = button(row, "▶ play", () => (playing ? stop() : play()));
  button(row, "mirror ↕", () => { base = base.map(p => mirrorPoly(p, "y", centre(base))); draw(); });
  button(row, "clear trails", () => { trails.forEach(t => (t.length = 0)); update(true); });
  htmlEl("div", "mo-lab", "examples", left).style.cssText = "font-size:15px;color:var(--qb-muted)";
  const prow = htmlEl("div", "qb-presets", "", left);

  // ---- dials ----
  const R = 80, trails: [number, number][][] = [];
  const dials = [...PAIRS.map(p => ({ label: p.label, speed: p.speed })), { label: "c₃₀·c₁₂³", speed: "3α − 3α = 0: stays" }].map((d, i) => {
    const box = htmlEl("div", "mp-dial", "", dialsBox);
    const t = htmlEl("div", "", "", box);
    t.innerHTML = `<b>${d.label}</b> <span style="color:var(--qb-muted)">${d.speed}</span>`;
    const svg = svgEl("svg", { viewBox: `${-R - 8} ${-R - 8} ${2 * R + 16} ${2 * R + 16}`, width: "100%" }, box);
    svg.style.maxWidth = "190px";
    const c = svgEl("circle", { cx: 0, cy: 0, r: R }, svg);
    c.style.cssText = "fill:none;stroke:var(--qb-grid,#ddd);stroke-width:1.5";
    for (const [x1, y1, x2, y2] of [[-R, 0, R, 0], [0, -R, 0, R]]) { const l = svgEl("line", { x1, y1, x2, y2 }, svg); l.style.cssText = "stroke:var(--qb-grid,#ccc);stroke-width:1.5"; }
    const re = svgEl("text", { x: R - 2, y: -6, "text-anchor": "end" }, svg); re.textContent = "Re"; re.style.cssText = "font-size:16px;fill:var(--qb-muted)";
    const im = svgEl("text", { x: 6, y: -R + 14 }, svg); im.textContent = "Im"; im.style.cssText = "font-size:16px;fill:var(--qb-muted)";
    const trail = svgEl("polyline", { points: "" }, svg);
    trail.style.cssText = "fill:none;stroke:var(--qb-c4,#ff7f0e);stroke-width:2.5;stroke-opacity:.6";
    const arrow = svgEl("g", {}, svg);
    const num = htmlEl("div", "mp-num", "", box);
    trails.push([]);
    return { svg, trail, arrow, num, i };
  });
  let scaleMax: number[] = [];

  function setAlpha(a: number): void {
    alpha = a;
    aS.set(Math.round(((a * 180 / Math.PI + 540) % 360) - 180));
    draw();
  }
  function draw(): void {
    const c = centre(base);
    sc.setPolys(base.map(p => rotatePoly(p, alpha, c)));   // → onChange(final) → update
    sc.selected = -1;
  }
  function update(_final: boolean): void {
    const vals = phaseMoments(sc.getPolys());
    if (!vals.length) return;
    // sizes relative to c11^{(p+q)/2} (dimensionless); each dial is scaled to its size when the
    // shape was loaded, but never below FLOOR, so a moment that is ≈ 0 shows as ≈ 0
    const c11 = vals[1].re, ORD = [2, 2, 3, 3, 3, 12], FLOOR = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25 ** 4];
    const rel = vals.map((v, i) => Math.hypot(v.re, v.im) / c11 ** (ORD[i] / 2));
    if (scaleMax.length !== vals.length) scaleMax = rel.map((r, i) => Math.max(r, FLOOR[i]));
    vals.forEach((v, i) => {
      const d = dials[i], mag = Math.hypot(v.re, v.im);
      scaleMax[i] = Math.max(scaleMax[i], rel[i]);
      const k = 0.9 * R / scaleMax[i] / c11 ** (ORD[i] / 2);
      const x = v.re * k, y = -v.im * k;
      const tr = trails[i];
      if (!tr.length || Math.hypot(tr[tr.length - 1][0] - x, tr[tr.length - 1][1] - y) > 0.5) tr.push([x, y]);
      if (tr.length > 720) tr.shift();
      d.trail.setAttribute("points", tr.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(" "));
      d.arrow.replaceChildren();
      const L = Math.hypot(x, y);
      if (L > 2) {
        const ux = x / L, uy = y / L, hl = 14, hw = 7;
        const ln = svgEl("line", { x1: 0, y1: 0, x2: x - ux * hl * 0.8, y2: y - uy * hl * 0.8 }, d.arrow);
        ln.style.cssText = "stroke:var(--qb-c1,#1f77b4);stroke-width:4;stroke-linecap:round";
        const hd = svgEl("polygon", { points: `${x},${y} ${x - ux * hl - uy * hw},${y - uy * hl + ux * hw} ${x - ux * hl + uy * hw},${y - uy * hl - ux * hw}` }, d.arrow);
        hd.style.setProperty("fill", "var(--qb-c1,#1f77b4)");
      } else {
        const dot = svgEl("circle", { cx: 0, cy: 0, r: 5 }, d.arrow); dot.style.setProperty("fill", "var(--qb-c1,#1f77b4)");
      }
      const ph = Math.atan2(v.im, v.re) * 180 / Math.PI;
      d.num.textContent = rel[i] < 1e-3 * FLOOR[i] ? "|c| ≈ 0" : `|c| = ${mag.toExponential(2)}, arg = ${ph.toFixed(0)}°`;
    });
  }

  function play(): void {
    playing = true; playBtn.textContent = "❚❚ pause";
    let lastT = performance.now();
    const step = (t: number) => {
      if (!playing) return;
      const dt = Math.min(0.1, (t - lastT) / 1000); lastT = t;
      let a = alpha + dt * 30 * Math.PI / 180;     // 30° per second
      if (a > Math.PI) a -= 2 * Math.PI;
      setAlpha(a);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }
  function stop(): void { playing = false; cancelAnimationFrame(raf); playBtn.textContent = "▶ play"; }

  const presets: { id: string; label: string; shapes: () => Poly[]; alpha: number; mirror?: boolean; play?: boolean }[] = [
    { id: "blob", label: "rotate slowly", shapes: () => [blob(32, 32, 14)], alpha: 0, play: true },
    { id: "mirror", label: "mirror", shapes: () => [blob(32, 32, 14)], alpha: 0, mirror: true },
    { id: "tri", label: "3-fold symmetry: c₂₀ = 0", shapes: () => [trefoil(32, 32, 16)], alpha: 0 },
    { id: "rect", label: "a rectangle: c₃₀ = 0", shapes: () => [[[18, 24], [46, 24], [46, 40], [18, 40]]], alpha: 0.4 },
  ];
  const btns = presets.map(p => button(prow, p.label, () => load(p)));
  const p0 = presets.find(p => p.id === host.dataset.preset) ?? presets[0];
  button(prow, "⟲ reset", () => load(p0, true));
  function load(p: typeof presets[number], noPlay = false): void {
    stop();
    btns.forEach((b, i) => b.classList.toggle("active", presets[i] === p));
    base = p.shapes(); scaleMax = []; trails.forEach(t => (t.length = 0));
    const a0 = p === p0 && host.dataset.alpha ? Number(host.dataset.alpha) * Math.PI / 180 : p.alpha;
    alpha = a0; aS.set(Math.round(a0 * 180 / Math.PI));
    draw();
    if (p.mirror) {
      // record the arrows before and after one reflection
      base = base.map(q => mirrorPoly(q, "y", centre(base))); draw();
    }
    if (p.play && !noPlay && host.dataset.autoplay !== "off") play();
  }
  load(p0, true);
}

autoMount(NAME, mount);
