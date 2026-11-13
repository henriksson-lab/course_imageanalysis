// Projections and the sinogram. Left: the object (128 x 128) with the rays of the current
// angle and the detector; click to drop point masses, or tick "drag moves the object" and
// drag to shift it. Middle: the projection p_theta(rho) at the current angle. Right: the
// sinogram, filled column by column as theta advances; every point's sinusoid
// rho = x cos(theta) + y sin(theta) is drawn over it in its own colour.
// Auto-mounts on <div data-demo="radonproj">. Options: data-preset="hand|centre|mirror|mass|shepp|real|linear|shift",
// data-object="points|point|disc|square|twodiscs|shepp|cell".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure, seriesColor } from "../lib/plot";
import * as R from "../lib/radon";
import { ObjectName, OBJECT_LABELS, objectImage, cellObject, shiftImage, imagePanel, sinoPanel, injectStyle, baseCss, onPointer, lineSeg, layout, CELL_CREDIT, uniqueClip } from "./view";

const NAME = "radonproj";
const N = 128;

interface Pt { x: number; y: number; m: number }
interface Preset { id: string; label: string; obj: ObjectName; pts: Pt[]; angle: number; full360?: boolean; shiftMode?: boolean; shift?: [number, number] }

// the by-hand example uses 16 pixels per unit: A = (2, 0) mass 1, B = (0, 1) mass 2, C = (-1, -1) mass 1
const HAND: Pt[] = [{ x: 32, y: 0, m: 1 }, { x: 0, y: 16, m: 2 }, { x: -16, y: -16, m: 1 }];
const PRESETS: Preset[] = [
  { id: "hand", label: "three points by hand", obj: "points", pts: HAND, angle: 45 },
  { id: "centre", label: "the centre does not move", obj: "points", pts: [{ x: 0, y: 0, m: 2 }, { x: 40, y: 0, m: 1 }, { x: -20, y: 30, m: 1 }], angle: 120 },
  { id: "mirror", label: "180°–360° mirrors 0°–180°", obj: "twodiscs", pts: [{ x: 36, y: 36, m: 6 }], angle: 359, full360: true },
  { id: "mass", label: "every column has the same sum", obj: "square", pts: [], angle: 30 },
  { id: "shepp", label: "Shepp–Logan sinogram", obj: "shepp", pts: [], angle: 179 },
  { id: "real", label: "a real image", obj: "cell", pts: [], angle: 179 },
  { id: "linear", label: "two discs: sinograms add", obj: "twodiscs", pts: [], angle: 179 },
  { id: "shift", label: "shift the object", obj: "disc", pts: [{ x: 10, y: 6, m: 8 }], angle: 179, shiftMode: true },
];

// Pages: the figures in one row and the controls in a column beside them (slides keep
// baseCss's figures row with the control columns below it).
const PD = `.qb-demo[data-demo="${NAME}"]:not(.reveal *)`;
const STYLE = baseCss(NAME) + `
${PD} { flex-direction: row; flex-wrap: wrap; align-items: flex-start; gap: .8em; }
${PD} .rd-figs { flex: 1 1 640px; width: auto; min-width: 0; gap: .6em; }
${PD} > .qb-panel { flex: 0 1 290px; gap: .3em; font-size: 15px; }
${PD} > .qb-panel .rd-sub { gap: .3em; }
${PD} > .qb-panel .qb-btn { font-size: 13px; padding: .12em .5em; }
${PD} > .qb-panel .qb-presets, ${PD} > .qb-panel .rd-row { gap: .25em; }
${PD} .qb-readout { font-size: 13px; }
${PD} .rd-fig figcaption { font-size: 15px; }
@media (min-width: 900px) {
  ${PD} .rd-figs { flex-wrap: nowrap; }
  ${PD} .rd-fig { min-width: 0; }
}
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const objAttr = host.dataset.object as ObjectName | undefined;
  const start: Preset = objAttr && OBJECT_LABELS[objAttr] && !host.dataset.preset ? { ...p0, obj: objAttr, pts: [], angle: 179 } : p0;
  const st = { obj: start.obj, pts: [] as Pt[], angle: 0, full360: false, shiftMode: false, shift: [0, 0] as [number, number], playing: false, mass: 1 };

  // ---- layout ----
  const onSlide = !!host.closest(".reveal");
  let L: { figs: HTMLElement; cols: HTMLElement[] };
  if (onSlide) L = layout(host, 3);
  else {
    const figs = htmlEl("div", "rd-figs", "", host), side = htmlEl("div", "qb-panel", "", host);
    L = { figs, cols: [0, 1, 2].map(() => htmlEl("div", "rd-sub", "", side)) };
  }
  const imgP = imagePanel(L.figs, "click: add a point · blue bar: detector", N, 14, "rd-fig img");
  const credit = htmlEl("div", "rd-credit", CELL_CREDIT, imgP.fig);
  const profFig = plotFigure(L.figs, "projection", "rd-fig prof");
  const prof = uniqueClip(new Plot(profFig, { font: 22, width: 320, height: 290, x: [-64, 64], y: [0, 1], xlabel: "ρ (pixels)", ylabel: "p(ρ)", xticks: [-64, -32, 0, 32, 64] }));
  const sinoP = sinoPanel(L.figs, "sinogram", N / 2, [0, 180], "rd-fig wide", 300);
  const [colA, panel, colC] = L.cols;

  htmlEl("div", "rd-lab", "presets", colA);
  const presetBtns = presets(colA, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "rd-lab", "object", panel);
  const objNames: ObjectName[] = ["points", "disc", "square", "twodiscs", "shepp", "cell"];
  const objRow = htmlEl("div", "qb-presets", "", panel);
  const objBtns = objNames.map(o => button(objRow, OBJECT_LABELS[o], () => { st.obj = o; st.shift = [0, 0]; rebuild(); }));
  const angS = slider(panel, "θ", 0, 179, 1, 0, v => { stop(); st.angle = v; draw(); }, v => v + "°");
  const ctl = htmlEl("div", "rd-row", "", panel);
  const playBtn = button(ctl, "▶ play", () => (st.playing ? stop() : play()));
  button(ctl, "+1°", () => { stop(); st.angle = Math.min(nAng() - 1, st.angle + 1); draw(); });
  button(ctl, "+45°", () => { stop(); st.angle = Math.min(nAng() - 1, st.angle + 45); draw(); });
  button(ctl, "0°", () => { stop(); st.angle = 0; draw(); });
  const massS = slider(colC, "point mass", 1, 8, 1, 1, v => { st.mass = v; });
  const shiftCb = checkbox(colC, "drag moves the object", false, v => { st.shiftMode = v; imgP.cap.textContent = v ? "drag: shift the object · blue: detector" : "click: add a point · blue bar: detector"; });
  const fullCb = checkbox(colC, "0°–360°", false, v => { st.full360 = v; st.angle = Math.min(st.angle, nAng() - 1); rebuild(); });
  const row2 = htmlEl("div", "rd-row", "", colC);
  button(row2, "clear points", () => { st.pts = []; rebuild(); });
  button(row2, "reset", () => load(start));
  const ro = htmlEl("div", "qb-readout", "", colC);

  const nAng = () => (st.full360 ? 360 : 180);

  // ---- model ----
  let base: Float64Array = new Float64Array(N * N);     // object without points, unshifted
  let img: Float64Array = new Float64Array(N * N);      // what is measured
  let sino: Float64Array = new Float64Array(180 * N);
  let geom = R.geometry(N, R.angleSet(180));
  let total = 0, pmax = 1, imax = 1;

  function rebuild(): void {
    if (st.obj === "cell") { cellObject().then(() => compute()); return; }
    compute();
  }

  function compute(): void {
    base = objectImage(st.obj, N);
    recompute();
  }

  function recompute(): void {
    img = shiftImage(base, N, st.shift[0], st.shift[1]);
    geom = R.geometry(N, R.angleSet(nAng(), nAng()));
    sino = R.project(img, geom);
    total = img.reduce((s, v) => s + v, 0);
    // point masses: exact deltas at rho = x cos + y sin, shared between the two nearest bins
    for (const p of st.pts) {
      const x = p.x + st.shift[0], y = p.y + st.shift[1];
      total += p.m;
      for (let a = 0; a < geom.angles.length; a++) {
        const u = R.binOf(R.pointRho(x, y, geom.angles[a]), N), b0 = Math.floor(u), f = u - b0;
        if (b0 >= 0 && b0 < N) sino[a * N + b0] += p.m * (1 - f);
        if (b0 + 1 >= 0 && b0 + 1 < N) sino[a * N + b0 + 1] += p.m * f;
      }
    }
    pmax = 0;
    for (const v of sino) pmax = Math.max(pmax, v);
    pmax = pmax > 0 ? pmax * 1.08 : 1;
    let bm = 0;
    for (const v of base) bm = Math.max(bm, v);
    imax = bm > 0 ? bm : 1;
    angS.input.max = String(nAng() - 1);
    sync();
    draw();
  }

  function sync(): void {
    objBtns.forEach((b, i) => b.classList.toggle("active", objNames[i] === st.obj));
    shiftCb.checked = st.shiftMode; fullCb.checked = st.full360;
    imgP.cap.textContent = st.shiftMode ? "drag: shift the object · blue: detector" : "click: add a point · blue bar: detector";
    credit.style.display = st.obj === "cell" ? "" : "none";
    massS.set(st.mass);
  }

  // overlays (created once, updated in draw)
  const rays = Array.from({ length: 9 }, () => imgP.plot.line([0, 0], [0, 0], { color: "#f2c200", width: 1.6, opacity: 0.85 }));
  const det = imgP.plot.line([0, 0], [0, 0], { color: "var(--qb-c1)", width: 7 });
  const beam = imgP.plot.arrow(0, 0, 0, 0, { color: "#f2c200", width: 3 });
  let ptMarks: { remove(): void }[] = [];
  const profLine = prof.step([0], [0], { color: "var(--qb-c1)", width: 2.5 }, "mid");
  const curLine = sinoP.plot.vline(0, { color: "var(--qb-c4)", width: 2, dash: "5 4" });

  function draw(): void {
    const na = nAng(), a = Math.min(st.angle, na - 1), th = a * Math.PI / 180;
    angS.set(a);
    // image
    const disp = Float64Array.from(img, v => v);
    imgP.show(disp, 0, imax);
    for (let k = 0; k < rays.length; k++) {
      const [xs, ys] = lineSeg(th, -56 + 14 * k, 70, -70);
      rays[k].set(xs, ys);
    }
    const c = Math.cos(th), s = Math.sin(th);
    // detector across the far end (t = -74); rho axis along (cos, sin)
    det.set([-64 * c + 74 * s, 64 * c + 74 * s], [-64 * s - 74 * c, 64 * s - 74 * c]);
    // beam direction: from the source side (t = +76) towards the detector (t = -74)
    beam.set(-76 * s, 76 * c, -58 * s, 58 * c);
    // point marks
    ptMarks.forEach(m => m.remove());
    ptMarks = [];
    const sinoX: number[] = Array.from({ length: na + 1 }, (_, k) => k);
    st.pts.forEach((p, i) => {
      const x = p.x + st.shift[0], y = p.y + st.shift[1], col = seriesColor(i + 1);
      ptMarks.push(imgP.plot.scatter([x], [y], { color: col, r: 5 + 1.5 * p.m ** 0.5 }));
      const rho = R.pointRho(x, y, th);
      ptMarks.push(imgP.plot.scatter([rho * c + 74 * s], [rho * s - 74 * c], { color: col, r: 6 }));
      ptMarks.push(prof.vline(rho, { color: col, width: 2.5, dash: "4 3" }));
      ptMarks.push(sinoP.plot.line(sinoX, sinoX.map(t => R.pointRho(x, y, t * Math.PI / 180)), { color: col, width: 2.2, opacity: 0.9 }));
    });
    // profile
    const rhos = Array.from({ length: N }, (_, b) => R.rhoOf(b, N));
    const pr = Array.from(sino.subarray(a * N, a * N + N));
    prof.setDomain(undefined, [0, pmax]);
    profLine.set(rhos, pr);
    (profFig.querySelector("figcaption") as HTMLElement).textContent = `projection at θ = ${a}°`;
    // sinogram
    sinoP.plot.setDomain([0, na], undefined);
    sinoP.plot.opts.xticks = na > 180 ? [0, 90, 180, 270, 360] : [0, 45, 90, 135, 180];
    sinoP.plot.setDomain();
    sinoP.show(sino, na, N, 0, na, a + 1, 0, pmax / 1.08);
    curLine.set(a + 0.5);
    // readout
    const sum = pr.reduce((u, v) => u + v, 0);
    let txt = `θ = ${a}°\nprojection sum ${sum.toFixed(1)}\nimage total    ${total.toFixed(1)}`;
    st.pts.forEach((p, i) => {
      const x = p.x + st.shift[0], y = p.y + st.shift[1];
      txt += `\n${"ABCDEFGHIJ"[i] ?? "·"} (${x.toFixed(0)}, ${y.toFixed(0)}) m${p.m}: ρ ${R.pointRho(x, y, th).toFixed(1)}`;
    });
    if (st.shift[0] || st.shift[1]) txt += `\nshift t = (${st.shift[0].toFixed(0)}, ${st.shift[1].toFixed(0)})`;
    ro.textContent = txt;
  }

  // ---- pointer ----
  let drag: [number, number, number, number] | null = null;
  let pendingFrame = false;
  onPointer(imgP.plot, {
    down: (x, y) => {
      presetBtns.forEach(b => b.classList.remove("active"));
      if (st.shiftMode) { drag = [x, y, st.shift[0], st.shift[1]]; return; }
      if (Math.hypot(x, y) > 62) return;
      if (st.pts.length >= 10) return;
      st.pts.push({ x: Math.round(x - st.shift[0]), y: Math.round(y - st.shift[1]), m: st.mass });
      recompute();
    },
    move: (x, y) => {
      if (!drag) return;
      st.shift = [Math.round(drag[2] + x - drag[0]), Math.round(drag[3] + y - drag[1])];
      st.shift = st.shift.map(v => Math.max(-30, Math.min(30, v))) as [number, number];
      if (!pendingFrame) { pendingFrame = true; requestAnimationFrame(() => { pendingFrame = false; recompute(); }); }
    },
    up: () => { drag = null; },
  });

  // ---- play ----
  let timer = 0;
  function play(): void {
    if (st.angle >= nAng() - 1) st.angle = 0;
    st.playing = true; playBtn.textContent = "❚❚ pause";
    const tick = () => {
      if (!st.playing) return;
      st.angle = Math.min(nAng() - 1, st.angle + 2);
      draw();
      if (st.angle >= nAng() - 1) { stop(); return; }
      timer = window.setTimeout(tick, 50);
    };
    tick();
  }
  function stop(): void { st.playing = false; playBtn.textContent = "▶ play"; clearTimeout(timer); }

  function load(p: Preset): void {
    stop();
    st.obj = p.obj; st.pts = p.pts.map(q => ({ ...q })); st.angle = p.angle; st.full360 = !!p.full360;
    st.shiftMode = !!p.shiftMode; st.shift = p.shift ? [...p.shift] as [number, number] : [0, 0];
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    rebuild();
  }

  load(start);
}

autoMount(NAME, mount);
