// Moving curves with the level set equation φ_t + F|∇φ| = 0 on a 128 × 128 grid (lib/levelset.ts).
// Speed F = c − ε κ: a constant part c (outward > 0) and a curvature part. Forward Euler in
// time, upwind (Godunov) or central differences in space; Δt as a multiple of the stability
// limit; reinitialisation to a signed distance every k steps. Draw closed curves with the
// mouse (they are added to the current shape) or pick a shape. Views: φ as a colour map with
// level lines every 4 pixels and the zero contour, or φ as a surface (WebGL2). Optional: the
// same motion done explicitly on a polygon (vertices moved along their normals, no surgery).
// Auto-mounts on <div data-demo="curveflow">. Optional: data-preset=<preset id>, data-view="map|surface",
// data-steps=<n> (run n steps of the preset at load: a snapshot for a figure or a check).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { diverging, divergingT } from "../lib/colour";
import { GLView, OrbitCamera } from "../lib/gl3d";
import * as LS from "../lib/levelset";
import { Contour } from "../lib/marching";
import { Animator } from "./anim";
import { ShapeId, shapePhi, contourPolys } from "./shapes";
import { PhiSurface } from "../levelsurface/surface";

const NAME = "curveflow";
const N = 128;
const CMAX = 20;
const S = 512;

const SHAPES: { id: ShapeId; label: string }[] = [
  { id: "circle", label: "circle" }, { id: "two", label: "two blobs" }, { id: "dumbbell", label: "dumbbell" },
  { id: "star", label: "star" }, { id: "square", label: "square" },
];

interface Params { F: number; eps: number; factor: number; scheme: LS.Scheme; reinit: number; spf: number; explicit: boolean }
interface Preset extends Params { id: string; label: string; shape: ShapeId; maxT?: number }
const BASE: Params = { F: 1, eps: 0, factor: 0.5, scheme: "upwind", reinit: 0, spf: 4, explicit: false };
const PRESETS: Preset[] = [
  { ...BASE, id: "merge", label: "two blobs merge", shape: "two", F: 1, explicit: true },
  { ...BASE, id: "split", label: "dumbbell splits", shape: "dumbbell", F: -1, explicit: true },
  { ...BASE, id: "nosplit", label: "dumbbell, curvature flow", shape: "dumbbell", F: 0, eps: 1, factor: 0.9, spf: 15, reinit: 25 },
  { ...BASE, id: "star", label: "star: corners round off", shape: "star", F: 0, eps: 1, factor: 0.9, spf: 10, explicit: true },
  { ...BASE, id: "swallowtail", label: "swallowtail", shape: "star", F: 1, explicit: true, maxT: 14 },
  { ...BASE, id: "cfl", label: "Δt too large", shape: "circle", F: 1, factor: 1.5, spf: 1 },
  { ...BASE, id: "central", label: "central differences", shape: "circle", F: -1, scheme: "central", spf: 2 },
  { ...BASE, id: "noreinit", label: "no reinitialisation", shape: "star", F: 0, eps: 1, factor: 0.9, spf: 10, maxT: 100 },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .cf-main { flex: 0 1 470px; min-width: 260px; position: relative; }
[data-demo="${NAME}"] .cf-main canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .cf-cap { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .cf-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .cf-sliders { display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .cf-row { display: flex; gap: .35em; flex-wrap: wrap; align-items: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .cf-main { flex-basis: 440px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; max-width: 560px; flex-basis: 520px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .5em; }
.reveal [data-demo="${NAME}"] .cf-sliders { display: grid; grid-template-columns: 1fr 1fr; column-gap: .8em; row-gap: .1em; }
.reveal [data-demo="${NAME}"] .cf-sliders .qb-slider { grid-template-columns: auto 1fr 2.6em; gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { ...p0 } as Preset;
  let viewMode: "map" | "surface" = host.dataset.view === "surface" ? "surface" : "map";

  // ---- layout ----
  const main = htmlEl("div", "cf-main", "", host);
  const cap = htmlEl("div", "cf-cap", "", main);
  const cv = htmlEl("canvas", "", "", main);
  cv.width = S; cv.height = S;
  const img = document.createElement("canvas"); img.width = N; img.height = N;
  const glBox = htmlEl("div", "", "", main);
  const view = new GLView(glBox, { aspect: 1, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: new OrbitCamera({ distance: 1.9, azimuth: 0.5, elevation: 0.6 }) }], fallback: "Use the map view." });
  const surface = new PhiSurface(view, CMAX, 1 / 80, CMAX);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cf-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "cf-lab", "start shape (or draw closed curves on the image)", panel);
  const shapeRow = htmlEl("div", "cf-row", "", panel);
  const shapeBtns = SHAPES.map(s => button(shapeRow, s.label, () => { st.shape = s.id; init(shapePhi(s.id)); }));
  button(shapeRow, "clear", () => { init(new Float64Array(N * N).fill(CMAX)); });
  const sl = htmlEl("div", "cf-sliders", "", panel);
  const FS = slider(sl, "speed c", -1, 1, 0.05, st.F, v => { st.F = v; changed(); });
  const ES = slider(sl, "curvature ε", 0, 2, 0.05, st.eps, v => { st.eps = v; changed(); });
  const DS = slider(sl, "Δt / limit", 0.2, 1.5, 0.05, st.factor, v => { st.factor = v; changed(); });
  const RS = slider(sl, "reinit every", 0, 50, 5, st.reinit, v => { st.reinit = v; changed(); }, v => v === 0 ? "never" : `${v}`);
  const SS = slider(sl, "steps/frame", 1, 20, 1, st.spf, v => { st.spf = v; });
  const optRow = htmlEl("div", "cf-row", "", panel);
  const schemeBtns = (["upwind", "central"] as LS.Scheme[]).map(s => button(optRow, s, () => { st.scheme = s; changed(); }));
  const viewBtns = (["map", "surface"] as const).map(v => button(optRow, v, () => { viewMode = v; layoutView(); draw(); }));
  const exChk = checkbox(optRow, "explicit polygon", st.explicit, v => { st.explicit = v; draw(); });
  const ctlRow = htmlEl("div", "cf-row", "", panel);
  const playBtn = button(ctlRow, "play", () => anim.toggle());
  button(ctlRow, "step", () => { anim.pause(); advance(1); draw(); });
  button(ctlRow, "reset", () => { anim.pause(); restart(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Blue: φ < 0 (inside), red: φ > 0. Black: the zero contour; grey: level lines every 4 pixels. Green: the explicit polygon moved with the same speed.", panel);

  const anim = new Animator(host, () => frame(), p => { playBtn.textContent = p ? "pause" : "play"; });

  // ---- state ----
  let phi: Float64Array = new Float64Array(N * N), phiInit: Float64Array = new Float64Array(N * N);
  let polys: LS.Poly[] = [], polysInit: LS.Poly[] = [];
  let steps = 0, t = 0, work = new Float64Array(N * N);

  function terms(): LS.SpeedTerms { return { F: st.F, eps: st.eps }; }
  function dtNow(): number { const lim = LS.cflLimit(terms()); return Number.isFinite(lim) ? st.factor * lim : 0; }

  function init(p: Float64Array): void {
    phiInit = p; polysInit = contourPolys(LS.zeroContour(p, N, N));
    restart();
  }
  function restart(): void {
    phi = Float64Array.from(phiInit); polys = polysInit.map(P => P.map(q => [q[0], q[1]] as [number, number]));
    steps = 0; t = 0;
    sync(); draw();
  }
  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); }

  function load(p: Preset): void {
    anim.pause();
    Object.assign(st, p);
    FS.set(st.F); ES.set(st.eps); DS.set(st.factor); RS.set(st.reinit); SS.set(st.spf); exChk.checked = st.explicit;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    init(shapePhi(p.shape));
  }

  function sync(): void {
    shapeBtns.forEach((b, i) => b.classList.toggle("active", SHAPES[i].id === st.shape));
    schemeBtns.forEach((b, i) => b.classList.toggle("active", ["upwind", "central"][i] === st.scheme));
    viewBtns.forEach((b, i) => b.classList.toggle("active", ["map", "surface"][i] === viewMode));
  }

  function advance(k: number): boolean {
    const dt = dtNow();
    if (dt <= 0) return false;
    for (let i = 0; i < k; i++) {
      if (st.maxT !== undefined && t >= st.maxT - 1e-9) return false;
      LS.lsStep(phi, N, N, terms(), dt, st.scheme, work);
      if (st.explicit) polys = polys.map(P => LS.explicitStep(P, st.F, st.eps, dt));
      steps++; t += dt;
      if (st.reinit > 0 && steps % st.reinit === 0) LS.reinitExact(phi, N, N);
      let bad = false;
      for (let p = 0; p < N * N; p += 97) if (!Number.isFinite(phi[p]) || Math.abs(phi[p]) > 1e6) { bad = true; break; }
      if (bad) return false;
    }
    return true;
  }

  function frame(): boolean {
    const ok = advance(st.spf);
    draw();
    // stop when nothing is inside any more (shrinking) or the grid is full (growing)
    const n = LS.insideCount(phi);
    if (steps > 0 && (n === 0 || n === N * N)) return false;
    return ok;
  }

  // ---- drawing curves with the mouse ----
  let drawing: LS.Poly | null = null;
  const toGrid = (ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * N - 0.5, (ev.clientY - r.top) / r.height * N - 0.5];
  };
  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault(); anim.pause();
    cv.setPointerCapture(ev.pointerId);
    drawing = [toGrid(ev)];
  });
  cv.addEventListener("pointermove", ev => {
    if (!drawing) return;
    const q = toGrid(ev), l = drawing[drawing.length - 1];
    if (Math.hypot(q[0] - l[0], q[1] - l[1]) >= 1) { drawing.push(q); draw(); }
  });
  const up = () => {
    if (!drawing) return;
    const P = drawing; drawing = null;
    if (P.length >= 8) {
      const d = LS.polygonSDF([P], N, N);
      presetBtns.forEach(b => b.classList.remove("active"));
      init(LS.union(phi, d));
    } else draw();
  };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);

  // ---- rendering ----
  function layoutView(): void {
    cv.style.display = viewMode === "map" ? "block" : "none";
    glBox.style.display = viewMode === "surface" ? "block" : "none";
    sync();
  }

  function strokeContour(ctx: CanvasRenderingContext2D, C: Contour): void {
    const k = S / N;
    ctx.beginPath();
    for (let s = 0; s < C.segments.length; s += 2) {
      const a = C.segments[s], b = C.segments[s + 1];
      ctx.moveTo((C.points[2 * a] + 0.5) * k, (C.points[2 * a + 1] + 0.5) * k);
      ctx.lineTo((C.points[2 * b] + 0.5) * k, (C.points[2 * b + 1] + 0.5) * k);
    }
    ctx.stroke();
  }

  function draw(): void {
    const finite = phi.every(Number.isFinite);
    if (viewMode === "map") {
      const ictx = img.getContext("2d")!, id = ictx.createImageData(N, N);
      for (let p = 0; p < N * N; p++) {
        const c = Number.isFinite(phi[p]) ? diverging(divergingT(phi[p], CMAX)) : [0, 0, 0];
        id.data[4 * p] = 255 * c[0]; id.data[4 * p + 1] = 255 * c[1]; id.data[4 * p + 2] = 255 * c[2]; id.data[4 * p + 3] = 255;
      }
      ictx.putImageData(id, 0, 0);
      const ctx = cv.getContext("2d")!;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, S, S);
      if (finite) {
        ctx.strokeStyle = "rgba(50,50,50,0.5)"; ctx.lineWidth = 1;
        for (let lv = -40; lv <= 80; lv += 4) if (lv !== 0) strokeContour(ctx, LS.zeroContour(phi, N, N, lv));
        ctx.strokeStyle = "#111"; ctx.lineWidth = 4;
        strokeContour(ctx, LS.zeroContour(phi, N, N));
      }
      if (st.explicit) {
        ctx.strokeStyle = "#009e73"; ctx.lineWidth = 2.5;
        for (const P of polys) {
          ctx.beginPath();
          P.forEach((q, i) => { const x = (q[0] + 0.5) * S / N, y = (q[1] + 0.5) * S / N; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.closePath(); ctx.stroke();
        }
      }
      if (drawing) {
        ctx.strokeStyle = "#7b2cbf"; ctx.lineWidth = 3; ctx.beginPath();
        drawing.forEach((q, i) => { const x = (q[0] + 0.5) * S / N, y = (q[1] + 0.5) * S / N; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.stroke();
      }
    } else if (finite) surface.set(phi, N, N, 2);

    cap.textContent = viewMode === "map" ? "φ (blue inside) and the zero contour: draw curves here" : "φ as a surface (drag to rotate)";
    const dt = dtNow(), lim = LS.cflLimit(terms());
    const C = finite ? LS.zeroContour(phi, N, N) : null;
    const pieces = C ? C.lines.length : 0;
    const area = finite ? LS.insideArea(phi, N, N) : NaN, len = C ? LS.frontLength(phi, N, N) : NaN;
    const g = finite ? LS.meanGradNearFront(phi, N, N) : NaN;
    let mx = 0; for (let p = 0; p < N * N; p++) mx = Math.max(mx, Math.abs(phi[p]));
    const r = (a: string, b: string) => a.padEnd(14) + b;
    ro.textContent = [
      r("speed", `F = ${st.F.toFixed(2)} − ${st.eps.toFixed(2)} κ`),
      r("Δt", `${dt.toFixed(3)} (limit ${Number.isFinite(lim) ? lim.toFixed(3) : "∞"})`),
      r("step, time", `${steps}, t = ${t.toFixed(1)}`),
      r("pieces", `${pieces}`),
      r("area", `${Number.isFinite(area) ? area.toFixed(0) : "–"} px²`),
      r("length", `${Number.isFinite(len) ? len.toFixed(1) : "–"} px`),
      r("|∇φ| at front", Number.isFinite(g) ? g.toFixed(2) : "–"),
      r("max |φ|", mx > 1e5 || !Number.isFinite(mx) ? "exploded" : mx.toFixed(1)),
    ].join("\n");
  }

  layoutView();
  load(p0);
  const n0 = Number(host.dataset.steps ?? 0);
  if (n0 > 0) { advance(n0); draw(); }
}

autoMount(NAME, mount);
