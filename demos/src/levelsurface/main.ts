// A curve as the zero level of a surface. Left: the surface z = φ(x, y) as a rotatable mesh
// (WebGL2, lib/gl3d.ts) with a translucent horizontal plane at height c and the contours
// φ = c (orange) and φ = 0 (black) drawn on it. Right: the top view, φ in a diverging colour
// map (blue inside, red outside), thin level lines every 0.5, the zero contour (solid black)
// and the contour φ = c (dashed orange), all from marching squares (lib/marching.ts). Click
// the top view: the normal n = ∇φ/|∇φ| there, |∇φ|, and the curvature κ = ∇·n of the level
// line through that point, with its osculating circle.
// Shapes: circle, ellipse, two circles (min of two distances), square, "egg crate"
// sin(πx/2) sin(πy/2). The same zero contour can come from different φ: the signed distance
// d, 3d, or d + d²/(2R) (for the circle this is (x² + y² − R²)/(2R)).
// Auto-mounts on <div data-demo="levelsurface">. Optional: data-preset="circle|split|samecurve|corner|egg",
// data-view="both|top" (top: no 3D view).
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { GLView, OrbitCamera, viewButtons, heightFieldMesh, MeshData } from "../lib/gl3d";
import { diverging, divergingT } from "../lib/colour";
import { marchingSquares, Contour } from "../lib/marching";
import { pointSegDist, sdfBox } from "../lib/levelset";

const NAME = "levelsurface";
const EXT = 3;            // world coordinates x, y in [−3, 3]
const NT = 121;           // top-view grid (spacing 0.05)
const NM = 49;            // mesh grid (spacing 0.125)
const ZS = 1 / 12;        // mesh height per unit of φ (world size 1 = 6 units)
const CLIP = 6;           // φ clipped to ±6 in the 3D view
const CMAX = 3;           // colour map saturates at |φ| = 3

type ShapeId = "circle" | "ellipse" | "two" | "square" | "egg";
type Mode = "sdf" | "x3" | "sq";
const SHAPES: { id: ShapeId; label: string; R: number }[] = [
  { id: "circle", label: "circle", R: 1.5 },
  { id: "ellipse", label: "ellipse", R: 2.2 },
  { id: "two", label: "two circles", R: 1 },
  { id: "square", label: "square", R: 1.5 },
  { id: "egg", label: "egg crate", R: 1 },
];
const MODES: { id: Mode; label: string }[] = [
  { id: "sdf", label: "distance d" },
  { id: "x3", label: "3 × d" },
  { id: "sq", label: "d + d²/2R" },
];

interface Preset { id: string; label: string; shape: ShapeId; mode: Mode; c: number; pt: [number, number] | null }
const PRESETS: Preset[] = [
  { id: "circle", label: "circle", shape: "circle", mode: "sdf", c: 0.5, pt: [1.2, -0.9] },
  { id: "split", label: "one becomes two", shape: "two", mode: "sdf", c: 0.4, pt: null },
  { id: "samecurve", label: "same curve, different φ", shape: "circle", mode: "x3", c: 0, pt: [1.2, -0.9] },
  { id: "corner", label: "corner: κ blows up", shape: "square", mode: "sdf", c: 0.1, pt: [1.6, -1.6] },
  { id: "egg", label: "egg crate", shape: "egg", mode: "sdf", c: 0, pt: null },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .ls-view { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .ls-top { flex: 0 1 340px; min-width: 220px; position: relative; }
[data-demo="${NAME}"] .ls-top canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .ls-cap { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .4em; }
[data-demo="${NAME}"] .ls-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .ls-view { flex-basis: 400px; }
.reveal [data-demo="${NAME}"] .ls-top { flex-basis: 330px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .5em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// ---- the functions ----
const ELL_A = 2.2, ELL_B = 1.2;
const ELLIPSE: number[] = (() => {
  const n = 720, P: number[] = [];
  for (let k = 0; k < n; k++) {
    const t0 = 2 * Math.PI * k / n, t1 = 2 * Math.PI * (k + 1) / n;
    P.push(ELL_A * Math.cos(t0), ELL_B * Math.sin(t0), ELL_A * Math.cos(t1), ELL_B * Math.sin(t1));
  }
  return P;
})();

/** Signed distance (or, for the egg crate, the given function) at world point (x, y); y up. */
function baseFn(shape: ShapeId, x: number, y: number): number {
  switch (shape) {
    case "circle": return Math.hypot(x, y) - 1.5;
    case "two": return Math.min(Math.hypot(x - 1.2, y) - 1, Math.hypot(x + 1.2, y) - 1);
    case "square": return sdfBox(x, y, 0, 0, 1.5, 1.5);
    case "egg": return Math.sin(Math.PI * x / 2) * Math.sin(Math.PI * y / 2);
    case "ellipse": {
      let d = Infinity;
      for (let s = 0; s < ELLIPSE.length; s += 4) d = Math.min(d, pointSegDist(x, y, ELLIPSE[s], ELLIPSE[s + 1], ELLIPSE[s + 2], ELLIPSE[s + 3]));
      return (x / ELL_A) ** 2 + (y / ELL_B) ** 2 < 1 ? -d : d;
    }
  }
}

function phiFn(shape: ShapeId, mode: Mode, x: number, y: number): number {
  const d = baseFn(shape, x, y);
  if (shape === "egg") return d;
  const R = SHAPES.find(s => s.id === shape)!.R;
  return mode === "x3" ? 3 * d : mode === "sq" ? d + d * d / (2 * R) : d;
}

/** Sample on an n × n grid; row 0 = top (y = +3), column 0 = left (x = −3). */
function sample(shape: ShapeId, mode: Mode, n: number): Float64Array {
  const v = new Float64Array(n * n), hs = 2 * EXT / (n - 1);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) v[j * n + i] = phiFn(shape, mode, -EXT + i * hs, EXT - j * hs);
  return v;
}

/** φ, gradient, |∇φ| and κ at a world point by central differences of the function (h = 0.01). */
function geometryAt(shape: ShapeId, mode: Mode, x: number, y: number) {
  const h = 0.01, f = (a: number, b: number) => phiFn(shape, mode, a, b);
  const c = f(x, y), l = f(x - h, y), r = f(x + h, y), d = f(x, y - h), u = f(x, y + h);
  const fx = (r - l) / (2 * h), fy = (u - d) / (2 * h);
  const fxx = (r - 2 * c + l) / (h * h), fyy = (u - 2 * c + d) / (h * h);
  const fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h);
  const g = Math.hypot(fx, fy);
  const k = (fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / (g * g * g);
  return { phi: c, fx, fy, g, k };
}

const f2 = (v: number) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const show3d = host.dataset.view !== "top";
  const st = { shape: p0.shape, mode: p0.mode, c: p0.c, pt: p0.pt as [number, number] | null };

  // ---- layout ----
  let view: GLView | null = null;
  if (show3d) {
    const vbox = htmlEl("div", "ls-view", "", host);
    htmlEl("div", "ls-cap", "surface z = φ(x, y), plane z = c", vbox);
    view = new GLView(vbox, { aspect: 1.1, viewports: [{ x: 0, y: 0, w: 1, h: 1, camera: new OrbitCamera({ distance: 2.1, azimuth: 0.5, elevation: 0.5 }) }], fallback: "The top view on the right shows the same function." });
    viewButtons(vbox, view, 0, ["front", "top", "iso"], true);
  }
  const top = htmlEl("div", "ls-top", "", host);
  const topCap = htmlEl("div", "ls-cap", "", top);
  topCap.style.whiteSpace = "nowrap";
  const cv = htmlEl("canvas", "", "", top);
  const S = 480; cv.width = S; cv.height = S;
  const img = document.createElement("canvas"); img.width = NT; img.height = NT;
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "ls-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ls-lab", "shape", panel);
  const shapeRow = htmlEl("div", "qb-presets", "", panel);
  const shapeBtns = SHAPES.map(s => button(shapeRow, s.label, () => { st.shape = s.id; clearPresets(); rebuild(); }));
  htmlEl("div", "ls-lab", "φ for the same curve", panel);
  const modeRow = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = MODES.map(m => button(modeRow, m.label, () => { st.mode = m.id; clearPresets(); rebuild(); }));
  const cS = slider(panel, "plane c", -1.5, 2.5, 0.05, st.c, v => { st.c = v; clearPresets(); redraw(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Blue: φ < 0 (inside), red: φ > 0. Solid black: φ = 0; dashed orange: φ = c. Click the top view for the normal and curvature there.", panel);

  function clearPresets(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function load(p: Preset): void {
    st.shape = p.shape; st.mode = p.mode; st.c = p.c; st.pt = p.pt ? [...p.pt] as [number, number] : null;
    cS.set(st.c);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    rebuild();
  }

  // ---- 3D objects ----
  const surf = view?.addMesh({ positions: [0, 0, 0], indices: [] }, { shading: "smooth", lighting: "lambert", vertexColors: true });
  const plane = view?.addMesh({ positions: [0, 0, 0], indices: [] }, { color: [0.95, 0.7, 0.2], lighting: "unlit", opacity: 0.3 });
  const zeroL = view?.addLines({ positions: [] }, { width: 4, color: [0.05, 0.05, 0.05] });
  const cL = view?.addLines({ positions: [] }, { width: 4, color: [0.9, 0.45, 0] });

  let tv: Float64Array = new Float64Array(NT * NT), mv: Float64Array = new Float64Array(NM * NM);

  function rebuild(): void {
    tv = sample(st.shape, st.mode, NT);
    mv = sample(st.shape, st.mode, NM);
    if (surf) {
      const clipped = Float64Array.from(mv, v => Math.max(-CLIP, Math.min(CLIP, v)));
      const m = heightFieldMesh(clipped, NM, NM, { scale: 2 * CLIP * ZS, lo: -CLIP, hi: CLIP });
      const P = m.positions as Float32Array, C = new Float32Array(P.length);
      for (let p = 0; p < NM * NM; p++) {
        P[3 * p + 1] -= CLIP * ZS;             // height = φ ZS, zero at y = 0
        const col = diverging(divergingT(mv[p], CMAX));
        C[3 * p] = col[0]; C[3 * p + 1] = col[1]; C[3 * p + 2] = col[2];
      }
      surf.setData({ ...m, colors: C } as MeshData);
    }
    redraw();
  }

  function contour3d(level: number): number[] {
    const C = marchingSquares(mv, NM, NM, level), out: number[] = [];
    const y = Math.max(-CLIP, Math.min(CLIP, level)) * ZS + 0.004, L = NM - 1;
    for (let s = 0; s < C.segments.length; s += 2) {
      for (const id of [C.segments[s], C.segments[s + 1]]) out.push(C.points[2 * id] / L - 0.5, y, C.points[2 * id + 1] / L - 0.5);
    }
    return out;
  }

  function redraw(): void {
    syncButtons();
    if (view && plane && zeroL && cL) {
      const y = Math.max(-CLIP, Math.min(CLIP, st.c)) * ZS;
      plane.setData({ positions: [-0.5, y, -0.5, 0.5, y, -0.5, 0.5, y, 0.5, -0.5, y, 0.5], indices: [0, 2, 1, 0, 3, 2] });
      zeroL.setData({ positions: contour3d(0) });
      cL.setData({ positions: contour3d(st.c) });
    }
    drawTop();
  }

  function syncButtons(): void {
    shapeBtns.forEach((b, i) => b.classList.toggle("active", SHAPES[i].id === st.shape));
    modeBtns.forEach((b, i) => { b.classList.toggle("active", MODES[i].id === st.mode); b.disabled = st.shape === "egg"; });
  }

  // ---- top view ----
  const toPx = (x: number, y: number): [number, number] => [(x + EXT) / (2 * EXT) * S, (EXT - y) / (2 * EXT) * S];
  function strokeContour(ctx: CanvasRenderingContext2D, C: Contour): void {
    const k = S / (NT - 1);
    ctx.beginPath();
    for (let s = 0; s < C.segments.length; s += 2) {
      const a = C.segments[s], b = C.segments[s + 1];
      ctx.moveTo(C.points[2 * a] * k, C.points[2 * a + 1] * k);
      ctx.lineTo(C.points[2 * b] * k, C.points[2 * b + 1] * k);
    }
    ctx.stroke();
  }

  function drawTop(): void {
    const ictx = img.getContext("2d")!, id = ictx.createImageData(NT, NT);
    for (let p = 0; p < NT * NT; p++) {
      const c = diverging(divergingT(tv[p], CMAX));
      id.data[4 * p] = 255 * c[0]; id.data[4 * p + 1] = 255 * c[1]; id.data[4 * p + 2] = 255 * c[2]; id.data[4 * p + 3] = 255;
    }
    ictx.putImageData(id, 0, 0);
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(img, 0, 0, S, S);
    // level lines every 0.5
    ctx.strokeStyle = "rgba(60,60,60,0.45)"; ctx.lineWidth = 1; ctx.setLineDash([]);
    for (let lv = -6; lv <= 12; lv += 0.5) if (lv !== 0) strokeContour(ctx, marchingSquares(tv, NT, NT, lv));
    ctx.strokeStyle = "#111"; ctx.lineWidth = 4;
    strokeContour(ctx, marchingSquares(tv, NT, NT, 0));
    ctx.strokeStyle = "#e07000"; ctx.lineWidth = 4; ctx.setLineDash([12, 8]);
    strokeContour(ctx, marchingSquares(tv, NT, NT, st.c));
    ctx.setLineDash([]);
    topCap.textContent = `top view: φ = 0 black, φ = ${st.c.toFixed(2)} orange`;
    const C0 = marchingSquares(tv, NT, NT, st.c);
    const pieces = C0.lines.length;
    const lines = [`shape: ${SHAPES.find(s => s.id === st.shape)!.label}`, `contour φ = c: ${pieces} piece${pieces === 1 ? "" : "s"}`];
    if (st.pt) {
      const [x, y] = st.pt, G = geometryAt(st.shape, st.mode, x, y);
      const [px, py] = toPx(x, y), nx = G.fx / G.g, ny = G.fy / G.g;
      // osculating circle of the level line through the point
      if (Math.abs(G.k) > 1e-3 && Math.abs(1 / G.k) < 20) {
        const r = 1 / G.k, cx = x - r * nx, cy = y - r * ny, [qx, qy] = toPx(cx, cy);
        ctx.strokeStyle = "rgba(0,110,0,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.arc(qx, qy, Math.abs(r) / (2 * EXT) * S, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      }
      const L = 0.7, [ex, ey] = toPx(x + L * nx, y + L * ny);
      ctx.strokeStyle = "#111"; ctx.fillStyle = "#111"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ex, ey); ctx.stroke();
      const ang = Math.atan2(ey - py, ex - px);
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 14 * Math.cos(ang - 0.4), ey - 14 * Math.sin(ang - 0.4));
      ctx.lineTo(ex - 14 * Math.cos(ang + 0.4), ey - 14 * Math.sin(ang + 0.4)); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      lines.push(
        `point (${x.toFixed(2)}, ${y.toFixed(2)})`,
        `φ       = ${f2(G.phi)}`,
        `∇φ      = (${f2(G.fx)}, ${f2(G.fy)})`,
        `|∇φ|    = ${f2(G.g)}`,
        `n       = (${f2(nx)}, ${f2(ny)})`,
        `κ       = ${Math.abs(G.k) > 999 ? "> 999" : f2(G.k)}`,
        `1/κ     = ${Math.abs(G.k) < 1e-3 ? "∞" : f2(1 / G.k)}`,
      );
    } else lines.push("click the top view for n and κ");
    ro.textContent = lines.join("\n");
  }

  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const r = cv.getBoundingClientRect();
    const x = -EXT + (ev.clientX - r.left) / r.width * 2 * EXT, y = EXT - (ev.clientY - r.top) / r.height * 2 * EXT;
    st.pt = [Math.round(x * 20) / 20, Math.round(y * 20) / 20];
    drawTop();
  });

  load(p0);
}

autoMount(NAME, mount);
