// Free-form deformation by hand (splines day, unit 4). A coarse grid of control points over
// a 256 × 256 image (checkerboard or micrograph); drag a control point and every pixel moves
// by the tensor-product cubic B-spline of the control-point displacements (degree 1:
// bilinear, for comparison). Shown: the deformed image with the deformed grid, and the
// Jacobian determinant det(I + ∇u) as a colour map (blue: compressed, red: stretched,
// black: folded, det < 0). The readout gives the displacement of the pixels at the dragged
// control point, at the next one and diagonally (4/9, 1/9, 1/36 of the drag for degree 3).
// <div data-demo="ffd" data-preset="one|bend|bulge|fold|translate|linear" data-image="checker|cell"></div>
import { autoMount, htmlEl, presets, checkbox, button, svgEl } from "../lib/ui";
import { Grey, drawGrey } from "../lib/imageops";
import { checkerboard } from "../lib/synth";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { diverging } from "../lib/colour";
import { FFDGrid, ffdGrid, ffdIndex, ffdAt, ffdWarp, ffdJacobianDet } from "../lib/splineimage";

const NAME = "ffd";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .7em; }
.qb-demo[data-demo="${NAME}"] .fd-imgs { flex: 1 1 640px; max-width: 760px; display: flex; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .fd-imgs figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin-bottom: .5em; }
.qb-demo[data-demo="${NAME}"] .fd-stack { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .fd-stack canvas, .qb-demo[data-demo="${NAME}"] .fd-stack svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .fd-stack svg { touch-action: none; overflow: visible; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 360px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .fd-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .fd-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .fd-row { display: flex; flex-wrap: wrap; gap: .2em .9em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .fd-imgs { flex: 0 0 720px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 380px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .04em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .45em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .fd-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const N = 256;
type Disp = [number, number, number, number][];   // k, l, dx, dy
interface State { image: "checker" | "cell"; h: number; degree: 1 | 3; disp: Disp; sel: [number, number]; grid: boolean; arrows: boolean }
const DEFAULT: State = { image: "checker", h: 32, degree: 3, disp: [], sel: [4, 4], grid: true, arrows: false };
function bend(): Disp { const d: Disp = []; for (let k = -1; k <= 9; k++) d.push([k, 4, 0, 14 * Math.sin(Math.PI * k / 8)]); return d; }
function bulge(): Disp {
  const d: Disp = [];
  for (const [k, l] of [[3, 3], [4, 3], [5, 3], [3, 4], [5, 4], [3, 5], [4, 5], [5, 5]]) { const a = Math.atan2(l - 4, k - 4); d.push([k, l, 12 * Math.cos(a), 12 * Math.sin(a)]); }
  return d;
}
function translation(): Disp { const d: Disp = []; for (let k = -1; k <= 9; k++) for (let l = -1; l <= 9; l++) d.push([k, l, 12, 6]); return d; }
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "one", label: "one point, 9 px", s: { disp: [[4, 4, 9, 0]], sel: [4, 4] } },
  { id: "bend", label: "a smooth bend", s: { disp: bend(), sel: [4, 4] } },
  { id: "bulge", label: "a local bulge", s: { disp: bulge(), sel: [5, 4] } },
  { id: "fold", label: "too far: the grid folds", s: { disp: [[4, 4, 44, 0], [5, 4, -44, 0]], sel: [4, 4] } },
  { id: "translate", label: "all the same vector: a translation", s: { disp: translation(), sel: [4, 4] } },
  { id: "linear", label: "degree 1 vs 3", s: { disp: [[4, 4, 20, 12]], degree: 1, sel: [4, 4] } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, ...p0.s, image: host.dataset.image === "cell" ? "cell" : "checker" };
  let cell: Grey | null = null, g: FFDGrid = ffdGrid(N, N, st.h, st.degree);

  const imgs = htmlEl("div", "fd-imgs", "", host);
  const f1 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "deformed image (drag the grid points)", f1);
  const s1 = htmlEl("div", "fd-stack", "", f1);
  const cvW = htmlEl("canvas", "", "", s1);
  const svg = svgEl("svg", { viewBox: `-0.5 -0.5 ${N} ${N}` }, s1);
  const f2 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "Jacobian determinant (black: fold)", f2);
  const s2 = htmlEl("div", "fd-stack", "", f2);
  const cvJ = htmlEl("canvas", "", "", s2);
  const credit = htmlEl("div", "fd-credit", CELL_CREDIT, f1);
  cvW.width = cvW.height = cvJ.width = cvJ.height = N;

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fd-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "fd-lab", "grid spacing · degree · image", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  const hBtns = [16, 32, 64].map(h => button(row, h + " px", () => { st.h = h; st.disp = []; rebuild(); }));
  const dBtns = ([3, 1] as const).map(d => button(row, d === 3 ? "cubic" : "linear", () => { st.degree = d; rebuild(); }));
  const row2 = htmlEl("div", "qb-presets", "", panel);
  const iBtns = (["checker", "cell"] as const).map(i => button(row2, i === "checker" ? "checkerboard" : "micrograph", () => { st.image = i; update(); }));
  const row3 = htmlEl("div", "fd-row", "", panel);
  const cks = [
    checkbox(row3, "grid", st.grid, v => { st.grid = v; overlay(); }),
    checkbox(row3, "arrows", st.arrows, v => { st.arrows = v; overlay(); }),
  ];
  const row4 = htmlEl("div", "qb-presets", "", panel);
  button(row4, "move all by (12, 6)", () => { st.disp = translation(); rebuild(); });
  button(row4, "reset", () => load(p0.s));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Handles sit at control point + its displacement; the grid lines show where the pixels actually go.", panel);

  function load(s: Partial<State>): void {
    const img = st.image;
    Object.assign(st, DEFAULT, s, { image: img });
    st.disp = (s.disp ?? []).map(d => [...d] as [number, number, number, number]);
    cks[0].checked = st.grid; cks[1].checked = st.arrows;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    rebuild();
  }
  function rebuild(): void {
    g = ffdGrid(N, N, st.h, st.degree);
    for (const [k, l, dx, dy] of st.disp) if (k >= -1 && k <= g.nx && l >= -1 && l <= g.ny) { const i = ffdIndex(g, k, l); g.dx[i] = dx; g.dy[i] = dy; }
    st.sel = [Math.min(st.sel[0], g.nx - 1), Math.min(st.sel[1], g.ny - 1)];
    update();
  }
  function storeDisp(): void {
    st.disp = [];
    for (let l = -1; l <= g.ny; l++) for (let k = -1; k <= g.nx; k++) { const i = ffdIndex(g, k, l); if (g.dx[i] || g.dy[i]) st.disp.push([k, l, g.dx[i], g.dy[i]]); }
  }

  let base: Grey = checkerboard(N, N, 16, 40, 215);
  function update(): void {
    if (st.image === "cell" && !cell) { loadCell(N).then(c => { cell = c; update(); }); }
    base = st.image === "cell" && cell ? cell : checkerboard(N, N, 16, 40, 215);
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    hBtns.forEach((b, i) => b.classList.toggle("active", [16, 32, 64][i] === st.h));
    dBtns.forEach((b, i) => b.classList.toggle("active", [3, 1][i] === st.degree));
    iBtns.forEach((b, i) => b.classList.toggle("active", ["checker", "cell"][i] === st.image));
    compute();
    overlay();
  }

  function compute(): void {
    drawGrey(cvW, ffdWarp(base, g, "bilinear"), { mode: "clip", lo: 0, hi: 255 });
    const ctx = cvJ.getContext("2d")!, id = ctx.createImageData(N, N);
    let minJ = Infinity, folds = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const d = ffdJacobianDet(g, x, y), p = 4 * (y * N + x);
      minJ = Math.min(minJ, d);
      if (d <= 0) { folds++; id.data[p] = id.data[p + 1] = id.data[p + 2] = 0; }
      else { const c = diverging(0.5 + 0.5 * Math.max(-1, Math.min(1, 3 * Math.log2(d)))); id.data[p] = 255 * c[0]; id.data[p + 1] = 255 * c[1]; id.data[p + 2] = 255 * c[2]; }
      id.data[p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const [k, l] = st.sel, i = ffdIndex(g, k, l), h = g.h;
    const mag = (x: number, y: number) => { const [u, v] = ffdAt(g, x, y); return Math.hypot(u, v); };
    ro.textContent = `control point (${k}, ${l}) at (${k * h}, ${l * h})\n  displacement ${Math.hypot(g.dx[i], g.dy[i]).toFixed(2)} px\n` +
      `pixel moves  at the point  ${mag(k * h, l * h).toFixed(2)} px\n` +
      `             next point    ${mag((k + 1) * h, l * h).toFixed(2)} px\n` +
      `             diagonal      ${mag((k + 1) * h, (l + 1) * h).toFixed(2)} px\n` +
      `min det J ${minJ.toFixed(3)}   folded pixels ${folds}`;
  }

  function overlay(): void {
    svg.replaceChildren();
    const h = g.h;
    if (st.grid) {
      // deformed grid lines (every h/2): forward map T(x) = x + u(x)
      const step = h / 2;
      for (let c = 0; c <= N; c += step) {
        let dv = "", dh = "";
        for (let s = 0; s <= N; s += 4) {
          const [u1, v1] = ffdAt(g, c, s), [u2, v2] = ffdAt(g, s, c);
          dv += (s ? "L" : "M") + (c + u1).toFixed(1) + "," + (s + v1).toFixed(1);
          dh += (s ? "L" : "M") + (s + u2).toFixed(1) + "," + (c + v2).toFixed(1);
        }
        for (const d of [dv, dh]) { const p = svgEl("path", { d, fill: "none" }, svg); p.style.cssText = `stroke:#ffcc00;stroke-width:${c % h === 0 ? 1.4 : 0.6};opacity:.9`; }
      }
    }
    if (st.arrows) for (let y = 8; y < N; y += 16) for (let x = 8; x < N; x += 16) {
      const [u, v] = ffdAt(g, x, y);
      if (Math.hypot(u, v) < 0.3) continue;
      const p = svgEl("path", { d: `M${x},${y}L${x + u},${y + v}` }, svg); p.style.cssText = "stroke:#00e5ff;stroke-width:1.5";
      const c = svgEl("circle", { cx: x + u, cy: y + v, r: 1.6 }, svg); c.style.cssText = "fill:#00e5ff";
    }
    // handles at control point + displacement (control points inside the image)
    for (let l = 0; l < g.ny; l++) for (let k = 0; k < g.nx; k++) {
      const i = ffdIndex(g, k, l), x = k * h + g.dx[i], y = l * h + g.dy[i];
      if (k * h > N - 1 || l * h > N - 1) continue;
      const sel = k === st.sel[0] && l === st.sel[1];
      if (g.dx[i] || g.dy[i]) { const ln = svgEl("path", { d: `M${k * h},${l * h}L${x},${y}` }, svg); ln.style.cssText = "stroke:#ff3b30;stroke-width:1.2;stroke-dasharray:2 2"; }
      const c = svgEl("circle", { cx: x, cy: y, r: h >= 32 ? 5.5 : 4 }, svg);
      c.style.cssText = `fill:${sel ? "#ff3b30" : "#ffffff"};stroke:#ff3b30;stroke-width:1.6;cursor:grab;touch-action:none`;
      c.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        c.setPointerCapture(ev.pointerId);
        st.sel = [k, l];
        let pending = false;
        const toSvg = (e: PointerEvent) => { const r = svg.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * N - 0.5, (e.clientY - r.top) / r.height * N - 0.5]; };
        const move = (e: PointerEvent) => {
          const [px, py] = toSvg(e);
          g.dx[i] = Math.round(px - k * h); g.dy[i] = Math.round(py - l * h);
          c.setAttribute("cx", String(k * h + g.dx[i])); c.setAttribute("cy", String(l * h + g.dy[i]));
          pbtns.forEach(b => b.classList.remove("active"));
          if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; compute(); }); }
        };
        const up = () => { c.removeEventListener("pointermove", move); c.removeEventListener("pointerup", up); storeDisp(); compute(); overlay(); };
        c.addEventListener("pointermove", move);
        c.addEventListener("pointerup", up);
        compute();
      });
    }
  }

  load(p0.s);
}

autoMount(NAME, mount);
