// raycomposite: one ray through a slice of the synthetic head, composited sample by sample
// (3D-graphics day, unit `volrender/`). Drag the ray's ends (E = eye side). Samples every Δs mm
// (trilinear in the volume). Plotted along the ray: the value, the opacity α_i of each sample
// (from the transfer function, corrected for the step), the transmittance T and the
// accumulated colour C. "Next sample" steps through the compositing; front-to-back or
// back-to-front (same final colour, different partial sums); MIP (a running maximum instead);
// early termination marks where the ray could stop. The same transfer-function editor as volview.
// Auto-mounts on <div data-demo="raycomposite">. Optional: data-preset="wall|fog|layers|mip".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { headPhantom, transferEditor, tfAlpha, tfColour, correctAlpha, frontToBack, backToFront } from "../lib/volrender";
import { State, DEF, PRESETS, at, raySamples } from "./ray";
import { Plot } from "../lib/plot";
import type { RGB } from "../lib/colour";

const NAME = "raycomposite";

const STYLE = `
.qb-demo[data-demo="${NAME}"] .rc-c1 { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rc-c2 { flex: 1 1 400px; min-width: 280px; max-width: 470px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] canvas.rc-slice { width: 100%; aspect-ratio: 1; display: block; touch-action: none; border-radius: 3px; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .rc-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .rc-sw { display: inline-block; width: 1.6em; height: 1.1em; border: 1px solid #555; vertical-align: middle; border-radius: 3px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 17px; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .rc-c1 { flex: 0 0 290px; }
.reveal .qb-demo[data-demo="${NAME}"] .rc-c2 { flex: 0 0 430px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .02em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const css = (c: RGB) => `rgb(${c.map(x => Math.round(255 * Math.max(0, Math.min(1, x)))).join(",")})`;
const f3 = (v: number) => v.toFixed(3);


function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init = (): State => ({ ...DEF, ...(p0?.s ?? {}), a: [...DEF.a] as [number, number], b: [...DEF.b] as [number, number] });
  let st = init();
  const vol = headPhantom();
  const ZMAX = vol.nz * vol.spacing[2], YMAX = vol.ny;
  let k = Infinity;   // samples composited so far (Infinity = all)

  // ---- column 1: the slice ----
  const c1 = htmlEl("div", "rc-c1", "", host);
  const cv = htmlEl("canvas", "rc-slice", "", c1);
  const img = document.createElement("canvas"); img.width = 128; img.height = 128;
  {
    const d = new Uint8ClampedArray(4 * 128 * 128);
    for (let j = 0; j < 128; j++) for (let i = 0; i < 128; i++) {
      const v = at(vol, (i + 0.5) / 128 * ZMAX, (j + 0.5) / 128 * YMAX), q = 4 * (j * 128 + i);
      d[q] = d[q + 1] = d[q + 2] = v; d[q + 3] = 255;
    }
    img.getContext("2d")!.putImageData(new ImageData(d, 128, 128), 0, 0);
  }
  const srow = htmlEl("div", "rc-row", "", c1);
  button(srow, "next sample", () => { if (k >= samples().v.length) k = 0; k++; draw(); });
  button(srow, "all", () => { k = Infinity; draw(); });
  button(srow, "start", () => { k = 0; draw(); });
  button(srow, "reset", () => { st = init(); k = Infinity; pb.forEach(b => b.classList.remove("active")); sync(); });
  const dsS = slider(c1, "step Δs (mm)", 0.5, 4, 0.5, st.ds, v => { st.ds = v; k = Infinity; draw(); });

  // ---- column 2: plots ----
  const c2 = htmlEl("div", "rc-c2", "", host);
  const pv = new Plot(c2, { width: 440, height: 150, x: [0, 80], y: [0, 255], ylabel: "value", margin: { l: 56, b: 26, t: 8, r: 10 }, font: 16, yticks: [0, 128, 255] });
  const vLine = pv.line([], [], { width: 2.5, color: "var(--qb-fg)" });
  const vDots = pv.scatter([], [], { r: 4 });
  const vCur = pv.vline(0, { color: "#e0a800", width: 2 });
  const pa = new Plot(c2, { width: 440, height: 170, x: [0, 80], y: [0, 1], xlabel: "distance from E (mm)", margin: { l: 56, b: 46, t: 8, r: 10 }, font: 16, yticks: [0, 0.5, 1] });
  const aBars = pa.bars([], [], { color: "var(--qb-c2)", barWidth: 0.6, label: "α_i" });
  const tLine = pa.step([], [], { color: "var(--qb-c1)", width: 3, label: "T" });
  const cLine = pa.line([], [], { color: "var(--qb-c3)", width: 3, label: "C (brightness)" });
  const eLine = pa.vline(-10, { color: "var(--qb-c4)", dash: "5 4", width: 2.5 });
  const aCur = pa.vline(0, { color: "#e0a800", width: 2 });
  pa.legend(undefined, "tr");
  const ro = htmlEl("div", "qb-readout", "", c2);

  // ---- column 3: transfer function and options ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { st = { ...DEF, ...p.s, a: [...DEF.a] as [number, number], b: [...DEF.b] as [number, number] }; k = Infinity; sync(); } })));
  const ed = transferEditor(panel, st.tf, vol.data, tf => { st.tf = tf; draw(); }, { width: 380, height: 150 });
  const orow = htmlEl("div", "rc-row", "", panel);
  const ftb = button(orow, "front to back", () => { st.order = "ftb"; k = Infinity; draw(); });
  const btf = button(orow, "back to front", () => { st.order = "btf"; k = Infinity; draw(); });
  const chk = htmlEl("div", "rc-row", "", panel);
  const mipC = checkbox(chk, "MIP", st.mip, v => { st.mip = v; draw(); });
  const earlyC = checkbox(chk, "early termination", st.early, v => { st.early = v; draw(); });
  const corrC = checkbox(chk, "opacity correction", st.correction, v => { st.correction = v; draw(); });
  const fin = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag the ends of the ray on the slice. The transfer function gives the opacity per 1 mm (one voxel).", panel);

  function sync(): void { dsS.set(st.ds); ed.set(st.tf); mipC.checked = st.mip; earlyC.checked = st.early; corrC.checked = st.correction; draw(); }

  const samples = () => raySamples(vol, st.a, st.b, st.ds);

  function draw(): void {
    ftb.classList.toggle("active", st.order === "ftb"); btf.classList.toggle("active", st.order === "btf");
    const { t, v, pts } = samples(), n = v.length;
    const L = Math.hypot(st.b[0] - st.a[0], st.b[1] - st.a[1]);
    const alphas = v.map(x => { const a = tfAlpha(st.tf, x); return st.correction ? correctAlpha(a, st.ds) : Math.min(1, a * 1); });
    const cols = v.map(x => tfColour(st.tf, x));
    const f2b = frontToBack(alphas, cols), b2f = backToFront(alphas, cols);
    const stopAt = frontToBack(alphas, cols, 0.05).used;
    const shown = Math.min(n, k);
    // slice canvas
    const dpr = window.devicePixelRatio || 1, cw = cv.clientWidth || 300;
    cv.width = Math.round(cw * dpr); cv.height = Math.round(cw * dpr);
    const g = cv.getContext("2d")!;
    g.imageSmoothingEnabled = true;
    g.drawImage(img, 0, 0, cv.width, cv.height);
    const S = (z: number, y: number): [number, number] => [z / ZMAX * cv.width, y / YMAX * cv.height];
    g.strokeStyle = "#ffe14a"; g.lineWidth = 2 * dpr; g.beginPath(); g.moveTo(...S(...st.a)); g.lineTo(...S(...st.b)); g.stroke();
    const order = st.order === "ftb" ? [...Array(n).keys()] : [...Array(n).keys()].reverse();
    order.forEach((i, m) => {
      const [x, y] = S(...pts[i]);
      g.beginPath(); g.arc(x, y, (m === shown - 1 ? 4.5 : 2.6) * dpr, 0, 7);
      g.fillStyle = m < shown ? (m === shown - 1 ? "#ffe14a" : css(cols[i])) : "rgba(255,255,255,0.35)"; g.fill();
    });
    if (st.early && !st.mip && stopAt < n && st.order === "ftb") { const [x, y] = S(...pts[stopAt - 1]); g.strokeStyle = "#ff3df5"; g.lineWidth = 2 * dpr; g.beginPath(); g.arc(x, y, 8 * dpr, 0, 7); g.stroke(); }
    for (const [p, lab] of [[st.a, "E"], [st.b, ""]] as [[number, number], string][]) {
      const [x, y] = S(...p);
      g.beginPath(); g.arc(x, y, 9 * dpr, 0, 7); g.fillStyle = "#fff"; g.fill(); g.strokeStyle = "#000"; g.lineWidth = 2 * dpr; g.stroke();
      if (lab) { g.fillStyle = "#000"; g.font = `bold ${15 * dpr}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(lab, x, y + 0.5); }
    }
    // plots
    const xmax = Math.max(10, Math.ceil(L / 10) * 10);
    pv.setDomain([0, xmax]); pa.setDomain([0, xmax]);
    vLine.set(t, v); vDots.set(t, v, cols.map(css));
    aBars.set(t, alphas);
    const lum = (c: RGB) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    if (st.order === "ftb") {
      const ts = [0, ...t.slice(0, shown)], Ts = [1, ...f2b.steps.slice(0, shown).map(s => s.T)];
      tLine.set(ts.map((x, i) => (i === 0 ? 0 : x + st.ds / 2)), Ts);
      cLine.set([0, ...t.slice(0, shown).map(x => x + st.ds / 2)], [0, ...f2b.steps.slice(0, shown).map(s => lum(s.C))]);
    } else {
      // back to front: partial sums over the far part of the ray, plotted at the sample reached
      const idx = [...Array(shown).keys()].map(m => n - 1 - m);
      tLine.set([], []);
      cLine.set(idx.map(i => t[i] - st.ds / 2), b2f.steps.slice(0, shown).map(s => lum(s.C)));
    }
    eLine.set(st.early && !st.mip && stopAt < n ? t[stopAt - 1] + st.ds / 2 : -10);
    const cur = shown > 0 ? order[shown - 1] : -1;
    vCur.set(cur >= 0 && shown < n ? t[cur] : -10); aCur.set(cur >= 0 && shown < n ? t[cur] : -10);
    // readouts
    const lines: string[] = [];
    if (cur >= 0 && shown < n) {
      const m = shown - 1, i = cur;
      lines.push(`sample ${i + 1} of ${n}: t = ${t[i].toFixed(1)} mm, value ${v[i].toFixed(1)}`);
      lines.push(`α = ${f3(alphas[i])}${st.correction ? ` (table ${f3(tfAlpha(st.tf, v[i]))} per mm)` : ""}, c = (${cols[i].map(f3).join(", ")})`);
      const s = st.order === "ftb" ? f2b.steps[m] : b2f.steps[m];
      lines.push(st.order === "ftb" ? `after it: C = (${s.C.map(f3).join(", ")}), T = ${f3(s.T)}` : `after it: C = (${s.C.map(f3).join(", ")})`);
    } else lines.push(`${n} samples, Δs = ${st.ds} mm along ${L.toFixed(1)} mm`);
    ro.textContent = lines.join("\n");
    const mx = n ? Math.max(...v) : 0;
    fin.innerHTML = "";
    const line = (label: string, c: RGB, txt: string) => { const d = htmlEl("div", "", "", fin); const sw = htmlEl("span", "rc-sw", "", d); sw.style.background = css(c); d.append(` ${label} ${txt}`); };
    line("front to back:", f2b.C, `C = (${f2b.C.map(f3).join(", ")}), T = ${f3(f2b.T)}`);
    line("back to front:", b2f.C, `C = (${b2f.C.map(f3).join(", ")})`);
    if (st.early) htmlEl("div", "", `stop at T < 0.05 after ${stopAt} of ${n} samples`, fin);
    if (st.mip) line("MIP:", [mx / 255, mx / 255, mx / 255], `max = ${mx.toFixed(1)}`);
  }

  // drag the ends
  let drag: "a" | "b" | null = null;
  const toSlice = (ev: PointerEvent): [number, number] => { const r = cv.getBoundingClientRect(); return [Math.max(0, Math.min(ZMAX, (ev.clientX - r.left) / r.width * ZMAX)), Math.max(0, Math.min(YMAX, (ev.clientY - r.top) / r.height * YMAX))]; };
  cv.addEventListener("pointerdown", ev => {
    const p = toSlice(ev), da = Math.hypot(p[0] - st.a[0], p[1] - st.a[1]), db = Math.hypot(p[0] - st.b[0], p[1] - st.b[1]);
    drag = da < db ? "a" : "b";
    cv.setPointerCapture(ev.pointerId); ev.preventDefault();
    st[drag] = p.map(x => Math.round(x)) as [number, number]; k = Infinity; draw();
  });
  cv.addEventListener("pointermove", ev => { if (!drag) return; st[drag] = toSlice(ev).map(x => Math.round(x)) as [number, number]; draw(); });
  const up = () => { drag = null; };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);
  if (p0) pb[PRESETS.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
