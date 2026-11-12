// zernikebasis (moments elective, unit zernike/): the Zernike polynomials and why |Z_nm| does
// not change under rotation.
// Tab "basis": V_nm for n ≤ 8 as a pyramid (rows n, columns m), shown as the real Zernike
// functions R_nm cos(mθ) (m ≥ 0) and R_n|m| sin(|m|θ) (m < 0) on a diverging map, or as
// |V_nm| = |R_nm(ρ)|. Click one: large view, radial profile R_nm(ρ) and its optics name.
// Tab "rotation": a test image and a copy rotated by α (optionally mirrored), rasterised
// anti-aliased on 64 × 64; bars of |Z_nm| of both (n ≤ 8, m ≥ 0), and a dial with the phase
// of one Z_nm before and after: it turns by −mα (θ counter-clockwise on screen).
// Options: data-tab (basis, rotation), data-preset (defocus, astig, rotate, six, mirror),
// data-presets (comma-separated ids: show only these preset buttons).
import { autoMount, htmlEl, button, slider, checkbox, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { diverging, viridis, colormapLUT, scalarToRGBA } from "../lib/colour";
import { zernikeImage, zernikeRadial } from "../lib/moments";
import { S, NZ, ShapeId, zernikeName, shapeImage, zernikeOf } from "./compute";

const NAME = "zernikebasis";
const NB = 8, TILE = 40;
const C0 = "var(--qb-c4, #ff7f0e)", C1 = "var(--qb-c1, #1f77b4)";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; gap: .9em; align-items: flex-start; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .zb-ctl { flex: 0 0 230px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .zb-main { flex: 1 1 640px; min-width: 0; display: flex; gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .zb-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.qb-demo[data-demo="${NAME}"] .zb-big { text-align: center; font-size: 16px; }
.qb-demo[data-demo="${NAME}"] canvas.zb-img { image-rendering: pixelated; border: 1px solid var(--qb-grid, #ddd); }
.qb-demo[data-demo="${NAME}"] .zb-pyr { flex: 0 1 470px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .zb-side { flex: 1 1 260px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .zb-pair { display: flex; gap: .6em; justify-content: center; text-align: center; font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const sub = (v: number) => (v < 0 ? "₋" : "") + String(Math.abs(v)).split("").map(c => SUB[+c]).join("");

/** Real Zernike function on a size² grid: R cos(mθ) for m ≥ 0, R sin(|m|θ) for m < 0. */
function realZernike(n: number, m: number, size: number): Float64Array {
  const z = zernikeImage(n, Math.abs(m), size);
  return m >= 0 ? z.re : z.im;
}

function toDataURL(v: Float64Array, size: number, mag: boolean): string {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const id = c.getContext("2d")!.createImageData(size, size);
  const vals = mag ? v.map(Math.abs) : v;
  scalarToRGBA(vals, { map: mag ? colormapLUT(viridis) : colormapLUT(diverging), lo: mag ? 0 : -1, hi: 1, nan: [0, 0, 0, 0] }, id.data);
  c.getContext("2d")!.putImageData(id, 0, 0);
  return c.toDataURL();
}

function greyCanvas(can: HTMLCanvasElement, f: Float64Array): void {
  can.width = can.height = S;
  const c2 = can.getContext("2d")!, id = c2.createImageData(S, S);
  for (let i = 0; i < S * S; i++) { const g = Math.round(255 * (1 - Math.min(1, f[i]))); id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = g; id.data[4 * i + 3] = 255; }
  c2.putImageData(id, 0, 0);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const ctl = htmlEl("div", "zb-ctl", "", host);
  const main = htmlEl("div", "zb-main", "", host);
  const st = { tab: "basis" as "basis" | "rotation", mag: false, n: 2, m: 0, shape: "blob" as ShapeId, alpha: 0, mirror: false, pn: 2, pm: 2 };

  htmlEl("div", "zb-lab", "view", ctl);
  const trow = htmlEl("div", "qb-presets", "", ctl);
  const tb = [button(trow, "basis", () => { st.tab = "basis"; render(); }), button(trow, "rotation", () => { st.tab = "rotation"; render(); })];
  // basis controls
  const bctl = htmlEl("div", "", "", ctl);
  bctl.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  const vrow = htmlEl("div", "qb-presets", "", bctl);
  const vb = [button(vrow, "cos / sin", () => { st.mag = false; render(); }), button(vrow, "|V| = |R(ρ)|", () => { st.mag = true; render(); })];
  htmlEl("div", "qb-note", "Click a polynomial to enlarge it.", bctl).style.fontSize = "14px";
  // rotation controls
  const rctl = htmlEl("div", "", "", ctl);
  rctl.style.cssText = "display:flex;flex-direction:column;gap:.35em";
  htmlEl("div", "zb-lab", "image", rctl);
  const srow = htmlEl("div", "qb-presets", "", rctl);
  const SHAPES: [ShapeId, string][] = [["blob", "blob"], ["six", "6"], ["F", "F"]];
  const sb = SHAPES.map(([id, l]) => button(srow, l, () => { st.shape = id; render(); }));
  const aS = slider(rctl, "rotate α", -180, 180, 5, 0, v => { st.alpha = v; render(); }, v => `${v}°`);
  const mC = checkbox(rctl, "mirror the copy", false, v => { st.mirror = v; render(); });
  htmlEl("div", "zb-lab", "phase of", rctl);
  const prow2 = htmlEl("div", "qb-presets", "", rctl);
  const PH: [number, number][] = [[1, 1], [2, 2], [3, 1], [3, 3], [4, 2], [4, 4]];
  const pb = PH.map(([n, m]) => button(prow2, `Z${sub(n)}${sub(m)}`, () => { st.pn = n; st.pm = m; render(); }));
  htmlEl("div", "zb-lab", "examples", ctl);
  const prow = htmlEl("div", "qb-presets", "", ctl);

  // ---- basis view ----
  const basisBox = htmlEl("div", "", "", main);
  basisBox.style.cssText = "display:flex;gap:.8em;flex-wrap:wrap;align-items:flex-start;width:100%";
  const pyrBox = htmlEl("div", "zb-pyr", "", basisBox);
  const W = (NB + 1) * TILE + 50, H = (NB + 1) * TILE + 24;
  const pyr = svgEl("svg", { viewBox: `-50 -4 ${W} ${H}`, width: "100%" }, pyrBox);
  const side = htmlEl("div", "zb-side", "", basisBox);
  const bigTitle = htmlEl("div", "zb-big", "", side);
  const bigCan = htmlEl("canvas", "zb-img", "", side);
  bigCan.width = bigCan.height = 128; bigCan.style.cssText = "width:170px;height:170px;border:none";
  const profBox = htmlEl("div", "", "", side);
  profBox.style.width = "100%";
  const prof = new Plot(profBox, { width: 380, height: 230, x: [0, 1], y: [-1.1, 1.1], xlabel: "ρ", ylabel: "R(ρ)", title: "radial part", font: 17, margin: { b: 44 } });
  prof.hline(0, { color: "var(--qb-axis, #444)", width: 1 });
  const profLine = prof.line([], [], { color: C1, width: 3 });
  const tiles = new Map<string, SVGImageElement>();
  const frames = new Map<string, SVGRectElement>();
  for (let n = 0; n <= NB; n++) {
    const t = svgEl("text", { x: -8, y: n * TILE + TILE / 2 + 6, "text-anchor": "end" }, pyr); t.textContent = `n=${n}`; t.style.cssText = "font-size:13px;fill:var(--qb-muted)";
    for (let m = -n; m <= n; m += 2) {
      const x = (m + NB) / 2 * TILE, y = n * TILE;
      const im = svgEl("image", { x: x + 1, y: y + 1, width: TILE - 2, height: TILE - 2 }, pyr);
      im.style.cursor = "pointer";
      const fr = svgEl("rect", { x: x + 0.5, y: y + 0.5, width: TILE - 1, height: TILE - 1, rx: 3 }, pyr);
      fr.style.cssText = "fill:none;stroke:transparent;stroke-width:2.5;pointer-events:none";
      im.addEventListener("click", () => { st.n = n; st.m = m; render(); });
      tiles.set(`${n},${m}`, im); frames.set(`${n},${m}`, fr);
    }
  }
  const mlab = svgEl("text", { x: W / 2 - 50, y: H - 6, "text-anchor": "middle" }, pyr);
  mlab.textContent = "m = −8 … 8 (columns)"; mlab.style.cssText = "font-size:13px;fill:var(--qb-muted)";
  let tilesMag: boolean | null = null;

  // ---- rotation view ----
  const rotBox = htmlEl("div", "", "", main);
  rotBox.style.cssText = "display:flex;gap:.8em;flex-wrap:wrap;align-items:flex-start;width:100%";
  const rotLeft = htmlEl("div", "", "", rotBox);
  rotLeft.style.cssText = "flex:0 1 330px;display:flex;flex-direction:column;gap:.4em;align-items:center";
  const pair = htmlEl("div", "zb-pair", "", rotLeft);
  const mk = (title: string, col: string) => { const d = htmlEl("div", "", "", pair); const t = htmlEl("div", "", "", d); t.innerHTML = `<b style="color:${col}">${title}</b>`; const c = htmlEl("canvas", "zb-img", "", d); c.style.cssText = "width:150px;height:150px"; return { c, t }; };
  const imA = mk("original", C0), imB = mk("rotated copy", C1);
  const dialBox = htmlEl("div", "", "", rotLeft);
  dialBox.style.cssText = "text-align:center;font-size:15px";
  const R = 70;
  const dial = svgEl("svg", { viewBox: `${-R - 6} ${-R - 6} ${2 * R + 12} ${2 * R + 12}`, width: "150" }, dialBox);
  const dialText = htmlEl("div", "", "", dialBox);
  const barBox = htmlEl("div", "", "", rotBox);
  barBox.style.cssText = "flex:1 1 420px;min-width:300px";
  const IDX: [number, number][] = [];
  for (let n = 0; n <= NZ; n++) for (let m = n % 2; m <= n; m += 2) IDX.push([n, m]);
  const bars = new Plot(barBox, { width: 620, height: 330, x: [-0.7, IDX.length - 0.3], y: [0, 1], ylabel: "|Z_nm|", font: 16, xticks: IDX.map((_, i) => i).filter(i => IDX[i][1] <= 1), xfmt: v => `${IDX[Math.round(v)]?.[0]}`, xlabel: "n (each group: m = n mod 2, …, n)", margin: { b: 50 } });
  const bA = bars.bars([], [], { color: C0, barWidth: 0.4, label: "original" });
  const bB = bars.bars([], [], { color: C1, barWidth: 0.4, label: "rotated copy" });
  bars.legend(undefined, "tr");
  const barNote = htmlEl("div", "", "", barBox);
  barNote.style.cssText = "font-size:15px;text-align:center";

  function renderBasis(): void {
    if (tilesMag !== st.mag) {
      tilesMag = st.mag;
      for (let n = 0; n <= NB; n++) for (let m = -n; m <= n; m += 2) tiles.get(`${n},${m}`)!.setAttribute("href", toDataURL(realZernike(n, m, TILE), TILE, st.mag));
    }
    frames.forEach((f, k) => f.style.setProperty("stroke", k === `${st.n},${st.m}` ? "var(--qb-fg, #222)" : "transparent"));
    const name = zernikeName(st.n, st.m);
    bigTitle.innerHTML = `<b>V${sub(st.n)}${sub(st.m)}</b>: R${sub(st.n)}${sub(Math.abs(st.m))}(ρ) ${st.m > 0 ? `cos ${st.m}θ` : st.m < 0 ? `sin ${-st.m}θ` : ""}${name ? ` — ${name}` : ""}`;
    const v = realZernike(st.n, st.m, 128), c2 = bigCan.getContext("2d")!, id = c2.createImageData(128, 128);
    scalarToRGBA(st.mag ? v.map(Math.abs) : v, { map: st.mag ? colormapLUT(viridis) : colormapLUT(diverging), lo: st.mag ? 0 : -1, hi: 1, nan: [0, 0, 0, 0] }, id.data);
    c2.putImageData(id, 0, 0);
    const rs = Array.from({ length: 201 }, (_, i) => i / 200);
    profLine.set(rs, rs.map(r => zernikeRadial(st.n, Math.abs(st.m), r)));
    prof.setLabels({ title: `radial part: ${(st.n - Math.abs(st.m)) / 2} zero(s) in 0 < ρ < 1` });
  }

  function renderRotation(): void {
    const a = st.alpha * Math.PI / 180;
    const fA = shapeImage(st.shape), fB = shapeImage(st.shape, a, st.mirror);
    greyCanvas(imA.c, fA); greyCanvas(imB.c, fB);
    imB.t.innerHTML = `<b style="color:${C1}">${st.mirror ? "mirrored, " : ""}rotated by ${st.alpha}°</b>`;
    const ZA = zernikeOf(fA), ZB = zernikeOf(fB);
    const magA = IDX.map((_, k) => Math.hypot(ZA.re[k], ZA.im[k])), magB = IDX.map((_, k) => Math.hypot(ZB.re[k], ZB.im[k]));
    const top = Math.max(...magA, ...magB) * 1.08;
    bars.setDomain(undefined, [0, top]);
    bA.set(IDX.map((_, i) => i - 0.2), magA); bB.set(IDX.map((_, i) => i + 0.2), magB);
    let worst = 0;
    magA.forEach((v, i) => { if (v > 0.02 * magA[0]) worst = Math.max(worst, Math.abs(magB[i] - v) / v); });
    barNote.textContent = `largest relative difference of |Z_nm| (moments above 2 % of |Z₀₀|): ${(100 * worst).toFixed(1)} %`;
    // phase dial
    const k = IDX.findIndex(([n, m]) => n === st.pn && m === st.pm);
    dial.replaceChildren();
    const circ = svgEl("circle", { cx: 0, cy: 0, r: R }, dial); circ.style.cssText = "fill:none;stroke:var(--qb-grid,#ccc);stroke-width:1.5";
    for (const [x1, y1, x2, y2] of [[-R, 0, R, 0], [0, -R, 0, R]]) svgEl("line", { x1, y1, x2, y2 }, dial).style.cssText = "stroke:var(--qb-grid,#ccc);stroke-width:1.5";
    const arrow = (re: number, im: number, col: string, scale: number) => {
      const x = re * scale, y = -im * scale, L = Math.hypot(x, y);
      if (L < 2) return;
      const ux = x / L, uy = y / L;
      svgEl("line", { x1: 0, y1: 0, x2: x - ux * 10, y2: y - uy * 10 }, dial).style.cssText = `stroke:${col};stroke-width:4;stroke-linecap:round`;
      svgEl("polygon", { points: `${x},${y} ${x - ux * 13 - uy * 6},${y - uy * 13 + ux * 6} ${x - ux * 13 + uy * 6},${y - uy * 13 - ux * 6}` }, dial).style.setProperty("fill", col);
    };
    const mA = Math.hypot(ZA.re[k], ZA.im[k]), mB = Math.hypot(ZB.re[k], ZB.im[k]), sc = 0.92 * R / Math.max(mA, mB, 1e-12);
    arrow(ZA.re[k], ZA.im[k], C0, sc); arrow(ZB.re[k], ZB.im[k], C1, sc);
    const pA = Math.atan2(ZA.im[k], ZA.re[k]), pB = Math.atan2(ZB.im[k], ZB.re[k]);
    let d = (pB - pA) * 180 / Math.PI; d = ((d + 540) % 360) - 180;
    let pred = -st.pm * st.alpha; pred = ((pred % 360) + 540) % 360 - 180;
    dialText.innerHTML = `Z${sub(st.pn)}${sub(st.pm)}: phase change <b>${d.toFixed(0).replace("-", "−")}°</b>` + (st.mirror ? ` (mirror ↔: phase φ → m·180° − φ)` : `, −mα = ${pred.toFixed(0).replace("-", "−")}°`);
  }

  function render(): void {
    tb.forEach((b, i) => b.classList.toggle("active", i === (st.tab === "basis" ? 0 : 1)));
    vb.forEach((b, i) => b.classList.toggle("active", i === (st.mag ? 1 : 0)));
    sb.forEach((b, i) => b.classList.toggle("active", SHAPES[i][0] === st.shape));
    pb.forEach((b, i) => b.classList.toggle("active", PH[i][0] === st.pn && PH[i][1] === st.pm));
    aS.set(st.alpha); mC.checked = st.mirror;
    const basis = st.tab === "basis";
    bctl.style.display = basis ? "flex" : "none"; rctl.style.display = basis ? "none" : "flex";
    basisBox.style.display = basis ? "flex" : "none"; rotBox.style.display = basis ? "none" : "flex";
    if (basis) renderBasis(); else renderRotation();
  }

  const presets: { id: string; label: string; s: Partial<typeof st> }[] = [
    { id: "defocus", label: "defocus (n = 2, m = 0)", s: { tab: "basis", n: 2, m: 0 } },
    { id: "astig", label: "astigmatism (2, 2)", s: { tab: "basis", n: 2, m: 2 } },
    { id: "rotate", label: "rotate: magnitudes stay", s: { tab: "rotation", shape: "blob", alpha: 60, pn: 2, pm: 2 } },
    { id: "six", label: "6 vs 9", s: { tab: "rotation", shape: "six", alpha: 180, pn: 1, pm: 1 } },
    { id: "mirror", label: "same |Z|, different shape", s: { tab: "rotation", shape: "F", alpha: 0, mirror: true, pn: 3, pm: 1 } },
  ];
  const only = host.dataset.presets?.split(",").map(x => x.trim());
  const shown = only ? presets.filter(p => only.includes(p.id)) : presets;
  const btns = shown.map(p => button(prow, p.label, () => load(p)));
  const p0 = presets.find(p => p.id === host.dataset.preset) ?? presets[0];
  button(prow, "⟲ reset", () => load(p0));
  function load(p: typeof presets[number]): void {
    Object.assign(st, { tab: "basis", mag: false, n: 2, m: 0, shape: "blob", alpha: 0, mirror: false, pn: 2, pm: 2 }, p.s);
    if (p === p0 && (host.dataset.tab === "basis" || host.dataset.tab === "rotation")) st.tab = host.dataset.tab;
    btns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    render();
  }
  load(p0);
}

autoMount(NAME, mount);
