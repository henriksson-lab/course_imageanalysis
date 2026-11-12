// granulo: granulometry, size filtering by openings with structuring elements of growing
// radius r (morphology day, unit morph-openclose). Left: the opening at the current r;
// pixels already removed are coloured by the radius at which they disappeared (a "size
// map"). Right: the area left a(r) and the pattern spectrum a(r) − a(r + 1), a histogram of
// object sizes without segmenting any object. "via distance transform" shows the distance
// transform with the erosion {DT > r} and its boundary (the iso-line DT = r).
// Options: data-preset (sizes | touching | lines | square | nuclei).
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { viridis, toCss } from "../lib/colour";
import { SEKind, openingFunction, granulometry, seDistance } from "./size";
import { SIZE, twoSizes, touching, lines, nucleiMask } from "./images";

const NAME = "granulo";
const RMAX = 24;

type ImgId = "sizes" | "touching" | "lines" | "nuclei";
const IMAGES: { id: ImgId; label: string; make: () => Uint8Array }[] = [
  { id: "sizes", label: "two sizes + specks", make: () => twoSizes() },
  { id: "touching", label: "touching discs", make: () => touching() },
  { id: "lines", label: "bars", make: () => lines() },
  { id: "nuclei", label: "nuclei (thresholded)", make: () => nucleiMask() },
];
const SES: { id: SEKind; label: string }[] = [
  { id: "disc", label: "disc" }, { id: "square", label: "square" }, { id: "diamond", label: "diamond" }, { id: "hline", label: "horizontal line" },
];
interface Preset { id: string; label: string; img: ImgId; se: SEKind; r: number; dt?: boolean }
const PRESETS: Preset[] = [
  { id: "sizes", label: "two sizes, two peaks", img: "sizes", se: "disc", r: 7 },
  { id: "touching", label: "touching discs: one bigger object?", img: "touching", se: "disc", r: 8 },
  { id: "lines", label: "lines: find the horizontal ones", img: "lines", se: "hline", r: 4 },
  { id: "square", label: "square SE on discs", img: "sizes", se: "square", r: 5 },
  { id: "nuclei", label: "nuclei mask", img: "nuclei", se: "disc", r: 6 },
  { id: "dt", label: "erosion = threshold of DT", img: "sizes", se: "disc", r: 5, dt: true },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] .gr-img { flex: 0 1 420px; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .gr-img canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; display: block; border: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] .gr-plots { flex: 1 1 300px; min-width: 0; max-width: 380px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 300px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .gr-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .gr-key { font-size: 14px; display: flex; align-items: center; gap: .4em; flex-wrap: wrap; margin-top: .2em; }
.qb-demo[data-demo="${NAME}"] .gr-key span.sw { display: inline-block; width: 14px; height: 14px; border-radius: 2px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
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
  const st = { img: p0.img, se: p0.se, r: p0.r, dt: false };
  const cache = new Map<ImgId, Uint8Array>();
  let mask: Uint8Array = new Uint8Array(0), of: Int32Array = new Int32Array(0), dist: Float32Array = new Float32Array(0), gran = { area: [0], spectrum: [0] };

  const imgBox = htmlEl("div", "gr-img", "", host);
  const canvas = htmlEl("canvas", "", "", imgBox);
  canvas.width = SIZE; canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;
  const key = htmlEl("div", "gr-key", "", imgBox);
  const plots = htmlEl("div", "gr-plots", "", host);
  const pa = new Plot(plots, { width: 380, height: 230, x: [0, RMAX], y: [0, 1], xlabel: "", ylabel: "area left (%)", xticks: [0, 5, 10, 15, 20], ariaLabel: "area left after opening" });
  const ps = new Plot(plots, { width: 380, height: 250, x: [-0.5, RMAX + 0.5], y: [0, 1], xlabel: "radius r", ylabel: "spectrum (%)", xticks: [0, 5, 10, 15, 20], ariaLabel: "pattern spectrum" });
  const aLine = pa.line([], [], { color: "#1f77b4", width: 3 });
  const aDots = pa.scatter([], [], { color: "#1f77b4", r: 4 });
  const aV = pa.vline(0, { color: "#d62728" });
  const barStyle = { colors: [] as string[] };
  const sBars = ps.bars([], [], barStyle);
  const sV = ps.vline(0, { color: "#d62728" });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "gr-lab", "image", panel);
  const imgBtns = presets(panel, IMAGES.map(m => ({ label: m.label, apply: () => { st.img = m.id; recompute(); } })));
  htmlEl("div", "gr-lab", "structuring element B_r", panel);
  const seBtns = presets(panel, SES.map(m => ({ label: m.label, apply: () => { st.se = m.id; recompute(); } })));
  const rS = slider(panel, "radius r", 0, RMAX, 1, st.r, v => { st.r = v; draw(); });
  const dtBox = checkbox(panel, "via distance transform", false, v => { st.dt = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "gr-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => load(p0));

  function load(p: Preset): void {
    st.img = p.img; st.se = p.se; st.r = p.r; st.dt = !!p.dt;
    rS.set(p.r); dtBox.checked = st.dt;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute();
  }

  function recompute(): void {
    if (!cache.has(st.img)) cache.set(st.img, IMAGES.find(m => m.id === st.img)!.make());
    mask = cache.get(st.img)!;
    of = openingFunction(mask, SIZE, SIZE, st.se, RMAX);
    const d = seDistance(mask, SIZE, SIZE, st.se);
    dist = st.se === "disc" ? d.map(v => Math.sqrt(v)) : d;
    gran = granulometry(of, RMAX);
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMAGES[i].id === st.img));
    seBtns.forEach((b, i) => b.classList.toggle("active", SES[i].id === st.se));
    draw();
  }

  function draw(): void {
    const r = st.r, id = ctx.createImageData(SIZE, SIZE), px = id.data, n = SIZE * SIZE;
    const total = gran.area[0] || 1;
    if (!st.dt) {
      for (let i = 0; i < n; i++) {
        let c: [number, number, number];
        if (of[i] < 0) c = [255, 255, 255];
        else if (of[i] >= r) c = [40, 50, 75];
        else { const v = viridis(Math.min(1, (of[i] + 1) / 16)); c = [v[0] * 255, v[1] * 255, v[2] * 255]; }
        px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
      }
      key.innerHTML = "";
      key.append("dark: kept by the opening · removed at radius ");
      for (const q of [1, 4, 8, 12, 16]) {
        const sw = htmlEl("span", "sw", "", key); sw.style.background = toCss(viridis(q / 16));
        key.append(q === 16 ? "16+" : String(q));
      }
    } else {
      let mx = 1;
      for (let i = 0; i < n; i++) if (Number.isFinite(dist[i])) mx = Math.max(mx, dist[i]);
      for (let i = 0; i < n; i++) {
        const x = i % SIZE, y = (i - x) / SIZE, v = dist[i];
        let c: [number, number, number];
        if (!mask[i]) c = [255, 255, 255];
        else {
          const g = 235 - 200 * Math.min(1, v / mx);
          const inside = v > r;
          c = inside ? [g * 0.55 + 100, g * 0.75 + 60, 70] : [g, g, g];
          // iso-line: inside pixel with a 4-neighbour outside
          if (inside && [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => {
            const xx = x + dx, yy = y + dy;
            return xx >= 0 && yy >= 0 && xx < SIZE && yy < SIZE && dist[yy * SIZE + xx] <= r;
          })) c = [214, 39, 40];
        }
        px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
      }
      const se = SES.find(s => s.id === st.se)!.label;
      key.textContent = `grey: distance to the background (${st.se === "disc" ? "Euclidean" : st.se === "square" ? "chessboard" : st.se === "diamond" ? "city-block" : "along the row"}), dark = far · green: DT > ${r} = the erosion by the ${se} of radius ${r} · red: its edge (DT ≈ ${r})`;
    }
    ctx.putImageData(id, 0, 0);
    // plots
    const rs = Array.from({ length: RMAX + 1 }, (_, k) => k);
    const pct = gran.area.slice(0, RMAX + 1).map(a => 100 * a / total);
    pa.setDomain([0, RMAX], [0, 100]);
    aLine.set(rs, pct); aDots.set(rs, pct); aV.set(r);
    const spec = gran.spectrum.map(a => 100 * a / total);
    const top = Math.max(5, ...spec.slice(1)) * 1.1;
    ps.setDomain([-0.5, RMAX + 0.5], [0, Math.ceil(top / 5) * 5]);
    barStyle.colors = rs.map(k => toCss(viridis(Math.min(1, (k + 1) / 16))));
    sBars.set(rs, spec); sV.set(r - 0.5);
    const kept = gran.area[r] ?? 0;
    ro.textContent = `r = ${r}: ${kept} of ${total} pixels left (${(100 * kept / total).toFixed(1)} %)\nremoved at radius ${r + 1}: ${gran.spectrum[r] ?? 0} pixels`;
    ro.style.whiteSpace = "pre-line";
  }

  load(p0);
}

autoMount(NAME, mount);
