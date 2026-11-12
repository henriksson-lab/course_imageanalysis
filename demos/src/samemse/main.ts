// Same MSE, different quality. One image distorted in five ways (brightness shift, contrast
// change, Gaussian noise, blur, JPEG), each tuned by bisection on its parameter to the same
// mean squared error, side by side with MSE, PSNR (all equal) and SSIM (very different).
// A slider sets the common MSE; "SSIM maps" shows where each distortion loses structure.
// Auto-mounts on <div data-demo="samemse">. Options: data-image="cell|card", data-mse="144",
// data-maps="1".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { ssim, psnrFromMse } from "../lib/metrics";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { card, SIZE } from "../jpeg/testimages";
import { DISTORTIONS, DistId, tune, describe, LABEL } from "./distort";

const NAME = "samemse";
const N = SIZE;
const MSES = [16, 25, 50, 100, 144, 200, 300, 400];

const STYLE = `
.qb-demo[data-demo="${NAME}"] .sm-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 12px; flex: 1 1 640px; min-width: 0; max-width: 800px; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.25; }
.qb-demo[data-demo="${NAME}"] figcaption b { font-size: 17px; }
.qb-demo[data-demo="${NAME}"] figcaption .ss { font-weight: 600; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; }
.qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 220px; max-width: none; }
.reveal .qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { max-width: 260px; }
.qb-demo[data-demo="${NAME}"] .sm-btns { display: flex; flex-wrap: wrap; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
.qb-demo[data-demo="${NAME}"] .sm-credit { font-size: 13px; color: var(--qb-muted); grid-column: 1 / -1; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .sm-grid { max-width: 660px; gap: 4px 10px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; line-height: 1.15; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption b { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 14px; line-height: 1.15; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption b { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function paint(c: HTMLCanvasElement, v: ArrayLike<number>, scale = 1): void {
  c.width = N; c.height = N;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const g = Math.max(0, Math.min(255, Math.round(v[i] * scale)));
    id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = g; id.data[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const m0 = Number(d.mse ?? 144);
  const init = { img: (d.image === "card" ? "card" : "cell") as "cell" | "card", mi: Math.max(0, MSES.indexOf(m0) >= 0 ? MSES.indexOf(m0) : 4), maps: d.maps === "1" };
  const st = { ...init };

  const grid = htmlEl("div", "sm-grid", "", host);
  const panes = ["original", ...DISTORTIONS].map(id => {
    const f = htmlEl("figure", "", "", grid);
    const c = htmlEl("canvas", "", "", f);
    const cap = htmlEl("figcaption", "", "", f);
    return { id, c, cap };
  });
  const credit = htmlEl("div", "sm-credit", "", grid);

  const panel = htmlEl("div", "qb-panel", "", host);
  presets(panel, [
    { label: "same MSE (Wang–Bovik)", apply: () => { Object.assign(st, { img: "cell", mi: 4, maps: false }); syncUI(); load(); } },
    { label: "small error", apply: () => { Object.assign(st, { img: "cell", mi: 1, maps: false }); syncUI(); load(); } },
    { label: "large error", apply: () => { Object.assign(st, { img: "cell", mi: 7, maps: false }); syncUI(); load(); } },
    { label: "SSIM maps", apply: () => { Object.assign(st, { maps: true }); syncUI(); compute(); } },
  ]);
  const imgRow = htmlEl("div", "sm-btns", "", panel);
  const imgBtns = [button(imgRow, "cell", () => { st.img = "cell"; syncUI(); load(); }), button(imgRow, "test card", () => { st.img = "card"; syncUI(); load(); })];
  const mS = slider(panel, "MSE", 0, MSES.length - 1, 1, st.mi, v => { st.mi = v; later(); }, v => String(MSES[v]));
  const mapBox = checkbox(panel, "show SSIM maps", st.maps, on => { st.maps = on; compute(); });
  button(panel, "reset", () => { Object.assign(st, init); syncUI(); load(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Each distortion's parameter is found by bisection so that its MSE matches the target (images rounded to whole grey levels). PSNR = 10·log10(255²/MSE). SSIM: 11 × 11 Gaussian window, σ = 1.5; map white = 1, black = 0.", panel);

  function syncUI(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("on", (i === 0) === (st.img === "cell")));
    mS.set(st.mi); mapBox.checked = st.maps;
  }

  let img: Float64Array = new Float64Array(N * N), cellImg: Float64Array | null = null;
  function load(): void {
    credit.textContent = st.img === "cell" ? CELL_CREDIT : "Synthetic test card.";
    if (st.img === "cell") {
      if (!cellImg) { loadCell().then(g => { cellImg = Float64Array.from(g.data, v => Math.round(v)); load(); }); return; }
      img = cellImg;
    } else img = card();
    compute();
  }
  let timer = 0;
  function later(): void {
    window.clearTimeout(timer);
    timer = window.setTimeout(compute, 120);
  }
  const cache = new Map<string, { out: Float64Array; p: number; mse: number; ss: number; map: Float64Array }>();
  function compute(): void {
    const T = MSES[st.mi];
    const lines: string[] = [`MSE ${T}, PSNR ${psnrFromMse(T, 255).toFixed(2)} dB`];
    for (const pane of panes) {
      if (pane.id === "original") {
        paint(pane.c, st.maps ? new Float64Array(N * N).fill(255) : img);
        pane.cap.innerHTML = `<b>original</b><br>${st.maps ? "(SSIM maps: white = 1)" : "&nbsp;"}<br>MSE 0 · PSNR ∞<br><span class="ss">SSIM 1</span>`;
        continue;
      }
      const id = pane.id as DistId, key = `${st.img}|${T}|${id}`;
      let r = cache.get(key);
      if (!r) {
        const t = tune(id, img, N, N, T);
        const s = ssim(img, t.out, N, N, { peak: 255 });
        r = { out: t.out, p: t.p, mse: t.mse, ss: s.mean, map: Float64Array.from(s.map.data) };
        cache.set(key, r);
      }
      paint(pane.c, st.maps ? r.map.map(v => 255 * Math.max(0, v)) : r.out);
      pane.cap.innerHTML = `<b>${LABEL[id]}</b><br>${describe(id, r.p, img)}<br>MSE ${r.mse.toFixed(1)} · PSNR ${psnrFromMse(r.mse, 255).toFixed(1)} dB<br><span class="ss">SSIM ${r.ss.toFixed(3)}</span>`;
      lines.push(`SSIM ${r.ss.toFixed(3)}  ${id}`);
    }
    ro.textContent = lines.join("\n");
  }
  syncUI();
  load();
}

autoMount(NAME, mount);
