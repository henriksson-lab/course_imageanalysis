// Leakage in validation: a pixel classifier (k nearest neighbours on a feature stack) for
// "nucleus or background" in twelve synthetic images from four days, tested with five
// ways of splitting the data into 4 folds: random pixels, random 8 × 8 blocks, blocks with a
// buffer zone, by image, by day. Left: the 12 images (rows = days) coloured by their role in
// the fold that tests the selected image (red = test, blue = train, grey = buffer). Middle: test
// accuracy of every split (bar = mean of 4 folds, whisker = range). Right: the prediction on
// the selected image. "Nothing to learn" makes the labels random blobs unrelated to the image.
// Auto-mounts on <div data-demo="leakage">. Options: data-preset="default|nothing|buffer|newday|noday|augment".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { toBytes } from "../lib/imageops";
import * as M from "./model";

const NAME = "leakage";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .lk-thumbs { flex: 0 1 250px; min-width: 200px; }
[data-demo="${NAME}"] .lk-grid { display: grid; grid-template-columns: auto repeat(3, 1fr); gap: 4px; align-items: center; }
[data-demo="${NAME}"] .lk-grid span { font-size: 14px; color: var(--qb-muted); writing-mode: vertical-rl; transform: rotate(180deg); }
[data-demo="${NAME}"] .lk-grid canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border-radius: 2px; cursor: pointer; outline: 3px solid transparent; }
[data-demo="${NAME}"] .lk-grid canvas.sel { outline-color: #ffd400; }
[data-demo="${NAME}"] .lk-bars { flex: 1 1 340px; max-width: 460px; min-width: 260px; }
[data-demo="${NAME}"] .lk-pred { flex: 0 1 190px; min-width: 150px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .lk-pred canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] .lk-key { font-size: 13px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .35em; }
[data-demo="${NAME}"] .lk-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .lk-busy { font-size: 14px; color: var(--qb-c2); min-height: 1.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
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
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let st: M.Settings = { ...M.DEFAULTS, ...p0.settings };
  let show: M.Strategy = p0.show ?? "pixels", selImg = 0;

  // ---- layout ----
  const thumbs = htmlEl("figure", "lk-thumbs", "", host);
  htmlEl("figcaption", "", "12 images, 4 days", thumbs);
  const grid = htmlEl("div", "lk-grid", "", thumbs);
  const cvs: HTMLCanvasElement[] = [];
  for (let d = 0; d < M.DAYS; d++) {
    htmlEl("span", "", "day " + (d + 1), grid);
    for (let j = 0; j < M.PER_DAY; j++) {
      const c = htmlEl("canvas", "", "", grid);
      c.width = M.S; c.height = M.S;
      const i = d * M.PER_DAY + j;
      c.addEventListener("click", () => { selImg = i; drawThumbs(); drawPred(); });
      cvs.push(c);
    }
  }
  htmlEl("div", "lk-key", "red: test · blue: train · grey: buffer (unused)", thumbs);
  const stratRow = htmlEl("div", "qb-presets", "", thumbs);
  stratRow.style.marginTop = ".3em";
  const stratBtns = M.STRATEGIES.map(s => button(stratRow, s.label, () => { show = s.id; drawThumbs(); drawPred(); drawBars(); }));

  const barsF = htmlEl("figure", "lk-bars", "", host);
  htmlEl("figcaption", "", "test accuracy of each way to split", barsF);
  const plot = new Plot(barsF, { x: [0.4, 5.6], y: [0.4, 1.04], yticks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], ylabel: "test accuracy", height: 380, xticks: [1, 2, 3, 4, 5], xfmt: v => ["", "pixels", "blocks", "buffer", "image", "day"][Math.round(v)] ?? "" });
  const busy = htmlEl("div", "lk-busy", "", barsF);

  const predF = htmlEl("figure", "lk-pred", "", host);
  htmlEl("figcaption", "", "selected image", predF);
  const cvImg = htmlEl("canvas", "", "", predF);
  cvImg.width = M.S; cvImg.height = M.S;
  htmlEl("figcaption", "", "prediction on its test pixels", predF);
  const cvPred = htmlEl("canvas", "", "", predF);
  cvPred.width = M.S; cvPred.height = M.S;
  const predRo = htmlEl("div", "lk-key", "", predF);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "lk-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const dayS = slider(panel, "day effect", 0, 2, 0.25, st.dayEffect, v => set({ dayEffect: v }));
  const kS = slider(panel, "neighbours k", 1, 15, 2, st.k, v => set({ k: v }));
  const bufS = slider(panel, "buffer (px)", 0, 12, 1, st.buffer, v => set({ buffer: v }));
  const nothingC = checkbox(panel, "nothing to learn (random labels)", st.nothing, v => set({ nothing: v }));
  const augC = checkbox(panel, "augment (rotate) before splitting", st.augment, v => set({ augment: v }));
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Synthetic images. Features: smoothed intensities, gradient magnitude and blob filters at several scales. Classifier: k nearest neighbours over all training pixels (k-d tree search, cut off after 400 nodes); 200 test pixels per fold.", panel);

  let data: M.DataSet = M.makeSet(st);
  let results: (M.StrategyResult | null)[] = M.STRATEGIES.map(() => null);
  let job = 0;

  function set(p: Partial<M.Settings>): void {
    const remake = p.dayEffect !== undefined || p.nothing !== undefined || p.augment !== undefined;
    st = { ...st, ...p };
    pbtn.forEach(b => b.classList.remove("active"));
    if (remake) data = M.makeSet(st);
    run();
  }
  function load(p: M.Preset): void {
    st = { ...M.DEFAULTS, ...p.settings }; show = p.show ?? "pixels";
    dayS.set(st.dayEffect); kS.set(st.k); bufS.set(st.buffer); nothingC.checked = st.nothing; augC.checked = st.augment;
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    data = M.makeSet(st);
    run(first);
    first = false;
  }
  let first = true;

  /** Index into data of original image i (copies follow their original when augmenting). */
  const idx = (i: number) => (data.n === M.DAYS * M.PER_DAY ? i : 4 * i);

  function run(sync = false): void {
    const my = ++job;
    if (sync) {
      results = M.STRATEGIES.map(q => M.summarise(q.id, M.evaluate(data, q.id, st)));
      drawThumbs(); drawPred(); drawBars();
      busy.textContent = "";
      return;
    }
    results = M.STRATEGIES.map(() => null);
    drawThumbs(); drawPred(); drawBars();
    busy.textContent = "computing…";
    let k = 0;
    const step = () => {
      if (my !== job) return;
      results[k] = M.summarise(M.STRATEGIES[k].id, M.evaluate(data, M.STRATEGIES[k].id, st));
      drawBars();
      k++;
      if (k < M.STRATEGIES.length) setTimeout(step, 0); else busy.textContent = "";
    };
    setTimeout(step, 0);
  }

  function drawThumbs(): void {
    stratBtns.forEach((b, i) => b.classList.toggle("active", M.STRATEGIES[i].id === show));
    const folds = M.assignFolds(data, show, 1);
    const cnt = [0, 0, 0, 0];
    folds[idx(selImg)].forEach(v => cnt[v]++);
    const f = cnt.indexOf(Math.max(...cnt));
    const roles = M.roles(data, folds, f, show, st.buffer);
    for (let i = 0; i < 12; i++) {
      const j = idx(i), bytes = toBytes(data.images[j], { mode: "clip", lo: 0, hi: 230 }), ro = roles[j];
      const ctx = cvs[i].getContext("2d")!, id = ctx.createImageData(M.S, M.S);
      for (let p = 0; p < M.S * M.S; p++) {
        const g = bytes[p], c = ro[p] === 1 ? [230, 40, 40] : ro[p] === 2 ? [150, 150, 150] : [40, 110, 230], a = ro[p] === 2 ? 0.6 : 0.4;
        id.data[4 * p] = g * (1 - a) + c[0] * a; id.data[4 * p + 1] = g * (1 - a) + c[1] * a; id.data[4 * p + 2] = g * (1 - a) + c[2] * a; id.data[4 * p + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      cvs[i].classList.toggle("sel", i === selImg);
    }
  }

  function drawPred(): void {
    const j = idx(selImg), bytes = toBytes(data.images[j], { mode: "clip", lo: 0, hi: 230 }), lab = data.labels[j];
    const ctx = cvImg.getContext("2d")!, id = ctx.createImageData(M.S, M.S);
    for (let p = 0; p < M.S * M.S; p++) {
      const x = p % M.S, y = (p - x) / M.S;
      const edge = lab[p] && ((x > 0 && !lab[p - 1]) || (x < M.S - 1 && !lab[p + 1]) || (y > 0 && !lab[p - M.S]) || (y < M.S - 1 && !lab[p + M.S]));
      const g = bytes[p];
      id.data[4 * p] = edge ? 255 : g; id.data[4 * p + 1] = edge ? 212 : g; id.data[4 * p + 2] = edge ? 0 : g; id.data[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const res = M.predictImage(data, show, st, j);
    const c2 = cvPred.getContext("2d")!, id2 = c2.createImageData(M.S, M.S);
    for (let p = 0; p < M.S * M.S; p++) {
      const test = res.roles[j][p] === 1, ok = res.pred[p] === lab[p];
      const base = res.pred[p] ? 255 : 0;
      const c = !test ? [128, 128, 160] : ok ? [base, base, base] : [230, 40, 40];
      id2.data[4 * p] = c[0]; id2.data[4 * p + 1] = c[1]; id2.data[4 * p + 2] = c[2]; id2.data[4 * p + 3] = 255;
    }
    c2.putImageData(id2, 0, 0);
    predRo.textContent = `${M.STRATEGIES.find(s => s.id === show)!.label}: ${(100 * res.acc).toFixed(1)} % correct on this image. White = predicted nucleus, black = background, red = wrong, blue-grey = not a test pixel. Truth outline yellow above.`;
  }

  function drawBars(): void {
    plot.clear();
    const maj = M.majority(data);
    plot.hline(maj, { color: "var(--qb-muted)", width: 1.5, text: maj > 0.6 ? `always "background": ${(100 * maj).toFixed(1)}` : "chance" });
    const xs: number[] = [], hs: number[] = [], cols: string[] = [];
    results.forEach((r, i) => { if (r) { xs.push(i + 1); hs.push(r.mean); cols.push(M.STRATEGIES[i].id === show ? seriesColor(1) : seriesColor(0)); } });
    plot.bars(xs, hs, { colors: cols, barWidth: 0.7, base: 0.4 });
    results.forEach((r, i) => {
      if (!r) return;
      plot.line([i + 1, i + 1], [r.min, r.max], { color: "var(--qb-fg)", width: 2.5 });
      plot.line([i + 0.85, i + 1.15], [r.min, r.min], { color: "var(--qb-fg)", width: 2.5 });
      plot.line([i + 0.85, i + 1.15], [r.max, r.max], { color: "var(--qb-fg)", width: 2.5 });
      plot.text(i + 1, r.max + 0.012, (100 * r.mean).toFixed(1), { anchor: "middle", size: 16 });
    });
  }

  load(p0);
}

autoMount(NAME, mount);
