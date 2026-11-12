// receptive: the receptive field of a stack of convolution and pooling layers. Bottom row: the
// input pixels (a 1D cut through the image); every row above is one layer. Click any unit: the
// units it depends on light up in every layer below, down to the input, and the size of its
// receptive field is printed per layer. A 2D view shows the same field as a square on a
// 64 × 64 input with a 20-pixel nucleus. Layers are added with the buttons or come from presets.
//
// <div data-demo="receptive" data-preset="three"></div>
import { autoMount, button, htmlEl, presets as presetRow, svgEl } from "../lib/ui";
import { receptiveField } from "../lib/cnn";

const NAME = "receptive";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .rf-main { flex: 1 1 640px; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .45em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Layer { kind: "conv" | "pool"; k: number; s: number; d: number }
const conv = (k: number, d = 1): Layer => ({ kind: "conv", k, s: 1, d });
const pool: Layer = { kind: "pool", k: 2, s: 2, d: 1 };
const PRESETS: { id: string; label: string; layers: Layer[]; n?: number }[] = [
  { id: "one", label: "one 3×3 layer", layers: [conv(3)] },
  { id: "three", label: "three 3×3 layers = 7", layers: [conv(3), conv(3), conv(3)] },
  { id: "seven", label: "one 7×7 layer", layers: [conv(7)] },
  { id: "pool", label: "with pooling", layers: [conv(3), pool, conv(3), pool, conv(3)] },
  { id: "nucleus", label: "how deep to see a whole nucleus?", layers: [conv(3), conv(3), pool, conv(3), conv(3), pool, conv(3), conv(3)] },
  { id: "dilated", label: "dilated: 1, 2, 4", layers: [conv(3, 1), conv(3, 2), conv(3, 4)] },
];
const NIN = 40, C = 16, NUCLEUS = 20;

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  let layers: Layer[] = start.layers.map(l => ({ ...l }));
  let sel: [number, number] | null = null;   // (layer index ≥ 1, unit)

  const main = htmlEl("div", "rf-main", "", host);
  const svg = svgEl("svg", { width: "100%", role: "img", "aria-label": "Layers of units and the receptive field of one unit" }, main);
  const panel = htmlEl("div", "qb-panel", "", host);
  const row1 = htmlEl("div", "qb-presets", "", panel);
  button(row1, "+ conv 3", () => add(conv(3)));
  button(row1, "+ conv 5", () => add(conv(5)));
  button(row1, "+ 2×2 pool", () => add(pool));
  button(row1, "+ conv 3, dilation 2", () => add(conv(3, 2)));
  button(row1, "remove top layer", () => { layers.pop(); sel = null; draw(); });
  const svg2 = svgEl("svg", { viewBox: "0 0 300 300", width: "100%", style: "max-width:260px", role: "img", "aria-label": "Receptive field in 2D" }, panel);
  const readout = htmlEl("div", "qb-readout", "", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => { layers = p.layers.map(l => ({ ...l })); sel = null; draw(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p)); } })));
  presetBtns[PRESETS.indexOf(start)]?.classList.add("active");

  function add(l: Layer): void { if (layers.length < 9) { layers.push({ ...l }); sel = null; draw(); } }

  // units per layer ("valid": no padding) and the first input pixel each unit sees
  function geometry(): { n: number[]; first: number[][] } {
    const n = [NIN], first: number[][] = [Array.from({ length: NIN }, (_, i) => i)];
    layers.forEach((L, l) => {
      const m = Math.max(0, Math.floor((n[l] - L.d * (L.k - 1) - 1) / L.s) + 1);
      n.push(m);
      first.push(Array.from({ length: m }, (_, i) => first[l][i * L.s]));
    });
    return { n, first };
  }

  function draw(): void {
    const { n, first } = geometry(), rf = receptiveField(layers), NL = layers.length;
    const rowH = Math.min(56, 400 / Math.max(1, NL)), H = 70 + rowH * NL + 40, W = 900, x0 = 150, cw = (W - x0 - 20) / NIN;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    if (!sel && NL > 0 && n[NL] > 0) sel = [NL, Math.floor((n[NL] - 1) / 2)];
    // which units are in the selected unit's field, per layer
    const lit: Set<number>[] = Array.from({ length: NL + 1 }, () => new Set<number>());
    if (sel) {
      lit[sel[0]].add(sel[1]);
      for (let l = sel[0]; l >= 1; l--) {
        const L = layers[l - 1];
        for (const i of lit[l]) for (let u = 0; u < L.k; u++) lit[l - 1].add(i * L.s + u * L.d);
      }
    }
    const Y = (l: number) => H - 50 - l * rowH;
    for (let l = 0; l <= NL; l++) {
      const y = Y(l), sz = Math.min(cw * 0.8, rowH * 0.55);
      const label = l === 0 ? "input" : layers[l - 1].kind === "pool" ? `pool 2, s 2` : `conv ${layers[l - 1].k}` + (layers[l - 1].d > 1 ? `, d ${layers[l - 1].d}` : "");
      const t = svgEl("text", { x: 8, y: y + 6 }, svg); t.textContent = label;
      // the jump: unit i sits at the centre of its field
      const r = rf.r[l];
      for (let i = 0; i < n[l]; i++) {
        const cx = x0 + (first[l][i] + (r - 1) / 2 + 0.5) * cw;
        const on = lit[l].has(i), isSel = sel && sel[0] === l && sel[1] === i;
        const rect = svgEl("rect", { x: cx - sz / 2, y: y - sz / 2, width: sz, height: sz, rx: 3, fill: isSel ? "#d62728" : on ? "#ff7f0e" : "#e6e6e6", stroke: "#999" }, svg);
        if (l > 0) { rect.style.cursor = "pointer"; rect.addEventListener("pointerdown", ev => { ev.preventDefault(); sel = [l, i]; draw(); }); }
      }
    }
    // bracket on the input
    if (sel && lit[0].size) {
      const a = Math.min(...lit[0]), b = Math.max(...lit[0]), y = Y(0) + 22;
      svgEl("path", { d: `M${x0 + a * cw + 2},${y - 6} V${y} H${x0 + (b + 1) * cw - 2} V${y - 6}`, fill: "none", stroke: "#d62728", "stroke-width": 2.5 }, svg);
      const t = svgEl("text", { x: x0 + (a + b + 1) * cw / 2, y: y + 20, "text-anchor": "middle", fill: "#d62728" }, svg);
      t.textContent = `receptive field ${b - a + 1}`;
    }
    // 2D view
    svg2.replaceChildren();
    const S = 300, px = S / 64;
    svgEl("rect", { x: 0, y: 0, width: S, height: S, fill: "#f3f3f3", stroke: "#999" }, svg2);
    svgEl("circle", { cx: S / 2, cy: S / 2, r: NUCLEUS / 2 * px, fill: "#9ecae1", stroke: "#3182bd", "stroke-width": 2 }, svg2);
    const rTop = sel ? rf.r[sel[0]] : 1;
    svgEl("rect", { x: S / 2 - rTop / 2 * px, y: S / 2 - rTop / 2 * px, width: rTop * px, height: rTop * px, fill: "#ff7f0e", "fill-opacity": 0.25, stroke: "#d62728", "stroke-width": 2.5 }, svg2);
    const t2 = svgEl("text", { x: 6, y: S - 8, style: "font-size:17px" }, svg2);
    t2.textContent = `nucleus ${NUCLEUS} px, field ${rTop} × ${rTop}`;
    // readout
    let w = 0;
    const lines = layers.map((L, l) => {
      const wl = L.kind === "pool" ? 0 : L.k * L.k * C * C + C;
      w += wl;
      return `${String(l + 1).padStart(2)} ${(L.kind === "pool" ? "pool 2" : `conv ${L.k}${L.d > 1 ? " d" + L.d : ""}`).padEnd(9)} ${String(rf.r[l + 1]).padStart(3)} ${String(rf.j[l + 1]).padStart(4)} ${L.kind === "pool" ? "" : String(wl).padStart(6)}`;
    });
    readout.textContent = `   layer   field jump weights\n` + lines.join("\n") + `\ntotal ${w} weights (2D, ${C} channels in and out)` + (sel ? `\nselected unit (layer ${sel[0]}): field ${rf.r[sel[0]]} pixels` : "");
  }
  draw();
}

autoMount(NAME, mount);
