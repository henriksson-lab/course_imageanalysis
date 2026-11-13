// Small DOM helpers shared by the demos of the diffusion-models day (scorematch1d,
// noisespectrum, samplerpaths, genmetrics, guidance1d, latentdiff, posterior1d,
// virtualstain). Candidates for lib/ui.ts later (see plan/build-issues.md, "local helpers").
import { htmlEl, button } from "../lib/ui";

/** Add a <style> once per demo name. Selectors in `css` should start with .qb-demo[data-demo="name"]. */
export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name;
  s.textContent = css;
  document.head.appendChild(s);
}

export interface Choice<T> { set(v: T): void; get(): T; row: HTMLElement }

/** A labelled row of toggle buttons; exactly one is active. */
export function choiceRow<T extends string | number>(parent: Element, label: string, items: { v: T; label: string }[], value: T, onPick: (v: T) => void): Choice<T> {
  const row = htmlEl("div", "dv-choice", "", parent);
  if (label) htmlEl("span", "dv-choice-lab", label, row);
  let cur = value;
  const btns = items.map(it => button(row, it.label, () => { cur = it.v; mark(); onPick(it.v); }));
  const mark = () => btns.forEach((b, i) => b.classList.toggle("active", items[i].v === cur));
  mark();
  return { set: (v: T) => { cur = v; mark(); }, get: () => cur, row };
}

/** A captioned canvas showing a grey image (values mapped linearly from [lo, hi]). */
export interface ImgPanel { fig: HTMLElement; canvas: HTMLCanvasElement; cap: HTMLElement; draw(v: ArrayLike<number>, w: number, h: number, lo: number, hi: number, map?: (t: number) => [number, number, number]): void }
export function imgPanel(parent: Element, caption: string, cls = "dv-img"): ImgPanel {
  const fig = htmlEl("figure", cls, "", parent);
  const canvas = htmlEl("canvas", "", "", fig);
  const cap = htmlEl("figcaption", "", caption, fig);
  return {
    fig, canvas, cap,
    draw(v, w, h, lo, hi, map) {
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const ctx = canvas.getContext("2d")!, im = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        const t = Math.min(1, Math.max(0, (v[i] - lo) / (hi - lo || 1)));
        const [r, g, b] = map ? map(t) : [255 * t, 255 * t, 255 * t];
        im.data[4 * i] = r; im.data[4 * i + 1] = g; im.data[4 * i + 2] = b; im.data[4 * i + 3] = 255;
      }
      ctx.putImageData(im, 0, 0);
    },
  };
}

/** Shared CSS for the helpers (scoped to one demo name). */
export function helperCss(name: string): string {
  const s = `.qb-demo[data-demo="${name}"]`;
  return `
${s} .dv-choice { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${s} .dv-choice-lab { margin-right: .2em; }
${s} figure.dv-img { margin: 0; display: flex; flex-direction: column; align-items: center; }
${s} figure.dv-img canvas { width: 100%; image-rendering: pixelated; border-radius: 3px; display: block; }
${s} figure.dv-img figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin-top: .15em; }
.reveal ${s} figure.dv-img figcaption { font-size: 18px; }
`;
}

/** Format a number with a fixed number of decimals, or "–" when not finite. */
export const fmt = (v: number, d = 2): string => Number.isFinite(v) ? v.toFixed(d) : "–";
