// Small UI helpers shared by the point-detection day's demos (peaks1d, spotdetect, falsespots,
// spotfit, crowdfit, spotmatch). The image panel is the edges day's ImgView.
import { htmlEl, slider, Slider } from "../lib/ui";
export { ImgView, injectStyle } from "../canny/imgview";

/** Detection colours: matched (TP), false (FP), missed truth (FN), neutral candidate. */
export const TP_COL = "#2ca02c", FP_COL = "#e8402a", FN_COL = "#ffd400", CAND_COL = "#5b8ff9";

/** A slider on a log scale between lo and hi (u = 0..100 internally). */
export function logSlider(parent: Element, label: string, lo: number, hi: number, value: number, onInput: (v: number) => void,
  fmt: (v: number) => string = v => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2))): Slider & { setValue(v: number): void } {
  const of = (u: number) => lo * Math.pow(hi / lo, u / 100), uo = (v: number) => 100 * Math.log(v / lo) / Math.log(hi / lo);
  const s = slider(parent, label, 0, 100, 0.5, uo(value), u => onInput(round3(of(u))), u => fmt(of(u)));
  return { ...s, setValue: (v: number) => s.set(uo(v)) };
}
const round3 = (v: number) => Number(v.toPrecision(3));

/** A row of tab buttons; returns a function that marks the active one. */
export function tabRow(parent: Element, labels: string[], onPick: (i: number) => void): (i: number) => void {
  const row = htmlEl("div", "qb-presets pd-tabs", "", parent);
  const btns = labels.map((l, i) => {
    const b = htmlEl("button", "qb-btn", l, row);
    b.type = "button";
    b.addEventListener("click", () => { onPick(i); b.blur(); });
    return b;
  });
  return (i: number) => btns.forEach((b, j) => b.classList.toggle("active", i === j));
}

/** Only these preset ids (data-presets="a,b,c"), or all. */
export function shownIds(host: HTMLElement): string[] | null {
  return host.dataset.presets ? host.dataset.presets.split(/[\s,]+/).filter(Boolean) : null;
}

/** Label above a control group. */
export const label = (parent: Element, text: string): HTMLElement => htmlEl("div", "ev-lab", text, parent);
