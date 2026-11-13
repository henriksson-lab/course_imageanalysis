// UI helpers shared by the optical flow day's demos (lkflow, hsflow, flowviz, ffdreg):
// the flow colour-wheel legend, arrows of a flow on an ImgView, and per-demo CSS injection.
// Candidate for a shared lib together with canny/imgview.ts.
import { htmlEl, svgEl } from "../lib/ui";
import { Flow, flowColour, quiver } from "../lib/flow";
import { ImgView, IMGVIEW_CSS } from "../canny/imgview";

/** Inject the image-view CSS once and this demo's CSS once. */
export function injectCss(name: string, css: string): void {
  if (!document.getElementById("qb-style-ev")) {
    const s = document.createElement("style"); s.id = "qb-style-ev"; s.textContent = IMGVIEW_CSS; document.head.appendChild(s);
  }
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + name; s.textContent = css; document.head.appendChild(s);
}

/** A filled colour wheel (canvas): hue = direction (CCW on screen), saturation = speed; with a scale label. */
export function flowWheel(parent: Element, size = 84): { el: HTMLElement; setScale(max: number): void } {
  const el = htmlEl("div", "fw-legend", "", parent);
  const c = htmlEl("canvas", "fw-wheel", "", el);
  const N = 96; c.width = N; c.height = N;
  c.style.width = size + "px"; c.style.height = size + "px";
  const ctx = c.getContext("2d")!, id = ctx.createImageData(N, N), R = N / 2 - 1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dx = (x + 0.5 - N / 2) / R, dy = (y + 0.5 - N / 2) / R, r = Math.hypot(dx, dy), p = 4 * (y * N + x);
    if (r > 1) { id.data[p + 3] = 0; continue; }
    const col = flowColour(dx, dy, 1);
    id.data[p] = col[0]; id.data[p + 1] = col[1]; id.data[p + 2] = col[2]; id.data[p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  const lab = htmlEl("div", "fw-lab", "", el);
  return { el, setScale: (m: number) => { lab.textContent = `hue = direction\nedge of wheel = ${fmtSpeed(m)} px/frame`; } };
}

export function fmtSpeed(v: number): string {
  return v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2);
}

/** Draw arrows of a flow on an ImgView overlay: every `step` px, length × `scale`. */
export function drawQuiver(view: ImgView, f: Flow, step: number, scale: number, color = "#111", width = 1.6, mask?: ArrayLike<number>): void {
  const g = svgEl("g", {}, view.over);
  for (const q of quiver(f, step)) {
    const i = q.y * f.w + q.x;
    if (mask && !mask[i]) continue;
    const L = Math.hypot(q.u, q.v) * scale;
    if (!(L > 0.25)) { view.add("circle", { cx: q.x + 0.5, cy: q.y + 0.5, r: 0.35, fill: color }, g); continue; }
    const x0 = q.x + 0.5, y0 = q.y + 0.5, x1 = x0 + q.u * scale, y1 = y0 + q.v * scale, ux = q.u * scale / L, uy = q.v * scale / L;
    const hl = Math.min(L * 0.45, Math.max(1.2, step * 0.3)), hw = hl * 0.5;
    view.add("line", { x1: x0, y1: y0, x2: x1 - ux * hl * 0.7, y2: y1 - uy * hl * 0.7, stroke: color, "stroke-width": width, "stroke-linecap": "round" }, g);
    svgEl("polygon", { points: `${x1},${y1} ${x1 - ux * hl - uy * hw},${y1 - uy * hl + ux * hw} ${x1 - ux * hl + uy * hw},${y1 - uy * hl - ux * hw}`, fill: color }, g);
  }
}

/** CSS of the wheel legend (prefixed per demo by the caller). */
export const WHEEL_CSS = (sel: string) => `
${sel} .fw-legend { display: flex; align-items: center; gap: .5em; }
${sel} .fw-wheel { display: block; flex: none; }
${sel} .fw-lab { font-size: 15px; white-space: pre-line; color: var(--qb-muted); line-height: 1.25; }
.reveal ${sel} .fw-lab { font-size: 18px; }
`;
