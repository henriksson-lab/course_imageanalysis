// Label / seed painting on a canvas over an image (owner: random-forest day; used by
// pixelclass and, later, graphcut). Several labels with their own colours, brush size,
// eraser, undo per stroke, clear. The label map is a Uint8Array of the image size
// (0 = unlabelled, otherwise the label's value). The canvas shows, from bottom to top: the
// image, an optional overlay (e.g. a result tint), the painted labels.
//
//   const p = new LabelPainter(host, { width: 128, height: 128, labels: [
//     { value: 1, name: "object", color: "#ff7f0e" }, { value: 2, name: "background", color: "#1f77b4" }],
//     onChange: (map, final) => { if (final) recompute(map); } });
//   p.setImage(greyPixels);          // 0..255 grey values, w*h
//
// Pointer mapping uses getBoundingClientRect (reveal.js scales slides with a CSS transform).
// Controls are built with the ui.ts helpers, so the keyboard goes back to reveal.js.
import { htmlEl, button, slider, Slider } from "./ui";
import { hexToRgb } from "./colour";

export interface PaintLabel {
  /** Value written into the label map (1..255). */
  value: number;
  name: string;
  /** CSS hex colour, e.g. "#ff7f0e". */
  color: string;
}

export interface PainterOptions {
  /** Size of the label map in pixels (the image size). */
  width: number;
  height: number;
  labels: PaintLabel[];
  /** Brush radius in pixels (default 2); the slider runs from 0 to maxBrush. */
  brush?: number;
  maxBrush?: number;
  /** Build the label buttons, brush slider, erase/undo/clear (default true). */
  controls?: boolean;
  /** Where the controls go (default: a panel after the canvas). */
  controlsParent?: HTMLElement;
  /** Opacity of the painted labels over the image (default 0.85). */
  opacity?: number;
  /** Called during a stroke (final = false, at most once per frame) and when it ends (final = true). */
  onChange?: (labels: Uint8Array, final: boolean) => void;
  /** Initial label map. */
  initial?: Uint8Array;
  /** Caption above the canvas. */
  caption?: string;
  /** Maximum number of undo steps (default 40). */
  undoDepth?: number;
}

/** Set every pixel within `radius` of the segment (x0,y0)-(x1,y1) to `value` (pixel units;
 *  pixel (x, y) has its centre at (x + 0.5, y + 0.5)). radius 0 paints single pixels. */
export function paintSegment(map: Uint8Array, w: number, h: number, x0: number, y0: number, x1: number, y1: number, radius: number, value: number): void {
  const r = Math.max(0.5, radius + 0.5), dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy;
  const xa = Math.max(0, Math.floor(Math.min(x0, x1) - r)), xb = Math.min(w - 1, Math.ceil(Math.max(x0, x1) + r));
  const ya = Math.max(0, Math.floor(Math.min(y0, y1) - r)), yb = Math.min(h - 1, Math.ceil(Math.max(y0, y1) + r));
  for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
    const px = x + 0.5, py = y + 0.5;
    let t = L2 > 0 ? ((px - x0) * dx + (py - y0) * dy) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const ex = px - (x0 + t * dx), ey = py - (y0 + t * dy);
    if (ex * ex + ey * ey <= r * r) map[y * w + x] = value;
  }
}

/** Count of pixels per label value (index = value, 0..255). */
export function labelCounts(map: Uint8Array): number[] {
  const c = new Array<number>(256).fill(0);
  for (const v of map) c[v]++;
  return c;
}

/** A painting canvas with its controls. */
export class LabelPainter {
  readonly canvas: HTMLCanvasElement;
  readonly root: HTMLElement;
  readonly w: number;
  readonly h: number;
  labels: Uint8Array;
  /** Value painted by the brush (0 = erase). */
  active: number;
  brush: number;
  readonly opts: PainterOptions;
  private image: ImageData;
  private overlay: ImageData | null = null;
  private undoStack: Uint8Array[] = [];
  private ctx: CanvasRenderingContext2D;
  private labelBtns: HTMLButtonElement[] = [];
  private eraseBtn: HTMLButtonElement | null = null;
  private brushS: Slider | null = null;
  private rgb = new Map<number, [number, number, number]>();
  private pending = false;

  constructor(parent: HTMLElement, opts: PainterOptions) {
    this.opts = opts; this.w = opts.width; this.h = opts.height;
    this.labels = opts.initial ? Uint8Array.from(opts.initial) : new Uint8Array(this.w * this.h);
    this.active = opts.labels[0]?.value ?? 1;
    this.brush = opts.brush ?? 2;
    for (const l of opts.labels) this.rgb.set(l.value, hexToRgb(l.color).map(v => Math.round(255 * v)) as [number, number, number]);

    this.root = htmlEl("figure", "qb-paint", "", parent);
    if (opts.caption) htmlEl("figcaption", "", opts.caption, this.root);
    this.canvas = htmlEl("canvas", "qb-paint-canvas", "", this.root);
    this.canvas.width = this.w; this.canvas.height = this.h;
    this.ctx = this.canvas.getContext("2d")!;
    this.image = new ImageData(this.w, this.h);
    this.image.data.fill(128);
    for (let p = 3; p < this.image.data.length; p += 4) this.image.data[p] = 255;
    this.installPointer();
    if (opts.controls !== false) this.buildControls(opts.controlsParent ?? htmlEl("div", "qb-panel qb-paint-panel", "", parent));
    this.draw();
  }

  /** Background image: ImageData, a canvas/image element, or grey values 0..255 (w*h). */
  setImage(img: ImageData | HTMLCanvasElement | HTMLImageElement | ArrayLike<number>): void {
    if (img instanceof ImageData) this.image = img;
    else if (img instanceof HTMLCanvasElement || img instanceof HTMLImageElement) {
      const c = document.createElement("canvas");
      c.width = this.w; c.height = this.h;
      const cx = c.getContext("2d")!;
      cx.drawImage(img, 0, 0, this.w, this.h);
      this.image = cx.getImageData(0, 0, this.w, this.h);
    } else {
      const id = new ImageData(this.w, this.h);
      for (let p = 0; p < this.w * this.h; p++) { const g = img[p]; id.data[4 * p] = g; id.data[4 * p + 1] = g; id.data[4 * p + 2] = g; id.data[4 * p + 3] = 255; }
      this.image = id;
    }
    this.draw();
  }

  /** RGBA overlay blended over the image by its alpha (e.g. a segmentation tint), or null. */
  setOverlay(o: ImageData | null): void { this.overlay = o; this.draw(); }

  /** Replace the label map (pushes an undo step). */
  setLabels(map: Uint8Array): void { this.pushUndo(); this.labels = Uint8Array.from(map); this.draw(); this.changed(true); }

  clear(): void { this.pushUndo(); this.labels = new Uint8Array(this.w * this.h); this.draw(); this.changed(true); }

  undo(): void {
    const prev = this.undoStack.pop();
    if (!prev) return;
    this.labels = prev; this.draw(); this.changed(true);
  }

  /** Choose the painted label (0 = erase). */
  setActive(value: number): void { this.active = value; this.syncButtons(); }

  setBrush(r: number): void { this.brush = r; this.brushS?.set(r); }

  /** CSS colour of a label value. */
  colorOf(value: number): string { return this.opts.labels.find(l => l.value === value)?.color ?? "#000"; }

  private pushUndo(): void {
    this.undoStack.push(Uint8Array.from(this.labels));
    if (this.undoStack.length > (this.opts.undoDepth ?? 40)) this.undoStack.shift();
  }

  private changed(final: boolean): void {
    if (final) { this.opts.onChange?.(this.labels, true); return; }
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.opts.onChange?.(this.labels, false); });
  }

  /** Redraw the canvas. */
  draw(): void {
    const out = new ImageData(this.w, this.h), d = out.data, im = this.image.data, ov = this.overlay?.data, a = this.opts.opacity ?? 0.85;
    for (let p = 0; p < this.w * this.h; p++) {
      let r = im[4 * p], g = im[4 * p + 1], b = im[4 * p + 2];
      if (ov) { const t = ov[4 * p + 3] / 255; r += t * (ov[4 * p] - r); g += t * (ov[4 * p + 1] - g); b += t * (ov[4 * p + 2] - b); }
      const l = this.labels[p];
      if (l) { const c = this.rgb.get(l); if (c) { r += a * (c[0] - r); g += a * (c[1] - g); b += a * (c[2] - b); } }
      d[4 * p] = r; d[4 * p + 1] = g; d[4 * p + 2] = b; d[4 * p + 3] = 255;
    }
    this.ctx.putImageData(out, 0, 0);
  }

  /** Pointer position in label-map pixels. */
  toPixel(ev: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * this.w, (ev.clientY - r.top) / r.height * this.h];
  }

  private installPointer(): void {
    const cv = this.canvas;
    let last: [number, number] | null = null;
    cv.addEventListener("pointerdown", ev => {
      ev.preventDefault();
      cv.setPointerCapture(ev.pointerId);
      this.pushUndo();
      last = this.toPixel(ev);
      paintSegment(this.labels, this.w, this.h, ...last, ...last, this.brush, this.active);
      this.draw(); this.changed(false);
    });
    cv.addEventListener("pointermove", ev => {
      if (!last) return;
      const q = this.toPixel(ev);
      paintSegment(this.labels, this.w, this.h, ...last, ...q, this.brush, this.active);
      last = q;
      this.draw(); this.changed(false);
    });
    const up = () => { if (last) { last = null; this.changed(true); } };
    cv.addEventListener("pointerup", up);
    cv.addEventListener("pointercancel", up);
  }

  private buildControls(panel: HTMLElement): void {
    const row = htmlEl("div", "qb-presets", "", panel);
    this.labelBtns = this.opts.labels.map(l => {
      const b = button(row, l.name, () => this.setActive(l.value), "qb-paint-label");
      b.style.setProperty("--qb-label", l.color);
      return b;
    });
    this.eraseBtn = button(row, "erase", () => this.setActive(0));
    this.brushS = slider(panel, "brush", 0, this.opts.maxBrush ?? 10, 1, this.brush, v => { this.brush = v; });
    const row2 = htmlEl("div", "qb-presets", "", panel);
    button(row2, "undo", () => this.undo());
    button(row2, "clear", () => this.clear());
    this.syncButtons();
  }

  private syncButtons(): void {
    this.labelBtns.forEach((b, i) => b.classList.toggle("active", this.opts.labels[i].value === this.active));
    this.eraseBtn?.classList.toggle("active", this.active === 0);
  }
}
