// A histogram of the grey values with a draggable iso-value c (a vertical line): click or drag
// in the plot to move c. Shared by the marching-cubes day's demos (isocontour, marchcubes,
// meshmeasure). Counts are drawn as log10(1 + count), so the small objects are visible next to
// the large background peak. Built on lib/plot.ts and lib/histogram.ts.
import { Plot, seriesColor } from "../lib/plot";
import { histogram } from "../lib/histogram";

export interface HistC {
  plot: Plot;
  /** Replace the data (and optionally the value range). */
  setValues(v: ArrayLike<number>, lo?: number, hi?: number): void;
  /** Move the c line (no callback). */
  setC(c: number): void;
}

export function histC(parent: Element, o: { lo: number; hi: number; c: number; bins?: number; width?: number; height?: number; xlabel?: string; onChange: (c: number) => void; fmt?: (c: number) => string }): HistC {
  const bins = o.bins ?? 64;
  let lo = o.lo, hi = o.hi, c = o.c;
  const fmt = o.fmt ?? ((v: number) => v.toFixed(Math.abs(hi - lo) < 5 ? 2 : 0));
  const plot = new Plot(parent, { width: o.width ?? 420, height: o.height ?? 190, x: [lo, hi], y: [0, 1], xlabel: o.xlabel ?? "grey value (drag to set c)", yticks: [], margin: { l: 14 } });
  const bars = plot.histogram([lo, hi], [0], { color: seriesColor(0) });
  const line = plot.vline(c, { color: seriesColor(1), text: "c = " + fmt(c), dash: "" });
  const setValues = (v: ArrayLike<number>, l?: number, h?: number) => {
    if (l !== undefined && h !== undefined && (l !== lo || h !== hi)) { lo = l; hi = h; plot.setDomain([lo, hi], [0, 1]); }
    const H = histogram(v, { bins, min: lo, max: hi });
    const counts = Array.from(H.counts, k => Math.log10(1 + k));
    const m = Math.max(...counts) || 1;
    const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * H.binWidth);
    plot.setDomain([lo, hi], [0, m * 1.08]);
    bars.set(edges, counts);
  };
  const setC = (v: number) => { c = v; (line.style as { text?: string }).text = "c = " + fmt(c); line.set(c); };
  let dragging = false;
  const fromEv = (ev: PointerEvent) => {
    const [x] = plot.toData(ev);
    const v = Math.max(lo, Math.min(hi, x));
    setC(v); o.onChange(v);
  };
  plot.svg.style.touchAction = "none";
  plot.svg.style.cursor = "ew-resize";
  plot.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; plot.svg.setPointerCapture(ev.pointerId); fromEv(ev); });
  plot.svg.addEventListener("pointermove", ev => { if (dragging) fromEv(ev); });
  const up = () => { dragging = false; };
  plot.svg.addEventListener("pointerup", up);
  plot.svg.addEventListener("pointercancel", up);
  return { plot, setValues, setC };
}
