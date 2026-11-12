// The watershed demo's computation (no DOM, so the unit's numbers can be computed in node):
// landscape → markers → flooding (lib/watershed.ts).
import { Grey, gaussianBlur, gradient } from "../lib/imageops";
import { floodWatershed, minimaMarkers, hMinimaMarkers, countLabels } from "../lib/watershed";

export type Land = "gradient" | "intensity" | "inverted";
export type MarkerMode = "minima" | "hmin" | "clicked";
/** A clicked marker: a disc of radius r at (x, y) with a label (1 = background). */
export interface Click { x: number; y: number; label: number; r: number }

/** The landscape to flood: ‖∇(G_σ * I)‖, G_σ * I, or 255 − G_σ * I. */
export function landscape(img: Grey, land: Land, sigma: number): Float32Array {
  if (land === "gradient") return Float32Array.from(gradient(img, sigma).mag.data);
  const b = sigma > 0 ? gaussianBlur(img, sigma) : img;
  return land === "intensity" ? Float32Array.from(b.data) : Float32Array.from(b.data, v => 255 - v);
}

export function clickMarkers(w: number, h: number, clicks: Click[]): Int32Array {
  const m = new Int32Array(w * h);
  for (const c of clicks) {
    const R = Math.ceil(c.r);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const x = Math.round(c.x) + dx, y = Math.round(c.y) + dy;
      if (x < 0 || y < 0 || x >= w || y >= h || dx * dx + dy * dy > c.r * c.r) continue;
      m[y * w + x] = c.label;
    }
  }
  return m;
}

export interface WsRun {
  land: Float32Array;
  markers: Int32Array;
  /** Regional minima of the landscape (before h). */
  nMinima: number;
  /** Number of markers used. */
  nMarkers: number;
  labels: Int32Array;
  level: Float64Array;
  nRegions: number;
  /** Milliseconds for markers + flooding. */
  ms: number;
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

// the number of regional minima of the last landscape (it does not change with h or the clicks)
let lastLand: Float32Array | null = null, lastMinima = 0;
function nMinimaOf(land: Float32Array, w: number, h: number): number {
  if (land !== lastLand) { lastMinima = minimaMarkers(land, w, h, 8).count; lastLand = land; }
  return lastMinima;
}

export function markersFor(land: Float32Array, w: number, h: number, mode: MarkerMode, hh: number, clicks: Click[]): { markers: Int32Array; nMinima: number; nMarkers: number } {
  if (mode === "minima") { const all = minimaMarkers(land, w, h, 8); lastLand = land; lastMinima = all.count; return { markers: all.data, nMinima: all.count, nMarkers: all.count }; }
  if (mode === "hmin") { const m = hMinimaMarkers(land, w, h, hh, 8); return { markers: m.data, nMinima: nMinimaOf(land, w, h), nMarkers: m.count }; }
  const m = clickMarkers(w, h, clicks);
  return { markers: m, nMinima: nMinimaOf(land, w, h), nMarkers: countLabels(m) };
}

export function runWatershed(land: Float32Array, w: number, h: number, mode: MarkerMode, hh: number, clicks: Click[]): WsRun {
  const t0 = now();
  const mk = markersFor(land, w, h, mode, hh, clicks);
  const r = floodWatershed(land, w, h, mk.markers, { conn: 8 });
  const ms = now() - t0;
  return { land, markers: mk.markers, nMinima: mk.nMinima, nMarkers: mk.nMarkers, labels: r.labels, level: r.level, nRegions: countLabels(r.labels), ms };
}
