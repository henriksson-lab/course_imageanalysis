// Fast grey erosion/dilation by a flat disc (no DOM; tested against lib/morph.ts erode/dilate
// with seDisk in demos/test/morphology-test.ts). The disc {dx² + dy² ≤ r²} is a stack of
// horizontal chords; each chord is a 1D sliding min/max along the rows, computed with the
// van Herk / Gil–Werman algorithm (3 comparisons per pixel, whatever the length), then the
// result is the min/max over the 2r + 1 shifted rows. Outside the image is neutral.
// Local helper: lib/morph.ts has only the direct O(n·|B|) version.

/** Sliding min (or max) over the window [x − hw, x + hw] of every row, neutral border. */
export function rowExtremum(f: ArrayLike<number>, w: number, h: number, hw: number, isMax: boolean): Float32Array {
  const out = new Float32Array(w * h);
  if (hw === 0) { for (let i = 0; i < w * h; i++) out[i] = f[i]; return out; }
  const k = 2 * hw + 1, L = Math.ceil((w + 2 * hw) / k) * k, sg = isMax ? -1 : 1;
  // work with sg·f so that one min-routine serves both (max f = −min(−f))
  const p = new Float64Array(L), g = new Float64Array(L), s = new Float64Array(L);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let i = 0; i < L; i++) { const x = i - hw; p[i] = x >= 0 && x < w ? sg * f[o + x] : Infinity; }
    for (let b = 0; b < L; b += k) {
      let m = p[b]; g[b] = m;
      for (let i = b + 1; i < b + k; i++) { const v = p[i]; if (v < m) m = v; g[i] = m; }
      m = p[b + k - 1]; s[b + k - 1] = m;
      for (let i = b + k - 2; i >= b; i--) { const v = p[i]; if (v < m) m = v; s[i] = m; }
    }
    for (let x = 0; x < w; x++) { const a = s[x], c = g[x + k - 1]; out[o + x] = sg * (a < c ? a : c); }
  }
  return out;
}

function discExtremum(f: ArrayLike<number>, w: number, h: number, r: number, isMax: boolean): Float32Array {
  const R = Math.floor(r), rows = new Map<number, Float32Array>(), hwOf: number[] = [];
  for (let dy = -R; dy <= R; dy++) {
    const hw = Math.floor(Math.sqrt(Math.max(0, r * r - dy * dy)) + 1e-9);
    hwOf.push(hw);
    if (!rows.has(hw)) rows.set(hw, rowExtremum(f, w, h, hw, isMax));
  }
  const out = new Float32Array(w * h).fill(isMax ? -Infinity : Infinity);
  for (let j = 0; j <= 2 * R; j++) {
    const dy = j - R, src = rows.get(hwOf[j])!;
    for (let y = Math.max(0, -dy); y < Math.min(h, h - dy); y++) {
      const o = y * w, so = (y + dy) * w;
      if (isMax) for (let x = 0; x < w; x++) { const v = src[so + x]; if (v > out[o + x]) out[o + x] = v; }
      else for (let x = 0; x < w; x++) { const v = src[so + x]; if (v < out[o + x]) out[o + x] = v; }
    }
  }
  return out;
}

export const erodeDisc = (f: ArrayLike<number>, w: number, h: number, r: number) => discExtremum(f, w, h, r, false);
export const dilateDisc = (f: ArrayLike<number>, w: number, h: number, r: number) => discExtremum(f, w, h, r, true);
export const openDisc = (f: ArrayLike<number>, w: number, h: number, r: number) => dilateDisc(erodeDisc(f, w, h, r), w, h, r);
export const closeDisc = (f: ArrayLike<number>, w: number, h: number, r: number) => erodeDisc(dilateDisc(f, w, h, r), w, h, r);
