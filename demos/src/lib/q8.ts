// Int8 weights with one float scale per output unit / output channel, base64-encoded: the
// format written by the offline training scripts (vaeimages/train_*.py, cnnviewer/train_cnn.py,
// unetseg/train_unet.py). Owner: neural-networks day. No DOM except atob (also in node ≥ 16).
//
// A layer stores nOut × nIn weights row by row (row o = all weights of output o: for a
// convolution, its Cin × k × k kernel), quantised as q = round(w / s_o) with s_o =
// max|w_o| / 127.

export interface Q8Layer { q: string; s: number[]; b: number[]; nIn: number; nOut: number }

/** Dequantised weights and biases of a layer. */
export function deq(l: Q8Layer): { W: Float32Array; b: Float32Array; nIn: number; nOut: number } {
  const bin = atob(l.q), W = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const q = bin.charCodeAt(i), v = q > 127 ? q - 256 : q;
    W[i] = v * l.s[Math.floor(i / l.nIn)];
  }
  return { W, b: Float32Array.from(l.b), nIn: l.nIn, nOut: l.nOut };
}

/** Base64 of raw bytes → Uint8Array (for stored example images). */
export function bytesFromB64(s: string): Uint8Array {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
