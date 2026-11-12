// CPU side of the texformats demo: turn grey values in [0, 1] into the bytes that are
// uploaded for each texture format, and decode them again (what the GPU returns when it
// samples the texture). The demo's readout uses exactly these functions, so the numbers
// shown are the ones stored on the GPU.
import { F16, encode, decode } from "../floatbits/fp";

export type FormatId = "RGBA8" | "SRGB8_ALPHA8" | "RGB565" | "RGB10_A2" | "R16F" | "R32F" | "R16UI";

export interface FormatInfo {
  id: FormatId;
  label: string;          // short conceptual name shown to students
  bytes: number;          // bytes per texel
  kind: "normalised" | "float" | "integer" | "normalised (sRGB)";
  channels: string;
  filterable: "yes" | "no" | "extension";   // R32F needs OES_texture_float_linear
  note: string;
}

export const FORMATS: FormatInfo[] = [
  { id: "RGBA8", label: "8 bit", bytes: 4, kind: "normalised", channels: "R8 G8 B8 A8", filterable: "yes", note: "k / 255" },
  { id: "SRGB8_ALPHA8", label: "8 bit sRGB", bytes: 4, kind: "normalised (sRGB)", channels: "R8 G8 B8 A8", filterable: "yes", note: "8-bit sRGB code (more codes for darks), decoded by the GPU before filtering" },
  { id: "RGB565", label: "5-6-5", bytes: 2, kind: "normalised", channels: "R5 G6 B5", filterable: "yes", note: "red and blue k / 31, green k / 63, in 16 bits" },
  { id: "RGB10_A2", label: "10 bit", bytes: 4, kind: "normalised", channels: "R10 G10 B10 A2", filterable: "yes", note: "k / 1023 per channel, 2-bit alpha" },
  { id: "R16F", label: "half float", bytes: 2, kind: "float", channels: "R16F", filterable: "yes", note: "1 sign, 5 exponent, 10 fraction bits" },
  { id: "R32F", label: "float32", bytes: 4, kind: "float", channels: "R32F", filterable: "extension", note: "1 sign, 8 exponent, 23 fraction bits" },
  { id: "R16UI", label: "16-bit integer", bytes: 2, kind: "integer", channels: "R16UI", filterable: "no", note: "integer 0..65535; the shader divides by 65535" },
];

export const unorm = (x: number, bits: number) => Math.round(Math.min(1, Math.max(0, x)) * (2 ** bits - 1));

/** sRGB encoding of a linear value (IEC 61966-2-1), and its inverse. */
export function srgbEncode(x: number): number {
  const v = Math.min(1, Math.max(0, x));
  return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
}
export function srgbDecode(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** Stored code(s) for one grey value, as a short hex/binary string, and the decoded value
 *  per channel (for grey input, R = G = B except for RGB565). */
export function storeTexel(id: FormatId, x: number): { stored: string; decoded: number[] } {
  switch (id) {
    case "RGBA8": { const k = unorm(x, 8); return { stored: `R=G=B=${k} (0x${hex(k, 2)}), A=255`, decoded: [k / 255] }; }
    case "SRGB8_ALPHA8": {
      const k = unorm(srgbEncode(x), 8);
      return { stored: `R=G=B=${k} (0x${hex(k, 2)}) sRGB code`, decoded: [srgbDecode(k / 255)] };
    }
    case "RGB565": {
      const r = unorm(x, 5), g = unorm(x, 6), b = unorm(x, 5);
      const w = (r << 11) | (g << 5) | b;
      return { stored: `0x${hex(w, 4)}: R=${r} G=${g} B=${b}`, decoded: [r / 31, g / 63, b / 31] };
    }
    case "RGB10_A2": {
      const k = unorm(x, 10);
      const w = ((3 << 30) | (k << 20) | (k << 10) | k) >>> 0;
      return { stored: `0x${hex(w, 8)}: R=G=B=${k}, A=3`, decoded: [k / 1023] };
    }
    case "R16F": {
      const b = encode(F16, x);
      return { stored: `0x${hex(b, 4)}`, decoded: [decode(F16, b)] };
    }
    case "R32F": {
      const v = Math.fround(x);
      const dv = new DataView(new ArrayBuffer(4));
      dv.setFloat32(0, v);
      return { stored: `0x${hex(dv.getUint32(0), 8)}`, decoded: [v] };
    }
    case "R16UI": { const k = unorm(x, 16); return { stored: `${k} (0x${hex(k, 4)})`, decoded: [k / 65535] }; }
  }
}

const hex = (v: number, n: number) => v.toString(16).toUpperCase().padStart(n, "0");

/** Upload arrays for a whole w x h grey image (row-major, row 0 at the top). */
export function uploadData(id: FormatId, x: ArrayLike<number>): ArrayBufferView {
  const n = x.length;
  switch (id) {
    case "RGBA8": case "SRGB8_ALPHA8": {
      const a = new Uint8Array(4 * n);
      for (let i = 0; i < n; i++) {
        const k = unorm(id === "RGBA8" ? x[i] : srgbEncode(x[i]), 8);
        a[4 * i] = a[4 * i + 1] = a[4 * i + 2] = k; a[4 * i + 3] = 255;
      }
      return a;
    }
    case "RGB565": {
      const a = new Uint16Array(n);
      for (let i = 0; i < n; i++) a[i] = (unorm(x[i], 5) << 11) | (unorm(x[i], 6) << 5) | unorm(x[i], 5);
      return a;
    }
    case "RGB10_A2": {
      const a = new Uint32Array(n);
      for (let i = 0; i < n; i++) { const k = unorm(x[i], 10); a[i] = ((3 << 30) | (k << 20) | (k << 10) | k) >>> 0; }
      return a;
    }
    case "R16F": { const a = new Uint16Array(n); for (let i = 0; i < n; i++) a[i] = encode(F16, x[i]); return a; }
    case "R32F": return Float32Array.from(x as ArrayLike<number>);
    case "R16UI": { const a = new Uint16Array(n); for (let i = 0; i < n; i++) a[i] = unorm(x[i], 16); return a; }
  }
}

/** Bytes for a w x h texture, with or without a full mipmap chain. */
export function textureBytes(id: FormatId, w: number, h: number, mip: boolean): number {
  const b = FORMATS.find(f => f.id === id)!.bytes;
  if (!mip) return b * w * h;
  let t = 0;
  for (;;) { t += b * w * h; if (w === 1 && h === 1) break; w = Math.max(1, w >> 1); h = Math.max(1, h >> 1); }
  return t;
}
